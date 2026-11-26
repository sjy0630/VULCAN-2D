"""Select additional state/transport corrections and freeze gate predictions.

All model comparisons fix raw sheet partitions first. The 1.1 V gate is
withheld from the endpoint transfer experiment; its earlier cycles can train
a separate same-gate fit and the final three-anchor prospective model.
"""
import argparse
from dataclasses import asdict, replace
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from . import cycle_quality as quality
from . import model_shared_device as model
from . import predict_gate as predictor
from . import review_gate_model as review
from . import validate_gate_prediction as original
from . import validate_optimized_device as opt
from . import validate_shared_device as shared
from . import validate_transition_regions as audit

PHASES = ("SET", "RESET")
ENDPOINTS = (.9, 1.3)
GATES = (.9, 1.1, 1.3)
SET_APPROACHES = ("reference", "anchored", "threshold_population")
RESET_APPROACHES = ("reference", "anchored", "feedback")


def _refit_anchors(groups, circuits, phase, starts, windows):
    amplitude_fields = (("off_set_a", "soft_set_a") if phase == "SET"
                        else ("off_reset_a", "soft_reset_a"))
    gates = tuple(sorted(starts))
    out = {}
    shapes = {}
    for m, (side, interval) in enumerate(windows):
        pieces = [review._read_window(groups["1T1M",g,phase],
                                      circuits["1T1M",g,phase], side, interval) for g in gates]
        if not all(len(p) for p in pieces):
            raise ValueError("Transport anchor has no usable observations")
        use_field = m == 0
        init = [np.log10(getattr(starts[g],amplitude_fields[m])) for g in gates]
        init = np.clip(init, -17.9, -3.1).tolist()
        fit_start = init + ([6.] if use_field else [1.3])
        low = [-18.]*len(gates) + ([0.] if use_field else [.6])
        high = [-3.]*len(gates) + ([12.] if use_field else [2.7])

        def values(p):
            shape = p[-1]
            return np.concatenate([
                (p[j] + (np.log10(v)+shape*np.sqrt(v)/np.log(10) if use_field else shape*np.log10(v)) - y)
                * np.sqrt(350/len(v)) for j,(v,y) in enumerate(sample.T for sample in pieces)
            ])
        result = least_squares(values,fit_start,bounds=(low,high), loss="soft_l1", f_scale=.15)
        if not result.success:
            raise RuntimeError("SET transport anchor failure")
        shapes[m] = result.x[-1]
        for j,g in enumerate(gates):
            out.setdefault(g,{})[amplitude_fields[m]] = float(10**result.x[j])
    return shapes,{g:replace(starts[g],**out[g]) for g in gates}


def _lookup_coordinates(row, circuit):
    drain_v, drain_i = circuit.transistor
    levels, indices = np.unique(drain_i,return_index=True)
    v,current = np.abs(row.voltage),np.abs(row.current)
    supported = current <= np.interp(v,drain_v,drain_i)
    local = v - np.interp(current,levels,drain_v[indices])
    use = (row.masks["use_unclamped"] & row.masks["branch_outward"] & supported
           & (local >= .25))
    return v,local,current,use


def _latent_thresholds(rows, device, prep, circuit):
    """Estimate one threshold and width per independent SET sweep."""
    fits = []
    for row in rows:
        _,local,current,use = _lookup_coordinates(row,circuit)
        local_use,current = local[use],current[use]
        if not len(local_use):
            continue
        off = model.device_current(local_use,0,0,device,prep,"SET")
        on = model.device_current(local_use,1,0,device,prep,"SET")
        active = (on>5*off) & (local_use<2.45)
        use_shape = active & (current >= off*.25) & (current <= on*1.2)
        x = local_use[use_shape]
        measured = current[use_shape]
        if len(x) < 20:
            continue
        lo, hi = off[use_shape],on[use_shape]
        estimated = (measured-lo)/(hi-lo)
        if (np.count_nonzero(estimated < .15) < 4 or
                np.count_nonzero(estimated > .65) < 4):
            continue

        def loss(p):
            d = replace(device,set_center_v=p[0],set_width_v=p[1])
            s = model._soft_formation(x,0.,d,prep)
            obs = (1-s)*lo+s*hi
            return np.log10(obs)-np.log10(measured)

        solution = least_squares(loss,[1.6,.075],bounds=([.42,.008],[2.20,.33]),
                                 loss="soft_l1",f_scale=.3, max_nfev=80)
        if solution.success:
            fits.append((row.label,float(solution.x[0]),float(solution.x[1])))
    return fits


def _fit_set_anchored(groups, circuits, calibration, method):
    starts = dict(zip(calibration.gates_v,calibration.preparations))
    transport,preps = _refit_anchors(
        groups,circuits,"SET",starts,((True,(.21,.63)),(False,(.38,1.40))))
    d = replace(calibration.device,beta_set=transport[0],soft_exponent=transport[1])
    gate_group = {g:groups["1T1M",g,"SET"] for g in starts}
    traces = {g:review._weighted_trace(rows,"SET") for g,rows in gate_group.items()}
    fits = {g:_latent_thresholds(rows,d,preps[g],circuits["1T1M",g,"SET"])
            for g,rows in gate_group.items()}

    def terminal_residual(p):
        device = replace(d,set_center_v=p[0],set_width_v=p[1])
        result = []
        for g,(v,use,y,w) in traces.items():
            predicted = model.simulate_sweep(v,"SET",device,preps[g],circuits["1T1M",g,"SET"])
            result.append(w*(np.log10(np.abs(predicted["current_A"][use]))-y))
        return np.concatenate(result)

    if method=="threshold_population" and all(len(f)>=4 for f in fits.values()):
        # The width of an individual transition and the distribution of
        # transition centers are different observables.
        center = float(np.mean([np.median([p[1] for p in f]) for f in fits.values()]))
        width = float(np.mean([np.median([p[2] for p in f]) for f in fits.values()]))
    else:
        r = least_squares(terminal_residual,[1.66,.08],bounds=([.52,.008],[2.10,.30]),
                          max_nfev=130,diff_step=3e-5)
        if not r.success:
            raise RuntimeError(r.message)
        center,width = map(float,r.x)
    return replace(calibration, device=replace(d,set_center_v=center,set_width_v=width),
                   preparations=tuple(preps[g] for g in starts)), fits


def _fit_reset_feedback(groups,circuits,cal):
    d = cal.device
    gate_data = {g:review._weighted_trace(groups["1T1M",g,"RESET"],"RESET")
                 for g in cal.gates_v}
    states = {g:review._starting_state(g,groups,circuits,cal) for g in cal.gates_v}

    def parse(x):
        return replace(d, reset_center_v=x[0], reset_width_v=x[1],
                       reset_hazard_per_v=np.exp(x[2]))

    def residual(x):
        device = parse(x)
        parts = []
        for g,(v,use,y,w) in gate_data.items():
            curve = model.simulate_sweep(v,"RESET",device,cal.preparation(g),
                                         circuits["1T1M",g,"RESET"],states[g])
            parts.append(w*(np.log10(np.abs(curve["current_A"][use]))-y))
        return np.concatenate(parts)

    start = [min(max(d.reset_center_v-.12,.41),1.45),.09,np.log(13.)]
    r = least_squares(residual,start,bounds=([.18,.025,np.log(.6)],
                                             [1.55,.50,np.log(85.)]),
                     max_nfev=160, diff_step=2e-5, ftol=2e-8)
    if not r.success:
        raise RuntimeError(r.message)
    return replace(cal,device=parse(r.x))


def _fit_choices(groups,circuits,reference):
    set_anchor,set_fits = _fit_set_anchored(groups,circuits,reference,"anchored")
    population,_ = _fit_set_anchored(groups,circuits,reference,"threshold_population")
    anchored = review._fit_reset_anchors(groups,circuits,reference)
    feedback = _fit_reset_feedback(groups,circuits,anchored)
    return {"SET":{"reference":reference,"anchored":set_anchor,"threshold_population":population},
            "RESET":{"reference":reference,"anchored":anchored,"feedback":feedback}},set_fits


def _metric(s):
    m = {k:s[k]["pooled_rmse_decade"] for k in ("use_transition","use_unclamped","eligible","outward","return")}
    m["event"] = s["event"]
    return m


def _score_endpoint(cal,phase,traces,circuits,g):
    k = ("1T1M",g,phase)
    rows, set_wave = traces[k],traces["1T1M",g,"SET"][0].voltage
    p = review._phase_predictions(rows,phase,g,cal,circuits,set_wave)
    return _metric(review._phase_scores(rows,p,phase,circuits[k]))


def _choose(options):
    # Preserve transition accuracy while still charging the model for fitting
    # an entire trace badly. An undetected transition carries a fixed cost.
    def mean_score(by_gate):
        return float(np.mean([
            .50*s["use_transition"]+.35*s["use_unclamped"]+
            .15*(s["event"]["mae_V"] if s["event"]["mae_V"] is not None else .75)
            for s in by_gate.values()
        ]))
    scores = {method:mean_score(s) for method,s in options.items()}
    best = min(scores.values())
    # Avoid promoting a more complicated form on a negligible development gain.
    selected = next(name for name in scores if scores[name] <= best*1.005)
    return selected,scores


def _blend(set_calibration,reset_calibration):
    assert set_calibration.gates_v==reset_calibration.gates_v
    ps=tuple(replace(a,off_reset_a=b.off_reset_a,soft_reset_a=b.soft_reset_a)
             for a,b in zip(set_calibration.preparations,reset_calibration.preparations))
    fields=("beta_reset","reset_soft_alpha","reset_center_v","reset_width_v",
            "reset_off_power","reset_on_power","reset_late_fraction",
            "reset_center_gap_v","reset_hazard_per_v")
    def merge(a, b):
        return replace(a, **{k:getattr(b,k) for k in fields})

    gates=set_calibration.gates_v
    per_gate=()
    if set_calibration.anchor_devices or reset_calibration.anchor_devices:
        per_gate=tuple(merge(set_calibration.device_at(g),reset_calibration.device_at(g)) for g in gates)
    return predictor.GateCalibration(gates,merge(set_calibration.device,reset_calibration.device),ps,per_gate)


def _score_cycle(calibration, gate, data, circuits, arrays, prefix):
    keys=tuple(("1T1M",gate,phase) for phase in PHASES)
    written,erased=(data[k] for k in keys)
    if [s.label[1:] for s in written]!=[r.label[1:] for r in erased]:
        raise ValueError("SET and RESET cycle identifiers do not align")
    pair=tuple(circuits[k] for k in keys)
    results=[calibration.predict(gate,s.voltage,r.voltage,circuits=pair) for s,r in zip(written,erased)]
    scores={}
    for k,predictions in zip(keys,zip(*results)):
        scores[k[2]]=_metric(review._phase_scores(data[k],predictions,k[2],circuits[k]))
        shared._record_curves(arrays,f"{prefix}_{k[2]}",data[k],predictions)
    return scores


def _read_data():
    raw,circuits=shared.load_groups(include_abnormal_pairs=True)
    flags=quality.screen_groups(raw)
    usable = {k:quality.split_with_screen(rows,flags[k]) for k,rows in raw.items()}
    first = {k:quality.split_with_screen(rows,flags[k],stop=quality.TRAINING_BOUNDARY[k[1]])
             for k,rows in raw.items()}
    last = {k:quality.split_with_screen(rows,flags[k],start=quality.TRAINING_BOUNDARY[k[1]])
            for k,rows in raw.items()}
    return raw,usable,first,last,circuits


def _gate_cv(usable,first,last,circuits,arrays):
    endpoint_data = lambda data: {k:rows for k,rows in data.items() if k[1] in ENDPOINTS}
    endpoint_circuits = endpoint_data(circuits)
    fit_ref,_ = original.fit_endpoints(endpoint_data(first),endpoint_circuits)
    choices,_ = _fit_choices(endpoint_data(first),circuits,fit_ref)
    endpoint_scores={phase:{name:{str(g):_score_endpoint(cal,phase,last,circuits,g) for g in ENDPOINTS}
                            for name,cal in choices[phase].items()} for phase in PHASES}
    chosen, objective = {}, {}
    for phase in PHASES:
        chosen[phase], objective[phase] = _choose(endpoint_scores[phase])
        print("Endpoint validation",phase,chosen[phase],objective[phase],flush=True)
    all_ref,_ = original.fit_endpoints(endpoint_data(usable),endpoint_circuits)
    final_choices, spread = _fit_choices(endpoint_data(usable),circuits,all_ref)
    results = {}
    for phase in PHASES:
        k = ("1T1M",1.1,phase)
        rows = last[k]
        set_wave = last["1T1M",1.1,"SET"][0].voltage
        results[phase] = {}
        for method,cal in final_choices[phase].items():
            predicted = review._phase_predictions(rows,phase,1.1,cal,circuits,set_wave)
            results[phase][method] = _metric(review._phase_scores(rows,predicted,phase,circuits[k]))
            shared._record_curves(arrays,f"gate_holdout_{phase}_{method}",rows,predicted)
    # Score the coherent cycle: the selected SET endpoint really initializes RESET.
    combined = _blend(final_choices["SET"][chosen["SET"]], final_choices["RESET"][chosen["RESET"]])
    coherent=_score_cycle(combined,1.1,last,circuits,arrays,"gate_holdout_selected")
    local=_all_gate_calibration(usable,circuits,chosen,ENDPOINTS)
    local_metrics=_score_cycle(local,1.1,last,circuits,arrays,"gate_holdout_interpolated_shape")
    # Unlike the phase-family choices, this prospective architecture choice
    # is informed by the historical target-gate errors. No target currents
    # enter either calibration used to compute those errors.
    shared_objective={phase:_choose({"shared":{"1.1":coherent[phase]},
                                    "per_gate":{"1.1":local_metrics[phase]}})
                      for phase in PHASES}
    print("Historical gate transfer",shared_objective,flush=True)
    feedback_model=_blend(final_choices["SET"][chosen["SET"]],final_choices["RESET"]["feedback"])
    reset_key=("1T1M",1.1,"RESET")
    peak_by_gate={str(g):float(np.max(np.abs(usable["1T1M",g,"RESET"][0].voltage))) for g in GATES}
    common_peak=min(peak_by_gate.values())
    matched_outward={}
    for name in final_choices["RESET"]:
        rows=last[reset_key]
        currents=arrays[f"gate_holdout_RESET_{name}_current_A"]
        pairs=[]
        for row,current in zip(rows,currents):
            keep=(row.masks["use_unclamped"] & row.masks["branch_outward"]
                  & (np.abs(row.voltage)<=common_peak+1e-9))
            pairs.append((np.abs(row.current[keep]),np.abs(current[keep])))
        matched_outward[name]=audit._summarize_residuals(pairs)
    return ({"chosen":chosen,"endpoint_objective":objective,"endpoint_scores":endpoint_scores,
             "1p1_holdout_36_to_53":results,"coherent_selected":coherent,
             "interpolated_shape_benchmark":local_metrics,
             "prospective_shape_choice":{p:shared_objective[p][0] for p in PHASES},
             "shape_objective":{p:shared_objective[p][1] for p in PHASES},
             "reset_peak_abs_V":peak_by_gate,
             "reset_outward_common_amplitude":matched_outward,
             "individual_training_thresholds":{str(g):rr for g,rr in spread.items()},
             "feedback_grid_check_1p1V":_grid_audit(1.1,feedback_model,-1.5)},combined)


def _single_reference(groups,circuits,gate):
    ks,kr=("1T1M",gate,"SET"),("1T1M",gate,"RESET")
    if gate == 1.1:
        dev,prep = opt.fit_set(groups[ks],circuits[ks],"recent6")
    else:
        dev,prep = shared.fit_soft_set(groups[ks],circuits[ks])
    start=model.simulate_sweep(groups[ks][0].voltage,"SET",dev,prep,circuits[ks])["final"]
    dev,prep=shared.fit_soft_reset(groups[kr],circuits[kr],dev,prep,start)
    return predictor.GateCalibration((gate,),dev,(prep,))


def _single_gate_cv(first,last,circuits,arrays):
    gate=1.1
    keys=[("1T1M",gate,p) for p in PHASES]
    early={k:first[k][:-8] for k in keys}
    internal={k:first[k][-8:] for k in keys}
    training={k:first[k] for k in keys}
    e,_=_fit_choices(early,circuits,_single_reference(early,circuits,gate))
    options={phase:{name:{str(gate):_score_endpoint(cal,phase,internal,circuits,gate)}
                    for name,cal in e[phase].items()} for phase in PHASES}
    chosen,scores={},{}
    for phase in PHASES:
        chosen[phase],scores[phase]=_choose(options[phase])
        print("Same-gate temporal validation",phase,chosen[phase],scores[phase],flush=True)
    trained,_=_fit_choices(training,circuits,_single_reference(training,circuits,gate))
    results={}
    for phase in PHASES:
        k=("1T1M",gate,phase)
        results[phase]={}
        for method,cal in trained[phase].items():
            predictions=review._phase_predictions(last[k],phase,gate,cal,circuits,last[keys[0]][0].voltage)
            results[phase][method]=_metric(review._phase_scores(last[k],predictions,phase,circuits[k]))
            shared._record_curves(arrays,f"same_gate_{phase}_{method}",last[k],predictions)
    calibration=_blend(trained["SET"][chosen["SET"]],trained["RESET"][chosen["RESET"]])
    coherent=_score_cycle(calibration,gate,last,circuits,arrays,"same_gate_selected")
    return {"chosen":chosen,"development_objectives":scores,"development_scores":options,
            "1p1_holdout_36_to_53":results,"coherent_selected":coherent},calibration


def _bare_fit(rows, circuits, device, prep):
    sk,rk=("1M",None,"SET"),("1M",None,"RESET")
    dev,p=shared.fit_bare_set(rows[sk],circuits[sk],device,prep)
    return shared.fit_bare_reset(rows[rk],circuits[rk],dev,p,model.State(hard=1.))


def _fit_bare_pretransport(rows, device, preparation):
    parts=[]
    for row in rows:
        voltage=np.abs(row.voltage)
        keep=row.masks["use_unclamped"] & row.masks["branch_outward"]
        onset=shared._clip_voltage(row.voltage,row.current)
        mask=keep & (voltage>=.4) & (voltage<=onset-.15+1e-7)
        parts.append(np.column_stack((voltage[mask],np.log10(np.abs(row.current[mask])))))
    v,y=np.concatenate(parts).T

    def residual(p):
        return (p[0]+np.log10(v)+p[1]*np.sqrt(v)/np.log(10)-y)

    r=least_squares(residual,[-11.5,4.],bounds=([-17,0],[-6,12]),loss="soft_l1",f_scale=.25)
    if not r.success:
        raise RuntimeError(r.message)
    return (replace(device,beta_set=r.x[1]),
            replace(preparation,off_set_a=10**r.x[0],soft_available=0.))


def _bare_set_score(rows, d, prep, circuit):
    curves=[shared.simulation(row,"SET",d,prep,circuit) for row in rows]
    result=shared.scores(rows,curves,"SET",circuit)
    pre_pairs=[]
    for row,pred in zip(rows,curves):
        onset=shared._clip_voltage(row.voltage,row.current)
        ok=(row.masks["use_unclamped"] & row.masks["branch_outward"]
            & (row.voltage <= onset-.15+1e-7))
        pre_pairs.append((np.abs(row.current[ok]), np.abs(pred["current_A"][ok])))
    result["pre_compliance"] = audit._summarize_residuals(pre_pairs)
    return result,curves


def _bare_reset_candidates(rows,circuit,device,prep,initial):
    d,p=shared.fit_bare_reset(rows,circuit,device,prep,initial)
    candidates={"typical":(d,p)}
    # If the terminal drop is abrupt, a smooth least-squares optimum can
    # advance it to accommodate a minority of early switching cycles.
    # Compare a prior on the late rupture location from training events.
    events=[row.event["event_abs_voltage_V"] for row in rows if "event_abs_voltage_V" in row.event]
    early,late=d.hard_reset_centers_v
    if events:
        anchor=min(max(float(np.median(events)),early+.04),2.75)
        candidates["event_anchored"]=(replace(d,hard_reset_centers_v=(early,anchor)),p)
    return candidates


def _bare_comparison(first,last,circuits,reference,arrays):
    sk,rk=("1M",None,"SET"),("1M",None,"RESET")
    dev,p=reference.device,reference.preparations[0]
    fixed,pfixed=_bare_fit(first,circuits,dev,p)
    anchored,pa=_fit_bare_pretransport(first[sk],fixed,pfixed)
    variants=(("shared_pretransport",fixed,pfixed),("bare_pretransport",anchored,pa))
    scores={}
    for name,d,prep in variants:
        scores[name],curves=_bare_set_score(last[sk],d,prep,circuits[sk])
        shared._record_curves(arrays,"bare_SET_"+name,last[sk],curves)

    train_set,valid_set=first[sk][:-4],first[sk][-4:]
    inner=shared.fit_bare_set(train_set,circuits[sk],dev,p)
    option_pre=(inner,_fit_bare_pretransport(train_set,*inner))
    set_dev={name:_bare_set_score(valid_set,*cal,circuits[sk])[0] for (name,_,_),cal in zip(variants,option_pre)}
    set_dev_score={name:.5*s["pre_compliance"]["pooled_rmse_decade"]+
                   .5*s["use_unclamped"]["pooled_rmse_decade"] for name,s in set_dev.items()}
    cutoff=min(set_dev_score.values())*1.005
    set_choice=next(name for name,s in set_dev_score.items() if s<=cutoff)
    set_device,set_prep=next((d,p) for n,d,p in variants if n==set_choice)

    hard = model.State(hard=1.)
    train_reset,valid_reset=first[rk][:-4],first[rk][-4:]
    reset_dev={}
    for name,(d,p) in _bare_reset_candidates(train_reset,circuits[rk],set_device,set_prep,hard).items():
        curves=[shared.simulation(row,"RESET",d,p,circuits[rk],hard) for row in valid_reset]
        reset_dev[name]={"validation":_metric(review._phase_scores(valid_reset,curves,"RESET",circuits[rk]))}
    reset_choice,reset_dev_scores=_choose(reset_dev)
    reset_params=_bare_reset_candidates(first[rk],circuits[rk],set_device,set_prep,hard)
    reset_scores={}
    for name,(d,p) in reset_params.items():
        curves=[shared.simulation(row,"RESET",d,p,circuits[rk],hard) for row in last[rk]]
        reset_scores[name]=shared.scores(last[rk],curves,"RESET",circuits[rk])
        shared._record_curves(arrays,"bare_RESET_"+name,last[rk],curves)
    best_device,best_prep=reset_params[reset_choice]
    samples=[]
    failed=[]
    for row in first[rk]:
        try:
            d,p=shared.fit_bare_reset([row],circuits[rk],fixed,pfixed,hard)
        except (ValueError,RuntimeError) as exc:
            failed.append({"sheet":row.label,"reason":str(exc)})
            continue
        samples.append(shared.simulation(last[rk][0],"RESET",d,p,circuits[rk],hard))
    ensemble,quantiles=shared.score_ensemble(last[rk],samples,"RESET",circuits[rk])
    arrays["bare_RESET_population_band_A"]=quantiles
    calibrated={"device":asdict(best_device),"preparation":asdict(best_prep),
                "SET_compliance_A":circuits[sk].compliance_a,"RESET_compliance_A":circuits[rk].compliance_a}
    return {"SET":scores,"RESET":{**reset_scores,"population":ensemble,"training_fit_failures":failed},
            "chosen":{"SET":set_choice,"RESET":reset_choice},
            "training_objective":{"SET":set_dev_score,"RESET":reset_dev_scores},
            "bare_uses_same_equations":True,
            "interpretation":"1 mA SET plateau is a bound; a post-SET intrinsic current cannot be scored."},calibrated


def _all_gate_calibration(usable,circuits,gate_choice,gates=GATES):
    devices,preps = [],[]
    for gate in gates:
        data={("1T1M",gate,phase):usable["1T1M",gate,phase] for phase in PHASES}
        base=_single_reference(data,circuits,gate)
        if gate_choice["SET"]=="reference":
            s=base
        else:
            s,_=_fit_set_anchored(data,circuits,base,gate_choice["SET"])
        reset_choice=gate_choice["RESET"]
        r=base if reset_choice=="reference" else review._fit_reset_anchors(data,circuits,s)
        if reset_choice=="feedback":
            r=_fit_reset_feedback(data,circuits,r)
        both=_blend(s,r)
        devices.append(both.device)
        preps.append(both.preparations[0])
    # All anchors use the same equation family. Only calibrated coefficients
    # are interpolated between gates.
    return predictor.GateCalibration(gates,devices[len(gates)//2],tuple(preps),tuple(devices))


def _prospective_model(usable,circuits,family,shape):
    measured={k:rows for k,rows in usable.items() if k[0]=="1T1M"}
    models={}
    if "per_gate" in shape.values():
        models["per_gate"]=_all_gate_calibration(measured,circuits,family)
    if "shared" in shape.values():
        limits={g:{p:len(measured["1T1M",g,p]) for p in PHASES} for g in GATES}
        dev,preps,set_devices=opt.fit_all_gates(measured,circuits,limits,"SET")
        dev,preps,_=opt.fit_all_gates(measured,circuits,limits,"RESET",dev,preps,set_by_gate=set_devices)
        base=predictor.GateCalibration(GATES,dev,tuple(preps[g] for g in GATES))
        s=base
        if family["SET"]!="reference":
            s,_=_fit_set_anchored(measured,circuits,base,family["SET"])
        r=base
        if family["RESET"]!="reference":
            r=review._fit_reset_anchors(measured,circuits,s)
        if family["RESET"]=="feedback":
            r=_fit_reset_feedback(measured,circuits,r)
        models["shared"]=_blend(s,r)
    return _blend(models[shape["SET"]],models[shape["RESET"]])


def _write_prediction(path, gate, cal, set_wave, reset_wave):
    columns=("voltage_V","current_A","device_voltage_V","soft_state","hard_state")
    written,erased=cal.predict(gate,set_wave,reset_wave)
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f)
        w.writerow(("phase","gate_V",*columns))
        for phase,result in zip(PHASES,(written,erased)):
            w.writerows((phase,gate,*row) for row in zip(*(result[c] for c in columns)))
    return written,erased


def _grid_audit(gate, cal, reset_peak):
    coarse_step=.02
    coarse_waves=(predictor.triangle(5.,coarse_step),predictor.triangle(reset_peak,coarse_step))
    fine_waves=(predictor.triangle(5.,.005),predictor.triangle(reset_peak,.005))
    cs,cr=predictor.circuits_at_gate(gate)
    a=cal.predict(gate,*coarse_waves,circuits=(cs,cr))
    b=cal.predict(gate,*fine_waves,circuits=(cs,cr))
    summaries={}
    for phase,coarse,fine,circuit in zip(PHASES,a,b,(cs,cr)):
        net=np.abs(coarse["voltage_V"])-np.abs(coarse["device_voltage_V"])
        kcl=float(np.max(np.abs(np.abs(coarse["current_A"])-circuit.transistor_current(net))))
        if kcl>=2e-10:
            raise ValueError(f"{gate} V {phase}: voltage/current balance failed")
        cv,fv=np.abs(coarse["voltage_V"]),np.abs(fine["voltage_V"])
        aidx,bidx=int(cv.argmax()),int(fv.argmax())
        model_values = np.r_[np.interp(cv[:aidx+1],fv[:bidx+1],np.abs(fine["current_A"][:bidx+1])),
                             np.interp(cv[aidx+1:],fv[bidx:][::-1],np.abs(fine["current_A"][bidx:][::-1]))]
        iv=np.abs(coarse["current_A"])
        valid=(iv>1e-10)&(model_values>1e-10)
        error=np.log10(iv[valid]/model_values[valid])
        summaries[phase]={"max_KCL_residual_A":kcl,
                          "coarse_vs_fine_rmse_decade_above_0p1nA":float(np.sqrt(np.mean(error**2)))}
    assert np.all(np.diff(a[1]["soft_state"])<=1e-8)
    assert np.isclose(a[0]["final"].soft,a[1]["soft_state"][0],rtol=0.,atol=2e-9)
    return summaries


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    out=parser.parse_args(argv).output_dir
    out.mkdir(parents=True,exist_ok=True)
    raw,usable,first,last,circuits=_read_data()
    arrays={}
    gate_report,gate_model=_gate_cv(usable,first,last,circuits,arrays)
    within_report,within_model=_single_gate_cv(first,last,circuits,arrays)
    bare,bare_params=_bare_comparison(first,last,circuits,gate_model,arrays)
    prospective=_prospective_model(usable,circuits,gate_report["chosen"],gate_report["prospective_shape_choice"])
    gate_model.to_file(out/"endpoint_calibration.json")
    within_model.to_file(out/"1p1V_calibration.json")
    prospective.to_file(out/"new_gate_calibration.json")
    (out/"bare_1M_calibration.json").write_text(json.dumps(bare_params,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    grid={}
    for g in (1.,1.2):
        _write_prediction(out/f"prediction_{g:.1f}V.csv",g,prospective,
                          predictor.triangle(5.,.01),predictor.triangle(-1.5,.01))
        grid[str(g)]=_grid_audit(g,prospective,-1.5)
    source_files=[audit.ROOT/w[3] for w in audit.WORKBOOKS]
    source_files.extend(Path(__file__).parent/"data"/s for s in audit.TRANSISTOR_FILES)
    results={
        "calibration_scope":"The endpoint comparison withholds every 1.1 V current. The chronological 1.1 V comparison trains through cycle 35. The final 3-anchor prediction fits all available screened measurements, with no validation at 1.0/1.2 V.",
        "selection":"Fixed comparison weighting on transition and unclamped data; simplest candidate within 0.5% of the best development score. Only previous data used. New predictions frozen under +5/-1.5 V, 0.01 V nominal step.",
        "retained_sweeps":[{"topology":k[0],"gate_v":k[1],"phase":k[2],
                            "raw":len(raw[k]),"retained":len(v)} for k,v in usable.items()],
        "cross_gate":gate_report,
        "known_gate":within_report,
        "bare":bare,
        "prediction_grid_check":grid,
        "input_sha256":{v.name:hashlib.sha256(v.read_bytes()).hexdigest() for v in source_files},
    }
    (out/"final_validation.json").write_text(json.dumps(results,indent=2,ensure_ascii=False,
                                                        allow_nan=False)+"\n",encoding="utf-8")
    np.savez_compressed(out/"validated_curves.npz",**arrays)
    print("Waveform, calibration and validation package complete",flush=True)


if __name__=="__main__":
    main()
