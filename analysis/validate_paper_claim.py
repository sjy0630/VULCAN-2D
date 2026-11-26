"""Audit typical IV41 loops and freeze a write-gate intervention forecast."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

from analysis import refit_iv41 as refit
from analysis import validate_programming_bridge as bridge
from analysis.fit_reset_population import load_data
from vulcan2d import gate_only
from vulcan2d import model_shared_device as model
from vulcan2d import predict_gate
from vulcan2d import reset_population
from vulcan2d import validate_transition_regions as audit

PHASES = ("SET", "RESET")
READ_V = .15
READ_WINDOW = (.10, .20)
NEW_GATES = (1.05, 1.25)
WRITE_GATES = (.95, 1.15, 1.35)
CONTROL_GATE = 1.10
PLAN_SEED = 412024


def phase_median(rows):
    """Raw pointwise median on positions supported by at least half the sweeps."""
    y = np.stack([np.where(r.masks["use_unclamped"], refit.log_i(r.current), np.nan)
                  for r in rows])
    support = np.sum(np.isfinite(y), axis=0) >= np.ceil(len(rows)/2)
    typical = np.full(y.shape[1], np.nan)
    typical[support] = np.nanmedian(y[:, support], axis=0)
    return typical, support


def event(wave, current, phase, use):
    v = abs(wave)
    peak = int(v.argmax())
    valid = use & (v>0.) & np.isfinite(current) & (np.abs(current)>=audit.CURRENT_FLOOR_A[phase])
    log_g = np.full(len(v), np.nan)
    logs = np.full(len(v), np.nan)
    logs[valid] = refit.log_i(current[valid])
    preview = audit.median_preview(logs[:peak+1])
    log_g[:peak+1] = preview - np.log10(np.maximum(v[:peak+1],1e-18))
    found = audit.locate_event(v, log_g, peak, phase)
    return float(v[found[0]]) if found else None


def reads(wave, current, use, fixed=False):
    v = abs(wave)
    out = np.arange(len(v)) <= int(v.argmax())
    band = ((v>=READ_WINDOW[0]-1e-9)&(v<=READ_WINDOW[1]+1e-9)
            if not fixed else np.isclose(v,READ_V,atol=1e-9,rtol=0))
    available = use & band & (v>0.) & (abs(current)>0.)
    result = {}
    for flag,key in ((out,"log10_lrs_s"),(~out,"log10_hrs_s")):
        mask = available & flag
        result[key] = (float(np.median(refit.log_i(current[mask])-np.log10(v[mask])))
                       if mask.sum() >= (1 if fixed else 3) else None)
    return result


def measures(wave, current, phase, valid, fixed_read=False):
    flip = {f"{phase.lower()}_event_v":event(wave,current,phase,valid)}
    if phase == "RESET":
        flip.update(reads(wave,current,valid,fixed_read))
    return flip


def observed_measures(rows, phase):
    items = [measures(row.voltage,row.current,phase,row.masks["use_unclamped"]) for row in rows]
    return {key:[r[key] for r in items if r[key] is not None] for key in items[0]}


def summarize(values, n):
    return {"detected":len(values),"total":n,
            "q10_q50_q90":np.quantile(values,[.1,.5,.9]).tolist() if len(values) else None}


def compare_point_measures(expected, predicted):
    result={}
    for key,estimate in predicted.items():
        observed=expected[key]["q10_q50_q90"]
        result[key] = {"value":estimate,
                       "abs_error":abs(estimate-observed[1]) if estimate is not None and observed is not None else None}
    return result


def distribution_results(expected, simulated, n):
    return {key:{**summarize(values,n),
                 "wasserstein":float(wasserstein_distance(expected[key],values))
                 if len(values) and len(expected[key]) else None}
            for key,values in simulated.items()}


def ensemble(cal, set_gate, reset_gate, waves):
    d = model.Device(**cal["device"])
    p = model.Preparation(**cal["preparation"])
    cs,_ = refit.circuits(set_gate)
    _,cr = refit.circuits(reset_gate)
    programmed = model.simulate_sweep(waves["SET"],"SET",d,p,cs)
    peak = float(np.max(np.abs(programmed["current_A"])))
    amplitudes = reset_population.prepared_distribution(cal["program_distribution"],peak)
    draw = reset_population.simulate_ensemble(waves["RESET"],d,p,cr,programmed["final"],amplitudes)
    return draw,peak


def historical(data, gate_cfg):
    results={}
    for gate in bridge.ASSESS_GATES:
        prediction = gate_only.predict(gate_cfg,gate)
        waves = {phase:prediction[phase][0] for phase in PHASES}
        pair = refit.circuits(gate)
        physical = gate_only.original(gate_cfg,gate,waves,pair)
        phase_results={}
        for i,phase in enumerate(PHASES):
            rows=data[gate][phase][1]
            median,mask=phase_median(rows)
            v,new=prediction[phase]
            if not np.array_equal(v,rows[0].voltage):
                raise ValueError("Scoring needs the frozen waveform grid")
            apex=int(np.argmax(abs(v)))
            out=np.arange(len(v))<=apex
            anchors={float(k):a[phase] for k,a in sorted(
                gate_cfg["anchors"].items(),key=lambda item:float(item[0]))}
            empirical=gate_only.current_for_method(v,pair[i],physical[phase],anchors,gate,"terminal_mean",0.)
            curves={"gate_only":new,"physical_prior":np.sign(v)*10**physical[phase],
                    "terminal_interpolation":empirical}
            actual=observed_measures(rows,phase)
            observed={k:summarize(y,len(rows)) for k,y in actual.items()}
            scored={}
            for name,current in curves.items():
                squared=(refit.log_i(current)-median)**2
                central={k:float(np.sqrt(np.mean(squared[region]))) if region.any() else None
                         for k,region in (("whole",mask),("outward",mask&out),("return",mask&~out))}
                scored[name]={"median_curve_rmse_decade":central,
                              "key_metrics":compare_point_measures(observed,measures(v,current,phase,mask))}
            entry={"scans":len(rows),"supported_points":int(mask.sum()),
                   "observed":observed,"models":scored}
            if phase=="RESET":
                draws,_=ensemble(gate_cfg["physical_prior"],gate,gate,waves)
                keys=[measures(v,d,phase,mask) for d in draws]
                samples={k:[d[k] for d in keys if d[k] is not None] for k in keys[0]}
                entry["physical_population"]=distribution_results(actual,samples,len(draws))
            phase_results[phase]=entry
        results[gate]=phase_results
    return results


def planned_runs(path):
    rng=np.random.default_rng(PLAN_SEED)
    fields=("cycle_id","experiment","block","gate_set_V","gate_reset_V","set_peak_V",
            "reset_peak_V","step_V","read_voltage_V","device_id","physical_order","status")
    with path.open("w",newline="") as handle:
        writer=csv.writer(handle)
        writer.writerow(fields)
        for exp,gates in (("typical",NEW_GATES),("control",WRITE_GATES)):
            for block in range(1,31):
                order=rng.permutation(gates)
                for j,g in enumerate(order,start=1):
                    writer.writerow((f"{exp}-{block:02}-{j}",exp,block,g,
                                     g if exp=="typical" else CONTROL_GATE,
                                     5.,-1.5,.01,-READ_V,"","",""))


def freeze(cfg, output):
    waves={k:np.asarray(cfg["waveforms"][k]) for k in PHASES}
    cal=cfg["physical_prior"]
    destination=output/"frozen_predictions.csv"
    rows={}
    with destination.open("w",newline="") as handle:
        writer=csv.writer(handle)
        writer.writerow(("experiment","method","gate_set_V","gate_reset_V","phase","branch",
                         "voltage_V","typical_current_A","p10_abs_current_A","p90_abs_current_A"))
        for g in NEW_GATES:
            pair = refit.circuits(g)
            original = gate_only.original(cfg,g,waves,pair)
            for i,(phase,(v,current)) in enumerate(gate_only.predict(cfg,g).items()):
                a=int(abs(v).argmax())
                anchors={float(k):value[phase] for k,value in sorted(
                    cfg["anchors"].items(),key=lambda item:float(item[0]))}
                baseline=gate_only.current_for_method(
                    v,pair[i],original[phase],anchors,g,"terminal_mean",0.)
                for name,values in (("gate_only",current),("terminal_interpolation",baseline)):
                    for j,(voltage,intensity) in enumerate(zip(v,values)):
                        writer.writerow(("typical",name,g,g,phase,"outward" if j<=a else "return",
                                         voltage,intensity,"",""))
        # The null retains the programmed state at the reference SET gate.
        # Under a fixed RESET gate all null readouts and RESET sweeps coincide.
        fixed,_=ensemble(cal,1.15,CONTROL_GATE,waves)
        v=waves["RESET"]
        a=int(abs(v).argmax())
        for g in WRITE_GATES:
            drawn,current=ensemble(cal,g,CONTROL_GATE,waves)
            record={"predicted_set_peak_current_a":current,"models":{}}
            for name,draws in (("shared_state",drawn),("fixed_state_load_line",fixed)):
                q=np.quantile(draws,[.1,.5,.9],axis=0)
                used=np.ones(len(v),dtype=bool)
                items=[measures(v,draw,"RESET",used,fixed_read=True) for draw in draws]
                record["models"][name] = {k:summarize([m[k] for m in items if m[k] is not None],len(items))
                                           for k in items[0]}
                for j,val in enumerate(v):
                    writer.writerow(("control",name,g,CONTROL_GATE,"RESET",
                                     "outward" if j<=a else "return",val,
                                     np.sign(val)*q[1,j],q[0,j],q[2,j]))
            rows[g]=record
    planned_runs(output/"measurement_schedule.csv")
    return rows


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook",type=Path)
    parser.add_argument("gate_only",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--cache",type=Path,required=True)
    args=parser.parse_args(argv)
    args.output.mkdir(parents=True,exist_ok=True)
    original=refit.origin_cache(args.workbook,args.cache)
    cfg=json.loads(args.gate_only.read_text())
    evidence=historical(load_data(original,args.cache),cfg)
    control=freeze(cfg,args.output)
    summary={
        "historical_gate_audit":evidence,
        "frozen_experiment":{"new_gates_v":NEW_GATES,"write_gates_v":WRITE_GATES,
                             "control_reset_gate_v":CONTROL_GATE,"terminal_read_v":-READ_V,
                             "control":control,"planned_scans_each_condition":30},
        "sha256":{
            "workbook":refit.sha_file(args.workbook),
            "gate_calibration":refit.sha_file(args.gate_only),
            "forecast":refit.sha_file(args.output/"frozen_predictions.csv"),
            "schedule":refit.sha_file(args.output/"measurement_schedule.csv"),
            "lookup":{name:refit.sha_file(audit.ROOT/"vulcan2d/data"/name)
                      for name in audit.TRANSISTOR_FILES},
        }
    }
    (args.output/"paper_evidence.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    for gate,phases in evidence.items():
        print(gate,[(phase,{k:round(d["median_curve_rmse_decade"]["whole"],3)
                             for k,d in case["models"].items()})
                    for phase,case in phases.items()],flush=True)
    for gate,case in control.items():
        print("Frozen",gate,case["models"]["shared_state"],flush=True)


if __name__=="__main__":
    main()
