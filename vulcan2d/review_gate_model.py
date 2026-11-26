"""Diagnose and test bounded changes to the two-anchor 1T1M gate predictor.

This experiment keeps the original calibration file intact. Candidate choice
uses only the 0.9 V and 1.3 V groups, with separate trailing validation blocks.
"""
from dataclasses import asdict, replace
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.optimize import least_squares

from . import cycle_quality as quality
from . import model_shared_device as model
from . import predict_gate as predictor
from . import validate_gate_prediction as original
from . import validate_shared_device as shared
from . import validate_transition_regions as audit

ENDPOINTS = (.9, 1.3)
PHASES = ("SET", "RESET")
BASELINE_ARTIFACT = audit.ROOT.parent / "output" / "vulcan_gate_prediction.zip"
METHODS = {"SET": ("reference", "branch_weighted", "skewed"),
           "RESET": ("reference", "branch_weighted", "transport", "distributed", "anchored")}


def _weighted_trace(rows, phase):
    """Give the early outgoing branch and return explicit leverage."""
    v = rows[0].voltage
    if any(not np.array_equal(r.voltage, v) for r in rows):
        raise ValueError("Incomplete voltage alignment")
    keep = np.array([r.masks["use_unclamped"] for r in rows])
    event = np.array([r.masks["use_transition"] for r in rows])
    outward = np.array([r.masks["branch_outward"] for r in rows])
    early_range = (.15, .65) if phase == "SET" else (.15, .45)
    early = (np.abs(v) >= early_range[0]) & (np.abs(v) <= early_range[1])
    masks = ((keep, .25), (event, .35), (keep & ~outward, .25),
             (keep & outward & early[None, :], .15))
    if not all(np.any(region) for region, _ in masks):
        raise ValueError("Calibration region has no valid data")
    weights = sum(w*region/region.sum() for region, w in masks)
    raw = np.abs([r.current for r in rows])
    y = np.zeros_like(raw)
    y[keep] = np.log10(raw[keep])
    totals = weights.sum(axis=0)
    use = totals > 0
    return v, use, (weights*y).sum(axis=0)[use] / totals[use], np.sqrt(totals[use])


def _parameterize(phase, method, device):
    if phase == "SET":
        fields = ("beta_set", "soft_exponent", "set_center_v", "set_width_v")
        start = [device.beta_set, device.soft_exponent, device.set_center_v, device.set_width_v]
        bounds = ([2, .6, .45, .008], [10, 2.4, 2.10, .30])
        if method == "skewed":
            fields += ("set_skew",)
            start.append(1.)
            bounds[0].append(.25)
            bounds[1].append(4.)
    else:
        transport = method in ("transport", "distributed")
        if transport:
            fields = ("beta_reset", "reset_off_power", "reset_on_power",
                      "reset_center_v", "reset_width_v")
            start = [min(device.beta_reset, 6.), 1., 1.35, device.reset_center_v, .09]
            bounds = ([0., .55, .7, .18, .025], [11., 3.2, 3., 1.60, .5])
        else:
            fields = ("beta_reset", "reset_soft_alpha", "reset_center_v", "reset_width_v")
            start = [device.beta_reset, device.reset_soft_alpha, device.reset_center_v,
                     max(device.reset_width_v, .027)]
            bounds = ([.3, .35, .15, .025], [12., 5., 1.60, .55])
        if method == "distributed":
            fields += ("reset_late_fraction", "reset_center_gap_v")
            start += [.30, .38]
            bounds[0].extend([.05, .12])
            bounds[1].extend([.85, .75])
    return fields, start, bounds


def _starting_state(gate, groups, circuits, calibration):
    k = ("1T1M", gate, "SET")
    written = model.simulate_sweep(groups[k][0].voltage, "SET", calibration.device,
                                   calibration.preparation(gate), circuits[k])
    return written["final"]


def _read_window(rows, circuit, outward, window):
    grid, drain = circuit.transistor
    values, indices = np.unique(drain, return_index=True)
    data = []
    for row in rows:
        voltage, current = np.abs(row.voltage), np.abs(row.current)
        use = (row.masks["use_unclamped"] & (row.masks["branch_outward"] == outward)
               & (voltage >= window[0]-1e-9) & (voltage <= window[1]+1e-9)
               & (current <= values[-1]))
        measured = current[use]
        local = voltage[use] - np.interp(measured, values, grid[indices])
        good = local > .001
        data.append(np.column_stack((local[good], np.log10(measured[good]))))
    return np.concatenate(data)


def _fit_reset_anchors(groups, circuits, initial_model):
    """Calibrate transport windows independently before fitting RESET survival."""
    anchors = []
    gates = initial_model.gates_v
    n_gate = len(gates)
    # These measured segments are readout proxies, not proof of pure states.
    for outward, window in ((True, (.18,.42)), (False, (.30,1.0))):
        bands = [_read_window(groups["1T1M",g,"RESET"], circuits["1T1M",g,"RESET"],
                              outward, window) for g in gates]
        if not all(len(b) for b in bands):
            raise ValueError("Empty RESET transport anchor")
        if outward:
            center = np.linspace(-8.,-5.8,n_gate).tolist() if n_gate>1 else [-6.]
            start, lo, hi = center+[1.35], [-15.]*n_gate+[.8], [-4.]*n_gate+[3.]
        else:
            center = np.linspace(-11.9,-11.,n_gate).tolist() if n_gate>1 else [-10.5]
            start, lo, hi = center+[1.1,5.], [-18.]*n_gate+[.55,0.], [-7.]*n_gate+[3.2,11.]

        def objective(p):
            beta = 0. if outward else p[n_gate+1]
            return np.concatenate([
                (p[i]+p[n_gate]*np.log10(v)+beta*np.sqrt(v)/np.log(10)-y)
                * np.sqrt(350/len(v))
                for i,(v,y) in enumerate(b.T for b in bands)
            ])

        fit = least_squares(objective, start, bounds=(lo,hi), loss="soft_l1", f_scale=.14)
        if not fit.success:
            raise RuntimeError(fit.message)
        anchors.append(fit.x)
    on, off = anchors
    dev = replace(initial_model.device, beta_reset=off[n_gate+1], reset_off_power=off[n_gate],
                  reset_on_power=on[n_gate])
    prep = tuple(replace(p, off_reset_a=10**off[i], soft_reset_a=10**on[i])
                 for i,p in enumerate(initial_model.preparations))
    base = predictor.GateCalibration(gates,dev,prep)
    data = {g:_weighted_trace(groups["1T1M",g,"RESET"],"RESET") for g in gates}
    init = {g:_starting_state(g,groups,circuits,base) for g in gates}

    def objective(p):
        updated = replace(dev,reset_center_v=p[0],reset_width_v=p[1])
        errors = []
        for g,(v,use,y,w) in data.items():
            current = model.simulate_sweep(v,"RESET",updated,base.preparation(g),
                                           circuits["1T1M",g,"RESET"],init[g])["current_A"]
            errors.append(w*(np.log10(np.abs(current[use]))-y))
        return np.concatenate(errors)

    result = least_squares(objective, [1.20,.10], bounds=([.20,.025],[1.60,.55]))
    if not result.success:
        raise RuntimeError(result.message)
    return replace(base,device=replace(dev,reset_center_v=result.x[0],reset_width_v=result.x[1]))


def fit_candidate(groups, circuits, initial_model, phase, method):
    if method == "reference":
        return initial_model
    if method == "anchored":
        return _fit_reset_anchors(groups,circuits,initial_model)
    fields, start, (low, high) = _parameterize(phase, method, initial_model.device)
    prep_fields = (("off_set_a", "soft_set_a") if phase == "SET"
                   else ("off_reset_a", "soft_reset_a"))
    local_low = [-13.5, -6.7] if phase == "SET" else [-15., -10.]
    local_high = [-7.5, -3.5] if phase == "SET" else [-8.3, -4.3]
    data, initial = {}, {}
    for g in ENDPOINTS:
        data[g] = _weighted_trace(groups["1T1M", g, phase], phase)
        initial[g] = (model.State() if phase == "SET" else _starting_state(
            g, groups, circuits, initial_model))
        prep = initial_model.preparation(g)
        amplitude = [np.log10(getattr(prep, f)) for f in prep_fields]
        if initial_model.device.reset_on_power is None and "reset_on_power" in fields:
            amplitude[1] += np.log10(initial_model.device.reset_soft_alpha)
        start.extend(np.clip(amplitude, np.array(local_low)+1e-4, np.array(local_high)-1e-4))
        low += local_low
        high += local_high

    def decode(p):
        device = replace(initial_model.device, **dict(zip(fields, p[:len(fields)])))
        preps = tuple(replace(prep, **dict(zip(prep_fields, 10 ** row)))
                      for prep, row in zip(initial_model.preparations, p[len(fields):].reshape(2,2)))
        return predictor.GateCalibration(ENDPOINTS, device, preps)

    def objective(p):
        cal = decode(p)
        residuals = []
        for g, (v, use, observations, w) in data.items():
            curve = model.simulate_sweep(v, phase, cal.device, cal.preparation(g),
                                         circuits["1T1M", g, phase], initial[g])
            wave = np.abs(curve["current_A"][use])
            residuals.append(w * (np.log10(wave) - observations))
        return np.concatenate(residuals)

    result = least_squares(objective, start, bounds=(low,high), x_scale="jac",
                           diff_step=5e-5, max_nfev=240, ftol=1e-7, xtol=1e-7, gtol=1e-7)
    if not result.success or not np.isfinite(result.cost):
        raise RuntimeError(f"{phase} {method}: {result.message}")
    return decode(result.x)


def _phase_predictions(rows, phase, gate, calibration, circuits, set_wave):
    cs, cr = circuits["1T1M", gate, "SET"], circuits["1T1M", gate, "RESET"]
    prep = calibration.preparation(gate)
    if phase == "SET":
        return [model.simulate_sweep(row.voltage, phase, calibration.device, prep, cs)
                for row in rows]
    end = model.simulate_sweep(set_wave, "SET", calibration.device, prep, cs)["final"]
    return [model.simulate_sweep(row.voltage, phase, calibration.device, prep, cr, end)
            for row in rows]


def _phase_scores(rows, predictions, phase, circuit):
    s = shared.scores(rows, predictions, phase, circuit)
    for name, boundary in (("return", "return"), ("outward", "outward")):
        values = []
        for row,pred in zip(rows,predictions):
            branch = row.masks["branch_outward"]
            use = row.masks["use_unclamped"] & (branch if boundary=="outward" else ~branch)
            values.append((np.abs(row.current[use]), np.abs(pred["current_A"][use])))
        s[name] = audit._summarize_residuals(values)
    return s


def _score(s):
    return {k: s[k]["pooled_rmse_decade"] for k in ("use_transition","eligible","outward","return")}


def _endpoint_selection(phase, training, validation, circuits, baseline):
    results = {}
    for method in METHODS[phase]:
        fit = fit_candidate(training,circuits,baseline,phase,method)
        per_gate = {}
        for g in ENDPOINTS:
            key = ("1T1M",g,phase)
            test_set = training["1T1M",g,"SET"][0].voltage
            fit_rows = validation[key]
            pred = _phase_predictions(fit_rows, phase, g, fit, circuits, test_set)
            per_gate[str(g)] = _phase_scores(fit_rows,pred,phase,circuits[key])
        results[method] = per_gate
    summaries = {}
    for method, scores in results.items():
        # Earlier iterations showed a small transition-only loss can hide a
        # large return error. Keep both in the endpoint model-selection score.
        summaries[method] = float(np.mean([
            .45*_score(s)["use_transition"] + .25*_score(s)["outward"] + .30*_score(s)["return"]
            for s in scores.values()]))
    selected = min(METHODS[phase],key=summaries.__getitem__)
    return selected, summaries, results


def _diagnose(rows, fit, circuits):
    r = rows["1T1M",1.1,"RESET"]
    circuit = circuits["1T1M",1.1,"RESET"]
    grid, drain = circuit.transistor
    levels, indices = np.unique(drain,return_index=True)
    v = np.abs(r[0].voltage)
    apex = np.argmax(v)
    raw = np.abs(np.array([row.current for row in r]))
    med = np.median(raw,axis=0)
    prep = fit.preparation(1.1)
    rows_out = []
    for value in (.40,.60,.80,1.,1.20,1.40,1.44,1.50,1.70):
        i = int(np.argmin(np.abs(v[:apex+1]-value)))
        observed = raw[:,i]
        physically_supported = (observed <= np.interp(v[i],grid,drain)) & np.isfinite(observed)
        observed = observed[physically_supported]
        vh = v[i] - np.interp(observed,levels,grid[indices])
        low = model.device_current(vh,0,0,fit.device,prep,"RESET")
        high = model.device_current(vh,1,0,fit.device,prep,"RESET")
        above = observed>np.maximum(low,high)
        below = observed<np.minimum(low,high)
        rows_out.append({"voltage_abs_V": float(v[i]),"measured_median_A": float(med[i]),
                         "model_HRS_limit_median_A": float(np.median(low)),
                         "model_LRS_limit_median_A": float(np.median(high)),
                         "above_envelopes": int(above.sum()),"below_envelopes": int(below.sum()),
                         "n_sweeps": len(observed)})
    return rows_out


def _persistent_crossing(voltage, current, threshold):
    v = np.abs(voltage)
    peak = int(np.argmax(v)) + 1
    v, current = v[:peak], np.abs(current[:peak])
    hits = np.flatnonzero((current[:-2]>=threshold) & (current[1:-1]>=threshold)
                          & (current[2:]>=threshold))
    if not len(hits):
        return np.nan
    i = int(hits[0])
    if i == 0 or not np.isfinite(current[i-1]) or current[i] == current[i-1]:
        return float(v[i])
    return float(v[i-1]+(v[i]-v[i-1])*(threshold-current[i-1])/(current[i]-current[i-1]))


def _set_widths(training, arrays):
    thresholds = (1e-6,5e-6,10e-6,15e-6)

    def describe(voltages,currents):
        crosses = np.array([[_persistent_crossing(v,i,t) for t in thresholds]
                            for v,i in zip(voltages,currents)])
        return {"n_curves":len(crosses),
                "counts":[int(x) for x in np.isfinite(crosses).sum(axis=0)],
                "crossing_median_V":np.nanmedian(crosses,axis=0).tolist(),
                "width_median_V":float(np.nanmedian(crosses[:,-1] - crosses[:,0]))}

    result={"thresholds_A":thresholds}
    for g in ENDPOINTS:
        rows=training["1T1M",g,"SET"]
        result[str(g)+"_train"]=describe([r.voltage for r in rows],[r.current for r in rows])
    v=arrays["SET_reference_voltage"][32:]
    result["1.1_measured"]=describe(v,arrays["SET_reference_measured"][32:])
    for k in METHODS["SET"]:
        result[k+"_forecast"]=describe(v,arrays["SET_"+k+"_current_A"][32:])
    return result


def _reset_stress_reference(groups, circuits, cal):
    gs, gr = ("1T1M",1.1,"SET"), ("1T1M",1.1,"RESET")
    set_result, reset_result = cal.predict(
        1.1,groups[gs][0].voltage,groups[gr][0].voltage,circuits=(circuits[gs],circuits[gr]))
    v = np.abs(reset_result["voltage_V"])
    peak = np.argmax(v)+1
    state = set_result["final"]
    _, frozen = circuits[gr].operating_point(v[:peak], lambda x: model.device_current(
        x,state.soft,state.hard,cal.device,cal.preparation(1.1),"RESET"))
    sample = int(np.argmin(np.abs(v[:peak]-1.4)))
    actual = np.abs(reset_result["device_voltage_V"][:peak])
    return {"at_applied_abs_V":float(v[sample]),"state_frozen_stress_V":float(frozen[sample]),
            "actual_model_internal_V":float(actual[sample]),
            "largest_difference_V":float(np.max(np.abs(actual-frozen)))}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True,type=Path)
    args = p.parse_args(argv)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with ZipFile(BASELINE_ARTIFACT) as zip_file:
        old = json.loads(zip_file.read("gate_calibration.json"))
    baseline = predictor.GateCalibration(tuple(old["gates_v"]), model.Device(**old["device"]),
                                         tuple(model.Preparation(**p) for p in old["preparations"]))
    raw, circuits = shared.load_groups(include_abnormal_pairs=True)
    ep = {k:r for k,r in raw.items() if k[1] in ENDPOINTS}
    flags = quality.screen_groups(ep)
    training = {k:quality.split_with_screen(r,flags[k],stop=quality.TRAINING_BOUNDARY[k[1]])
                for k,r in ep.items()}
    validation = {k:quality.split_with_screen(r,flags[k],start=quality.TRAINING_BOUNDARY[k[1]])
                  for k,r in ep.items()}
    development = {k:training[k]+validation[k] for k in ep}
    train_reference, _ = original.fit_endpoints(training,{k:circuits[k] for k in ep})
    selected, criteria, detail = {}, {}, {}
    for phase in PHASES:
        selected[phase], criteria[phase], detail[phase] = _endpoint_selection(
            phase,training,validation,circuits,train_reference)
        print(phase,selected[phase],criteria[phase],flush=True)

    # Assess an endpoint-only phase-wise choice on all retained 1.1 V curves.
    # Keep every ablation in the report to avoid treating that gate as a new
    # prospective holdout when the modeling family was developed on it.
    target = {k:r for k,r in raw.items() if k[1] == 1.1}
    tf = quality.screen_groups(target)
    retained = {k:quality.split_with_screen(r,tf[k]) for k,r in target.items()}
    scores, arrays = {}, {}
    tested_calibrations = {}
    for phase in PHASES:
        k=("1T1M",1.1,phase)
        rows=retained[k]
        start=retained["1T1M",1.1,"SET"][0].voltage
        scores[phase]={}
        tested_calibrations[phase]={}
        for method in METHODS[phase]:
            fit=fit_candidate(development,circuits,baseline,phase,method)
            tested_calibrations[phase][method]=asdict(fit)
            pred=_phase_predictions(rows,phase,1.1,fit,circuits,start)
            scores[phase][method]={
                "all":_phase_scores(rows,pred,phase,circuits[k]),
                "36_to_53":_phase_scores(rows[32:],pred[32:],phase,circuits[k])}
            shared._record_curves(arrays,f"{phase}_{method}",rows,pred)
    result={
        "contract": "All candidate parameters, region weights and selection use endpoint-gate sweeps only. Changes are experimental; the prior blind calibration is not overwritten.",
        "data_split": {f"{key[1]:.1f}_{key[2]}":{"training":len(training[key]),"validation":len(validation[key])}
                       for key in ep},
        "selection": selected,
        "endpoint_ranking":criteria,
        "endpoint_validation":detail,
        "diagnostic":_diagnose({k:r[32:] for k,r in retained.items()},baseline,circuits),
        "SET_width_diagnostic":_set_widths(training,arrays),
        "RESET_stress_reference":_reset_stress_reference(retained,circuits,baseline),
        "reused_baseline_sha256":hashlib.sha256(BASELINE_ARTIFACT.read_bytes()).hexdigest(),
        "holdout_1p1":scores,
        "experimental_calibrations":tested_calibrations,
    }
    (args.output_dir/"model_review.json").write_text(
        json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    np.savez_compressed(args.output_dir/"candidate_curves.npz",**arrays)


if __name__ == "__main__":
    main()
