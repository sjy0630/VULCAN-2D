"""Test the log-conductance state in paired SET and RESET predictions."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit

from analysis import audit_log_soft as readout
from vulcan2d import model_shared_device as model
from vulcan2d import validate_shared_device as shared


FEATURES = ("peak_current", "current_path_integral", "power_path_integral")


def data():
    groups, circuits = shared.load_groups()
    paired = {}
    for gate in (.9, 1.1, 1.3):
        sk, rk = [("1T1M", gate, phase) for phase in ("SET", "RESET")]
        sets, resets = [readout.numbered(groups[key]) for key in (sk, rk)]
        if sets.keys() != resets.keys():
            raise ValueError(f"SET and RESET labels differ at {gate:g} V")
        paired[gate] = [(sets[i], resets[i], circuits[sk], circuits[rk])
                        for i in sorted(sets)]
    return paired


def set_features(set_row, circuit, device, prep):
    trace = model.simulate_sweep(set_row.voltage, "SET", device, prep, circuit)
    end = np.argmax(np.abs(set_row.voltage)) + 1
    current = np.abs(trace["current_A"][:end])
    local = np.abs(trace["device_voltage_V"][:end])
    applied = np.abs(set_row.voltage[:end])
    values = [np.max(current), np.trapezoid(current, applied),
              np.trapezoid(current * local, applied)]
    return np.log10(values), trace


def observed_state(reset, circuit):
    ids, v, i = readout.intrinsic(reset, circuit, outward=True)
    ref = readout.reference_current(ids, v, i)
    return None if ref is None else np.log10(ref / readout.REFERENCE_V / 1e-6)


def current_law(local, state, anchors, shapes):
    """Interpolate log conductance and use an ohmic continuation outside the voltage grid."""
    local, state = np.broadcast_arrays(local, state)
    grid = readout.V_GRID
    right = np.clip(np.searchsorted(grid, local), 1, len(grid)-1)
    left = right-1
    v = np.clip(local, grid[0], grid[-1])
    ratio = (v-grid[left])/(grid[right]-grid[left])
    a = np.clip(np.searchsorted(anchors, state), 1, len(anchors)-1)
    mix = np.clip((state-anchors[a-1])/(anchors[a]-anchors[a-1]), 0., 1.)
    low = (1-mix)*shapes[a-1,left]+mix*shapes[a,left]
    high = (1-mix)*shapes[a-1,right]+mix*shapes[a,right]
    log_i = state + low + ratio*(high-low)
    return readout.REFERENCE_V * 1e-6 * 10**log_i * local/v


def stress_and_apex(reset, circuit, state, anchors, shapes):
    applied = np.abs(reset.voltage)
    apex = int(applied.argmax())
    _, stress = circuit.operating_point(
        applied[:apex+1], lambda v: current_law(v, state, anchors, shapes))
    return stress, apex


def erase(state, stress, apex, length, parameters):
    floor, center, width = parameters
    start = expit(-center/width)
    progress = (expit((stress-center)/width)-start)/(1-start)
    progress = np.maximum.accumulate(progress)
    path = state + (min(floor, state)-state)*progress
    return np.r_[path, np.full(length-apex-1, path[-1])]


def fit_erase(rows, anchors, shapes):
    observations = []
    for _, reset, _, circuit in rows:
        state = observed_state(reset, circuit)
        if state is None:
            continue
        stress, apex = stress_and_apex(reset, circuit, state, anchors, shapes)
        points = [readout.intrinsic(reset,circuit,outward=flag) for flag in (True,False)]
        ids = np.concatenate([p[0] for p in points])
        _, v, i = points[0]
        if len(ids):
            observations.append((state, stress, apex, len(v), ids, v[ids], np.log10(i[ids])))
    if not observations:
        raise ValueError("No valid training RESET records")
    def residual(parameters):
        return np.concatenate([
            (np.log10(current_law(v, erase(state, stress, apex, n, parameters)[ids],
                                  anchors, shapes))-target)/np.sqrt(len(ids))
            for state,stress,apex,n,ids,v,target in observations])
    result = least_squares(residual, [-3.,.9,.22],
                           bounds=([-7.,.15,.02],[-1.,2.5,1.2]),
                           diff_step=1e-4, ftol=1e-5, max_nfev=1200)
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def log_errors(row, prediction, mask):
    return np.log10(np.abs(prediction[mask]/row.current[mask]))


def reset_predict(reset, circuit, initial, parameters, anchors, shapes):
    stress, apex = stress_and_apex(reset,circuit,initial,anchors,shapes)
    state = erase(initial,stress,apex,len(reset.voltage),parameters)
    return circuit.operating_point(
        np.abs(reset.voltage),lambda v: current_law(v,state,anchors,shapes))[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path)
    args = parser.parse_args()
    pairs = data()
    train = pairs[.9][:70]+pairs[1.1][:32]
    test = {.9:pairs[.9][70:], 1.1:pairs[1.1][32:], 1.3:pairs[1.3]}
    anchors, shapes = readout.learn_anchors([readout.readout_record(r,rc) for _,r,_,rc in train])
    states = np.linspace(anchors[0]-2,anchors[-1]+6,40)
    assert np.allclose(current_law(readout.REFERENCE_V,states,anchors,shapes)/readout.REFERENCE_V,
                       1e-6*10**states)
    assert np.all(np.diff(current_law(np.linspace(0,2.5,251)[None,:],
                                      states[:,None],anchors,shapes),axis=1)>=0)
    nominal = pairs[1.1][:32]
    device, prep = shared.fit_soft_set([s for s,_,_,_ in nominal],nominal[0][2])
    set_end = model.simulate_sweep(nominal[0][0].voltage,"SET",device,prep,nominal[0][2])["final"]
    device, prep = shared.fit_soft_reset([r for _,r,_,_ in nominal],nominal[0][3],device,prep,set_end)
    x,y = [],[]
    for s,r,sc,rc in train:
        measured = observed_state(r,rc)
        if measured is not None:
            values,_ = set_features(s,sc,device,prep)
            x.append(values)
            y.append(measured)
    coefficients = np.array([np.polyfit(column,y,1) for column in np.asarray(x).T])
    parameters = fit_erase(train,anchors,shapes)
    report = {}
    for gate, rows in test.items():
        errors = {key:[] for key in ("SET_original","RESET_original","RESET_measured_entry",*FEATURES)}
        entry = {key:[] for key in FEATURES}
        predicted_states = {key:[] for key in FEATURES}
        measured_states = []
        matched = {key:[] for key in ("original","measured_entry",*FEATURES)}
        for s,r,sc,rc in rows:
            features, written = set_features(s,sc,device,prep)
            mask = r.masks["use_unclamped"]
            mask_late = mask & (np.abs(r.voltage)>=.3)
            errors["SET_original"].extend(log_errors(s,written["current_A"],s.masks["use_unclamped"]))
            erased = model.simulate_sweep(r.voltage,"RESET",device,prep,rc,written["final"])
            errors["RESET_original"].extend(log_errors(r,erased["current_A"],mask))
            observed = observed_state(r,rc)
            if observed is not None:
                measured_states.append(observed)
                pred = reset_predict(r,rc,observed,parameters,anchors,shapes)
                matched["measured_entry"].extend(log_errors(r,pred,mask_late))
                matched["original"].extend(log_errors(r,erased["current_A"],mask_late))
            for feature,value,(slope,offset) in zip(FEATURES,features,coefficients):
                state = slope*value+offset
                predicted_states[feature].append(state)
                pred = reset_predict(r,rc,state,parameters,anchors,shapes)
                errors[feature].extend(log_errors(r,pred,mask))
                if observed is not None:
                    entry[feature].append(state-observed)
                    matched[feature].extend(log_errors(r,pred,mask_late))
        report[str(gate)] = {
            "cycles":len(rows),"cycles_with_entry":len(measured_states),
            "entry_observed_median_uS":float(10**np.median(measured_states)),
            "entry_predicted_median_uS":{k:float(10**np.median(v)) for k,v in predicted_states.items()},
            "entry_rmse_decade":{k:readout.rmse(v) for k,v in entry.items()},
            "out_of_anchor_range_predicted_cycle_counts": {
                k:int(np.count_nonzero((np.asarray(v)<anchors[0])|(np.asarray(v)>anchors[-1])))
                for k,v in predicted_states.items()},
            "whole_branch_rmse_decade":{k:readout.rmse(v) for k,v in errors.items() if v},
            "same_late_reset_points_rmse_decade":{k:readout.rmse(v) for k,v in matched.items()},
        }
    output = {
        "protocol":"Train nominal SET on cycles 4-35 at 1.1 V; train shared readout and writer/eraser on 0.9 V first 70 and 1.1 V cycles 4-35. Freeze all parameters, hold out 1.3 V and later blocks. No measured currents enter unconditioned predictions. RESET state is inferred for the intervention only; its scores use |Vapp| >= 0.3 V on matching traces. Same-numbered SET and RESET paired, no cross-cycle carryover. Path integrals are taken over the outward applied-voltage ramp and are not charge or energy.",
        "anchors":anchors.tolist(),
        "writer_slope_offset":dict(zip(FEATURES,coefficients.tolist())),
        "reset_floor_center_width":parameters.tolist(),"results":report}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    print(json.dumps(output,indent=2))


if __name__ == "__main__":
    main()
