"""Add high-conductance readout curves from early RESET and test on held-out cycles."""
import argparse
import json
from pathlib import Path

import numpy as np

from analysis import audit_log_soft as readout
from analysis import validate_log_soft_cycle as cycle
from vulcan2d import model_shared_device as model
from vulcan2d import validate_shared_device as shared


MIN_SUPPORT = 4
TRANSITION_MARGIN_V = .10


def early_record(row, circuit):
    ids, local, current = readout.intrinsic(row, circuit, outward=True)
    event = row.event.get("event_abs_voltage_V")
    if event is None:
        # A missing event gives no basis for extending the presumed fixed-state region.
        ids = ids[local[ids] <= .25]
    else:
        ids = ids[np.abs(row.voltage[ids]) <= event-TRANSITION_MARGIN_V]
    ref = readout.reference_current(ids, local, current)
    if ref is None:
        return None
    order = ids[np.argsort(local[ids])]
    voltage, unique = np.unique(local[order], return_index=True)
    if len(voltage) < 2:
        return None
    shape = np.interp(readout.V_GRID, voltage, np.log10(current[order[unique]]/ref),
                      left=np.nan, right=np.nan)
    return float(np.log10(ref/readout.REFERENCE_V/1e-6)), shape


def add_entry_anchors(records, anchors, shapes):
    grid = readout.V_GRID
    ref_index = np.flatnonzero(grid == readout.REFERENCE_V)[0]
    output = []
    for lo, hi in zip(range(-2, 4), range(-1, 5)):
        rows = [(s, shape) for s, shape in records if max(lo, anchors[-1]) < s <= hi]
        if len(rows) < MIN_SUPPORT:
            continue
        states, samples = zip(*rows)
        samples = np.asarray(samples)
        support = np.sum(np.isfinite(samples), axis=0)
        valid = np.flatnonzero(support >= MIN_SUPPORT)
        # Require actual observations around the reference and above it.
        if ref_index not in valid or not np.any(grid[valid] >= .20):
            continue
        end = next((i-1 for i in range(ref_index, len(grid)) if support[i]<MIN_SUPPORT), len(grid)-1)
        start = next((i+1 for i in range(ref_index, -1, -1) if support[i]<MIN_SUPPORT), 0)
        used = np.arange(start,end+1)
        if grid[end] < .20:
            continue
        sample_at_edge = [s for s, shape in rows if np.isfinite(shape[end])]
        state = float(np.median(sample_at_edge))
        median = np.array([np.nanmedian(samples[:,i]) for i in used])
        candidate = shapes[-1].copy()
        candidate[used] = median
        candidate[:start] += median[0]-shapes[-1,start]
        candidate[end+1:] += median[-1]-shapes[-1,end]
        candidate = np.maximum.accumulate(candidate)
        candidate -= candidate[ref_index]
        output.append((state,candidate,{"state":state,"conductance_uS":10**state,
                                         "rows_in_bin":len(rows),"rows_at_last_point":int(support[end]),
                                         "max_measured_local_V":float(grid[end]),
                                         "min_measured_local_V":float(grid[start])}))
    if not output:
        raise ValueError("No new early RESET anchors satisfy coverage")
    return (np.r_[anchors,[r[0] for r in output]], np.vstack([shapes,*[r[1] for r in output]]),
            [r[2] for r in output])


def evaluate(rows, low, expanded, original_erase, expanded_erase, device, prep, writer):
    predicted = {key:[] for key in ("original_compact","low_only","added_anchors_fixed_erase",
                                     "added_anchors_refit_erase")}
    early = {"low_only":[],"added_anchors":[]}
    early_in_range, inputs = 0, 0
    measured_states, early_cycles = [], 0
    for s,r,sc,rc in rows:
        x, written = cycle.set_features(s,sc,device,prep)
        state = float(writer[0]*x[1]+writer[1])
        inputs += int(expanded[0][0] <= state <= expanded[0][-1])
        mask = r.masks["use_unclamped"]
        base = model.simulate_sweep(r.voltage,"RESET",device,prep,rc,written["final"])
        predicted["original_compact"].extend(cycle.log_errors(r,base["current_A"],mask))
        for name, family, eraser in (("low_only",low,original_erase),
                                    ("added_anchors_fixed_erase",expanded,original_erase),
                                    ("added_anchors_refit_erase",expanded,expanded_erase)):
            value = cycle.reset_predict(r,rc,state,eraser,*family)
            predicted[name].extend(cycle.log_errors(r,value,mask))
        record = early_record(r,rc)
        if record is None:
            continue
        measured, _ = record
        ids, local, observed = readout.intrinsic(r,rc,outward=True)
        event = r.event.get("event_abs_voltage_V")
        before = np.abs(r.voltage[ids])<=event-TRANSITION_MARGIN_V if event is not None else local[ids]<=.25
        ids = ids[before & (local[ids]>=.20) & (local[ids]<=.35)]
        if not len(ids):
            continue
        early_cycles += 1
        measured_states.append(measured)
        early_in_range += int(expanded[0][0] <= measured <= expanded[0][-1])
        for name, family in (("low_only",low),("added_anchors",expanded)):
            current, _ = rc.operating_point(
                np.abs(r.voltage[ids]),lambda v:cycle.current_law(v,measured,*family))
            early[name].extend(np.log10(current/observed[ids]))
    return {"test_cycles":len(rows),"writer_states_in_new_anchor_range":inputs,
            "whole_reset_rmse_decade":{k:readout.rmse(v) for k,v in predicted.items()},
            "early_readout_given_measured_entry":{
                "cycles":early_cycles,"points":len(early["low_only"]),
                "cycles_in_expanded_anchor_range":early_in_range,
                "observed_state_median_uS":float(10**np.median(measured_states)) if early_cycles else None,
                "rmse_decade":{k:readout.rmse(v) for k,v in early.items()}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    pairs = cycle.data()
    nominal = pairs[1.1][:32]
    train = pairs[.9][:70] + nominal
    device, prep = shared.fit_soft_set([s for s,_,_,_ in nominal],nominal[0][2])
    start = model.simulate_sweep(nominal[0][0].voltage,"SET",device,prep,nominal[0][2])["final"]
    device, prep = shared.fit_soft_reset([r for _,r,_,_ in nominal],nominal[0][3],device,prep,start)
    x,y = [],[]
    for s,r,sc,rc in train:
        state = cycle.observed_state(r,rc)
        if state is not None:
            x.append(cycle.set_features(s,sc,device,prep)[0][1])
            y.append(state)
    writer = np.polyfit(x,y,1)
    low = readout.learn_anchors([readout.readout_record(r,rc) for _,r,_,rc in train])
    results = {}
    for key, training, test in (
        ("unseen_1p3",train,pairs[1.3]),
        ("seen_1p3_later_cycles",train+pairs[1.3][:60],pairs[1.3][60:]),
    ):
        records = [record for _,r,_,rc in training if (record:=early_record(r,rc)) is not None]
        anchors, shapes, coverage = add_entry_anchors(records,*low)
        states = np.r_[anchors, (anchors[:-1]+anchors[1:])/2]
        assert np.allclose(cycle.current_law(readout.REFERENCE_V,states,anchors,shapes),
                           readout.REFERENCE_V*1e-6*10**states,rtol=1e-12,atol=0.)
        np.testing.assert_array_less(-1e-16, np.diff(cycle.current_law(
            np.linspace(0,2.5,251)[None,:],states[:,None],anchors,shapes),axis=1))
        original = cycle.fit_erase(training,*low)
        expanded = cycle.fit_erase(training,anchors,shapes)
        results[key] = {"added_anchors":coverage,"early_train_records":len(records),
                        "original_erase":original.tolist(),"expanded_erase":expanded.tolist(),
                        "results":evaluate(test,low,(anchors,shapes),original,expanded,device,prep,writer)}
    report = {"protocol":"Early RESET points with de-embedded local voltage, ending 0.10 V before the detected terminal-voltage transition. Missing transition uses at most 0.25 V local. Low-current return anchors and the SET writer were built from 0.9 V and 1.1 V training only. The unseen test keeps all 1.3 V data out. The seen test permits the first 60 high-gate cycles for early anchors and eraser training and scores only the last 41. For a given higher state, voltages outside directly supported points inherit the neighboring low-state voltage dependence. Above the largest state anchor only amplitude extends. Early readout is conditioned on one measured low-voltage point; complete RESET forecasts use only predicted SET state.",
              "low_anchor_range_uS":(10**low[0][[0,-1]]).tolist(),"splits":results}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
