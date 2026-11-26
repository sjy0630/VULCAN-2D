"""Validate a common logarithmic readout state and a separate SET-to-state rule."""
import argparse
import json
from pathlib import Path
import re

import numpy as np

from vulcan2d import model_shared_device as model
from vulcan2d import validate_shared_device as shared


REFERENCE_V = 0.15
V_GRID = np.round(np.arange(.075, .6251, .025), 3)
MIN_ANCHOR_SAMPLES = 8


def numbered(rows):
    items = {int(re.search(r"\d+$", row.label).group()): row for row in rows}
    if len(items) != len(rows):
        raise ValueError("Duplicate cycle numbers")
    return items


def intrinsic(row, circuit, outward):
    applied, current = np.abs(row.voltage), np.abs(row.current)
    vt, it = circuit.transistor
    levels, first = np.unique(it, return_index=True)
    local = applied - np.interp(current, levels, vt[first])
    keep = (row.masks["use_unclamped"] & (row.masks["branch_outward"] == outward)
            & (current > 0) & (current < .95*np.interp(applied, vt, it)) & (local > 0))
    return np.flatnonzero(keep), local, current


def reference_current(ids, volts, currents):
    for left, right in zip(ids[:-1], ids[1:]):
        if right != left + 1:
            continue
        a, b = sorted((left, right), key=lambda i: volts[i])
        if volts[a] <= REFERENCE_V <= volts[b] and volts[a] < volts[b]:
            return float(np.interp(REFERENCE_V, volts[[a,b]], currents[[a,b]]))
    return None


def readout_record(row, circuit):
    ids, volts, currents = intrinsic(row, circuit, outward=False)
    ref = reference_current(ids, volts, currents)
    if ref is None:
        return None
    order = ids[np.argsort(volts[ids])]
    v, first = np.unique(volts[order], return_index=True)
    if v[0] > V_GRID[0] or v[-1] < V_GRID[-1]:
        return None
    s = np.log10(ref / REFERENCE_V / 1e-6)
    log_i = np.interp(V_GRID, v, np.log10(currents[order[first]]))
    shape = log_i - np.interp(REFERENCE_V, V_GRID, log_i)
    return float(s), shape


def learn_anchors(records):
    records = sorted((rec for rec in records if rec is not None), key=lambda rec: rec[0])
    if len(records) < 3*MIN_ANCHOR_SAMPLES:
        raise ValueError("Insufficient complete RESET return records to form three anchors")
    anchors, shapes = [], []
    for ids in np.array_split(np.arange(len(records)), min(5, len(records)//MIN_ANCHOR_SAMPLES)):
        anchors.append(float(np.median([records[i][0] for i in ids])))
        shape = np.median([records[i][1] for i in ids], axis=0)
        shapes.append(np.maximum.accumulate(shape))
    shapes = np.asarray(shapes)
    ref = np.flatnonzero(V_GRID == REFERENCE_V)[0]
    return np.asarray(anchors), shapes - shapes[:, ref, None]


def current_law(s, anchors, shapes):
    """Positive magnitude; outside the state anchors extend amplitude only."""
    shape = np.array([np.interp(s, anchors, k) for k in shapes.T])
    current = np.r_[0., REFERENCE_V * 1e-6 * 10**(s+shape)]
    volts = np.r_[0., V_GRID]
    return lambda v: np.interp(v, volts, current)


def rmse(errors):
    return float(np.sqrt(np.mean(np.square(errors)))) if len(errors) else None


def evaluate_readout(rows, circuit, anchors, shapes):
    interpolated, constant, in_range, outside, in_range_constant = [], [], [], [], []
    count = 0
    for row in rows:
        rec = readout_record(row, circuit)
        if rec is None:
            continue
        s, _ = rec
        ids, local, observed = intrinsic(row, circuit, outward=False)
        ids = ids[(local[ids] >= .2) & (local[ids] <= .60)]
        if not len(ids):
            continue
        predicted, pred_v = circuit.operating_point(
            np.abs(row.voltage[ids]), current_law(s, anchors, shapes))
        base, base_v = circuit.operating_point(
            np.abs(row.voltage[ids]), current_law(s, np.array([0.]), np.median(shapes, axis=0)[None,:]))
        # Both laws are tabulated only over the same local-voltage interval.
        ok = (pred_v <= V_GRID[-1]) & (base_v <= V_GRID[-1])
        delta = np.log10(predicted[ok] / observed[ids[ok]])
        reference_delta = np.log10(base[ok] / observed[ids[ok]])
        constant.extend(reference_delta)
        interpolated.extend(delta)
        if anchors[0] <= s <= anchors[-1]:
            in_range.extend(delta)
            in_range_constant.extend(reference_delta)
        else:
            outside.extend(delta)
        count += bool(len(delta))
    return {"cycles": count, "points":len(interpolated),
            "interpolated_shape_rmse":rmse(interpolated), "common_shape_rmse":rmse(constant),
            "in_anchor_range_points":len(in_range), "in_anchor_range_rmse":rmse(in_range),
            "in_anchor_range_common_shape_rmse":rmse(in_range_constant),
            "edge_shape_points":len(outside), "edge_shape_rmse":rmse(outside)}


def write_features(records, device, prep):
    result = []
    for set_row, reset_row, sc, rc in records:
        ids, local, current = intrinsic(reset_row, rc, outward=True)
        ref = reference_current(ids, local, current)
        if ref is None:
            continue
        set_prediction = model.simulate_sweep(set_row.voltage, "SET", device, prep, sc)
        peak = np.max(np.abs(set_prediction["current_A"]))
        result.append((np.log10(peak/1e-6), np.log10(ref/REFERENCE_V/1e-6)))
    return np.asarray(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    groups, circuits = shared.load_groups()
    paired = {}
    for gate in (.9, 1.1, 1.3):
        sk, rk = [("1T1M", gate, phase) for phase in ("SET", "RESET")]
        s, r = numbered(groups[sk]), numbered(groups[rk])
        if s.keys() != r.keys():
            raise ValueError(f"Incompatible SET and RESET labels at {gate:g} V")
        paired[gate] = [(s[i], r[i], circuits[sk], circuits[rk]) for i in sorted(s)]
    # First chronological block from each seen gate. Entire high gate is held out.
    train = paired[.9][:70]+paired[1.1][:32]
    test = {".9":paired[.9][70:], "1.1":paired[1.1][32:], "1.3":paired[1.3]}
    anchors, shapes = learn_anchors([readout_record(r, rc) for s,r,sc,rc in train])
    for s in np.linspace(anchors[0]-1., anchors[-1]+1., 31):
        law = current_law(s, anchors, shapes)
        assert np.isclose(law(REFERENCE_V)/REFERENCE_V/1e-6, 10**s, rtol=1e-12)
        assert np.all(np.diff(law(np.r_[0., V_GRID])) >= 0)
    nominal = paired[1.1][:32]
    d,p = shared.fit_soft_set([s for s,r,sc,rc in nominal],nominal[0][2])
    samples = write_features(train,d,p)
    if np.ptp(samples[:,0]) < .01:
        raise ValueError("Training lacks independent variation in predicted SET current")
    slope, offset = np.polyfit(samples[:,0], samples[:,1], 1)
    results = {}
    for name, records in test.items():
        reader = evaluate_readout([r for s,r,sc,rc in records],records[0][3],anchors,shapes)
        x,y = write_features(records,d,p).T
        prediction = offset+slope*x
        results[name] = {"return_conditioned":reader,
                         "reset_entry_from_predicted_set": {
                             "cycles":len(y), "rmse_decade":rmse(prediction-y),
                             "predicted_median_uS":float(10**np.median(prediction)),
                             "observed_log_median_uS":float(10**np.median(y)),
                             "constant_training_median_rmse":rmse(np.median(samples[:,1])-y)}}
    output = {
        "protocol":"Train on the first 70 cycles at 0.9 V and cycles 4–35 at 1.1 V. Hold out the rest and all of 1.3 V. State is log10(G at local 0.15 V / 1 uS). Return shape uses the held-out return low-bias read as input and scores separate higher-bias points. Edge templates extend only in amplitude. SET diagnostic predicts RESET entry using the frozen nominal SET peak, with one shared log-linear rule trained at both seen gates.",
        "training_labels": {"0.9":[s.label for s,r,sc,rc in paired[.9][:70]],
                            "1.1":[s.label for s,r,sc,rc in nominal]},
        "local_grid_v": V_GRID.tolist(), "soft_anchors":anchors.tolist(),
        "log_normalized_current_templates":shapes.tolist(),
        "set_peak_to_state":{"samples":len(samples),"slope":float(slope),"offset":float(offset)},
        "results":results}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"anchors":anchors.tolist(),"results":results},indent=2))


if __name__ == "__main__":
    main()
