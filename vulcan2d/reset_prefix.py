"""Forecast the unread portion of a RESET sweep from its outgoing prefix."""
import argparse
import csv
from pathlib import Path

import numpy as np

from . import predict_gate
from .validate_transition_regions import CURRENT_FLOOR_A, NEAR_TRANSISTOR_CAP

LOG_FLOOR = -18.
BIN_COUNT = 8


def features(voltage, current, transistor, cutoff):
    """Extract binned log current and inferred conductance from available samples."""
    v = np.abs(np.asarray(voltage, dtype=float))
    data = np.abs(np.asarray(current, dtype=float))
    if (v.ndim != 1 or not v.size or data.shape != v.shape or not np.isfinite(v).all()
            or np.any(np.diff(v) < -1e-10)
            or v[-1] < cutoff - .005 or v[-1] > cutoff + 1e-9):
        raise ValueError("Use only the measured outgoing RESET prefix up to the requested cutoff")
    grid, drain = transistor
    cap = np.interp(v, grid, drain)
    good = (np.isfinite(data) & (data >= CURRENT_FLOOR_A["RESET"])
            & (data < NEAR_TRANSISTOR_CAP * cap))
    ids = np.flatnonzero(v >= .05 - 1e-9)
    bins = np.array_split(ids, BIN_COUNT)
    log_i = np.log10(np.maximum(data, 10**LOG_FLOOR))
    levels, idx = np.unique(drain, return_index=True)
    local = v - np.interp(data, levels, grid[idx])
    result = []
    for indices in bins:
        kept = indices[good[indices]]
        result.append(np.median(log_i[kept]) if kept.size else np.nan)
    for indices in (bins[0], bins[-1]):
        kept = indices[good[indices] & (local[indices] >= .03)]
        result.append(np.median(log_i[kept] - np.log10(local[kept])) if kept.size else np.nan)
    return np.asarray(result)


def normalize(x, center, scale, quadratic=False):
    u = np.nan_to_num(np.clip((x-center)/scale,-4.,4.), nan=0.)
    if quadratic:
        a,b = np.triu_indices(u.shape[1])
        u = np.concatenate((u, .5*u[:,a]*u[:,b]), axis=1)
    return u


def fit(features, logs, keep, method, size):
    """Fit one conditional predictor with training-only feature normalization."""
    center, scale = [], []
    for column in features.T:
        valid = column[np.isfinite(column)]
        center.append(np.median(valid) if len(valid) else 0.)
        scale.append(max(.25, np.std(valid)) if len(valid) else 1.)
    center, scale = np.asarray(center), np.asarray(scale)
    x = normalize(features, center, scale, method == "quadratic")
    mask = keep.astype(float)
    weights = mask.sum(axis=0)
    used = mask * logs
    mean = np.divide(used.sum(axis=0), weights, out=np.full(logs.shape[1], LOG_FLOOR),
                     where=weights > 0.)
    fitted = dict(method=np.array(method), size=np.array(size), center=center, scale=scale, mean=mean)
    if method == "neighbors":
        fitted.update(x=x, mask=mask, target=used)
    else:
        design = np.c_[np.ones(len(x)),x]
        gram = np.einsum("ni,nj,nk->kij",design,design,mask)
        target = np.einsum("ni,nk->ki",design,used)
        penalty = np.eye(design.shape[1])*size*len(design)
        penalty[0,0] = 1e-10
        fitted.update(coefficients=np.linalg.solve(gram+penalty,target[:,:,None])[:,:,0],
                      supported=weights>0.)
    return fitted


def predict(fitted, feature_matrix):
    method = str(fitted["method"])
    x = normalize(np.atleast_2d(feature_matrix), fitted["center"], fitted["scale"],
                  method == "quadratic")
    if method == "neighbors":
        dist = np.mean((x[:,None,:]-fitted["x"][None,:,:])**2,axis=2)
        n = min(int(fitted["size"]), dist.shape[1])-1
        h = np.maximum(.025,np.partition(dist,n,axis=1)[:,n,None])
        weights = np.exp(-.5*dist/h)
        support = np.einsum("ij,jk->ik",weights,fitted["mask"])
        sums = np.einsum("ij,jk->ik",weights,fitted["target"])
        result = np.divide(sums,support,out=np.broadcast_to(fitted["mean"],sums.shape).copy(),where=support>0.)
    else:
        design = np.c_[np.ones(len(x)),x]
        result = np.einsum("ij,kj->ik",design,fitted["coefficients"])
        result[:,~fitted["supported"]] = fitted["mean"][~fitted["supported"]]
    if not np.isfinite(result).all():
        raise RuntimeError("RESET prefix prediction produced nonfinite currents")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration",type=Path)
    parser.add_argument("prefix",type=Path, help="CSV columns voltage_V and current_A, outgoing only")
    parser.add_argument("--gate", type=float, help="Target gate for a cross-gate calibration")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    with np.load(args.calibration,allow_pickle=False) as loaded:
        cal=dict(loaded)
    voltage, current = np.loadtxt(args.prefix,delimiter=",",skiprows=1,ndmin=2).T
    wave=cal["wave"]
    a=int(np.argmax(np.abs(wave)))
    if "gate" in cal:
        gate = float(cal["gate"])
        if args.gate is not None and not np.isclose(args.gate, gate, rtol=0, atol=1e-9):
            parser.error(f"This same-gate calibration requires {gate:g} V")
    else:
        gate = args.gate
        if gate is None or not float(cal["minimum_gate"]) <= gate <= float(cal["maximum_gate"]):
            parser.error("Provide a target gate within the range of the validation")
    _, reset_circuit=predict_gate.circuits_at_gate(gate)
    cutoff=float(cal["cutoff"])
    x=features(voltage,current,reset_circuit.transistor,cutoff)
    if not np.isfinite(x).any():
        raise ValueError("The prefix has no valid non-limited readings")
    future=(np.arange(len(wave))>a) | (np.abs(wave)>cutoff+1e-9)
    y=predict(cal,x)[0]
    # Saturate at the transistor's available current in the same load line.
    current=np.minimum(10**np.minimum(y,0.),reset_circuit.supply_bound(np.abs(wave)))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(("voltage_V","predicted_current_A","branch"))
        for index in np.flatnonzero(future):
            writer.writerow((wave[index],np.sign(wave[index])*current[index],
                             "outward" if index<=a else "return"))


if __name__ == "__main__":
    main()
