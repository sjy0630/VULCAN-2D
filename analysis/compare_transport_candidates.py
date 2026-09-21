"""Cycle-held and voltage-held transport comparison under a saved design.

This is a subsequent internal discrimination analysis on previously used data,
not an external blind test or a complete-cycle autonomous switching model.
"""
from __future__ import annotations
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from diagnose_cross_polarity_physics import GateLookup, read_pairs
from vulcan2d.transistor import load_default_lookup, load_reverse_lookup
from vulcan2d.transport_candidates import (
    Curve, NAMES, TransportAssumptions, conditional_physics,
    fit_shared_shape, log10_shape, profile_amplitude,
)

OUT = ROOT / "analysis"
ASSUMPTIONS = TransportAssumptions()
BRANCHES = {
    "set_pre": {"phase": "set", "return": False, "low": .50, "high": 1.20, "cap": 1.25},
    "set_connected": {"phase": "set", "return": True, "low": .15, "high": .80, "cap": 5.0},
    "reset_initial": {"phase": "reset", "return": False, "low": .10, "high": .35, "cap": .45},
    "reset_hrs": {"phase": "reset", "return": True, "low": .50, "high": 1.20, "cap": 1.7},
}


def extract_points():
    """Keep native samples and explicit reasons, before candidate fitting."""
    lookup = {p: GateLookup(f(), 1.1) for p, f in
              (("set", load_default_lookup), ("reset", load_reverse_lookup))}
    rows = []
    for pair in read_pairs("1.1"):
        for name, config in BRANCHES.items():
            raw = pair[config["phase"]]
            vapp, signed_i = np.asarray(raw["voltage"]), np.asarray(raw["current"])
            vapp, imag = np.abs(vapp), np.abs(signed_i)
            zero = float(np.max(imag[(vapp <= 1e-12) & np.isfinite(imag)]))
            apex = int(np.nanargmax(vapp))
            last_apex = len(vapp) - 1 - int(np.nanargmax(vapp[::-1]))
            indices = range(last_apex, len(vapp)) if config["return"] else range(apex + 1)
            for index in indices:
                v, i = float(vapp[index]), float(imag[index])
                if v > config["cap"]:
                    continue
                row = {"branch": name, "cycle_id": pair["cycle_id"], "raw_index": index,
                       "applied_abs_voltage_V": v, "signed_current_A": float(signed_i[index]),
                       "current_A": i, "zero_reference_A": zero,
                       "hbn_voltage_V": None, "transistor_voltage_V": None,
                       "transistor_fraction": None, "capacity_fraction": None,
                       "valid": True, "reason": ""}
                reasons = []
                if not np.isfinite(v) or not np.isfinite(i):
                    reasons.append("nonfinite")
                elif v <= 1e-12:
                    reasons.append("zero_voltage")
                elif i <= 0:
                    reasons.append("nonpositive_magnitude")
                else:
                    try:
                        vd = lookup[config["phase"]].inverse(i)
                        vh = v - vd
                        row.update(hbn_voltage_V=vh, transistor_voltage_V=vd,
                                   transistor_fraction=vd/v,
                                   capacity_fraction=i/lookup[config["phase"]].at_voltage(v))
                        if vh <= 0:
                            reasons.append("nonpositive_internal_voltage")
                        if vd/v > .5:
                            reasons.append("transistor_fraction_gt_0p5")
                        if row["capacity_fraction"] > .7:
                            reasons.append("capacity_fraction_gt_0p7")
                    except ValueError:
                        reasons.append("lookup_out_of_domain")
                row["valid"], row["reason"] = not reasons, ";".join(reasons)
                rows.append(row)
    return rows


def assemble_curves(rows, branch, window, signal):
    config = dict(BRANCHES[branch])
    if window == "wide" and branch in ("set_pre", "reset_hrs"):
        config["low"] = .30
    threshold = config["low"] + .60 * (config["high"] - config["low"])
    curves, selection = [], []
    for cycle_id in range(1, 54):
        eligible = [r for r in rows if r["cycle_id"] == cycle_id and r["branch"] == branch
                    and r["valid"] and config["low"] <= r["hbn_voltage_V"] <= config["high"]]
        selected = [r for r in eligible if signal == "all_positive" or r["current_A"] > 10*r["zero_reference_A"]]
        selected.sort(key=lambda r: r["hbn_voltage_V"])
        v = np.array([r["hbn_voltage_V"] for r in selected])
        y = np.log10([r["current_A"] for r in selected])
        low_n, high_n = int(np.sum(v <= threshold)), int(np.sum(v > threshold))
        # The common eligible set is fixed across candidates and evaluations.
        accepted = len(v) >= 8 and low_n >= 4 and high_n >= 3
        if accepted:
            curves.append(Curve(cycle_id, v, y))
        selection.append({"branch": branch, "window": window, "signal": signal,
                          "cycle_id": cycle_id, "n_numerical_valid_in_window": len(eligible),
                          "n_selected": len(selected), "n_low": low_n, "n_high": high_n,
                          "included_all_candidates": accepted,
                          "reason": "" if accepted else "insufficient_common_low_high_coverage"})
    return curves, selection, config, threshold


def subset(curve, low, threshold):
    mask = curve.voltage_v <= threshold if low else curve.voltage_v > threshold
    return Curve(curve.cycle_id, curve.voltage_v[mask], curve.log10_current_a[mask])


def score(curve, fit, amplitude, protocol, threshold, assumptions=ASSUMPTIONS):
    prediction = amplitude + log10_shape(curve.voltage_v, fit["name"], fit["parameter"], assumptions)
    residual = prediction - curve.log10_current_a
    row = {"cycle_id": curve.cycle_id, "candidate": fit["name"], "protocol": protocol,
           "parameter": fit["parameter"], "parameter_at_boundary": fit["parameter_at_boundary"],
           "log10_reference_amplitude": amplitude, "n_scored": len(residual),
           "rmse_decade": float(np.sqrt(np.mean(residual**2))),
           "bias_decade": float(np.mean(residual)),
           "max_abs_error_decade": float(np.max(np.abs(residual))),
           "n_training_cycles": fit["n_training_cycles"]}
    for tag, mask in (("low", curve.voltage_v <= threshold), ("high", curve.voltage_v > threshold)):
        row[f"{tag}_rmse_decade"] = (float(np.sqrt(np.mean(residual[mask]**2))) if np.any(mask) else None)
    row.update(conditional_physics(fit["name"], fit["parameter"], assumptions))
    return row


def evaluate_loco(curves, name, threshold, assumptions=ASSUMPTIONS):
    rows = []
    for target in curves:
        training = [c for c in curves if c.cycle_id != target.cycle_id]
        fit = fit_shared_shape(training, name, assumptions)
        assert target.cycle_id not in fit["training_cycle_ids"]
        rows.append(score(target, fit, fit["log10_reference_amplitude_center"],
                          "loco_autonomous", threshold, assumptions))
        low_training = [subset(c, True, threshold) for c in training]
        fit = fit_shared_shape(low_training, name, assumptions)
        low_target = subset(target, True, threshold)
        amplitude = profile_amplitude(low_target, name, fit["parameter"], assumptions)
        rows.append(score(subset(target, False, threshold), fit, amplitude,
                          "loco_low_read_conditioned_high_voltage", threshold, assumptions))
    return rows


def summarize(rows):
    if not rows:
        return {"n_cycles": 0}
    rmses = np.array([r["rmse_decade"] for r in rows])
    pars = [r["parameter"] for r in rows if r["parameter"] is not None]
    return {"n_cycles": len(rows), "target_cycle_ids": [r["cycle_id"] for r in rows],
            "equal_cycle_rmse_decade": float(np.sqrt(np.mean(rmses**2))),
            "median_cycle_rmse_decade": float(np.median(rmses)),
            "cycle_rmse_q10_q90": np.quantile(rmses, [.1, .9]).tolist(),
            "max_cycle_rmse_decade": float(np.max(rmses)),
            "mean_signed_bias_decade": float(np.mean([r["bias_decade"] for r in rows])),
            "parameter_q10_median_q90": np.quantile(pars,[.1,.5,.9]).tolist() if pars else None,
            "n_shape_at_parameter_boundary": sum(r["parameter_at_boundary"] for r in rows)}


def write_csv(path, rows):
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run():
    native = extract_points()
    scores, selections, groups = [], [], {}
    for window in ("primary", "wide"):
        groups[window] = {}
        for signal in ("signal_guard", "all_positive"):
            groups[window][signal] = {}
            for branch in BRANCHES:
                curves, selection, config, threshold = assemble_curves(native, branch, window, signal)
                selections.extend(selection)
                result = {"voltage_window": config, "low_high_split_internal_V": threshold,
                          "eligible_cycle_ids": [c.cycle_id for c in curves], "candidates": {}}
                for name in NAMES:
                    fitted = fit_shared_shape(curves, name)
                    rows = evaluate_loco(curves, name, threshold)
                    training = [c for c in curves if c.cycle_id <= 35]
                    fit_early = fit_shared_shape(training, name)
                    for c in curves:
                        if c.cycle_id > 35:
                            rows.append(score(c, fit_early, fit_early["log10_reference_amplitude_center"],
                                              "chronological_autonomous", threshold))
                    for c in curves:
                        amp = profile_amplitude(c, name, fitted["parameter"])
                        rows.append(score(c, fitted, amp, "retrospective_profiled", threshold))
                    for row in rows:
                        row.update(branch=branch, window=window, signal=signal)
                    scores.extend(rows)
                    result["candidates"][name] = {
                        "all_data_shape_fit": fitted,
                        "evaluation": {protocol: summarize([r for r in rows if r["protocol"] == protocol])
                                       for protocol in ("loco_autonomous", "loco_low_read_conditioned_high_voltage",
                                                        "chronological_autonomous", "retrospective_profiled")}}
                groups[window][signal][branch] = result
                print(f"Completed {window}/{signal}/{branch}: {len(curves)}/53 cycles", flush=True)
    report = {
        "purpose": "Internal transport discrimination; no change to production M13 equations",
        "assumptions": asdict(ASSUMPTIONS), "candidates": NAMES,
        "design": "TRANSPORT_COMPARISON_DESIGN_2026-09-20.md",
        "method": {"weight": "equal per cycle within each branch; squared log10-current residual",
                   "amplitude_center": "median of training-cycle profiled log reference currents",
                   "target_selection": "common candidate-independent signal/voltage/coverage masks; all exclusions reported",
                   "voltage_holdout": "shape from other cycles' low-voltage 60%; target low reads set amplitude; target high40% only scoring",
                   "chronology": "cycles <=35 train; >35 score",
                   "all_data_fit": "diagnostic only, not a predictive validation score"},
        "limitations": [
            "All data have been used in historical development; internal evaluation is not external validation.",
            "Low-read-conditioned transport prediction is not voltage-only complete-cycle prediction.",
            "Candidate stable windows assume frozen transport state; this is not independently measured.",
            "Signal guard is a processing sensitivity, not a calibrated noise model; excluded anomalous cycles remain a research gap.",
            "Apparent epsilon and hopping length assume uniform field and 300 K; not measured material parameters.",
            "Schottky zero-bias balance is a finite-current branch extension; two-interface electrostatics and asymmetric states remain unresolved.",
            "No candidate selection is justified solely by ranking; parameter validity and shared residual structure matter.",
        ],
        "input_sha256": {}, "groups": groups,
    }
    inputs = ("1T1M写入.xlsx", "1T1M擦除.xlsx", "vulcan2d/data/fig2b_transistor_lookup.json",
              "vulcan2d/data/bookoutput2_reverse_transistor_lookup.json", "vulcan2d/transport_candidates.py",
              "analysis/compare_transport_candidates.py", "analysis/diagnose_cross_polarity_physics.py",
              "analysis/TRANSPORT_COMPARISON_DESIGN_2026-09-20.md")
    report["input_sha256"] = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in inputs}
    write_csv(OUT/"transport_candidate_native_points.csv", native)
    write_csv(OUT/"transport_candidate_selection.csv", selections)
    write_csv(OUT/"transport_candidate_cycle_scores.csv", scores)
    (OUT/"transport_candidate_comparison.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    print(f"Saved {len(scores)} cycle-score records", flush=True)
    return report


if __name__ == "__main__":
    run()
