"""Measured-lookup-only SET-return / RESET-initial polarity diagnostic.

Run from any directory with the research Python environment. No h-BN
transport envelope, latent-state fit, fitted gate correction, or model current
is used. All ratios are retrospective measured-data diagnostics, not forecasts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vulcan2d.transistor import load_default_lookup, load_reverse_lookup


DATASETS = {
    "1.1": ("1T1M写入.xlsx", "1T1M擦除.xlsx", "confirmed_Si_Ri_pairs"),
    "0.9": ("0.9V栅压.xlsx", "0.9V栅压RESET.xlsx", "exploratory_matching_Sheet_identifiers"),
    "1.3": ("1.3V栅压.xlsx", "1.3V栅压RESET.xlsx", "exploratory_matching_Sheet_identifiers"),
}
WINDOWS = {"primary": (0.10, 0.30), "lower": (0.08, 0.25), "higher": (0.15, 0.35)}
GRID = np.round(np.arange(0.08, 0.35001, 0.01), 10)
ZERO_V = 1e-12
CAPS = {"set": 0.60, "reset": 0.45}
MAX_CAPACITY_FRACTION = 0.50
MAX_TRANSISTOR_VOLTAGE_FRACTION = 0.25
MAX_INVERSE_CONDITION = 1.0
NUMERICAL_CURRENT_PERTURBATION = 0.01
PREFIX = "cross_polarity_physics"


class GateLookup:
    """Exact piecewise-linear inverse of the processed lookup at fixed gate."""

    def __init__(self, lookup, gate):
        self.lookup, self.gate = lookup, gate
        nodes = np.unique(np.concatenate([
            curve["drain_voltage_v"] for curve in lookup.curves()]))
        self.voltage = nodes[(nodes >= lookup.vds_min) & (nodes <= lookup.vds_max)]
        self.current = np.array([lookup.current(v, gate) for v in self.voltage])

    def at_voltage(self, voltage):
        if not self.voltage[0] <= voltage <= self.voltage[-1]:
            raise ValueError("applied voltage outside transistor lookup domain")
        return float(np.interp(voltage, self.voltage, self.current))

    def inverse(self, current):
        if not np.isfinite(current) or not 0 <= current <= self.current[-1]:
            raise ValueError("current outside transistor lookup domain")
        index = int(np.searchsorted(self.current, current, side="left"))
        if index == 0:
            return float(self.voltage[0])
        lo, hi = self.current[index - 1], self.current[index]
        if hi == lo:
            raise ValueError("flat transistor segment makes inverse ambiguous")
        weight = (current - lo) / (hi - lo)
        return float(self.voltage[index - 1]
                     + weight * (self.voltage[index] - self.voltage[index - 1]))


def read_pairs(gate):
    set_file, reset_file, pairing = DATASETS[gate]
    sets = pd.read_excel(ROOT / set_file, sheet_name=None)
    resets = pd.read_excel(ROOT / reset_file, sheet_name=None)

    def indexed(frames):
        result = {}
        for name, frame in frames.items():
            match = re.fullmatch(r"(?:S|R|Sheet)\s*(\d+)", str(name), re.I)
            if match:
                index = int(match.group(1))
                if index in result:
                    raise ValueError(f"Duplicate numeric sheet identifier: {name}")
                result[index] = (name, frame)
        return result

    sets, resets = indexed(sets), indexed(resets)
    if sets.keys() != resets.keys():
        raise ValueError(f"Unmatched SET/RESET sheet identifiers at gate {gate}")
    pairs = []
    for index in sorted(sets):
        pair = {"gate_V": float(gate), "cycle_id": index, "pairing": pairing}
        for phase, frames in (("set", sets), ("reset", resets)):
            name, frame = frames[index]
            voltage = frame.iloc[:, 0].to_numpy(float)
            current = frame.iloc[:, 1].to_numpy(float)
            pair[phase] = {"sheet": name, "voltage": voltage, "current": current}
        pairs.append(pair)
    return pairs


def deembed_branch(raw, phase, lookup, offset_sign=0):
    voltage = np.abs(np.asarray(raw["voltage"], dtype=float))
    observed = np.abs(np.asarray(raw["current"], dtype=float))
    zeros = observed[np.isfinite(voltage) & (voltage <= ZERO_V) & np.isfinite(observed)]
    zero_scale = float(np.max(zeros)) if len(zeros) else 0.0
    zero_median = float(np.median(zeros)) if len(zeros) else None
    apex = int(np.nanargmax(voltage))
    if phase == "set":
        last_apex = len(voltage) - 1 - int(np.nanargmax(voltage[::-1]))
        # Reverse the return branch into increasing applied magnitude.
        indices = np.arange(len(voltage) - 1, last_apex - 1, -1)
    else:
        indices = np.arange(apex + 1)
    points = []
    for index in indices:
        vapp = float(voltage[index])
        if np.isfinite(vapp) and vapp > CAPS[phase]:
            continue
        current = float(observed[index] + offset_sign * zero_scale)
        item = {
            "raw_index": int(index), "applied_abs_voltage_V": vapp,
            "observed_abs_current_A": float(observed[index]),
            "perturbed_current_A": current,
            "transistor_voltage_V": None, "hbn_voltage_V": None,
            "transistor_voltage_fraction": None, "transistor_capacity_fraction": None,
            "inverse_condition": None, "status": "valid", "reason": None,
        }
        reasons = []
        if not np.isfinite(vapp) or not np.isfinite(current):
            reasons.append("nonfinite_observation")
        elif vapp <= ZERO_V:
            reasons.append("zero_bias")
        elif current <= 0:
            reasons.append("nonpositive_current")
        else:
            try:
                vds = lookup.inverse(current)
                vh = vapp - vds
                item["transistor_voltage_V"] = vds
                item["hbn_voltage_V"] = vh
                item["transistor_voltage_fraction"] = vds / vapp
                capacity = lookup.at_voltage(vapp)
                item["transistor_capacity_fraction"] = current / capacity
                if vh <= 0:
                    reasons.append("nonpositive_hbn_voltage")
                if item["transistor_voltage_fraction"] > MAX_TRANSISTOR_VOLTAGE_FRACTION:
                    reasons.append("transistor_voltage_fraction_guard")
                if item["transistor_capacity_fraction"] > MAX_CAPACITY_FRACTION:
                    reasons.append("transistor_capacity_guard")
                eps = NUMERICAL_CURRENT_PERTURBATION
                lower, upper = lookup.inverse(current * (1 - eps)), lookup.inverse(current * (1 + eps))
                if vh > 0:
                    condition = (upper - lower) / (2 * eps * vh)
                    item["inverse_condition"] = condition
                    if condition > MAX_INVERSE_CONDITION:
                        reasons.append("inverse_sensitivity_guard")
            except ValueError as error:
                reasons.append(str(error))
        if reasons:
            item["status"] = "excluded"
            item["reason"] = "; ".join(reasons)
        points.append(item)
    return {
        "points": points, "zero_bias_n": len(zeros),
        "zero_bias_abs_current_median_A": zero_median,
        "zero_bias_abs_current_max_A": zero_scale,
    }


def interpolate_branch(branch, target):
    """Interpolate only unique, increasing, valid native adjacent segments."""
    candidates = []
    points = branch["points"]
    fold_crossing = False
    for left, right in zip(points[:-1], points[1:]):
        if left["status"] != "valid" or right["status"] != "valid":
            continue
        if abs(left["raw_index"] - right["raw_index"]) != 1:
            continue
        low, high = left["hbn_voltage_V"], right["hbn_voltage_V"]
        if min(low, high) - 1e-10 <= target <= max(low, high) + 1e-10:
            if high <= low:
                fold_crossing = True
                continue
            weight = float(np.clip((target - low) / (high - low), 0.0, 1.0))
            i0, i1 = left["perturbed_current_A"], right["perturbed_current_A"]
            candidates.append({
                "current_A": float(10 ** ((1 - weight) * np.log10(i0) + weight * np.log10(i1))),
                "linear_current_A": float((1 - weight) * i0 + weight * i1),
                "raw_left_index": left["raw_index"], "raw_right_index": right["raw_index"],
                "transistor_voltage_V": float((1 - weight) * left["transistor_voltage_V"]
                                               + weight * right["transistor_voltage_V"]),
                "transistor_voltage_fraction": float((1 - weight) * left["transistor_voltage_fraction"]
                                                      + weight * right["transistor_voltage_fraction"]),
                "transistor_capacity_fraction": float((1 - weight) * left["transistor_capacity_fraction"]
                                                       + weight * right["transistor_capacity_fraction"]),
                "inverse_condition": max(left["inverse_condition"], right["inverse_condition"]),
            })
    if fold_crossing:
        return None, "internal_voltage_fold"
    if not candidates:
        return None, "no_valid_adjacent_bracket_no_extrapolation"
    if len(candidates) > 1:
        logs = [np.log10(item["current_A"]) for item in candidates]
        if max(logs) - min(logs) > 1e-8:
            return None, "multiple_internal_voltage_crossings"
    return candidates[0], None


def ratio_grid(pair, policy, branches, sensitivity_branches):
    rows = []
    for voltage in GRID:
        row = {"gate_V": pair["gate_V"], "cycle_id": pair["cycle_id"],
               "lookup_branch": policy, "hbn_voltage_V": float(voltage),
               "pairing": pair["pairing"], "status": "valid", "reason": None}
        interpolated = {}
        for phase in ("set", "reset"):
            item, reason = interpolate_branch(branches[phase], voltage)
            interpolated[phase] = item
            if reason:
                row["status"] = "unavailable"
                row["reason"] = (row["reason"] + "; " if row["reason"] else "") + f"{phase}: {reason}"
            if item is not None:
                row.update({f"{phase}_{key}": value for key, value in item.items()})
                row[f"{phase}_zero_reference_fraction"] = (
                    branches[phase]["zero_bias_abs_current_max_A"] / item["current_A"])
        if row["status"] == "valid":
            set_item, reset_item = interpolated["set"], interpolated["reset"]
            row["eta"] = reset_item["current_A"] / set_item["current_A"]
            row["log10_eta"] = float(np.log10(row["eta"]))
            row["eta_linear_current_interpolation"] = (
                reset_item["linear_current_A"] / set_item["linear_current_A"])
            row["zero_reference_exceeds_10percent_of_either_current"] = any(
                row[f"{phase}_zero_reference_fraction"] >= 0.1 for phase in ("set", "reset"))
            row["log_vs_linear_interpolation_difference_decade"] = float(np.log10(
                row["eta"] / row["eta_linear_current_interpolation"]))
            alternative = []
            for set_sign in (-1, 1):
                for reset_sign in (-1, 1):
                    set_perturbed, _ = interpolate_branch(sensitivity_branches["set"][set_sign], voltage)
                    reset_perturbed, _ = interpolate_branch(sensitivity_branches["reset"][reset_sign], voltage)
                    if set_perturbed is not None and reset_perturbed is not None:
                        alternative.append(reset_perturbed["current_A"] / set_perturbed["current_A"])
            row["zero_reference_sensitivity_available_combinations"] = len(alternative)
            # An incomplete set of perturbations cannot furnish two-sided
            # sensitivity bounds (e.g. I - zero-reference becomes nonpositive).
            row["zero_reference_sensitivity_eta_min"] = min(alternative) if len(alternative) == 4 else None
            row["zero_reference_sensitivity_eta_max"] = max(alternative) if len(alternative) == 4 else None
        rows.append(row)
    return rows


def summarize_cycle(pair, policy, name, limits, ratios, branches):
    low, high = limits
    selected = [row for row in ratios if low - 1e-10 <= row["hbn_voltage_V"] <= high + 1e-10]
    good = [row for row in selected if row["status"] == "valid"]
    result = {
        "gate_V": pair["gate_V"], "cycle_id": pair["cycle_id"], "pairing": pair["pairing"],
        "lookup_branch": policy, "window": name, "window_low_V": low, "window_high_V": high,
        "n_grid_requested": len(selected), "n_grid_valid": len(good),
        "full_window_coverage": len(good) == len(selected),
        "set_recorded_peak_abs_voltage_V": float(np.nanmax(np.abs(pair["set"]["voltage"]))),
        "reset_recorded_peak_abs_voltage_V": float(np.nanmax(np.abs(pair["reset"]["voltage"]))),
        "status": "insufficient_data", "reason": None,
    }
    for phase in ("set", "reset"):
        branch = branches[phase]
        points = branch["points"]
        result[f"{phase}_native_points_in_window"] = sum(
            p["status"] == "valid" and low <= p["hbn_voltage_V"] <= high for p in points)
        result.update({f"{phase}_{key}": value for key, value in branch.items() if key != "points"})
        brackets = set()
        for row in good:
            brackets.update((row[f"{phase}_raw_left_index"], row[f"{phase}_raw_right_index"]))
        result[f"{phase}_distinct_bracketing_native_points"] = len(brackets)
    if len(good) < 5 or min(result[f"{phase}_distinct_bracketing_native_points"]
                            for phase in ("set", "reset")) < 3:
        result["reason"] = "Requires at least 5 matched grid points and 3 native points per branch"
        return result
    x = np.array([row["hbn_voltage_V"] for row in good])
    y = np.array([row["log10_eta"] for row in good])
    constant = float(np.mean(y))
    result.update({
        "status": "complete_window" if result["full_window_coverage"] else "partial_window",
        "observed_grid_min_V": float(x[0]), "observed_grid_max_V": float(x[-1]),
        "identity_rmse_decade": float(np.sqrt(np.mean(y ** 2))),
        "best_constant_eta": float(10 ** constant),
        "median_eta": float(10 ** np.median(y)),
        "constant_ratio_rmse_decade": float(np.sqrt(np.mean((y - constant) ** 2))),
        "constant_ratio_max_abs_residual_decade": float(np.max(np.abs(y - constant))),
        "eta_min": float(10 ** np.min(y)), "eta_max": float(10 ** np.max(y)),
        "log10_eta_range_decade": float(np.ptp(y)),
        "log10_eta_slope_decade_per_V": float(np.polyfit(x, y, 1)[0]),
        "log10_eta_last_minus_first_decade": float(y[-1] - y[0]),
        "log_vs_linear_interpolation_max_difference_decade": max(
            abs(row["log_vs_linear_interpolation_difference_decade"]) for row in good),
    })
    for phase in ("set", "reset"):
        for field in ("transistor_voltage_fraction", "transistor_capacity_fraction",
                      "inverse_condition", "zero_reference_fraction"):
            result[f"{phase}_{field}_max"] = max(row[f"{phase}_{field}"] for row in good)
    shifts = []
    for row in good:
        for bound in ("min", "max"):
            value = row["zero_reference_sensitivity_eta_" + bound]
            if value is not None:
                shifts.append(abs(float(np.log10(value / row["eta"]))))
    result["zero_reference_sensitivity_max_shift_decade"] = max(shifts) if shifts else None
    result["zero_reference_sensitivity_full_4_combinations_grid_points"] = sum(
        row["zero_reference_sensitivity_available_combinations"] == 4 for row in good)
    result["n_grid_zero_reference_exceeds_10percent"] = sum(
        row["zero_reference_exceeds_10percent_of_either_current"] for row in good)
    return result


def quantiles(values):
    values = [value for value in values if value is not None and np.isfinite(value)]
    return [float(value) for value in np.quantile(values, [0.1, 0.5, 0.9])] if values else None


def summarize_group(cycles, ratio_rows, gate, policy, window, cycle_ids):
    selected = [row for row in cycles if row["gate_V"] == gate and row["lookup_branch"] == policy
                and row["window"] == window and row["cycle_id"] in cycle_ids]
    full = [row for row in selected if row["status"] == "complete_window"]
    usable_ids = {row["cycle_id"] for row in full}
    low, high = WINDOWS[window]
    points = [row for row in ratio_rows if row["gate_V"] == gate and row["lookup_branch"] == policy
              and row["cycle_id"] in usable_ids and row["status"] == "valid"
              and low - 1e-10 <= row["hbn_voltage_V"] <= high + 1e-10]
    result = {
        "requested_cycle_ids": cycle_ids, "n_cycles": len(selected),
        "n_complete_window_cycles": len(full),
        "complete_window_cycle_ids": sorted(usable_ids),
        "partial_or_insufficient_cycles": [{"cycle_id": row["cycle_id"], "status": row["status"],
                                             "n_grid_valid": row["n_grid_valid"]}
                                            for row in selected if row["status"] != "complete_window"],
        "n_interpolated_grid_points": len(points),
    }
    for key in ("best_constant_eta", "identity_rmse_decade", "constant_ratio_rmse_decade",
                "log10_eta_range_decade", "log10_eta_slope_decade_per_V",
                "set_transistor_voltage_fraction_max", "reset_transistor_voltage_fraction_max",
                "set_zero_reference_fraction_max", "reset_zero_reference_fraction_max",
                "zero_reference_sensitivity_max_shift_decade",
                "log_vs_linear_interpolation_max_difference_decade"):
        result[key + "_q10_median_q90"] = quantiles([row[key] for row in full])
    if points:
        y = np.array([row["log10_eta"] for row in points])
        result["universal_constant_eta"] = float(10 ** np.mean(y))
        result["universal_constant_rmse_decade"] = float(np.std(y))
        result["identity_pooled_rmse_decade"] = float(np.sqrt(np.mean(y ** 2)))
        result["per_cycle_constant_pooled_rmse_decade"] = float(np.sqrt(np.mean([
            row["constant_ratio_rmse_decade"] ** 2 for row in full])))
        result["n_cycles_constant_residual_rmse_above_0p1_decade"] = sum(
            row["constant_ratio_rmse_decade"] > 0.1 for row in full)
    return result


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis")
    parser.add_argument("--gates", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    args = parser.parse_args(argv)
    all_ratios, all_cycles, all_native, loaded = [], [], [], {}
    for gate_text in args.gates:
        pairs = read_pairs(gate_text)
        loaded[gate_text] = [pair["cycle_id"] for pair in pairs]
        for policy in ("mean", "up"):
            lookups = {
                "set": GateLookup(load_default_lookup(branch=policy), float(gate_text)),
                "reset": GateLookup(load_reverse_lookup(branch=policy), float(gate_text)),
            }
            for pair in pairs:
                branches = {phase: deembed_branch(pair[phase], phase, lookups[phase])
                            for phase in ("set", "reset")}
                sensitive = {phase: {
                    sign: deembed_branch(pair[phase], phase, lookups[phase], offset_sign=sign)
                    for sign in (-1, 1)} for phase in ("set", "reset")}
                ratios = ratio_grid(pair, policy, branches, sensitive)
                all_ratios.extend(ratios)
                for phase, branch in branches.items():
                    all_native.extend({"gate_V": pair["gate_V"], "cycle_id": pair["cycle_id"],
                                       "lookup_branch": policy, "phase": phase, **row}
                                      for row in branch["points"])
                for name, limits in WINDOWS.items():
                    all_cycles.append(summarize_cycle(pair, policy, name, limits, ratios, branches))
            print(f"gate={gate_text} lookup={policy}: {len(pairs)} pairs", flush=True)
    actual_files = {filename for gate in args.gates for filename in DATASETS[gate][:2]}
    actual_files.update(("vulcan2d/data/fig2b_transistor_lookup.json",
                         "vulcan2d/data/bookoutput2_reverse_transistor_lookup.json"))
    report = {
        "purpose": "Retrospective measured-lookup-only cross-polarity transport diagnostic",
        "method": {
            "definition": "eta_i(v)=abs(I_RESET_initial_i(v))/abs(I_SET_return_i(v))",
            "v_definition": "h-BN internal voltage magnitude after subtracting measured-lookup transistor drop",
            "hbn_fitted_envelopes_used": False,
            "gate_mapping": "literal applied gate, no fitted affine gate correction or series resistance",
            "transistor_preprocessing": (
                "Existing lookup: mean up/down branch, forced zero origin, cumulative maximum; "
                "up-only sensitivity repeats the analysis. Fixed-gate inverse uses exact piecewise-linear lookup segments."),
            "branch_selection": {"SET": "return branch after final maximum, reversed for increasing voltage",
                                 "RESET": "initial branch before first maximum", "applied_voltage_caps_V": CAPS},
            "windows_V": WINDOWS, "grid_step_V": 0.01,
            "window_interpretation": (
                "Prespecified candidate initial low-bias windows; no independent proof that internal state is perfectly frozen. "
                "Window comparisons are sensitivity diagnostics, not optimization for agreement."),
            "zero_bias_excluded_abs_voltage_le_V": ZERO_V,
            "guards": {"maximum_transistor_capacity_fraction": MAX_CAPACITY_FRACTION,
                       "maximum_transistor_voltage_fraction": MAX_TRANSISTOR_VOLTAGE_FRACTION,
                       "maximum_inverse_condition": MAX_INVERSE_CONDITION,
                       "inverse_condition": "[Vds(1.01 I)-Vds(0.99 I)]/(0.02 VhBN)",
                       "interpretation": "Prespecified numerical guards, not material thresholds or measurement uncertainty"},
            "interpolation": (
                "Linear log10(current) interpolation only on valid adjacent native segments with increasing internal voltage; "
                "folds and multiple-valued inverses excluded; no extrapolation and no interpolation across excluded points. "
                "Linear-current interpolation reported as sensitivity. Grid points are correlated, not independent measurements."),
            "zero_reference_sensitivity": (
                "No offsets subtracted in main analysis. For sensitivity only, perturb each branch's current magnitude by "
                "plus/minus its own maximum absolute zero-bias readout, re-invert and re-interpolate, and report four combinations. "
                "Two-sided min/max are null unless all four combinations remain calculable. The 10-percent flag is descriptive "
                "and does not remove points or cycles. This is an illustrative processing sensitivity, not an estimated noise level, "
                "confidence interval, or calibration uncertainty."),
            "fits": (
                "Identity eta=1; per-cycle best constant exp10(mean(log10 eta)); universal constant across cycles; "
                "linear trend of log10 eta versus voltage. These are descriptive same-data fits, not validation of a mechanism."),
            "adequacy": (
                "No statistical rejection test without measurement error/repeat-read estimates. Residuals and a descriptive "
                "0.1-decade benchmark are reported, not a physical acceptance threshold."),
            "first_three_split": (
                "Post-hoc sensitivity motivated by the earlier v0.13 error audit. All 53 remain in the primary report; "
                "cycles 4-53 are a descriptive comparison, not a replacement dataset or independent test."),
        },
        "limitations": [
            "The ratio is cross-polarity and cross-scan; it does not isolate time relaxation from polarity dependence.",
            "Numerical guards do not establish correspondence of standalone lookup devices to the target cell or quantify gate interpolation error.",
            "0.9/1.3-V equal Sheet identifiers give exploratory pairs; only 1.1-V Si/Ri pairing is treated as confirmed.",
            "The gate groups are not established as a controlled paired-gate experiment; do not infer a causal gate law from group medians or differently filtered subsets.",
            "A constant eta can describe an amplitude relation without proving identical microscopic states or transport mechanisms.",
            "Voltage-dependent eta does not uniquely distinguish contact asymmetry, transport changes, and changing state during the read branches.",
        ],
        "input_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                         for name in sorted(actual_files)},
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ("analysis/diagnose_cross_polarity_physics.py", "vulcan2d/transistor.py")},
        "groups": {},
        "outputs": {"ratios": PREFIX + "_eta_grid.csv", "cycles": PREFIX + "_cycles.csv",
                    "native_points": PREFIX + "_native_points.csv"},
    }
    for gate, ids in loaded.items():
        report["groups"][gate] = {}
        for policy in ("mean", "up"):
            report["groups"][gate][policy] = {}
            for name in WINDOWS:
                report["groups"][gate][policy][name] = {
                    "all_cycles": summarize_group(all_cycles, all_ratios, float(gate), policy, name, ids),
                    "first_three": summarize_group(all_cycles, all_ratios, float(gate), policy, name, [i for i in ids if i <= 3]),
                    "cycles_four_onward": summarize_group(all_cycles, all_ratios, float(gate), policy, name, [i for i in ids if i >= 4]),
                }
    report["1p1V_first_three_cycle_details"] = [row for row in all_cycles
        if row["gate_V"] == 1.1 and row["cycle_id"] <= 3 and row["lookup_branch"] == "mean"]
    report["1p1V_first_three_higher_voltage_checks"] = [row for row in all_ratios
        if row["gate_V"] == 1.1 and row["cycle_id"] <= 3 and row["lookup_branch"] == "mean"
        and row["hbn_voltage_V"] in (0.25, 0.30, 0.35)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / report["outputs"]["ratios"], all_ratios)
    write_csv(args.output_dir / report["outputs"]["cycles"], all_cycles)
    write_csv(args.output_dir / report["outputs"]["native_points"], all_native)
    path = args.output_dir / (PREFIX + "_summary.json")
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"wrote {path}")
    return report


if __name__ == "__main__":
    main()
