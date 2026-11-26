"""Reproduce the fixed-source 1T/1T1R circuit-consistency audit.

No fitting is performed. Cycles 4--53 are scored descriptively with the old
full-group calibration; they are not an independent validation set. The two
transistor data sources have not been confirmed to be the same physical cell.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vulcan2d import model as M3
from vulcan2d import model_v7 as M7
from vulcan2d import model_v11 as M11
from vulcan2d import model_v12 as M12
from vulcan2d import model_v13 as M13
from vulcan2d.circuit import solve_series_load_line
from vulcan2d.transistor import measured_transistor_mapping

OBSERVATIONS = ROOT/"analysis"/"full_cycle_validation_audit_v13_curves.npz"
CURRENT_FLOORS = {"SET": 1e-11, "RESET": 1e-12}


def point_metrics(voltage, observed, predicted, floor):
    """Use the archived v13 nonzero-voltage/current-floor scoring convention."""
    voltage = np.asarray(voltage, float)
    observed, predicted = np.abs(observed), np.abs(predicted)
    if observed.shape != voltage.shape or predicted.shape != voltage.shape:
        raise ValueError("Unaligned scoring arrays")
    zero = np.abs(voltage) <= 1e-12
    observed_finite = np.isfinite(voltage) & np.isfinite(observed)
    eligible = observed_finite & ~zero & (observed >= floor)
    prediction_valid = np.isfinite(predicted) & (predicted > 0)
    scored = eligible & prediction_valid
    invalid = int(np.count_nonzero(eligible & ~prediction_valid))
    nonfinite = int(np.count_nonzero(~np.isfinite(predicted)))
    residual = np.log10(predicted[scored]/observed[scored])
    status = "invalid_prediction" if invalid or nonfinite else "ok" if len(residual) else "no_scoreable_observations"
    return {"status": status, "current_floor_A": floor,
            "n_points": len(voltage), "n_zero_bias_points": int(np.count_nonzero(zero)),
            "n_nonfinite_observed": int(np.count_nonzero(~observed_finite)),
            "n_below_current_floor": int(np.count_nonzero(observed_finite & ~zero & (observed < floor))),
            "n_scored_points": len(residual), "n_invalid_predictions_at_eligible_points": invalid,
            "n_nonfinite_predictions_all_points": nonfinite,
            "rmse_decade": float(np.sqrt(np.mean(residual**2))) if status == "ok" else None}


def archived_raw_pairs():
    """Native observed arrays saved by validate_v13, without target predictions.

    The current workbook paths are macOS dataless placeholders; this explicit
    archive avoids pretending a fresh read of the unavailable spreadsheets.
    """
    with np.load(OBSERVATIONS) as archive:
        return [{"cycle": cycle, **{
            f"{phase}_{kind}": archive[f"observed_cycle_{cycle}_{phase}_{kind}"].copy()
            for phase in ("set", "reset") for kind in ("voltage", "current")}}
                for cycle in range(1, 54)]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def direct_current(mapping, vds, gate):
    """Independent branch selection against the two original lookup objects."""
    lookup = mapping.lookup if vds >= 0 else mapping.reset_lookup
    return float(np.sign(vds) * lookup.current(abs(vds), gate))


def circuit_checks(mapping):
    calibration = M12.default_calibration()
    transport = calibration.set_transport
    reset_p = M11.params_with_transport_law(M11.Params(), calibration.reset_transport)
    lrs = calibration.set_reset_link.predict_reset_lrs_amplitude(
        transport.connected_amplitude(), transport.connected_exponent,
        reset_p.reset_alpha_lrs_identified)
    hrs = M12._cap_reset_endpoint(calibration.reset_endpoint.predict(), lrs, reset_p)
    comparisons = []
    for gate in (0.9, 1.1, 1.3):
        for state in (0.0, 0.5, 1.0):
            for applied in (-1.7, -1.0, -0.25, 0.25, 1.0, 5.0):
                if applied > 0:
                    hbn = lambda vh: np.sign(vh) * M12.set_hbn_current(
                        abs(vh), state, transport.pre_amplitude(),
                        transport.connected_amplitude(), transport)
                    current, vh = M12.divider_set(
                        applied, state, transport.pre_amplitude(),
                        transport.connected_amplitude(), gate, transport,
                        mapping=mapping)
                else:
                    hbn = lambda vh: np.sign(vh) * M7.reset_hbn_current(
                        abs(vh), state, lrs, hrs, reset_p)
                    current, vh = M7.divider_reset(
                        abs(applied), state, lrs, hrs, gate,
                        p=reset_p, mapping=mapping)
                    current, vh = -current, -vh
                root = brentq(
                    lambda v: hbn(v) - direct_current(mapping, applied-v, gate),
                    min(0.0, applied), max(0.0, applied), xtol=5e-15)
                comparisons.append({
                    "gate_V": gate, "state": state, "applied_V": applied,
                    "current_A": float(current), "hbn_voltage_V": float(vh),
                    "KCL_residual_A": float(current-direct_current(mapping, applied-vh, gate)),
                    "independent_root_current_difference_A": float(current-hbn(root)),
                    "independent_root_voltage_difference_V": float(vh-root),
                })
    short_limit = []
    for applied in (-1.0, 1.0):
        standalone = direct_current(mapping, applied, 1.1)
        for resistance in (1e6, 1e3, 1.0, 0.01):
            current, vh = solve_series_load_line(
                applied, 1.1, lambda v: v/resistance, mapping, nbis=60)
            short_limit.append({
                "applied_V": applied, "hbn_resistance_ohm": resistance,
                "current_A": float(current), "hbn_voltage_V": float(vh),
                "standalone_current_A": standalone,
                "relative_short_limit_error": float(abs(current/standalone-1)),
                "KCL_residual_A": float(current-direct_current(mapping, applied-vh, 1.1)),
            })
    max_residual = max(abs(r["KCL_residual_A"]) for r in comparisons)
    max_root_difference = max(abs(r["independent_root_current_difference_A"]) for r in comparisons)
    if max(max_residual, max_root_difference) > 2e-10:
        raise AssertionError("Circuit identity failed the 2e-10 A numerical gate")
    for applied in (-1.0, 1.0):
        local = [r for r in short_limit if r["applied_V"] == applied]
        if not all(a["relative_short_limit_error"] > b["relative_short_limit_error"]
                   for a, b in zip(local, local[1:])):
            raise AssertionError("Ohmic short limit did not converge to standalone 1T")
    return {"independent_frozen_load_points": comparisons,
            "max_KCL_residual_A": max_residual,
            "max_independent_root_current_difference_A": max_root_difference,
            "ohmic_short_limit": short_limit}


def normal_cycle_checks(mapping, baseline_path=None):
    raw = archived_raw_pairs()
    selected = [row for row in raw if 4 <= row["cycle"] <= 53]
    if [row["cycle"] for row in selected] != list(range(4, 54)):
        raise ValueError("Expected every normal-operation cycle 4--53")
    baseline = np.load(baseline_path) if baseline_path else None
    cache, arrays, records = {}, {}, []
    max_kcl = 0.0
    baseline_max_difference = ({"current_A": 0.0, "applied_voltage_V": 0.0,
                                "hbn_voltage_V": 0.0} if baseline is not None else None)
    passive = {phase: {"n_checked": 0, "n_over_capacity": 0,
                       "max_ratio": 0.0, "worst_point": None}
               for phase in ("set", "reset")}
    for row in selected:
        wave_key = hashlib.sha256(
            row["set_voltage"].tobytes()+row["reset_voltage"].tobytes()).hexdigest()
        if wave_key not in cache:
            cache[wave_key] = M13.simulate_calibrated_cycle(
                row["set_voltage"], row["reset_voltage"])
        prediction = cache[wave_key]
        record = {"cycle_id": row["cycle"], "metrics": {}}
        for phase in ("set", "reset"):
            result = prediction[phase]
            voltage, predicted, vh = (result[key] for key in ("Vs", "Is", "Vhs"))
            observed = row[f"{phase}_current"]
            for key in ("Vs", "Is", "Vhs"):
                name = f"cycle{row['cycle']}_{phase}_{key}"
                arrays[name] = result[key]
                if baseline is not None:
                    np.testing.assert_array_equal(
                        result[key], baseline[name],
                        err_msg=f"Default prediction changed relative to pre-fix baseline: {name}")
                    quantity = {"Is": "current_A", "Vs": "applied_voltage_V", "Vhs": "hbn_voltage_V"}[key]
                    baseline_max_difference[quantity] = max(
                        baseline_max_difference[quantity],
                        float(np.max(np.abs(result[key]-baseline[name]))))
            arrays[f"cycle{row['cycle']}_{phase}_observed_A"] = observed
            record["metrics"][phase] = point_metrics(
                voltage, observed, predicted, CURRENT_FLOORS[phase.upper()])
            for v, h, current in zip(voltage, vh, predicted):
                max_kcl = max(max_kcl, abs(current-direct_current(mapping, v-h, 1.1)))
            # Necessary bound for any passive series h-BN branch, not a fit.
            for index, (v, current) in enumerate(zip(voltage, observed)):
                if not np.isfinite(current) or abs(v) <= 1e-12:
                    continue
                if abs(current) < CURRENT_FLOORS[phase.upper()]:
                    continue
                capacity = abs(direct_current(mapping, v, 1.1))
                ratio = float(abs(current)/capacity)
                item = passive[phase]
                item["n_checked"] += 1
                item["n_over_capacity"] += int(ratio > 1.0+1e-9)
                if ratio > item["max_ratio"]:
                    item["max_ratio"] = ratio
                    item["worst_point"] = {"cycle_id": row["cycle"], "point_index": index,
                                           "applied_V": float(v), "observed_A": float(current),
                                           "standalone_capacity_A": capacity}
        records.append(record)
    pooled = {}
    for phase in ("set", "reset"):
        rows = [row["metrics"][phase] for row in records]
        if any(row["status"] != "ok" for row in rows):
            raise AssertionError(f"Incomplete normal-cycle scoring: {phase}")
        n = sum(row["n_scored_points"] for row in rows)
        pooled[phase] = {"n_points": n, "rmse_decade": float(np.sqrt(sum(
            row["n_scored_points"]*row["rmse_decade"]**2 for row in rows)/n))}
    if max_kcl > 2e-10:
        raise AssertionError("Full-cycle KCL failed the 2e-10 A numerical gate")
    if baseline is not None:
        baseline.close()
    return {"cycle_ids": [r["cycle"] for r in selected],
            "raw_startup_cycle_ids_preserved_but_not_scored": [1, 2, 3],
            "calibration": "unchanged historical full-group default; NO REFIT; descriptive, not held out",
            "history": "v13 predicts separate protocol pairs; no claim of continuous inter-pair state",
            "unique_voltage_grid_pairs": len(cache), "pooled": pooled, "per_cycle": records,
            "max_KCL_residual_A": float(max_kcl),
            "before_after_max_difference": baseline_max_difference,
            "passive_series_capacity_check": passive,
            "capacity_check_caveat": "No experimental uncertainty model; a violation flags transfer/provenance/noise, not proof of active h-BN"}, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"analysis"/"unified_transistor_2026-10-29")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--baseline-source-revision", help="Recorded source revision used to generate the supplied baseline")
    args = parser.parse_args()
    mapping = measured_transistor_mapping()
    inherited = M7.transistor_mapping()
    historical = M3.Params()
    source_paths = [ROOT/"vulcan2d"/"data"/name for name in (
        "fig2b_transistor_lookup.json", "bookoutput2_reverse_transistor_lookup.json")]
    report = {"scope": "Fixed-source terminal constitutive and circuit consistency; no new microscopic mechanism",
              "terminal_convention": "Physical source grounded; Vds=Vd, Vgs=Vg; no source/drain relabel on negative drain voltage",
              "source_metadata": [{k: v for k, v in json.loads(p.read_text()).items() if k != "curves"}
                                  for p in source_paths],
              "comparisons": []}
    for vd in (0.1, 1.0, 2.0):
        report["comparisons"].append({
            "gate_V": 1.1, "drain_magnitude_V": vd,
            "standalone_positive_A": direct_current(mapping, vd, 1.1),
            "canonical_positive_A": mapping.drain_current(vd, 1.1),
            "canonical_negative_A": mapping.drain_current(-vd, 1.1),
            "historical_v7_affine_positive_A": inherited.current(vd, 1.1, side="set"),
            "legacy_default_Params_positive_A": M3.i_tr(vd, "set", historical, 1.0),
            "legacy_default_Params_negative_magnitude_A": M3.i_tr(vd, "reset", historical, 1.0),
        })
    report["legacy_comparison_scope"] = "model.Params() defaults; UI CAL file was not loaded"
    report["circuit_checks"] = circuit_checks(mapping)
    report["normal_cycles"], arrays = normal_cycle_checks(mapping, args.baseline)
    report["baseline_provenance"] = ({
        "prediction_arrays": str(args.baseline),
        "prediction_arrays_sha256": sha256(args.baseline),
        "source_revision_recorded_by_caller": args.baseline_source_revision,
        "reproduction": (
            "Extract `git archive <source_revision> vulcan2d` into a separate temporary directory. "
            "Run that directory's model_v13.simulate_calibrated_cycle on observed_cycle_{4..53}_{set,reset}_voltage "
            "from analysis/full_cycle_validation_audit_v13_curves.npz. Save phase Vs/Is/Vhs under "
            "cycle{index}_{phase}_{key}; caching only identical voltage-grid pairs is equivalent. "
            "Pass this NPZ with --baseline and record the extracted revision with --baseline-source-revision."
        ),
        "calibration": "Historical default, evaluated on identical native voltage grids; no refit",
    } if args.baseline else None)
    report["observation_source"] = {
        "path": str(OBSERVATIONS.relative_to(ROOT)),
        "arrays_used": "observed_cycle_{1..53}_{set,reset}_{voltage,current}; no archived predictions used",
        "availability": "Raw XLSX files are dataless placeholders on this machine; no fresh workbook read was completed",
        "interpretation": "Re-evaluation of archived native measurements, not independent source verification",
    }
    sources = source_paths + [Path(__file__), OBSERVATIONS]
    sources += [ROOT/"vulcan2d"/name for name in (
        "transistor.py", "circuit.py", "model.py", "model_v7.py", "model_v12.py", "model_v13.py", "validate_v13.py", "__init__.py")]
    report["source_sha256"] = {str(p.relative_to(ROOT)): sha256(p) for p in sources}
    report["limitations"] = [
        "Positive and reverse datasets are from different sources; same cell, body/well wiring and exact device match remain unconfirmed.",
        "A shared measured port law is a circuit consistency result, not a microscopic MOS theory or evidence for a unique parasitic mechanism.",
        "The h-BN model still has separate polarity envelopes and empirical SET-to-RESET handoff.",
        "First three cycles are outside the primary normal-operation score by advisor-informed scope; raw records remain untouched.",
        "Current parameters were trained on historical groups including startup data. These numbers are not a new cycles4--53 refit or held-out prediction.",
        "Legacy desktop defaults remain historical and must not be described as the shared research law.",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir/"curves.npz", **arrays)
    path = args.output_dir/"summary.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    print(path)
    print(json.dumps({"pooled": report["normal_cycles"]["pooled"],
                      "KCL_A": report["normal_cycles"]["max_KCL_residual_A"],
                      "before_after": report["normal_cycles"]["before_after_max_difference"],
                      "capacity": report["normal_cycles"]["passive_series_capacity_check"]}, indent=2))


if __name__ == "__main__":
    main()
