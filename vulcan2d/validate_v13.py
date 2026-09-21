"""Audit the v0.12 numerical laws without discarding failed RESET labels.

Training eligibility remains the legacy 51-pair rule.  Prediction/scoring uses
all 53 raw pairs, excluding each target SET and RESET from every fitted law.
The first-35 chronological split is unchanged.  RESET state reconstruction is
a separate, retrospective diagnostic and never supplies prediction inputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from . import cycle_link as CL
from . import validate_full_cycle as VF
from . import validate_v12 as V12
from .reset_identification import identify_pf_reset_state, reconstruct_reset


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_NAME = "full_cycle_validation_audit_v13.json"
CYCLES_NAME = "full_cycle_validation_audit_v13_cycles.csv"
CURVES_NAME = "full_cycle_validation_audit_v13_curves.npz"
MODES = ("same_protocol", "cross_gate_prior")
CURRENT_FLOORS = {"SET": 1e-11, "RESET": 1e-12}
ZERO_BIAS_TOLERANCE_V = 1e-12


def _load_raw_pairs():
    """Keep all paired labels, with no envelope or current-based admission."""
    set_frames = pd.read_excel(ROOT / VF.SET_FILE, sheet_name=None)
    reset_frames = pd.read_excel(ROOT / VF.RESET_FILE, sheet_name=None)
    rows = []
    for index, set_sheet, reset_sheet in CL.pair_cycle_sheets(
            set_frames, reset_frames):
        row = {"cycle": index, "set_sheet": set_sheet,
               "reset_sheet": reset_sheet}
        for phase, frame in (("set", set_frames[set_sheet]),
                             ("reset", reset_frames[reset_sheet])):
            voltage = frame.iloc[:, 0].to_numpy(float)
            current = frame.iloc[:, 1].to_numpy(float)
            # The simulation grid must not depend on target current values.
            finite_voltage = np.isfinite(voltage)
            row[f"{phase}_voltage"] = voltage[finite_voltage]
            row[f"{phase}_current"] = current[finite_voltage]
            row[f"{phase}_raw_rows"] = len(voltage)
            row[f"{phase}_nonfinite_voltage_rows"] = int(
                np.count_nonzero(~finite_voltage))
        rows.append(row)
    return rows


def _select_training(named_set, accepted_records, reset_by_id,
                     target_cycle=None, last_training_cycle=None):
    """Select by cycle identity, including when the target was rejected."""
    def include(index):
        return (index != target_cycle
                and (last_training_cycle is None
                     or index <= last_training_cycle))

    training_set = [
        item for item in named_set
        if include(int(item[0].upper().replace("S", "")))
    ]
    training_records = [
        row for row in accepted_records if include(row["cycle"])
    ]
    training_resets = [reset_by_id[row["cycle"]]
                       for row in training_records]
    set_ids = [int(name.upper().replace("S", ""))
               for name, _ in training_set]
    reset_ids = [row["cycle"] for row in training_records]
    return training_set, training_records, training_resets, {
        "training_SET_cycle_ids": set_ids,
        "training_RESET_cycle_ids": reset_ids,
        "training_SET_cycles": len(set_ids),
        "training_paired_RESET_cycles": len(reset_ids),
    }


def _simulate(*args, **kwargs):
    # Lazy import also lets the validation helpers be tested independently.
    from . import model_v13 as M13
    return M13.run_autonomous_full_cycle(*args, **kwargs)


def _run(row, bundle, base_params, reset_mapping, mode):
    if mode not in MODES:
        raise ValueError("Unsupported RESET transition mode")
    transition = (bundle["reset_transition"]
                  if mode == "same_protocol" else None)
    return _simulate(
        row["set_voltage"], row["reset_voltage"], VF.GATE_VOLTAGE,
        bundle["set_transport"], bundle["set_transition"], bundle["link"],
        bundle["endpoint"], bundle["reset_transport"],
        reset_base_params=base_params, reset_mapping=reset_mapping,
        reset_transition_law=transition,
    )


def _point_metrics(voltage, observed, predicted, current_floor,
                   branch="all"):
    """Expose invalid predictions instead of reporting their masked score."""
    voltage = np.asarray(voltage, dtype=float)
    observed = np.abs(np.asarray(observed, dtype=float))
    predicted = np.abs(np.asarray(predicted, dtype=float))
    if (voltage.ndim != 1 or observed.shape != voltage.shape
            or predicted.shape != voltage.shape):
        raise ValueError("Scoring requires aligned one-dimensional arrays")
    selected = np.ones(len(voltage), dtype=bool)
    if branch != "all":
        if branch not in ("forward", "return"):
            raise ValueError("Unsupported branch")
        apex = int(np.argmax(np.abs(voltage)))
        selected[:] = False
        selected[:apex + 1] = branch == "forward"
        if branch == "return":
            selected[apex:] = True
    observed_valid = np.isfinite(observed) & np.isfinite(voltage)
    zero_bias = (selected & np.isfinite(voltage)
                 & (np.abs(voltage) <= ZERO_BIAS_TOLERANCE_V))
    zero_observed = observed[zero_bias & observed_valid]
    eligible = (selected & ~zero_bias & observed_valid
                & (observed >= current_floor))
    prediction_valid = np.isfinite(predicted) & (predicted > 0.0)
    scoreable = eligible & prediction_valid
    invalid = eligible & ~prediction_valid
    residual = np.log10(predicted[scoreable]) - np.log10(observed[scoreable])
    n_invalid = int(np.count_nonzero(invalid))
    n_nonfinite_predictions = int(np.count_nonzero(
        selected & ~np.isfinite(predicted)))
    status = ("invalid_prediction" if n_invalid or n_nonfinite_predictions else
              "ok" if len(residual) else "no_scoreable_observations")
    result = {
        "status": status,
        "current_floor_A": float(current_floor),
        "n_points": int(np.count_nonzero(selected)),
        "zero_bias_tolerance_V": ZERO_BIAS_TOLERANCE_V,
        "n_zero_bias_points": int(np.count_nonzero(zero_bias)),
        "zero_bias_observed_abs_current_median_A": (
            float(np.median(zero_observed)) if len(zero_observed) else None),
        "zero_bias_observed_abs_current_max_A": (
            float(np.max(zero_observed)) if len(zero_observed) else None),
        "n_nonfinite_observed": int(np.count_nonzero(
            selected & ~observed_valid)),
        "n_below_current_floor": int(np.count_nonzero(
            selected & ~zero_bias & observed_valid & (observed < current_floor))),
        "n_eligible_points": int(np.count_nonzero(eligible)),
        "n_scored_points": int(len(residual)),
        "n_invalid_predictions_at_eligible_points": n_invalid,
        "n_nonfinite_predictions_all_points": n_nonfinite_predictions,
        "rmse_decade": None,
        "median_absolute_decade": None,
        "median_signed_decade": None,
    }
    if status == "ok":
        result.update({
            "rmse_decade": float(np.sqrt(np.mean(residual ** 2))),
            "median_absolute_decade": float(np.median(np.abs(residual))),
            "median_signed_decade": float(np.median(residual)),
        })
    return result


def _prediction_record(row, prediction, split, mode, provenance,
                       eligible_ids, rejection_reasons):
    index = row["cycle"]
    record = {
        "split": split,
        "cycle_id": index,
        "set_sheet": row["set_sheet"],
        "reset_sheet": row["reset_sheet"],
        "reset_transition_mode": mode,
        "legacy_training_eligible": index in eligible_ids,
        "legacy_rejection_reason": rejection_reasons.get(index),
        **provenance,
        "target_excluded": (
            index not in provenance["training_SET_cycle_ids"]
            and index not in provenance["training_RESET_cycle_ids"]),
        "status": "ok",
        "reason": None,
        "metrics": {},
        "handoff": prediction.get("handoff", {}),
    }
    if not record["target_excluded"]:
        raise ValueError("Target cycle leaked into training")
    for phase in ("SET", "RESET"):
        key = phase.lower()
        voltage = row[f"{key}_voltage"]
        if not np.array_equal(prediction[key]["Vs"], voltage):
            raise ValueError("Prediction voltage grid differs from target grid")
        record["metrics"][phase] = {
            branch: _point_metrics(
                voltage, row[f"{key}_current"], prediction[key]["Is"],
                CURRENT_FLOORS[phase], branch)
            for branch in ("all", "forward", "return")
        }
    statuses = [value["status"]
                for phase in record["metrics"].values()
                for value in phase.values()]
    nonfinite = any(
        phase["all"]["n_nonfinite_predictions_all_points"]
        for phase in record["metrics"].values())
    if nonfinite or "invalid_prediction" in statuses:
        record["status"] = "invalid_prediction"
        record["reason"] = "Nonfinite prediction or nonpositive eligible current"
    elif any(status != "ok" for status in statuses):
        record["status"] = "incomplete_scoring"
        record["reason"] = "At least one branch has no scoreable observations"
    return record


def _failed_record(row, split, mode, provenance, eligible_ids,
                   rejection_reasons, status, error):
    index = row["cycle"]
    return {
        "split": split, "cycle_id": index,
        "set_sheet": row["set_sheet"], "reset_sheet": row["reset_sheet"],
        "reset_transition_mode": mode,
        "legacy_training_eligible": index in eligible_ids,
        "legacy_rejection_reason": rejection_reasons.get(index),
        **provenance,
        "target_excluded": (
            index not in provenance["training_SET_cycle_ids"]
            and index not in provenance["training_RESET_cycle_ids"]),
        "status": status, "reason": f"{type(error).__name__}: {error}",
        "metrics": {}, "handoff": getattr(error, "diagnostics", {}),
    }


def _evaluate(raw_pairs, named_set, accepted_records, reset_by_id,
              base_params, reset_mapping, rejection_reasons,
              last_training_cycle=None):
    split = ("leave_one_cycle_out" if last_training_cycle is None
             else "chronological_first35_to_later")
    eligible_ids = {row["cycle"] for row in accepted_records}
    targets = [row for row in raw_pairs if last_training_cycle is None
               or row["cycle"] > last_training_cycle]
    fixed_bundle, fixed_provenance = None, None
    if last_training_cycle is not None:
        training = _select_training(
            named_set, accepted_records, reset_by_id,
            last_training_cycle=last_training_cycle)
        fixed_provenance = training[3]
        try:
            fixed_bundle = V12._fit_bundle(*training[:3], base_params)
        except (ValueError, RuntimeError, FloatingPointError) as error:
            fixed_provenance = {
                **fixed_provenance, "reference_calibration_status": "failed",
                "reference_calibration_reason": f"{type(error).__name__}: {error}",
            }
            failures = [_failed_record(
                row, split, mode, fixed_provenance, eligible_ids,
                rejection_reasons, "calibration_failed", error)
                for row in targets for mode in MODES]
            return failures, {}, None, fixed_provenance
        fixed_provenance = {
            **fixed_provenance, "reference_calibration_status": "ok",
            "reference_calibration_reason": None,
        }
    records, predictions = [], {}
    for position, row in enumerate(targets):
        if fixed_bundle is None:
            training = _select_training(
                named_set, accepted_records, reset_by_id,
                target_cycle=row["cycle"])
            provenance = training[3]
            try:
                bundle = V12._fit_bundle(*training[:3], base_params)
            except (ValueError, RuntimeError, FloatingPointError) as error:
                records.extend(_failed_record(
                    row, split, mode, provenance, eligible_ids,
                    rejection_reasons, "calibration_failed", error)
                    for mode in MODES)
                continue
        else:
            bundle, provenance = fixed_bundle, fixed_provenance
        for mode in MODES:
            try:
                prediction = _run(row, bundle, base_params, reset_mapping, mode)
                record = _prediction_record(
                    row, prediction, split, mode, provenance,
                    eligible_ids, rejection_reasons)
                predictions[(split, row["cycle"], mode)] = prediction
            except (ValueError, RuntimeError, FloatingPointError) as error:
                record = _failed_record(
                    row, split, mode, provenance, eligible_ids,
                    rejection_reasons, "prediction_failed", error)
            records.append(record)
        if (position + 1) % 10 == 0 or position + 1 == len(targets):
            print(f"v0.13 {split} {position + 1}/{len(targets)}", flush=True)
    return records, predictions, fixed_bundle, fixed_provenance


def _group_ids(raw_pairs, eligible_ids):
    all_ids = [row["cycle"] for row in raw_pairs]
    return {
        "all_pairs": all_ids,
        "legacy_accepted": [index for index in all_ids if index in eligible_ids],
        "legacy_rejected": [index for index in all_ids if index not in eligible_ids],
    }


def _summarize_group(cycle_ids, split, records, predictions, raw_by_id):
    result = {"cycle_ids": cycle_ids, "n_pairs": len(cycle_ids), "modes": {}}
    for mode in MODES:
        selected = [row for row in records
                    if row["split"] == split
                    and row["reset_transition_mode"] == mode
                    and row["cycle_id"] in cycle_ids]
        successful = [row["cycle_id"] for row in selected if row["status"] == "ok"]
        failures = [{"cycle_id": row["cycle_id"], "status": row["status"],
                     "reason": row["reason"]}
                    for row in selected if row["status"] != "ok"]
        missing = sorted(set(cycle_ids) - {row["cycle_id"] for row in selected})
        complete = len(successful) == len(cycle_ids) and not failures and not missing
        item = {
            "status": "empty" if not cycle_ids else "ok" if complete else "incomplete",
            "n_successful_pairs": len(successful), "failed_pairs": failures,
            "missing_cycle_ids": missing, "scores": None,
        }
        # Do not silently substitute an easier subset when prediction fails.
        if cycle_ids and complete:
            actual_set, predicted_set, actual_reset, predicted_reset = [], [], [], []
            for index in cycle_ids:
                row = raw_by_id[index]
                predicted = predictions[(split, index, mode)]
                for phase, observed_curves, predicted_curves in (
                        ("set", actual_set, predicted_set),
                        ("reset", actual_reset, predicted_reset)):
                    voltage = row[f"{phase}_voltage"]
                    current = row[f"{phase}_current"]
                    # Missing labels and predefined zero-bias points are
                    # omitted only at scoring time, independently of prediction.
                    finite = (np.isfinite(current) & np.isfinite(voltage)
                              & (np.abs(voltage) > ZERO_BIAS_TOLERANCE_V))
                    observed_curves.append((voltage[finite], current[finite]))
                    predicted_curves.append((
                        np.asarray(predicted[phase]["Vs"])[finite],
                        np.asarray(predicted[phase]["Is"])[finite]))
            item["scores"] = V12._score(
                actual_set, predicted_set, actual_reset, predicted_reset)
        result["modes"][mode] = item
    return result


def _retrospective_diagnostics(raw_pairs, reference_bundle,
                               base_params, reset_mapping,
                               reference_failure_reason=None):
    diagnostics = []
    for row in raw_pairs:
        item = {
            "cycle_id": row["cycle"],
            "reference": "first35_training_transport",
            "uses_measured_target_current": True,
            "used_for_prediction": False,
            "status": "identified", "reason": None,
            "identified_e50_mvc": None, "identified_width_mvc": None,
            "state_fit_rms": None, "reconstruction_valid_fraction": None,
        }
        if reference_bundle is None:
            item["status"] = "reference_unavailable"
            item["reason"] = (
                "First-35 training reference unavailable: "
                + (reference_failure_reason or "calibration did not return a bundle"))
            diagnostics.append(item)
            continue
        try:
            cycle = reconstruct_reset(
                row["reset_voltage"], row["reset_current"],
                VF.GATE_VOLTAGE, reset_mapping)
            item["reconstruction_valid_fraction"] = float(np.mean(cycle.valid))
            state = identify_pf_reset_state(
                cycle, base_params.reset_alpha_lrs_identified,
                reference_bundle["reset_transport"])
            item.update({
                "identified_e50_mvc": float(state.e50_mvc),
                "identified_width_mvc": float(state.width_mvc),
                "state_fit_rms": float(state.state_fit_rms),
            })
        except (ValueError, RuntimeError, FloatingPointError) as error:
            item["status"] = "not_identifiable"
            item["reason"] = f"{type(error).__name__}: {error}"
        diagnostics.append(item)
    return diagnostics


def _diagnostic_summary(diagnostics, cycle_ids):
    selected = [row for row in diagnostics if row["cycle_id"] in cycle_ids]
    identified = [row for row in selected if row["status"] == "identified"]
    result = {
        "cycle_ids": cycle_ids, "n_attempted": len(selected),
        "n_identified": len(identified),
        "not_identifiable": [{"cycle_id": row["cycle_id"], "reason": row["reason"]}
                             for row in selected if row["status"] != "identified"],
    }
    for field in ("identified_e50_mvc", "identified_width_mvc", "state_fit_rms"):
        values = [row[field] for row in identified]
        result[f"{field}_q25_median_q75"] = (
            [float(value) for value in np.quantile(values, [0.25, 0.5, 0.75])]
            if values else None)
    return result


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_metadata():
    def git(*args):
        result = subprocess.run(["git", *args], cwd=ROOT, text=True,
                                capture_output=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    paths = [VF.SET_FILE, VF.RESET_FILE, *VF.TRAIN_RESET_FILES.values(),
             "vulcan2d/data/fig2b_transistor_lookup.json",
             "vulcan2d/data/bookoutput2_reverse_transistor_lookup.json"]
    source_paths = sorted(str(path.relative_to(ROOT))
                          for path in (ROOT / "vulcan2d").glob("*.py"))
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "working_tree_dirty": bool(git("status", "--porcelain")),
        "input_sha256": {path: _sha256(ROOT / path) for path in paths},
        "audit_source_sha256": {path: _sha256(ROOT / path)
                                for path in source_paths if (ROOT / path).exists()},
    }


def _save_curves(path, raw_pairs, predictions):
    arrays = {}
    for row in raw_pairs:
        for phase in ("set", "reset"):
            for field in ("voltage", "current"):
                arrays[f"observed_cycle_{row['cycle']}_{phase}_{field}"] = (
                    row[f"{phase}_{field}"])
    for (split, index, mode), prediction in predictions.items():
        for phase in ("set", "reset"):
            prefix = f"{split}_{mode}_cycle_{index}_{phase}"
            arrays[f"{prefix}_voltage"] = prediction[phase]["Vs"]
            arrays[f"{prefix}_current"] = prediction[phase]["Is"]
    np.savez_compressed(path, **arrays)


def _csv_rows(records, diagnostics):
    diagnostic_by_id = {row["cycle_id"]: row for row in diagnostics}
    for record in records:
        flat = {key: value for key, value in record.items()
                if key not in ("metrics", "handoff")}
        for key, value in list(flat.items()):
            if isinstance(value, list):
                flat[key] = json.dumps(value)
        for phase, branches in record["metrics"].items():
            for branch, metrics in branches.items():
                flat.update({f"{phase}_{branch}_{key}": value
                             for key, value in metrics.items()})
        for key in ("set_end_state", "reference_current_relative_error"):
            flat[f"handoff_{key}"] = record["handoff"].get(key)
        flat.update({f"diagnostic_{key}": value
                     for key, value in diagnostic_by_id[record["cycle_id"]].items()
                     if key != "cycle_id"})
        yield flat


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis")
    args = parser.parse_args(argv)
    base_params, reset_mapping = VF._fit_reset_parameters()
    accepted, rejected = VF._load_paired_cycles(base_params, reset_mapping)
    raw_pairs = _load_raw_pairs()
    raw_by_id = {row["cycle"]: row for row in raw_pairs}
    eligible_ids = {row["cycle"] for row in accepted}
    if not eligible_ids <= raw_by_id.keys():
        raise ValueError("Training cycles missing from raw scoring data")
    reasons = {row["cycle"]: row["reason"] for row in rejected}
    named_set, _ = V12._load_set_cycles()
    reset_by_id = dict(zip(
        [row["cycle"] for row in accepted],
        V12._reconstruct_resets(accepted, reset_mapping)))
    loo, loo_predictions, _, _ = _evaluate(
        raw_pairs, named_set, accepted, reset_by_id,
        base_params, reset_mapping, reasons)
    chronological, chronological_predictions, reference, provenance = _evaluate(
        raw_pairs, named_set, accepted, reset_by_id,
        base_params, reset_mapping, reasons, last_training_cycle=35)
    records = loo + chronological
    predictions = {**loo_predictions, **chronological_predictions}
    groups = _group_ids(raw_pairs, eligible_ids)
    diagnostics = _retrospective_diagnostics(
        raw_pairs, reference, base_params, reset_mapping,
        reference_failure_reason=provenance.get("reference_calibration_reason"))
    diagnostic_groups = {
        f"{name}_{period}": _diagnostic_summary(
            diagnostics, [index for index in indices
                          if (index <= 35) == (period == "early")])
        for name, indices in groups.items() for period in ("early", "late")
    }
    early = diagnostic_groups["legacy_accepted_early"][
        "identified_e50_mvc_q25_median_q75"]
    late = diagnostic_groups["legacy_accepted_late"][
        "identified_e50_mvc_q25_median_q75"]
    report = {
        "audit_version": "0.13.0",
        "numerical_calibration_laws": "v0.12; v0.13 explicit SET/RESET handoff",
        "provenance": _source_metadata(),
        "scope": "1.1-V gate; recorded +5-V SET and -1.7-V RESET DC protocols",
        "validation_contract": {
            "training_eligibility": "unchanged legacy separable-envelope rule",
            "scoring_eligibility": "all paired raw cycles; no envelope rejection",
            "target_currents_enter_simulator": False,
            "target_currents_used_for_scoring_and_retrospective_diagnostics": True,
            "model_selection_is_retrospective": True,
            "new_independent_test_data": False,
            "current_floors_A": CURRENT_FLOORS,
            "zero_bias_policy": {
                "excluded_from_log_scoring_abs_voltage_le_V": ZERO_BIAS_TOLERANCE_V,
                "selection_depends_on_prediction": False,
                "observed_offset_reported_separately": True,
                "interpretation": (
                    "The static model has I(0)=0. Measured zero-bias offset is "
                    "reported as absolute-current median/max, not fitted or log-scored. "
                    "Nonfinite predictions remain invalid at every voltage."),
            },
            "invalid_prediction_policy": (
                "report failed/invalid cycles; aggregate scores null for incomplete groups"),
            "median_loop_definition": (
                "RMSE between measured and predicted pointwise median curves; "
                "not the median individual-cycle RMSE"),
            "pointwise_branch_definition": (
                "native voltage points; apex belongs to forward and return; "
                "whole-cycle score counts every point once"),
        },
        "data": {
            "all_pairs": len(raw_pairs), "legacy_accepted_pairs": len(accepted),
            "legacy_rejected_pairs": rejected,
            "groups": groups,
        },
        "leave_one_cycle_out": {
            name: _summarize_group(indices, "leave_one_cycle_out", records,
                                   predictions, raw_by_id)
            for name, indices in groups.items()
        },
        "chronological_first35_to_later": {
            "last_training_cycle": 35, **provenance,
            "later_test_pairs": sum(row["cycle"] > 35 for row in raw_pairs),
            "groups": {
                name: _summarize_group(
                    [index for index in indices if index > 35],
                    "chronological_first35_to_later", records, predictions, raw_by_id)
                for name, indices in groups.items()
            },
        },
        "retrospective_RESET_diagnostics": {
            "reference": "first35_training_transport", **provenance,
            "uses_measured_target_current": True, "used_for_prediction": False,
            "interpretation": (
                "Observable-state reconstruction with a fixed early-training transport "
                "law; descriptive drift audit, not a predicted microscopic threshold"),
            "groups": diagnostic_groups,
            "legacy_accepted_late_minus_early_median_e50_mvc": (
                late[1] - early[1] if early is not None and late is not None else None),
            "cycles": diagnostics,
        },
        "cycle_predictions": records,
        "curve_archive": {
            "filename": CURVES_NAME,
            "observed_key": "observed_cycle_{cycle_id}_{set|reset}_{voltage|current}",
            "predicted_key": (
                "{split}_{reset_transition_mode}_cycle_{cycle_id}_"
                "{set|reset}_{voltage|current}"),
            "contains": "all raw observed pairs and successfully returned predictions",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / SUMMARY_NAME
    summary_path.write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                       allow_nan=False), encoding="utf-8")
    csv_rows = list(_csv_rows(records, diagnostics))
    fieldnames = list(dict.fromkeys(key for row in csv_rows for key in row))
    csv_path = args.output_dir / CYCLES_NAME
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    _save_curves(args.output_dir / CURVES_NAME, raw_pairs, predictions)
    print(f"wrote {summary_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {args.output_dir / CURVES_NAME}")
    return report


if __name__ == "__main__":
    main()
