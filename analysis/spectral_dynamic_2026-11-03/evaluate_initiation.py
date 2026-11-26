"""Frozen terminal-fit full32 training audit; never selects on held-out data.

Refuses running fit artifacts, preserves the best recorded training parameter
vector, evaluates its actual full32 trajectory, and reports every predeclared
primary feature failure. Lower total cost does not override failed curve gates.
An optional table path supports later numerical refinement at the same q.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from joint_problem import SpectralJointProblem, ROOT, HERE, TRAIN
from feature_constraints import branch_indices, read_at, FLOORS, LIMIT_MASK_A
from vulcan2d.shared_spectral import SpectralShape, SpectralLaw, DEFAULT_SHAPE_PATH
from vulcan2d.shared_spectral_ivp import SpectralTable, TabulatedSpectralLaw

FEATURES = ROOT/"analysis/constrained_shared_2026-11-03/feature_constraints.json"
DEFAULT_TABLE = HERE/"transport_g010_v0025.npz"
GAP_BASELINES = ROOT/"analysis/shared_gap_refinement_2026-11-03"
FINISHED_STATUS = "bounded_training_trial_finished_requires_full32_validation"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_finished_fit(path):
    """Fail before any evaluation if the optimization snapshot is not final."""
    payload = json.loads(Path(path).read_text())
    if payload.get("status") != FINISHED_STATUS:
        raise ValueError("Fit is running, incomplete or not the declared terminal trial; evaluation refused")
    if any(row.get("status") == "running" or any(call.get("status") == "running" for call in row.get("calls", []))
           for row in payload.get("runs", [])):
        raise ValueError("Terminal fit contains unfinished running calls; evaluation refused")
    best = payload.get("best_training")
    if not isinstance(best, dict) or "q" not in best or "cost" not in best:
        raise ValueError("Terminal fit has no valid best_training candidate")
    q = np.asarray(best["q"], dtype=float)
    if q.shape != (8,) or not np.all(np.isfinite(q)) or not np.isfinite(best["cost"]):
        raise ValueError("Best training candidate is not a finite declared eight-parameter vector")
    matching = [call for row in payload.get("runs", []) if row.get("start") == best.get("start")
                for call in row.get("calls", []) if call.get("call") == best.get("call")]
    if len(matching) != 1 or matching[0].get("status") != "valid":
        raise ValueError("best_training does not identify one completed valid objective call")
    if not np.array_equal(np.asarray(matching[0]["q"]), q) or matching[0]["cost"] != best["cost"]:
        raise ValueError("best_training and its recorded objective call disagree")
    if not payload.get("source_sha256"):
        raise ValueError("Terminal fit lacks frozen source hashes")
    return payload


def load_training_predictions(path, problem):
    """Read only declared training record arrays from a fit-produced NPZ."""
    predictions = {}
    with np.load(path, allow_pickle=False) as archive:
        legacy_gap_schema = "1r_cycle1_set_voltage" in archive.files
        for top, ids in TRAIN.items():
            for record_index, cid in enumerate(ids):
                for phase in ("set", "reset"):
                    prefix = f"{top}_cycle{cid}_{phase}_" if legacy_gap_schema else f"{top}_{cid}_{phase}_"
                    trace = {}
                    for key in ("command_voltage_v", "supply_voltage_v", "hbn_voltage_v",
                                "transistor_voltage_v", "current_a", "compliance_drop_v",
                                "compliance_limited", "state_u"):
                        if prefix+key in archive.files:
                            trace[key] = archive[prefix+key].copy()
                    if legacy_gap_schema:
                        trace["command_voltage_v"] = archive[prefix+"voltage"].copy()
                        stored_observed = abs(archive[prefix+"observed_current_a"])
                        if not np.array_equal(stored_observed, problem.raw_training[top, phase][record_index]):
                            raise ValueError(f"Baseline stored training observations differ for {prefix}")
                    if not {"current_a", "state_u", "command_voltage_v"} <= set(trace):
                        raise ValueError(f"Missing native current/state/command for {prefix}")
                    command = problem.waveforms[top, phase]
                    if not np.array_equal(trace["command_voltage_v"], command):
                        raise ValueError(f"Stored prediction native waveform differs for {prefix}")
                    if trace["current_a"].shape != command.shape or trace["state_u"].shape != command.shape:
                        raise ValueError(f"Stored prediction shape differs for {prefix}")
                    if not np.all(np.isfinite(trace["state_u"])):
                        raise ValueError(f"Stored state contains nonfinite entries for {prefix}")
                    trace["final_u"] = float(trace["state_u"][-1])
                    predictions[top, cid, phase] = trace
    return predictions


def score_cost(problem, predictions):
    residual, blocks = problem.score_predictions(predictions)
    return float(np.dot(residual, residual)/2.), residual, blocks


def phase_metrics(problem, predictions):
    """Original-record plain log errors; masks use observed data, never predictions."""
    result = {}
    for top, ids in TRAIN.items():
        for phase in ("set", "reset"):
            v, observed = problem.waveforms[top, phase], problem.raw_training[top, phase]
            predicted = abs(np.asarray([predictions[top, cid, phase]["current_a"] for cid in ids]))
            eligible = np.isfinite(observed) & (observed >= FLOORS[phase]) & (abs(v)[None, :] > 1e-12)
            outward = np.arange(len(v)) <= np.argmax(abs(v))
            masks = {"all_eligible": eligible, "outward": eligible & outward,
                     "return": eligible & ~outward}
            if (top, phase) == ("1r", "set"):
                nonlimited = observed < LIMIT_MASK_A
                masks.update(nonlimited=eligible & nonlimited,
                             nonlimited_outward=eligible & nonlimited & outward,
                             nonlimited_return=eligible & nonlimited & ~outward,
                             limited=eligible & ~nonlimited)
            groups = {}
            for name, mask in masks.items():
                n = int(mask.sum())
                if not n:
                    groups[name] = {"n": 0, "plain_RMSE_decade": None}
                    continue
                if np.any(~np.isfinite(predicted[mask])) or np.any(predicted[mask] <= 0.):
                    raise ValueError("Invalid prediction at an eligible training observation")
                logs_observed, logs_predicted = np.zeros_like(observed), np.zeros_like(predicted)
                logs_observed[mask], logs_predicted[mask] = np.log10(observed[mask]), np.log10(predicted[mask])
                errors = logs_predicted[mask]-logs_observed[mask]
                count = mask.sum(axis=0)
                mo = logs_observed.sum(axis=0)/np.maximum(count, 1)
                mp = logs_predicted.sum(axis=0)/np.maximum(count, 1)
                do, dp = np.where(mask, logs_observed-mo, 0.), np.where(mask, logs_predicted-mp, 0.)
                bias_mse = float(np.sum(count*(mp-mo)**2)/n)
                observed_variance = float(np.sum(do*do)/n)
                predicted_variance = float(np.sum(dp*dp)/n)
                covariance = float(np.sum(do*dp)/n)
                mse = float(np.mean(errors**2))
                per_record = [{"record": cid, "n": int(row.sum()),
                               "plain_RMSE_decade": float(np.sqrt(np.mean((np.log10(predicted[k, row])-np.log10(observed[k, row]))**2))) if np.any(row) else None}
                              for k, (cid, row) in enumerate(zip(ids, mask))]
                groups[name] = {"n": n, "plain_RMSE_decade": math.sqrt(mse),
                    "mean_signed_log_error_decade": float(np.mean(errors)),
                    "max_abs_log_error_decade": float(np.max(abs(errors))),
                    "mean_curve_bias_RMS_decade": math.sqrt(bias_mse),
                    "empirical_observation_spread_RMS_decade": math.sqrt(observed_variance),
                    "prediction_spread_RMS_decade": math.sqrt(predicted_variance),
                    "observation_prediction_covariance_decade2": covariance,
                    "MSE_decomposition_residual": mse-(bias_mse+observed_variance+predicted_variance-2*covariance),
                    "per_record": per_record}
            result[f"{top}_{phase}"] = {"groups": groups,
                "n_raw_points": int(observed.size), "n_zero_voltage_points": int(len(ids)*np.sum(abs(v) <= 1e-12)),
                "n_observations_below_historical_floor": int(np.sum(observed < FLOORS[phase])),
                "historical_scoring_floor_A": FLOORS[phase],
                "scope": "Plain original-record log RMSE with unchanged observed-data masks; no flooring of measured or predicted current. Scatter is descriptive, not irreducible noise."}
    return result


def predicted_feature(identifier, feature, voltage, current):
    """Use the frozen feature definitions and their shared native-grid helpers."""
    v, i = np.asarray(voltage), abs(np.asarray(current))
    branch = feature.get("branch", "outward")
    if np.any(~np.isfinite(i)):
        raise ValueError("Nonfinite predicted current")
    if "magnitude_V" in feature:
        value, index = read_at(v, i, feature["magnitude_V"], branch)
        return value, {"native_index": index}
    if ".alpha_" in identifier:
        a, b = feature["window_abs_V"]
        ia, ja = read_at(v, i, a, branch); ib, jb = read_at(v, i, b, branch)
        if min(ia, ib) <= 0.:
            raise ValueError("Nonpositive current in trajectory slope")
        return float(np.log(ib/ia)/np.log(abs(v[jb]/v[ja]))), {"indices": [ja, jb]}
    if ".geomean_current_" in identifier or ".log_current_q90_q10_span_" in identifier:
        a, b = feature["window_abs_V"]
        indices = branch_indices(v, branch)
        indices = indices[(abs(v[indices]) >= a-1e-9) & (abs(v[indices]) <= b+1e-9)]
        if not len(indices) or np.any(i[indices] <= 0.):
            raise ValueError("Empty or nonpositive model plateau window")
        logs = np.log10(i[indices])
        value = 10.**logs.mean() if ".geomean_current_" in identifier else np.quantile(logs, .9)-np.quantile(logs, .1)
        return float(value), {"indices": indices.tolist()}
    if identifier.endswith(".peak_current") or identifier.endswith(".peak_abs_voltage"):
        indices = branch_indices(v, "outward")
        j = int(indices[np.argmax(i[indices])])
        value = i[j] if identifier.endswith(".peak_current") else abs(v[j])
        return float(value), {"native_index": j, "endpoint_maximum": j == int(np.argmax(abs(v)))}
    if "current_threshold_A" in feature:
        indices = branch_indices(v, "outward")
        hits = np.flatnonzero(i[indices] >= feature["current_threshold_A"])
        if not len(hits):
            raise ValueError("Declared current threshold is never reached")
        k = int(hits[0]); j = int(indices[k]); previous = int(indices[max(k-1, 0)])
        return float((v[previous]+v[j])/2.), {"native_index": j,
            "last_below_voltage_V": float(v[previous]), "first_crossing_voltage_V": float(v[j]),
            "bracket_width_V": float(v[j]-v[previous])}
    if identifier.endswith(".R_0p05V"):
        current05, j = read_at(v, i, .05, "return")
        if current05 <= 0.:
            raise ValueError("Zero current makes read resistance undefined")
        return float(abs(v[j])/current05), {"native_index": j}
    if ".R_fit_" in identifier:
        low, high = feature["records"][0]["window_abs_V"]
        indices = branch_indices(v, "outward")
        indices = indices[(abs(v[indices]) >= low-3e-7) & (abs(v[indices]) <= high+3e-7)]
        denominator = float(np.dot(abs(v[indices]), i[indices]))
        if denominator <= 0.:
            raise ValueError("Nonpositive low-voltage fitted conductance")
        return float(np.dot(v[indices], v[indices])/denominator), {"indices": indices.tolist()}
    if identifier.endswith(".R_25mV_over_R_50mV"):
        i25, j25 = read_at(v, i, .025, "outward"); i50, j50 = read_at(v, i, .05, "outward")
        if min(i25, i50) <= 0.:
            raise ValueError("Nonpositive initial RESET current")
        return float((abs(v[j25])/i25)/(abs(v[j50])/i50)), {"indices": [j25, j50]}
    raise ValueError(f"Unsupported frozen feature definition: {identifier}")


def feature_gates(problem, predictions, constraints):
    primary = constraints["proposed_success_criteria_before_next_fit"]["primary_feature_ids"]
    extras = ["1r.reset.outward.R_fit_0p025_0p2V_sensitivity", "1r.reset.outward.R_25mV_over_R_50mV"]
    output = {}
    for identifier in primary+extras:
        feature = constraints["features"][identifier]
        top, phase = feature["topology"], feature["phase"]
        rows = []
        for cid in TRAIN[top]:
            try:
                value, metadata = predicted_feature(identifier, feature, problem.waveforms[top, phase],
                                                     predictions[top, cid, phase]["current_a"])
                if not np.isfinite(value):
                    raise ValueError("Nonfinite model feature")
                rows.append({"record": cid, "value": value, "valid": True, **metadata})
            except ValueError as exc:
                rows.append({"record": cid, "value": None, "valid": False, "failure": str(exc)})
        values = [row["value"] for row in rows if row["valid"]]
        median = float(np.median(values)) if values else None
        low, high = feature["proposed_empirical_acceptance_interval"]
        valid_all = len(values) == len(rows)
        passed = bool(valid_all and low <= median <= high)
        output[identifier] = {"primary_gate": identifier in primary,
            "description": feature["description"], "unit": feature["unit"],
            "fixed_acceptance_interval": [low, high], "acceptance_mode": feature["acceptance_mode"],
            "model_median": median, "model_min_max": [min(values), max(values)] if values else None,
            "observed_training_median": feature["distribution"]["median"],
            "observed_full_eligible_min_max": [feature["distribution"]["min"], feature["distribution"]["max"]],
            "observed_all_raw_record_min_max": [min(row["value"] for row in feature["records"]),
                                                max(row["value"] for row in feature["records"])],
            "passed": passed, "n_invalid_model_records": len(rows)-len(values), "records": rows,
            "scope": "Every declared model training record is retained. Undefined/never-crossing model features fail rather than being censored away. Fixed empirical bands are not confidence intervals."}
    failed = [name for name in primary if not output[name]["passed"]]
    return {"n_primary": len(primary), "n_primary_passed": len(primary)-len(failed),
            "all_primary_passed": not failed, "failed_primary_ids": failed, "features": output,
            "noncompensation_rule": "A lower aggregate loss cannot compensate any failed predeclared primary feature."}


def compare_predictions(problem, full, stored):
    rows, max_current, max_state, max_endpoint = {}, 0., 0., 0.
    for top, ids in TRAIN.items():
        for phase in ("set", "reset"):
            current, state, endpoint = 0., 0., 0.
            for cid in ids:
                a, b = full[top, cid, phase], stored[top, cid, phase]
                current = max(current, problem._current_log_difference([a], [b]))
                state = max(state, float(np.max(abs(a["state_u"]-b["state_u"]))))
                endpoint = max(endpoint, abs(float(a["final_u"])-float(b["final_u"])))
            rows[f"{top}_{phase}"] = {"max_abs_log_current_difference_natural": current,
                "max_abs_state_difference_u": state, "max_abs_endpoint_difference_u": endpoint}
            max_current, max_state, max_endpoint = max(max_current, current), max(max_state, state), max(max_endpoint, endpoint)
    return {"phases": rows, "max_abs_log_current_difference_natural": max_current,
            "max_abs_state_difference_u": max_state, "max_abs_endpoint_difference_u": max_endpoint,
            "passes_original_current_and_state_tolerances": bool(max_current <= math.log1p(problem.periodic_current_rtol)
                and max(max_state, max_endpoint) <= problem.periodic_state_atol),
            "periodic_current_rtol": problem.periodic_current_rtol, "periodic_state_atol": problem.periodic_state_atol}


def save_predictions(path, predictions, q, residual):
    arrays = {"frozen_q": np.asarray(q), "full32_training_residual": residual}
    for (top, cid, phase), trace in predictions.items():
        for key, value in trace.items():
            if isinstance(value, np.ndarray):
                arrays[f"{top}_{cid}_{phase}_{key}"] = value
        arrays[f"{top}_{cid}_{phase}_final_u"] = np.asarray(trace["final_u"])
    np.savez_compressed(path, **arrays)


def gap_baseline_metrics(problem, constraints, label, directory=GAP_BASELINES):
    """Rescore a fixed fitted GapLaw using training arrays only, never rerun it.

    Legacy NPZs also contain holdouts, but the loader indexes only TRAIN IDs.
    Saved observations for those IDs must exactly match the current raw cache.
    Only training metrics from the legacy report are consulted for replay.
    """
    directory = Path(directory)
    report_path, arrays_path = directory/f"{label}_evaluation.json", directory/f"{label}_evaluation.npz"
    saved = json.loads(report_path.read_text())
    if saved.get("status") != "full_grid_no_refit_evaluation_completed":
        raise ValueError(f"Gap baseline {label} evaluation is not complete")
    if any(saved["split"][top]["train"] != ids for top, ids in TRAIN.items()):
        raise ValueError(f"Gap baseline {label} training IDs differ")
    predictions = load_training_predictions(arrays_path, problem)
    cost, _, blocks = score_cost(problem, predictions)
    metrics = phase_metrics(problem, predictions)
    replay = {}
    for top in TRAIN:
        for phase in ("set", "reset"):
            computed = metrics[f"{top}_{phase}"]["groups"]["all_eligible"]["plain_RMSE_decade"]
            original = saved["metrics"][top]["train"][phase]["all_eligible"]["rmse_decade"]
            if original is None or not math.isclose(computed, original, rel_tol=1e-11, abs_tol=1e-11):
                raise ValueError(f"Gap baseline {label} training raw-RMSE replay differs: {top}/{phase}")
            replay[f"{top}_{phase}"] = {"saved_training_plain_RMSE_decade": original,
                "recomputed_plain_RMSE_decade": computed, "difference": computed-original}
    fit_path = ROOT/saved["fit_file"]
    fit_digest = saved["source_sha256"].get(saved["fit_file"])
    if fit_digest is None or not fit_path.is_file() or sha256(fit_path) != fit_digest:
        raise ValueError(f"Gap baseline {label} fit artifact no longer matches its saved evaluation")
    fitted = json.loads(fit_path.read_text())
    selected = [row for row in fitted["fits"] if row["start"] == fitted["selected_start"]]
    if len(selected) != 1 or not selected[0].get("valid", False):
        raise ValueError(f"Gap baseline {label} lacks one valid selected training fit")
    if fitted["selected_law"] != saved["shared_material"]:
        raise ValueError(f"Gap baseline {label} material parameters differ between fit and evaluation")
    return {"label": label, "baseline_kind": "previously_fitted_common_GapLaw",
        "training_only_comparable_cost": cost, "scoring_blocks": blocks,
        "phase_plain_log_metrics": metrics,
        "primary_feature_gates": feature_gates(problem, predictions, constraints),
        "saved_training_RMSE_replay": replay,
        "selected_training_fit": {"status": fitted["status"], "selected_start": fitted["selected_start"],
            "parameter_names": fitted["parameter_names"], "q": selected[0]["q"],
            "shared_material": fitted["selected_law"]},
        "training_input_alignment": "Stored training voltage and raw observation arrays exactly match the current objective; held-out NPZ arrays were not accessed.",
        "directly_reused": ["Frozen full native training prediction arrays", "Selected fitted parameters and material record", "Training plain RMSE only as an equality cross-check"],
        "recomputed_for_comparison": ["Exact current four-phase/three-group weighted training cost", "Plain phase/branch and observed-limit/nonlimit RMSE", "Every original primary feature gate"],
        "not_reused": ["Legacy bias-only or optimizer cost as a comparable objective", "Held-out predictions or held-out scores", "Old median-curve scores as raw-record accuracy"],
        "source_sha256": {str(path.resolve()): sha256(path) for path in (report_path, arrays_path, fit_path)},
        "historical_context": "These are retrospective project artifacts with historical exposure noted in their original reports; rescoring training records does not create a new prospective validation set.",
        "scope": "No GapLaw reoptimization or ODE rerun. Both previously fitted candidates are included; any baseline ranking uses only this common training score."}


def evaluate_terminal_fit(fit_path, output_path, *, table_path=DEFAULT_TABLE,
                          stored_predictions_path=None, baseline_json=None, baseline_predictions=None,
                          gap_baseline_directory=GAP_BASELINES):
    started = time.monotonic()
    fit_path, output_path, table_path = Path(fit_path).resolve(), Path(output_path).resolve(), Path(table_path).resolve()
    fit = load_finished_fit(fit_path)
    fit_digest = sha256(fit_path)
    q = np.array(fit["best_training"]["q"], dtype=float, copy=True)
    q.flags.writeable = False
    stored_predictions_path = Path(stored_predictions_path or fit_path.parent/"initiation_best_training.npz").resolve()
    protected_inputs = {fit_path, table_path, stored_predictions_path, FEATURES.resolve(), Path(DEFAULT_SHAPE_PATH).resolve()}
    if baseline_json is not None:
        protected_inputs.add(Path(baseline_json).resolve())
    if baseline_predictions is not None:
        protected_inputs.add(Path(baseline_predictions).resolve())
    for label in ("block", "accurate"):
        protected_inputs.update((Path(gap_baseline_directory)/f"{label}_{suffix}").resolve()
                                for suffix in ("evaluation.json", "evaluation.npz", "fit.json"))
    if output_path in protected_inputs or output_path.with_suffix(".npz") in protected_inputs:
        raise ValueError("Evaluation output must not overwrite frozen inputs or baseline artifacts")
    for relative, digest in fit["source_sha256"].items():
        source = ROOT/relative
        if not source.is_file() or sha256(source) != digest:
            raise ValueError(f"Frozen fitting source changed or disappeared: {relative}")
    constraints = json.loads(FEATURES.read_text())
    if constraints["training_ids"] != TRAIN or constraints.get("test_arrays_loaded") is not False:
        raise ValueError("Feature registry training scope differs")
    shape = SpectralShape.from_json()
    table = SpectralTable.load(table_path, shape)
    def factory(parameters, ron):
        if abs(ron-shape.ron_ohm) > 1e-8:
            raise ValueError("Training Ron differs from frozen spectral shape")
        exact = SpectralLaw(shape, *10.**np.asarray(parameters[:5]), reset_initiation_log=float(parameters[7]))
        return TabulatedSpectralLaw(exact, table)
    problem = SpectralJointProblem(factory, include_reset_initiation=True, mode="full",
        integrator_options=fit["integrator_options"],
        periodic_current_rtol=fit["objective"]["periodic_current_rtol"],
        periodic_state_atol=fit["objective"]["periodic_state_atol"],
        additional_source_paths=[FEATURES, DEFAULT_SHAPE_PATH, table_path, fit_path])
    if problem.training_arrays_sha256 != fit["objective"]["training_arrays_sha256"]:
        raise ValueError("Training data changed since fitting")
    # These are fitted shared-model references, not the deliberately poor
    # unoptimized diagnostic. Rescore both before the new full32 solve.
    gap_references = {label: gap_baseline_metrics(problem, constraints, label, gap_baseline_directory)
                      for label in ("block", "accurate")}
    stored = load_training_predictions(stored_predictions_path, problem)
    stored_cost, _, stored_blocks = score_cost(problem, stored)
    if not math.isclose(stored_cost, fit["best_training"]["cost"], rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError("Stored best predictions do not reproduce the terminal best_training cost")
    frozen_inputs = {str(path): sha256(path) for path in (fit_path, stored_predictions_path, FEATURES, Path(DEFAULT_SHAPE_PATH), table_path)}
    print(json.dumps({"status": "frozen_terminal_candidate_full32_evaluation_started",
                      "q": q.tolist(), "table": str(table_path), "holdout_loaded": False}), flush=True)
    full = problem.evaluate(q, mode="full")
    if full["details"]["used_mode"] != "full32" or full["details"]["n_integrated_1t_pairs"] != 32:
        raise RuntimeError("Evaluator failed to propagate all32 original training pairs")
    for path, digest in frozen_inputs.items():
        if sha256(path) != digest:
            raise RuntimeError(f"Frozen input changed during evaluation: {path}")
    if sha256(fit_path) != fit_digest:
        raise RuntimeError("Terminal fit snapshot changed during evaluation")
    comparison = compare_predictions(problem, full["predictions"], stored)
    comparison.update(stored_best_training_cost=stored_cost,
        full32_training_cost=full["details"]["verified_cost"],
        cost_difference_full_minus_stored=full["details"]["verified_cost"]-stored_cost,
        stored_mode=fit["best_training"]["details"]["used_mode"],
        same_table_as_fit=sha256(table_path) == fit["source_sha256"][str(DEFAULT_TABLE.relative_to(ROOT))],
        stored_scoring_blocks=stored_blocks)
    comparison["scope"] = ("Same table and parameters: difference audits the saved prefix/full prediction approximation and numerical reproducibility."
        if comparison["same_table_as_fit"] else
        "Changed numerical table, unchanged parameters: difference combines table refinement and saved-prefix approximation; it cannot be attributed solely to prefix reuse.")
    report = {"status": "completed_frozen_full32_training_evaluation_no_holdout_no_refit",
        "frozen_best_training": fit["best_training"], "q": q.tolist(),
        "parameter_selection": "The terminal fit's recorded best_training only; no selection using this reevaluation or held-out data.",
        "fit_terminal_status": fit["status"], "fit_runs_termination_status": [row["status"] for row in fit["runs"]],
        "checkpoint_metadata": problem.checkpoint_metadata(), "full32_details": full["details"],
        "stored_vs_full32": comparison,
        "phase_plain_log_metrics": phase_metrics(problem, full["predictions"]),
        "primary_feature_gates": feature_gates(problem, full["predictions"], constraints),
        "fitted_GapLaw_baselines": gap_references,
        "trace_integrator_metadata": {f"{top}_{cid}_{phase}": trace.get("metadata", {})
            for (top, cid, phase), trace in full["predictions"].items()},
        "holdout_arrays_loaded": False,
        "interpretation": "Improvement must be assessed against both frozen fitted GapLaw baselines, not only the unoptimized diagnostic. Global training loss improvement is not full model success. Every primary feature failure remains a failure; a RESET-only initiation term does not remove the existing SET state-order obstruction. Passing training gates would still not identify a microscopic mechanism."}
    strongest_reference = min(gap_references, key=lambda key: gap_references[key]["training_only_comparable_cost"])
    for label, reference in gap_references.items():
        reference["full32_spectral_cost_minus_this_GapLaw_cost"] = full["details"]["verified_cost"]-reference["training_only_comparable_cost"]
        reference["spectral_minus_GapLaw_plain_RMSE_decade"] = {
            phase: {group: values["plain_RMSE_decade"]-reference["phase_plain_log_metrics"][phase]["groups"][group]["plain_RMSE_decade"]
                    for group, values in summary["groups"].items() if values["n"]}
            for phase, summary in report["phase_plain_log_metrics"].items()}
    report["strongest_fitted_GapLaw_training_reference"] = {
        "label": strongest_reference, "selection": "Lower recomputed original-record training cost among the two preexisting frozen fitted candidates; no held-out score used.",
        "reference_training_cost": gap_references[strongest_reference]["training_only_comparable_cost"],
        "spectral_full32_cost": full["details"]["verified_cost"],
        "spectral_minus_reference_cost": full["details"]["verified_cost"]-gap_references[strongest_reference]["training_only_comparable_cost"],
        "does_not_override_feature_failures": True}
    if (baseline_json is None) != (baseline_predictions is None):
        raise ValueError("Supply both baseline JSON and native-prediction NPZ, or neither")
    if baseline_json is not None:
        baseline_json, baseline_predictions = Path(baseline_json).resolve(), Path(baseline_predictions).resolve()
        baseline = json.loads(baseline_json.read_text())
        if baseline.get("status") != "completed_diagnostic_full32_not_fitted" or baseline.get("details", {}).get("used_mode") != "full32" or baseline["details"].get("n_integrated_1t_pairs") != 32:
            raise ValueError("Baseline must be the completed full32 diagnostic")
        baseline_curves = load_training_predictions(baseline_predictions, problem)
        baseline_cost, _, baseline_blocks = score_cost(problem, baseline_curves)
        if not math.isclose(baseline_cost, baseline["details"]["verified_cost"], rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("Baseline stored predictions fail their original-record cost replay")
        report["unoptimized_diagnostic_comparison"] = {"status": "same_training_targets_native_grids_masks_group_weights_and_feature_bands",
            "baseline_status": baseline["status"], "baseline_q": baseline["q"],
            "baseline_weighted_cost": baseline_cost, "baseline_scoring_blocks": baseline_blocks,
            "full32_cost_change_from_baseline": full["details"]["verified_cost"]-baseline_cost,
            "baseline_phase_plain_log_metrics": phase_metrics(problem, baseline_curves),
            "baseline_primary_feature_gates": feature_gates(problem, baseline_curves, constraints),
            "baseline_source_sha256_as_recorded": baseline.get("source_sha256", {}),
            "scope": "Stored previous seven-parameter full32 diagnostic, rescored on exactly the same current frozen training observations. It is not a newly optimized baseline or held-out comparison."}
        frozen_inputs.update({str(baseline_json): sha256(baseline_json), str(baseline_predictions): sha256(baseline_predictions)})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path = output_path.with_suffix(".npz")
    save_predictions(trace_path, full["predictions"], q, full["residual"])
    report.update(full_native_training_predictions=str(trace_path), training_predictions_sha256=sha256(trace_path),
        elapsed_s=time.monotonic()-started, input_sha256=frozen_inputs,
        evaluator_source_sha256={str(path.relative_to(ROOT)): sha256(path) for path in
            (Path(__file__), ROOT/"analysis/constrained_shared_2026-11-03/feature_constraints.py")})
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, default=HERE/"initiation_joint_fit.json")
    parser.add_argument("--output", type=Path, help="Separate JSON/NPZ output prefix; a nondefault table gets a distinct default filename")
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--stored-predictions", type=Path)
    parser.add_argument("--baseline-json", type=Path, default=HERE/"diagnostic_compromise.json")
    parser.add_argument("--baseline-predictions", type=Path, default=HERE/"diagnostic_compromise.npz")
    parser.add_argument("--no-diagnostic-baseline", "--no-baseline", dest="no_baseline", action="store_true",
                        help="Skip only the unoptimized diagnostic; fitted block/accurate GapLaw references remain mandatory")
    parser.add_argument("--gap-baseline-directory", type=Path, default=GAP_BASELINES,
                        help="Directory containing both frozen block/accurate evaluation JSON+NPZ and their recorded fit files; always included")
    args = parser.parse_args()
    output = args.output or HERE/("initiation_full32_evaluation.json" if args.table.resolve() == DEFAULT_TABLE.resolve()
                                 else f"initiation_full32_evaluation_{args.table.stem}.json")
    report = evaluate_terminal_fit(args.fit, output, table_path=args.table,
        stored_predictions_path=args.stored_predictions,
        baseline_json=None if args.no_baseline else args.baseline_json,
        baseline_predictions=None if args.no_baseline else args.baseline_predictions,
        gap_baseline_directory=args.gap_baseline_directory)
    print(json.dumps({"status": report["status"], "output": str(output.resolve()),
        "full32_cost": report["full32_details"]["verified_cost"],
        "primary_features_passed": report["primary_feature_gates"]["n_primary_passed"],
        "primary_features_total": report["primary_feature_gates"]["n_primary"],
        "all_primary_passed": report["primary_feature_gates"]["all_primary_passed"],
        "holdout_loaded": False}, indent=2))


if __name__ == "__main__":
    main()
