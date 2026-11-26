"""Training-only reconstruction of two-pair surrogate and ordered-record loss.

No trajectories are recomputed and no parameters are fit or selected. Only
explicit training-record arrays are loaded from completed evaluation caches.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = ROOT / "analysis/shared_hbn_2026-11-03/data/native_observations.npz"
TRAIN = {"1r": [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
         "1t1r": list(range(4, 36))}


def group_metrics(observed, predicted, mask, repeated_pairs):
    count = mask.sum(axis=0)
    n = int(mask.sum())
    eligible_index = count > 0
    if np.any(mask & (~np.isfinite(predicted) | (predicted <= 0.))):
        raise ValueError("Invalid scored prediction")
    y, p = np.zeros_like(observed), np.zeros_like(predicted)
    y[mask] = np.log10(observed[mask])
    p[mask] = np.log10(predicted[mask])
    mean_y = np.sum(y, axis=0)/np.maximum(count, 1)
    mean_p = np.sum(p, axis=0)/np.maximum(count, 1)
    dy, dp = np.where(mask, y-mean_y, 0.), np.where(mask, p-mean_p, 0.)
    scatter = float(np.sum(dy**2)/n)
    actual_total = float(np.sum((p[mask]-y[mask])**2)/n)
    bias = float(np.sum(count*(mean_p-mean_y)**2)/n)
    pred_variance = float(np.sum(dp**2)/n)
    covariance = float(np.sum(dp*dy)/n)
    reconstructed = bias+scatter+pred_variance-2*covariance
    if not np.isclose(actual_total, reconstructed, atol=2e-13, rtol=2e-13):
        raise AssertionError("Ordered error decomposition failed")
    # Each surrogate pair is compared against every eligible training record
    # through the exact same weighted per-waveform-index log means as the fit.
    pair_bias, pair_full = [], []
    for index in range(repeated_pairs):
        log_prediction = np.log10(predicted[index, eligible_index])
        pair_bias.append(float(np.sum(count[eligible_index]
            *(log_prediction-mean_y[eligible_index])**2)/n))
        full = 0.
        for j in np.flatnonzero(eligible_index):
            full += float(np.sum((np.log10(predicted[index, j])-y[mask[:, j], j])**2))
        pair_full.append(full/n)
        if not np.isclose(pair_full[-1], pair_bias[-1]+scatter, atol=2e-13, rtol=2e-13):
            raise AssertionError("Surrogate sufficient-statistics identity failed")
    surrogate_bias = float(np.mean(pair_bias))
    surrogate_total = float(np.mean(pair_full))
    result = {
        "eligible_records_points": n,
        "n_native_indices": int(eligible_index.sum()),
        "observed_within_index_variance": scatter,
        "actual_ordered_total_mse": actual_total,
        "actual_ordered_model_dependent_part": actual_total-scatter,
        "actual_ordered_mean_curve_bias_mse": bias,
        "actual_ordered_prediction_cycle_variance": pred_variance,
        "actual_ordered_prediction_observation_covariance": covariance,
        "surrogate_pair_bias_mse": pair_bias,
        "surrogate_bias_mse": surrogate_bias,
        "surrogate_total_mse_with_record_variance": surrogate_total,
        "surrogate_minus_actual_total_mse": surrogate_total-actual_total,
        "first_training_record_actual_score_weight": float(mask[0].sum()/n),
        "first_pair_surrogate_weight": 1./repeated_pairs,
        "decomposition_absolute_error": abs(actual_total-reconstructed),
    }
    if repeated_pairs == 2:
        log_all = np.log10(predicted[:, eligible_index])
        shape_difference = log_all[0]-log_all[1]
        result.update(
            first_vs_second_pair_log_shape_mse=float(
                np.sum(count[eligible_index]*shape_difference**2)/n),
            max_abs_first_vs_second_pair_log_difference=float(np.max(abs(shape_difference))),
            max_abs_later_training_vs_second_pair_log_difference=float(
                np.max(abs(log_all[1:]-log_all[1]))),
        )
    return result


def audit(label, native):
    metadata_path = OUT / f"{label}_evaluation.json"
    array_path = metadata_path.with_suffix(".npz")
    metadata = json.loads(metadata_path.read_text())
    if {k: v["train"] for k, v in metadata["split"].items()} != TRAIN:
        raise ValueError("Unexpected training split")
    fit_path = ROOT / metadata["fit_file"]
    fit = json.loads(fit_path.read_text())
    selected = next(row for row in fit["fits"] if row["start"] == fit["selected_start"])
    phases = {}
    digest = hashlib.sha256()
    with np.load(array_path) as cache:
        for top, ids in TRAIN.items():
            for phase in ("set", "reset"):
                voltages, observations, predictions = [], [], []
                for cid in ids:
                    key = f"{top}_cycle{cid}_{phase}"
                    v = cache[key+"_voltage"]
                    obs = abs(cache[key+"_observed_current_a"])
                    pred = abs(cache[key+"_current_a"])
                    raw_key = f"{top}_cycle_{cid}_{phase}"
                    if not np.array_equal(v, native[raw_key+"_voltage"], equal_nan=True):
                        raise ValueError("Voltage differs from native training source")
                    if not np.array_equal(obs, abs(native[raw_key+"_current"]), equal_nan=True):
                        raise ValueError("Observation differs from native training source")
                    for suffix, value in (("voltage", v), ("observed", obs), ("predicted", pred)):
                        digest.update((key+suffix).encode())
                        digest.update(value.tobytes())
                    voltages.append(v); observations.append(obs); predictions.append(pred)
                vv, obs, pred = map(np.asarray, (voltages, observations, predictions))
                if not np.all(vv == vv[0]):
                    raise ValueError("Training waveforms differ")
                floor = 1e-11 if phase == "set" else 1e-12
                valid = (abs(vv) > 1e-12) & np.isfinite(obs) & (obs >= floor)
                if (top, phase) == ("1r", "set"):
                    up = np.arange(vv.shape[1]) <= np.argmax(abs(vv[0]))
                    masks = {"nonlimited_outward": valid & (obs < .995e-3) & up,
                             "nonlimited_return": valid & (obs < .995e-3) & ~up,
                             "limited": valid & (obs >= .995e-3)}
                else:
                    masks = {"all_eligible": valid}
                groups = {name: group_metrics(obs, pred, mask, 1 if top == "1r" else 2)
                          for name, mask in masks.items()}
                means = {name: float(np.mean([row[name] for row in groups.values()]))
                         for name in ("observed_within_index_variance", "actual_ordered_total_mse",
                                      "actual_ordered_model_dependent_part", "surrogate_bias_mse",
                                      "surrogate_total_mse_with_record_variance",
                                      "surrogate_minus_actual_total_mse")}
                phases[f"{top}_{phase}"] = {"groups": groups, "group_weight": 1./len(groups),
                                            "phase_weighted_values": means}
    summed = {name: sum(p["phase_weighted_values"][name] for p in phases.values())
              for name in next(iter(phases.values()))["phase_weighted_values"]}
    costs = {name+"_cost_half_sum_four_phases": value/2. for name, value in summed.items()}
    costs["original_optimizer_reported_bias_cost"] = selected["verified_bias_cost"]
    costs["reconstructed_surrogate_bias_minus_reported_cost"] = (
        summed["surrogate_bias_mse"]/2.-selected["verified_bias_cost"])
    costs["equal_four_phase_ordered_root_mean_square_decade"] = float(
        np.sqrt(summed["actual_ordered_total_mse"]/4.))
    costs["equal_four_phase_surrogate_root_mean_square_decade"] = float(
        np.sqrt(summed["surrogate_total_mse_with_record_variance"]/4.))
    return {"candidate": label, "fit_file": str(fit_path.relative_to(ROOT)),
            "fit_sha256": hashlib.sha256(fit_path.read_bytes()).hexdigest(),
            "evaluation_metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
            "training_loaded_arrays_sha256": digest.hexdigest(),
            "phases": phases, "objective_costs": costs}


def main():
    candidates = [label for label in ("accurate", "block", "access", "access_profile")
                  if (OUT/f"{label}_evaluation.json").exists()
                  and (OUT/f"{label}_evaluation.npz").exists()]
    with np.load(DATA) as native:
        reports = {label: audit(label, native) for label in candidates}
    result = {
        "status": "training_only_objective_scope_reconstruction_no_fit_or_selection",
        "training_ids": TRAIN, "test_observation_arrays_loaded": False,
        "units": "Squared log10-current error; optimizer cost=one half of sum of four equally weighted phase MSEs, not their average.",
        "weighting": "Bare SET: nonlimited outward, nonlimited return, limited each1/3. Other phases: all eligible native points. Floors and observed-compliance masks match JointProblem exactly.",
        "surrogate": "Bare first model pair is compared against all14 records; first2 continuous1T model pairs each receive1/2 weight against all32 training records through per-index log means.",
        "actual_ordered": "Each model trajectory is paired with its own training record (1T cycles4-35), retaining exact observation masks. Model-dependent part=total minus per-index observation variance; may be negative for a successful cycle-dependent predictor.",
        "scope": "No test arrays inspected or used for selection. Reconstructed surrogate uses frozen evaluation trajectories, so last digits can differ from fitting-tolerance residuals.",
        "candidates": reports,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUT/"objective_scope_audit.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({label: report["objective_costs"] for label, report in reports.items()}, indent=2))


if __name__ == "__main__":
    main()
