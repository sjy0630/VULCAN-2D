"""Training-only variability and exact log-error decomposition; no model fit.

The oracle here is the per-waveform-index arithmetic mean of log10(current)
among observations eligible for the requested score. It is an empirical lower
bound for a *single deterministic waveform*, not a measurement-noise estimate
or a lower bound for models that can explain cycle-dependent state changes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "analysis/shared_hbn_2026-11-03"
OUT = Path(__file__).resolve().parent / "data_diagnostics.json"
TRAIN = {
    "1r": [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    "1t1r": list(range(4, 36)),
}
FLOORS = {"set": 1e-11, "reset": 1e-12}


def stats(values):
    x = np.asarray(values, float)
    return {"n": int(x.size), "min": float(x.min()),
            "q25": float(np.quantile(x, .25)), "median": float(np.median(x)),
            "mean": float(np.mean(x)), "q75": float(np.quantile(x, .75)),
            "max": float(x.max()), "population_sd": float(np.std(x))}


def digest_arrays(items):
    digest = hashlib.sha256()
    for name, array in sorted(items):
        value = np.ascontiguousarray(array)
        digest.update(name.encode())
        digest.update(str(value.shape).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def decomposition(voltage, observed, predicted, phase, selection=None,
                  point_details=False):
    """Exact weighted identity, allowing record-specific eligibility masks.

    With y=log10(observed), p=log10(predicted), means at each native waveform
    index j, and score weights n_j / N:
    MSE = E[(mean(p)-mean(y))^2] + E[var(y)] + E[var(p)] - 2 E[cov(p,y)].
    Population variances use denominator n_j (not n_j-1).
    """
    observed, predicted = np.abs(observed), np.abs(predicted)
    voltage = np.asarray(voltage)
    if observed.shape != predicted.shape or observed.shape[1:] != voltage.shape:
        raise ValueError("Unaligned native waveform arrays")
    mask = (abs(voltage)[None, :] > 1e-12) & np.isfinite(observed)
    mask &= observed >= FLOORS[phase]
    if selection is not None:
        mask &= np.broadcast_to(selection, observed.shape)
    n = int(mask.sum())
    invalid = mask & (~np.isfinite(predicted) | (predicted <= 0))
    if np.any(invalid) or n == 0:
        raise ValueError("Invalid prediction or empty selected training region")
    ly = np.zeros(observed.shape)
    lp = np.zeros(observed.shape)
    ly[mask] = np.log10(observed[mask])
    lp[mask] = np.log10(predicted[mask])
    counts = mask.sum(axis=0)
    denominator = np.maximum(counts, 1)
    mean_y = ly.sum(axis=0) / denominator
    mean_p = lp.sum(axis=0) / denominator
    dy = np.where(mask, ly-mean_y, 0.)
    dp = np.where(mask, lp-mean_p, 0.)
    error = lp[mask]-ly[mask]
    bias_sse = float(np.sum(counts*(mean_p-mean_y)**2))
    obs_sse = float(np.sum(dy**2))
    pred_sse = float(np.sum(dp**2))
    covariance_sum = float(np.sum(dp*dy))
    residual_scatter_sse = float(np.sum((dp-dy)**2))
    sse = float(np.sum(error**2))
    reconstructed = bias_sse+obs_sse+pred_sse-2*covariance_sum
    if not np.isclose(sse, reconstructed, rtol=5e-13, atol=1e-11):
        raise AssertionError("Exact squared-error identity failed")
    result = {
        "n": n, "n_waveform_indices": int(np.count_nonzero(counts)),
        "n_indices_with_partial_record_eligibility": int(np.count_nonzero(
            (counts > 0) & (counts < observed.shape[0]))),
        "invalid_predictions": 0, "current_floor_A": FLOORS[phase],
        "model_rmse_decade": float(np.sqrt(sse/n)), "model_mse_decade2": sse/n,
        "systematic_mean_curve_mse_decade2": bias_sse/n,
        "systematic_mean_curve_rmse_decade": float(np.sqrt(bias_sse/n)),
        "fixed_waveform_oracle_mse_decade2": obs_sse/n,
        "fixed_waveform_oracle_rmse_decade": float(np.sqrt(obs_sse/n)),
        "model_cycle_variance_decade2": pred_sse/n,
        "model_observation_cycle_covariance_decade2": covariance_sum/n,
        "residual_scatter_mse_decade2": residual_scatter_sse/n,
        "systematic_mse_fraction": bias_sse/sse if sse else None,
        "fixed_waveform_variability_over_model_mse": obs_sse/sse if sse else None,
        "mse_difference_from_fixed_waveform_oracle": (sse-obs_sse)/n,
        "exact_identity_abs_error_decade2": abs(sse-reconstructed)/n,
        "sse_decade2": sse,
        "mean_signed_log_error_decade": float(np.mean(error)),
    }
    if point_details:
        apex = int(np.argmax(abs(voltage)))
        rows = [{
            "index": int(j), "voltage_V": float(voltage[j]),
            "branch": "outward" if j <= apex else "return",
            "n_eligible_records": int(counts[j]),
            "mean_observed_log10_A": float(mean_y[j]),
            "mean_predicted_log10_A": float(mean_p[j]),
            "bias_decade": float(mean_p[j]-mean_y[j]),
            "observation_population_sd_decade": float(np.sqrt(np.sum(dy[:, j]**2)/counts[j])),
            "model_population_sd_decade": float(np.sqrt(np.sum(dp[:, j]**2)/counts[j])),
            "systematic_sse_decade2": float(counts[j]*(mean_p[j]-mean_y[j])**2),
        } for j in np.flatnonzero(counts)]
        result["waveform_points"] = rows
        result["largest_systematic_sse_points"] = sorted(
            rows, key=lambda row: row["systematic_sse_decade2"], reverse=True)[:12]
    return result


def reset_peak_diagnostic(voltage, observed, predicted, ids):
    apex = int(np.argmax(abs(voltage)))
    obs, pred = abs(observed[:, :apex+1]), abs(predicted[:, :apex+1])
    index_o, index_p = np.argmax(obs, axis=1), np.argmax(pred, axis=1)
    io = obs[np.arange(len(ids)), index_o]
    ip = pred[np.arange(len(ids)), index_p]
    vo, vp = voltage[index_o], voltage[index_p]
    low = int(np.argmin(abs(voltage[:apex+1]+.2)))
    rows = [{"record": int(cid), "observed_peak_A": float(io[k]),
             "predicted_peak_A": float(ip[k]), "observed_peak_V": float(vo[k]),
             "predicted_peak_V": float(vp[k]),
             "observed_initial_abs_current_at_minus0p2V_A": float(obs[k, low]),
             "predicted_initial_abs_current_at_minus0p2V_A": float(pred[k, low])}
            for k, cid in enumerate(ids)]
    return {
        "definition": "Peak magnitude on outward RESET including first apex; no physical parameter fit",
        "observed_peak_current_A": stats(io), "predicted_peak_current_A": stats(ip),
        "observed_peak_voltage_V": stats(vo), "predicted_peak_voltage_V": stats(vp),
        "peak_current_log_error_rmse_decade": float(np.sqrt(np.mean(np.log10(ip/io)**2))),
        "peak_voltage_error_rmse_V": float(np.sqrt(np.mean((vp-vo)**2))),
        "per_training_record": rows,
    }


def main():
    baseline = json.loads((BASE / "gap_joint_evaluation.json").read_text())
    if {k: v["train"] for k, v in baseline["split"].items()} != TRAIN:
        raise ValueError("Training split changed")
    training_arrays = []
    results, peaks, prediction_variation = {}, {}, {}
    with np.load(BASE / "gap_joint_evaluation.npz") as cache, np.load(
            BASE / "data/native_observations.npz") as native:
        for top, ids in TRAIN.items():
            results[top], prediction_variation[top] = {}, {}
            for phase in ("set", "reset"):
                prefix = f"{top}_train_{phase}"
                # Only *_train_* cache arrays and explicitly named train raw
                # records are loaded. No test observations are inspected.
                v = cache[prefix+"_voltage"]
                observed = cache[prefix+"_observed"]
                predicted = cache[prefix+"_predicted"]
                for k, cid in enumerate(ids):
                    raw_prefix = f"{top}_cycle_{cid}_{phase}"
                    if not np.array_equal(v, native[raw_prefix+"_voltage"], equal_nan=True):
                        raise ValueError("Native voltage mismatch")
                    if not np.array_equal(observed[k], abs(native[raw_prefix+"_current"]), equal_nan=True):
                        raise ValueError("Native observation mismatch")
                training_arrays.extend([(prefix+"_"+name, val) for name, val in
                    [("voltage", v), ("observed", observed), ("predicted", predicted)]])
                outward = np.arange(len(v)) <= int(np.argmax(abs(v)))
                selections = {"all_eligible": None, "outward": outward, "return": ~outward}
                if top == "1r" and phase == "set":
                    for label, selected in (("nonlimited", observed < .995e-3),
                                            ("limited", observed >= .995e-3)):
                        selections[label] = selected
                        selections[label+"_outward"] = selected & outward
                        selections[label+"_return"] = selected & ~outward
                if top == "1r" and phase == "reset":
                    selections["without_initial_25mV"] = ~(
                        np.isclose(v, -.025, atol=1e-6) & outward)
                regions = {name: decomposition(v, observed, predicted, phase, selected,
                            point_details=name == "all_eligible")
                           for name, selected in selections.items()}
                # Reproduce the existing baseline count and all-point score.
                for name in ("all_eligible", "nonlimited", "limited", "without_initial_25mV"):
                    if name not in regions:
                        continue
                    expected = baseline["metrics"][top]["train"][phase][name]
                    measured = regions[name]
                    if measured["n"] != expected["n"] or not np.isclose(
                            measured["model_rmse_decade"], expected["rmse_decade"], atol=2e-14, rtol=0):
                        raise AssertionError("Existing score not reproduced")
                total = regions["all_eligible"]
                partitions = [["outward", "return"]]
                if top == "1r" and phase == "set":
                    partitions += [["nonlimited", "limited"],
                        ["nonlimited_outward", "nonlimited_return", "limited_outward", "limited_return"]]
                checks = []
                for names in partitions:
                    count = sum(regions[name]["n"] for name in names)
                    sse = sum(regions[name]["sse_decade2"] for name in names)
                    if count != total["n"] or not np.isclose(sse, total["sse_decade2"], rtol=5e-14):
                        raise AssertionError("Region score recombination failed")
                    checks.append({"regions": names, "n": count,
                        "reconstructed_all_point_rmse_decade": float(np.sqrt(sse/count))})
                results[top][phase] = {"regions": regions, "partition_recombination": checks}
                nonzero = abs(v) > 1e-12
                lp = np.log10(abs(predicted[:, nonzero]))
                prediction_variation[top][phase] = {
                    "n_distinct_prediction_rows_exact": int(np.unique(predicted, axis=0).shape[0]),
                    "max_log_range_at_same_native_index_decade": float(np.max(np.ptp(lp, axis=0))),
                    "max_abs_difference_from_first_training_curve_A": float(np.max(abs(predicted-predicted[0]))),
                }
                if phase == "reset":
                    peaks[top] = reset_peak_diagnostic(v, observed, predicted, ids)
    components = [results[top][phase]["regions"]["all_eligible"]
                  for top in TRAIN for phase in ("set", "reset")]
    count = sum(r["n"] for r in components)
    component_names = ["model_mse_decade2", "systematic_mean_curve_mse_decade2",
                      "fixed_waveform_oracle_mse_decade2", "model_cycle_variance_decade2",
                      "model_observation_cycle_covariance_decade2", "residual_scatter_mse_decade2"]
    pooled = {name: sum(r["n"]*r[name] for r in components)/count for name in component_names}
    pooled["n"] = count
    pooled["model_rmse_decade"] = float(np.sqrt(pooled["model_mse_decade2"]))
    pooled["fixed_waveform_oracle_rmse_decade"] = float(np.sqrt(pooled["fixed_waveform_oracle_mse_decade2"]))
    history = [r for r in baseline["continuous_1t1r_history"] if r["cycle"] in TRAIN["1t1r"]]
    report = {
        "status": "training_only_diagnostic_no_physical_refit",
        "training_ids": TRAIN, "test_observations_loaded": False,
        "baseline": "analysis/shared_hbn_2026-11-03/gap_joint_evaluation.{json,npz}",
        "training_arrays_sha256": digest_arrays(training_arrays),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "score": "Same native-grid point score as evaluate_shared_gap_joint.score: nonzero V; finite |Iobs|; >=1e-11A SET / >=1e-12A RESET. Prediction must be finite and >0. No resampling or clipping.",
        "branch_rule": "Indices <= first |V| apex are outward; all later indices return. Repeated apex and all eligible points occur in exactly one branch.",
        "identity": "MSE = systematic mean-curve MSE + observed within-index variance + predicted within-index variance - 2*within-index covariance; weights are each index's eligible record count divided by total count.",
        "interpretation_limits": [
            "Oracle log mean is estimated and evaluated on the same training sample; it is a descriptive empirical optimum, not a held-out achievable accuracy guarantee.",
            "Observed scatter combines cycle/device variability, unmodelled state and instrument effects. It is not identified as irreducible measurement noise.",
            "Cycle-dependent models may beat the fixed-waveform oracle through positive model/observation covariance. The residual-scatter identity explicitly includes this possibility.",
            "Floor eligibility is record-specific. Every region recomputes its pointwise means with its own exact eligible mask; excluded low currents are not replaced by the floor.",
            "Limited/nonlimited classification depends on observed current; the two conditional oracle curves are not one deployable deterministic predictor. Their separate oracle variances must not be summed and advertised as the global fixed-waveform bound.",
            "Bare-R chronology across records is unverified; first14 is a fixed record split, not independently confirmed chronological validation.",
            "1T1R first3 excluded. Only records4-35 are used in all calculations; test arrays are not loaded.",
        ],
        "decomposition": results,
        "pooled_all_training_points": pooled,
        "equal_four_phase_mean_mse_decade2": {name: float(np.mean([r[name] for r in components])) for name in component_names},
        "reset_peaks": peaks,
        "model_prediction_cycle_variation": prediction_variation,
        "continuous_training_1t1r_state_history": history,
        "training_state_statistics": {name: stats([r[name] for r in history])
                                      for name in ("initial_u", "after_set_u", "after_reset_u")},
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    compact = {top: {phase: {name: {key: value for key, value in row.items()
               if key in ("n", "model_rmse_decade", "fixed_waveform_oracle_rmse_decade",
                          "systematic_mean_curve_rmse_decade", "systematic_mse_fraction")}
               for name, row in result["regions"].items()}
               for phase, result in phases.items()} for top, phases in results.items()}
    print(json.dumps({"decomposition": compact, "pooled": pooled,
                      "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
