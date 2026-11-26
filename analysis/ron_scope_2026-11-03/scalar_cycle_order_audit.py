"""Training-only optimistic loss floor for deterministic scalar repeated flows.

Independent nodewise isotonic regression relaxes the shared state/ODE. Each
continuous segment still has one direction across both phases and all nodes.
Observed masks and original per-observation loss weights remain fixed.
"""
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np
from scipy.optimize import isotonic_regression

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "analysis/parameter_transfer_2026-11-03"))
from group_problem import make_problem


def fit_segment(y, weights, increasing):
    predicted = np.zeros_like(y)
    for col in range(y.shape[1]):
        eligible = weights[:, col] > 0
        if np.any(eligible):
            predicted[eligible, col] = isotonic_regression(
                y[eligible, col], weights=weights[eligible, col], increasing=increasing).x
    return predicted, float(np.sum(weights*(predicted-y)**2)/2.)


def interval_check(values, fraction, increasing):
    lower, upper = values*(1-fraction), values*(1+fraction)
    if increasing:
        return bool(np.all(np.maximum.accumulate(lower) <= upper))
    return bool(np.all(np.minimum.accumulate(upper) >= lower))


def band_requirement(values, increasing):
    # Existence of an ordered sequence within symmetric relative envelopes.
    # This is a tolerance diagnostic, not a calibrated noise confidence band.
    required = 0.
    witness = None
    for i in range(len(values)):
        for j in range(i+1, len(values)):
            ratio = values[i]/values[j] if increasing else values[j]/values[i]
            delta = max(0., float((ratio-1)/(ratio+1)))
            if delta > required:
                required, witness = delta, [i, j]
    return required, witness


def main():
    started = time.monotonic()
    base = make_problem("1r", mode="full").joint  # Loads only original training arrays; no ODE.
    segments = {"1r": [[1,2,3], list(range(5,16))], "1t1r": [list(range(4,36))]}
    weights, logged = {}, {}
    for key, observed in base.raw_training.items():
        w = np.zeros_like(observed)
        for group in base.groups[key]:
            w[group["mask"]] += 1./group["normalization"]**2
        y = np.zeros_like(observed)
        y[w > 0] = np.log10(observed[w > 0])
        weights[key], logged[key] = w, y
    arrays, rows, chosen_predictions = {}, [], {}
    for top, sequences in segments.items():
        for ids in sequences:
            indices = [base.train_ids[top].index(cid) for cid in ids]
            results, predictions = {}, {}
            for increasing in (True, False):
                direction = "increasing_abs_current" if increasing else "decreasing_abs_current"
                scores = {}
                for phase in ("set", "reset"):
                    key = top, phase
                    pred, cost = fit_segment(logged[key][indices], weights[key][indices], increasing)
                    predictions[direction, phase] = pred
                    scores[phase] = cost
                results[direction] = dict(phase_cost=scores, total_cost=sum(scores.values()))
            selected = min(results, key=lambda name:results[name]["total_cost"])
            for phase in ("set", "reset"):
                key = top, phase
                chosen_predictions.setdefault(key, np.zeros_like(logged[key]))[indices] = predictions[selected, phase]
                arrays[f"{top}_{ids[0]}_{ids[-1]}_{phase}_log10_relaxed_prediction"] = predictions[selected, phase]
            rows.append(dict(topology=top, record_ids=ids, direction_results=results,
                             selected_relaxed_direction=selected,
                             optimistic_cost_floor=results[selected]["total_cost"]))
    total = sum(row["optimistic_cost_floor"] for row in rows)
    direct_cost = sum(float(np.sum(weights[k]*(chosen_predictions[k]-logged[k])**2)/2.) for k in weights)
    assert np.isclose(total, direct_cost, rtol=1e-14, atol=1e-15)

    phase_floors = {}
    for key in logged:
        top, phase = key
        # Further relaxation: choose a separate orientation per phase and
        # segment, with equal weight at eligible points. This supplies a lower
        # bound for raw phase RMSE, not the joint-direction optimum above.
        sse = 0.
        for ids in segments[top]:
            idx = [base.train_ids[top].index(cid) for cid in ids]
            binary = (weights[key][idx] > 0).astype(float)
            sse += 2*min(fit_segment(logged[key][idx],binary,inc)[1] for inc in (True,False))
        n = int(np.count_nonzero(weights[key]))
        phase_floors[f"{top}_{phase}"] = dict(n=n, raw_RMSE_lower_bound_decade=float(np.sqrt(sse/n)),
            scope="Independent phase/segment directions; weaker than shared-direction scalar requirement")

    low_reads = []
    top = "1r"
    for phase, branch in (("set","return"),("reset","outward")):
        v = base.waveforms[top,phase]
        turning = int(np.argmax(abs(v)))
        mask = np.isclose(abs(v),.05,rtol=0,atol=1e-8)
        mask &= np.arange(len(v)) > turning if branch == "return" else np.arange(len(v)) <= turning
        columns = np.flatnonzero(mask); assert len(columns) == 1
        col = int(columns[0])
        for ids in segments[top]:
            idx = [base.train_ids[top].index(cid) for cid in ids]
            currents = base.raw_training[top,phase][idx,col]
            assert np.all(weights[top,phase][idx,col] > 0)
            assert phase != "set" or np.all(currents < .995e-3)
            directions = {}
            for inc in (True,False):
                need,witness = band_requirement(currents,inc)
                directions["increasing" if inc else "decreasing"] = dict(
                    minimum_symmetric_fractional_band=need,
                    controlling_record_pair=[ids[i] for i in witness] if witness else None,
                    sensitivities={str(delta):interval_check(currents,delta,inc) for delta in (.05,.10,.20)})
            low_reads.append(dict(phase=phase,branch=branch,voltage_v=float(v[col]),record_ids=ids,
                currents_a=currents.tolist(),directions=directions,
                minimum_band_over_either_direction=min(x["minimum_symmetric_fractional_band"] for x in directions.values())))

    combined_reads = []
    for ids in segments["1r"]:
        selected = [row for row in low_reads if row["record_ids"] == ids]
        by_direction = {direction:max(row["directions"][direction]["minimum_symmetric_fractional_band"]
                                      for row in selected)
                        for direction in ("increasing", "decreasing")}
        combined_reads.append(dict(record_ids=ids, required_bands_by_common_direction=by_direction,
                                   minimum_band_for_both_low_read_nodes=min(by_direction.values())))
    paired_set = np.concatenate([np.asarray(row["currents_a"]) for row in low_reads if row["phase"] == "set"])
    paired_reset = np.concatenate([np.asarray(row["currents_a"]) for row in low_reads if row["phase"] == "reset"])
    log_ratio = np.log10(paired_reset/paired_set)
    pair_stats = dict(n=14, median_RESET_over_SET=float(np.median(paired_reset/paired_set)),
        log10_ratio_RMSE=float(np.sqrt(np.mean(log_ratio**2))),
        log10_ratio_std=float(np.std(log_ratio,ddof=1)),
        log10_current_Pearson_correlation=float(np.corrcoef(np.log10(paired_set),np.log10(paired_reset))[0,1]),
        interpretation="Paired opposite-polarity low reads co-vary across cycles. This is descriptive, not proof of independent-noise variance, zero tail dynamics, or a particular microscopic mechanism.")

    # Validate library semantics and the weighting convention with a known case.
    test = isotonic_regression(np.array([3.,1.,2.]),weights=np.array([1.,3.,1.]),increasing=True).x
    assert np.allclose(test,[1.5,1.5,2.],atol=1e-14,rtol=0)
    report = dict(status="completed_training_scalar_order_relaxation_no_physical_fit_no_ODE",
        assumptions=["One scalar deterministic state with unique order-preserving flows",
                     "Identical complete command/protocol per cycle within each segment, including unchanged dwell/path-clock assumption",
                     "A common direction of monotone port-current magnitude versus state at every nonzero phase node",
                     "Fixed parameters/circuits; no stochastic forcing or unmodeled history variables"],
        unknowns=["Measured dwell and waits unavailable", "Error envelopes are sensitivity choices, not measured instrument uncertainty"],
        segments=rows,total_original_joint_cost_lower_bound=total,original_n_residuals=base.n_residuals,
        weight_policy="Each point retains sum(1/normalization**2) from original masks; no per-node or per-segment renormalization; cost=0.5*SSE",
        raw_phase_lower_bounds=phase_floors,low_bias_checks=low_reads,
        low_bias_common_direction_checks=combined_reads,paired_low_read_statistics=pair_stats,
        scope="Independent isotonic node traces relax all cross-node state/ODE links. Necessary optimistic loss floors, not a new physical fit or proof against stochastic/multistate/nonrepeated protocols. Holdouts never read.",
        elapsed_s=time.monotonic()-started,
        source_training_arrays_sha256=base.training_arrays_sha256,
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__),ROOT/"analysis/bare_history_2026-11-03/identity_confirmation.json",
             ROOT/"analysis/constrained_shared_2026-11-03/ordered_objective.py")})
    (HERE/"SCALAR_CYCLE_ORDER_AUDIT.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    np.savez_compressed(HERE/"SCALAR_CYCLE_ORDER_RELAXATION.npz",**arrays)
    print(json.dumps({k:report[k] for k in ("status","elapsed_s","total_original_joint_cost_lower_bound","raw_phase_lower_bounds","low_bias_checks")},indent=2))


if __name__ == "__main__":
    main()
