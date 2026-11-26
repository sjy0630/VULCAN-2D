"""Separate, locally checked prefix reuse for the ordered training objective.

This does not change OrderedJointProblem's periodic algorithm or tolerances.
Every prefix record is propagated and scored separately. Reuse needs three
consecutive adjacent-pair trajectory checks plus a local contraction screen,
or three completely identical numerical repetitions. Neither supplies a
rigorous global error bound; frozen candidates still require full32 comparison.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

ADJACENT_TRANSITIONS_REQUIRED = 3
STATE_ATOL_CEILING = 1e-7
CURRENT_RTOL_CEILING = 1e-4
CONTRACTION_RATIO_CEILING = .5
ROUNDOFF_ULPS = 32.


def _tolerances(problem):
    # Explicitly preserve stricter caller settings, never relax the original
    # 1e-7 state / ln(1.0001) current thresholds for this separate shortcut.
    state = min(float(problem.periodic_state_atol), STATE_ATOL_CEILING)
    current = min(float(problem.periodic_current_rtol), CURRENT_RTOL_CEILING)
    if not math.isfinite(state) or state <= 0. or not math.isfinite(current) or current <= 0.:
        raise ValueError("Positive finite periodic tolerances required")
    return state, math.log1p(current)


def _trajectory_comparison(problem, first, second):
    state_difference = max(float(np.max(abs(np.asarray(a["state_u"])-np.asarray(b["state_u"]))))
                           for a, b in zip(first, second))
    endpoint_difference = abs(float(first[-1]["final_u"])-float(second[-1]["final_u"]))
    current_difference = problem._current_log_difference(first, second)
    identical = (endpoint_difference == 0. and all(
        np.array_equal(a["state_u"], b["state_u"]) and
        np.array_equal(a["current_a"], b["current_a"])
        for a, b in zip(first, second)))
    return {"max_state_difference_u": state_difference,
            "endpoint_difference_u": endpoint_difference,
            "max_current_log_difference_natural": float(current_difference),
            "exactly_identical_numerical_trajectory": bool(identical)}


def check_prefix_tail(problem, pairs):
    """Check the last four continuous pairs without asserting global accuracy."""
    state_atol, log_current_atol = _tolerances(problem)
    required_pairs = ADJACENT_TRANSITIONS_REQUIRED+1
    if len(pairs) < required_pairs:
        return {"passed": False, "reason": "fewer_than_four_integrated_pairs"}
    window = pairs[-required_pairs:]
    comparisons = [_trajectory_comparison(problem, a, b) for a, b in zip(window[:-1], window[1:])]
    trajectory_passed = all(max(row["max_state_difference_u"], row["endpoint_difference_u"]) <= state_atol
                            and row["max_current_log_difference_natural"] <= log_current_atol
                            for row in comparisons)
    endpoints = np.array([float(pair[-1]["final_u"]) for pair in window])
    differences = np.array([row["endpoint_difference_u"] for row in comparisons])
    roundoff = ROUNDOFF_ULPS*np.finfo(float).eps*max(1., float(np.max(abs(endpoints))))
    exact_repeat = all(row["exactly_identical_numerical_trajectory"] for row in comparisons)
    ratios, ratio_status = [], []
    for previous, current in zip(differences[:-1], differences[1:]):
        if min(previous, current) <= roundoff:
            ratios.append(None)
            ratio_status.append("zero_or_roundoff_dominated_inconclusive")
        else:
            ratios.append(float(current/previous))
            ratio_status.append("resolved")
    contraction_passed = all(r is not None and r < CONTRACTION_RATIO_CEILING for r in ratios)
    passed = trajectory_passed and (contraction_passed or exact_repeat)
    if passed:
        reason = "three_identical_numerical_repetitions" if exact_repeat else "three_trajectory_checks_and_resolved_local_contraction"
    elif not trajectory_passed:
        reason = "trajectory_tolerance_not_met"
    elif any(r is None for r in ratios):
        reason = "contraction_unresolved_near_roundoff_and_no_exact_repeat"
    else:
        reason = "endpoint_contraction_ratio_not_below_half"
    return {"passed": bool(passed), "reason": reason,
            "n_integrated_pairs_at_check": len(pairs),
            "required_consecutive_adjacent_comparisons": ADJACENT_TRANSITIONS_REQUIRED,
            "comparisons": comparisons,
            "state_atol": state_atol, "current_log_atol_natural": log_current_atol,
            "endpoint_roundoff_threshold_u": roundoff,
            "endpoint_local_contraction_ratios": ratios,
            "endpoint_ratio_status": ratio_status,
            "contraction_ratio_strict_ceiling": CONTRACTION_RATIO_CEILING,
            "resolved_local_contraction_passed": bool(contraction_passed),
            "exact_numerical_repetitions_passed": bool(exact_repeat),
            "scope": "Local trajectory and contraction evidence only; not a rigorous global error bound or an analytical periodic-orbit proof."}


def evaluate_prefix_checked(problem, q, max_prefix=12):
    """Continuous checked prefix, otherwise full32; never read held-out data.

    At least four pairs are integrated so three consecutive pair-to-pair
    comparisons can pass. All actual prefix traces keep their own record IDs;
    only subsequent records reuse the latest pair. The training targets and
    normalization are exactly problem.score_predictions's original masks.
    """
    if isinstance(max_prefix, bool) or not isinstance(max_prefix, (int, np.integer)) or max_prefix < 4:
        raise ValueError("max_prefix must be an integer of at least four")
    q = problem.validate_q(q) if hasattr(problem, "validate_q") else np.asarray(q, dtype=float)
    state_atol, log_current_atol = _tolerances(problem)
    ids = problem.train_ids
    series_ids = list(ids["1t1r"])
    if len(series_ids) < 4:
        raise ValueError("Prefix check requires at least four ordered series records")
    law = problem.law_factory(q, problem.ron)
    bare = problem._pair(law, float(q[problem.initial_indices["1r"]]), "1r")
    predictions = {("1r", cid, phase): trace for cid in ids["1r"]
                   for phase, trace in zip(("set", "reset"), bare)}
    state = float(q[problem.initial_indices["1t1r"]])
    pairs, checks = [], []
    search_limit = min(int(max_prefix), len(series_ids))
    shortcut = False
    for index in range(len(series_ids)):
        pair = problem._pair(law, state, "1t1r")
        pairs.append(pair)
        state = float(pair[-1]["final_u"])
        if 4 <= len(pairs) <= search_limit:
            check = check_prefix_tail(problem, pairs)
            checks.append(check)
            if check["passed"] and len(pairs) < len(series_ids):
                shortcut = True
                break
        # Once search_limit is reached without a certificate, all remaining
        # pairs are truly propagated. Later accidental closeness is not reused.
    for index, cid in enumerate(series_ids):
        pair = pairs[index] if index < len(pairs) else pairs[-1]
        for phase, trace in zip(("set", "reset"), pair):
            predictions["1t1r", cid, phase] = trace
    residual, blocks = problem.score_predictions(predictions)
    cost = float(np.dot(residual, residual)/2.)
    detail = {
        "requested_mode": "separate_prefix_checked_shortcut",
        "used_mode": "locally_checked_prefix_reuse" if shortcut else "full32",
        "approximation_used": shortcut,
        "objective_is_exact_original_record_score": not shortcut,
        "objective_scope": "Original-record score with locally checked continuation approximation; full32 frozen-candidate comparison required" if shortcut else "Full original-record ordered training score",
        "max_prefix_requested": int(max_prefix), "prefix_search_limit": search_limit,
        "max_prefix_checked": checks[-1]["n_integrated_pairs_at_check"] if checks else 0,
        "prefix_length": len(pairs),
        "n_integrated_1t_pairs": len(pairs),
        "n_reused_1t_pairs": len(series_ids)-len(pairs),
        "propagated_series_record_ids": series_ids[:len(pairs)],
        "reused_series_record_ids": series_ids[len(pairs):],
        "reused_trace_record_id": series_ids[len(pairs)-1] if shortcut else None,
        "all_actual_prefix_predictions_preserved": True,
        "series_state_reset_between_records": False,
        "fell_back_to_full32": not shortcut,
        "prefix_checks": checks,
        "state_atol": state_atol, "current_log_atol_natural": log_current_atol,
        "verified_cost": cost, "cost_includes_record_scatter": True,
        "model_dependent_cost_after_subtracting_fixed_scatter": cost-problem.scatter_constant_cost,
        "blocks": blocks,
        "parameter_names": list(getattr(problem, "parameter_names", ())),
        "evaluation_split": "training_only",
        "objective_source_sha256": problem.objective_source_sha256,
        "prefix_shortcut_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "final_frozen_candidate_requires_full32_comparison": True,
    }
    return {"residual": residual, "details": detail, "predictions": predictions}


def compare_prefix_to_full(problem, q, max_prefix=12):
    """Frozen-parameter audit of this separate shortcut against actual full32."""
    full = problem.evaluate(q, mode="full")
    prefix = evaluate_prefix_checked(problem, q, max_prefix=max_prefix)
    state_atol, log_current_atol = _tolerances(problem)
    largest_state, largest_current, largest_endpoint = 0., 0., 0.
    for key in full["predictions"]:
        a, b = full["predictions"][key], prefix["predictions"][key]
        comparison = _trajectory_comparison(problem, [a], [b])
        largest_state = max(largest_state, comparison["max_state_difference_u"])
        largest_current = max(largest_current, comparison["max_current_log_difference_natural"])
        largest_endpoint = max(largest_endpoint, comparison["endpoint_difference_u"])
    return {"status": "frozen_parameter_prefix_vs_full32_training_audit_no_refit",
            "q": np.asarray(q).tolist(), "full": full["details"], "prefix": prefix["details"],
            "checkpoint_metadata": problem.checkpoint_metadata(),
            "max_abs_state_difference_u": largest_state,
            "max_abs_endpoint_difference_u": largest_endpoint,
            "max_abs_log_current_difference_natural": largest_current,
            "cost_difference_prefix_minus_full": prefix["details"]["verified_cost"]-full["details"]["verified_cost"],
            "full32_comparison_passed": bool(max(largest_state, largest_endpoint) <= state_atol
                                             and largest_current <= log_current_atol),
            "comparison_scope": "Only this frozen parameter vector, original training waveforms and declared numerical settings."}
