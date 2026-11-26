"""Injected-trace tests for prefix reuse, not material-model evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from joint_problem import SpectralJointProblem, ROOT, HERE, TRAIN
from prefix_periodic import evaluate_prefix_checked, check_prefix_tail, compare_prefix_to_full
from validate_joint_problem import SyntheticCircuit, synthetic_law_factory


class PrefixToySimulator:
    def __init__(self, behavior):
        self.behavior, self.calls = behavior, []

    def __call__(self, command, initial_u, law, circuit, **options):
        del law, options
        final = initial_u
        if circuit.topology == "1t1r" and circuit.phase == "reset":
            if self.behavior == "exact_after_first":
                final = .25
            elif self.behavior == "contracting":
                final = .25+.1*(initial_u-.25)
            elif self.behavior == "small_drift":
                final = initial_u+1e-9
            elif self.behavior == "roundoff_drift":
                final = initial_u+1e-14
            else:
                raise ValueError("Unknown toy behavior")
        state = np.linspace(initial_u, final, len(command))
        trace = {"state_u": state, "current_a": np.asarray(command)*1e-5*np.exp(-state), "final_u": final}
        self.calls.append({"topology": circuit.topology, "phase": circuit.phase,
                           "initial_u": initial_u, "final_u": final, "trace": trace})
        return trace


def problem_for(behavior):
    simulator = PrefixToySimulator(behavior)
    circuits = {top: tuple(SyntheticCircuit(top, phase) for phase in ("set", "reset")) for top in TRAIN}
    return SpectralJointProblem(synthetic_law_factory, simulate_fn=simulator, circuits=circuits), simulator


def main():
    q = np.array([-3., -4., -6., 0., 0., 1., 2.])
    checks, examples = {}, {}
    exact, simulator = problem_for("exact_after_first")
    result = evaluate_prefix_checked(exact, q)
    detail = result["details"]
    assert detail["used_mode"] == "locally_checked_prefix_reuse"
    assert detail["n_integrated_1t_pairs"] == 5
    assert detail["prefix_checks"][-1]["reason"] == "three_identical_numerical_repetitions"
    assert len(simulator.calls) == 2+2*5
    series_calls = [c for c in simulator.calls if c["topology"] == "1t1r"]
    assert all(b["initial_u"] == a["final_u"] for a, b in zip(series_calls[:-1], series_calls[1:]))
    for pair_index, cid in enumerate(TRAIN["1t1r"][:5]):
        for phase_index, phase in enumerate(("set", "reset")):
            assert result["predictions"]["1t1r", cid, phase] is series_calls[2*pair_index+phase_index]["trace"]
    assert result["predictions"]["1t1r", 9, "set"] is result["predictions"]["1t1r", 8, "set"]
    assert result["predictions"]["1t1r", 4, "set"] is not result["predictions"]["1t1r", 8, "set"]
    checks["every_actual_prefix_trace_and_initial_transient_preserved_without_reset"] = True
    rescored, _ = exact.score_predictions(result["predictions"])
    np.testing.assert_array_equal(rescored, result["residual"])
    full = exact.evaluate(q, mode="full")
    np.testing.assert_array_equal(full["residual"], result["residual"])
    overwritten = dict(result["predictions"])
    for cid in TRAIN["1t1r"]:
        for phase in ("set", "reset"):
            overwritten["1t1r", cid, phase] = result["predictions"]["1t1r", 8, phase]
    incorrectly_reset_residual, _ = exact.score_predictions(overwritten)
    assert not np.array_equal(incorrectly_reset_residual, result["residual"])
    checks["exact_record_targets_weights_and_full32_score_preserved"] = True
    examples["exact_after_first"] = detail

    contracting, _ = problem_for("contracting")
    result = evaluate_prefix_checked(contracting, q)
    assert result["details"]["n_integrated_1t_pairs"] == 12
    assert result["details"]["prefix_checks"][-1]["resolved_local_contraction_passed"]
    assert result["details"]["approximation_used"]
    assert not result["details"]["objective_is_exact_original_record_score"]
    comparison = compare_prefix_to_full(contracting, q)
    assert comparison["full32_comparison_passed"]
    checks["resolved_contracting_tail_uses_explicitly_labeled_approximation"] = True
    checks["frozen_prefix_vs_full32_checks_state_current_endpoint_and_cost"] = True
    examples["contracting"] = comparison

    drifting, _ = problem_for("small_drift")
    result = evaluate_prefix_checked(drifting, q)
    assert result["details"]["n_integrated_1t_pairs"] == 32
    assert result["details"]["used_mode"] == "full32"
    assert result["details"]["n_reused_1t_pairs"] == 0
    assert all(row["reason"] == "endpoint_contraction_ratio_not_below_half" for row in result["details"]["prefix_checks"])
    full = drifting.evaluate(q, mode="full")
    np.testing.assert_array_equal(result["residual"], full["residual"])
    checks["small_drift_within_local_tolerances_still_falls_back_without_contraction"] = True
    examples["small_drift"] = result["details"]

    rounding, _ = problem_for("roundoff_drift")
    result = evaluate_prefix_checked(rounding, q)
    assert result["details"]["used_mode"] == "full32"
    assert all(row["reason"] == "contraction_unresolved_near_roundoff_and_no_exact_repeat" for row in result["details"]["prefix_checks"])
    checks["roundoff_dominated_nonidentical_traces_do_not_fake_contraction"] = True
    jittered_pairs = []
    for j in range(4):
        phase = {"state_u": np.array([.25+j*1e-12, .25]),
                 "current_a": np.array([1e-5*(1+j*1e-12), 2e-5]), "final_u": .25}
        jittered_pairs.append([phase, phase])
    zero_endpoint_check = check_prefix_tail(rounding, jittered_pairs)
    assert not zero_endpoint_check["passed"]
    assert all(r is None for r in zero_endpoint_check["endpoint_local_contraction_ratios"])
    checks["zero_over_zero_endpoints_require_exact_entire_trajectory_repetitions"] = True

    original_mode = exact.mode
    assert exact.periodic_state_atol == 1e-7 and exact.periodic_current_rtol == 1e-4
    evaluate_prefix_checked(exact, q, max_prefix=4)
    assert exact.mode == original_mode
    assert exact.periodic_state_atol == 1e-7 and exact.periodic_current_rtol == 1e-4
    for invalid in (3, 4.5, True):
        try:
            evaluate_prefix_checked(exact, q, max_prefix=invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid prefix limit accepted")
    checks["original_mode_tolerances_unchanged_and_insufficient_prefix_rejected"] = True
    report = {"status": "synthetic_prefix_shortcut_validation_passed_not_physics_validation",
              "all_passed": all(checks.values()), "checks": checks, "examples": examples,
              "training_arrays_sha256": exact.training_arrays_sha256,
              "n_residuals": exact.n_residuals,
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                  [Path(__file__), HERE/"prefix_periodic.py", HERE/"joint_problem.py", HERE/"validate_joint_problem.py",
                   ROOT/"analysis/constrained_shared_2026-11-03/ordered_objective.py"]},
              "scope": "Synthetic injected trajectories validate shortcut wiring and conservative decisions only. A frozen real candidate still requires full32 comparison."}
    (HERE/"prefix_periodic_validation.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status": report["status"], "checks": checks,
                      "contracting_full32_comparison": {k: v for k, v in comparison.items() if k.startswith("max_") or k in ("full32_comparison_passed", "cost_difference_prefix_minus_full")}}, indent=2))


if __name__ == "__main__":
    main()
