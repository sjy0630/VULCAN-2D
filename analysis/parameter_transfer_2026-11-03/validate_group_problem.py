"""Validate selected-group wiring/scoring, without real ODE runs or fitting."""
from pathlib import Path
import hashlib
import json
import math
import re
import tempfile
import time
from unittest.mock import patch

import numpy as np

from group_problem import GroupSpectralProblem, SpectralJointProblem, TRAIN, ROOT, make_problem
from validate_joint_problem import SyntheticCircuit, SyntheticSimulator, synthetic_law_factory

HERE = Path(__file__).resolve().parent


def joint_problem(behavior):
    simulator = SyntheticSimulator(behavior)
    circuits = {top: tuple(SyntheticCircuit(top, phase) for phase in ("set", "reset")) for top in TRAIN}
    problem = SpectralJointProblem(synthetic_law_factory, simulate_fn=simulator, circuits=circuits)
    return problem, simulator


def verify_calls(calls, topology, initial, expected):
    assert len(calls) == expected
    assert {c["topology"] for c in calls} == {topology}, "Inactive topology was simulated"
    assert len({c["law_id"] for c in calls}) == 1
    assert calls[0]["initial_u"] == initial
    assert all(b["initial_u"] == a["final_u"] for a, b in zip(calls[:-1], calls[1:]))


def verify_selected_outputs(result, topology):
    assert {key[0] for key in result["predictions"]} == {topology}
    assert set(result["details"]["blocks"]) == {topology+"_set", topology+"_reset"}
    assert result["details"]["simulated_topologies"] == [topology]


def main():
    start = time.monotonic(); checks = {}; results = []
    accessed = []
    original_load = np.load

    class GuardedTrainingArchive:
        def __init__(self, archive):
            self.archive = archive
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.archive.close()
        def __getitem__(self, key):
            match = re.fullmatch(r"(1r|1t1r)_cycle_(\d+)_(set|reset)_(voltage|current)", key)
            assert match and int(match[2]) in TRAIN[match[1]], "Nontraining array requested"
            accessed.append(key)
            return self.archive[key]

    with patch("ordered_objective.np.load", lambda *a, **kw: GuardedTrainingArchive(original_load(*a, **kw))):
        joint, simulator = joint_problem("periodic")
    assert len(accessed) == (14+32)*4
    checks["inherited_loader_requests_only_original_training_arrays"] = True
    q7 = np.array([-3., -4., -6., 0., 0., 1., 2.])
    full_reference = joint.evaluate(q7, mode="full")
    periodic_reference = joint.evaluate(q7, mode="periodic")
    groups = {top: GroupSpectralProblem(joint, top) for top in TRAIN}
    assert joint.n_residuals == 27023
    assert sum(group.n_residuals for group in groups.values()) == joint.n_residuals
    assert math.isclose(sum(group.scatter_constant_cost for group in groups.values()),
                        joint.scatter_constant_cost, rel_tol=1e-15)
    for topology, group in groups.items():
        q6 = group.from_joint_q(q7)
        mapped = group.to_joint_q(q6)
        assert np.array_equal(mapped[:5], q7[:5])
        active_index = 5 if topology == "1r" else 6
        inactive_index = 6 if topology == "1r" else 5
        assert mapped[active_index] == q7[active_index] and mapped[inactive_index] == 0.
        simulator.calls.clear()
        full = group.evaluate(q6, mode="full")
        verify_calls(simulator.calls, topology, q6[5], 2 if topology == "1r" else 64)
        verify_selected_outputs(full, topology)
        reference = full_reference["residual"][group.joint_residual_slice]
        assert full["residual"].tobytes() == reference.tobytes()
        assert full["details"]["verified_cost"] == float(np.dot(reference, reference)/2.)
        for key, value in full["details"]["blocks"].items():
            assert value == full_reference["details"]["blocks"][key]
        for key, trace in full["predictions"].items():
            assert np.array_equal(trace["current_a"], full_reference["predictions"][key]["current_a"])
            assert np.array_equal(trace["state_u"], full_reference["predictions"][key]["state_u"])
        simulator.calls.clear()
        fast = group.evaluate(q6, mode="periodic")
        verify_calls(simulator.calls, topology, q6[5], 2 if topology == "1r" else 6)
        verify_selected_outputs(fast, topology)
        assert fast["residual"].tobytes() == periodic_reference["residual"][group.joint_residual_slice].tobytes()
        if topology == "1t1r":
            assert not np.array_equal(fast["predictions"][topology, 4, "set"]["current_a"],
                                      fast["predictions"][topology, 5, "set"]["current_a"])
            assert fast["predictions"][topology, 5, "set"] is fast["predictions"][topology, 35, "set"]
        comparison = group.compare_modes(q6)
        assert comparison["comparison_passed_current_tolerance"]
        assert comparison["cost_difference_shortcut_minus_full"] == 0.
        assert set(comparison["phase_differences"]) == {topology+"_set", topology+"_reset"}
        for invalid in (q6[:-1], np.r_[q6, 1.], np.r_[q6[:5], -1.], np.r_[np.nan, q6[1:]]):
            try:
                group.validate_q(invalid)
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid group coordinates accepted")
        try:
            group.score_predictions(full_reference["predictions"])
        except ValueError:
            pass
        else:
            raise AssertionError("Unselected predictions accepted by group scorer")
        with tempfile.TemporaryDirectory(prefix="group-objective-") as temp:
            path = Path(temp)/"group.json"
            report = group.export_training_evaluation(q6, path, compare_modes=True)
            assert set(report["details"]["blocks"]) == {topology+"_set", topology+"_reset"}
            with np.load(path.with_suffix(".npz")) as archive:
                for key in archive.files:
                    if key not in ("q", "residual"):
                        assert key.startswith(topology+"_cycle_")
                assert archive["residual"].tobytes() == full["residual"].tobytes()
        results.append(dict(topology=topology, n_residuals=group.n_residuals,
                            cost=full["details"]["verified_cost"],
                            residual_slice=[group.joint_residual_slice.start, group.joint_residual_slice.stop]))
    assert math.isclose(sum(row["cost"] for row in results), full_reference["details"]["verified_cost"], rel_tol=2e-15)
    checks["both_group_residuals_bit_equal_to_joint_slices_with_original_weights"] = True
    checks["group_costs_and_scatter_sum_to_joint_without_renormalization"] = True
    checks["inactive_topology_never_simulated_and_inactive_u0_fixed_zero"] = True
    checks["single_prepared_bare_pair_scored_against_all14_records"] = True
    checks["series_cycle4_initial_once_continuous32_and_periodic_transient_preserved"] = True
    checks["all_returned_details_metrics_predictions_and_exports_selected_only"] = True
    checks["invalid_q_and_unselected_prediction_input_rejected"] = True

    drift, simulator = joint_problem("drifting")
    group = GroupSpectralProblem(drift, "1t1r")
    q6 = group.from_joint_q(q7)
    reference = drift.evaluate(q7, mode="full")
    simulator.calls.clear()
    fallback = group.evaluate(q6, mode="periodic")
    verify_calls(simulator.calls, "1t1r", q6[5], 64)
    assert fallback["details"]["fell_back_to_full32"]
    assert fallback["residual"].tobytes() == reference["residual"][group.joint_residual_slice].tobytes()
    checks["failed_periodic_check_falls_back_to_full32_without_bare_simulation_or_state_reset"] = True

    frozen = make_problem("1r")  # Constructs only; performs no real ODE.
    law = frozen.joint.law_factory(frozen.to_joint_q(groups["1r"].from_joint_q(q7)), frozen.ron)
    assert law.exact_law.beta == 1. and law.exact_law.reset_initiation_log == 0.
    metadata = frozen.checkpoint_metadata()
    assert metadata["beta_fixed"] == 1. and metadata["reset_initiation_log_fixed"] == 0.
    assert metadata["source_joint_n_residuals"] == 27023
    assert metadata["inactive_topology_simulated"] is False
    assert metadata["n_parameters"] == 6
    checks["default_factory_fixed_shape_table_beta1_A0_and_checkpoint_hashes"] = True
    report = dict(status="group_objective_wiring_validated_no_real_ODE_no_optimization",
                  all_passed=all(checks.values()), checks=checks, group_results=results,
                  original_joint_n_residuals=27023, original_joint_synthetic_cost=full_reference["details"]["verified_cost"],
                  accessed_training_array_count=len(accessed), elapsed_s=time.monotonic()-start,
                  checkpoint_example=metadata,
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [Path(__file__), HERE/"group_problem.py"]},
                  limitations=["Synthetic simulations verify wiring/scoring, not physical accuracy.",
                               "No real candidate evaluation, optimization, heldout access or claim of group feasibility."])
    (HERE/"group_validation.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key:report[key] for key in ("status", "all_passed", "checks", "group_results", "elapsed_s")}, indent=2))


if __name__ == "__main__":
    main()
