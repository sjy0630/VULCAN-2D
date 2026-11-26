"""Validate objective wiring with declared synthetic traces, never physics fits.

The raw observations loaded for scoring are training records only. Synthetic
simulators make periodic reuse, first-record weighting and fallback behavior
testable without depending on a material integrator or touching test data.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from joint_problem import SpectralJointProblem, ROOT, HERE, TRAIN


@dataclass(frozen=True)
class SyntheticCircuit:
    topology: str
    phase: str


def synthetic_law_factory(q, ron):
    return {"q": np.asarray(q).copy(), "ron": ron}


class SyntheticSimulator:
    """Explicitly synthetic: its parameters have no material interpretation."""
    def __init__(self, behavior):
        self.behavior, self.calls = behavior, []

    def __call__(self, command, initial_u, law, circuit, **options):
        if circuit.topology == "1r":
            final_u = initial_u*.8
        elif self.behavior == "periodic":
            final_u = .25 if circuit.phase == "reset" else initial_u
        elif self.behavior == "drifting":
            final_u = initial_u+.01
        else:
            raise ValueError("Unknown synthetic behavior")
        state = np.linspace(initial_u, final_u, len(command))
        current = command*1e-5*np.exp(-state)
        self.calls.append({"topology": circuit.topology, "phase": circuit.phase,
                           "initial_u": float(initial_u), "final_u": float(final_u),
                           "law_q": law["q"].tolist(),
                           "law_id": id(law), "initial_command_voltage_v": options["initial_command_voltage_v"]})
        return {"current_a": current, "state_u": state, "final_u": final_u}


def make_problem(behavior, *, include_reset_initiation=False):
    simulator = SyntheticSimulator(behavior)
    circuits = {top: tuple(SyntheticCircuit(top, phase) for phase in ("set", "reset")) for top in TRAIN}
    return SpectralJointProblem(synthetic_law_factory, simulate_fn=simulator, circuits=circuits,
                                include_reset_initiation=include_reset_initiation), simulator


def assert_handoffs(calls, q):
    assert len({x["law_id"] for x in calls}) == 1, "Both topologies must receive the identical material object"
    assert calls[0]["initial_u"] == q[5]
    assert calls[1]["initial_u"] == calls[0]["final_u"]
    series = [x for x in calls if x["topology"] == "1t1r"]
    assert series[0]["initial_u"] == q[6]
    assert all(b["initial_u"] == a["final_u"] for a, b in zip(series[:-1], series[1:]))


def main():
    q = np.array([-3., -4., -6., 0., 0., 1., 2.])
    checks = {}
    periodic, simulator = make_problem("periodic")
    fast = periodic.evaluate(q, mode="periodic")
    assert fast["details"]["used_mode"] == "checked_periodic_shortcut"
    assert fast["details"]["n_integrated_1t_pairs"] == 3
    assert len(simulator.calls) == 8
    assert_handoffs(simulator.calls, q)
    first = fast["predictions"]["1t1r", 4, "set"]["current_a"]
    second = fast["predictions"]["1t1r", 5, "set"]["current_a"]
    assert not np.array_equal(first, second), "A first-record transient must not be erased"
    assert fast["predictions"]["1t1r", 5, "set"] is fast["predictions"]["1t1r", 35, "set"]
    checks["first_record_transient_preserved_with_locally_periodic_remaining_records"] = True
    checks["same_law_object_and_continuous_series_state_handoffs"] = True
    simulator.calls.clear()
    full = periodic.evaluate(q, mode="full")
    assert len(simulator.calls) == 66
    assert_handoffs(simulator.calls, q)
    assert np.array_equal(fast["residual"], full["residual"])
    assert fast["details"]["verified_cost"] == full["details"]["verified_cost"]
    assert len(full["predictions"]) == (14+32)*2
    assert len(full["residual"]) == periodic.n_residuals
    checks["exact_periodic_and_full32_scores_identical_on_all_original_record_masks"] = True
    comparison = periodic.compare_modes(q)
    assert comparison["full32_comparison_passed_current_tolerance"]
    assert comparison["cost_difference_shortcut_minus_full"] == 0.
    checks["compare_modes_uses_new_simulator_and_preserves_tolerance_contract"] = True

    drifting, simulator = make_problem("drifting")
    fallback = drifting.evaluate(q, mode="periodic")
    assert fallback["details"]["fell_back_to_full32"]
    assert fallback["details"]["used_mode"] == "full32"
    assert fallback["details"]["n_integrated_1t_pairs"] == 32
    assert len(simulator.calls) == 66
    assert_handoffs(simulator.calls, q)
    direct = drifting.evaluate(q, mode="full")
    assert np.array_equal(fallback["residual"], direct["residual"])
    checks["failed_periodic_check_falls_back_to_all32_without_state_reset"] = True

    for invalid in (q[:6], np.r_[q, 1.], np.array([np.nan, *q[1:]]), np.r_[q[:5], -1., q[6]]):
        try:
            periodic.evaluate(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid seven-parameter coordinate accepted")
    checks["invalid_q_shape_nonfinite_and_negative_initial_state_rejected"] = True

    extended, extended_simulator = make_problem("periodic", include_reset_initiation=True)
    q8 = np.r_[q, 2.3]
    extended_result = extended.evaluate(q8, mode="periodic")
    assert extended.n_parameters == 8 and periodic.n_parameters == 7
    assert extended.parameter_names[-1] == "reset_initiation_log"
    assert extended.initial_indices == {"1r": 5, "1t1r": 6}
    assert all(call["law_q"] == q8.tolist() for call in extended_simulator.calls)
    assert_handoffs(extended_simulator.calls, q8)
    # The synthetic material intentionally ignores A: this checks that the
    # adapter passes it unchanged, not whether the new material term is valid.
    assert np.array_equal(extended_result["residual"], fast["residual"])
    for invalid in (q, np.r_[q, -1.], np.r_[q, np.nan], np.r_[q, np.inf]):
        try:
            extended.validate_q(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid opted-in RESET initiation coordinate accepted")
    try:
        periodic.validate_q(q8)
    except ValueError:
        pass
    else:
        raise AssertionError("Default seven-parameter interface silently enabled A_RESET")
    checks["explicit_eight_parameter_opt_in_passes_untransformed_A_and_keeps_initial_indices"] = True
    checks["dimension_mismatch_negative_nonfinite_A_rejected_and_default_stays_seven"] = True

    with tempfile.TemporaryDirectory(prefix="spectral-objective-wiring-") as temp:
        destination = Path(temp)/"synthetic_training_evaluation.json"
        report = periodic.export_training_evaluation(q, destination, compare_modes=True)
        artifact = np.load(destination.with_suffix(".npz"))
        assert report["details"]["used_mode"] == "full32"
        assert np.array_equal(artifact["residual"], full["residual"])
        for key in artifact.files:
            if key.startswith("1t1r_cycle_"):
                assert int(key.split("_")[2]) in TRAIN["1t1r"]
            elif key.startswith("1r_cycle_"):
                assert int(key.split("_")[2]) in TRAIN["1r"]
        artifact.close()
        checks["export_defaults_to_full32_and_only_contains_training_record_ids"] = True
        extended_path = Path(temp)/"synthetic_extended_training_evaluation.json"
        extended_report = extended.export_training_evaluation(q8, extended_path)
        assert extended_report["n_parameters"] == 8
        assert extended_report["include_reset_initiation"] is True
        assert extended_report["parameter_names"][-1] == "reset_initiation_log"
        assert extended_report["checkpoint_metadata"]["reset_initiation_parameter_index"] == 7
        assert extended_report["details"]["parameter_coordinate_types"][-1] == "natural_log_drive_amplitude_untransformed"
        with np.load(extended_path.with_suffix(".npz")) as extended_artifact:
            np.testing.assert_array_equal(extended_artifact["q"], q8)
        checks["extended_dimension_coordinate_metadata_and_export_match_opt_in"] = True

    metadata = periodic.checkpoint_metadata()
    assert metadata["initial_parameter_indices"] == {"1r": 5, "1t1r": 6}
    assert metadata["periodic_current_rtol"] == 1e-4
    assert metadata["periodic_state_atol"] == 1e-7
    assert metadata["fallback_full"] is True
    checks["initial_indices_and_preexisting_periodic_tolerances_preserved"] = True
    extended_metadata = extended.checkpoint_metadata()
    assert extended_metadata["periodic_current_rtol"] == metadata["periodic_current_rtol"]
    assert extended_metadata["periodic_state_atol"] == metadata["periodic_state_atol"]
    assert extended_metadata["fallback_full"] is True
    output = {"status": "synthetic_objective_wiring_validation_passed_not_physics_validation",
              "checks": checks, "all_passed": all(checks.values()),
              "n_residuals": periodic.n_residuals,
              "training_arrays_sha256": periodic.training_arrays_sha256,
              "scatter_constant_cost": periodic.scatter_constant_cost,
              "shared_Ron_from_training_ohm": periodic.ron,
              "synthetic_periodic_cost": fast["details"]["verified_cost"],
              "synthetic_full32_cost": full["details"]["verified_cost"],
              "synthetic_fallback_cost": fallback["details"]["verified_cost"],
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                  [Path(__file__), HERE/"joint_problem.py", ROOT/"analysis/constrained_shared_2026-11-03/ordered_objective.py"]},
              "limitations": ["Synthetic simulator only verifies objective wiring, never validates the spectral material model or its numerical integration.",
                              "No test or held-out record arrays are loaded or evaluated."]}
    (HERE/"objective_validation.json").write_text(json.dumps(output, indent=2, allow_nan=False)+"\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
