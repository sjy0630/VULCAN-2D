"""Check real training masks and synthetic history handoffs; no fitting."""
from pathlib import Path
import json
import re
import time
from unittest.mock import patch

import numpy as np

from segmented_problem import (SegmentedBareSpectralProblem, GroupSpectralProblem,
                               TRAINING_SEGMENTS, HERE, make_problem)
from validate_joint_problem import make_problem as synthetic_joint
from ordered_objective import TRAIN


def main():
    started = time.monotonic()
    checks, accessed = {}, []
    original_load = np.load

    class TrainingOnly:
        def __init__(self, archive):
            self.archive = archive
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.archive.close()
        def __getitem__(self, key):
            match = re.fullmatch(r"(1r|1t1r)_cycle_(\d+)_(set|reset)_(voltage|current)", key)
            assert match and int(match[2]) in TRAIN[match[1]], "Nontraining data requested"
            accessed.append(key)
            return self.archive[key]

    with patch("ordered_objective.np.load", lambda *a, **kw: TrainingOnly(original_load(*a, **kw))):
        joint, simulator = synthetic_joint("drifting")
    group = GroupSpectralProblem(joint, "1r")
    problem = SegmentedBareSpectralProblem(group)
    q = np.array([-3., -4., -6., 0., 0., 1., 2.])
    result = problem.evaluate(q)
    calls = simulator.calls.copy()
    assert len(calls) == 28 and len({c["law_id"] for c in calls}) == 1
    assert all(c["topology"] == "1r" for c in calls)
    offset = 0
    for i, ids in enumerate(TRAINING_SEGMENTS):
        segment = calls[offset:offset + 2*len(ids)]
        assert segment[0]["initial_u"] == q[5+i]
        assert all(b["initial_u"] == a["final_u"] for a, b in zip(segment[:-1], segment[1:]))
        offset += 2*len(ids)
    assert calls[6]["initial_u"] != calls[5]["final_u"], "Synthetic gap should expose false bridging"
    assert [x["record_id"] for x in result["details"]["state_boundaries"]] == TRAIN["1r"]
    assert len(result["predictions"]) == 28
    checks["one_material_object_and_exact_handoffs_inside_both_segments"] = True
    checks["only_cycle1_and_unobserved_cycle5_have_initial_state_inputs"] = True
    checks["no_missing_cycle4_or_inactive_series_simulation"] = True

    # Recompute the raw-record vector directly, independently of the old
    # scorer's temporary inactive slots. Weights and ordering are unchanged.
    manual = []
    for phase in ("set", "reset"):
        predicted = abs(np.array([result["predictions"]["1r", cid, phase]["current_a"]
                                  for cid in TRAIN["1r"]]))
        observed = problem.raw_training["1r", phase]
        for score_group in group.groups["1r", phase]:
            mask = score_group["mask"]
            manual.append((np.log10(predicted[mask]) - np.log10(observed[mask]))
                          / score_group["normalization"])
    manual = np.concatenate(manual)
    assert len(manual) == 5583
    assert np.array_equal(manual, result["residual"])
    checks["5583_residuals_bit_equal_to_independent_original_mask_calculation"] = True

    changed = q.copy(); changed[6] += 1.
    second = problem.evaluate(changed)
    assert all(np.array_equal(result["predictions"]["1r", cid, phase]["state_u"],
                              second["predictions"]["1r", cid, phase]["state_u"])
               for cid in TRAINING_SEGMENTS[0] for phase in ("set", "reset"))
    assert not np.array_equal(result["predictions"]["1r", 5, "set"]["state_u"],
                              second["predictions"]["1r", 5, "set"]["state_u"])
    assert result["details"]["state_boundaries"][-1]["RESET_final_u"] == result["predictions"]["1r", 15, "reset"]["final_u"]
    checks["gap_state_affects_second_segment_only_and_cycle15_terminal_state_retained"] = True

    for invalid in (q[:-1], np.r_[q, 0.], np.r_[q[:5], -1., 2.],
                    np.r_[q[:6], np.nan]):
        try:
            problem.validate_q(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid history coordinates accepted")
    checks["invalid_state_coordinates_rejected"] = True
    assert len(accessed) == 184
    checks["only_original_training_arrays_accessed"] = True

    injected = problem.checkpoint_metadata()
    assert injected["beta_fixed"] is None and injected["reset_initiation_log_fixed"] is None
    checks["injected_factory_not_mislabeled_as_fixed_physical_law"] = True

    real = make_problem()  # Construction only: no real ODE here.
    metadata = real.checkpoint_metadata()
    assert metadata["n_parameters"] == 7 and metadata["n_residuals"] == 5583
    assert metadata["beta_fixed"] == 1. and metadata["reset_initiation_log_fixed"] == 0.
    assert metadata["training_segments"] == [list(x) for x in TRAINING_SEGMENTS]
    checks["production_factory_and_identity_provenance_preserved"] = True
    report = dict(status="segmented_history_wiring_validated_no_fit_no_real_ODE",
                  checks=checks, all_passed=all(checks.values()),
                  n_checks=len(checks), training_arrays_accessed=len(accessed),
                  real_ODE_pairs=0, elapsed_s=time.monotonic()-started,
                  checkpoint_example=metadata)
    (HERE/"segmented_validation.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:report[k] for k in ("status","all_passed","n_checks","elapsed_s")},indent=2))


if __name__ == "__main__":
    main()
