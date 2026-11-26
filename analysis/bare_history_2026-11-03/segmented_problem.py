"""Training-only bare spectral history, with one explicit missing-cycle boundary.

Cycles 1--3 and 5--15 each propagate continuously. Cycle4 was measured but
not supplied: initial_u5 is an unobserved gap-entry state, not a simulated
reset or a prediction across that gap. No material parameter varies by cycle.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREVIOUS = ROOT / "analysis/parameter_transfer_2026-11-03"
if str(PREVIOUS) not in sys.path:
    sys.path.insert(0, str(PREVIOUS))
from group_problem import GroupSpectralProblem, make_problem as make_group

IDENTITY = HERE / "identity_confirmation.json"
TRAINING_SEGMENTS = ((1, 2, 3), tuple(range(5, 16)))
PARAMETER_NAMES = ("log10_gamma_c_ev", "log10_J_SET_A", "log10_J_RESET_A",
                   "log10_k_SET_per_V", "log10_k_RESET_per_V",
                   "initial_u_cycle1", "unobserved_gap_entry_u_cycle5")


class SegmentedBareSpectralProblem:
    def __init__(self, group):
        if not isinstance(group, GroupSpectralProblem) or group.topology != "1r":
            raise TypeError("Requires the original bare training group")
        identity = json.loads(IDENTITY.read_text())
        confirmed = identity["confirmed"]
        if not (confirmed["bare_physical_device_count"] == 1
                and confirmed["numeric_chronological_order"]
                and confirmed["same_label_SET_then_RESET"]
                and confirmed["missing_measured_cycles"] == [4, 23]):
            raise ValueError("The recorded history does not support this segmentation")
        self.group = group
        self.train_ids = {"1r": [cid for segment in TRAINING_SEGMENTS for cid in segment]}
        if self.train_ids != group.train_ids:
            raise ValueError("Training records must match the frozen original masks")
        self.waveforms, self.raw_training = group.waveforms, group.raw_training
        if any(self.waveforms["1r", phase][0] != 0.
               or self.waveforms["1r", phase][-1] != 0. for phase in ("set", "reset")):
            raise ValueError("Pair-to-pair propagation requires the native zero endpoints")
        self.n_residuals = group.n_residuals
        self.n_parameters = len(PARAMETER_NAMES)
        self.parameter_names = PARAMETER_NAMES
        self.options = group.options

    def validate_q(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (7,) or not np.all(np.isfinite(q)) or np.any(q[5:] < 0.):
            raise ValueError("Need five finite log10 material coordinates and two nonnegative segment states")
        return q

    def evaluate(self, q):
        q = self.validate_q(q)
        # The old factory's inactive series initial state is unused. Only its
        # first five coordinates select the material; all 14 pairs share it.
        joint_q = self.group.to_joint_q(np.r_[q[:5], q[5]])
        law = self.group.joint.law_factory(joint_q, self.group.ron)
        predictions, boundaries, segments = {}, [], []
        for index, ids in enumerate(TRAINING_SEGMENTS):
            state = float(q[5 + index])
            initial = state
            for cid in ids:
                pair_initial = state
                pair = self.group.joint._pair(law, state, "1r")
                if (float(pair[0]["state_u"][0]) != pair_initial
                        or float(pair[1]["state_u"][0]) != float(pair[0]["final_u"])):
                    raise RuntimeError("Simulator failed state inheritance at a native zero endpoint")
                for phase, trace in zip(("set", "reset"), pair):
                    predictions["1r", cid, phase] = trace
                state = float(pair[-1]["final_u"])
                boundaries.append(dict(record_id=cid, SET_initial_u=pair_initial,
                                       SET_final_RESET_initial_u=float(pair[0]["final_u"]),
                                       RESET_final_u=state))
            segments.append(dict(record_ids=list(ids), initial_u=initial, final_u=state,
                                 initial_source="first_record_preparation" if index == 0
                                 else "unobserved_state_after_missing_measured_cycle4"))
        residual, blocks = self.group.score_predictions(predictions)
        details = dict(
            status="conditional_segmented_bare_training_evaluation_not_a_complete_joint_model",
            evaluation_split="original_training_only", n_integrated_pairs=14,
            n_material_parameters=5, n_segment_initial_states=2, n_parameters=7,
            parameter_names=list(PARAMETER_NAMES), n_residuals=self.n_residuals,
            training_segments=segments, state_boundaries=boundaries,
            within_segment_state_resets=False, missing_measured_cycle4_simulated=False,
            gap_entry_state_is_observed=False, parameters_shared_across_all_segments=True,
            verified_cost=float(np.dot(residual, residual) / 2.), blocks=blocks,
            objective_normalization="Original 5583 bare residual entries, masks and weights unchanged",
            repeated_pair_scatter_subtracted=False,
            timing_policy="Common effective command-path rates; no physical seconds or waiting dynamics inferred",
            future_holdout_state_source={"cycle16": "inherit cycle15 RESET final_u",
                                         "cycle24": "unidentified after missing measured cycle23; not an unconditional prediction"})
        return dict(residual=residual, details=details, predictions=predictions)

    def residual(self, q):
        return self.evaluate(q)["residual"]

    def checkpoint_metadata(self):
        original = self.group.checkpoint_metadata()
        hashes = dict(original["source_sha256"])
        hashes.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (IDENTITY, Path(__file__).resolve())})
        return dict(status="segmented_bare_history_adapter_not_a_fitted_model",
                    source_sha256=hashes, train_ids=self.train_ids,
                    training_arrays_sha256=original["source_joint_training_arrays_sha256"],
                    training_segments=[list(x) for x in TRAINING_SEGMENTS],
                    parameter_names=list(PARAMETER_NAMES), n_parameters=7,
                    n_residuals=self.n_residuals, integrator_options=self.options,
                    source_joint_residual_slice=original["source_joint_residual_slice"],
                    material_factory_scope=("frozen_production_spectral_factory"
                                            if "frozen_spectral_shape" in original
                                            else "injected_factory_physical_parameters_unspecified"),
                    beta_fixed=original.get("beta_fixed"),
                    reset_initiation_log_fixed=original.get("reset_initiation_log_fixed"),
                    frozen_spectral_shape=original.get("frozen_spectral_shape"),
                    loaded_data_scope=original["loaded_data_scope"],
                    scope="Same material object for all14 pairs; only initial states at1 and after missing4. No data from holdout, no periodic reuse, no missing waveform invention.")


def make_problem(integrator_options=None):
    return SegmentedBareSpectralProblem(make_group("1r", integrator_options=integrator_options, mode="full"))
