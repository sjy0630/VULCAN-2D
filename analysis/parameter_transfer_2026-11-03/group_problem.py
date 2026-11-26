"""Training-only group existence/transfer diagnostic, not a joint final model.

The original joint scorer defines every mask, residual order and normalization.
Only the chosen topology is simulated. Its residual is an unchanged slice of
the original 27,023-dimensional vector; no group renormalization is applied.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SPECTRAL_DIR = ROOT / "analysis/spectral_dynamic_2026-11-03"
for directory in (ROOT, SPECTRAL_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from joint_problem import SpectralJointProblem, PARAMETER_NAMES, TRAIN
from vulcan2d.shared_spectral import SpectralShape, SpectralLaw
from vulcan2d.shared_spectral_ivp import SpectralTable, TabulatedSpectralLaw

TABLE_PATH = SPECTRAL_DIR / "transport_g010_v0025.npz"
GROUP_PARAMETER_NAMES = PARAMETER_NAMES[:5] + ("initial_u0",)


class GroupSpectralProblem:
    """Compose one selected group with an unchanged SpectralJointProblem.

    Supplying a custom joint problem is useful for simulator-wiring tests.
    Production callers should use make_problem(), which fixes beta=1, A=0,
    spectral shape/table and the original circuits.
    """

    def __init__(self, joint_problem, topology):
        if topology not in TRAIN:
            raise ValueError("topology must be '1r' or '1t1r'")
        if not isinstance(joint_problem, SpectralJointProblem):
            raise TypeError("joint_problem must be SpectralJointProblem")
        if joint_problem.n_parameters != 7 or joint_problem.include_reset_initiation:
            raise ValueError("Group adapter requires the original seven-coordinate A_RESET=0 interface")
        self.joint = joint_problem
        self.topology = topology
        self.mode = joint_problem.mode
        self.parameter_names = GROUP_PARAMETER_NAMES
        self.n_parameters = 6
        self.ron = joint_problem.ron
        self.options = joint_problem.options
        self.train_ids = {topology: list(TRAIN[topology])}
        self.initial_indices = {topology: 5}
        self.waveforms = {key: value for key, value in joint_problem.waveforms.items() if key[0] == topology}
        self.groups = {key: value for key, value in joint_problem.groups.items() if key[0] == topology}
        self.raw_training = {key: value for key, value in joint_problem.raw_training.items() if key[0] == topology}
        self.periodic_current_rtol = joint_problem.periodic_current_rtol
        self.periodic_state_atol = joint_problem.periodic_state_atol
        self.fallback_full = joint_problem.fallback_full
        offset, start, stop = 0, None, None
        for top in TRAIN:
            size = sum(group["n"] for phase in ("set", "reset")
                       for group in joint_problem.groups[top, phase])
            if top == topology:
                start, stop = offset, offset + size
            offset += size
        if offset != joint_problem.n_residuals:
            raise ValueError("Joint residual layout differs from inherited training groups")
        self.joint_residual_slice = slice(start, stop)
        self.n_residuals = stop - start
        self.scatter_constant_cost = sum(
            sum(group["observation_scatter_mse"] for group in groups)/(2.*len(groups))
            for groups in self.groups.values())
        self.prediction_keys = {(topology, cid, phase) for cid in TRAIN[topology]
                                for phase in ("set", "reset")}

    def validate_q(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (6,) or not np.all(np.isfinite(q)) or q[5] < 0.:
            raise ValueError("Group q requires five finite log10 coordinates and one nonnegative initial u")
        return q

    def to_joint_q(self, q):
        q = self.validate_q(q)
        result = np.zeros(7)
        result[:5] = q[:5]
        result[5 if self.topology == "1r" else 6] = q[5]
        return result

    def from_joint_q(self, q):
        q = self.joint.validate_q(q)
        return np.r_[q[:5], q[5 if self.topology == "1r" else 6]]

    def score_predictions(self, predictions):
        """Use the exact original scorer, retaining only selected-group outputs.

        Unselected training observations fill its unused API slots and yield
        exactly zero residuals. They are not simulated predictions, are never
        exported, and are not included in any returned metric or cost.
        """
        if set(predictions) != self.prediction_keys:
            raise ValueError("Predictions must contain exactly the selected training topology/records/phases")
        temporary = dict(predictions)
        for top, ids in TRAIN.items():
            if top == self.topology:
                continue
            for phase in ("set", "reset"):
                for row, cid in enumerate(ids):
                    temporary[top, cid, phase] = self.joint.raw_training[top, phase][row]
        vector, all_details = self.joint.score_predictions(temporary)
        selected = self.joint_residual_slice
        if np.any(vector[:selected.start] != 0.) or np.any(vector[selected.stop:] != 0.):
            raise RuntimeError("Inactive scorer compatibility slots did not give exactly zero residual")
        details = {f"{self.topology}_{phase}": all_details[f"{self.topology}_{phase}"]
                   for phase in ("set", "reset")}
        return vector[selected].copy(), details

    def evaluate(self, q, *, mode=None):
        q = self.validate_q(q)
        mode = self.mode if mode is None else mode
        if mode not in ("full", "periodic"):
            raise ValueError("mode must be full or periodic")
        joint_q = self.to_joint_q(q)
        law = self.joint.law_factory(joint_q, self.ron)
        predictions = {}
        if self.topology == "1r":
            pair = self.joint._pair(law, float(q[5]), "1r")
            for cid in TRAIN["1r"]:
                for phase, trace in zip(("set", "reset"), pair):
                    predictions["1r", cid, phase] = trace
            periodic = None
            use_periodic = fallback = False
            n_pairs, used_mode = 1, "single_prepared_bare_pair"
        else:
            state, pairs = float(q[5]), []
            for _ in range(3):
                pair = self.joint._pair(law, state, "1t1r")
                pairs.append(pair)
                state = pair[-1]["final_u"]
            periodic = self.joint._periodicity(pairs)
            use_periodic = mode == "periodic" and periodic["passed"]
            fallback = mode == "periodic" and not periodic["passed"]
            if fallback and not self.fallback_full:
                raise RuntimeError("Third-pair periodicity check failed; use full32 mode")
            if not use_periodic:
                for _ in range(3, len(TRAIN["1t1r"])):
                    pair = self.joint._pair(law, state, "1t1r")
                    pairs.append(pair)
                    state = pair[-1]["final_u"]
            for index, cid in enumerate(TRAIN["1t1r"]):
                pair = pairs[min(index, 1)] if use_periodic else pairs[index]
                for phase, trace in zip(("set", "reset"), pair):
                    predictions["1t1r", cid, phase] = trace
            n_pairs = len(pairs)
            used_mode = "checked_periodic_shortcut" if use_periodic else "full32"
        residual, blocks = self.score_predictions(predictions)
        cost = float(np.dot(residual, residual)/2.)
        exact = not use_periodic or periodic["exact_numerical_fixed_point"]
        details = dict(
            status="separate_group_existence_or_transfer_diagnostic_not_final_joint_model",
            selected_topology=self.topology, evaluation_split="training_only",
            parameter_names=list(self.parameter_names), n_parameters=6,
            requested_mode=mode, used_mode=used_mode, periodicity=periodic,
            objective_is_exact_original_record_score=exact,
            fell_back_to_full32=fallback, n_integrated_selected_pairs=n_pairs,
            verified_cost=cost, cost_includes_record_scatter=True,
            model_dependent_cost_after_subtracting_fixed_scatter=cost-self.scatter_constant_cost,
            blocks=blocks, n_residuals=self.n_residuals,
            objective_normalization="unchanged slice of original joint residual; no group rescaling",
            simulated_topologies=[self.topology],
            source_joint_residual_slice=[self.joint_residual_slice.start, self.joint_residual_slice.stop],
        )
        if self.topology == "1t1r":
            details.update(n_integrated_1t_pairs=n_pairs, first_record_id=4,
                           remaining_record_ids=list(range(5, 36)),
                           series_state_reset_between_records=False,
                           final_candidate_requires_full32_comparison=True)
        else:
            details["bare_record_chronology_policy"] = "One prepared pair scored against14 records; no inter-record chronology assumed"
        return {"residual": residual, "details": details, "predictions": predictions}

    def residual(self, q, return_details=False, *, mode=None):
        result = self.evaluate(q, mode=mode)
        return (result["residual"], result["details"]) if return_details else result["residual"]

    def compare_modes(self, q):
        full, shortcut = self.evaluate(q, mode="full"), self.evaluate(q, mode="periodic")
        differences, maximum = {}, 0.
        for phase in ("set", "reset"):
            worst = max(self.joint._current_log_difference(
                [full["predictions"][self.topology, cid, phase]],
                [shortcut["predictions"][self.topology, cid, phase]]) for cid in TRAIN[self.topology])
            maximum = max(maximum, worst)
            key = f"{self.topology}_{phase}"
            differences[key] = dict(max_abs_log_current_difference_natural=worst,
                weighted_total_mse_difference_shortcut_minus_full=
                    shortcut["details"]["blocks"][key]["weighted_total_mse"]-
                    full["details"]["blocks"][key]["weighted_total_mse"])
        return dict(status="selected_training_group_full_vs_periodic_no_refit",
                    selected_topology=self.topology, full=full["details"], shortcut=shortcut["details"],
                    phase_differences=differences,
                    cost_difference_shortcut_minus_full=shortcut["details"]["verified_cost"]-full["details"]["verified_cost"],
                    max_abs_log_current_difference_natural=maximum,
                    comparison_passed_current_tolerance=bool(maximum <= np.log1p(self.periodic_current_rtol)))

    def checkpoint_metadata(self):
        source = self.joint.checkpoint_metadata()
        hashes = dict(source["source_sha256"])
        hashes[str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        metadata = dict(
            status="group_diagnostic_not_final_shared_parameter_model", selected_topology=self.topology,
            parameter_names=list(self.parameter_names), n_parameters=6,
            parameter_coordinate_types=["log10"]*5+["linear_state"],
            source_joint_parameter_names=list(PARAMETER_NAMES),
            source_joint_initial_index=5 if self.topology == "1r" else 6,
            inactive_joint_initial_state=0., inactive_topology_simulated=False,
            train_ids=self.train_ids, n_residuals=self.n_residuals,
            source_joint_n_residuals=self.joint.n_residuals,
            source_joint_residual_slice=[self.joint_residual_slice.start, self.joint_residual_slice.stop],
            source_joint_training_arrays_sha256=source["training_arrays_sha256"],
            source_sha256=hashes, requested_mode=self.mode, integrator_options=self.options,
            periodic_current_rtol=self.periodic_current_rtol, periodic_state_atol=self.periodic_state_atol,
            fallback_full=self.fallback_full, cost_includes_record_scatter=True,
            observation_scatter_constant_cost=self.scatter_constant_cost,
            objective="Selected topology's exact original masks/branch weights; no normalization change",
            scorer_compatibility="Unused slots receive their training observations only inside original scorer; zero inactive residual discarded; no inactive metrics or predictions exposed",
            loaded_data_scope="Inherited loader reads both training groups only; selected group scored, no holdout arrays accessed",
            initial_state_policy=("one prepared bare pair for14 records" if self.topology == "1r" else "cycle4 initial state used once; continuous through35"),
            timing_policy="Effective voltage-path rates, no physical seconds inferred",
        )
        if hasattr(self, "frozen_spectral_shape"):
            metadata.update(frozen_spectral_shape=self.frozen_spectral_shape,
                            beta_fixed=1., reset_initiation_log_fixed=0.,
                            frozen_table_path=str(TABLE_PATH))
        return metadata

    def export_training_evaluation(self, q, output_json, *, mode="full", compare_modes=False, export_traces=True):
        q = self.validate_q(q)
        result = self.evaluate(q, mode=mode)
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        report = dict(status="frozen_selected_group_training_evaluation_not_final_joint_model",
                      q=q.tolist(), selected_topology=self.topology,
                      checkpoint_metadata=self.checkpoint_metadata(), details=result["details"])
        if compare_modes:
            report["full_vs_periodic"] = self.compare_modes(q)
        if export_traces:
            arrays = {"q": q, "residual": result["residual"]}
            for (top, cid, phase), trace in result["predictions"].items():
                prefix = f"{top}_cycle_{cid}_{phase}"
                arrays[prefix+"_command_V"] = self.waveforms[top, phase]
                for key in ("current_a", "state_u", "hbn_voltage_v", "transistor_voltage_v"):
                    if key in trace:
                        arrays[prefix+"_"+key] = np.asarray(trace[key])
                arrays[prefix+"_final_u"] = np.asarray(trace["final_u"])
            np.savez_compressed(path.with_suffix(".npz"), **arrays)
            report["training_trace_artifact"] = str(path.with_suffix(".npz").resolve())
        path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        return report


def make_problem(topology, integrator_options=None, mode="periodic"):
    """Fixed-spectrum beta1/A0 group objective; no simulation on construction."""
    if topology not in TRAIN:
        raise ValueError("topology must be '1r' or '1t1r'")
    shape = SpectralShape.from_json()
    table = SpectralTable.load(TABLE_PATH, shape)

    def factory(q, ron):
        if abs(ron-shape.ron_ohm) > 1e-8:
            raise ValueError("Frozen spectral shape and original training Ron differ")
        exact = SpectralLaw(shape, *10.**np.asarray(q[:5]), reset_initiation_log=0.)
        if exact.beta != 1.:
            raise ValueError("Group diagnostic requires frozen beta=1")
        return TabulatedSpectralLaw(exact, table)

    sources = [Path(__file__), TABLE_PATH,
               ROOT/"analysis/constrained_shared_2026-11-03/lognormal_spectral_refinement.json",
               ROOT/"vulcan2d/shared_spectral.py", ROOT/"vulcan2d/shared_spectral_ivp.py",
               ROOT/"vulcan2d/shared_gap.py", ROOT/"vulcan2d/shared_gap_ivp.py",
               ROOT/"vulcan2d/data/fig2b_transistor_lookup.json",
               ROOT/"vulcan2d/data/bookoutput2_reverse_transistor_lookup.json"]
    joint = SpectralJointProblem(factory, mode=mode, integrator_options=integrator_options,
                                  additional_source_paths=sources, include_reset_initiation=False)
    result = GroupSpectralProblem(joint, topology)
    result.frozen_spectral_shape = asdict(shape)
    return result
