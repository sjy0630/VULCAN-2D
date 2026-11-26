"""Training-only ordered objective adapter for the shared spectral model.

All observation masks, group weights, record ordering, scatter terms and
periodic/full32 scoring are inherited from the frozen OrderedJointProblem.
Only the simulator/circuits and explicit parameter coordinate convention change.
No held-out record is read by this module, including its export/CLI interface.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ORDERED_DIR = ROOT/"analysis/constrained_shared_2026-11-03"
for directory in (ROOT, ORDERED_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from ordered_objective import OrderedJointProblem, TRAIN
from vulcan2d.transistor import measured_transistor_mapping

PARAMETER_NAMES = ("log10_gamma_c_ev", "log10_J_SET_A", "log10_J_RESET_A",
                   "log10_k_SET_per_V", "log10_k_RESET_per_V",
                   "initial_u_1r", "initial_u_1t1r_cycle4")
PARAMETER_NAMES_WITH_RESET_INITIATION = PARAMETER_NAMES+("reset_initiation_log",)


def _source_file(callable_or_object):
    """Best-effort provenance for injected functions/classes, never code eval."""
    try:
        source = inspect.getsourcefile(callable_or_object)
    except (TypeError, OSError):
        source = None
    return Path(source).resolve() if source and Path(source).is_file() else None


class SpectralJointProblem(OrderedJointProblem):
    """Shared-law factory plus interchangeable signed simulation backend.

    law_factory(q, ron) returns one common material object. By default q has
    exactly seven values in PARAMETER_NAMES order; the first five are log10.
    Explicit include_reset_initiation=True appends q[7]=A_RESET, a nonnegative
    natural-log drive amplitude passed without a log10 transformation. Initial
    states always remain q[5]/q[6]. The factory must consume the opted-in term;
    the objective never invents or modifies a material constitutive law.
    simulate_fn(command, initial_u, law, circuit, **options) must return
    current_a, state_u, final_u; currents may be signed, as scoring uses |I|.

    Native bare records are compared to one explicitly prepared bare SET/RESET
    pair because their inter-record chronology is unknown. Series cycles 4-35
    propagate continuously. No per-series-cycle state resets are introduced.
    """

    def __init__(self, law_factory, *, simulate_fn=None, circuits=None,
                 mode="periodic", transistor_mapping=None,
                 integrator_options=None, periodic_current_rtol=1e-4,
                 periodic_state_atol=1e-7, additional_source_paths=(),
                 include_reset_initiation=False):
        if not callable(law_factory):
            raise TypeError("law_factory(q, ron) must be callable")
        if not isinstance(include_reset_initiation, bool):
            raise TypeError("include_reset_initiation must be an explicit boolean")
        self.include_reset_initiation = include_reset_initiation
        self.parameter_names = (PARAMETER_NAMES_WITH_RESET_INITIATION if include_reset_initiation
                                else PARAMETER_NAMES)
        self.n_parameters = len(self.parameter_names)
        if simulate_fn is None or circuits is None:
            from vulcan2d.shared_spectral_ivp import (
                BareSpectralCircuit, LookupSpectralCircuit, simulate_spectral_ivp)
            if simulate_fn is None:
                simulate_fn = simulate_spectral_ivp
            if circuits is None:
                mapping = measured_transistor_mapping() if transistor_mapping is None else transistor_mapping
                series = LookupSpectralCircuit(mapping, 1.1)
                circuits = {"1r": (BareSpectralCircuit(.001), BareSpectralCircuit()),
                            "1t1r": (series, series)}
        if not callable(simulate_fn):
            raise TypeError("simulate_fn must be callable")
        if set(circuits) != set(TRAIN) or any(len(circuits[top]) != 2 for top in TRAIN):
            raise ValueError("circuits must supply SET/RESET pairs for 1r and 1t1r")
        self.simulate_fn = simulate_fn
        self.additional_source_paths = tuple(Path(p).resolve() for p in additional_source_paths)
        if any(not p.is_file() for p in self.additional_source_paths):
            raise ValueError("Additional provenance sources must already exist")
        super().__init__(law_factory, mode=mode, circuits=circuits,
                         integrator_options=integrator_options,
                         initial_indices={"1r": 5, "1t1r": 6},
                         periodic_current_rtol=periodic_current_rtol,
                         periodic_state_atol=periodic_state_atol,
                         fallback_full=True)
        self.inherited_objective_source_sha256 = self.objective_source_sha256
        self.objective_source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def validate_q(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (self.n_parameters,) or not np.all(np.isfinite(q)):
            raise ValueError(f"Spectral q must be a finite {self.n_parameters}-element vector")
        if np.any(q[5:7] < 0):
            raise ValueError("Prepared initial states must be nonnegative")
        if self.include_reset_initiation and q[7] < 0:
            raise ValueError("RESET initiation log amplitude must be nonnegative")
        return q

    def _pair(self, law, initial_u, topology):
        traces, previous_voltage = [], 0.
        for phase, circuit in zip(("set", "reset"), self.circuits[topology]):
            command = self.waveforms[topology, phase]
            trace = self.simulate_fn(command, initial_u, law, circuit,
                                    initial_command_voltage_v=previous_voltage,
                                    **self.options)
            if not isinstance(trace, dict):
                raise TypeError("Spectral simulator must return a trace dictionary")
            if not {"current_a", "state_u", "final_u"} <= set(trace):
                raise ValueError("Simulator trace lacks current_a, state_u or final_u")
            current, state = np.asarray(trace["current_a"]), np.asarray(trace["state_u"])
            if current.shape != command.shape or state.shape != command.shape:
                raise ValueError("Spectral trace does not preserve the native command grid")
            if not np.all(np.isfinite(current)) or not np.all(np.isfinite(state)):
                raise ValueError("Spectral trace contains nonfinite current or state")
            final_u = float(trace["final_u"])
            if not np.isfinite(final_u) or final_u < 0. or np.any(state < 0.):
                raise ValueError("Spectral trace contains an invalid state")
            traces.append(trace)
            initial_u, previous_voltage = final_u, float(command[-1])
        return traces

    def evaluate(self, q, *, mode=None):
        result = super().evaluate(self.validate_q(q), mode=mode)
        result["details"].update(
            model_family="shared_log_normal_spectral_beta1_dynamic_candidate",
            parameter_names=list(self.parameter_names),
            n_parameters=self.n_parameters,
            include_reset_initiation=self.include_reset_initiation,
            reset_initiation_parameter_index=7 if self.include_reset_initiation else None,
            parameters_are_log10_except_initial_states=not self.include_reset_initiation,
            parameter_coordinate_types=(["log10"]*5+["linear_state"]*2
                                        +(["natural_log_drive_amplitude_untransformed"]
                                          if self.include_reset_initiation else [])),
            initial_parameter_indices=self.initial_indices,
            inherited_scoring_source_sha256=self.inherited_objective_source_sha256,
            evaluation_split="training_only",
            series_state_reset_between_records=False)
        return result

    def checkpoint_metadata(self):
        paths = {Path(__file__).resolve(), (ORDERED_DIR/"ordered_objective.py").resolve(),
                 (ROOT/"vulcan2d/transistor.py").resolve(), *self.additional_source_paths}
        for item in (self.law_factory, self.simulate_fn):
            source = _source_file(item)
            if source is not None:
                paths.add(source)
        for pair in self.circuits.values():
            for circuit in pair:
                source = _source_file(type(circuit))
                if source is not None:
                    paths.add(source)
        return {
            "objective": "four-phase equal weight, bareSET three equal groups, exact original-record masks",
            "objective_source_sha256": self.objective_source_sha256,
            "inherited_objective_source_sha256": self.inherited_objective_source_sha256,
            "training_arrays_sha256": self.training_arrays_sha256,
            "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)},
            "train_ids": self.train_ids, "n_residuals": self.n_residuals,
            "cost_includes_record_scatter": True,
            "observation_scatter_constant_cost": self.scatter_constant_cost,
            "requested_mode": self.mode, "integrator_options": self.options,
            "periodic_current_rtol": self.periodic_current_rtol,
            "periodic_state_atol": self.periodic_state_atol,
            "fallback_full": True, "initial_parameter_indices": self.initial_indices,
            "parameter_names": list(self.parameter_names),
            "n_parameters": self.n_parameters,
            "include_reset_initiation": self.include_reset_initiation,
            "reset_initiation_parameter_index": 7 if self.include_reset_initiation else None,
            "reset_initiation_coordinate": "nonnegative natural-log drive amplitude, passed without log10 transformation" if self.include_reset_initiation else "absent: original seven-parameter model",
            "first_five_coordinates": "base-10 logarithms",
            "ron_from_training_ohm": self.ron,
            "series_record_order": list(TRAIN["1t1r"]),
            "series_initial_state_used_once_at_record": 4,
            "bare_record_chronology_policy": "One prepared bare SET/RESET prediction scored against all raw bare records; inter-record continuity is not assumed.",
            "evaluation_split": "training_only_no_heldout_arrays_loaded",
            "timing_policy": "No sampling times or waits inferred or introduced by the objective adapter.",
        }

    def export_training_evaluation(self, q, output_json, *, mode="full",
                                   compare_modes=False, export_traces=True):
        """Evaluate frozen q on training records only; perform no optimization.

        Full32 is the export default even if the optimization mode is periodic.
        compare_modes additionally verifies the shortcut against full32 at the
        same q. This method intentionally has no test/holdout-selection option.
        """
        q = self.validate_q(q)
        result = self.evaluate(q, mode=mode)
        destination = Path(output_json)
        destination.parent.mkdir(parents=True, exist_ok=True)
        report = {"status": "frozen_parameter_training_evaluation_no_refit_no_holdout",
                  "q": q.tolist(), "parameter_names": list(self.parameter_names),
                  "n_parameters": self.n_parameters,
                  "include_reset_initiation": self.include_reset_initiation,
                  "checkpoint_metadata": self.checkpoint_metadata(),
                  "details": result["details"]}
        if compare_modes:
            report["full32_vs_periodic"] = self.compare_modes(q)
        if export_traces:
            arrays = {"residual": result["residual"], "q": q}
            for (top, cid, phase), trace in result["predictions"].items():
                prefix = f"{top}_cycle_{cid}_{phase}"
                arrays[prefix+"_command_V"] = self.waveforms[top, phase]
                for key in ("current_a", "state_u", "hbn_voltage_v", "transistor_voltage_v"):
                    if key in trace:
                        arrays[prefix+"_"+key] = np.asarray(trace[key])
                arrays[prefix+"_final_u"] = np.asarray(trace["final_u"])
            trace_path = destination.with_suffix(".npz")
            np.savez_compressed(trace_path, **arrays)
            report["training_trace_artifact"] = str(trace_path.resolve())
            report["training_trace_sha256"] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
        destination.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory", required=True, help="Importable module:law_factory; factory(q,ron) supplies the frozen common spectral law")
    parser.add_argument("--q-json", required=True, type=Path, help="JSON parameter list or object with key q; seven values by default, eight with explicit RESET initiation")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("full", "periodic"), default="full")
    parser.add_argument("--compare-modes", action="store_true")
    parser.add_argument("--no-traces", action="store_true")
    parser.add_argument("--include-reset-initiation", action="store_true",
                        help="Explicitly append q[7]=nonnegative RESET initiation natural-log drive amplitude")
    args = parser.parse_args()
    module_name, separator, factory_name = args.factory.partition(":")
    if not separator or not module_name or not factory_name:
        parser.error("--factory must be module:callable")
    factory = getattr(importlib.import_module(module_name), factory_name)
    payload = json.loads(args.q_json.read_text())
    q = payload["q"] if isinstance(payload, dict) else payload
    problem = SpectralJointProblem(factory, mode=args.mode,
                                   additional_source_paths=[args.q_json],
                                   include_reset_initiation=args.include_reset_initiation)
    report = problem.export_training_evaluation(q, args.output, mode=args.mode,
                                               compare_modes=args.compare_modes,
                                               export_traces=not args.no_traces)
    print(json.dumps({"output": str(args.output.resolve()), "evaluation_split": "training_only",
                      "used_mode": report["details"]["used_mode"],
                      "cost": report["details"]["verified_cost"]}, indent=2))


if __name__ == "__main__":
    main()
