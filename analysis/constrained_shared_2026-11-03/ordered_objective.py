"""Exact record-weighted shared-model objective, with checked periodic shortcut.

The full objective pairs every prediction with its own training record. The
shortcut scores cycle4 once and the second continuous pair against records5-35
individually; it never compares two equally weighted pairs to the same mean.
Its third-pair check is local evidence of a periodic orbit, not a global error
bound. Final candidates must also run compare_modes() against all32 cycles.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vulcan2d.shared_gap import BareGapCircuit, LookupGapCircuit
from vulcan2d.shared_gap_ivp import simulate_gap_ivp
from vulcan2d.transistor import measured_transistor_mapping

TRAIN = {"1r": [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
         "1t1r": list(range(4, 36))}
DATA = ROOT / "analysis/shared_hbn_2026-11-03/data/native_observations.npz"


class OrderedJointProblem:
    """law_factory(q, ron) -> one common material object for both circuits.

    q[7]/q[8] are initial bare/cycle4 states unless initial_indices is supplied.
    residual() always returns the same raw-record vector length in both modes,
    including when a failed periodic check falls back to full propagation.
    Costs include record scatter: cost=0.5*sum(four phase MSEs), with bare SET
    nonlimited outward/return and limited groups each weighted1/3.
    """

    def __init__(self, law_factory, *, mode="periodic", transistor_mapping=None,
                 circuits=None, integrator_options=None, initial_indices=None,
                 periodic_current_rtol=1e-4, periodic_state_atol=1e-7,
                 fallback_full=True):
        if mode not in ("full", "periodic"):
            raise ValueError("mode must be full or periodic")
        if periodic_current_rtol <= 0. or periodic_state_atol <= 0.:
            raise ValueError("Periodic tolerances must be positive")
        self.law_factory, self.mode = law_factory, mode
        self.initial_indices = {"1r": 7, "1t1r": 8} if initial_indices is None else dict(initial_indices)
        self.periodic_current_rtol, self.periodic_state_atol = periodic_current_rtol, periodic_state_atol
        self.fallback_full = bool(fallback_full)
        self.options = dict(method="Radau", state_rtol=1e-6, state_atol=1e-8, current_rtol=5e-4)
        if integrator_options:
            self.options.update(integrator_options)
        self.train_ids = {top: ids.copy() for top, ids in TRAIN.items()}
        self.split = {top: {"train": ids.copy()} for top, ids in TRAIN.items()}
        self.waveforms, self.raw_training, self.log_observed, self.groups = {}, {}, {}, {}
        digest = hashlib.sha256()
        self.n_residuals = 0
        self.scatter_constant_cost = 0.
        with np.load(DATA) as raw:
            for top, ids in TRAIN.items():
                for phase in ("set", "reset"):
                    key = top, phase
                    vv = np.array([raw[f"{top}_cycle_{cid}_{phase}_voltage"] for cid in ids])
                    observed = abs(np.array([raw[f"{top}_cycle_{cid}_{phase}_current"] for cid in ids]))
                    if not np.all(vv == vv[0]):
                        raise ValueError("This periodic objective requires identical native training waveforms")
                    v = vv[0]
                    self.waveforms[key], self.raw_training[key] = v, observed
                    digest.update(str(key).encode()); digest.update(vv.tobytes()); digest.update(observed.tobytes())
                    mask = np.isfinite(observed) & (abs(v)[None, :] > 1e-12)
                    mask &= observed >= (1e-11 if phase == "set" else 1e-12)
                    logs = np.zeros_like(observed)
                    logs[mask] = np.log10(observed[mask])
                    self.log_observed[key] = logs
                    if key == ("1r", "set"):
                        outward = np.arange(len(v)) <= np.argmax(abs(v))
                        named_masks = [("nonlimited_outward", mask & (observed < .995e-3) & outward),
                                       ("nonlimited_return", mask & (observed < .995e-3) & ~outward),
                                       ("limited", mask & (observed >= .995e-3))]
                    else:
                        named_masks = [("all_eligible", mask)]
                    groups = []
                    for name, selected in named_masks:
                        n = int(selected.sum())
                        if n == 0:
                            raise ValueError("Empty objective group")
                        count = selected.sum(axis=0)
                        mean = np.where(selected, logs, 0.).sum(axis=0)/np.maximum(count, 1)
                        scatter = float(np.sum(np.where(selected, (logs-mean)**2, 0.))/n)
                        group = {"name": name, "mask": selected, "n": n,
                            "normalization": np.sqrt(n*len(named_masks)),
                            "observation_scatter_mse": scatter,
                            "first_record_point_weight": float(selected[0].sum()/n),
                            "remaining_records_point_weight": float(selected[1:].sum()/n)}
                        groups.append(group)
                        self.n_residuals += n
                        self.scatter_constant_cost += scatter/(2.*len(named_masks))
                    self.groups[key] = groups
        self.training_arrays_sha256 = digest.hexdigest()
        v, current = self.waveforms["1r", "reset"], self.raw_training["1r", "reset"]
        keep = (np.arange(len(v)) <= np.argmax(abs(v))) & (abs(v) >= .049) & (abs(v) <= .201)
        slopes = np.sum(abs(v[keep])*current[:, keep], axis=1)/np.sum(v[keep]**2)
        self.ron = float(np.median(1./slopes))
        if circuits is None:
            mapping = measured_transistor_mapping() if transistor_mapping is None else transistor_mapping
            transistor = LookupGapCircuit(mapping, 1.1)
            self.circuits = {"1r": (BareGapCircuit(.001), BareGapCircuit()), "1t1r": (transistor, transistor)}
        else:
            self.circuits = circuits
        self.objective_source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def checkpoint_metadata(self):
        sources = [Path(__file__), ROOT/"vulcan2d/shared_gap_ivp.py", ROOT/"vulcan2d/shared_gap.py",
                   ROOT/"vulcan2d/transistor.py"]
        factory_file = inspect.getsourcefile(self.law_factory)
        if factory_file and Path(factory_file).is_file():
            sources.append(Path(factory_file))
        return {"objective": "four-phase equal weight, bareSET three equal groups, exact original-record masks",
            "objective_source_sha256": self.objective_source_sha256,
            "training_arrays_sha256": self.training_arrays_sha256,
            "source_sha256": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            "train_ids": self.train_ids, "n_residuals": self.n_residuals,
            "cost_includes_record_scatter": True, "observation_scatter_constant_cost": self.scatter_constant_cost,
            "requested_mode": self.mode, "integrator_options": self.options,
            "periodic_current_rtol": self.periodic_current_rtol,
            "periodic_state_atol": self.periodic_state_atol, "fallback_full": self.fallback_full,
            "initial_parameter_indices": self.initial_indices}

    def _pair(self, law, initial_u, topology):
        traces = []
        previous_voltage = 0.
        for phase, circuit in zip(("set", "reset"), self.circuits[topology]):
            command = self.waveforms[topology, phase]
            trace = simulate_gap_ivp(command, initial_u, law, circuit,
                initial_command_voltage_v=previous_voltage, **self.options)
            traces.append(trace)
            initial_u, previous_voltage = trace["final_u"], float(command[-1])
        return traces

    @staticmethod
    def _current_log_difference(first, second):
        values = []
        for a, b in zip(first, second):
            x, y = abs(a["current_a"]), abs(b["current_a"])
            nonzero = (x != 0.) | (y != 0.)
            if np.any((x[nonzero] <= 0.) | (y[nonzero] <= 0.)):
                return float("inf")
            if np.any(nonzero):
                values.append(float(np.max(abs(np.log(x[nonzero])-np.log(y[nonzero])))))
        return max(values, default=0.)

    def _periodicity(self, pairs):
        first, second, third = pairs[:3]
        endpoint12 = abs(first[-1]["final_u"]-second[-1]["final_u"])
        endpoint23 = abs(second[-1]["final_u"]-third[-1]["final_u"])
        log_difference = self._current_log_difference(second, third)
        state_difference = max(float(np.max(abs(a["state_u"]-b["state_u"]))) for a, b in zip(second, third))
        passed = (max(endpoint12, endpoint23, state_difference) <= self.periodic_state_atol
                  and log_difference <= np.log1p(self.periodic_current_rtol))
        return {"third_pair_checked": True, "passed": bool(passed),
            "endpoint_difference_first_second_u": float(endpoint12),
            "endpoint_difference_second_third_u": float(endpoint23),
            "second_third_max_state_difference_u": state_difference,
            "second_third_max_current_log_difference_natural": log_difference,
            "exact_numerical_fixed_point": bool(endpoint12 == 0. and endpoint23 == 0.
                and state_difference == 0. and log_difference == 0.),
            "scope": "Third-pair local periodicity check only; full32 comparison required for final candidate"}

    def score_predictions(self, predictions):
        """Score explicit (topology,record,phase)->trace/array on original masks."""
        residuals, details = [], {}
        for top, ids in TRAIN.items():
            for phase in ("set", "reset"):
                key = top, phase
                arrays = [predictions[top, cid, phase] for cid in ids]
                predicted = abs(np.array([a["current_a"] if isinstance(a, dict) else a for a in arrays]))
                if predicted.shape != self.raw_training[key].shape:
                    raise ValueError("Prediction shape differs from original training records")
                phase_sse, group_rows = 0., []
                for group in self.groups[key]:
                    mask = group["mask"]
                    if np.any(~np.isfinite(predicted[mask])) or np.any(predicted[mask] <= 0.):
                        raise ValueError("Invalid model current at an eligible original observation")
                    errors = np.log10(predicted[mask])-self.log_observed[key][mask]
                    block = errors/group["normalization"]
                    residuals.append(block)
                    weighted = float(np.dot(block, block)); phase_sse += weighted
                    group_rows.append({"name": group["name"], "n": group["n"],
                        "raw_rmse_decade": float(np.sqrt(np.mean(errors**2))),
                        "weighted_group_mse": weighted,
                        "first_record_point_weight": group["first_record_point_weight"],
                        "remaining_records_point_weight": group["remaining_records_point_weight"]})
                details[f"{top}_{phase}"] = {"weighted_total_mse": phase_sse, "groups": group_rows}
        vector = np.concatenate(residuals)
        if len(vector) != self.n_residuals or not np.all(np.isfinite(vector)):
            raise ValueError("Invalid fixed-dimensional ordered residual")
        return vector, details

    def evaluate(self, q, *, mode=None):
        mode = self.mode if mode is None else mode
        if mode not in ("full", "periodic"):
            raise ValueError("mode must be full or periodic")
        law = self.law_factory(q, self.ron)  # same object for both topologies
        bare = self._pair(law, float(q[self.initial_indices["1r"]]), "1r")
        predictions = {(top, cid, phase): trace for top in ("1r",) for cid in TRAIN[top]
                       for phase, trace in zip(("set", "reset"), bare)}
        state = float(q[self.initial_indices["1t1r"]]); pairs = []
        for _ in range(3):
            pair = self._pair(law, state, "1t1r"); pairs.append(pair); state = pair[-1]["final_u"]
        periodic = self._periodicity(pairs)
        use_periodic = mode == "periodic" and periodic["passed"]
        fallback = mode == "periodic" and not periodic["passed"]
        if fallback and not self.fallback_full:
            raise RuntimeError("Third-pair periodicity check failed; use full32 mode")
        if not use_periodic:
            for _ in range(3, len(TRAIN["1t1r"])):
                pair = self._pair(law, state, "1t1r"); pairs.append(pair); state = pair[-1]["final_u"]
        for index, cid in enumerate(TRAIN["1t1r"]):
            pair = pairs[min(index, 1)] if use_periodic else pairs[index]
            for phase, trace in zip(("set", "reset"), pair):
                predictions["1t1r", cid, phase] = trace
        residual, blocks = self.score_predictions(predictions)
        cost = float(np.dot(residual, residual)/2.)
        metadata = {"requested_mode": mode, "used_mode": "checked_periodic_shortcut" if use_periodic else "full32",
            "objective_is_exact_original_record_score": not use_periodic or periodic["exact_numerical_fixed_point"],
            "objective_scope": "original-record score with locally checked periodic prediction approximation" if use_periodic
                and not periodic["exact_numerical_fixed_point"] else "full original-record ordered training score",
            "periodicity": periodic, "fell_back_to_full32": fallback,
            "n_integrated_1t_pairs": len(pairs), "first_record_id": 4, "remaining_record_ids": list(range(5, 36)),
            "verified_cost": cost, "cost_includes_record_scatter": True,
            "model_dependent_cost_after_subtracting_fixed_scatter": cost-self.scatter_constant_cost,
            "blocks": blocks, "objective_source_sha256": self.objective_source_sha256}
        return {"residual": residual, "details": metadata, "predictions": predictions}

    def residual(self, q, return_details=False, *, mode=None):
        result = self.evaluate(q, mode=mode)
        return (result["residual"], result["details"]) if return_details else result["residual"]

    def compare_modes(self, q):
        """Final frozen-candidate audit: periodic approximation vs actual32 pairs."""
        full, shortcut = self.evaluate(q, mode="full"), self.evaluate(q, mode="periodic")
        differences = {}
        maximum = 0.
        for top, ids in TRAIN.items():
            for phase in ("set", "reset"):
                worst = 0.
                for cid in ids:
                    a, b = full["predictions"][top, cid, phase], shortcut["predictions"][top, cid, phase]
                    value = self._current_log_difference([a], [b]); worst = max(worst, value)
                maximum = max(maximum, worst)
                differences[f"{top}_{phase}"] = {"max_abs_log_current_difference_natural": worst,
                    "weighted_total_mse_difference_shortcut_minus_full": shortcut["details"]["blocks"][f"{top}_{phase}"]["weighted_total_mse"]
                        -full["details"]["blocks"][f"{top}_{phase}"]["weighted_total_mse"]}
        return {"status": "frozen_parameter_full32_vs_periodic_objective_audit_no_refit",
            "checkpoint_metadata": self.checkpoint_metadata(), "q": np.asarray(q).tolist(),
            "full": full["details"], "shortcut": shortcut["details"], "phase_differences": differences,
            "cost_difference_shortcut_minus_full": shortcut["details"]["verified_cost"]-full["details"]["verified_cost"],
            "max_abs_log_current_difference_natural": maximum,
            "full32_comparison_passed_current_tolerance": bool(maximum <= np.log1p(self.periodic_current_rtol))}
