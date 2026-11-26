"""Finite saved-node common-lambda diagnostic. Prepared only; requires --run + freeze.

No physical model imports, inversions, circuit roots, ODEs, or optimizers.
The explicitly counted bisections ARE parameter-scalar root solving.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "empirical_integral_2026-11-04"
INPUT_NAMES = ("empirical_integral_diagnostic.json", "empirical_integral_arrays.npz",
               "LAMBDA_NECESSARY_BOUND.json")
ITERATIONS = 48
GRID_POINTS = 31
MAX_H_BRACKETS = 4
H_TOL = 1e-8
B_TOL = 5e-9
ROOT_PADDING = 1e-10
POS_IDS = ("set4_outward_0p2_to_0p8", "set4_outward_0p8_to_1p6")
NEG_IDS = ("reset4_outward_0p2_to_0p8", "reset4_outward_0p8_to_1p6")
NUMERICAL_ERRORS = (ValueError, ArithmeticError, np.linalg.LinAlgError)


class BudgetExceeded(RuntimeError):
    def __init__(self, scope):
        self.scope = scope
        super().__init__(f"{scope} numerical budget exhausted")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def valid_h(context):
    return context["all_positive"] and context["H"] is not None and math.isfinite(context["H"])


def freeze_check():
    manifest_path = HERE / "SOURCE_FREEZE.json"
    manifest = json.loads(manifest_path.read_text())
    frozen = manifest["source_sha256"]
    required = [Path(__file__).resolve(), HERE / "EXPERIMENT_PLAN.md"]
    required += [OLD / name for name in INPUT_NAMES]
    for path in required:
        if frozen.get(str(path)) != digest(path):
            raise RuntimeError(f"Missing/stale freeze entry: {path}")
    return {"manifest_sha256": digest(manifest_path),
            "source_sha256": {str(p): digest(p) for p in required}}


def identify(rows, names, ledger, result):
    ledger["linear_identification_attempts"] += 1
    x = np.array([[r["A"], -r["B"]] for r in rows], dtype=float)
    y = np.array([r["y"] for r in rows], dtype=float)
    result.update(window_ids=[r["window_id"] for r in rows], raw_X_A_minus_B=x,
                  y=y, available=False, all_positive=False, status="attempted")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        result["status"] = "nonfinite_input"
        return result
    norms = np.linalg.norm(x, axis=0)
    if np.any(norms <= 0):
        result["status"] = "zero_column"
        return result
    scaled = x / norms[None, :]
    singular = np.linalg.svd(scaled, compute_uv=False)
    tolerance = 2 * np.finfo(float).eps * singular[0]
    rank = int(np.count_nonzero(singular > tolerance))
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else math.inf
    result.update(column_normalized_condition_2=condition, numerical_rank=rank,
                  severely_ill_conditioned=condition > 1e8)
    if rank != 2:
        result["status"] = "rank_deficient"
        return result
    ledger["two_by_two_linear_solve_attempts"] += 1
    try:
        coefficients = np.linalg.solve(scaled, y) / norms
    except np.linalg.LinAlgError as exc:
        result.update(status="linear_solve_failure", error=str(exc))
        return result
    ledger["two_by_two_linear_solves_returned"] += 1
    finite = bool(np.all(np.isfinite(coefficients)))
    result.update(status="identified" if finite else "nonfinite_coefficients",
                  available=finite, coefficients=dict(zip(names, coefficients)),
                  all_positive=finite and bool(np.all(coefficients > 0)),
                  identification_residual=x @ coefficients - y)
    return result


def closure(row, coefficients):
    first, second = coefficients[0] * row["A"], coefficients[1] * row["B"]
    predicted = first - second
    error = row["y"] - predicted
    denominator = abs(row["y"]) + abs(first) + abs(second)
    return {"window_id": row["window_id"], "A": row["A"], "B": row["B"],
            "delta_u": row["y"], "recovery_term": first, "capture_term": second,
            "predicted_delta_u": predicted, "E": error,
            "normalization_denominator": denominator,
            "normalized_E": error / denominator if denominator else None,
            "used_for_identification": row["window_id"] in POS_IDS + NEG_IDS}


class Configuration:
    def __init__(self, diagnostic, bound, arrays, ledger, elapsed_before):
        self.interpolant, self.gauss = bound["interpolant"], bound["gauss_order"]
        self.bound, self.ledger = bound, ledger
        self.lambda0 = diagnostic["fixed_material"]["lambda_per_v"]
        self.windows = {}
        self.a_cache, self.h_cache = {}, {}
        self.trace = []
        self.report = {}
        self.started = None
        self.root_started = None
        self.elapsed_before = elapsed_before
        for row in diagnostic["window_variants"]:
            if (row["interpolant"], row["gauss_order"]) != (self.interpolant, self.gauss):
                continue
            prefix = row["array_prefix"]
            voltage = np.abs(arrays[prefix + "local_voltage_v"])
            mass0 = arrays[prefix + "weights_path_v"] * arrays[prefix + "F"]
            if not np.all(voltage > 0) or not np.all(mass0 > 0):
                raise ValueError("Nonpositive saved quadrature support/mass")
            if not np.all(arrays[prefix + "completed_mask"]):
                raise ValueError("Incomplete saved node input")
            self.windows[row["window_id"]] = {
                "v": voltage, "log_mass0": np.log(mass0),
                "B": float(row["B_a_v2"]), "y": float(row["delta_u"]),
                "A0": float(row["A_v2"]),
            }
        if len(self.windows) != 12:
            raise ValueError("Expected exactly 12 saved training windows")

    def check(self):
        now = time.monotonic()
        if self.started is not None:
            elapsed = now - self.started
            if self.elapsed_before + elapsed >= 60:
                raise BudgetExceeded("total")
            if elapsed >= 15:
                raise BudgetExceeded("configuration")
        if self.root_started is not None and now - self.root_started >= 2:
            raise BudgetExceeded("root")

    def integral(self, window_id, lam):
        self.check()
        key = (window_id, float(lam))
        if key in self.a_cache:
            self.ledger["saved_sum_cache_hits"] += 1
            return self.a_cache[key]
        self.ledger["new_saved_window_sum_evaluations"] += 1
        w = self.windows[window_id]
        self.ledger["saved_node_terms_reweighted"] += int(w["v"].size)
        z = w["log_mass0"] + (lam - self.lambda0) * w["v"]
        shift = float(np.max(z))
        log_a = shift + math.log(float(np.sum(np.exp(z - shift))))
        a = math.exp(log_a)
        row = {"window_id": window_id, "lambda_per_v": float(lam),
               "log_A": log_a, "A": a, "B": w["B"], "y": w["y"]}
        self.a_cache[key] = row
        return row

    def log_ratio_minus(self, lam, threshold):
        self.ledger["reset_boundary_scalar_function_evaluations"] += 1
        lo, hi = [self.integral(w, lam) for w in NEG_IDS]
        return hi["log_A"] - lo["log_A"] - math.log(threshold)

    def connected_positive_proof(self):
        low, high = [self.windows[w] for w in POS_IDS]
        ratio0 = high["A0"] / low["A0"]
        s = high["B"] / low["B"]
        t = high["y"] / low["y"]
        separated = float(np.min(high["v"])) > float(np.max(low["v"]))
        passed = low["y"] < 0 and high["y"] < 0 and t < s < ratio0 and separated
        return {"SET_y_low": low["y"], "SET_y_high": high["y"],
                "SET_R0": ratio0, "SET_S": s, "SET_T": t,
                "SET_support_strictly_separated": separated,
                "SET_positive_for_all_lambda_ge_lambda0": bool(passed),
                "RESET_positive_only_in_open_S_T_interval": True,
                "scope": "Exact finite-sum algebra; numerical singularities still break a search bracket."}

    def evaluate_h(self, lam, purpose):
        self.check()
        lam = float(lam)
        if lam in self.h_cache:
            self.ledger["H_context_cache_hits"] += 1
            return self.h_cache[lam]
        self.ledger["H_parameter_point_evaluations"] += 1
        context = {"lambda_per_v": lam, "purpose": purpose, "status": "attempted",
                   "positive": {"status": "unattempted", "available": False, "all_positive": False},
                   "negative": {"status": "unattempted", "available": False, "all_positive": False},
                   "all_positive": False, "H": None, "b_F": None, "b_G": None}
        # Publish before either linear identification: partial evidence survives exceptions.
        self.h_cache[lam] = context
        self.trace.append(context)
        for side, ids, names in (("positive", POS_IDS, ("kappa", "r")),
                                 ("negative", NEG_IDS, ("alpha", "beta"))):
            branch = context[side]
            branch.update(status="attempted", window_integrals=[])
            try:
                for window_id in ids:
                    branch["window_integrals"].append(self.integral(window_id, lam))
                identify(branch["window_integrals"], names, self.ledger, branch)
            except BudgetExceeded as exc:
                branch.update(status=f"{exc.scope}_budget_exhausted_unresolved", error=str(exc))
                context["status"] = "budget_interrupted_partial_identification"
                raise
            except NUMERICAL_ERRORS as exc:
                branch.update(status="numerical_failure_unresolved", available=False, all_positive=False,
                              error=f"{type(exc).__name__}: {exc}")
                self.ledger["linear_identification_numerical_failures"] += 1
        pos, neg = context["positive"], context["negative"]
        context["all_positive"] = pos["all_positive"] and neg["all_positive"]
        context["status"] = "completed" if pos["available"] and neg["available"] else "partial_or_unresolved_identification"
        if context["all_positive"]:
            try:
                k, r = [float(pos["coefficients"][s]) for s in ("kappa", "r")]
                alpha, beta = [float(neg["coefficients"][s]) for s in ("alpha", "beta")]
                lk, lr, la, lb = map(math.log, (k, r, alpha, beta))
                context.update(H=la + lb - lk - lr, b_F=(la - lk) / 2, b_G=(lr - lb) / 2)
            except NUMERICAL_ERRORS as exc:
                context.update(status="H_arithmetic_failure_unresolved", error=f"{type(exc).__name__}: {exc}")
        self.check()
        return context

    def boundary_root(self, kind):
        b = self.bound["pairs"]["reset_outward"]
        if kind == "S":
            threshold = b["S_B_high_over_low"]
            lo = b["strict_necessary_lambda_lower_bound_jensen_per_v"]
            hi = b["lambda_above_this_suffices_for_R_greater_than_S_per_v"]
        else:
            threshold = b["T_y_high_over_low"]
            lo = b["T_crossing_lambda_lower_bound_jensen_per_v"]
            hi = b["strict_necessary_lambda_upper_bound_jensen_per_v"]
        lo, hi = lo - ROOT_PADDING, hi + ROOT_PADDING
        history = []
        result = {"kind": kind, "threshold": threshold, "initial_bracket": [lo, hi],
                  "endpoint_padding_per_v": ROOT_PADDING,
                  "evaluations": history, "status": "attempted", "completed": False}
        self.report["boundary_roots"][kind] = result
        self.ledger["parameter_scalar_root_problems_attempted"] += 1
        self.root_started = time.monotonic()
        try:
            flo = self.log_ratio_minus(lo, threshold)
            history.append({"lambda": lo, "f": flo})
            fhi = self.log_ratio_minus(hi, threshold)
            history.append({"lambda": hi, "f": fhi})
            if not (math.isfinite(flo) and math.isfinite(fhi) and flo <= 0 <= fhi):
                result["status"] = "invalid_initial_sign_bracket_no_extension"
                return result
            reason = "iteration_limit_48"
            if flo == 0 or fhi == 0:
                lo = hi = lo if flo == 0 else hi
                flo = fhi = 0.0
                reason = "floating_exact_zero_endpoint"
            else:
                for _ in range(ITERATIONS):
                    mid = (lo + hi) / 2
                    if mid == lo or mid == hi:
                        reason = "floating_midpoint_stagnation"
                        break
                    fm = self.log_ratio_minus(mid, threshold)
                    self.ledger["reset_boundary_bisection_steps"] += 1
                    history.append({"lambda": mid, "f": fm})
                    if not math.isfinite(fm):
                        result["status"] = "nonfinite_midpoint_unresolved"
                        return result
                    if fm == 0:
                        lo = hi = mid
                        flo = fhi = fm
                        reason = "floating_exact_zero_midpoint"
                        break
                    if fm < 0:
                        lo, flo = mid, fm
                    else:
                        hi, fhi = mid, fm
            self.check()
            result.update(status="completed_parameter_root", completed=True, stopping_reason=reason,
                          bracket=[lo, hi], endpoint_values=[flo, fhi], bracket_width=hi-lo)
            self.ledger["parameter_scalar_roots_completed"] += 1
        except BudgetExceeded as exc:
            result.update(status=f"{exc.scope}_budget_exhausted_unresolved", error=str(exc))
            if exc.scope != "root":
                raise
        except NUMERICAL_ERRORS as exc:
            result.update(status="numerical_failure_unresolved", error=f"{type(exc).__name__}: {exc}")
        finally:
            result["numerical_wall_s"] = time.monotonic() - self.root_started
            self.root_started = None
        return result

    def h_root(self, left, right, bracket_index):
        self.ledger["parameter_scalar_root_problems_attempted"] += 1
        history = []
        lo, hi = left, right
        result = {"bracket_index": bracket_index, "initial_bracket":
                  [left["lambda_per_v"], right["lambda_per_v"]],
                  "status": "attempted", "completed": False, "evaluations": history}
        self.report["H_root_attempts"].append(result)
        self.root_started = time.monotonic()
        try:
            self.check()
            valid = all(c["all_positive"] and c["H"] is not None and math.isfinite(c["H"]) for c in (lo, hi))
            if not valid or lo["H"] * hi["H"] >= 0:
                result["status"] = "invalid_cached_endpoint_bracket_no_search"
                return result
            result["verified_cached_endpoint_H"] = [lo["H"], hi["H"]]
            reason = "iteration_limit_48"
            for _ in range(ITERATIONS):
                midpoint = (lo["lambda_per_v"] + hi["lambda_per_v"]) / 2
                if midpoint == lo["lambda_per_v"] or midpoint == hi["lambda_per_v"]:
                    reason = "floating_midpoint_stagnation"
                    break
                current = self.evaluate_h(midpoint, "H_bisection")
                self.ledger["H_bisection_steps"] += 1
                history.append({"lambda": midpoint, "H": current["H"], "all_positive": current["all_positive"]})
                if not current["all_positive"] or current["H"] is None or not math.isfinite(current["H"]):
                    result["status"] = "nonpositive_or_nonfinite_hole_unresolved_no_crossing"
                    return result
                if current["H"] == 0:
                    lo = hi = current
                    reason = "floating_exact_zero_midpoint"
                    break
                if math.copysign(1, current["H"]) == math.copysign(1, lo["H"]):
                    lo = current
                else:
                    hi = current
            self.check()
            selected = min((lo, hi), key=lambda c: abs(c["H"]))
            result.update(status="completed_parameter_root", completed=True, stopping_reason=reason,
                          bracket=[lo["lambda_per_v"], hi["lambda_per_v"]],
                          final_endpoint_H=[lo["H"], hi["H"]],
                          bracket_width=hi["lambda_per_v"] - lo["lambda_per_v"],
                          candidate_lambda=selected["lambda_per_v"], candidate_H=selected["H"])
            self.ledger["parameter_scalar_roots_completed"] += 1
        except BudgetExceeded as exc:
            result.update(status=f"{exc.scope}_budget_exhausted_unresolved", error=str(exc))
            if exc.scope != "root":
                raise
        except NUMERICAL_ERRORS as exc:
            result.update(status="numerical_failure_unresolved", error=f"{type(exc).__name__}: {exc}")
        finally:
            result["last_retained_bracket"] = [lo["lambda_per_v"], hi["lambda_per_v"]]
            result["last_retained_endpoint_H"] = [lo["H"], hi["H"]]
            result["numerical_wall_s"] = time.monotonic() - self.root_started
            self.root_started = None
        return result

    def candidate(self, context, origin):
        self.check()
        self.ledger["candidate_checks"] += 1
        result = {"origins": origin, "lambda_per_v": context["lambda_per_v"], "H": context["H"],
                  "identified_context": context, "status": "attempted",
                  "eligible_for_continuous_forward_check": False,
                  "separate_polarity_coefficients_closure": [], "forced_common_b_closure": [],
                  "window_plan": [{"window_id": w, "status": "unattempted"} for w in sorted(self.windows)],
                  "not_independent_holdout": True, "not_a_full_model_pass": True}
        self.report["candidates"].append(result)
        try:
            self._candidate_compute(context, result)
        except BudgetExceeded as exc:
            result.update(status=f"{exc.scope}_budget_exhausted_partial", error=str(exc))
            raise
        except NUMERICAL_ERRORS as exc:
            result.update(status="numerical_failure_unresolved", error=f"{type(exc).__name__}: {exc}")
        return result

    def _candidate_compute(self, context, result):
        k, r = [float(context["positive"]["coefficients"][s]) for s in ("kappa", "r")]
        alpha, beta = [float(context["negative"]["coefficients"][s]) for s in ("alpha", "beta")]
        b_f, b_g = context["b_F"], context["b_G"]
        b = (b_f + b_g) / 2
        result.update(b_F=b_f, b_G=b_g, b_common_midpoint=b,
                      direct_common_tuple={"lambda_per_v": context["lambda_per_v"], "kappa": k, "J_a": k/r, "b": b},
                      positive_coefficients={"kappa": k, "r": r, "alpha": alpha, "beta": beta})
        common_neg = (k * math.exp(2 * b), r * math.exp(-2 * b))
        projection_relative = {"alpha": (common_neg[0] - alpha) / alpha,
                               "beta": (common_neg[1] - beta) / beta}
        numeric_shared = abs(context["H"]) <= H_TOL and abs(b_f-b_g) <= B_TOL
        b_valid = 0 <= b_f <= math.log(10) and 0 <= b_g <= math.log(10)
        separate, common = result["separate_polarity_coefficients_closure"], result["forced_common_b_closure"]
        result.update({
                  "both_b_estimates_in_0_to_ln10": b_valid,
                  "b_common_in_0_to_ln10": 0 <= b <= math.log(10),
                  "numeric_shared_relation_satisfied": numeric_shared,
                  "eligible_for_continuous_forward_check": False,
                  "common_b_negative_coefficients": {"alpha": common_neg[0], "beta": common_neg[1]},
                  "projection_relative_difference": projection_relative,
                  "max_abs_projection_relative_difference": max(map(abs, projection_relative.values())),
                  "status": "attempted"})
        for window in result["window_plan"]:
            window_id = window["window_id"]
            window["status"] = "attempted"
            try:
                row = self.integral(window_id, context["lambda_per_v"])
                negative = window_id.startswith("reset")
                separate.append(closure(row, (alpha, beta) if negative else (k, r)))
                common.append(closure(row, common_neg if negative else (k, r)))
                self.ledger["candidate_window_closure_pairs"] += 1
                window["status"] = "completed"
            except BudgetExceeded as exc:
                window.update(status=f"{exc.scope}_budget_exhausted_partial", error=str(exc))
                raise
            except NUMERICAL_ERRORS as exc:
                window.update(status="numerical_failure_unresolved", error=f"{type(exc).__name__}: {exc}")
        self.check()
        if not all(w["status"] == "completed" for w in result["window_plan"]):
            result["status"] = "partial_window_closures_unresolved"
            return
        result.update(status="completed", eligible_for_continuous_forward_check=numeric_shared and b_valid)
        result["closure_max_abs_normalized_E"] = {
            name: {"identification_windows": max(abs(c["normalized_E"]) for c in checks if c["used_for_identification"]),
                   "remaining_eight_training_windows": max(abs(c["normalized_E"]) for c in checks if not c["used_for_identification"])}
            for name, checks in (("separate", separate), ("common_b", common))}
        return result

    def run(self):
        self.started = time.monotonic()
        counts_before = dict(self.ledger)
        try:
            self._run()
        except BudgetExceeded as exc:
            self.report.update(status=f"{exc.scope}_budget_exhausted_partial", error=str(exc),
                               unattempted_reason="Numerical budget exhausted; no automatic retry")
        except NUMERICAL_ERRORS as exc:
            self.report.update(status="numerical_failure_partial_configuration", error=f"{type(exc).__name__}: {exc}",
                               unattempted_reason="Configuration numerical failure; independent configurations may continue")
        finally:
            self.report["numerical_wall_s"] = time.monotonic() - self.started
            self.report["work_counts"] = {k: v-counts_before.get(k, 0) for k, v in self.ledger.items()}
        return self.report

    def _run(self):
        proof = self.connected_positive_proof()
        result = self.report
        result.update({"interpolant": self.interpolant, "gauss_order": self.gauss,
                  "connected_positive_coefficient_argument": proof, "H_evaluation_trace": self.trace,
                  "boundary_roots": {k: {"status": "unattempted", "completed": False} for k in ("S", "T")},
                  "grid": [], "H_root_attempts": [], "candidates": [],
                  "grid_plan": {"planned": GRID_POINTS, "attempted": 0, "completed": 0},
                  "candidate_plan": []})
        self.check()
        b = self.bound["pairs"]["reset_outward"]
        result["RESET_ordering_premise"] = {
            "y_low": b["low_window"]["y_delta_u"], "y_high": b["high_window"]["y_delta_u"],
            "S": b["S_B_high_over_low"], "T": b["T_y_high_over_low"],
            "strict_support_gap_v": b["strict_support_gap_v"]}
        reset_premise = (b["high_window"]["y_delta_u"] > b["low_window"]["y_delta_u"] > 0
                         and b["T_y_high_over_low"] > b["S_B_high_over_low"]
                         and b["strict_support_gap_v"] > 0)
        if not reset_premise:
            result["status"] = "RESET_ordering_premise_failed_no_search"
            return result
        if not proof["SET_positive_for_all_lambda_ge_lambda0"]:
            result["status"] = "positive_component_argument_failed_no_search"
            return result
        roots = result["boundary_roots"]
        roots["S"] = self.boundary_root("S")
        roots["T"] = self.boundary_root("T")
        if any(not v["completed"] for v in roots.values()):
            result["status"] = "boundary_unresolved_no_search"
            return result
        lo, hi = roots["S"]["bracket"][1], roots["T"]["bracket"][0]
        result["conservative_open_positive_interval"] = [lo, hi]
        if not self.lambda0 < lo < hi:
            result["status"] = "empty_or_invalid_inner_interval_no_search"
            return result
        result["grid"] = [{"lambda_per_v": lo + (hi-lo)*i/(GRID_POINTS+1), "H": None,
                           "all_positive": False, "status": "unattempted"} for i in range(1, GRID_POINTS+1)]
        registered_candidates = {}

        def register_candidate(lam, origin):
            if lam in registered_candidates:
                registered_candidates[lam]["origins"].append(origin)
            else:
                planned = {"lambda_per_v": lam, "origins": [origin], "status": "unattempted"}
                result["candidate_plan"].append(planned)
                registered_candidates[lam] = planned

        grid = []
        for slot in result["grid"]:
            result["grid_plan"]["attempted"] += 1
            slot["status"] = "attempted"
            try:
                c = self.evaluate_h(slot["lambda_per_v"], "31_point_grid")
                slot.update(H=c["H"], all_positive=c["all_positive"], status=c["status"])
                result["grid_plan"]["completed"] += 1
            except BudgetExceeded as exc:
                slot.update(status=f"{exc.scope}_budget_exhausted_partial", error=str(exc))
                raise
            except NUMERICAL_ERRORS as exc:
                slot.update(status="numerical_failure_unresolved", error=f"{type(exc).__name__}: {exc}")
                c = self.h_cache.get(slot["lambda_per_v"], slot)
            grid.append(c)
            if valid_h(c) and abs(c["H"]) <= H_TOL:
                register_candidate(c["lambda_per_v"], "exact_zero_grid" if c["H"] == 0 else "near_zero_grid_candidate_not_a_located_root")
        brackets, rejected = [], []
        for left, right in zip(grid, grid[1:]):
            if not valid_h(left) or not valid_h(right):
                rejected.append([left["lambda_per_v"], right["lambda_per_v"]])
            elif left["H"] * right["H"] < 0:
                brackets.append((left, right))
        result["nonpositive_adjacent_pairs_not_bridged"] = rejected
        result["H_sign_change_brackets_found"] = len(brackets)
        result["H_sign_change_brackets_not_attempted_due_to_cap"] = [
            [a["lambda_per_v"], b["lambda_per_v"]] for a, b in brackets[MAX_H_BRACKETS:]]
        result["H_root_plan"] = [{"index": index, "bracket": [a["lambda_per_v"], b["lambda_per_v"]], "status": "unattempted"}
                                 for index, (a, b) in enumerate(brackets[:MAX_H_BRACKETS])]
        for index, (left, right) in enumerate(brackets[:MAX_H_BRACKETS]):
            result["H_root_plan"][index]["status"] = "attempted"
            solved = self.h_root(left, right, index)
            result["H_root_plan"][index]["status"] = solved["status"]
            if solved["completed"]:
                register_candidate(solved["candidate_lambda"], f"H_sign_change_bracket_{index}")
        for planned in sorted(result["candidate_plan"], key=lambda row: row["lambda_per_v"]):
            planned["status"] = "attempted"
            try:
                checked = self.candidate(self.h_cache[planned["lambda_per_v"]], planned["origins"])
                planned["status"] = checked["status"]
            except BudgetExceeded as exc:
                planned.update(status=f"{exc.scope}_budget_exhausted_partial", error=str(exc))
                raise
            except NUMERICAL_ERRORS as exc:
                planned.update(status="numerical_failure_unresolved", error=f"{type(exc).__name__}: {exc}")
        valid = [c for c in grid if valid_h(c)]
        if valid:
            best = min(valid, key=lambda c: abs(c["H"]))
            result["smallest_abs_H_grid_point_not_a_root_candidate"] = {
                "lambda_per_v": best["lambda_per_v"], "H": best["H"], "b_F": best["b_F"], "b_G": best["b_G"]}
        result.update(status="completed_finite_saved_node_diagnostic", no_root_does_not_prove_absence=True,
                      finite_grid_may_miss_tangent_or_multiple_roots=True)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Run only after source/plan freeze and review")
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return
    destination = HERE / "SAVED_LAMBDA_FEASIBILITY.json"
    marker = HERE / "RUN_STARTED.json"
    if destination.exists() or marker.exists():
        raise RuntimeError("Refusing repeat execution: run marker or result already exists")
    loading_start = time.monotonic()
    frozen = freeze_check()
    # Exclusive creation is the execution lock, including concurrent invocations.
    with marker.open("x") as stream:
        json.dump({"status": "run_started_no_automatic_retry", "source_freeze": frozen}, stream, indent=2)
        stream.write("\n")
    diagnostic = json.loads((OLD / INPUT_NAMES[0]).read_text())
    bounds = json.loads((OLD / INPUT_NAMES[2]).read_text())
    ledger = Counter({key: 0 for key in (
        "linear_identification_attempts", "two_by_two_linear_solve_attempts",
        "two_by_two_linear_solves_returned", "new_saved_window_sum_evaluations",
        "reset_boundary_scalar_function_evaluations", "H_parameter_point_evaluations",
        "parameter_scalar_root_problems_attempted", "parameter_scalar_roots_completed",
        "reset_boundary_bisection_steps", "H_bisection_steps", "candidate_checks",
        "candidate_window_closure_pairs", "saved_node_terms_reweighted")})
    result = {"status": "running", "source_freeze": frozen,
              "configurations": [{"interpolant": b["interpolant"], "gauss_order": b["gauss_order"],
                                  "status": "unattempted"} for b in bounds["configurations"]],
              "work_counts": ledger, "new_ODEs": 0, "new_state_inversions": 0,
              "new_static_circuit_roots": 0, "new_nonlinear_forward_fits": 0,
              "new_quadrature_nodes": 0,
              "parameter_scalar_root_solving_performed": True,
              "full_unified_model_achieved": False, "unconditional_mechanism_rejection": False,
              "scope": "Frozen readout/interpolated trajectories/path clock, finite saved nodes, training windows only.",
              "numerical_tolerances_not_physical_thresholds": {"abs_H": H_TOL, "abs_b_F_minus_b_G": B_TOL},
              "numerical_budgets_s": {"total": 60, "configuration": 15, "scalar_root": 2},
              "finite_search_is_not_complete_root_enumeration": True}
    numerical_elapsed = 0.0
    io_elapsed = 0.0
    try:
        with np.load(OLD / INPUT_NAMES[1], allow_pickle=False) as arrays:
            result["input_loading_and_hash_wall_s"] = time.monotonic() - loading_start
            for index, bound in enumerate(bounds["configurations"]):
                if numerical_elapsed >= 60:
                    result["configurations"][index]["unattempted_reason"] = "Total numerical budget exhausted"
                    continue
                try:
                    config = Configuration(diagnostic, bound, arrays, ledger, numerical_elapsed)
                except NUMERICAL_ERRORS as exc:
                    result["configurations"][index].update(status="input_preparation_numerical_failure",
                        error=f"{type(exc).__name__}: {exc}", numerical_wall_s=0.0)
                    continue
                config.report = result["configurations"][index]
                try:
                    config.run()
                finally:
                    numerical_elapsed = sum(c.get("numerical_wall_s", 0.0) for c in result["configurations"])
                io_start = time.monotonic()
                destination.write_text(json.dumps(clean(result), indent=2) + "\n")
                io_elapsed += time.monotonic() - io_start
                print(f"Completed {config.interpolant}/Gauss{config.gauss}", flush=True)
        io_start = time.monotonic()
        result["sources_unchanged"] = freeze_check() == frozen
        io_elapsed += time.monotonic() - io_start
        complete = all(c["status"] == "completed_finite_saved_node_diagnostic" for c in result["configurations"])
        if not result["sources_unchanged"]:
            result["status"] = "source_integrity_failed_results_not_validated"
        else:
            result["status"] = "completed_finite_saved_node_diagnostic" if complete else "completed_partial_finite_diagnostic_unresolved_items_retained"
    except Exception as exc:
        result.update(status="failed_partial_results_retained_no_automatic_retry", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["numerical_wall_s"] = sum(c.get("numerical_wall_s", 0.0) for c in result["configurations"])
        result["artifact_IO_and_final_hash_wall_s"] = io_elapsed
        result["parameter_scalar_root_solving_performed"] = ledger["parameter_scalar_root_problems_attempted"] > 0
        result["parameter_scalar_function_evaluations"] = (
            ledger["reset_boundary_scalar_function_evaluations"] + ledger["H_parameter_point_evaluations"])
        destination.write_text(json.dumps(clean(result), indent=2) + "\n")
    print(json.dumps(clean({"status": result["status"], "work_counts": ledger}), indent=2), flush=True)


if __name__ == "__main__":
    main()
