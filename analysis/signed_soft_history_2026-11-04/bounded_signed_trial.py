"""Prepared bounded RESET4 calibration and continuous SET5 check.

The sole contract is EXPERIMENT_PLAN.md. No model runs unless explicitly
invoked with --run-authorized after root/reviewer approval. SET5 never enters
the b objective or candidate selection. No previous-stage source is modified.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
from scipy.optimize import minimize_scalar
from signed_forward_kernel import SignedSoftKernel, Precision, PhaseFailure

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "cooperative_relaxation_2026-11-04"
BMAX = math.log(10.0)
LIMITS = {"log_current_decade": 1e-4, "state_u": 1e-3,
          "legacy_log_current_decade": 5e-7, "local_voltage_v": 1e-9,
          "KCL_a": 1e-13, "KVL_v": 1e-12, "state_join_u": 1e-12}


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    return None if isinstance(value, float) and not math.isfinite(value) else value


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def parallel_hash(paths):
    paths = sorted({str(Path(p).resolve()) for p in paths})
    with ThreadPoolExecutor(max_workers=4) as pool:
        return dict(zip(paths, pool.map(hash_file, paths)))


class Artifacts:
    """Hash exactly the serialized bytes being written; never reread outputs."""
    def __init__(self):
        self.io_wall_s = 0.0
        self.catalog = {}

    def _save(self, path, producer, *, replace=False, catalog=True):
        started = time.monotonic()
        try:
            payload = producer()
            digest = hashlib.sha256(payload).hexdigest()
            with path.open("wb" if replace else "xb") as handle:
                handle.write(payload)
            item = {"path": str(path), "sha256": digest, "bytes": len(payload)}
            if catalog:
                self.catalog[str(path)] = item
            return item
        finally:
            # Serialization/compression, hashing serialized bytes and disk waits
            # belong to artifact I/O, never to the numerical model budget.
            self.io_wall_s += time.monotonic() - started

    def json(self, path, value, *, replace=False, catalog=True):
        return self._save(path, lambda: (json.dumps(clean(value), indent=2,
                          ensure_ascii=False, allow_nan=False) + "\n").encode(),
                          replace=replace, catalog=catalog)

    def npz(self, path, arrays):
        def produce():
            stream = BytesIO()
            np.savez_compressed(stream, **arrays)
            return stream.getvalue()
        return self._save(path, produce)


class BudgetExceeded(TimeoutError):
    pass


class StopSearch(RuntimeError):
    pass


class NumericalBudget:
    """Only active numerical calls accrue; no active timer survives an I/O call."""
    def __init__(self):
        self.total_s, self.search_s = 0.0, 0.0
        self.calls = []
        self.active = None

    def check(self):
        if self.active is None:
            raise RuntimeError("Budget callback used outside a numerical operation")
        elapsed = time.monotonic() - self.active["started"]
        if self.total_s + elapsed >= 240.0:
            raise BudgetExceeded("Cumulative 240 s numerical budget exhausted")
        if self.active["search"] and self.search_s + elapsed >= 120.0:
            raise BudgetExceeded("Cumulative 120 s RESET4 search budget exhausted")
        limit = self.active["per_call_limit_s"]
        if limit is not None and elapsed >= limit:
            raise BudgetExceeded(f"Per-phase {limit:g} s numerical budget exhausted")

    def run(self, label, function, *, search=False, per_call_limit_s=None):
        if self.active is not None:
            raise RuntimeError("Nested numerical accounting is prohibited")
        self.active = dict(label=label, started=time.monotonic(), search=search,
                           per_call_limit_s=per_call_limit_s)
        state = "completed"
        try:
            self.check()
            result = function(self.check)
            self.check()  # A completed call that overshot its budget is not a fit candidate.
            return result
        except Exception:
            state = "failed_or_budget_stopped"
            raise
        finally:
            elapsed = time.monotonic() - self.active["started"]
            self.total_s += elapsed
            if search:
                self.search_s += elapsed
            self.calls.append(dict(label=label, elapsed_s=elapsed, search=search,
                                   per_call_limit_s=per_call_limit_s, status=state))
            self.active = None


def point_integrity(point):
    a, i, v, t = (float(point[k]) for k in
                   ("command_voltage_v", "current_a", "local_voltage_v", "transistor_voltage_v"))
    finite = all(math.isfinite(x) for x in (a, i, v, t, point["KCL_residual_a"], point["KVL_residual_v"]))
    sign = (i == 0.0 and v == 0.0 and t == 0.0) if a == 0.0 else (
        i * a > 0.0 and v * a > 0.0 and t * a >= 0.0)
    return dict(passed=bool(finite and sign and i * v >= 0.0 and i * t >= 0.0
                           and abs(a - v - t) <= LIMITS["KVL_v"]
                           and abs(point["KVL_residual_v"]) <= LIMITS["KVL_v"]
                           and abs(point["KCL_residual_a"]) <= LIMITS["KCL_a"]),
                finite=finite, signs_match_actual_command=bool(sign),
                port_and_component_passivity=bool(i * a >= 0.0 and i * v >= 0.0 and i * t >= 0.0),
                KVL_abs_v=abs(a - v - t))


def phase_integrity(kernel, result, data, initial_u):
    a, s = result["arrays"], result["summary"]
    v, mask = data["command_voltage_v"], data["scoring_mask"]
    np.testing.assert_array_equal(a["command_voltage_v"], v)
    np.testing.assert_array_equal(a["observed_current_a"], data["observed_current_a"])
    np.testing.assert_array_equal(a["original_scoring_mask"], mask)
    keys = ("current_a", "local_voltage_v", "transistor_voltage_v", "KCL_residual_a",
            "KVL_residual_v", "state_u", "state_q")
    shapes = all(a[k].shape == v.shape for k in keys)
    finite = shapes and all(np.all(np.isfinite(a[k])) for k in keys)
    points = [point_integrity({k: (float(v[j]) if k == "command_voltage_v" else float(a[k][j]))
                              for k in ("command_voltage_v", "current_a", "local_voltage_v",
                                        "transistor_voltage_v", "KCL_residual_a", "KVL_residual_v")})
              for j in range(len(v))]
    gammas = np.asarray([kernel.gamma(u) for u in a["state_u"]])
    q_valid = bool(np.all((a["state_q"] > 0) & (a["state_q"] < 1)))
    np.testing.assert_allclose(a["state_q"], np.exp(-a["state_u"]), rtol=1e-13, atol=0)
    repeated = np.flatnonzero(np.diff(v) == 0.0)
    repeated_error = float(max(abs(a["state_u"][repeated + 1] - a["state_u"][repeated]), default=0.0))
    entrance_error = abs(float(a["state_u"][0]) - initial_u)
    terminal_error = abs(float(a["state_u"][-1]) - s["terminal_u"])
    count_ok = s["n_native"] == len(v) and s["n_scored"] == int(mask.sum())
    rhs_ok, rhs_omission_count = True, 0
    if "rhs_trace" in a:
        rows = a["rhs_trace"]
        present = np.all(np.isfinite(rows[:, [2, 3, 4, 6]]), axis=1)
        missing = rows[~present]
        rhs_omission_count = len(missing)
        rhs_ok = bool(rhs_omission_count == s["smallfield_omission"]["n_RHS_omissions"]
                      and np.all(abs(missing[:, 0]) < kernel.vmin)
                      and np.all(missing[:, 0] != 0.0))
        for row in rows[present]:
            rhs_ok = rhs_ok and point_integrity(dict(command_voltage_v=row[0], current_a=row[2],
                local_voltage_v=row[3], transistor_voltage_v=row[4], KCL_residual_a=row[6],
                KVL_residual_v=row[0] - row[3] - row[4]))["passed"]
    passed = bool(finite and count_ok and q_valid and all(p["passed"] for p in points) and rhs_ok
                  and s["max_rhs_KCL_residual_a"] <= LIMITS["KCL_a"]
                  and s["max_segment_state_join_error"] <= LIMITS["state_join_u"]
                  and max(repeated_error, entrance_error, terminal_error) <= LIMITS["state_join_u"]
                  and s["initial_u"] == initial_u and s["no_conditioned_anchor"]
                  and s["no_hidden_state_reset"])
    return dict(passed=passed, native_arrays_match_data=True, native_shapes_and_finiteness=finite,
                exact_native_and_scored_counts=count_ok, native_points_passive_and_signed=all(p["passed"] for p in points),
                state_domain_checked_without_clipping=True, gamma_min_ev=float(gammas.min()), gamma_max_ev=float(gammas.max()),
                state_q_valid=q_valid, actual_initial_u=initial_u, entrance_state_error=entrance_error,
                terminal_state_error=terminal_error, maximum_duplicate_voltage_state_error=repeated_error,
                RHS_passivity_and_declared_omissions_passed=rhs_ok, RHS_omission_rows=rhs_omission_count)


def add_diagnostics(result, data, protocol):
    """Saved native arithmetic only; never an objective for parameter selection."""
    a, s = result["arrays"], result["summary"]
    v, obs, current, mask = a["command_voltage_v"], abs(a["observed_current_a"]), abs(a["current_a"]), a["original_scoring_mask"]
    outward = np.arange(len(v)) <= np.argmax(abs(v))
    groups = {"all_eligible": mask, "outward": mask & outward, "return": mask & ~outward,
              "outward_low_0p2": mask & outward & (abs(v) <= .2 + 1e-12),
              "return_low_0p2": mask & ~outward & (abs(v) <= .2 + 1e-12),
              "both_branches_low_0p2": mask & (abs(v) <= .2 + 1e-12),
              "remaining_above_0p2": mask & (abs(v) > .2 + 1e-12)}
    s["original_mask_diagnostics"] = {name: dict(n=int(use.sum()), RMSE_decade=float(
        np.sqrt(np.mean(np.log10(current[use] / obs[use]) ** 2)))) for name, use in groups.items() if np.any(use)}
    feature_rows = []
    for f in protocol["applicable_single_record_features"]:
        if (f["record"], f["phase"]) != (data["record_id"], data["phase"]):
            continue
        name, indices = f["identifier"], np.asarray(f["native_indices"], dtype=int)
        if name.endswith(".peak_current") or name.endswith(".peak_abs_voltage"):
            domain = np.flatnonzero(outward)
            j = int(domain[np.argmax(current[domain])])
            value = float(current[j] if name.endswith(".peak_current") else abs(v[j]))
            used = [j]
        elif ".alpha_" in name:
            value = float(np.log(current[indices[1]] / current[indices[0]]) / np.log(abs(v[indices[1]] / v[indices[0]])))
            used = indices.tolist()
        elif "geomean_current" in name:
            value, used = float(10 ** np.mean(np.log10(current[indices]))), indices.tolist()
        elif "log_current_q90_q10_span" in name:
            ly = np.log10(current[indices]); value, used = float(np.quantile(ly, .9) - np.quantile(ly, .1)), indices.tolist()
        else:
            value, used = float(current[indices[0]]), indices.tolist()
        feature_rows.append(dict(identifier=name, predicted_value=value, observed_value=f["observed_value"],
                                 unit=f["unit"], predicted_native_indices=used,
                                 observed_native_indices=f["native_indices"],
                                 original_aggregate_band_reference_only=f["original_aggregate_acceptance_interval"]))
    s["single_record_feature_references"] = feature_rows
    s["feature_scope"] = "Single-record diagnostics only; no original20 aggregate gate pass/fail"
    if data["phase"] == "set":
        crossings = []
        for threshold in (1e-6, 1e-5):
            row = {"threshold_a": threshold}
            for label, trace in (("observed", obs), ("predicted", current)):
                hits = np.flatnonzero(outward & mask & (trace >= threshold))
                row[label] = dict(reached=False) if not len(hits) else dict(reached=True,
                    last_below_voltage_v=float(v[hits[0] - 1]), first_at_or_above_voltage_v=float(v[hits[0]]))
            crossings.append(row)
        s["outward_current_crossings"] = crossings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-authorized", action="store_true")
    args = parser.parse_args()
    if not args.run_authorized:
        raise SystemExit("Prepared only; root/reviewer approval required before --run-authorized.")
    destination, planpath, trialdir = HERE / "bounded_signed_trial.json", HERE / "bounded_signed_trial.plan.json", HERE / "phase_trials"
    if destination.exists() or planpath.exists() or trialdir.exists():
        raise FileExistsError("Preserve all existing attempts; no restart or output overwrite")
    store, budget = Artifacts(), NumericalBudget()
    load_start = time.monotonic()
    kernel = SignedSoftKernel.load()
    data = {"set4": kernel.load_training_phase(4, "set"), "reset4": kernel.load_training_phase(4, "reset"),
            "set5": kernel.load_training_phase(5, "set")}
    reference = kernel.load_set4_reference_arrays()
    protocol_path = HERE / "DATA_PROTOCOL_AUDIT.json"
    protocol = json.loads(protocol_path.read_text())
    if protocol["original_holdout_arrays_read"] is not False:
        raise ValueError("Data audit scope must be training-only")
    cachepath = HERE / "training_reset4_set5_native.npz"
    if hash_file(cachepath) != protocol["small_cache"]["sha256"]:
        raise ValueError("Audited two-record cache changed")
    with np.load(cachepath, allow_pickle=False) as archive:
        for label in ("reset4", "set5"):
            phase = data[label]; prefix = f'1t1r_cycle_{phase["record_id"]}_{phase["phase"]}'
            np.testing.assert_array_equal(phase["command_voltage_v"], archive[prefix + "_voltage"])
            np.testing.assert_array_equal(phase["observed_current_a"], archive[prefix + "_current"])
            np.testing.assert_array_equal(phase["scoring_mask"], archive[prefix + "_original_scoring_mask"])
    for label, native, scored in (("set4", 503, 500), ("reset4", 173, 170), ("set5", 503, 500)):
        phase = data[label]
        if len(phase["command_voltage_v"]) != native or int(phase["scoring_mask"].sum()) != scored:
            raise ValueError("Original phase counts changed")
    np.testing.assert_array_equal(data["set4"]["command_voltage_v"], reference["command_voltage_v"])
    np.testing.assert_array_equal(data["set4"]["observed_current_a"], reference["observed_current_a"])
    np.testing.assert_array_equal(data["set4"]["scoring_mask"], reference["original_scoring_mask"])
    frozen = kernel.set4_reference_summary
    reset_initial = float(frozen["terminal_u"])
    if reset_initial != protocol["state_handoff"]["SET4_saved_terminal_u"] or reset_initial != float(reference["state_u"][-1]):
        raise ValueError("Frozen SET4 terminal handoff differs from audited state")
    sources = dict(kernel.source_sha256)
    additions = [HERE / "EXPERIMENT_PLAN.md", protocol_path, cachepath, Path(__file__)]
    loaded_hashes = parallel_hash([*sources, *additions])
    if any(loaded_hashes[p] != digest for p, digest in sources.items()):
        raise ValueError("Actual inherited input hash changed")
    sources.update({str(p.resolve()): loaded_hashes[str(p.resolve())] for p in additions})
    loading_s = time.monotonic() - load_start
    search_precision = Precision("search", 2e-6, 2e-7, .04)
    precisions = [Precision("nominal", 1e-7, 1e-8, .02), Precision("refined", 1e-8, 1e-9, .01)]
    static_points = [(command, state) for state in (reset_initial, float(frozen["conditional_anchor_u"]), 25.)
                     for command in (.2, 5., -.02, -2.)]
    plan = dict(contract="EXPERIMENT_PLAN.md", contract_sha256=sources[str((HERE / "EXPERIMENT_PLAN.md").resolve())],
        scope="Frozen m2 signed SET4-terminal -> RESET4 -> SET5; RESET4-only b calibration",
        frozen_positive_parameters=asdict(kernel.parameters), fixed_m=2, parallel_path_a_s=0.,
        b_domain=[0., BMAX], b_parameter_count=1, SET5_used_for_selection=False,
        RESET4_incoming_u=reset_initial, SET4_regression_initial_u=float(frozen["derived_initial_u"]),
        SET5_incoming_u="Each precision's actual RESET4 terminal_u; no measured anchor or reset",
        static_points=[dict(command_v=a, state_u=u) for a, u in static_points],
        RHS_b_values=[0., BMAX], RHS_check="Independent unified q equation versus signed u RHS on selected already-solved 12 points",
        RHS_arithmetic_tolerance="5e-12*(abs(recovery_term)+abs(capture_term))+1e-12; no physical accuracy claim",
        search_precision=asdict(search_precision), verification_precisions=[asdict(x) for x in precisions],
        regression_precision=asdict(precisions[-1]), regression_b=0.,
        endpoint_objective_calls=2, bounded_optimizer_max_calls=18, maximum_RESET4_search_calls=20,
        optimizer=dict(method="bounded", maxiter=18, xatol=1e-4, restarts=0),
        objective="Mean squared log10(|I_model|/|I_observed|) on all170 original RESET4 eligible points only",
        invalid_objective="positive infinity for optimizer, JSON null with explicit failure status; no finite penalty/clipping",
        maximum_phase_ODE_attempts=25, search_phase_limit_s=12., regression_and_final_phase_limit_s=30.,
        numerical_total_budget_s=240., numerical_search_budget_s=120.,
        timing="Accumulate active static/RHS/phase/diagnostic/legacy numerical call wall time only. Input loading, byte serialization, hashing and disk I/O separately recorded; no active timer during I/O.",
        timeout_policy="Any search phase timeout stops the entire search; a complete prior best may enter reserved final verification; no retries",
        limits=LIMITS, phase_counts={"SET4": [503, 500], "RESET4": [173, 170], "SET5": [503, 500]},
        onset_diagnostics_A=[1e-6, 1e-5], feature_scope="13 single-record references, never original20 aggregate validation",
        original_holdout_accessed=False, bare_records_accessed=False, new_inversions=0,
        output_hash_policy="SHA256 of exactly serialized bytes recorded on every write; no repeated output reads required",
        no_domain_grid_parameter_or_model_expansion=True)
    plan_item = store.json(planpath, plan)
    sources[str(planpath)] = plan_item["sha256"]
    trialdir.mkdir()
    report = dict(status="running_bounded_signed_history_trial", plan=plan, source_sha256=sources,
                  input_loading_and_hash_wall_s=loading_s, trials=[], final_pairs=[],
                  phase_ODE_attempts=0, search_objective_calls=0, optimizer_objective_calls=0,
                  numerical_validation_passed=False, full_unified_model_achieved=False,
                  original_holdout_accessed=False, SET5_used_for_b_selection=False)
    overall_start = time.monotonic()
    best = None

    def persist():
        report["timing"] = dict(numerical_total_s=budget.total_s, numerical_search_s=budget.search_s,
            output_serialization_and_IO_s=store.io_wall_s, active_wall_since_plan_s=time.monotonic() - overall_start,
            numerical_calls=budget.calls)
        report["artifact_catalog"] = dict(store.catalog)
        return store.json(destination, report, replace=True, catalog=False)

    def phase_call(label, phase_name, b, initial_u, precision, *, search=False):
        nonlocal best
        row = dict(trial_id=f'{len(report["trials"]) + 1:03d}_{label}', label=label, phase=phase_name,
                   b=float(b), initial_u=float(initial_u), initial_command_voltage_v=0.,
                   precision=asdict(precision), search_objective=search,
                   phase_call_started=False, objective_MSE=None)
        report["trials"].append(row)
        arrays, holder, result = {}, {}, None
        failure_stops_search = False
        try:
            if not math.isfinite(b) or not 0. <= b <= BMAX:
                row.update(status="outside_declared_b_domain", objective_for_optimizer="positive_infinity")
                return None
            if report["phase_ODE_attempts"] >= 25:
                raise BudgetExceeded("Hard 25 phase-ODE attempt limit reached")
            if search:
                if phase_name != "reset4" or report["search_objective_calls"] >= 20:
                    raise BudgetExceeded("Search is RESET4-only with at most20 calls")
                report["search_objective_calls"] += 1
            def calculate(check):
                report["phase_ODE_attempts"] += 1
                row["phase_call_started"] = True
                phase = data[phase_name]
                value = kernel.simulate_phase(phase["command_voltage_v"], circuit, initial_u, b, precision,
                    initial_command_voltage_v=0., observed_current_a=phase["observed_current_a"],
                    scoring_mask=phase["scoring_mask"], check_budget=check, collect_rhs=not search)
                holder["result"] = value
                return value
            result = budget.run(row["trial_id"], calculate, search=search,
                                per_call_limit_s=12. if search else 30.)
            arrays = result["arrays"]
            mse = float(np.mean(result["residual"] ** 2))
            if not math.isfinite(mse):
                raise ValueError("Completed phase has no finite original-mask score")
            row.update(status="completed", summary=deepcopy(result["summary"]), objective_MSE=mse)
            if search and (best is None or mse < best["MSE"]):
                best = dict(b=float(b), MSE=mse, RMSE_decade=math.sqrt(mse), source_trial=row["trial_id"])
                report["best_RESET4_search_candidate"] = dict(best)
            return result
        except PhaseFailure as exc:
            arrays = exc.arrays
            row.update(status="numerical_failure_preserved", summary=deepcopy(exc.summary),
                       objective_for_optimizer="positive_infinity", objective_MSE=None)
            failure_stops_search = exc.summary.get("failure_scope") == "caller_budget_exhausted"
            return None
        except Exception as exc:
            if "result" in holder:
                arrays = holder["result"]["arrays"]
                row["summary"] = deepcopy(holder["result"]["summary"])
            row.update(status="caller_budget_exhausted" if isinstance(exc, TimeoutError) else "caller_failure_preserved",
                       error_type=type(exc).__name__, error=str(exc), objective_MSE=None,
                       objective_for_optimizer="positive_infinity")
            failure_stops_search = isinstance(exc, TimeoutError)
            return None
        finally:
            row["numerical_budget_total_s_after"] = budget.total_s
            row["numerical_budget_search_s_after"] = budget.search_s
            row["arrays_artifact"] = store.npz(trialdir / (row["trial_id"] + ".npz"), arrays)
            store.json(trialdir / (row["trial_id"] + ".json"), row)
            persist()
            print(json.dumps(clean(dict(event="phase_terminal", trial_id=row["trial_id"],
                b=row["b"], status=row["status"], RMSE_decade=(None if row["objective_MSE"] is None
                else math.sqrt(row["objective_MSE"])), phase_ODE_attempts=report["phase_ODE_attempts"]))), flush=True)
            if search and failure_stops_search:
                raise StopSearch("Search timeout/call budget: preserve attempts and stop; no restart")

    def audit_final(label, value, phase_name, incoming):
        def compute(check):
            integrity = phase_integrity(kernel, value, data[phase_name], incoming)
            check()
            legacy, rows = kernel.audit_native_legacy(value, check_budget=check)
            value["arrays"]["native_legacy_audit"] = rows
            legacy_ok = bool(not legacy["failures"] and legacy["completed_points"] == len(data[phase_name]["command_voltage_v"])
                and legacy.get("max_log_current_difference_decade", math.inf) <= LIMITS["legacy_log_current_decade"]
                and legacy.get("max_local_voltage_difference_v", math.inf) <= LIMITS["local_voltage_v"]
                and legacy.get("max_reference_KCL_residual_a", math.inf) <= LIMITS["KCL_a"])
            value["summary"]["native_integrity"] = integrity
            value["summary"]["native_legacy_audit"] = legacy
            value["summary"]["native_legacy_audit_passed"] = legacy_ok
            add_diagnostics(value, data[phase_name], protocol)
            return bool(integrity["passed"] and legacy_ok)
        ok = False
        try:
            ok = budget.run(label + "_legacy_and_integrity", compute)
        except Exception as exc:
            value["summary"]["postphase_audit_failure"] = dict(error_type=type(exc).__name__, error=str(exc))
        value["summary"]["final_phase_validation_passed"] = ok
        json_item = store.json(HERE / (label + ".json"), value["summary"])
        npz_item = store.npz(HERE / (label + ".npz"), value["arrays"])
        return ok, dict(summary=json_item, arrays=npz_item)

    persist()
    try:
        static = budget.run("12_static_points", lambda check: kernel.validate_circuit(static_points, check_budget=check))
        for mode in ("fast", "strict_bisection42"):
            chosen = [row for row in static["rows"] if row["mode"] == mode]
            for row in chosen:
                row["signed_passivity_and_KVL"] = point_integrity(row["point"])
            static["modes"][mode]["full_physical_passed"] = bool(static["modes"][mode]["passed"]
                and len(chosen) == 12 and all(row["signed_passivity_and_KVL"]["passed"] for row in chosen))
        report["static_validation"] = static
        selected_mode = next((mode for mode in ("fast", "strict_bisection42") if static["modes"][mode]["full_physical_passed"]), None)
        if selected_mode is None:
            raise RuntimeError("Neither circuit mode passed all12 static checks; no phase ODE")
        report["selected_circuit_mode"] = selected_mode
        circuit = kernel.make_circuit(mode=selected_mode)
        def rhs_algebra(check):
            rows = []
            p = kernel.parameters
            for source in [x for x in static["rows"] if x["mode"] == selected_mode]:
                state, point = source["state_u"], source["point"]
                q, v, current = math.exp(-state), abs(point["local_voltage_v"]), abs(point["current_a"])
                sign = 1 if point["local_voltage_v"] > 0 else -1
                rates = []
                for b in (0., BMAX):
                    check()
                    actual = kernel.signed_rate(state, point, b)
                    kq, j0 = p.kappa_per_v2 * math.exp(b), p.J_positive_a * math.exp(2 * b)
                    capture = kq * v * (current / j0) * math.exp(sign * b) * (1 - q) / q
                    recovery = kq * v * math.exp(-sign * b + p.lambda_per_v * v) * q
                    expected = recovery - capture
                    tolerance = 5e-12 * (abs(recovery) + abs(capture)) + 1e-12
                    rates.append(actual)
                    rows.append(dict(command_v=source["command_v"], state_u=state, b=b,
                        actual_signed_u_rate=actual, unified_q_transformed_u_rate=expected,
                        absolute_difference=abs(actual - expected), tolerance=tolerance,
                        passed=abs(actual - expected) <= tolerance))
                if sign > 0 and rates[0] != rates[1]:
                    raise ValueError("Positive-voltage RHS changed with b")
            if len(rows) != 24 or not all(row["passed"] for row in rows):
                raise ValueError("Declared unified RHS algebra check failed")
            return dict(passed=True, rows=rows, positive_b_invariance_exact=True, new_roots=0)
        report["RHS_algebra_validation"] = budget.run("24_RHS_algebra_values", rhs_algebra)
        persist()
        regression = phase_call("SET4_regression", "set4", 0., float(frozen["derived_initial_u"]), precisions[-1])
        if regression is None:
            raise RuntimeError("Positive regression failed; stop before b calibration")
        def compare_regression(check):
            check()
            a, mask = regression["arrays"], data["set4"]["scoring_mask"]
            current_difference = float(np.max(abs(np.log10(abs(a["current_a"][mask] / reference["current_a"][mask])))))
            state_difference = float(np.max(abs(a["state_u"] - reference["state_u"])))
            integrity = phase_integrity(kernel, regression, data["set4"], float(frozen["derived_initial_u"]))
            return dict(max_log_current_difference_decade=current_difference, max_state_u_difference=state_difference,
                native_integrity=integrity, passed=bool(current_difference <= LIMITS["log_current_decade"]
                and state_difference <= LIMITS["state_u"] and integrity["passed"]),
                formal_RESET4_entry_remains_frozen_terminal_u=reset_initial,
                regression_terminal_u_not_used_as_formal_entry=regression["summary"]["terminal_u"])
        report["positive_regression"] = budget.run("SET4_regression_comparison", compare_regression)
        store.json(HERE / "SET4_regression_validation.json", report["positive_regression"])
        persist()
        if not report["positive_regression"]["passed"]:
            raise RuntimeError("Positive regression tolerance failed; stop before b calibration")
        try:
            for b, label in ((0., "b_zero"), (BMAX, "b_ln10")):
                phase_call(label, "reset4", b, reset_initial, search_precision, search=True)
            def objective(b):
                if report["optimizer_objective_calls"] >= 18:
                    raise StopSearch("Hard18 bounded optimizer objective calls reached")
                report["optimizer_objective_calls"] += 1
                value = phase_call("bounded_b", "reset4", float(b), reset_initial, search_precision, search=True)
                return math.inf if value is None else float(np.mean(value["residual"] ** 2))
            optimization = minimize_scalar(objective, bounds=(0., BMAX), method="bounded",
                options={"maxiter": 18, "xatol": 1e-4})
            report["optimizer"] = dict(success=bool(optimization.success), message=str(optimization.message),
                nfev=int(optimization.nfev), nit=int(optimization.nit), b=float(optimization.x),
                objective_MSE=float(optimization.fun), global_optimum_claim=False)
        except StopSearch as exc:
            report["search_stop_reason"] = str(exc)
        persist()
        if best is None:
            report["status"] = "completed_no_finite_RESET4_candidate_no_physical_rejection"
        else:
            report["selected_b"] = dict(best, source="Minimum170-point RESET4 MSE over completed search calls only",
                SET5_used=False, exact_boundary=best["b"] in (0., BMAX),
                normalized_domain_margin=min(best["b"], BMAX - best["b"]) / BMAX)
            verified = {}
            for precision in precisions:
                pair = dict(precision=asdict(precision), b=best["b"], status="started", phases={})
                report["final_pairs"].append(pair)
                reset = phase_call("final_" + precision.name + "_reset4", "reset4", best["b"], reset_initial, precision)
                if reset is None:
                    pair["status"] = "RESET4_failed_no_SET5_entry"; persist(); continue
                ok_reset, reset_artifacts = audit_final("final_" + precision.name + "_reset4", reset, "reset4", reset_initial)
                pair["phases"]["reset4"] = dict(validation_passed=ok_reset, artifacts=reset_artifacts, summary=reset["summary"])
                incoming_set5 = float(reset["summary"]["terminal_u"])
                set5 = phase_call("final_" + precision.name + "_set5", "set5", best["b"], incoming_set5, precision)
                if set5 is None:
                    pair["status"] = "SET5_failed_after_completed_RESET4"; persist(); continue
                ok_set5, set5_artifacts = audit_final("final_" + precision.name + "_set5", set5, "set5", incoming_set5)
                pair["phases"]["set5"] = dict(validation_passed=ok_set5, artifacts=set5_artifacts, summary=set5["summary"])
                join = abs(set5["summary"]["initial_u"] - reset["summary"]["terminal_u"])
                command_join = abs(set5["summary"]["initial_command_voltage_v"] - reset["summary"]["final_command_voltage_v"])
                pair.update(status="completed_pair", phase_state_join_error=join, phase_command_join_error_v=command_join,
                    validation_passed=bool(ok_reset and ok_set5 and max(join, command_join) <= LIMITS["state_join_u"]))
                verified[precision.name] = dict(reset4=reset, set5=set5)
                persist()
            if set(verified) == {"nominal", "refined"}:
                def compare_precision(check):
                    comparisons = {}
                    for name in ("reset4", "set5"):
                        check(); a, z = (verified[p][name]["arrays"] for p in ("nominal", "refined"))
                        mask = data[name]["scoring_mask"]
                        logdiff = float(np.max(abs(np.log10(abs(a["current_a"][mask] / z["current_a"][mask])))))
                        udiff = float(np.max(abs(a["state_u"] - z["state_u"])))
                        comparisons[name] = dict(n_scored=int(mask.sum()), max_log_current_difference_decade=logdiff,
                            max_state_u_difference=udiff, passed=bool(logdiff <= LIMITS["log_current_decade"] and udiff <= LIMITS["state_u"]))
                    return comparisons
                report["precision_comparison"] = budget.run("two_phase_precision_comparison", compare_precision)
                passed = all(c["passed"] for c in report["precision_comparison"].values()) and all(p.get("validation_passed", False) for p in report["final_pairs"])
                report["numerical_validation_passed"] = bool(passed)
                if passed:
                    report["verified_scores"] = {name: verified["refined"][name]["summary"]["original_mask_diagnostics"] for name in ("reset4", "set5")}
                    report["score_roles"] = dict(reset4="Single-record calibration", set5="Training-internal continuous next-phase check, not original holdout")
            report["status"] = "completed_bounded_signed_attempt_no_full_model_claim"
    except Exception as exc:
        report.update(status="partial_or_failed_signed_attempt_preserved", error_type=type(exc).__name__, error=str(exc))
    finally:
        hash_start = time.monotonic()
        try:
            final_hashes = parallel_hash(sources)
            report["source_files_unchanged"] = {path: final_hashes[path] == value for path, value in sources.items()}
            if not all(report["source_files_unchanged"].values()):
                report["status"] = "invalid_input_source_changed"
                report["numerical_validation_passed"] = False
                report.pop("verified_scores", None)
        except Exception as exc:
            report["source_recheck_failure"] = dict(error_type=type(exc).__name__, error=str(exc))
            report["numerical_validation_passed"] = False
            report.pop("verified_scores", None)
        report["final_source_hash_IO_wall_s"] = time.monotonic() - hash_start
        report["search_complete_calls"] = sum(t["status"] == "completed" and t["search_objective"] for t in report["trials"])
        report["completed_phase_calls"] = sum(t["status"] == "completed" for t in report["trials"])
        report["failed_phase_calls"] = sum(t["status"] != "completed" for t in report["trials"])
        final_item = persist()
    print(json.dumps(clean(dict(status=report["status"], selected_b=report.get("selected_b"),
        numerical_validation_passed=report["numerical_validation_passed"], phase_ODE_attempts=report["phase_ODE_attempts"],
        verified_scores=report.get("verified_scores"), numerical_total_s=budget.total_s,
        report_artifact=final_item)), indent=2), flush=True)


if __name__ == "__main__":
    main()
