"""Prepared one-run terminal-candidate shutdown diagnostic; no model imports."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULT_PATH = HERE / "calibration_run/RESULT.json"
TABLE_PATH = Path("/private/tmp/vulcan-runtime-cache/transport_g010_v0025.npz")
TABLE_SHA = "00d6d9ea0b1dfe855181e7016afc6dd5e23d169f83f19cfcedfc9d3b35f9e4e2"
CONTRACT_PATH = ROOT / "analysis/constrained_shared_2026-11-03/feature_constraints.json"
PLAN_PATH = HERE / "SHUTDOWN_DIAGNOSTIC_PLAN.md"
TAG = "1r.reset.return.I_0p2V"
RECORDS = (1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
EXPECTED = {"a_max_s": 0.0013206747378601963, "kappa_a_per_v2": 3.60746202417315,
            "rho": 1.0629325349343612, "b": 1.1692973784703131}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def require(condition, message):
    if not condition:
        raise ValueError(message)


def B(voltage, lam):
    if lam == 0:
        return voltage * voltage / 2
    z = lam * voltage
    return (z + (z-1) * math.expm1(z)) / (lam*lam)


def write_markdown(result):
    lines = ["# Selected-candidate RESET shutdown diagnostic", "",
             f"Status: `{result['status']}`. Selected coarse request 26; final refined request 35.",
             "No new ODE, fit, inverse or root was performed. New frozen soft-readout evaluations are counted below.", "",
             "The lower bounds are conditional on the selected parameters, actual RESET entrances, frozen table and path clock.",
             "They are not parameter-family exclusions or experimental uncertainty intervals.", "",
             "| Record | q entry | q lower | q saved | a lower / S | a saved / S | Soft lower / A | Total lower / A | Saved total / A | Observed / A |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in result["records"]:
        if row["status"] != "completed":
            lines.append(f"| {row['record_id']} | unresolved: {row['status']} | | | | | | | | |")
            continue
        values = [row[k] for k in ("q_entry", "q_min", "q_saved", "a_min_s", "a_saved_s",
                  "soft_current_lower_a", "total_current_lower_a", "saved_total_abs_a", "observed_abs_a")]
        lines.append("| " + str(row["record_id"]) + " | " + " | ".join(f"{v:.9g}" for v in values) + " |")
    lines += ["", "## Original median gate", ""]
    aggregate = result.get("aggregate", {})
    if aggregate.get("defined"):
        for name, value in aggregate.items():
            if name != "defined":
                lines.append(f"- {name}: {value}")
        lines += ["", result["interpretation"]]
    else:
        lines.append("Unresolved: all original eligible records are required; no failed record is dropped.")
    lines += ["", "The saved actual soft current is distinct from the newly evaluated soft current at q_min.",
              "An a-state gap above its envelope includes continued formation and the reduction of loss by (1-q).",
              "Saved-point flux ratios cannot establish integrated contributions over the phase.",
              "Per-record observed-current ratios are diagnostics only; the original median gate is unchanged.", "",
              "## Work and provenance", "", "```json", json.dumps(result["counts"], indent=2), "```", ""]
    for path, digest in result.get("input_sha256_before", {}).items():
        lines.append(f"- `{path}`: `{digest}`")
    lines.append(f"\nInputs unchanged: `{result.get('inputs_unchanged')}`. Floating-point arithmetic; no interval-error certificate.\n")
    (HERE / "SHUTDOWN_DIAGNOSTIC.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--confirmed-terminal", action="store_true",
                        help="Use only after the root agent explicitly confirms terminal calibration")
    args = parser.parse_args()
    if not (args.run and args.confirmed_terminal):
        parser.print_help()
        return
    destination = HERE / "SHUTDOWN_DIAGNOSTIC.json"
    marker = HERE / "SHUTDOWN_DIAGNOSTIC_RUN_STARTED.json"
    require(not destination.exists() and not marker.exists(), "Diagnostic already started; no automatic retry")
    terminal = json.loads(RESULT_PATH.read_text())
    require(terminal["status"] == "bounded_search_and_final_checks_finished", "Calibration is not at the required terminal status")
    require(terminal["sources_unchanged"] and terminal["numerical_final_candidate_verified"], "Final candidate is not numerically verified with unchanged sources")
    selected = terminal["last_accepted_center"]
    require(selected["request_id"] == 26, "Unexpected formally accepted candidate")
    require(all(selected["parameters"][k] == v for k, v in EXPECTED.items()), "Accepted parameters differ from the predeclared diagnostic")
    request = next(r for r in terminal["requests"] if r["request_id"] == 35)
    require(request["eligible_for_selection"] and request["kind"] == "final" and request["precision"]["name"] == "refined", "Missing eligible final refined request")
    require(request["parameters"] == selected["parameters"], "Refined request changed candidate parameters")
    prefix = Path(request["artifact_prefix"]).resolve()
    require(prefix.parent == (HERE / "calibration_run/cases").resolve(), "Unexpected final artifact directory")
    case_path, arrays_path = prefix.with_suffix(".json"), prefix.with_suffix(".npz")
    case = json.loads(case_path.read_text())
    contract = json.loads(CONTRACT_PATH.read_text())
    require(case["features"]["contract_sha256"] == sha(CONTRACT_PATH), "Original feature contract changed")
    feature = contract["features"][TAG]
    gate = case["features"]["features"][TAG]
    require(gate["acceptance_interval"] == feature["proposed_empirical_acceptance_interval"], "Original tail gate changed")
    original_rows = {int(r["record"]): r for r in feature["records"] if r["eligible"]}
    require(tuple(sorted(original_rows)) == RECORDS and gate["n_eligible"] == 14, "Original tail eligibility differs")
    material = case["summary"]["material"]
    params = material["parameters"]
    require(all(params[k] == v for k, v in selected["parameters"].items()), "Saved material parameters differ")
    require(params["enable_a"] is True, "Selected case must have a enabled")
    ron, gc = material["readout"]["Ron_ohm"], material["readout"]["gamma_c_ev"]
    phase_summaries = {(r["topology"], int(r["record_id"]), r["phase"]): r["summary"] for r in case["phases"]}
    inputs = [RESULT_PATH, case_path, arrays_path, CONTRACT_PATH, TABLE_PATH, Path(__file__).resolve(), PLAN_PATH]
    before = {str(p): sha(p) for p in inputs}
    require(before[str(TABLE_PATH)] == TABLE_SHA, "Frozen table hash mismatch")
    with marker.open("x") as stream:
        json.dump({"status": "started_no_retry", "input_sha256": before}, stream, indent=2)
    result = {"status": "running", "selected_parameters": params, "records": [],
              "input_sha256_before": before, "table_monotonicity_audits": {},
              "original_gate": {"id": TAG, "acceptance_interval": gate["acceptance_interval"],
                                "saved_model_median": gate["model_median"], "saved_gate_passed": gate["passed"]},
              "counts": {"records_planned": 14, "records_attempted": 0, "records_completed": 0,
                         "new_soft_current_evaluations_attempted": 0, "new_soft_current_evaluations_completed": 0,
                         "saved_point_rate_arithmetic": 0, "ODEs": 0, "fits": 0, "inverse_mappings": 0,
                         "scalar_roots": 0, "circuit_roots": 0, "new_spectral_quadratures": 0},
              "floating_point_not_interval_certificate": True,
              "experimental_noise_certificate": False, "parameter_family_rejection": False}
    compute_elapsed = 0.0
    try:
        with np.load(TABLE_PATH, allow_pickle=False) as archive:
            lg, lv, li = [archive[k].copy() for k in ("log_gamma", "log_voltage", "log_current")]
            metadata = json.loads(str(archive["metadata"]))
        require(np.all(np.diff(lg)>0) and np.all(np.diff(lv)>0) and np.all(np.isfinite(li)), "Invalid frozen spectral axes/values")
        negative = li[1]
        shift = math.log(metadata["shape"]["ron_ohm"] / ron)
        vmax_domain = metadata["shape"]["voltage_limit_v"]
        require(math.log(vmax_domain) == float(lv[-1]), "Original table endpoint metadata mismatch")
        with np.load(arrays_path, allow_pickle=False) as arrays:
            for cid in RECORDS:
                row = {"record_id": cid, "status": "unattempted"}
                result["records"].append(row)
                if compute_elapsed >= 30:
                    row["reason"] = "Declared numerical budget exhausted"
                    continue
                prefix_key = f"1r_cycle_{cid}_reset__"
                # Decompression/loading is outside the arithmetic budget.
                saved = {key: arrays[prefix_key+key] for key in (
                    "command_voltage_v", "local_voltage_v", "compliance_limited", "state_q", "state_a_s",
                    "soft_current_a", "hard_current_a", "current_a", "observed_current_a", "original_scoring_mask")}
                started = time.monotonic()
                result["counts"]["records_attempted"] += 1
                row["status"] = "attempted"
                try:
                    v = saved["command_voltage_v"]
                    index = int(original_rows[cid]["native_index"])
                    apex = int(np.argmax(abs(v)))
                    require(index > apex and abs(abs(v[index])-.2) <= 3e-7, "Original return sample is invalid")
                    require(v[0] == 0 and v[-1] == 0 and np.all(v <= 0), "Unexpected bare RESET waveform")
                    require(np.all(np.diff(abs(v[:apex+1])) >= 0) and np.all(np.diff(abs(v[apex:])) <= 0), "RESET path is not the declared two monotone branches")
                    require(np.array_equal(v, saved["local_voltage_v"]) and not np.any(saved["compliance_limited"]), "Bare RESET is not actual-command unregulated drive")
                    require(saved["original_scoring_mask"][index], "Original return sample is not scored")
                    q0, a0 = float(saved["state_q"][0]), float(saved["state_a_s"][0])
                    q, a = float(saved["state_q"][index]), float(saved["state_a_s"][index])
                    phase = phase_summaries[("1r", cid, "reset")]
                    require(q0 == phase["initial_state_q"] and a0 == phase["initial_state_a_s"], "RESET entry does not match saved actual phase entry")
                    require(0<q0<1 and 0<q<1 and 0<=a0<=params["a_max_s"] and 0<=a<=params["a_max_s"], "State outside physical domain")
                    peak, read = float(max(abs(v))), float(abs(v[index]))
                    Dq = 2*B(peak, params["lambda_per_v"]) - B(read, params["lambda_per_v"])
                    Dv = peak*peak - read*read/2
                    increment = params["kappa_positive_per_v2"]*math.exp(2*params["b"])*Dq
                    q_min = 1/(1/q0+increment)
                    loss_budget = params["kappa_a_per_v2"]*params["rho"]*math.exp(params["b"])*Dv
                    log_a_min = math.log(a0)-loss_budget if a0>0 else None
                    a_min = math.exp(log_a_min) if log_a_min is not None else 0.0
                    soft, hard, total, observed = [abs(float(saved[key][index])) for key in (
                        "soft_current_a", "hard_current_a", "current_a", "observed_current_a")]
                    row.update(native_index=index, read_abs_voltage_v=read, peak_abs_voltage_v=peak,
                        q_entry=q0, a_entry_s=a0, q_saved=q, a_saved_s=a, q_min=q_min, a_min_s=a_min,
                        log_a_min_s=log_a_min, Dq_v2=Dq, Dv_v2=Dv,
                        q_inverse_denominator_increment=increment, a_max_loss_log_budget=loss_budget,
                        saved_soft_abs_a=soft, saved_hard_abs_a=hard, saved_total_abs_a=total,
                        observed_abs_a=observed, q_saved_over_q_min=q/q_min,
                        log_a_saved_over_a_min=(math.log(a)-log_a_min) if a>0 and log_a_min is not None else None)
                    require(observed>0 and math.isclose(observed, original_rows[cid]["value"], rel_tol=1e-14), "Original observed feature value mismatch")
                    require(math.isclose(total,soft+hard,rel_tol=1e-12,abs_tol=1e-18) and
                            math.isclose(hard,a*read,rel_tol=1e-12,abs_tol=1e-18), "Saved branch decomposition mismatch")
                    # Native-point rate arithmetic only; no new dynamic or circuit evaluation.
                    k, ka, b, rho = [params[key] for key in ("kappa_positive_per_v2", "kappa_a_per_v2", "b", "rho")]
                    capture = k*read*(total/params["J_positive_a"])*math.exp(-2*b)*(1-q)
                    recovery = k*read*math.exp(2*b+params["lambda_per_v"]*read)*q*q
                    formation = ka*read*math.exp(-b)*q*(params["a_max_s"]-a)
                    loss = ka*read*rho*math.exp(b)*(1-q)*a
                    row["saved_point_flux_arithmetic"] = dict(capture_q=capture,recovery_q=recovery,
                        formation_a=formation,loss_a=loss,q_net=capture-recovery,a_net=formation-loss,
                        capture_over_recovery=capture/recovery if recovery else None,
                        formation_over_loss=formation/loss if loss else None)
                    result["counts"]["saved_point_rate_arithmetic"] += 1
                    require(q >= q_min and a >= a_min, "Saved state violates evaluated analytical bound; unresolved numerics")
                    log_v = math.log(read)
                    require(lv[0] <= log_v <= lv[-1] and read <= vmax_domain, "Read voltage outside original table")
                    vi = min(max(int(np.searchsorted(lv,log_v)-1),0),len(lv)-2)
                    tv = float((log_v-lv[vi])/(lv[vi+1]-lv[vi]))
                    require(0<=tv<=1, "Voltage interpolation fraction outside table cell")
                    audit_key = str(vi)
                    if audit_key not in result["table_monotonicity_audits"]:
                        differences = np.diff(negative[:,vi:vi+2],axis=0)
                        result["table_monotonicity_audits"][audit_key] = dict(
                            negative_table_index=1, voltage_column_indices=[vi,vi+1],
                            voltage_log_endpoints=lv[vi:vi+2].tolist(),
                            gamma_log_domain=[float(lg[0]),float(lg[-1])],
                            n_gamma_increments_per_column=len(lg)-1,
                            minimum_log_current_increment=float(np.min(differences)),
                            negative_increment_count=int(np.count_nonzero(differences<0)),
                            monotone=bool(np.all(differences>=0)), tolerance_or_repair_used=False)
                    require(result["table_monotonicity_audits"][audit_key]["monotone"], "Negative table is not monotone in Gamma at the relevant voltage cell")
                    log_gamma_min = math.log(gc*q_min/(1-q_min))
                    log_gamma_saved = math.log(gc*q/(1-q))
                    require(lg[0]<=log_gamma_min<=log_gamma_saved<=lg[-1], "Bound/saved Gamma is outside monotone frozen table domain")
                    gi = min(max(int(np.searchsorted(lg,log_gamma_min)-1),0),len(lg)-2)
                    tg = float((log_gamma_min-lg[gi])/(lg[gi+1]-lg[gi]))
                    require(0<=tg<=1, "Gamma interpolation fraction outside table cell")
                    require(result["counts"]["new_soft_current_evaluations_attempted"] < 14, "Readout quota exhausted")
                    result["counts"]["new_soft_current_evaluations_attempted"] += 1
                    # Exactly one newly evaluated soft readout per record, at q_min.
                    log_current = ((1-tg)*((1-tv)*negative[gi,vi]+tv*negative[gi,vi+1])
                                   +tg*((1-tv)*negative[gi+1,vi]+tv*negative[gi+1,vi+1])+shift)
                    soft_lower = math.exp(float(log_current))
                    result["counts"]["new_soft_current_evaluations_completed"] += 1
                    require(math.isfinite(soft_lower) and soft_lower>0, "Invalid lower-bound soft readout")
                    hard_lower = read*a_min
                    total_lower = soft_lower+hard_lower
                    require(soft_lower<=soft*(1+1e-12) and total_lower<=total*(1+1e-12), "Readout bound exceeds saved current beyond declared roundoff comparison")
                    row.update(status="completed", soft_current_lower_a=soft_lower,
                        hard_current_lower_a=hard_lower,total_current_lower_a=total_lower,
                        lower_to_exact_observed_ratio=total_lower/observed,
                        soft_lower_to_exact_observed_ratio=soft_lower/observed,
                        table_interpolation=dict(gamma_lower_index=gi,voltage_lower_index=vi,
                            log_gamma_min=log_gamma_min,log_gamma_saved=log_gamma_saved,
                            gamma_fraction=tg,voltage_fraction=tv,ron_log_shift=shift),
                        bound_scope="Selected candidate and actual entry, monotone frozen interpolant; floating-point evaluation")
                    result["counts"]["records_completed"] += 1
                except Exception as exc:
                    row.update(status="unresolved_retained", error=f"{type(exc).__name__}: {exc}")
                finally:
                    compute_elapsed += time.monotonic()-started
        complete = len(result["records"])==14 and all(r["status"]=="completed" for r in result["records"])
        result["aggregate"] = {"defined": complete}
        if complete:
            medians = {name:float(np.median([r[key] for r in result["records"]])) for name,key in (
                ("soft_lower_median_a","soft_current_lower_a"),("hard_lower_median_a","hard_current_lower_a"),
                ("total_lower_median_a","total_current_lower_a"),("saved_total_median_a","saved_total_abs_a"),
                ("saved_soft_median_a","saved_soft_abs_a"),("saved_hard_median_a","saved_hard_abs_a"),
                ("observed_median_a","observed_abs_a"))}
            upper = float(gate["acceptance_interval"][1])
            require(math.isclose(medians["saved_total_median_a"],gate["model_median"],rel_tol=1e-12), "Saved original gate median mismatch")
            result["aggregate"].update(medians,original_upper_gate_a=upper,
                soft_recovery_bound_exceeds_upper=medians["soft_lower_median_a"]>upper,
                parallel_max_loss_bound_exceeds_upper=medians["hard_lower_median_a"]>upper,
                combined_bound_exceeds_upper=medians["total_lower_median_a"]>upper)
            clauses=[]
            if medians["soft_lower_median_a"]>upper:
                clauses.append("The selected q recovery envelope already exceeds the original median upper gate from its actual RESET entrances, even with capture removed.")
            if medians["hard_lower_median_a"]>upper:
                clauses.append("The selected parallel maximum-loss envelope also exceeds that gate, even with formation removed.")
            if medians["total_lower_median_a"]>upper and not clauses:
                clauses.append("Only the combined conditional lower bound exceeds the original median upper gate.")
            if not clauses:
                clauses.append("The conservative lower bound permits the original gate; it does not establish reachability. Compare saved states and fluxes for capture, formation and q-dependent loss suppression.")
            clauses.append("These are fixed-candidate/actual-entry statements, not exclusions of the parameter family or a new model-family recommendation.")
            result["interpretation"]=" ".join(clauses)
        result["status"]="completed_conditional_shutdown_diagnostic" if complete else "completed_partial_unresolved_records_retained"
    except Exception as exc:
        result.update(status="failed_partial_diagnostic_no_retry",error=f"{type(exc).__name__}: {exc}")
    finally:
        result["numerical_wall_s"]=compute_elapsed
        result["input_sha256_after"]={str(p):sha(p) for p in inputs}
        result["inputs_unchanged"]=result["input_sha256_after"]==before
        if not result["inputs_unchanged"]:
            result["status"]="input_integrity_failure_no_validated_conclusion"
        destination.write_text(json.dumps(clean(result),indent=2)+"\n")
        write_markdown(clean(result))
    print(json.dumps(clean({"status":result["status"],"counts":result["counts"],"aggregate":result.get("aggregate")}),indent=2))


if __name__ == "__main__":
    main()
