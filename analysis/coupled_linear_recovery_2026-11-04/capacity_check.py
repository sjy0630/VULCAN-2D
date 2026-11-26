"""Prepared one-readout q1 capacity check; no model imports or new trajectories."""
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
CALIBRATION = ROOT / "analysis/coupled_training_calibration_2026-11-04/calibration_run"
RESULT_PATH = CALIBRATION / "RESULT.json"
CONTRACT_PATH = ROOT / "analysis/constrained_shared_2026-11-03/feature_constraints.json"
TABLE_PATH = Path("/private/tmp/vulcan-runtime-cache/transport_g010_v0025.npz")
TABLE_SHA = "00d6d9ea0b1dfe855181e7016afc6dd5e23d169f83f19cfcedfc9d3b35f9e4e2"
KERNEL_PATH = ROOT / "analysis/coupled_native_training_2026-11-04/coupled_forward_v2.py"
KERNEL_SHA = "c6873e8cec07965a3637bb695c942fab2fd75d699f0413ae1a2b0332097c2d2e"
ADAPTER_PATH = KERNEL_PATH.with_name("training_problem_v2.py")
ADAPTER_SHA = "de4539d573e3b0c13fbfbf8a9d89b28d29a2baa8d3458ef3145bb0e9f32480f7"
PLAN_PATH = HERE / "CAPACITY_PLAN.md"
TAG = "1r.set.return.R_0p05V"
RECORDS = (1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
EXPECTED = {"lambda_per_v": 0.4470398752554136,
    "kappa_positive_per_v2": 57.67798793605963,
    "J_positive_a": 0.0037505935471625162,
    "a_max_s": 0.0013206747378601963,
    "kappa_a_per_v2": 3.60746202417315,
    "rho": 1.0629325349343612,
    "b": 1.1692973784703131}
ENTRIES = {"bare1": [4.298219923945171e-05, 0.0],
           "bare5": [0.00025057076016083026, 0.0],
           "series4": [7.755279151592979e-05, 0.0]}
ICC = 0.001


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_report(result):
    lines = ["# Static q¹ low-resistance capacity check", "",
             f"Status: `{result['status']}`.", "",
             "No q¹ trajectory or fit was run. The shared-parameter statements below concern the bare 1R gate only.", ""]
    if "error" in result:
        lines += [f"Unresolved failure: {result['error']}", ""]
    for key in ("rectangle", "positive_readout", "selected_capacity", "necessary_A"):
        if key in result:
            lines += [f"## {key}", "", "```json", json.dumps(result[key], indent=2), "```", ""]
    lines += [result.get("interpretation", "No capacity conclusion was completed."), "",
              "A necessary bound is not an attainable-state claim or a sufficient fit criterion.",
              "All statements retain the fixed J, b, rho, lambda, readout, entries and original protocol.",
              "Changing these conditions requires rechecking the inequalities.", "",
              "```json", json.dumps(result["counts"], indent=2), "```", "",
              f"Inputs unchanged: `{result.get('inputs_unchanged')}`. Ordinary floating point, not interval arithmetic.", ""]
    (HERE / "CAPACITY_CHECK.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--review-confirmed", action="store_true")
    args = parser.parse_args()
    if not (args.run and args.review_confirmed):
        parser.print_help()
        return
    destination = HERE / "CAPACITY_CHECK.json"
    marker = HERE / "CAPACITY_CHECK_RUN_STARTED.json"
    require(not destination.exists() and not marker.exists(), "Capacity check already started; no automatic retry")
    result = {"status": "running", "protocol_records": [],
        "counts": {"saved_protocol_checks": 0, "negative_stored_node_maxima": 0,
                   "positive_monotonicity_audits": 0, "positive_readouts_attempted": 0,
                   "positive_readouts_completed": 0, "negative_forward_readouts": 0,
                   "ODEs": 0, "fits": 0, "inverses": 0, "roots": 0, "quadratures": 0},
        "evaluated_recovery_order": 1, "new_trajectories": False,
        "q1_states_replaced_by_q2_states": False,
        "floating_point_not_interval_certificate": True}
    before = {}
    numerical_wall = 0.0
    with marker.open("x") as stream:
        json.dump({"status": "started_no_retry", "script_sha256": sha(__file__),
                   "plan_sha256": sha(PLAN_PATH)}, stream, indent=2)
    try:
        for path in (RESULT_PATH, CONTRACT_PATH, TABLE_PATH, KERNEL_PATH, ADAPTER_PATH,
                     Path(__file__).resolve(), PLAN_PATH):
            before[str(path)] = sha(path)
        require(before[str(TABLE_PATH)] == TABLE_SHA and before[str(KERNEL_PATH)] == KERNEL_SHA and
                before[str(ADAPTER_PATH)] == ADAPTER_SHA, "Frozen table or circuit/adapter source changed")
        terminal = json.loads(RESULT_PATH.read_text())
        require(terminal["status"] == "bounded_search_and_final_checks_finished" and
                terminal["sources_unchanged"] and terminal["numerical_final_candidate_verified"], "Missing verified terminal q2 result")
        selected = terminal["last_accepted_center"]
        require(selected["request_id"] == 26 and selected["parameters"] == EXPECTED, "Unexpected selected parameters")
        request = next(r for r in terminal["requests"] if r["request_id"] == 35)
        require(request["eligible_for_selection"] and request["kind"] == "final" and
                request["precision"]["name"] == "refined" and request["parameters"] == EXPECTED, "Unexpected refined source")
        prefix = Path(request["artifact_prefix"]).resolve()
        require(prefix.parent == (CALIBRATION / "cases").resolve(), "Unexpected artifact directory")
        case_path, arrays_path = prefix.with_suffix(".json"), prefix.with_suffix(".npz")
        for path in (case_path, arrays_path):
            before[str(path)] = sha(path)
        case = json.loads(case_path.read_text())
        contract = json.loads(CONTRACT_PATH.read_text())
        require(case["features"]["contract_sha256"] == before[str(CONTRACT_PATH)], "Original feature contract changed")
        original_feature, feature = contract["features"][TAG], case["features"]["features"][TAG]
        require(feature["acceptance_interval"] == original_feature["proposed_empirical_acceptance_interval"], "Original resistance gate changed")
        eligible = {int(r["record"]): r for r in original_feature["records"] if r["eligible"]}
        require(tuple(sorted(eligible)) == RECORDS and feature["n_eligible"] == 14, "Original resistance eligibility differs")
        summary = case["summary"]
        require(summary["initial_states"] == ENTRIES and summary["missing_bare4_bridged"] is False,
                "Original entries or missing-history policy changed")
        checkpoint = summary["checkpoint_metadata"]
        require(checkpoint["bare_set_actual_circuit_compliance_a"] == ICC and
                checkpoint["reset_compliance_a"] is None, "Original bare compliance differs")
        material = summary["material"]
        require(all(material["parameters"][k] == v for k, v in EXPECTED.items()) and
                material["fixed_recovery_order"] == 2 and material["parameters"]["enable_a"] is True,
                "Reference q2 material differs")
        require(EXPECTED["lambda_per_v"] >= 0 and EXPECTED["b"] >= 0 and EXPECTED["rho"] > 0 and
                EXPECTED["J_positive_a"] > 0 and EXPECTED["kappa_a_per_v2"] >= 0, "Required signs do not hold")
        ron, gc = material["readout"]["Ron_ohm"], material["readout"]["gamma_c_ev"]
        require(ron == 58.463033350725695 and gc == 0.03127031375421752, "Frozen readout parameters differ")
        with np.load(arrays_path, allow_pickle=False) as arrays:
            for cid in RECORDS:
                sv = arrays[f"1r_cycle_{cid}_set__command_voltage_v"]
                rv = arrays[f"1r_cycle_{cid}_reset__command_voltage_v"]
                started = time.monotonic()
                require(numerical_wall < 30, "Arithmetic budget exhausted")
                si, ri = int(np.argmax(abs(sv))), int(np.argmax(abs(rv)))
                idx = int(eligible[cid]["native_index"])
                require(sv[0] == sv[-1] == rv[0] == rv[-1] == 0 and np.all(sv >= 0) and np.all(rv <= 0), "Unexpected waveform polarity/endpoints")
                require(np.all(np.diff(sv[:si+1]) >= 0) and np.all(np.diff(sv[si:]) <= 0) and
                        np.all(np.diff(abs(rv[:ri+1])) >= 0) and np.all(np.diff(abs(rv[ri:])) <= 0), "Unexpected waveform branches")
                require(idx > si and abs(float(sv[idx])-.05) <= 3e-7 and max(abs(rv)) <= 2.5,
                        "Read point or RESET peak violates declared protocol")
                result["protocol_records"].append({"record_id": cid, "native_index": idx,
                    "read_command_voltage_v": float(sv[idx]), "reset_peak_abs_voltage_v": float(max(abs(rv)))})
                result["counts"]["saved_protocol_checks"] += 1
                numerical_wall += time.monotonic()-started
        with np.load(TABLE_PATH, allow_pickle=False) as archive:
            lg, lv, li = [archive[k].copy() for k in ("log_gamma", "log_voltage", "log_current")]
            metadata = json.loads(str(archive["metadata"]))
        started = time.monotonic()
        try:
            require(numerical_wall < 30, "Arithmetic budget exhausted")
            reads = {r["read_command_voltage_v"] for r in result["protocol_records"]}
            require(len(reads) == 1, "Different read voltages require more than the authorized one readout")
            read = next(iter(reads))
            reset_max = max(r["reset_peak_abs_voltage_v"] for r in result["protocol_records"])
            require(np.all(np.diff(lg) > 0) and np.all(np.diff(lv) > 0) and np.all(np.isfinite(li)), "Invalid table axes/values")
            vmax_table = metadata["shape"]["voltage_limit_v"]
            require(math.log(vmax_table) == float(lv[-1]) and reset_max <= vmax_table, "Invalid original endpoint domain")
            qcap = ICC/(ICC+EXPECTED["J_positive_a"])
            fcap = qcap/(qcap+EXPECTED["rho"]*math.exp(-2*EXPECTED["b"])*(1-qcap))
            require(0 < qcap < 1 and 0 < fcap < 1 and
                    all(0 < ENTRIES[key][0] <= qcap and ENTRIES[key][1] == 0 for key in ("bare1", "bare5")),
                    "Original bare entries fail the proposed rectangle")
            log_g = math.log(gc*qcap/(1-qcap))
            log_v = math.log(read)
            require(lg[0] <= log_g <= lg[-1] and lv[0] <= log_v <= lv[-1], "Capacity point outside frozen table")
            shift = math.log(metadata["shape"]["ron_ohm"] / ron)
            negative_right_column = int(np.searchsorted(lv, math.log(reset_max), side="left"))
            require(0 <= negative_right_column < len(lv), "RESET upper bracket outside table")
            negative_nodes = li[1, :, :negative_right_column+1]
            neg_flat = int(np.argmax(negative_nodes))
            neg_index = np.unravel_index(neg_flat, negative_nodes.shape)
            neg_max = math.exp(float(li[1][neg_index])+shift)
            result["counts"]["negative_stored_node_maxima"] += 1
            neg_rhs = ICC*math.exp(4*EXPECTED["b"])
            selected_a = EXPECTED["a_max_s"]
            neg_selected_lhs = neg_max+selected_a*fcap*reset_max
            selected_rectangle = neg_selected_lhs <= neg_rhs
            result["rectangle"] = {"qcap": qcap, "fcap": fcap,
                "selected_a_upper_s": selected_a*fcap,
                "negative_protocol_node_envelope_max_a": neg_max,
                "negative_max_node_indices": [int(i) for i in neg_index],
                "negative_voltage_columns_inclusive": [0, negative_right_column],
                "negative_envelope_upper_node_voltage_v": math.exp(float(lv[negative_right_column])),
                "negative_envelope_all_gamma_rows": True,
                "negative_envelope_includes_nearzero_linear_extension": True,
                "negative_envelope_includes_right_bracketing_column": True,
                "reset_max_v": reset_max, "negative_boundary_rhs_lower_a": neg_rhs,
                "selected_negative_boundary_lhs_upper_a": neg_selected_lhs,
                "selected_negative_boundary_margin_a": neg_rhs-neg_selected_lhs,
                "selected_joint_rectangle_sufficient_check_passed": selected_rectangle,
                "negative_check_is_sufficient_not_necessary": True}
            vi = min(max(int(np.searchsorted(lv, log_v)-1), 0), len(lv)-2)
            gi = min(max(int(np.searchsorted(lg, log_g)-1), 0), len(lg)-2)
            tv = float((log_v-lv[vi])/(lv[vi+1]-lv[vi]))
            tg = float((log_g-lg[gi])/(lg[gi+1]-lg[gi]))
            require(0 <= tv <= 1 and 0 <= tg <= 1, "Invalid interpolation fractions")
            increments = np.diff(li[0, :, vi:vi+2], axis=0)
            result["positive_monotonicity"] = {"voltage_indices": [vi, vi+1],
                "minimum_increment": float(np.min(increments)),
                "negative_increment_count": int(np.count_nonzero(increments < 0)),
                "tolerance_or_repair_used": False}
            result["counts"]["positive_monotonicity_audits"] += 1
            require(np.all(increments >= 0), "Positive readout is not monotone in Gamma")
            result["counts"]["positive_readouts_attempted"] += 1
            log_i = ((1-tg)*((1-tv)*li[0, gi, vi]+tv*li[0, gi, vi+1])+
                     tg*((1-tv)*li[0, gi+1, vi]+tv*li[0, gi+1, vi+1])+shift)
            fsoft = math.exp(float(log_i))
            require(fsoft > 0 and math.isfinite(fsoft), "Nonfinite positive capacity readout")
            result["counts"]["positive_readouts_completed"] += 1
            r_upper = float(feature["acceptance_interval"][1])
            required_current = read/r_upper
            require(r_upper > 0 and 0 < required_current < ICC, "Gate cannot use the planned below-compliance proof")
            observed_R = [float(eligible[cid]["value"]) for cid in RECORDS]
            require(all(r > 0 and math.isfinite(r) for r in observed_R), "Invalid original eligible observed resistance")
            observed_R_median = float(np.median(observed_R))
            capacity_current = fsoft+read*selected_a*fcap
            excluded = selected_rectangle and capacity_current < required_current
            result["positive_readout"] = {"read_voltage_v": read, "soft_at_qcap_a": fsoft,
                "gamma_lower_index": gi, "voltage_lower_index": vi,
                "gamma_fraction": tg, "voltage_fraction": tv,
                "feature_id": TAG, "original_R_gate_upper_ohm": r_upper,
                "original_observed_R_eligible_ohm": observed_R,
                "original_observed_R_median_ohm": observed_R_median,
                "fcap": fcap,
                "negative_protocol_node_envelope_max_a": neg_max,
                "current_required_for_upper_R_a": required_current}
            result["selected_capacity"] = {"selected_A_s": selected_a,
                "total_read_current_upper_a": capacity_current,
                "upper_below_compliance": capacity_current < ICC,
                "unconstrained_read_resistance_lower_ohm": read/capacity_current,
                "current_capacity_less_than_required": capacity_current < required_current,
                "all_positive_kappa_excluded_for_original_R_gate": excluded,
                "applies_to_all_original_eligible_records_hence_median": excluded}
            a_req = (1/r_upper-fsoft/read)/fcap
            result["necessary_A"] = {"raw_A_threshold_s": a_req, "positive_threshold": a_req > 0,
                "verified_necessary_lower_bound_s": None, "attainability_or_sufficiency_claim": False}
            if a_req > 0:
                neg_req_lhs = neg_max+a_req*fcap*reset_max
                req_rectangle = neg_req_lhs <= neg_rhs
                result["necessary_A"].update(
                    threshold_negative_boundary_lhs_upper_a=neg_req_lhs,
                    threshold_negative_boundary_margin_a=neg_rhs-neg_req_lhs,
                    threshold_joint_rectangle_sufficient_check_passed=req_rectangle,
                    verified_necessary_lower_bound_s=a_req if req_rectangle else None,
                    bound_scope="All A below threshold, fixed J/b/rho/lambda/readout/entries/protocol; any kappa>0 and kappa_a>=0")
            clauses = []
            if excluded:
                clauses.append("The selected A is capacity-excluded from the original bare low-resistance median gate for every positive kappa under the joint invariant rectangle.")
            else:
                clauses.append("This finite check does not exclude the selected A; a failed sufficient rectangle test is unresolved, not evidence that the rectangle is violated.")
            if result["necessary_A"]["verified_necessary_lower_bound_s"] is not None:
                clauses.append("The positive A threshold is a verified necessary lower bound under the recorded fixed conditions; reaching it does not guarantee a valid fit.")
            result["interpretation"] = " ".join(clauses)
        finally:
            numerical_wall += time.monotonic()-started
        require(numerical_wall <= 30, "Arithmetic budget exceeded; no validated capacity conclusion")
        result["status"] = "completed_static_conditional_capacity_check"
    except Exception as exc:
        result.update(status="failed_partial_capacity_check_no_retry", error=f"{type(exc).__name__}: {exc}")
    finally:
        result["numerical_wall_s"] = numerical_wall
        result["input_sha256_before"] = before
        result["input_sha256_after"] = {p: sha(p) for p in before}
        result["inputs_unchanged"] = result["input_sha256_after"] == before
        if not result["inputs_unchanged"]:
            result["status"] = "input_integrity_failure_no_validated_capacity_conclusion"
        destination.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        write_report(result)
    print(json.dumps({"status": result["status"], "counts": result["counts"],
        "selected_capacity": result.get("selected_capacity"), "necessary_A": result.get("necessary_A"),
        "inputs_unchanged": result["inputs_unchanged"]}, indent=2))


if __name__ == "__main__":
    main()
