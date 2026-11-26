"""One reviewed analytic q1 seed derivation; no model imports or simulation."""
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
PLAN_PATH = HERE / "SEED_PLAN.md"
RECORDS = (1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
MULTIPLIERS = (1, 3, 10)
TAG = "1r.reset.return.I_0p2V"
EXPECTED_PARAMETERS = {
    "lambda_per_v": 0.4470398752554136,
    "kappa_positive_per_v2": 57.67798793605963,
    "J_positive_a": 0.0037505935471625162,
    "a_max_s": 0.0013206747378601963,
    "kappa_a_per_v2": 3.60746202417315,
    "rho": 1.0629325349343612,
    "b": 1.1692973784703131,
}
EXPECTED_ENTRIES = {
    "bare1": [4.298219923945171e-05, 0.0],
    "bare5": [0.00025057076016083026, 0.0],
    "series4": [7.755279151592979e-05, 0.0],
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def B(voltage, lam):
    if lam == 0:
        return voltage * voltage / 2
    z = lam * voltage
    return (z + (z-1)*math.expm1(z)) / (lam*lam)


def write_report(result):
    lines = ["# Fixed q¹ seed derivation", "", f"Status: `{result['status']}`.", "",
             "This is a deleted-capture reference from saved q² RESET entrances, not a q¹ fit or trajectory.",
             "The shared median-gate upper limit defines a common seed target only; no per-record gate is added.", ""]
    if "error" in result:
        lines += [f"Unresolved failure: {result['error']}", ""]
    if "target" in result:
        target = result["target"]
        lines += [f"- Native return voltage: {target['signed_voltage_v']:.17g} V",
                  f"- Original median-gate upper limit: {target['current_upper_a']:.17g} A",
                  f"- q target at a = 0: {target['q_target']:.17g}", ""]
    if "kappa1_reference_per_v2" in result:
        lines += [f"Reference κ₁: **{result['kappa1_reference_per_v2']:.17g} V⁻²**.", "",
                  "| Fixed multiplier | Shared κ / V⁻² |", "|---:|---:|"]
        for candidate in result["candidates"]:
            lines.append(f"| {candidate['multiplier']} | {candidate['parameters']['kappa_positive_per_v2']:.17g} |")
        lines += ["", "All other parameters and all three original segment entries are unchanged."]
    lines += ["", "κ scales capture and recovery in both polarities. Actual q¹ entrances, capture feedback and a histories will change.",
              "This reference is not a family-wide rate bound, a reachability guarantee or a physical-time calibration.",
              "The subsequent three full forward comparisons require separate execution authorization.", "",
              "```json", json.dumps(result["counts"], indent=2), "```", "",
              f"Inputs unchanged: `{result.get('inputs_unchanged')}`. No interval-arithmetic certificate.", ""]
    (HERE / "SEED_DERIVATION.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--review-confirmed", action="store_true",
                        help="Use only after the root agent reviews and authorizes this one derivation")
    args = parser.parse_args()
    if not (args.run and args.review_confirmed):
        parser.print_help()
        return
    destination = HERE / "SEED_DERIVATION.json"
    marker = HERE / "SEED_DERIVATION_RUN_STARTED.json"
    require(not destination.exists() and not marker.exists(), "Seed derivation already started; no automatic retry")
    before = {}
    result = {"status": "running", "records": [], "candidates": [],
              "counts": {"saved_record_checks": 0, "monotonicity_audits": 0,
                         "analytic_inverse_attempts": 0, "analytic_inverse_completed": 0,
                         "scalar_rate_calculations": 0, "candidate_declarations": 0,
                         "ODEs": 0, "fits": 0, "forward_trajectories": 0,
                         "iterative_inverses": 0, "scalar_roots": 0,
                         "circuit_roots": 0, "spectral_quadratures": 0},
              "q1_entry_states_inferred": False, "physical_time_calibration": False,
              "fit_or_validation_score": False, "parameter_family_bound": False,
              "floating_point_not_interval_certificate": True}
    with marker.open("x") as stream:
        json.dump({"status": "started_no_retry", "script_sha256": sha(__file__),
                   "plan_sha256": sha(PLAN_PATH)}, stream, indent=2)
    numerical_wall = 0.0
    try:
        for path in (RESULT_PATH, CONTRACT_PATH, TABLE_PATH, Path(__file__).resolve(), PLAN_PATH):
            before[str(path)] = sha(path)
        require(before[str(TABLE_PATH)] == TABLE_SHA, "Frozen table hash mismatch")
        terminal = json.loads(RESULT_PATH.read_text())
        require(terminal["status"] == "bounded_search_and_final_checks_finished", "Calibration not at terminal state")
        require(terminal["sources_unchanged"] and terminal["numerical_final_candidate_verified"], "Final q2 candidate not verified")
        selected = terminal["last_accepted_center"]
        require(selected["request_id"] == 26 and selected["parameters"] == EXPECTED_PARAMETERS, "Unexpected selected q2 candidate")
        request = next(r for r in terminal["requests"] if r["request_id"] == 35)
        require(request["eligible_for_selection"] and request["kind"] == "final" and
                request["precision"]["name"] == "refined" and request["parameters"] == EXPECTED_PARAMETERS,
                "Unexpected final refined q2 request")
        prefix = Path(request["artifact_prefix"]).resolve()
        require(prefix.parent == (CALIBRATION / "cases").resolve(), "Unexpected final artifact location")
        case_path, arrays_path = prefix.with_suffix(".json"), prefix.with_suffix(".npz")
        for path in (case_path, arrays_path):
            before[str(path)] = sha(path)
        case = json.loads(case_path.read_text())
        contract = json.loads(CONTRACT_PATH.read_text())
        require(case["features"]["contract_sha256"] == before[str(CONTRACT_PATH)], "Original feature contract changed")
        feature, gate = contract["features"][TAG], case["features"]["features"][TAG]
        require(feature["proposed_empirical_acceptance_interval"] == gate["acceptance_interval"], "Original median gate changed")
        original = {int(r["record"]): r for r in feature["records"] if r["eligible"]}
        require(tuple(sorted(original)) == RECORDS and gate["n_eligible"] == 14, "Original eligibility changed")
        material = case["summary"]["material"]
        require(all(material["parameters"][k] == v for k, v in EXPECTED_PARAMETERS.items()) and
                material["parameters"]["enable_a"] is True and material["fixed_recovery_order"] == 2,
                "Saved material differs from selected q2 candidate")
        require(case["summary"]["initial_states"] == EXPECTED_ENTRIES, "Original three segment entries differ")
        ron, gc = material["readout"]["Ron_ohm"], material["readout"]["gamma_c_ev"]
        require(ron == 58.463033350725695 and gc == 0.03127031375421752, "Frozen readout parameters changed")
        phases = {(p["topology"], int(p["record_id"]), p["phase"]): p["summary"] for p in case["phases"]}
        with np.load(arrays_path, allow_pickle=False) as arrays:
            for cid in RECORDS:
                key = f"1r_cycle_{cid}_reset__"
                saved = {name: arrays[key+name] for name in
                         ("command_voltage_v", "local_voltage_v", "compliance_limited", "state_q", "state_a_s")}
                started = time.monotonic()
                require(numerical_wall < 30, "Numerical arithmetic budget exhausted")
                v = saved["command_voltage_v"]
                idx = int(original[cid]["native_index"])
                apex = int(np.argmax(abs(v)))
                require(idx > apex and abs(abs(v[idx])-.2) <= 3e-7, "Unexpected original return index")
                require(v[0] == 0 and v[-1] == 0 and np.all(v <= 0) and
                        np.all(np.diff(abs(v[:apex+1])) >= 0) and np.all(np.diff(abs(v[apex:])) <= 0),
                        "RESET waveform differs from declared two monotone branches")
                require(np.array_equal(v, saved["local_voltage_v"]) and not np.any(saved["compliance_limited"]),
                        "Bare RESET is not unregulated actual-command drive")
                q0, a0 = float(saved["state_q"][0]), float(saved["state_a_s"][0])
                phase = phases[("1r", cid, "reset")]
                require(q0 == phase["initial_state_q"] and a0 == phase["initial_state_a_s"], "Saved q2 entrance mismatch")
                require(0 < q0 < 1 and 0 <= a0 <= EXPECTED_PARAMETERS["a_max_s"], "Invalid saved entrance")
                result["records"].append({"record_id": cid, "native_index": idx,
                    "q2_actual_reset_entry_q": q0, "q2_actual_reset_entry_a_s": a0,
                    "read_abs_voltage_v": float(abs(v[idx])), "peak_abs_voltage_v": float(max(abs(v)))})
                result["counts"]["saved_record_checks"] += 1
                numerical_wall += time.monotonic()-started
        with np.load(TABLE_PATH, allow_pickle=False) as table:
            lg, lv, li = [table[k].copy() for k in ("log_gamma", "log_voltage", "log_current")]
            metadata = json.loads(str(table["metadata"]))
        started = time.monotonic()
        try:
            require(numerical_wall < 30, "Numerical arithmetic budget exhausted")
            read_values = {row["read_abs_voltage_v"] for row in result["records"]}
            require(len(read_values) == 1, "Records require different inverses; one-inverse plan cannot proceed")
            read = next(iter(read_values))
            upper = float(gate["acceptance_interval"][1])
            require(upper > 0 and math.isfinite(upper), "Invalid original gate upper limit")
            require(np.all(np.diff(lg) > 0) and np.all(np.diff(lv) > 0) and np.all(np.isfinite(li)), "Invalid frozen table")
            require(math.log(metadata["shape"]["voltage_limit_v"]) == float(lv[-1]), "Frozen endpoint metadata mismatch")
            log_v = math.log(read)
            require(lv[0] <= log_v <= lv[-1], "Read voltage outside frozen table")
            vi = min(max(int(np.searchsorted(lv, log_v)-1), 0), len(lv)-2)
            tv = float((log_v-lv[vi])/(lv[vi+1]-lv[vi]))
            require(0 <= tv <= 1, "Invalid voltage interpolation fraction")
            differences = np.diff(li[1, :, vi:vi+2], axis=0)
            result["monotonicity"] = {"negative_table_index": 1, "voltage_indices": [vi, vi+1],
                "minimum_increment": float(np.min(differences)),
                "negative_increment_count": int(np.count_nonzero(differences < 0)),
                "tolerance_or_repair_used": False}
            result["counts"]["monotonicity_audits"] += 1
            require(np.all(differences >= 0), "Negative frozen table is not monotone")
            shift = math.log(metadata["shape"]["ron_ohm"] / ron)
            j_at_read = (1-tv)*li[1, :, vi] + tv*li[1, :, vi+1] + shift
            target_log_i = math.log(upper)
            require(j_at_read[0] < target_log_i < j_at_read[-1], "Target outside invertible frozen current range")
            gi = int(np.searchsorted(j_at_read, target_log_i, side="right")-1)
            require(0 <= gi < len(lg)-1 and j_at_read[gi+1] > j_at_read[gi], "Target inverse cell has no positive slope")
            result["counts"]["analytic_inverse_attempts"] += 1
            fraction = float((target_log_i-j_at_read[gi])/(j_at_read[gi+1]-j_at_read[gi]))
            log_gamma = float(lg[gi]+fraction*(lg[gi+1]-lg[gi]))
            require(0 <= fraction <= 1 and lg[0] <= log_gamma <= lg[-1], "Analytic inverse left its frozen cell")
            gamma = math.exp(log_gamma)
            q_target = gamma/(gamma+gc)
            require(0 < q_target < 1 and math.isfinite(q_target), "Invalid inverse target state")
            result["counts"]["analytic_inverse_completed"] += 1
            result["target"] = {"feature_id": TAG, "original_gate_is_median_not_per_record": True,
                "current_upper_a": upper, "signed_voltage_v": -read, "a_assumed_s": 0.0,
                "gamma_target_ev": gamma, "q_target": q_target,
                "inversion": {"method": "one analytic inverse of log-bilinear frozen table",
                    "gamma_lower_index": gi, "voltage_lower_index": vi, "voltage_fraction": tv,
                    "current_fraction": fraction, "log_gamma_target": log_gamma, "ron_log_shift": shift}}
            for row in result["records"]:
                dose = 2*B(row["peak_abs_voltage_v"], EXPECTED_PARAMETERS["lambda_per_v"]) - B(read, EXPECTED_PARAMETERS["lambda_per_v"])
                log_ratio = math.log(row["q2_actual_reset_entry_q"]/q_target)
                rate = log_ratio/(math.exp(2*EXPECTED_PARAMETERS["b"])*dose)
                require(dose > 0 and log_ratio > 0 and rate > 0 and math.isfinite(rate), "Nonpositive or nonfinite reference rate; no clipping")
                row.update(D_v2=dose, log_entry_over_target=log_ratio, kappa1_reference_per_v2=rate)
                result["counts"]["scalar_rate_calculations"] += 1
            reference = float(np.median([row["kappa1_reference_per_v2"] for row in result["records"]]))
            require(reference > 0 and math.isfinite(reference), "Invalid median reference")
            result["kappa1_reference_per_v2"] = reference
            result["q1_kappa_reference"] = reference
            result["q1_kappa_reference_unit"] = "V^-2"
            result["initial_states"] = EXPECTED_ENTRIES
            result["seed_parameters"] = dict(EXPECTED_PARAMETERS, kappa_positive_per_v2=reference)
            for multiplier in MULTIPLIERS:
                parameters = dict(EXPECTED_PARAMETERS)
                parameters["kappa_positive_per_v2"] = multiplier*reference
                require(math.isfinite(parameters["kappa_positive_per_v2"]), "Nonfinite declared candidate")
                result["candidates"].append({"label": f"q1_kappa_reference_x{multiplier}",
                    "fixed_recovery_order": 1, "multiplier": multiplier, "parameters": parameters,
                    "initial_states": EXPECTED_ENTRIES,
                    "only_parameter_replacement": "kappa_positive_per_v2",
                    "capture_and_recovery_in_both_polarities_scaled_together": True,
                    "forward_evaluated": False})
                result["counts"]["candidate_declarations"] += 1
        finally:
            numerical_wall += time.monotonic()-started
        require(numerical_wall <= 30, "Numerical arithmetic budget exceeded; no validated seed declaration")
        result["status"] = "completed_reference_seed_declaration_no_forward_evaluation"
    except Exception as exc:
        result.update(status="failed_partial_seed_derivation_no_retry", error=f"{type(exc).__name__}: {exc}")
    finally:
        result["numerical_wall_s"] = numerical_wall
        result["input_sha256_before"] = before
        result["input_sha256_after"] = {path: sha(path) for path in before}
        result["inputs_unchanged"] = result["input_sha256_after"] == before
        if not result["inputs_unchanged"]:
            result["status"] = "input_integrity_failure_no_validated_seed_declaration"
        destination.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        write_report(result)
    print(json.dumps({"status": result["status"], "counts": result["counts"],
                      "kappa1_reference_per_v2": result.get("kappa1_reference_per_v2"),
                      "inputs_unchanged": result["inputs_unchanged"]}, indent=2))


if __name__ == "__main__":
    main()
