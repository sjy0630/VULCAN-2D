"""One declared common-law probe, not a joint fit or a mechanism selection scan.

This demonstrates both circuit embeddings at the same parameters, then checks
why a positive fixed field threshold plus ideal compliance cannot develop the
historical bare1R hard conductance. The counterexample is conditional on these
chosen thresholds and this rate family; it does not exclude all field models.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vulcan2d.shared_hbn import SharedHbnLaw, simulate_1r, simulate_1t1r
from vulcan2d.transistor import measured_transistor_mapping


def run():
    # Transport scales and threshold are historical bare1R reference values.
    # Rates/acceleration below are DECLARED uncalibrated probe parameters.
    law = SharedHbnLaw(
        leakage_scale_a=9.99e-11, leakage_alpha_per_v=3.315,
        on_conductance_s=1./96., weights=(1.,),
        set_thresholds_v=(2.675,), reset_thresholds_v=(.55,),
        set_rate_per_v=2., reset_rate_per_v=2., acceleration_per_v=4.)
    forward = np.linspace(0., 5., 51)
    voltage = np.r_[forward, forward[-2::-1]]
    limit = 1e-3
    mapping = measured_transistor_mapping()
    traces, rows = {}, {}
    for topology in ("1R", "1T1R"):
        # Tighten adaptive tolerances as well as the step cap. A cap-only
        # comparison may leave actual accepted steps and error unchanged.
        values = []
        settings = ((.02, 1e-8, 1e-5), (.01, 1e-9, 1e-6),
                    (.005, 1e-10, 1e-7))
        for step, atol, rtol in settings:
            if topology == "1R":
                result = simulate_1r(voltage, [0.], law, compliance_a=limit,
                                     max_step_v=step, state_atol=atol, state_rtol=rtol)
            else:
                result = simulate_1t1r(voltage, [0.], law, mapping, 1.1,
                                       max_step_v=step, state_atol=atol, state_rtol=rtol)
            values.append(result)
        result = values[-1]
        conductivity = law.leakage_scale_a*law.leakage_alpha_per_v + (
            law.on_conductance_s * np.dot(law.weights, result["final_state"]))
        intrinsic_residual = max(abs(i-law.current(v, state))
                                 for i, v, state in zip(result["current_a"],
                                     result["hbn_voltage_v"], result["state"]))
        kcl = (0. if topology == "1R" else max(abs(i-mapping.drain_current(v,1.1))
               for i,v in zip(result["current_a"],result["transistor_voltage_v"])))
        rows[topology] = {
            "initial_state": [0.], "same_material_parameters": True,
            "external_gate_V": 1.1 if topology == "1T1R" else None,
            "external_instrument_compliance_A": limit if topology == "1R" else None,
            "final_state": result["final_state"].tolist(),
            "peak_current_A": float(np.max(abs(result["current_a"]))),
            "peak_internal_voltage_V": float(np.max(abs(result["hbn_voltage_v"]))),
            "post_SET_intrinsic_small_signal_R_ohm": float(1./conductivity),
            "max_intrinsic_current_residual_A": float(intrinsic_residual),
            "max_transistor_KCL_residual_A": float(kcl),
            "max_command_voltage_closure_residual_V": float(np.max(abs(
                result["command_voltage_v"]-result["hbn_voltage_v"]
                -result["transistor_voltage_v"]-result["compliance_drop_v"]))),
            "refinement": [{"max_step_V": step,
                "state_atol": atol, "state_rtol": rtol,
                "accepted_substeps": value["n_substeps"],
                "rejected_substeps": value["n_rejected_substeps"],
                "max_current_difference_from_finest_A": float(np.max(abs(
                    value["current_a"]-result["current_a"]))),
                "max_state_difference_from_finest": float(np.max(abs(
                    value["state"]-result["state"])))}
                for (step,atol,rtol),value in zip(settings,values)],
        }
        state_error = float(np.max(abs(values[-2]["state"] - result["state"])))
        current_error = float(np.max(abs(values[-2]["current_a"]-result["current_a"])))
        rows[topology]["last_refinement_check"] = {
            "max_state_change": state_error, "max_current_change_A": current_error,
            "state_tolerance": 2e-4, "current_tolerance_A": 1e-5,
            "passed": state_error <= 2e-4 and current_error <= 1e-5,
        }
        for field in ("command_voltage_v", "hbn_voltage_v", "transistor_voltage_v",
                      "supply_voltage_v", "compliance_drop_v", "current_a", "state"):
            traces[f"{topology}_{field}"] = result[field]
        if max(intrinsic_residual, kcl) > 1e-11:
            raise AssertionError("Constitutive or transistor continuity failed")
    # If z >= Icomp/(G_on*Vth), passivity plus compliance implies Vh <= Vth,
    # so the positive-threshold growth rate vanishes in the continuous limit.
    cap = limit/(law.on_conductance_s*law.set_thresholds_v[0])
    # The leakage branch makes this bound conservative (possibly stricter).
    report = {
        "status": "uncalibrated_single_common_law_probe_not_a_joint_fit",
        "integration": result["integration"],
        "supersedes": "Earlier midpoint-predictor outputs gave false zero 1T1R growth; those numerical outputs are invalid, not physical evidence.",
        "law": asdict(law), "records": rows,
        "conditional_bound": {
            "assumptions": ["one connectivity initially below the bound",
                "fixed positive internal threshold 2.675V",
                "no growth below threshold", "ideal instantaneous1mA compliance",
                "passive additive leakage and G_on*z*V conduction",
                "continuous command-path evolution; numerical overshoot must refine away"],
            "connectivity_upper_bound": cap,
            "conductive_branch_resistance_lower_bound_ohm": law.set_thresholds_v[0]/limit,
            "historical_bare_R_reference_ohm": 96.,
            "can_reach_hard_reference_under_these_assumptions": False,
        },
        "conclusion": "The common implementation is circuit-consistent but this declared threshold-only candidate does not reproduce the historical bare1R hard state. Do not replace the default calibrated model with it.",
        "limitations": [
            "No raw bare1R workbook was read or fitted;96ohm is an archived reference.",
            "Nature SI Fig6 reports1kohm LRS;96ohm has not been mapped to that exact example. The paper supports shared nominal Au-Ti-hBN-W geometry, not identical states of the measured cells.",
            "Bare forming and conditioned1T1R initial states need not be equal experimentally. Equal zero state is a controlled counterfactual here.",
            "The observed2.675V forming crossing is not independently identified as an intrinsic rate cutoff; assigning it as such is an explicit hypothesis.",
            "No implication that a unique thermal, ionic or transient-overshoot mechanism has been proved.",
            "Path coordinate is command-voltage variation, not seconds; ideal compliance cannot simulate instrument response or overshoot.",
            "No new accuracy score or universal transfer claim is supported.",
            "Fine-step self-comparison is a numerical check only. Any failed final refinement is retained and must not be called certified.",
        ],
    }
    sources = [Path(__file__), ROOT/'vulcan2d/shared_hbn.py',ROOT/'vulcan2d/circuit.py',
               ROOT/'vulcan2d/transistor.py', ROOT/'vulcan2d/model_v4.py']
    sources += list((ROOT/'vulcan2d/data').glob('*transistor_lookup.json'))
    report["source_sha256"] = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sources}
    folder=ROOT/'analysis/shared_hbn_2026-11-03'
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'candidate_probe.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(folder/'candidate_probe.npz',**traces)
    print(json.dumps({"records":rows,"bound":report["conditional_bound"]},indent=2),flush=True)


if __name__ == '__main__':
    run()
