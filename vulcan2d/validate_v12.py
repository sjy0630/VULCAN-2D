"""Strict voltage-only validation of the autonomous VULCAN-2D loop.

For each accepted target pair, every law is refitted without that pair.  The
target contributes only its voltage grids to the simulator; both target
current traces are held back until scoring.  A chronological first-35/later
test is reported separately.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

from . import cycle_link as CL
from . import hrs_transport as HT
from . import model_v12 as M12
from . import validate_full_cycle as VF
from . import validate_v5 as V5
from .reset_identification import (
    fit_pf_reset_transition_law,
    reconstruct_reset,
)
from .set_identification import (
    estimate_set_amplitudes,
    fit_set_transition_law,
    fit_set_transport_law,
    reconstruct_set,
)


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "analysis" / "full_cycle_autonomous_summary.json"
FIGURE_PATH = ROOT / "analysis" / "figures" / "20_vulcan_autonomous_loop.png"
SET_PROTOCOL = "0 to +5 to 0 V DC SET at Vg=1.1 V"
RESET_PROTOCOL = "0 to -1.7 to 0 V DC RESET at Vg=1.1 V"


def _finite_xy(frame):
    voltage = frame.iloc[:, 0].to_numpy(float)
    current = frame.iloc[:, 1].to_numpy(float)
    valid = np.isfinite(voltage) & np.isfinite(current)
    return voltage[valid], current[valid]


def _load_set_cycles():
    frames = pd.read_excel(ROOT / VF.SET_FILE, sheet_name=None)
    named = []
    indexed = {}
    for name, frame in frames.items():
        voltage, current = _finite_xy(frame)
        cycle = reconstruct_set(voltage, current, VF.GATE_VOLTAGE)
        index = int(str(name).upper().replace("S", ""))
        named.append((str(name), cycle))
        indexed[index] = cycle
    return named, indexed


def _reconstruct_resets(records, mapping):
    return [
        reconstruct_reset(
            row["reset_voltage"], row["reset_current"],
            VF.GATE_VOLTAGE, mapping)
        for row in records
    ]


def _fit_bundle(named_set, reset_records, reset_cycles, base_params):
    if len(named_set) < 4 or len(reset_records) < 4:
        raise ValueError("Training bundle needs at least four complete cycles")
    set_transport = fit_set_transport_law(
        [cycle for _, cycle in named_set],
        VF.GATE_VOLTAGE, SET_PROTOCOL)
    set_transition = fit_set_transition_law(
        named_set, set_transport, VF.GATE_VOLTAGE, SET_PROTOCOL)
    reset_transport = HT.fit_effective_pf_law(
        reset_cycles, VF.GATE_VOLTAGE, RESET_PROTOCOL)
    reset_transition = fit_pf_reset_transition_law(
        [(f"R{row['cycle']}", cycle)
         for row, cycle in zip(reset_records, reset_cycles)],
        base_params.reset_alpha_lrs_identified,
        reset_transport,
        VF.GATE_VOLTAGE,
        RESET_PROTOCOL,
    )
    reset_amplitudes = [
        HT.estimate_hrs_amplitude(cycle, reset_transport)
        for cycle in reset_cycles
    ]
    endpoint = CL.fit_reset_endpoint_attractor(
        reset_amplitudes, VF.GATE_VOLTAGE, RESET_PROTOCOL)
    by_index = {
        int(name.upper().replace("S", "")): cycle
        for name, cycle in named_set
    }
    paired_set_amplitudes = [
        estimate_set_amplitudes(
            by_index[row["cycle"]], set_transport)[1]
        for row in reset_records
    ]
    link = M12.fit_set_reset_link_law(
        paired_set_amplitudes,
        [row["read_g_lrs"] for row in reset_records],
        set_transport.connected_exponent,
        base_params.reset_alpha_lrs_identified,
        SET_PROTOCOL,
        RESET_PROTOCOL,
    )
    return {
        "set_transport": set_transport,
        "set_transition": set_transition,
        "reset_transport": reset_transport,
        "reset_transition": reset_transition,
        "endpoint": endpoint,
        "link": link,
    }


def _run(row, bundle, base_params, reset_mapping,
         reset_transition_mode="same_protocol"):
    if reset_transition_mode == "same_protocol":
        transition = bundle["reset_transition"]
    elif reset_transition_mode == "cross_gate_prior":
        transition = None
    else:
        raise ValueError("Unsupported RESET transition mode")
    return M12.run_autonomous_full_cycle(
        row["set_voltage"], row["reset_voltage"], VF.GATE_VOLTAGE,
        bundle["set_transport"], bundle["set_transition"], bundle["link"],
        bundle["endpoint"], bundle["reset_transport"],
        reset_base_params=base_params, reset_mapping=reset_mapping,
        reset_transition_law=transition,
    )


def _curve_error(actual_curves, predicted_curves, current_floor):
    residuals = []
    cycle_rmse = []
    for (_, observed), (_, predicted) in zip(
            actual_curves, predicted_curves):
        observed = np.abs(np.asarray(observed, dtype=float))
        predicted = np.abs(np.asarray(predicted, dtype=float))
        valid = (
            np.isfinite(observed) & np.isfinite(predicted)
            & (observed >= current_floor) & (predicted > 0.0)
        )
        local = np.log10(predicted[valid]) - np.log10(observed[valid])
        if len(local) == 0:
            continue
        residuals.extend(local)
        cycle_rmse.append(float(np.sqrt(np.mean(local ** 2))))
    residuals = np.asarray(residuals, dtype=float)
    cycle_rmse = np.asarray(cycle_rmse, dtype=float)
    return {
        "current_floor_A": float(current_floor),
        "n_points": int(len(residuals)),
        "pooled_rmse_decade": float(np.sqrt(np.mean(residuals ** 2))),
        "pooled_median_absolute_decade": float(np.median(np.abs(residuals))),
        "pooled_median_signed_decade": float(np.median(residuals)),
        "cycle_rmse_q10_median_q90_decade": [
            float(value) for value in np.quantile(cycle_rmse, (0.1, 0.5, 0.9))
        ],
    }


def _score(actual_set, predicted_set, actual_reset, predicted_reset):
    return {
        "SET_pointwise": _curve_error(
            actual_set, predicted_set, current_floor=1e-11),
        "RESET_pointwise": _curve_error(
            actual_reset, predicted_reset, current_floor=1e-12),
        "SET_median_loop_rmse_decade": V5._loop_rmse(
            actual_set, predicted_set, vmax=5.0),
        "RESET_median_loop_rmse_decade": V5._loop_rmse(
            actual_reset, predicted_reset, vmax=1.7),
    }


def _evaluate_loo(named_set, records, reset_cycles, base_params,
                  reset_mapping):
    actual_set, predicted_set = [], []
    actual_reset, predicted_reset, prior_reset = [], [], []
    predictions = []
    for target, row in enumerate(records):
        cycle_index = row["cycle"]
        training_set = [
            item for item in named_set
            if int(item[0].upper().replace("S", "")) != cycle_index
        ]
        training_records = [
            value for index, value in enumerate(records) if index != target
        ]
        training_resets = [
            value for index, value in enumerate(reset_cycles)
            if index != target
        ]
        bundle = _fit_bundle(
            training_set, training_records, training_resets, base_params)
        prediction = _run(row, bundle, base_params, reset_mapping)
        prior_prediction = _run(
            row, bundle, base_params, reset_mapping,
            reset_transition_mode="cross_gate_prior")
        predictions.append(prediction)
        actual_set.append((row["set_voltage"], row["set_current"]))
        predicted_set.append((prediction["set"]["Vs"],
                              prediction["set"]["Is"]))
        actual_reset.append((row["reset_voltage"], row["reset_current"]))
        predicted_reset.append((prediction["reset"]["Vs"],
                                prediction["reset"]["Is"]))
        prior_reset.append((prior_prediction["reset"]["Vs"],
                            prior_prediction["reset"]["Is"]))
        if (target + 1) % 10 == 0:
            print(f"autonomous LOO {target + 1}/{len(records)}")
    scores = {
        "same_protocol_transition": _score(
            actual_set, predicted_set, actual_reset, predicted_reset),
        "cross_gate_transition_prior": _score(
            actual_set, predicted_set, actual_reset, prior_reset),
    }
    return (scores,
            predictions, actual_set, predicted_set,
            actual_reset, predicted_reset)


def _evaluate_chronological(named_set, records, reset_cycles, base_params,
                            reset_mapping, last_training_cycle=35):
    training_set = [
        item for item in named_set
        if int(item[0].upper().replace("S", "")) <= last_training_cycle
    ]
    training_positions = [
        index for index, row in enumerate(records)
        if row["cycle"] <= last_training_cycle
    ]
    test_positions = [
        index for index, row in enumerate(records)
        if row["cycle"] > last_training_cycle
    ]
    training_records = [records[index] for index in training_positions]
    training_resets = [reset_cycles[index] for index in training_positions]
    bundle = _fit_bundle(
        training_set, training_records, training_resets, base_params)
    actual_set, predicted_set = [], []
    actual_reset, predicted_reset, prior_reset = [], [], []
    for index in test_positions:
        row = records[index]
        prediction = _run(row, bundle, base_params, reset_mapping)
        prior_prediction = _run(
            row, bundle, base_params, reset_mapping,
            reset_transition_mode="cross_gate_prior")
        actual_set.append((row["set_voltage"], row["set_current"]))
        predicted_set.append((prediction["set"]["Vs"],
                              prediction["set"]["Is"]))
        actual_reset.append((row["reset_voltage"], row["reset_current"]))
        predicted_reset.append((prediction["reset"]["Vs"],
                                prediction["reset"]["Is"]))
        prior_reset.append((prior_prediction["reset"]["Vs"],
                            prior_prediction["reset"]["Is"]))
    return {
        "last_training_cycle": int(last_training_cycle),
        "training_SET_cycles": int(len(training_set)),
        "training_paired_RESET_cycles": int(len(training_records)),
        "later_test_pairs": int(len(test_positions)),
        "same_protocol_transition": _score(
            actual_set, predicted_set, actual_reset, predicted_reset),
        "cross_gate_transition_prior": _score(
            actual_set, predicted_set, actual_reset, prior_reset),
    }


def _plot(actual_set, predicted_set, actual_reset, predicted_reset,
          example, report):
    if plt is None:
        return
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    for axis, actual, predicted, vmax, title in (
            (axes[0, 0], actual_set, predicted_set, 5.0,
             "SET: voltage-only leave-one-cycle-out"),
            (axes[0, 1], actual_reset, predicted_reset, 1.7,
             "RESET: voltage-only leave-one-cycle-out")):
        grid = np.arange(0.1, vmax + 1e-9, 0.02)
        for forward, linestyle in ((True, "-"), (False, "--")):
            measured, low, high = V5._median_branch(actual, grid, forward)
            modeled, _, _ = V5._median_branch(predicted, grid, forward)
            axis.fill_between(
                grid, low, high, color="#9aa0a6", alpha=0.20,
                label="measured 10-90%" if forward else None)
            axis.plot(grid, measured, color="#4b5563", linestyle=linestyle,
                      linewidth=1.8,
                      label="measured median" if forward else None)
            axis.plot(grid, modeled, color="#d97706", linestyle=linestyle,
                      linewidth=2.0,
                      label="autonomous median" if forward else None)
        axis.set_yscale("log")
        axis.set_xlabel("|Applied voltage| (V)")
        axis.set_ylabel("|Current| (A)")
        axis.set_title(title)
        axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)

    set_result = example["set"]
    reset_result = example["reset"]
    set_apex = int(np.argmax(np.abs(set_result["Vs"])))
    reset_apex = int(np.argmax(np.abs(reset_result["Vs"])))
    axes[1, 0].plot(
                    set_result["Vs"][:set_apex + 1],
                    set_result["xs"][:set_apex + 1],
                    color="#2563eb", label="SET connected fraction")
    axes[1, 0].plot(
                    np.abs(reset_result["Vs"][:reset_apex + 1]),
                    reset_result["pbs"][:reset_apex + 1],
                    color="#dc2626", label="RESET connected fraction")
    axes[1, 0].set_xlabel("|Applied voltage| (V)")
    axes[1, 0].set_ylabel("Observable connected fraction")
    axes[1, 0].set_ylim(-0.04, 1.04)
    axes[1, 0].set_title("Irreversible coarse-grained state")
    axes[1, 0].legend(frameon=False)
    axes[1, 0].grid(alpha=0.2)

    axes[1, 1].axis("off")
    selected_score = report["leave_one_cycle_out"][
        "same_protocol_transition"]
    set_error = selected_score[
        "SET_median_loop_rmse_decade"]
    reset_error = selected_score[
        "RESET_median_loop_rmse_decade"]
    parameter = report["full_group_calibration"]
    text = (
        "Strict target exclusion\n"
        "Target SET/RESET currents used only for scoring\n\n"
        f"SET median-loop RMSE: {set_error['forward_decade']:.3f} / "
        f"{set_error['return_decade']:.3f} decade\n"
        f"RESET median-loop RMSE: {reset_error['forward_decade']:.3f} / "
        f"{reset_error['return_decade']:.3f} decade\n\n"
        f"SET beta: {parameter['SET_pre_beta_per_sqrt_V']:.3f} V^-1/2\n"
        f"Connected exponent n: {parameter['SET_connected_exponent']:.3f}\n"
        f"SET exposure E50: {parameter['SET_E50_V2']:.3f} V^2\n"
        f"SET->RESET factor: {parameter['SET_RESET_retention_factor']:.3f}\n\n"
        "Effective laws; no unique microscopic assignment"
    )
    axes[1, 1].text(0.02, 0.98, text, va="top", fontsize=10.5,
                    linespacing=1.45)
    figure.suptitle("VULCAN-2D autonomous quasi-static SET-RESET loop",
                    fontsize=14, fontweight="bold")
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=220)
    plt.close(figure)


def main():
    base_params, reset_mapping = VF._fit_reset_parameters()
    records, rejected = VF._load_paired_cycles(base_params, reset_mapping)
    named_set, _ = _load_set_cycles()
    reset_cycles = _reconstruct_resets(records, reset_mapping)
    full = _fit_bundle(named_set, records, reset_cycles, base_params)
    (loo_score, predictions, actual_set, predicted_set,
     actual_reset, predicted_reset) = _evaluate_loo(
        named_set, records, reset_cycles, base_params, reset_mapping)
    chronological = _evaluate_chronological(
        named_set, records, reset_cycles, base_params, reset_mapping)

    report = {
        "model": "VULCAN-2D autonomous quasi-static voltage-domain loop",
        "version": "0.12.0",
        "data": {
            "SET_workbook": VF.SET_FILE,
            "RESET_workbook": VF.RESET_FILE,
            "SET_cycles": int(len(named_set)),
            "paired_RESET_cycles_accepted": int(len(records)),
            "rejected_RESET_cycles": rejected,
            "gate_voltage_V": VF.GATE_VOLTAGE,
        },
        "prediction_contract": {
            "runtime_inputs": (
                "SET voltage waveform, RESET voltage waveform, gate voltage"),
            "target_SET_current_used": False,
            "target_RESET_current_used": False,
            "validation": (
                "leave one complete target S_i/R_i pair out of every fitted "
                "SET transport, SET transition, SET-RESET link, RESET HRS "
                "transport, RESET transition and RESET endpoint law"),
            "scope": (
                "same 1.1-V gate and recorded 5-V SET/1.7-V RESET DC protocols"),
            "RESET_transition_modes": {
                "same_protocol_transition": (
                    "1.1-V PF-envelope observable-state law; best random "
                    "target-excluding accuracy"),
                "cross_gate_transition_prior": (
                    "legacy 0.9/1.3-V transition prior; better temporal "
                    "robustness across the observed late-cycle shift"),
            },
        },
        "Claude_handoff_corrections": [
            "interpolate the literal Fig. 2b lookup at Vg=1.1 V instead of "
            "reusing the v0.6 cross-gate affine endpoint correction",
            "fit separate pre-SET PF-like and connected power-law envelopes",
            "do not renormalize every identified state trajectory to 0 and 1",
            "label integral |V_hBN| dV_app as V^2 voltage exposure, not time",
            "score autonomous target-excluding current prediction rather than "
            "the current-derived latent-state reconstruction",
        ],
        "full_group_calibration": {
            "SET_pre_beta_per_sqrt_V": (
                full["set_transport"].beta_pre_per_sqrt_v),
            "SET_connected_exponent": (
                full["set_transport"].connected_exponent),
            "SET_pre_shape_RMSE_decade": (
                full["set_transport"].pre_shape_rmse_decade),
            "SET_connected_shape_RMSE_decade": (
                full["set_transport"].connected_shape_rmse_decade),
            "SET_E50_V2": full["set_transition"].e50_v2,
            "SET_width_V2": full["set_transition"].width_v2,
            "SET_state_identification_RMS": (
                full["set_transition"].residual_rms_state),
            "SET_RESET_retention_factor": full["link"].retention_factor,
            "SET_RESET_link_sigma_decade": (
                full["link"].residual_sigma_decade),
            "RESET_PF_beta_per_sqrt_V": (
                full["reset_transport"].beta_per_sqrt_v),
            "RESET_E50_MV_cm": full["reset_transition"].e50_mvc,
            "RESET_width_MV_cm": full["reset_transition"].width_mvc,
            "RESET_state_identification_RMS": (
                full["reset_transition"].residual_rms_state),
            "RESET_endpoint_sigma_decade": (
                full["endpoint"].residual_sigma_decade),
        },
        "leave_one_cycle_out": loo_score,
        "chronological_first35_to_later": chronological,
        "interpretation_boundary": M12.claim_scope(),
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _plot(actual_set, predicted_set, actual_reset, predicted_reset,
          predictions[len(predictions) // 2], report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote {SUMMARY_PATH}")
    if plt is not None:
        print(f"wrote {FIGURE_PATH}")


if __name__ == "__main__":
    main()
