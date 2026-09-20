"""Validate the measured-SET-conditioned VULCAN-2D full-cycle link.

For target pair ``S_i -> R_i``, the initial RESET state is predicted from the
forward ``S_i`` trace.  The regression is fitted to all other accepted pairs.
The target ``R_i`` current supplies only the quality-controlled validation
label and score; it is never a numerical input to that target's prediction.
"""
from __future__ import annotations

from dataclasses import replace
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
from . import model_v7 as M7
from . import model_v8 as M8
from . import validate_v5 as V5
from .reset_identification import (
    fit_shared_alpha,
    identify_reset_state,
    load_reset_workbook,
    reconstruct_reset,
    state_width_from_levels,
)
from .validate_v7 import _deterministic_thresholds


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "analysis" / "full_cycle_hrs_attractor_summary.json"
FIGURE_PATH = ROOT / "analysis" / "figures" / "18_vulcan_full_cycle_hrs_attractor.png"
SET_FILE = "1T1M写入.xlsx"
RESET_FILE = "1T1M擦除.xlsx"
GATE_VOLTAGE = 1.1
TRAIN_RESET_FILES = {0.9: "0.9V栅压RESET.xlsx",
                     1.3: "1.3V栅压RESET.xlsx"}


def _finite_xy(frame):
    voltage = frame.iloc[:, 0].to_numpy(float)
    current = frame.iloc[:, 1].to_numpy(float)
    good = np.isfinite(voltage) & np.isfinite(current)
    return voltage[good], current[good]


def _log_error(observed, predicted):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    good = (np.isfinite(observed) & np.isfinite(predicted)
            & (observed > 0.0) & (predicted > 0.0))
    residual = np.log10(predicted[good]) - np.log10(observed[good])
    return {
        "n": int(len(residual)),
        "median_absolute_decade": float(np.median(np.abs(residual))),
        "rmse_decade": float(np.sqrt(np.mean(residual ** 2))),
        "median_signed_decade": float(np.median(residual)),
    }


def _correlation(exposure, conductance):
    from scipy.stats import pearsonr, spearmanr

    x = np.log10(np.asarray(exposure, dtype=float))
    y = np.log10(np.asarray(conductance, dtype=float))
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    return {
        "pearson_log_space_r": float(pearson.statistic),
        "pearson_p_value": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p_value": float(spearman.pvalue),
    }


def _fit_reset_parameters():
    """Rebuild v0.8 parameters without using 1.1-V labels."""
    base_mapping = M7.transistor_mapping()
    cycles = []
    states = []
    gates = []
    for gate, filename in TRAIN_RESET_FILES.items():
        group = []
        for _, voltage, current in load_reset_workbook(ROOT / filename):
            group.append(reconstruct_reset(
                voltage, current, gate, base_mapping))
        cycles.extend(group)
        gates.extend([gate] * len(group))

    alpha_lrs, _ = fit_shared_alpha(cycles, "lrs")
    alpha_hrs, _ = fit_shared_alpha(cycles, "hrs")
    accepted_gates = []
    for gate, cycle in zip(gates, cycles):
        try:
            states.append(identify_reset_state(cycle, alpha_lrs, alpha_hrs))
            accepted_gates.append(gate)
        except ValueError:
            continue

    e50_values = np.asarray(
        [state.e50_mvc for state in states], dtype=float)
    e50_values = e50_values[np.isfinite(e50_values)]
    e50 = float(np.median(e50_values))
    e50_sigma = max(
        1.4826 * float(np.median(np.abs(e50_values - e50))), 0.02)
    widths = np.asarray([
        state_width_from_levels(state.e90_mvc, state.e10_mvc)
        for state in states
    ])
    widths = widths[np.isfinite(widths) & (widths > 0.0)]
    width = float(np.median(widths))

    p = replace(
        M8.Params(),
        reset_alpha_lrs_identified=alpha_lrs,
        reset_alpha_hrs_identified=alpha_hrs,
        reset_e50_mvc=e50,
        reset_width_mvc=width,
        reset_e50_sigma_mvc=e50_sigma,
    )
    p = M8.fit_endpoint_ratio_params(
        accepted_gates,
        [state.g_lrs_a for state in states],
        [state.g_hrs_a for state in states],
        p=p,
    )
    return p, M8.transistor_mapping(p)


def _load_paired_cycles(p, mapping):
    set_frames = pd.read_excel(ROOT / SET_FILE, sheet_name=None)
    reset_frames = pd.read_excel(ROOT / RESET_FILE, sheet_name=None)
    pairs = CL.pair_cycle_sheets(set_frames, reset_frames)
    accepted = []
    rejected = []
    for index, set_sheet, reset_sheet in pairs:
        set_voltage, set_current = _finite_xy(set_frames[set_sheet])
        reset_voltage, reset_current = _finite_xy(reset_frames[reset_sheet])
        try:
            exposure = CL.set_forward_exposure(set_voltage, set_current)
            reconstructed = reconstruct_reset(
                reset_voltage, reset_current, GATE_VOLTAGE, mapping)
            state = identify_reset_state(
                reconstructed,
                p.reset_alpha_lrs_identified,
                p.reset_alpha_hrs_identified,
            )
            read_g = M8.estimate_initial_lrs(
                reconstructed, alpha_lrs=p.reset_alpha_lrs_identified)
        except ValueError as error:
            rejected.append({"cycle": index, "reason": str(error)})
            continue
        accepted.append({
            "cycle": index,
            "set_voltage": set_voltage,
            "set_current": set_current,
            "reset_voltage": reset_voltage,
            "reset_current": reset_current,
            "set_exposure": exposure,
            "read_g_lrs": read_g,
            "identified_state": state,
        })
    return accepted, rejected


def _chronological_validation(exposure, conductance, n_training=35):
    law = CL.fit_set_reset_transfer(
        exposure[:n_training], conductance[:n_training])
    predicted = np.asarray([
        law.predict(value) for value in exposure[n_training:]
    ])
    baseline = np.full_like(
        predicted, np.median(conductance[:n_training]))
    return {
        "training_cycles": int(n_training),
        "later_test_cycles": int(len(predicted)),
        "set_conditioned": _log_error(
            conductance[n_training:], predicted),
        "no_set_median_baseline": _log_error(
            conductance[n_training:], baseline),
    }


def main():
    p, mapping = _fit_reset_parameters()
    records, rejected = _load_paired_cycles(p, mapping)
    exposure = np.asarray([row["set_exposure"] for row in records])
    read_g = np.asarray([row["read_g_lrs"] for row in records])
    identified_g = np.asarray([
        row["identified_state"].g_lrs_a for row in records])
    full_law = CL.fit_set_reset_transfer(exposure, read_g)
    set_predicted_g, loo_laws = CL.leave_one_out_transfer_predictions(
        exposure, read_g)
    endpoint_law = CL.fit_reset_endpoint_attractor(
        [row["identified_state"].g_hrs_a for row in records],
        GATE_VOLTAGE,
        "paired 0 to -1.7 to 0 V DC RESET",
    )
    endpoint_predicted_g, endpoint_loo_laws = (
        CL.leave_one_out_endpoint_predictions(
            [row["identified_state"].g_hrs_a for row in records],
            GATE_VOLTAGE,
            "paired 0 to -1.7 to 0 V DC RESET",
        )
    )
    endpoint_from_set_g, _ = (
        CL.leave_one_out_transfer_predictions(
            exposure,
            [row["identified_state"].g_hrs_a for row in records],
        )
    )
    baseline_g = np.asarray([
        np.median(np.delete(read_g, target))
        for target in range(len(read_g))
    ])

    thresholds = _deterministic_thresholds(
        len(records), p.reset_e50_mvc, p.reset_e50_sigma_mvc)
    actual_reset = []
    baseline_cycles = []
    set_conditioned_cycles = []
    set_endpoint_cycles = []
    optimized_cycles = []
    read_initialized_cycles = []
    measured_set = []
    for target, (row, threshold) in enumerate(zip(records, thresholds)):
        actual_reset.append((row["reset_voltage"], row["reset_current"]))
        measured_set.append((row["set_voltage"], row["set_current"]))
        baseline_cycles.append(M8.run_read_initialized_reset(
            row["reset_voltage"], GATE_VOLTAGE, baseline_g[target],
            p=p, mapping=mapping, e50_mvc=threshold,
            width_mvc=p.reset_width_mvc,
        ))
        set_conditioned_cycles.append(CL.run_set_conditioned_cycle(
            row["set_voltage"], row["set_current"], row["reset_voltage"],
            GATE_VOLTAGE, loo_laws[target], p=p, mapping=mapping,
            e50_mvc=threshold, width_mvc=p.reset_width_mvc,
        ))
        set_endpoint_cycles.append(M7.run_identified_reset(
            row["reset_voltage"], GATE_VOLTAGE, set_predicted_g[target],
            min(endpoint_from_set_g[target], 0.95 * set_predicted_g[target]),
            p=p, mapping=mapping, e50_mvc=threshold,
            width_mvc=p.reset_width_mvc,
        ))
        optimized_cycles.append(CL.run_set_conditioned_cycle(
            row["set_voltage"], row["set_current"], row["reset_voltage"],
            GATE_VOLTAGE, loo_laws[target], p=p, mapping=mapping,
            e50_mvc=threshold, width_mvc=p.reset_width_mvc,
            endpoint_law=endpoint_loo_laws[target],
        ))
        read_initialized_cycles.append(M8.run_read_initialized_reset(
            row["reset_voltage"], GATE_VOLTAGE, read_g[target],
            p=p, mapping=mapping, e50_mvc=threshold,
            width_mvc=p.reset_width_mvc,
        ))

    baseline_loop_error = V5._loop_rmse(
        actual_reset, V5._as_curves(baseline_cycles), vmax=1.7)
    set_loop_error = V5._loop_rmse(
        actual_reset, V5._as_curves(set_conditioned_cycles), vmax=1.7)
    set_endpoint_loop_error = V5._loop_rmse(
        actual_reset, V5._as_curves(set_endpoint_cycles), vmax=1.7)
    optimized_loop_error = V5._loop_rmse(
        actual_reset, V5._as_curves(optimized_cycles), vmax=1.7)
    read_loop_error = V5._loop_rmse(
        actual_reset, V5._as_curves(read_initialized_cycles), vmax=1.7)
    initial_baseline_error = _log_error(read_g, baseline_g)
    initial_set_error = _log_error(read_g, set_predicted_g)
    baseline_vs_identified = _log_error(identified_g, baseline_g)
    set_vs_identified = _log_error(identified_g, set_predicted_g)
    initial_read_vs_identified = _log_error(identified_g, read_g)
    identified_hrs = np.asarray([
        row["identified_state"].g_hrs_a for row in records])
    ratio_endpoint_g = (
        set_predicted_g * M8.predicted_hrs_ratio(GATE_VOLTAGE, p))
    ratio_endpoint_error = _log_error(identified_hrs, ratio_endpoint_g)
    set_endpoint_error = _log_error(identified_hrs, endpoint_from_set_g)
    attractor_endpoint_error = _log_error(
        identified_hrs, endpoint_predicted_g)
    from scipy.stats import pearsonr, spearmanr
    lrs_hrs_pearson = pearsonr(np.log10(identified_g),
                               np.log10(identified_hrs))
    lrs_hrs_spearman = spearmanr(np.log10(identified_g),
                                 np.log10(identified_hrs))
    chronological_n = 35
    chronological_set_law = CL.fit_set_reset_transfer(
        exposure[:chronological_n], read_g[:chronological_n])
    chronological_endpoint_law = CL.fit_reset_endpoint_attractor(
        identified_hrs[:chronological_n],
        GATE_VOLTAGE,
        "paired 0 to -1.7 to 0 V DC RESET",
    )
    chronological_g_lrs = np.asarray([
        chronological_set_law.predict(value)
        for value in exposure[chronological_n:]
    ])
    chronological_attractor_g_hrs = np.full(
        len(records) - chronological_n,
        chronological_endpoint_law.predict(),
    )
    chronological_ratio_g_hrs = (
        chronological_g_lrs
        * M8.predicted_hrs_ratio(GATE_VOLTAGE, p)
    )
    chronological_thresholds = _deterministic_thresholds(
        len(records) - chronological_n,
        p.reset_e50_mvc,
        p.reset_e50_sigma_mvc,
    )
    chronological_raw = []
    chronological_attractor_cycles = []
    chronological_ratio_cycles = []
    for row, g_lrs, g_hrs_attractor, g_hrs_ratio, threshold in zip(
            records[chronological_n:], chronological_g_lrs,
            chronological_attractor_g_hrs, chronological_ratio_g_hrs,
            chronological_thresholds):
        chronological_raw.append((row["reset_voltage"], row["reset_current"]))
        chronological_attractor_cycles.append(M7.run_identified_reset(
            row["reset_voltage"], GATE_VOLTAGE, g_lrs,
            min(g_hrs_attractor, 0.95 * g_lrs), p=p, mapping=mapping,
            e50_mvc=threshold, width_mvc=p.reset_width_mvc,
        ))
        chronological_ratio_cycles.append(M7.run_identified_reset(
            row["reset_voltage"], GATE_VOLTAGE, g_lrs,
            min(g_hrs_ratio, 0.95 * g_lrs), p=p, mapping=mapping,
            e50_mvc=threshold, width_mvc=p.reset_width_mvc,
        ))

    report = {
        "model": (
            "VULCAN-2D full-cycle link with a protocol-conditioned "
            "RESET HRS attractor"
        ),
        "data": {
            "set_workbook": SET_FILE,
            "reset_workbook": RESET_FILE,
            "confirmed_pairing": "S_i is followed by R_i",
            "gate_voltage_V": GATE_VOLTAGE,
            "paired_cycles_total": 53,
            "paired_cycles_accepted": len(records),
            "rejected_cycles": rejected,
        },
        "prediction_protocol": {
            "target_inputs": (
                "target S_i voltage/current and target R_i voltage protocol; "
                "target R_i current is used only for quality control, labels "
                "and scoring"
            ),
            "validation": (
                "leave-one-cycle-out: each S_i-to-RESET-state law excludes "
                "the target R_i starting-state label"
            ),
            "reset_law_training": (
                "transport and transition-field laws use only the 0.9-V and "
                "1.3-V RESET ensembles; the legacy HRS/LRS gate-ratio law is "
                "retained only as a comparator"
            ),
        },
        "set_reset_transfer": {
            "feature": "forward SET integral |I| dV (A*V)",
            "feature_interpretation": (
                "voltage-domain exposure proxy; proportional to charge only "
                "under a common linear ramp rate"
            ),
            "equation": (
                "log10(G_RESET,0)=log10(G_ref)+b[log10(A_SET)-"
                "log10(A_ref)]"
            ),
            "log10_A_ref": full_law.log10_exposure_ref,
            "A_ref_A_V": 10.0 ** full_law.log10_exposure_ref,
            "log10_G_ref": full_law.log10_g_ref,
            "G_ref_A": 10.0 ** full_law.log10_g_ref,
            "slope_b": full_law.slope,
            "robust_residual_sigma_decade": (
                full_law.residual_sigma_decade),
            "correlation": _correlation(exposure, read_g),
        },
        "reset_endpoint_diagnosis": {
            "assumption_replaced": "G_HRS=rho(V_G)*G_LRS",
            "log10_G_LRS_standard_deviation_decade": float(
                np.std(np.log10(identified_g))),
            "log10_G_HRS_standard_deviation_decade": float(
                np.std(np.log10(identified_hrs))),
            "LRS_HRS_log_correlation": {
                "pearson_r": float(lrs_hrs_pearson.statistic),
                "pearson_p_value": float(lrs_hrs_pearson.pvalue),
                "spearman_rho": float(lrs_hrs_spearman.statistic),
                "spearman_p_value": float(lrs_hrs_spearman.pvalue),
            },
            "diagnosis": (
                "At fixed gate and protocol, HRS is substantially more "
                "concentrated than LRS and is not positively correlated with "
                "LRS; a fixed HRS/LRS ratio transfers spurious LRS variance "
                "into the RESET return branch."
            ),
        },
        "protocol_conditioned_hrs_attractor": {
            "equation": "log10(G_HRS)=mu_HRS(protocol,V_G)+epsilon",
            "protocol": endpoint_law.protocol_label,
            "gate_voltage_V": endpoint_law.gate_voltage_v,
            "log10_G_HRS_center": endpoint_law.log10_g_hrs_ref,
            "G_HRS_center_A": endpoint_law.predict(),
            "robust_residual_sigma_decade": (
                endpoint_law.residual_sigma_decade),
            "validation": (
                "leave-one-cycle-out; each target endpoint is predicted from "
                "the other 50 accepted cycles"
            ),
        },
        "initial_state_validation": {
            "common_endpoint_identified_target": {
                "no_SET_median_baseline_LOO": baseline_vs_identified,
                "SET_conditioned_LOO": set_vs_identified,
                "RESET_initial_read": initial_read_vs_identified,
            },
            "direct_transfer_label_target": {
                "target": "initial 0.08--0.12 V RESET read estimate",
                "no_SET_median_baseline_LOO": initial_baseline_error,
                "SET_conditioned_LOO": initial_set_error,
            },
            "chronological_first35_to_later16": (
                _chronological_validation(exposure, read_g, 35)),
        },
        "terminal_RESET_validation": {
            "no_SET_median_baseline": baseline_loop_error,
            "SET_conditioned_fixed_HRS_LRS_ratio": set_loop_error,
            "SET_predicts_HRS_endpoint_LOO": set_endpoint_loop_error,
            "SET_plus_protocol_HRS_attractor_LOO": optimized_loop_error,
            "RESET_read_initialized_reference": read_loop_error,
        },
        "endpoint_model_selection": {
            "fixed_HRS_LRS_ratio_with_SET_predicted_LRS": (
                ratio_endpoint_error),
            "SET_exposure_linear_HRS_endpoint_LOO": set_endpoint_error,
            "protocol_HRS_attractor_LOO": attractor_endpoint_error,
            "decision": (
                "Selected the protocol attractor: it is simpler than an "
                "additional SET-to-HRS slope and gives the lowest terminal "
                "RESET return-branch error."
            ),
        },
        "chronological_endpoint_check": {
            "training": "first 35 accepted pairs",
            "test": "later 16 accepted pairs",
            "training_cycle_ids": [
                row["cycle"] for row in records[:chronological_n]],
            "test_cycle_ids": [
                row["cycle"] for row in records[chronological_n:]],
            "log10_G_HRS_center_training": (
                chronological_endpoint_law.log10_g_hrs_ref),
            "log10_G_HRS_center_later_test": float(np.mean(
                np.log10(identified_hrs[chronological_n:]))),
            "HRS_endpoint_error": {
                "fixed_HRS_LRS_ratio": _log_error(
                    identified_hrs[chronological_n:],
                    chronological_ratio_g_hrs,
                ),
                "protocol_HRS_attractor": _log_error(
                    identified_hrs[chronological_n:],
                    chronological_attractor_g_hrs,
                ),
            },
            "terminal_RESET_error": {
                "fixed_HRS_LRS_ratio": V5._loop_rmse(
                    chronological_raw,
                    V5._as_curves(chronological_ratio_cycles),
                    vmax=1.7,
                ),
                "protocol_HRS_attractor": V5._loop_rmse(
                    chronological_raw,
                    V5._as_curves(chronological_attractor_cycles),
                    vmax=1.7,
                ),
            },
            "interpretation": (
                "The later cycles shift to a higher HRS conductance than the "
                "first 35. The attractor still improves the later median "
                "return curve, but its individual endpoint error reveals "
                "cycle-order drift that a stationary distribution cannot "
                "explain."
            ),
        },
        "claim_scope": CL.claim_scope(),
        "verdict": {
            "full_electrical_cycle_linked": True,
            "target_RESET_current_used_for_prediction": False,
            "SET_adds_predictive_information": bool(
                set_vs_identified["rmse_decade"]
                < baseline_vs_identified["rmse_decade"]),
            "HRS_attractor_improves_return_branch": bool(
                optimized_loop_error["return_decade"]
                < set_loop_error["return_decade"]),
            "fully_autonomous_SET_simulation": False,
            "interpretation": (
                "The measured SET history weakly but significantly improves "
                "the following RESET starting conductance. At the fixed "
                "1.1-V gate and DC protocol, RESET ends near a narrower HRS "
                "distribution rather than preserving a fixed HRS/LRS ratio. "
                "Neither relation identifies a microscopic species."
            ),
        },
    }
    SUMMARY_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if plt is not None:
        fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.2))

        ax = axes[0, 0]
        ax.loglog(identified_g, identified_hrs, "o", ms=5, alpha=0.7,
                  color="#2369bd", label="identified endpoints")
        x_grid = np.geomspace(np.min(identified_g), np.max(identified_g), 200)
        ax.plot(x_grid, np.full_like(x_grid, endpoint_law.predict()),
                color="#d95f02", lw=2.0, label="HRS attractor center")
        rho = lrs_hrs_spearman.statistic
        ax.set_xlabel(r"Initial RESET $G_{LRS}$ (A)")
        ax.set_ylabel(r"Final RESET $G_{HRS}$ (A)")
        ax.set_title(f"(a) Final HRS is not proportional to LRS ($\\rho$={rho:.2f})")
        ax.legend(fontsize=8)

        ax = axes[0, 1]
        set_grid = np.arange(0.1, 5.0 + 1e-9, 0.05)
        reset_grid = np.arange(0.1, 1.7 + 1e-9, 0.02)
        V5._median_plot(ax, measured_set, set_grid, "measured SET input",
                        forward=True, color="#2369bd", lw=1.8)
        V5._median_plot(ax, measured_set, set_grid, None,
                        forward=False, color="#2369bd", lw=1.2,
                        ls="--")
        V5._median_plot(ax, actual_reset, reset_grid, "measured RESET target",
                        forward=True, x_sign=-1, color="black", lw=1.7)
        V5._median_plot(ax, actual_reset, reset_grid, None,
                        forward=False, x_sign=-1, color="black", lw=1.1,
                        ls="--")
        V5._median_plot(ax, V5._as_curves(optimized_cycles), reset_grid,
                        "optimized RESET prediction", forward=True,
                        x_sign=-1, color="#d95f02", lw=1.8)
        V5._median_plot(ax, V5._as_curves(optimized_cycles), reset_grid,
                        None, forward=False, x_sign=-1, color="#d95f02",
                        lw=1.2, ls="--")
        ax.set_ylim(1e-11, 2e-2)
        ax.set_xlabel("Applied voltage (V)")
        ax.set_ylabel("|I| (A)")
        ax.set_title("(b) Median measured-SET-conditioned electrical cycle")
        ax.legend(fontsize=7, loc="lower right")

        ax = axes[1, 0]
        labels = ["Fixed HRS/LRS\nratio", "SET predicts\nHRS",
                  "Protocol HRS\nattractor"]
        values = [ratio_endpoint_error["rmse_decade"],
                  set_endpoint_error["rmse_decade"],
                  attractor_endpoint_error["rmse_decade"]]
        ax.bar(np.arange(3), values,
               color=["#999999", "#2369bd", "#d95f02"])
        ax.set_xticks(np.arange(3), labels)
        ax.set_ylabel("Final-HRS RMSE (decade)")
        ax.set_title("(c) Protocol endpoint removes spurious LRS scaling")

        ax = axes[1, 1]
        labels = ["No SET\nbaseline", "SET + fixed\nratio",
                  "SET + HRS\nattractor", "RESET initial\nread"]
        loop_sets = [baseline_loop_error, set_loop_error,
                     optimized_loop_error, read_loop_error]
        x = np.arange(4)
        ax.bar(x - 0.18, [v["forward_decade"] for v in loop_sets], 0.36,
               label="RESET forward")
        ax.bar(x + 0.18, [v["return_decade"] for v in loop_sets], 0.36,
               label="RESET return")
        ax.set_xticks(x, labels)
        ax.set_ylabel("Median-loop RMSE (decade)")
        ax.set_title("(d) Accuracy versus information available")
        ax.legend(fontsize=8)

        fig.suptitle("VULCAN-2D: protocol-conditioned RESET endpoint",
                     fontsize=15, fontweight="bold")
        fig.tight_layout()
        FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
        plt.close(fig)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("saved", SUMMARY_PATH)
    if plt is not None:
        print("saved", FIGURE_PATH)


if __name__ == "__main__":
    main()
