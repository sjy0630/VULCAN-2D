"""Validate RESET prediction using only the initial low-voltage read state."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

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
from .validate_v7 import _deterministic_thresholds, _quantiles


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "analysis" / "v8_read_initialized_reset_summary.json"
FIGURE_PATH = ROOT / "analysis" / "figures" / "16_vulcan_read_initialized_reset.png"
RESET_FILES = {
    0.9: "0.9V栅压RESET.xlsx",
    1.1: "1T1M擦除.xlsx",
    1.3: "1.3V栅压RESET.xlsx",
}
TRAIN_GATES = (0.9, 1.3)
HOLDOUT_GATE = 1.1


def _log_error(observed, predicted):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    good = (
        np.isfinite(observed) & np.isfinite(predicted)
        & (observed > 0.0) & (predicted > 0.0)
    )
    residual = np.log10(predicted[good]) - np.log10(observed[good])
    return {
        "n": int(len(residual)),
        "median_absolute_decade": float(np.median(np.abs(residual))),
        "rmse_decade": float(np.sqrt(np.mean(residual ** 2))),
        "median_signed_decade": float(np.median(residual)),
    }


def _group_ratio_medians(states):
    return {
        gate: float(np.median([
            np.log10(state.g_hrs_a / state.g_lrs_a)
            for state in states[gate]
        ]))
        for gate in TRAIN_GATES
    }


def _training_emax_cv(states, read_lrs):
    """Check whether adding reconstructed maximum field travels across groups."""
    out = {}
    for train_gate, test_gate in ((0.9, 1.3), (1.3, 0.9)):
        train_state = states[train_gate]
        test_state = states[test_gate]
        train_x = np.asarray([np.nanmax(s.field_mvc) for s in train_state])
        train_y = np.asarray([
            np.log10(s.g_hrs_a / s.g_lrs_a) for s in train_state
        ])
        test_x = np.asarray([np.nanmax(s.field_mvc) for s in test_state])
        x0 = float(np.median(train_x))
        variance = float(np.var(train_x - x0))
        slope = 0.0 if variance <= 1e-20 else float(
            np.cov(train_x - x0, train_y, ddof=0)[0, 1] / variance)
        slope = min(slope, 0.0)
        intercept = float(np.median(train_y - slope * (train_x - x0)))
        constant = float(np.median(train_y))
        actual = np.asarray([s.g_hrs_a for s in test_state])
        pred_constant = read_lrs[test_gate] * 10.0 ** constant
        pred_emax = read_lrs[test_gate] * 10.0 ** (
            intercept + slope * (test_x - x0))
        out[f"{train_gate:g}_to_{test_gate:g}"] = {
            "constant_ratio_rmse_decade": _log_error(
                actual, pred_constant)["rmse_decade"],
            "emax_ratio_rmse_decade": _log_error(
                actual, pred_emax)["rmse_decade"],
        }
    return out


def main():
    base_mapping = M7.transistor_mapping()
    reconstructed = {gate: [] for gate in RESET_FILES}
    raw_curves = {gate: [] for gate in RESET_FILES}
    for gate, filename in RESET_FILES.items():
        for sheet, voltage, current in load_reset_workbook(ROOT / filename):
            reconstructed[gate].append((
                sheet,
                reconstruct_reset(voltage, current, gate, base_mapping),
                (voltage, current),
            ))

    training_cycles = [cycle for gate in TRAIN_GATES
                       for _, cycle, _ in reconstructed[gate]]
    alpha_lrs, alpha_lrs_residual = fit_shared_alpha(training_cycles, "lrs")
    alpha_hrs, alpha_hrs_residual = fit_shared_alpha(training_cycles, "hrs")

    states = {gate: [] for gate in RESET_FILES}
    accepted_cycles = {gate: [] for gate in RESET_FILES}
    accepted_raw = {gate: [] for gate in RESET_FILES}
    rejected = {gate: [] for gate in RESET_FILES}
    for gate, cycles in reconstructed.items():
        for sheet, cycle, raw in cycles:
            try:
                state = identify_reset_state(cycle, alpha_lrs, alpha_hrs)
                read_lrs = M8.estimate_initial_lrs(
                    cycle, alpha_lrs=alpha_lrs)
            except ValueError as error:
                rejected[gate].append({"sheet": sheet, "reason": str(error)})
                continue
            states[gate].append(state)
            accepted_cycles[gate].append(cycle)
            accepted_raw[gate].append(raw)

    train_states = [state for gate in TRAIN_GATES for state in states[gate]]
    train_e50 = np.asarray([state.e50_mvc for state in train_states], dtype=float)
    train_e50 = train_e50[np.isfinite(train_e50)]
    e50 = float(np.median(train_e50))
    mad = float(np.median(np.abs(train_e50 - e50)))
    e50_sigma = max(1.4826 * mad, 0.02)
    train_width = np.asarray([
        state_width_from_levels(state.e90_mvc, state.e10_mvc)
        for state in train_states
    ])
    train_width = train_width[np.isfinite(train_width) & (train_width > 0.0)]
    width = float(np.median(train_width))

    p = replace(
        M8.Params(),
        reset_alpha_lrs_identified=alpha_lrs,
        reset_alpha_hrs_identified=alpha_hrs,
        reset_e50_mvc=e50,
        reset_width_mvc=width,
        reset_e50_sigma_mvc=e50_sigma,
    )
    p = M8.fit_endpoint_ratio_params(
        gate_voltage=[gate for gate in TRAIN_GATES for _ in states[gate]],
        g_lrs=[state.g_lrs_a for gate in TRAIN_GATES for state in states[gate]],
        g_hrs=[state.g_hrs_a for gate in TRAIN_GATES for state in states[gate]],
        p=p,
    )
    mapping = M8.transistor_mapping(p)

    read_lrs = {
        gate: np.asarray([
            M8.estimate_initial_lrs(cycle, alpha_lrs=alpha_lrs)
            for cycle in accepted_cycles[gate]
        ])
        for gate in RESET_FILES
    }
    predicted_hrs = {
        gate: read_lrs[gate] * M8.predicted_hrs_ratio(gate, p)
        for gate in RESET_FILES
    }

    validation = {}
    simulated = {}
    for gate in RESET_FILES:
        thresholds = _deterministic_thresholds(
            len(states[gate]), e50, e50_sigma)
        model_cycles = []
        for raw, g_lrs, threshold in zip(
                accepted_raw[gate], read_lrs[gate], thresholds):
            model_cycles.append(M8.run_read_initialized_reset(
                raw[0], gate, g_lrs, p=p, mapping=mapping,
                e50_mvc=threshold, width_mvc=width,
            ))
        simulated[gate] = model_cycles
        validation[str(gate)] = {
            "identified_cycles": len(states[gate]),
            "rejected_cycles": rejected[gate],
            "initial_LRS_read_error": _log_error(
                [state.g_lrs_a for state in states[gate]], read_lrs[gate]),
            "predicted_HRS_error": _log_error(
                [state.g_hrs_a for state in states[gate]],
                predicted_hrs[gate]),
            "predicted_HRS_LRS_ratio": M8.predicted_hrs_ratio(gate, p),
            "read_initialized_terminal_rmse_decade": V5._loop_rmse(
                accepted_raw[gate], V5._as_curves(model_cycles),
                vmax={0.9: 1.5, 1.1: 1.7, 1.3: 1.5}[gate],
            ),
        }

    previous = json.loads(
        (ROOT / "analysis" / "v7_reset_identification_summary.json").read_text(
            encoding="utf-8"))
    report = {
        "model": "VULCAN-2D read-initialized RESET prediction",
        "data_split": {
            "training_gates_V": list(TRAIN_GATES),
            "held_out_gate_V": HOLDOUT_GATE,
            "holdout_policy": (
                "1.1-V labels are evaluated only after fitting transport, "
                "state and endpoint-ratio parameters on 0.9/1.3 V"
            ),
        },
        "identified_law": {
            "alpha_LRS_per_V": alpha_lrs,
            "alpha_HRS_per_V": alpha_hrs,
            "alpha_LRS_within_cycle_MAD_decade": alpha_lrs_residual,
            "alpha_HRS_within_cycle_MAD_decade": alpha_hrs_residual,
            "E50_MV_cm": e50,
            "E50_robust_sigma_MV_cm": e50_sigma,
            "logistic_width_MV_cm": width,
            "training_group_log10_HRS_LRS_medians": _group_ratio_medians(states),
            "log10_HRS_LRS_at_1p1V": p.reset_log10_ratio_ref,
            "log10_ratio_slope_per_gate_V": p.reset_log10_ratio_slope_per_v,
        },
        "model_selection": {
            "maximum_internal_field_candidate": _training_emax_cv(
                states, read_lrs),
            "decision": (
                "Rejected Emax-dependent endpoint ratio because it worsened "
                "both leave-one-training-gate-group-out tests; retained the "
                "two-anchor gate interpolation"
            ),
        },
        "validation": validation,
        "comparison": {
            "endpoint_conditioned_terminal_rmse_decade": {
                gate: previous["validation"][gate][
                    "endpoint_conditioned_terminal_rmse_decade"]
                for gate in ("0.9", "1.1", "1.3")
            },
            "v0.6_unconditioned_terminal_rmse_decade": previous[
                "comparison"]["v0.6_unconditioned_terminal_rmse_decade"],
        },
        "claim_scope": M8.claim_scope(p),
        "verdict": {
            "post_reset_endpoint_leakage_closed": True,
            "additional_experiment_required_for_this_upgrade": False,
            "holdout_gate": "1.1 V",
            "remaining_limit": (
                "The initial state still comes from a low-voltage electrical "
                "read and the HRS ratio is an empirical two-anchor interpolation"
            ),
        },
    }
    SUMMARY_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if plt is not None:
        gate = HOLDOUT_GATE
        actual_lrs = np.asarray([state.g_lrs_a for state in states[gate]])
        actual_hrs = np.asarray([state.g_hrs_a for state in states[gate]])
        fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.2))

        ax = axes[0, 0]
        ax.loglog(actual_lrs, read_lrs[gate], "o", ms=4, alpha=0.65)
        lo = min(np.min(actual_lrs), np.min(read_lrs[gate]))
        hi = max(np.max(actual_lrs), np.max(read_lrs[gate]))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2)
        ax.set_xlabel("Endpoint-identified $G_{LRS}$ (A)")
        ax.set_ylabel("Initial-read $G_{LRS}$ (A)")
        ax.set_title("(a) Initial 0.1-V segment sets the starting state")

        ax = axes[0, 1]
        ax.loglog(actual_hrs, predicted_hrs[gate], "o", ms=4, alpha=0.65,
                  color="#d95f02")
        lo = min(np.min(actual_hrs), np.min(predicted_hrs[gate]))
        hi = max(np.max(actual_hrs), np.max(predicted_hrs[gate]))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2)
        ax.set_xlabel("Endpoint-identified $G_{HRS}$ (A)")
        ax.set_ylabel("Predicted $G_{HRS}$ (A)")
        ax.set_title("(b) Final HRS is predicted, not read from return data")

        ax = axes[1, 0]
        grid = np.arange(0.0, 1.7 + 1e-9, 0.02)
        V5._median_plot(ax, accepted_raw[gate], grid, "measured 1.1 V",
                        x_sign=-1, color="black", lw=2.0)
        V5._median_plot(ax, V5._as_curves(simulated[gate]), grid,
                        "read-initialized prediction", x_sign=-1,
                        color="#d95f02", lw=1.8)
        ax.set_ylim(1e-11, 1e-4)
        ax.set_xlabel("Applied voltage (V)")
        ax.set_ylabel("|I| (A)")
        ax.set_title("(c) Held-out 1.1-V terminal RESET loop")
        ax.legend(fontsize=8)

        ax = axes[1, 1]
        labels = ["Old unconditioned", "Endpoint-conditioned",
                  "Read-initialized"]
        old = previous["comparison"][
            "v0.6_unconditioned_terminal_rmse_decade"]["1.1"]
        conditioned = previous["validation"]["1.1"][
            "endpoint_conditioned_terminal_rmse_decade"]
        blind = validation["1.1"]["read_initialized_terminal_rmse_decade"]
        forward = [old["forward_decade"], conditioned["forward_decade"],
                   blind["forward_decade"]]
        returned = [old["return_decade"], conditioned["return_decade"],
                    blind["return_decade"]]
        x = np.arange(len(labels))
        ax.bar(x - 0.18, forward, 0.36, label="forward")
        ax.bar(x + 0.18, returned, 0.36, label="return")
        ax.set_xticks(x, labels, rotation=12, ha="right")
        ax.set_ylabel("Median-loop RMSE (decade)")
        ax.set_title("(d) Accuracy versus information available to the model")
        ax.legend(fontsize=8)

        fig.suptitle("VULCAN-2D read-initialized RESET prediction",
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
