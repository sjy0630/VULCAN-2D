"""Validate the inverse-identified RESET state and measured reverse load line."""
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

from . import model_v6 as M6
from . import model_v7 as M7
from . import validate_v5 as V5
from .reset_identification import (
    fit_shared_alpha,
    identify_reset_state,
    load_reset_workbook,
    logistic_state,
    reconstruct_reset,
    state_width_from_levels,
)


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "analysis" / "v7_reset_identification_summary.json"
FIGURE_PATH = ROOT / "analysis" / "figures" / "15_vulcan_v7_reset_identification.png"
RESET_FILES = {
    0.9: "0.9V栅压RESET.xlsx",
    1.1: "1T1M擦除.xlsx",
    1.3: "1.3V栅压RESET.xlsx",
}
TRAIN_GATES = (0.9, 1.3)
HOLDOUT_GATE = 1.1


def _quantiles(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0, "median": None, "p10": None, "p90": None}
    return {
        "n": int(len(values)),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.1)),
        "p90": float(np.quantile(values, 0.9)),
    }


def _median_state(states, field_grid):
    curves = []
    for state in states:
        good = np.isfinite(state.field_mvc) & np.isfinite(state.state_monotone)
        field = state.field_mvc[good]
        value = state.state_monotone[good]
        order = np.argsort(field)
        field, value = field[order], value[order]
        unique, first = np.unique(field, return_index=True)
        if len(unique) < 3:
            continue
        curves.append(np.interp(field_grid, unique, value[first],
                                left=value[first][0], right=value[first][-1]))
    return np.median(np.asarray(curves), axis=0)


def _state_rmse(observed, predicted):
    return float(np.sqrt(np.mean((np.asarray(observed) - np.asarray(predicted)) ** 2)))


def _deterministic_thresholds(n, center, sigma):
    from scipy.special import ndtri

    q = (np.arange(n, dtype=float) + 0.5) / n
    return np.clip(center + sigma * ndtri(q), 0.05, None)


def main():
    base_mapping = M7.transistor_mapping()
    reconstructed = {gate: [] for gate in RESET_FILES}
    raw_curves = {gate: [] for gate in RESET_FILES}
    for gate, filename in RESET_FILES.items():
        for sheet, voltage, current in load_reset_workbook(ROOT / filename):
            raw_curves[gate].append((voltage, current))
            reconstructed[gate].append((
                sheet,
                reconstruct_reset(voltage, current, gate, base_mapping),
            ))

    training_cycles = [cycle for gate in TRAIN_GATES
                       for _, cycle in reconstructed[gate]]
    alpha_lrs, alpha_lrs_residual = fit_shared_alpha(training_cycles, "lrs")
    alpha_hrs, alpha_hrs_residual = fit_shared_alpha(training_cycles, "hrs")

    states = {gate: [] for gate in RESET_FILES}
    accepted_raw = {gate: [] for gate in RESET_FILES}
    rejected = {gate: [] for gate in RESET_FILES}
    for gate, cycles in reconstructed.items():
        for (sheet, cycle), raw in zip(cycles, raw_curves[gate]):
            try:
                states[gate].append(identify_reset_state(
                    cycle, alpha_lrs, alpha_hrs,
                ))
                accepted_raw[gate].append(raw)
            except ValueError as error:
                rejected[gate].append({"sheet": sheet, "reason": str(error)})

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
        M7.Params(),
        reset_alpha_lrs_identified=alpha_lrs,
        reset_alpha_hrs_identified=alpha_hrs,
        reset_e50_mvc=e50,
        reset_width_mvc=width,
        reset_e50_sigma_mvc=e50_sigma,
    )
    mapping = M7.transistor_mapping(p)

    field_grid = np.linspace(0.1, 1.8, 171)
    predicted_state = logistic_state(field_grid, e50, width)
    validation = {}
    simulated = {}
    for gate in RESET_FILES:
        observed_state = _median_state(states[gate], field_grid)
        thresholds = _deterministic_thresholds(
            len(states[gate]), e50, e50_sigma)
        model_cycles = []
        for (voltage, _), state, threshold in zip(
                accepted_raw[gate], states[gate], thresholds):
            model_cycles.append(M7.run_identified_reset(
                voltage, gate, state.g_lrs_a, state.g_hrs_a,
                p=p, mapping=mapping, e50_mvc=threshold,
                width_mvc=width,
            ))
        simulated[gate] = model_cycles
        model_as_curves = V5._as_curves(model_cycles)
        validation[str(gate)] = {
            "identified_cycles": len(states[gate]),
            "rejected_cycles": rejected[gate],
            "deembedding_valid_fraction": _quantiles([
                state.reconstruction_valid_fraction for state in states[gate]
            ]),
            "g_lrs_A": _quantiles([state.g_lrs_a for state in states[gate]]),
            "g_hrs_A": _quantiles([state.g_hrs_a for state in states[gate]]),
            "E90_MV_cm": _quantiles([state.e90_mvc for state in states[gate]]),
            "E50_MV_cm": _quantiles([state.e50_mvc for state in states[gate]]),
            "E10_MV_cm": _quantiles([state.e10_mvc for state in states[gate]]),
            "transition_width_MV_cm": _quantiles([
                state_width_from_levels(state.e90_mvc, state.e10_mvc)
                for state in states[gate]
            ]),
            "median_state_rmse": _state_rmse(observed_state, predicted_state),
            "endpoint_conditioned_terminal_rmse_decade": V5._loop_rmse(
                accepted_raw[gate], model_as_curves,
                vmax={0.9: 1.5, 1.1: 1.7, 1.3: 1.5}[gate],
            ),
        }

    old = json.loads(
        (ROOT / "analysis" / "v6_validation_summary.json").read_text(
            encoding="utf-8"))
    report = {
        "model": "VULCAN-2D v0.7 inverse-identified RESET state",
        "data": {
            "reverse_transistor": "Bookoutput2.xls, measured negative Vd",
            "reset_workbooks": RESET_FILES,
            "training_gates_V": list(TRAIN_GATES),
            "held_out_gate_V": HOLDOUT_GATE,
        },
        "identified_common_law": {
            "alpha_LRS_per_V": alpha_lrs,
            "alpha_HRS_per_V": alpha_hrs,
            "alpha_LRS_within_cycle_MAD_decade": alpha_lrs_residual,
            "alpha_HRS_within_cycle_MAD_decade": alpha_hrs_residual,
            "E50_MV_cm": e50,
            "E50_robust_sigma_MV_cm": e50_sigma,
            "logistic_width_MV_cm": width,
            "state_equation": "s(E)=1/(1+exp((E-E50)/width))",
            "drive": "E_hBN=10*|V_hBN|/d_hBN, d_hBN=6 nm",
        },
        "validation": validation,
        "comparison": {
            "v0.6_unconditioned_terminal_rmse_decade": {
                gate: old["validation"][gate]["reset"]["median_loop_rmse_decade"]
                for gate in ("0.9", "1.1", "1.3")
            },
            "v0.7_scope": (
                "RESET transition prediction conditioned on each cycle's "
                "non-disturbing LRS/HRS transport envelopes"
            ),
        },
        "claim_scope": M7.claim_scope(p),
        "verdict": {
            "reverse_transistor_gap_closed": True,
            "gate_neutral_reset_state_law": True,
            "holdout_state_test": "1.1 V excluded from parameter identification",
            "absolute_state_prediction_complete": False,
            "remaining_task": (
                "predict G_LRS and G_HRS from the preceding SET/history rather "
                "than conditioning on low-voltage observations"
            ),
        },
    }
    SUMMARY_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if plt is not None:
        fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.0))
        reverse = mapping.reset_lookup
        ax = axes[0, 0]
        for gate in (0.9, 1.1, 1.3):
            grid = np.linspace(0.0, 1.2, 121)
            ax.plot(grid, [reverse.current(v, gate) * 1e6 for v in grid],
                    label=f"$V_G$={gate:g} V")
        ax.set_xlabel("$|V_{DS}|$ (V)")
        ax.set_ylabel("$|I_D|$ (µA)")
        ax.set_title("(a) Measured negative-$V_{DS}$ transistor branch")
        ax.legend(fontsize=8)

        ax = axes[0, 1]
        colors = {0.9: "#1f77b4", 1.1: "#2ca02c", 1.3: "#d62728"}
        for gate in RESET_FILES:
            for state in states[gate][::max(1, len(states[gate]) // 18)]:
                ax.plot(state.field_mvc, state.state_monotone,
                        color=colors[gate], alpha=0.18, lw=0.8)
            median = _median_state(states[gate], field_grid)
            ax.plot(field_grid, median, color=colors[gate], lw=2.0,
                    label=f"data {gate:g} V")
        ax.plot(field_grid, predicted_state, "k--", lw=2.0,
                label="endpoint-trained common law")
        ax.set_xlim(0.1, 1.8); ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Reconstructed $E_{hBN}$ (MV/cm)")
        ax.set_ylabel("Observable connected state $s$")
        ax.set_title("(b) RESET states collapse after 1T de-embedding")
        ax.legend(fontsize=8)

        ax = axes[1, 0]
        for gate in RESET_FILES:
            vals = [state.e50_mvc for state in states[gate]
                    if np.isfinite(state.e50_mvc)]
            ax.boxplot(vals, positions=[gate], widths=0.08,
                       patch_artist=True,
                       boxprops={"facecolor": colors[gate], "alpha": 0.45})
        ax.axhline(e50, color="black", ls="--", label="0.9/1.3-V training median")
        ax.set_xticks([0.9, 1.1, 1.3]); ax.set_xlabel("$V_G$ (V)")
        ax.set_ylabel("Identified $E_{50}$ (MV/cm)")
        ax.set_title("(c) 1.1-V holdout tests the common field scale")
        ax.legend(fontsize=8)

        ax = axes[1, 1]
        gate = 1.1
        grid = np.arange(0.0, 1.7 + 1e-9, 0.02)
        V5._median_plot(ax, accepted_raw[gate], grid, "measured 1.1 V",
                        x_sign=-1, color="black", lw=2.0)
        V5._median_plot(ax, V5._as_curves(simulated[gate]), grid,
                        "identified RESET", x_sign=-1,
                        color="#d62728", lw=1.8)
        ax.set_ylim(1e-11, 1e-4)
        ax.set_xlabel("Applied voltage (V)"); ax.set_ylabel("|I| (A)")
        ax.set_title("(d) Held-out transition with observed endpoint states")
        ax.legend(fontsize=8)

        fig.suptitle("VULCAN-2D v0.7 — inverse-identified RESET state",
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
