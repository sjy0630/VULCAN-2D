"""Plot saved cross-polarity diagnostics without refitting or running a model.

The 4--53 comparison is descriptive and post hoc; all 53 cycles remain shown.
Shading summarizes between-cycle variation, not measurement uncertainty.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
COLORS = {1: "#D55E00", 2: "#0072B2", 3: "#CC79A7"}
MARKERS = {1: "o", 2: "s", 3: "^"}
STYLES = {1: "-", 2: "--", 3: "-."}


def load_inputs(directory):
    grid = pd.read_csv(directory / "cross_polarity_physics_eta_grid.csv")
    cycles = pd.read_csv(directory / "cross_polarity_physics_cycles.csv")
    summary = json.loads((directory / "cross_polarity_physics_summary.json").read_text())
    grid = grid.loc[np.isclose(grid.gate_V, 1.1) & grid.lookup_branch.eq("mean")
                    & grid.hbn_voltage_V.between(0.1 - 1e-10, 0.35 + 1e-10)].copy()
    cycles = cycles.loc[np.isclose(cycles.gate_V, 1.1)
                        & cycles.lookup_branch.eq("mean")
                        & cycles.window.eq("primary")].sort_values("cycle_id")
    expected = set(range(1, 54))
    if set(cycles.cycle_id) != expected or len(cycles) != 53:
        raise ValueError("Expected exactly 53 primary-window cycles at 1.1 V")
    if not cycles.status.eq("complete_window").all() or not grid.status.eq("valid").all():
        raise ValueError("This figure requires complete saved diagnostic coverage")
    if grid.duplicated(["cycle_id", "hbn_voltage_V"]).any():
        raise ValueError("Duplicate cycle/voltage entries in the saved grid")
    if not np.all(np.isfinite(grid.eta) & (grid.eta > 0)):
        raise ValueError("Positive finite ratios are required for logarithmic axes")
    if not np.all(np.isfinite(cycles.best_constant_eta) & (cycles.best_constant_eta > 0)):
        raise ValueError("Positive finite constant ratios are required")
    if not (np.isclose(cycles.window_low_V, 0.1).all()
            and np.isclose(cycles.window_high_V, 0.3).all()):
        raise ValueError("Primary window must be 0.10--0.30 V")
    if summary["method"]["hbn_fitted_envelopes_used"]:
        raise ValueError("Expected the measured-lookup-only diagnostic")
    return grid, cycles


def make_figure(grid, cycles):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.labelsize": 9, "axes.titlesize": 9.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 5.15))
    fig.subplots_adjust(left=0.093, right=0.982, bottom=0.33, top=0.80, wspace=0.30)
    fig.suptitle("Cross-polarity current ratios at matched h-BN voltage",
                 x=0.5, y=0.963, fontsize=11, fontweight="bold")
    fig.text(0.5, 0.910,
             r"$V_G=1.1$ V  |  mean transistor lookup  |  all 53 confirmed pairs",
             ha="center", fontsize=8.5, color="#444444")
    ax = axes[0]
    later = grid.loc[grid.cycle_id.between(4, 53)].pivot(
        index="hbn_voltage_V", columns="cycle_id", values="eta")
    if set(later.columns) != set(range(4, 54)) or later.isna().any().any():
        raise ValueError("Cycles 4--53 must cover the same plotted voltage grid")
    quantiles = np.quantile(later.to_numpy(), [0.1, 0.5, 0.9], axis=1)
    ax.fill_between(later.index, quantiles[0], quantiles[2], color="#737B85",
                    alpha=0.22, linewidth=0)
    ax.plot(later.index, quantiles[1], color="#262C34", lw=1.8, zorder=4)
    for cycle_id in (1, 2, 3):
        rows = grid.loc[grid.cycle_id.eq(cycle_id)].sort_values("hbn_voltage_V")
        sensitive = ((rows.set_zero_reference_fraction >= 0.1)
                     | (rows.reset_zero_reference_fraction >= 0.1)).to_numpy()
        x, y = rows.hbn_voltage_V.to_numpy(), rows.eta.to_numpy()
        ax.plot(x, y, color=COLORS[cycle_id], ls=STYLES[cycle_id], lw=1.15, zorder=3)
        for mask, hollow in ((sensitive, True), (~sensitive, False)):
            ax.scatter(x[mask], y[mask], s=14, marker=MARKERS[cycle_id],
                       facecolor="white" if hollow else COLORS[cycle_id],
                       edgecolor=COLORS[cycle_id], linewidth=0.8, zorder=5)
    ax.set(xlim=(0.095, 0.355), ylim=(3.5e-5, 3.0), yscale="log",
           xlabel=r"Internal voltage magnitude, $|V_{\mathrm{hBN}}|$ (V)",
           ylabel=r"Current ratio, $\eta(v)=|I_R(v)|/|I_S(v)|$")
    ax.set_xticks([0.10, 0.15, 0.20, 0.25, 0.30, 0.35])
    ax.set_title("a  Ratio across voltage", loc="left", pad=9, fontweight="bold")

    ax = axes[1]
    later_cycles = cycles.loc[cycles.cycle_id >= 4]
    ax.plot(later_cycles.cycle_id, later_cycles.best_constant_eta,
            color="#8B9096", lw=0.7, zorder=2)
    ax.scatter(later_cycles.cycle_id, later_cycles.best_constant_eta,
               s=12, color="#343A41", zorder=3)
    label_offsets = {1: (9, 9), 2: (14, 5), 3: (18, -8)}
    for cycle_id in (1, 2, 3):
        value = float(cycles.loc[cycles.cycle_id.eq(cycle_id), "best_constant_eta"].iloc[0])
        ax.scatter(cycle_id, value, s=28, color=COLORS[cycle_id],
                   marker=MARKERS[cycle_id], zorder=5)
        ax.annotate(f"R{cycle_id}", (cycle_id, value), xytext=label_offsets[cycle_id],
                    textcoords="offset points", color=COLORS[cycle_id], fontsize=8)
    ax.axhline(1.0, color="#575F68", lw=0.9, ls=(0, (3, 3)), zorder=1)
    ax.text(52.5, 1.10, r"$\eta=1$", ha="right", va="bottom", fontsize=8)
    ax.set(xlim=(-0.8, 54.5), ylim=(3.5e-4, 3.0), yscale="log",
           xlabel="Paired cycle number",
           ylabel=r"Best constant ratio, $\eta_i$ (0.10–0.30 V)")
    ax.set_xticks([1, 10, 20, 30, 40, 53])
    ax.set_title("b  Ratio for each cycle", loc="left", pad=9, fontweight="bold")
    for axis in axes:
        axis.grid(axis="y", which="major", color="#E4E7EB", linewidth=0.5, zorder=0)
        axis.tick_params(which="both", direction="out", width=0.6)

    handles = [Line2D([], [], color=COLORS[i], marker=MARKERS[i], ls=STYLES[i],
                      markersize=4, lw=1.0, label=f"Pair {i}") for i in (1, 2, 3)]
    handles += [Line2D([], [], color="#262C34", lw=1.8, label="Cycles 4–53: median"),
                Patch(facecolor="#737B85", alpha=0.22, edgecolor="none",
                      label="Cycles 4–53: 10–90% spread"),
                Line2D([], [], color="#555555", marker="o", markerfacecolor="white",
                       ls="none", markersize=4, label="Hollow (a): readout-sensitive")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.185),
               ncol=3, frameon=False, fontsize=7.5, columnspacing=1.6, handlelength=2.2)
    fig.text(0.093, 0.139,
             "Retrospective data diagnostic: SET return vs RESET initial sweep; polarity and scan stage both differ.",
             ha="left", fontsize=7.5, color="#333333")
    fig.text(0.093, 0.102,
             "Shading is cycle-to-cycle spread, not a confidence interval. The 4–53 comparison is post hoc.",
             ha="left", fontsize=7.5, color="#333333")
    fig.text(0.093, 0.065,
             "Hollow points: zero-bias reference is at least 10% of either current; a sensitivity flag, not uncertainty.",
             ha="left", fontsize=7.5, color="#333333")
    fig.text(0.093, 0.029,
             r"Panel b uses saved $10^{\mathrm{mean}(\log_{10}\eta(v))}$; no model parameters are refitted.",
             ha="left", fontsize=7.5, color="#333333")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=HERE)
    parser.add_argument("--output-dir", type=Path, default=HERE / "figures")
    args = parser.parse_args()
    grid, cycles = load_inputs(args.input_dir)
    fig = make_figure(grid, cycles)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        path = args.output_dir / f"23_cross_polarity_physics.{suffix}"
        fig.savefig(path, dpi=360, metadata={"Creator": "VULCAN-2D saved-data diagnostic"})
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
