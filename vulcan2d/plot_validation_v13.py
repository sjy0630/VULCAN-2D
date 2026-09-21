"""Plot saved v0.13 audit outputs without fitting or simulating again."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .validate_v13 import CURVES_NAME, SUMMARY_NAME


COLORS = {"same_protocol": "#0072B2", "cross_gate_prior": "#D55E00"}
LABELS = {"same_protocol": "Same-protocol law",
          "cross_gate_prior": "Cross-gate prior"}


def _save(figure, directory, name):
    for suffix in ("png", "pdf"):
        path = directory / f"{name}.{suffix}"
        figure.savefig(path, dpi=300, facecolor="white", bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(figure)


def plot_cycle_diagnostics(report, output_dir):
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 7.5), sharex=True)
    for mode in COLORS:
        rows = sorted((row for row in report["cycle_predictions"]
                       if row["split"] == "leave_one_cycle_out"
                       and row["reset_transition_mode"] == mode
                       and row["status"] == "ok"), key=lambda row: row["cycle_id"])
        axes[0].plot(
            [row["cycle_id"] for row in rows],
            [row["metrics"]["RESET"]["all"]["rmse_decade"] for row in rows],
            marker="o" if mode == "same_protocol" else "s", markersize=3.5,
            linewidth=1, color=COLORS[mode], label=LABELS[mode])
        rejected = [row for row in rows if not row["legacy_training_eligible"]]
        axes[0].scatter(
            [row["cycle_id"] for row in rejected],
            [row["metrics"]["RESET"]["all"]["rmse_decade"] for row in rejected],
            s=95, facecolors="none", edgecolors="#333333", linewidths=1.2,
            label="Excluded from legacy training" if mode == "same_protocol" else None)
    axes[0].set_title("a  RESET prediction error: all 53 target-excluding folds", loc="left")
    axes[0].set_ylabel("Whole-cycle RMSE (decade)")
    axes[0].legend(frameon=False, fontsize=9)

    diagnostic = report["retrospective_RESET_diagnostics"]
    rows = [row for row in diagnostic["cycles"] if row["status"] == "identified"]
    eligible = set(report["data"]["groups"]["legacy_accepted"])
    for retained in (True, False):
        selected = [row for row in rows if (row["cycle_id"] in eligible) == retained]
        axes[1].scatter(
            [row["cycle_id"] for row in selected],
            [row["identified_e50_mvc"] for row in selected],
            s=24 if retained else 45, marker="o" if retained else "x",
            color="#333333" if retained else "#D55E00",
            label=("Reconstructed from measured current" if retained
                   else "Legacy-ineligible: diagnostic only"))
    for period, limits, color in (("early", (1, 35), "#009E73"),
                                  ("late", (36, 53), "#CC79A7")):
        values = diagnostic["groups"][f"legacy_accepted_{period}"][
            "identified_e50_mvc_q25_median_q75"]
        if values is not None:
            axes[1].hlines(values[1], *limits, color=color, linewidth=2,
                           label=f"{period.capitalize()} accepted median: {values[1]:.3f}")
    axes[1].set_title("b  Retrospective RESET state diagnosis (fixed early transport law)", loc="left")
    axes[1].set_ylabel(r"Identified $E_{50}$ (MV/cm)")
    axes[1].set_xlabel("Measured cycle index")
    axes[1].legend(frameon=False, fontsize=9)
    for axis in axes:
        axis.axvline(35.5, color="#777777", linewidth=1, linestyle="--")
        axis.set_xlim(0, 54)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("VULCAN-2D: full-population validation audit", fontsize=14)
    fig.text(0.5, 0.018,
             "Dashed line: fixed first-35/later split. Panel b uses measured labels for diagnosis only;\n"
             "it is not a prediction of microscopic thresholds. Both mode choices remain retrospective.",
             ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.075, 1, 0.95))
    _save(fig, output_dir, "21_v13_cycle_error_and_reset_drift")


def plot_previously_excluded(report, curves, output_dir):
    ids = report["data"]["groups"]["legacy_rejected"]
    if not ids:
        return
    fig, axes = plt.subplots(2, len(ids), figsize=(5.2 * len(ids), 7.8), squeeze=False)
    for column, index in enumerate(ids):
        for row, phase in enumerate(("set", "reset")):
            ax = axes[row, column]
            prefix = f"observed_cycle_{index}_{phase}"
            voltage = np.abs(curves[f"{prefix}_voltage"])
            current = np.abs(curves[f"{prefix}_current"])
            valid = (voltage > 0) & (current > 0) & np.isfinite(current)
            ax.plot(voltage[valid], current[valid], color="#333333", linewidth=1.6,
                    label="Measured", zorder=3)
            for mode in COLORS:
                if phase == "set" and mode == "cross_gate_prior":
                    continue
                prefix = f"leave_one_cycle_out_{mode}_cycle_{index}_{phase}"
                if f"{prefix}_current" not in curves:
                    ax.text(0.04, 0.1, f"{LABELS[mode]}: prediction failed",
                            transform=ax.transAxes, fontsize=9)
                    continue
                v = np.abs(curves[f"{prefix}_voltage"])
                i = np.abs(curves[f"{prefix}_current"])
                valid = (v > 0) & (i > 0) & np.isfinite(i)
                ax.plot(v[valid], i[valid], color=COLORS[mode],
                        linestyle="--" if mode == "same_protocol" else ":",
                        linewidth=1.7,
                        label="Autonomous SET" if phase == "set" else LABELS[mode])
            ax.set_title(f"{phase.upper()} cycle {index}", loc="left")
            ax.set_yscale("log")
            ax.set_xlabel("|Applied voltage| (V)")
            ax.set_ylabel("|Current| (A)")
            ax.grid(alpha=0.18)
            ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Previously excluded pairs: scored without identifying target envelopes", fontsize=13)
    fig.text(0.5, 0.02,
             "S2/R2 and S3/R3 are each excluded from their own parameter fits.\n"
             "RESET training still uses the 51 legacy-eligible cycles; failed labels are retained for scoring.",
             ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    _save(fig, output_dir, "22_v13_previously_excluded_pairs")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "analysis")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    output_dir = args.output_dir or args.input_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads((args.input_dir / SUMMARY_NAME).read_text())
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    plot_cycle_diagnostics(report, output_dir)
    with np.load(args.input_dir / CURVES_NAME, allow_pickle=False) as curves:
        plot_previously_excluded(report, curves, output_dir)


if __name__ == "__main__":
    main()
