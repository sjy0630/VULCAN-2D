#!/usr/bin/env python3
"""Plot two saved necessary-bound results without any new model calculations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
SOURCES = {
    "growth": HERE.parent / "population_result.json",
    "capture": HERE.parent / "capture_budget_result.json",
}
STEM = "fig_bidirectional_state_limits"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    hashes = {key: digest(path) for key, path in SOURCES.items()}
    data = {key: json.loads(path.read_text()) for key, path in SOURCES.items()}
    growth, capture = data["growth"]["bridges"], data["capture"]["bridges"]
    x = np.array([b["reset_record"] for b in growth])
    assert list(x) == list(range(4, 35))
    assert all(a["reset_record"] == b["reset_record"] and a["set_record"] == b["set_record"]
               for a, b in zip(growth, capture))
    assert len(growth) == len(capture) == 31
    y_growth = np.array([b["budget_ratio"] for b in growth])
    y_capture = np.array([b["required_decrease_over_upper"] for b in capture])
    vg = np.array([b["status"] == "conditional_violation" for b in growth])
    vc = np.array([b["capture_bound_violated"] for b in capture])
    assert np.all(vg == (y_growth > 1)) and np.all(vc == (y_capture > 1))
    assert vg.sum() == 24 and vc.sum() == 7 and (vg & vc).sum() == 0
    assert (vg | vc).sum() == 31
    assert all(b["either_direction_violated"] for b in capture)
    assert data["capture"]["source_sha256"][str(SOURCES["growth"])] == hashes["growth"]
    assert data["growth"]["parameters"] == data["capture"]["parameters"]

    red, blue, grey, ink = "#D55E00", "#0072B2", "#A7ADB3", "#243240"
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 10,
        "axes.labelsize": 10.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#7C8690", "text.color": ink, "axes.labelcolor": ink,
        "xtick.color": ink, "ytick.color": ink, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300,
    })
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 8.5), sharex=True)
    fig.subplots_adjust(left=0.125, right=0.975, top=0.78, bottom=0.245, hspace=0.36)
    fig.suptitle("Bidirectional state limits in one training history", x=0.125,
                 y=0.974, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.125, 0.932,
             "Fixed a = 0 readout, κ, J and declared path clock; no waiting evolution",
             fontsize=10.4, ha="left")
    fig.text(0.125, 0.894,
             "24 growth violations + 7 capture violations = 31/31 conditionally excluded bridges",
             fontsize=10.3, ha="left")
    handles = [
        Line2D([], [], color=red, marker="o", linestyle="none", markersize=6,
               label="Growth violation"),
        Line2D([], [], color=blue, marker="s", linestyle="none", markersize=6,
               label="Capture violation"),
        Line2D([], [], color=grey, marker="o", markerfacecolor="white", linestyle="none",
               markersize=6, label="Not excluded by this panel alone"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.115, 0.86),
               ncol=3, frameon=False, fontsize=9.2, columnspacing=1.35, handletextpad=0.45)

    configurations = [
        (axes[0], y_growth, vg, red, "o", (-127, 207), [-100, -50, 0, 50, 100, 150, 200],
         "a   Growth bound: 24/31 violations", r"$\Delta Z_{\mathrm{required,lower}}\,/\,U_{\mathrm{rec}}$"),
        (axes[1], y_capture, vc, blue, "s", (-115, 36), [-100, -75, -50, -25, 0, 25],
         "b   Capture bound: 7/31 violations", r"$-\Delta u_{\mathrm{required,upper}}\,/\,L_{\mathrm{capture}}$"),
    ]
    for ax, y, violation, color, marker, ylim, yticks, title, ylabel in configurations:
        for lo, hi in ((3.5, 13.5), (23.5, 34.5)):
            ax.axvspan(lo, hi, color="#F4F5F6", zorder=0)
        for boundary in (13.5, 23.5):
            ax.axvline(boundary, color="#D8DDE1", linewidth=0.7, zorder=1)
        ax.grid(axis="y", color="#DDE1E5", linewidth=0.6, alpha=0.75, zorder=0)
        ax.plot(x, y, color="#B7BDC2", linewidth=0.9, zorder=2)
        ax.scatter(x[~violation], y[~violation], color="white", marker="o", s=26,
                   edgecolors=grey, linewidths=0.85, zorder=3)
        ax.scatter(x[violation], y[violation], color=color, marker=marker, s=32,
                   edgecolors="white", linewidths=0.5, zorder=4)
        ax.axhline(1, color=ink, linewidth=1.1, linestyle=(0, (5, 3)), zorder=3)
        ax.set_xlim(3.5, 34.5)
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        ax.set_ylabel("Signed ratio\n" + ylabel, labelpad=9)
        ax.set_title(title, fontsize=10.6, fontweight="bold", loc="left", pad=7)
        ax.set_xticks(x)
        ax.tick_params(axis="x", labelsize=8, length=3)
    axes[0].annotate("Necessary limit: 1", xy=(4.1, 1), xytext=(4.1, -29),
                     fontsize=8.7, ha="left", va="top")
    axes[1].annotate("Necessary limit: 1", xy=(4.1, 1), xytext=(4.1, 10.5),
                     fontsize=8.7, ha="left", va="bottom")
    axes[0].annotate("1.21", xy=(32, y_growth[28]), xytext=(31.6, 40),
                     fontsize=8.7, color=red, ha="right",
                     arrowprops={"arrowstyle": "-", "color": red, "lw": 0.7})
    axes[1].annotate("2.60", xy=(30, y_capture[26]), xytext=(29.7, 17),
                     fontsize=8.7, color=blue, ha="right",
                     arrowprops={"arrowstyle": "-", "color": blue, "lw": 0.7})
    axes[1].set_xlabel("RESET record k  (next SET record = k + 1)", labelpad=8)

    fig.text(0.125, 0.170,
             r"Bridge: RESET return at $-0.2$ V $\rightarrow$ next SET outward at $+0.2$ V.",
             fontsize=9.4, ha="left")
    fig.text(0.125, 0.133,
             r"Different coordinates: panel a uses $Z=e^u$; panel b uses $u$. Both limits are 1.",
             fontsize=9.4, ha="left")
    fig.text(0.125, 0.096,
             "Growth bound: frozen m = 2 recovery. Capture bound: any nonnegative recovery allowed.",
             fontsize=9.1, ha="left")
    fig.text(0.125, 0.059,
             "All signed ratios retained; neither ratio is a current-error factor.",
             fontsize=9.4, ha="left", color="#536171")
    fig.text(0.125, 0.025,
             "The same 31 chronological bridges appear in both panels; they are not independent experiments.",
             fontsize=9.1, ha="left", color="#536171")

    output_hashes = {}
    for suffix in ("png", "pdf"):
        out = HERE / f"{STEM}.{suffix}"
        fig.savefig(out, dpi=300, facecolor="white")
        output_hashes[out.name] = digest(out)
    plt.close(fig)
    assert hashes == {key: digest(path) for key, path in SOURCES.items()}
    metadata = {
        "description": "Two saved necessary state-budget ratios for the same 31 training bridges",
        "sources": {key: {"path": str(path), "sha256": hashes[key]} for key, path in SOURCES.items()},
        "script_sha256": digest(Path(__file__)),
        "output_sha256": output_hashes,
        "software": {"matplotlib": matplotlib.__version__, "numpy": np.__version__},
        "size_inches": [8.6, 8.5], "png_dpi": 300,
        "plotted_records": [
            {"reset_record": int(k), "growth_ratio": float(g), "capture_ratio": float(c),
             "growth_violated": bool(a), "capture_violated": bool(b)}
            for k, g, c, a, b in zip(x, y_growth, y_capture, vg, vc)
        ],
        "counts": data["capture"]["counts"],
        "interpretation": [
            "Fixed a=0 readout, kappa, J, and declared path clock; no waiting evolution.",
            "Growth coordinate Z=exp(u); capture coordinate u. Magnitudes across panels are not comparable errors.",
            "Growth bound uses frozen m=2 recovery. Capture bound allows any nonnegative recovery.",
            "Both necessary limits are 1; a nonviolation of either panel alone does not establish feasibility.",
            "The same 31 chronological bridges, not independent experiments, appear in both panels.",
            "Neither ratio is a current-error factor. Negative values are retained.",
            "No new inverse, circuit solve, ODE, fit, or raw-observation access.",
        ],
    }
    (HERE / f"{STEM}.metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"sources": hashes, "outputs": output_hashes}, indent=2))


if __name__ == "__main__":
    main()
