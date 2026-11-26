#!/usr/bin/env python3
"""Plot only saved population results; no inverse, ODE, fitting, or raw-data reads."""

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
SOURCE = HERE.parent / "population_result.json"
STEM = "fig_training_state_budget"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source_bytes = SOURCE.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    result = json.loads(source_bytes)
    bridges = result["bridges"]
    x = np.array([b["reset_record"] for b in bridges])
    y = np.array([b["budget_ratio"] for b in bridges])
    excluded = np.array([b["status"] == "conditional_violation" for b in bridges])
    assert list(x) == list(range(4, 35))
    assert len(x) == 31 and excluded.sum() == 24 and (y < 0).sum() == 7
    assert np.all(excluded == (y > 1))
    assert all(b["set_record"] == b["reset_record"] + 1 for b in bridges)

    # Okabe--Ito: vermillion and blue, distinguished additionally by marker shape.
    red, blue, ink = "#D55E00", "#0072B2", "#243240"
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 10.5,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#7C8690",
        "text.color": ink,
        "axes.labelcolor": ink,
        "xtick.color": ink,
        "ytick.color": ink,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    })
    fig, ax = plt.subplots(figsize=(8.6, 5.5))
    fig.subplots_adjust(left=0.115, right=0.975, top=0.74, bottom=0.265)
    fig.suptitle("State-budget conflict across the training sequence", x=0.115,
                 y=0.975, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.115, 0.922, "Conditional fixed readout (a = 0); no waiting evolution",
             fontsize=10.5, ha="left")

    # Fixed chronological groups from the saved result, never selected by outcome.
    groups = result["chronological_groups"]
    for name, face in (("early", "#F4F5F6"), ("middle", "white"), ("late", "#F4F5F6")):
        group = groups[name]
        lo, hi = group["reset_labels"]
        ax.axvspan(lo - 0.5, hi + 0.5, color=face, zorder=0)
        ax.text((lo + hi) / 2, 1.015,
                f"{name.capitalize()}: {group['conditional_violation']}/{group['n']} violations",
                transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=9.7)
    for boundary in (13.5, 23.5):
        ax.axvline(boundary, color="#CDD2D7", linewidth=0.8, zorder=1)

    ax.grid(axis="y", color="#DDE1E5", linewidth=0.6, alpha=0.75, zorder=0)
    ax.plot(x, y, color="#A8AFB5", linewidth=1, zorder=2)
    ax.scatter(x[excluded], y[excluded], color=red, marker="o", s=32,
               edgecolors="white", linewidths=0.5, zorder=4)
    ax.scatter(x[~excluded], y[~excluded], color=blue, marker="s", s=32,
               edgecolors="white", linewidths=0.5, zorder=4)
    ax.axhline(1, color=ink, linewidth=1.1, linestyle=(0, (5, 3)), zorder=3)
    ax.annotate("Necessary limit: 1", xy=(4.3, 1), xytext=(4.3, -24),
                fontsize=9, color=ink, ha="left", va="top",
                bbox={"facecolor": "#F4F5F6", "edgecolor": "none", "pad": 1.5})
    ax.annotate("1.21", xy=(32, y[list(x).index(32)]), xytext=(31.6, 33),
                fontsize=9, color=red, ha="right",
                arrowprops={"arrowstyle": "-", "color": red, "lw": 0.75})

    ax.set_xlim(3.5, 34.5)
    ax.set_ylim(-127, 208)
    ax.set_xticks(x)
    ax.tick_params(axis="x", labelsize=8, length=3)
    ax.set_yticks([-100, -50, 0, 50, 100, 150, 200])
    ax.set_ylabel("Signed state-budget ratio\n" +
                  r"$\Delta Z_{\mathrm{required,lower}}\,/\,\Delta Z_{\mathrm{allowed,upper}}$",
                  labelpad=9)
    ax.set_xlabel("RESET record k  (next SET record = k + 1)", labelpad=8)

    handles = [
        Line2D([], [], color=red, marker="o", linestyle="none", markersize=6,
               label="Conditional violation: 24"),
        Line2D([], [], color=blue, marker="s", linestyle="none", markersize=6,
               label="Not excluded by this condition: 7"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.108, 0.88),
               ncol=2, frameon=False, fontsize=10, columnspacing=2, handletextpad=0.5)
    fig.text(0.115, 0.145,
             r"Bridge: RESET return at $-0.2$ V $\rightarrow$ next SET outward at $+0.2$ V.",
             fontsize=9.2, ha="left")
    fig.text(0.115, 0.097,
             "Ratio > 1 violates the necessary condition; ratio ≤ 1 does not establish feasibility.",
             fontsize=9.2, ha="left")
    fig.text(0.115, 0.049,
             "All seven negative ratios are retained. This ratio is not a current-error factor.",
             fontsize=9.2, ha="left", color="#536171")

    outputs = {}
    for suffix in ("pdf", "png"):
        path = HERE / f"{STEM}.{suffix}"
        fig.savefig(path, dpi=300, facecolor="white")
        outputs[path.name] = sha256(path)
    plt.close(fig)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == source_hash

    metadata = {
        "description": "Signed saved state-budget ratios in native training-record order",
        "source": str(SOURCE),
        "source_sha256": source_hash,
        "script_sha256": sha256(Path(__file__)),
        "output_sha256": outputs,
        "software": {"matplotlib": matplotlib.__version__, "numpy": np.__version__},
        "size_inches": [8.6, 5.5],
        "png_dpi": 300,
        "plotted_fields": ["reset_record", "set_record", "budget_ratio", "status"],
        "plotted_records": [
            {key: b[key] for key in ("reset_record", "set_record", "budget_ratio", "status")}
            for b in bridges
        ],
        "chronological_groups": groups,
        "conditional_violations": int(excluded.sum()),
        "not_excluded": int((~excluded).sum()),
        "negative_ratios_retained": int((y < 0).sum()),
        "necessary_limit": 1,
        "interpretation": [
            "Conditional fixed readout with a=0 and no waiting evolution.",
            "Ratio is a state-growth budget comparison, not a current-error factor.",
            "Not excluded is not a sufficient feasibility result.",
            "Chronological bridges are not asserted to be independent experiments.",
            "No confidence interval, new inverse, ODE, parameter fit, or raw observation read.",
        ],
    }
    (HERE / f"{STEM}.metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"source_sha256": source_hash, "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
