"""Plot the measured cycle-3/cycle-4 SET and RESET branches.

The comparison follows the same branch definition used by the model audits:
SET uses the return branch after the positive-voltage apex, while RESET uses
the outward branch before the negative-voltage apex.  The inset highlights
the native 0.10--0.30 V magnitude window used to quantify the transition.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORKSPACE = ROOT.parent
OUTPUT = WORKSPACE / "output" / "figures"
sys.path.insert(0, str(HERE))

from diagnose_cross_polarity_physics import read_pairs  # noqa: E402


COLORS = {3: "#1768AC", 4: "#E4572E"}


def log_tick(value: float, _position: float) -> str:
    """Use plain-ASCII exponents so labels render reliably on macOS CJK fonts."""
    if value <= 0:
        return ""
    exponent = int(round(np.log10(value)))
    return f"1e{exponent}" if np.isclose(value, 10.0**exponent) else ""


def branch_data(raw: dict, phase: str) -> tuple[np.ndarray, np.ndarray]:
    voltage = np.asarray(raw["voltage"], dtype=float)
    current = np.abs(np.asarray(raw["current"], dtype=float))
    magnitude = np.abs(voltage)
    apex = np.flatnonzero(
        np.isclose(magnitude, np.nanmax(magnitude), atol=1e-10, rtol=0)
    )
    indices = np.arange(len(voltage))
    if phase == "set":
        mask = indices > apex[-1]
    else:
        mask = indices < apex[0]
    mask &= np.isfinite(voltage) & np.isfinite(current) & (current > 0)
    return voltage[mask], current[mask]


def geometric_mean_in_window(voltage: np.ndarray, current: np.ndarray) -> float:
    magnitude = np.abs(voltage)
    mask = (magnitude >= 0.10 - 1e-10) & (magnitude <= 0.30 + 1e-10)
    return float(np.exp(np.mean(np.log(current[mask]))))


def format_current(value: float) -> str:
    if value >= 1e-6:
        return f"{value * 1e6:.3g} μA"
    return f"{value * 1e9:.3g} nA"


def main() -> None:
    pairs = read_pairs("1.1")
    curves: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {
        "set": {},
        "reset": {},
    }
    for phase in curves:
        for cycle in (3, 4):
            curves[phase][cycle] = branch_data(pairs[cycle - 1][phase], phase)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial Unicode MS",
                "Hiragino Sans GB",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "axes.linewidth": 0.9,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.9), constrained_layout=True)

    settings = {
        "set": {
            "axis": axes[0],
            "title": "a  SET 回程（扫描方向：+5 V → 0 V）",
            "xlim": (0.0, 5.0),
            "highlight": (0.10, 0.30),
            "inset_pos": [0.50, 0.47, 0.46, 0.43],
            "inset_xlim": (0.09, 0.31),
            "inset_ylim": (3e-7, 1.7e-6),
        },
        "reset": {
            "axis": axes[1],
            "title": "b  RESET 去程（扫描方向：0 V → -1.68 V）",
            "xlim": (-1.70, 0.0),
            "highlight": (-0.30, -0.10),
            "inset_pos": [0.07, 0.47, 0.46, 0.43],
            "inset_xlim": (-0.31, -0.09),
            "inset_ylim": (3e-10, 1.2e-6),
        },
    }

    for phase, config in settings.items():
        ax = config["axis"]
        ax.axvspan(*config["highlight"], color="#7A5195", alpha=0.10, zorder=0)
        means = {}
        for cycle in (3, 4):
            voltage, current = curves[phase][cycle]
            ax.plot(
                voltage,
                current,
                color=COLORS[cycle],
                lw=2.0,
                label=f"第 {cycle} 圈",
                zorder=2,
            )
            means[cycle] = geometric_mean_in_window(voltage, current)

        ax.set_yscale("log")
        ax.set_xlim(*config["xlim"])
        ax.set_xlabel(r"外加电压 $V_{app}$ (V)")
        ax.set_ylabel(r"电流幅值 $|I|$ (A)")
        ax.set_title(config["title"], loc="left", fontweight="semibold", pad=10)
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=9))
        ax.yaxis.set_major_formatter(FuncFormatter(log_tick))
        ax.grid(True, which="major", ls="--", lw=0.6, alpha=0.28)
        ax.grid(True, which="minor", axis="y", ls=":", lw=0.45, alpha=0.18)
        ax.legend(loc="lower right", frameon=False)

        inset = ax.inset_axes(config["inset_pos"])
        for cycle in (3, 4):
            voltage, current = curves[phase][cycle]
            magnitude = np.abs(voltage)
            mask = (magnitude >= 0.10 - 1e-10) & (magnitude <= 0.30 + 1e-10)
            inset.plot(
                voltage[mask],
                current[mask],
                color=COLORS[cycle],
                marker="o",
                ms=3.2,
                lw=1.45,
            )
        inset.set_yscale("log")
        inset.set_xlim(*config["inset_xlim"])
        inset.set_ylim(*config["inset_ylim"])
        inset.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
        inset.yaxis.set_major_formatter(FuncFormatter(log_tick))
        inset.grid(True, which="both", ls=":", lw=0.45, alpha=0.35)
        inset.tick_params(labelsize=7.2, length=2.5)
        inset.set_title("低压比较区：$|V|=0.10$–$0.30$ V", fontsize=8.5, pad=4)

        ratio = means[4] / means[3]
        annotation = (
            f"几何平均：第3圈 {format_current(means[3])}\n"
            f"　　　　　第4圈 {format_current(means[4])}\n"
            f"第4/第3圈 = {ratio:.2f}×" if ratio < 10 else
            f"几何平均：第3圈 {format_current(means[3])}\n"
            f"　　　　　第4圈 {format_current(means[4])}\n"
            f"第4/第3圈 = {ratio:.0f}×"
        )
        inset.text(
            0.04,
            0.05,
            annotation,
            transform=inset.transAxes,
            fontsize=7.4,
            va="bottom",
            ha="left",
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "0.80", "alpha": 0.92},
        )

    fig.suptitle(
        "第3圈与第4圈实测 I–V 支路对比（$V_G=1.1$ V）",
        fontsize=15,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        -0.02,
        "紫色阴影及小窗为异常判别所用的低压区间；SET 基本重合，而 RESET 在第4圈整体跃升。",
        ha="center",
        fontsize=10,
        color="#444444",
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    png = OUTPUT / "cycle3_cycle4_set_reset_comparison.png"
    svg = OUTPUT / "cycle3_cycle4_set_reset_comparison.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    set_ratio = geometric_mean_in_window(*curves["set"][4]) / geometric_mean_in_window(*curves["set"][3])
    reset_ratio = geometric_mean_in_window(*curves["reset"][4]) / geometric_mean_in_window(*curves["reset"][3])
    print(png)
    print(svg)
    print(f"SET cycle4/cycle3 = {set_ratio:.8f}")
    print(f"RESET cycle4/cycle3 = {reset_ratio:.8f}")


if __name__ == "__main__":
    main()
