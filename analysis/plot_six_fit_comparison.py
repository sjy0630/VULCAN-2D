"""Draw a six-panel, like-for-like comparison of HRS and LRS fits.

Every score is computed on the same response, log10(|I|), so the R^2 values
can be compared directly.  The input curves are the voltage-binned median
branches saved by ``conduction_models.py``.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATA = HERE / "median_branches.npz"
OUT = (
    ROOT
    / "vulcan_pdf_v013_edit_2026-09-21"
    / "figs"
    / "conduction_comparison.png"
)


def r2_log(current: np.ndarray, prediction: np.ndarray) -> float:
    """Coefficient of determination evaluated on log10 current."""
    y = np.log10(current)
    yhat = np.log10(prediction)
    return float(1.0 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))


def fit_linear_basis(
    voltage: np.ndarray,
    current: np.ndarray,
    basis,
    offset=None,
):
    """Fit log(I) = intercept + slope*basis(V) + optional offset(V)."""
    log_i = np.log(current)
    fixed = np.zeros_like(voltage) if offset is None else offset(voltage)
    slope, intercept = np.polyfit(basis(voltage), log_i - fixed, 1)
    prediction = np.exp(intercept + slope * basis(voltage) + fixed)
    return prediction, (intercept, slope), r2_log(current, prediction)


def fit_sinh(voltage: np.ndarray, current: np.ndarray):
    """Fit I=A*sinh(alpha*V) in log-current space."""
    log_i = np.log(current)

    def residual(theta):
        log_a, log_alpha = theta
        alpha = np.exp(log_alpha)
        return log_a + np.log(np.sinh(alpha * voltage)) - log_i

    result = least_squares(
        residual,
        x0=[np.log(6.0e-10), np.log(6.7)],
        max_nfev=100_000,
    )
    amplitude = np.exp(result.x[0])
    alpha = np.exp(result.x[1])
    prediction = amplitude * np.sinh(alpha * voltage)
    return prediction, (amplitude, alpha), r2_log(current, prediction)


def fit_transistor_envelope(voltage: np.ndarray, current: np.ndarray):
    """Fit an effective 1T-limited knee in log-current space.

    This is a compact terminal envelope, not a standalone MOSFET extraction.
    """
    log_i = np.log(current)

    def evaluate(theta, v):
        amplitude, knee, modulation = np.exp(theta)
        return amplitude * (1.0 + modulation * v) * np.tanh(v / knee)

    result = least_squares(
        lambda theta: np.log(evaluate(theta, voltage)) - log_i,
        x0=[np.log(4.5e-7), np.log(0.76), np.log(27.0)],
        max_nfev=100_000,
    )
    prediction = evaluate(result.x, voltage)
    return prediction, tuple(np.exp(result.x)), r2_log(current, prediction)


def draw_panel(ax, voltage, current, prediction, title, formula, score, color):
    ax.semilogy(
        voltage,
        current,
        "o",
        ms=3.4,
        color="#565656",
        alpha=0.85,
        label="median data",
        zorder=3,
    )
    ax.semilogy(
        voltage,
        prediction,
        "-",
        lw=2.2,
        color=color,
        label="fit",
        zorder=4,
    )
    ax.set_title(title, fontsize=11.2, fontweight="bold", pad=4)
    ax.text(
        0.04,
        0.94,
        formula + "\n" + rf"$R^2_{{\log I}}={score:.3f}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.1,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", alpha=0.92),
    )
    ax.set_xlabel(r"terminal voltage $V_{app}$ (V)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, which="both", alpha=0.20, linewidth=0.6)


def main():
    data = np.load(DATA)
    vh, ih = data["Vh"], data["Ih"]
    vl, il = data["Vl"], data["Il"]

    hrs_sinh, _, hrs_sinh_r2 = fit_sinh(vh, ih)
    hrs_sch, _, hrs_sch_r2 = fit_linear_basis(vh, ih, np.sqrt)
    hrs_power, _, hrs_power_r2 = fit_linear_basis(vh, ih, np.log)

    lrs_transistor, _, lrs_transistor_r2 = fit_transistor_envelope(vl, il)
    lrs_power, _, lrs_power_r2 = fit_linear_basis(vl, il, np.log)
    lrs_sch, _, lrs_sch_r2 = fit_linear_basis(vl, il, np.sqrt)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(18, 6.6), sharey="row")

    draw_panel(
        axes[0, 0],
        vh,
        ih,
        hrs_sinh,
        "HRS | strongest fit",
        r"$I=A\sinh(\alpha V)$",
        hrs_sinh_r2,
        "#c62828",
    )
    draw_panel(
        axes[0, 1],
        vh,
        ih,
        hrs_sch,
        "HRS | alternative good fit",
        r"$I=A\exp(\beta\sqrt{V})$",
        hrs_sch_r2,
        "#ef6c00",
    )
    draw_panel(
        axes[0, 2],
        vh,
        ih,
        hrs_power,
        "HRS | visibly poorer fit",
        r"$I=AV^m$",
        hrs_power_r2,
        "#8d6e63",
    )
    draw_panel(
        axes[1, 0],
        vl,
        il,
        lrs_transistor,
        "LRS | strongest fit",
        r"$I=A(1+\lambda V)\tanh(V/V_k)$",
        lrs_transistor_r2,
        "#1565c0",
    )
    draw_panel(
        axes[1, 1],
        vl,
        il,
        lrs_power,
        "LRS | intermediate fit",
        r"$I=AV^m$",
        lrs_power_r2,
        "#00838f",
    )
    draw_panel(
        axes[1, 2],
        vl,
        il,
        lrs_sch,
        "LRS | visibly poorer fit",
        r"$I=A\exp(\beta\sqrt{V})$",
        lrs_sch_r2,
        "#607d8b",
    )

    axes[0, 0].set_ylabel(r"HRS median $|I|$ (A)", fontsize=9.5)
    axes[1, 0].set_ylabel(r"LRS median $|I|$ (A)", fontsize=9.5)
    axes[0, 0].set_ylim(3.0e-10, 2.5e-7)
    axes[1, 0].set_ylim(3.0e-7, 8.0e-5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    fig.text(
        0.5,
        0.005,
        "All scores use the same metric: R² on log10|I|. Curves use terminal voltage and do not uniquely identify a microscopic mechanism.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#555555",
    )
    fig.subplots_adjust(left=0.058, right=0.992, top=0.91, bottom=0.13, wspace=0.16, hspace=0.53)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, facecolor="white")
    print(f"saved {OUT}")
    print(
        "scores:",
        {
            "hrs_sinh": hrs_sinh_r2,
            "hrs_schottky": hrs_sch_r2,
            "hrs_power": hrs_power_r2,
            "lrs_transistor": lrs_transistor_r2,
            "lrs_power": lrs_power_r2,
            "lrs_schottky": lrs_sch_r2,
        },
    )


if __name__ == "__main__":
    main()
