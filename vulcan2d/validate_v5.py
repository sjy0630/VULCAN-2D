"""Audit the new data constraints added after the v0.4 pull request.

This audit does three things that the original validation did not:

1. verifies the confirmed 1 mA standalone-1R SET compliance against raw data;
2. holds out the 1.1 V gate condition and predicts it by interpolation from
   0.9/1.3 V endpoint anchors;
3. injects the CAFM spot-area distribution as an optional geometric prior and
   reports sensitivity without calling a CAFM spot a physical filament.

Run from the repository root, optionally with the locally held CAFM data:

    python -m vulcan2d.validate_v5 --cafm /path/to/spot_distribution.txt
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # numeric audit remains usable in minimal runtimes
    plt = None

from . import features as F
from . import model_v4 as M
from .spatial_prior import SpotAreaDistribution


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "analysis" / "v5_validation_summary.json"
FIGURE_PATH = ROOT / "analysis" / "figures" / "13_vulcan_v5_constraints_audit.png"


def _load_cycles(filename):
    book = pd.ExcelFile(ROOT / filename)
    cycles = []
    for sheet in book.sheet_names:
        frame = book.parse(sheet)
        voltage = frame.iloc[:, 0].to_numpy(float)
        current = frame.iloc[:, 1].to_numpy(float)
        good = np.isfinite(voltage) & np.isfinite(current)
        cycles.append((voltage[good], current[good]))
    return cycles


def _at_branch(voltage, current, target, forward):
    apex = int(np.argmax(np.abs(voltage)))
    branch = slice(None, apex + 1) if forward else slice(apex, None)
    branch_v = np.asarray(voltage[branch])
    branch_i = np.abs(np.asarray(current[branch]))
    return float(branch_i[np.argmin(np.abs(branch_v - target))])


def _first_cross(voltage, current, threshold):
    apex = int(np.argmax(np.abs(voltage)))
    hit = np.where(np.abs(current[:apex + 1]) >= threshold)[0]
    return float(voltage[hit[0]]) if len(hit) else np.nan


def _summary_1t(curves):
    rows = []
    for voltage, current in curves:
        feat = F.set_feat(np.asarray(voltage), np.asarray(current))
        rows.append({
            "Imax_A": float(np.max(np.abs(current))),
            "Vset_dlog_V": float(feat["Vset"]),
        })
    return rows


def _summary_1r(curves):
    rows = []
    for voltage, current in curves:
        rows.append({
            "Imax_A": float(np.max(np.abs(current))),
            "Vhard_0p5mA_V": _first_cross(
                np.asarray(voltage), np.asarray(current), 0.5e-3,
            ),
        })
    return rows


def _summary_reset(curves):
    rows = []
    for voltage, current in curves:
        voltage = np.asarray(voltage)
        current = np.asarray(current)
        apex = int(np.argmax(np.abs(voltage)))
        forward_i = np.abs(current[:apex + 1])
        peak = int(np.argmax(forward_i))
        rows.append({
            "I_LRS_m0p2V_A": _at_branch(voltage, current, -0.2, True),
            "I_HRS_return_m0p2V_A": _at_branch(voltage, current, -0.2, False),
            "Ireset_peak_A": float(forward_i[peak]),
        })
    return rows


def _stats(rows):
    result = {}
    for key in rows[0]:
        values = np.asarray([row[key] for row in rows], dtype=float)
        values = values[np.isfinite(values)]
        result[key] = {"n": int(len(values)), "median": float(np.median(values))}
    return result


def _median_branch(curves, grid, forward):
    all_log = []
    for voltage, current in curves:
        apex = int(np.argmax(np.abs(voltage)))
        branch = slice(None, apex + 1) if forward else slice(apex, None)
        branch_v = np.abs(np.asarray(voltage[branch]))
        branch_i = np.abs(np.asarray(current[branch]))
        order = np.argsort(branch_v)
        all_log.append(np.interp(
            grid, branch_v[order], np.log10(np.clip(branch_i[order], 1e-13, None)),
        ))
    values = np.asarray(all_log)
    return (10 ** np.median(values, axis=0),
            10 ** np.quantile(values, 0.1, axis=0),
            10 ** np.quantile(values, 0.9, axis=0))


def _loop_rmse(data, model, vmax=5.0):
    grid = np.arange(0.1, vmax + 1e-9, 0.02)
    errors = []
    for forward in (True, False):
        data_median, _, _ = _median_branch(data, grid, forward)
        model_median, _, _ = _median_branch(model, grid, forward)
        errors.append(float(np.sqrt(np.mean(
            (np.log10(model_median) - np.log10(data_median)) ** 2
        ))))
    return {"forward_decade": errors[0], "return_decade": errors[1]}


def _as_curves(cycles):
    return [(np.asarray(cycle["Vs"]), np.asarray(cycle["Is"])) for cycle in cycles]


def endpoint_interpolated_params(p=None):
    """Remove every 1.1 V gate anchor by log-interpolating 0.9/1.3 V values."""

    p = M.Params() if p is None else p
    endpoints = {
        "Icc_ref": (p.Icc_low, p.Icc_high),
        "Ireset_peak_ref": (p.Ireset_peak_low, p.Ireset_peak_high),
        "reset_connected_scale_ref": (
            p.reset_connected_scale_low, p.reset_connected_scale_high,
        ),
        "reset_perc_exponent_ref": (
            p.reset_perc_exponent_low, p.reset_perc_exponent_high,
        ),
        "reset_off_scale_ref": (p.reset_off_scale_low, p.reset_off_scale_high),
        "alpha_reset_ref_v4": (p.alpha_reset_low_v4, p.alpha_reset_high_v4),
        "alpha_reset_on_ref_v4": (
            p.alpha_reset_on_low_v4, p.alpha_reset_on_high_v4,
        ),
        "reset_drive_scale_ref": (p.reset_drive_scale_low, p.reset_drive_scale_high),
        "reset_threshold_factor_ref": (
            p.reset_threshold_factor_low, p.reset_threshold_factor_high,
        ),
    }
    # Vg_ref=1.1 V is the midpoint of the measured 0.9--1.3 V interval, so
    # log-linear interpolation reduces to the geometric mean.
    updates = {name: float(np.sqrt(low * high))
               for name, (low, high) in endpoints.items()}
    return replace(p, **updates)


def _metrics(data, model, reset=False):
    if reset:
        data_stats = _stats(_summary_reset(data))
        model_stats = _stats(_summary_reset(model))
        rmse = _loop_rmse(data, model, vmax=1.7)
        anchors = {
            "initial_current_at_-0.2V_A": {
                "data": data_stats["I_LRS_m0p2V_A"]["median"],
                "model": model_stats["I_LRS_m0p2V_A"]["median"],
            },
            "return_current_at_-0.2V_A": {
                "data": data_stats["I_HRS_return_m0p2V_A"]["median"],
                "model": model_stats["I_HRS_return_m0p2V_A"]["median"],
            },
        }
    else:
        data_stats = _stats(_summary_1t(data))
        model_stats = _stats(_summary_1t(model))
        rmse = _loop_rmse(data, model)
        anchors = {
            "apex_current_A": {
                "data": data_stats["Imax_A"]["median"],
                "model": model_stats["Imax_A"]["median"],
            },
            "set_voltage_V": {
                "data": data_stats["Vset_dlog_V"]["median"],
                "model": model_stats["Vset_dlog_V"]["median"],
            },
        }
    return {"anchors": anchors, "median_loop_rmse_decade": rmse}


def _simulate_1p1(p, spatial_weights=None):
    sets, _ = M.simulate_1t1r_set_ensemble(
        p, Vg=1.1, n_cycles=53, seed=110, step=0.02,
        spatial_weights=spatial_weights,
    )
    resets, _ = M.simulate_1t1r_reset_ensemble(
        p, Vg=1.1, n_cycles=53, seed=110, set_step=0.02,
        reset_step=0.02, reset_vpeak=1.7, spatial_weights=spatial_weights,
    )
    return _as_curves(sets), _as_curves(resets)


def _median_plot(ax, curves, grid, label, forward=True, x_sign=1.0, **kwargs):
    median, _, _ = _median_branch(curves, grid, forward)
    ax.semilogy(x_sign * grid, median, label=label, **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cafm", type=Path)
    args = parser.parse_args(argv)

    p = M.Params()
    measured_set = _load_cycles("1T1M写入.xlsx")
    measured_reset = _load_cycles("1T1M擦除.xlsx")
    measured_1r = _load_cycles("1R数据.xlsx")

    simulated_1r = _as_curves(M.simulate_1r_set_ensemble(
        p, n_cycles=len(measured_1r), seed=110, step=0.05,
    ))
    data_1r = _stats(_summary_1r(measured_1r))
    model_1r = _stats(_summary_1r(simulated_1r))
    measured_clamp = data_1r["Imax_A"]["median"]
    model_clamp = model_1r["Imax_A"]["median"]

    direct_set, direct_reset = _simulate_1p1(p)
    holdout_p = endpoint_interpolated_params(p)
    holdout_set, holdout_reset = _simulate_1p1(holdout_p)

    report = {
        "model": "VULCAN-2D v0.5 data-constraint audit",
        "standalone_1r_compliance": {
            "confirmed_protocol_A": p.Icomp_1r,
            "measured_median_apex_A": measured_clamp,
            "model_median_apex_A": model_clamp,
            "measured_relative_error_to_protocol": (
                measured_clamp / p.Icomp_1r - 1.0
            ),
            "model_never_exceeds_protocol": bool(
                max(np.max(np.abs(current)) for _, current in simulated_1r)
                <= p.Icomp_1r * (1.0 + 1e-12)
            ),
        },
        "gate_holdout_1p1V": {
            "protocol": (
                "0.9 V and 1.3 V parameter anchors retained; every explicit "
                "1.1 V gate anchor replaced by endpoint log interpolation"
            ),
            "predicted_set_ceiling_A": holdout_p.Icc_ref,
            "measured_set_ceiling_anchor_A": p.Icc_ref,
            "predicted_reset_peak_anchor_A": holdout_p.Ireset_peak_ref,
            "measured_reset_peak_anchor_A": p.Ireset_peak_ref,
            "direct_anchor_model": {
                "set": _metrics(measured_set, direct_set),
                "reset": _metrics(measured_reset, direct_reset, reset=True),
            },
            "endpoint_interpolation_model": {
                "set": _metrics(measured_set, holdout_set),
                "reset": _metrics(measured_reset, holdout_reset, reset=True),
            },
            "claim_scope": (
                "interpolation holdout inside the measured gate window; not "
                "extrapolation to a new transistor or gate range"
            ),
        },
        "cafm_spatial_prior": {
            "status": "not run",
            "claim_scope": (
                "CAFM area distribution can constrain geometric heterogeneity "
                "but cannot identify a filament diameter or defect chemistry"
            ),
        },
    }

    cafm_set = None
    cafm_reset = None
    prior_weights = None
    if args.cafm is not None:
        distribution = SpotAreaDistribution.from_text(args.cafm)
        spatial_summary = distribution.summary(k=p.K)
        prior_weights = np.asarray(
            spatial_summary["coarse_prior"]["area_weights"], dtype=float,
        )
        cafm_set, cafm_reset = _simulate_1p1(p, prior_weights)
        report["cafm_spatial_prior"] = {
            "status": "run",
            "source_basename": args.cafm.name,
            "distribution": spatial_summary,
            "sensitivity_at_1p1V": {
                "set": _metrics(measured_set, cafm_set),
                "reset": _metrics(measured_reset, cafm_reset, reset=True),
            },
            "claim_scope": spatial_summary["claim_scope"],
        }

    SUMMARY_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    if plt is None:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print("figure skipped: matplotlib is not installed in this runtime")
        print("saved", SUMMARY_PATH)
        return

    grid = np.arange(0.0, 5.0 + 1e-9, 0.02)
    reset_grid = np.arange(0.0, 1.7 + 1e-9, 0.02)
    direct_color = "#4d4d4d"
    holdout_color = "#d95f02"
    cafm_color = "#1f78b4"
    data_color = "#111111"
    fig, axes = plt.subplots(2, 2, figsize=(14.8, 9.2))

    ax = axes[0, 0]
    for voltage, current in measured_1r:
        ax.semilogy(voltage, np.abs(current), color="#d6d6d6", lw=0.45)
    _median_plot(
        ax, simulated_1r, grid, "model median", color=holdout_color, lw=2,
    )
    ax.axhline(1e-3, color=data_color, ls=":", label="confirmed 1 mA clamp")
    ax.set_title("(a) Standalone 1R SET protocol")
    ax.set_xlabel("Applied voltage (V)")
    ax.set_ylabel("|I| (A)")
    ax.set_ylim(1e-12, 2e-3)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    _median_plot(ax, measured_set, grid, "data 1.1 V", color=data_color, lw=2)
    _median_plot(
        ax, direct_set, grid, "direct-anchor model",
        color=direct_color, lw=1.5, ls="-.",
    )
    _median_plot(
        ax, holdout_set, grid, "endpoint-only holdout",
        color=holdout_color, lw=1.6, ls="--",
    )
    if cafm_set is not None:
        _median_plot(
            ax, cafm_set, grid, "CAFM-area prior",
            color=cafm_color, lw=1.5, ls=":",
        )
    ax.set_title("(b) 1.1 V SET holdout")
    ax.set_xlabel("Applied voltage (V)")
    ax.set_ylabel("|I| (A)")
    ax.set_ylim(1e-11, 2e-4)
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    _median_plot(
        ax, measured_reset, reset_grid, "data 1.1 V",
        x_sign=-1.0, color=data_color, lw=2,
    )
    _median_plot(
        ax, direct_reset, reset_grid, "direct-anchor model",
        x_sign=-1.0, color=direct_color, lw=1.5, ls="-.",
    )
    _median_plot(
        ax, holdout_reset, reset_grid, "endpoint-only holdout",
        x_sign=-1.0, color=holdout_color, lw=1.6, ls="--",
    )
    if cafm_reset is not None:
        _median_plot(
            ax, cafm_reset, reset_grid, "CAFM-area prior",
            x_sign=-1.0, color=cafm_color, lw=1.5, ls=":",
        )
    ax.set_title("(c) 1.1 V RESET holdout")
    ax.set_xlabel("Applied voltage (V)")
    ax.set_ylabel("|I| (A)")
    ax.set_ylim(1e-13, 2e-5)
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    categories = ["SET\nforward", "SET\nreturn", "RESET\nforward", "RESET\nreturn"]
    direct_metrics = report["gate_holdout_1p1V"]["direct_anchor_model"]
    holdout_metrics = report["gate_holdout_1p1V"]["endpoint_interpolation_model"]

    def errors(metrics):
        return [
            metrics["set"]["median_loop_rmse_decade"]["forward_decade"],
            metrics["set"]["median_loop_rmse_decade"]["return_decade"],
            metrics["reset"]["median_loop_rmse_decade"]["forward_decade"],
            metrics["reset"]["median_loop_rmse_decade"]["return_decade"],
        ]

    series = [
        ("direct anchor", errors(direct_metrics), direct_color),
        ("endpoint holdout", errors(holdout_metrics), holdout_color),
    ]
    if cafm_set is not None:
        cafm_metrics = report["cafm_spatial_prior"]["sensitivity_at_1p1V"]
        series.append(("CAFM prior", errors(cafm_metrics), cafm_color))
    x = np.arange(len(categories))
    width = 0.24 if len(series) == 3 else 0.34
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2.0) * width
    for offset, (label, values, color) in zip(offsets, series):
        bars = ax.bar(
            x + offset, values, width, label=label,
            color=color, edgecolor="#333333", linewidth=0.45,
        )
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)
    ax.set_title("(d) Median-loop error by validation mode")
    ax.set_xticks(x, categories)
    ax.set_ylabel("RMSE (decade)")
    ax.set_ylim(0.0, 1.2)
    ax.legend(fontsize=7)

    for ax in axes.ravel():
        ax.grid(True, which="both", alpha=0.2)
    fig.suptitle(
        "VULCAN-2D v0.5: protocol, gate holdout, and CAFM-prior audit",
        fontsize=14, y=0.985,
    )
    fig.text(
        0.5, 0.952,
        "Measured medians vs direct anchors, 0.9/1.3→1.1 V interpolation, "
        "and optional CAFM geometry",
        ha="center", va="top", fontsize=9, color="#4d4d4d",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.935))
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=170, facecolor="white")
    plt.close(fig)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("saved", FIGURE_PATH)
    print("saved", SUMMARY_PATH)


if __name__ == "__main__":
    main()
