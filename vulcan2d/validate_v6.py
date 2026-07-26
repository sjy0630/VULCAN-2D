"""Validate the v0.6 measured-transistor load line and its claim boundary.

The 1.1-V condition is held out from the affine 1T-to-1T1R gate mapping and
from all h-BN RESET parameters.  v0.6 is considered structurally successful if
gate voltage enters RESET only through the transistor lookup.  Numerical
improvement is reported separately and is never assumed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

from . import model_v4 as M5
from . import model_v6 as M6
from . import validate_v5 as V5
from .transistor import load_default_lookup


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "analysis" / "v6_validation_summary.json"
FIGURE_PATH = ROOT / "analysis" / "figures" / "14_vulcan_v6_transistor_loadline.png"

SET_FILES = {0.9: "0.9V栅压.xlsx", 1.1: "1T1M写入.xlsx", 1.3: "1.3V栅压.xlsx"}
RESET_FILES = {0.9: "0.9V栅压RESET.xlsx", 1.1: "1T1M擦除.xlsx", 1.3: "1.3V栅压RESET.xlsx"}
STEPS = {0.9: 0.01, 1.1: 0.02, 1.3: 0.01}
RESET_PEAKS = {0.9: 1.5, 1.1: 1.7, 1.3: 1.5}


def _simulate_v6(gate, p):
    measured_set = V5._load_cycles(SET_FILES[gate])
    measured_reset = V5._load_cycles(RESET_FILES[gate])
    sets, _ = M6.simulate_1t1r_set_ensemble(
        p, Vg=gate, n_cycles=len(measured_set), seed=110, step=STEPS[gate],
    )
    resets, _ = M6.simulate_1t1r_reset_ensemble(
        p, Vg=gate, n_cycles=len(measured_reset), seed=110,
        set_step=STEPS[gate], reset_step=STEPS[gate],
        reset_vpeak=RESET_PEAKS[gate],
    )
    return (measured_set, measured_reset,
            V5._as_curves(sets), V5._as_curves(resets))


def _hbn_gate_spread(p):
    families = {
        "reset_connected_scale": [p.reset_connected_scale_low,
                                  p.reset_connected_scale_ref,
                                  p.reset_connected_scale_high],
        "reset_off_scale": [p.reset_off_scale_low, p.reset_off_scale_ref,
                            p.reset_off_scale_high],
        "alpha_reset": [p.alpha_reset_low_v4, p.alpha_reset_ref_v4,
                        p.alpha_reset_high_v4],
        "alpha_reset_on": [p.alpha_reset_on_low_v4,
                           p.alpha_reset_on_ref_v4,
                           p.alpha_reset_on_high_v4],
        "reset_drive_scale": [p.reset_drive_scale_low,
                              p.reset_drive_scale_ref,
                              p.reset_drive_scale_high],
        "reset_threshold_factor": [p.reset_threshold_factor_low,
                                   p.reset_threshold_factor_ref,
                                   p.reset_threshold_factor_high],
    }
    return {name: float(max(values) - min(values))
            for name, values in families.items()}


def main():
    p = M6.Params()
    mapping = M6.transistor_mapping(p)
    lookup = load_default_lookup()
    results = {}
    cached = {}
    for gate in (0.9, 1.1, 1.3):
        measured_set, measured_reset, model_set, model_reset = _simulate_v6(gate, p)
        cached[gate] = (measured_set, measured_reset, model_set, model_reset)
        results[str(gate)] = {
            "effective_gate_voltage_V": mapping.effective_gate_voltage(gate),
            "set": V5._metrics(measured_set, model_set),
            "reset": V5._metrics(measured_reset, model_reset, reset=True),
        }

    v5_summary = json.loads(
        (ROOT / "analysis" / "v5_validation_summary.json").read_text(encoding="utf-8"))
    old = v5_summary["gate_holdout_1p1V"]
    v6_reset_rmse = results["1.1"]["reset"]["median_loop_rmse_decade"]
    old_direct = old["direct_anchor_model"]["reset"]["median_loop_rmse_decade"]
    old_holdout = old["endpoint_interpolation_model"]["reset"]["median_loop_rmse_decade"]

    holdout_mean = float(np.mean(list(old_holdout.values())))
    v6_mean = float(np.mean(list(v6_reset_rmse.values())))
    report = {
        "model": "VULCAN-2D v0.6 measured-transistor load-line audit",
        "source": {
            "dataset": "Nature Figure 2b standalone 1T I_D(V_DS,V_G)",
            "doi": "10.5281/zenodo.7607096",
            "license": "CC-BY-4.0",
            "sha256": "4009913b0d35ae02f1030d25e2fadba5a9fdd2dcf830f050d3e59c41eb51a518",
            "gate_curves_V": lookup.gate_voltages.tolist(),
        },
        "architecture": {
            **M6.claim_scope(p),
            "hbn_gate_parameter_spread": _hbn_gate_spread(p),
            "explicit_gate_dependent_hbn_reset_families": 0,
            "v0.5_explicit_gate_dependent_hbn_reset_families": 6,
        },
        "validation": results,
        "reset_1p1V_comparison_decade": {
            "v0.5_direct_anchor": old_direct,
            "v0.5_endpoint_holdout": old_holdout,
            "v0.6_fig2b_gate_neutral_hbn": v6_reset_rmse,
        },
        "verdict": {
            "structural_correction": True,
            "mean_reset_rmse_change_vs_v0.5_holdout_decade": v6_mean - holdout_mean,
            "material_numerical_reset_improvement": bool(
                v6_mean < holdout_mean - 0.05),
            "interpretation": (
                "The screenshot correctly identifies gate double counting in v0.5. "
                "v0.6 removes it, but the public positive-V_DS Fig. 2b curves do "
                "not resolve RESET polarity/device mapping and do not materially "
                "improve the held-out RESET loop. Reverse-orientation 1T data or "
                "a verified terminal/body mapping is still required."
            ),
        },
    }
    SUMMARY_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if plt is None:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    measured_set, measured_reset, model_set, model_reset = cached[1.1]
    holdout_p = V5.endpoint_interpolated_params(M5.Params())
    _, v5_holdout_reset = V5._simulate_1p1(holdout_p)
    direct_set, v5_direct_reset = V5._simulate_1p1(M5.Params())
    grid = np.arange(0.0, 1.7 + 1e-9, 0.02)
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.2))

    ax = axes[0, 0]
    for curve in lookup.curves():
        ax.plot(curve["drain_voltage_v"], curve["drain_current_a"] * 1e6,
                lw=1.4, label=f'{curve["gate_voltage_v"]:g} V')
    ax.set_title("(a) Public Fig. 2b transistor lookup")
    ax.set_xlabel("$V_{DS}$ (V)")
    ax.set_ylabel("$I_D$ (µA)")
    ax.legend(ncol=2, fontsize=8)

    ax = axes[0, 1]
    actual = np.array([0.9, 1.1, 1.3])
    effective = np.array([mapping.effective_gate_voltage(x) for x in actual])
    ax.plot(actual, effective, "o-", color="#1f78b4")
    ax.scatter([0.9, 1.3], effective[[0, 2]], color="#d95f02", zorder=3,
               label="endpoint fit")
    ax.scatter([1.1], [effective[1]], facecolors="none", edgecolors="black",
               s=75, label="held out")
    ax.set_title("(b) Explicit 1T-to-1T1R gate mapping")
    ax.set_xlabel("Applied $V_G$ (V)")
    ax.set_ylabel("Fig. 2b effective $V_G$ (V)")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    V5._median_plot(ax, measured_reset, grid, "data 1.1 V", x_sign=-1,
                    color="black", lw=2)
    V5._median_plot(ax, v5_direct_reset, grid, "v0.5 direct anchors", x_sign=-1,
                    color="#777777", lw=1.4, ls="-.")
    V5._median_plot(ax, v5_holdout_reset, grid, "v0.5 endpoint holdout", x_sign=-1,
                    color="#d95f02", lw=1.5, ls="--")
    V5._median_plot(ax, model_reset, grid, "v0.6 Fig. 2b", x_sign=-1,
                    color="#1f78b4", lw=1.7)
    ax.set_title("(c) 1.1 V RESET: structural fix is not a fit")
    ax.set_xlabel("Applied voltage (V)")
    ax.set_ylabel("|I| (A)")
    ax.set_ylim(1e-12, 1e-4)
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    labels = ["v0.5\ndirect", "v0.5\nholdout", "v0.6\nFig. 2b"]
    forward = [old_direct["forward_decade"], old_holdout["forward_decade"],
               v6_reset_rmse["forward_decade"]]
    returning = [old_direct["return_decade"], old_holdout["return_decade"],
                 v6_reset_rmse["return_decade"]]
    x = np.arange(3); width = 0.36
    ax.bar(x - width / 2, forward, width, label="forward")
    ax.bar(x + width / 2, returning, width, label="return")
    ax.set_xticks(x, labels)
    ax.set_ylabel("RMSE (decade)")
    ax.set_title("(d) Predictive honesty: RESET remains unresolved")
    ax.legend(fontsize=8)

    fig.suptitle("VULCAN-2D v0.6 — measured transistor, gate-neutral h-BN",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("saved", SUMMARY_PATH)
    print("saved", FIGURE_PATH)


if __name__ == "__main__":
    main()
