"""Validate VULCAN-2D v0.4 against multi-gate SET/RESET and 1R data.

Run from the repository root:

    python -m vulcan2d.validate_v4

The validation intentionally separates direct fits from predictions:

* SET current ceilings, RESET peak/current-level anchors, XTEM geometry and the
  1R forming/hard-state scales are extracted constraints;
* the full I-V branch shapes, internal-field reconstruction, state continuity,
  grid invariance and coexistence of soft/hard regimes are consequences to audit.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import features as F
from . import model_v4 as M


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG_DIR = os.path.join(ROOT, "analysis", "figures")
SUMMARY_PATH = os.path.join(ROOT, "analysis", "v4_validation_summary.json")


def _load_cycles(filename):
    book = pd.ExcelFile(os.path.join(ROOT, filename))
    cycles = []
    for sheet in book.sheet_names:
        frame = book.parse(sheet)
        voltage = frame.iloc[:, 0].to_numpy(float)
        current = frame.iloc[:, 1].to_numpy(float)
        good = np.isfinite(voltage) & np.isfinite(current)
        cycles.append((voltage[good], current[good]))
    return cycles


def _at_branch(V, I, target, forward):
    apex = int(np.argmax(np.abs(V)))
    sl = slice(None, apex + 1) if forward else slice(apex, None)
    branch_v, branch_i = np.asarray(V[sl]), np.abs(np.asarray(I[sl]))
    return float(branch_i[np.argmin(np.abs(branch_v - target))])


def _first_cross(V, I, threshold):
    apex = int(np.argmax(np.abs(V)))
    hit = np.where(np.abs(I[:apex + 1]) >= threshold)[0]
    return float(V[hit[0]]) if len(hit) else np.nan


def _summary_1t(curves):
    rows = []
    for V, I in curves:
        feat = F.set_feat(np.asarray(V), np.asarray(I))
        rows.append({
            "Imax_A": float(np.max(np.abs(I))),
            "Vset_dlog_V": float(feat["Vset"]),
            "V10uA_V": _first_cross(np.asarray(V), np.asarray(I), 10e-6),
            "Iret_1V_A": _at_branch(V, I, 1.0, False),
            "Iret_2V_A": _at_branch(V, I, 2.0, False),
        })
    return rows


def _summary_1r(curves):
    rows = []
    for V, I in curves:
        rows.append({
            "Imax_A": float(np.max(np.abs(I))),
            "Vhard_0p5mA_V": _first_cross(np.asarray(V), np.asarray(I), 0.5e-3),
            "Ifwd_1V_A": _at_branch(V, I, 1.0, True),
            "Ifwd_2V_A": _at_branch(V, I, 2.0, True),
            "Iret_0p05V_A": _at_branch(V, I, 0.05, False),
        })
    return rows


def _summary_reset(curves):
    rows = []
    for V, I in curves:
        V = np.asarray(V); I = np.asarray(I)
        apex = int(np.argmax(np.abs(V)))
        forward_i = np.abs(I[:apex + 1])
        peak = int(np.argmax(forward_i))
        ilrs = _at_branch(V, I, -0.2, True)
        ihrs = _at_branch(V, I, -0.2, False)
        rows.append({
            "I_LRS_m0p2V_A": ilrs,
            "I_HRS_return_m0p2V_A": ihrs,
            "reset_ratio_m0p2V": ilrs / max(ihrs, 1e-30),
            "Ireset_peak_A": float(forward_i[peak]),
            "Vreset_at_Ipeak_V": float(V[peak]),
            "Ireset_apex_A": float(forward_i[-1]),
        })
    return rows


def _summary_1r_reset(curves):
    rows = _summary_reset(curves)
    for row in rows:
        row["R_LRS_m0p2V_ohm"] = 0.2 / max(row["I_LRS_m0p2V_A"], 1e-30)
        row["terminal_peak_power_W"] = (
            abs(row["Vreset_at_Ipeak_V"]) * row["Ireset_peak_A"]
        )
    return rows


def _column(rows, key):
    return np.asarray([row[key] for row in rows], float)


def _stats(rows):
    out = {}
    for key in rows[0]:
        values = _column(rows, key)
        values = values[np.isfinite(values)]
        out[key] = {
            "n": int(len(values)),
            "median": float(np.median(values)),
            "mean": float(np.mean(values)),
            "q10": float(np.quantile(values, 0.1)),
            "q90": float(np.quantile(values, 0.9)),
        }
    return out


def _median_branch(curves, grid, forward):
    all_log = []
    for V, I in curves:
        apex = int(np.argmax(np.abs(V)))
        sl = slice(None, apex + 1) if forward else slice(apex, None)
        vb, ib = np.abs(np.asarray(V[sl])), np.abs(np.asarray(I[sl]))
        order = np.argsort(vb)
        all_log.append(np.interp(grid, vb[order],
                                 np.log10(np.clip(ib[order], 1e-13, None))))
    values = np.asarray(all_log)
    return (10 ** np.median(values, axis=0),
            10 ** np.quantile(values, 0.1, axis=0),
            10 ** np.quantile(values, 0.9, axis=0))


def _loop_rmse(data, model, vmax=5.0):
    grid = np.arange(0.1, vmax + 1e-9, 0.02)
    errors = []
    for forward in (True, False):
        dmed, _, _ = _median_branch(data, grid, forward)
        mmed, _, _ = _median_branch(model, grid, forward)
        errors.append(float(np.sqrt(np.mean((np.log10(mmed) - np.log10(dmed)) ** 2))))
    return {"forward_decade": errors[0], "return_decade": errors[1]}


def _as_curves(cycles):
    return [(np.asarray(c["Vs"]), np.asarray(c["Is"])) for c in cycles]


def _median_curve_peak(curves):
    apex = int(np.argmax(np.abs(curves[0][0])))
    voltage = np.asarray(curves[0][0][:apex + 1])
    current = np.asarray([np.abs(I[:apex + 1]) for _, I in curves])
    median = np.median(current, axis=0)
    peak = int(np.argmax(median))
    return {"V": float(voltage[peak]), "I_A": float(median[peak]),
            "I_apex_A": float(median[-1])}


def _print_pair(label, data_stats, model_stats, key, scale=1.0, unit=""):
    d = data_stats[key]["median"] * scale
    m = model_stats[key]["median"] * scale
    print(f"  {label:18s} data={d:10.4g}  model={m:10.4g} {unit}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    p = M.Params()
    measured = {
        0.9: _load_cycles("0.9V栅压.xlsx"),
        1.1: _load_cycles("1T1M写入.xlsx"),
        1.3: _load_cycles("1.3V栅压.xlsx"),
    }
    measured_reset = {
        0.9: _load_cycles("0.9V栅压RESET.xlsx"),
        1.1: _load_cycles("1T1M擦除.xlsx"),
        1.3: _load_cycles("1.3V栅压RESET.xlsx"),
    }
    steps = {0.9: 0.01, 1.1: 0.02, 1.3: 0.01}
    reset_vpeaks = {0.9: 1.5, 1.1: 1.7, 1.3: 1.5}
    simulated = {}
    simulated_reset = {}
    for gate, curves in measured.items():
        cycles, _ = M.simulate_1t1r_set_ensemble(
            p, Vg=gate, n_cycles=len(curves), seed=110, step=steps[gate],
        )
        simulated[gate] = _as_curves(cycles)
        reset_cycles, _ = M.simulate_1t1r_reset_ensemble(
            p, Vg=gate, n_cycles=len(measured_reset[gate]), seed=110,
            set_step=steps[gate], reset_step=steps[gate],
            reset_vpeak=reset_vpeaks[gate],
        )
        simulated_reset[gate] = _as_curves(reset_cycles)

    measured_1r = _load_cycles("1R数据.xlsx")
    simulated_1r = _as_curves(M.simulate_1r_set_ensemble(
        p, n_cycles=len(measured_1r), seed=110, step=0.05,
    ))
    measured_1r_reset = _load_cycles("裸RRESET.xlsx")
    simulated_1r_reset = _as_curves(M.simulate_1r_reset_ensemble(
        p, n_cycles=len(measured_1r_reset), seed=110, step=0.025, vpeak=2.5,
    ))

    report = {"model": "VULCAN-2D v0.4", "evidence_scope": {
        "multi_gate": "paired quasi-static 1T1R SET and RESET ensembles",
        "standalone_1r": "forming/SET plus negative RESET ensembles",
        "xtem": "as-fabricated approximately 18-layer, 6-nm Au-Ti/h-BN/W stack",
        "not_identified": ["pulse time", "local temperature",
                           "unique metal-ion versus vacancy identity",
                           "state-resolved conductive-path chemistry"],
    }, "one_t_one_r": {}, "one_r": {}, "one_r_reset": {}}

    print("=" * 78)
    print("VULCAN-2D v0.4: multi-gate soft regime + standalone hard regime")
    print("=" * 78)
    for gate in (0.9, 1.1, 1.3):
        ds = _stats(_summary_1t(measured[gate]))
        ms = _stats(_summary_1t(simulated[gate]))
        rmse = _loop_rmse(measured[gate], simulated[gate])
        report["one_t_one_r"][str(gate)] = {
            "cycles": len(measured[gate]), "data": ds, "model": ms,
            "median_loop_rmse": rmse,
        }
        print(f"\nV_G={gate:.1f} V  ({len(measured[gate])} measured cycles)")
        _print_pair("apex current", ds, ms, "Imax_A", 1e6, "uA")
        _print_pair("dlogI V_SET", ds, ms, "Vset_dlog_V", 1.0, "V")
        _print_pair("10-uA crossing", ds, ms, "V10uA_V", 1.0, "V")
        _print_pair("return I at 1 V", ds, ms, "Iret_1V_A", 1e6, "uA")
        _print_pair("return I at 2 V", ds, ms, "Iret_2V_A", 1e6, "uA")
        print(f"  loop RMSE          forward={rmse['forward_decade']:.3f}, "
              f"return={rmse['return_decade']:.3f} decade")

        dr = _stats(_summary_reset(measured_reset[gate]))
        mr = _stats(_summary_reset(simulated_reset[gate]))
        reset_rmse = _loop_rmse(measured_reset[gate], simulated_reset[gate],
                                vmax=reset_vpeaks[gate])
        data_peak = _median_curve_peak(measured_reset[gate])
        model_peak = _median_curve_peak(simulated_reset[gate])
        report["one_t_one_r"][str(gate)]["reset"] = {
            "cycles": len(measured_reset[gate]), "data": dr, "model": mr,
            "median_curve_peak": {"data": data_peak, "model": model_peak},
            "median_loop_rmse": reset_rmse,
        }
        print("  RESET")
        _print_pair("initial LRS @-0.2", dr, mr, "I_LRS_m0p2V_A", 1e6, "uA")
        _print_pair("return HRS @-0.2", dr, mr, "I_HRS_return_m0p2V_A", 1e9, "nA")
        print(f"  {'median-curve peak':18s} data={data_peak['V']:10.4g}  "
              f"model={model_peak['V']:10.4g} V")
        print(f"  {'peak current':18s} data={data_peak['I_A']*1e6:10.4g}  "
              f"model={model_peak['I_A']*1e6:10.4g} uA")
        print(f"  RESET loop RMSE    forward={reset_rmse['forward_decade']:.3f}, "
              f"return={reset_rmse['return_decade']:.3f} decade")

    ds_1r = _stats(_summary_1r(measured_1r))
    ms_1r = _stats(_summary_1r(simulated_1r))
    rmse_1r = _loop_rmse(measured_1r, simulated_1r)
    report["one_r"] = {"cycles": len(measured_1r), "data": ds_1r,
                       "model": ms_1r, "median_loop_rmse": rmse_1r}
    print(f"\nStandalone 1R  ({len(measured_1r)} measured cycles)")
    _print_pair("hard crossing", ds_1r, ms_1r, "Vhard_0p5mA_V", 1.0, "V")
    _print_pair("pre-form I at 1 V", ds_1r, ms_1r, "Ifwd_1V_A", 1e9, "nA")
    _print_pair("pre-form I at 2 V", ds_1r, ms_1r, "Ifwd_2V_A", 1e9, "nA")
    _print_pair("return I at 0.05 V", ds_1r, ms_1r, "Iret_0p05V_A", 1e6, "uA")
    print(f"  loop RMSE          forward={rmse_1r['forward_decade']:.3f}, "
          f"return={rmse_1r['return_decade']:.3f} decade")

    ds_1rr = _stats(_summary_1r_reset(measured_1r_reset))
    ms_1rr = _stats(_summary_1r_reset(simulated_1r_reset))
    rmse_1rr = _loop_rmse(measured_1r_reset, simulated_1r_reset, vmax=2.5)
    dpeak_1rr = _median_curve_peak(measured_1r_reset)
    mpeak_1rr = _median_curve_peak(simulated_1r_reset)
    report["one_r_reset"] = {
        "cycles": len(measured_1r_reset), "data": ds_1rr, "model": ms_1rr,
        "median_curve_peak": {"data": dpeak_1rr, "model": mpeak_1rr},
        "median_loop_rmse": rmse_1rr,
    }
    print(f"\nStandalone 1R RESET  ({len(measured_1r_reset)} measured cycles)")
    _print_pair("initial R @-0.2", ds_1rr, ms_1rr, "R_LRS_m0p2V_ohm", 1.0, "ohm")
    _print_pair("peak current", ds_1rr, ms_1rr, "Ireset_peak_A", 1e3, "mA")
    _print_pair("return HRS @-0.2", ds_1rr, ms_1rr,
                "I_HRS_return_m0p2V_A", 1e12, "pA")
    _print_pair("terminal peak power", ds_1rr, ms_1rr,
                "terminal_peak_power_W", 1e3, "mW")
    print(f"  {'median-curve peak':18s} data={dpeak_1rr['V']:10.4g}  "
          f"model={mpeak_1rr['V']:10.4g} V")
    print(f"  RESET loop RMSE    forward={rmse_1rr['forward_decade']:.3f}, "
          f"return={rmse_1rr['return_decade']:.3f} decade")

    # Grid-invariance diagnostic: the same deterministic device/ramp resampled.
    fine, _ = M.simulate_1t1r_set_ensemble(p, 1.1, 1, seed=7, step=0.01)
    coarse, _ = M.simulate_1t1r_set_ensemble(p, 1.1, 1, seed=7, step=0.02)
    grid = np.arange(0.1, 5.0 + 1e-9, 0.02)
    fm, _, _ = _median_branch(_as_curves(fine), grid, True)
    cm, _, _ = _median_branch(_as_curves(coarse), grid, True)
    grid_rmse = float(np.sqrt(np.mean((np.log10(fm) - np.log10(cm)) ** 2)))
    report["grid_invariance_forward_rmse_decade"] = grid_rmse
    report["regime_anchors"] = M.regime_summary(p)
    print(f"\nGrid invariance (0.01 vs 0.02 V): {grid_rmse:.4f} decade")

    # ----------------------------- figure ---------------------------------
    colors = {0.9: "#2b83ba", 1.1: "#33a02c", 1.3: "#d7191c"}
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.4))

    ax = axes[0, 0]
    grid = np.arange(0.0, 5.0 + 1e-9, 0.02)
    for gate in (0.9, 1.1, 1.3):
        dmed, _, _ = _median_branch(measured[gate], grid, True)
        mmed, _, _ = _median_branch(simulated[gate], grid, True)
        ax.semilogy(grid, dmed, color=colors[gate], lw=2,
                    label=f"data {gate:.1f} V")
        ax.semilogy(grid, mmed, color=colors[gate], lw=1.4, ls="--",
                    label=f"model {gate:.1f} V")
    ax.set_title("(a) 1T1R SET: gate-selected soft regime")
    ax.set_xlabel("Applied voltage (V)"); ax.set_ylabel("|I| (A)")
    ax.set_ylim(1e-11, 2e-4); ax.legend(fontsize=7, ncol=2)

    ax = axes[0, 1]
    for gate in (0.9, 1.1, 1.3):
        rgrid = np.arange(0.0, reset_vpeaks[gate] + 1e-9, steps[gate])
        dmed, _, _ = _median_branch(measured_reset[gate], rgrid, True)
        mmed, _, _ = _median_branch(simulated_reset[gate], rgrid, True)
        ax.semilogy(-rgrid, dmed, color=colors[gate], lw=2,
                    label=f"data {gate:.1f} V")
        ax.semilogy(-rgrid, mmed, color=colors[gate], lw=1.4, ls="--",
                    label=f"model {gate:.1f} V")
    ax.set_title("(b) 1T1R RESET: explicit V_G and state continuity")
    ax.set_xlabel("Applied voltage (V)"); ax.set_ylabel("|I| (A)")
    ax.set_ylim(1e-13, 1e-4); ax.legend(fontsize=7, ncol=2)

    ax = axes[0, 2]
    for V, I in measured_1r:
        ax.semilogy(V, np.abs(I), color="0.84", lw=0.5)
    med, lo, hi = _median_branch(simulated_1r, grid, True)
    ax.fill_between(grid, lo, hi, color="#ef8a62", alpha=0.28)
    ax.semilogy(grid, med, color="#b2182b", lw=2, label="model median")
    ax.axhline(p.Icomp_1r, color="black", ls=":", label="10 mA SET clamp")
    ax.set_title("(c) Standalone 1R SET: hard-path formation")
    ax.set_xlabel("Applied voltage (V)"); ax.set_ylabel("|I| (A)")
    ax.set_ylim(1e-12, 2e-3); ax.legend(fontsize=8)

    ax = axes[1, 0]
    rgrid = np.arange(0.0, 2.5 + 1e-9, 0.025)
    for forward, ls, label in ((True, "-", "forward"), (False, ":", "return")):
        dmed, _, _ = _median_branch(measured_1r_reset, rgrid, forward)
        mmed, _, _ = _median_branch(simulated_1r_reset, rgrid, forward)
        ax.semilogy(-rgrid, dmed, color="0.35", lw=1.8, ls=ls,
                    label=f"data {label}")
        ax.semilogy(-rgrid, mmed, color="#b2182b", lw=1.5, ls=ls,
                    label=f"model {label}")
    ax.set_title("(d) Standalone 1R RESET: staged hard-link rupture")
    ax.set_xlabel("Applied voltage (V)"); ax.set_ylabel("|I| (A)")
    ax.set_ylim(1e-13, 2e-2); ax.legend(fontsize=7)

    ax = axes[1, 1]
    example_soft, _ = M.simulate_1t1r_reset_ensemble(
        p, 1.1, 1, seed=110, reset_step=0.02, reset_vpeak=1.7,
    )
    c = example_soft[0]; apex = int(np.argmax(np.abs(c["Vs"])))
    ax.plot(c["Vs"][:apex + 1], c["pbs"][:apex + 1],
            color="#5e3c99", lw=2, label="weighted connectivity")
    ax2 = ax.twinx()
    ax2.plot(c["Vs"][:apex + 1], c["Ehs_MV_cm"][:apex + 1],
             color="#e66101", lw=1.4, label="mean h-BN field")
    ax.set_title("(e) XTEM geometry enters through E_hBN=V_hBN/d")
    ax.set_xlabel("Applied voltage (V)"); ax.set_ylabel("weighted phi")
    ax2.set_ylabel("E_hBN (MV/cm)")

    ax = axes[1, 2]
    example = M.simulate_1r_reset_ensemble(p, 1, seed=110, step=0.005)[0]
    apex = int(np.argmax(np.abs(example["Vs"])))
    for idx in range(example["links"].shape[1]):
        ax.plot(example["Vs"][:apex + 1], example["links"][:apex + 1, idx],
                lw=0.9, alpha=0.65)
    ax.plot(example["Vs"][:apex + 1], example["hard"][:apex + 1],
            color="black", lw=2, label="weighted hard connectivity")
    ax.set_title("(f) Coarse links span the layered stack, not single layers")
    ax.set_xlabel("Applied voltage (V)"); ax.set_ylabel("surviving link state")
    ax.set_ylim(-0.03, 1.03); ax.legend(fontsize=8)

    for ax in axes.ravel():
        ax.grid(True, which="both", alpha=0.2)
    fig.suptitle("VULCAN-2D v0.4 - XTEM-constrained field + soft/hard SET-RESET",
                 fontsize=13, y=0.995)
    fig.tight_layout()
    figure_path = os.path.join(FIG_DIR, "12_vulcan_v4_multiregime_validation.png")
    fig.savefig(figure_path, dpi=150)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print("\nsaved", figure_path)
    print("saved", SUMMARY_PATH)


if __name__ == "__main__":
    main()
