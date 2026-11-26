"""End-to-end execution, validation, and archiving for Iteration 2026-11-25-02.

Runs from VULCAN-2D-handoff-2026-11-24 using main .venv Python:
  ../VULCAN-2D/.venv/bin/python -m analysis.run_iteration_2026_11_25_02
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

from collections import Counter, defaultdict
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from analysis import audit_iv41_high_conductance as reads
from analysis import refit_iv41 as iv
from analysis import validate_joint_gap_set as joint
from analysis import validate_joint_gap_width as jgw
from analysis.et_two_channel_model import (
    ElectroThermalResetRamp,
    FieldScaledET2ChSetRamp,
    StandaloneNMOS,
)
from vulcan2d import validate_shared_device as shared
from vulcan2d.transistor import load_default_lookup, load_reverse_lookup

HANDOFF_ROOT = Path(__file__).resolve().parents[1]
MAIN_REPO = HANDOFF_ROOT.parent / "VULCAN-2D"
CACHE_DIR = HANDOFF_ROOT.parent / "outputs" / "vulcan-handoff" / "iv41_cache"
WORKBOOK_IV41 = Path("/Users/sujiangyu/Desktop/VULCAN-2D/不同栅压_原始曲线数据.xlsx")
EVIDENCE_DIR = MAIN_REPO / "docs" / "model-exploration" / "evidence"
ITER_DIR = (
    MAIN_REPO
    / "docs"
    / "model-exploration"
    / "iterations"
    / "2026-11-25-02-数据配对与电热双通道统一模型"
)
RNG_SEED = 20261125


def sha256_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_simple_xlsx(path: Path):
    out = []
    ns = {
        "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    }
    with ZipFile(path) as z:
        wb_xml = ET.fromstring(z.read("xl/workbook.xml"))
        rels = {
            el.attrib["Id"]: el.attrib["Target"]
            for el in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        }
        for sh in wb_xml.findall(".//m:sheet", ns):
            name = sh.attrib["name"]
            rid = sh.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
            target = "xl/" + rels[rid].lstrip("/")
            if not target.startswith("xl/worksheets/"):
                target = "xl/worksheets/" + rels[rid].split("/")[-1]
            root = ET.fromstring(z.read(target))
            vs, is_ = [], []
            for row in root.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row"):
                nums = []
                for c in row:
                    if c.attrib.get("t") == "s":
                        continue
                    vel = c.find("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v")
                    if vel is not None and vel.text is not None:
                        try:
                            nums.append(float(vel.text))
                        except ValueError:
                            pass
                if len(nums) >= 2:
                    vs.append(nums[0])
                    is_.append(nums[1])
            if vs:
                out.append((name, np.asarray(vs, float), np.asarray(is_, float)))
    return out


def low_bias_g015(v: np.ndarray, i: np.ndarray):
    ap = int(np.argmax(np.abs(v)))
    vo, io = np.abs(v[: ap + 1]), np.abs(i[: ap + 1])
    vr, ir = np.abs(v[ap:][::-1]), np.abs(i[ap:][::-1])
    g_in = max(float(np.interp(0.15, vo, io) / 0.15), 1e-13)
    g_ret = max(float(np.interp(0.15, vr, ir) / 0.15), 1e-13)
    return g_in, g_ret


def reach_half_lrs_v(v: np.ndarray, i: np.ndarray) -> float | None:
    ap = int(np.argmax(v))
    vo, io = v[: ap + 1], np.abs(i[: ap + 1])
    vr, ir = v[ap:][::-1], np.abs(i[ap:][::-1])
    ir_interp = np.interp(vo, vr, ir)
    cross = np.flatnonzero((vo >= 0.45) & (io >= 0.5 * ir_interp))
    return float(vo[cross[0]]) if len(cross) else None


def wasserstein_1d(u: np.ndarray, v: np.ndarray) -> float:
    q = np.linspace(0.01, 0.99, 99)
    return float(np.mean(np.abs(np.quantile(u, q) - np.quantile(v, q))))


def run_part1_audit(idx: dict, nmos: StandaloneNMOS) -> dict:
    """Audit historical Fig.2c mispairing, IV41 pairing/blocks, VG=5V bridge, and 1T lookup."""
    # 1. Sheet 3 Fig.2c mapping audit
    s_map, r_map = {}, {}
    with ZipFile(WORKBOOK_IV41) as z:
        ss = []
        root_ss = ET.fromstring(z.read("xl/sharedStrings.xml"))
        for si in root_ss:
            ss.append(
                "".join(
                    t.text or ""
                    for t in si.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")
                )
            )
        root_s3 = ET.fromstring(z.read("xl/worksheets/sheet3.xml"))
        for r_el in list(
            root_s3.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row")
        )[1:]:
            vals = []
            for c in r_el:
                vel = c.find("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v")
                vals.append(
                    ss[int(vel.text)]
                    if (vel is not None and c.attrib.get("t") == "s")
                    else (vel.text if vel is not None else None)
                )
            if vals[0].startswith("Fig.2c set"):
                s_map[int(vals[1][1:])] = vals[10]
            elif vals[0].startswith("Fig.2c reset"):
                r_map[int(vals[1][1:])] = vals[10]

    exact_matches = []
    offsets = []
    for k in range(1, 54):
        sid = int(s_map[k].split("@")[1]) if "@" in s_map[k] else 1
        rid = int(r_map[k].split("@")[1]) if "@" in r_map[k] else 1
        offsets.append(sid - rid)
        if sid == rid:
            exact_matches.append(k)

    # Historical Fig.2c correlations
    h11_s = read_simple_xlsx(HANDOFF_ROOT / "1T1M写入.xlsx")
    h11_r = read_simple_xlsx(HANDOFF_ROOT / "1T1M擦除.xlsx")
    h11_s_in, h11_s_ret = zip(*(low_bias_g015(v, i) for _, v, i in h11_s))
    h11_r_in, h11_r_ret = zip(*(low_bias_g015(v, i) for _, v, i in h11_r))
    fig2c_corr_sr = float(np.corrcoef(np.log10(h11_s_ret), np.log10(h11_r_in))[0, 1])
    fig2c_corr_rs = float(np.corrcoef(np.log10(h11_r_ret[:-1]), np.log10(h11_s_in[1:]))[0, 1])

    # Full 152-cycle Cell 322 & IV41 0.9-1.5V + 5.0V pairing & block stats
    iv41_pairing = {}
    vset_vs_vrst = {}
    for label, s_idx, obj_id, vrst in [
        ("0.9", 5, 2, 1.5),
        ("1.0", 6, 3, 1.5),
        ("1.1", 7, 4, 1.5),
        ("1.2", 8, 5, 1.5),
        ("1.3", 9, 6, 1.5),
        ("1.4", 10, 7, 1.5),
        ("1.5", 11, 8, 1.5),
        ("5.0_IV41", 12, 10, 3.0),
        ("1.1_Cell322_full", 19, 17, 1.7),
    ]:
        c_meta = {str(r[4]): r for r in idx["curves"][1:] if r[1] == obj_id}
        s_in, s_ret, r_in, r_end, vsets = [], [], [], [], []
        with np.load(CACHE_DIR / f"s{s_idx}.npz") as data:
            for k in sorted(data.files, key=int):
                _, v, i = data[k].T
                gi, gr = low_bias_g015(v, i)
                if c_meta[k][6] == "正":
                    s_in.append(gi)
                    s_ret.append(gr)
                    vs = reach_half_lrs_v(v, i)
                    if vs is not None:
                        vsets.append(vs)
                else:
                    r_in.append(gi)
                    r_end.append(gr)
        n = min(len(s_ret), len(r_in))
        corr_sr = float(np.corrcoef(np.log10(s_ret[:n]), np.log10(r_in[:n]))[0, 1])
        corr_rs = float(np.corrcoef(np.log10(r_end[: n - 1]), np.log10(s_in[1:n]))[0, 1])
        h_flags = [g * 1e6 > 30.0 for g in s_ret]
        runs = 1 + sum(a != b for a, b in zip(h_flags[:-1], h_flags[1:])) if h_flags else 0
        cut = int(0.7 * len(h_flags))
        iv41_pairing[label] = {
            "n_set": len(s_ret),
            "n_reset": len(r_in),
            "corr_set_ret_to_reset_in_lag0": corr_sr,
            "corr_reset_end_to_next_set_in": corr_rs,
            "high_conductance_gt30uS_fraction": float(np.mean(h_flags)),
            "high_low_runs": int(runs),
            "first_70pct_high_fraction": float(np.mean(h_flags[:cut])) if cut else None,
            "last_30pct_high_fraction": float(np.mean(h_flags[cut:])) if cut else None,
            "median_R_ret_ohm": float(1.0 / np.median(s_ret)),
        }
        vset_vs_vrst[label] = {
            "reset_stop_abs_V": vrst,
            "vset_50pct_lrs_q10_q50_q90_V": np.quantile(vsets, [0.1, 0.5, 0.9]).tolist(),
        }

    bare_s = read_simple_xlsx(HANDOFF_ROOT / "1R数据.xlsx")
    bare_r = read_simple_xlsx(HANDOFF_ROOT / "裸RRESET.xlsx")
    bs_in, bs_ret = zip(*(low_bias_g015(v, i) for _, v, i in bare_s))
    br_in, br_end = zip(*(low_bias_g015(v, i) for _, v, i in bare_r))
    bare_onsets = [
        float(v[joint.clamp_indices(v, np.abs(i), 1e-3)])
        for _, v, i in bare_s
        if joint.clamp_indices(v, np.abs(i), 1e-3) is not None
    ]
    vset_vs_vrst["Bare_1M"] = {
        "reset_stop_abs_V": 2.5,
        "vset_clamp_q10_q50_q90_V": np.quantile(bare_onsets, [0.1, 0.5, 0.9]).tolist(),
        "median_R_reset_entry_ohm": float(1.0 / np.median(br_in)),
    }

    # 1T linear interpolation distortion vs StandaloneNMOS
    fwd_raw = load_default_lookup()
    t_audit = {}
    for g, s_idx, obj_id in [
        (0.9, 5, 2),
        (1.0, 6, 3),
        (1.1, 7, 4),
        (1.2, 8, 5),
        (1.3, 9, 6),
        (1.4, 10, 7),
        (1.5, 11, 8),
        (5.0, 12, 10),
    ]:
        c_meta = {str(r[4]): r for r in idx["curves"][1:] if r[1] == obj_id}
        with np.load(CACHE_DIR / f"s{s_idx}.npz") as data:
            imaxes = [np.abs(data[k][:, 2]).max() * 1e6 for k in data.files if c_meta[k][6] == "正"]
        t_audit[str(g)] = {
            "iv41_measured_imax_p50_p90_uA": np.quantile(imaxes, [0.5, 0.9]).tolist(),
            "old_lookup_5V_uA": float(fwd_raw.current(5.0, g) * 1e6) if g <= 3.0 else None,
            "smooth_nmos_4p2V_calibrated_uA": float(
                nmos.drain_current(np.array([4.2]), g, scale=1.05, offset_v=-0.095)[0] * 1e6
            ),
        }

    # Plot 01: Data Pairing, Block Switching, 1T Loadline Fix, and V_set vs |V_rst|
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0), layout="constrained")

    # Panel A: Fig.2c broken pairing vs IV41 1.4V exact pairing
    ax = axes[0, 0]
    c_meta_14 = {str(r[4]): r for r in idx["curves"][1:] if r[1] == 7}
    with np.load(CACHE_DIR / "s10.npz") as d14:
        s14 = [low_bias_g015(*d14[k][:, 1:].T)[1] * 1e6 for k in sorted(d14.files, key=int) if c_meta_14[k][6] == "正"]
        r14 = [low_bias_g015(*d14[k][:, 1:].T)[0] * 1e6 for k in sorted(d14.files, key=int) if c_meta_14[k][6] == "负"]
    ax.scatter(
        np.asarray(h11_s_ret) * 1e6,
        np.asarray(h11_r_in) * 1e6,
        s=22,
        alpha=0.75,
        color="#c0392b",
        label=f"Historical Fig.2c (Cell 322, r={fig2c_corr_sr:+.2f}, 2/53 paired)",
    )
    ax.scatter(
        s14,
        r14,
        s=20,
        alpha=0.75,
        color="#1f6f8b",
        label=f"IV41 1.4 V raw order (r={iv41_pairing['1.4']['corr_set_ret_to_reset_in_lag0']:+.3f})",
    )
    grid_g = np.logspace(-2, 2.5, 50)
    ax.plot(grid_g, grid_g, "k--", lw=1.0, alpha=0.6, label="1:1 state continuity")
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="SET return G(0.15 V) (µS)",
        ylabel="Same-index RESET entry G(-0.15 V) (µS)",
        title="(a) Cycle-pairing audit: Fig.2c vs IV41",
    )
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.2)

    # Panel B: Contiguous H/l mode blocks & cross-cycle memory in IV41 1.4V and 1.5V
    ax = axes[0, 1]
    c_meta_15 = {str(r[4]): r for r in idx["curves"][1:] if r[1] == 8}
    with np.load(CACHE_DIR / "s11.npz") as d15:
        s15 = [low_bias_g015(*d15[k][:, 1:].T)[1] * 1e6 for k in sorted(d15.files, key=int) if c_meta_15[k][6] == "正"]
    ax.plot(np.arange(1, len(s14) + 1), s14, "-o", ms=3, lw=1.1, color="#8e44ad", label="IV41 1.4 V (5 runs)")
    ax.plot(np.arange(1, len(s15) + 1), s15, "-s", ms=3, lw=1.1, color="#16a085", label="IV41 1.5 V (2 runs)")
    ax.axhline(30.0, color="k", ls=":", lw=1.0, label="30 µS Filamentary/Distributed split")
    ax.axvline(70.7, color="#c0392b", ls="--", lw=1.0, label="Old 70/30 row split")
    ax.set(
        yscale="log",
        xlabel="Cycle index in IV41 sequence",
        ylabel="Post-SET G(0.15 V) (µS)",
        title="(b) Block regime switching vs 70/30 row cut",
    )
    ax.legend(fontsize=7.5, loc="lower left")
    ax.grid(alpha=0.2)

    # Panel C: 1T saturation current vs Gate Voltage (0.9-5.0 V)
    ax = axes[1, 0]
    gates_plot = [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 5.0]
    meas_p90 = [t_audit[str(g)]["iv41_measured_imax_p50_p90_uA"][1] for g in gates_plot]
    old_5v = [t_audit[str(g)]["old_lookup_5V_uA"] for g in gates_plot[:-1]]
    vg_fine = np.linspace(0.85, 5.05, 120)
    cal_curve = [
        float(nmos.drain_current(np.array([4.2]), g, scale=1.05, offset_v=-0.095)[0] * 1e6)
        for g in vg_fine
    ]
    ax.plot(gates_plot, meas_p90, "ko", ms=6, label="IV41 measured SET I_max (p90)")
    ax.plot(gates_plot[:-1], old_5v, "s--", color="#c0392b", ms=5, label="Old Fig.2b linear-interp I_1T(5 V)")
    ax.plot(vg_fine, cal_curve, "-", color="#1f6f8b", lw=1.8, label="Calibrated smooth NMOS (up to 5 V)")
    ax.set(
        xlabel="Gate voltage V_G (V)",
        ylabel="Peak current (µA)",
        title="(c) 1T loadline calibration & VG=5 V extension",
    )
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.2)

    # Panel D: SET threshold V_set vs preceding RESET stop magnitude |V_rst|
    ax = axes[1, 1]
    labels_d = ["0.9", "1.1", "1.3", "1.5", "Bare_1M", "5.0_IV41"]
    names_d = ["IV41 0.9V\n(-1.5V)", "IV41 1.1V\n(-1.5V)", "IV41 1.3V\n(-1.5V)", "IV41 1.5V\n(-1.5V)", "Bare 1M\n(-2.5V)", "IV41 5.0V\n(-3.0V)"]
    q_mat = np.array(
        [
            vset_vs_vrst[k].get("vset_50pct_lrs_q10_q50_q90_V")
            or vset_vs_vrst[k].get("vset_clamp_q10_q50_q90_V")
            for k in labels_d
        ]
    )
    x_pos = np.arange(len(labels_d))
    colors_d = ["#486d93"] * 4 + ["#c0392b", "#16a085"]
    ax.bar(x_pos, q_mat[:, 1], yerr=[q_mat[:, 1] - q_mat[:, 0], q_mat[:, 2] - q_mat[:, 1]], capsize=4, color=colors_d, alpha=0.82)
    ax.set_xticks(x_pos, names_d, fontsize=8)
    ax.set(
        ylabel="SET switching voltage V_set (V)",
        title="(d) Same-cell IV41 5V proves V_set scales with |V_reset|",
    )
    ax.grid(axis="y", alpha=0.2)

    fig.suptitle("Iteration 2026-11-25-02 | Data Pairing, Regime Blocks, 1T Loadline & RESET-Depth Audit", weight="bold")
    fig.savefig(ITER_DIR / "01_data_pairing_and_1t_audit.png", dpi=180)
    plt.close(fig)

    return {
        "fig2c_vs_cell322_audit": {
            "total_pairs": 53,
            "exact_same_cycle_pairs": len(exact_matches),
            "matching_pair_indices": exact_matches,
            "cycle_index_offset_min_median_max": [int(min(offsets)), float(np.median(offsets)), int(max(offsets))],
            "corr_set_ret_to_reset_in": fig2c_corr_sr,
            "corr_reset_end_to_next_set_in": fig2c_corr_rs,
        },
        "iv41_and_cell322_pairing": iv41_pairing,
        "bare_1m_pairing": {
            "n_cycles": len(bare_s),
            "corr_set_ret_to_reset_in": float(np.corrcoef(np.log10(bs_ret), np.log10(br_in))[0, 1]),
            "corr_reset_end_to_next_set_in": float(np.corrcoef(np.log10(br_end[:-1]), np.log10(bs_in[1:]))[0, 1]),
        },
        "vset_vs_reset_stop": vset_vs_vrst,
        "transistor_lookup_audit": t_audit,
        "standalone_nmos_parameters": asdict(nmos),
    }


def calibrate_set_variant(train: dict, nmos: StandaloneNMOS, *, ablation_isothermal: bool = False):
    """Calibrate either the full Field-Scaled ET-2Ch model or the isothermal raw-1T ablation."""
    fwd_raw = load_default_lookup()
    blocks = []
    for name, (items, circuit) in train.items():
        bare = name == "1M"
        if not bare:
            sample = np.linspace(0, len(items) - 1, min(15, len(items)), dtype=int)
            items = [items[idx] for idx in sample]
        voltage, actual, mask, initial = joint.observations(items, circuit)
        subsets = list(joint.regions(voltage, bare).values())
        raw_targets = [(mask & sub, 1.0) for sub in subsets if np.any(mask & sub)]
        top_w = 0.35 if bare else 0.65 / (len(train) - 1)
        tot_w = sum(w for _, w in raw_targets) + (3.0 if bare else 0.0)
        compiled = [
            (keep, np.log10(actual[keep]), w * top_w / (tot_w * keep.sum()))
            for keep, w in raw_targets
        ]
        onset_info = None
        if bare:
            pts = np.array([joint.clamp_indices(voltage, row, circuit.compliance_a) for row in actual])
            onset_info = (
                voltage[: int(voltage.argmax()) + 1],
                voltage[pts],
                3.0 * top_w / (tot_w * len(pts)),
            )
        raw_curve = None
        if not bare and ablation_isothermal:
            vd_g = np.linspace(0.0, 5.0, 251)
            id_g = np.array([fwd_raw.current(float(v), float(name)) for v in vd_g])
            raw_curve = (vd_g, id_g)
        ramp = FieldScaledET2ChSetRamp(
            voltage,
            None if (bare or ablation_isothermal) else nmos,
            None if bare else float(name),
            2.5 if bare else 1.5,
            step=0.04,
            raw_1t_curve=raw_curve,
            enable_vrst_scaling=not ablation_isothermal,
            enable_two_channel=not ablation_isothermal,
            enable_thermal=not ablation_isothermal,
        )
        blocks.append((ramp, initial, compiled, onset_info))

    def loss_fn(p):
        mse = 0.0
        for ramp, init, compiled, onset_info in blocks:
            cur, _, cur_unc, _, _ = ramp.predict(init, p, iterations=22)
            for keep, act, w in compiled:
                mse += w * float(np.sum((np.log10(np.maximum(cur[keep], 1e-18)) - act) ** 2))
            if onset_info is not None:
                v_out, v_meas, w_on = onset_info
                log_iu = np.maximum.accumulate(
                    np.log10(np.maximum(cur_unc[:, : len(v_out)], 1e-18)), axis=1
                )
                v_pred = np.array([np.interp(-3.0, row, v_out) for row in log_iu])
                mse += w_on * float(np.sum(((v_pred - v_meas) / 0.14) ** 2))
        return mse

    p0 = np.array(
        [
            8.0,
            np.log(65.0),
            0.66,
            np.log(8.0e-8),
            0.65,
            -8.5,
            3.2,
            3.8,
            np.log(6.0e4),
            9.0e-4,
            1.08,
            -0.095,
        ]
    )
    bounds = [
        (2.5, 13.5),
        (np.log(15.0), np.log(400.0)),
        (0.0, 2.0),
        (np.log(1e-9), np.log(2e-6)),
        (0.0, 0.98),
        (-16.0, 4.0),
        (1.5, 28.0),
        (0.5, 18.0),
        (np.log(1e3), np.log(1e7)),
        (0.0, 2e-2),
        (0.55, 1.20),
        (-0.30, 0.15),
    ]
    maxfev = 180 if ablation_isothermal else 460
    sol = minimize(loss_fn, p0, method="Powell", bounds=bounds, options={"maxfev": maxfev, "ftol": 5e-4})
    return sol.x, float(sol.fun), int(sol.nfev)


def run_part2_set_and_bridge(idx: dict, nmos: StandaloneNMOS) -> tuple[dict, np.ndarray]:
    """Fit unified SET + ablation, evaluate held-out benchmark, blocked splits, VG=5V bridge, and 15-cell D2D."""
    prior_json = json.loads((EVIDENCE_DIR / "single_gap_series_set.json").read_text())["fit"]
    jgw_full = json.loads((EVIDENCE_DIR / "joint_gap_width.json").read_text())
    jgw_params = np.array(
        [
            jgw_full["fit"]["effective_barrier_eV"],
            np.log(jgw_full["fit"]["resistance_ohm"]),
            np.log(jgw_full["fit"]["closed_low_bias_conductance_S"]),
            jgw_full["fit"]["log_rate_V_inv"],
            jgw_full["fit"]["field_sensitivity_nm_per_V"],
        ]
    )
    old_params = np.array(
        [
            prior_json["beta_per_V"],
            np.log(prior_json["resistance_ohm"]),
            *prior_json["log_growth_rate_field_slope_capacity"],
        ]
    )
    train, test, _, fallback = joint.load_split(CACHE_DIR, old_params[0])

    x_ablation, loss_abl, nfev_abl = calibrate_set_variant(train, nmos, ablation_isothermal=True)
    x_main, loss_main, nfev_main = calibrate_set_variant(train, nmos, ablation_isothermal=False)

    fwd_raw = load_default_lookup()
    heldout_scores = {}
    plot_panels = {}

    for name, (items, circuit) in test.items():
        bare = name == "1M"
        voltage, actual, mask, initial = joint.observations(items, circuit)
        # Baseline 1: Prior 1T1M-only GapRamp
        prior_cur = joint.GapRamp(voltage, circuit, 0.0125).predict(initial, old_params)[0]
        # Baseline 2: Joint Gap-Width
        jgw_cur = jgw.WidthRamp(voltage, circuit, 0.0125).predict(initial, jgw_params)[0]
        # Ablation: Isothermal single-channel + raw 1T
        raw_curve = None
        if not bare:
            vd_g = np.linspace(0.0, 5.0, 251)
            raw_curve = (vd_g, np.array([fwd_raw.current(float(v), float(name)) for v in vd_g]))
        abl_ramp = FieldScaledET2ChSetRamp(
            voltage,
            None,
            None if bare else float(name),
            2.5 if bare else 1.5,
            step=0.0125,
            raw_1t_curve=raw_curve,
            enable_vrst_scaling=False,
            enable_two_channel=False,
            enable_thermal=False,
        )
        abl_cur = abl_ramp.predict(initial, x_ablation)[0]
        # Main candidate: Field-Scaled ET-2Ch (step=0.0125 and step=0.00625 refinement check)
        main_ramp = FieldScaledET2ChSetRamp(
            voltage,
            None if bare else nmos,
            None if bare else float(name),
            2.5 if bare else 1.5,
            step=0.0125,
        )
        main_cur, vh, _, _, v1t = main_ramp.predict(initial, x_main, iterations=26)
        fine_cur = FieldScaledET2ChSetRamp(
            voltage,
            None if bare else nmos,
            None if bare else float(name),
            2.5 if bare else 1.5,
            step=0.00625,
        ).predict(initial, x_main, iterations=26)[0]

        # Check KCL voltage balance on unclamped points
        rc = float(np.exp(x_main[1]))
        bound = 1e-3 if bare else nmos.monotone_table(float(name), x_main[10], x_main[11], vmax=6.0)[1][-1]
        bal_mask = mask & (main_cur < 0.998 * bound)
        v_sum = vh + rc * main_cur + v1t
        max_kcl_err = float(np.max(np.abs((v_sum - voltage)[bal_mask]))) if np.any(bal_mask) else 0.0
        max_ref_dec = float(np.max(np.abs(np.log10(main_cur[mask] / fine_cur[mask]))))

        subsets = {
            "all_unclamped": np.ones_like(voltage, dtype=bool),
            "outward": np.arange(len(voltage)) <= voltage.argmax(),
            **joint.regions(voltage, bare),
        }
        candidates = {
            "prior_1t1m_only": prior_cur,
            "joint_gap_width": jgw_cur,
            "ablation_isothermal_raw1t": abl_cur,
            "field_scaled_et2ch": main_cur,
        }
        reg_scores = {}
        for skey, sub in subsets.items():
            use = mask & sub
            if np.any(use):
                reg_scores[skey] = {
                    "points": int(use.sum()),
                    **{
                        cname: float(np.sqrt(np.mean(np.log10(cval[use] / actual[use]) ** 2)))
                        for cname, cval in candidates.items()
                    },
                }

        entry = {
            "scans": len(items),
            "rmse_decade": reg_scores,
            "max_step_refinement_decade": max_ref_dec,
            "max_kcl_voltage_error_V": max_kcl_err,
        }
        if bare:
            onset_dict = {}
            for cname, cval in {"measured": actual, **candidates}.items():
                idxs = [joint.clamp_indices(voltage, r, 1e-3) for r in cval]
                onset_dict[cname] = [float(voltage[i]) if i is not None else None for i in idxs]
            entry["clamp_onset_V"] = onset_dict
            entry["clamp_onset_mae_V"] = {
                cname: (
                    float(
                        np.mean(
                            [abs(a - b) for a, b in zip(onset_dict["measured"], onset_dict[cname]) if b is not None]
                        )
                    )
                    if any(b is not None for b in onset_dict[cname])
                    else None
                )
                for cname in candidates
            }
        if not bare and float(name) in (1.3, 1.4, 1.5):
            # Blocked / regime-separated evaluation (Distributed l vs Filamentary H)
            g_ret_015 = np.array([reads.terminal_estimate(row, False, 0.15) for _, row, _ in items])
            out_mask = mask & (np.arange(len(voltage)) <= voltage.argmax())
            regime_split = {}
            for rlabel, rkeep in (("distributed_le30uS", g_ret_015 <= 30.0), ("filamentary_gt30uS", g_ret_015 > 30.0)):
                if np.any(rkeep):
                    use_r = out_mask[rkeep]
                    regime_split[rlabel] = {
                        "scans": int(rkeep.sum()),
                        "joint_gap_width_outward_rmse": float(
                            np.sqrt(np.mean(np.log10(jgw_cur[rkeep][use_r] / actual[rkeep][use_r]) ** 2))
                        ),
                        "field_scaled_et2ch_outward_rmse": float(
                            np.sqrt(np.mean(np.log10(main_cur[rkeep][use_r] / actual[rkeep][use_r]) ** 2))
                        ),
                    }
            if float(name) == 1.5:
                at_1p4 = int(np.argmin(np.abs(voltage[: int(voltage.argmax()) + 1] - 1.4)))
                low_keep = (g_ret_015 <= 30.0) & mask[:, at_1p4]
                regime_split["low_group_current_at_1p4V_median_uA"] = {
                    "measured": float(np.median(actual[low_keep, at_1p4]) * 1e6),
                    "prior_1t1m_only": float(np.median(prior_cur[low_keep, at_1p4]) * 1e6),
                    "joint_gap_width": float(np.median(jgw_cur[low_keep, at_1p4]) * 1e6),
                    "field_scaled_et2ch": float(np.median(main_cur[low_keep, at_1p4]) * 1e6),
                }
            entry["regime_separated"] = regime_split

        heldout_scores[name] = entry
        if name in ("1M", "0.9", "1.1", "1.4", "1.5"):
            plot_panels[name] = (voltage, mask, actual, prior_cur, jgw_cur, main_cur)

    # Zero-shot Bridge Test on Same-Cell IV41 VG = 5.0 V (s12.npz, V_rst = -3.0 V)
    c_meta_5v = {str(r[4]): r for r in idx["curves"][1:] if r[1] == 10}
    v5_curves, v5_inits, v5_meas_vset = [], [], []
    with np.load(CACHE_DIR / "s12.npz") as d5:
        for k in sorted(d5.files, key=int):
            if c_meta_5v[k][6] != "正":
                continue
            _, v, i = d5[k].T
            ap = int(np.argmax(v))
            vo, io = v[: ap + 1], np.abs(i[: ap + 1])
            # Initial conductance at terminal 0.5V (since VG=5V 1T R_on ~ 1.5 kOhm << hBN pre-SET R ~ 50 kOhm)
            g05 = max(float(np.interp(0.5, vo, io) / 0.5), 1e-11)
            v5_curves.append(np.abs(i))
            v5_inits.append(g05)
            vs = reach_half_lrs_v(v, i)
            if vs is not None:
                v5_meas_vset.append(vs)
        v5_grid = d5[sorted([k for k in d5.files if c_meta_5v[k][6] == "正"], key=int)[0]][:, 1]

    v5_actual = np.stack(v5_curves)
    v5_inits_arr = np.asarray(v5_inits)
    # Predict zero-shot with V_rst = 3.0 V vs counterfactual V_rst = 1.5 V!
    ramp_5v_true = FieldScaledET2ChSetRamp(v5_grid, nmos, 5.0, 3.0, step=0.015)
    ramp_5v_no_vrst = FieldScaledET2ChSetRamp(v5_grid, nmos, 5.0, 1.5, step=0.015)
    pred_5v_true = ramp_5v_true.predict(v5_inits_arr, x_main, iterations=26)[0]
    pred_5v_novrst = ramp_5v_no_vrst.predict(v5_inits_arr, x_main, iterations=26)[0]

    ap5 = int(np.argmax(v5_grid))
    out5_mask = (np.arange(len(v5_grid)) <= ap5) & (v5_grid > 0.55) & (v5_actual < 0.72e-3)
    ret5_mask = (np.arange(len(v5_grid)) > ap5) & (v5_grid >= 0.15)
    pred_5v_vsets = [
        v for row in pred_5v_true if (v := reach_half_lrs_v(v5_grid, row)) is not None
    ]
    pred_5v_novrst_vsets = [
        v for row in pred_5v_novrst if (v := reach_half_lrs_v(v5_grid, row)) is not None
    ]
    bridge_5v_report = {
        "scans": len(v5_curves),
        "protocol": "100% zero-shot out-of-sample test on IV41 VG=5.0 V (V_reset=-3.0 V); old Fig.2b lookup cannot run at VG>3.0 V.",
        "outward_unclamped_rmse_decade": {
            "field_scaled_et2ch_vrst_3p0V": float(
                np.sqrt(np.mean(np.log10(pred_5v_true[out5_mask] / v5_actual[out5_mask]) ** 2))
            ),
            "counterfactual_vrst_1p5V": float(
                np.sqrt(np.mean(np.log10(pred_5v_novrst[out5_mask] / v5_actual[out5_mask]) ** 2))
            ),
        },
        "return_rmse_decade": float(
            np.sqrt(np.mean(np.log10(pred_5v_true[:, ret5_mask] / v5_actual[:, ret5_mask]) ** 2))
        ),
        "vset_50pct_lrs_median_V": {
            "measured": float(np.median(v5_meas_vset)),
            "predicted_vrst_3p0V": float(np.median(pred_5v_vsets)),
            "counterfactual_vrst_1p5V": float(np.median(pred_5v_novrst_vsets)),
        },
        "peak_current_at_6p5V_median_uA": {
            "measured": float(np.median(v5_actual[:, ap5]) * 1e6),
            "predicted": float(np.median(pred_5v_true[:, ap5]) * 1e6),
        },
    }

    # Plot 02: 6-panel Unified SET I-V Comparison (Bare 1M, 0.9V, 1.1V, 1.4V, 1.5V, and Zero-Shot IV41 5.0V)
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 7.6), layout="constrained")
    panel_order = ["1M", "0.9", "1.1", "1.4", "1.5"]
    for ax, key in zip(axes.flat[:5], panel_order):
        v, m, obs, p_old, p_jgw, p_new = plot_panels[key]
        ap = int(v.argmax())
        out = np.arange(len(v)) <= ap
        show_m = (np.any(m, axis=0) | ((key == "1M") & (v > 2.4))) & out
        show_p = (v > 0.55) & out
        q10, q50, q90 = np.quantile(obs[:, show_m], [0.1, 0.5, 0.9], axis=0) * 1e6
        ax.fill_between(v[show_m], q10, q90, color=".78", alpha=0.45, label="Measured p10–p90")
        ax.plot(v[show_m], q50, color=".12", lw=1.7, label="Measured median")
        ax.plot(v[show_p], np.median(p_old[:, show_p], axis=0) * 1e6, "--", color="#486d93", lw=1.35, label="Prior 1T1M-only (23 kΩ)")
        ax.plot(v[show_p], np.median(p_jgw[:, show_p], axis=0) * 1e6, ":", color="#7f8c8d", lw=1.5, label="Prev Joint Gap-Width")
        ax.plot(v[show_p], np.median(p_new[:, show_p], axis=0) * 1e6, "-", color="#b8253a", lw=1.8, label="Field-Scaled ET-2Ch (Ours)")
        rmse_j = heldout_scores[key]["rmse_decade"]["outward"]["joint_gap_width"]
        rmse_n = heldout_scores[key]["rmse_decade"]["outward"]["field_scaled_et2ch"]
        title_str = "Bare 1M (held-out)" if key == "1M" else f"1T1M · {key} V (held-out)"
        ax.set(
            title=f"{title_str}\nRMSE: {rmse_j:.2f} → {rmse_n:.2f} dec",
            yscale="log",
            xlabel="Applied SET voltage (V)",
            ylabel="Current magnitude (µA)",
        )
        ax.grid(alpha=0.2)

    # 6th panel: Zero-shot IV41 VG = 5.0 V bridge!
    ax = axes.flat[5]
    out5 = np.arange(len(v5_grid)) <= ap5
    show5 = (v5_grid >= 0.55) & out5
    q10_5, q50_5, q90_5 = np.quantile(v5_actual[:, show5], [0.1, 0.5, 0.9], axis=0) * 1e6
    ax.fill_between(v5_grid[show5], q10_5, q90_5, color=".78", alpha=0.45)
    ax.plot(v5_grid[show5], q50_5, color=".12", lw=1.7, label="Measured median")
    ax.plot(
        v5_grid[show5],
        np.median(pred_5v_novrst[:, show5], axis=0) * 1e6,
        "--",
        color="#7f8c8d",
        lw=1.35,
        label="No V_rst scaling (-1.5V)",
    )
    ax.plot(
        v5_grid[show5],
        np.median(pred_5v_true[:, show5], axis=0) * 1e6,
        "-",
        color="#16a085",
        lw=1.8,
        label="Zero-shot ET-2Ch (-3.0V)",
    )
    ax.set(
        title=f"IV41 · 5.0 V Bridge (Zero-Shot)\nV_set med: meas {bridge_5v_report['vset_50pct_lrs_median_V']['measured']:.2f}V vs pred {bridge_5v_report['vset_50pct_lrs_median_V']['predicted_vrst_3p0V']:.2f}V",
        yscale="log",
        xlabel="Applied SET voltage (V)",
        ylabel="Current magnitude (µA)",
    )
    ax.legend(fontsize=7.2, loc="lower right")
    ax.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=7.2, loc="lower right")

    fig.suptitle(
        "Unified SET Across Bare 1M, 0.9–1.5 V 1T1M, and Zero-Shot IV41 5.0 V Bridge",
        weight="bold",
    )
    fig.savefig(ITER_DIR / "02_unified_set_iv_comparison.png", dpi=180)
    plt.close(fig)

    param_names = [
        "beta0_per_V",
        "R_contact_ohm",
        "gamma_vrst_per_V",
        "G_res_thresh_S",
        "res_ohmic_fraction",
        "log_k_set",
        "s_field_per_V",
        "z_capacity",
        "R_th_K_per_W",
        "c_surge_coeff",
        "transistor_current_scale",
        "transistor_gate_offset_V",
    ]
    decoded = [
        float(x_main[0]),
        float(np.exp(x_main[1])),
        float(x_main[2]),
        float(np.exp(x_main[3])),
        float(x_main[4]),
        float(x_main[5]),
        float(x_main[6]),
        float(x_main[7]),
        float(np.exp(x_main[8])),
        float(x_main[9]),
        float(x_main[10]),
        float(x_main[11]),
    ]
    return {
        "protocol": (
            "Exact chronological/workbook splits and point masks from joint_gap_width.json: "
            "first 14 Bare 1M cycles and first 70% scans at 0.9-1.3 V for calibration; "
            "last 8 Bare 1M, last 30% at 0.9-1.3 V, all 101 scans at 1.4 V, all 101 scans at 1.5 V, "
            "and all 141 scans at IV41 5.0 V held out. Single scalar 0.5 V initial read per scan; "
            "known measurement RESET stop voltage (1.5 V, 2.5 V, 3.0 V) enters barrier scaling lambda."
        ),
        "bare_anchor_fallback_cycles": fallback,
        "fitted_parameters": dict(zip(param_names, decoded)),
        "raw_parameter_vector": x_main.tolist(),
        "optimization": {
            "main_loss": loss_main,
            "main_nfev": nfev_main,
            "ablation_isothermal_loss": loss_abl,
            "ablation_isothermal_nfev": nfev_abl,
        },
        "heldout_benchmark": heldout_scores,
        "iv41_5v_zero_shot_bridge": bridge_5v_report,
    }, x_main


def run_part3_reset_ensemble_and_d2d(idx: dict, nmos: StandaloneNMOS, x_set: np.ndarray) -> dict:
    """Evaluate paired Electro-Thermal RESET, Stochastic Bimodal Ensemble, and 15-cell 1.1V D2D transfer."""
    rc_ohm = float(np.exp(x_set[1]))
    rev_lookup = load_reverse_lookup()

    def rev_curve_at(gate_v: float):
        vd = np.linspace(0.0, 2.0, 201)
        id_c = np.array([rev_lookup.current(float(v), float(gate_v)) for v in vd])
        return vd, id_c

    # Load paired IV41 SET+RESET (0.9..1.5V) + Bare 1M SET+RESET
    groups_1m, _ = shared.load_groups()
    bare_set_rows = groups_1m[("1M", None, "SET")]
    bare_rst_rows = groups_1m[("1M", None, "RESET")]

    iv41_cohorts = {}
    for g in (0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5):
        coh, _ = iv.load_cohort(idx, CACHE_DIR, round((g - 0.9) * 10) + 5, g)
        iv41_cohorts[g] = coh

    # Fit unified Electro-Thermal RESET on training pairs: Bare 1M [:14] + IV41 0.9, 1.1, 1.3 [:70%]
    rst_train_blocks = []
    b_v = bare_rst_rows[0].voltage
    b_obs = np.abs(np.stack([r.current for r in bare_rst_rows[:14]]))
    b_mask = np.stack([r.masks["use_unclamped"] for r in bare_rst_rows[:14]])
    b_gin = np.array([low_bias_g015(s.voltage, s.current)[1] for s in bare_set_rows[:14]])
    b_gend = np.array([low_bias_g015(r.voltage, r.current)[1] for r in bare_rst_rows[:14]])
    rst_train_blocks.append((ElectroThermalResetRamp(b_v, None, rc_ohm, step=0.02), b_gin, b_gend, b_obs, b_mask, 1e-11, 0.30))

    for g in (0.9, 1.1, 1.3):
        s_rows = iv41_cohorts[g]["SET"]
        r_rows = iv41_cohorts[g]["RESET"]
        n_tr = int(0.7 * min(len(s_rows), len(r_rows)))
        idx_sub = np.linspace(0, n_tr - 1, 20, dtype=int)
        rv = r_rows[0].voltage
        robs = np.abs(np.stack([r_rows[i].current for i in idx_sub]))
        rmask = np.stack([r_rows[i].masks["use_unclamped"] for i in idx_sub])
        # Use measured RESET entry read (-0.15 V) on training to decouple erasure law from small polarity offset
        rgin = np.array([low_bias_g015(r_rows[i].voltage, r_rows[i].current)[0] for i in idx_sub])
        rgend = np.array([low_bias_g015(r_rows[i].voltage, r_rows[i].current)[1] for i in idx_sub])
        rst_train_blocks.append(
            (ElectroThermalResetRamp(rv, rev_curve_at(g), rc_ohm, step=0.02), rgin, rgend, robs, rmask, 3e-12, 0.70 / 3.0)
        )

    def rst_loss(p):
        tot = 0.0
        for ramp, gin, gend, obs, msk, gfl, w in rst_train_blocks:
            pred, _, g_traj = ramp.predict(gin, gfl, p, enable_thermal=True)
            curve_mse = float(np.mean((np.log10(pred[msk] / obs[msk])) ** 2))
            end_mse = float(np.mean((np.log10(np.maximum(g_traj[:, -1], 1e-14) / gend)) ** 2))
            tot += w * (0.75 * curve_mse + 0.25 * end_mse)
        return tot

    p0_rst = np.array([3.2, -0.5, 8.5, np.log(2.2e5), 0.52])
    bnds_rst = [(0.8, 8.5), (-8.0, 6.0), (1.0, 24.0), (np.log(2e3), np.log(5e6)), (0.15, 1.15)]
    sol_rst = minimize(rst_loss, p0_rst, method="Powell", bounds=bnds_rst, options={"maxfev": 280, "ftol": 5e-4})
    x_rst = sol_rst.x

    # Evaluate RESET on held-out Bare 1M [14:] and IV41 (0.9..1.5V) alongside Deterministic Floor!
    reset_results = {}
    # Bare 1M held-out [14:]
    b_obs_te = np.abs(np.stack([r.current for r in bare_rst_rows[14:]]))
    b_msk_te = np.stack([r.masks["use_unclamped"] for r in bare_rst_rows[14:]])
    b_gin_paired = np.array([low_bias_g015(s.voltage, s.current)[1] for s in bare_set_rows[14:]])
    b_gin_wave = np.full(len(b_gin_paired), float(np.median(b_gin)))
    b_ramp = ElectroThermalResetRamp(b_v, None, rc_ohm, step=0.01)
    b_pred_paired, _, _ = b_ramp.predict(b_gin_paired, 1e-11, x_rst, enable_thermal=True)
    b_pred_wave, _, _ = b_ramp.predict(b_gin_wave, 1e-11, x_rst, enable_thermal=True)
    b_pred_isoth, _, _ = b_ramp.predict(b_gin_paired, 1e-11, x_rst, enable_thermal=False)
    b_floor = shared.common_curve_floor(bare_rst_rows[14:])["use_unclamped"]
    reset_results["Bare_1M"] = {
        "cycles": len(bare_rst_rows[14:]),
        "deterministic_floor_decade": float(b_floor),
        "waveform_only_rmse_decade": float(np.sqrt(np.mean(np.log10(b_pred_wave[b_msk_te] / b_obs_te[b_msk_te]) ** 2))),
        "paired_set_handoff_rmse_decade": float(np.sqrt(np.mean(np.log10(b_pred_paired[b_msk_te] / b_obs_te[b_msk_te]) ** 2))),
        "paired_isothermal_ablation_rmse_decade": float(np.sqrt(np.mean(np.log10(b_pred_isoth[b_msk_te] / b_obs_te[b_msk_te]) ** 2))),
    }

    iv41_rst_plot_data = {}
    for g in (0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5):
        s_rows = iv41_cohorts[g]["SET"]
        r_rows = iv41_cohorts[g]["RESET"]
        n_pair = min(len(s_rows), len(r_rows))
        cut = int(0.7 * n_pair) if g <= 1.3 else 0
        s_te = s_rows[cut:n_pair]
        r_te = r_rows[cut:n_pair]
        rv = r_te[0].voltage
        robs = np.abs(np.stack([r.current for r in r_te]))
        rmsk = np.stack([r.masks["use_unclamped"] for r in r_te])
        gin_paired = np.array([low_bias_g015(s.voltage, s.current)[1] for s in s_te])
        # Waveform-only entry uses median SET return from training/predicted baseline
        gin_wave = np.full(len(s_te), float(np.median([low_bias_g015(s.voltage, s.current)[1] for s in s_rows[: max(cut, 10)]])))
        rramp = ElectroThermalResetRamp(rv, rev_curve_at(g), rc_ohm, step=0.01)
        p_paired, _, _ = rramp.predict(gin_paired, 3e-12, x_rst, enable_thermal=True)
        p_wave, _, _ = rramp.predict(gin_wave, 3e-12, x_rst, enable_thermal=True)
        fl = shared.common_curve_floor(r_te)["use_unclamped"]
        reset_results[str(g)] = {
            "cycles": len(r_te),
            "deterministic_floor_decade": float(fl),
            "waveform_only_rmse_decade": float(np.sqrt(np.mean(np.log10(p_wave[rmsk] / robs[rmsk]) ** 2))),
            "paired_set_handoff_rmse_decade": float(np.sqrt(np.mean(np.log10(p_paired[rmsk] / robs[rmsk]) ** 2))),
        }
        if g in (1.1, 1.4):
            iv41_rst_plot_data[g] = (np.abs(rv), rmsk, robs, p_wave, p_paired)

    # Plot 03: Electro-Thermal RESET (Bare 1M thermal rupture + Paired IV41 vs Floor)
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.3), layout="constrained")
    # Panel A: Bare 1M RESET outward & return
    ax = axes[0]
    bv_abs = np.abs(b_v)
    b_ap = int(bv_abs.argmax())
    for sl, ls, lbl_sfx in ((slice(None, b_ap + 1), "-", "out"), (slice(b_ap, None), "--", "ret")):
        q10, q50, q90 = np.quantile(b_obs_te[:, sl], [0.1, 0.5, 0.9], axis=0) * 1e6
        if lbl_sfx == "out":
            ax.fill_between(bv_abs[sl], q10, q90, color=".80", alpha=0.45, label="Measured p10–p90")
        ax.plot(bv_abs[sl], q50, ls, color=".12", lw=1.7, label="Measured median" if lbl_sfx == "out" else None)
        ax.plot(
            bv_abs[sl],
            np.median(b_pred_isoth[:, sl], axis=0) * 1e6,
            ls,
            color="#486d93",
            lw=1.35,
            label="Isothermal ablation (no R_th)" if lbl_sfx == "out" else None,
        )
        ax.plot(
            bv_abs[sl],
            np.median(b_pred_paired[:, sl], axis=0) * 1e6,
            ls,
            color="#b8253a",
            lw=1.8,
            label="Electro-Thermal RESET" if lbl_sfx == "out" else None,
        )
    ax.set(
        yscale="log",
        xlabel="Bare 1M |V_reset| (V)",
        ylabel="Current magnitude (µA)",
        title=f"(a) Bare 1M RESET (8 held-out)\nIsoth {reset_results['Bare_1M']['paired_isothermal_ablation_rmse_decade']:.2f} → ET {reset_results['Bare_1M']['paired_set_handoff_rmse_decade']:.2f} dec",
    )
    ax.legend(fontsize=7.3)
    ax.grid(alpha=0.2)

    # Panel B: IV41 1.4 V Paired RESET (p10-p90 spread vs paired prediction)
    ax = axes[1]
    rv14, rmsk14, robs14, pwave14, ppair14 = iv41_rst_plot_data[1.4]
    ap14 = int(rv14.argmax())
    out14 = np.arange(len(rv14)) <= ap14
    q10, q50, q90 = np.quantile(robs14[:, out14], [0.1, 0.5, 0.9], axis=0) * 1e6
    pq10, pq50, pq90 = np.quantile(ppair14[:, out14], [0.1, 0.5, 0.9], axis=0) * 1e6
    ax.fill_between(rv14[out14], q10, q90, color=".80", alpha=0.45, label="Measured outward p10–p90")
    ax.fill_between(rv14[out14], pq10, pq90, color="#eab3bc", alpha=0.55, label="Paired ET model p10–p90")
    ax.plot(rv14[out14], q50, "-", color=".12", lw=1.7, label="Measured median")
    ax.plot(rv14[out14], pq50, "-", color="#b8253a", lw=1.8, label="Paired ET median")
    ax.set(
        yscale="log",
        xlabel="Reverse voltage |V_app| (V)",
        ylabel="Current magnitude (µA)",
        title=f"(b) IV41 1.4 V Paired RESET Outward\nFloor {reset_results['1.4']['deterministic_floor_decade']:.2f} | Paired {reset_results['1.4']['paired_set_handoff_rmse_decade']:.2f} dec",
    )
    ax.legend(fontsize=7.3)
    ax.grid(alpha=0.2)

    # Panel C: Summary bar chart — Waveform-only Floor vs Waveform-only vs Paired SET->RESET
    ax = axes[2]
    g_keys = ["0.9", "1.1", "1.3", "1.4", "1.5", "Bare_1M"]
    x_idx = np.arange(len(g_keys))
    w_bar = 0.26
    ax.bar(x_idx - w_bar, [reset_results[k]["deterministic_floor_decade"] for k in g_keys], w_bar, color="#95a5a6", label="Waveform-only floor (bound)")
    ax.bar(x_idx, [reset_results[k]["waveform_only_rmse_decade"] for k in g_keys], w_bar, color="#486d93", label="ET RESET (waveform-only)")
    ax.bar(x_idx + w_bar, [reset_results[k]["paired_set_handoff_rmse_decade"] for k in g_keys], w_bar, color="#b8253a", label="ET RESET (paired SET→RESET)")
    ax.set_xticks(x_idx, ["0.9V", "1.1V", "1.3V", "1.4V", "1.5V", "Bare 1M"], fontsize=8.5)
    ax.set(
        ylabel="RESET RMSE (decade)",
        title="(c) Paired SET→RESET vs waveform floor",
    )
    ax.legend(fontsize=7.3)
    ax.grid(axis="y", alpha=0.2)

    fig.suptitle("Iteration 2026-11-25-02 | Unified Electro-Thermal RESET & Paired State Handoff", weight="bold")
    fig.savefig(ITER_DIR / "03_electrothermal_reset_and_bare_loop.png", dpi=180)
    plt.close(fig)

    # Part 3B: Stochastic + Bimodal Markov Ensemble (seed = RNG_SEED)
    rng = np.random.default_rng(RNG_SEED)
    prior_json = json.loads((EVIDENCE_DIR / "single_gap_series_set.json").read_text())["fit"]
    old_p = np.array(
        [
            prior_json["beta_per_V"],
            np.log(prior_json["resistance_ohm"]),
            *prior_json["log_growth_rate_field_slope_capacity"],
        ]
    )
    train_set, test_set, _, _ = joint.load_split(CACHE_DIR, old_p[0])

    # Estimate C2C threshold std sigma_e (~0.11 V) from training SET switching spread
    sigma_e = 0.11
    n_mc = 160
    ensemble_metrics = {}
    mc_plot_cache = {}
    for key in ("1M", "0.9", "1.3", "1.4", "1.5"):
        bare = key == "1M"
        items, circuit = test_set[key]
        voltage, actual, mask, initial = joint.observations(items, circuit)
        # Sample initial states from low-bias pool + lognormal C2C perturbation
        sampled_idx = rng.integers(0, len(initial), size=n_mc)
        g0_mc = initial[sampled_idx] * (10.0 ** rng.normal(0.0, 0.18, size=n_mc))
        e_jitter = rng.normal(0.0, sigma_e, size=n_mc)

        # Markov 2-state chain (Distributed l <-> Filamentary H) for gate >= 1.3 V
        fil_mode = np.zeros(n_mc, dtype=float)
        if not bare and float(key) >= 1.25:
            # Stationary high-mode probability grows with gate current availability, persistent dwell p_stay=0.96
            p_enter = 0.035 * max(float(key) - 1.15, 0.0) / 0.25
            p_exit = 0.030
            state_h = int(rng.random() < (p_enter / (p_enter + p_exit)))
            for t in range(n_mc):
                if state_h == 0 and rng.random() < p_enter:
                    state_h = 1
                elif state_h == 1 and rng.random() < p_exit:
                    state_h = 0
                fil_mode[t] = float(state_h)

        ramp = FieldScaledET2ChSetRamp(
            voltage, None if bare else nmos, None if bare else float(key), 2.5 if bare else 1.5, step=0.015
        )
        mc_cur, _, _, _, _ = ramp.predict(g0_mc, x_set, iterations=24, field_shift=e_jitter, filament_mode=fil_mode, g_fil_high_s=1.35e-4)

        q10_mc, q50_mc, q90_mc = np.quantile(mc_cur, [0.10, 0.50, 0.90], axis=0)
        out_m = mask & (np.arange(len(voltage)) <= voltage.argmax())
        covered = (actual >= q10_mc[None, :]) & (actual <= q90_mc[None, :])
        cov_rate = float(np.sum(covered & out_m) / np.sum(out_m))

        # Switching voltage distribution comparison (Wasserstein-1 in Volts)
        if bare:
            vset_meas = np.array([voltage[joint.clamp_indices(voltage, r, 1e-3)] for r in actual])
            vset_sim = np.array([voltage[idx_c] for r in mc_cur if (idx_c := joint.clamp_indices(voltage, r, 1e-3)) is not None])
        else:
            vset_meas = np.array([v for r in actual if (v := reach_half_lrs_v(voltage, r)) is not None])
            vset_sim = np.array([v for r in mc_cur if (v := reach_half_lrs_v(voltage, r)) is not None])
        w1_vset = wasserstein_1d(vset_meas, vset_sim) if (len(vset_meas) and len(vset_sim)) else None

        # Post-SET G(0.15 V) distribution on return branch
        ap = int(voltage.argmax())
        vr = voltage[ap:][::-1]
        g_ret_meas = np.array([np.interp(0.15, vr, r[ap:][::-1]) / 0.15 * 1e6 for r in actual])
        g_ret_sim = np.array([np.interp(0.15, vr, r[ap:][::-1]) / 0.15 * 1e6 for r in mc_cur])
        w1_log_glrs = wasserstein_1d(np.log10(np.maximum(g_ret_meas, 1e-4)), np.log10(np.maximum(g_ret_sim, 1e-4)))

        ensemble_metrics[key] = {
            "mc_samples": n_mc,
            "outward_p10_p90_coverage": cov_rate,
            "vset_measured_q10_q50_q90_V": np.quantile(vset_meas, [0.1, 0.5, 0.9]).tolist(),
            "vset_simulated_q10_q50_q90_V": np.quantile(vset_sim, [0.1, 0.5, 0.9]).tolist() if len(vset_sim) else None,
            "vset_wasserstein1_V": w1_vset,
            "post_set_log10_G015_wasserstein1_decade": w1_log_glrs,
            "simulated_high_mode_fraction": float(np.mean(g_ret_sim > 30.0)),
            "measured_high_mode_fraction": float(np.mean(g_ret_meas > 30.0)),
        }
        if key in ("1M", "1.3", "1.4"):
            mc_plot_cache[key] = (voltage, mask, actual, q10_mc, q50_mc, q90_mc, vset_meas, vset_sim, g_ret_meas, g_ret_sim)

    # Part 3C: 15-Cell VG = 1.1 V (sheets 14..28) Hierarchical 1-DOF Few-Read D2D Adaptation
    d2d_cells = {}
    for s_idx in range(14, 29):
        sname = idx["names"][s_idx - 1].split("_", 1)[1]
        coh, _ = iv.load_cohort(idx, CACHE_DIR, s_idx, 1.1)
        s_rows = coh["SET"]
        if len(s_rows) < 12:
            continue
        v_c = s_rows[0].voltage
        ap_c = int(v_c.argmax())
        # Skip first 3 startup conditioning cycles (per project protocol) and use 5 scans [3:8] for 1-DOF adaptation
        cal_rows, te_rows = s_rows[3:8], s_rows[8:]
        g05_cal = np.array([max(float(np.interp(0.5, v_c[: ap_c + 1], np.abs(r.current[: ap_c + 1])) / 0.5), 1e-12) for r in cal_rows])
        g05_te = np.array([max(float(np.interp(0.5, v_c[: ap_c + 1], np.abs(r.current[: ap_c + 1])) / 0.5), 1e-12) for r in te_rows])
        obs_te = np.abs(np.stack([r.current for r in te_rows]))
        msk_te = np.stack([r.masks["use_unclamped"] for r in te_rows]) & (v_c > 0.55) & (np.arange(len(v_c)) <= ap_c)

        ramp_c = FieldScaledET2ChSetRamp(v_c, nmos, 1.1, 1.5, step=0.02)
        g05_pop = np.full(len(te_rows), 1.2e-8)
        pred_zero = ramp_c.predict(g05_pop, x_set)[0]
        obs_cal = np.abs(np.stack([r.current for r in cal_rows]))
        msk_cal = np.stack([r.masks["use_unclamped"] for r in cal_rows]) & (v_c > 0.55) & (np.arange(len(v_c)) <= ap_c)
        g05_med_cal = np.full(len(cal_rows), float(np.median(g05_cal)))
        best_de = minimize(
            lambda de: float(
                np.mean(
                    (np.log10(ramp_c.predict(g05_med_cal, x_set, field_shift=float(de[0]))[0][msk_cal] / obs_cal[msk_cal])) ** 2
                )
            ),
            [0.0],
            method="Powell",
            bounds=[(-0.65, 0.65)],
            options={"maxfev": 30},
        ).x[0]
        pred_5read = ramp_c.predict(np.full(len(te_rows), float(np.median(g05_cal))), x_set, field_shift=float(best_de))[0]
        pred_per_scan = ramp_c.predict(g05_te, x_set, field_shift=float(best_de))[0]
        fl_c = float(
            np.sqrt(
                np.mean(
                    (
                        np.log10(obs_te)
                        - np.mean(np.log10(np.maximum(obs_te, 1e-18)), axis=0, keepdims=True)
                    )[msk_te]
                    ** 2
                )
            )
        )
        d2d_cells[sname] = {
            "held_out_scans": len(te_rows),
            "deterministic_floor_decade": fl_c,
            "zero_shot_population_rmse_decade": float(np.sqrt(np.mean(np.log10(pred_zero[msk_te] / obs_te[msk_te]) ** 2))),
            "five_scan_1dof_adapted_rmse_decade": float(np.sqrt(np.mean(np.log10(pred_5read[msk_te] / obs_te[msk_te]) ** 2))),
            "adapted_plus_lowbias_read_rmse_decade": float(np.sqrt(np.mean(np.log10(pred_per_scan[msk_te] / obs_te[msk_te]) ** 2))),
            "fitted_cell_field_offset_V": float(best_de),
        }

    # Plot 04: Stochastic + Bimodal Ensemble & 15-Cell D2D Adaptation
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.8), layout="constrained")
    # Panel A: 1.4 V Stochastic p10-p90 Ensemble Band vs Measured p10-p90 Band
    ax = axes[0, 0]
    v14, m14, obs14, q10_14, q50_14, q90_14, vs_m14, vs_s14, gr_m14, gr_s14 = mc_plot_cache["1.4"]
    out14 = (np.arange(len(v14)) <= v14.argmax()) & (v14 >= 0.55)
    mq10, mq50, mq90 = np.quantile(obs14[:, out14], [0.1, 0.5, 0.9], axis=0) * 1e6
    ax.fill_between(v14[out14], mq10, mq90, color=".78", alpha=0.5, label="Measured p10–p90 (101 scans)")
    ax.fill_between(v14[out14], q10_14[out14] * 1e6, q90_14[out14] * 1e6, color="#eab3bc", alpha=0.55, label="MC Ensemble p10–p90")
    ax.plot(v14[out14], mq50, color=".12", lw=1.7, label="Measured median")
    ax.plot(v14[out14], q50_14[out14] * 1e6, color="#b8253a", lw=1.8, label="MC Ensemble median")
    ax.set(
        yscale="log",
        xlabel="Applied SET voltage (V)",
        ylabel="Current magnitude (µA)",
        title=f"(a) 1.4 V Stochastic Band (p10–p90 cov = {ensemble_metrics['1.4']['outward_p10_p90_coverage']*100:.1f}%)",
    )
    ax.legend(fontsize=7.4)
    ax.grid(alpha=0.2)

    # Panel B: V_set Switching Voltage CDFs (0.9 V, 1.4 V, Bare 1M)
    ax = axes[0, 1]
    for k_cdf, col, lbl in (("1.4", "#1f6f8b", "1T1M 1.4V"), ("1M", "#b8253a", "Bare 1M")):
        _, _, _, _, _, _, vm_c, vs_c, _, _ = mc_plot_cache[k_cdf]
        ax.plot(np.sort(vm_c), np.linspace(0, 1, len(vm_c)), "-", color=col, lw=1.8, label=f"{lbl} measured")
        ax.plot(np.sort(vs_c), np.linspace(0, 1, len(vs_c)), "--", color=col, lw=1.6, label=f"{lbl} MC model")
    ax.set(
        xlabel="SET switching threshold V_set (V)",
        ylabel="Empirical CDF",
        title=f"(b) V_set Distributions (W1: 1.4V={ensemble_metrics['1.4']['vset_wasserstein1_V']:.2f}V, 1M={ensemble_metrics['1M']['vset_wasserstein1_V']:.2f}V)",
    )
    ax.legend(fontsize=7.4)
    ax.grid(alpha=0.2)

    # Panel C: Bimodal Post-SET G(0.15 V) CDF on 1.3 V & 1.4 V (Markov Distributed <-> Filamentary)
    ax = axes[1, 0]
    for k_cdf, col, lbl in (("1.3", "#16a085", "1.3 V"), ("1.4", "#8e44ad", "1.4 V")):
        _, _, _, _, _, _, _, _, grm, grs = mc_plot_cache[k_cdf]
        ax.plot(np.sort(grm), np.linspace(0, 1, len(grm)), "-", color=col, lw=1.8, label=f"{lbl} measured")
        ax.plot(np.sort(grs), np.linspace(0, 1, len(grs)), "--", color=col, lw=1.6, label=f"{lbl} Markov+ET MC")
    ax.axvline(30.0, color="k", ls=":", lw=1.0, label="30 µS H/l boundary")
    ax.set(
        xscale="log",
        xlabel="Post-SET G(0.15 V) (µS)",
        ylabel="Empirical CDF",
        title="(c) Bimodal Distributed/Filamentary LRS Distribution",
    )
    ax.legend(fontsize=7.4)
    ax.grid(alpha=0.2)

    # Panel D: 15-Cell VG = 1.1 V D2D Generalization
    ax = axes[1, 1]
    c_names = list(d2d_cells.keys())
    xc = np.arange(len(c_names))
    ax.plot(xc, [d2d_cells[c]["zero_shot_population_rmse_decade"] for c in c_names], "o--", color="#7f8c8d", ms=4.5, label="Zero-shot (IV41 prior)")
    ax.plot(xc, [d2d_cells[c]["five_scan_1dof_adapted_rmse_decade"] for c in c_names], "s-", color="#1f6f8b", ms=4.5, label="5-scan 1-DOF cell adapted")
    ax.plot(xc, [d2d_cells[c]["adapted_plus_lowbias_read_rmse_decade"] for c in c_names], "^-", color="#b8253a", ms=5, label="Adapted + 0.5V read")
    ax.plot(xc, [d2d_cells[c]["deterministic_floor_decade"] for c in c_names], "k:", lw=1.2, label="Single-trace floor")
    c_labels = [c.replace("阵列_", "Arr_") for c in c_names]
    ax.set_xticks(xc, c_labels, rotation=40, ha="right", fontsize=7.5)
    ax.set(
        ylabel="Held-out SET outward RMSE (dec)",
        title="(d) Cross-Device Transfer Across 14 Array Cells (VG=1.1V)",
    )
    ax.legend(fontsize=7.0)
    ax.grid(alpha=0.2)

    fig.suptitle(
        "Iteration 2026-11-25-02 | Stochastic Bimodal Ensemble & Multi-Cell Hierarchical Transfer",
        weight="bold",
    )
    fig.savefig(ITER_DIR / "04_stochastic_bimodal_ensemble_and_d2d.png", dpi=180)
    plt.close(fig)

    return {
        "rng_seed": RNG_SEED,
        "electrothermal_reset": {
            "fitted_parameters": {
                "beta_reset_per_V": float(x_rst[0]),
                "log_k_reset": float(x_rst[1]),
                "s_field_reset_per_V": float(x_rst[2]),
                "R_th_reset_K_per_W": float(np.exp(x_rst[3])),
                "v_onset_V": float(x_rst[4]),
            },
            "heldout_scores": reset_results,
        },
        "stochastic_bimodal_ensemble": ensemble_metrics,
        "fifteen_cell_1p1V_d2d_transfer": {
            "summary_median_rmse_decade": {
                "deterministic_floor": float(np.median([v["deterministic_floor_decade"] for v in d2d_cells.values()])),
                "zero_shot_population": float(np.median([v["zero_shot_population_rmse_decade"] for v in d2d_cells.values()])),
                "five_scan_1dof_adapted": float(np.median([v["five_scan_1dof_adapted_rmse_decade"] for v in d2d_cells.values()])),
                "adapted_plus_lowbias_read": float(np.median([v["adapted_plus_lowbias_read_rmse_decade"] for v in d2d_cells.values()])),
            },
            "per_cell": d2d_cells,
        },
    }


def main():
    t_start = time.time()
    ITER_DIR.mkdir(parents=True, exist_ok=True)
    idx = json.loads((CACHE_DIR / "index.json").read_text(encoding="utf-8"))
    nmos = StandaloneNMOS.fit_fig2b(HANDOFF_ROOT / "vulcan2d/data/fig2b_transistor_lookup.json")
    print(f"[1/3] Standalone NMOS fitted (RMSE={nmos.fit_rmse_ua:.2f} uA). Running data & circuit audit...", flush=True)

    audit_report = run_part1_audit(idx, nmos)
    (ITER_DIR / "data_and_circuit_audit.json").write_text(
        json.dumps(audit_report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"[2/3] Audit saved ({time.time() - t_start:.1f}s). Calibrating Unified Field-Scaled ET-2Ch SET & VG=5V bridge...", flush=True)

    set_report, x_set = run_part2_set_and_bridge(idx, nmos)
    set_report["provenance_sha256"] = {
        "iv41_workbook": sha256_file(WORKBOOK_IV41),
        "iv41_cache_index": sha256_file(CACHE_DIR / "index.json"),
        "bare_set_workbook": sha256_file(HANDOFF_ROOT / "1R数据.xlsx"),
        "bare_reset_workbook": sha256_file(HANDOFF_ROOT / "裸RRESET.xlsx"),
    }
    (ITER_DIR / "unified_set_ablation_and_bridge.json").write_text(
        json.dumps(set_report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"[3/3] SET & 5V bridge saved ({time.time() - t_start:.1f}s). Running paired ET-RESET, MC ensemble & 15-cell D2D...", flush=True)

    cycle_report = run_part3_reset_ensemble_and_d2d(idx, nmos, x_set)
    (ITER_DIR / "closed_loop_reset_and_ensemble.json").write_text(
        json.dumps(cycle_report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"All complete in {time.time() - t_start:.1f}s! Artifacts written to {ITER_DIR}", flush=True)


if __name__ == "__main__":
    main()
