"""End-to-end calibration, full GPT-protocol verification, and archiving for Iteration 2026-11-27-02 (v4.3).

Validates the de-bloated 12-parameter closed-cycle model (v4.3) against BOTH v4.1 (12 params)
and v4.2 (14 params) across every single metric from the 2026-11-27 independent audit:
1. Held-out SET across Bare 1M + 7 gates (0.9-1.5 V) + merged 288,608-point 1T1M RMSE;
2. +0.25 V SET return readback median current & RMSE across all 8 groups;
3. Autonomous predicted_SET_end -> RESET vs measured_SET_return -> RESET across all 8 groups +
   merged 104,740-point 1T1M RESET RMSE (with ZERO hardcoded Bare/1T1M floor split!);
4. Both SET and RESET step-halving stability;
5. Zero-shot IV41 5.0 V SET outward & return;
6. Frozen cross-cell replay across all 15 exported 1.1 V device groups (2,156 scans).
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.optimize import minimize

from analysis import refit_iv41 as iv
from analysis import validate_joint_gap_set as joint
from analysis.et_two_channel_model import StandaloneNMOS
from analysis.unified_compact_v43 import NMOS_GATE_OFFSET_V, UnifiedCompactV43
from vulcan2d import validate_shared_device as shared
from vulcan2d.transistor import load_reverse_lookup

REPO_ROOT = Path(__file__).resolve().parents[1]
HANDOFF_DATA_ROOT = REPO_ROOT.parent / "VULCAN-2D-handoff-2026-11-24"
CACHE_DIR = REPO_ROOT.parent / "outputs" / "vulcan-handoff" / "iv41_cache"
WORKBOOK_IV41 = Path("/Users/sujiangyu/Desktop/VULCAN-2D/不同栅压_原始曲线数据.xlsx")
AUDIT_V41_DIR = REPO_ROOT.parent / "outputs" / "VULCAN-2D-v4.1-只读测评-2026-11-27"
ITER06_JSON = (
    REPO_ROOT
    / "docs"
    / "model-exploration"
    / "iterations"
    / "2026-11-27-01-回程坍塌与全自动整圈交接"
    / "iteration_2026_11_27_01_results.json"
)
EVIDENCE_DIR = REPO_ROOT / "docs" / "model-exploration" / "evidence"
ITER07_DIR = (
    REPO_ROOT
    / "docs"
    / "model-exploration"
    / "iterations"
    / "2026-11-27-02-十二参数精简整圈闭环模型"
)


def sha256_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_i_at(v: np.ndarray, i: np.ndarray, target_v: float, branch: str) -> float:
    ap = int(np.argmax(np.abs(v)))
    if branch == "outward":
        vx, ix = np.abs(v[: ap + 1]), np.abs(i[: ap + 1])
    else:
        vx, ix = np.abs(v[ap:][::-1]), np.abs(i[ap:][::-1])
    return max(float(np.interp(target_v, vx, ix)), 1.0e-14)


def rmse(pred: np.ndarray, meas: np.ndarray, mask: np.ndarray) -> float | None:
    if not np.any(mask):
        return None
    r = np.maximum(pred[mask], 1.0e-18) / np.maximum(meas[mask], 1.0e-18)
    return float(np.sqrt(np.mean(np.log10(r) ** 2)))


def suffix(meta: list) -> int:
    name = meta[7]
    return int(name.rsplit("@", 1)[1]) if "@" in name else 1


def calibrate_v43(
    train_set: dict,
    iv41_cohorts: dict,
    bare_set_tr: list,
    bare_rst_tr: list,
    nmos: StandaloneNMOS,
    rev_1t_fn,
    fwd_1t_fn,
) -> tuple[np.ndarray, float, int]:
    """Joint closed-loop calibration strictly on training splits (fast max_sub=20 during search)."""
    set_blocks = []
    for name, (items, circuit) in train_set.items():
        bare = name == "1M"
        if not bare:
            sub = np.linspace(0, len(items) - 1, min(14, len(items)), dtype=int)
            items = [items[i] for i in sub]
        voltage, actual, mask_gt055, init_g05 = joint.observations(items, circuit)
        raw_unclamped = np.stack([row.masks["use_unclamped"] for _, row, _ in items])
        ap = int(voltage.argmax())
        ret_low_mask = raw_unclamped & (np.arange(len(voltage)) > ap) & (voltage >= 0.20) & (voltage <= 0.55)

        i05_raw = (
            init_g05 * 0.50
            if bare
            else np.array([read_i_at(row.voltage, row.current, 0.50, "outward") for _, row, _ in items])
        )
        reg = joint.regions(voltage, bare)
        raw_t = []
        for k, sub_m in reg.items():
            if np.any(mask_gt055 & sub_m):
                w = 1.8 if k in ("pre_switch", "low_outward", "transition_outward") else 1.2
                raw_t.append((mask_gt055 & sub_m, w))
        if not bare and np.any(ret_low_mask):
            raw_t.append((ret_low_mask, 1.8))

        top_w = 0.28 if bare else (0.24 if name == "0.9" else 0.48 / (len(train_set) - 2))
        tot_w = sum(w for _, w in raw_t) + (3.4 if bare else 0.0)
        compiled = [
            (keep, np.log10(np.maximum(actual[keep], 1e-16)), w * top_w / (tot_w * keep.sum()))
            for keep, w in raw_t
        ]
        onset_info = None
        if bare:
            pts = np.array([joint.clamp_indices(voltage, r, circuit.compliance_a) for r in actual])
            idx_235 = int(np.argmin(np.abs(voltage[: ap + 1] - 2.35)))
            onset_info = (
                voltage[: ap + 1],
                voltage[pts],
                np.log10(np.maximum(actual[:, idx_235], 1e-15)),
                idx_235,
                3.4 * top_w / (tot_w * len(pts)),
            )
        set_blocks.append((voltage, bare, None if bare else float(name), i05_raw, compiled, onset_info))

    rst_blocks = [
        (
            bare_rst_tr[0].voltage,
            True,
            None,
            None,
            np.array([read_i_at(s.voltage, s.current, 0.25, "return") for s in bare_set_tr]),
            np.abs(np.stack([r.current for r in bare_rst_tr])),
            np.stack([r.masks["use_unclamped"] for r in bare_rst_tr]),
            0.28,
        )
    ]
    for g in (0.9, 1.1, 1.3):
        sr, rr = iv41_cohorts[g]["SET"], iv41_cohorts[g]["RESET"]
        n = min(len(sr), len(rr))
        sub = np.linspace(0, int(0.7 * n) - 1, 14, dtype=int)
        rst_blocks.append(
            (
                rr[0].voltage,
                False,
                fwd_1t_fn(g),
                rev_1t_fn(g),
                np.array([read_i_at(sr[i].voltage, sr[i].current, 0.25, "return") for i in sub]),
                np.abs(np.stack([rr[i].current for i in sub])),
                np.stack([rr[i].masks["use_unclamped"] for i in sub]),
                0.72 / 3.0,
            )
        )

    p0 = np.array(
        [
            np.log(32.0),      # 0: log_rc (ohm)
            7.45,              # 1: beta0
            1.28,              # 2: beta_fil
            np.log(4.5e-9),    # 3: log_g_cap
            1.84,              # 4: p_vrst
            1.57,              # 5: v_set0
            10.9,              # 6: s_set
            0.80,              # 7: v_rst0
            3.25,              # 8: s_rst
            np.log(1.9e5),     # 9: log_rth_10ms
            np.log(2.7e-5),    # 10: log_g_consolidate (~27 uS)
            8.5,               # 11: k_ret_relax_per_V
        ]
    )
    bounds = [
        (np.log(18.0), np.log(68.0)),
        (6.2, 8.6),
        (0.85, 1.65),
        (np.log(1.5e-9), np.log(1.0e-8)),
        (1.40, 2.10),
        (1.30, 1.85),
        (6.0, 14.5),
        (0.55, 1.05),
        (2.0, 7.5),
        (np.log(5.0e4), np.log(5.5e5)),
        (np.log(1.8e-5), np.log(3.8e-5)),
        (5.5, 18.0),
    ]

    def eval_set_loss(p):
        mdl = UnifiedCompactV43(p)
        mse = 0.0
        for voltage, bare, gv, i05_raw, compiled, onset_info in set_blocks:
            vrst = 2.5 if bare else 1.5
            lam = (vrst / 1.5) ** mdl.p_vrst
            c1t = None if bare else fwd_1t_fn(gv)
            g0_in = mdl.invert_read_to_g0(0.50, i05_raw, c1t, lam=lam)
            cur, _, cur_unc, _, _ = mdl.simulate_set(
                voltage, g0_in, None if bare else nmos, gv, vrst, step=0.035, iterations=19, max_sub=18
            )
            for keep, act, w in compiled:
                mse += w * float(np.sum((np.log10(np.maximum(cur[keep], 1e-18)) - act) ** 2))
            if onset_info is not None:
                v_out, v_meas, log_i235_m, idx_235, w_on = onset_info
                log_iu = np.maximum.accumulate(np.log10(np.maximum(cur_unc[:, : len(v_out)], 1e-18)), axis=1)
                v_pred = np.array([np.interp(-3.0, row, v_out) for row in log_iu])
                mse += w_on * float(np.sum(((v_pred - v_meas) / 0.09) ** 2))
                mse += 0.35 * w_on * float(np.sum((np.log10(np.maximum(cur[:, idx_235], 1e-18)) - log_i235_m) ** 2))
        return mse

    def eval_rst_loss(p):
        mdl = UnifiedCompactV43(p)
        tot = 0.0
        for r_volt, bare, fc_1t, rc_1t, i_set25, obs, msk, w in rst_blocks:
            g_at_25 = mdl.invert_read_to_g0(0.25, i_set25, fc_1t, lam=1.0)
            g_entry = mdl.relax_from_read_to_zero(g_at_25, 0.25, bare=bare)
            pred, _, _ = mdl.simulate_reset(r_volt, g_entry, rc_1t, step=0.025, max_sub=20)
            tot += w * float(np.mean((np.log10(np.maximum(pred[msk], 1e-18) / np.maximum(obs[msk], 1e-18))) ** 2))
        return tot

    print(f"  Initial v4.3 SET loss = {eval_set_loss(p0):.4f}, RESET loss = {eval_rst_loss(p0):.4f}", flush=True)
    # Stage 1: Optimize SET + Markov necking parameters [0..6, 10, 11]
    idx_s1 = [0, 1, 2, 3, 4, 5, 6, 10, 11]

    def f_s1(x_sub):
        p = p0.copy()
        p[idx_s1] = x_sub
        return eval_set_loss(p) + 0.12 * eval_rst_loss(p)

    s1 = minimize(f_s1, p0[idx_s1], method="Powell", bounds=[bounds[i] for i in idx_s1], options={"maxfev": 115, "ftol": 6e-4})
    p_mid = p0.copy()
    p_mid[idx_s1] = s1.x
    print(f"  Stage 1 joint loss = {s1.fun:.4f} (nfev={s1.nfev}); Stage 2 RESET + electrothermal...", flush=True)

    # Stage 2: Optimize RESET + thermal + consolidation [1, 4, 7, 8, 9, 10, 11]
    idx_s2 = [1, 4, 7, 8, 9, 10, 11]

    def f_s2(x_sub):
        p = p_mid.copy()
        p[idx_s2] = x_sub
        return eval_rst_loss(p) + 1.40 * eval_set_loss(p)

    s2 = minimize(f_s2, p_mid[idx_s2], method="Powell", bounds=[bounds[i] for i in idx_s2], options={"maxfev": 95, "ftol": 6e-4})
    p_after2 = p_mid.copy()
    p_after2[idx_s2] = s2.x
    print(f"  Stage 2 joint loss = {s2.fun:.4f} (nfev={s2.nfev}); Stage 3 final polish...", flush=True)

    # Stage 3: Final polish on [5, 6, 7, 8, 10, 11]
    idx_s3 = [5, 6, 7, 8, 10, 11]

    def f_s3(x_sub):
        p = p_after2.copy()
        p[idx_s3] = x_sub
        return eval_rst_loss(p) + 1.60 * eval_set_loss(p)

    s3 = minimize(f_s3, p_after2[idx_s3], method="Powell", bounds=[bounds[i] for i in idx_s3], options={"maxfev": 60, "ftol": 6e-4})
    p_final = p_after2.copy()
    p_final[idx_s3] = s3.x
    return p_final, float(s3.fun), int(s1.nfev + s2.nfev + s3.nfev)


def run_all(replay_frozen: bool = False):
    t0 = time.time()
    ITER07_DIR.mkdir(parents=True, exist_ok=True)
    idx = json.loads((CACHE_DIR / "index.json").read_text(encoding="utf-8"))
    nmos = StandaloneNMOS.fit_fig2b(REPO_ROOT / "vulcan2d/data/fig2b_transistor_lookup.json")
    rev_lookup = load_reverse_lookup()
    v41_audit = json.loads((AUDIT_V41_DIR / "scores.json").read_text(encoding="utf-8"))
    v41_cross = json.loads((AUDIT_V41_DIR / "cross_cell_scores.json").read_text(encoding="utf-8"))
    v42_res = json.loads(ITER06_JSON.read_text(encoding="utf-8"))

    def fwd_1t(g: float):
        return nmos.monotone_table(g, scale=1.0, offset_v=NMOS_GATE_OFFSET_V, vmax=6.0)

    def rev_1t(g: float):
        vd = np.linspace(0.0, 2.0, 201)
        return vd, np.array([rev_lookup.current(float(v), float(g)) for v in vd])

    prior = json.loads((EVIDENCE_DIR / "single_gap_series_set.json").read_text())["fit"]
    old_cwd = Path.cwd()
    os.chdir(HANDOFF_DATA_ROOT)
    try:
        train_set, test_set, _, _ = joint.load_split(CACHE_DIR, prior["beta_per_V"])
        groups_1m, _ = shared.load_groups()
    finally:
        os.chdir(old_cwd)

    bs, br = groups_1m[("1M", None, "SET")], groups_1m[("1M", None, "RESET")]
    iv41_cohorts = {
        g: iv.load_cohort(idx, CACHE_DIR, round((g - 0.9) * 10) + 5, g)[0]
        for g in (0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5)
    }

    if replay_frozen and (ITER07_DIR / "iteration_2026_11_27_02_results.json").exists():
        prev_j = json.loads((ITER07_DIR / "iteration_2026_11_27_02_results.json").read_text(encoding="utf-8"))
        mdl = UnifiedCompactV43.from_dict(prev_j["parameters"])
        print("[1/4] Loaded frozen calibrated 12-parameter v4.3 model...", flush=True)
    else:
        print("[1/4] Calibrating 12-parameter v4.3 closed-cycle model...", flush=True)
        x_v43, joint_loss_f, tot_nfev = calibrate_v43(
            train_set, iv41_cohorts, bs[:14], br[:14], nmos, rev_1t, fwd_1t
        )
        mdl = UnifiedCompactV43(x_v43)
        print(f"  Converged in {time.time() - t0:.1f}s (joint_loss={joint_loss_f:.4f}, nfev={tot_nfev})", flush=True)

    print("[2/4] Running full GPT-audit replay on held-out SET, +0.25V readback, & Autonomous SET->RESET...", flush=True)
    sets_out, resets_out, sim_end_states = {}, {}, {}
    set_plot_data, rst_plot_data = {}, {}
    merged_set_sq, merged_set_pts = 0.0, 0
    merged_rst_meas_sq, merged_rst_pred_sq, merged_rst_pts = 0.0, 0.0, 0

    for name, (items, circuit) in test_set.items():
        bare = name == "1M"
        gate = None if bare else float(name)
        fc = None if bare else fwd_1t(gate)
        peak = 2.5 if bare else 1.5
        v, obs, mask, anchor = joint.observations(items, circuit)
        raw_r = (
            anchor * 0.50
            if bare
            else np.array([read_i_at(row.voltage, row.current, 0.50, "outward") for _, row, _ in items])
        )
        lam = (peak / 1.5) ** mdl.p_vrst
        g0 = mdl.invert_read_to_g0(0.50, raw_r, fc, lam=lam)

        pred, vh, _, g_end0, v1t = mdl.simulate_set(v, g0, None if bare else nmos, gate, peak, step=0.0125, iterations=25)
        fine = mdl.simulate_set(v, g0, None if bare else nmos, gate, peak, step=0.00625, iterations=25)[0]
        sim_end_states[name] = {row.label: g_e for (_, row, _), g_e in zip(items, g_end0)}

        ap = int(v.argmax())
        out = np.arange(v.size) <= ap
        regions = {"all_unclamped": np.ones_like(out), "outward": out, **joint.regions(v, bare)}
        scores_d = {}
        for k, reg_m in regions.items():
            sm = mask & reg_m
            if np.any(sm):
                scores_d[k] = {
                    "points": int(sm.sum()),
                    "v41_audit": v41_audit["SET"][name]["scores"][k]["v41_replayed"],
                    "v42": v42_res["SET"][name]["scores"][k]["v42"],
                    "v43": rmse(pred, obs, sm),
                }
        if not bare:
            merged_set_sq += float(np.sum(np.log10(pred[mask] / obs[mask]) ** 2))
            merged_set_pts += int(mask.sum())

        ret_pred_025 = np.array([read_i_at(v, row, 0.25, "return") for row in pred])
        ret_meas_025 = np.array([read_i_at(v, row, 0.25, "return") for row in obs])
        threshold = 1e-3 if bare else fc[1][-1]
        bal = mask & (pred < 0.998 * threshold)

        sets_out[name] = {
            "scans": len(items),
            "scores": scores_d,
            "SET_step_halving_max_decade": {
                "v41_audit": v41_audit["SET"][name]["SET_step_halving_max_decade"],
                "v43": float(np.max(np.abs(np.log10(pred[mask] / fine[mask])))),
            },
            "SET_max_voltage_balance_V": float(np.max(np.abs((vh + mdl.rc * pred + v1t - v)[bal]))),
            "SET_return_0p25V_readback": {
                "measured_median_uA": float(np.median(ret_meas_025) * 1e6),
                "v41_prediction_median_uA": v41_audit["SET"][name]["SET_return_0p25V_readback"]["prediction_median_uA"],
                "v42_prediction_median_uA": v42_res["SET"][name]["SET_return_0p25V_readback"]["v42_prediction_median_uA"],
                "v43_prediction_median_uA": float(np.median(ret_pred_025) * 1e6),
                "v41_RMSE_decade": v41_audit["SET"][name]["SET_return_0p25V_readback"]["RMSE_decade"],
                "v42_RMSE_decade": v42_res["SET"][name]["SET_return_0p25V_readback"]["v42_RMSE_decade"],
                "v43_RMSE_decade": rmse(ret_pred_025, ret_meas_025, np.ones(len(pred), dtype=bool)),
            },
        }
        if bare:
            m_idx = [joint.clamp_indices(v, c, 1e-3) for c in obs]
            p_idx = [joint.clamp_indices(v, c, 1e-3) for c in pred]
            pairs = [(v[x], v[y]) for x, y in zip(m_idx, p_idx) if y is not None]
            sets_out[name]["clamp_mae_V"] = float(np.mean([abs(x - y) for x, y in pairs]))
        if name in ("1M", "0.9", "1.1", "1.4"):
            raw_uncl = np.stack([row.masks["use_unclamped"] for _, row, _ in items])
            disp_m = (mask | (out & (v >= 2.4))) if bare else (mask | ((~out) & raw_uncl & (v >= 0.10)))
            set_plot_data[name] = (v, obs, disp_m, pred)

    # Full RESET Handoff Replay (ZERO hardcoded g_floor split!)
    for name in test_set:
        bare = name == "1M"
        gate = None if bare else float(name)
        fc = None if bare else fwd_1t(gate)
        rc_c = None if bare else rev_1t(gate)
        if bare:
            sr, rr = bs[14:], br[14:]
        else:
            sheet = round((gate - 0.9) * 10) + 5
            sr, rr = iv41_cohorts[gate]["SET"], iv41_cohorts[gate]["RESET"]
            meta = {str(r[4]): r for r in idx["curves"][1:] if r[13] == idx["names"][sheet - 1]}
            assert [suffix(meta[r.label]) for r in sr] == [suffix(meta[r.label]) for r in rr]
            cut = int(0.7 * len(sr)) if gate <= 1.3 else 0
            sr, rr = sr[cut:], rr[cut:]

        rv = rr[0].voltage
        robs = np.abs(np.stack([r.current for r in rr]))
        rmask = np.stack([r.masks["use_unclamped"] for r in rr])
        read_025 = np.array([read_i_at(s.voltage, s.current, 0.25, "return") for s in sr])
        entry_meas = mdl.relax_from_read_to_zero(mdl.invert_read_to_g0(0.25, read_025, fc, lam=1.0), 0.25, bare=bare)

        anchored = mdl.simulate_reset(rv, entry_meas, rc_c, step=0.012)[0]
        refined_r = mdl.simulate_reset(rv, entry_meas, rc_c, step=0.006)[0]
        rst_step_max = float(np.max(np.abs(np.log10(anchored[rmask] / refined_r[rmask]))))

        m_idx = np.asarray([i for i, s in enumerate(sr) if s.label in sim_end_states[name]], dtype=int)
        obs_m, mask_m, anc_m = robs[m_idx], rmask[m_idx], anchored[m_idx]
        entry_pred = np.array([sim_end_states[name][sr[i].label] for i in m_idx])
        cascade = mdl.simulate_reset(rv, entry_pred, rc_c, step=0.012)[0]

        if not bare:
            merged_rst_meas_sq += float(np.sum(np.log10(anc_m[mask_m] / obs_m[mask_m]) ** 2))
            merged_rst_pred_sq += float(np.sum(np.log10(cascade[mask_m] / obs_m[mask_m]) ** 2))
            merged_rst_pts += int(mask_m.sum())

        resets_out[name] = {
            "n_paired": int(len(m_idx)),
            "points": int(mask_m.sum()),
            "RESET_step_halving_max_decade": {
                "v41_audit": v41_audit["RESET"][name]["RESET_step_halving_max_decade"],
                "v42": v42_res["RESET"][name]["RESET_step_halving_max_decade"]["v42"],
                "v43": rst_step_max,
            },
            "RESET_step_halving_rms_decade": rmse(anchored, refined_r, rmask),
            "measured_SET_return_RMSE": {
                "v41_audit": v41_audit["RESET"][name]["matched_RESET"]["scoring"]["all"]["measured_SET_return"],
                "v42": v42_res["RESET"][name]["measured_SET_return_RMSE"]["v42"],
                "v43": rmse(anc_m, obs_m, mask_m),
            },
            "predicted_SET_end_RMSE": {
                "v41_audit": v41_audit["RESET"][name]["matched_RESET"]["scoring"]["all"]["predicted_SET_end"],
                "v42": v42_res["RESET"][name]["predicted_SET_end_RMSE"]["v42"],
                "v43": rmse(cascade, obs_m, mask_m),
            },
            "entry_G_uS_median": {
                "measured_SET_return": float(np.median(entry_meas[m_idx]) * 1e6),
                "v41_predicted_SET_end": v41_audit["RESET"][name]["entry_G_uS_med"]["predicted_SET_end"],
                "v42_predicted_SET_end": v42_res["RESET"][name]["entry_G_uS_median"]["v42_predicted_SET_end"],
                "v43_predicted_SET_end": float(np.median(entry_pred) * 1e6),
            },
        }
        if name in ("1M", "0.9", "1.1", "1.4"):
            rst_plot_data[name] = (rv, obs_m, mask_m, anc_m, cascade)

    # Zero-shot IV41 5.0 V Bridge
    meta5 = {str(r[4]): r for r in idx["curves"][1:] if r[13] == idx["names"][11]}
    with np.load(CACHE_DIR / "s12.npz") as f5:
        tr5 = [f5[k][:, 1:] for k in sorted(f5.files, key=int) if meta5[k][6] == "正"]
    v5, obs5 = tr5[0][:, 0], np.abs(np.stack([t[:, 1] for t in tr5]))
    st5 = np.array([read_i_at(v5, row, 0.50, "outward") for row in obs5])
    g0_5 = mdl.invert_read_to_g0(0.50, st5, fwd_1t(5.0), lam=2.0 ** mdl.p_vrst)
    pr5 = mdl.simulate_set(v5, g0_5, nmos, 5.0, 3.0, step=0.015, iterations=25)[0]
    fn5 = mdl.simulate_set(v5, g0_5, nmos, 5.0, 3.0, step=0.0075, iterations=25)[0]
    out5 = np.arange(v5.size) <= v5.argmax()
    mo5 = out5 & (v5 > 0.55) & (obs5 < 0.72e-3)
    mr5 = np.broadcast_to(~out5 & (v5 >= 0.15), obs5.shape)
    bridge_5v = {
        "scans": len(tr5),
        "outward_RMSE": {
            "v41_audit": v41_audit["IV41_5V"]["outward"]["v41_replayed"],
            "v42": v42_res["IV41_5V"]["outward_RMSE"]["v42"],
            "v43": rmse(pr5, obs5, mo5),
        },
        "return_RMSE": {
            "v41_audit": v41_audit["IV41_5V"]["return"]["v41_replayed"],
            "v42": v42_res["IV41_5V"]["return_RMSE"]["v42"],
            "v43": rmse(pr5, obs5, mr5),
        },
        "step_halving_outward_decade": float(np.max(np.abs(np.log10(pr5[mo5] / fn5[mo5])))),
        "step_halving_return_decade": float(np.max(np.abs(np.log10(pr5[mr5] / fn5[mr5])))),
    }

    print("[3/4] Running frozen cross-cell replay across all 15 exported 1.1 V device groups (2,156 scans)...", flush=True)
    fc_1p1 = fwd_1t(1.1)
    cross_records = {}
    for sheet in range(14, 29):
        coh, _ = iv.load_cohort(idx, CACHE_DIR, sheet, 1.1)
        rows = coh["SET"]
        v_c = rows[0].voltage
        rev_pk = float(np.max(np.abs(coh["RESET"][0].voltage)))
        init_c = np.array([read_i_at(r.voltage, r.current, 0.50, "outward") for r in rows])
        g_c = mdl.invert_read_to_g0(0.50, init_c, fc_1p1, lam=(rev_pk / 1.5) ** mdl.p_vrst)
        pred_c = mdl.simulate_set(v_c, g_c, nmos, 1.1, rev_pk, step=0.0125, iterations=25)[0]
        meas_c = np.stack([np.abs(r.current) for r in rows])
        msk_c = np.stack([r.masks["use_unclamped"] for r in rows]) & (v_c > 0.55)
        out_c = np.arange(len(v_c)) <= v_c.argmax()
        lab = idx["names"][sheet - 1].split("_", 1)[1]
        cross_records[lab] = {
            "scans": len(rows),
            "v41_all_rmse": v41_cross["groups"][lab]["all"]["rmse_decade"],
            "v42_all_rmse": v42_res["cross_cell_15_groups"][lab]["v42_all_rmse"],
            "v43_all_rmse": rmse(pred_c, meas_c, msk_c),
            "v43_outward_rmse": rmse(pred_c, meas_c, msk_c & out_c),
            "v43_return_rmse": rmse(pred_c, meas_c, msk_c & ~out_c),
        }

    v43_cross_vals = [r["v43_all_rmse"] for r in cross_records.values()]
    v42_cross_vals = [r["v42_all_rmse"] for r in cross_records.values()]
    v41_cross_vals = [r["v41_all_rmse"] for r in cross_records.values()]

    print("[4/4] Generating side-by-side verification figures...", flush=True)
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.6), layout="constrained")
    for ax, key in zip(axes.flat, ["1M", "0.9", "1.1", "1.4"]):
        v, obs, dm, pr = set_plot_data[key]
        ap = int(v.argmax())
        for outward in (True, False):
            br_m = (np.arange(v.size) <= ap) if outward else (np.arange(v.size) > ap)
            cols = np.flatnonzero(br_m & dm.any(axis=0))
            if not len(cols):
                continue
            filt = dm[:, cols]
            ls = "-" if outward else "--"
            lo, med, hi = np.nanquantile(np.where(filt, obs[:, cols], np.nan), [0.1, 0.5, 0.9], axis=0) * 1e6
            if outward:
                ax.fill_between(v[cols], lo, hi, color=".82", alpha=0.55, label="Measured p10–p90")
            ax.plot(v[cols], med, color=".12", ls=ls, lw=1.7, label="Measured out" if outward else "Measured return")
            p_med = np.nanmedian(np.where(filt, pr[:, cols], np.nan), axis=0) * 1e6
            ax.plot(v[cols], p_med, color="#b8253a", ls=ls, lw=1.9, label="v4.3 out (12p)" if outward else "v4.3 return (12p)")
        if key == "1M":
            t_s = f"Bare 1M SET (held-out)\nOutward RMSE: {sets_out['1M']['scores']['outward']['v43']:.2f} dec"
        else:
            r_all = sets_out[key]["scores"]["all_unclamped"]["v43"]
            i25_m = sets_out[key]["SET_return_0p25V_readback"]["measured_median_uA"]
            i25_p = sets_out[key]["SET_return_0p25V_readback"]["v43_prediction_median_uA"]
            t_s = f"1T1M · {key} V SET (down to 0.10 V!)\nRMSE {r_all:.2f} dec | I(0.25V): meas {i25_m:.2f}, v4.3 {i25_p:.2f} µA"
        ax.set(title=t_s, yscale="log", xlabel="Applied SET voltage (V)", ylabel="Current magnitude (µA)")
        ax.grid(alpha=0.2)
    axes[0, 1].legend(fontsize=7.6, loc="lower right")
    fig.suptitle("Iteration 2026-11-27-02 (v4.3, 12 Params) | Full-Loop SET With Markov Nanowire Necking", weight="bold")
    fig.savefig(ITER07_DIR / "01_v43_12p_full_set.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.6), layout="constrained")
    for ax, key in zip(axes.flat, ["1M", "0.9", "1.1", "1.4"]):
        rv, robs, rmsk, anc, casc = rst_plot_data[key]
        rv_abs = np.abs(rv)
        ap = int(rv_abs.argmax())
        for outward in (True, False):
            br_m = (np.arange(rv_abs.size) <= ap) if outward else (np.arange(rv_abs.size) > ap)
            cols = np.flatnonzero(br_m & rmsk.any(axis=0))
            if not len(cols):
                continue
            filt = rmsk[:, cols]
            ls = "-" if outward else "--"
            lo, med, hi = np.nanquantile(np.where(filt, robs[:, cols], np.nan), [0.1, 0.5, 0.9], axis=0) * 1e6
            if outward:
                ax.fill_between(rv_abs[cols], lo, hi, color=".85", alpha=0.65)
            ax.plot(rv_abs[cols], med, color=".12", ls=ls, lw=1.7)
            ax.plot(rv_abs[cols], np.nanmedian(np.where(filt, anc[:, cols], np.nan), axis=0) * 1e6, color="#9f3546", ls=ls, lw=1.6)
            ax.plot(rv_abs[cols], np.nanmedian(np.where(filt, casc[:, cols], np.nan), axis=0) * 1e6, color="#2b6cb0", ls=ls, lw=1.6)
        rm_v = resets_out[key]["measured_SET_return_RMSE"]["v43"]
        rp_v41 = resets_out[key]["predicted_SET_end_RMSE"]["v41_audit"]
        rp_v42 = resets_out[key]["predicted_SET_end_RMSE"]["v42"]
        rp_v43 = resets_out[key]["predicted_SET_end_RMSE"]["v43"]
        lbl = "Bare 1M" if key == "1M" else f"1T1M · {key} V"
        ax.set(
            title=f"{lbl} RESET (12p, No Floor Split)\nPred-end: v4.1 {rp_v41:.2f} → v4.2 {rp_v42:.2f} → v4.3 {rp_v43:.2f} (Meas {rm_v:.2f})",
            yscale="log",
            xlabel="Applied RESET voltage magnitude (V)",
            ylabel="Current magnitude (µA)",
        )
        ax.grid(alpha=0.2)
    handles = [
        Line2D([], [], color=".12", lw=1.7, label="Measured median"),
        Line2D([], [], color="#9f3546", lw=1.6, label="Measured SET return → RESET"),
        Line2D([], [], color="#2b6cb0", lw=1.6, label="Autonomous Predicted SET end → RESET"),
        Line2D([], [], color=".35", ls="--", lw=1.4, label="Return sweep"),
    ]
    axes[0, 0].legend(handles=handles, fontsize=7.6)
    fig.suptitle("Iteration 2026-11-27-02 (v4.3, 12 Params) | Closed-Cycle RESET With Mott-Cabrera Self-Limiting Gap", weight="bold")
    fig.savefig(ITER07_DIR / "02_v43_12p_closed_cycle_reset.png", dpi=180)
    plt.close(fig)

    merged_summary = {
        "seven_gate_1T1M_SET_all_unclamped_decade": {
            "points": merged_set_pts,
            "v41_audit_12p": 0.227068,
            "v42_14p": v42_res["merged_headline_metrics"]["seven_gate_1T1M_SET_all_unclamped_decade"]["v42"],
            "v43_12p": float(np.sqrt(merged_set_sq / merged_set_pts)),
        },
        "seven_gate_1T1M_RESET_from_measured_SET_return_decade": {
            "points": merged_rst_pts,
            "v41_audit_12p": 1.16316,
            "v42_14p": v42_res["merged_headline_metrics"]["seven_gate_1T1M_RESET_from_measured_SET_return_decade"]["v42"],
            "v43_12p": float(np.sqrt(merged_rst_meas_sq / merged_rst_pts)),
        },
        "seven_gate_1T1M_RESET_from_predicted_SET_end_decade": {
            "points": merged_rst_pts,
            "v41_audit_12p": 1.51506,
            "v42_14p": v42_res["merged_headline_metrics"]["seven_gate_1T1M_RESET_from_predicted_SET_end_decade"]["v42"],
            "v43_12p": float(np.sqrt(merged_rst_pred_sq / merged_rst_pts)),
        },
        "cross_cell_15_groups_2156_scans_median_decade": {
            "v41_audit_median": float(np.median(v41_cross_vals)),
            "v42_median": float(np.median(v42_cross_vals)),
            "v43_median": float(np.median(v43_cross_vals)),
            "v43_min_max": [float(np.min(v43_cross_vals)), float(np.max(v43_cross_vals))],
        },
    }

    full_json = {
        "protocol": (
            "De-bloated 12-parameter closed-cycle model (v4.3): deletes beta_wkb_per_V, deletes "
            "zero_bias_retain / G_nonvol_S step jump, deletes non-Markov i_peak memory, and deletes "
            "the hardcoded Bare/1T1M g_floor split. Trained strictly on the same training splits."
        ),
        "parameter_count": 12,
        "parameters": mdl.to_dict(),
        "NMOS_fit": asdict(nmos),
        "merged_headline_metrics": merged_summary,
        "SET": sets_out,
        "RESET": resets_out,
        "IV41_5V": bridge_5v,
        "cross_cell_15_groups": cross_records,
        "provenance_sha256": {
            "iv41_workbook": sha256_file(WORKBOOK_IV41),
            "iv41_cache_index": sha256_file(CACHE_DIR / "index.json"),
            "bare_set_workbook": sha256_file(HANDOFF_DATA_ROOT / "1R数据.xlsx"),
            "bare_reset_workbook": sha256_file(HANDOFF_DATA_ROOT / "裸RRESET.xlsx"),
        },
    }
    (ITER07_DIR / "iteration_2026_11_27_02_results.json").write_text(
        json.dumps(full_json, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"All complete in {time.time() - t0:.1f}s! Saved to {ITER07_DIR}", flush=True)


if __name__ == "__main__":
    import sys

    run_all(replay_frozen="--replay-frozen" in sys.argv)
