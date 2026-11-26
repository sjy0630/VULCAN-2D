"""Unified Electro-Thermal Two-Channel (ET-2Ch) compact model for Au/Ti/h-BN/W memristors.

Implements:
1. StandaloneNMOS: Smooth monotone velocity-saturated short-channel NMOS model fitted strictly
   to standalone 1T measurements (Fig. 2b forward + Bookoutput2 reverse), avoiding 0.5 V linear
   V_G interpolation artifacts and extending smoothly to V_G = 5.0 V.
2. FieldScaledET2ChSetRamp: Unified SET constitutive and evolution solver with:
   - Barrier depletion scaling E_eff = V_h / lambda(|V_reset,stop|) from preceding RESET depth;
   - Parallel hopping barrier + un-erased ohmic filament residual channel;
   - Local electro-thermal feedback T_eff = T_0 + R_th * (I * V_h + P_cap_surge) with C_par.
3. ElectroThermalResetRamp: Unified continuous electro-thermal RESET solver across both branches
   for µW 1T1M gradual erasure and mW Bare 1M / 5V thermal filament rupture.
4. StochasticBimodalEnsemble: Cycle-to-cycle activation-energy dispersion + 2-state Markov
   (Distributed <-> Filamentary) switching for distributional (p10-p90, V_set, G_LRS) prediction.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit


@dataclass(frozen=True)
class StandaloneNMOS:
    vt0: float
    dibl: float
    subvt_slope: float
    gm_sat: float
    g_lin: float
    clm: float
    p_tail: float
    v_tail: float
    fit_rmse_ua: float

    @classmethod
    def fit_fig2b(cls, lookup_json_path: Path) -> "StandaloneNMOS":
        payload = json.loads(Path(lookup_json_path).read_text(encoding="utf-8"))
        vg_all, vd_all, id_all = [], [], []
        for c in payload["curves"]:
            vg = float(c["gate_voltage_v"])
            v = np.asarray(c["drain_voltage_v"], dtype=float)
            i = np.asarray(c["drain_current_a"], dtype=float)
            ap = int(np.argmax(v))
            vu = v[: ap + 1]
            iu = 0.5 * (i[: ap + 1] + np.interp(vu, v[ap:][::-1], i[ap:][::-1]))
            iu = np.maximum.accumulate(np.maximum(iu, 0.0))
            keep = vu >= 0.05
            vg_all.append(np.full(keep.sum(), vg))
            vd_all.append(vu[keep])
            id_all.append(iu[keep])
        vg_arr, vd_arr, id_arr = map(np.concatenate, (vg_all, vd_all, id_all))

        p0 = [0.62, 0.04, 0.09, 1.75e-4, 2.2e-4, 0.05, 1e-6, 4.0]
        lo = [0.20, 0.00, 0.03, 5.0e-5, 3.0e-5, 0.00, 0.00, 2.0]
        hi = [1.00, 0.20, 0.30, 5.0e-4, 1.0e-3, 0.40, 1e-4, 7.0]

        def eval_p(vd, vg, p):
            vt0, dibl, ss, gm_sat, g_lin, clm, p_tail, v_tail = p
            vov = ss * np.logaddexp(0.0, (vg - (vt0 - dibl * vd)) / ss)
            isat = gm_sat * vov
            ilin = g_lin * (vov + 0.15) * vd
            id_ch = (ilin * isat) / np.power(
                np.power(np.maximum(ilin, 0.0), 1.6) + np.power(isat + 1e-18, 1.6), 1.0 / 1.6
            )
            tail = p_tail * vd * np.exp(np.clip(vd - v_tail, -30.0, 20.0)) * np.exp(0.35 * vg)
            return id_ch * (1.0 + clm * vd) + np.where(vd > 0.0, tail, 0.0)

        sol = least_squares(
            lambda p: (eval_p(vd_arr, vg_arr, p) - id_arr) * 1e5,
            p0,
            bounds=(lo, hi),
            max_nfev=800,
        )
        rmse_ua = float(np.sqrt(np.mean((eval_p(vd_arr, vg_arr, sol.x) - id_arr) ** 2)) * 1e6)
        return cls(*map(float, sol.x), fit_rmse_ua=rmse_ua)

    def drain_current(self, vd: np.ndarray, vg: float, scale: float = 1.0, offset_v: float = 0.0) -> np.ndarray:
        vd_pos = np.maximum(np.asarray(vd, dtype=float), 0.0)
        vg_eff = max(float(vg) + float(offset_v), 0.10)
        vov = self.subvt_slope * np.logaddexp(
            0.0, (vg_eff - (self.vt0 - self.dibl * vd_pos)) / self.subvt_slope
        )
        isat = self.gm_sat * vov
        ilin = self.g_lin * (vov + 0.15) * vd_pos
        id_ch = (ilin * isat) / np.power(
            np.power(np.maximum(ilin, 0.0), 1.6) + np.power(isat + 1e-18, 1.6), 1.0 / 1.6
        )
        tail = (
            self.p_tail
            * vd_pos
            * np.exp(np.clip(vd_pos - self.v_tail, -30.0, 20.0))
            * np.exp(0.35 * min(vg_eff, 3.5))
        )
        out = scale * (id_ch * (1.0 + self.clm * vd_pos) + tail)
        return np.where(vd_pos > 0.0, out, 0.0)

    def monotone_table(self, vg: float, scale: float = 1.0, offset_v: float = 0.0, vmax: float = 7.0, n: int = 351):
        vd = np.linspace(0.0, vmax, n)
        id_c = self.drain_current(vd, vg, scale=scale, offset_v=offset_v)
        id_c[0] = 0.0
        id_c = np.maximum.accumulate(id_c)
        return vd, id_c


class FieldScaledET2ChSetRamp:
    """Unified SET model with E = V_h / lambda(|V_rst|), parallel ohmic residual, and Joule heating."""

    def __init__(
        self,
        terminal: np.ndarray,
        nmos: StandaloneNMOS | None,
        gate_v: float | None,
        v_reset_stop: float,
        step: float = 0.025,
        compliance_a: float | None = None,
        c_par_pf: float | None = None,
        raw_1t_curve: tuple[np.ndarray, np.ndarray] | None = None,
        enable_vrst_scaling: bool = True,
        enable_two_channel: bool = True,
        enable_thermal: bool = True,
    ):
        self.terminal = np.asarray(terminal, dtype=float)
        self.nmos = nmos
        self.bare = gate_v is None and raw_1t_curve is None
        self.gate_v = gate_v
        self.raw_1t_curve = raw_1t_curve
        self.v_reset_stop = float(v_reset_stop)
        self.apex = int(self.terminal.argmax())
        self.grid = np.linspace(
            0.0, self.terminal[self.apex], int(np.ceil(self.terminal[self.apex] / step)) + 1
        )
        self.compliance_a = (1e-3 if self.bare else None) if compliance_a is None else compliance_a
        self.c_par_pf = (300.0 if self.bare else 50.0) if c_par_pf is None else float(c_par_pf)
        self.enable_vrst_scaling = enable_vrst_scaling
        self.enable_two_channel = enable_two_channel
        self.enable_thermal = enable_thermal

    def predict(
        self,
        initial_g05: np.ndarray,
        params: np.ndarray,
        iterations: int = 24,
        field_shift: np.ndarray | float = 0.0,
        filament_mode: np.ndarray | None = None,
        g_fil_high_s: float = 1.25e-4,
    ):
        (
            beta0,
            log_rc,
            gamma_vrst,
            f_res_thresh_log,
            res_frac,
            log_k_set,
            s_field,
            z_cap,
            log_rth,
            c_surge,
            t_scale,
            t_offset,
        ) = params

        initial_g05 = np.asarray(initial_g05, dtype=float)
        rc = float(np.exp(log_rc))
        rth = float(np.exp(log_rth)) if self.enable_thermal else 0.0
        lam = max(1.0 + gamma_vrst * (self.v_reset_stop - 1.5), 0.5) if self.enable_vrst_scaling else 1.0
        beta_eff = beta0 / lam

        if self.enable_two_channel:
            g_th = float(np.exp(f_res_thresh_log))
            w_ohmic = res_frac * (initial_g05 / (initial_g05 + g_th))
        else:
            w_ohmic = np.zeros_like(initial_g05)
        g_ohmic0 = initial_g05 * w_ohmic
        g_hop0 = np.maximum(initial_g05 - g_ohmic0, 1e-15)

        v05 = np.maximum(0.5 - rc * initial_g05 * 0.5, 0.10)
        amp0 = (g_hop0 * v05) / np.maximum(np.sinh(beta_eff * v05), 1e-12)

        if not self.bare:
            if self.raw_1t_curve is not None:
                vd_tab, id_tab = self.raw_1t_curve
            else:
                vd_tab, id_tab = self.nmos.monotone_table(
                    self.gate_v, scale=t_scale, offset_v=t_offset, vmax=max(5.5, float(self.terminal.max()) + 0.5)
                )
            levels, idx_u = np.unique(id_tab, return_index=True)
            v_of_i = vd_tab[idx_u]
            imax_1t = float(levels[-1])

        def hbn_i(vh, z, g_extra=0.0):
            a0 = amp0[:, None] if vh.ndim == 2 else amp0
            go_base = g_ohmic0[:, None] if vh.ndim == 2 else g_ohmic0
            ge = g_extra[:, None] if (vh.ndim == 2 and np.ndim(g_extra) == 1) else g_extra
            # Initial barrier hopping + newly formed soft/filamentary links (weakly nonlinear, ~V + 0.6 V^2)
            i_init_bar = a0 * np.sinh(np.clip(beta_eff * vh, 0.0, 40.0))
            g_formed = a0 * np.expm1(np.maximum(z, 0.0)) * np.sinh(beta_eff * 1.0)
            i_formed = g_formed * vh * (0.35 + 0.65 * np.sinh(np.clip(1.6 * vh, 0.0, 25.0)) / np.sinh(1.6))
            return i_init_bar + i_formed + (go_base + ge) * vh

        def solve(applied, z, g_extra=0.0):
            app = np.asarray(applied, dtype=float)
            lo = np.zeros_like(z + app)
            hi = np.broadcast_to(app, lo.shape).copy()
            for _ in range(iterations):
                vh = 0.5 * (lo + hi)
                it = hbn_i(vh, z, g_extra)
                if self.bare:
                    below = (it < self.compliance_a) & (vh + rc * it < app)
                else:
                    v1t = np.interp(np.minimum(it, imax_1t), levels, v_of_i)
                    below = (it < imax_1t * 0.99995) & (vh + rc * it + v1t < app)
                lo = np.where(below, vh, lo)
                hi = np.where(below, hi, vh)
            vh = 0.5 * (lo + hi)
            it_unc = hbn_i(vh, z, g_extra)
            it_cl = np.minimum(it_unc, self.compliance_a) if self.compliance_a is not None else it_unc
            v1t_final = np.zeros_like(vh) if self.bare else np.interp(np.minimum(it_cl, imax_1t), levels, v_of_i)
            return it_cl, vh, it_unc, v1t_final

        z = np.zeros_like(initial_g05)
        g_fil = np.zeros_like(initial_g05)
        z_tr, gf_tr = [z], [g_fil]
        c_exc = max(self.c_par_pf - 50.0, 0.0) / 250.0 if self.enable_thermal else 0.0
        cap_eff = z_cap + (6.0 if self.bare and self.enable_thermal else 0.0) * min(c_surge * 1e3, 2.0)
        fil_mask = np.zeros_like(initial_g05) if filament_mode is None else np.asarray(filament_mode, dtype=float)

        for a, step in zip(self.grid[:-1], np.diff(self.grid)):
            it, vh, _, _ = solve(a, z, g_fil)
            p_eff = it * vh + c_surge * c_exc * (vh**2) * (it / (it + 2e-6))
            t_rat = 1.0 + (rth / 300.0) * p_eff
            e_eff = vh / lam - field_shift
            r1 = np.exp(np.clip(log_k_set + s_field * (e_eff - 1.0) + 16.0 * (1.0 - 1.0 / t_rat), -35.0, 40.0))
            z_mid = z + (cap_eff - z) * (-np.expm1(-r1 * step * 0.5))
            gf_mid = g_fil + fil_mask * (g_fil_high_s - g_fil) * (-np.expm1(-0.35 * r1 * step * 0.5))

            it2, vh2, _, _ = solve(a + 0.5 * step, z_mid, gf_mid)
            p_eff2 = it2 * vh2 + c_surge * c_exc * (vh2**2) * (it2 / (it2 + 2e-6))
            t_rat2 = 1.0 + (rth / 300.0) * p_eff2
            e_eff2 = vh2 / lam - field_shift
            r2 = np.exp(np.clip(log_k_set + s_field * (e_eff2 - 1.0) + 16.0 * (1.0 - 1.0 / t_rat2), -35.0, 40.0))
            z = z + (cap_eff - z) * (-np.expm1(-r2 * step))
            g_fil = g_fil + fil_mask * (g_fil_high_s - g_fil) * (-np.expm1(-0.35 * r2 * step))
            z_tr.append(z)
            gf_tr.append(g_fil)

        z_tr = np.asarray(z_tr).T
        gf_tr = np.asarray(gf_tr).T
        z_map = np.array([np.interp(self.terminal[: self.apex + 1], self.grid, row) for row in z_tr])
        gf_map = np.array([np.interp(self.terminal[: self.apex + 1], self.grid, row) for row in gf_tr])
        n_ret = len(self.terminal) - self.apex - 1
        z_map = np.c_[z_map, np.broadcast_to(z[:, None], (len(z), n_ret))]
        gf_map = np.c_[gf_map, np.broadcast_to(g_fil[:, None], (len(g_fil), n_ret))]
        cur, vh, cur_unc, v1t = solve(self.terminal, z_map, gf_map)
        return cur, vh, cur_unc, z_map, v1t


class ElectroThermalResetRamp:
    """Continuous electro-thermal RESET model across outward + return branches for 1T1M and Bare 1M."""

    def __init__(
        self,
        voltage: np.ndarray,
        reverse_1t_curve: tuple[np.ndarray, np.ndarray] | None,
        rc_ohm: float,
        step: float = 0.015,
    ):
        self.wave = np.abs(np.asarray(voltage, dtype=float))
        self.apex = int(self.wave.argmax())
        self.bare = reverse_1t_curve is None
        self.rc = float(rc_ohm)
        if not self.bare:
            vd, id_c = reverse_1t_curve
            self.levels, idx = np.unique(id_c, return_index=True)
            self.v_of_i = vd[idx]
            self.imax_1t = float(self.levels[-1])
        # Build fine path grid across BOTH outward and return limbs
        up = np.linspace(0.0, self.wave[self.apex], int(np.ceil(self.wave[self.apex] / step)) + 1)
        dn = up[-2::-1]
        self.grid_v = np.r_[up, dn]
        self.grid_apex = len(up) - 1

    def predict(self, g_entry_s: np.ndarray, g_floor_s: float, params: np.ndarray, enable_thermal: bool = True):
        """Propagate log-conductance and metallic filament fraction across full RESET waveform."""
        beta_rst, log_k_rst, s_field_rst, log_rth_rst, v_onset = params
        rth = float(np.exp(log_rth_rst)) if enable_thermal else 0.0
        g_entry_s = np.maximum(np.asarray(g_entry_s, dtype=float), g_floor_s * 1.01)

        # Split entry conductance into ohmic/metallic filament (>30 uS) + nonlinear soft state
        g_fil = np.maximum(g_entry_s - 2.5e-5, 0.0)
        g_soft = np.maximum(g_entry_s - g_fil, g_floor_s * 1.01)

        def hbn_i(vh, gs, gf):
            # Normalized at 0.15 V so I(0.15 V) / 0.15 V == gs + gf
            sh_ratio = np.sinh(np.clip(beta_rst * vh, 0.0, 35.0)) / max(np.sinh(beta_rst * 0.15), 1e-12)
            return gs * 0.15 * sh_ratio + gf * vh

        def solve_step(app_v, gs, gf):
            lo = np.zeros_like(gs + app_v)
            hi = np.full_like(lo, app_v)
            for _ in range(22):
                vh = 0.5 * (lo + hi)
                it = hbn_i(vh, gs, gf)
                if self.bare:
                    below = vh + self.rc * it < app_v
                else:
                    v1t = np.interp(np.minimum(it, self.imax_1t), self.levels, self.v_of_i)
                    below = (it < self.imax_1t * 0.9999) & (vh + self.rc * it + v1t < app_v)
                lo = np.where(below, vh, lo)
                hi = np.where(below, hi, vh)
            vh = 0.5 * (lo + hi)
            return hbn_i(vh, gs, gf), vh

        gs = g_soft.copy()
        gf = g_fil.copy()
        gs_tr, gf_tr = [gs], [gf]
        for j in range(len(self.grid_v) - 1):
            a = self.grid_v[j]
            ds = abs(self.grid_v[j + 1] - a)
            it, vh = solve_step(a, gs, gf)
            t_rat = 1.0 + (rth / 300.0) * (it * vh)
            drive = s_field_rst * np.maximum(vh - v_onset, -0.25) + 18.0 * (1.0 - 1.0 / t_rat)
            rate = np.exp(np.clip(log_k_rst + drive, -35.0, 35.0))
            # Filament ruptures thermally & field-wise first; soft conductance decays toward g_floor_s in log-space
            gf = gf * np.exp(-1.8 * rate * ds)
            log_gs = np.log(g_floor_s) + (np.log(np.maximum(gs, g_floor_s)) - np.log(g_floor_s)) * np.exp(-rate * ds)
            gs = np.exp(log_gs)
            gs_tr.append(gs)
            gf_tr.append(gf)

        gs_tr = np.asarray(gs_tr).T
        gf_tr = np.asarray(gf_tr).T
        # Map outward and return separately onto measured voltage grid
        up_grid = self.grid_v[: self.grid_apex + 1]
        dn_grid = self.grid_v[self.grid_apex :][::-1]  # ascending for np.interp
        w_up = self.wave[: self.apex + 1]
        w_dn = self.wave[self.apex + 1 :]

        gs_up = np.array([np.interp(w_up, up_grid, r[: self.grid_apex + 1]) for r in gs_tr])
        gf_up = np.array([np.interp(w_up, up_grid, r[: self.grid_apex + 1]) for r in gf_tr])
        gs_dn = np.array([np.interp(w_dn, dn_grid, r[self.grid_apex :][::-1]) for r in gs_tr])
        gf_dn = np.array([np.interp(w_dn, dn_grid, r[self.grid_apex :][::-1]) for r in gf_tr])
        gs_full = np.c_[gs_up, gs_dn]
        gf_full = np.c_[gf_up, gf_dn]

        # Vectorized point operating solve on measured waveform
        app = np.broadcast_to(self.wave, gs_full.shape)
        lo = np.zeros_like(app)
        hi = app.copy()
        for _ in range(24):
            vh = 0.5 * (lo + hi)
            sh_ratio = np.sinh(np.clip(beta_rst * vh, 0.0, 35.0)) / max(np.sinh(beta_rst * 0.15), 1e-12)
            it = gs_full * 0.15 * sh_ratio + gf_full * vh
            if self.bare:
                below = vh + self.rc * it < app
            else:
                v1t = np.interp(np.minimum(it, self.imax_1t), self.levels, self.v_of_i)
                below = (it < self.imax_1t * 0.9999) & (vh + self.rc * it + v1t < app)
            lo = np.where(below, vh, lo)
            hi = np.where(below, hi, vh)
        vh = 0.5 * (lo + hi)
        sh_ratio = np.sinh(np.clip(beta_rst * vh, 0.0, 35.0)) / max(np.sinh(beta_rst * 0.15), 1e-12)
        cur = gs_full * 0.15 * sh_ratio + gf_full * vh
        return np.maximum(cur, 1e-16), vh, gs_full + gf_full
