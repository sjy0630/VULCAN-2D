"""Iteration 2026-11-25-03: De-bulged Sharp Electro-Thermal SET + Geometry-Scaled R_th(G) RESET.

Resolves the visual and physical shortcomings identified after Iteration 2026-11-25-02:
1. Pre-switching bulge elimination & vertical Bare 1M jump:
   - Caps deep-barrier hopping amplitude at G_deep_cap (~25-60 nS) while routing excess
     shallow-reset conductance (e.g. 250 nS at 1.5 V, 20 uS at 5.0 V) through a low-slope
     residual bridge (beta_res ~ 2.2 V^-1), eliminating the 0.8-1.8 V upward bulge.
   - Power-law RESET-depletion scaling lambda_trans = (|V_rst|/1.5)^p_t keeps Bare 1M
     pre-switching slope gentle (~3.1 V^-1, <0.1 uA up to 2.55 V) before electro-thermal +
     300 pF capacitive runaway triggers a razor-sharp vertical jump at ~2.65 V.
2. Cross-cycle two-channel handoff (RESET[k-1] -> SET[k]):
   - Extracts both deep-barrier and residual-bridge states from preceding RESET return limb,
     leveraging IV41's r = 0.89-0.991 cross-cycle continuity.
3. Constriction geometry-scaled RESET thermal resistance R_th(G) = R_th_10mS * (10 mS / G)^eta:
   - Simultaneously ruptures 95-ohm metallic filaments in Bare 1M and 10-kohm filaments in
     1.3-1.5 V 1T1M, with intrinsic h-BN low-bias de-embedding across SET->RESET polarity flip.
"""
from __future__ import annotations

import numpy as np

from analysis.et_two_channel_model import StandaloneNMOS


def intrinsic_low_bias_g(
    v: np.ndarray,
    i: np.ndarray,
    circuit_1t: tuple[np.ndarray, np.ndarray] | None = None,
    rc_ohm: float = 65.0,
    v_read: float = 0.15,
) -> tuple[float, float]:
    """De-embed intrinsic h-BN conductance at |V_app| = v_read on outward and return limbs."""
    ap = int(np.argmax(np.abs(v)))
    vo, io = np.abs(v[: ap + 1]), np.abs(i[: ap + 1])
    vr, ir = np.abs(v[ap:][::-1]), np.abs(i[ap:][::-1])
    if circuit_1t is not None:
        vd_tab, id_tab = circuit_1t
        levels, idx_u = np.unique(id_tab, return_index=True)
        v_of_i = vd_tab[idx_u]
        imax = float(levels[-1])

    out = []
    for vx, ix in ((vo, io), (vr, ir)):
        i_at = max(float(np.interp(v_read, vx, ix)), 1e-14)
        v_drop = rc_ohm * i_at
        if circuit_1t is not None:
            v_drop += float(np.interp(min(i_at, imax), levels, v_of_i))
        vh_eff = max(v_read - v_drop, 0.025)
        out.append(max(i_at / vh_eff, 1e-13))
    return out[0], out[1]


class DebulgedSharpET2ChSetRamp:
    """Unified SET model without pre-switching upward bulge and with sharp electro-thermal knee."""

    def __init__(
        self,
        terminal: np.ndarray,
        nmos: StandaloneNMOS | None,
        gate_v: float | None,
        v_reset_stop: float,
        step: float = 0.025,
        compliance_a: float | None = None,
        c_par_pf: float | None = None,
    ):
        self.terminal = np.asarray(terminal, dtype=float)
        self.nmos = nmos
        self.bare = gate_v is None
        self.gate_v = gate_v
        self.v_reset_stop = float(v_reset_stop)
        self.apex = int(self.terminal.argmax())
        self.grid = np.linspace(
            0.0, self.terminal[self.apex], int(np.ceil(self.terminal[self.apex] / step)) + 1
        )
        self.compliance_a = (1e-3 if self.bare else None) if compliance_a is None else compliance_a
        self.c_par_pf = (300.0 if self.bare else 50.0) if c_par_pf is None else float(c_par_pf)

    def predict(
        self,
        initial_g05: np.ndarray,
        params: np.ndarray,
        iterations: int = 23,
        field_shift: np.ndarray | float = 0.0,
        filament_mode: np.ndarray | float | None = None,
        g_fil_high_s: float = 1.35e-4,
    ):
        (
            beta_deep,
            beta_res,
            log_rc,
            p_vrst_trans,
            p_vrst_field,
            log_g_deep_cap,
            log_g_seed,
            log_k_set,
            s_field,
            v_crit0,
            z_fb,
            log_g_on_base,
            v_hold_lrs,
            log_rth,
            c_surge,
            t_scale,
            t_offset,
        ) = params

        initial_g05 = np.asarray(initial_g05, dtype=float)
        rc = float(np.exp(log_rc))
        rth = float(np.exp(log_rth))
        g_on_base = float(np.exp(log_g_on_base))
        g_seed = float(np.exp(log_g_seed))
        lam_trans = (self.v_reset_stop / 1.5) ** p_vrst_trans
        lam_field = (self.v_reset_stop / 1.5) ** p_vrst_field
        beta_eff = beta_deep / lam_trans

        g_cap = float(np.exp(log_g_deep_cap))
        g_deep0 = g_cap * np.tanh(initial_g05 / g_cap)
        g_res0 = np.maximum(initial_g05 - g_deep0, 0.0)

        # Residual parallel shunt delays primary breakdown slightly at high G_0 (1.5V / 5.0V)
        v_shunt_shift = 0.055 * np.log1p(g_res0 / g_cap)

        v05 = np.maximum(0.5 - rc * initial_g05 * 0.5, 0.10)
        amp_deep = (g_deep0 * v05) / np.maximum(np.sinh(beta_eff * v05), 1e-12)
        sv_res05 = np.maximum(np.sinh(beta_res * v05), 1e-12)

        if not self.bare:
            vd_tab, id_tab = self.nmos.monotone_table(
                self.gate_v,
                scale=t_scale,
                offset_v=t_offset,
                vmax=max(5.5, float(self.terminal.max()) + 0.5),
            )
            levels, idx_u = np.unique(id_tab, return_index=True)
            v_of_i = vd_tab[idx_u]
            imax_1t = float(levels[-1])

        def hbn_i(vh, z, g_extra=0.0):
            ad = amp_deep[:, None] if vh.ndim == 2 else amp_deep
            gr = g_res0[:, None] if vh.ndim == 2 else g_res0
            v0 = v05[:, None] if vh.ndim == 2 else v05
            svr = sv_res05[:, None] if vh.ndim == 2 else sv_res05
            ge = g_extra[:, None] if (vh.ndim == 2 and np.ndim(g_extra) == 1) else g_extra

            i_deep = ad * np.sinh(np.clip(beta_eff * vh, 0.0, 38.0))
            i_res = (gr * v0) * np.sinh(np.clip(beta_res * vh, 0.0, 35.0)) / svr
            # Intrinsic h-BN defect nucleation seed g_seed (~10 nS) grows by expm1(z) toward self-limiting LRS capacity
            g_sw = g_seed * np.expm1(np.clip(z, 0.0, 22.0))
            i_sw = g_sw * vh * (0.38 + 0.62 * np.sinh(np.clip(1.45 * vh, 0.0, 25.0)) / np.sinh(1.45))
            return i_deep + i_res + i_sw + ge * vh

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
            v1t_out = np.zeros_like(vh) if self.bare else np.interp(np.minimum(it_cl, imax_1t), levels, v_of_i)
            return it_cl, vh, it_unc, v1t_out

        z = np.zeros_like(initial_g05)
        g_fil = np.zeros_like(initial_g05)
        g_cap_cur = np.full_like(initial_g05, 1.5e-2 if self.bare else g_on_base)
        z_tr, gf_tr = [z], [g_fil]
        c_exc = max(self.c_par_pf - 50.0, 0.0) / 250.0
        fil_mask = np.zeros_like(initial_g05) if filament_mode is None else np.asarray(filament_mode, dtype=float)

        for a, step in zip(self.grid[:-1], np.diff(self.grid)):
            it, vh, _, _ = solve(a, z, g_fil)
            if not self.bare:
                g_cap_cur = np.maximum(g_cap_cur, g_on_base + it / max(v_hold_lrs, 0.20))
            z_cap_dyn = np.maximum(np.log1p(g_cap_cur / g_seed), 0.0)
            # Capacitive surge & tip feedback ignite ONLY once dielectric breakdown nucleates (z > 0.35),
            # never prematurely on noisy high-G0 fallback scans!
            nuc_gate = z / (z + 0.35)
            p_eff = it * vh + c_surge * c_exc * (vh**2) * nuc_gate
            t_rat = 1.0 + (rth / 300.0) * p_eff
            fb_term = z_fb * c_exc * nuc_gate * np.minimum(z, 6.0)
            e_eff = (vh / lam_field) - v_crit0 - v_shunt_shift + fb_term - field_shift
            r1 = np.exp(np.clip(log_k_set + s_field * e_eff + 22.0 * (1.0 - 1.0 / t_rat), -38.0, 45.0))
            rem1 = np.maximum(z_cap_dyn - z, 0.0)
            eff_rem1 = np.minimum(rem1, 6.5)
            z_mid = np.minimum(z + eff_rem1 * (-np.expm1(-r1 * step * 0.5)), z_cap_dyn)
            gf_mid = g_fil + fil_mask * (g_fil_high_s - g_fil) * (-np.expm1(-0.4 * r1 * step * 0.5))

            it2, vh2, _, _ = solve(a + 0.5 * step, z_mid, gf_mid)
            if not self.bare:
                g_cap_cur = np.maximum(g_cap_cur, g_on_base + it2 / max(v_hold_lrs, 0.20))
            z_cap_dyn2 = np.maximum(np.log1p(g_cap_cur / g_seed), 0.0)
            nuc_gate2 = z_mid / (z_mid + 0.35)
            p_eff2 = it2 * vh2 + c_surge * c_exc * (vh2**2) * nuc_gate2
            t_rat2 = 1.0 + (rth / 300.0) * p_eff2
            fb_term2 = z_fb * c_exc * nuc_gate2 * np.minimum(z_mid, 6.0)
            e_eff2 = (vh2 / lam_field) - v_crit0 - v_shunt_shift + fb_term2 - field_shift
            r2 = np.exp(np.clip(log_k_set + s_field * e_eff2 + 22.0 * (1.0 - 1.0 / t_rat2), -38.0, 45.0))
            rem2 = np.maximum(z_cap_dyn2 - z, 0.0)
            eff_rem2 = np.minimum(rem2, 6.5)
            z = np.minimum(z + eff_rem2 * (-np.expm1(-r2 * step)), z_cap_dyn2)
            g_fil = g_fil + fil_mask * (g_fil_high_s - g_fil) * (-np.expm1(-0.4 * r2 * step))
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


class GeomThermalResetRamp:
    """Electro-thermal RESET with constriction geometry thermal scaling R_th(G) = R_th_10mS * (10 mS / G)^eta."""

    def __init__(
        self,
        voltage: np.ndarray,
        rev_1t_curve: tuple[np.ndarray, np.ndarray] | None,
        rc_ohm: float = 65.0,
        step: float = 0.015,
    ):
        self.wave = np.abs(np.asarray(voltage, dtype=float))
        self.apex = int(self.wave.argmax())
        self.bare = rev_1t_curve is None
        self.rc = float(rc_ohm)
        if not self.bare:
            vd, id_c = rev_1t_curve
            self.levels, idx_u = np.unique(id_c, return_index=True)
            self.v_of_i = vd[idx_u]
            self.imax_1t = float(self.levels[-1])
        up = np.linspace(0.0, self.wave[self.apex], int(np.ceil(self.wave[self.apex] / step)) + 1)
        self.grid_v = np.r_[up, up[-2::-1]]
        self.grid_apex = len(up) - 1

    def predict(
        self,
        g_entry_s: np.ndarray,
        g_floor_s: float,
        params: np.ndarray,
        enable_thermal: bool = True,
        enable_geom_scaling: bool = True,
    ):
        beta_hrs, beta_lrs, log_k_rst, s_field, v_on, log_rth_10ms, eta_geom = params
        rth_10ms = float(np.exp(log_rth_10ms)) if enable_thermal else 0.0
        eta = float(eta_geom) if enable_geom_scaling else 0.0
        g_in = np.maximum(np.asarray(g_entry_s, dtype=float), g_floor_s * 1.01)

        def hbn_i(vh, g_state):
            w_fil = g_state / (g_state + 8.0e-6)
            sh_hrs = np.sinh(np.clip(beta_hrs * vh, 0.0, 35.0)) / max(np.sinh(beta_hrs * 0.15), 1e-12)
            sh_lrs = np.sinh(np.clip(beta_lrs * vh, 0.0, 35.0)) / max(np.sinh(beta_lrs * 0.15), 1e-12)
            return g_state * 0.15 * ((1.0 - w_fil) * sh_hrs + w_fil * sh_lrs)

        def solve_pt(app_v, g_state):
            lo = np.zeros_like(g_state + app_v)
            hi = np.broadcast_to(app_v, lo.shape).copy()
            for _ in range(20):
                vh = 0.5 * (lo + hi)
                it = hbn_i(vh, g_state)
                if self.bare:
                    below = vh + self.rc * it < app_v
                else:
                    v1t = np.interp(np.minimum(it, self.imax_1t), self.levels, self.v_of_i)
                    below = (it < self.imax_1t * 0.9999) & (vh + self.rc * it + v1t < app_v)
                lo = np.where(below, vh, lo)
                hi = np.where(below, hi, vh)
            vh = 0.5 * (lo + hi)
            return np.maximum(hbn_i(vh, g_state), 1e-16), vh

        g = g_in.copy()
        g_tr = [g]
        for j in range(len(self.grid_v) - 1):
            a = self.grid_v[j]
            ds = abs(self.grid_v[j + 1] - a)
            it, vh = solve_pt(a, g)
            # Reference 10 mS (~100 ohm Bare 1M metallic filament); thinner 1T1M filaments scale up R_th
            fil_act = g / (g + 1.0e-5)
            rth_eff = rth_10ms * fil_act * np.power(1.0e-2 / np.maximum(g, 5.0e-6), eta)
            t_rat = 1.0 + (rth_eff / 300.0) * (it * vh)
            f_drive = s_field * (vh - v_on)
            th_drive = 18.0 * (1.0 - 1.0 / t_rat)
            rate = np.exp(np.clip(log_k_rst + f_drive + th_drive, -30.0, 25.0))
            g = np.maximum(g * np.exp(-rate * ds), g_floor_s)
            g_tr.append(g)

        g_tr = np.asarray(g_tr).T
        up_g = self.grid_v[: self.grid_apex + 1]
        dn_g = self.grid_v[self.grid_apex :][::-1]
        g_up = np.array([np.interp(self.wave[: self.apex + 1], up_g, r[: self.grid_apex + 1]) for r in g_tr])
        g_dn = np.array([np.interp(self.wave[self.apex + 1 :], dn_g, r[self.grid_apex :][::-1]) for r in g_tr])
        g_full = np.c_[g_up, g_dn]
        cur, vh = solve_pt(np.broadcast_to(self.wave, g_full.shape), g_full)
        return cur, vh, g_full
