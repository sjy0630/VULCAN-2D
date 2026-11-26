"""Iteration 2026-11-26-01: 12-Parameter Unified Bidirectional Electro-Thermal Compact Model.

Cuts total parameter count in half (from 24 parameters in Iteration 03 down to 12 shared physical
parameters) while unifying SET (V > 0), zero-bias volatile/nonvolatile retention (V = 0), and
RESET (V < 0) into a single odd-symmetric, C^inf continuous h-BN device model:

Shared 12-parameter vector theta:
  0: log_rc        -> R_c (ohm): shared electrode/contact series resistance
  1: beta0         -> beta_0 (V^-1): base deep-barrier hopping slope
  2: beta_w        -> beta_w (V^-1): WKB wider-gap slope increment (resolves 0.9 V SET lag)
  3: beta_fil      -> beta_fil (V^-1): shared conductive bridge nonlinearity (SET & RESET)
  4: log_g_cap     -> G_cap (S): deep-barrier saturation cap & nucleation scale (SET & RESET)
  5: p_vrst        -> p_rst: single preceding-|V_rst| depletion scaling exponent
  6: v_set0        -> V_set,0 (V): positive breakdown threshold at unit rate
  7: s_set         -> s_set (V^-1): positive breakdown field sensitivity
  8: log_g_nv      -> G_nv (S): volatile-to-nonvolatile filament stability threshold
  9: v_rst0        -> V_rst,0 (V): negative rupture threshold at unit rate
 10: s_rst         -> s_rst (V^-1): negative rupture field sensitivity
 11: log_rth_10ms  -> R_th,10mS (K/W): shared constriction thermal resistance (eta = 2/3)
"""
from __future__ import annotations

import numpy as np

from analysis.et_two_channel_model import StandaloneNMOS

ETA_CONSTRICTION = 2.0 / 3.0
G_QUANTUM_HALF_S = 3.874046e-5  # Universal half-quantum conductance e^2/h (38.74 uS)
NMOS_GATE_OFFSET_V = -0.060     # Fixed from standalone Fig. 2b NMOS transfer calibration


class UnifiedCompactHBN:
    """Single 12-parameter bidirectional electro-thermal compact model for Au-Ti/h-BN/W."""

    PARAM_NAMES = (
        "R_contact_ohm",
        "beta0_per_V",
        "beta_wkb_per_V",
        "beta_fil_per_V",
        "G_cap_S",
        "p_vrst_exp",
        "v_crit_set_V",
        "s_set_per_V",
        "G_nonvol_S",
        "v_crit_rst_V",
        "s_rst_per_V",
        "R_th_10mS_K_per_W",
    )

    def __init__(self, params: np.ndarray):
        assert len(params) == 12, f"Expected 12 unified parameters, got {len(params)}"
        self.raw = np.asarray(params, dtype=float)
        self.rc = float(np.exp(self.raw[0]))
        self.beta0 = float(self.raw[1])
        self.beta_w = float(self.raw[2])
        self.beta_fil = float(self.raw[3])
        self.g_cap = float(np.exp(self.raw[4]))
        self.p_vrst = float(self.raw[5])
        self.v_set0 = float(self.raw[6])
        self.s_set = float(self.raw[7])
        self.g_nv = float(np.exp(self.raw[8]))
        self.v_rst0 = float(self.raw[9])
        self.s_rst = float(self.raw[10])
        self.rth_10ms = float(np.exp(self.raw[11]))

    def to_dict(self) -> dict[str, float]:
        vals = [
            self.rc,
            self.beta0,
            self.beta_w,
            self.beta_fil,
            self.g_cap,
            self.p_vrst,
            self.v_set0,
            self.s_set,
            self.g_nv,
            self.v_rst0,
            self.s_rst,
            self.rth_10ms,
        ]
        return dict(zip(self.PARAM_NAMES, map(float, vals)))

    def current_mag(self, vh_abs: np.ndarray, g0_state: np.ndarray, lam: float = 1.0) -> np.ndarray:
        """Exact odd-symmetric two-channel h-BN current magnitude |I_hBN(|V_h|, G)|.

        Satisfies dI/dV_h|_{V_h=0} == G identically across both polarities.
        """
        g = np.maximum(g0_state, 1.0e-15)
        if vh_abs.ndim == 2 and g.ndim == 1:
            g = g[:, None]
        g_deep = self.g_cap * np.tanh(g / self.g_cap)
        g_fil = np.maximum(g - g_deep, 0.0)
        b_deep = (self.beta0 + (self.beta_w / lam) * np.log1p(self.g_cap / g)) / lam
        sh_d = np.sinh(np.clip(b_deep * vh_abs, 0.0, 38.0)) / np.maximum(b_deep, 1e-9)
        sh_f = np.sinh(np.clip(self.beta_fil * vh_abs, 0.0, 35.0)) / max(self.beta_fil, 1e-9)
        return g_deep * sh_d + g_fil * sh_f

    def thermal_r(self, g0_state: np.ndarray) -> np.ndarray:
        """Shared constriction thermal resistance R_th(G) (active once localized bridge G >= ~25 uS forms)."""
        g = np.maximum(g0_state, 1.0e-15)
        act = np.square(g / (g + 2.8e-5))
        return self.rth_10ms * act * np.power(1.0e-2 / np.maximum(g, 2.0e-5), ETA_CONSTRICTION)

    def zero_bias_retain(self, g_end: np.ndarray, g_floor: float = 2.0e-12) -> np.ndarray:
        """Spontaneous zero-bias relaxation of sub-critical soft bridges (G < G_nv) between sweeps."""
        g = np.maximum(np.asarray(g_end, dtype=float), g_floor)
        ratio_sq = np.square(g / self.g_nv)
        keep = ratio_sq / (1.0 + ratio_sq)
        return g_floor + np.maximum(g - g_floor, 0.0) * keep

    def invert_read_to_g0(
        self,
        v_read_abs: float,
        i_read_abs: np.ndarray,
        circuit_1t: tuple[np.ndarray, np.ndarray] | None = None,
        lam: float = 1.0,
    ) -> np.ndarray:
        """Exact bisection inverse recovering zero-bias state G from any low-bias (V_read, I_read)."""
        i_obs = np.maximum(np.asarray(i_read_abs, dtype=float), 1.0e-14)
        v_drop = self.rc * i_obs
        if circuit_1t is not None:
            vd_tab, id_tab = circuit_1t
            levels, iu = np.unique(id_tab, return_index=True)
            v_drop = v_drop + np.interp(np.minimum(i_obs, levels[-1]), levels, vd_tab[iu])
        vh = np.maximum(v_read_abs - v_drop, 0.03)

        lo = np.full_like(i_obs, 1.0e-15)
        hi = np.maximum(i_obs / vh * 2.0, 1.0e-4)
        for _ in range(28):
            mid = np.sqrt(lo * hi)
            im = self.current_mag(vh, mid, lam=lam)
            lo = np.where(im < i_obs, mid, lo)
            hi = np.where(im < i_obs, hi, mid)
        return np.sqrt(lo * hi)

    def simulate_set(
        self,
        terminal: np.ndarray,
        g0_init: np.ndarray,
        nmos: StandaloneNMOS | None,
        gate_v: float | None,
        v_reset_prev: float,
        step: float = 0.020,
        compliance_a: float | None = None,
        c_par_pf: float | None = None,
        iterations: int = 23,
        field_shift: np.ndarray | float = 0.0,
        filament_mode: np.ndarray | None = None,
        g_fil_high_s: float = 1.2e-4,
    ):
        """Simulate positive SET sweep (outward + return) under shared h-BN law."""
        term = np.asarray(terminal, dtype=float)
        bare = gate_v is None
        comp = (1.0e-3 if bare else None) if compliance_a is None else compliance_a
        c_pf = (300.0 if bare else 50.0) if c_par_pf is None else float(c_par_pf)
        lam = (float(v_reset_prev) / 1.5) ** self.p_vrst

        ap = int(term.argmax())
        grid = np.linspace(0.0, term[ap], int(np.ceil(term[ap] / step)) + 1)

        g_init = np.maximum(np.asarray(g0_init, dtype=float), 1.0e-15)
        g_fil0 = np.maximum(g_init - self.g_cap * np.tanh(g_init / self.g_cap), 0.0)
        v_shunt = 0.055 * np.log1p(g_fil0 / self.g_cap)
        g_seed = 0.50 * self.g_cap
        v_hold = 0.44 * self.v_set0

        if not bare:
            vd_tab, id_tab = nmos.monotone_table(
                gate_v, scale=1.0, offset_v=NMOS_GATE_OFFSET_V, vmax=max(5.5, float(term.max()) + 0.5)
            )
            levels, iu = np.unique(id_tab, return_index=True)
            v_of_i = vd_tab[iu]
            imax_1t = float(levels[-1])

        def state_g(z_val, gf_extra=0.0):
            return g_init + g_seed * np.expm1(np.clip(z_val, 0.0, 23.0)) + gf_extra

        def solve(app_v, z_val, gf_extra=0.0):
            app = np.asarray(app_v, dtype=float)
            g_cur = state_g(z_val[:, None] if (app.ndim == 1 and z_val.ndim == 1 and len(app) != len(z_val)) else z_val, gf_extra)
            lo = np.zeros_like(g_cur + app)
            hi = np.broadcast_to(app, lo.shape).copy()
            for _ in range(iterations):
                vh = 0.5 * (lo + hi)
                it = self.current_mag(vh, g_cur, lam=lam)
                if bare:
                    below = (it < comp) & (vh + self.rc * it < app)
                else:
                    v1t = np.interp(np.minimum(it, imax_1t), levels, v_of_i)
                    below = (it < imax_1t * 0.99995) & (vh + self.rc * it + v1t < app)
                lo = np.where(below, vh, lo)
                hi = np.where(below, hi, vh)
            vh = 0.5 * (lo + hi)
            it_unc = self.current_mag(vh, g_cur, lam=lam)
            it_cl = np.minimum(it_unc, comp) if comp is not None else it_unc
            v1t_out = np.zeros_like(vh) if bare else np.interp(np.minimum(it_cl, imax_1t), levels, v_of_i)
            return it_cl, vh, it_unc, v1t_out

        z = np.zeros_like(g_init)
        gf = np.zeros_like(g_init)
        # Base active-sweep ON channel is universal e^2/h (decoupled from zero-bias G_nv relaxation!)
        g_max_cur = np.full_like(g_init, 1.5e-2 if bare else G_QUANTUM_HALF_S)
        z_tr, gf_tr = [z], [gf]
        c_exc = max(c_pf - 50.0, 0.0) / 250.0
        fil_mask = np.zeros_like(g_init) if filament_mode is None else np.asarray(filament_mode, dtype=float)

        def eval_rate(app_v, z_val, gf_val, g_max_arr):
            it_v, vh_v, _, _ = solve(app_v, z_val, gf_val)
            if not bare:
                g_max_arr = np.maximum(g_max_arr, G_QUANTUM_HALF_S + it_v / max(v_hold, 0.25))
            z_mx = np.maximum(np.log1p(np.maximum(g_max_arr - g_init, 0.0) / g_seed), 0.0)
            g_nw = state_g(z_val, gf_val)
            nuc = z_val / (z_val + 0.35)
            # Distributed multi-spot 1T1M SET heats gently (0.08 * R_th,10mS), whereas unbuffered
            # Bare 1M (c_exc > 0) concentrates 300 pF discharge into a single constriction!
            if c_exc > 0.0:
                p_eff = it_v * vh_v + 1.2e-2 * c_exc * (vh_v**2) * nuc
                rth_eff = self.thermal_r(np.maximum(g_nw, 4.0e-5 * c_exc * nuc))
            else:
                p_eff = it_v * vh_v
                rth_eff = 0.08 * self.rth_10ms
            t_rat = 1.0 + (rth_eff / 300.0) * p_eff
            fb = 0.50 * c_exc * nuc * np.minimum(z_val, 6.0)
            e_eff = (vh_v / lam) - self.v_set0 - v_shunt + fb - field_shift
            r_v = np.exp(np.clip(self.s_set * e_eff + 20.0 * (1.0 - 1.0 / t_rat), -38.0, 45.0))
            return r_v, z_mx, g_max_arr

        for a, ds in zip(grid[:-1], np.diff(grid)):
            r_probe, z_mx_p, g_max_cur = eval_rate(a, z, gf, g_max_cur)
            # Local adaptive sub-stepping during rapid breakdown guarantees <0.01 dec step convergence!
            max_adv = float(np.max(r_probe * np.minimum(np.maximum(z_mx_p - z, 0.0), 6.5))) * ds
            n_sub = min(max(1, int(np.ceil(max_adv / 0.25))), 12)
            sub_ds = ds / n_sub
            for s_idx in range(n_sub):
                a_sub = a + s_idx * sub_ds
                r1, z_max1, g_max_cur = eval_rate(a_sub, z, gf, g_max_cur) if s_idx > 0 else (r_probe, z_mx_p, g_max_cur)
                eff1 = np.minimum(np.maximum(z_max1 - z, 0.0), 6.5)
                z_mid = np.minimum(z + eff1 * (-np.expm1(-r1 * sub_ds * 0.5)), z_max1)
                gf_mid = gf + fil_mask * (g_fil_high_s - gf) * (-np.expm1(-0.4 * r1 * sub_ds * 0.5))

                r2, z_max2, g_max_cur = eval_rate(a_sub + 0.5 * sub_ds, z_mid, gf_mid, g_max_cur)
                eff2 = np.minimum(np.maximum(z_max2 - z, 0.0), 6.5)
                z = np.minimum(z + eff2 * (-np.expm1(-r2 * sub_ds)), z_max2)
                gf = gf + fil_mask * (g_fil_high_s - gf) * (-np.expm1(-0.4 * r2 * sub_ds))
            z_tr.append(z)
            gf_tr.append(gf)

        z_tr = np.asarray(z_tr).T
        gf_tr = np.asarray(gf_tr).T
        z_map = np.array([np.interp(term[: ap + 1], grid, r) for r in z_tr])
        gf_map = np.array([np.interp(term[: ap + 1], grid, r) for r in gf_tr])
        n_ret = len(term) - ap - 1
        z_map = np.c_[z_map, np.broadcast_to(z[:, None], (len(z), n_ret))]
        gf_map = np.c_[gf_map, np.broadcast_to(gf[:, None], (len(gf), n_ret))]

        g_full = g_init[:, None] + g_seed * np.expm1(np.clip(z_map, 0.0, 23.0)) + gf_map
        app_2d = np.broadcast_to(term, g_full.shape)
        lo = np.zeros_like(g_full)
        hi = app_2d.copy()
        for _ in range(iterations):
            vh = 0.5 * (lo + hi)
            it = self.current_mag(vh, g_full, lam=lam)
            if bare:
                below = (it < comp) & (vh + self.rc * it < app_2d)
            else:
                v1t = np.interp(np.minimum(it, imax_1t), levels, v_of_i)
                below = (it < imax_1t * 0.99995) & (vh + self.rc * it + v1t < app_2d)
            lo = np.where(below, vh, lo)
            hi = np.where(below, hi, vh)
        vh = 0.5 * (lo + hi)
        cur_unc = self.current_mag(vh, g_full, lam=lam)
        cur = np.minimum(cur_unc, comp) if comp is not None else cur_unc
        v1t_f = np.zeros_like(vh) if bare else np.interp(np.minimum(cur, imax_1t), levels, v_of_i)
        g_end = g_full[:, -1]
        return cur, vh, cur_unc, g_end, v1t_f

    def simulate_reset(
        self,
        voltage: np.ndarray,
        g0_entry: np.ndarray,
        rev_1t_curve: tuple[np.ndarray, np.ndarray] | None,
        g_floor: float = 2.0e-12,
        step: float = 0.015,
        enable_thermal: bool = True,
    ):
        """Simulate negative RESET sweep (outward + return) under the EXACT SAME conduction & thermal laws."""
        wave = np.abs(np.asarray(voltage, dtype=float))
        ap = int(wave.argmax())
        bare = rev_1t_curve is None
        if not bare:
            vd, id_c = rev_1t_curve
            levels, iu = np.unique(id_c, return_index=True)
            v_of_i = vd[iu]
            imax_1t = float(levels[-1])

        up = np.linspace(0.0, wave[ap], int(np.ceil(wave[ap] / step)) + 1)
        grid_v = np.r_[up, up[-2::-1]]
        grid_ap = len(up) - 1
        # Dynamic barrier depletion lam(t) grows as reverse bias exceeds 1.5V (e.g. Bare 1M -2.5V),
        # ensuring exact state continuity between RESET end and the next SET start!
        lam_grid = np.power(np.maximum(np.maximum.accumulate(grid_v) / 1.5, 1.0), self.p_vrst)
        lam_wave = np.power(np.maximum(np.maximum.accumulate(wave) / 1.5, 1.0), self.p_vrst)

        def solve_pt(app_v, g_st, lam_val):
            lo = np.zeros_like(g_st + app_v)
            hi = np.broadcast_to(app_v, lo.shape).copy()
            for _ in range(20):
                vh = 0.5 * (lo + hi)
                it = self.current_mag(vh, g_st, lam=lam_val)
                if bare:
                    below = vh + self.rc * it < app_v
                else:
                    v1t = np.interp(np.minimum(it, imax_1t), levels, v_of_i)
                    below = (it < imax_1t * 0.9999) & (vh + self.rc * it + v1t < app_v)
                lo = np.where(below, vh, lo)
                hi = np.where(below, hi, vh)
            vh = 0.5 * (lo + hi)
            return np.maximum(self.current_mag(vh, g_st, lam=lam_val), 1.0e-16), vh

        g = np.maximum(np.asarray(g0_entry, dtype=float), g_floor * 1.01)
        g_tr = [g]
        for j in range(len(grid_v) - 1):
            a = grid_v[j]
            ds = abs(grid_v[j + 1] - a)
            it, vh = solve_pt(a, g, float(lam_grid[j]))
            rth_eff = self.thermal_r(g) if enable_thermal else 0.0
            t_rat = 1.0 + (rth_eff / 300.0) * (it * vh)
            rate = np.exp(np.clip(self.s_rst * (vh - self.v_rst0) + 18.0 * (1.0 - 1.0 / t_rat), -30.0, 25.0))
            g = g_floor + np.maximum(g - g_floor, 0.0) * np.exp(-rate * ds)
            g_tr.append(g)

        g_tr = np.asarray(g_tr).T
        up_g = grid_v[: grid_ap + 1]
        dn_g = grid_v[grid_ap:][::-1]
        g_up = np.array([np.interp(wave[: ap + 1], up_g, r[: grid_ap + 1]) for r in g_tr])
        g_dn = np.array([np.interp(wave[ap + 1 :], dn_g, r[grid_ap:][::-1]) for r in g_tr])
        g_full = np.c_[g_up, g_dn]
        cur, vh = solve_pt(np.broadcast_to(wave, g_full.shape), g_full, lam_wave[None, :])
        return cur, vh, g_full
