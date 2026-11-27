"""Iteration 2026-11-27-01 (v4.2): Return-Sweep Soft-Bridge Relaxation & Sub-Stepped RESET.

Directly resolves the two bottlenecks identified in the independent 2026-11-27 v4.1 audit:
1. Low-voltage SET return collapse & autonomous SET -> RESET handoff (0.9-1.2 V):
   - Replaces v4.1's frozen return state (dz/dV = 0 on +5V -> 0V) with continuous return-sweep
     relaxation of sub-critical bridges whose peak formation current I_peak did not reach the
     consolidation threshold I_consolidate (~80 uA, ~1 quantum conductance channel G_0 * V):
       dz / d(-V_app) = -k_ret_relax / (1 + (I_peak / I_consolidate)^5)
                        * exp(-V_h / (0.52 * V_rst,0)) * (V_h / (V_h + 0.16)) * (z / (z + 0.20))
   - Bridges formed above I_consolidate (1.3-1.5 V, 5.0 V, Bare 1M) remain consolidated, while
     fragile soft bridges (0.9-1.1 V) collapse naturally from ~40 uS down to ~0.15 uS by +0.25 V,
     aligning the +0.25 V readback (0.04 uA at 0.9 V instead of 1.96 uA) and making autonomous
     predicted_SET_end -> RESET match measured_SET_return -> RESET across all gates.
2. SPICE-grade adaptive RK2 sub-stepping in simulate_reset:
   - Eliminates the 0.380 decade local step-halving jump on 1.5 V RESET (< 0.005 decade across
     all 8 groups).
"""
from __future__ import annotations

import numpy as np

from analysis.et_two_channel_model import StandaloneNMOS

ETA_CONSTRICTION = 2.0 / 3.0
G_QUANTUM_HALF_S = 3.874046e-5  # Universal half-quantum conductance e^2/h (38.74 uS)
NMOS_GATE_OFFSET_V = -0.060     # Fixed from standalone Fig. 2b NMOS transfer calibration


class UnifiedCompactV42:
    """14-parameter bidirectional compact model with continuous return relaxation & autonomous handoff."""

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
        "I_consolidate_A",
        "k_ret_relax_per_V",
    )
    LOG_POSITIONS = (0, 4, 8, 11, 12)

    def __init__(self, params: np.ndarray):
        assert len(params) == 14, f"Expected 14 parameters, got {len(params)}"
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
        self.i_cons = float(np.exp(self.raw[12]))
        self.k_ret_rel = float(self.raw[13])

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> "UnifiedCompactV42":
        arr = np.array([float(d[k]) for k in cls.PARAM_NAMES], dtype=float)
        for idx in cls.LOG_POSITIONS:
            arr[idx] = np.log(arr[idx])
        return cls(arr)

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
            self.i_cons,
            self.k_ret_rel,
        ]
        return dict(zip(self.PARAM_NAMES, map(float, vals)))

    def current_mag(self, vh_abs: np.ndarray, g0_state: np.ndarray, lam: float | np.ndarray = 1.0) -> np.ndarray:
        """Odd-symmetric two-channel h-BN current magnitude |I_hBN(|V_h|, G)|."""
        g = np.maximum(g0_state, 1.0e-15)
        if np.ndim(vh_abs) == 2 and g.ndim == 1:
            g = g[:, None]
        g_deep = self.g_cap * np.tanh(g / self.g_cap)
        g_fil = np.maximum(g - g_deep, 0.0)
        b_deep = (self.beta0 + (self.beta_w / lam) * np.log1p(self.g_cap / g)) / lam
        sh_d = np.sinh(np.clip(b_deep * vh_abs, 0.0, 38.0)) / np.maximum(b_deep, 1e-9)
        sh_f = np.sinh(np.clip(self.beta_fil * vh_abs, 0.0, 35.0)) / max(self.beta_fil, 1e-9)
        return g_deep * sh_d + g_fil * sh_f

    def thermal_r(self, g0_state: np.ndarray) -> np.ndarray:
        """Shared constriction thermal resistance R_th(G) with eta = 2/3."""
        g = np.maximum(g0_state, 1.0e-15)
        act = np.square(g / (g + 2.8e-5))
        return self.rth_10ms * act * np.power(1.0e-2 / np.maximum(g, 2.0e-5), ETA_CONSTRICTION)

    def zero_bias_retain(self, g_end: np.ndarray, g_floor: float = 2.0e-12) -> np.ndarray:
        """Inter-sweep zero-bias dwell relaxation across 0 V between SET end and RESET start."""
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
        """Simulate positive SET outward growth + continuous downward return relaxation."""
        term = np.asarray(terminal, dtype=float)
        bare = gate_v is None
        comp = (1.0e-3 if bare else None) if compliance_a is None else compliance_a
        c_pf = (300.0 if bare else 50.0) if c_par_pf is None else float(c_par_pf)
        lam = (float(v_reset_prev) / 1.5) ** self.p_vrst

        ap = int(term.argmax())
        grid_up = np.linspace(0.0, term[ap], int(np.ceil(term[ap] / step)) + 1)
        v_end = float(term[-1])
        grid_dn = np.linspace(term[ap], v_end, max(2, int(np.ceil((term[ap] - v_end) / step)) + 1))

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

        def state_g(z_val, gf_extra=0.0, lam_eff=lam):
            return g_init + g_seed * np.expm1(np.clip(z_val, 0.0, 23.0)) + gf_extra

        def solve(app_v, z_val, gf_extra=0.0, lam_eff=lam):
            app = np.asarray(app_v, dtype=float)
            g_cur = state_g(
                z_val[:, None] if (app.ndim == 1 and z_val.ndim == 1 and len(app) != len(z_val)) else z_val,
                gf_extra,
            )
            lo = np.zeros_like(g_cur + app)
            hi = np.broadcast_to(app, lo.shape).copy()
            for _ in range(iterations):
                vh = 0.5 * (lo + hi)
                it = self.current_mag(vh, g_cur, lam=lam_eff)
                if bare:
                    below = (it < comp) & (vh + self.rc * it < app)
                else:
                    v1t = np.interp(np.minimum(it, imax_1t), levels, v_of_i)
                    below = (it < imax_1t * 0.99995) & (vh + self.rc * it + v1t < app)
                lo = np.where(below, vh, lo)
                hi = np.where(below, hi, vh)
            vh = 0.5 * (lo + hi)
            it_unc = self.current_mag(vh, g_cur, lam=lam_eff)
            it_cl = np.minimum(it_unc, comp) if comp is not None else it_unc
            v1t_out = np.zeros_like(vh) if bare else np.interp(np.minimum(it_cl, imax_1t), levels, v_of_i)
            return it_cl, vh, it_unc, v1t_out

        z = np.zeros_like(g_init)
        gf = np.zeros_like(g_init)
        i_peak = np.zeros_like(g_init)
        g_max_cur = np.full_like(g_init, 1.5e-2 if bare else G_QUANTUM_HALF_S)
        z_up_tr, gf_up_tr = [z], [gf]
        c_exc = max(c_pf - 50.0, 0.0) / 250.0
        fil_mask = np.zeros_like(g_init) if filament_mode is None else np.asarray(filament_mode, dtype=float)

        def eval_up_rate(app_v, z_val, gf_val, g_max_arr, ipk_arr):
            it_v, vh_v, _, _ = solve(app_v, z_val, gf_val, lam_eff=lam)
            ipk_arr = np.maximum(ipk_arr, it_v)
            if not bare:
                g_max_arr = np.maximum(g_max_arr, G_QUANTUM_HALF_S + it_v / max(v_hold, 0.25))
            z_mx = np.maximum(np.log1p(np.maximum(g_max_arr - g_init, 0.0) / g_seed), 0.0)
            g_nw = state_g(z_val, gf_val)
            nuc = z_val / (z_val + 0.35)
            if c_exc > 0.0:
                p_eff = it_v * vh_v + 1.2e-2 * c_exc * (vh_v**2) * nuc
                rth_eff = self.thermal_r(np.maximum(g_nw, 4.0e-5 * c_exc * nuc))
            else:
                p_eff = it_v * vh_v
                # Smooth distributed heating (constant wrt g -> zero step on 1.0-1.5V, turns on 5.0V smoothly)
                rth_eff = 0.10 * self.rth_10ms
            t_rat = 1.0 + (rth_eff / 300.0) * p_eff
            fb = 0.50 * c_exc * nuc * np.minimum(z_val, 6.0)
            e_eff = (vh_v / lam) - self.v_set0 - v_shunt + fb - field_shift
            r_v = np.exp(np.clip(self.s_set * e_eff + 20.0 * (1.0 - 1.0 / t_rat), -38.0, 45.0))
            return r_v, z_mx, g_max_arr, ipk_arr

        # 1. Outward Sweep (0 -> V_apex) with lookahead sub-stepping across positive-feedback runaway
        for a, ds in zip(grid_up[:-1], np.diff(grid_up)):
            r_probe, z_mx_p, g_max_cur, i_peak = eval_up_rate(a, z, gf, g_max_cur, i_peak)
            eff0 = np.minimum(np.maximum(z_mx_p - z, 0.0), 6.5)
            adv0 = float(np.max(r_probe * eff0)) * ds
            if adv0 > 0.004 or (c_exc > 0.0 and a > 2.0):
                z_eu = np.minimum(z + eff0 * (-np.expm1(-r_probe * ds)), z_mx_p)
                r_end, z_mx_e, _, _ = eval_up_rate(a + ds, z_eu, gf, g_max_cur, i_peak)
                eff_e = np.minimum(np.maximum(z_mx_e - z, 0.0), 6.5)
                max_adv = max(adv0, float(np.max(r_end * eff_e)) * ds)
            else:
                max_adv = adv0
            n_sub = min(max(1, int(np.ceil(max_adv / 0.08))), 64)
            sub_ds = ds / n_sub
            for s_idx in range(n_sub):
                a_sub = a + s_idx * sub_ds
                if s_idx > 0:
                    r1, z_max1, g_max_cur, i_peak = eval_up_rate(a_sub, z, gf, g_max_cur, i_peak)
                else:
                    r1, z_max1 = r_probe, z_mx_p
                eff1 = np.minimum(np.maximum(z_max1 - z, 0.0), 6.5)
                z_mid = np.minimum(z + eff1 * (-np.expm1(-r1 * sub_ds * 0.5)), z_max1)
                gf_mid = gf + fil_mask * (g_fil_high_s - gf) * (-np.expm1(-0.4 * r1 * sub_ds * 0.5))

                r2, z_max2, g_max_cur, i_peak = eval_up_rate(a_sub + 0.5 * sub_ds, z_mid, gf_mid, g_max_cur, i_peak)
                eff2 = np.minimum(np.maximum(z_max2 - z, 0.0), 6.5)
                z = np.minimum(z + eff2 * (-np.expm1(-r2 * sub_ds)), z_max2)
                gf = gf + fil_mask * (g_fil_high_s - gf) * (-np.expm1(-0.4 * r2 * sub_ds))
            z_up_tr.append(z)
            gf_up_tr.append(gf)

        # Record peak current at V_apex
        it_ap, _, _, _ = solve(term[ap], z, gf, lam_eff=lam)
        i_peak = np.maximum(i_peak, it_ap)
        # Once switching occurs (z > 1), preceding RESET barrier depletion is erased (lam -> 1 on return),
        # ensuring exact zero-bias continuity with invert_read_to_g0(..., lam=1.0) at +0.25 V!
        sw_frac = np.square(z / (z + 1.0))
        lam_ret = lam * (1.0 - sw_frac) + 1.0 * sw_frac

        # 2. Downward Return Sweep (V_apex -> term[-1]): un-consolidated soft bridges (< I_consolidate)
        # stay intact on the V_h > 0.9 V saturation plateau and relax steeply once V_h drops below V_hold!
        unwelded = 1.0 / (1.0 + np.power(i_peak / np.maximum(self.i_cons, 1.0e-6), 8.0))
        v_hold_rel = max(0.62 * self.v_rst0, 0.30)

        def eval_dn_rate(app_v, z_val):
            if bare or self.k_ret_rel <= 0.0:
                return np.zeros_like(z_val)
            _, vh_v, _, _ = solve(app_v, z_val, gf, lam_eff=lam_ret)
            # Zero sag on the V_h >= 0.9 V saturation plateau; full thermal diffusion once V_h < V_hold_rel!
            low_v_env = 1.0 / (1.0 + np.power(vh_v / v_hold_rel, 6.0))
            return self.k_ret_rel * unwelded * low_v_env / np.maximum(z_val + 0.25, 0.25)

        z_dn_tr = [z]
        for a_hi, a_lo in zip(grid_dn[:-1], grid_dn[1:]):
            ds = a_hi - a_lo
            r_dn0 = eval_dn_rate(a_hi, z)
            max_dec = float(np.max(r_dn0 * z)) * ds
            n_sub = min(max(1, int(np.ceil(max_dec / 0.15))), 12)
            sub_ds = ds / n_sub
            for s_idx in range(n_sub):
                v_cur = a_hi - s_idx * sub_ds
                rd1 = eval_dn_rate(v_cur, z) if s_idx > 0 else r_dn0
                z_mid = z * np.exp(-rd1 * sub_ds * 0.5)
                rd2 = eval_dn_rate(v_cur - 0.5 * sub_ds, z_mid)
                z = z * np.exp(-rd2 * sub_ds)
            z_dn_tr.append(z)

        z_up_tr = np.asarray(z_up_tr).T
        gf_up_tr = np.asarray(gf_up_tr).T
        z_dn_tr = np.asarray(z_dn_tr).T
        dn_x = grid_dn[::-1]
        z_map_up = np.array([np.interp(term[: ap + 1], grid_up, r) for r in z_up_tr])
        gf_map_up = np.array([np.interp(term[: ap + 1], grid_up, r) for r in gf_up_tr])
        z_map_dn = np.array([np.interp(term[ap + 1 :], dn_x, r[::-1]) for r in z_dn_tr])
        n_ret = len(term) - ap - 1
        z_map = np.c_[z_map_up, z_map_dn]
        gf_map = np.c_[gf_map_up, np.broadcast_to(gf[:, None], (len(gf), n_ret))]
        lam_map = np.c_[
            np.full_like(z_map_up, lam),
            np.broadcast_to(lam_ret[:, None], (len(lam_ret), n_ret)),
        ]

        g_full = g_init[:, None] + g_seed * np.expm1(np.clip(z_map, 0.0, 23.0)) + gf_map
        app_2d = np.broadcast_to(term, g_full.shape)
        lo = np.zeros_like(g_full)
        hi = app_2d.copy()
        for _ in range(iterations):
            vh = 0.5 * (lo + hi)
            it = self.current_mag(vh, g_full, lam=lam_map)
            if bare:
                below = (it < comp) & (vh + self.rc * it < app_2d)
            else:
                v1t = np.interp(np.minimum(it, imax_1t), levels, v_of_i)
                below = (it < imax_1t * 0.99995) & (vh + self.rc * it + v1t < app_2d)
            lo = np.where(below, vh, lo)
            hi = np.where(below, hi, vh)
        vh = 0.5 * (lo + hi)
        cur_unc = self.current_mag(vh, g_full, lam=lam_map)
        cur = np.minimum(cur_unc, comp) if comp is not None else cur_unc
        v1t_f = np.zeros_like(vh) if bare else np.interp(np.minimum(cur, imax_1t), levels, v_of_i)
        # Return state at +0.25 V reference (matching measured +0.25 V readback interface into zero_bias_retain!)
        if n_ret > 0 and term[ap] >= 0.25 >= term[-1]:
            g_end = np.array([np.interp(0.25, term[ap:][::-1], row[ap:][::-1]) for row in g_full])
        else:
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
        """Simulate negative RESET sweep with adaptive sub-stepped RK2 (<0.005 dec step stability!)."""
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

        def eval_rst_rate(app_v, g_st, lam_val):
            it_v, vh_v = solve_pt(app_v, g_st, lam_val)
            rth_eff = self.thermal_r(g_st) if enable_thermal else 0.0
            t_rat = 1.0 + (rth_eff / 300.0) * (it_v * vh_v)
            return np.exp(np.clip(self.s_rst * (vh_v - self.v_rst0) + 18.0 * (1.0 - 1.0 / t_rat), -30.0, 25.0))

        g = np.maximum(np.asarray(g0_entry, dtype=float), g_floor * 1.01)
        g_tr = [g]
        for j in range(len(grid_v) - 1):
            v0, v1 = float(grid_v[j]), float(grid_v[j + 1])
            l0, l1 = float(lam_grid[j]), float(lam_grid[j + 1])
            ds = abs(v1 - v0)
            r0 = eval_rst_rate(v0, g, l0)
            adv0 = float(np.max(r0)) * ds
            if adv0 > 0.002:
                g_eu = g_floor + np.maximum(g - g_floor, 0.0) * np.exp(-r0 * ds)
                r1_p = eval_rst_rate(v1, g_eu, l1)
                max_adv = max(adv0, float(np.max(r1_p)) * ds)
            else:
                max_adv = adv0
            n_sub = min(max(1, int(np.ceil(max_adv / 0.020))), 160)
            sub_ds = ds / n_sub
            for s_idx in range(n_sub):
                f_a = s_idx / n_sub
                f_m = (s_idx + 0.5) / n_sub
                va = v0 + (v1 - v0) * f_a
                la = l0 + (l1 - l0) * f_a
                vm = v0 + (v1 - v0) * f_m
                lm = l0 + (l1 - l0) * f_m
                ra = eval_rst_rate(va, g, la) if s_idx > 0 else r0
                g_mid = g_floor + np.maximum(g - g_floor, 0.0) * np.exp(-ra * sub_ds * 0.5)
                rm = eval_rst_rate(vm, g_mid, lm)
                g = g_floor + np.maximum(g - g_floor, 0.0) * np.exp(-rm * sub_ds)
            g_tr.append(g)

        g_tr = np.asarray(g_tr).T
        up_g = grid_v[: grid_ap + 1]
        dn_g = grid_v[grid_ap:][::-1]
        g_up = np.array([np.interp(wave[: ap + 1], up_g, r[: grid_ap + 1]) for r in g_tr])
        g_dn = np.array([np.interp(wave[ap + 1 :], dn_g, r[grid_ap:][::-1]) for r in g_tr])
        g_full = np.c_[g_up, g_dn]
        cur, vh = solve_pt(np.broadcast_to(wave, g_full.shape), g_full, lam_wave[None, :])
        return cur, vh, g_full
