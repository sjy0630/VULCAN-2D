"""VULCAN-2D v0.4: current-controlled soft/hard switching regimes in h-BN.

This module extends, rather than silently replaces, the calibrated v0.3 model.
The measured evidence now contains three current-limited 1T1R SET ensembles
(``V_G`` = 0.9, 1.1, and 1.3 V) and a standalone 1R forming ensemble measured
with a 10 mA instrument compliance.  Those data identify two *device-level
working regimes*:

* 1T1R: a transistor-limited, high-resistance, progressive soft-path regime;
* 1R: an abrupt hard-path regime once a local runaway/percolation threshold is
  crossed, after which the measured SET current can be dominated by the 10 mA
  instrument clamp.  The standalone RESET sweep has no user-set compliance.

The state variables are deliberately mechanism-neutral:

``phi_k``
    occupation/connectivity of high-transmission configurations in local weak
    region k.  Literature-compatible candidates include intrinsic vacancy
    clusters/inter-layer bridges and confined metal-assisted paths.

``h_j`` / ``h``
    survival of coarse hard links and their weighted total connectivity.  SET
    creates the hard state; RESET ruptures the links over a distributed voltage
    interval.  A metal-rich CNF and a strongly percolated intrinsic-defect path
    remain competing microscopic realizations until state-resolved
    XTEM/EDS/EELS is available.

The evidence set now also includes multi-gate 1T1R RESET sweeps, standalone 1R
RESET sweeps, and the cross-sectional TEM inset in Fig. 1h of Zhu et al.  The
TEM constrains the as-fabricated Au-Ti/h-BN/W stack to approximately 18 layers
and 6 nm.  It is structural evidence, not a state-resolved image of a conductive
filament.  The voltage-domain exposure below removes numerical dependence on the
workbook voltage increment, but is not a substitute for pulse or dwell time.
"""

from dataclasses import dataclass
from math import gamma
from typing import Optional

import numpy as np

from . import model as v3


@dataclass
class Params(v3.Params):
    """v0.4 parameters.

    v0.3 parameters are inherited so the already validated 1.1 V transport,
    variability, and RESET model remain available.  New parameters are either
    directly extracted from the added workbooks or explicitly marked as
    provisional regime-transition parameters.
    """

    # Fig. 1h XTEM / paper geometry.  ``K`` inherited from v0.3 remains the
    # number of lateral coarse-grained weak-region quadrature points; it is not
    # replaced by the approximately 18 atomically resolved h-BN layers.
    hbn_thickness_nm: float = 6.0
    hbn_reference_thickness_nm: float = 6.0
    hbn_layers: int = 18
    via_diameter_nm: float = 260.0
    device_area_um2: float = 0.053

    # Measured 1T1R SET apex-current anchors (medians of 100/53/101 cycles).
    # Interpolation is performed in log(I), which preserves positivity and is
    # only intended inside the measured 0.9--1.3 V gate window.
    Vg_low: float = 0.9
    Vg_ref: float = 1.1
    Vg_high: float = 1.3
    Icc_low: float = 36.912e-6
    Icc_ref: float = 51.46e-6
    Icc_high: float = 93.304e-6
    # The h-BN retains a finite voltage at the sweep apex, so V_DS is below 5 V.
    # This single load-line headroom factor maps the target terminal ceiling to
    # the internal transistor branch; it is not a MOSFET mobility parameter.
    loadline_headroom: float = 1.105
    # Return-branch conductance grows faster than the current ceiling: at higher
    # V_G the transistor not only passes more terminal current but leaves a more
    # connected h-BN state.  The exponent is constrained by the measured 1 V
    # return-current ratio across 0.9/1.3 V.  It is an effective path-development
    # law, consistent with compliance-controlled incomplete/complete paths, not
    # proof of a particular migrating ion.
    gate_path_exponent: float = 0.72

    # Measured multi-gate RESET current maxima.  These are terminal-current
    # anchors, not standalone MOSFET parameters.  Their ratios scale the
    # calibrated v0.3 negative-polarity transistor branch, while the load-line
    # solution determines the internal h-BN voltage and the RESET onset.
    Ireset_peak_low: float = 0.42509e-6
    Ireset_peak_ref: float = 3.7743e-6
    Ireset_peak_high: float = 21.524e-6
    Vreset_peak_low: float = 1.410
    Vreset_peak_ref: float = 1.120
    Vreset_peak_high: float = 1.040
    reset_loadline_headroom: float = 1.04
    Vk_reset_v4: float = 0.22
    lam_reset_v4: float = 0.03
    # Polarity-specific effective path factors are required by the measured
    # low-voltage RESET initial/final currents, especially close to the MOSFET
    # threshold at V_G=0.9 V.  They describe the observable 1T1R branch; they
    # must not be re-labelled as intrinsic h-BN conductivities.
    reset_connected_scale_low: float = 0.0022
    reset_connected_scale_ref: float = 0.30
    reset_connected_scale_high: float = 0.42
    reset_perc_exponent_low: float = 2.0
    reset_perc_exponent_ref: float = 2.0
    reset_perc_exponent_high: float = 2.0
    reset_off_scale_low: float = 0.008
    reset_off_scale_ref: float = 0.74
    reset_off_scale_high: float = 0.95
    # Effective RESET-return field coefficients obtained from the three median
    # terminal return branches.  They remain device-level transport parameters
    # because the series voltage partition is not independently measured.
    alpha_reset_low_v4: float = 7.53
    alpha_reset_ref_v4: float = 6.00
    alpha_reset_high_v4: float = 6.51
    alpha_reset_on_low_v4: float = 10.0
    alpha_reset_on_ref_v4: float = 7.666
    alpha_reset_on_high_v4: float = 7.666
    # Map the measured terminal RESET peak positions onto a common local-stress
    # threshold after the self-consistent voltage division is solved.
    reset_drive_scale_low: float = 0.62
    reset_drive_scale_ref: float = 0.91
    reset_drive_scale_high: float = 0.90
    # The weak state written at 0.9 V erases at a lower reconstructed stress.
    # This is a state-stability factor, not a second measured activation energy.
    reset_threshold_factor_low: float = 0.44
    reset_threshold_factor_ref: float = 1.00
    reset_threshold_factor_high: float = 1.00

    # Slower voltage-domain RESET kinetics resolve the peak-and-collapse shape;
    # the old v0.3 value (2.0 per 0.02-V row) collapsed the state too abruptly.
    Kc_reset: float = 0.20

    # Quasi-static voltage exposure.  Kc_set/reset in v0.3 were calibrated per
    # 0.02 V sample.  Scaling every update by |dV|/dv_ref makes resampling the
    # same ramp at 0.01 or 0.05 V approximately invariant.
    dv_ref: float = 0.02

    # State drive uses the reconstructed h-BN voltage.  A small terminal-field
    # fraction is retained as a reduced-order representation of unresolved
    # local field concentration.  It is provisional and must not be presented
    # as an independently measured electrostatic factor.
    local_field_fraction: float = 0.20

    # Standalone 1R pre-forming transport, fitted to the median forward branch
    # below 2.4 V across 22 sweeps: RMSE ~0.13 decade.
    I0_1r: float = 9.99e-11
    alpha_1r: float = 3.315

    # Standalone 1R hard-path branch.  The crossing statistics come from the
    # first point with I >= 0.5 mA; RESET independently constrains the hard-state
    # low-voltage scale to approximately 96 ohm.  The sub-grid width encodes an
    # abrupt avalanche, not an atomistic barrier measurement.
    Vhard_mean: float = 2.675
    Vhard_sigma: float = 0.087
    hard_width: float = 0.008
    Rhard_median: float = 96.0
    Rhard_sigma_ln: float = 0.40
    # Measurement protocol confirmed by the experimenter: 10 mA compliance is
    # applied during standalone SET/forming.
    Icomp_1r: float = 10.0e-3

    # Standalone 1R RESET.  The initial negative-bias branch independently
    # confirms an approximately 96-ohm hard path.  A few parallel coarse links
    # reproduce the observed stepwise rupture without identifying each link as
    # one atomic layer or one uniquely metallic filament.
    hard_reset_links: int = 6
    Vhard_reset_peak_observed: float = 0.5125
    hard_reset_onset_mean: float = 0.55
    hard_reset_onset_sigma: float = 0.105
    hard_reset_completion_mean: float = 1.45
    hard_reset_completion_sigma: float = 0.385
    hard_reset_width: float = 0.018
    hard_reset_dominant_weight: float = 0.65
    # Measurement protocol confirmed by the experimenter: no user-set current
    # compliance is applied during standalone RESET.  ``None`` means that the
    # simulated terminal current is not numerically clipped.  The analyser's
    # finite hardware/current range still exists, but it is not a compliance
    # parameter in this reduced-order model.
    Icomp_1r_reset: Optional[float] = None


def _positive_log_interp(x, x0, y0, x1, y1):
    t = (x - x0) / (x1 - x0)
    return float(np.exp(np.log(y0) + t * (np.log(y1) - np.log(y0))))


def gate_current_ceiling(Vg, p):
    """Measured-data interpolation of the 1T1R SET current ceiling.

    Values outside 0.9--1.3 V are clipped because the present data do not
    identify threshold/subthreshold MOSFET behavior there.
    """

    vg = float(np.clip(Vg, p.Vg_low, p.Vg_high))
    if vg <= p.Vg_ref:
        return _positive_log_interp(vg, p.Vg_low, p.Icc_low,
                                    p.Vg_ref, p.Icc_ref)
    return _positive_log_interp(vg, p.Vg_ref, p.Icc_ref,
                                p.Vg_high, p.Icc_high)


def gate_path_factor(Vg, p):
    """Compliance-dependent development of the post-SET soft-path conductance."""

    ratio = gate_current_ceiling(Vg, p) / p.Icc_ref
    return float(ratio ** p.gate_path_exponent)


def reset_peak_anchor(Vg, p):
    """Interpolate the measured terminal RESET-current maximum in log space."""

    vg = float(np.clip(Vg, p.Vg_low, p.Vg_high))
    if vg <= p.Vg_ref:
        return _positive_log_interp(vg, p.Vg_low, p.Ireset_peak_low,
                                    p.Vg_ref, p.Ireset_peak_ref)
    return _positive_log_interp(vg, p.Vg_ref, p.Ireset_peak_ref,
                                p.Vg_high, p.Ireset_peak_high)


def _gate_interp(Vg, low, ref, high, p):
    vg = float(np.clip(Vg, p.Vg_low, p.Vg_high))
    if vg <= p.Vg_ref:
        return _positive_log_interp(vg, p.Vg_low, low, p.Vg_ref, ref)
    return _positive_log_interp(vg, p.Vg_ref, ref, p.Vg_high, high)


def reset_connected_path_factor(Vg, p):
    return _gate_interp(Vg, p.reset_connected_scale_low,
                        p.reset_connected_scale_ref,
                        p.reset_connected_scale_high, p)


def reset_off_path_factor(Vg, p):
    return _gate_interp(Vg, p.reset_off_scale_low, p.reset_off_scale_ref,
                        p.reset_off_scale_high, p)


def reset_drive_factor(Vg, p):
    return _gate_interp(Vg, p.reset_drive_scale_low, p.reset_drive_scale_ref,
                        p.reset_drive_scale_high, p)


def reset_threshold_factor(Vg, p):
    return _gate_interp(Vg, p.reset_threshold_factor_low,
                        p.reset_threshold_factor_ref,
                        p.reset_threshold_factor_high, p)


def _transport_params(side, Vg, p):
    if side != "reset":
        return p.I_s, p.alpha
    alpha = _gate_interp(Vg, p.alpha_reset_low_v4,
                         p.alpha_reset_ref_v4,
                         p.alpha_reset_high_v4, p)
    return p.I_s * p.I_s_reset_scale, alpha


def _G_ensemble(phi, lng, w, p, Gon_eff, side, Vg):
    """v0.4 path prefactor(s) with measured RESET observability factors.

    SET retains the v0.3 single transport channel.  RESET returns separate
    off-state and connected-state prefactors because the new return branches
    require a lower HRS field coefficient while the pre-rupture LRS remains
    strongly nonlinear.  A single ``G*sinh(alpha*V)`` cannot fit both limits.
    """

    if side != "reset":
        return v3.G_ensemble(phi, lng, w, p, Gon_eff, side)
    I_s, _ = _transport_params(side, Vg, p)
    exponent = _gate_interp(Vg, p.reset_perc_exponent_low,
                            p.reset_perc_exponent_ref,
                            p.reset_perc_exponent_high, p)
    theta = np.power(np.clip(phi, 0.0, 1.0), exponent)
    off = reset_off_path_factor(Vg, p)
    connected = reset_connected_path_factor(Vg, p)
    Goff = I_s * np.sum(w * off * np.exp(lng) * (1.0 - theta))
    Gon = I_s * np.sum(w * connected * Gon_eff * theta)
    return float(Goff), float(Gon)


def _hbn_current(V_h, G_state, Vg, side, p):
    vh = abs(float(V_h))
    if side != "reset":
        lim = p.alpha * p.Vclip
        return float(G_state) * np.sinh(min(p.alpha * vh, lim))
    Goff, Gon = G_state
    _, alpha_off = _transport_params("reset", Vg, p)
    alpha_on = _gate_interp(Vg, p.alpha_reset_on_low_v4,
                            p.alpha_reset_on_ref_v4,
                            p.alpha_reset_on_high_v4, p)
    return (Goff * np.sinh(min(alpha_off * vh, alpha_off * p.Vclip)) +
            Gon * np.sinh(min(alpha_on * vh, alpha_on * p.Vclip)))


def i_tr(V_tr, Vg, side, p, isat_mult=1.0):
    """Effective transistor output characteristic with an explicit gate input.

    For SET, the current at V_DS=5 V equals the measured gate-dependent ceiling.
    The V_DS shape retains the v0.3 empirical output law.  For RESET, the v0.3
    negative-polarity shape is scaled by the three measured terminal RESET peak
    anchors.  This is an effective circuit calibration, not a claim that those
    peaks are direct measurements of the standalone transistor saturation law.
    """

    Vt = abs(float(V_tr))
    if side == "reset":
        # The old v0.3 RESET branch had a very large linear-output factor and
        # no gate input.  The new data support a bounded, gate-indexed effective
        # current ceiling.  The small lambda term retains finite output slope.
        shape = ((1.0 + p.lam_reset_v4 * Vt) *
                 np.tanh(Vt / max(p.Vk_reset_v4, 1e-12)))
        return (isat_mult * p.reset_loadline_headroom *
                reset_peak_anchor(Vg, p) * shape)
    shape = (1.0 + p.lam_set * Vt) * np.tanh(Vt / p.Vk_set)
    shape_at_5 = (1.0 + 5.0 * p.lam_set) * np.tanh(5.0 / p.Vk_set)
    return (isat_mult * p.loadline_headroom * gate_current_ceiling(Vg, p) *
            shape / shape_at_5)


def divider_1t1r(V_app, G_ens, Vg, p, side="set", isat_mult=1.0, nbis=28):
    """Self-consistent series load line for the h-BN element and MOSFET."""

    Va = abs(float(V_app))
    if Va < 1e-12:
        return 0.0, 0.0
    lo, hi = 0.0, Va
    for _ in range(nbis):
        vh = 0.5 * (lo + hi)
        ih = _hbn_current(vh, G_ens, Vg, side, p)
        it = i_tr(Va - vh, Vg, side, p, isat_mult)
        if ih < it:
            lo = vh
        else:
            hi = vh
    Vh = 0.5 * (lo + hi)
    current = _hbn_current(Vh, G_ens, Vg, side, p)
    return current, np.sign(V_app) * Vh


def voltage_exposure(V_now, V_prev, p):
    """Dimensionless quasi-static exposure relative to the calibrated 0.02 V grid."""

    if V_prev is None:
        return 0.0
    return abs(float(V_now) - float(V_prev)) / max(p.dv_ref, 1e-12)


def hbn_field_MV_cm(V_h, p):
    """Average out-of-plane field across the XTEM-constrained h-BN thickness."""

    return 10.0 * abs(float(V_h)) / max(float(p.hbn_thickness_nm), 1e-12)


def _thickness_equivalent_voltage(V, p):
    """Map field at ``hbn_thickness_nm`` to an equivalent voltage at 6 nm.

    Existing kinetic thresholds were calibrated in volts for the approximately
    6-nm device.  This transform makes thickness scaling explicit while leaving
    the default calibration unchanged.
    """

    scale = p.hbn_reference_thickness_nm / max(p.hbn_thickness_nm, 1e-12)
    return abs(float(V)) * scale


def _state_drive(V_app, V_h, p, side="set", Vg=None):
    """XTEM-thickness-normalized local stress, expressed as equivalent volts."""

    eta = float(np.clip(p.local_field_fraction, 0.0, 1.0))
    mixed = (1.0 - eta) * abs(V_h) + eta * abs(V_app)
    drive = _thickness_equivalent_voltage(mixed, p)
    if side == "reset":
        if Vg is None:
            raise ValueError("Vg is required for the gate-dependent RESET drive")
        drive *= reset_drive_factor(Vg, p)
    return drive


def step_soft(phi, V_drive, exposure, T, Vth_patch, side, p, phi_min=None):
    """Grid-invariant bounded soft-path kinetics."""

    phi_min = p.phi_floor if phi_min is None else phi_min
    if exposure <= 0:
        return np.clip(phi.copy(), phi_min, 1.0)
    if side == "set":
        arr = np.exp((p.Ea_set / v3.KB) * (1.0 / p.T0 - 1.0 / T))
        drive = np.sinh(np.clip(p.beta_set * (V_drive - Vth_patch), -39, 39))
        rate = p.Kc_set * exposure * arr * np.clip(drive, 0, None)
        equilibrium = np.ones_like(phi)
    else:
        arr = np.exp((p.Ea_reset / v3.KB) * (1.0 / p.T0 - 1.0 / T))
        drive = np.sinh(np.clip(p.beta_reset * (V_drive - Vth_patch), -39, 39))
        rate = p.Kc_reset * exposure * arr * np.clip(drive, 0, None)
        equilibrium = np.full_like(phi, phi_min)
    return np.clip(equilibrium + (phi - equilibrium) * np.exp(-rate), phi_min, 1.0)


def run_1t1r_sweep(Vwave, phi0, lng, w, p, side, Vg, isat_mult,
                    Vth_c, dtheta, Gon_eff, phi_min=None):
    """Integrate one 1T1R voltage sweep and expose internal-voltage diagnostics."""

    phi = np.asarray(phi0, float).copy()
    Vwave = np.asarray(Vwave, float)
    phi_min = p.phi_floor if phi_min is None else phi_min
    Vth_patch = Vth_c + dtheta
    I = np.empty_like(Vwave)
    Vh = np.empty_like(Vwave)
    pb = np.empty_like(Vwave)
    drive_hist = np.empty_like(Vwave)
    field_hist = np.empty_like(Vwave)
    previous = None
    for idx, Va in enumerate(Vwave):
        Gens = _G_ensemble(phi, lng, w, p, Gon_eff, side, Vg)
        current, vh = divider_1t1r(Va, Gens, Vg, p, side, isat_mult)
        temperature = (p.T0 if p.Rth <= 0 else
                       min(p.T0 + p.Rth * abs(current * vh), p.Tmax))
        drive = _state_drive(Va, vh, p, side, Vg)
        exposure = voltage_exposure(Va, previous, p)
        phi = step_soft(phi, drive, exposure, temperature, Vth_patch,
                        side, p, phi_min)
        I[idx], Vh[idx] = np.sign(Va) * current, vh
        pb[idx] = float(np.sum(w * phi))
        drive_hist[idx] = drive
        field_hist[idx] = hbn_field_MV_cm(vh, p)
        previous = Va
    return I, Vh, pb, drive_hist, field_hist, phi


def _cycle_thresholds(p, rng, n_cycles, side="set"):
    mean = p.Vth_set0 if side == "set" else p.Vth_reset0
    shape = p.m_set if side == "set" else p.m_reset
    eta = mean / gamma(1.0 + 1.0 / shape)
    previous = mean
    for _ in range(n_cycles):
        sample = eta * rng.weibull(shape)
        current = (mean + p.rho_acf * (previous - mean) +
                   np.sqrt(1.0 - p.rho_acf ** 2) * (sample - mean))
        previous = current
        yield current


def simulate_1t1r_set_ensemble(p=None, Vg=1.1, n_cycles=53, seed=0,
                               step=0.02, device_seed=2026):
    """Independent positive SET sweeps for multi-gate comparison.

    The 0.9/1.3 V files do not contain the intervening RESET waveform.  Each
    sweep is therefore initialized from an HRS distribution, while the frozen
    patch geometry remains the same device.  This is a protocol assumption,
    not evidence that the material returns to a pristine atomic state.
    """

    p = Params() if p is None else p
    rng = np.random.default_rng(seed)
    w, dtheta = v3.make_frozen(p, np.random.default_rng(device_seed))
    wave = v3.triangle(5.0, step=step)
    cycles = []
    for Vth_c in _cycle_thresholds(p, rng, n_cycles):
        lng = rng.normal(0.0, p.sigma_lnG, p.K)
        isat_mult = rng.normal(1.0, p.icc_noise)
        Gon_eff = max(p.Gon * gate_path_factor(Vg, p) *
                      np.exp(rng.normal(0.0, p.sigma_Gon)), 1.0)
        phi0 = np.full(p.K, p.phi_floor)
        I, Vh, pb, drive, field, phi_final = run_1t1r_sweep(
            wave, phi0, lng, w, p, "set", Vg, isat_mult,
            Vth_c, dtheta, Gon_eff, p.phi_floor,
        )
        cycles.append(dict(Vs=wave.copy(), Is=I, Vhs=Vh, pbs=pb,
                           Vdrive=drive, Ehs_MV_cm=field,
                           phi_final=phi_final.copy(), lng=lng.copy(),
                           Gon_eff=Gon_eff, isat_mult=isat_mult,
                           Vth_set_c=Vth_c, Vg=float(Vg)))
    return cycles, dict(w=w, dtheta=dtheta)


def simulate_1t1r_cycle_ensemble(p=None, Vg=1.1, n_cycles=53, seed=0,
                                 set_step=0.02, reset_step=0.02,
                                 reset_vpeak=1.7, device_seed=2026):
    """Sequential SET then RESET sweeps with state continuity.

    The final ``phi_k`` from SET is the initial condition for RESET.  This is
    now testable because the 0.9-V and 1.3-V workbooks contain matching RESET
    ensembles.  SET and RESET thresholds remain separate stochastic draws;
    the electrical data do not identify one shared atomistic barrier.
    """

    p = Params() if p is None else p
    rng = np.random.default_rng(seed)
    reset_rng = np.random.default_rng(seed + 100003)
    w, dtheta = v3.make_frozen(p, np.random.default_rng(device_seed))
    set_thresholds = _cycle_thresholds(p, rng, n_cycles, "set")
    reset_thresholds = _cycle_thresholds(p, reset_rng, n_cycles, "reset")
    set_wave = v3.triangle(5.0, step=set_step)
    reset_wave = v3.triangle(-abs(reset_vpeak), step=reset_step)
    cycles = []
    for Vth_set_c, Vth_reset_c in zip(set_thresholds, reset_thresholds):
        lng = rng.normal(0.0, p.sigma_lnG, p.K)
        isat_mult = rng.normal(1.0, p.icc_noise)
        Gon_eff = max(p.Gon * gate_path_factor(Vg, p) *
                      np.exp(rng.normal(0.0, p.sigma_Gon)), 1.0)
        phi0 = np.full(p.K, p.phi_floor)
        Is, Vhs, pbs, sdrive, sfield, phi_set = run_1t1r_sweep(
            set_wave, phi0, lng, w, p, "set", Vg, isat_mult,
            Vth_set_c, dtheta, Gon_eff, p.phi_floor,
        )
        Ir, Vhr, pbr, rdrive, rfield, phi_reset = run_1t1r_sweep(
            reset_wave, phi_set, lng, w, p, "reset", Vg, isat_mult,
            Vth_reset_c * reset_threshold_factor(Vg, p),
            dtheta * reset_threshold_factor(Vg, p),
            Gon_eff, p.phi_floor,
        )
        cycles.append(dict(
            Vs=set_wave.copy(), Is=Is, Vhs=Vhs, pbs=pbs,
            Vset_drive=sdrive, Eset_MV_cm=sfield,
            Vr=reset_wave.copy(), Ir=Ir, Vhr=Vhr, pbr=pbr,
            Vreset_drive=rdrive, Ereset_MV_cm=rfield,
            phi_after_set=phi_set.copy(), phi_after_reset=phi_reset.copy(),
            Vth_set_c=Vth_set_c, Vth_reset_c=Vth_reset_c,
            Gon_eff=Gon_eff, Vg=float(Vg),
        ))
    return cycles, dict(w=w, dtheta=dtheta)


def simulate_1t1r_reset_ensemble(p=None, Vg=1.1, n_cycles=53, seed=0,
                                 set_step=0.02, reset_step=0.02,
                                 reset_vpeak=1.7, device_seed=2026):
    """Convenience wrapper returning the RESET half of sequential cycles."""

    cycles, frozen = simulate_1t1r_cycle_ensemble(
        p, Vg, n_cycles, seed, set_step, reset_step, reset_vpeak, device_seed,
    )
    out = []
    for c in cycles:
        out.append(dict(Vs=c["Vr"], Is=c["Ir"], Vhs=c["Vhr"], pbs=c["pbr"],
                        Vdrive=c["Vreset_drive"], Ehs_MV_cm=c["Ereset_MV_cm"],
                        phi_initial=c["phi_after_set"],
                        phi_final=c["phi_after_reset"], Vg=c["Vg"]))
    return out, frozen


def _hard_activation(V, Vcrit, width):
    z = np.clip((abs(V) - Vcrit) / max(width, 1e-9), -60.0, 60.0)
    return float(1.0 / (1.0 + np.exp(-z)))


def simulate_1r_set_ensemble(p=None, n_cycles=22, seed=0, step=0.05,
                             vpeak=5.0, compliance=None):
    """Standalone 1R forming/SET sweeps with an explicit hard-path branch.

    ``h`` is monotone during this positive sweep.  Its negative-polarity rupture
    is implemented separately in :func:`simulate_1r_reset_ensemble`, so the
    measured SET and RESET instrument envelopes remain distinct.
    """

    p = Params() if p is None else p
    rng = np.random.default_rng(seed)
    wave = v3.triangle(vpeak, step=step)
    icomp = p.Icomp_1r if compliance is None else float(compliance)
    cycles = []
    for _ in range(n_cycles):
        Vcrit = float(np.clip(rng.normal(p.Vhard_mean, p.Vhard_sigma), 2.35, 3.10))
        Rhard = float(p.Rhard_median * np.exp(rng.normal(0.0, p.Rhard_sigma_ln)))
        # A modest log scatter represents pre-existing weak-region variability.
        I0 = float(p.I0_1r * np.exp(rng.normal(0.0, 0.45)))
        I = np.empty_like(wave)
        hard_hist = np.empty_like(wave)
        raw_hist = np.empty_like(wave)
        hard = 0.0
        for idx, Va in enumerate(wave):
            if Va >= 0:
                hard = max(hard, _hard_activation(Va, Vcrit, p.hard_width))
            pre = I0 * np.sinh(np.clip(p.alpha_1r * abs(Va), 0.0, 40.0))
            hard_current = abs(Va) / max(Rhard, 1e-12)
            raw = (1.0 - hard) * pre + hard * hard_current
            I[idx] = np.sign(Va) * min(raw, icomp)
            hard_hist[idx], raw_hist[idx] = hard, raw
        cycles.append(dict(Vs=wave.copy(), Is=I, hard=hard_hist, Iraw=raw_hist,
                           Vhard=Vcrit, Rhard=Rhard, compliance=icomp))
    return cycles


def _hard_survival(V, Vcrit, width):
    """Smooth survival of hard links under RESET stress."""

    return 1.0 - _hard_activation(V, Vcrit, width)


def simulate_1r_reset_ensemble(p=None, n_cycles=22, seed=0, step=0.025,
                               vpeak=2.5, compliance=None):
    """Standalone 1R RESET with stochastic rupture of coarse hard links.

    Each link represents a complete coarse-grained path through the layered
    stack, not one h-BN layer.  Their weights sum to the measured low-voltage
    hard conductance.  Once ruptured on the negative sweep a link stays broken,
    so the return branch follows the defect-assisted HRS transport law.
    """

    p = Params() if p is None else p
    rng = np.random.default_rng(seed)
    wave = v3.triangle(-abs(vpeak), step=step)
    icomp = p.Icomp_1r_reset if compliance is None else float(compliance)
    nlinks = max(int(p.hard_reset_links), 2)
    cycles = []
    for _ in range(n_cycles):
        Rhard = float(p.Rhard_median * np.exp(rng.normal(0.0, p.Rhard_sigma_ln)))
        I0 = float(p.I0_1r * np.exp(rng.normal(0.0, 0.45)))
        onset = float(np.clip(rng.normal(p.hard_reset_onset_mean,
                                         p.hard_reset_onset_sigma), 0.25, 0.95))
        completion = float(np.clip(rng.normal(p.hard_reset_completion_mean,
                                               p.hard_reset_completion_sigma),
                                   onset + 0.25, abs(vpeak) - 0.05))
        # The first, dominant branch sets the macroscopic current maximum.  The
        # remaining weaker links span the later conductance-collapse interval.
        dominant = float(np.clip(rng.normal(p.hard_reset_dominant_weight, 0.06),
                                 0.45, 0.82))
        tail = rng.dirichlet(np.full(nlinks - 1, 0.8)) * (1.0 - dominant)
        weights = np.concatenate([[dominant], tail])
        positions = np.linspace(0.0, 1.0, nlinks) ** 0.78
        thresholds = onset + (completion - onset) * positions
        if nlinks > 2:
            thresholds[1:-1] += rng.normal(0.0, 0.035, nlinks - 2)
            thresholds[1:-1] = np.clip(thresholds[1:-1], onset, completion)
            thresholds = np.maximum.accumulate(thresholds)
        thresholds[-1] = completion

        I = np.empty_like(wave)
        hard_hist = np.empty_like(wave)
        power_hist = np.empty_like(wave)
        link_hist = np.empty((len(wave), nlinks))
        survival = np.ones(nlinks)
        for idx, Va in enumerate(wave):
            if Va <= 0:
                candidate = np.asarray([
                    _hard_survival(Va, vc, p.hard_reset_width)
                    for vc in thresholds
                ])
                survival = np.minimum(survival, candidate)
            hard_fraction = float(np.dot(weights, survival))
            stress = _thickness_equivalent_voltage(Va, p)
            pre = I0 * np.sinh(np.clip(p.alpha_1r * stress, 0.0, 40.0))
            raw = pre + hard_fraction * abs(Va) / max(Rhard, 1e-12)
            terminal = raw if icomp is None else min(raw, icomp)
            I[idx] = np.sign(Va) * terminal
            hard_hist[idx] = hard_fraction
            power_hist[idx] = abs(Va) * terminal
            link_hist[idx] = survival
        cycles.append(dict(
            Vs=wave.copy(), Is=I, hard=hard_hist, links=link_hist,
            link_weights=weights, rupture_thresholds=thresholds,
            terminal_power_W=power_hist, Rhard=Rhard,
            Vreset_onset=onset, Vreset_completion=completion,
            compliance=icomp,
        ))
    return cycles


def regime_summary(p=None):
    """Return the directly constrained v0.4 regime anchors for reporting/UI use."""

    p = Params() if p is None else p
    return {
        "gate_window_V": [p.Vg_low, p.Vg_high],
        "gate_current_ceiling_A": {
            str(p.Vg_low): p.Icc_low,
            str(p.Vg_ref): p.Icc_ref,
            str(p.Vg_high): p.Icc_high,
        },
        "reset_peak_current_A": {
            str(p.Vg_low): p.Ireset_peak_low,
            str(p.Vg_ref): p.Ireset_peak_ref,
            str(p.Vg_high): p.Ireset_peak_high,
        },
        "xtem_geometry": {
            "hbn_thickness_nm": p.hbn_thickness_nm,
            "hbn_layers_approx": p.hbn_layers,
            "mean_layer_pitch_nm": p.hbn_thickness_nm / p.hbn_layers,
            "via_diameter_nm": p.via_diameter_nm,
            "device_area_um2_max": p.device_area_um2,
            "claim_scope": "as-fabricated layered stack; not state-resolved filament evidence",
        },
        "standalone_1r": {
            "forming_threshold_mean_V": p.Vhard_mean,
            "forming_threshold_sigma_V": p.Vhard_sigma,
            "instrument_compliance_A": p.Icomp_1r,
            "hard_path_resistance_median_ohm": p.Rhard_median,
            "reset_instrument_compliance_A": p.Icomp_1r_reset,
            "reset_current_peak_observed_V": p.Vhard_reset_peak_observed,
            "reset_first_link_threshold_mean_V": p.hard_reset_onset_mean,
            "reset_link_count": p.hard_reset_links,
        },
        "interpretation": {
            "phi_k": "coarse-grained local high-transmission connectivity",
            "hard": "coarse hard-path connectivity; chemical species unresolved",
            "xtem": "layer count/thickness/interface constraint; no unique migrating species",
        },
    }
