"""VULCAN-2D v0.3 - distributed soft-breakdown dynamic device
model for the h-BN 1T1M memristor. See analysis/MODEL_REVIEW_2026-07-14.md for
the evidence, limits, and calibration audit.

Core ideas (mechanism-neutral at the microscopic geometry level):
  * The h-BN slab is K parallel local-region sub-populations ("patches"); each patch k
    has a soft-breakdown fraction phi_k in [0,1] (0=pristine HRS, 1=soft-broken
    LRS). A patch can coarse-grain an intrinsic interlayer bridge, a metal-assisted
    confined path, or a CAFM hotspot. Conduction is the sum of patch currents;
    w_k is an effective contribution weight (roughly local area times local
    conductivity), and phi_bar = sum_k w_k phi_k.
  * Switching = progressive soft breakdown: patches cross a spread of thresholds
    -> phi_bar(V) is a smooth sigmoid. Device-level progressiveness does not by
    itself rule out nanoscale localized bridges or paths.
  * A series MOSFET (load-line divider) caps the current -> I_cc is set by the
    transistor alone, structurally decoupled from the defect ensemble.

HONESTY (verified by the design + code review). Three kinds of quantities:

  CALIBRATED per-cycle inputs (NOT emergent; finite-N counting self-averages to
  ~1-5% CV so it cannot make these):
    - per-cycle Weibull threshold draw  -> V_set/V_reset spread (CV~25%);
    - per-cycle log-conductance residual sigma_lnG -> R spread WIDTH;
    - per-cycle LRS conductance spread sigma_Gon -> R_LRS CV.
    - I_s/Gon -> resistance levels; icc_noise -> the measured I_cc noise floor;
    - cE/cf/cg -> the observed 53-cycle endurance slopes.
  By CONSTRUCTION (direct consequence of an input assumption, not a prediction):
    - R is approximately LOG-NORMAL because ln(conductance)~Normal is injected;
    - V_set/V_reset are independent because SET and RESET thresholds are drawn
      independently;
    - the transition is progressive because K patches cross distributed thresholds.
  STRUCTURAL CONSEQUENCE (not directly targeted by a calibration knob):
    - the DECOUPLING I_cc CV (~1%) << V_set/R CV (25-46%): the transistor caps the
      apex current largely independently of the random defect ensemble.
"""
from dataclasses import dataclass, asdict
import numpy as np

KB = 8.617333e-5  # eV/K


# ----------------------------------------------------------------------------
@dataclass
class Params:
    # --- transport: effective defect-assisted law (sinh fit, R2_log=0.998) ---
    # A single-temperature I-V sweep cannot uniquely distinguish TAT, hopping,
    # and related field-assisted mechanisms.  "sinh" is the validated transport
    # law; the microscopic label remains provisional until variable-T data exist.
    I_s: float = 2.61e-10      # pristine patch scale [A], ensemble-calibrated
    alpha: float = 6.747       # sinh field coefficient [1/V]  (FIXED)
    I_s_reset_scale: float = 1.60  # polarity asymmetry from RESET-return HRS fit
    alpha_reset: float = 7.666     # RESET-return field coefficient [1/V]
    K: int = 10                # quadrature/coarse sub-populations; not a channel count
    Gon: float = 2.931e3       # full-LRS conductance lift (HRS->LRS swing)
    phi_floor: float = 1.0e-5  # residual breakdown fraction in HRS (must be << 1/Gon)
    Vclip: float = 5.8         # clip |V_h| in sinh(alpha*V_h) to avoid overflow
    p_perc: float = 1.0        # areal-mixing exponent: Theta(phi)=phi**p_perc
                               # 1.0 = parallel-area mixture; >1 is provisional
                               # until area/percolation scaling data are available

    # --- effective series-transistor output law ---
    # SET starts from the direct LRS-return fit.  The former (Isat=0.43 uA,
    # lambda=27.4/V) fit reached 51 uA via an unphysical channel-length factor and
    # amplified Icc variability through the divider.
    Isat_set: float = 24.84e-6
    Vk_set: float = 1.773
    lam_set: float = 0.261
    Isat_reset: float = 50.0e-9
    Vk_reset: float = 0.355
    lam_reset: float = 119.2

    # --- kinetics (per-step effective rate constants Kc = nu0*exp(-Ea/kT0)*dt) ---
    Kc_set: float = 2.0        # SET generation rate constant (sharp per-patch;
    Kc_reset: float = 2.0      #   progressiveness comes from threshold spread)
    beta_set: float = 4.0      # field acceleration in SET gate [1/V]
    beta_reset: float = 5.0    # field acceleration in RESET gate [1/V]
    Ea_set: float = 1.00       # SET activation energy [eV] (thermal accel only)
    Ea_reset: float = 0.92     # RESET activation energy [eV]
    sigma_theta: float = 0.16  # frozen per-patch threshold spread [V] -> progressive width

    # --- latent threshold distribution; calibrated through the full I-V extractor ---
    Vth_set0: float = 1.294    # undamaged SET threshold mean [V]
    Vth_reset0: float = 0.564  # |RESET| threshold mean [V]
    m_set: float = 4.80        # Weibull shape after output-estimator calibration
    m_reset: float = 2.05      # divider compresses the observed RESET CV
    rho_acf: float = 0.20      # AR(1) on per-cycle threshold -> lag-1 ACF

    # --- variability widths: CALIBRATED (sigma_lnG, sigma_Gon) ---
    sigma_lnG: float = 1.075   # HRS-only log-conductance residual -> R_HRS CV
    sigma_Gon: float = 0.355   # LRS-only conductance-lift log-spread -> R_LRS CV
    # --- calibrated measurement-noise proxy (not emergent) ---
    icc_noise: float = 0.00656 # transistor Isat noise floor; calibrated to I_cc CV
    dirichlet_conc: float = 0.6  # effective-weight concentration (few dominant weak links)

    # --- thermal (optional; disabled until variable-T data can identify it) ---
    T0: float = 300.0
    Rth: float = 0.0
    Tmax: float = 1500.0

    # --- endurance damage (R5), calibrated last on the 53-cycle sequence ---
    # D is measured in equivalent full SET/RESET cycles: with kappa_dmg=0.5,
    # a complete 0->1->0 state excursion increments D by approximately one.
    kappa_dmg: float = 0.5
    cE: float = 0.0073         # D -> SET threshold drift [V/equivalent cycle]
    cf: float = 1.43e-5        # D -> residual soft-breakdown fraction rise
    cg: float = 0.0100         # D -> exponential Gon loss [1/equivalent cycle]

    def to_npz(self, path):
        np.savez(path, **{k: v for k, v in asdict(self).items()})

    @classmethod
    def from_npz(cls, path):
        d = dict(np.load(path))
        return cls(**{k: (int(v) if k == "K" else float(v)) for k, v in d.items()})


# ----------------------------------------------------------------------------
# conduction
def _theta(phi, p):
    """Per-patch HRS->LRS weight.

    Linear p_perc=1 is the parallel-area mixture. Exponents above one are only
    phenomenological knee sharpening until area/percolation data are available.
    """
    return phi if p.p_perc == 1.0 else np.power(np.clip(phi, 0, 1), p.p_perc)


def _transport_params(side, p):
    if side == "reset":
        return p.I_s * p.I_s_reset_scale, p.alpha_reset
    return p.I_s, p.alpha


def G_ensemble(phi, lng, w, p, Gon_eff, side="set"):
    """Effective local-region transport prefactor, independent of V_h.

    The normalized weights coarse-grain both projected area and local current
    density; CAFM spot area alone is therefore insufficient to set ``w``.

    HRS trap-landscape fluctuations and LRS soft-breakdown fluctuations are
    distinct random variables.  The old multiplicative form exp(lng)*Gon made
    the HRS residual control the LRS too, inflating Icc variability.  This
    state interpolation keeps both limits explicit and continuous.
    """
    I_s, _ = _transport_params(side, p)
    theta = _theta(phi, p)
    return I_s * np.sum(w * (np.exp(lng) * (1.0 - theta) + Gon_eff * theta))


def i_hbn(V_h, phi, lng, w, p, Gon_eff, side="set"):
    """Area-summed effective defect-assisted current (scalar V_h).

    sinh(alpha*V_h) is the detailed-balance form of a symmetric hopping flux;
    alpha remains an effective potential-drop parameter at one temperature.
    """
    _, alpha = _transport_params(side, p)
    lim = alpha * p.Vclip
    return G_ensemble(phi, lng, w, p, Gon_eff, side) * np.sinh(
        np.clip(alpha * V_h, -lim, lim))


def i_tr(V_tr, side, p, isat_mult):
    if side == "set":
        Isat, Vk, lam = p.Isat_set, p.Vk_set, p.lam_set
    else:
        Isat, Vk, lam = p.Isat_reset, p.Vk_reset, p.lam_reset
    Vt = abs(V_tr)
    return isat_mult * Isat * (1.0 + lam * Vt) * np.tanh(Vt / Vk)


def divider(V_app, G_ens, p, side, isat_mult, nbis=24):
    """Series load-line: find V_h in [0,|V_app|] with G_ens*sinh(a*V_h)=I_tr(|V_app|-V_h).
    Both branches monotone in V_h -> scalar bisection. Returns (I_dev, V_h_signed)."""
    Va = abs(V_app)
    if Va < 1e-9:
        return 0.0, 0.0
    _, a = _transport_params(side, p)
    lim = a * p.Vclip
    if side == "set":
        Isat, Vk, lam = p.Isat_set, p.Vk_set, p.lam_set
    else:
        Isat, Vk, lam = p.Isat_reset, p.Vk_reset, p.lam_reset
    Isat *= isat_mult
    lo, hi = 0.0, Va
    for _ in range(nbis):
        mid = 0.5 * (lo + hi)
        vtr = Va - mid
        ihb = G_ens * np.sinh(min(a * mid, lim))
        itr = Isat * (1.0 + lam * vtr) * np.tanh(vtr / Vk)
        if ihb - itr < 0:
            lo = mid
        else:
            hi = mid
    Vh = 0.5 * (lo + hi)
    Idev = G_ens * np.sinh(min(a * Vh, lim))
    return Idev, np.sign(V_app) * Vh


# ----------------------------------------------------------------------------
# kinetics: mean-field soft breakdown.  The defect state is driven by the
# APPLIED terminal stress V_app (the soft-degradation field stress), while the
# series transistor independently limits the CURRENT — this decouples state
# evolution from the post-breakdown V_h collapse, giving a reproducible LRS.
# Patches cross a SPREAD of thresholds -> phi_bar(V) is a smooth (progressive)
# sigmoid.  Exact exponential-Euler keeps phi in [phi_floor,1] for any rate.
def step_phi(phi, V_app, T, Vth_patch, side, p, phi_min=None):
    phi_min = p.phi_floor if phi_min is None else phi_min
    if side == "set":                                  # generation toward 1
        arr = np.exp((p.Ea_set / KB) * (1.0 / p.T0 - 1.0 / T))
        drive = np.sinh(np.clip(p.beta_set * (V_app - Vth_patch), -39, 39))
        g = p.Kc_set * arr * np.clip(drive, 0, None)
        a, b = g, g
    else:                                              # recovery toward floor
        arr = np.exp((p.Ea_reset / KB) * (1.0 / p.T0 - 1.0 / T))
        drive = np.sinh(np.clip(p.beta_reset * (abs(V_app) - Vth_patch), -39, 39))
        r = p.Kc_reset * arr * np.clip(drive, 0, None)
        a, b = r * phi_min, r
    out = phi.copy()
    nz = b > 1e-12
    e = np.exp(-b[nz])
    out[nz] = phi[nz] * e + (a[nz] / b[nz]) * (1.0 - e)
    return np.clip(out, phi_min, 1.0)


# ----------------------------------------------------------------------------
def triangle(vpeak, step=0.02):
    # up[-2::-1] drops the duplicated apex point (avoids dx=0 in downstream
    # np.gradient and a redundant double-dwell at the peak)
    up = np.arange(0, vpeak + np.sign(vpeak) * 1e-9, np.sign(vpeak) * step)
    return np.concatenate([up, up[-2::-1]])


def run_sweep(Vwave, phi0, lng, w, p, side, isat_mult, Vth_c, dtheta, Gon_eff,
              phi_min=None):
    """Integrate phi over a voltage waveform; return I_dev, V_h, phi_bar arrays."""
    phi = phi0.copy()
    n = len(Vwave)
    I = np.empty(n); Vh = np.empty(n); pb = np.empty(n)
    Vth_patch = Vth_c + dtheta            # per-patch thresholds this cycle
    phi_min = p.phi_floor if phi_min is None else phi_min
    for k, Va in enumerate(Vwave):
        G_ens = G_ensemble(phi, lng, w, p, Gon_eff, side)
        Idev, vh = divider(Va, G_ens, p, side, isat_mult)
        T = p.T0 if p.Rth <= 0 else min(p.T0 + p.Rth * abs(Idev * vh), p.Tmax)
        phi = step_phi(phi, Va, T, Vth_patch, side, p, phi_min)  # reduced-order stress drive
        I[k], Vh[k], pb[k] = Idev, vh, float(np.sum(w * phi))
    return I, Vh, pb, phi


# ----------------------------------------------------------------------------
def make_frozen(p, rng):
    """Frozen device structure (same physical cell): patch weights + thresholds.
    Physical correlation: heavier patches are WEAK LINKS that break FIRST, so
    their threshold offset is lower (anti-correlated with weight). This makes the
    dominant patch always switch -> reproducible LRS -> tame R_LRS spread, while
    sub-dominant patches switching at a spread of thresholds keep the transition
    progressive."""
    w = rng.dirichlet(np.full(p.K, p.dirichlet_conc))
    z = rng.normal(0, 1, p.K)                          # idiosyncratic spread
    wz = (w - w.mean()) / (w.std() + 1e-12)            # standardized weight
    dtheta = p.sigma_theta * (0.7 * (-wz) + 0.7 * z)   # weak links break first
    dtheta -= np.sum(w * dtheta)                       # zero area-weighted mean
    return w, dtheta


def simulate_cycles(p, n_cycles=53, seed=0, device_seed=2026):
    """Sequential 53-cycle Monte-Carlo for ONE cell (C2C-dominant).
    Per cycle, draw C2C: threshold (AR(1) Weibull), ln-G residual, Isat noise.
    The frozen patch structure is keyed by device_seed, while seed controls only
    the cycle sequence. Accumulate endurance damage D across cycles."""
    rng = np.random.default_rng(seed)
    # Consume the legacy frozen-structure draws so seed=2026 preserves the
    # calibrated C2C sequence, then use an independent fixed device structure.
    make_frozen(p, rng)
    w, dtheta = make_frozen(p, np.random.default_rng(device_seed))
    D = 0.0
    prev_set = p.Vth_set0
    prev_rst = p.Vth_reset0
    cycles = []
    # Weibull scale eta so that mean = Vth0: mean = eta*Gamma(1+1/m)
    from math import gamma
    eta_set = p.Vth_set0 / gamma(1 + 1 / p.m_set)
    eta_rst = p.Vth_reset0 / gamma(1 + 1 / p.m_reset)
    phi = np.full(p.K, p.phi_floor)
    for c in range(n_cycles):
        # --- per-cycle C2C draws ---
        wbl_s = eta_set * rng.weibull(p.m_set)
        wbl_r = eta_rst * rng.weibull(p.m_reset)
        # AR(1) for lag-1 ACF around the (damage-drifted) mean
        mean_set = p.Vth_set0 + p.cE * D
        Vth_set_c = mean_set + p.rho_acf * (prev_set - mean_set) + \
            np.sqrt(1 - p.rho_acf ** 2) * (wbl_s - p.Vth_set0)
        Vth_rst_c = p.Vth_reset0 + p.rho_acf * (prev_rst - p.Vth_reset0) + \
            np.sqrt(1 - p.rho_acf ** 2) * (wbl_r - p.Vth_reset0)
        prev_set, prev_rst = Vth_set_c, Vth_rst_c
        lng = rng.normal(0, p.sigma_lnG, p.K)         # C2C conductance residual
        isat_mult = rng.normal(1.0, p.icc_noise)      # I_cc 1% noise
        floor_eff = min(p.phi_floor + p.cf * D, 0.5)
        # per-cycle LRS conductance-lift spread (breakdown-configuration C2C);
        # acts only in LRS (phi>0), leaves the HRS read at phi=floor untouched
        Gon_eff = max(p.Gon * np.exp(rng.normal(0, p.sigma_Gon) - p.cg * D), 1.0)
        pp = p
        # reset phi to pristine floor at the start of each SET (cell is in HRS)
        phi = np.full(p.K, floor_eff)
        # --- SET sweep 0->+5->0 ---
        Vs = triangle(5.0)
        Is, Vhs, pbs, phi = run_sweep(Vs, phi, lng, w, pp, "set", isat_mult,
                                      Vth_set_c, dtheta, Gon_eff, floor_eff)
        dphi_set = abs(pbs[len(Vs) // 2] - pbs[0])
        # --- RESET sweep 0->-1.7->0 ---
        Vr = triangle(-1.7)
        Ir, Vhr, pbr, phi = run_sweep(Vr, phi, lng, w, pp, "reset", isat_mult,
                                      Vth_rst_c, dtheta, Gon_eff, floor_eff)
        dphi_rst = abs(pbr[len(Vr) // 2] - pbr[0])
        # endurance damage
        D += p.kappa_dmg * (dphi_set + dphi_rst)
        cycles.append(dict(Vs=Vs, Is=Is, pbs=pbs, Vr=Vr, Ir=Ir, pbr=pbr,
                           Vth_set_c=Vth_set_c, Vth_rst_c=Vth_rst_c, D=D))
    return cycles, dict(w=w, dtheta=dtheta)
