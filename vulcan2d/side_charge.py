"""One-reservoir equilibrium charge readout of two frozen WKB configurations.

The structural addition energy is a local parameter, not an evolution law.
Each configuration current is averaged explicitly; no mean-barrier shortcut.
"""
from functools import lru_cache
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.constants import Boltzmann, elementary_charge, electron_mass, hbar
from scipy.special import expit

LN10 = np.log(10.)
VT = Boltzmann*300/elementary_charge


@lru_cache(maxsize=8)
def nodes(order):
    return leggauss(order)


def configuration_supply(voltage, height_ev, width_nm, *, temperature_k=300., mass_ratio=.5, order=24):
    """C, d(log10 C)/dh and d(log10 C)/dg, broadcast over V and h.

    Same supply integral as barrier_state.supply_conductance. Moving energy
    partitions are a quadrature device; derivatives integrate the physical
    action derivative and do not differentiate numerical partition locations.
    """
    v, h = np.broadcast_arrays(np.asarray(voltage, float), np.asarray(height_ev, float))
    if np.any(~np.isfinite(v)) or np.any(~np.isfinite(h)) or np.any(v < 0) or np.any(h <= 0):
        raise ValueError('Finite nonnegative voltages and positive heights required')
    if width_nm < 0 or mass_ratio <= 0 or temperature_k <= 0 or order < 4:
        raise ValueError('Valid geometry, mass, temperature and quadrature required')
    shape = v.shape; v, h = v.ravel()[:, None], h.ravel()[:, None]
    vt = Boltzmann*temperature_k/elementary_charge
    lo, hi = -v-60*vt, np.full_like(v, 60*vt)
    knots = np.sort(np.clip(np.concatenate((lo, -v-8*vt, -v, -v+8*vt,
                          np.full_like(v, -8*vt), np.zeros_like(v), np.full_like(v, 8*vt), h-v, h, hi), axis=1), lo, hi), axis=1)
    z, w = nodes(order)
    mid, half = (knots[:, :-1]+knots[:, 1:])/2, np.diff(knots)/2
    energy = (mid[:, :, None]+half[:, :, None]*z).reshape(len(v), -1)
    weight = (half[:, :, None]*w).reshape(len(v), -1)
    ap, bp = np.maximum(h-energy, 0), np.maximum(h-energy-v, 0)
    sa, sb = np.sqrt(ap), np.sqrt(bp)
    root = np.zeros_like(ap); dh = np.zeros_like(ap)
    both = (bp > 0) | (v == 0)
    np.divide(2*(ap+sa*sb+bp)/3, sa+sb, out=root, where=both & (sa+sb > 0))
    np.divide(1, sa+sb, out=dh, where=both & (sa+sb > 0))
    crossing = ~both & (v > 0)
    np.divide(2*ap*sa/3, v, out=root, where=crossing)
    np.divide(sa, v, out=dh, where=crossing)
    coefficient = 2*np.sqrt(2*mass_ratio*electron_mass*elementary_charge)/hbar*1e-9
    prefactor = np.full_like(v, 1/vt)
    np.divide(-np.expm1(-v/vt), v, out=prefactor, where=v > 0)
    integrand = np.exp(-coefficient*width_nm*root)*expit(-energy/vt)*expit((energy+v)/vt)*prefactor*weight
    c = integrand.sum(axis=1)
    if np.any(c <= 0) or np.any(c > 1+1e-10):
        raise RuntimeError('Supply outside passive range')
    grad_h = (-coefficient*width_nm*integrand*dh).sum(axis=1)/c/LN10
    grad_g = (-coefficient*integrand*root).sum(axis=1)/c/LN10
    return tuple(x.reshape(shape) for x in (c, grad_h, grad_g))


def shape_and_gradients(voltage, shared, energies_ev, *, mode='equilibrium', order=24):
    """log10|I/G| and gradients for shared (g, low Phi, Delta Phi, a), energy."""
    v = np.asarray(voltage, float)
    if v.ndim != 1 or np.any(v == 0):
        raise ValueError('A nonzero signed voltage vector is required')
    width, low, delta, lever = np.asarray(shared, float)
    if delta < 0 or abs(lever) > 1+1e-12 or mode not in ('equilibrium', 'frozen'):
        raise ValueError('Ordered barriers, passive lever arm and known mode required')
    energies = np.atleast_1d(energies_ev).astype(float)
    heights = np.array([low, low+delta])[:, None]
    c, ch, cg = configuration_supply(np.abs(v)[None, :], heights, width, order=order)
    a = lever if mode == 'equilibrium' else 0.
    u = (a*v[None, :]-energies[:, None])/VT
    logp, log1p = -np.logaddexp(0, -u), -np.logaddexp(0, u)
    lc = np.log(c)
    mixture = np.logaddexp(logp+lc[0], log1p+lc[1])
    p = expit(u); fraction = np.exp(logp+lc[0]-mixture)
    value = (np.log(np.abs(v))[None, :]+mixture)/LN10
    gate_derivative = (fraction-p)/(VT*LN10)
    gradients = np.stack((fraction*cg[0]+(1-fraction)*cg[1],
                          fraction*ch[0]+(1-fraction)*ch[1],
                          (1-fraction)*ch[1],
                          gate_derivative*v[None, :] if mode == 'equilibrium' else np.zeros_like(value)), axis=-1)
    return value, gradients, -gate_derivative, p


def predict(voltage, shared, states, *, mode='equilibrium', order=24):
    states = np.atleast_2d(states)
    if states.shape[1] != 2:
        raise ValueError('Each state is [log10 supply S, addition energy eV]')
    return states[:, 0, None]+shape_and_gradients(voltage, shared, states[:, 1], mode=mode, order=order)[0]


def stationary_charge(voltage, energy_ev, lever_arm, gamma=1.):
    """Occupation and balanced in/out flux for a single equilibrium reservoir."""
    if gamma <= 0 or abs(lever_arm) > 1:
        raise ValueError('Positive exchange scale and passive lever arm required')
    p = expit((lever_arm*np.asarray(voltage)-energy_ev)/VT)
    return p, gamma*p*(1-p), gamma*(1-p)*p


def symmetric_log_ratio_bound(voltage, lever_arm):
    if abs(lever_arm) > 1:
        raise ValueError('Passive lever arm required')
    return 2*abs(lever_arm)*np.abs(voltage)/VT
