"""Independent four-state channels with a bounded conducting-energy spectrum.

Each channel has its own reciprocal side charge. The exact stationary closure
is integrated over channel energies, not evaluated at their average energy.
Analytic derivatives use positive log sums even near Coulomb blockade.
"""
from functools import lru_cache
import numpy as np
from scipy.special import expit, logsumexp
from .coupled_charge import charging, VT, LN10


@lru_cache(None)
def quadrature(panels=8, order=12):
    if panels < 1 or order < 1:
        raise ValueError('Positive quadrature panel and order counts required')
    x, w = np.polynomial.legendre.leggauss(order)
    centers = -1 + (2*np.arange(panels)+1)/panels
    nodes = (centers[:, None]+x[None, :]/panels).ravel()
    weights = np.tile(w/(2*panels), panels)
    return nodes, weights


def add(a, b):
    """Positive log-sum and its derivative vector."""
    value = np.logaddexp(a[0], b[0])
    grad = np.exp(a[0]-value)[..., None]*a[1] + np.exp(b[0]-value)[..., None]*b[1]
    return value, grad


def multiply(a, b):
    return a[0]+b[0], a[1]+b[1]


def fermi(energy, grad):
    z = energy/VT
    f = expit(-z)
    return ((-np.logaddexp(0., z), (f-1)[..., None]*grad/VT),
            (-np.logaddexp(0., -z), f[..., None]*grad/VT))


def shape_and_gradient(voltage, shared, trap_energies, *, panels=8, order=12):
    """Log-current shape and derivatives: Ed,xd,xt,u,ln GL/R,width,Et.

    Voltage is 1D and energies is a vector of local retained structural states.
    Current is normalized by e*N*Gamma_R. A fitted amplitude supplies that
    unseparated scale. No time evolution of E_t is implied.
    """
    shared = np.asarray(shared, float)
    voltage = np.asarray(voltage, float)
    energy = np.atleast_1d(np.asarray(trap_energies, float))
    if shared.shape != (6,) or np.any(~np.isfinite(shared)) or shared[5] < 0:
        raise ValueError('Six finite shared parameters and nonnegative half-width required')
    if voltage.ndim != 1 or energy.ndim != 1 or np.any(~np.isfinite(voltage)) or np.any(~np.isfinite(energy)):
        raise ValueError('Finite voltage and energy vectors required')
    ed, xd, xt, _, gl, width = shared
    u, du0 = charging(np.r_[shared[:5], 0.])
    du = np.r_[du0[:5], 0., 0.]
    nodes, weights = quadrature(panels, order)
    v = voltage[None, :, None]
    level = ed+(1-xd)*v+width*nodes[None, None, :]
    side = energy[:, None, None]+(1-xt)*v
    dl = np.zeros((*level.shape, 7)); ds = np.zeros((*side.shape, 7))
    dl[..., 0] = 1.; dl[..., 1] = -v; dl[..., 5] = nodes
    ds[..., 2] = -v; ds[..., 6] = 1.
    l0, lc0 = fermi(level-v, dl); r0, rc0 = fermi(level, dl)
    l1, lc1 = fermi(level+u-v, dl+du); r1, rc1 = fermi(level+u, dl+du)
    c0, c0c = fermi(side, ds); c1, c1c = fermi(side+u, ds+du)
    dg = np.zeros(7); dg[4] = 1.
    norm = np.logaddexp(gl, 0.); dn = expit(gl)*dg
    def average(left, right):
        a, da = add((left[0]+gl, left[1]+dg), right)
        return a-norm, da-dn
    a0, a0c = average(l0, r0), average(lc0, rc0)
    a1, a1c = average(l1, r1), average(lc1, rc1)
    first = add(a0c, a1)
    delta_base = average(multiply(l0, lc1), multiply(r0, rc1))
    term = multiply(delta_base, add(c0c, c1))
    shift = -np.expm1(-u/VT)
    with np.errstate(divide='ignore'):
        denominator = np.logaddexp(first[0], term[0]+np.log(shift))
    ddenominator = (np.exp(first[0]-denominator)[..., None]*first[1]
                    + np.exp(term[0]-denominator)[..., None]
                    * (shift*term[1]+np.exp(-u/VT)*du/VT))
    m = add(multiply(c0, a0c), multiply(c1, a0))
    mc = add(multiply(c0c, a1c), multiply(c1c, a1))
    m = m[0]-denominator, m[1]-ddenominator
    mc = mc[0]-denominator, mc[1]-ddenominator
    positive = voltage[None, :, None] >= 0
    def current_pair(l, rc, r, lc):
        return (np.where(positive, l[0]+rc[0], r[0]+lc[0]),
                np.where(positive[..., None], l[1]+rc[1], r[1]+lc[1]))
    current = add(multiply(mc, current_pair(l0, rc0, r0, lc0)),
                  multiply(m, current_pair(l1, rc1, r1, lc1)))
    spectral_log = current[0]+np.log(weights)
    averaged = logsumexp(spectral_log, axis=-1)
    gradient = np.sum(np.exp(spectral_log-averaged[..., None])[..., None]*current[1], axis=-2)
    with np.errstate(divide='ignore'):
        drive = np.log(-np.expm1(-np.abs(voltage)/VT))
    log_current = gl-norm+averaged+drive
    return {'log10_abs_current_scale': log_current/LN10,
            'derivatives': (gradient+dg-dn)/LN10,
            'current_scale': np.sign(voltage)*np.exp(log_current)}


def predict(voltage, shared, states, *, panels=8, order=12):
    states = np.atleast_2d(np.asarray(states, float))
    if states.ndim != 2 or states.shape[1] != 2 or np.any(~np.isfinite(states)):
        raise ValueError('Two finite state coordinates: log10 amplitude A, E_t eV')
    return states[:, 0, None] + shape_and_gradient(
        voltage, shared, states[:, 1], panels=panels, order=order)['log10_abs_current_scale']
