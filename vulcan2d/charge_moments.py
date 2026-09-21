"""Exact mean-charge dynamics for one independent four-state channel.

This is an occupancy propagator, not a terminal transient-current model.
Time is dimensionless Gamma_R*t unless Gamma_R is independently specified.
"""
import numpy as np
from scipy.linalg import expm
from scipy.special import expit
from .coupled_charge import charging, VT
from .coupled_charge_reduced import stationary_current, log_fermi_pair


def coefficients(voltage, shared, trap_energy, log_side_rate=0.):
    shared = np.asarray(shared, float)
    if shared.shape != (5,) or np.any(~np.isfinite(shared)) or not np.isfinite(log_side_rate):
        raise ValueError('Five finite shared parameters and finite log side rate required')
    ed, xd, xt, _, gl = shared
    u, _ = charging(np.r_[shared, 0.])
    v, et = np.broadcast_arrays(np.asarray(voltage, float), np.asarray(trap_energy, float))
    if np.any(~np.isfinite(v)) or np.any(~np.isfinite(et)):
        raise ValueError('Finite voltage and energy required')
    level, side = ed+(1-xd)*v, et+(1-xt)*v
    l0, lc0 = log_fermi_pair((level-v)/VT); r0, rc0 = log_fermi_pair(level/VT)
    l1, _ = log_fermi_pair((level+u-v)/VT); r1, _ = log_fermi_pair((level+u)/VT)
    norm = np.logaddexp(gl, 0.)
    a0 = np.logaddexp(gl+l0, r0)-norm
    a0c = np.logaddexp(gl+lc0, rc0)-norm
    a1 = np.logaddexp(gl+l1, r1)-norm
    c0, c0c = log_fermi_pair(side/VT); c1, c1c = log_fermi_pair((side+u)/VT)
    with np.errstate(divide='ignore'):
        shift = np.log(-np.expm1(-u/VT))
    delta_a = np.logaddexp(gl+l0-np.logaddexp(0., -(level+u-v)/VT),
                          r0-np.logaddexp(0., -(level+u)/VT))+shift-norm
    delta_c = c0+c1c+shift
    logdet = np.logaddexp(np.logaddexp(a0c, a1), delta_a+np.logaddexp(c0c, c1))
    g, gamma = np.exp(norm), np.exp(log_side_rate)
    product = np.exp(delta_a+delta_c)
    fast = .5*(g+gamma+np.hypot(g-gamma, 2*np.sqrt(g*gamma*product)))
    slow = g*gamma*np.exp(logdet)/fast
    return {'F0': np.exp(a0), 'F1': np.exp(a1), 'c0': np.exp(c0), 'c1': np.exp(c1),
            'B': -np.exp(delta_a), 'D': -np.exp(delta_c),
            'closure_determinant': np.exp(logdet),
            'conductor_attempt_rate': g, 'side_attempt_rate': gamma,
            'fast_decay_rate': fast, 'slow_decay_rate': slow,
            'slow_rate_relative_to_uncoupled': slow/min(g, gamma)}


def propagate_means(voltage, dwell, shared, trap_energy, initial, log_side_rate=0.):
    """Piecewise constant voltage; mean charges preserved across every segment.

    Dwell is Gamma_R*delta_t, not seconds or voltage-domain dose. Initial means
    must be explicit. No reset boundary or measured-current reinitialization.
    """
    voltage = np.asarray(voltage, float)
    dwell = np.broadcast_to(np.asarray(dwell, float), voltage.shape)
    state = np.asarray(initial, float).copy()
    if voltage.ndim != 1 or state.shape != (2,) or np.any(~np.isfinite(state)) or np.any((state < 0) | (state > 1)):
        raise ValueError('Voltage vector and two initial mean occupations in [0,1] required')
    if np.any(~np.isfinite(dwell)) or np.any(dwell < 0):
        raise ValueError('Finite nonnegative dimensionless dwell required')
    rows = []
    for v, duration in zip(voltage, dwell):
        c = coefficients(v, shared, trap_energy, log_side_rate)
        g, gamma = c['conductor_attempt_rate'], c['side_attempt_rate']
        matrix = np.array([[-g, g*c['B']], [gamma*c['D'], -gamma]])
        fixed = stationary_current(v, shared, trap_energy)
        equilibrium = np.array([fixed['mean_conductor_occupation'], fixed['mean_side_occupation']])
        state = equilibrium+expm(matrix*duration)@(state-equilibrium)
        rows.append(state.copy())
    return np.array(rows)
