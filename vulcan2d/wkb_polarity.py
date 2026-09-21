"""Asymmetric frozen WKB profiles at matched opposite internal biases.

Uses a common mean chemical potential, energy-independent supply weight and
passive linear voltage division. It does not model resonant interference,
energy-dependent contact supply, charge rearrangement or scan-state changes.
"""
import numpy as np
from scipy.constants import elementary_charge, electron_mass, hbar, Boltzmann
from scipy.special import logsumexp
from numpy.polynomial.legendre import leggauss


def linear_root_average(left, right):
    """Integral_0^1 sqrt(max((1-x)*left+x*right, 0)) dx."""
    a, b = np.broadcast_arrays(np.asarray(left, dtype=float), np.asarray(right, dtype=float))
    ap, bp = np.maximum(a, 0.), np.maximum(b, 0.); sa, sb = np.sqrt(ap), np.sqrt(bp)
    both = (a >= 0) & (b >= 0); result = np.zeros_like(a)
    np.divide((2./3.)*(ap+sa*sb+bp), sa+sb, out=result, where=both & (sa+sb > 0))
    crossing = (a > 0) ^ (b > 0)
    np.divide((2./3.)*(ap*sa+bp*sb), np.abs(a-b), out=result, where=crossing)
    return result


def log_fermi_window(energy_ev, voltage_v, temperature_k=300.):
    v = float(voltage_v)
    if v <= 0 or temperature_k <= 0:
        raise ValueError('Positive matched voltage and temperature required')
    vt = Boltzmann*temperature_k/elementary_charge; e = np.asarray(energy_ev)
    return -np.logaddexp(0., (e-v/2)/vt)-np.logaddexp(0., -(e+v/2)/vt)+np.log(-np.expm1(-v/vt))


def log_transmission(energy_ev, voltage_v, phi_left_ev, phi_right_ev, width_nm, *, sign=1, mass_ratio=.5):
    if sign not in (-1, 1) or width_nm < 0 or mass_ratio <= 0 or voltage_v < 0:
        raise ValueError('Valid sign, width, mass and voltage required')
    energy = np.asarray(energy_ev)
    left = phi_left_ev+sign*voltage_v/2-energy
    right = phi_right_ev-sign*voltage_v/2-energy
    coefficient = 2*np.sqrt(2*mass_ratio*electron_mass*elementary_charge)/hbar*1e-9
    return -coefficient*width_nm*linear_root_average(left, right)


def log_current_integral(voltage_v, phi_left_ev, phi_right_ev, width_nm, *, sign=1, mass_ratio=.5,
                         temperature_k=300., order=48):
    """Natural log of positive current divided by common supply prefactor (V)."""
    v = float(voltage_v); vt = Boltzmann*temperature_k/elementary_charge
    if v <= 0 or temperature_k <= 0 or order < 4 or min(phi_left_ev, phi_right_ev) < 0:
        raise ValueError('Positive voltage/temperature and nonnegative equilibrium endpoint heights required')
    left, right = phi_left_ev+sign*v/2, phi_right_ev-sign*v/2
    lo, hi = -v/2-60*vt, max(v/2, left, right)+60*vt
    knots = np.unique(np.clip([lo, -v/2-8*vt, -v/2, -v/2+8*vt, 0., v/2-8*vt,
                               v/2, v/2+8*vt, left-8*vt, left, left+8*vt,
                               right-8*vt, right, right+8*vt, hi], lo, hi))
    z, w = leggauss(order); mid, half = .5*(knots[:-1]+knots[1:]), .5*np.diff(knots)
    energy = (mid[:, None]+half[:, None]*z).ravel(); weight = (half[:, None]*w).ravel()
    log_integrand = log_transmission(energy, v, phi_left_ev, phi_right_ev, width_nm, sign=sign, mass_ratio=mass_ratio)
    log_integrand += log_fermi_window(energy, v, temperature_k)
    return float(logsumexp(log_integrand+np.log(weight)))


def thermal_log_ratio_bound(voltage_v, temperature_k=300.):
    if temperature_k <= 0 or np.any(np.asarray(voltage_v) < 0):
        raise ValueError('Positive temperature and nonnegative voltage required')
    return np.asarray(voltage_v)*elementary_charge/(Boltzmann*temperature_k)
