"""Finite-temperature supply through a trapezoidal WKB barrier.

This research constitutive law treats barrier width as a frozen state within
a specified low-bias branch. It supplies neither a width evolution law nor a
microscopic species assignment. Widths are conditional on effective mass.
"""
import numpy as np
from scipy.constants import elementary_charge, electron_mass, hbar, Boltzmann
from scipy.special import expit
from numpy.polynomial.legendre import leggauss

LN10 = np.log(10.)


def mean_barrier_root(energy_ev, voltage_v, height_ev):
    """Integral_0^1 sqrt(max(height-E-v*x, 0)) dx, in sqrt(eV)."""
    energy, voltage = np.broadcast_arrays(energy_ev, voltage_v)
    if np.any(voltage < 0) or height_ev <= 0:
        raise ValueError('Positive barrier height and nonnegative voltage required')
    a = np.maximum(height_ev-energy, 0.); b = np.maximum(height_ev-energy-voltage, 0.)
    sa, sb = np.sqrt(a), np.sqrt(b)
    same_side = (b > 0) | (voltage == 0)
    ans = np.zeros_like(a, dtype=float)
    np.divide((2./3.)*(a+sa*sb+b), sa+sb, out=ans, where=same_side & (sa+sb > 0))
    crossing = ~same_side & (voltage > 0)
    np.divide((2./3.)*a*sa, voltage, out=ans, where=crossing)
    return ans


def transmission(energy_ev, voltage_v, width_nm, height_ev, mass_ratio=.5):
    if mass_ratio <= 0 or np.any(np.asarray(width_nm) < 0):
        raise ValueError('Positive mass and nonnegative width required')
    coefficient = 2*np.sqrt(2*mass_ratio*electron_mass*elementary_charge)/hbar*1e-9
    return np.exp(-coefficient*np.asarray(width_nm)*mean_barrier_root(energy_ev, voltage_v, height_ev))


def supply_conductance(voltage_v, widths_nm, height_ev, *, mass_ratio=.5, temperature_k=300., order=24):
    """I/(G_prefactor*v), including the analytic v=0 conductance limit.

    Energy quadrature is split at chemical potentials, thermal windows and
    barrier turning points. Finite-temperature over-barrier supply is kept.
    The energy tail is truncated at 60 kBT outside the transport window.
    """
    v = float(voltage_v); width = np.asarray(widths_nm, dtype=float)
    if v < 0 or temperature_k <= 0 or order < 4:
        raise ValueError('Valid nonnegative voltage, temperature and quadrature order required')
    vt = Boltzmann*temperature_k/elementary_charge
    lo, hi = -v-60*vt, 60*vt
    knots = np.unique(np.clip([lo, -v-8*vt, -v, -v+8*vt, -8*vt, 0., 8*vt,
                               height_ev-v, height_ev, hi], lo, hi))
    z, w = leggauss(order)
    mid, half = .5*(knots[1:]+knots[:-1]), .5*np.diff(knots)
    energy = (mid[:, None]+half[:, None]*z).ravel(); weight = (half[:, None]*w).ravel()
    supply = expit(-energy/vt)*expit((energy+v)/vt)
    supply *= -np.expm1(-v/vt)/v if v > 0 else 1./vt
    t = transmission(energy[None, :], v, width.ravel()[:, None], height_ev, mass_ratio)
    result = (t @ (weight*supply)).reshape(width.shape)
    if np.any(result <= 0) or np.any(result > 1.+1e-11):
        raise RuntimeError('Invalid integrated transmission')
    return result


class BarrierTable:
    def __init__(self, height_ev, *, mass_ratio=.5, temperature_k=300., max_voltage=.32,
                 max_width_nm=6., voltage_step=.0025, width_step_nm=.025, order=24):
        self.height_ev = float(height_ev); self.mass_ratio = float(mass_ratio)
        self.voltage = np.linspace(0., max_voltage, int(np.ceil(max_voltage/voltage_step))+1)
        self.width = np.linspace(0., max_width_nm, int(np.ceil(max_width_nm/width_step_nm))+1)
        self.log_conductance = np.array([np.log10(supply_conductance(v, self.width, height_ev,
                                             mass_ratio=mass_ratio, temperature_k=temperature_k, order=order))
                                        for v in self.voltage])
        self.order = int(order); self.temperature_k = float(temperature_k)

    def interpolate(self, voltage, width):
        v, g = np.broadcast_arrays(voltage, width)
        if np.any(~np.isfinite(v)) or np.any(~np.isfinite(g)) or np.any(v < 0) or np.any(v > self.voltage[-1]+1e-12):
            raise ValueError('Finite voltage inside table domain required')
        if np.any(g < 0) or np.any(g > self.width[-1]+1e-12):
            raise ValueError('Width inside table domain required')
        dv, dg = self.voltage[1], self.width[1]
        iv = np.clip((v/dv).astype(int), 0, len(self.voltage)-2)
        ig = np.clip((g/dg).astype(int), 0, len(self.width)-2)
        fv, fg = (v-self.voltage[iv])/dv, (g-self.width[ig])/dg
        a, b = self.log_conductance[iv, ig], self.log_conductance[iv+1, ig]
        c, d = self.log_conductance[iv, ig+1], self.log_conductance[iv+1, ig+1]
        value = (1-fv)*((1-fg)*a+fg*c)+fv*((1-fg)*b+fg*d)
        deriv_v = ((1-fg)*(b-a)+fg*(d-c))/dv
        deriv_g = ((1-fv)*(c-a)+fv*(d-b))/dg
        return value, deriv_v, deriv_g

    def internal(self, voltage, log10_prefactor_S, width_nm):
        v = np.asarray(voltage, dtype=float)
        logc, dlogc_dv, dlogc_dg = self.interpolate(v, width_nm)
        conductance = 10**(log10_prefactor_S+logc)
        current = v*conductance
        derivative = conductance*(1+LN10*v*dlogc_dv)
        if np.any(derivative <= 0):
            raise RuntimeError('Nonmonotone internal transport interpolation')
        return current, derivative, np.ones_like(current), dlogc_dg


def effective_internal(voltage, shared, local, candidate):
    """Amplitude/shape controls; local is log10 current at reference 0.2 V."""
    v, amplitude = np.broadcast_arrays(voltage, local)
    positive = v > 0; safe = np.where(positive, v, .2)
    if candidate == 'power':
        if shared <= 0:
            raise ValueError('Positive power required')
        shape = shared*np.log10(safe/.2)
        current = 10**(amplitude+shape); derivative = current*shared/safe
        dg = np.log10(safe/.2)
    elif candidate == 'pf_like':
        if shared < 0:
            raise ValueError('Nonnegative effective PF-like curvature required')
        shape = np.log10(safe/.2)+shared*(np.sqrt(safe)-np.sqrt(.2))/LN10
        current = 10**(amplitude+shape)
        derivative = current*(1/safe+shared/(2*np.sqrt(safe)))
        dg = (np.sqrt(safe)-np.sqrt(.2))/LN10
    else:
        raise ValueError(candidate)
    return np.where(positive, current, 0.), derivative, dg, np.ones_like(current)


def forward_circuit(vapp, internal_function, transistor_voltage, transistor_current, iterations=52):
    """Solve KCL in transistor voltage and differentiate the implicit solution.

    internal_function(vh) returns I, dI/dVh and log-current derivatives for one
    shared and one local parameter. No observed memristor current is accepted.
    """
    va = np.asarray(vapp, dtype=float); tv, ti = np.asarray(transistor_voltage), np.asarray(transistor_current)
    if np.any(~np.isfinite(va)) or np.any(va < 0) or np.any(va > tv[-1]) or iterations < 1:
        raise ValueError('Valid applied voltage and iterations required')
    if tv[0] != 0 or ti[0] != 0 or np.any(np.diff(tv) <= 0) or np.any(np.diff(ti) < 0):
        raise ValueError('Monotone transistor lookup from origin required')
    before, _, _, _ = internal_function(va)
    low = np.zeros_like(va); high = np.minimum(va, np.interp(before, ti, tv))
    middle = .5*(low+high)
    for _ in range(iterations):
        ih, slope, _, _ = internal_function(va-middle)
        it = np.interp(middle, tv, ti)
        converged = np.abs(ih-it) <= 2e-11*np.maximum(np.maximum(ih, it), 1e-300)
        if np.all(converged):
            break
        move = ih > it
        low = np.where(move, middle, low); high = np.where(move, high, middle)
        ix = np.clip(np.searchsorted(tv, middle, side='right'), 1, len(tv)-1)
        tslope = (ti[ix]-ti[ix-1])/(tv[ix]-tv[ix-1])
        newton = middle+(ih-it)/(slope+tslope)
        trial = np.where((newton > low) & (newton < high), newton, .5*(low+high))
        middle = np.where(converged, middle, trial)
    vd = middle; vh = va-vd
    current, slope, global_derivative, local_derivative = internal_function(vh)
    idx = np.clip(np.searchsorted(tv, vd, side='right'), 1, len(tv)-1)
    transistor_slope = (ti[idx]-ti[idx-1])/(tv[idx]-tv[idx-1])
    fraction = transistor_slope/(slope+transistor_slope)
    reference = np.interp(vd, tv, ti)
    relative = np.abs(current-reference)/np.maximum(np.maximum(current, reference), 1e-300)
    return {'current_a':current, 'internal_voltage_v':vh, 'transistor_voltage_v':vd,
            'dlog10_current_dshared':fraction*global_derivative,
            'dlog10_current_dlocal':fraction*local_derivative,
            'max_kcl_relative_error':float(relative.max())}
