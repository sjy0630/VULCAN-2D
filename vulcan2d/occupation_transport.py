"""Research-only stationary occupation and matched-complexity return kernels.

The fast electronic occupation is not a slow structural memory state. Rate
and energy conversions assume a single effective level and symmetric bias.
This module does not modify the production M13 switching model.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit

from .transport_candidates import (Curve, TransportAssumptions, LN10, Q,
    log10_shape as baseline_shape, fit_shared_shape as baseline_fit)

NAMES = ("power", "schottky_free", "occupation", "occupation_background", "rollover",
         "full_bias_level", "full_bias_level_background")


def domain(name, assumptions=TransportAssumptions()):
    """Optimization coordinates; background parameter is log10(gamma)."""
    wmin = 2 * assumptions.thermal_voltage_v
    if wmin >= 1.5:
        raise ValueError("Assumed temperature incompatible with declared width domain")
    if name == "occupation":
        return np.array([0., wmin]), np.array([4., 1.5])
    if name == "occupation_background":
        return np.array([0., wmin, -6.]), np.array([4., 1.5, 1.])
    if name == "rollover":
        return np.array([1., .05, .15]), np.array([5., 1., 4.])
    if name == "full_bias_level":
        return np.array([0., .01]), np.array([1., .99])
    if name == "full_bias_level_background":
        return np.array([0., .01, -6.]), np.array([1., .99, 1.])
    raise ValueError(f"Not a nonlinear occupation comparison kernel: {name}")


def initial_points(name, assumptions=TransportAssumptions()):
    # Fixed before data fitting; no target-dependent warm starts.
    base = [(0.4, .15), (1.0, .3), (1.8, .5), (3.0, .9)]
    if name == "occupation":
        values = base
    elif name == "occupation_background":
        values = [(v, w, g) for (v, w), g in zip(base, (-3., -2., -1., -4.))]
    elif name == "rollover":
        values = [(1.5, .3, .6), (2., .6, 1.2), (3., .9, 2.), (4., .1, 3.)]
    elif name.startswith("full_bias_level"):
        values = [(.03,.15),(.1,.1),(.2,.5),(.5,.85)]
        if name.endswith("background"):
            values = [(*p,g) for p,g in zip(values,(-3.,-2.,-1.,-4.))]
    else:
        raise ValueError(name)
    lo, hi = domain(name, assumptions)
    return [np.clip(np.array(x), lo + 1e-8, hi - 1e-8) for x in values]


def _logcosh(z):
    return np.logaddexp(z, -z) - np.log(2.)


def _log_kernel(v, name, p, assumptions):
    if name.startswith("full_bias_level"):
        vt=assumptions.thermal_voltage_v
        a=(p[0]-p[1]*v)/vt
        b=(p[0]+(1-p[1])*v)/vt
        out=b+np.log(-np.expm1(-v/vt))-np.logaddexp(0.,a)-np.logaddexp(0.,b)
        if name.endswith("background"):
            out=np.logaddexp(out,p[2]*LN10+np.log(v/assumptions.reference_voltage_v))
        return out
    if name.startswith("occupation"):
        z = v / p[1]
        logsinh = z + np.log(-np.expm1(-2*z)) - np.log(2.)
        out = logsinh - np.logaddexp(_logcosh(z), _logcosh(p[0]/p[1]))
        if name == "occupation_background":
            out = np.logaddexp(out, p[2]*LN10 + np.log(v / assumptions.reference_voltage_v))
        return out
    nlo, nhi, vc = p
    return nlo*np.log(v) - .5*(nlo-nhi)*np.logaddexp(0., 2*np.log(v/vc))


def log10_shape(voltage_magnitude, name, parameters, assumptions=TransportAssumptions()):
    v = np.asarray(voltage_magnitude, dtype=float)
    if np.any(~np.isfinite(v)) or np.any(v <= 0):
        raise ValueError("Strictly positive finite voltage required")
    if name in ("power", "schottky_free"):
        return baseline_shape(v, name, float(np.asarray(parameters).reshape(-1)[0]), assumptions)
    p = np.asarray(parameters, dtype=float)
    lo, hi = domain(name, assumptions)
    if p.shape != lo.shape or np.any(~np.isfinite(p)) or np.any(p < lo-1e-12) or np.any(p > hi+1e-12):
        raise ValueError("Parameters outside declared domain")
    return (_log_kernel(v, name, p, assumptions)
            - _log_kernel(assumptions.reference_voltage_v, name, p, assumptions)) / LN10


def current(voltage, log10_reference_current_a, name, parameters,
            assumptions=TransportAssumptions()):
    v = np.asarray(voltage, dtype=float)
    if np.any(~np.isfinite(v)) or not np.isfinite(log10_reference_current_a):
        raise ValueError("Finite voltage and amplitude required")
    result = np.zeros_like(v)
    mask = v != 0
    logs = log10_reference_current_a + log10_shape(np.abs(v[mask]), name, parameters, assumptions)
    if name.startswith("full_bias_level"):
        # Full-bias asymmetric energy alignment must NOT be made artificially
        # odd. Reverse bias swaps the electrostatic lever arms, but the positive
        # reference-current normalization and channel rates remain unchanged.
        negative=v[mask]<0
        p=np.asarray(parameters,dtype=float);reverse=p.copy();reverse[1]=1-p[1]
        logs[negative]=log10_reference_current_a+(
            _log_kernel(np.abs(v[mask][negative]),name,reverse,assumptions)
            -_log_kernel(assumptions.reference_voltage_v,name,p,assumptions))/LN10
    if np.any(logs > np.log10(np.finfo(float).max)):
        raise OverflowError("Current overflows; no clipping")
    result[mask] = np.sign(v[mask])*10.**logs
    return float(result) if result.ndim == 0 else result


def stationary_level(voltage, energy_ev, voltage_coupling, gamma_left_hz,
                     gamma_right_hz, number_levels=1., assumptions=TransportAssumptions()):
    """Explicit rates for a local junction seeing voltage_coupling * voltage.

    If coupling < 1 this is NOT a full two-terminal voltage-drop solution.
    """
    values = (energy_ev, voltage_coupling, gamma_left_hz, gamma_right_hz, number_levels)
    if not all(np.isfinite(x) for x in values) or energy_ev < 0 or not 0 < voltage_coupling <= 1:
        raise ValueError("Invalid energy or voltage coupling")
    if min(gamma_left_hz, gamma_right_hz, number_levels) <= 0:
        raise ValueError("Rates and channel count must be positive")
    v = np.asarray(voltage, dtype=float)
    if np.any(~np.isfinite(v)):
        raise ValueError("Voltage must be finite")
    vt = assumptions.thermal_voltage_v
    fl = expit((voltage_coupling*v/2-energy_ev)/vt)
    fr = expit((-voltage_coupling*v/2-energy_ev)/vt)
    p = (gamma_left_hz*fl+gamma_right_hz*fr)/(gamma_left_hz+gamma_right_hz)
    net = Q*number_levels*gamma_left_hz*gamma_right_hz/(gamma_left_hz+gamma_right_hz)*(fl-fr)
    return p, net


def profile_amplitude(curve, name, parameters, assumptions=TransportAssumptions()):
    return float(np.mean(curve.log10_current_a-log10_shape(curve.voltage_v,name,parameters,assumptions)))


def series_current(applied_voltage, transistor_voltage_grid, transistor_current_grid,
                   log10_reference_current_a, name, parameters,
                   assumptions=TransportAssumptions()):
    """Solve positive terminal I-V with a fixed measured transistor law.

    The solve receives no target current. A monotone branch allows vectorized
    bisection with KVL enforced at every step; it is still a frozen-state branch,
    not an autonomous SET/RESET history model.
    """
    va=np.asarray(applied_voltage,dtype=float)
    vt=np.asarray(transistor_voltage_grid,dtype=float)
    it=np.asarray(transistor_current_grid,dtype=float)
    if vt.ndim!=1 or it.shape!=vt.shape or len(vt)<2:
        raise ValueError("Invalid transistor grid")
    if np.any(~np.isfinite(vt)) or np.any(~np.isfinite(it)) or np.any(np.diff(vt)<=0) or np.any(np.diff(it)<0):
        raise ValueError("Transistor grid must be finite and monotone")
    if vt[0]!=0 or it[0]!=0 or np.any(it<0):
        raise ValueError("Positive transistor law must start at the origin")
    if np.any(~np.isfinite(va)) or np.any(va<0) or np.any(va>vt[-1]):
        raise ValueError("Applied voltage outside transistor domain")
    lo=np.zeros_like(va);hi=va.copy()
    for _ in range(60):
        mid=(lo+hi)/2
        ih=current(mid,log10_reference_current_a,name,parameters,assumptions)
        itr=np.interp(va-mid,vt,it)
        lo=np.where(ih<itr,mid,lo);hi=np.where(ih>=itr,mid,hi)
    vh=(lo+hi)/2
    ih=current(vh,log10_reference_current_a,name,parameters,assumptions)
    return ih,vh


def conditional_physics(name, parameters, assumptions=TransportAssumptions()):
    if name.startswith("full_bias_level"):
        out={"assumed_temperature_K":assumptions.temperature_k,
             "conditional_level_energy_eV":float(parameters[0]),
             "conditional_electrostatic_lever_arm":float(parameters[1]),
             "interpretation":"Full reservoir bias difference; effective single level, not microscopic identification"}
        if name.endswith("background"):out["background_to_saturation_at_reference"]=float(10.**parameters[2])
        return out
    if not name.startswith("occupation"):
        return {"interpretation": "Comparison kernel; not microscopic identification"}
    p = np.asarray(parameters)
    coupling = 2*assumptions.thermal_voltage_v/p[1]
    out = {"assumed_temperature_K": assumptions.temperature_k,
           "conditional_voltage_coupling": float(coupling),
           "conditional_level_energy_eV": float(coupling*p[0]/2),
           "interpretation": "Local junction with assumed constant partial voltage; remaining drop not solved; N and rates not separately identified",
           "equivalent_temperature_if_full_symmetric_bias_K":float(assumptions.temperature_k/coupling)}
    if name == "occupation_background":
        out["background_to_saturation_at_reference"] = float(10.**p[2])
    return out


def fit_shared_shape(curves, name, assumptions=TransportAssumptions()):
    curves = tuple(curves)
    if len(curves) < 2 or len({c.cycle_id for c in curves}) != len(curves):
        raise ValueError("At least two unique training cycles required")
    if name in ("power", "schottky_free"):
        fit = baseline_fit(curves, name, assumptions)
        fit["parameters"] = [fit["parameter"]]
        return fit
    lower, upper = domain(name, assumptions)
    v = np.concatenate([c.voltage_v for c in curves])
    y = np.concatenate([c.log10_current_a for c in curves])
    counts = np.array([len(c.voltage_v) for c in curves])
    groups = np.repeat(np.arange(len(curves)), counts)
    weights = 1/np.sqrt(len(curves)*counts[groups])

    def residual(p):
        r = y-log10_shape(v, name, p, assumptions)
        means = np.bincount(groups, weights=r)/counts
        return (r-means[groups])*weights

    attempts = [least_squares(residual, p, bounds=(lower, upper), x_scale=upper-lower,
                  max_nfev=300, ftol=1e-9, xtol=1e-9, gtol=1e-9)
                for p in initial_points(name, assumptions)]
    converged = [a for a in attempts if a.success and np.isfinite(a.cost)]
    if not converged:
        raise RuntimeError(f"No converged {name} fit")
    best = min(converged, key=lambda a:a.cost)
    p = best.x
    # Coordinates normalized by declared domain width; local diagnostic only.
    singular = np.linalg.svd(best.jac*(upper-lower)[None,:], compute_uv=False)
    condition = float(singular[0]/singular[-1]) if singular[-1] > 0 else None
    amplitudes = [profile_amplitude(c,name,p,assumptions) for c in curves]
    return {"name": name, "parameters": p.tolist(), "parameter_bounds": [lower.tolist(),upper.tolist()],
            "parameter_at_boundary": bool(np.any(np.minimum(p-lower,upper-p)/(upper-lower)<1e-5)),
            "n_training_cycles": len(curves), "training_cycle_ids": [c.cycle_id for c in curves],
            "log10_reference_amplitudes": dict(zip([c.cycle_id for c in curves],amplitudes)),
            "log10_reference_amplitude_center": float(np.median(amplitudes)),
            "profiled_equal_cycle_rmse_decade": float(np.sqrt(2*best.cost)),
            "conditional_physics": conditional_physics(name,p,assumptions),
            "scaled_jacobian_singular_values": singular.tolist(), "scaled_jacobian_condition": condition,
            "multistart": [{"parameters": a.x.tolist(),"rmse_decade": float(np.sqrt(2*a.cost)),
                            "success": bool(a.success),"nfev": int(a.nfev)} for a in attempts]}
