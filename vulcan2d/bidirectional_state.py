"""Energy-related two-state rates coupled to a measured series transistor.

Research negative-sweep candidates, not a unified SET/RESET model. Time is
applied-voltage path length x (V), so rates have units 1/V. An absolute time
interpretation needs a known constant sweep rate and waits. The scalar state
is a conducting-configuration fraction, not an identified atomic species.

Parameters: log10(A_H/A), beta/(sqrt(V)), log10(A_C/A), alpha/(V),
log(peak rate / (1/V)), delta0/eV, effective charge/e, saddle fraction theta,
common saddle lowering lambda/(eV/V^2). DeltaG(v)=delta0-charge*v for
v=|negative internal voltage|. Lambda=0 is the single tilted-barrier control.
"""
from pathlib import Path
import ctypes
import hashlib
import shutil
import subprocess
import sys
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import expit
from .transport_candidates import TransportAssumptions

THERMAL_VOLTAGE = TransportAssumptions().thermal_voltage_v
NAMES = ('single_hop', 'polarizable_saddle')
_CORE = None


def parameter_bounds(name):
    if name not in NAMES:
        raise ValueError(name)
    lower = [-14., 0., -12., .05, -12., -1., -1., 0.]
    upper = [-4., 15., -3., 20., 18., 1., 1., 1.]
    if name == 'polarizable_saddle':
        lower += [0.]; upper += [.5]
    return np.array(lower), np.array(upper)


def complete_parameters(parameters, name):
    p = np.asarray(parameters, dtype=float)
    expected = 8 if name == 'single_hop' else 9 if name == 'polarizable_saddle' else -1
    if p.shape != (expected,) or not np.all(np.isfinite(p)):
        raise ValueError('Finite parameter vector required for the named candidate')
    if p[1] < 0 or p[3] <= 0 or not 0 <= p[7] <= 1 or (len(p) == 9 and p[8] < 0):
        raise ValueError('Invalid transport or saddle parameters')
    return np.r_[p, 0.] if len(p) == 8 else p.copy()


def rate_reference(p, max_internal_voltage=1.7, thermal_voltage=THERMAL_VOLTAGE):
    """Normalize the maximum individual rate on [0,vmax] to exp(p[4]).

    Each log rate is convex in v for lambda>=0, hence its maximum is at an
    endpoint. This is only a stable parametrization of the same energy law.
    The fitting cap exp(18)/V is declared as a numerical/parameter prior;
    boundary hits must be profiled, not interpreted as measured rates.
    """
    if max_internal_voltage <= 0 or thermal_voltage <= 0:
        raise ValueError('Positive voltage domain and temperature required')
    v = np.array([0., max_internal_voltage])
    offsets = np.r_[(-p[5]/2 + p[7]*p[6]*v + p[8]*v*v)/thermal_voltage,
                    (p[5]/2 - (1-p[7])*p[6]*v + p[8]*v*v)/thermal_voltage]
    return float(p[4] - offsets.max())


def log_rates(v, parameters, name, max_internal_voltage=1.7, thermal_voltage=THERMAL_VOLTAGE):
    p = complete_parameters(parameters, name); v = np.asarray(v, dtype=float)
    if np.any(~np.isfinite(v)) or np.any(v < 0) or np.any(v > max_internal_voltage):
        raise ValueError('Internal voltage outside declared rate domain')
    ell = rate_reference(p, max_internal_voltage, thermal_voltage)
    build = ell + (-p[5]/2 + p[7]*p[6]*v + p[8]*v*v)/thermal_voltage
    remove = ell + (p[5]/2 - (1-p[7])*p[6]*v + p[8]*v*v)/thermal_voltage
    return build, remove


def exact_fixed_voltage_step(state, duration_path, log_build, log_remove):
    s, dx = np.asarray(state), np.asarray(duration_path)
    if np.any((s < 0) | (s > 1)) or np.any(dx < 0):
        raise ValueError('State in [0,1] and nonnegative path required')
    total = np.logaddexp(log_build, log_remove)
    equilibrium = expit(np.asarray(log_build)-np.asarray(log_remove))
    with np.errstate(divide='ignore'):
        log_z = total + np.log(dx)
    # exp(-z) is numerically zero for z>745; this evaluates that exact limit.
    z = np.exp(np.minimum(log_z, np.log(745.)))
    return s*np.exp(-z) + equilibrium*(-np.expm1(-z))


def _core():
    global _CORE
    if _CORE is not None:
        return _CORE
    source = Path(__file__).with_name('_bidirectional_core.cpp')
    key = hashlib.sha256(source.read_bytes()).hexdigest()[:20]
    folder = source.parents[1] / '.venv/.cache/bidirectional'
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / ('core_'+key+('.dylib' if sys.platform == 'darwin' else '.so'))
    if not output.exists():
        compiler = shutil.which('c++')
        if compiler is None:
            raise RuntimeError('A C++ compiler is required for the numerical helper')
        flags = ['-dynamiclib'] if sys.platform == 'darwin' else ['-shared', '-fPIC']
        subprocess.run([compiler, '-O3', '-std=c++11', *flags, str(source), '-o', str(output)], check=True, capture_output=True)
    lib = ctypes.CDLL(str(output)); pointer = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
    lib.bd_load.argtypes = [ctypes.c_double, ctypes.c_double, pointer, pointer, pointer, ctypes.c_int, pointer]
    lib.bd_load.restype = None
    lib.bd_rhs.argtypes = [ctypes.c_double, ctypes.c_double, pointer, ctypes.c_double, ctypes.c_double,
                           pointer, pointer, ctypes.c_int]
    lib.bd_rhs.restype = ctypes.c_double
    _CORE = lib
    return lib


def trajectory(vapp, parameters, name, transistor_voltage, transistor_current, *,
               max_internal_voltage=1.7, initial_log_state=0., rtol=2e-6, atol=2e-8, method='LSODA',
               max_rhs_calls=30000):
    """Forward full-history prediction; accepts no observed memristor current.

    The state is integrated as log(s), preserving very small conducting
    fractions. The ODE solver controls that log-state error, rather than an
    absolute population error that can hide large current errors.
    """
    va = np.asarray(vapp, dtype=float); p = np.ascontiguousarray(complete_parameters(parameters, name))
    tv, ti = [np.ascontiguousarray(a, dtype=float) for a in (transistor_voltage, transistor_current)]
    if va.ndim != 1 or len(va) < 2 or np.any(~np.isfinite(va)) or np.any(va < 0):
        raise ValueError('Finite one-dimensional nonnegative applied voltages required')
    if np.max(va) > max_internal_voltage or not np.isfinite(initial_log_state) or initial_log_state > 0:
        raise ValueError('Applied-voltage domain or initial state invalid')
    if tv.ndim != 1 or ti.shape != tv.shape or len(tv) < 2 or tv[0] != 0 or ti[0] != 0:
        raise ValueError('Lookup arrays starting at the physical origin required')
    if np.any(np.diff(tv) <= 0) or np.any(np.diff(ti) < 0) or np.max(va) > tv[-1]:
        raise ValueError('Monotone lookup covering the applied domain required')
    if not np.all(np.isfinite(tv)) or not np.all(np.isfinite(ti)):
        raise ValueError('Finite transistor data required')
    ell = rate_reference(p, max_internal_voltage); lib = _core()
    # Repeated voltages carry no specified dwell duration in these records.
    path = np.r_[0., np.cumsum(np.abs(np.diff(va)))]
    unique, indices, inverse = np.unique(path, return_index=True, return_inverse=True)
    drive = va[indices]
    if len(unique) == 1:
        state = np.full(len(va), initial_log_state); success = True; nfev = 0; message = 'Zero path'
    else:
        calls = 0
        def rhs(x, logstate):
            nonlocal calls
            calls += 1
            if calls > max_rhs_calls:
                raise RuntimeError('State integration exceeded the declared evaluation budget')
            voltage = float(np.interp(x, unique, drive))
            return [lib.bd_rhs(voltage, float(logstate[0]), p, THERMAL_VOLTAGE, ell, tv, ti, len(tv))]
        result = solve_ivp(rhs, (0., float(unique[-1])), [initial_log_state], method=method,
                           t_eval=unique, rtol=rtol, atol=atol,
                           max_step=float(np.max(np.diff(unique))))
        success, nfev, message = bool(result.success), result.nfev, result.message
        if not success or result.y.shape[1] != len(unique):
            raise RuntimeError('State integration failed: '+message)
        state = result.y[0, inverse]
        if np.any(~np.isfinite(state)) or np.max(state) > 10*atol:
            raise RuntimeError('Invalid integrated state')
    internal, current = np.empty(len(va)), np.empty(len(va)); out = np.empty(2)
    for j, (voltage, lp) in enumerate(zip(va, state)):
        lib.bd_load(float(voltage), float(lp), p, tv, ti, len(tv), out)
        internal[j], current[j] = out
    it = np.interp(va-internal, tv, ti)
    kcl = np.abs(current-it)/np.maximum(np.maximum(current, it), 1e-20)
    return {'current_a': current, 'internal_voltage_v': internal, 'log_state': state,
            'max_kcl_relative_error': float(kcl.max()), 'converged': success, 'nfev': nfev,
            'rate_log_geometric_zero_per_V': ell, 'method': method, 'rtol': rtol, 'atol': atol}


def conditional_barriers(v, parameters, name, *, attempt_per_path=1e13, max_internal_voltage=1.7):
    """Conditional activation energies; attempt_per_path=nu/sweep_rate, 1/V.

    This does not measure a barrier or assume a known experimental scan speed.
    Negative values are reported, never separately clipped to zero.
    """
    if not np.isfinite(attempt_per_path) or attempt_per_path <= 0:
        raise ValueError('Positive attempt rate per voltage path required')
    build, remove = log_rates(v, parameters, name, max_internal_voltage)
    return THERMAL_VOLTAGE*(np.log(attempt_per_path)-build), THERMAL_VOLTAGE*(np.log(attempt_per_path)-remove)
