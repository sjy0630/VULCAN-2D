"""Slow joint side-charge/configuration dynamics; only conductor is adiabatic.

State order is (q,m)=00,01,10,11. The effective generator averages microscopic
rates over the conditional conductor occupation and preserves equilibrium
partition functions. Time remains applied-voltage path (rates per volt).
"""
from pathlib import Path
import ctypes
import hashlib
import shutil
import subprocess
import sys
import numpy as np
from scipy.special import logsumexp
from .coupled_charge import charging, VT, LN10
from .coupled_charge_reduced import log_fermi_pair
from .charge_spectrum import quadrature
from .activated_configuration import LO as Q_LO, HI as Q_HI

LO = np.r_[Q_LO, -40.]
HI = np.r_[Q_HI, 20.]
EDGES = ((0, 1), (1, 0), (2, 3), (3, 2), (0, 2), (2, 0), (1, 3), (3, 1))
_CORE = None


def surfaces(voltage, shared, parameters, *, panels=8, order=8):
    """Conditional currents and log rates, axes state/edge, voltage, node."""
    s, p = np.asarray(shared, float), np.asarray(parameters, float)
    v = np.asarray(voltage, float)
    if s.shape != (6,) or np.any(~np.isfinite(s)) or s[5] < 0:
        raise ValueError('Six finite electronic baseline parameters required')
    if p.shape != (9,) or np.any(~np.isfinite(p)) or p[1] < 0 or p[7] < 0:
        raise ValueError('Nine finite parameters and nonnegative shift/saddle polarizability required')
    if v.ndim != 1 or np.any(~np.isfinite(v)):
        raise ValueError('Finite signed voltage vector required')
    nodes, weights = quadrature(panels, order); n = len(nodes)
    u = charging(np.r_[s[:5], 0.])[0]; norm = np.logaddexp(s[4], 0.)
    q = np.array([0, 0, 1, 1])[:, None, None]; m = np.array([0, 1, 0, 1])[:, None, None]
    x = v[None, :, None]
    ed = s[0]+s[5]*nodes[None, None, :]+(1-q)*p[1]
    energy = ed+(1-s[1])*x+u*m
    lf, lc = log_fermi_pair((energy-x)/VT); rf, rc = log_fermi_pair(energy/VT)
    occupied = np.logaddexp(s[4]+lf, rf)-norm
    empty = np.logaddexp(s[4]+lc, rc)-norm
    with np.errstate(divide='ignore'):
        drive = np.log(-np.expm1(-np.abs(x)/VT))
    pair = np.where(x >= 0, lf+rc, rf+lc)
    current = np.exp(p[0]*LN10+s[4]-norm+pair+drive)
    rates = []
    for config in (0, 1):
        side = p[2+config]+(1-s[2])*v[:, None]
        c0, cc0 = log_fermi_pair(side/VT); c1, cc1 = log_fermi_pair((side+u)/VT)
        j = 2*config
        rates.append(p[8]+np.logaddexp(empty[j]+c0, occupied[j]+c1))
        rates.append(p[8]+np.logaddexp(empty[j+1]+cc0, occupied[j+1]+cc1))
    base = p[4]+p[7]*v[:, None]**2/VT
    for side_state in (0, 1):
        delta0 = p[5]-p[6]*v[:, None]+side_state*(p[3]-p[2])
        delta1 = delta0-p[1]
        j0, j1 = side_state, 2+side_state
        rates.append(base+np.logaddexp(empty[j0]-delta0/(2*VT), occupied[j0]-delta1/(2*VT)))
        rates.append(base+np.logaddexp(empty[j1]+delta0/(2*VT), occupied[j1]+delta1/(2*VT)))
    return {'current': current, 'log_rates': np.array(rates), 'n_log_probability': np.array([empty, occupied]),
            'nodes': nodes, 'weights': weights, 'voltage': v}


def generator(voltage, shared, parameters, *, panels=1, order=1):
    """Independent Python generator representation, columns are source states."""
    s = surfaces(np.atleast_1d(voltage), shared, parameters, panels=panels, order=order)
    q = np.zeros((len(s['voltage']), len(s['nodes']), 4, 4))
    for (a, b), rate in zip(EDGES, np.exp(s['log_rates'])):
        q[:, :, b, a] += rate; q[:, :, a, a] -= rate
    return q


def initial_distribution(shared, parameters, *, panels=8, order=8, initial_m=None):
    """q=0 and its zero-bias equilibrium m marginal (or declared sensitivity)."""
    s, p = np.asarray(shared), np.asarray(parameters)
    z, w = quadrature(panels, order); ed = s[0]+s[5]*z+p[1]
    u = charging(np.r_[s[:5], 0.])[0]
    logz0 = np.logaddexp(0., -ed/VT)
    logz1 = -p[2]/VT+np.logaddexp(0., -(ed+u)/VT)
    den = np.logaddexp(logz0, logz1)
    result = np.zeros((4, len(z)))
    result[0] = np.exp(logz0-den); result[1] = np.exp(logz1-den)
    if initial_m is not None:
        if not np.isfinite(initial_m) or not 0 <= initial_m <= 1: raise ValueError('Initial m probability must be in [0,1]')
        result[0] = 1-initial_m; result[1] = initial_m
    return result


def _core():
    global _CORE
    if _CORE is not None: return _CORE
    source = Path(__file__).with_name('_retained_charge_configuration_core.cpp')
    dependency = source.with_name('_activated_configuration_core.cpp')
    key = hashlib.sha256(source.read_bytes()+dependency.read_bytes()).hexdigest()[:20]
    folder = source.parents[1]/'.venv/.cache/retained_charge_configuration'; folder.mkdir(parents=True, exist_ok=True)
    output = folder/('core_'+key+('.dylib' if sys.platform == 'darwin' else '.so'))
    if not output.exists():
        compiler = shutil.which('c++')
        if not compiler: raise RuntimeError('C++ compiler required')
        flags = ['-dynamiclib'] if sys.platform == 'darwin' else ['-shared', '-fPIC']
        subprocess.run([compiler, '-O3', '-std=c++11', *flags, str(source), '-o', str(output)], check=True, capture_output=True)
    lib = ctypes.CDLL(str(output)); a = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    i = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
    lib.rc_run.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, a, a, a, a, a, a,
                          ctypes.c_int, a, i, i, ctypes.c_int, a, a, ctypes.c_int, a, a, a,
                          a, a, a, a, a, a, ctypes.c_int]
    lib.rc_run.restype = ctypes.c_int
    lib.rc_propagator.argtypes = [a, ctypes.c_double, a]; lib.rc_propagator.restype = ctypes.c_int
    _CORE = lib; return lib


def propagator(log_rates, duration):
    """Native positive four-state exponential, exposed for independent tests."""
    rate = np.ascontiguousarray(log_rates, dtype=float)
    if rate.shape != (8,) or np.any(~np.isfinite(rate)) or not np.isfinite(duration) or duration < 0:
        raise ValueError('Eight finite rates and finite nonnegative duration required')
    out = np.empty((4, 4)); code = _core().rc_propagator(rate, duration, out)
    if code: raise RuntimeError(f'Markov exponential failed: {code}')
    return out


def trajectory(applied, shared, parameters, positive_lookup, negative_lookup, *,
               table_step=.005, max_step=.01, panels=8, order=8, initial_m=None, diagnostics=False):
    """Continuous native sequence; propagator tables are convex interpolants.

    Fixed-voltage Markov exponentials are cached for the substep durations.
    This preserves positivity at arbitrary rate stiffness. Voltage-table
    refinement remains required because exponentials, not log rates, interpolate.
    """
    va = np.ascontiguousarray(applied, dtype=float)
    if va.ndim != 1 or len(va) < 2 or va[0] != 0 or np.any(~np.isfinite(va)):
        raise ValueError('Finite voltage sequence beginning at zero required')
    if not np.all(np.isfinite([table_step, max_step])) or min(table_step, max_step) <= 0:
        raise ValueError('Positive finite voltage table and path steps required')
    lookup = []
    for tv, ti in (positive_lookup, negative_lookup):
        tv, ti = np.ascontiguousarray(tv, dtype=float), np.ascontiguousarray(ti, dtype=float)
        if (tv.ndim != 1 or len(tv) < 2 or ti.shape != tv.shape or np.any(~np.isfinite(tv)) or
                np.any(~np.isfinite(ti)) or tv[0] != 0 or ti[0] != 0 or np.any(np.diff(tv) <= 0) or np.any(np.diff(ti) < 0)):
            raise ValueError('Monotone finite transistor tables with physical origin required')
        lookup.append((tv, ti))
    if np.max(va) > lookup[0][0][-1] or -np.min(va) > lookup[1][0][-1]: raise ValueError('Applied voltage outside lookup')
    limit = max(float(abs(va).max()), table_step); count = int(np.ceil(limit/table_step))
    grid = np.linspace(-count*table_step, count*table_step, 2*count+1)
    s = surfaces(grid, shared, parameters, panels=panels, order=order)
    initial = np.ascontiguousarray(initial_distribution(shared, parameters, panels=panels, order=order, initial_m=initial_m))
    j0 = np.ascontiguousarray(np.concatenate(s['current'][:2], axis=1))
    j1 = np.ascontiguousarray(np.concatenate(s['current'][2:], axis=1))
    rates = np.ascontiguousarray(s['log_rates'].transpose(1, 2, 0))
    delta = np.r_[0., np.diff(va)]; nsub = np.ceil(abs(delta)/max_step-1e-10).astype(np.int32)
    nsub[(delta != 0) & (nsub == 0)] = 1
    raw = abs(delta[nsub > 0])/nsub[nsub > 0]
    rounded = np.array([float(f'{x:.12g}') for x in raw])
    durations, ids = np.unique(rounded, return_inverse=True)
    step_ids = np.full(len(va), -1, dtype=np.int32); step_ids[nsub > 0] = ids
    n, k = len(va), len(s['weights'])
    if len(durations)*len(grid)*k*32*8 > 600_000_000:
        raise ValueError('Too many distinct step durations for propagator cache')
    current = np.empty(n); vh = np.empty(n); vd = np.empty(n); states = np.empty((n, 4, k))
    errors = np.empty(n); roots = np.empty(n)
    code = _core().rc_run(n, k, len(grid), va, np.ascontiguousarray(grid), j0, j1, rates,
                         np.ascontiguousarray(np.tile(s['weights'], 2)), len(durations), np.ascontiguousarray(durations),
                         np.ascontiguousarray(step_ids), np.ascontiguousarray(nsub),
                         len(lookup[0][0]), *lookup[0], len(lookup[1][0]), *lookup[1], initial,
                         current, vh, vd, states, errors, roots, int(diagnostics))
    if code: raise RuntimeError(f'Retained-charge trajectory failed: {code}')
    return {'current_A': current, 'internal_voltage_V': vh, 'transistor_voltage_V': vd, 'states': states,
            'maximum_kcl_relative_error': float(errors.max()), 'root_count': roots,
            'quadrature_weights': s['weights'], 'table_step': table_step, 'max_step': max_step,
            'distinct_substep_durations_V': durations,
            'maximum_relative_duration_rounding': float(np.max(abs(rounded/raw-1))) if len(raw) else 0.}
