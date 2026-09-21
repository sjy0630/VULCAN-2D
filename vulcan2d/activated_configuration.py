"""Activated retained configurations coupled to fast correlated charge readout.

The fast states are conditionally stationary. Slow effective rates are averages
of microscopic detailed-balance rates, preserving the equilibrium free-energy
relation at zero bias. Time is applied-voltage path, not calibrated seconds.
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

LO = np.array([-10., 0., -.6, -.6, -40., -.6, -.3, 0.])
HI = np.array([-2., .6, .6, .6, 5., .6, .3, .03])
_CORE = None


def fast_joint(voltage, ed, et, shared, log_side_rate=0.):
    """Positive-term stationary joint probabilities; ed/et can broadcast.

    shared = [Ed center,xd,xt,U fraction,ln Gamma_L/R,width]. This function
    takes channel ed separately and returns currents in e*N*Gamma_R units.
    """
    shared = np.asarray(shared, float)
    v, ed, et = np.broadcast_arrays(np.asarray(voltage, float), np.asarray(ed, float), np.asarray(et, float))
    xd, xt, gl = shared[1], shared[2], shared[4]
    u, _ = charging(np.r_[shared[:5], 0.]); norm = np.logaddexp(gl, 0.)
    level, side = ed+(1-xd)*v, et+(1-xt)*v
    l0, lc0 = log_fermi_pair((level-v)/VT); r0, rc0 = log_fermi_pair(level/VT)
    l1, lc1 = log_fermi_pair((level+u-v)/VT); r1, rc1 = log_fermi_pair((level+u)/VT)
    a0 = np.logaddexp(gl+l0, r0)-norm; a0c = np.logaddexp(gl+lc0, rc0)-norm
    a1 = np.logaddexp(gl+l1, r1)-norm; a1c = np.logaddexp(gl+lc1, rc1)-norm
    c0, c0c = log_fermi_pair(side/VT); c1, c1c = log_fermi_pair((side+u)/VT)
    with np.errstate(divide='ignore'):
        shift = np.log(-np.expm1(-u/VT)); drive = np.log(-np.expm1(-np.abs(v)/VT))
    delta_a = np.logaddexp(gl+l0+lc1, r0+rc1)+shift-norm
    den = np.logaddexp(np.logaddexp(a0c, a1), delta_a+np.logaddexp(c0c, c1))
    m = np.logaddexp(c0+a0c, c1+a0)-den
    mc = np.logaddexp(c0c+a1c, c1c+a1)-den
    n = np.logaddexp(a0+c0c, a1+c0)-den
    nc = np.logaddexp(a0c+c1c, a1c+c1)-den
    both_norm = np.logaddexp(norm, log_side_rate)
    joint = np.array([np.logaddexp(norm+a0c+mc, log_side_rate+c0c+nc),
                      np.logaddexp(norm+a0+mc, log_side_rate+c1c+n),
                      np.logaddexp(norm+a1c+m, log_side_rate+c0+nc),
                      np.logaddexp(norm+a1+m, log_side_rate+c1+n)])-both_norm
    d0 = np.where(v >= 0, l0+rc0, r0+lc0)
    d1 = np.where(v >= 0, l1+rc1, r1+lc1)
    logcurrent = gl-norm+np.logaddexp(mc+d0, m+d1)+drive
    return {'log_probability': joint, 'log_abs_current_scale': logcurrent}


def surfaces(voltage, shared, parameters, *, panels=4, order=8, log_side_rate=0.):
    """Current surfaces and slow configuration rates for independent channels."""
    p = np.asarray(parameters, float); shared = np.asarray(shared, float)
    if p.shape != (8,) or np.any(~np.isfinite(p)) or p[1] < 0 or p[7] < 0:
        raise ValueError('Eight finite parameters; nonnegative energy shift and saddle polarizability required')
    if shared.shape != (6,) or shared[5] < 0 or np.any(~np.isfinite(shared)):
        raise ValueError('Six finite shared readout parameters required')
    v = np.asarray(voltage, float)
    if v.ndim != 1 or np.any(~np.isfinite(v)): raise ValueError('Finite voltage vector required')
    z, w = quadrature(panels, order); ed = shared[0]+shared[5]*z
    a = fast_joint(v[:, None], ed[None, :]+p[1], p[2], shared, log_side_rate)
    b = fast_joint(v[:, None], ed[None, :], p[3], shared, log_side_rate)
    n = np.array([0, 1, 0, 1])[:, None, None]; m = np.array([0, 0, 1, 1])[:, None, None]
    delta = p[5]-p[6]*v[None, :, None]-n*p[1]+m*(p[3]-p[2])
    barrier = p[4]+p[7]*v*v/VT
    log01 = barrier[:, None]+logsumexp(a['log_probability']-delta/(2*VT), axis=0)
    log10 = barrier[:, None]+logsumexp(b['log_probability']+delta/(2*VT), axis=0)
    currents = np.exp(np.array([a['log_abs_current_scale'], b['log_abs_current_scale']])+LN10*p[0])
    return {'current': currents, 'log_rates': np.array([log01, log10]),
            'log_probability': np.array([a['log_probability'], b['log_probability']]),
            'nodes': z, 'weights': w, 'voltage': v}


def _core():
    global _CORE
    if _CORE is not None: return _CORE
    source = Path(__file__).with_name('_activated_configuration_core.cpp')
    key = hashlib.sha256(source.read_bytes()).hexdigest()[:20]
    folder = source.parents[1]/'.venv/.cache/activated_configuration'; folder.mkdir(parents=True, exist_ok=True)
    output = folder/('core_'+key+('.dylib' if sys.platform == 'darwin' else '.so'))
    if not output.exists():
        compiler = shutil.which('c++')
        if not compiler: raise RuntimeError('C++ compiler required for the trajectory helper')
        flags = ['-dynamiclib'] if sys.platform == 'darwin' else ['-shared', '-fPIC']
        subprocess.run([compiler, '-O3', '-std=c++11', *flags, str(source), '-o', str(output)], check=True, capture_output=True)
    lib = ctypes.CDLL(str(output)); a = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    lib.ac_run.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, a, a, a, a, a, a,
                          ctypes.c_int, a, a, ctypes.c_int, a, a, ctypes.c_double, ctypes.c_double,
                          a, a, a, a, a, a, ctypes.c_int]
    lib.ac_run.restype = ctypes.c_int
    _CORE = lib
    return lib


def trajectory(applied, shared, parameters, positive_lookup, negative_lookup, *,
               table_step=.01, max_step=.02, panels=4, order=8, initial=0., diagnostics=False):
    """Native-voltage complete sequence. No observed current is accepted.

    Rates have units 1/V of applied path. Repeated voltages imply no known
    dwell. Root continuation follows scalar capacitive-load attraction.
    """
    va = np.ascontiguousarray(applied, dtype=float)
    if va.ndim != 1 or len(va) < 2 or va[0] != 0 or np.any(~np.isfinite(va)):
        raise ValueError('Finite voltage sequence starting at zero required')
    if not np.all(np.isfinite([table_step, max_step, initial])) or table_step <= 0 or max_step <= 0 or not 0 <= initial <= 1:
        raise ValueError('Positive discretization and initial probability in [0,1] required')
    # Include both polarities in a uniformly spaced table, with exact zero.
    limit = max(float(np.max(np.abs(va))), table_step)
    count = int(np.ceil(limit/table_step)); grid = np.linspace(-count*table_step, count*table_step, 2*count+1)
    s = surfaces(grid, shared, parameters, panels=panels, order=order)
    j0 = np.ascontiguousarray(s['current'][0]); j1 = np.ascontiguousarray(s['current'][1])
    rates = np.ascontiguousarray(s['log_rates'].transpose(1, 0, 2))
    n = len(va); k = len(s['weights'])
    current = np.empty(n); vh = np.empty(n); vd = np.empty(n)
    states = np.empty((n, 2, k)); errors = np.empty(n); roots = np.empty(n)
    pv, pi = [np.ascontiguousarray(x, dtype=float) for x in positive_lookup]
    nv, ni = [np.ascontiguousarray(x, dtype=float) for x in negative_lookup]
    for tv, ti in ((pv, pi), (nv, ni)):
        if (tv.ndim != 1 or len(tv) < 2 or ti.shape != tv.shape or
                np.any(~np.isfinite(tv)) or np.any(~np.isfinite(ti)) or
                tv[0] != 0 or ti[0] != 0 or np.any(np.diff(tv) <= 0) or np.any(np.diff(ti) < 0)):
            raise ValueError('Monotone transistor lookup with physical origin required')
    if np.max(va) > pv[-1] or -np.min(va) > nv[-1]: raise ValueError('Applied voltage outside lookup')
    code = _core().ac_run(n, k, len(grid), va, np.ascontiguousarray(grid), j0, j1, rates, np.ascontiguousarray(s['weights']),
                         len(pv), pv, pi, len(nv), nv, ni, max_step, initial,
                         current, vh, vd, states, errors, roots, int(diagnostics))
    if code: raise RuntimeError(f'Activated configuration trajectory failed: {code}')
    return {'current_A': current, 'internal_voltage_V': vh, 'transistor_voltage_V': vd, 'states': states,
            'maximum_kcl_relative_error': float(errors.max()), 'root_count': roots,
            'quadrature_weights': s['weights'], 'table_step': table_step, 'max_step': max_step}
