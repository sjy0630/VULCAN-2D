"""Instantaneous finite-domain current support, without invented path dynamics.

Parameters: log10 B, spectrum center, halfwidth, passive lever, Delta, U.
B = A*expit(contact log ratio). Four states have energy shifts Delta, Delta+U,
0, U. Inferring a closest configuration from observed current is an optimistic
support calculation and is deliberately separate from the forward readout.
"""
from pathlib import Path
import ctypes
import hashlib
import shutil
import subprocess
import sys
import numpy as np
from .charge_spectrum import quadrature
from .coupled_charge_reduced import log_fermi_pair
from .coupled_charge import VT, LN10

LO = np.array([-10., -.5, 0., .01, 0., 0.])
HI = np.array([-2., 1.5, 1.5, .99, 1.2, 1.2])
_CORE = None


def conditional_currents(voltage, parameters, *, panels=12, order=8):
    p = np.asarray(parameters, float); v = np.asarray(voltage, float)
    if p.shape != (6,) or not np.isfinite(p).all() or np.any(p[[2, 4, 5]] < 0) or not 0 <= p[3] <= 1:
        raise ValueError('Finite physical readout parameters required')
    if v.ndim != 1 or not np.isfinite(v).all(): raise ValueError('Finite voltage vector required')
    nodes, weights = quadrature(panels, order)
    shift = np.array([p[4], p[4]+p[5], 0., p[5]])
    energy = p[1]+p[2]*nodes[None, None, :]+shift[:, None, None]
    vv = v[None, :, None]
    lf, lc = log_fermi_pair((energy-p[3]*vv)/VT)
    rf, rc = log_fermi_pair((energy+(1-p[3])*vv)/VT)
    with np.errstate(divide='ignore'):
        drive = np.log(-np.expm1(-abs(vv)/VT))
    pair = np.where(vv >= 0, lf+rc, rf+lc)
    current = np.exp(p[0]*LN10+pair+drive)
    return current@weights


def fractions(domain_count):
    if isinstance(domain_count, bool) or int(domain_count) != domain_count or domain_count < 1:
        raise ValueError('Positive integer domain count required')
    n = int(domain_count)
    counts = [(a, b, c, n-a-b-c) for a in range(n+1) for b in range(n-a+1) for c in range(n-a-b+1)]
    return np.array(counts, float)/n


def _core():
    global _CORE
    if _CORE is not None: return _CORE
    source = Path(__file__).with_name('_finite_readout_core.cpp'); dependency = source.with_name('_activated_configuration_core.cpp')
    key = hashlib.sha256(source.read_bytes()+dependency.read_bytes()).hexdigest()[:20]
    folder = source.parents[1]/'.venv/.cache/finite_readout'; folder.mkdir(parents=True, exist_ok=True)
    output = folder/('core_'+key+('.dylib' if sys.platform == 'darwin' else '.so'))
    if not output.exists():
        compiler = shutil.which('c++')
        if not compiler: raise RuntimeError('C++ compiler required')
        flags = ['-dynamiclib'] if sys.platform == 'darwin' else ['-shared', '-fPIC']
        subprocess.run([compiler, '-O3', '-std=c++11', *flags, str(source), '-o', str(output)], check=True, capture_output=True)
    lib = ctypes.CDLL(str(output)); a = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    lib.fr_load.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, a, a, a, a,
                            ctypes.c_int, a, a, ctypes.c_int, a, a, a, a, a, a]
    lib.fr_load.restype = ctypes.c_int; _CORE = lib
    return lib


def table_load(applied, grid, currents, populations, positive_lookup, negative_lookup):
    """Solve each fixed four-state population; no measured current is accepted."""
    va, grid, cur, pop = [np.ascontiguousarray(x, dtype=float) for x in (applied, grid, currents, populations)]
    if va.ndim != 1 or not va.size or not np.isfinite(va).all(): raise ValueError('Finite applied voltage vector required')
    if grid.ndim != 1 or len(grid) < 3 or not np.isfinite(grid).all(): raise ValueError('Finite uniform grid required')
    if np.any(np.diff(grid) <= 0) or not np.allclose(np.diff(grid), grid[1]-grid[0], rtol=1e-10, atol=1e-14):
        raise ValueError('Uniform voltage grid required')
    if min(va) < grid[0] or max(va) > grid[-1] or not grid[0] < 0 < grid[-1]: raise ValueError('Applied voltages outside signed table')
    if cur.shape != (len(grid), 4) or not np.isfinite(cur).all() or np.any(cur < 0): raise ValueError('Four nonnegative finite currents per grid point required')
    if pop.ndim != 2 or pop.shape[1] != 4 or not len(pop) or not np.isfinite(pop).all() or np.any(pop < 0) or not np.allclose(pop.sum(axis=1), 1., atol=1e-13, rtol=0):
        raise ValueError('Four-state normalized nonnegative populations required')
    look = []
    for tv, ti in (positive_lookup, negative_lookup):
        tv, ti = np.ascontiguousarray(tv, dtype=float), np.ascontiguousarray(ti, dtype=float)
        if tv.ndim != 1 or len(tv) < 2 or tv.shape != ti.shape or not np.isfinite(tv).all() or not np.isfinite(ti).all() or tv[0] != 0 or ti[0] != 0 or np.any(np.diff(tv) <= 0) or np.any(np.diff(ti) < 0):
            raise ValueError('Finite monotone transistor tables required')
        look.append((tv, ti))
    if max(va) > look[0][0][-1] or -min(va) > look[1][0][-1]: raise ValueError('Applied voltage outside transistor table')
    shape = (len(pop), len(va)); current, internal, transistor, error = [np.empty(shape) for _ in range(4)]
    code = _core().fr_load(len(pop), len(va), len(grid), pop, va, grid, cur,
                           len(look[0][0]), *look[0], len(look[1][0]), *look[1], current, internal, transistor, error)
    if code: raise RuntimeError(f'Finite-readout load failed: {code}')
    return {'current_A': current, 'internal_voltage_V': internal, 'transistor_voltage_V': transistor,
            'maximum_kcl_relative_error': float(error.max())}


def readout(applied, parameters, populations, positive_lookup, negative_lookup, *, table_step=.005, panels=12, order=8):
    va = np.asarray(applied, float)
    if not np.isfinite(table_step) or table_step <= 0: raise ValueError('Positive table step required')
    limit = max(float(np.max(abs(va))), table_step); n = int(np.ceil(limit/table_step))
    grid = np.linspace(-n*table_step, n*table_step, 2*n+1)
    current = conditional_currents(grid, parameters, panels=panels, order=order)
    return table_load(va, grid, current.T, populations, positive_lookup, negative_lookup)


def nearest_configuration(observed, currents):
    """Oracle lower bound; labels consume observed current and are not predictions."""
    y = np.asarray(observed, float); values = np.asarray(currents, float)
    if y.ndim != 1 or values.ndim != 2 or values.shape[1] != len(y) or not np.isfinite(y).all() or not np.isfinite(values).all() or np.any(y <= 0) or np.any(values <= 0):
        raise ValueError('Positive finite currents required for logarithmic comparison')
    residual = np.log10(values/y[None, :]); index = np.argmin(abs(residual), axis=0)
    return index, residual[index, np.arange(len(y))]
