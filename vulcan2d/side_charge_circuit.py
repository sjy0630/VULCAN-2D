"""Load-root audit for possibly nonmonotone equilibrium charge readout."""
import numpy as np
from scipy.optimize import brentq
from .side_charge import shape_and_gradients


def load_roots(applied, voltage, device_current, transistor_voltage, transistor_current):
    """All roots of the piecewise-linear device and load tables, with stability.

    Stable here means positive incremental device+load conductance under a
    scalar capacitive load-line dynamics, not microscopic dynamical stability.
    """
    va = float(applied)
    v, current = np.asarray(voltage), np.asarray(device_current)
    tv, ti = np.asarray(transistor_voltage), np.asarray(transistor_current)
    if va <= 0 or va > min(v[-1], tv[-1]) or v[0] != 0 or tv[0] != 0:
        raise ValueError('Positive applied voltage inside both tables required')
    shifted = va-tv
    grid = np.unique(np.r_[0., v[(v > 0) & (v < va)], shifted[(shifted > 0) & (shifted < va)], va])
    difference = np.interp(grid, v, current)-np.interp(va-grid, tv, ti)
    intervals = np.flatnonzero(difference[:-1]*difference[1:] < 0)
    roots = []
    for j in intervals:
        slope = (difference[j+1]-difference[j])/(grid[j+1]-grid[j])
        root = grid[j]-difference[j]/slope
        roots.append({'internal_voltage_V':float(root), 'stable_load_line':bool(slope > 0)})
    for j in np.flatnonzero(difference == 0):
        lo, hi = max(j-1, 0), min(j+1, len(grid)-1)
        roots.append({'internal_voltage_V':float(grid[j]),
                      'stable_load_line':bool(difference[hi] > difference[lo])})
    return sorted(roots, key=lambda r:r['internal_voltage_V'])


def forward_initial(applied_voltage, shared, state, lookup, *, mode='equilibrium', step=.0005, order=48):
    """Enumerate load roots, and refine unique solutions using direct WKB.

    No measured current is accepted. Multiple roots are reported as ambiguous,
    with no silently chosen branch or target-current-based root selection.
    """
    signed = np.asarray(applied_voltage, float)
    if np.any(signed == 0) or np.any(np.sign(signed) != np.sign(signed[0])):
        raise ValueError('One nonzero-polarity branch required')
    sign = np.sign(signed[0]); va = np.abs(signed)
    grid = np.linspace(0, va.max(), int(np.ceil(va.max()/step))+1)
    values = shape_and_gradients(sign*grid[1:], shared, [state[1]], mode=mode, order=order)[0][0]
    current = np.r_[0., 10**(state[0]+values)]
    predictions, diagnostics, max_kcl = [], [], 0.
    for voltage in va:
        roots = load_roots(voltage, grid, current, lookup.voltage, lookup.current)
        record = {'applied_abs_voltage_V':float(voltage), 'roots':roots}
        if len(roots) != 1 or not roots[0]['stable_load_line']:
            predictions.append(None); diagnostics.append(record); continue
        def internal(vh):
            if vh <= 0:
                return 0.
            shape = shape_and_gradients([sign*vh], shared, [state[1]], mode=mode, order=order)[0][0, 0]
            return float(10**(state[0]+shape))
        # Solve in transistor voltage to preserve low-current relative accuracy.
        vd = brentq(lambda x:internal(voltage-x)-np.interp(x, lookup.voltage, lookup.current),
                    0., voltage, xtol=1e-30, rtol=1e-13)
        ih = internal(voltage-vd); it = float(np.interp(vd, lookup.voltage, lookup.current))
        kcl = abs(ih-it)/max(ih, it)
        max_kcl = max(max_kcl, kcl)
        record.update(internal_voltage_V=float(voltage-vd), transistor_voltage_V=float(vd),
                      kcl_relative_error=kcl)
        predictions.append(ih); diagnostics.append(record)
    return {'predicted_current_A':predictions, 'root_diagnostics':diagnostics,
            'ambiguous_count':sum(v is None for v in predictions), 'max_kcl_relative_error':max_kcl}
