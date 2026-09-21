"""Zero-bias continuation of joint charge/configuration probability.

This sibling wrapper preserves the archived retained-charge implementation.
Contact changes here are imposed counterfactuals, not a generative event law.
"""
import numpy as np
from scipy.special import log_expit
from . import retained_charge_configuration as rc


def compensated_contact(shared, parameters, log_ratio):
    """Change g while preserving B=A*expit(g); never clip compensated A."""
    s, p = np.array(shared, dtype=float), np.array(parameters, dtype=float)
    if s.shape != (6,) or p.shape != (9,) or not np.all(np.isfinite(np.r_[s, p, log_ratio])):
        raise ValueError('Finite six electronic and nine physical parameters required')
    p[0] += (log_expit(s[4])-log_expit(log_ratio))/np.log(10.)
    s[4] = log_ratio
    return s, p


def continue_from_zero(applied, shared, parameters, positive_lookup, negative_lookup,
                       initial_state, *, table_step=.005, max_step=.01, panels=8,
                       order=8, diagnostics=False):
    """Propagate arbitrary full P(q,m;E), specified at the first zero point.

    No initialization, state fitting, or normalization is performed. The caller
    must preserve the energy-node basis when handing off a previous state. Only
    zero-bias handoffs are supported: these also reset the load root reference
    unambiguously. Duplicate zero points carry exactly zero path duration.
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
                np.any(~np.isfinite(ti)) or tv[0] != 0 or ti[0] != 0 or
                np.any(np.diff(tv) <= 0) or np.any(np.diff(ti) < 0)):
            raise ValueError('Monotone finite transistor tables with physical origin required')
        lookup.append((tv, ti))
    if np.max(va) > lookup[0][0][-1] or -np.min(va) > lookup[1][0][-1]:
        raise ValueError('Applied voltage outside lookup')
    limit = max(float(abs(va).max()), table_step); count = int(np.ceil(limit/table_step))
    grid = np.linspace(-count*table_step, count*table_step, 2*count+1)
    surf = rc.surfaces(grid, shared, parameters, panels=panels, order=order)
    n, k = len(va), len(surf['weights'])
    initial = np.ascontiguousarray(initial_state, dtype=float)
    if (initial.shape != (4, k) or np.any(~np.isfinite(initial)) or np.any(initial < 0) or
            not np.allclose(initial.sum(axis=0), 1., atol=2e-11, rtol=0)):
        raise ValueError('Full nonnegative state (4,n_energy) with normalized columns required')
    # Engine columns are m=0 nodes then m=1 nodes within each q. Weights
    # repeat without a factor 1/2: each energy node has four state probabilities.
    j0 = np.ascontiguousarray(np.concatenate(surf['current'][:2], axis=1))
    j1 = np.ascontiguousarray(np.concatenate(surf['current'][2:], axis=1))
    rates = np.ascontiguousarray(surf['log_rates'].transpose(1, 2, 0))
    delta = np.r_[0., np.diff(va)]
    nsub = np.ceil(abs(delta)/max_step-1e-10).astype(np.int32)
    nsub[(delta != 0) & (nsub == 0)] = 1
    raw = abs(delta[nsub > 0])/nsub[nsub > 0]
    rounded = np.array([float(f'{x:.12g}') for x in raw])
    durations, ids = np.unique(rounded, return_inverse=True)
    step_ids = np.full(n, -1, dtype=np.int32); step_ids[nsub > 0] = ids
    if len(durations)*len(grid)*k*32*8 > 600_000_000:
        raise ValueError('Too many distinct step durations for propagator cache')
    current = np.empty(n); vh = np.empty(n); vd = np.empty(n)
    states = np.empty((n, 4, k)); errors = np.empty(n); roots = np.empty(n)
    code = rc._core().rc_run(n, k, len(grid), va, np.ascontiguousarray(grid), j0, j1, rates,
        np.ascontiguousarray(np.tile(surf['weights'], 2)), len(durations),
        np.ascontiguousarray(durations), np.ascontiguousarray(step_ids), np.ascontiguousarray(nsub),
        len(lookup[0][0]), *lookup[0], len(lookup[1][0]), *lookup[1], initial,
        current, vh, vd, states, errors, roots, int(diagnostics))
    if code:
        raise RuntimeError(f'Retained-contact continuation failed: {code}')
    return {'current_A': current, 'internal_voltage_V': vh, 'transistor_voltage_V': vd,
            'states': states, 'maximum_kcl_relative_error': float(errors.max()),
            'root_count': roots, 'quadrature_weights': surf['weights'],
            'table_step': table_step, 'max_step': max_step,
            'distinct_substep_durations_V': durations,
            'maximum_relative_duration_rounding': float(np.max(abs(rounded/raw-1))) if len(raw) else 0.}


def switched_trajectory(applied, shared, parameters, positive_lookup, negative_lookup,
                        split_index, first_log_ratio, second_log_ratio, **settings):
    """Impose one contact change between native duplicate-zero observations.

    Every native point is retained. Energy nodes and all non-contact physical
    parameters stay fixed. The boundary is prescribed, not predicted.
    """
    va = np.asarray(applied, dtype=float)
    if (va.ndim != 1 or not isinstance(split_index, (int, np.integer)) or
            not 2 <= split_index <= len(va)-2 or va[split_index-1] != 0 or va[split_index] != 0):
        raise ValueError('Split must lie between two native zero-bias points')
    s0, p0 = compensated_contact(shared, parameters, first_log_ratio)
    s1, p1 = compensated_contact(shared, parameters, second_log_ratio)
    initial = rc.initial_distribution(s0, p0, panels=settings.get('panels', 8), order=settings.get('order', 8))
    first = continue_from_zero(va[:split_index], s0, p0, positive_lookup, negative_lookup, initial, **settings)
    last = continue_from_zero(va[split_index:], s1, p1, positive_lookup, negative_lookup, first['states'][-1], **settings)
    result = {name: np.concatenate([first[name], last[name]]) for name in
              ('current_A', 'internal_voltage_V', 'transistor_voltage_V', 'states', 'root_count')}
    result.update(quadrature_weights=first['quadrature_weights'],
                  maximum_kcl_relative_error=max(first['maximum_kcl_relative_error'], last['maximum_kcl_relative_error']),
                  maximum_relative_duration_rounding=max(first['maximum_relative_duration_rounding'], last['maximum_relative_duration_rounding']),
                  boundary_state_change=float(np.max(abs(first['states'][-1]-last['states'][0]))),
                  segment_parameters=[p0.tolist(), p1.tolist()], segment_shared_parameters=[s0.tolist(), s1.tolist()],
                  split_index=int(split_index))
    return result
