"""Discrete conducting bottleneck versus a deterministic population mean.

Two configurations have distinct frozen-state load lines. At each applied
voltage, formation is evaluated at the OFF configuration's internal voltage,
removal at the ON configuration's internal voltage. The conditional fixed-
internal-voltage rate law is inherited from bidirectional_state. This does
not establish equilibrium thermodynamics of the entire biased circuit.
"""
import numpy as np
from .bidirectional_state import _core, log_rates


def frozen_load(vapp, transport, state, transistor_voltage, transistor_current):
    va = np.asarray(vapp, dtype=float); s = np.broadcast_to(np.asarray(state, dtype=float), va.shape)
    if len(transport) != 4 or np.any((s < 0) | (s > 1)) or np.any(va < 0):
        raise ValueError('Four transport parameters, nonnegative voltage and state in [0,1] required')
    p = np.ascontiguousarray([*transport, 0., 0., 0., .5, 0.], dtype=float)
    tv, ti = [np.ascontiguousarray(a, dtype=float) for a in (transistor_voltage, transistor_current)]
    if np.any(~np.isfinite(va)) or np.any(~np.isfinite(p)) or p[1] < 0 or p[3] <= 0:
        raise ValueError('Finite valid transport inputs required')
    if tv[0] != 0 or ti[0] != 0 or np.any(np.diff(tv) <= 0) or np.any(np.diff(ti) < 0) or np.max(va) > tv[-1]:
        raise ValueError('Invalid monotone transistor domain')
    vh, current = np.empty(va.size), np.empty(va.size); result = np.empty(2); lib = _core()
    for j, (voltage, state_value) in enumerate(zip(va.ravel(), s.ravel())):
        lp = float(np.log(state_value)) if state_value > 0 else -745.
        lib.bd_load(float(voltage), lp, p, tv, ti, len(tv), result)
        vh[j], current[j] = result
    return vh.reshape(va.shape), current.reshape(va.shape)


def clock_grid(vapp, transport, transistor_voltage, transistor_current, substeps=8):
    """Midpoint internal voltages for BOTH frozen configurations, before fitting rates."""
    va = np.asarray(vapp, dtype=float)
    if va.ndim != 1 or len(va) < 2 or substeps < 1:
        raise ValueError('One-dimensional voltage history and positive substeps required')
    fractions = (np.arange(substeps)+.5)/substeps
    middle = va[:-1, None] + (va[1:]-va[:-1])[:, None]*fractions
    v0, _ = frozen_load(middle, transport, 0., transistor_voltage, transistor_current)
    v1, _ = frozen_load(middle, transport, 1., transistor_voltage, transistor_current)
    return {'off_voltage': v0, 'on_voltage': v1, 'path_increment': np.abs(np.diff(va))/substeps}


def kinetic_parameters(kinetic, candidate):
    k = np.asarray(kinetic, dtype=float)
    expected = {'constant': 2, 'single_hop': 3, 'polarizable_saddle': 4}.get(candidate)
    if k.shape != (expected,):
        raise ValueError('Kinetic parameter vector does not match candidate')
    peak, delta = k[:2]; charge = k[2] if len(k) >= 3 else 0.; lowering = k[3] if len(k) == 4 else 0.
    return np.array([-9., 6., -6., 5., peak, delta, charge, .5, lowering])


def step_log_probabilities(log_build, log_remove, path_increment):
    """Exact two-state propagator at fixed rates, with log-domain stay probabilities."""
    a, b, dx = np.broadcast_arrays(log_build, log_remove, path_increment)
    if np.any(dx < 0) or np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)):
        raise ValueError('Finite log rates and nonnegative path increment required')
    total = np.logaddexp(a, b)
    with np.errstate(divide='ignore'):
        z = np.exp(np.minimum(total+np.log(dx), 700.))
        relax = np.log(-np.expm1(-z))
    out = np.empty((*a.shape, 2, 2))
    out[..., 0, 1] = a-total+relax
    out[..., 1, 0] = b-total+relax
    out[..., 0, 0] = np.logaddexp(-z, b-total+relax)
    out[..., 1, 1] = np.logaddexp(-z, a-total+relax)
    return out


def transition_log_probabilities(grid, kinetic, candidate):
    p = kinetic_parameters(kinetic, candidate)
    a, _ = log_rates(grid['off_voltage'], p, 'polarizable_saddle')
    _, b = log_rates(grid['on_voltage'], p, 'polarizable_saddle')
    steps = step_log_probabilities(a, b, grid['path_increment'][:, None])
    n, substeps = a.shape
    total = np.full((n, 2, 2), -np.inf); total[:, 0, 0] = 0.; total[:, 1, 1] = 0.
    for j in range(substeps):
        # Matrix product in log space, summing both possible intermediate states.
        total = np.logaddexp(total[:, :, 0, None]+steps[:, j, 0, None, :],
                             total[:, :, 1, None]+steps[:, j, 1, None, :])
    return total


def negative_log_likelihood(log_transitions, states, interval_mask=None):
    z = np.asarray(states, dtype=int)
    if len(z) != len(log_transitions)+1 or np.any((z != 0) & (z != 1)):
        raise ValueError('Binary observed state at each voltage point required')
    chosen = log_transitions[np.arange(len(z)-1), z[:-1], z[1:]]
    if interval_mask is not None:
        chosen = chosen[np.asarray(interval_mask, dtype=bool)]
    return float(-np.sum(chosen))


def population(log_transitions, initial_probability=1.):
    if not 0 <= initial_probability <= 1:
        raise ValueError('Initial probability must be in [0,1]')
    p = np.array([1-initial_probability, initial_probability]); result = [initial_probability]
    for transition in np.exp(log_transitions):
        p = p @ transition; result.append(float(p[1]))
    return np.array(result)


def sample_paths(log_transitions, n_paths=10000, seed=317):
    rng = np.random.default_rng(seed); states = np.ones((n_paths, len(log_transitions)+1), dtype=np.int8)
    probabilities = np.exp(log_transitions)
    for j, transition in enumerate(probabilities):
        probability_on = transition[states[:, j], 1]
        states[:, j+1] = rng.random(n_paths) < probability_on
    return states
