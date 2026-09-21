"""Finite equally weighted conducting contributions with circuit feedback.

The birth/death count n has current H+(n/N)C. Holding total C fixed makes N
a granularity parameter in this comparison, not an inferred atom/path count.
Rate energies are defined conditionally at a common internal voltage; using
state-dependent load lines is not a proof of equilibrium of the biased circuit.
"""
import numpy as np
from .stochastic_bottleneck import frozen_load, kinetic_parameters
from .bidirectional_state import log_rates


def generator(internal_voltage, count, kinetic, candidate):
    v = np.asarray(internal_voltage, dtype=float)
    if count < 1 or int(count) != count or v.shape[-1] != count+1:
        raise ValueError('Positive integer count and one voltage per count state required')
    p = kinetic_parameters(kinetic, candidate)
    build, remove = log_rates(v, p, 'polarizable_saddle')
    n = np.arange(count+1)
    birth = (count-n)*np.exp(build); death = n*np.exp(remove)
    q = np.zeros((*v.shape, count+1)); index = np.arange(count+1)
    q[..., index, index] = -(birth+death)
    q[..., index[:-1], index[1:]] = birth[..., :-1]
    q[..., index[1:], index[:-1]] = death[..., 1:]
    return q


def stochastic_exponential(generators, duration):
    """Positive uniformization with scaling/squaring, batched row generators.

    The scaled Poisson mean is <=0.5. Sixteen terms leave <1e-18 tail mass.
    Tiny row-mass roundoff is corrected only below a checked tolerance and its
    maximum is returned. No negative entries are clipped.
    """
    q = np.asarray(generators, dtype=float)
    if q.ndim != 3 or q.shape[1] != q.shape[2] or np.any(~np.isfinite(q)):
        raise ValueError('Finite batch of square generators required')
    dt = np.broadcast_to(np.asarray(duration, dtype=float), (len(q),))
    if np.any(dt < 0) or np.any(~np.isfinite(dt)):
        raise ValueError('Finite nonnegative durations required')
    size = q.shape[-1]; ix = np.arange(size); eye = np.broadcast_to(np.eye(size), q.shape)
    rates = -q[:, ix, ix]
    off = q.copy(); off[:, ix, ix] = 0.
    if np.any(off < 0) or np.any(rates < 0) or np.max(np.abs(q.sum(axis=-1))) > 1e-10*max(1., float(rates.max())):
        raise ValueError('Nonnegative transition rates and zero row sums required')
    lam = rates.max(axis=1); z = lam*dt
    squarings = np.maximum(0, np.ceil(np.log2(np.maximum(z/.5, 1.)))).astype(int)
    small_z = z/np.exp2(squarings)
    discrete = eye + q/np.where(lam > 0, lam, 1.)[:, None, None]
    coefficient = np.exp(-small_z); power = eye.copy(); result = coefficient[:, None, None]*power
    for k in range(1, 17):
        power = power @ discrete
        coefficient *= small_z/k
        result += coefficient[:, None, None]*power
    max_drift = 0.
    def normalize(matrix):
        nonlocal max_drift
        sums = matrix.sum(axis=-1); drift = float(np.max(np.abs(sums-1.)))
        max_drift = max(max_drift, drift)
        if drift > 1e-7 or np.min(matrix) < 0:
            raise RuntimeError('Matrix exponential lost probability conservation/positivity')
        return matrix/sums[..., None]
    result = normalize(result)
    for k in range(int(squarings.max())):
        active = squarings > k
        result[active] = normalize(result[active] @ result[active])
    return result, {'maximum_raw_row_sum_roundoff': max_drift, 'maximum_scaling_squarings': int(squarings.max())}


def transitions(vapp, transport, kinetic, candidate, count, transistor_voltage, transistor_current, substeps=8):
    va = np.asarray(vapp, dtype=float)
    if va.ndim != 1 or len(va) < 2 or np.any(~np.isfinite(va)) or np.any(va < 0) or substeps < 1:
        raise ValueError('Finite nonnegative voltage history and positive substeps required')
    if count < 1 or int(count) != count: raise ValueError('Positive integer count required')
    count = int(count); fractions = (np.arange(substeps)+.5)/substeps
    mid = va[:-1, None]+np.diff(va)[:, None]*fractions
    drive = np.broadcast_to(mid[:, :, None], (*mid.shape, count+1))
    state = np.broadcast_to(np.arange(count+1)/count, drive.shape)
    vh, unused = frozen_load(drive, transport, state, transistor_voltage, transistor_current)
    q = generator(vh, count, kinetic, candidate)
    dt = np.abs(np.diff(va))/substeps
    result = np.broadcast_to(np.eye(count+1), (len(va)-1, count+1, count+1)).copy()
    max_roundoff, max_squarings = 0., 0
    for j in range(substeps):
        transition, diagnostic = stochastic_exponential(q[:, j], dt)
        result = result @ transition
        max_roundoff = max(max_roundoff, diagnostic['maximum_raw_row_sum_roundoff'])
        max_squarings = max(max_squarings, diagnostic['maximum_scaling_squarings'])
    return result, {'maximum_raw_row_sum_roundoff': max_roundoff,
                    'maximum_scaling_squarings': max_squarings,
                    'maximum_transition_row_sum_error': float(np.max(np.abs(result.sum(axis=-1)-1.)))}


def marginal(transitions, initial_count=None):
    pmat = np.asarray(transitions); count = pmat.shape[-1]-1
    initial = count if initial_count is None else initial_count
    if int(initial) != initial or not 0 <= initial <= count: raise ValueError('Invalid initial count')
    prob = np.zeros(count+1); prob[int(initial)] = 1.
    rows = [prob.copy()]
    for step in pmat:
        prob = prob @ step; rows.append(prob.copy())
    return np.array(rows)


def sample_counts(transitions, n_paths=20000, seed=512, initial_count=None):
    pmat = np.asarray(transitions); count = pmat.shape[-1]-1
    initial = count if initial_count is None else initial_count
    if int(initial) != initial or not 0 <= initial <= count: raise ValueError('Invalid initial count')
    states = np.empty((n_paths, len(pmat)+1), dtype=np.int16); states[:, 0] = initial
    rng = np.random.default_rng(seed)
    for j, transition in enumerate(pmat):
        cdf = np.cumsum(transition, axis=1); cdf[:, -1] = 1.
        states[:, j+1] = np.sum(rng.random(n_paths)[:, None] > cdf[states[:, j]], axis=1)
    return states


def state_currents(vapp, transport, count, transistor_voltage, transistor_current):
    va = np.asarray(vapp)
    drive = np.broadcast_to(va[:, None], (len(va), count+1))
    state = np.broadcast_to(np.arange(count+1)/count, drive.shape)
    return frozen_load(drive, transport, state, transistor_voltage, transistor_current)
