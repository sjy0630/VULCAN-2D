"""Conserved four-state transport/side-charge network with reciprocal feedback.

Each lead transition obeys local detailed balance. The side reservoir is R.
Tree and cycle sums avoid cancellation of nearly equal gross currents.
"""
from collections import Counter
from itertools import product
import numpy as np
from scipy.special import expit, logsumexp
from scipy.constants import Boltzmann, elementary_charge

VT = Boltzmann*300/elementary_charge
LN10 = np.log(10.)
# Elementary edges: L/R conductor transitions for m=0 and m=1; R side transitions.
EDGES = ((0,1),(1,0),(0,1),(1,0),(2,3),(3,2),(2,3),(3,2),(0,2),(2,0),(1,3),(3,1))


def tree_and_current_terms():
    trees = []
    for root in range(4):
        states = [s for s in range(4) if s != root]
        choices = [[k for k, (a,b) in enumerate(EDGES) if a == s] for s in states]
        group = []
        for selection in product(*choices):
            arrows = {EDGES[k][0]:EDGES[k][1] for k in selection}
            valid = True
            for start in states:
                current, visited = start, set()
                while current != root and current not in visited:
                    visited.add(current); current = arrows[current]
                if current != root:valid = False; break
            if valid:group.append(tuple(selection))
        trees.append(np.array(group, int))
    numerator = Counter()
    for state, edge, sign in ((0,0,1),(1,1,-1),(2,4,1),(3,5,-1)):
        for term in trees[state]:
            numerator[tuple(sorted((*term, edge)))] += sign
    positive, negative = [], []
    for term, count in numerator.items():
        if count > 0:positive.extend([term]*count)
        elif count < 0:negative.extend([term]*(-count))
    if len(positive) != len(negative):
        raise RuntimeError('Unbalanced current cycle polynomial')
    return tuple(trees), np.array(positive, int), np.array(negative, int)


TREES, POSITIVE_TERMS, NEGATIVE_TERMS = tree_and_current_terms()


def charging(shared):
    _, xd, xt, fraction, _, _ = np.asarray(shared, float)
    if not 0 <= fraction <= 1 or not 0 <= xd <= 1 or not 0 <= xt <= 1:
        raise ValueError('Passive positions and bounded mutual energy required')
    distance = 6*abs(xd-xt)
    cap = 1.439964/(3.1*max(distance, .25)); value = fraction*cap
    derivative = np.zeros(6); derivative[3] = cap
    if distance > .25:
        derivative[1] = -value/(xd-xt); derivative[2] = value/(xd-xt)
    return value, derivative


def log_rates(voltage, shared, trap_energy):
    """Elementary log rates and derivatives for 6 shared + one trap energy."""
    shared = np.asarray(shared, float)
    if shared.shape != (6,) or np.any(~np.isfinite(shared)):
        raise ValueError('Six finite shared parameters required')
    v, et = np.broadcast_arrays(np.asarray(voltage, float), np.asarray(trap_energy, float))
    if np.any(~np.isfinite(v)) or np.any(~np.isfinite(et)):
        raise ValueError('Finite voltage and trap energy required')
    ed, xd, xt, _, gl, gt = shared; u, du = charging(shared)
    rates, grads = [], []
    for occupied in (0, 1):
        energy = ed+(1-xd)*v+u*occupied
        de = np.zeros((*v.shape, 7)); de[..., :6] += occupied*du
        de[..., 0] += 1; de[..., 1] -= v
        for mu, prefactor, index in ((v, gl, 4), (0., 0., None)):
            z = (energy-mu)/VT; f = expit(-z)
            base = np.zeros_like(de)
            if index is not None:base[..., index] = 1
            rates.extend((prefactor-np.logaddexp(0., z), prefactor-np.logaddexp(0., -z)))
            grads.extend((base+(f-1)[..., None]*de/VT, base+f[..., None]*de/VT))
    for occupied in (0, 1):
        energy = et+(1-xt)*v+u*occupied
        de = np.zeros((*v.shape, 7)); de[..., :6] += occupied*du
        de[..., 2] -= v; de[..., 6] = 1
        z = energy/VT; f = expit(-z); base = np.zeros_like(de); base[..., 5] = 1
        rates.extend((gt-np.logaddexp(0., z), gt-np.logaddexp(0., -z)))
        grads.extend((base+(f-1)[..., None]*de/VT, base+f[..., None]*de/VT))
    return np.array(rates), np.array(grads)


def _sum_terms(logrates, derivatives, indices):
    terms = logrates[indices].sum(axis=1)
    gradient = derivatives[indices].sum(axis=1)
    total = logsumexp(terms, axis=0)
    weights = np.exp(terms-total)
    return total, (weights[..., None]*gradient).sum(axis=0)


def stationary(voltage, shared, trap_energy, *, derivatives=True):
    """Probabilities, signed net current/Gamma_R and stable log magnitude.

    Multiply net current by e*N*Gamma_R for physical amperes. State energy,
    parallel count and absolute rate remain conditional, unmeasured quantities.
    """
    v, et = np.broadcast_arrays(np.asarray(voltage, float), np.asarray(trap_energy, float))
    logs, dr = log_rates(v, shared, et)
    state_logs, state_grad = zip(*[_sum_terms(logs, dr, indices) for indices in TREES])
    state_logs, state_grad = np.array(state_logs), np.array(state_grad)
    z = logsumexp(state_logs, axis=0); probability = np.exp(state_logs-z)
    dz = np.sum(probability[..., None]*state_grad, axis=0)
    pos, dpos = _sum_terms(logs, dr, POSITIVE_TERMS)
    neg, dneg = _sum_terms(logs, dr, NEGATIVE_TERMS)
    selected = np.where(v >= 0, pos, neg)
    dselected = np.where((v >= 0)[..., None], dpos, dneg)
    with np.errstate(divide='ignore'):
        drive = np.log(-np.expm1(-np.abs(v)/VT))
    log_abs = selected+drive-z
    current = np.sign(v)*np.exp(log_abs)
    gradients = (dselected-dz)/LN10
    # Gross edge currents remain available for independent conservation checks.
    k = np.exp(logs); p0,p1,p2,p3 = probability
    left = k[0]*p0-k[1]*p1+k[4]*p2-k[5]*p3
    right = k[2]*p0-k[3]*p1+k[6]*p2-k[7]*p3
    side = k[8]*p0-k[9]*p2+k[10]*p1-k[11]*p3
    return {'probabilities':probability, 'log10_abs_current_scale':log_abs/LN10,
            'current_scale':current, 'derivatives':gradients,
            'gross_left_current':left, 'gross_right_current':right, 'gross_side_current':side,
            'forward_reverse_cycle_log_ratio_residual':pos-neg-v/VT}


def predict(voltage, shared, states):
    v = np.asarray(voltage, float); states = np.atleast_2d(states)
    if v.ndim != 1 or np.any(v == 0) or states.shape[1] != 2:
        raise ValueError('Nonzero voltage vector; states=[log10 current scale A, trap energy eV]')
    result = stationary(v[None, :], shared, states[:, 1, None])
    return states[:, 0, None]+result['log10_abs_current_scale']
