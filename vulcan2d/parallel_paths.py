"""Research parallel localized-level ensembles with shared polarity parameters.

Uniform energy bands are effective alternatives, not measured defect spectra.
Every element has the complete reservoir voltage difference and fixed contact
rates. Frozen-path mixtures do not supply a switching or relaxation equation.
"""
from functools import lru_cache
import numpy as np
from scipy.special import expit, logsumexp
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix, hstack
from .transport_candidates import TransportAssumptions, LN10

NAMES = ('band_amplitude', 'two_band_fixed', 'two_band_conserved',
         'two_band_free', 'two_level_free')


@lru_cache(None)
def quadrature(nodes):
    x, w = np.polynomial.legendre.leggauss(nodes)
    return x, np.log(w / 2)


def band_log10_shape(voltage, parameters, assumptions=TransportAssumptions(), nodes=64):
    """Uniform-band integral of f_L-f_R, relative to positive reference bias."""
    v = np.asarray(voltage, float)
    p = np.asarray(parameters, float)
    if p.shape != (3,) or not np.all(np.isfinite(p)):
        raise ValueError('Expected finite E0, width, lever arm')
    energy, width, eta = p
    if not 0 <= energy <= 1 or not 0 <= width <= 1 or not .01 <= eta <= .99:
        raise ValueError('Band parameters outside the declared search domain')
    if np.any(~np.isfinite(v)) or np.any(v == 0):
        raise ValueError('Finite nonzero voltage required')
    if nodes < 8:
        raise ValueError('At least eight quadrature nodes required')
    x, lw = quadrature(nodes)
    energies = energy + width * x / 2
    vt = assumptions.thermal_voltage_v

    def raw(values):
        a = (energies - eta * values[..., None]) / vt
        b = (energies + (1-eta) * values[..., None]) / vt
        # Positive magnitude of the exact Fermi difference; no cancellation
        # of nearly equal occupancies in far-off-resonance tails.
        logs = (np.maximum(a, b) + np.log(-np.expm1(-np.abs(values[..., None])/vt))
                - np.logaddexp(0., a) - np.logaddexp(0., b))
        return logsumexp(logs + lw, axis=-1)

    return (raw(v)-raw(np.array(assumptions.reference_voltage_v))) / LN10


def global_domain(name):
    if name not in NAMES:
        raise ValueError(name)
    if name == 'band_amplitude':
        return np.array([0., 0., .01]), np.array([1., 1., .99])
    if name == 'two_level_free':
        return np.array([0., .01, 0., .01]), np.array([1., .99, 1., .99])
    lo, hi = [0., 0., .01] * 2, [1., 1., .99] * 2
    if name == 'two_band_fixed':
        lo += [-20.]; hi += [20.]
    elif name == 'two_band_conserved':
        lo += [-18., -18.]; hi += [-2., -2.]
    return np.array(lo), np.array(hi)


def state_domain(name):
    if name in ('band_amplitude', 'two_band_fixed'):
        return np.array([-18.]), np.array([-2.])
    if name == 'two_band_conserved':
        return np.array([-20.]), np.array([20.])
    if name in ('two_band_free', 'two_level_free'):
        return np.array([-18., -20.]), np.array([-2., 20.])
    raise ValueError(name)


def component_shapes(v, name, parameters, nodes=64):
    p = np.asarray(parameters)
    if name == 'two_level_free':
        return (band_log10_shape(v, [p[0], 0., p[1]], nodes=nodes),
                band_log10_shape(v, [p[2], 0., p[3]], nodes=nodes))
    first = band_log10_shape(v, p[:3], nodes=nodes)
    return first, None if name == 'band_amplitude' else band_log10_shape(v, p[3:6], nodes=nodes)


def _evaluate(shapes, name, parameters, states):
    s1, s2 = shapes
    q = np.asarray(states, float)
    if q.ndim != 2 or q.shape[1] != len(state_domain(name)[0]):
        raise ValueError('Invalid state array')
    if name == 'band_amplitude':
        pred = q[:, :1] + s1
        return pred, np.ones((*pred.shape, 1))
    if name == 'two_band_fixed':
        t = np.full((len(q), 1), parameters[-1]); amplitude = q[:, :1]
    elif name == 'two_band_conserved':
        t = q[:, :1]; amplitude = 0.
        s1 = s1 + parameters[-2]; s2 = s2 + parameters[-1]
    else:
        amplitude = q[:, :1]; t = q[:, 1:2]
    z = expit(t)
    l1, l2 = LN10*s1 - np.logaddexp(0., t), LN10*s2 - np.logaddexp(0., -t)
    pred = amplitude + np.logaddexp(l1, l2)/LN10
    deriv = (expit(l2-l1)-z)/LN10
    if name == 'two_band_conserved':
        local = deriv[..., None]
    elif name == 'two_band_fixed':
        local = np.ones((*pred.shape, 1))
    else:
        local = np.stack((np.ones_like(pred), deriv), axis=-1)
    return pred, local


def predict(voltage, name, parameters, states, nodes=64):
    lo, hi = global_domain(name)
    p = np.asarray(parameters, float); q = np.atleast_2d(states)
    qlo, qhi = state_domain(name)
    if p.shape != lo.shape or np.any(~np.isfinite(p)) or np.any(p < lo) or np.any(p > hi):
        raise ValueError('Invalid shared parameters')
    if np.any(~np.isfinite(q)) or np.any(q < qlo) or np.any(q > qhi):
        raise ValueError('Invalid local state')
    return _evaluate(component_shapes(voltage, name, p, nodes), name, p, q)[0]


def current(voltage, name, parameters, state, nodes=64):
    v = np.asarray(voltage, float)
    if np.any(~np.isfinite(v)):
        raise ValueError('Finite voltage required')
    ans = np.zeros_like(v); mask = v != 0
    if np.any(mask):
        logs = predict(v[mask], name, parameters, state, nodes)[0]
        if np.any(logs > np.log10(np.finfo(float).max)):
            raise OverflowError('Current overflow; no clipping')
        ans[mask] = np.sign(v[mask])*10**logs
    return float(ans) if ans.ndim == 0 else ans


def starts(name, y, weights):
    a = np.sum(y * weights, axis=1)
    bands = [([.10, .12, .10], [.20, .30, .80]),
             ([.20, .35, .08], [.08, .05, .70]),
             ([.07, .03, .75], [.35, .50, .15]),
             ([.15, .25, .35], [.40, .70, .90])]
    for k, (b1, b2) in enumerate(bands):
        if name == 'band_amplitude':
            p = b1; q = a[:, None]
        elif name == 'two_level_free':
            p = [b1[0], b1[2], b2[0], b2[2]]
            q = np.column_stack((a, np.full(len(y), (-2., 2., -5., 5.)[k])))
        else:
            p = b1 + b2
            if name == 'two_band_fixed':
                p += [(-2., 2., -5., 5.)[k]]; q = a[:, None]
            elif name == 'two_band_conserved':
                p += [float(np.median(a)-.5), float(np.median(a)+.5)]
                q = np.linspace(-2., 2., len(y))[:, None] * (1 if k % 2 else -1)
            else:
                q = np.column_stack((a, np.full(len(y), (-2., 2., -5., 5.)[k])))
        yield np.array(p), q


def fit_shared(voltage, y, weights, cycle_ids, name, max_nfev=600, nstarts=4):
    """Joint training fit with sparse local-state and global-shape Jacobian."""
    v, y, w = np.asarray(voltage), np.asarray(y), np.asarray(weights)
    ids = list(cycle_ids); n, m = y.shape
    if n < 2 or len(ids) != n or len(set(ids)) != n or m != len(v):
        raise ValueError('Distinct training pairs and common grid required')
    w = np.broadcast_to(w, y.shape)
    if not np.all(np.isfinite(y)) or np.any(w < 0) or not np.allclose(w.sum(axis=1), 1):
        raise ValueError('Finite observations and normalized nonnegative row weights required')
    glo, ghi = global_domain(name); slo, shi = state_domain(name)
    ng, ns = len(glo), len(slo)
    lo = np.r_[glo, np.tile(slo, n)]; hi = np.r_[ghi, np.tile(shi, n)]
    sw = np.sqrt(w / n); local_rows = np.repeat(np.arange(n*m), ns)
    local_cols = np.repeat(np.arange(n), m*ns)*ns + np.tile(np.arange(ns), n*m)

    def values(x):
        return _evaluate(component_shapes(v, name, x[:ng]), name, x[:ng], x[ng:].reshape(n, ns))

    def residual(x):
        return ((values(x)[0] - y)*sw).ravel()

    def jacobian(x):
        base, local = values(x)
        global_jac = []
        for j in range(ng):
            delta = 1e-6*(ghi[j]-glo[j])
            if x[j]+delta > ghi[j]:
                delta = -delta
            other = x.copy(); other[j] += delta
            global_jac.append(((values(other)[0]-base)/delta*sw).ravel())
        lj = csr_matrix(((local*sw[:, :, None]).ravel(), (local_rows, local_cols)), shape=(n*m, n*ns))
        return hstack((csr_matrix(np.array(global_jac).T), lj), format='csr')

    attempts = []
    for k, (p, q) in enumerate(starts(name, y, w)):
        if k >= nstarts:break
        initial = np.clip(np.r_[p, q.ravel()], lo+1e-8, hi-1e-8)
        f = least_squares(residual, initial, jac=jacobian, bounds=(lo, hi),
                          tr_solver='lsmr', x_scale=np.r_[ghi-glo, np.tile(shi-slo, n)],
                          max_nfev=max_nfev, ftol=2e-7, xtol=2e-7, gtol=2e-7,
                          tr_options={'atol': 1e-6, 'btol': 1e-6, 'maxiter': 300})
        attempts.append(f)
    good = [f for f in attempts if f.success and np.isfinite(f.cost)]
    # Preserve capped runs for inspection; a capped best run cannot silently
    # become a claimed converged physical optimum.
    best = min(good or attempts, key=lambda f: f.cost)
    return {'name': name, 'parameters': best.x[:ng].tolist(),
            'states': best.x[ng:].reshape(n, ns).tolist(), 'training_cycle_ids': ids,
            'converged': bool(best.success), 'rmse_decade': float(np.sqrt(2*best.cost)),
            'shared_parameter_at_boundary': bool(np.any(np.minimum(best.x[:ng]-glo, ghi-best.x[:ng])/(ghi-glo)<1e-5)),
            'multistart': [{'success': bool(f.success), 'nfev': int(f.nfev), 'optimality': float(f.optimality),
                           'rmse_decade': float(np.sqrt(2*f.cost)), 'parameters': f.x[:ng].tolist()}
                          for f in attempts]}


def identify_state(voltage, observed, weights, fit, target_cycle_id=None):
    """Identify state from ONLY the explicitly provided observations.

    Call with positive-only coordinates for a RESET forecast. The function
    takes no target negative currents and rejects targets in the shared fit.
    """
    if target_cycle_id is not None and target_cycle_id in fit['training_cycle_ids']:
        raise ValueError('Target included in shared-shape training')
    name = fit['name']; p = np.array(fit['parameters']); v = np.asarray(voltage)
    y, w = np.asarray(observed), np.asarray(weights)
    w = w/w.sum(); sw = np.sqrt(w)
    slo, shi = state_domain(name); shapes = component_shapes(v, name, p)
    def residual(q):return (_evaluate(shapes, name, p, q[None, :])[0][0]-y)*sw
    def jac(q):return _evaluate(shapes, name, p, q[None, :])[1][0]*sw[:, None]
    attempts = []
    amp = float(np.clip(np.mean(y), -17.9, -2.1))
    for t in (-10., -2., 2., 10.):
        q = [t] if name == 'two_band_conserved' else [amp] if len(slo) == 1 else [amp, t]
        attempts.append(least_squares(residual, q, jac=jac, bounds=(slo, shi), max_nfev=250,
                                     ftol=1e-10, xtol=1e-10, gtol=1e-10))
    f = min(attempts, key=lambda f:f.cost)
    singular = np.linalg.svd(f.jac, compute_uv=False)
    return {'state': f.x.tolist(), 'converged': bool(f.success), 'fit_rmse_decade': float(np.sqrt(2*f.cost)),
            'state_jacobian_singular_values': singular.tolist(),
            'at_boundary': bool(np.any(np.minimum(f.x-slo, shi-f.x)/(shi-slo)<1e-5))}
