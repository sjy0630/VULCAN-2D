"""Research irreversible RESET survival laws with explicit scan history.

Common HRS/connected transport shapes are effective nuisance envelopes. This
module does not identify PF dielectric parameters or a full SET/RESET state.
Rate clocks use voltage-path length; a time interpretation requires constant
sweep speed and known dwell/gap times, which are absent from these records.
"""
import numpy as np
from scipy.special import exprel
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix, hstack
from .transport_candidates import TransportAssumptions, LN10

NAMES = ('field_weibull', 'thermal_rate', 'power_damage')
REFERENCE = .2
POWER_REFERENCE = 1e-6
LN2 = np.log(2.)


def domains(name):
    if name not in NAMES:raise ValueError(name)
    glo, ghi = [0., .05], [15., 20.]
    if name == 'field_weibull':glo += [.25]; ghi += [12.]
    if name == 'thermal_rate':glo += [.1]; ghi += [1/(2*TransportAssumptions().thermal_voltage_v)]
    scale = (-1.7, .5) if name == 'field_weibull' else (-12., 3.) if name == 'thermal_rate' else (-8., 3.)
    return np.array(glo), np.array(ghi), np.array([-14., -14., scale[0]]), np.array([-5., -3., scale[1]])


def log_shapes(v, shared):
    v = np.asarray(v, float)
    if np.any(~np.isfinite(v)) or np.any(v < 0):raise ValueError('Finite nonnegative internal voltage required')
    beta, alpha = shared[:2]; u = np.where(v > 0, v, REFERENCE)
    h = np.log(u/REFERENCE)+beta*(np.sqrt(u)-np.sqrt(REFERENCE))
    def logsinh(z):return z+np.log(-np.expm1(-2*z))-LN2
    c = logsinh(alpha*u)-logsinh(alpha*REFERENCE)
    return np.where(v > 0, h, -np.inf), np.where(v > 0, c, -np.inf)


def thermal_clock(vapp, vh, gamma):
    """Exact integral for piecewise linear internal voltage vs applied path."""
    ds = np.abs(np.diff(vapp, axis=-1)); dv = np.diff(vh, axis=-1)
    increments = ds*np.exp(gamma*(vh[..., :-1]-.5))*exprel(gamma*dv)
    return np.concatenate((np.zeros((*vh.shape[:-1], 1)), np.cumsum(increments, axis=-1)), axis=-1)


def power_step(logp, v, ds, shared, local):
    """Exact positive solution dp/ds=-k*v*(H+C*p)*p/P_ref at fixed v."""
    lh, lc = log_shapes(v, shared)
    lh = lh+LN10*local[:, 0]; lc = lc+LN10*local[:, 1]
    k = LN2*10**(-local[:, 2])
    z = k*v*np.exp(lh)/POWER_REFERENCE*ds
    active = (v > 0) & (ds > 0)
    loss = np.zeros_like(logp)
    # This log form stays stable for tiny H and for very large C/H.
    loss[active] = np.logaddexp(0., logp[active]+lc[active]-lh[active]
                               +np.log(-np.expm1(-z[active])))
    return logp-z-loss


def predict_internal(vapp, vh, name, shared, local, return_state=False):
    vapp, vh = np.asarray(vapp, float), np.asarray(vh, float)
    local = np.atleast_2d(local)
    if vapp.shape != vh.shape or vh.ndim != 2 or len(vh) != len(local):
        raise ValueError('Cycle-by-point arrays and one local parameter vector per cycle required')
    if name == 'field_weibull':
        clock = np.maximum.accumulate(vh, axis=1)
        logp = -LN2*(clock/10**local[:, 2:3])**shared[2]
    elif name == 'thermal_rate':
        clock = thermal_clock(vapp, vh, shared[2])
        logp = -LN2*clock/10**local[:, 2:3]
    elif name == 'power_damage':
        logp = np.zeros_like(vh)
        for j in range(1, vh.shape[1]):
            logp[:, j] = power_step(logp[:, j-1], (vh[:, j-1]+vh[:, j])/2,
                                     np.abs(vapp[:, j]-vapp[:, j-1]), shared, local)
    else:raise ValueError(name)
    h, c = log_shapes(vh, shared)
    out = np.logaddexp(LN10*local[:, 0:1]+h, LN10*local[:, 1:2]+c+logp)/LN10
    return (out, logp) if return_state else out


def initial_local(vapp, vh, y, w, name, shared, scale_voltage=.6):
    n = len(y); result = []
    h, c = log_shapes(vh, shared)
    clocks = thermal_clock(vapp, vh, shared[2]) if name == 'thermal_rate' else None
    for k in range(n):
        apex = int(np.argmax(vapp[k])); indices = np.arange(len(y[k])); valid = w[k] > 0
        tail = valid & (indices > apex) & (vh[k] >= .12) & (vh[k] <= .4)
        if sum(tail) < 3:tail = valid & (vh[k] >= .8) & (vh[k] <= 1.3)
        if not np.any(tail):tail = valid
        ah = float(np.mean(y[k, tail]-h[k, tail]/LN10))
        low = valid & (indices <= apex) & (vh[k] >= .08) & (vh[k] <= .3)
        residual = 10**y[k, low]-10**ah*np.exp(h[k, low])
        ac = float(np.mean(np.log10(np.maximum(residual, 1e-14))-c[k, low]/LN10))
        idx = int(np.argmin(np.abs(vh[k, :apex+1]-scale_voltage)))
        if name == 'field_weibull':scale = scale_voltage
        elif name == 'thermal_rate':scale = max(clocks[k, idx], 1e-10)
        else:
            vv = vh[k, :idx+1]; va = vapp[k, :idx+1]
            # Initialization only, using available training/conditioning data.
            scale = max(float(np.trapezoid(vv*10**y[k, :idx+1]/POWER_REFERENCE, va)), 1e-7)
        result.append([ah, ac, np.log10(scale)])
    return np.array(result)


def fit_history(vapp, vh, y, weights, cycle_ids, name, max_nfev=400, nstarts=4):
    vapp, vh, y, w = map(lambda a:np.asarray(a, float), (vapp, vh, y, weights))
    if y.shape != vh.shape or y.shape != vapp.shape or y.shape != w.shape:
        raise ValueError('Matching cycle-by-point arrays required')
    if np.any(~np.isfinite(y)) or np.any(w < 0) or not np.allclose(w.sum(axis=1), 1):
        raise ValueError('Finite currents and normalized nonnegative weights required')
    n, m = y.shape; glo, ghi, slo, shi = domains(name); ng = len(glo)
    if len(set(cycle_ids)) != n:raise ValueError('Distinct training cycle ids required')
    lo, hi = np.r_[glo, np.tile(slo, n)], np.r_[ghi, np.tile(shi, n)]
    sw = np.sqrt(w/n)
    rows = np.repeat(np.arange(n*m), 3)
    cols = np.repeat(np.arange(n), m*3)*3+np.tile(np.arange(3), n*m)
    def values(x):
        ans = predict_internal(vapp, vh, name, x[:ng], x[ng:].reshape(n, 3))
        return np.where(w > 0, ans, 0.)
    target = np.where(w > 0, y, 0.)
    def residual(x):return ((values(x)-target)*sw).ravel()
    def jacobian(x):
        base = values(x); gj = []
        for j in range(ng):
            dx = 1e-6*(ghi[j]-glo[j]); dx = -dx if x[j]+dx > ghi[j] else dx
            other = x.copy(); other[j] += dx
            gj.append(((values(other)-base)/dx*sw).ravel())
        lj = np.empty((n, m, 3))
        q = x[ng:].reshape(n, 3)
        for j in range(3):
            dx = np.where(q[:, j]+1e-5 > shi[j], -1e-5, 1e-5)
            other = x.copy(); other[ng:].reshape(n, 3)[:, j] += dx
            lj[:, :, j] = (values(other)-base)/dx[:, None]*sw
        return hstack((csr_matrix(np.array(gj).T), csr_matrix((lj.ravel(), (rows, cols)), shape=(n*m, n*3))), format='csr')
    guesses = [(6., 4., 3.), (5., 8., 6.), (8., 2., 10.), (4., 12., 1.)]
    attempts = []
    for k, (beta, alpha, kinetic) in enumerate(guesses[:nstarts]):
        p = np.array([beta, alpha] if name == 'power_damage' else [beta, alpha, kinetic])
        q = initial_local(vapp, vh, y, w, name, p, (.6, .5, .7, .8)[k])
        x0 = np.clip(np.r_[p, q.ravel()], lo+1e-7, hi-1e-7)
        f = least_squares(residual, x0, jac=jacobian, bounds=(lo, hi), x_scale=hi-lo,
                          tr_solver='lsmr', tr_options={'atol': 1e-6, 'btol': 1e-6, 'maxiter': 300},
                          max_nfev=max_nfev, ftol=2e-7, gtol=2e-7, xtol=2e-7)
        attempts.append(f)
    good = [f for f in attempts if f.success and np.isfinite(f.cost)]
    best = min(good or attempts, key=lambda f:f.cost)
    return {'name': name, 'shared_parameters': best.x[:ng].tolist(), 'local_parameters': best.x[ng:].reshape(n, 3).tolist(),
            'training_cycle_ids': list(cycle_ids), 'converged': bool(best.success), 'rmse_decade': float(np.sqrt(2*best.cost)),
            'shared_parameter_at_boundary': bool(np.any(np.minimum(best.x[:ng]-glo, ghi-best.x[:ng])/(ghi-glo)<1e-5)),
            'multistart': [{'success': bool(f.success), 'nfev': int(f.nfev), 'rmse_decade': float(np.sqrt(2*f.cost)),
                           'shared_parameters': f.x[:ng].tolist(), 'optimality': float(f.optimality)} for f in attempts]}


def identify_from_outward(vapp, vh, y, weights, fit, target_id):
    if target_id in fit['training_cycle_ids']:raise ValueError('Target in shared-parameter training')
    apex = int(np.argmax(vapp)); cut = slice(0, apex+1)
    # Discard target return arrays before guesses, fitting or diagnostics.
    va, v, obs, w = [np.asarray(a)[cut][None, :] for a in (vapp, vh, y, weights)]
    w = w/w.sum(); sw = np.sqrt(w); name = fit['name']; p = np.array(fit['shared_parameters'])
    _, _, lo, hi = domains(name)
    def residual(q):
        pred = predict_internal(va, v, name, p, q[None, :])
        return (np.where(w > 0, pred-obs, 0.)*sw).ravel()
    attempts = []
    for scale in (.35, .55, .75, 1.):
        q = np.clip(initial_local(va, v, obs, w, name, p, scale)[0], lo+1e-7, hi-1e-7)
        attempts.append(least_squares(residual, q, bounds=(lo, hi), max_nfev=300,
                                     ftol=1e-9, xtol=1e-9, gtol=1e-9))
    good = [f for f in attempts if f.success]; f = min(good or attempts, key=lambda f:f.cost)
    return {'local_parameters': f.x.tolist(), 'converged': bool(f.success), 'outward_fit_rmse_decade': float(np.sqrt(2*f.cost)),
            'scaled_jacobian_singular_values': np.linalg.svd(f.jac*(hi-lo)[None, :], compute_uv=False).tolist(),
            'parameter_at_boundary': bool(np.any(np.minimum(f.x-lo, hi-f.x)/(hi-lo)<1e-5))}


def terminal_trajectory(vapp, name, shared, local, transistor_voltage, transistor_current, substeps=4):
    """Forward load-line/irreversible-state trajectory; no measured device I.

    Rate states use midpoint load line per substep. Athermal max-field damage
    relaxes monotonically at each new applied voltage, permitting avalanches.
    Every final stored current solves a frozen-state monotone load line.
    """
    va = np.asarray(vapp, float); local = np.atleast_2d(local)
    if va.ndim != 2 or len(local) != len(va) or substeps < 1:raise ValueError('Invalid trajectory inputs')
    n, m = va.shape; currents = np.zeros_like(va); internal = np.zeros_like(va); states = np.zeros_like(va)
    logp = np.zeros(n); maximum = np.zeros(n); largest_relaxation_steps = 0; unconverged = 0
    def load(applied, lp):
        lo, hi = np.zeros(n), applied.copy()
        for unused in range(42):
            mid = (lo+hi)/2; h, c = log_shapes(mid, shared)
            ih = np.exp(np.logaddexp(LN10*local[:, 0]+h, LN10*local[:, 1]+c+lp))
            it = np.interp(applied-mid, transistor_voltage, transistor_current)
            less = ih < it; lo = np.where(less, mid, lo); hi = np.where(less, hi, mid)
        v = (lo+hi)/2; h, c = log_shapes(v, shared)
        current = np.exp(np.logaddexp(LN10*local[:, 0]+h, LN10*local[:, 1]+c+lp))
        return v, current
    for j in range(m):
        prior = np.zeros(n) if j == 0 else va[:, j-1]
        ds = np.abs(va[:, j]-prior)/substeps
        for k in range(substeps):
            endpoint = prior+(va[:, j]-prior)*(k+1)/substeps
            midpoint = prior+(va[:, j]-prior)*(k+.5)/substeps
            if name == 'field_weibull':
                settled = False
                for iteration in range(80):
                    v, unused = load(endpoint, logp)
                    maximum = np.maximum(maximum, v)
                    updated = -LN2*(maximum/10**local[:, 2])**shared[2]
                    # A tiny surviving fraction may multiply a huge connected
                    # envelope. Absolute |delta p| can falsely declare
                    # convergence while the current still changes materially.
                    settled = bool(np.max(np.abs(updated-logp)) < 1e-9)
                    logp = updated
                    if settled:break
                largest_relaxation_steps = max(largest_relaxation_steps, iteration+1)
                unconverged += not settled
            else:
                v, unused = load(midpoint, logp)
                # Predict half-step state, then re-solve the midpoint load
                # line. Freezing the OLD state at the midpoint voltage is
                # only first order when current loss changes voltage sharing.
                if name == 'thermal_rate':
                    half = logp-LN2*ds/2*np.exp(shared[2]*(v-.5))/10**local[:, 2]
                    v, unused = load(midpoint, half)
                    logp -= LN2*ds*np.exp(shared[2]*(v-.5))/10**local[:, 2]
                else:
                    half = power_step(logp, v, ds/2, shared, local)
                    v, unused = load(midpoint, half)
                    logp = power_step(logp, v, ds, shared, local)
        v, current = load(va[:, j], logp)
        internal[:, j], currents[:, j], states[:, j] = v, current, logp
    transistor = np.interp(va-internal, transistor_voltage, transistor_current)
    scale = np.maximum(np.maximum(currents, transistor), 1e-20)
    return {'current_a': currents, 'internal_voltage_v': internal, 'log_survival': states,
            'max_kcl_relative_error': float(np.max(np.abs(currents-transistor)/scale)),
            'largest_athermal_relaxation_steps': largest_relaxation_steps, 'unconverged_athermal_steps': int(unconverged)}
