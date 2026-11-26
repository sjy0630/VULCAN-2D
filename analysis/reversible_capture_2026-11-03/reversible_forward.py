"""Prepared one-seed reversible SET forward probe; no implicit rate selection.

Run only after the parent has reviewed reversible_budget.json and written an
explicit fixed-rate plan JSON. This is internal experiment accounting.  This is a conditional cycle-4 SET experiment,
not a complete or independently validated common 1R/1T1R model.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import signal
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CACHE = Path('/private/tmp/vulcan-runtime-cache')
PREVIOUS = ROOT / 'analysis/reset_shape_2026-11-03'
BUDGET_STATUS = 'completed_one_seed_reversible_rate_budget_no_ODE_or_optimization'
PRECISIONS = (
    dict(name='nominal', rtol=1e-7, atol=1e-8, max_step=.02),
    dict(name='refined', rtol=1e-8, atol=1e-9, max_step=.01),
)


class NumericalDomainError(ValueError):
    """The finite numerical interface cannot evaluate this state or voltage."""


class NumericalUnknown(ValueError):
    """The requested positive local voltage is below the existing table."""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj, *, exclusive=False):
    with Path(path).open('x' if exclusive else 'w') as handle:
        json.dump(obj, handle, indent=2, allow_nan=False)
        handle.write('\n')


def load_fixed_rate_plan(path, budget_path, budget):
    """A caller must supply both the exact reviewed budget and an explicit fixed rate."""
    fixed_plan = json.loads(Path(path).read_text())
    if fixed_plan.get('budget_sha256') != sha(budget_path):
        raise ValueError('Fixed-rate plan must identify the exact completed budget hash')
    if budget['status'] != BUDGET_STATUS:
        raise ValueError('The necessary reversible budget must have completed first')
    kappa = fixed_plan.get('kappa_per_v2')
    if isinstance(kappa, bool) or not isinstance(kappa, (int, float)) or not math.isfinite(kappa) or kappa <= 0:
        raise ValueError('Parent must explicitly choose a positive finite kappa_per_v2')
    for key in ('selection_rule', 'decision_reference'):
        if not isinstance(fixed_plan.get(key), str) or not fixed_plan[key].strip():
            raise ValueError(f'Parent must supply {key}; there is no default rate rule')
    return fixed_plan, float(kappa)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixed-rate-plan', type=Path, required=True,
                        help='Parent-written JSON containing budget_sha256, '
                             'kappa_per_v2, selection_rule, decision_reference')
    args = parser.parse_args()
    budget_path = HERE / 'reversible_budget.json'
    budget = json.loads(budget_path.read_text())
    fixed_rate_plan, kappa = load_fixed_rate_plan(args.fixed_rate_plan, budget_path, budget)
    # Imports and transport loading occur only after reading the fixed-rate plan.
    sys.dont_write_bytecode = True  # Importing frozen modules must not create old-stage pyc files.
    import numpy as np
    from scipy.integrate import solve_ivp
    from scipy.optimize import brentq
    sys.path[:0] = [str(ROOT), str(ROOT / 'analysis/mechanism_selection_2026-11-03')]
    from SOFT_FEEDBACK_LOW_BIAS_BUDGET import cached_transport

    base_path = PREVIOUS / 'soft_field_mobility_probe.json'
    base = json.loads(base_path.read_text())
    old_budget_path = PREVIOUS / 'soft_beta_relaxed_budget.json'
    old_budget = json.loads(old_budget_path.read_text())
    seed = budget['plan']['seed']
    lam, js, ron, gc = [float(seed[k]) for k in
                       ('lambda_per_v', 'J_positive_a', 'Ron_ohm', 'gamma_c_ev')]
    if seed['parallel_path_a_s'] != 0. or lam < 0 or min(js, ron, gc) <= 0:
        raise ValueError('This probe supports only the declared soft-only positive seed')
    if any(not math.isfinite(x) for x in (lam, js, ron, gc)):
        raise ValueError('Invalid seed')
    anchor = float(old_budget['anchors_u']['initial_0p2'])
    nominal_budget = next(row for row in budget['cases']
                          if row['material_envelope'] == 0. and row['grid']['name'] == 'refined')
    if abs(math.exp(-anchor) - nominal_budget['first_passage']['q0_optimistic_upper']) > 1e-16:
        raise ValueError('Nominal saved anchor and budget initial state differ')

    t, lookups = cached_transport(CACHE)
    for path in (t.table_path, *lookups):
        if sha(path) != budget['source_sha256'][str(path)]:
            raise ValueError(f'Budget source changed: {path}')
    with np.load(base_path.with_suffix('.npz')) as archive:
        v = archive['command_voltage_v'].copy()
        s = archive['path_voltage_s'].copy()
        signed_obs = archive['observed_current_a'].copy()
        mask = archive['original_scoring_mask'].copy()
    obs = abs(signed_obs)
    out = np.arange(len(v)) <= np.argmax(v)
    np.testing.assert_array_equal(mask, np.isfinite(obs) & (obs >= 1e-11) & (v > 1e-12))
    if (len(v), int(mask.sum()), int((mask & out).sum()), int((mask & ~out).sum())) != (503, 500, 250, 250):
        raise ValueError('Native cycle-4 sample identity/count changed')
    np.testing.assert_allclose(s[out], v[out], atol=1e-12, rtol=0)
    np.testing.assert_allclose(s[~out], 10.-v[~out], atol=1e-12, rtol=0)
    anchor_refs_path = CACHE / 'RON_SCALING_REVIEW.json'
    anchors = next(row for row in json.loads(anchor_refs_path.read_text())['rows']
                   if row['Ron_ohm'] == ron)['anchor_reinversions']
    feature_ref = next(row for row in base['runs']
                       if row['precision']['name'] == 'refined' and 'metrics' in row)['reference_features']
    frozen_sources = [base_path, base_path.with_suffix('.npz'), old_budget_path,
                      anchor_refs_path, CACHE / 'RON_SCALING_REVIEW.npz', t.table_path, *lookups,
                      ROOT / 'analysis/soft_drive_structure_2026-11-03/sinh_field_probe.py',
                      ROOT / 'analysis/soft_drive_structure_2026-11-03/sinh_field_probe.json',
                      ROOT / 'analysis/soft_drive_structure_2026-11-03/sinh_field_probe.npz',
                      ROOT / 'analysis/mechanism_selection_2026-11-03/SOFT_FEEDBACK_LOW_BIAS_BUDGET.py',
                      ROOT / 'analysis/ron_scope_2026-11-03/ron_scaling_review.py',
                      ROOT / 'vulcan2d/transistor.py']
    frozen = {str(path): sha(path) for path in frozen_sources}
    plan = dict(
        scope='One two-point-calibrated, single-cycle conditional soft SET probe. '
              'No RESET, bare history, holdout, 32-cycle aggregate, or full unified-model validation.',
        seed=seed, kappa_per_v2=kappa, kappa_selection_rule=fixed_rate_plan['selection_rule'],
        fixed_rate_plan=fixed_rate_plan, budget_sha256=sha(budget_path),
        state_law='q=exp(-u); Gamma=gc/expm1(u); '
                  'du/ds=kappa*v*(exp(lambda*v)-I*expm1(u)/J)',
        equivalent_q_law='dq/ds=kappa*v*((I/J)*(1-q)-exp(lambda*v)*q)',
        signs='Both signs are retained; no positive-part operator or monotone-state assertion.',
        physical_change='Reversible q dynamics is a new physical hypothesis, not just a coordinate change.',
        path_clock='s is cumulative command-voltage path, not known physical time.',
        initialization='Condition on saved outward .2V state; reverse-integrate same signed ODE .2->0; '
                       'then integrate complete outward 0->5 and return s=5->10 with A=10-s.',
        anchor_command_v=.2, anchor_u=anchor, anchor_q=math.exp(-anchor),
        anchor_gamma_ev=old_budget['anchors_gamma_ev']['initial_0p2'],
        apex='Continuous state; duplicated native endpoints add no dwell.',
        domain='0<u<=80 and actual Gamma inside original table; u=0 is a physical endpoint '
               'outside this finite-table interface. No clipping, projection, or extrapolated current.',
        precisions=list(PRECISIONS), method='DOP853', max_physical_seeds=1,
        max_solver_attempts=2, max_ODE_segments=6,
        zero_command='At A=0 exact I=0 and u_prime=0, after validating state/Gamma.',
        smallfield_bound='For 0<A<vmin: |u_prime|<=kappa*A*'
                         '(exp(lambda*A)+pi*gc/(Ron*J)); '
                         'B=.5*kappa*vmin^2*(exp(lambda*vmin)+pi*gc/(Ron*J)). '
                         'Only B<1e-12 and a valid state/Gamma allow RHS omission.',
        smallfield_scope='Bounded numerical omission, never exact inactivity. '
                         'Count physical monotone microsegments, not RHS retries; '
                         'local integral bound is not a full-trajectory error certificate.',
        circuit_comparison='36 fixed command/state pairs versus original42 bisection; '
                           'fallback to legacy on comparison failure.',
        fast_root_domain='Solve transistor drop on [0,min(Tmax,A-vmin)] with inward '
                         'floating endpoint adjustment only when subtraction places local voltage '
                         'below vmin. No current evaluation in 0<v<vmin.',
        circuit_tolerances=dict(max_log_current_difference_decade=5e-7,
                                max_local_voltage_difference_v=1e-9, max_KCL_residual_a=1e-13),
        scoring='All original500 native eligible points, including conditioned anchor and low currents; '
                'no floor, baseline subtraction, or point exclusion.',
        group_names=['all500', 'outward250', 'return250', 'outward_low10',
                     'both_branches_low20', 'remaining480', 'outward0p5_to1p5_50'],
        outward_current_thresholds_a=[1e-6, 1e-5],
        reference_scope='Three old endpoint checks and all six 32-record median reference bands; '
                        'a single-cycle comparison is not an aggregate gate.',
        calibration_scope='Saved return2/4 observations define lambda/J. They are calibration points, '
                          'not independent predictions; saved4V equilibrium is repelling,2V attracting.',
        compute_budget_s=120,
        budget_start='After input loads, source hashes and plan write, before circuit checks and ODE.',
        stopping_policy='Preserve every failure; no new seed, rate adjustment, scan, fit, '
                        'domain expansion, solver recovery or automatic rerun.')
    plan_path = HERE / 'reversible_forward.plan.json'
    dest = HERE / 'reversible_forward.json'
    if any(path.exists() for path in (plan_path, dest, dest.with_suffix('.npz'), dest.with_suffix('.md'))):
        raise FileExistsError('An existing forward attempt must be preserved; refusing overwrite/rerun')
    dump(plan_path, plan, exclusive=True)
    sources = dict(frozen)
    sources.update({str(path): sha(path) for path in
                    (budget_path, args.fixed_rate_plan.resolve(), plan_path, Path(__file__))})
    report = dict(status='running_one_predeclared_reversible_forward', plan=plan,
                  source_sha256=sources, frozen_source_sha256=frozen, runs=[])
    dump(dest, report, exclusive=True)
    arrays = dict(command_voltage_v=v, path_voltage_s=s, observed_current_a=signed_obs,
                  original_scoring_mask=mask)
    lg, lv, li = t.lg, t.lv, t.li[0]
    tx, ti = t.knots['set']
    vmin = math.exp(lv[0])
    log_scale = math.log(t.r0/ron)

    def gamma(u):
        if not math.isfinite(u) or not 0 < u <= 80:
            raise NumericalDomainError(f'State outside finite interface 0<u<=80: {u!r}')
        g = gc / math.expm1(u)
        if not lg[0] <= math.log(g) <= lg[-1]:
            raise NumericalDomainError(f'Gamma outside existing table: {g!r}')
        return g

    def material(voltage, log_gamma):
        if voltage == 0:
            return 0.
        if voltage < 0 or math.log(voltage) > lv[-1]:
            raise NumericalDomainError('Material voltage outside existing table')
        y = math.log(voltage)
        if y < lv[0]:
            raise NumericalUnknown('Positive local voltage below table resolution')
        ix = min(max(int(np.searchsorted(lg, log_gamma)-1), 0), len(lg)-2)
        iy = min(max(int(np.searchsorted(lv, y)-1), 0), len(lv)-2)
        a = (log_gamma-lg[ix])/(lg[ix+1]-lg[ix])
        b = (y-lv[iy])/(lv[iy+1]-lv[iy])
        return math.exp((1-a)*((1-b)*li[ix, iy]+b*li[ix, iy+1])
                        +a*((1-b)*li[ix+1, iy]+b*li[ix+1, iy+1])+log_scale)

    def fast(command, u):
        log_gamma = math.log(gamma(float(u)))
        command = float(command)
        if command == 0:
            return 0., 0., 0.
        if command < 0 or math.log(command) > lv[-1]:
            raise NumericalDomainError('Command outside material table domain')
        if math.log(command) < lv[0]:
            raise NumericalUnknown('Positive command below table resolution')
        def residual(transistor_drop):
            return float(np.interp(transistor_drop, tx, ti))-material(command-transistor_drop, log_gamma)
        # This shrinks only the root-search interval; it does not clip a state,
        # command, or solved voltage and never evaluates an invented subtable I.
        material_top = command-vmin
        material_limited = material_top <= float(tx[-1])
        top = min(material_top, float(tx[-1]))
        if top > 0. and command-top < vmin:
            top = math.nextafter(top, -math.inf)
        low_residual = residual(0.)
        if low_residual > 0.:
            raise NumericalDomainError('KCL root requires negative transistor voltage')
        if top <= 0.:
            if low_residual == 0.:
                current = material(command, log_gamma)
                return current, command, low_residual
            raise NumericalUnknown('At command=vmin the KCL root requires subtable local voltage')
        if command-top < vmin:
            raise NumericalUnknown('Cannot represent a root bracket inside the table voltage domain')
        high_residual = residual(top)
        if high_residual < 0.:
            if material_limited:
                raise NumericalUnknown('KCL root requires local voltage below the existing table')
            raise NumericalDomainError('KCL root requires transistor drop above its table')
        vt = brentq(residual, 0., top, xtol=1e-15, rtol=1e-13, maxiter=100)
        vh = command-vt
        current = material(vh, log_gamma)
        return current, vh, current-float(np.interp(vt, tx, ti))

    def legacy(command, u):
        g = gamma(float(u))
        command = float(command)
        if command == 0:
            return 0., 0., 0.
        if command < 0 or math.log(command) > lv[-1]:
            raise NumericalDomainError('Command outside material table domain')
        if math.log(command) < lv[0]:
            raise NumericalUnknown('Positive command below table resolution')
        current, vh = t.series(command, g, ron)
        current, vh = float(current), float(vh)
        if vh <= 0 or math.log(vh) < lv[0]:
            raise NumericalUnknown('Legacy local voltage below table resolution')
        if math.log(vh) > lv[-1] or not 0 <= command-vh <= tx[-1]:
            raise NumericalDomainError('Legacy root outside material/transistor domains')
        return current, vh, current-float(np.interp(command-vh, tx, ti))

    def timeout(*_):
        raise TimeoutError('120second predeclared probe budget exhausted; no restart')

    begun = time.monotonic()
    previous_handler = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 120.)
    try:
        # Lorentz-area bound cancels Gamma*expm1(u)=gc and is state-uniform.
        bound = .5*kappa*vmin*vmin*(math.exp(lam*vmin)+math.pi*gc/(ron*js))
        report['smallfield_bound'] = dict(vmin_v=vmin,
            local_integral_upper_per_monotone_microsegment=bound,
            below_declared_1e_minus12=bound < 1e-12,
            scope='Local omitted RHS integral only; not exact inactivity or a global error certificate')
        if not math.isfinite(bound) or bound >= 1e-12:
            raise NumericalUnknown('The mandatory near-zero microsegments exceed the omission tolerance')
        fixed_states = (anchor, old_budget['anchors_u']['outward_2V_lower'],
                        old_budget['anchors_u']['return_5V_lower'],
                        old_budget['anchors_u']['return_2V_upper'], 15., 25.)
        comparisons, comparison_failures = [], []
        for command in (.02, .1, .2, 1., 2., 5.):
            for state in fixed_states:
                b = legacy(command, state)
                try:
                    a = fast(command, state)
                except (NumericalDomainError, NumericalUnknown, RuntimeError, ValueError) as exc:
                    comparison_failures.append(dict(command_v=command, state_u=state,
                        error_type=type(exc).__name__, error=str(exc)))
                    comparisons.append([command, state, float('nan'), float('nan'), float('nan')])
                    continue
                comparisons.append([command, state, abs(math.log10(a[0]/b[0])),
                                    abs(a[1]-b[1]), abs(a[2])])
        comparison_array = np.asarray(comparisons)
        valid_comparisons = comparison_array[np.all(np.isfinite(comparison_array[:, 2:]), axis=1), 2:]
        maxima = np.max(valid_comparisons, axis=0) if len(valid_comparisons) else [None, None, None]
        passed = bool(not comparison_failures and maxima[0] <= 5e-7
                      and maxima[1] <= 1e-9 and maxima[2] <= 1e-13)
        report['KCL_comparison'] = dict(n_points=36, completed_fast_points=len(valid_comparisons),
            max_log_current_difference_decade=float(maxima[0]) if maxima[0] is not None else None,
            max_local_voltage_difference_v=float(maxima[1]) if maxima[1] is not None else None,
            max_abs_KCL_residual_a=float(maxima[2]) if maxima[2] is not None else None,
            failures=comparison_failures, passed=passed,
            selected='scalar_brent' if passed else 'legacy42_bisection')
        arrays['fixed_circuit_comparison'] = comparison_array
        series = fast if passed else legacy
        for precision in PRECISIONS:
            run = dict(precision=precision, status='running', segments=[])
            report['runs'].append(run)
            prefix = precision['name']
            started = time.monotonic()
            last, omissions, rhs_rows = {}, [], []
            segment_name = None
            calls = 0
            rhs_kcl = 0.
            print(json.dumps(dict(stage='ODE_start', precision=prefix, kappa_per_v2=kappa)), flush=True)

            def rate(command, u):
                nonlocal calls, rhs_kcl
                calls += 1
                last.update(segment=segment_name, command_v=float(command), u=float(u))
                gamma(u)  # Also validate state at zero and bounded-omission points.
                if command == 0:
                    rhs_rows.append([command, u, 0., 0., 0., 0.])
                    return 0.
                if 0 < command < vmin:
                    if bound >= 1e-12:
                        raise NumericalUnknown('Subtable RHS omission bound exceeds declared tolerance')
                    omissions.append(dict(segment=segment_name, command_v=float(command), u=float(u),
                        bounded_numerical_omission=True, exact_inactive=False,
                        out_of_domain_current_evaluated=False))
                    # NaN records an unevaluated current/local voltage, never an invented zero.
                    rhs_rows.append([command, u, float('nan'), float('nan'), 0., float('nan')])
                    return 0.
                current, vh, kcl = series(command, u)
                rhs_kcl = max(rhs_kcl, abs(kcl))
                derivative = kappa*vh*(math.exp(lam*vh)-current*math.expm1(u)/js)
                if not math.isfinite(derivative):
                    raise NumericalDomainError('Nonfinite signed reversible derivative')
                rhs_rows.append([command, u, current, vh, derivative, kcl])
                return derivative

            def integrate(name, span, u0, command):
                nonlocal segment_name
                segment_name = name
                row = dict(name=name, span=list(span), initial_u=float(u0), status='running')
                run['segments'].append(row)
                solution = solve_ivp(lambda coordinate, y: [rate(command(coordinate), float(y[0]))],
                    span, [u0], method='DOP853', rtol=precision['rtol'], atol=precision['atol'],
                    max_step=precision['max_step'], dense_output=True)
                row.update(status='completed' if solution.success else 'solver_failed',
                    success=bool(solution.success), message=str(solution.message), nfev=int(solution.nfev),
                    accepted_steps=len(solution.t)-1, final_u=float(solution.y[0, -1]))
                arrays[prefix+'_'+name+'_coordinate'] = solution.t
                arrays[prefix+'_'+name+'_state_u'] = solution.y[0]
                arrays[prefix+'_'+name+'_state_q'] = np.exp(-solution.y[0])
                if not solution.success:
                    raise RuntimeError(solution.message)
                return solution

            try:
                back = integrate('backward', (.2, 0.), anchor, lambda coordinate: coordinate)
                u0 = float(back.y[0, -1])
                run['derived_initial_u'] = u0
                forward = integrate('outward', (0., 5.), u0, lambda coordinate: coordinate)
                returned = integrate('return', (5., 10.), float(forward.y[0, -1]), lambda coordinate: 10.-coordinate)
                u = np.empty_like(v)
                u[out] = forward.sol(s[out])[0]
                u[~out] = returned.sol(s[~out])[0]
                values = np.asarray([series(command, state) for command, state in zip(v, u)])
                current, vh, kcl = values.T
                if not (np.all(np.isfinite(current)) and np.all(current >= 0) and np.all(current[mask] > 0)):
                    raise NumericalDomainError('Invalid native current')
                groups = dict(all500=mask, outward250=mask & out, return250=mask & ~out,
                    outward_low10=mask & out & (v <= .2+1e-12),
                    both_branches_low20=mask & (v <= .2+1e-12), remaining480=mask & (v > .2+1e-12),
                    outward0p5_to1p5_50=mask & out & (v > .5+1e-12) & (v <= 1.5+1e-12))
                metrics = {}
                for name, selected in groups.items():
                    error = np.log10(current[selected]/obs[selected])
                    metrics[name] = dict(n=int(selected.sum()), RMSE_decade=float(np.sqrt(np.mean(error*error))),
                        mean_signed_log_error_decade=float(np.mean(error)),
                        median_absolute_factor=float(10**np.median(abs(error))))
                references = {}
                for name, reference in feature_ref.items():
                    indices = reference['native_indices']
                    value = (float(current[indices[0]]) if len(indices) == 1 else
                             float(math.log(current[indices[1]]/current[indices[0]]) /
                                   math.log(v[indices[1]]/v[indices[0]])))
                    low, high = reference['original32_aggregate_band']
                    references[name] = dict(single_cycle_prediction=value,
                        original32_aggregate_band=[low, high], native_indices=indices,
                        single_prediction_inside_reference_band=low <= value <= high,
                        scope='Reference context only, not an original32-record aggregate evaluation')
                actual = []
                for name, command, branch, direction in (
                    ('outward_2V_lower', 2., out, 'at_least'),
                    ('return_5V_lower', 5., ~out, 'at_least'),
                    ('return_2V_upper', 2., ~out, 'at_most')):
                    index = int(np.argmin(np.where(branch, abs(v-command), np.inf)))
                    if v[index] != command:
                        raise ValueError('Missing native endpoint')
                    limit, value = anchors[name]['current_a'], float(current[index])
                    actual.append(dict(name=name, native_index=index, predicted_current_a=value,
                        bound_current_a=limit, direction=direction, actual_u=float(u[index]),
                        bound_u=float(math.log1p(gc/anchors[name]['rederived_gamma_ev'])),
                        condition_satisfied=value >= limit if direction == 'at_least' else value <= limit))
                crossings = []
                for threshold in plan['outward_current_thresholds_a']:
                    item = dict(threshold_a=threshold)
                    for name, trace in (('observed', obs), ('predicted', current)):
                        hits = np.flatnonzero(out & mask & (trace >= threshold))
                        item[name] = (dict(reached=False) if not len(hits) else
                            dict(reached=True, last_below_v=float(v[hits[0]-1]),
                                 first_at_or_above_v=float(v[hits[0]])))
                    crossings.append(item)
                anchor_index = int(np.argmin(np.where(out, abs(v-.2), np.inf)))
                audited = np.asarray([legacy(command, state) for command, state in zip(v, u)])
                run.update(status='completed_conditional_reversible_single_cycle', metrics=metrics,
                    reference_features=references, actual_budget_endpoints=actual,
                    all_three_actual_endpoints_satisfied=all(row['condition_satisfied'] for row in actual),
                    outward_current_crossings=crossings, terminal_u=float(u[-1]), terminal_q=float(math.exp(-u[-1])),
                    forward_anchor_state_error=float(u[anchor_index]-anchor),
                    forward_anchor_current_over_observed=float(current[anchor_index]/obs[anchor_index]),
                    apex_state_continuity_error=float(returned.y[0, 0]-forward.y[0, -1]),
                    max_native_KCL_residual_a=float(max(abs(kcl))), max_rhs_KCL_residual_a=rhs_kcl,
                    max_native_log_current_difference_vs_legacy=float(max(abs(np.log10(current[mask]/audited[mask, 0])))),
                    max_native_local_voltage_difference_vs_legacy=float(max(abs(vh-audited[:, 1]))),
                    failed_single_cycle_reference_names=[name for name, row in references.items()
                                                        if not row['single_prediction_inside_reference_band']],
                    interpretation='Completed integration is not acceptance of the mechanism or unified model')
                arrays[prefix+'_current_a'] = current
                arrays[prefix+'_state_u'] = u
                arrays[prefix+'_state_q'] = np.exp(-u)
                arrays[prefix+'_local_voltage_v'] = vh
                arrays[prefix+'_KCL_residual_a'] = kcl
                arrays[prefix+'_legacy_native_current_a'] = audited[:, 0]
                arrays[prefix+'_legacy_native_local_voltage_v'] = audited[:, 1]
                np.testing.assert_allclose(np.linalg.norm(np.log10(current[mask])-np.log10(obs[mask]))/math.sqrt(500),
                                           metrics['all500']['RMSE_decade'], rtol=1e-12)
                run['arithmetic_audit_passed'] = True
            except Exception as exc:
                category = ('numerical_domain_unresolved_not_physical_rejection' if isinstance(exc, NumericalDomainError)
                    else 'numerical_resolution_unresolved_not_physical_rejection' if isinstance(exc, NumericalUnknown)
                    else 'time_budget_exhausted_not_physical_rejection' if isinstance(exc, TimeoutError)
                    else 'solver_or_audit_failure_not_physical_rejection')
                run.update(status='failed_preserved', error_type=type(exc).__name__, error=str(exc),
                           last_rhs_evaluation=dict(last), failure_scope=category)
                if run['segments'] and run['segments'][-1]['status'] == 'running':
                    run['segments'][-1]['status'] = 'exception_before_completion'
                if isinstance(exc, TimeoutError):
                    raise
            finally:
                completed = [row['name'] for row in run['segments'] if row['status'] == 'completed']
                # Every completed prescribed segment crosses one 0<->vmin microsegment.
                # For a failed segment, count at most one potential crossing, conservatively.
                omission_segments = sorted(set(completed) | {row['segment'] for row in omissions})
                run.update(elapsed_s=time.monotonic()-started, n_rhs_calls=calls,
                    smallfield_certificate=dict(RHS_omission_evaluation_count=len(omissions), exact_inactive_count=0,
                        physical_microsegment_names=omission_segments,
                        local_integral_bound_per_microsegment=bound,
                        sum_local_bounds=len(omission_segments)*bound,
                        scope='At most one microsegment per prescribed monotone segment; no retry multiplier. '
                              'A failed partial segment is conservatively counted once if omission was used. '
                              'These are local omission bounds, not a trajectory-error certificate.',
                        omission_evaluations=omissions))
                arrays[prefix+'_rhs_trace'] = np.asarray(rhs_rows).reshape((-1, 6))
                run['rhs_trace_columns'] = ['command_v', 'state_u', 'current_a', 'local_voltage_v',
                                            'signed_u_derivative_per_v', 'KCL_residual_a']
                run['rhs_trace_nan_meaning'] = 'Current/local voltage not evaluated during bounded subtable omission'
        if 'nominal_current_a' in arrays and 'refined_current_a' in arrays:
            report['precision_comparison'] = dict(
                max_native_log_current_change_decade=float(max(abs(np.log10(
                    arrays['nominal_current_a'][mask]/arrays['refined_current_a'][mask])))),
                max_native_state_u_change=float(max(abs(arrays['nominal_state_u']-arrays['refined_state_u']))),
                max_native_state_q_change=float(max(abs(arrays['nominal_state_q']-arrays['refined_state_q']))),
                terminal_state_u_change=float(arrays['nominal_state_u'][-1]-arrays['refined_state_u'][-1]),
                max_native_local_voltage_change_v=float(max(abs(
                    arrays['nominal_local_voltage_v']-arrays['refined_local_voltage_v']))))
        report['status'] = 'completed_two_prescribed_attempts_all_failures_retained'
    except Exception as exc:
        report.update(status='partial_or_failed_probe_preserved_no_restart', error_type=type(exc).__name__, error=str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        signal.signal(signal.SIGALRM, previous_handler)
        report['compute_elapsed_s'] = time.monotonic()-begun
        unchanged = {path: sha(path) == value for path, value in frozen.items()}
        report['frozen_source_files_unchanged'] = unchanged
        if not all(unchanged.values()):
            report['status'] = 'invalid_frozen_source_changed'
        report['n_completed_runs'] = sum(row['status'] == 'completed_conditional_reversible_single_cycle'
                                         for row in report['runs'])
        report['n_failed_runs'] = sum(row['status'] == 'failed_preserved' for row in report['runs'])
        dump(dest, report)
        np.savez_compressed(dest.with_suffix('.npz'), **arrays)

    lines = ['# 单种子可逆 q：条件单圈 SET forward', '',
             f'状态：{report["status"]}；计算 {report["compute_elapsed_s"]:.6f} s；上限 120 s。', '',
             f'固定 κ={kappa:.17g} V⁻²；选择规则：{fixed_rate_plan["selection_rule"]}。', '',
             'q=e⁻ᵘ，Γ=Γc/expm1(u)，u′=κv[exp(λv)−I expm1(u)/J]。保留导数双向变化，未用正部截断。', '',
             'λ/J 使用已保存返扫 2 V、4 V 两点校准；不是独立预测。4 V 局部平衡排斥，2 V 吸引，不能说二者均稳定。', '',
             '| 精度 | 状态 | 全500 RMSE/dec | 外扫250 | 返扫250 | 外扫低10 | 其余480 |',
             '|---|---|---:|---:|---:|---:|---:|']
    for row in report['runs']:
        metrics = row.get('metrics')
        values = [f'{metrics[key]["RMSE_decade"]:.8f}' for key in
                  ('all500', 'outward250', 'return250', 'outward_low10', 'remaining480')] if metrics else ['—']*5
        lines.append('| '+' | '.join([row['precision']['name'], row['status'], *values])+' |')
        if not metrics:
            lines.extend(['', f'失败类型：{row.get("failure_scope")}；{row.get("error")}。'])
            continue
        lines.extend(['', f'{row["precision"]["name"]} 三个实际端点：', '',
                      '| 端点 | 预测/µA | 原界/µA | 满足 |', '|---|---:|---:|---|'])
        for point in row['actual_budget_endpoints']:
            lines.append(f'| {point["name"]} | {point["predicted_current_a"]*1e6:.8f} | '
                         f'{point["bound_current_a"]*1e6:.8f} | {point["condition_satisfied"]} |')
        lines.extend(['', '| 六个返扫参考 | 单圈值 | 原32圈聚合带 | 在带内 |', '|---|---:|---|---|'])
        for name, reference in row['reference_features'].items():
            lines.append(f'| {name} | {reference["single_cycle_prediction"]:.9g} | '
                         f'{reference["original32_aggregate_band"]} | '
                         f'{reference["single_prediction_inside_reference_band"]} |')
        for crossing in row['outward_current_crossings']:
            lines.extend(['', f'{crossing["threshold_a"]:g} A 首次原生跨越：'
                          f'观测 {crossing["observed"]}；预测 {crossing["predicted"]}。'])
    if 'precision_comparison' in report:
        lines.extend(['', '两精度比较：`'+json.dumps(report['precision_comparison'])+'`。'])
    lines.extend(['', '原生 500 点全部保留，包含条件初值锚与低电流点。'
                  '原三个端点及六个返扫比较仅是单圈参考，不能替代原32圈聚合门槛。', '',
                  '近零仅使用 Lorentz 面积上界约束数值遗漏，未赋域外电流；局部界不是全轨迹误差保证。'
                  '数值域/分辨率失败不构成物理反证。s 不是秒，未补造扫描时间。', '',
                  '本结果不包括 RESET、裸1R、跨圈状态衔接、保留集或完整统一模型验证；'
                  '没有根据输出追加速率/尺度/种子，所有失败保留。'])
    dest.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=report['status'], completed=report['n_completed_runs'],
        failed=report['n_failed_runs'], compute_elapsed_s=report['compute_elapsed_s'],
        precision_comparison=report.get('precision_comparison'),
        runs=[{key: row[key] for key in ('precision', 'status', 'metrics', 'actual_budget_endpoints',
                                       'outward_current_crossings', 'error', 'failure_scope') if key in row}
              for row in report['runs']]), indent=2))


if __name__ == '__main__':
    main()
