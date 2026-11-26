"""At most30 fixed-spectrum diagnostic inversions; no ODE or parameter fit."""
from pathlib import Path
import hashlib
import json
import math
import signal
import sys
import time

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PREVIOUS = ROOT / 'analysis/reset_shape_2026-11-03'
CACHE = Path('/private/tmp/vulcan-runtime-cache')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analysis/mechanism_selection_2026-11-03'))
from SOFT_FEEDBACK_LOW_BIAS_BUDGET import cached_transport

NODES = [.2, .4, .6, .8, 1., 1.4, 2., 2.5, 3., 3.5, 4., 4.5, 5.]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    psource = PREVIOUS / 'soft_field_mobility_probe.json'
    p = json.loads(psource.read_text())
    ron, gc, beta, js = [p['plan'][key] for key in ('Ron_ohm', 'gamma_c_ev', 'beta', 'JS_A')]
    plan = dict(
        phase='cycle4 positive SET only, both chronological branches', commands_v=NODES,
        observations=26, original_return4_band_endpoint_inversions=2,
        optional_equal_state_log_error_balancing_roots=2, maximum_inverse_problems=30,
        frozen=dict(Ron_ohm=ron, gamma_c_ev=gc, beta=beta, JS_A=js),
        scope='Same-cycle observed-current inversion is a fixed-spectrum/fixed-transistor-table diagnostic, not a prediction or material-law-wide rejection.',
        native_selection='Nearest sample in declared branch must be within3e-7V; record mismatch and do not interpolate observed current.',
        inverse='Invert the same monotone piecewise-linear transistor lookup, then invert the same log-bilinear material table at measured local voltage; no extrapolation.',
        pair='Outward2V precedes return4V; positive mobility requires u_out2>=u_return4.',
        tradeoff='Only if target states violate order, use the shared-u equal-and-opposite log-current-error root to obtain the optimistic minimum maximum log error. This is an analytic order relaxation, not an ODE fit or an optimizer.',
        band_scope='Original32-record return4V empirical band plus previously declared outward2V lower anchor. The latter is identified separately, not relabelled a primary aggregate gate.',
        no_tolerance_gate=True, no_ODE=True, no_optimizer=True, no_holdout=True,
        compute_budget_s=30,
    )
    plan_path = HERE / 'DRIVE_STRUCTURE_INVERSION.plan.json'
    plan_path.write_text(json.dumps(plan, indent=2) + '\n')
    t, lookups = cached_transport(CACHE)
    for path in (t.table_path, *lookups):
        assert sha(path) == p['source_sha256'][str(path)]
    with np.load(psource.with_suffix('.npz')) as z:
        voltage = z['command_voltage_v'].copy()
        observed = abs(z['observed_current_a'].copy())
        original_mask = z['original_scoring_mask'].copy()
    outward = np.arange(len(voltage)) <= int(np.argmax(voltage))
    assert int(original_mask.sum()) == 500
    np.testing.assert_array_equal(original_mask, np.isfinite(observed) & (observed >= 1e-11) & (abs(voltage) > 1e-12))
    existing = json.loads((PREVIOUS / 'soft_beta_conditional_forward.json').read_text())
    run = next(r for r in existing['runs'] if r['status'] == 'completed_conditional_single_cycle_forward')
    band4 = run['reference_features']['1t1r.set.return.I_4V']['original32_aggregate_band']
    source = json.loads((CACHE / 'RON_SCALING_REVIEW.json').read_text())
    anchors = next(r for r in source['rows'] if r['Ron_ohm'] == ron)['anchor_reinversions']
    outward_lower = anchors['outward_2V_lower']['current_a']
    tx, ti = t.knots['set']; lg, lv, li = t.lg, t.lv, t.li[0]
    logfactor = math.log(t.r0 / ron)
    assert np.all(np.diff(tx) > 0) and np.all(np.diff(ti) >= 0)
    count = 0

    def inverse(A, I):
        nonlocal count
        count += 1
        assert count <= 30
        if not math.isfinite(I) or I <= 0:
            raise ValueError('Nonpositive/nonfinite diagnostic current')
        if I >= float(np.interp(min(A, tx[-1]), tx, ti)) or I <= ti[0]:
            raise ValueError('Observed current incompatible with positive material voltage inside transistor lookup')
        if np.count_nonzero(ti == I) > 1:
            raise ValueError('Transistor inverse not unique at a lookup plateau')
        vt = float(np.interp(I, ti, tx)); vh = A - vt
        if vh <= 0 or not lv[0] <= math.log(vh) <= lv[-1]:
            raise ValueError('Inferred material voltage outside original table')
        iv = min(max(int(np.searchsorted(lv, math.log(vh)) - 1), 0), len(lv)-2)
        fv = (math.log(vh) - lv[iv]) / (lv[iv+1] - lv[iv])
        profile = (1-fv)*li[:, iv] + fv*li[:, iv+1] + logfactor
        logi = math.log(I)
        if not profile[0] <= logi <= profile[-1]:
            raise ValueError('Required Gamma lies outside original table; no extrapolation')
        ig = min(max(int(np.searchsorted(profile, logi) - 1), 0), len(lg)-2)
        if profile[ig+1] <= profile[ig]:
            raise ValueError('Material inverse is not locally unique')
        fg = (logi-profile[ig])/(profile[ig+1]-profile[ig])
        logg = float((1-fg)*lg[ig]+fg*lg[ig+1]); gamma = math.exp(logg)
        u = math.log1p(gc/gamma)
        q = float((profile[ig+1]-profile[ig])/(lg[ig+1]-lg[ig]))
        n = float(((1-fg)*(li[ig, iv+1]-li[ig, iv])+fg*(li[ig+1, iv+1]-li[ig+1, iv]))/(lv[iv+1]-lv[iv]))
        jt = min(max(int(np.searchsorted(tx, vt)-1), 0), len(tx)-2)
        gt = float((ti[jt+1]-ti[jt])/(tx[jt+1]-tx[jt])); gh = I*n/vh
        H = q/(-math.expm1(-u)); chi = H*gt/(gh+gt)
        alpha = gh*gt/(I*(gh+gt)); drive = math.log(I/js)+beta*u
        predicted, checkv = t.series(A, gamma, ron)
        return dict(command_v=A, current_a=I, transistor_drop_v=vt, material_voltage_v=vh,
            gamma_ev=gamma, u=u, D=drive, drive_positive=drive > 0,
            q_intrinsic_dlogI_dlogGamma=q, n_intrinsic_dlogI_dlogV=n,
            g_h_s=gh, g_t_s=gt, intrinsic_H_minus_dlogI_du=H,
            circuit_alpha_dlogI_dA_per_v=alpha, circuit_chi_minus_dlogI_du=chi,
            D_u_beta_minus_chi=beta-chi, D_A_per_v=alpha,
            local_implicit_tangent_du_dA=-alpha/(beta-chi) if beta != chi else None,
            local_implicit_tangent_terminal_log_slope=A*beta*alpha/(beta-chi) if beta != chi else None,
            tangent_scope='Formula evaluated at this observed-inverted state; this point need not lie on D=0.',
            inverse_forward_current_log10_error=float(math.log10(float(predicted)/I)),
            inverse_forward_material_voltage_error_v=float(float(checkv)-vh))

    def predict(A, u):
        gamma = gc/math.expm1(u)
        if not lg[0] <= math.log(gamma) <= lg[-1]:
            raise ValueError('Tradeoff state outside table')
        value, _ = t.series(A, gamma, ron)
        return float(value)

    def tradeoff(label, first, later):
        nonlocal count
        ua, ub = first['u'], later['u']
        IA, IB = first['current_a'], later['current_a']
        row = dict(label=label, earlier_command_v=2., later_command_v=4.,
            earlier_target_a=IA, later_target_a=IB, earlier_required_u=ua, later_required_u=ub,
            required_later_minus_earlier_u=ub-ua, order_conflict=ub > ua,
            holding_outward_target_return4_minimum_a=predict(4., ua),
            holding_return_target_outward2_maximum_a=predict(2., ub))
        if ub <= ua:
            row['scope']='No necessary order conflict for these endpoints; no balancing root performed.'
            return row
        count += 1
        assert count <= 30
        def errors(u):
            return np.array([math.log10(predict(2., u)/IA), math.log10(predict(4., u)/IB)])
        common = float(brentq(lambda u: float(errors(u).sum()), ua, ub, xtol=1e-12, rtol=1e-13))
        error = errors(common); delta = float(max(abs(error)))
        row.update(shared_state_at_minimax_u=common, signed_log10_current_errors=error.tolist(),
            minimum_possible_max_absolute_log10_current_error=delta,
            corresponding_symmetric_multiplicative_factor=10**delta,
            earlier_balanced_prediction_a=predict(2., common), later_balanced_prediction_a=predict(4., common),
            optimistic_two_point_RMSE_lower_bound_decade=delta/math.sqrt(2.),
            feasible_equal_state_two_point_RMSE_decade=float(np.sqrt(np.mean(error**2))),
            optimistic_500_point_RMSE_lower_bound_from_this_pair_only_decade=delta/math.sqrt(500.),
            scope='Exact minimax order relaxation conditional on fixed spectrum/table and monotone readout. The RMSE lower bounds are deliberately weaker; no least-squares optimizer or full trajectory feasibility is claimed.')
        return row

    def timeout(*_):
        raise TimeoutError('30-second diagnostic budget exhausted; no restart')
    started = time.monotonic(); signal.signal(signal.SIGALRM, timeout); signal.setitimer(signal.ITIMER_REAL, 30.)
    report = dict(status='running', plan=plan, observations=[], band_inversions=[], pair_tradeoffs=[],
        original_scoring_count=500, original_scoring_mask_modified=False)
    try:
        for branch, take in [('outward', outward), ('return', ~outward)]:
            for A in (NODES if branch == 'outward' else NODES[::-1]):
                j = int(np.argmin(np.where(take, abs(voltage-A), np.inf)))
                base = dict(branch=branch, requested_command_v=A, native_index=j,
                    native_voltage_mismatch_v=float(voltage[j]-A), original_scoring_eligible=bool(original_mask[j]))
                try:
                    if abs(voltage[j]-A) > 3e-7 or not original_mask[j]:
                        raise ValueError('Declared native point absent or outside unchanged scoring mask')
                    base.update(status='inverted', **inverse(float(voltage[j]), float(observed[j])))
                except ValueError as exc:
                    base.update(status='inversion_failed_preserved', error=str(exc))
                report['observations'].append(base)
        for side, value in [('original_return4_lower', band4[0]), ('original_return4_upper', band4[1])]:
            row = dict(source=side, scope='Original32-record empirical reference band endpoint; not a cycle4 error bar.')
            try:
                row.update(status='inverted', **inverse(4., value))
            except ValueError as exc:
                row.update(status='inversion_failed_preserved', error=str(exc))
            report['band_inversions'].append(row)
        out2 = next(r for r in report['observations'] if r['branch']=='outward' and r['requested_command_v']==2.)
        ret4 = next(r for r in report['observations'] if r['branch']=='return' and r['requested_command_v']==4.)
        if out2['status']==ret4['status']=='inverted':
            report['pair_tradeoffs'].append(tradeoff('cycle4_exact_observations', out2, ret4))
        bandhigh = report['band_inversions'][1]
        if bandhigh['status']=='inverted':
            existing_out = dict(current_a=outward_lower,
                u=math.log1p(gc/anchors['outward_2V_lower']['rederived_gamma_ev']))
            report['pair_tradeoffs'].append(tradeoff('previous_outward2_lower_anchor_vs_original_return4_upper_band', existing_out, bandhigh))
        slopes=[]
        for a,b in zip(report['observations'][:-1], report['observations'][1:]):
            if a['status']!='inverted' or b['status']!='inverted':continue
            da=abs(b['command_v']-a['command_v'])
            if da==0:continue
            slopes.append(dict(earlier_native_index=a['native_index'],later_native_index=b['native_index'],
                earlier_branch=a['branch'],later_branch=b['branch'],du_required=b['u']-a['u'],
                delta_command_path_v=da, secant_du_required_per_path_v=(b['u']-a['u'])/da,
                monotone_SET_order_violated=b['u']>a['u'],
                scope='Diagnostic finite difference of exact-observation inversions; not a new pointwise acceptance gate or measured state derivative.'))
        report.update(status='completed_bounded_diagnostic_inversions_no_ODE_no_fit', selected_state_secants=slopes)
    except Exception as exc:
        report.update(status='partial_diagnostic_failure_preserved',error_type=type(exc).__name__,error=str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0.);report['compute_elapsed_s']=time.monotonic()-started
    report['inverse_problem_count']=count
    report['source_sha256']={str(path):sha(path) for path in (psource,psource.with_suffix('.npz'),
        PREVIOUS/'soft_beta_conditional_forward.json',CACHE/'RON_SCALING_REVIEW.json',
        t.table_path,*lookups,plan_path,Path(__file__))}
    report['interpretation']=[
        'Conditional on this spectrum, Ron and positive-transistor current table. Gate1.1V is interpolated; no measurement/calibration confidence interval is assigned.',
        'The inferred u is an effective inverse coordinate, not an independently measured gap or microscopic state.',
        'Same-cycle observations used for inversion and diagnostic residual tradeoffs are not predictions.',
        'No new tolerance gate. Original return4 aggregate band and prior outward2 lower anchor retain distinct provenance.',
        'No ODE, parameter optimizer, additional candidate, mapping scan or original holdout access.',
    ]
    dest=HERE/'DRIVE_STRUCTURE_INVERSION.json'
    dest.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],inverse_problem_count=count,elapsed_s=report['compute_elapsed_s'],
        observations=[{k:r[k] for k in ('branch','requested_command_v','status','current_a','u','D','D_u_beta_minus_chi') if k in r} for r in report['observations']],
        pair_tradeoffs=report['pair_tradeoffs']),indent=2))


if __name__=='__main__':
    main()
