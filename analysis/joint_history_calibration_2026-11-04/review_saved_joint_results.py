"""Read saved artifacts; never import a model, solver, optimizer, or runner."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
import hashlib
import json
import math
import numpy as np

HERE = Path(__file__).resolve().parent
CAL = ('set4', 'reset4', 'set5')
ALL = CAL + ('reset5', 'set6')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    result_bytes = (HERE / 'bounded_joint_calibration.json').read_bytes()
    report = json.loads(result_bytes)
    checks, failures = 0, []

    def ck(name, condition):
        nonlocal checks
        checks += 1
        if not bool(condition):
            failures.append(name)

    def close(name, a, b, rtol=1e-11, atol=1e-13):
        ck(name, math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=atol))

    def maximum(values):
        return float(np.max(np.abs(values), initial=0.))

    def domain(u):
        u = np.asarray(u)
        if not np.all(np.isfinite(u)) or np.any(u <= 0.) or np.any(u > 80.):
            return False
        gamma = report['fixed_material']['gamma_c_ev'] / np.expm1(u)
        return np.all(np.isfinite(gamma)) and np.all((gamma >= 1e-24) & (gamma <= 1e6))

    def ports(v, i, h, t, kcl, kvl):
        z, nz = v == 0., v != 0.
        return (all(np.all(np.isfinite(x)) for x in (v, i, h, t, kcl, kvl))
                and np.all(i[nz] * v[nz] > 0.) and np.all(h[nz] * v[nz] > 0.)
                and np.all(t[nz] * v[nz] >= 0.) and np.all(i[z] == 0.)
                and np.all(h[z] == 0.) and np.all(t[z] == 0.)
                and np.all(i * h >= 0.) and np.all(i * t >= 0.)
                and maximum(kcl) <= 1e-13 and maximum(kvl) <= 1e-12
                and maximum(v - h - t) <= 1e-12)

    # Read each generated artifact once. Only this new stage is decoded.
    catalog = report['artifact_catalog']
    def read_record(item):
        path, record = item
        data = Path(path).read_bytes()
        return path, record, data
    payloads = {}
    artifact_records = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for path, record, data in pool.map(read_record, catalog.items()):
            ck('artifact hash ' + path, digest(data) == record['sha256'])
            ck('artifact bytes ' + path, len(data) == record['bytes'])
            payloads[path] = data
            artifact_records.append({'path': path, 'sha256': digest(data), 'matched': digest(data) == record['sha256']})
    sources = []
    for path, expected in report['source_sha256'].items():
        actual = digest(Path(path).read_bytes())
        ck('input/end source hash ' + path, actual == expected)
        sources.append({'path': path, 'sha256': actual, 'matched': actual == expected})
    print('Generated artifacts and actual source hashes checked', flush=True)

    current = (HERE / 'bounded_joint_calibration.py').read_bytes()
    snapshot = (HERE / 'EXECUTED_RUNNER_SNAPSHOT.py').read_bytes()
    note = (HERE / 'RUNNER_SOURCE_TIMING_NOTE.md').read_bytes()
    old = b"report.update(status='stopped_with_preserved_exception', error_type=type(exc).__name__, error=str(exc))"
    new = b"report.update(status='stopped_with_preserved_exception', error_type=type(exc).__name__,\n                      error=str(exc), numerical_validation_passed=False)"
    ck('exact isolated exceptional flag diff', current.count(new) == 1 and current.replace(new, old) == snapshot)
    ck('reviewed/startup snapshot hash', digest(snapshot) == '4ffd87a98aca8550ccb04b62da1590159cc5746e7a731f8d8f2aefd14ef2b98a')
    ck('current runner hash', digest(current) == '5b99ed81c3f90f01f750774669967b03132a48c7f187c59d0239c1f488ec91cd')
    ck('recorded input runner is current version', report['source_sha256'][str(HERE / 'bounded_joint_calibration.py')] == digest(current))
    ck('terminal normal completion', report['status'] == 'completed_bounded_joint_diagnostic' and report['numerical_validation_passed'])
    ck('scope flags', all(report[k] is False for k in ('full_unified_model_achieved', 'full_joint_contract_evaluated',
        'original_holdout_accessed', 'bare_data_accessed', 'continuation_used_for_selection', 'old_trajectory_NPZ_read')))
    ck('no new state inversions', report['new_state_inversions'] == 0)
    source_qualification = {
        'reviewed_startup_snapshot_sha256': digest(snapshot), 'current_and_input_hash_sha256': digest(current),
        'timing_note_sha256': digest(note), 'difference_only_top_level_exception_flag': True,
        'normal_numerical_path_unchanged': True, 'entire_startup_source_frozen_claim': False,
        'scope': 'Input/end hashes match the edited file, not proof of unchanged startup bytes. The reconstructed snapshot matches the reviewed startup hash; no model rerun was used to erase this provenance limitation.',
    }

    with np.load(HERE / 'training_phases.npz', allow_pickle=False) as archive:
        native = {key: archive[key].copy() for key in archive.files}
    requests = report['requests']
    by_id = {r['request_id']: r for r in requests}
    ck('88 contiguous requests', list(by_id) == list(range(1, 89)))
    objectives = [r for r in requests if r['objective_request']]
    ck('85 objective requests', len(objectives) == report['objective_requests'] == 85)
    ck('objective request indices contiguous', [r['objective_request_index'] for r in objectives] == list(range(1, 86)))
    ck('only DOP control and final outside objective', [r['label'] for r in requests if not r['objective_request']]
       == ['control_DOP853', 'final_nominal', 'final_refined'])
    statuses = Counter(r['status'] for r in requests)
    ck('actual request failure ledger empty', statuses == {'completed_chain': 87, 'completed_cache_request': 1})
    phase_records = {}
    for r in requests:
        record_path = str(HERE / 'requests' / f"{r['request_id']:03d}_{r['label']}.json")
        ck(f"request {r['request_id']} snapshot", json.loads(payloads[record_path]) == r)
        ck(f"request {r['request_id']} domain", len(r['x_normalized']) == 4 and all(0 <= x <= 1 for x in r['x_normalized']))
        is_final = r['label'].startswith('final_')
        ck(f"request {r['request_id']} phase selection", r['phases'] == list(ALL if is_final else CAL))
        ck(f"request {r['request_id']} continuation excluded", not is_final or r['continuation_used_for_selection'] is False)
        if r['status'] == 'completed_cache_request':
            source = by_id[r['cache_source_request']]
            ck('exact cache identity', all(r[key] == source[key] for key in ('x_normalized', 'method', 'precision', 'phases', 'scores', 'handoffs', 'phase_results')))
            ck('cache target earlier and search-only', source['request_id'] < r['request_id'] and r['objective_request'] and not is_final)
        for p in r['phase_results']:
            ck('phase ledger completed', p['status'] == 'completed')
            arrays_path, summary_path = p['artifacts']['arrays']['path'], p['artifacts']['summary']['path']
            if arrays_path not in phase_records:
                phase_records[arrays_path] = (summary_path, p['phase'])
            ck('phase array catalog identity', p['artifacts']['arrays'] == catalog[arrays_path])
            ck('phase summary catalog identity', p['artifacts']['summary'] == catalog[summary_path])
    ck('265 unique physical phases', len(phase_records) == 265)

    phases = {}
    phase_reports = []
    all_max = defaultdict(float)
    rhs_total = independent_native_total = independent_nonzero_total = accepted_segments = 0
    for arrays_path, (summary_path, name) in phase_records.items():
        s = json.loads(payloads[summary_path])
        with np.load(BytesIO(payloads[arrays_path]), allow_pickle=False) as archive:
            a = {key: archive[key].copy() for key in archive.files}
        phases[arrays_path] = (s, a)
        tag = Path(arrays_path).stem + ' '
        v, obs, mask = a['command_voltage_v'], a['observed_current_a'], a['original_scoring_mask']
        is_set = name.startswith('set')
        n_expected, scored_expected = (503, 500) if is_set else (173, 170)
        ck(tag + 'original native identity', all(np.array_equal(a[key], native[f'{name}_{other}']) for key, other in (
            ('command_voltage_v', 'command_voltage_v'), ('observed_current_a', 'observed_current_a'), ('original_scoring_mask', 'scoring_mask'))))
        original = np.isfinite(abs(obs)) & (abs(obs) >= (1e-11 if is_set else 1e-12)) & (abs(v) > 1e-12)
        ck(tag + 'unaltered original mask', mask.dtype == np.bool_ and np.array_equal(mask, original))
        ck(tag + 'counts', len(v) == n_expected == s['n_native'] and int(mask.sum()) == scored_expected == s['n_scored'])
        low = mask & (abs(v) <= .2 + 1e-12)
        ck(tag + 'all low20 retained', low.sum() == 20 and s['metrics']['low_abs_le_0p2']['n'] == 20)
        u, q, i, h, t = (a[key] for key in ('state_u', 'state_q', 'current_a', 'local_voltage_v', 'transistor_voltage_v'))
        ck(tag + 'native state domain', domain(u))
        ck(tag + 'q mapping', np.allclose(q, np.exp(-u), rtol=1e-13, atol=0))
        ck(tag + 'native ports', ports(v, i, h, t, a['KCL_residual_a'], a['KVL_residual_v']))
        nz = v != 0.
        ck(tag + 'native readout finite domains', np.all(abs(h[nz]) >= 1e-9 * (1-1e-12))
           and np.all(abs(h[nz]) <= 5.) and np.all(abs(t) <= (5. if is_set else 2.)))
        clock = np.cumsum(abs(np.diff(np.r_[s['initial_command_voltage_v'], v])))
        ck(tag + 'path clock', np.allclose(clock, a['path_voltage_s'], rtol=0, atol=1e-12))
        repeated = np.diff(v) == 0.
        ck(tag + 'duplicates unchanged', np.all(np.diff(u)[repeated] == 0.))
        ck(tag + 'initial/terminal identity', u[0] == s['initial_u'] and u[-1] == s['terminal_u'] == s['final_u'])
        ck(tag + 'no solver failures', s['status'] == 'completed' and s['root_failures'] == [] and s['state_domain_failures'] == [])
        ck(tag + 'fixed shared material', s['fixed_material'] == report['fixed_material'] and s['fixed_recovery_order'] == 2 and s['parallel_path_a_s'] == 0.)
        previous_u, previous_s = s['initial_u'], 0.
        for segment in s['segments']:
            accepted_segments += 1
            index = segment['index']
            xs, us, qs = (a[f'segment_{index}_{suffix}'] for suffix in ('coordinate', 'state_u', 'state_q'))
            ck(tag + f'accepted segment {index}', len(xs) == segment['accepted_steps'] + 1 and np.all(np.diff(xs) > 0)
               and domain(us) and np.allclose(qs, np.exp(-us), rtol=1e-13, atol=0)
               and us[0] == previous_u == segment['initial_u'] and us[-1] == segment['final_u']
               and xs[0] == previous_s == segment['path_start_v'] and xs[-1] == segment['path_stop_v']
               and segment['status'] == 'completed' and segment['solver_status'] == 'finished')
            previous_u, previous_s = segment['final_u'], segment['path_stop_v']
        ck(tag + 'segment chain terminal', previous_u == s['terminal_u'] and abs(previous_s-clock[-1]) <= 1e-12)
        error = np.full(len(v), np.nan)
        error[mask] = np.log10(abs(i[mask]) / abs(obs[mask]))
        ck(tag + 'stored residual', np.allclose(error, a['log10_current_residual'], rtol=1e-12, atol=1e-12, equal_nan=True))
        outward = np.arange(len(v)) <= np.argmax(abs(v))
        groups = {'all_eligible': mask, 'low_abs_le_0p2': low, 'outward': mask & outward, 'return': mask & ~outward}
        for group, use in groups.items():
            stat = s['metrics'][group]
            ck(tag + group + ' count', stat['n'] == int(use.sum()))
            close(tag + group + ' RMSE', np.sqrt(np.mean(error[use]**2)), stat['RMSE_decade'])
            close(tag + group + ' signed error', np.mean(error[use]), stat['mean_signed_log_error_decade'])
            close(tag + group + ' median factor', 10**np.median(abs(error[use])), stat['median_absolute_factor'])
        audit = s['phase_integrity']
        ck(tag + 'saved integrity checks', audit['passed'] and all(x['passed'] for x in audit['checks']))
        params, b, omission = s['parameters'], s['b'], s['smallfield_omission']
        sign = 1 if is_set else -1
        bound = .5 * params['kappa_per_v2'] * 1e-18 * (
            math.exp((1-sign)*b + params['lambda_per_v']*1e-9)
            + math.pi * report['fixed_material']['gamma_c_ev'] / (report['fixed_material']['Ron_ohm']*params['J_positive_a']) * math.exp((sign-1)*b))
        ck(tag + 'explicit nearzero bound budget', s['nearzero_segment_budget'] == 1e-10 and 0 < bound < 1e-10)
        close(tag + 'nearzero bound arithmetic', bound, s['smallfield_bounds_by_sign'][str(sign)], atol=1e-30)
        ck(tag + 'physical omission accounting', omission['physical_microsegment_indices'] == [0, 1] and omission['exact_inactive_claim'] is False)
        close(tag + 'omission sum', 2*bound, omission['local_bound_sum'], atol=1e-30)
        if 'rhs_trace' in a:
            rows = a['rhs_trace']; rhs_total += len(rows)
            ck(tag + 'complete RHS trace count', len(rows) == s['n_rhs_calls'] and rows.shape[1] == 7 and domain(rows[:,1]))
            full = np.all(np.isfinite(rows[:,[2,3,4,6]]), axis=1)
            missing, z = rows[~full], rows[full]
            ck(tag + 'RHS omission rows', len(missing) == omission['n_RHS_omissions'] == len(omission['evaluations'])
               and np.all(np.isnan(missing[:,[2,3,4,6]])) and np.all(missing[:,5] == 0.)
               and np.all((abs(missing[:,0]) > 0.) & (abs(missing[:,0]) < 1e-9*(1+1e-12))))
            ck(tag + 'RHS ports', ports(z[:,0], z[:,2], z[:,3], z[:,4], z[:,6], z[:,0]-z[:,3]-z[:,4]))
            shift = np.where(z[:,3] < 0., 2*b, 0.)
            recovery = params['kappa_per_v2']*abs(z[:,3])*np.exp(shift + params['lambda_per_v']*abs(z[:,3])-z[:,1])
            capture = (params['kappa_per_v2']/params['J_positive_a'])*abs(z[:,3])*abs(z[:,2])*np.exp(-shift)*np.expm1(z[:,1])
            delta = abs(z[:,5] - (recovery-capture))
            ck(tag + 'saved RHS algebra', np.all(delta <= 5e-12*(abs(recovery)+abs(capture))+1e-12))
            all_max['RHS_algebra_absolute_difference'] = max(all_max['RHS_algebra_absolute_difference'], maximum(delta))
        else:
            ck(tag + 'search explicitly native-only', audit['scope'] == 'native_only_search_not_final_validation' and not audit['complete_integrity_passed'])
        if audit['independent_requested']:
            independent_native_total += len(v); independent_nonzero_total += int(nz.sum())
            ck(tag + 'final complete RHS and independent', 'rhs_trace' in a and audit['complete_integrity_passed'] and s['independent_native_readout']['passed'])
            ck(tag + 'all independent roots saved', np.all(a['independent_completed_mask'])
               and np.all(a['independent_bisections'][nz] == 48) and np.all(a['independent_bisections'][~nz] == 0))
            ic, ih, it, ik, iv = (a['independent_'+key] for key in ('current_a', 'local_voltage_v', 'transistor_voltage_v', 'KCL_residual_a', 'KVL_residual_v'))
            ck(tag + 'independent ports', ports(v, ic, ih, it, ik, iv))
            logdiff, vdiff = maximum(np.log10(abs(i[nz]/ic[nz]))), maximum(h-ih)
            ck(tag + 'independent agreement', logdiff <= 5e-7 and vdiff <= 1e-9)
            widths = a['independent_local_bracket_width_v']
            ck(tag + 'independent bracket width consistent with 48 steps', np.all(widths >= 0.) and np.all(widths <= abs(v)*2**-48+2*np.spacing(abs(v))))
            saved = s['independent_native_readout']
            ck(tag + 'independent count ledger', saved['attempted_native_points'] == saved['completed_native_points'] == len(v) and not saved['failures'])
            close(tag + 'independent log diff saved', logdiff, saved['max_log_current_difference_decade'])
            close(tag + 'independent voltage diff saved', vdiff, saved['max_local_voltage_difference_v'])
            all_max['independent_log_current_decade'] = max(all_max['independent_log_current_decade'], logdiff)
            all_max['independent_local_voltage_v'] = max(all_max['independent_local_voltage_v'], vdiff)
            all_max['independent_KCL_a'] = max(all_max['independent_KCL_a'], maximum(ik))
        all_max['native_KCL_a'] = max(all_max['native_KCL_a'], maximum(a['KCL_residual_a']))
        all_max['native_KVL_v'] = max(all_max['native_KVL_v'], maximum(a['KVL_residual_v']))
        all_max['RHS_KCL_a'] = max(all_max['RHS_KCL_a'], s['max_rhs_KCL_residual_a'])
        phase_reports.append({'path': arrays_path, 'phase': name, 'n_native': len(v), 'n_scored': int(mask.sum()),
                              'low20': int(low.sum()), 'initial_u': s['initial_u'], 'terminal_u': s['terminal_u'],
                              'RMSE_decade': s['metrics']['all_eligible']['RMSE_decade'], 'independent': audit['independent_requested']})
    print('All 265 saved physical phases checked', flush=True)

    for r in requests:
        incoming, incoming_command = r['parameters']['initial_u'], 0.
        ck(f"request {r['request_id']} full handoff count", len(r['handoffs']) == len(r['phases']))
        for phase_row, handoff in zip(r['phase_results'], r['handoffs']):
            s, a = phases[phase_row['artifacts']['arrays']['path']]
            ck(f"request {r['request_id']} actual handoff {phase_row['phase']}",
               phase_row['initial_u'] == s['initial_u'] == incoming == a['state_u'][0]
               and phase_row['initial_command_voltage_v'] == s['initial_command_voltage_v'] == incoming_command
               and handoff['passed'] and handoff['expected_initial_u'] == incoming
               and handoff['native_initial_u'] == incoming and handoff['expected_initial_command_voltage_v'] == incoming_command)
            ck('candidate shared parameters every phase', s['parameters'] == {key: r['parameters'][key] for key in ('kappa_per_v2', 'J_positive_a')} | {'lambda_per_v': report['fixed_material']['lambda_per_v']} and s['b'] == r['parameters']['b'])
            incoming, incoming_command = s['terminal_u'], s['final_command_voltage_v']
        scores = r['scores']; metric = scores['phase_metrics']
        ck('score phase keys calibration-only', tuple(metric) == CAL and scores['calibration_only'])
        values, ns = [], []
        for name in CAL:
            p = next(p for p in r['phase_results'] if p['phase'] == name)
            s = phases[p['artifacts']['arrays']['path']][0]
            ck('score metrics from actual phase', metric[name] == s['metrics'])
            values.append(metric[name]['all_eligible']['RMSE_decade']**2)
            ns.append(metric[name]['all_eligible']['n'])
        ck('1170 objective counts', ns == [500, 170, 500])
        close('phase equal weighted objective arithmetic', sum(values)/3, scores['mean_phase_MSE_decade2'])
        close('objective field arithmetic', sum(values)/3, r['objective_mean_phase_MSE'])
        close('1170 RMSE arithmetic', math.sqrt(sum(x*n for x,n in zip(values,ns))/1170), scores['all1170_RMSE_decade'])
        if r['label'].startswith('final_'):
            ck('continuation separate and unselected', set(r['continuation_metrics']) == {'reset5','set6'} and not r['continuation_used_for_selection'])
    eligible = [r for r in objectives if r['finite_objective_available']]
    best = min(eligible, key=lambda r: r['objective_mean_phase_MSE'])
    ck('selected minimum all completed target requests', best['request_id'] == report['selected_candidate']['request_id'] == 86)
    ck('selected exact parameter identity', best['parameters'] == report['selected_candidate']['parameters'])
    for optimizer in report['optimizers']:
        actual = [r for r in objectives if r['block'] == optimizer['block']]
        ck('40 actual optimizer requests '+optimizer['block'], len(actual) == optimizer['scipy_nfev'] == optimizer['objective_requests_in_block'] == 40)
        ck('optimizer not converged', optimizer['scipy_success'] is False and optimizer['global_optimum_claim'] is False)

    labels = {r['label']:r for r in requests}
    comparisons = {}
    for one, two, names, expected in (
        ('control_DOP853','control_Radau',CAL,report['control_regression']['phases']),
        ('final_nominal','final_refined',ALL,report['final_precision_comparison']['phases'])):
        rows = {}
        for name in names:
            ra = next(p for p in labels[one]['phase_results'] if p['phase'] == name)
            rz = next(p for p in labels[two]['phase_results'] if p['phase'] == name)
            sa,a = phases[ra['artifacts']['arrays']['path']]; sz,z = phases[rz['artifacts']['arrays']['path']]
            ck(one+two+name+' data identity', all(np.array_equal(a[key],z[key]) for key in ('command_voltage_v','observed_current_a','original_scoring_mask','path_voltage_s')))
            ck(one+two+name+' same dynamics', sa['parameters'] == sz['parameters'] and sa['b'] == sz['b'])
            mask = a['original_scoring_mask']
            ld,ud = maximum(np.log10(abs(a['current_a'][mask]/z['current_a'][mask]))),maximum(a['state_u']-z['state_u'])
            ck(one+two+name+' precision gates',ld<=1e-4 and ud<=1e-3)
            saved = expected[name]['cross_method'] if one.startswith('control') else expected[name]
            close('saved precision current difference',ld,saved['max_log_current_difference_decade'])
            close('saved precision state difference',ud,saved['max_state_u_difference'])
            rows[name]={'max_log_current_difference_decade':ld,'max_state_u_difference':ud}
        comparisons[one+'_vs_'+two]=rows
    ck('final common overall initial state', labels['final_nominal']['parameters'] == labels['final_refined']['parameters'] == best['parameters'])
    ck('3710 saved independent native roots, 3680 nonzero', independent_native_total==3710 and independent_nonzero_total==3680)

    timing = report['timing']; summed = defaultdict(float)
    for call in timing['calls']:
        summed[call['block']] += call['elapsed_s']
        ck('all actual numerical calls completed', call['status']=='completed')
        if 'per_call_limit_s' in call:
            ck('per call budget',0<=call['elapsed_s']<call['per_call_limit_s'])
    for block,used in timing['used_s'].items():
        close('timing sum '+block,summed[block],used)
        ck('block budget '+block,used<timing['limits_s'][block])
    fit=sum(timing['used_s'][k] for k in ('setup','new_optimizer','old_optimizer'))
    close('fitting time sum',fit,timing['fit_total_s'])
    ck('total numerical budget',fit<300 and fit+timing['used_s']['final']<400 and not timing['stopped_blocks'])
    control,final=labels['control_Radau'],labels['final_refined']
    score_comparison={'control':control['scores'],'final_refined':final['scores'],
        'mean_phase_MSE_relative_reduction':1-final['scores']['mean_phase_MSE_decade2']/control['scores']['mean_phase_MSE_decade2'],
        'all1170_RMSE_relative_reduction':1-final['scores']['all1170_RMSE_decade']/control['scores']['all1170_RMSE_decade'],
        'continuation_metrics_separate':final['continuation_metrics']}
    review={'status':'passed_with_startup_source_provenance_qualification' if not failures else 'failed',
        'checks_completed':checks,'failures':failures,'review_script_sha256':digest(Path(__file__).read_bytes()),
        'terminal_result_sha256':digest(result_bytes),'source_provenance_qualification':source_qualification,
        'actual_source_count':len(sources),'source_hash_checks':sources,'artifact_count':len(artifact_records),
        'artifact_hash_checks':artifact_records,'request_status_counts':dict(statuses),'objective_requests':len(objectives),
        'unique_saved_phases_audited':len(phases),'accepted_segments_audited':accepted_segments,
        'final_saved_RHS_rows_audited':rhs_total,'independent_saved_native_points':independent_native_total,
        'independent_nonzero_48step_points':independent_nonzero_total,'selected_request':best['request_id'],
        'selected_parameters':best['parameters'],'phase_reviews':phase_reports,'maxima':dict(all_max),
        'precision_comparisons':comparisons,'timing_used_s':timing['used_s'],'total_numerical_s':fit+timing['used_s']['final'],
        'same_precision_control_comparison':score_comparison,
        'optimizers_converged':False,'new_audit_roots':0,'new_audit_ODEs':0,'new_audit_inversions':0,
        'scope':'All generated artifacts and all 265 unique phase arrays in this new stage; source hashes only for earlier files. No model imports, new roots, ODEs, inversions, fits, retries or old-stage archive traversal.',
        'limitations':['Both 40-call local optimizers reached their evaluation caps without convergence; no global or family-wide exclusion.',
            'Equal-phase objective improved with worse SET4; original 46-pair/27023-residual/20-gate contract is not evaluated.',
            'RESET5/SET6 are training continuation checks excluded from ranking, not original holdout or independent pristine predictions.',
            'Startup runner changed only in exceptional report handling; input/end hash equality cannot establish frozen source throughout startup.']}
    destination=HERE/('ACTUAL_RESULT_REVIEW.json' if not failures else 'ACTUAL_RESULT_REVIEW_FAILURE.json')
    with destination.open('x') as handle:json.dump(review,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write('\n')
    if failures:raise AssertionError(failures)
    rows='\n'.join(f"| {name.upper()} | {control['scores']['phase_metrics'][name]['all_eligible']['RMSE_decade']:.9f} | {final['scores']['phase_metrics'][name]['all_eligible']['RMSE_decade']:.9f} |" for name in CAL)
    text=f'''# 连续共同参数校准：保存结果独立审计

结论：**数值与账本检查通过，启动源码来源附明确限定**。{checks} 项检查完成；{len(sources)} 项实际来源哈希、622 项生成产物哈希相符，265 个独立相位的保存数组均已审计。未执行模型或重新求根、反演、ODE、拟合。

## 实际结果与改进范围

最终 refined 与实际 control_Radau 使用相同数值精度比较：

| 相位 | 实际 control RMSE / decade | 最终 refined RMSE / decade |
|---|---:|---:|
{rows}

等相位 MSE 从 {control['scores']['mean_phase_MSE_decade2']:.12f} 降到 {final['scores']['mean_phase_MSE_decade2']:.12f}，下降 {100*score_comparison['mean_phase_MSE_relative_reduction']:.3f}%；原 1170 点加权 RMSE 从 {control['scores']['all1170_RMSE_decade']:.9f} 降到 {final['scores']['all1170_RMSE_decade']:.9f}。**SET4 变差，RESET4 与 SET5 改善，这是共同参数下的曲线折中，并非统一模型完成。**

训练内延续 RESET5/SET6 的 RMSE 分别为 {final['continuation_metrics']['reset5']['all_eligible']['RMSE_decade']:.9f} / {final['continuation_metrics']['set6']['all_eligible']['RMSE_decade']:.9f}；这 670 点没有参与排名，不是原保留集或全新独立预测，不合并为新目标分数。

## 请求、状态与数值验证

88 条记录为 85 次目标请求、1 次 DOP853 控制、2 条最终精度链；85 次目标含 1 次精确缓存命中。两个局部优化器各完成 40 次请求，均因评估次数上限而未收敛。请求86（第85次目标评估）是全部已完成目标评估的最低值；选择依据仅为前三相位 1170 点的等相位 MSE，未使用延续误差。该候选可优于优化器终止时返回的 simplex 摘要，不能仅用 scipy_fun 代替真实请求账本。

265 个独立保存相位（缓存复用未重复计数）、{accepted_segments} 个已接受积分段、全部原生 mask 与低压20点检查通过。每相位都实际传递前一终态和命令电压，没有重锚；两条最终五相位链共享同一参数与整体初态。全阶段无失败或超时记录。

最终两精度保存 RHS 共 {rhs_total} 行，独立代数与端口检查通过；独立读出覆盖 3710 个原生点，其中3680个非零点各保存48步二分记录，零压点原样保留。两精度差、独立读出差、KCL/KVL、q/Γ有限域与近零微段计数全部通过。近零预算是局部遗漏界，未将其称为全轨迹误差证明。

实际数值时间：setup {timing['used_s']['setup']:.6f} s，新起点 {timing['used_s']['new_optimizer']:.6f} s，旧起点 {timing['used_s']['old_optimizer']:.6f} s，final {timing['used_s']['final']:.6f} s；总 {review['total_numerical_s']:.6f} s，符合预声明限额。哈希、序列化与文件I/O按原约定另计。

## 启动源码来源限定

审核/启动快照 SHA 为 `{digest(snapshot)}`；运行保存的输入哈希与当前文件 SHA 为 `{digest(current)}`。逐字节差异仅为顶层异常出口增加 `numerical_validation_passed=False`，不改变正常数值路径、方程、数据、参数、预算或排名。实际本次正常完成，无该异常出口记录。

**不能声称整个启动期间源码冻结。** `source_files_unchanged=True` 只支持输入哈希采集之后到结束的文件一致性；不能据此证明启动执行源码从未变化。重建快照精确匹配先前审核哈希，时序说明原样保留；本审计未通过重跑掩盖这一限制。

本轮仍是限定参数域与两起点预算内的有限试验。优化未收敛，不能排除同方程全部参数族；没有运行裸1R或完整46对/27023残差/20项聚合合同。
'''
    with (HERE/'ACTUAL_RESULT_REVIEW.md').open('x') as handle:handle.write(text)
    print(json.dumps({'status':review['status'],'checks':checks,'phases':len(phases),'artifacts':len(artifact_records),
        'sources':len(sources),'RHS_rows':rhs_total,'total_numerical_s':review['total_numerical_s']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
