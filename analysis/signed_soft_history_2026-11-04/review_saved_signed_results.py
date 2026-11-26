"""Saved-result arithmetic only. No material/kernel/runner imports or new solves."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from io import BytesIO
from pathlib import Path
import hashlib
import json
import math
import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parent/'cooperative_relaxation_2026-11-04'


def main():
    content = {}
    catalog_checked = {}
    report = json.loads((HERE/'bounded_signed_trial.json').read_text())
    assert report['status'] == 'completed_bounded_signed_attempt_no_full_model_claim'
    catalog = report['artifact_catalog']

    def raw(path):
        path = str(Path(path).resolve())
        if path not in content:
            content[path] = Path(path).read_bytes()
        value = content[path]
        if path in catalog:
            assert len(value) == catalog[path]['bytes']
            assert hashlib.sha256(value).hexdigest() == catalog[path]['sha256']
            catalog_checked[path] = True
        return value

    def js(path):
        return json.loads(raw(path))

    def npz(path):
        with np.load(BytesIO(raw(path)), allow_pickle=False) as archive:
            return {key: archive[key].copy() for key in archive.files}

    def close(a, b, atol=2e-12, rtol=2e-12):
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)

    def load_source(item):
        path, expected = item
        payload = Path(path).read_bytes()
        return path, payload, hashlib.sha256(payload).hexdigest() == expected

    with ThreadPoolExecutor(max_workers=4) as pool:
        loaded = list(pool.map(load_source, report['source_sha256'].items()))
    source_checks = {}
    for path, payload, passed in loaded:
        content[str(Path(path).resolve())] = payload
        source_checks[path] = passed
    assert all(source_checks.values())
    print(f'Actual {len(source_checks)} source hashes verified.', flush=True)
    plan = js(HERE/'bounded_signed_trial.plan.json')
    assert plan == report['plan']
    protocol = js(HERE/'DATA_PROTOCOL_AUDIT.json')
    observed_cache = npz(HERE/'training_reset4_set5_native.npz')
    old_summary = js(OLD/'final_refined.json')
    old_arrays = npz(OLD/'final_refined.npz')
    frozen_initial = old_summary['terminal_u']
    parameters = old_summary['parameters']
    lam, kappa, jref = [parameters[k] for k in ('lambda_per_v','kappa_per_v2','J_positive_a')]
    gc = .03127031375421752
    with np.load(BytesIO(raw('/private/tmp/vulcan-runtime-cache/transport_g010_v0025.npz'))) as table:
        lg = table['log_gamma'].copy(); delta = math.exp(float(table['log_voltage'][0]))
    trials = report['trials']
    assert len(trials) == report['phase_ODE_attempts'] == 15
    assert all(t['status'] == 'completed' and t['phase_call_started'] for t in trials)
    search = [t for t in trials if t['search_objective']]
    assert len(search) == report['search_objective_calls'] == 10
    assert all(t['phase'] == 'reset4' and t['initial_u'] == frozen_initial for t in search)
    assert len([t for t in search if t['label'] == 'bounded_b']) == report['optimizer_objective_calls'] == report['optimizer']['nfev'] == 8
    assert [search[0]['b'],search[1]['b']] == plan['b_domain']
    assert all(0 <= t['b'] <= math.log(10.) for t in search)
    best = min(search,key=lambda t:t['objective_MSE'])
    assert best['trial_id'] == report['selected_b']['source_trial'] == '010_bounded_b'
    assert best['b'] == report['selected_b']['b'] == report['optimizer']['b']
    bstar = best['b']
    assert report['SET5_used_for_b_selection'] is False and report['selected_b']['SET5_used'] is False
    segments = 0
    for trial in trials:
        assert js(HERE/'phase_trials'/f"{trial['trial_id']}.json") == trial
        assert trial['summary']['parameters'] == parameters and trial['summary']['b'] == trial['b']
        assert trial['summary']['initial_u'] == trial['initial_u']
        close(trial['summary']['metrics']['all_eligible']['RMSE_decade']**2,trial['objective_MSE'])
        assert len(trial['summary']['segments']) == 2
        assert all(s['status']=='completed' and s['success'] for s in trial['summary']['segments'])
        segments += len(trial['summary']['segments'])
    times = report['timing']['numerical_calls']
    close(sum(c['elapsed_s'] for c in times),report['timing']['numerical_total_s'])
    close(sum(c['elapsed_s'] for c in times if c['search']),report['timing']['numerical_search_s'])
    assert all(c['status']=='completed' for c in times)
    assert report['timing']['numerical_total_s'] < 240 and report['timing']['numerical_search_s'] < 120
    assert all(c['per_call_limit_s'] is None or c['elapsed_s'] < c['per_call_limit_s'] for c in times)

    def point_checks(a, i, h, t, kcl, kvl):
        assert all(np.all(np.isfinite(x)) for x in (a,i,h,t,kcl,kvl))
        assert np.all(i*a >= 0) and np.all(i*h >= 0) and np.all(i*t >= 0)
        assert np.all(a*h >= 0) and np.all(a*t >= 0)
        assert max(abs(a-h-t),default=0.) <= 1e-12
        assert max(abs(kcl),default=0.) <= 1e-13 and max(abs(kvl),default=0.) <= 1e-12
        assert np.all(abs(t[a<0])<=2.) and np.all(abs(t[a>0])<=5.)
        nonzero = a != 0
        assert np.all(i[nonzero]*a[nonzero]>0) and np.all(h[nonzero]*a[nonzero]>0)
        assert np.all(i[~nonzero]==0) and np.all(h[~nonzero]==0) and np.all(t[~nonzero]==0)

    static = report['static_validation']
    assert static['n_reference_points'] == 12 and len(static['rows']) == 24 and not static['failures']
    for row in static['rows']:
        point = row['point']
        point_checks(*[np.array([point[k]]) for k in ('command_voltage_v','current_a','local_voltage_v','transistor_voltage_v','KCL_residual_a','KVL_residual_v')])
    for mode, summary in static['modes'].items():
        selected = [row for row in static['rows'] if row['mode']==mode]
        assert len(selected)==12 and summary['passed'] and summary['full_physical_passed']
        for key in ('log_current_difference_decade','local_voltage_difference_v','KCL_residual_a','KVL_residual_v'):
            close(max(row[key] for row in selected),summary[key],atol=0.)
    for row in report['RHS_algebra_validation']['rows']:
        source = next(x for x in static['rows'] if x['mode']==report['selected_circuit_mode'] and x['command_v']==row['command_v'] and x['state_u']==row['state_u'])
        point, state, b = source['point'], row['state_u'], row['b']
        v, i, sign = abs(point['local_voltage_v']), abs(point['current_a']), np.sign(point['local_voltage_v'])
        rate = kappa*v*(math.exp((1-sign)*b+lam*v-state)-(i/jref)*math.exp((sign-1)*b)*math.expm1(state))
        close(rate,row['actual_signed_u_rate'])
        assert abs(rate-row['unified_q_transformed_u_rate']) <= row['tolerance']

    phase_results, loaded_arrays = {}, {}
    def review_phase(label, data, summary, trial, expected_entry):
        name = trial['phase']; v = data['command_voltage_v']; obs = data['observed_current_a']
        if name == 'set4':
            ev, eo = old_arrays['command_voltage_v'], old_arrays['observed_current_a']
        else:
            prefix = '1t1r_cycle_'+('4_reset' if name=='reset4' else '5_set')
            ev, eo = observed_cache[prefix+'_voltage'],observed_cache[prefix+'_current']
        np.testing.assert_array_equal(v,ev); np.testing.assert_array_equal(obs,eo)
        mask = np.isfinite(abs(obs)) & (abs(obs)>=(1e-12 if name=='reset4' else 1e-11)) & (abs(v)>1e-12)
        np.testing.assert_array_equal(data['original_scoring_mask'],mask)
        assert int(mask.sum()) == (170 if name=='reset4' else 500)
        path = np.cumsum(abs(np.diff(np.r_[0.,v])))
        close(data['path_voltage_s'],path,atol=1e-12,rtol=0.)
        u,q = data['state_u'],data['state_q']; current = data['current_a']
        np.testing.assert_array_equal(q,np.exp(-u))
        loggamma=np.log(gc/np.expm1(u)); assert np.all((loggamma>=lg[0])&(loggamma<=lg[-1]))
        assert np.all((q>0)&(q<1))
        point_checks(v,current,data['local_voltage_v'],data['transistor_voltage_v'],data['KCL_residual_a'],data['KVL_residual_v'])
        assert u[0] == expected_entry == summary['initial_u']
        close(u[-1],summary['terminal_u'],atol=1e-12,rtol=0.)
        repeats=np.flatnonzero(np.diff(v)==0)
        assert np.all(u[repeats+1]==u[repeats])
        previous=expected_entry
        for segment in summary['segments']:
            k=segment['index']; times=data[f'segment_{k}_coordinate']; states=data[f'segment_{k}_state_u']
            assert states[0]==segment['initial_u']==previous
            assert states[-1]==segment['final_u']
            close([times[0],times[-1]],[segment['path_start_v'],segment['path_stop_v']],atol=0.)
            assert np.all(np.diff(times)>0) and len(times)==segment['accepted_steps']+1
            np.testing.assert_array_equal(data[f'segment_{k}_state_q'],np.exp(-states))
            previous=segment['final_u']
        assert previous==summary['terminal_u']
        errors=np.full(len(v),np.nan); errors[mask]=np.log10(abs(current[mask])/abs(obs[mask]))
        close(data['log10_current_residual'],errors)
        mse=float(np.mean(errors[mask]**2)); close(mse,trial['objective_MSE'])
        close(math.sqrt(mse),summary['metrics']['all_eligible']['RMSE_decade'])
        out=np.arange(len(v))<=np.argmax(abs(v))
        groups={'all_eligible':mask,'outward':mask&out,'return':mask&~out,
                'outward_low_0p2':mask&out&(abs(v)<=.2+1e-12),'return_low_0p2':mask&~out&(abs(v)<=.2+1e-12),
                'both_branches_low_0p2':mask&(abs(v)<=.2+1e-12),'remaining_above_0p2':mask&(abs(v)>.2+1e-12)}
        calculated={k:dict(n=int(use.sum()),RMSE_decade=float(np.sqrt(np.mean(errors[use]**2))),
                           SSE_fraction=float(np.sum(errors[use]**2)/np.sum(errors[mask]**2))) for k,use in groups.items()}
        if 'original_mask_diagnostics' in summary:
            for key,value in calculated.items():
                close(value['RMSE_decade'],summary['original_mask_diagnostics'][key]['RMSE_decade'])
                assert value['n']==summary['original_mask_diagnostics'][key]['n']
        rhs_delta=None
        if 'rhs_trace' in data:
            trace=data['rhs_trace']; assert len(trace)==summary['n_rhs_calls']
            finite=np.isfinite(trace).all(axis=1); omissions=trace[~finite]
            assert len(omissions)==summary['smallfield_omission']['n_RHS_omissions']
            assert np.all((abs(omissions[:,0])<delta)&(omissions[:,0]!=0)) and np.all(omissions[:,5]==0)
            a,s,i,h,t,d,k=trace[finite].T
            point_checks(a,i,h,t,k,a-h-t)
            sign=np.sign(h); b=trial['b']; vv=abs(h)
            rhs=kappa*vv*(np.exp((1-sign)*b+lam*vv-s)-(abs(i)/jref)*np.exp((sign-1)*b)*np.expm1(s))
            rhs_delta=float(max(abs(rhs-d),default=0.)); assert rhs_delta<1e-10
            expected_bounds={str(sign):.5*kappa*delta**2*(math.exp((1-sign)*b+lam*delta)+math.pi*gc/(58.463033350725695*jref)*math.exp((sign-1)*b)) for sign in (-1,1)}
            for sign,value in summary['smallfield_bounds_by_sign'].items():close(value,expected_bounds[sign],atol=0.)
            counted=summary['smallfield_omission']['physical_microsegment_indices']; assert counted==[0,1]
            sign=summary['segments'][0]['sign']; close(summary['smallfield_omission']['local_bound_sum'],2*expected_bounds[str(sign)],atol=0.)
        legacy_result=None
        if 'native_legacy_audit' in data:
            legacy=data['native_legacy_audit']; assert legacy.shape==(len(v),5)
            np.testing.assert_array_equal(legacy[:,0],np.arange(len(v)))
            point_checks(v,legacy[:,1],legacy[:,2],legacy[:,3],legacy[:,4],v-legacy[:,2]-legacy[:,3])
            legacy_result=dict(max_log_current_difference_decade=float(max(abs(np.log10(abs(current[mask]/legacy[mask,1]))))),
                               max_local_voltage_difference_v=float(max(abs(data['local_voltage_v']-legacy[:,2]))),
                               max_reference_KCL_residual_a=float(max(abs(legacy[:,4]))))
            for key,value in legacy_result.items():close(value,summary['native_legacy_audit'][key],atol=0.)
        diagnostics={}
        if name=='reset4':
            indices=np.flatnonzero(out); peak=int(indices[np.argmax(abs(current[indices]))]); obspeak=int(indices[np.argmax(abs(obs[indices]))])
            tail=int(np.flatnonzero((~out)&np.isclose(abs(v),.2,rtol=0,atol=1e-12))[0])
            diagnostics=dict(predicted_peak_index=peak,predicted_peak_a=float(abs(current[peak])),predicted_peak_command_v=float(abs(v[peak])),
                             observed_peak_index=obspeak,observed_peak_a=float(abs(obs[obspeak])),observed_peak_command_v=float(abs(v[obspeak])),
                             tail_index=tail,predicted_tail_a=float(abs(current[tail])),observed_tail_a=float(abs(obs[tail])))
        else:
            crossings=[]
            for threshold in (1e-6,1e-5):
                item={'threshold_a':threshold}
                for key,trace in [('predicted',abs(current)),('observed',abs(obs))]:
                    hits=np.flatnonzero(out&mask&(trace>=threshold))
                    item[key]=[float(v[hits[0]-1]),float(v[hits[0]])] if len(hits) else None
                crossings.append(item)
            diagnostics=dict(crossings=crossings,outward_0p2_predicted_a=float(abs(current[10])),outward_0p2_observed_a=float(abs(obs[10])))
        phase_results[label]=dict(phase=name,b=trial['b'],incoming_u=expected_entry,terminal_u=summary['terminal_u'],
            groups=calculated,max_rhs_algebra_difference=rhs_delta,legacy=legacy_result,
            max_native_KCL_a=float(max(abs(data['KCL_residual_a']))),max_native_KVL_v=float(max(abs(data['KVL_residual_v']))),
            max_rhs_KCL_a=summary['max_rhs_KCL_residual_a'],state_u_range=[float(min(u)),float(max(u))],diagnostics=diagnostics)
        loaded_arrays[label]=data
        print('Verified saved arrays:',label,flush=True)

    regression_trial=trials[0]
    regression=npz(HERE/'phase_trials'/f"{regression_trial['trial_id']}.npz")
    review_phase('SET4_regression',regression,regression_trial['summary'],regression_trial,old_summary['derived_initial_u'])
    rmask=regression['original_scoring_mask']
    regression_diff=dict(max_log_current_difference_decade=float(max(abs(np.log10(abs(regression['current_a'][rmask]/old_arrays['current_a'][rmask]))))),
                         max_state_u_difference=float(max(abs(regression['state_u']-old_arrays['state_u']))))
    for k,v in regression_diff.items():close(v,report['positive_regression'][k],atol=0.)
    best_arrays=npz(HERE/'phase_trials'/f"{best['trial_id']}.npz")
    review_phase('selected_RESET4_search',best_arrays,best['summary'],best,frozen_initial)
    for precision in ('nominal','refined'):
        pair=next(x for x in report['final_pairs'] if x['precision']['name']==precision)
        incoming=frozen_initial
        for phase in ('reset4','set5'):
            label=f'final_{precision}_{phase}'
            summary=js(HERE/f'{label}.json'); data=npz(HERE/f'{label}.npz')
            assert summary==pair['phases'][phase]['summary']
            trial=next(t for t in trials if t['label']==label)
            assert trial['b']==bstar and trial['initial_u']==incoming
            review_phase(label,data,summary,trial,incoming)
            incoming=summary['terminal_u']
        assert pair['phase_state_join_error']==pair['phase_command_join_error_v']==0.
    precision_results={}
    for name in ('reset4','set5'):
        a,z=[loaded_arrays[f'final_{precision}_{name}'] for precision in ('nominal','refined')]
        mask=a['original_scoring_mask']
        differences=dict(max_log_current_difference_decade=float(max(abs(np.log10(abs(a['current_a'][mask]/z['current_a'][mask]))))),
                         max_state_u_difference=float(max(abs(a['state_u']-z['state_u']))))
        for k,v in differences.items():close(v,report['precision_comparison'][name][k],atol=0.)
        assert differences['max_log_current_difference_decade']<=1e-4 and differences['max_state_u_difference']<=1e-3
        precision_results[name]=differences
    diagnostic=phase_results['final_refined_reset4']['diagnostics']
    peak_ratio=diagnostic['observed_peak_a']/diagnostic['predicted_peak_a']
    tail_ratio=diagnostic['predicted_tail_a']/diagnostic['observed_tail_a']
    output=dict(scope='No new roots/inversions/ODEs/fits/scans; all15 trial metadata, selected search + regression + four final array sets only',
        actual_source_hash_checks=source_checks,catalog_files_read_and_hash_checked=catalog_checked,
        phase_attempts=15,completed_segments=segments,search_calls=10,bounded_optimizer_calls=8,selected_b=bstar,
        selected_trial=best['trial_id'],selected_from_RESET4_only=True,all_trial_JSON_snapshots_match=True,
        positive_regression=regression_diff,static_12_points_and_24_RHS_algebra_passed=True,
        phases=phase_results,two_precision=precision_results,
        conditional_tradeoff=dict(observed_peak_over_predicted=peak_ratio,predicted_tail_over_observed=tail_ratio,
                                  minimum_worst_feature_factor=min(peak_ratio,tail_ratio)),
        numerical_validation_reproduced=True,physical_accuracy_not_established=True)
    (HERE/'SIGNED_ACTUAL_RESULT_AUDIT.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k not in ('actual_source_hash_checks','catalog_files_read_and_hash_checked')},indent=2),flush=True)


if __name__=='__main__':
    main()
