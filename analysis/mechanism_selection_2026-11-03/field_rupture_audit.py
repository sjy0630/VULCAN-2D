"""Read-only numerical audit of the completed trial; never fits or simulates.

Writes separate audit files; the executed trial source, plan and results stay
unchanged. Branch-product diagnostics use its already fitted background and
are descriptive, not a new parameter search or an uncertainty-aware test.
"""
from pathlib import Path
import hashlib
import json
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-15)


def main():
    trial = HERE / 'field_rupture_trial.json'
    r = json.loads(trial.read_text())
    plan = json.loads(trial.with_suffix('.plan.json').read_text())
    assert r['predeclared_plan'] == plan
    assert r['status'] == 'completed_predeclared_conditional_RESET_trial'
    assert r['fit_calls'] == 90
    for name in ('field_rupture_trial.py', 'field_rupture_trial.plan.json'):
        p = HERE / name
        assert hashlib.sha256(p.read_bytes()).hexdigest() == r['source_sha256'][str(p.relative_to(ROOT))]
    with np.load(trial.with_suffix('.npz'), allow_pickle=False) as archive:
        z = {key: archive[key] for key in archive.files}
    ids = z['record_ids'].tolist()
    assert ids == plan['training_ids']
    v = abs(z['command_voltage_v'])
    apex = int(np.argmax(v))
    outward = np.arange(len(v)) <= apex
    obs = z['observed_reset_abs_current_a']
    mask = z['original_reset_mask']
    np.testing.assert_array_equal(mask, np.isfinite(obs) & (obs >= 1e-12) & (v[None, :] > 1e-12))
    assert mask.sum() == 2799 and (mask & outward).sum() == 1400 and (mask & ~outward).sum() == 1399
    np.testing.assert_array_equal(v, v[::-1])
    set_g = z['prior_SET_conductance_s']
    result = dict(scope='Pure arithmetic verification of saved artifacts; no fitting, new model simulation, bounds expansion or held-out data access.',
                  LOO_scope_correction='Only the three new RESET parameters are refitted within each fold. Ron=58.463033350725695 ohm was selected using all14 training records and the spectral shape was selected previously. Thus this is conditional LOO under a fixed full-training preselected backbone, not leakage-free end-to-end cross-validation, original-holdout validation or chronological prediction.',
                  source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                 (trial, trial.with_suffix('.npz'), trial.with_suffix('.plan.json'), HERE/'field_rupture_trial.py')},
                  mask_counts=dict(all=2799, outward=1400, return_=1399, per_record=mask.sum(axis=1).tolist()),
                  models={})
    nstarts = 0
    for name, model in r['models'].items():
        assert len(model['LOO_folds']) == 14
        folds = [model['full_training_fit']] + model['LOO_folds']
        for fold in folds:
            excluded = fold['excluded_record']
            expected = [cid for cid in ids if cid != excluded]
            assert fold['fit_records'] == expected
            expected_g = float(np.exp(np.mean(np.log(set_g[[ids.index(cid) for cid in expected]]))))
            close(fold['fit_SET_geomean_conductance_s'], expected_g)
            assert len(fold['starts']) == 3
            for start, declared in zip(fold['starts'], plan['fixed_starts']):
                assert start['start'] == declared
                assert start['solver_success'] and start['status'] == 'finite_bounded_result'
                theta = np.asarray(start['parameters'])
                assert np.all(theta >= plan['lower']) and np.all(theta <= plan['upper'])
                assert start['n_residuals'] == sum(mask[ids.index(cid)].sum() for cid in expected)
                nstarts += 1
            assert not fold['selection_uses_excluded_RESET']
            if excluded is not None:
                cond = fold['test_conditioning']
                assert cond['used_test_SET'] == (name == 'prior_SET_conditioned')
                assert cond['test_RESET_used_only_for_final_scoring']
                g = set_g[ids.index(excluded)] if cond['used_test_SET'] else expected_g
                close(cond['input_conductance_s'], g)
                assert not cond['clipped_to_zero'] and cond['initial_read_error_vs_conditioning_g_s'] == 0
        assert model['training_initial_state_clipping']['n'] == 0
        assert model['training_initial_state_clipping']['max_initial_read_conductance_error_s'] == 0
        m = dict(all_fold_exclusions_and_three_fixed_starts_verified=True,
                 all_initial_state_clipping_counts=0, all_initial_conditioning_errors_s=0,
                 recomputed_metrics={}, feature_gates={}, physical_array_checks={})
        result['models'][name] = m
        for split in ('training', 'LOO'):
            pre = name + '_' + split
            cur, a, a0, dose, leak, theta = [z[pre+'_'+key] for key in
                                            ('abs_current', 'a', 'a0', 'D', 'leak_abs_current', 'parameters')]
            for arr in (cur, a, a0, dose, leak):
                assert np.all(np.isfinite(arr)) and np.all(arr >= 0)
            assert np.min(np.diff(dose, axis=-1)) >= -1e-12
            assert np.max(np.diff(a, axis=-1)) <= 1e-12
            k = 10**theta[..., 1]
            close(a, a0[:, None] * np.exp(-k[..., None]*dose))
            close(cur, leak + a*v)
            assert np.all(cur[mask] > 0)
            m['physical_array_checks'][split] = dict(finite_nonnegative=True, dose_nondecreasing=True,
                                                    path_conductance_nonincreasing=True, analytic_decay_and_additive_current_verified=True)
            m['recomputed_metrics'][split] = {}
            for group, eligible in (('all_eligible', mask), ('outward', mask & outward), ('return', mask & ~outward)):
                e = np.log10(cur[eligible])-np.log10(obs[eligible])
                scored = dict(n=int(eligible.sum()), raw_RMSE_decade=float(np.sqrt(np.mean(e**2))),
                              median_absolute_factor_error=float(10**np.median(abs(e))))
                for key, value in scored.items():
                    close(value, model[split+'_metrics'][group][key])
                m['recomputed_metrics'][split][group] = scored
            m['feature_gates'][split] = {}
            for feature, saved in model[split+'_feature_gates']['features'].items():
                values = []
                for row, detail in zip(cur, saved['record_values']):
                    if '.R_fit_' in feature:
                        j = np.array(detail['indices'])
                        value = float(np.dot(v[j],v[j])/np.dot(v[j],row[j]))
                    else:
                        j = detail['native_index']
                        value = float(v[j] if feature.endswith('.peak_abs_voltage') else row[j])
                    close(value, detail['value'])
                    values.append(value)
                value = float(np.median(values))
                close(value, saved['model_median'])
                low, high = saved['original_aggregate_interval']
                assert saved['aggregate_gate_passed'] == (low <= value <= high)
                m['feature_gates'][split][feature] = dict(value=value, interval=[low,high], passed=low <= value <= high)
        pre = name + '_training'
        theta = z[pre+'_parameters']; a0 = z[pre+'_a0']; a = z[pre+'_a']; leak = z[pre+'_leak_abs_current']
        tail_index = model['training_feature_gates']['features']['1r.reset.return.I_0p2V']['record_values'][0]['native_index']
        pure_peak = (theta[0]+np.sqrt(theta[0]**2+4/10**theta[1]))/2
        path_tail = a[:,tail_index]*v[tail_index]
        m['structural_diagnostics'] = dict(Vr_v=float(theta[0]), kappa_per_v2=float(10**theta[1]),
            pure_path_continuous_peak_v=float(pure_peak),
            tail_voltage_v=float(v[tail_index]), frozen_background_tail_a=float(leak[tail_index]),
            median_path_tail_a=float(np.median(path_tail)),
            median_background_fraction_of_tail=float(np.median(leak[tail_index]/z[pre+'_abs_current'][:,tail_index])),
            median_a0_s=float(np.median(a0)),
            interpretation='At this fitted parameter vector the tail is background limited, not residual-path limited. The fit has no identified positive field threshold.')
        # Diagnostic of the necessary positive-path condition under the *fixed*
        # fitted background. Raw noise/negative residuals are retained, not logged.
        all_eligible = mask & (v[None,:] > 0)
        m['fixed_background_data_diagnostic'] = dict(
            n_eligible_observations_below_background=int(np.sum(all_eligible & (obs < leak))),
            n_eligible_return_observations_below_background=int(np.sum(all_eligible & ~outward & (obs < leak))),
            warning='Descriptive mismatch of the already fitted common background; not a formal uncertainty-aware rejection of every background value or every exponential-decay law.')
    assert nstarts == 90
    result['verified_start_runs'] = nstarts
    result['next_falsifiable_conditions'] = dict(
        model_scope='Any nonnegative common frozen background b(v) plus path a*v, with common history-independent nonnegative hazard lambda(v) per unit path-voltage, symmetric outward/return sweep, no wait.',
        background_positivity='b(v) <= observed_abs_I(v) within measurement uncertainty; otherwise inferred path conductance is negative.',
        inferred_path='A_out(v)=[I_out(v)-b(v)]/v, A_return(v)=[I_return(v)-b(v)]/v, where both are positive.',
        branch_product='log A_out(v)+log A_return(v)=2 log a0-2 H(Vmax), independent of matched v within each record; H(v)=integral_0^v lambda(w) dw.',
        shared_normalized_outward='log[A_out,j(v)/a0,j]=-H(v), identical across records up to uncertainty; cannot be repaired by changing only a0.',
        linear_field_specialization='d log A_out/dv=-kappa*(v-Vr)_+; deembedded log path has nonpositive slope and constant curvature -kappa above Vr.',
        exact_total_peak_condition='At an interior total-current maximum: lambda(vp)=[1+b_prime(vp)/A_out(vp)]/vp. The simpler lambda(vp)=1/vp applies only to a path-only peak.',
        tail_condition='b(vt)<I_tail_max and 2H(Vmax)-H(vt) >= log[a0*vt/(I_tail_max-b(vt))] when the RHS is positive.',
        interpretation='First test these necessary shape/history conditions using credible background and measurement uncertainty. If violated, changing kappa/Vr cannot repair the assumed common voltage-only hazard with frozen background; this does not select a microscopic replacement mechanism.')
    (HERE/'field_rupture_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='all_saved_artifact_checks_passed_no_refit',start_runs=nstarts,
                         diagnostics={name:m['structural_diagnostics'] for name,m in result['models'].items()},
                         fixed_background={name:m['fixed_background_data_diagnostic'] for name,m in result['models'].items()}),indent=2))


if __name__ == '__main__':
    main()
