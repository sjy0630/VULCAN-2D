"""Post-run arithmetic only; no model imports, roots, inversions or fitting."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    sources = {}

    def read(path, npz=False):
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        if npz:
            with np.load(path, allow_pickle=False) as z:
                return {k: z[k].copy() for k in z.files}
        return json.loads(path.read_text())

    d = read(HERE / 'bounded_joint_calibration.json')
    assert d['status'] == 'completed_bounded_joint_diagnostic'
    rows = {r['label']: r for r in d['requests']}
    old, final = rows['control_Radau'], rows['final_refined']
    scores = {}
    for name in ('set4', 'reset4', 'set5'):
        a = old['scores']['phase_metrics'][name]['all_eligible']['RMSE_decade']
        b = final['scores']['phase_metrics'][name]['all_eligible']['RMSE_decade']
        scores[name] = {'old_RMSE_decade': a, 'new_RMSE_decade': b,
                        'relative_RMSE_change_percent': 100*(b/a-1),
                        'new_10_power_RMSE': 10**b}
    aggregate = {}
    for name in ('mean_phase_MSE_decade2', 'root_mean_phase_MSE_decade', 'all1170_RMSE_decade'):
        a, b = old['scores'][name], final['scores'][name]
        aggregate[name] = {'old': a, 'new': b, 'decrease_percent': 100*(1-b/a)}
    arrays = {}
    summaries = {}
    for row in final['phase_results']:
        name = row['phase']
        arrays[name] = read(Path(row['artifacts']['arrays']['path']), True)
        summaries[name] = read(Path(row['artifacts']['summary']['path']))
    features = {}
    for name, a in arrays.items():
        v = abs(a['command_voltage_v'])
        pivot = np.argmax(v)
        mask = a['original_scoring_mask']
        i, obs = abs(a['current_a']), abs(a['observed_current_a'])
        out = np.arange(len(v)) <= pivot
        f = {'native_peak_predicted_a': float(max(i)), 'native_peak_observed_a': float(max(obs)),
             'native_peak_prediction_over_observation': float(max(i)/max(obs)),
             'initial_u': summaries[name]['initial_u'], 'terminal_u': summaries[name]['terminal_u'],
             'low20_RMSE_decade': summaries[name]['metrics']['low_abs_le_0p2']['RMSE_decade']}
        for branch, use in [('outward', out), ('return', ~out)]:
            idx = np.flatnonzero(mask & use & np.isclose(v, .2, atol=1e-12, rtol=0))
            assert len(idx) == 1
            j = int(idx[0])
            f[branch+'_0p2V'] = {'native_index': j, 'prediction_a': float(i[j]),
                'observation_a': float(obs[j]), 'prediction_over_observation': float(i[j]/obs[j])}
        features[name] = f
    repeat = {}
    for first, second in [('reset4', 'reset5'), ('set5', 'set6')]:
        a, z = arrays[first], arrays[second]
        assert np.array_equal(a['command_voltage_v'], z['command_voltage_v'])
        mask = a['original_scoring_mask'] & z['original_scoring_mask']
        model_delta = np.log10(abs(a['current_a'][mask]/z['current_a'][mask]))
        data_delta = np.log10(abs(a['observed_current_a'][mask]/z['observed_current_a'][mask]))
        repeat[first+'__'+second] = {
            'same_command_waveform': True, 'n_joint_scored': int(mask.sum()),
            'max_predicted_log_current_difference_decade': float(max(abs(model_delta))),
            'max_predicted_u_difference': float(max(abs(a['state_u']-z['state_u']))),
            'observed_log_current_difference_RMS_decade': float(np.sqrt(np.mean(data_delta**2))),
            'observed_max_log_current_difference_decade': float(max(abs(data_delta))),
            'interpretation': 'Selected candidate is numerically near a repeatable cycle; not a proof for the entire equation family.'}
    reference = [old['scores']['phase_metrics'][n]['all_eligible']['RMSE_decade'] for n in ('set4','reset4','set5')]
    pareto = []
    for r in d['requests']:
        if r.get('objective_request') and r.get('finite_objective_available') and r['status'] == 'completed_chain':
            values = [r['scores']['phase_metrics'][n]['all_eligible']['RMSE_decade'] for n in ('set4','reset4','set5')]
            if all(a < b for a, b in zip(values, reference)):
                pareto.append({'request_id': r['request_id'], 'label': r['label'],
                    'RMSE_SET4_RESET4_SET5': values, 'mean_phase_MSE': r['scores']['mean_phase_MSE_decade2'],
                    'precision': r['precision'], 'not_final_independently_validated': True})
    final_summaries = []
    for label in ('final_nominal', 'final_refined'):
        for row in rows[label]['phase_results']:
            final_summaries.append(read(Path(row['artifacts']['summary']['path'])))
    independent = [s['independent_native_readout'] for s in final_summaries]
    result = {
        'description': 'Saved arithmetic only; no candidate reselection or new physical calculation.',
        'source_sha256': sources, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'calibration_scores': scores, 'aggregate_calibration_scores': aggregate,
        'training_continuation_scores': final['continuation_metrics'], 'fixed_candidate_features': features,
        'same_waveform_cycle_differences': repeat,
        'search_only_candidates_improving_all_three_phases': pareto,
        'independent_native_points': sum(s['completed_native_points'] for s in independent),
        'max_independent_log_current_difference_decade': max(s['max_log_current_difference_decade'] for s in independent),
        'max_independent_local_voltage_difference_v': max(s['max_local_voltage_difference_v'] for s in independent),
        'max_independent_KCL_residual_a': max(s['max_KCL_residual_a'] for s in independent),
        'max_independent_KVL_residual_v': max(s['max_KVL_residual_v'] for s in independent),
        'max_final_precision_log_current_difference_decade': max(s['max_log_current_difference_decade'] for s in d['final_precision_comparison']['phases'].values()),
        'full_unified_model_achieved': False,
    }
    (HERE / 'SAVED_RESULT_DIAGNOSTICS.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('aggregate_calibration_scores', 'fixed_candidate_features', 'same_waveform_cycle_differences', 'search_only_candidates_improving_all_three_phases', 'independent_native_points')}, indent=2))


if __name__ == '__main__':
    main()
