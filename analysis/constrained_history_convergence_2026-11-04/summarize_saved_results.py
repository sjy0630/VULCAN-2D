"""Arithmetic on saved three-phase results only; no model imports or solves."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    sources = {}
    def read(path, arrays=False):
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        if arrays:
            with np.load(path, allow_pickle=False) as z:
                return {k: z[k].copy() for k in z.files}
        return json.loads(path.read_text())
    result = read(HERE/'bounded_coordinate_check.json')
    assert result['status'] == 'completed_finite_coordinate_diagnostic'
    old = result['verification_chains']['old']['refined']
    seed = result['verification_chains']['request57']['refined']
    final = result['final_chains']['refined']
    scores = {}
    for name in ('set4','reset4','set5'):
        e = {k:r['scores']['phase_metrics'][name]['all_eligible']['RMSE_decade']
             for k,r in [('old',old),('request57',seed),('final',final)]}
        scores[name] = {**e, 'relative_RMSE_reduction_from_old_percent':100*(1-e['final']/e['old']),
                       'final_10_power_RMSE':10**e['final']}
    aggregate = {}
    for key in ('mean_phase_MSE_decade2','all1170_RMSE_decade'):
        vals = {k:r['scores'][key] for k,r in [('old',old),('request57',seed),('final',final)]}
        aggregate[key] = {**vals, 'relative_reduction_from_old_percent':100*(1-vals['final']/vals['old']),
                         'relative_reduction_from_seed_percent':100*(1-vals['final']/vals['request57'])}
    features = {}
    states = []
    for row in final['phase_results']:
        name = row['phase']
        a = read(Path(row['artifacts']['arrays']['path']), True)
        s = read(Path(row['artifacts']['summary']['path']))
        v, p, o = abs(a['command_voltage_v']), abs(a['current_a']), abs(a['observed_current_a'])
        mask = a['original_scoring_mask']
        outward = np.arange(len(v)) <= np.argmax(v)
        f = {'native_peak_observed_a':float(max(o)), 'native_peak_predicted_a':float(max(p)),
             'native_peak_prediction_over_observation':float(max(p)/max(o)),
             'low20_RMSE_decade':s['metrics']['low_abs_le_0p2']['RMSE_decade'],
             'outward_RMSE_decade':s['metrics']['outward']['RMSE_decade'],
             'return_RMSE_decade':s['metrics']['return']['RMSE_decade']}
        for label, branch in [('outward',outward),('return',~outward)]:
            ids = np.flatnonzero(mask & branch & np.isclose(v,.2,rtol=0,atol=1e-12))
            assert len(ids)==1
            i=int(ids[0]);f[label+'_0p2V']={'native_index':i,'observed_a':float(o[i]),
                'predicted_a':float(p[i]),'prediction_over_observation':float(p[i]/o[i])}
        features[name]=f
        states.append({'phase':name,'initial_u':s['initial_u'],'terminal_u':s['terminal_u']})
    statuses={}
    for r in result['requests']:
        statuses[r['status']]=statuses.get(r['status'],0)+1
    output={'source_sha256':sources,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'description':'Saved arithmetic, no new roots/inversions/ODEs/optimization.',
        'phase_RMSE_comparison':scores,'aggregate_scores':aggregate,'selected_features':features,
        'actual_state_chain':states,'request_status_counts':statuses,'search_outcome':result['search_outcome'],
        'original_holdout_accessed':False,'full_unified_model_achieved':False}
    (HERE/'SAVED_RESULT_DIAGNOSTICS.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:output[k] for k in ('aggregate_scores','selected_features','actual_state_chain','request_status_counts')},indent=2))


if __name__=='__main__': main()
