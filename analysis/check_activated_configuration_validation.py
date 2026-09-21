"""Independent scoring, frozen-load enumeration, and numerical sensitivities."""
from pathlib import Path
import json
import hashlib
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'analysis'))
from compare_activated_configuration import assemble, lookups, baseline, scores
from vulcan2d.activated_configuration import trajectory, surfaces
from vulcan2d.side_charge_circuit import load_roots


def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run():
    data = assemble(); lookup = lookups(); mask = data['weights'] > 0
    rows = {}; hashes = {}
    for split in ('all_descriptive', 'chronological'):
        for saddle in ('single', 'polarizable'):
            stem = f'activated_configuration_{split}_{saddle}'
            reconciled = ROOT/'analysis'/f'{stem}_reconciled.json'
            if reconciled.exists(): stem += '_reconciled'
            path = ROOT/'analysis'/f'{stem}.json'
            if not path.exists(): continue
            source = json.loads(path.read_text())
            if source['status'] != 'complete': continue
            for name, sha in source['input_sha256'].items(): assert digest(ROOT/name) == sha, name
            stored = np.load(path.with_suffix('.npz')); assert digest(path.with_suffix('.npz')) == source['prediction_sha256']
            hashes[str(path.relative_to(ROOT))] = digest(path)
            hashes[str(path.with_suffix('.npz').relative_to(ROOT))] = digest(path.with_suffix('.npz'))
            shared, p = source['shared_parameters'], source['parameters']; settings = source['settings']
            np.testing.assert_array_equal(shared, baseline(split))
            if split == 'chronological':
                old = json.loads((ROOT/'analysis/charge_spectrum_expanded_groups.json').read_text())['splits'][split]
                np.testing.assert_array_equal(old['training_ids'], source['training_ids'])
                assert not set(source['training_ids']) & set(source['target_ids'])
            for name, key in [('voltage', 'voltage'), ('observed', 'observed'), ('cycle', 'cycle'), ('window', 'window'), ('weights', 'weights')]:
                np.testing.assert_array_equal(stored[name], data[key])
            # Recompute the scalar objective directly without the score helper.
            train = np.isin(data['cycle'], source['training_ids']) & mask
            e = np.log10(stored['current'][train]/data['observed'][train]); w = data['weights'][train]
            independent_rmse = float(np.sqrt(np.dot(w, e*e)/w.sum()))
            assert abs(independent_rmse-source['training_scores']['rmse_decade']) < 1e-12
            states = stored['states']; weights = stored['quadrature_weights']
            assert np.max(abs(states.sum(axis=1)-1)) < 2e-13
            assert np.all((states >= 0) & (states <= 1))
            repeats = np.flatnonzero(np.diff(data['voltage']) == 0)+1
            np.testing.assert_array_equal(states[repeats], states[repeats-1])
            records = []
            for label, change in [('energy_refined', {'panels': 2*settings['panels']}),
                                  ('all_refined', {'panels': 2*settings['panels'], 'table_step': settings['table_step']/2, 'max_step': settings['max_step']/2}),
                                  ('finest', {'panels': 3*settings['panels'], 'table_step': settings['table_step']/4, 'max_step': settings['max_step']/4})]:
                config = dict(settings); config.update(change)
                refined = trajectory(data['voltage'], shared, p, *lookup, **config, diagnostics=True)
                err = np.log10(refined['current_A'][mask]/stored['current'][mask]); ww = data['weights'][mask]
                records.append({'label': label, 'settings': config,
                                'maximum_log_change': float(abs(err).max()),
                                'weighted_log_change': float(np.sqrt(np.dot(ww, err*err)/ww.sum())),
                                'training_scores': scores(data, refined['current_A'], source['training_ids']),
                                'target_scores': scores(data, refined['current_A'], source['target_ids']),
                                'maximum_kcl_relative_error': refined['maximum_kcl_relative_error'],
                                'multiple_root_point_count': int(np.count_nonzero(refined['root_count'] > 1))})
            # Independently enumerate every piecewise-linear root for sampled
            # frozen configurations; observed current never selects a root.
            candidates = np.flatnonzero(mask)
            index = np.unique(np.r_[candidates[np.linspace(0, len(candidates)-1, 180).astype(int)],
                                    np.flatnonzero(stored['root_count'] > 1)[::max(1, np.count_nonzero(stored['root_count'] > 1)//40)]])
            direct = surfaces(stored['internal_voltage'][index], shared, p, panels=settings['panels'], order=settings['order'])
            direct_i = np.sum(weights[None, :]*(states[index, 0]*direct['current'][0]+states[index, 1]*direct['current'][1]), axis=1)
            direct_error = np.log10(direct_i/stored['current'][index])
            root_checks = []; step = settings['table_step']; grid = np.arange(int(round(5/step))+1)*step
            tables = {sign: surfaces(sign*grid, shared, p, panels=settings['panels'], order=settings['order']) for sign in (-1, 1)}
            for j in index:
                sign = 1 if data['voltage'][j] > 0 else -1; surf = tables[sign]
                current = np.sum(weights[None, :]*(states[j, 0]*surf['current'][0]+states[j, 1]*surf['current'][1]), axis=1)
                tv, ti = lookup[0 if sign > 0 else 1]
                roots = load_roots(abs(data['voltage'][j]), grid, current, tv, ti)
                best = min(roots, key=lambda x: abs(x['internal_voltage_V']-abs(stored['internal_voltage'][j])))
                error = abs(best['internal_voltage_V']-abs(stored['internal_voltage'][j]))
                assert error < 1e-8 and best['stable_load_line'], (j, roots)
                assert len(roots) == stored['root_count'][j], (j, roots, stored['root_count'][j])
                root_checks.append(error)
            sensitivity = []
            for initial in (.01, .5, 1.):
                altered = trajectory(data['voltage'], shared, p, *lookup, **settings, initial=initial)
                sensitivity.append({'initial_q1': initial, 'refitted': False,
                                    'full_scores': scores(data, altered['current_A']),
                                    'target_scores': scores(data, altered['current_A'], source['target_ids'])})
            mean_q = states[:, 1]@weights
            end_set = np.array([np.flatnonzero((data['cycle'] == c) & (data['voltage'] > 0))[-1]+1 for c in range(1, 54)])
            end_pair = np.array([np.flatnonzero(data['cycle'] == c)[-1] for c in range(1, 54)])
            reshaped = stored['current'].reshape(53, -1); valid = data['weights'].reshape(53, -1) > 0
            repeat_difference = []
            for j in range(1, 53):
                m = valid[j] & valid[j-1]
                repeat_difference.append(float(np.max(abs(np.log10(reshaped[j, m]/reshaped[j-1, m])))))
            def cycle34_ratio(current, window):
                value = []
                for c in (3, 4):
                    m = (data['cycle'] == c) & (data['window'] == window) & (abs(data['voltage']) >= .10-1e-10)
                    value.append(np.mean(np.log10(current[m])))
                return float(10**(value[1]-value[0]))
            rows[stem] = {'independent_training_rmse': independent_rmse, 'refinements': records,
                          'source_fit_converged': source['starts'][source['selected_start']]['converged'],
                          'independent_frozen_root_checks': len(index), 'maximum_root_voltage_error_V': max(root_checks),
                          'maximum_direct_readout_log_error': float(abs(direct_error).max()),
                          'initial_condition_sensitivity': sensitivity,
                          'mean_q1_after_set': mean_q[end_set].tolist(), 'mean_q1_after_pair': mean_q[end_pair].tolist(),
                          'maximum_log_change_from_previous_pair': repeat_difference,
                          'cycle4_over_cycle3_native_0.10_to_0.30V': {
                              phase: {'observed': cycle34_ratio(data['observed'], window),
                                      'predicted': cycle34_ratio(stored['current'], window)}
                              for phase, window in [('SET_return', 2), ('RESET_outward', 4)]},
                          'parameters_near_bound_within_2e_5_fraction': [int(i) for i in range(8)
                              if (saddle != 'single' or i != 7) and min((p[i]-source['bounds'][0][i]),
                                  (source['bounds'][1][i]-p[i]))/(source['bounds'][1][i]-source['bounds'][0][i]) < 2e-5],
                          'normalization_max_error': float(np.max(abs(states.sum(axis=1)-1)))}
            print(stem, independent_rmse, [r['maximum_log_change'] for r in records], flush=True)
    required = ['vulcan2d/coupled_charge.py', 'vulcan2d/coupled_charge_reduced.py', 'vulcan2d/charge_spectrum.py',
                'vulcan2d/transistor.py', 'vulcan2d/data/fig2b_transistor_lookup.json',
                'vulcan2d/data/bookoutput2_reverse_transistor_lookup.json', str(Path(__file__).relative_to(ROOT)),
                'analysis/activated_configuration_reference.py', 'analysis/activated_configuration_reference.json',
                'tests/test_activated_configuration.py']
    hashes.update({p: digest(ROOT/p) for p in required})
    result = {'candidate_count': len(rows), 'input_sha256': hashes, 'candidates': rows}
    path = ROOT/'analysis/activated_configuration_validation_audit.json'
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__': run()
