"""Independent objective, histories, roots and numerical convergence checks."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'analysis'))
from compare_activated_configuration import assemble, lookups, baseline, scores
from check_retained_charge_contact import ratio34
from vulcan2d.retained_charge_configuration import trajectory, surfaces
from vulcan2d.side_charge_circuit import load_roots


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(paths, output):
    d = assemble(); lookup = lookups(); mask = d['weights'] > 0; rows = {}; hashes = {}
    for name in paths:
        print('VALIDATE', name, flush=True)
        path = ROOT/'analysis'/name; source = json.loads(path.read_text()); assert source['status'] == 'complete'
        for key, sha in source['input_sha256'].items(): assert digest(ROOT/key) == sha, key
        assert digest(path.with_suffix('.npz')) == source['prediction_sha256']
        stored = np.load(path.with_suffix('.npz'))
        hashes[str(path.relative_to(ROOT))] = digest(path)
        hashes[str(path.with_suffix('.npz').relative_to(ROOT))] = digest(path.with_suffix('.npz'))
        shared, p, settings = source['shared_parameters'], source['parameters'], source['settings']
        expected = np.array(baseline(source['split']))
        if source.get('sensitivity') == 'contact': expected[4] = shared[4]
        np.testing.assert_array_equal(shared, expected)
        if source['split'] == 'chronological':
            assert source['training_ids'] == list(range(1, 36))
            assert source['target_ids'] == list(range(36, 54))
        for field in ('voltage', 'observed', 'cycle', 'window', 'weights', 'low_signal'):
            np.testing.assert_array_equal(stored[field], d[field])
        train = mask & np.isin(d['cycle'], source['training_ids'])
        error = np.log10(stored['current'][train]/d['observed'][train]); w = d['weights'][train]
        independent_score = float(np.sqrt(np.dot(w, error**2)/w.sum()))
        assert abs(independent_score-source['training_scores']['rmse_decade']) < 1e-12
        states, qw = stored['states'], stored['quadrature_weights']
        norm_error = float(np.max(abs(states.sum(axis=1)-1)))
        assert norm_error < 1e-9 and states.min() >= 0 and states.max() <= 1+1e-9
        repeated = np.flatnonzero(np.diff(d['voltage']) == 0)+1
        np.testing.assert_array_equal(states[repeated], states[repeated-1])
        refinements = []
        for label, changes in [('energy_refined', {'panels': 2*settings['panels']}),
                              ('all_refined', {'panels': 2*settings['panels'], 'table_step': settings['table_step']/2, 'max_step': settings['max_step']/2}),
                              ('finest', {'panels': 3*settings['panels'], 'table_step': settings['table_step']/4, 'max_step': settings['max_step']/4})]:
            config = dict(settings); config.update(changes)
            print('REFINE', name, label, config, flush=True)
            r = trajectory(d['voltage'], shared, p, *lookup, **config, diagnostics=True)
            err = np.log10(r['current_A'][mask]/stored['current'][mask]); ww = d['weights'][mask]
            max_change = float(abs(err).max()); weighted_change = float(np.sqrt(np.dot(ww, err**2)/ww.sum()))
            refinements.append({'label': label, 'settings': config, 'maximum_log_change': max_change,
                                'weighted_log_change': weighted_change,
                                'passes_declared_tolerances': max_change < .005 and weighted_change < .001,
                                'training_scores': scores(d, r['current_A'], source['training_ids']),
                                'target_scores': scores(d, r['current_A'], source['target_ids']),
                                'maximum_kcl_relative_error': r['maximum_kcl_relative_error'],
                                'multiple_root_point_count': int(np.count_nonzero(r['root_count'] > 1))})
            del r
        candidates = np.flatnonzero(mask); index = candidates[np.linspace(0, len(candidates)-1, 120).astype(int)]
        direct = surfaces(stored['internal_voltage'][index], shared, p, panels=settings['panels'], order=settings['order'])
        direct_i = np.einsum('isn,sin,n->i', states[index], direct['current'], qw)
        direct_error = float(np.max(abs(np.log10(direct_i/stored['current'][index]))))
        step = settings['table_step']; grid = np.arange(int(round(5/step))+1)*step
        tables = {sign: surfaces(sign*grid, shared, p, panels=settings['panels'], order=settings['order']) for sign in (-1, 1)}
        root_errors = []
        for j in index:
            sign = 1 if d['voltage'][j] > 0 else -1
            frozen_current = np.einsum('sn,svn,n->v', states[j], tables[sign]['current'], qw)
            roots = load_roots(abs(d['voltage'][j]), grid, frozen_current, *lookup[0 if sign > 0 else 1])
            best = min(roots, key=lambda root: abs(root['internal_voltage_V']-abs(stored['internal_voltage'][j])))
            error = abs(best['internal_voltage_V']-abs(stored['internal_voltage'][j]))
            assert error < 1e-8 and best['stable_load_line'] and len(roots) == stored['root_count'][j]
            root_errors.append(error)
        sensitivities = []
        for initial in (0., .5, 1.):
            r = trajectory(d['voltage'], shared, p, *lookup, **settings, initial_m=initial)
            sensitivities.append({'initial_m1': initial, 'refitted': False, 'full_scores': scores(d, r['current_A']),
                                  'target_scores': scores(d, r['current_A'], source['target_ids']),
                                  'RESET_outward_4_over_3': ratio34(d, r['current_A'], 4)})
        end_pair = np.array([np.flatnonzero(d['cycle'] == c)[-1] for c in range(1, 54)])
        end_set = np.array([np.flatnonzero((d['cycle'] == c) & (d['voltage'] > 0))[-1]+1 for c in range(1, 54)])
        qmean = (states[:, 2]+states[:, 3])@qw; mmean = (states[:, 1]+states[:, 3])@qw
        currents = stored['current'].reshape(53, -1); valid = mask.reshape(53, -1)
        differences = [float(np.max(abs(np.log10(currents[c, valid[c]&valid[c-1]]/currents[c-1, valid[c]&valid[c-1]])))) for c in range(1, 53)]
        rows[path.stem] = {'independent_training_rmse': independent_score, 'normalization_max_error': norm_error,
                           'source_fit_converged': source['starts'][source['selected_start']]['converged'],
                           'refinements': refinements, 'all_refinements_pass': all(r['passes_declared_tolerances'] for r in refinements),
                           'independent_frozen_root_checks': len(index), 'maximum_root_voltage_error_V': max(root_errors),
                           'maximum_direct_readout_log_error': direct_error, 'initial_condition_sensitivity': sensitivities,
                           'mean_q1_after_set': qmean[end_set].tolist(), 'mean_q1_after_pair': qmean[end_pair].tolist(),
                           'mean_m1_after_set': mmean[end_set].tolist(), 'mean_m1_after_pair': mmean[end_pair].tolist(),
                           'maximum_log_change_from_previous_pair': differences,
                           'cycle4_over_cycle3_native_0.10_to_0.30V': {
                               phase: {'observed': ratio34(d, d['observed'], window), 'predicted': ratio34(d, stored['current'], window)}
                               for phase, window in [('SET_return', 2), ('RESET_outward', 4)]}}
        print(path.stem, independent_score, [r['maximum_log_change'] for r in refinements], flush=True)
        stored.close()
    for p in [Path(__file__), ROOT/'analysis/check_retained_charge_contact.py', ROOT/'tests/test_retained_charge_configuration.py']:
        hashes[str(p.relative_to(ROOT))] = digest(p)
    result = {'candidate_count': len(rows), 'input_sha256': hashes, 'candidates': rows}
    (ROOT/'analysis'/output).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('paths', nargs='+'); parser.add_argument('--output', required=True)
    args = parser.parse_args(); run(args.paths, args.output)
