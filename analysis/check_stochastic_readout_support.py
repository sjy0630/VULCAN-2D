"""Validate conditional readout fits and enumerate finite-domain support."""
from pathlib import Path
import hashlib
import json
import numpy as np
from fit_stochastic_readout_support import ROOT, inherited
from compare_activated_configuration import assemble, lookups, scores
from vulcan2d.finite_readout_support import readout, fractions, nearest_configuration, conditional_currents
from vulcan2d.side_charge_circuit import load_roots


def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def oracle(d, table, inverse):
    mask = d['weights'] > 0; current = np.zeros(len(mask)); labels = np.full(len(mask), -1)
    # Bound memory by one cycle instead of materializing configurations x 35k.
    for cycle in range(1, 54):
        select = mask & (d['cycle'] == cycle); index = np.flatnonzero(select)
        labels[select], _ = nearest_configuration(d['observed'][select], table[:, inverse[select]])
        current[select] = table[labels[select], inverse[select]]
    return current, labels


def coverage(d, current, ids):
    m = (d['weights'] > 0) & np.isin(d['cycle'], ids)
    error = abs(np.log10(current[m]/d['observed'][m])); w = d['weights'][m]
    return {str(tolerance): {'points_outside': int(np.count_nonzero(error > tolerance)),
                            'weighted_fraction_outside': float(w[error > tolerance].sum()/w.sum())}
            for tolerance in (.1, .2, .3, .5)} if m.any() else {}


def run():
    d = assemble(); lookup = lookups(); mask = d['weights'] > 0
    voltage, inverse = np.unique(d['voltage'], return_inverse=True)
    candidates = {}; domains = {}; hashes = {}; archive = {}
    for split in ('all_descriptive', 'chronological'):
        for states in ('two', 'four'):
            stem = f'stochastic_readout_support_{split}_{states}_reconciled'
            path = ROOT/'analysis'/f'{stem}.json'; source = json.loads(path.read_text()); assert source['status'] == 'complete'
            for name, sha in source['input_sha256'].items(): assert digest(ROOT/name) == sha, name
            assert digest(path.with_suffix('.npz')) == source['readout_sha256']
            for p in (path, path.with_suffix('.npz')): hashes[str(p.relative_to(ROOT))] = digest(p)
            stored = np.load(path.with_suffix('.npz')); p, settings = source['parameters'], source['settings']
            for key in ('voltage', 'observed', 'cycle', 'weights', 'window', 'low_signal'):
                np.testing.assert_array_equal(stored[key], d[key])
            prediction, labels = oracle(d, stored['conditional_current'][:, np.unique(inverse, return_index=True)[1]], inverse)
            np.testing.assert_array_equal(labels, stored['oracle_state'])
            np.testing.assert_allclose(prediction, stored['current'], rtol=0, atol=0)
            train = mask & np.isin(d['cycle'], source['training_ids']); error = np.log10(prediction[train]/d['observed'][train]); w = d['weights'][train]
            independent = float(np.sqrt(np.dot(w, error**2)/w.sum()))
            assert abs(independent-source['training_support_scores']['rmse_decade']) < 1e-12
            refs = []
            for label, changes in [('energy_refined', {'panels': 24}), ('table_refined', {'panels': 24, 'table_step': .0025}),
                                   ('finest', {'panels': 36, 'table_step': .00125})]:
                config = dict(settings); config.update(changes)
                table = readout(voltage, p, np.eye(4), *lookup, **config)
                refined, _ = oracle(d, table['current_A'], inverse)
                delta = np.log10(refined[mask]/stored['current'][mask]); ww = d['weights'][mask]
                # At ties the nearest-state identity can change discontinuously.
                # The minimal absolute error, not signed chosen current, is the objective.
                old_error = abs(np.log10(stored['current'][mask]/d['observed'][mask]))
                new_error = abs(np.log10(refined[mask]/d['observed'][mask]))
                objective_change = new_error-old_error
                signed = table['current_A'][:, inverse[mask]]/stored['conditional_current'][:, mask]
                max_kernel_change = float(np.max(abs(np.log10(signed))))
                weighted_objective = float(np.sqrt(np.dot(ww, objective_change**2)/ww.sum()))
                refs.append({'label': label, 'settings': config,
                             'maximum_conditional_current_log_change': max_kernel_change,
                             'maximum_absolute_residual_change': float(np.max(abs(objective_change))),
                             'weighted_absolute_residual_change': weighted_objective,
                             'maximum_oracle_chosen_current_log_change': float(np.max(abs(delta))),
                             'passes': max_kernel_change < .005 and weighted_objective < .001,
                             'training_support_scores': scores(d, refined, source['training_ids']),
                             'target_support_scores': scores(d, refined, source['target_ids']),
                             'maximum_kcl_relative_error': table['maximum_kcl_relative_error']})
            # Direct physical currents and independent root enumeration.
            index = np.flatnonzero(voltage != 0)[np.linspace(0, np.count_nonzero(voltage)-1, 30).astype(int)]
            main_table = readout(voltage, p, np.eye(4), *lookup, **settings)
            grid = np.arange(int(round(5/settings['table_step']))+1)*settings['table_step']; root_errors = []
            surfaces = {sign: conditional_currents(sign*grid, p, panels=settings['panels'], order=settings['order']) for sign in (-1, 1)}
            for state in range(4):
                for j in index:
                    sign = 1 if voltage[j] > 0 else -1
                    roots = load_roots(abs(voltage[j]), grid, surfaces[sign][state], *lookup[0 if sign > 0 else 1])
                    assert len(roots) == 1 and roots[0]['stable_load_line']
                    error = abs(roots[0]['internal_voltage_V']-abs(main_table['internal_voltage_V'][state, j]))
                    assert error < 1e-8; root_errors.append(error)
            swapped = np.array(p); swapped[[4, 5]] = swapped[[5, 4]]
            alt = readout(voltage, swapped, np.eye(4), *lookup, **settings)
            valid = voltage != 0
            swap_error = float(np.max(abs(np.log10(alt['current_A'][:, valid]/main_table['current_A'][[3, 1, 2, 0]][:, valid]))))
            assert swap_error < 1e-10
            candidates[stem] = {'independent_training_support_rmse': independent, 'refinements': refs,
                                'all_refinements_pass': all(row['passes'] for row in refs),
                                'independent_frozen_root_count': len(root_errors), 'maximum_root_voltage_error_V': max(root_errors),
                                'delta_U_swap_maximum_log_error': swap_error,
                                'training_coverage': coverage(d, prediction, source['training_ids']),
                                'target_coverage': coverage(d, prediction, source['target_ids'])}
            print(stem, independent, [row['maximum_conditional_current_log_change'] for row in refs], flush=True)
            if states == 'four':
                group = []; previous_absolute = None
                for n in (1, 2, 4, 8, 16):
                    pop = fractions(n); table = readout(voltage, p, pop, *lookup, **settings)
                    best, assignment = oracle(d, table['current_A'], inverse)
                    absolute = abs(np.log10(best[mask]/d['observed'][mask]))
                    if previous_absolute is not None: assert np.max(absolute-previous_absolute) < 1e-10
                    previous_absolute = absolute
                    group.append({'domains': n, 'configuration_count': len(pop),
                                  'training_support_scores': scores(d, best, source['training_ids']),
                                  'target_support_scores': scores(d, best, source['target_ids']),
                                  'full_support_scores': scores(d, best),
                                  'training_coverage': coverage(d, best, source['training_ids']),
                                  'target_coverage': coverage(d, best, source['target_ids']),
                                  'maximum_kcl_relative_error': table['maximum_kcl_relative_error']})
                    archive[f'{split}_N{n}_current'] = best; archive[f'{split}_N{n}_oracle_index'] = assignment
                    archive[f'{split}_N{n}_fractions'] = pop
                    print(split, 'domains', n, group[-1]['full_support_scores']['rmse_decade'], flush=True)
                low, high = main_table['current_A'].min(axis=0), main_table['current_A'].max(axis=0)
                continuous = np.clip(d['observed'], low[inverse], high[inverse])
                assert np.max(abs(np.log10(continuous[mask]/d['observed'][mask]))-previous_absolute) < 1e-10
                inherited_table = readout(voltage, inherited(split), np.eye(4), *lookup, **settings)
                old_best, _ = oracle(d, inherited_table['current_A'], inverse)
                domains[split] = {'fixed_fitted_parameters': p, 'finite_populations': group,
                                  'inherited_single_domain_full_support': scores(d, old_best),
                                  'continuous_arbitrary_population_support': {
                                      'training_support_scores': scores(d, continuous, source['training_ids']),
                                      'target_support_scores': scores(d, continuous, source['target_ids']),
                                      'full_support_scores': scores(d, continuous)}}
                archive[f'{split}_continuous_current'] = continuous
            stored.close()
    for p in (Path(__file__), ROOT/'vulcan2d/finite_readout_support.py', ROOT/'tests/test_finite_readout_support.py',
              ROOT/'analysis/fit_stochastic_readout_support.py'):
        hashes[str(p.relative_to(ROOT))] = digest(p)
    archive.update(voltage=d['voltage'], observed=d['observed'], cycle=d['cycle'], weights=d['weights'], window=d['window'], low_signal=d['low_signal'])
    output = ROOT/'analysis/stochastic_readout_support_validation.json'
    np.savez_compressed(output.with_suffix('.npz'), **archive)
    result = {'scope': 'Conditional readout feasibility, with observed-current oracle assignment; no stochastic prediction',
              'input_sha256': hashes, 'candidates': candidates, 'domain_support': domains,
              'arrays_sha256': digest(output.with_suffix('.npz'))}
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__': run()
