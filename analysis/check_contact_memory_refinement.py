"""Refine both swap directions and fixed controls; independent load checks."""
from pathlib import Path
import json
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'analysis'))
from check_contact_memory_counterfactual import digest
from compare_activated_configuration import assemble, lookups, scores
from check_retained_charge_contact import ratio34
from vulcan2d.retained_contact_memory import switched_trajectory
from vulcan2d.retained_charge_configuration import surfaces
from vulcan2d.side_charge_circuit import load_roots


def run():
    source_path = ROOT/'analysis/contact_memory_counterfactual.json'
    source = json.loads(source_path.read_text()); inputs = [source_path, source_path.with_suffix('.npz'), Path(__file__)]
    for name, sha in source['input_sha256'].items():
        assert digest(ROOT/name) == sha, name
    assert digest(source_path.with_suffix('.npz')) == source['arrays_sha256']
    saved = np.load(source_path.with_suffix('.npz')); d = assemble(); lookup = lookups()
    for k in ('voltage', 'observed', 'cycle', 'window', 'weights', 'low_signal'):
        np.testing.assert_array_equal(saved[k], d[k])
    mask = d['weights'] > 0; weight = d['weights'][mask]; rows = {}; arrays = {}
    # Stratify checks across early suppression and ordinary later cycles.
    early = np.flatnonzero(mask & (d['cycle'] <= 3)); late = np.flatnonzero(mask & (d['cycle'] > 3))
    index = np.r_[early[np.linspace(0, len(early)-1, 48).astype(int)], late[np.linspace(0, len(late)-1, 48).astype(int)]]
    for name, entry in source['candidates'].items():
        baseline = saved[f'{name}_current_A']; source_fit = json.loads((ROOT/entry['parameter_source']).read_text())
        settings = entry['settings']; s, p = source_fit['shared_parameters'], source_fit['parameters']
        g0, g1 = [part['ln_left_over_right'] for part in entry['identity_checks']]
        stored_path = ROOT/entry['states_file']; assert digest(stored_path) == entry['states_sha256']; inputs.append(stored_path)
        with np.load(stored_path) as stored:
            states, qw = stored['states'], stored['weights']
        repeated = np.flatnonzero(np.diff(d['voltage']) == 0)+1
        np.testing.assert_array_equal(states[repeated], states[repeated-1])
        # Conditional readout is contact-invariant, so original s,p apply
        # to both sides of the imposed switch for these frozen-load checks.
        step = settings['table_step']; grid = np.arange(int(round(5/step))+1)*step
        table = {sign: surfaces(sign*grid, s, p, panels=settings['panels'], order=settings['order']) for sign in (-1, 1)}
        root_errors = []; direct = surfaces(saved[f'{name}_internal_voltage_V'][index], s, p,
                                            panels=settings['panels'], order=settings['order'])
        direct_i = np.einsum('isn,sin,n->i', states[index], direct['current'], qw)
        for j in index:
            sign = 1 if d['voltage'][j] > 0 else -1
            frozen = np.einsum('sn,svn,n->v', states[j], table[sign]['current'], qw)
            roots = load_roots(abs(d['voltage'][j]), grid, frozen, *lookup[0 if sign > 0 else 1])
            vh = abs(saved[f'{name}_internal_voltage_V'][j]); chosen = min(roots, key=lambda r: abs(r['internal_voltage_V']-vh))
            error = abs(chosen['internal_voltage_V']-vh)
            assert error < 1e-8 and chosen['stable_load_line'] and len(roots) == saved[f'{name}_root_count'][j]
            root_errors.append(error)
        rows[name] = {'independent_frozen_root_checks': len(index), 'maximum_root_voltage_error_V': max(root_errors),
                      'maximum_direct_readout_log_error': float(np.max(abs(np.log10(direct_i/baseline[index])))),
                      'normalization_max_error': float(np.max(abs(states.sum(axis=1)-1.))),
                      'refinements': []}
        # Retain state-resolved diagnostics at the ends of SET and RESET.
        ends = np.flatnonzero((d['voltage'] == 0) & np.r_[False, np.diff(d['voltage']) != 0])
        rows[name]['endpoint_cycle'] = d['cycle'][ends].tolist()
        rows[name]['endpoint_q1'] = ((states[ends, 2]+states[ends, 3])@qw).tolist()
        rows[name]['endpoint_m1'] = ((states[ends, 1]+states[ends, 3])@qw).tolist()
        del states, table, direct
        for label, factor in [('refined', 2), ('finest', 4)]:
            config = dict(settings, panels=settings['panels']*(2 if factor == 2 else 3),
                          table_step=settings['table_step']/factor, max_step=settings['max_step']/factor)
            print(name, label, 'start', flush=True)
            r = switched_trajectory(d['voltage'], s, p, *lookup, source['split_index'], g0, g1, **config, diagnostics=True)
            err = np.log10(r['current_A'][mask]/baseline[mask]); maximum = float(abs(err).max())
            weighted = float(np.sqrt(np.dot(weight, err**2)/weight.sum()))
            row = {'label': label, 'settings': config, 'maximum_log_change': maximum, 'weighted_log_change': weighted,
                   'maximum_probability_sum_error': float(np.max(abs(r['states'].sum(axis=1)-1.))),
                   'minimum_probability': float(r['states'].min()),
                   'maximum_kcl_relative_error': r['maximum_kcl_relative_error'],
                   'boundary_state_change': r['boundary_state_change'],
                   'full_scores': scores(d, r['current_A']),
                   'SET_return_4_over_3': ratio34(d, r['current_A'], 2),
                   'RESET_outward_4_over_3': ratio34(d, r['current_A'], 4)}
            row['passes_declared_tolerances'] = bool(maximum < .005 and weighted < .001 and
                row['maximum_probability_sum_error'] < 2e-11 and row['maximum_kcl_relative_error'] < 1e-8 and
                row['minimum_probability'] >= 0 and row['boundary_state_change'] == 0)
            rows[name]['refinements'].append(row); arrays[f'{name}_{label}_current_A'] = r['current_A']
            print(name, label, maximum, weighted, row['passes_declared_tolerances'], flush=True)
            del r
        rows[name]['all_refinements_pass'] = all(r['passes_declared_tolerances'] for r in rows[name]['refinements'])
        # Checkpoint after each completed candidate, including any failure.
        result = {'scope': 'Numerical verification, not physical validation', 'candidates': rows,
                  'input_sha256': {str(p.relative_to(ROOT)): digest(p) for p in inputs}}
        (ROOT/'analysis/contact_memory_refinement.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    destination = ROOT/'analysis/contact_memory_refinement.npz'
    np.savez_compressed(destination, **arrays); result['arrays_sha256'] = digest(destination)
    result['all_refinements_pass'] = all(r['all_refinements_pass'] for r in rows.values())
    (ROOT/'analysis/contact_memory_refinement.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    assert result['all_refinements_pass']


if __name__ == '__main__': run()
