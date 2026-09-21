"""No-refit, prescribed contact swap; all native observations retained."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'analysis'))
from compare_activated_configuration import assemble, lookups, scores
from check_retained_charge_contact import ratio34
from vulcan2d import retained_charge_configuration as rc
from vulcan2d.retained_contact_memory import compensated_contact, switched_trajectory


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    d = assemble(); lookup = lookups(); boundary = int(np.flatnonzero(d['cycle'] == 4)[0])
    assert boundary == 2028 and len(d['voltage']) == 35828
    assert d['voltage'][boundary-1] == d['voltage'][boundary] == 0.
    inputs = [Path(__file__), ROOT/'vulcan2d/retained_contact_memory.py',
              ROOT/'analysis/CONTACT_MEMORY_IMPLEMENTATION_DESIGN_2026-09-21.md',
              ROOT/'analysis/CONTACT_ASYMMETRY_MEMORY_DESIGN_2026-09-21.md',
              ROOT/'tests/test_retained_contact_memory.py',
              ROOT/'analysis/check_retained_charge_contact.py']
    output = {'scope': 'No-refit prescribed contact swap, not event-time prediction or fitted stochastic path',
              'native_point_count': len(d['voltage']), 'split_index': boundary,
              'observed_SET_return_4_over_3': ratio34(d, d['observed'], 2),
              'observed_RESET_outward_4_over_3': ratio34(d, d['observed'], 4), 'candidates': {}}
    arrays = {k: d[k] for k in ('voltage', 'observed', 'cycle', 'window', 'weights', 'low_signal')}
    nonzero = abs(d['voltage']) > 0; valid = d['weights'] > 0
    for split in ('all_descriptive', 'chronological'):
        source_path = ROOT/'analysis'/f'retained_charge_configuration_{split}_single_contact.json'
        source = json.loads(source_path.read_text()); assert source['status'] == 'complete'
        inputs += [source_path, source_path.with_suffix('.npz')]
        for name, sha in source['input_sha256'].items():
            assert digest(ROOT/name) == sha, name; inputs.append(ROOT/name)
        assert digest(source_path.with_suffix('.npz')) == source['prediction_sha256']
        s = np.array(source['shared_parameters']); p = np.array(source['parameters']); g = abs(s[4])
        settings = source['settings']
        archive = np.load(source_path.with_suffix('.npz'))
        original = rc.surfaces(np.linspace(-5., 5., 2001), s, p)
        zero_original = rc.surfaces([0.], s, p)
        for label, g0, g1 in [('fixed_positive', g, g), ('fixed_negative', -g, -g),
                              ('negative_to_positive', -g, g), ('positive_to_negative', g, -g)]:
            name = f'{split}_{label}'
            r = switched_trajectory(d['voltage'], s, p, *lookup, boundary, g0, g1, **settings, diagnostics=True)
            identity = []
            for gl in (g0, g1):
                ss, pp = compensated_contact(s, p, gl)
                surface = rc.surfaces(np.linspace(-5., 5., 2001), ss, pp)
                zero = rc.surfaces([0.], ss, pp); mask = original['current'] > 0
                identity.append({'log10A': pp[0], 'ln_left_over_right': gl,
                    'log10A_within_old_bounds': bool(rc.LO[0] <= pp[0] <= rc.HI[0]),
                    'maximum_frozen_current_change_decade': float(np.max(abs(np.log10(surface['current'][mask]/original['current'][mask])))),
                    'maximum_zero_bias_log_occupation_change': float(np.max(abs(zero['n_log_probability']-zero_original['n_log_probability']))),
                    'maximum_zero_bias_log_rate_change': float(np.max(abs(zero['log_rates']-zero_original['log_rates'])))})
            row = {'parameter_source': str(source_path.relative_to(ROOT)), 'settings': settings,
                   'contact_label': label, 'boundary_cycle': 3, 'identity_checks': identity,
                   'full_scores': scores(d, r['current_A']),
                   'training_scores': scores(d, r['current_A'], source['training_ids']),
                   'last18_scores': scores(d, r['current_A'], list(range(36, 54))),
                   'SET_return_4_over_3': ratio34(d, r['current_A'], 2),
                   'RESET_outward_4_over_3': ratio34(d, r['current_A'], 4),
                   'maximum_kcl_relative_error': r['maximum_kcl_relative_error'],
                   'multiple_root_point_count': int(np.count_nonzero(r['root_count'] > 1)),
                   'boundary_state_change': r['boundary_state_change'],
                   'minimum_probability': float(r['states'].min()),
                   'maximum_probability_sum_error': float(np.max(abs(r['states'].sum(axis=1)-1.)))}
            assert row['maximum_kcl_relative_error'] < 1e-8
            assert row['boundary_state_change'] == 0 and row['maximum_probability_sum_error'] < 2e-11
            for record in identity:
                assert record['maximum_frozen_current_change_decade'] < 1e-12
                assert record['maximum_zero_bias_log_occupation_change'] < 1e-12
                assert record['maximum_zero_bias_log_rate_change'] < 1e-12
            if label.startswith('fixed'):
                ss, pp = compensated_contact(s, p, g0)
                unsplit = rc.trajectory(d['voltage'], ss, pp, *lookup, **settings)
                for field in ('current_A', 'states', 'internal_voltage_V'):
                    np.testing.assert_array_equal(r[field], unsplit[field])
                row['split_equals_unsplit_exactly'] = True
            if label == 'fixed_positive':
                np.testing.assert_array_equal(r['current_A'], archive['current'])
                np.testing.assert_array_equal(r['states'], archive['states'])
                row['matches_archived_prediction_exactly'] = True
            for field in ('current_A', 'internal_voltage_V', 'transistor_voltage_V', 'root_count'):
                arrays[f'{name}_{field}'] = r[field]
            # Save all full distributions, not only their unidentifiable averages.
            state_path = ROOT/'analysis'/f'contact_memory_{name}_states.npz'
            np.savez_compressed(state_path, states=r['states'], weights=r['quadrature_weights'])
            row['states_file'] = str(state_path.relative_to(ROOT)); row['states_sha256'] = digest(state_path)
            output['candidates'][name] = row
            print(name, 'all', row['full_scores']['rmse_decade'], 'first3', row['full_scores']['first3_rmse'],
                  'S4/S3', row['SET_return_4_over_3'], 'R4/R3', row['RESET_outward_4_over_3'], flush=True)
    destination = ROOT/'analysis/contact_memory_counterfactual.npz'
    np.savez_compressed(destination, **arrays); output['arrays_sha256'] = digest(destination)
    output['input_sha256'] = {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(inputs))}
    (ROOT/'analysis/contact_memory_counterfactual.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__': run()
