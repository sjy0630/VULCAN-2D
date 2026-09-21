"""Source identity, grouped scores, circuit roots and honest failure accounting."""
from pathlib import Path
import hashlib
import json
import re
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'analysis'))
from compare_parallel_paths import assemble, VOLTAGE, FOLDS
from compare_side_charge_readout import scores
from diagnose_cross_polarity_physics import GateLookup
from vulcan2d.transistor import load_reverse_lookup
from vulcan2d.side_charge import predict, VT


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(actual, expected):
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:compare(actual[key], expected[key])
    else:
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)


def run():
    names = ['side_charge_readout.json', 'side_charge_validation.json', 'side_charge_lever_profiles.json',
             'side_charge_maximal_gate.json', 'side_charge_high_voltage.json']
    paths = [ROOT/'analysis'/name for name in names]
    main, validation, profiles, maximal, high = reports = [json.loads(p.read_text()) for p in paths]
    provenance = []
    for path, report in zip(paths, reports):
        for name, sha in report['input_sha256'].items():
            assert digest(ROOT/name) == sha, f'{path.name}: {name} changed'
        provenance.append({'report':path.name, 'input_count':len(report['input_sha256'])})
    old = []
    for stem in ('barrier_height_polarity', 'barrier_state', 'finite_path', 'bidirectional'):
        path = ROOT/f'analysis/{stem}_artifact_manifest.json'; manifest = json.loads(path.read_text())
        count = 0
        for key, base in [('artifacts', ROOT), ('external_workspace_logs', ROOT.parent)]:
            for name, record in manifest.get(key, {}).items():
                assert digest(base/name) == record['sha256'], f'Old artifact changed: {name}'
                count += 1
        old.append({'manifest':path.name, 'sha256':digest(path), 'verified_file_count':count})
    ids, y, weights, selection = assemble()
    assert main['status'] == 'complete' and main['selection'] == selection
    np.testing.assert_array_equal(main['observed_log10_current_A'], y)
    np.testing.assert_array_equal(main['weights'], weights)
    assert np.count_nonzero(weights) == 5072 and sum(len(r['missing']) for r in selection) == 16
    labels = {'all_descriptive', 'chronological'} | {f'fold_{lo}_{hi}' for lo, hi in FOLDS}
    shared_starts, fit_count, local_count = [], 0, 0
    for mode, candidate in main['candidates'].items():
        assert set(candidate['splits']) == labels
        target_ids = []; pooled = {name:np.empty_like(y) for name in ('SET_only', 'joint_descriptive')}
        for label, split in candidate['splits'].items():
            training, target = np.array(split['training_ids']), np.array(split['target_ids'])
            if label == 'all_descriptive':
                np.testing.assert_array_equal(training, ids); np.testing.assert_array_equal(target, ids)
            else:
                assert not set(training) & set(target)
                if label == 'chronological':
                    np.testing.assert_array_equal(training, ids[ids <= 35]); np.testing.assert_array_equal(target, ids[ids > 35])
                else:
                    lo, hi = map(int, label.split('_')[1:])
                    np.testing.assert_array_equal(target, ids[(ids >= lo) & (ids <= hi)])
                    np.testing.assert_array_equal(training, ids[(ids < lo) | (ids > hi)])
                    target_ids.extend(target.tolist())
            fit = split['fit']; assert fit['converged'] and len(fit['states']) == len(training)
            shared_starts.extend(fit['multistart']); fit_count += 1
            predicted = predict(VOLTAGE, fit['shared_parameters'], fit['states'], mode=mode)
            compare(scores(y[training-1], predicted, weights[training-1], training), split['training_scores'])
            compare(split['training_scores']['all']['weighted_rmse_decade'], fit['rmse_decade'])
            for protocol, saved in split['targets'].items():
                local = saved['identification']; local_count += 1
                assert local['shared_parameters'] == fit['shared_parameters'] and local['converged']
                predicted = predict(VOLTAGE, fit['shared_parameters'], local['states'], mode=mode)
                compare(predicted, saved['predicted_log10_A'])
                compare(scores(y[target-1], predicted, weights[target-1], target), saved['scores'])
                mask = VOLTAGE > 0 if protocol == 'SET_only' else np.ones(len(VOLTAGE), bool)
                error = predicted[:, mask]-y[target-1][:, mask]; w = weights[target-1][:, mask]
                compare(np.sqrt(np.mean(np.sum(w*error**2, axis=1)/w.sum(axis=1))), local['rmse_decade'])
                if label.startswith('fold_'):pooled[protocol][target-1] = predicted
        assert sorted(target_ids) == ids.tolist()
        for protocol, predicted in pooled.items():
            compare(scores(y, predicted, weights, ids), candidate[protocol+'_five_fold_scores'])
    assert fit_count == 14 and len(shared_starts) == 56 and local_count == 24
    assert sum(not a['converged'] for a in shared_starts) == 7
    assert len(validation['quadrature_refinements']) == 38
    maximum_quad = max(r['maximum_24_48_difference_decade'] for r in validation['quadrature_refinements'])
    assert maximum_quad < 1e-6
    lookup = GateLookup(load_reverse_lookup(), 1.1); max_kcl = 0.; n_port = 0
    for mode, data in validation['native_RESET_port_forecasts'].items():
        assert sorted(r['cycle_id'] for r in data['cycles']) == ids.tolist()
        for row in data['cycles']:
            assert row['cycle_id'] not in row['training_ids'] and row['root_count_same_after_refinement']
            ref = row['refined']; assert ref['ambiguous_count'] == 0
            source = main['candidates'][mode]['splits'][row['split']]
            j = source['target_ids'].index(row['cycle_id'])
            assert row['state'] == source['targets']['SET_only']['identification']['states'][j]
            vh = [r['internal_voltage_V'] for r in ref['root_diagnostics']]
            predicted = 10**predict(-np.array(vh), source['fit']['shared_parameters'], row['state'], mode=mode, order=48)[0]
            np.testing.assert_allclose(predicted, ref['predicted_current_A'], rtol=1e-12, atol=0)
            vd = [r['transistor_voltage_V'] for r in ref['root_diagnostics']]
            it = np.interp(vd, lookup.voltage, lookup.current)
            max_kcl = max(max_kcl, float(np.max(np.abs(it-predicted)/predicted)))
            assert all(len(r['roots']) == 1 and r['roots'][0]['stable_load_line'] for r in ref['root_diagnostics'])
            err = np.log10(predicted/row['observed_current_A'])
            compare(err, row['errors_decade']); compare(np.sqrt(np.mean(err**2)), row['rmse_decade'])
            n_port += len(predicted)
        for group, condition in [('all', lambda c:True), ('first3', lambda c:c <= 3), ('later', lambda c:c > 3)]:
            rows = [r for r in data['cycles'] if condition(r['cycle_id'])]
            compare(np.sqrt(np.mean([r['rmse_decade']**2 for r in rows])), data['groups'][group]['equal_pair_rmse_decade'])
    assert n_port == 1166 and max_kcl < 1e-10
    for mode, rows in validation['SET_energy_information_profiles'].items():
        assert sorted(r['cycle_id'] for r in rows) == ids.tolist()
        for row in rows:
            cost = np.array(row['SET_rmse_profile_decade']); eps = np.array(row['epsilon_grid_ev'])
            retained = cost <= cost.min()+.005
            assert sum(retained) == row['n_retained_states']
            compare([eps[retained].min(), eps[retained].max()], row['retained_epsilon_min_max_ev'])
            compare(cost.min(), row['grid_best_SET_rmse_decade'])
    assert len(validation['symmetric_single_reservoir_lever_bound']) == 24
    for row in validation['symmetric_single_reservoir_lever_bound']:
        compare(VT*abs(np.log(row['favorable_ratio_upper']))/(2*row['internal_voltage_V']), row['necessary_abs_lever_arm_favorable'])
    assert len(profiles['profiles']) == 8 and len(maximal['splits']) == 6
    for row in profiles['profiles']:
        fit = row['fit']; assert fit['converged']
        chosen = ids if row['group'] == 'all53' else ids[:3]
        compare(scores(y[chosen-1], predict(VOLTAGE, fit['shared_parameters'], fit['states']), weights[chosen-1], chosen), row['scores'])
    pool = np.empty_like(y); targets = []
    for label, row in maximal['splits'].items():
        assert not set(row['training_ids']) & set(row['target_ids'])
        assert row['fit']['converged'] and row['identification']['converged']
        assert row['fit']['shared_parameters'][3] == 1
        predicted = predict(VOLTAGE, row['fit']['shared_parameters'], row['identification']['states'])
        compare(predicted, row['predicted_log10_A'])
        if label.startswith('fold_'):
            indices = np.array(row['target_ids'])-1; pool[indices] = predicted; targets.extend(row['target_ids'])
    assert sorted(targets) == ids.tolist()
    compare(scores(y, pool, weights, ids), maximal['five_fold_scores'])
    assert len(high['rows']) == 106 and all(r['n_points'] >= 5 for r in high['rows'])
    assert high['summary']['equilibrium']['n_observed_below_one'] == 53
    assert high['summary']['equilibrium']['n_predicted_below_one'] == 0
    log = ROOT.parent/'verification/2026-09-20/side-charge-tests-final.log'
    passed = re.findall(r'Ran (\d+) tests in ([\d.]+)s\s+OK', log.read_text())
    assert passed and int(passed[-1][0]) == 186
    audit = {'scope':'Conditional charge readout checked; failed RESET forecasts and slope constraints remain explicit',
             'input_provenance':provenance, 'prior_manifests_verified_without_rewriting':old,
             'n_pairs':53, 'n_valid_grid_points':5072, 'n_missing_high_voltage_points_retained':16,
             'primary_shared_fits':fit_count, 'primary_starts':len(shared_starts), 'unconverged_nonselected_starts':7,
             'selected_primary_fits_all_converged':True, 'target_identifications':local_count,
             'whole_pair_holdouts_and_training_only_shared_parameters_checked':True,
             'maximum_24_48_refinement_decade':maximum_quad, 'native_terminal_predictions_rescored':n_port,
             'maximum_recomputed_kcl_relative_error':max_kcl, 'SET_energy_profiles':106,
             'posthoc_fixed_lever_profiles':8, 'posthoc_maximal_gate_splits':6,
             'test_count':int(passed[-1][0]), 'test_seconds':float(passed[-1][1]),
             'test_log_sha256':digest(log), 'verifier_source_sha256':digest(Path(__file__))}
    audit_path = ROOT/'analysis/side_charge_validation_audit.json'
    audit_path.write_text(json.dumps(audit, indent=2)+'\n')
    files = paths+[Path(__file__), audit_path]+[ROOT/p for report in reports for p in report['input_sha256']]
    files += [ROOT/'analysis'/p for p in ['SIDE_CHARGE_READOUT_RESULTS_2026-09-21.md', 'plot_side_charge_evidence.py',
              'figures/36_side_charge_evidence.png', 'figures/36_side_charge_evidence.pdf']]
    logs = [ROOT.parent/'verification/2026-09-20'/p for p in ['side-charge-unit.log', 'side-charge-unit-final.log',
            'side-charge-circuit-unit.log', 'side-charge-comparison.log', 'side-charge-validation.log',
            'side-charge-lever-profiles.log', 'side-charge-maximal-gate.log', 'side-charge-high-voltage.log', 'side-charge-tests-final.log']]
    manifest = {'scope':audit['scope'], 'production_model_changed':False,
                'artifacts':{str(p.relative_to(ROOT)):{'sha256':digest(p), 'bytes':p.stat().st_size} for p in dict.fromkeys(files)},
                'external_workspace_logs':{str(p.relative_to(ROOT.parent)):{'sha256':digest(p), 'bytes':p.stat().st_size} for p in logs}}
    (ROOT/'analysis/side_charge_artifact_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    run()
