"""Audit spectral readout, conditional memory bounds and preserved artifacts."""
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
from vulcan2d.charge_spectrum import predict
from vulcan2d.coupled_charge import VT


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: same(a[k], b[k])
    else:
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-10)


def high_slopes(prediction, weights):
    result = []
    for p, w in zip(prediction, weights):
        mask = (VOLTAGE >= 2) & (VOLTAGE <= 3) & (w > 0)
        x = np.log10(VOLTAGE[mask]); x -= x.mean()
        result.append(float(p[mask]@x/(x@x)))
    return result


def relaxation_bound(shared):
    low = shared[0]-shared[5]
    f = np.exp(-np.logaddexp(0., low/VT))
    bound = (1+np.sqrt(f))*np.exp(np.logaddexp(0., -low/VT))
    return float(bound)


def run():
    paths = [ROOT/'analysis'/p for p in ['charge_spectrum_comparison.json',
             'charge_spectrum_validation.json', 'charge_spectrum_expanded_prior.json',
             'charge_spectrum_expanded_groups.json']]
    main, validation, expanded, wide_groups = reports = [json.loads(p.read_text()) for p in paths]
    for report in reports:
        for name, sha in report['input_sha256'].items():
            assert digest(ROOT/name) == sha, 'Changed source: '+name
    old = []
    for stem in ('coupled_charge', 'side_charge', 'barrier_height_polarity',
                 'barrier_state', 'finite_path', 'bidirectional'):
        p = ROOT/f'analysis/{stem}_artifact_manifest.json'; record = json.loads(p.read_text())
        count = 0
        for key, base in [('artifacts', ROOT), ('external_workspace_logs', ROOT.parent), ('external_workspace_sources', ROOT.parent)]:
            for name, info in record.get(key, {}).items():
                assert digest(base/name) == info['sha256'], 'Changed prior artifact: '+name
                count += 1
        old.append({'manifest': p.name, 'sha256': digest(p), 'verified_file_count': count})
    ids, y, w, selection = assemble()
    assert main['status'] == 'complete' and selection == main['selection']
    np.testing.assert_array_equal(main['ids'], ids); np.testing.assert_array_equal(main['voltage_V'], VOLTAGE)
    np.testing.assert_array_equal(main['observed_log10_current_A'], y); np.testing.assert_array_equal(main['weights'], w)
    assert y.shape == (53, 96) and np.count_nonzero(w) == 5072
    assert sum(len(r['missing']) for r in selection) == 16
    assert set(main['splits']) == {'all_descriptive', 'chronological'} | {f'fold_{a}_{b}' for a, b in FOLDS}
    reference = json.loads((ROOT/'analysis/coupled_charge_comparison.json').read_text())
    pools = {p: np.empty_like(y) for p in ('SET_only', 'joint_descriptive')}
    shared_attempts, local_attempts, target_ids = [], [], []
    for label, row in main['splits'].items():
        train, target = np.array(row['training_ids']), np.array(row['target_ids'])
        assert row['training_ids'] == reference['splits'][label]['training_ids']
        assert row['target_ids'] == reference['splits'][label]['target_ids']
        if label != 'all_descriptive':
            assert not set(train) & set(target)
            if label.startswith('fold_'): target_ids.extend(target.tolist())
        f = row['fit']; assert f['converged'] and len(f['states']) == len(train)
        shared_attempts.extend(f['multistart'])
        assert len(f['multistart']) == 5 and f['multistart'][-1]['source'] == 'same_training_zero_width_fit'
        assert f['rmse_decade'] <= reference['splits'][label]['fit']['rmse_decade']+1e-7
        p = predict(VOLTAGE, f['shared_parameters'], f['states'])
        same(scores(y[train-1], p, w[train-1], train), row['training_scores'])
        same(row['training_scores']['all']['weighted_rmse_decade'], f['rmse_decade'])
        if label == 'all_descriptive': same(p, row['predicted_log10_A'])
        for protocol, saved in row['targets'].items():
            local = saved['identification']; assert local['converged']
            assert local['shared_parameters'] == f['shared_parameters']
            assert len(local['states']) == len(target)
            local_attempts.extend(local['multistart'])
            prediction = predict(VOLTAGE, f['shared_parameters'], local['states'])
            same(prediction, saved['predicted_log10_A'])
            same(scores(y[target-1], prediction, w[target-1], target), saved['scores'])
            mask = VOLTAGE > 0 if protocol == 'SET_only' else np.ones(len(VOLTAGE), bool)
            err, ww = prediction[:, mask]-y[target-1][:, mask], w[target-1][:, mask]
            same(np.sqrt(np.mean(np.sum(ww*err**2, axis=1)/ww.sum(axis=1))), local['rmse_decade'])
            if label.startswith('fold_'): pools[protocol][target-1] = prediction
    assert len(shared_attempts) == 35 and len(local_attempts) == 12
    assert all(r['converged'] for r in shared_attempts+local_attempts)
    assert sorted(target_ids) == ids.tolist()
    for protocol, prediction in pools.items():
        same(scores(y, prediction, w, ids), main[protocol+'_five_fold_scores'])
        same(main[protocol+'_five_fold_scores'], validation['refined_384_point_five_fold_scores'][protocol])
    assert len(validation['quadrature_refinements']) == 19
    assert max(r['maximum_96_192_difference_decade'] for r in validation['quadrature_refinements']) < 2e-7
    assert len(validation['independent_adaptive_integrals']) == 42
    assert max(r['adaptive_vs_384_difference_decade'] for r in validation['independent_adaptive_integrals']) < 1e-8
    for row in validation['conditional_relaxation_bounds']:
        same(relaxation_bound(main['splits'][row['split']]['fit']['shared_parameters']), row['bias_to_zero_slow_rate_ratio_upper_bound'])
    assert validation['random_relaxation_bound_checks'] == 100
    assert validation['independent_four_state_propagation_checks'] == 100
    assert validation['maximum_mean_occupation_propagation_difference'] < 2e-11
    f = expanded['fit']; assert f['converged'] and len(f['multistart']) == 4
    assert all(r['converged'] for r in f['multistart'])
    prediction = predict(VOLTAGE, f['shared_parameters'], f['states'], panels=16)
    refined = predict(VOLTAGE, f['shared_parameters'], f['states'], panels=32)
    extra_refinement = float(np.max(np.abs(prediction-refined)))
    assert extra_refinement < 2e-7
    same(prediction, expanded['predicted_log10_A']); same(scores(y, prediction, w, ids), expanded['scores'])
    assert wide_groups['status'] == 'complete'
    assert set(wide_groups['splits']) == set(main['splits'])-{'all_descriptive'}
    wide_pools = {p: np.empty_like(y) for p in ('SET_only', 'joint_descriptive')}
    wide_checks, wide_starts, wide_local, wide_bounds = [], [], [], []
    for label, row in wide_groups['splits'].items():
        assert row['training_ids'] == main['splits'][label]['training_ids']
        assert row['target_ids'] == main['splits'][label]['target_ids']
        train, target = np.array(row['training_ids']), np.array(row['target_ids'])
        assert not set(train) & set(target)
        assert row['initial_shared_vectors'][0] == main['splits'][label]['fit']['shared_parameters']
        fwide = row['fit']; assert fwide['converged']
        wide_starts.extend(fwide['multistart'])
        wide_bounds.append({'split': label, 'bias_to_zero_slow_rate_ratio_upper_bound': relaxation_bound(fwide['shared_parameters'])})
        pw = predict(VOLTAGE, fwide['shared_parameters'], fwide['states'], panels=16)
        same(scores(y[train-1], pw, w[train-1], train), row['training_scores'])
        states = [('training', fwide['states'])]
        for protocol, saved in row['targets'].items():
            local = saved['identification']; assert local['converged']
            assert local['shared_parameters'] == fwide['shared_parameters']
            wide_local.extend(local['multistart'])
            pw = predict(VOLTAGE, fwide['shared_parameters'], local['states'], panels=16)
            same(pw, saved['predicted_log10_A'])
            same(scores(y[target-1], pw, w[target-1], target), saved['scores'])
            if label.startswith('fold_'): wide_pools[protocol][target-1] = pw
            states.append((protocol, local['states']))
        for scope, local in states:
            pw = predict(VOLTAGE, fwide['shared_parameters'], local, panels=16)
            refined_wide = predict(VOLTAGE, fwide['shared_parameters'], local, panels=32)
            difference = float(np.max(np.abs(pw-refined_wide)))
            assert difference < 2e-7
            wide_checks.append({'split': label, 'scope': scope, 'maximum_192_384_difference_decade': difference})
    assert len(wide_starts) == 30 and len(wide_local) == 12
    assert all(r['converged'] for r in wide_local)
    for protocol, pw in wide_pools.items():
        same(scores(y, pw, w, ids), wide_groups[protocol+'_five_fold_scores'])
    test_log = ROOT.parent/'verification/2026-09-20/charge-spectrum-tests-final.log'
    passed = re.findall(r'Ran (\d+) tests in ([\d.]+)s\s+OK', test_log.read_text())
    assert passed and int(passed[-1][0]) == 203
    audit = {'scope': 'Energy-spectrum and exact charge-dynamics diagnostics; no successful full-cycle mechanism or physical retention time',
             'n_pairs': 53, 'n_valid_points': 5072, 'n_masked_high_voltage_positions': 16,
             'shared_fits': 7, 'shared_starts': 35, 'target_identifications': 12,
             'all_selected_and_nonselected_primary_starts_converged': True,
             'whole_pair_groups_parameters_predictions_and_scores_verified': True,
             'prior_manifests_verified_without_rewriting': old,
             'spectral_state_groups_refined': 19, 'independent_adaptive_integrals': 42,
             'expanded_prior_starts': 4, 'expanded_prior_192_384_difference_decade': extra_refinement,
             'expanded_grouped_shared_fits': 6, 'expanded_grouped_starts': 30,
             'expanded_grouped_target_identifications': 12,
             'expanded_grouped_unconverged_nonselected_starts': sum(not r['converged'] for r in wide_starts),
             'expanded_grouped_refinements': wide_checks,
             'expanded_grouped_relaxation_bounds': wide_bounds,
             'observed_high_slopes_quantile_10_50_90': np.quantile(high_slopes(y, w), [.1, .5, .9]).tolist(),
             'primary_high_slopes_quantile_10_50_90': np.quantile(high_slopes(np.array(main['splits']['all_descriptive']['predicted_log10_A']), w), [.1, .5, .9]).tolist(),
             'expanded_high_slopes_quantile_10_50_90': np.quantile(high_slopes(prediction, w), [.1, .5, .9]).tolist(),
             'expanded_prior_slow_rate_acceleration_upper_bound': relaxation_bound(f['shared_parameters']),
             'expanded_prior_scope': 'Post-hoc all-data and grouped sensitivities; primary scores preserved, model choice not externally held out',
             'test_count': int(passed[-1][0]), 'test_seconds': float(passed[-1][1]),
             'test_log_sha256': digest(test_log), 'verifier_source_sha256': digest(Path(__file__))}
    audit_path = ROOT/'analysis/charge_spectrum_validation_audit.json'
    audit_path.write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
    files = paths+[Path(__file__), audit_path]+[ROOT/p for r in reports for p in r['input_sha256']]
    files += [ROOT/'analysis'/p for p in ['CHARGE_SPECTRUM_RESULTS_2026-09-21.md',
              'plot_charge_spectrum_evidence.py', 'write_charge_spectrum_report.py',
              'figures/38_charge_spectrum_evidence.png', 'figures/38_charge_spectrum_evidence.pdf',
              'CHARGE_DYNAMICS_LITERATURE_NOTE_2026-09-21.md', 'charge_dynamics_literature_discovery.json']]
    logs = [ROOT.parent/'verification/2026-09-20'/p for p in ['charge-spectrum-unit.log',
            'charge-spectrum-unit-final.log', 'charge-moment-unit.log', 'charge-spectrum-comparison.log',
            'charge-spectrum-validation.log', 'charge-spectrum-expanded.log', 'charge-spectrum-expanded-groups.log',
            'charge-spectrum-tests-final.log']]
    manifest = {'scope': audit['scope'], 'production_model_changed': False,
                'artifacts': {str(p.relative_to(ROOT)): {'sha256': digest(p), 'bytes': p.stat().st_size}
                              for p in dict.fromkeys(files)},
                'external_workspace_logs': {str(p.relative_to(ROOT.parent)): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in logs}}
    sources = [ROOT.parent/f'verification/2026-09-20/physics-sources/michalek2009-charge-dynamics.{ext}' for ext in ('pdf', 'txt')]
    manifest['external_workspace_sources'] = {str(p.relative_to(ROOT.parent)): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in sources}
    (ROOT/'analysis/charge_spectrum_artifact_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    run()
