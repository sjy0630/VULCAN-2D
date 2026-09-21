"""Recompute evidence, preserved provenance and side-flux cancellation scales.

This auditor does not refit the physical model or rewrite executed sources.
The information controls are cheap enough to refit all nested outer records.
"""
from decimal import Decimal, localcontext
from pathlib import Path
import hashlib
import json
import re
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analysis'))
from compare_parallel_paths import assemble, VOLTAGE, FOLDS
from compare_side_charge_readout import scores
from check_full_set_information import (features_and_target, select_and_fit,
                                        evaluate_ridge, summaries)
from diagnose_cross_polarity_physics import read_pairs
from vulcan2d.coupled_charge import predict, stationary, log_rates, EDGES
from vulcan2d.coupled_charge_reduced import predict as reduced_predict


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same(actual, expected):
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            same(actual[key], expected[key])
    elif isinstance(actual, list) and actual and isinstance(actual[0], dict):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            same(left, right)
    else:
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)


def decimal_side_flux(voltage, shared, energy, precision=160):
    """High-precision generator solve; net side flux relative to gross exchange.

    Rates are exponentiated from the same elementary double-precision log rates.
    This validates stationarity/subtraction, not accuracy of material inputs.
    """
    with localcontext() as ctx:
        ctx.prec = precision
        rates = [Decimal(str(float(x))).exp()
                 for x in log_rates(voltage, shared, energy)[0]]
        q = [[Decimal(0) for _ in range(4)] for _ in range(4)]
        for rate, (a, b) in zip(rates, EDGES):
            q[b][a] += rate
            q[a][a] -= rate
        q[3] = [Decimal(1)] * 4
        matrix = [row + [Decimal(i == 3)] for i, row in enumerate(q)]
        for col in range(4):
            pivot = max(range(col, 4), key=lambda i: abs(matrix[i][col]))
            matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
            divisor = matrix[col][col]
            matrix[col] = [x / divisor for x in matrix[col]]
            for i in range(4):
                if i != col:
                    factor = matrix[i][col]
                    matrix[i] = [a - factor*b for a, b in zip(matrix[i], matrix[col])]
        p = [r[-1] for r in matrix]
        entering = rates[8]*p[0] + rates[10]*p[1]
        leaving = rates[9]*p[2] + rates[11]*p[3]
        return float(abs(entering-leaving) / (entering+leaving))


def run():
    names = ['coupled_charge_comparison.json', 'full_set_information_audit.json',
             'coupled_charge_validation.json', 'coupled_charge_slope_tradeoff.json',
             'coupled_charge_reduction_check.json']
    paths = [ROOT / 'analysis' / name for name in names]
    main, information, validation, slope, reduction = reports = [
        json.loads(p.read_text()) for p in paths]
    provenance = []
    for path, report in zip(paths, reports):
        for name, sha in report['input_sha256'].items():
            assert digest(ROOT / name) == sha, f'{path.name}: source changed: {name}'
        provenance.append({'report': path.name, 'input_count': len(report['input_sha256'])})
    prior = []
    for stem in ('side_charge', 'barrier_height_polarity', 'barrier_state',
                 'finite_path', 'bidirectional'):
        path = ROOT / f'analysis/{stem}_artifact_manifest.json'
        manifest = json.loads(path.read_text())
        count = 0
        for key, base in [('artifacts', ROOT), ('external_workspace_logs', ROOT.parent)]:
            for name, record in manifest.get(key, {}).items():
                assert digest(base / name) == record['sha256'], f'Old artifact changed: {name}'
                count += 1
        prior.append({'manifest': path.name, 'sha256': digest(path), 'verified_file_count': count})

    ids, y, w, selection = assemble()
    assert main['status'] == 'complete' and main['selection'] == selection
    np.testing.assert_array_equal(main['ids'], ids)
    np.testing.assert_array_equal(main['voltage_V'], VOLTAGE)
    np.testing.assert_array_equal(main['observed_log10_current_A'], y)
    np.testing.assert_array_equal(main['weights'], w)
    assert y.shape == (53, 96) and np.count_nonzero(w) == 5072
    assert sum(len(r['missing']) for r in selection) == 16
    expected_labels = {'all_descriptive', 'chronological'} | {f'fold_{a}_{b}' for a, b in FOLDS}
    assert set(main['splits']) == expected_labels
    pool = {key: np.empty_like(y) for key in ('SET_only', 'joint_descriptive')}
    targets, shared_starts, local_starts, equivalence, fluxes = [], [], [], [], []
    for label, row in main['splits'].items():
        training, target = np.array(row['training_ids']), np.array(row['target_ids'])
        if label == 'all_descriptive':
            np.testing.assert_array_equal(training, ids)
            np.testing.assert_array_equal(target, ids)
            assert not row['targets']
        else:
            assert not set(training) & set(target)
            if label == 'chronological':
                expected_target = ids[ids > 35]
            else:
                lo, hi = map(int, label.split('_')[1:])
                expected_target = ids[(ids >= lo) & (ids <= hi)]
                targets.extend(target.tolist())
            np.testing.assert_array_equal(target, expected_target)
            np.testing.assert_array_equal(training, np.setdiff1d(ids, target))
            assert set(row['targets']) == set(pool)
        fit = row['fit']
        assert fit['converged'] and len(fit['states']) == len(training)
        shared_starts.extend(fit['multistart'])
        predicted = predict(VOLTAGE, fit['shared_parameters'], fit['states'])
        same(scores(y[training-1], predicted, w[training-1], training), row['training_scores'])
        same(row['training_scores']['all']['weighted_rmse_decade'], fit['rmse_decade'])
        if label == 'all_descriptive':
            same(predicted, row['predicted_log10_A'])
        groups = [('training', fit['states'])]
        for protocol, saved in row['targets'].items():
            local = saved['identification']
            assert local['shared_parameters'] == fit['shared_parameters'] and local['converged']
            assert len(local['states']) == len(target)
            local_starts.extend(local['multistart'])
            pred = predict(VOLTAGE, fit['shared_parameters'], local['states'])
            same(pred, saved['predicted_log10_A'])
            same(scores(y[target-1], pred, w[target-1], target), saved['scores'])
            mask = VOLTAGE > 0 if protocol == 'SET_only' else np.ones(len(VOLTAGE), bool)
            error, weights = pred[:, mask] - y[target-1][:, mask], w[target-1][:, mask]
            same(np.sqrt(np.mean(np.sum(weights*error**2, axis=1)/weights.sum(axis=1))), local['rmse_decade'])
            if label.startswith('fold_'):
                pool[protocol][target-1] = pred
            groups.append((protocol, local['states']))
        for scope, states in groups:
            state = np.array(states)
            old = predict(VOLTAGE, fit['shared_parameters'], state)
            reduced = reduced_predict(VOLTAGE, fit['shared_parameters'][:5], state)
            difference = float(np.max(np.abs(old-reduced)))
            assert difference < 1e-10
            equivalence.append({'split': label, 'scope': scope, 'n_state_vectors': len(state),
                                'maximum_difference_decade': difference})
            out = stationary(VOLTAGE[None, :], fit['shared_parameters'], state[:, 1, None])
            p = out['probabilities']
            k = np.exp(log_rates(VOLTAGE[None, :], fit['shared_parameters'], state[:, 1, None])[0])
            incoming, outgoing = k[8]*p[0]+k[10]*p[1], k[9]*p[2]+k[11]*p[3]
            gross = incoming+outgoing
            difference = np.abs(incoming-outgoing)
            sequential = np.abs(out['gross_side_current'])
            ratio = difference/gross
            assert float(np.max(ratio)) < 1e-12
            assert float(np.max(sequential/gross)) < 1e-12
            fluxes.append({'split': label, 'scope': scope,
                           'maximum_absolute_residual_in_Gamma_R_units': float(np.max(difference)),
                           'maximum_residual_relative_to_gross_exchange': float(np.max(ratio)),
                           'maximum_sequential_residual_relative_to_gross_exchange': float(np.max(sequential/gross)),
                           'maximum_sequential_residual_relative_to_net_transport':
                               float(np.max(sequential/np.abs(out['current_scale'])))})
    assert sorted(targets) == ids.tolist()
    assert len(shared_starts) == 28 and all(r['converged'] for r in shared_starts)
    assert len(local_starts) == 12 and all(r['converged'] for r in local_starts)
    for protocol, pred in pool.items():
        same(scores(y, pred, w, ids), main[protocol+'_five_fold_scores'])
    assert len(equivalence) == 19
    for actual, recorded in zip(equivalence, reduction['exact_reduction_checks']):
        assert actual['split'] == recorded['split'] and actual['scope'] == recorded['scope']
        same(actual['maximum_difference_decade'], recorded['maximum_difference_decade'])
    leakage = reduction['SET_information_leakage_checks']
    assert len(leakage) == 25 and all(not r['target_RESET_100_decade_perturbation_changes_prediction'] for r in leakage)

    decimal_checks = validation['independent_decimal_checks']
    assert len(decimal_checks) == 210
    assert max(r['maximum_probability_difference'] for r in decimal_checks) < 1e-12
    assert max(r['log10_current_difference_decade'] for r in decimal_checks) < 1e-12
    decimal_flux, refined_decimal_flux = [], []
    for row in decimal_checks:
        split = main['splits'][row['split']]
        j = split['training_ids'].index(row['training_cycle_id'])
        residual = decimal_side_flux(row['voltage_V'], split['fit']['shared_parameters'], split['fit']['states'][j][1])
        # Relative residual loses digits when side exchange is exponentially
        # small. Require both adequate accuracy and convergence with precision.
        refined = decimal_side_flux(row['voltage_V'], split['fit']['shared_parameters'],
                                    split['fit']['states'][j][1], precision=240)
        assert residual < 1e-110 and refined < 1e-190
        decimal_flux.append(residual)
        refined_decimal_flux.append(refined)

    operator = np.zeros_like(w)
    for j, weight in enumerate(w):
        use = (VOLTAGE >= 2) & (VOLTAGE <= 3) & (weight > 0)
        x = np.log10(VOLTAGE[use]); x -= x.mean()
        operator[j, use] = x/(x@x)
    observed_slope = np.sum(operator*y, axis=1)
    assert len(slope['fits']) == 3 and [r['penalty'] for r in slope['fits']] == [1, 10, 100]
    for row in slope['fits']:
        assert row['converged'] and all(r['converged'] for r in row['multistart'])
        pred = predict(VOLTAGE, row['shared_parameters'], row['states'])
        same(pred, row['predicted_log10_A'])
        same(scores(y, pred, w, ids), row['unpenalized_scores'])
        predicted_slope = np.sum(operator*pred, axis=1)
        same(predicted_slope, row['predicted_slopes'])
        same(observed_slope, row['observed_slopes'])
        same(np.sqrt(np.mean((predicted_slope-observed_slope)**2)), row['slope_rmse'])

    raw = features_and_target(read_pairs('1.1'))
    same(raw, information['raw_feature_rows'])
    iy = np.array([r['target_RESET_low_mean_log10_A'] for r in raw])
    outer_count = 0
    for candidate, columns in information['feature_groups'].items():
        assert all(not c.startswith('target_') for c in columns)
        x = np.array([[r[c] for c in columns] for r in raw])
        for protocol, saved in information['candidates'][candidate].items():
            predicted = np.full(53, np.nan)
            seen = []
            for row in saved['splits']:
                training, target = np.array(row['training_ids'])-1, np.array(row['target_ids'])-1
                assert not set(training) & set(target)
                np.testing.assert_array_equal(training, np.setdiff1d(np.arange(53), target))
                if protocol == 'chronological':
                    np.testing.assert_array_equal(training, np.arange(35))
                elif protocol == 'leave_one_pair_out':
                    assert len(target) == 1
                else:
                    lo, hi = map(int, row['label'].split('_')[1:])
                    np.testing.assert_array_equal(target, np.arange(lo-1, hi))
                model, selection = select_and_fit(x, iy, training)
                same(model, row['model']); same(selection, row['inner_penalty_selection'])
                predicted[target] = evaluate_ridge(x[target], model)
                seen.extend(target.tolist()); outer_count += 1
            expected = list(range(35, 53)) if protocol == 'chronological' else list(range(53))
            assert sorted(seen) == expected
            same(predicted, np.array(saved['predictions_log10_A'], float))
            same(summaries(iy, predicted, np.array(expected)), saved['groups'])
    assert outer_count == 295
    persistence = information['previous_measured_RESET_reference']
    same(np.array(persistence['predictions_log10_A'], float), np.r_[np.nan, iy[:-1]])
    same(summaries(iy, np.r_[np.nan, iy[:-1]], np.arange(1, 53)), persistence['groups'])
    test_log = ROOT.parent / 'verification/2026-09-20/coupled-charge-reduced-tests-final.log'
    passed = re.findall(r'Ran (\d+) tests in ([\d.]+)s\s+OK', test_log.read_text())
    assert passed and int(passed[-1][0]) == 194
    audit = {
        'scope': 'Four-state steady readout verified; predictive closure and microscopic mechanism remain unresolved',
        'input_provenance': provenance, 'prior_manifests_verified_without_rewriting': prior,
        'n_pairs': 53, 'n_valid_grid_points': 5072, 'n_missing_high_voltage_points_retained': 16,
        'primary_shared_fits': 7, 'primary_shared_starts': 28, 'target_identifications': 12,
        'all_primary_and_target_starts_converged': True,
        'whole_pair_exclusion_shared_parameters_predictions_and_scores_recomputed': True,
        'exact_equivalence_state_groups': 19,
        'maximum_recomputed_equivalence_difference_decade': max(r['maximum_difference_decade'] for r in equivalence),
        'independent_decimal_cases_verified': 210,
        'side_flux_scope': 'All 19 fitted training/target state groups; absolute flux in Gamma_R units, not amperes',
        'side_flux_roundoff_checks': fluxes,
        'maximum_side_residual_relative_to_gross_exchange': max(r['maximum_residual_relative_to_gross_exchange'] for r in fluxes),
        'independent_160_digit_side_flux_checks': 210,
        'maximum_160_digit_side_residual_relative_to_gross_exchange': max(decimal_flux),
        'independent_240_digit_side_flux_checks': 210,
        'maximum_240_digit_side_residual_relative_to_gross_exchange': max(refined_decimal_flux),
        'initial_auditor_failure': 'An initially chosen 1e-130 relative side-flux tolerance at 160 digits failed; maximum was 1.33e-124 for exponentially small exchange. Rechecked all cases at 240 digits, retaining original source and failure log.',
        'slope_penalty_fits_rescored': 3, 'slope_penalty_scope': 'Post-hoc descriptive, same observations reused',
        'information_nested_outer_records_refitted': outer_count,
        'information_raw_features_and_native_targets_recomputed': True,
        'information_target_perturbation_records_verified': 25,
        'native_terminal_validation_this_stage': False,
        'test_count': int(passed[-1][0]), 'test_seconds': float(passed[-1][1]),
        'test_log_sha256': digest(test_log), 'verifier_source_sha256': digest(Path(__file__))}
    audit_path = ROOT / 'analysis/coupled_charge_validation_audit.json'
    audit_path.write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
    files = paths + [Path(__file__), audit_path] + [ROOT/p for r in reports for p in r['input_sha256']]
    files += [ROOT/'analysis'/p for p in ['COUPLED_CHARGE_NETWORK_RESULTS_2026-09-21.md',
               'plot_coupled_charge_evidence.py', 'figures/37_coupled_charge_evidence.png',
               'figures/37_coupled_charge_evidence.pdf']]
    log_names = ['coupled-charge-unit.log', 'coupled-charge-comparison.log', 'full-set-information.log',
                 'coupled-charge-validation.log', 'coupled-charge-tests-final.log',
                 'coupled-charge-slope-tradeoff.log', 'coupled-charge-reduction-unit.log',
                 'coupled-charge-reduction-check.log', 'coupled-charge-reduced-tests-final.log',
                 'coupled-charge-artifact-audit.log', 'coupled-charge-verifier-initial.py.txt']
    logs = [ROOT.parent/'verification/2026-09-20'/p for p in log_names]
    manifest = {'scope': audit['scope'], 'production_model_changed': False,
                'artifacts': {str(p.relative_to(ROOT)): {'sha256': digest(p), 'bytes': p.stat().st_size}
                              for p in dict.fromkeys(files)},
                'external_workspace_logs': {str(p.relative_to(ROOT.parent)): {'sha256': digest(p), 'bytes': p.stat().st_size}
                                            for p in logs}}
    (ROOT/'analysis/coupled_charge_artifact_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k: v for k, v in audit.items() if k != 'side_flux_roundoff_checks'}, indent=2))


if __name__ == '__main__':
    run()
