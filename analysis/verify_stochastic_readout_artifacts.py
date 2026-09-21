"""Freeze the support/order evidence without claiming predictive closure."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np
from certify_cycle_order_permutation import certify
ROOT = Path(__file__).resolve().parents[1]


def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def info(p): return {'sha256': digest(p), 'bytes': p.stat().st_size}


def run():
    prior = []; files = set(); results = []; count = 0
    for stem in ('retained_charge_configuration', 'activated_configuration', 'charge_spectrum', 'coupled_charge', 'side_charge',
                 'barrier_height_polarity', 'barrier_state', 'finite_path', 'bidirectional'):
        path = ROOT/'analysis'/f'{stem}_artifact_manifest.json'; record = json.loads(path.read_text()); verified = 0
        for field, base in [('artifacts', ROOT), ('external_workspace_logs', ROOT.parent), ('external_workspace_sources', ROOT.parent)]:
            for name, expected in record.get(field, {}).items():
                assert digest(base/name) == expected['sha256'], name; verified += 1
        prior.append({'manifest': str(path.relative_to(ROOT)), 'sha256': digest(path), 'verified_files': verified})
    def verify_record(path):
        r = json.loads(path.read_text()); files.add(path)
        for name, expected in r['input_sha256'].items():
            assert digest(ROOT/name) == expected, name; files.add(ROOT/name)
        return r
    for split in ('all_descriptive', 'chronological'):
        for states in ('two', 'four'):
            originals = []
            for suffix in ('', '_reconciled'):
                path = ROOT/'analysis'/f'stochastic_readout_support_{split}_{states}{suffix}.json'
                r = verify_record(path); assert r['status'] == 'complete'
                assert digest(path.with_suffix('.npz')) == r['readout_sha256']; files.add(path.with_suffix('.npz'))
                if suffix:
                    assert r['starts'][:5] == originals[0]['starts'] and len(r['starts']) == 6; count += 1
                else: assert len(r['starts']) == 5; count += 5
                assert all(s['converged'] for s in r['starts'])
                best = r['starts'][r['selected_start']]
                assert best['rmse_decade'] <= min(s['rmse_decade'] for s in r['starts'])+1e-12
                np.testing.assert_array_equal(best['parameters'], r['parameters'])
                assert abs(best['rmse_decade']-r['training_support_scores']['rmse_decade']) < 1e-10
                originals.append(r)
                if suffix: results.append({'fit': path.stem, 'support_error': best['rmse_decade'], 'predictive_error': None})
        path = ROOT/'analysis'/f'stochastic_domain_support_{split}_N4.json'; r = verify_record(path)
        assert r['status'] == 'complete' and len(r['starts']) == 5; count += 5
        assert all(s['converged'] for s in r['starts'])
        best = r['starts'][r['selected_start']]; assert best['rmse_decade'] <= min(s['rmse_decade'] for s in r['starts'])+1e-12
        assert abs(best['rmse_decade']-r['training_support_scores']['rmse_decade']) < 1e-10
        assert digest(path.with_suffix('.npz')) == r['readout_sha256']; files.add(path.with_suffix('.npz'))
        results.append({'fit': path.stem, 'support_error': best['rmse_decade'], 'predictive_error': None})
    assert count == 34
    pure = verify_record(ROOT/'analysis/stochastic_readout_support_validation.json')
    finite = verify_record(ROOT/'analysis/stochastic_domain_refinement.json')
    for name, r in [('stochastic_readout_support_validation', pure), ('stochastic_domain_refinement', finite)]:
        path = ROOT/'analysis'/f'{name}.npz'; assert digest(path) == r['arrays_sha256']; files.add(path)
    assert len(pure['candidates']) == 4 and len(finite['candidates']) == 12
    for r in pure['candidates'].values():
        assert r['all_refinements_pass'] and r['independent_frozen_root_count'] == 120
        assert r['maximum_root_voltage_error_V'] < 1e-8 and r['delta_U_swap_maximum_log_error'] < 1e-10
        for v in r['refinements']:
            assert v['maximum_conditional_current_log_change'] < .005 and v['weighted_absolute_residual_change'] < .001
    for r in finite['candidates'].values():
        assert r['all_refinements_pass'] and r['maximum_mixed_root_voltage_error_V'] < 1e-8
        for v in r['refinements']:
            assert v['maximum_configuration_current_log_change'] < .005 and v['weighted_absolute_residual_change'] < .001
            assert v['maximum_kcl_relative_error'] < 1e-8
    verify_record(ROOT/'analysis/stochastic_readout_prefit.json')
    order = verify_record(ROOT/'analysis/cycle_order_evidence.json')
    exact = verify_record(ROOT/'analysis/cycle_order_exact_certificate.json')
    for name, row in order['analyses'].items():
        recomputed = certify(row['features']); assert recomputed == exact['analyses'][name]
        assert recomputed['certified'] and len(recomputed['extrema']) == 312
        assert len(recomputed['qualifying_prefix_set_events']) == 2
        assert recomputed['probability_numerator'] == 1 and recomputed['probability_denominator'] == 11713
    testlog = ROOT.parent/'verification/2026-09-20/stochastic_readout_verified_tests.log'
    match = re.search(r'Ran (\d+) tests in ([\d.]+)s\s+OK', testlog.read_text()); assert match and int(match[1]) == 35
    names = ['analysis/STOCHASTIC_READOUT_SUPPORT_DESIGN_2026-09-21.md', 'analysis/STOCHASTIC_DOMAIN_REFIT_DESIGN_2026-09-21.md',
             'analysis/CYCLE_ORDER_TEST_DESIGN_2026-09-21.md', 'analysis/STOCHASTIC_READOUT_RESULTS_2026-09-21.md',
             'analysis/CONTACT_ASYMMETRY_MEMORY_DESIGN_2026-09-21.md',
             'analysis/fit_stochastic_readout_support.py', 'analysis/fit_stochastic_domain_support.py',
             'analysis/check_stochastic_readout_prefit.py', 'analysis/check_stochastic_readout_support.py',
             'analysis/check_stochastic_domain_refinement.py', 'analysis/check_cycle_order_evidence.py',
             'analysis/certify_cycle_order_permutation.py', 'analysis/plot_stochastic_readout_evidence.py',
             'analysis/write_stochastic_readout_report.py', 'analysis/verify_stochastic_readout_artifacts.py',
             'vulcan2d/finite_readout_support.py', 'vulcan2d/_finite_readout_core.cpp',
             'tests/test_finite_readout_support.py', 'tests/test_cycle_order_evidence.py', 'tests/test_cycle_order_certificate.py',
             'analysis/figures/41_stochastic_readout_evidence.png', 'analysis/figures/41_stochastic_readout_evidence.pdf']
    files.update(ROOT/name for name in names)
    audit = {'scope': 'Current-support feasibility and cycle-order evidence; no validated stochastic microscopic prediction',
             'goal_complete': False, 'optimization_start_count': count, 'all_starts_converged': True,
             'related_test_count': 35, 'test_log_sha256': digest(testlog), 'fits': results,
             'prior_manifests_verified_without_rewriting': prior, 'cycle_order_exact_joint_p': 1/11713,
             'independent_root_checks': sum(r['independent_frozen_root_count'] for r in pure['candidates'].values())+
                                        sum(r['independent_mixed_root_checks'] for r in finite['candidates'].values()),
             'verifier_sha256': digest(Path(__file__))}
    path = ROOT/'analysis/stochastic_readout_artifact_audit.json'; path.write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n'); files.add(path)
    folder = ROOT.parent/'verification/2026-09-20'
    logs = sorted(set(folder.glob('stochastic_readout*.log')) | set(folder.glob('stochastic_domain*.log')) | set(folder.glob('cycle_order*.log')))
    manifest = {'scope': 'Conditional readout/order evidence; original research objective remains incomplete', 'production_model_changed': False,
                'artifacts': {str(p.relative_to(ROOT)): info(p) for p in sorted(files)},
                'external_workspace_logs': {str(p.relative_to(ROOT.parent)): info(p) for p in logs}}
    destination = ROOT/'analysis/stochastic_readout_artifact_manifest.json'; destination.write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    print('Verified', len(files), 'artifacts,', len(logs), 'logs,', len(prior), 'prior manifests;', count, 'starts; 35 tests;', audit['independent_root_checks'], 'roots')


if __name__ == '__main__': run()
