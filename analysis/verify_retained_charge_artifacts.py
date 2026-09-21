"""Freeze this evidence stage while preserving every earlier manifest."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np
ROOT = Path(__file__).resolve().parents[1]


def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def info(p): return {'sha256': digest(p), 'bytes': p.stat().st_size}


def run():
    prior = []
    for stem in ('activated_configuration', 'charge_spectrum', 'coupled_charge', 'side_charge',
                 'barrier_height_polarity', 'barrier_state', 'finite_path', 'bidirectional'):
        p = ROOT/'analysis'/f'{stem}_artifact_manifest.json'; old = json.loads(p.read_text()); count = 0
        for field, base in [('artifacts', ROOT), ('external_workspace_logs', ROOT.parent), ('external_workspace_sources', ROOT.parent)]:
            for name, expected in old.get(field, {}).items():
                assert digest(base/name) == expected['sha256'], f'Changed prior artifact {name}'
                count += 1
        prior.append({'manifest': str(p.relative_to(ROOT)), 'sha256': digest(p), 'verified_files': count})
    assert prior == json.loads((ROOT/'analysis/retained_charge_configuration_prior_integrity.json').read_text())
    numerical = {}; files = set(); fits = []; unselected = []; starts = 0
    for name in ('retained_charge_configuration_primary_validation.json', 'retained_charge_configuration_validation_audit.json',
                 'retained_charge_configuration_contact_validation.json', 'retained_charge_configuration_contact_sensitivity.json',
                 'retained_charge_configuration_contact_boundary.json', 'repeated_two_state_mean_bound.json'):
        path = ROOT/'analysis'/name; audit = json.loads(path.read_text()); files.add(path)
        for source, sha in audit['input_sha256'].items():
            assert digest(ROOT/source) == sha, source; files.add(ROOT/source)
        if name in ('retained_charge_configuration_validation_audit.json', 'retained_charge_configuration_contact_validation.json'):
            numerical.update(audit['candidates'])
    assert len(numerical) == 8
    for split in ('all_descriptive', 'chronological'):
        for suffix in ('single', 'polarizable', 'single_reconciled', 'polarizable_reconciled', 'single_wide', 'single_contact'):
            stem = f'retained_charge_configuration_{split}_{suffix}'
            path = ROOT/'analysis'/f'{stem}.json'; record = json.loads(path.read_text()); assert record['status'] == 'complete'
            for source, sha in record['input_sha256'].items():
                assert digest(ROOT/source) == sha, source; files.add(ROOT/source)
            assert digest(path.with_suffix('.npz')) == record['prediction_sha256']
            files.update((path, path.with_suffix('.npz')))
            best = record['starts'][record['selected_start']]
            assert best['converged']; assert best['rmse_decade'] <= min(s['rmse_decade'] for s in record['starts'])+1e-12
            np.testing.assert_array_equal(best['parameters'], record['parameters'])
            assert abs(best['rmse_decade']-record['training_scores']['rmse_decade']) < 1e-10
            assert record['maximum_kcl_relative_error'] < 1e-8
            if suffix.endswith('reconciled'):
                original = json.loads((ROOT/'analysis'/f'retained_charge_configuration_{split}_{suffix.removesuffix("_reconciled")}.json').read_text())
                assert record['starts'][:5] == original['starts'] and len(record['starts']) == 6
                starts += 1
            else:
                starts += len(record['starts'])
                for start in record['starts']:
                    if not start['converged']: unselected.append({'fit': stem, **start})
            if stem in numerical:
                a = numerical[stem]; assert a['all_refinements_pass'] and a['source_fit_converged']
                assert a['maximum_root_voltage_error_V'] < 1e-8
                assert a['independent_frozen_root_checks'] == 120
                for ref in a['refinements']:
                    assert ref['maximum_log_change'] < .005 and ref['weighted_log_change'] < .001
                    assert ref['maximum_kcl_relative_error'] < 1e-8
                fits.append({'fit': stem, 'training_rmse': record['training_scores']['rmse_decade'],
                             'target_rmse': record['target_scores']['rmse_decade'],
                             'parameters': record['parameters'], 'shared_parameters': record['shared_parameters'],
                             'selected_start': record['selected_start'],
                             'cycle34_ratios': a['cycle4_over_cycle3_native_0.10_to_0.30V']})
    assert starts == 32
    initial_sensitivity = ROOT.parent/'verification/2026-09-20/retained_charge_contact_initial_sensitivity.json'
    initial_test = ROOT.parent/'verification/2026-09-20/retained_charge_contact_initial_test_source.txt'
    archived = json.loads(initial_sensitivity.read_text())
    for source, sha in archived['input_sha256'].items():
        historical = initial_test if source == 'tests/test_retained_charge_contact.py' else ROOT/source
        assert digest(historical) == sha, source
    # Only the test-source identity changed; the physical sensitivity is identical.
    final_sensitivity = json.loads((ROOT/'analysis/retained_charge_configuration_contact_sensitivity.json').read_text())
    assert archived['candidates'] == final_sensitivity['candidates']
    bound = json.loads((ROOT/'analysis/repeated_two_state_mean_bound.json').read_text())
    assert bound['single_random_path_excluded'] is False
    reset = bound['data']['reset_outward']
    assert reset['favorable_reference_violation_count'] == len(reset['voltage_magnitude_V']) == 11
    assert reset['geometric_mean_ratio_after_favorable_reference_shift'] > 200
    testlog = ROOT.parent/'verification/2026-09-20/retained_charge_configuration_final_extended_tests.log'
    match = re.search(r'Ran (\d+) tests in ([\d.]+)s\s+OK', testlog.read_text()); assert match
    assert int(match[1]) == 39
    audit = {'scope': 'Retained charge/configuration and conditional repeated two-state mean-current bound',
             'goal_complete': False, 'all_selected_fits_converged': True, 'optimization_start_count': starts,
             'unconverged_nonselected_starts': unselected, 'fits': fits,
             'related_test_count': int(match[1]), 'test_log_sha256': digest(testlog),
             'prior_manifests_verified_without_rewriting': prior,
             'independent_frozen_load_checks': sum(r['independent_frozen_root_checks'] for r in numerical.values()),
             'maximum_refinement_log_change': max(s['maximum_log_change'] for r in numerical.values() for s in r['refinements']),
             'verifier_sha256': digest(Path(__file__))}
    output = ROOT/'analysis/retained_charge_configuration_artifact_audit.json'
    output.write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n'); files.add(output)
    names = ['analysis/RETAINED_CHARGE_CONFIGURATION_DESIGN_2026-09-21.md',
             'analysis/RETAINED_CHARGE_IMPLEMENTATION_DESIGN_2026-09-21.md',
             'analysis/RETAINED_CHARGE_SENSITIVITY_DESIGN_2026-09-21.md',
             'analysis/RETAINED_CHARGE_PHYSICS_2026-09-21.md', 'analysis/RETAINED_CHARGE_RESULTS_2026-09-21.md',
             'analysis/REPEATED_TWO_STATE_MEAN_BOUND_2026-09-21.md',
             'analysis/compare_retained_charge_configuration.py', 'analysis/refine_retained_charge_configuration.py',
             'analysis/check_retained_charge_validation.py', 'analysis/check_retained_charge_contact.py',
             'analysis/check_retained_contact_boundary.py', 'analysis/check_repeated_two_state_mean_bound.py',
             'analysis/retained_charge_configuration_prior_integrity.json',
             'analysis/retained_charge_configuration_prefit_refinement.json',
             'analysis/plot_retained_charge_evidence.py', 'analysis/write_retained_charge_report.py',
             'analysis/verify_retained_charge_artifacts.py',
             'analysis/figures/40_retained_charge_evidence.png', 'analysis/figures/40_retained_charge_evidence.pdf',
             'vulcan2d/retained_charge_configuration.py', 'vulcan2d/_retained_charge_configuration_core.cpp',
             'tests/test_retained_charge_configuration.py', 'tests/test_retained_charge_contact.py', 'tests/test_repeated_two_state_mean.py']
    files.update(ROOT/name for name in names)
    logs = sorted(set((ROOT.parent/'verification/2026-09-20').glob('retained_charge_configuration*.log')) |
                  set((ROOT.parent/'verification/2026-09-20').glob('repeated_two_state_mean_bound*.log')))
    manifest = {'scope': 'Conditional mechanism evidence; complete device explanation and novelty remain unresolved',
                'production_model_changed': False,
                'artifacts': {str(p.relative_to(ROOT)): info(p) for p in sorted(files)},
                'external_workspace_logs': {str(p.relative_to(ROOT.parent)): info(p) for p in logs},
                'external_workspace_sources': {str(p.relative_to(ROOT.parent)): info(p) for p in (initial_sensitivity, initial_test)}}
    destination = ROOT/'analysis/retained_charge_configuration_artifact_manifest.json'
    destination.write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    print('Verified', len(files), 'artifacts,', len(logs), 'logs,', len(prior), 'prior manifests;', starts, 'starts;', match[1], 'tests')


if __name__ == '__main__': run()
