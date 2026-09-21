"""Freeze the continuous-configuration evidence and verify prior identities."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np
ROOT = Path(__file__).resolve().parents[1]


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def info(path): return {'sha256': digest(path), 'bytes': path.stat().st_size}


def run():
    prior = []
    for stem in ('charge_spectrum', 'coupled_charge', 'side_charge', 'barrier_height_polarity',
                 'barrier_state', 'finite_path', 'bidirectional'):
        p = ROOT/f'analysis/{stem}_artifact_manifest.json'; r = json.loads(p.read_text()); count = 0
        for field, base in [('artifacts', ROOT), ('external_workspace_logs', ROOT.parent), ('external_workspace_sources', ROOT.parent)]:
            for name, expected in r.get(field, {}).items():
                assert digest(base/name) == expected['sha256'], f'Changed prior artifact: {name}'
                count += 1
        prior.append({'path': str(p.relative_to(ROOT)), 'sha256': digest(p), 'verified_file_count': count})
    audit_path = ROOT/'analysis/activated_configuration_validation_audit.json'
    audit = json.loads(audit_path.read_text()); assert audit['candidate_count'] == 4
    for name, expected in audit['input_sha256'].items(): assert digest(ROOT/name) == expected, name
    side = json.loads((ROOT/'analysis/activated_configuration_side_rate_sensitivity.json').read_text())
    assert len(side['candidates']) == 4
    for name, expected in side['input_sha256'].items(): assert digest(ROOT/name) == expected, name
    for records in side['candidates'].values():
        assert len(records) == 5
        zero = [r for r in records if r['ln_side_over_right_rate'] == 0.][0]
        assert zero['maximum_log_current_change'] < 1e-12
    records = []; failures = []; candidate_paths = []
    for split in ('all_descriptive', 'chronological'):
        for saddle in ('single', 'polarizable'):
            stem = f'activated_configuration_{split}_{saddle}'
            original_path = ROOT/f'analysis/{stem}.json'; original = json.loads(original_path.read_text())
            path = ROOT/f'analysis/{stem}_reconciled.json'; record = json.loads(path.read_text())
            assert original['status'] == record['status'] == 'complete'
            assert len(original['starts']) == 4 and len(record['starts']) == 5
            assert record['starts'][:4] == original['starts']
            for r in (original, record):
                for name, expected in r['input_sha256'].items(): assert digest(ROOT/name) == expected, name
            assert digest(path.with_suffix('.npz')) == record['prediction_sha256']
            best = record['starts'][record['selected_start']]
            assert best['rmse_decade'] <= min(x['rmse_decade'] for x in record['starts'])+1e-12
            assert best['converged'], f'Selected unconverged solution: {stem}'
            np.testing.assert_array_equal(best['parameters'], record['parameters'])
            assert abs(best['rmse_decade']-record['training_scores']['rmse_decade']) < 1e-10
            assert record['maximum_kcl_relative_error'] < 1e-8
            numerical = audit['candidates'][stem+'_reconciled']
            assert numerical['maximum_root_voltage_error_V'] < 1e-8
            for refinement in numerical['refinements']:
                assert refinement['maximum_log_change'] < .005
                assert refinement['weighted_log_change'] < .001
                assert refinement['maximum_kcl_relative_error'] < 1e-8
            for start in record['starts']:
                if not start['converged']: failures.append({'split': split, 'saddle': saddle, **start})
            records.append({'split': split, 'saddle': saddle, 'training_rmse': record['training_scores']['rmse_decade'],
                            'target_rmse': record['target_scores']['rmse_decade'], 'selected_start': record['selected_start'],
                            'starts': len(record['starts']), 'multiple_root_native_points': record['multiple_root_point_count']})
            candidate_paths += [path, path.with_suffix('.npz'), original_path, original_path.with_suffix('.npz')]
    testlog = ROOT.parent/'verification/2026-09-20/activated_configuration_final_tests.log'
    match = re.search(r'Ran (\d+) tests in ([\d.]+)s\s+OK', testlog.read_text()); assert match
    artifact_audit = {'scope': 'Continuous activated retained configurations; conditional candidate has not explained the full device',
                      'goal_complete': False, 'prior_manifests_verified_without_rewriting': prior,
                      'fits': records, 'total_starts_including_matched_split_reconciliation': 20,
                      'unconverged_nonselected_starts': failures, 'all_selected_fits_converged': True,
                      'related_test_count': int(match[1]), 'related_test_seconds': float(match[2]),
                      'test_log_sha256': digest(testlog), 'validation_audit_sha256': digest(audit_path),
                      'verifier_source_sha256': digest(Path(__file__))}
    audit_output = ROOT/'analysis/activated_configuration_artifact_audit.json'
    audit_output.write_text(json.dumps(artifact_audit, indent=2, allow_nan=False)+'\n')
    names = ['analysis/ACTIVATED_CONFIGURATION_DESIGN_2026-09-21.md',
             'analysis/ACTIVATED_CONFIGURATION_PHYSICS_2026-09-21.md',
             'analysis/ACTIVATED_CONFIGURATION_LITERATURE_NOTE_2026-09-21.md',
             'analysis/ACTIVATED_CONFIGURATION_RESULTS_2026-09-21.md',
             'analysis/RETAINED_CHARGE_CONFIGURATION_DESIGN_2026-09-21.md',
             'analysis/compare_activated_configuration.py', 'analysis/refine_activated_configuration_nested.py',
             'analysis/check_activated_configuration_validation.py', 'analysis/verify_activated_configuration_artifacts.py',
             'analysis/plot_activated_configuration_evidence.py', 'analysis/write_activated_configuration_report.py',
             'analysis/activated_configuration_reference.py', 'analysis/activated_configuration_reference.json',
             'analysis/activated_configuration_prefit_refinement.json', 'analysis/activated_configuration_literature_discovery.json',
             'analysis/check_activated_configuration_side_rate.py', 'analysis/activated_configuration_side_rate_sensitivity.json',
             'analysis/figures/39_activated_configuration_evidence.png', 'analysis/figures/39_activated_configuration_evidence.pdf',
             'vulcan2d/activated_configuration.py', 'vulcan2d/_activated_configuration_core.cpp',
             'tests/test_activated_configuration.py']
    files = set([ROOT/name for name in names]+candidate_paths+[audit_path, audit_output])
    for record in [json.loads(p.read_text()) for p in candidate_paths if p.suffix == '.json']:
        files.update(ROOT/name for name in record['input_sha256'])
    files.update(ROOT/name for name in audit['input_sha256'])
    logs = sorted((ROOT.parent/'verification/2026-09-20').glob('activated_configuration*.log'))
    sources = [ROOT.parent/f'verification/2026-09-20/physics-sources/ducry2022-configurations.{suffix}' for suffix in ('pdf', 'txt')]
    manifest = {'scope': 'Native continuous fixed-protocol model test; no successful complete microscopic explanation',
                'production_model_changed': False,
                'artifacts': {str(p.relative_to(ROOT)): info(p) for p in sorted(files)},
                'external_workspace_logs': {str(p.relative_to(ROOT.parent)): info(p) for p in logs},
                'external_workspace_sources': {str(p.relative_to(ROOT.parent)): info(p) for p in sources}}
    output = ROOT/'analysis/activated_configuration_artifact_manifest.json'
    output.write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    print(output, 'artifacts', len(files), 'logs', len(logs), 'sources', len(sources), 'failed starts', len(failures))


if __name__ == '__main__': run()
