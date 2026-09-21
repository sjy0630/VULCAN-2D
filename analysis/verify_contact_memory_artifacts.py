"""Independent scores/provenance audit and immutable stage manifest."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np
ROOT = Path(__file__).resolve().parents[1]


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def info(path): return {'sha256': digest(path), 'bytes': path.stat().st_size}


def run():
    previous = []
    for stem in ('stochastic_readout', 'retained_charge_configuration', 'activated_configuration',
                 'charge_spectrum', 'coupled_charge', 'side_charge', 'barrier_height_polarity',
                 'barrier_state', 'finite_path', 'bidirectional'):
        path = ROOT/'analysis'/f'{stem}_artifact_manifest.json'; record = json.loads(path.read_text()); count = 0
        for field, base in [('artifacts', ROOT), ('external_workspace_logs', ROOT.parent),
                            ('external_workspace_sources', ROOT.parent), ('verification_logs', ROOT.parent),
                            ('external_workspace_literature_sources', ROOT.parent)]:
            for name, expected in record.get(field, {}).items():
                assert digest(base/name) == expected['sha256'], name; count += 1
        previous.append({'manifest': str(path.relative_to(ROOT)), 'sha256': digest(path), 'verified_files': count})
    files = set()
    def read_record(stem):
        path = ROOT/'analysis'/f'{stem}.json'; r = json.loads(path.read_text()); files.add(path)
        for name, expected in r['input_sha256'].items():
            assert digest(ROOT/name) == expected, name; files.add(ROOT/name)
        array_path = path.with_suffix('.npz'); assert digest(array_path) == r['arrays_sha256']; files.add(array_path)
        return r, np.load(array_path)
    result, saved = read_record('contact_memory_counterfactual')
    refinement, fine = read_record('contact_memory_refinement')
    assert len(result['candidates']) == len(refinement['candidates']) == 8
    assert refinement['all_refinements_pass']
    mask = saved['weights'] > 0; weight = saved['weights']; cycle = saved['cycle']; voltage = saved['voltage']
    assert len(cycle) == 35828 and mask.sum() == 34662 and np.count_nonzero(mask & saved['low_signal']) == 787
    assert result['split_index'] == 2028 and voltage[2027] == voltage[2028] == 0.
    def score(y, selected):
        m = mask & selected; w = weight[m]; err = np.log10(y[m]/saved['observed'][m])
        return float(np.sqrt(np.dot(w, err**2)/w.sum()))
    def ratio(y, window):
        logs = [np.mean(np.log10(y[(cycle == c) & (saved['window'] == window) & (abs(voltage) >= .1-1e-10)])) for c in (3, 4)]
        return float(10**(logs[1]-logs[0]))
    for key, window in [('observed_SET_return_4_over_3', 2), ('observed_RESET_outward_4_over_3', 4)]:
        assert abs(result[key]-ratio(saved['observed'], window)) < 1e-10
    summaries = []; maximum = 0.; weighted_maximum = 0.; roots = 0
    for name, r in result['candidates'].items():
        y = saved[f'{name}_current_A']; stats = r['full_scores']
        for field, selected in [('rmse_decade', np.ones(len(cycle), bool)), ('first3_rmse', cycle <= 3), ('later50_rmse', cycle > 3)]:
            assert abs(stats[field]-score(y, selected)) < 1e-12
        for index, expected in enumerate(stats['windows'].values()):
            assert abs(expected-score(y, saved['window'] == index)) < 1e-12
        assert abs(r['last18_scores']['rmse_decade']-score(y, cycle > 35)) < 1e-12
        assert abs(r['SET_return_4_over_3']-ratio(y, 2)) < 1e-10
        assert abs(r['RESET_outward_4_over_3']-ratio(y, 4)) < 1e-10
        assert r['boundary_state_change'] == 0 and r['maximum_kcl_relative_error'] < 1e-8
        assert r['minimum_probability'] >= 0 and r['maximum_probability_sum_error'] < 2e-11
        if r['contact_label'].startswith('fixed'): assert r['split_equals_unsplit_exactly']
        if r['contact_label'] == 'fixed_positive': assert r['matches_archived_prediction_exactly']
        state_path = ROOT/r['states_file']; assert digest(state_path) == r['states_sha256']; files.add(state_path)
        a = refinement['candidates'][name]; assert a['all_refinements_pass'] and len(a['refinements']) == 2
        assert a['independent_frozen_root_checks'] == 96 and a['maximum_root_voltage_error_V'] < 1e-8
        roots += a['independent_frozen_root_checks']
        for f in a['refinements']:
            yy = fine[f'{name}_{f["label"]}_current_A']; err = np.log10(yy[mask]/y[mask])
            worst = float(np.max(abs(err))); weighted = float(np.sqrt(np.dot(weight[mask], err**2)/weight[mask].sum()))
            assert abs(worst-f['maximum_log_change']) < 1e-12 and abs(weighted-f['weighted_log_change']) < 1e-12
            assert f['passes_declared_tolerances'] and worst < .005 and weighted < .001
            assert f['maximum_kcl_relative_error'] < 1e-8 and f['maximum_probability_sum_error'] < 2e-11
            maximum = max(maximum, worst); weighted_maximum = max(weighted_maximum, weighted)
        summaries.append({'name': name, 'full_rmse_decade': stats['rmse_decade'],
                          'SET_4_over_3': ratio(y, 2), 'RESET_4_over_3': ratio(y, 4),
                          'selectivity_contrast': ratio(y, 4)/ratio(y, 2)})
    testlog = ROOT.parent/'verification/2026-09-20/contact_memory_initial_tests.log'
    match = re.search(r'Ran (\d+) tests in ([\d.]+)s\s+OK', testlog.read_text()); assert match and int(match[1]) == 15
    for name in ('analysis/CONTACT_MEMORY_RESULTS_2026-09-21.md', 'analysis/write_contact_memory_report.py',
                 'analysis/plot_contact_memory_evidence.py', 'analysis/verify_contact_memory_artifacts.py',
                 'analysis/figures/42_contact_memory_counterfactual.png', 'analysis/figures/42_contact_memory_counterfactual.pdf'):
        files.add(ROOT/name)
    audit = {'scope': 'No-refit contact swap fails selective-change constraint at inherited parameters',
             'goal_complete': False, 'parameter_refit': False, 'event_time_predicted': False,
             'native_point_count': len(cycle), 'scored_point_count': int(mask.sum()), 'related_test_count': int(match[1]),
             'test_log_sha256': digest(testlog), 'independent_root_checks': roots, 'refinement_count': 16,
             'maximum_refinement_change_decade': maximum, 'maximum_weighted_refinement_change_decade': weighted_maximum,
             'candidates': summaries, 'prior_manifests_verified_without_rewriting': previous,
             'verifier_sha256': digest(Path(__file__))}
    path = ROOT/'analysis/contact_memory_artifact_audit.json'; path.write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n'); files.add(path)
    logs = sorted((ROOT.parent/'verification/2026-09-20').glob('contact_memory*.log'))
    # The running verification log is deliberately not included in its own hash.
    logs = [p for p in logs if 'verification' not in p.name]
    manifest = {'scope': audit['scope'], 'production_model_changed': False, 'original_goal_achieved': False,
                'artifacts': {str(p.relative_to(ROOT)): info(p) for p in sorted(files)},
                'external_workspace_logs': {str(p.relative_to(ROOT.parent)): info(p) for p in logs}}
    (ROOT/'analysis/contact_memory_artifact_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    saved.close(); fine.close()
    print('Verified', len(files), 'artifacts,', len(logs), 'logs;', len(previous), 'unchanged prior manifests;',
          '15 tests,', roots, 'roots, 16 refinements. Max change:', maximum, 'Weighted:', weighted_maximum, flush=True)


if __name__ == '__main__': run()
