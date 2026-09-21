"""Recheck saved sources, all-fit integration accuracy and package provenance."""
from pathlib import Path
import csv
import hashlib
import json
import re
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vulcan2d.parallel_paths import predict


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    reports = ['parallel_path_comparison.json', 'parallel_rectification_bound.json', 'scan_memory_contrast.json']
    source_checks = []
    for name in reports:
        r = json.loads((ROOT/'analysis'/name).read_text())
        bad = [p for p, h in r['input_sha256'].items() if digest(ROOT/p) != h]
        if bad:raise RuntimeError(name+' source mismatch: '+str(bad))
        source_checks.append({'report': name, 'n_hashes_verified': len(r['input_sha256'])})
    r = json.loads((ROOT/'analysis/parallel_path_comparison.json').read_text()); v = np.array(r['voltage_V'])
    weights = np.array(r['weights_by_cycle'])
    np.testing.assert_allclose(weights.sum(axis=1), 1.)
    for mask, mass in [((v > 0) & (v <= .3), .25), (v > .3, .25), (v < 0, .5)]:
        np.testing.assert_allclose(weights[:, mask].sum(axis=1), mass)
    quadrature, failed_targets = [], []
    for name, c in r['candidates'].items():
        fits = [('all53', c['all_data_fit'])]+[(f['split'], f['fit']) for f in c['folds']]
        for split, f in fits:
            a = predict(v, name, f['parameters'], f['states'])
            b = predict(v, name, f['parameters'], f['states'], nodes=128)
            err = float(np.max(np.abs(a-b)))
            if err > 1e-6:raise RuntimeError('Quadrature tolerance exceeded')
            quadrature.append({'candidate': name, 'split': split, 'maximum_difference_decade': err,
                               'shared_fit_converged': f['converged'],
                               'n_successful_starts': sum(a['success'] for a in f['multistart'])})
        for fold in c['folds']:
            for t in fold['targets']:
                if t['cycle_id'] in fold['fit']['training_cycle_ids']:raise RuntimeError('Target leaked')
                if not t['converged']:failed_targets.append({'candidate': name, 'split': fold['split'], **t})
    with (ROOT/'analysis/parallel_path_scores.csv').open(encoding='utf-8-sig') as f:
        nrows = sum(1 for row in csv.DictReader(f))
    log = ROOT.parent/'verification/2026-09-20/parallel-path-tests-final.log'
    matches = re.findall(r'Ran (\d+) tests in ([\d.]+)s\s+OK', log.read_text())
    if not matches:raise RuntimeError('Complete passing test log not found')
    audit = {'source_checks': source_checks, 'n_score_rows': nrows,
             'missing_high_grid_points': int(np.sum(weights == 0)), 'n_pairs': len(weights),
             'all_fit_quadrature': quadrature, 'failed_target_state_fits': failed_targets,
             'test_count': int(matches[-1][0]), 'test_seconds': float(matches[-1][1]),
             'test_log': str(log.relative_to(ROOT.parent)), 'test_log_sha256': digest(log),
             'verifier_source_sha256': digest(Path(__file__))}
    out = ROOT/'analysis/parallel_path_numerical_audit.json'
    out.write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
    paths = [ROOT/'analysis'/name for name in reports]
    paths += [out, Path(__file__), ROOT/'vulcan2d/parallel_paths.py', ROOT/'tests/test_parallel_paths.py']
    paths += [ROOT/'analysis'/name for name in
              ['compare_parallel_paths.py', 'check_parallel_rectification_bound.py', 'check_scan_memory_contrast.py',
               'plot_parallel_path_evidence.py', 'PARALLEL_PATH_DESIGN_2026-09-20.md', 'PARALLEL_PATH_RESULTS_2026-09-21.md',
               'parallel_path_scores.csv', 'parallel_rectification_bound.csv', 'scan_memory_contrast.csv',
               'scan_memory_contrast_points.csv', 'figures/29_parallel_path_state_evidence.png', 'figures/29_parallel_path_state_evidence.pdf']]
    manifest = {'date': '2026-09-21', 'scope': 'Parallel-path and scan-history stage; active research goal unachieved',
                'production_model_changed': False, 'test_count': audit['test_count'],
                'artifacts': {str(p.relative_to(ROOT)): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in paths}}
    (ROOT/'analysis/parallel_path_artifact_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('Sources verified:', source_checks)
    print('Score rows:', nrows, 'paired cycles:', len(weights), 'masked high-voltage points:', audit['missing_high_grid_points'])
    print('Largest integration difference:', max(q['maximum_difference_decade'] for q in quadrature))
    print('Failed target-state fits:', len(failed_targets), 'tests:', audit['test_count'])


if __name__ == '__main__':run()
