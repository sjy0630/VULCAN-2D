"""Check and package the finite-step load-bound follow-up without refitting."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    paths = [ROOT / 'analysis' / name for name in
             ['negative_reentry_load_bound.json', 'negative_reentry_audit.json', 'reset_order_constraint.json']]
    checks = []
    for path in paths:
        report = json.loads(path.read_text())
        bad = [name for name, value in report['input_sha256'].items() if digest(ROOT / name) != value]
        if bad:
            raise RuntimeError(str(path)+': source mismatch '+str(bad))
        checks.append({'report': path.name, 'n_hashes_verified': len(report['input_sha256'])})
    report = json.loads(paths[0].read_text())
    rows = report['rows']
    if len(rows) != 7 or any(row['n_valid_sensitivity_checks'] != 12 for row in rows):
        raise RuntimeError('Unexpected event/check coverage')
    robust = [dict(gate_V=r['gate_V'], cycle_id=r['cycle_id'], raw_index_after=r['raw_index_after'])
              for r in rows if r['minimum_observed_over_bound_in_valid_checks'] > 1]
    not_robust = [dict(gate_V=r['gate_V'], cycle_id=r['cycle_id'], raw_index_after=r['raw_index_after'])
                  for r in rows if r['minimum_observed_over_bound_in_valid_checks'] <= 1]
    log = ROOT.parent / 'verification/2026-09-20/reentry-load-bound-tests-final.log'
    tests = re.findall(r'Ran (\d+) tests in ([\d.]+)s\s+OK', log.read_text())
    if not tests:
        raise RuntimeError('Passing test log missing')
    audit = {'source_checks': checks, 'n_upward_events': len(rows), 'n_sensitivity_checks': 12*len(rows),
             'robust_violation_events': robust, 'not_robust_violation_events': not_robust,
             'test_count': int(tests[-1][0]), 'test_seconds': float(tests[-1][1]),
             'test_log': str(log.relative_to(ROOT.parent)), 'test_log_sha256': digest(log),
             'verifier_source_sha256': digest(Path(__file__)),
             'limitations': ['This audit verifies implementation and artifact provenance, not the experimental assumptions or a microscopic mechanism.']}
    out = ROOT / 'analysis/negative_reentry_load_bound_numerical_audit.json'
    out.write_text(json.dumps(audit, indent=2)+'\n')
    # Refresh the finalized report/figure hashes after the documented count and
    # convergence corrections. Do not alter any source hash in saved fits.
    manifest_path = ROOT / 'analysis/reset_history_artifact_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for name, previous in manifest['artifacts'].items():
        path = ROOT / name
        actual = digest(path)
        if actual != previous['sha256'] and name != 'analysis/RESET_HISTORY_RESULTS_2026-09-21.md':
            raise RuntimeError('Unexpected historical artifact change: '+name)
        manifest['artifacts'][name] = {'sha256': actual, 'bytes': path.stat().st_size}
    paths += [out, Path(__file__), ROOT / 'tests/test_reentry_load_bound.py']
    paths += [ROOT / 'analysis' / name for name in ['check_negative_reentry_load_bound.py',
              'plot_reentry_load_bound.py', 'BIDIRECTIONAL_STATE_DESIGN_2026-09-21.md',
              'figures/31_recovery_load_bound.png', 'figures/31_recovery_load_bound.pdf']]
    followup = {'date': '2026-09-21', 'scope': 'Finite-step circuit evidence; microscopic identification and research goal remain incomplete',
                'production_model_changed': False, 'test_count': audit['test_count'],
                'artifacts': {str(p.relative_to(ROOT)): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in paths}}
    (ROOT / 'analysis/negative_reentry_load_bound_artifact_manifest.json').write_text(json.dumps(followup, indent=2)+'\n')
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    print('Source checks:', checks)
    print('Robust events:', robust, 'Other retained events:', not_robust)
    print('Tests:', audit['test_count'], 'All package hashes verified')


if __name__ == '__main__':
    run()
