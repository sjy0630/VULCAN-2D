"""Verify source/data provenance, saved numerical diagnostics and research scope."""
from pathlib import Path
import hashlib
import json
import platform
import re
import subprocess
import sys
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'analysis'))
from compare_bidirectional_states import CASES, prepare, metrics
from diagnose_cross_polarity_physics import DATASETS, GateLookup, read_pairs
from vulcan2d.transistor import load_reverse_lookup


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    reports, files, checks = [], [], []
    for gate, cid in CASES:
        for name in ('single_hop', 'polarizable_saddle'):
            path = ROOT / 'analysis' / f'bidirectional_{gate}_{cid}_{name}.json'
            files.append(path); reports.append(json.loads(path.read_text()))
    for name in ('stochastic_reentry_probe.json', 'stochastic_retention_diagnostic.json'):
        files.append(ROOT / 'analysis' / name); reports.append(json.loads(files[-1].read_text()))
    for path, report in zip(files, reports):
        bad = [p for p, h in report['input_sha256'].items() if digest(ROOT / p) != h]
        if bad: raise RuntimeError(path.name+' changed inputs '+str(bad))
        checks.append({'report': path.name, 'n_input_hashes_verified': len(report['input_sha256'])})
    raw, lookups, capacity = {}, {}, {}
    for gate in ('0.9', '1.1', '1.3'):
        raw[gate] = {p['cycle_id']: p for p in read_pairs(gate)}
        lookups[gate] = GateLookup(load_reverse_lookup(), float(gate)); failures = []
        for cid, pair in raw[gate].items():
            va = np.abs(pair['reset']['voltage']); obs = np.abs(pair['reset']['current'])
            cap = np.interp(va, lookups[gate].voltage, lookups[gate].current)
            invalid = (va > 1e-12) & (obs > cap); scored = invalid & (va >= .08-1e-10) & (obs > 0)
            if np.any(invalid): failures.append({'cycle_id': cid, 'raw_indices': np.flatnonzero(invalid).tolist(),
                                                'n_scored': int(sum(scored))})
        capacity[gate] = {'n_traces': len(raw[gate]), 'n_incompatible_nonzero_points': sum(len(r['raw_indices']) for r in failures),
                          'n_incompatible_scored_points': sum(r['n_scored'] for r in failures), 'traces': failures}
    numerical = []
    for report in reports[:22]:
        gate, cid = str(report['gate_V']), report['cycle_id']; fit = report['fit']
        data = prepare(raw[gate][cid], lookups[gate])
        np.testing.assert_array_equal(report['applied_abs_voltage_V'], data['vapp'])
        np.testing.assert_array_equal(report['observed_current_A'], data['observed'])
        np.testing.assert_array_equal(report['weights'], data['weights'])
        recomputed = metrics(data, np.array(fit['predicted_current_A']))
        for key, value in recomputed.items():
            if abs(value-fit['scores'][key]) > 1e-12: raise RuntimeError('Score mismatch')
        if not fit['numerical_check_passed'] or fit['max_kcl_relative_error'] > 1e-6:
            raise RuntimeError('Saved numerical validation failed')
        numerical.append({'gate_V': float(gate), 'cycle_id': cid, 'candidate': report['candidate'],
                          'rmse_decade': recomputed['rmse_decade'], 'n_starts': len(fit['multistart']),
                          'n_converged_starts': sum(r['success'] for r in fit['multistart']),
                          'parameter_boundary_indices': fit['parameter_boundary_indices'],
                          'maximum_refinement_difference_decade': fit['maximum_solver_refinement_difference_decade'],
                          'maximum_independent_solver_difference_decade': fit['maximum_independent_solver_difference_decade'],
                          'maximum_kcl_relative_error': fit['max_kcl_relative_error']})
    stochastic = reports[22]; retention = reports[23]
    for row in stochastic['candidates'].values():
        if row['quadrature_nll_difference'] > .001 or row['maximum_transition_probability_refinement_difference'] > .0001:
            raise RuntimeError('Stochastic integration refinement failed')
    retention_checks = []
    for row in retention['rows']:
        rate = row['conditional_maximum_sweep_rate_V_per_s']
        if rate is None: continue
        p = row['zero_stationary_high_probability']; total = row['zero_build_per_V']+row['zero_remove_per_V']
        value = p+(1-p)*np.exp(-total*rate*row['hypothetical_retention_seconds'])
        if abs(value-.95) > 1e-12: raise RuntimeError('Retention formula mismatch')
        retention_checks.append({'candidate': row['candidate'], 'retained_probability_at_computed_rate': float(value)})
    log = ROOT.parent / 'verification/2026-09-20/bidirectional-tests-final.log'
    match = re.findall(r'Ran (\d+) tests in ([\d.]+)s\s+OK', log.read_text())
    if not match: raise RuntimeError('Complete passing test log missing')
    audit = {'scope': 'Two deterministic candidates and a single selected stochastic trace; not a unified mechanism or publication-readiness audit',
             'source_checks': checks, 'deterministic_fits': numerical, 'capacity_mismatch_audit': capacity,
             'n_deterministic_fits': len(numerical), 'n_deterministic_starts': sum(r['n_starts'] for r in numerical),
             'n_converged_deterministic_starts': sum(r['n_converged_starts'] for r in numerical),
             'maximum_refinement_difference_decade': max(r['maximum_refinement_difference_decade'] for r in numerical),
             'maximum_independent_solver_difference_decade': max(r['maximum_independent_solver_difference_decade'] for r in numerical),
             'maximum_kcl_relative_error': max(r['maximum_kcl_relative_error'] for r in numerical),
             'stochastic_rate_fit_convergence': {name: row['fit']['converged'] for name, row in stochastic['candidates'].items()},
             'retention_formula_checks': retention_checks,
             'test_count': int(match[-1][0]), 'test_seconds': float(match[-1][1]),
             'test_log': str(log.relative_to(ROOT.parent)), 'test_log_sha256': digest(log),
             'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
                         'platform': platform.platform(), 'compiler': subprocess.check_output(['c++', '--version'], text=True).splitlines()[0]},
             'verifier_source_sha256': digest(Path(__file__))}
    out = ROOT / 'analysis/bidirectional_validation_audit.json'
    out.write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
    files += [out, Path(__file__), ROOT / 'tests/test_bidirectional_state.py', ROOT / 'tests/test_stochastic_bottleneck.py']
    files += sorted({ROOT / name for report in reports for name in report['input_sha256']})
    files += [ROOT / 'analysis' / name for name in ['BIDIRECTIONAL_STATE_RESULTS_2026-09-21.md',
              'reentry_literature_discovery.json', 'plot_bidirectional_evidence.py',
              'figures/32_bidirectional_stochastic_evidence.png', 'figures/32_bidirectional_stochastic_evidence.pdf']]
    files += [ROOT / DATASETS[g][1] for g in ('0.9', '1.1', '1.3')]
    literature = ROOT.parent / 'verification/2026-09-20/physics-sources'
    source_files = [literature / name for name in ['pan2017-crossref.json', 'ranjan2018-crossref.json', 'pazos2023-crossref.json',
                    'pazos2023-openalex.json', 'ranjan2018.html', 'ranjan2018.txt', 'pazos2023-repository.html',
                    'reentry-literature-fetch-log.json', 'reentry-literature-fulltext-log.json']]
    manifest = {'date': '2026-09-21', 'scope': audit['scope'], 'production_model_changed': False,
                'artifacts': {str(p.relative_to(ROOT)): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in dict.fromkeys(files)},
                'external_workspace_literature_sources': {str(p.relative_to(ROOT.parent)): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in source_files},
                'access_limitations': ['Pazos full XML returned HTTP 403; repository HTML is a bot-check page, not paper fulltext.',
                                       'Pan evidence in this stage is the Crossref publisher abstract; Ranjan fulltext was extracted and read.']}
    (ROOT / 'analysis/bidirectional_artifact_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('Fits:', audit['n_deterministic_fits'], 'starts:', audit['n_deterministic_starts'], 'converged:', audit['n_converged_deterministic_starts'])
    print('Max refinement / independent solver / KCL:', audit['maximum_refinement_difference_decade'], audit['maximum_independent_solver_difference_decade'], audit['maximum_kcl_relative_error'])
    print('Capacity mismatch counts:', {g:(r['n_incompatible_nonzero_points'],r['n_incompatible_scored_points']) for g,r in capacity.items()})
    print('Stochastic convergence:', audit['stochastic_rate_fit_convergence'], 'Tests:', audit['test_count'])


if __name__ == '__main__': run()
