"""Verify frozen provenance, full target coverage and finite-state numerics."""
from pathlib import Path
import hashlib
import json
import platform
import re
import sys
import numpy as np
import scipy
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'analysis'))
from check_finite_path_transfer import prepare, COUNTS, CANDIDATES
from finite_path_observables import log_readouts
from diagnose_cross_polarity_physics import GateLookup
from vulcan2d.transistor import load_reverse_lookup
from vulcan2d.finite_paths import state_currents, generator, stochastic_exponential


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    reports = [ROOT / 'analysis' / name for name in ('finite_path_transfer.json', 'finite_path_initial_match.json')]
    results = [json.loads(p.read_text()) for p in reports]
    checked = []
    for path, report in zip(reports, results):
        bad = [p for p, h in report['input_sha256'].items() if digest(ROOT / p) != h]
        if bad:
            raise RuntimeError(path.name+' has changed inputs '+str(bad))
        checked.append({'report': path.name, 'n_input_hashes_verified': len(report['input_sha256'])})
    result, matching = results
    source, va, ids, obs, zeros, ev, source_index, target = prepare()
    np.testing.assert_array_equal(result['target_cycle_ids'], ids[target])
    if int(target.sum()) != 100 or len(result['observed_traces']) != 101:
        raise RuntimeError('Missing observed targets')
    expected = {(candidate, count) for candidate in CANDIDATES for count in COUNTS}
    if len(result['results']) != 18 or {(r['candidate'],r['effective_count']) for r in result['results']} != expected:
        raise RuntimeError('Missing candidate/count setting')
    np.testing.assert_array_equal(result['frozen_transport_parameters'], source['transport_fit_outward_only']['parameters'])
    readouts = log_readouts(va, obs)
    for k, row in enumerate(result['observed_traces']):
        if row['cycle_id'] != ids[k] or row['role'] != ('target' if target[k] else 'source'):
            raise RuntimeError('Observation mapping changed')
        for key, value in readouts.items():
            if abs(row[key]-value[k]) > 1e-12:
                raise RuntimeError('Observed readout changed')
    threshold = {'maximum_transition_refinement_absolute_difference': 1e-4,
                 'maximum_marginal_refinement_absolute_difference': 1e-4,
                 'maximum_mean_current_refinement_decade': 1e-4,
                 'maximum_marginal_row_sum_error': 1e-10,
                 'maximum_kcl_relative_error': 1e-7,
                 'maximum_transport_relative_error': 1e-12,
                 'maximum_previous_two_state_absolute_difference': 1e-12,
                 'maximum_sampled_marginal_absolute_error': .025}
    maxima = {key: max(row['numerical_checks'].get(key, 0.) for row in result['results']) for key in threshold}
    for key, limit in threshold.items():
        if maxima[key] > limit:
            raise RuntimeError(key+' exceeds numerical audit tolerance')
    lookup = GateLookup(load_reverse_lookup(), 1.3)
    transport = result['frozen_transport_parameters']; independent = []
    step_indices = np.array([20, 60, 110, 149, 200, 280])
    substep_midpoints = va[step_indices]+(va[step_indices+1]-va[step_indices])/32
    dt = np.abs(va[step_indices+1]-va[step_indices])/16
    max_event_z, max_ecdf, max_median = 0., 0., 0.
    for row in result['results']:
        name, count = row['candidate'], row['effective_count']
        np.testing.assert_array_equal(row['frozen_kinetic_parameters'], source['candidates'][name]['fit']['parameters'])
        if row['numerical_checks']['minimum_transition_probability'] < 0:
            raise RuntimeError('Negative saved transition probability')
        vh, _ = state_currents(substep_midpoints, transport, count, lookup.voltage, lookup.current)
        q = generator(vh, count, row['frozen_kinetic_parameters'], name)
        actual, diagnostic = stochastic_exponential(q, dt)
        error = max(float(np.max(np.abs(actual[j]-expm(q[j]*dt[j])))) for j in range(len(q)))
        if error > 2e-8:
            raise RuntimeError('Independent matrix exponential disagreement')
        independent.append({'candidate':name, 'effective_count':count, 'maximum_absolute_difference':error,
                            'interval_raw_indices':step_indices.tolist(), 'diagnostic':diagnostic})
        replication = row['independent_model_sampling_replication']
        for kind in ('any_rise', 'drop_then_rise'):
            first = row['zero_reference_event_sensitivity']['source'][kind]
            second = replication['source_zero_reference_events'][kind]
            p, p2 = first['fraction'], second['fraction']
            se = np.sqrt(p*(1-p)/first['n']+p2*(1-p2)/second['n'])
            z = abs(p-p2)/se if se else 0.
            max_event_z = max(max_event_z, z)
            if abs(p-p2) > 6*se+2/first['n']:
                raise RuntimeError('Model Monte Carlo replication disagreement')
        max_ecdf = max(max_ecdf, *replication['two_sample_ecdf_max_difference'].values())
        for key in readouts:
            max_median = max(max_median, abs(row['readouts'][key]['quantiles']['0.5']-
                                           replication['readouts'][key]['quantiles']['0.5']))
    if max_ecdf > .025:
        raise RuntimeError('Model sampled readout distributions not replicated')
    apex = int(np.argmax(va)); window = (np.arange(len(va)) <= apex) & (va >= .1-1e-10) & (va <= .3+1e-10)
    initial = np.log10(obs[:, window]).mean(axis=1)
    for case in matching['cases']:
        factor = case['maximum_initial_current_factor_from_source']; mask = target.copy()
        if factor is not None:
            mask &= np.abs(initial-initial[source_index]) <= np.log10(factor)
        np.testing.assert_array_equal(case['target_cycle_ids'], ids[mask])
        if case['n_targets'] != int(mask.sum()):
            raise RuntimeError('Initial-match subset size mismatch')
    test_log = ROOT.parent / 'verification/2026-09-20/finite-path-tests-final.log'
    tests = re.findall(r'Ran (\d+) tests in ([\d.]+)s\s+OK', test_log.read_text())
    if not tests:
        raise RuntimeError('Full passing test log absent')
    audit = {'scope':'Numerical and provenance audit of a conditional frozen-parameter transfer test, not a mechanism validation',
             'source_checks': checked, 'n_target_traces': 100, 'n_candidate_count_settings': 18,
             'all_target_ids_retained': True, 'detector_exactly_matches_saved_101_trace_audit': True,
             'source_transport_and_kinetics_unchanged': True, 'numerical_maxima': maxima,
             'independent_matrix_exponential_checks': independent,
             'maximum_independent_exponential_absolute_difference': max(r['maximum_absolute_difference'] for r in independent),
             'maximum_model_replication_event_difference_in_standard_errors': max_event_z,
             'maximum_model_replication_readout_ecdf_difference': max_ecdf,
             'maximum_model_replication_readout_median_difference_decade': max_median,
             'test_count': int(tests[-1][0]), 'test_seconds': float(tests[-1][1]),
             'test_log': str(test_log.relative_to(ROOT.parent)), 'test_log_sha256': digest(test_log),
             'runtime': {'python':platform.python_version(), 'numpy':np.__version__, 'scipy':scipy.__version__, 'platform':platform.platform()},
             'verifier_source_sha256':digest(Path(__file__))}
    out = ROOT / 'analysis/finite_path_validation_audit.json'
    out.write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
    files = reports+[out, Path(__file__), ROOT / 'tests/test_finite_paths.py']
    files += [ROOT / path for report in results for path in report['input_sha256']]
    files += [ROOT / 'analysis' / path for path in ('FINITE_PATH_TRANSFER_RESULTS_2026-09-21.md',
               'finite_path_transfer_summary.csv', 'plot_finite_path_transfer.py',
               'figures/33_finite_path_transfer.png', 'figures/33_finite_path_transfer.pdf')]
    files += [ROOT / p for p in source['input_sha256']]
    manifest = {'date':'2026-09-21', 'scope':audit['scope'], 'production_model_changed':False,
                'artifacts':{str(p.relative_to(ROOT)):{'sha256':digest(p), 'bytes':p.stat().st_size} for p in dict.fromkeys(files)},
                'verification_logs':{str(test_log.relative_to(ROOT.parent)):{'sha256':digest(test_log), 'bytes':test_log.stat().st_size}}}
    (ROOT / 'analysis/finite_path_artifact_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k not in ('source_checks', 'independent_matrix_exponential_checks')}, indent=2))


if __name__ == '__main__':
    run()
