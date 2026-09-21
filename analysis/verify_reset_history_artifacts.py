"""Audit saved RESET evidence and independently extend capped field relaxations.

The original fitting/trajectory module is kept byte-for-byte unchanged so its
recorded input hashes remain valid. For flagged terminal groups only, a
verification copy of terminal_trajectory raises its iteration cap from 80 to
240. The exact transformed source is archived; this is a numerical sensitivity
check, not a refit or a change to the physical equations.
"""
from pathlib import Path
import csv
import hashlib
import inspect
import json
import re
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analysis'))
from vulcan2d import reset_history as kernel
from compare_reset_histories import assemble, score


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(name):
    with (ROOT / 'analysis' / name).open(encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def verify_extended_relaxation(comparison, terminal, data, lookup, saved_scores):
    original = inspect.getsource(kernel.terminal_trajectory)
    if original.count('range(80):') != 1:
        raise RuntimeError('Field iteration source changed; review verifier')
    extended = original.replace('range(80):', 'range(240):')
    namespace = dict(vars(kernel))
    exec(compile(extended, '<archived-240-iteration-trajectory>', 'exec'), namespace)
    extended_path = ROOT / 'analysis/reset_history_extended_iteration_source.txt'
    extended_path.write_text(extended)
    checks = []
    for diagnostic in terminal['diagnostics']:
        if diagnostic['unconverged_athermal_steps'] == 0:
            continue
        name, split = diagnostic['candidate'], diagnostic['split']
        fold = next(f for f in comparison['candidates'][name]['folds'] if f['split'] == split)
        protocols = list(fold['target_predictions'])
        ids = np.array(fold['target_predictions'][protocols[0]]['cycle_ids'])
        index = ids - 1
        local = np.concatenate([fold['target_predictions'][p]['local_parameters'] for p in protocols])
        va = np.tile(data['vapp'][index], (len(protocols), 1))
        args = (va, name, fold['fit']['shared_parameters'], local, lookup.voltage, lookup.current)
        print('Checking extended field relaxation:', name, split, flush=True)
        base = kernel.terminal_trajectory(*args, substeps=diagnostic['substeps'])
        more = namespace['terminal_trajectory'](*args, substeps=diagnostic['substeps'])
        if base['unconverged_athermal_steps'] != diagnostic['unconverged_athermal_steps']:
            raise RuntimeError('Original iteration flags did not reproduce')
        selected = np.tile(data['weights'][index] > 0, (len(protocols), 1))
        nonzero = va > 0
        difference = np.zeros_like(va)
        difference[nonzero] = np.abs(np.log10(base['current_a'][nonzero] / more['current_a'][nonzero]))
        max_score_reproduction_error = 0.
        with np.errstate(divide='ignore'):
            predicted = np.log10(base['current_a'])
        for pi, protocol in enumerate(protocols):
            for row, k in enumerate(index):
                computed = score(k+1, data['y'][k], predicted[pi*len(index)+row], data['weights'][k],
                                 data['vapp'][k], data['vh'][k], name, protocol, split, 'predicted_terminal')
                saved = next(r for r in saved_scores if r['candidate'] == name and r['split'] == split
                             and r['protocol'] == protocol and int(r['cycle_id']) == k+1)
                for key in ('rmse_decade', 'low_outward_rmse_decade', 'high_outward_rmse_decade',
                            'return_rmse_decade', 'bias_decade'):
                    max_score_reproduction_error = max(max_score_reproduction_error, abs(computed[key]-float(saved[key])))
        check = {'candidate': name, 'split': split, 'substeps': diagnostic['substeps'],
                 'original_iteration_cap': 80, 'extended_iteration_cap': 240,
                 'original_unconverged_substeps': base['unconverged_athermal_steps'],
                 'extended_unconverged_substeps': more['unconverged_athermal_steps'],
                 'extended_largest_relaxation_steps': more['largest_athermal_relaxation_steps'],
                 'maximum_scored_current_difference_decade': float(difference[selected].max()),
                 'maximum_nonzero_current_difference_decade': float(difference[nonzero].max()),
                 'extended_max_kcl_relative_error': more['max_kcl_relative_error'],
                 'maximum_original_score_reproduction_error_decade': max_score_reproduction_error}
        if more['unconverged_athermal_steps'] or difference[nonzero].max() > 1e-6 or max_score_reproduction_error > 1e-12:
            raise RuntimeError('Extended field relaxation did not pass: '+str(check))
        checks.append(check)
        print(check, flush=True)
    return checks, extended_path


def run():
    names = ['reset_history_comparison.json', 'reset_history_terminal.json',
             'reset_history_transport_profiles.json', 'reset_order_constraint.json', 'negative_reentry_audit.json']
    reports = {name: json.loads((ROOT / 'analysis' / name).read_text()) for name in names}
    source_checks = []
    for name, report in reports.items():
        bad = [p for p, h in report['input_sha256'].items() if digest(ROOT / p) != h]
        if bad:
            raise RuntimeError(name+' source mismatch: '+str(bad))
        source_checks.append({'report': name, 'n_hashes_verified': len(report['input_sha256'])})
    comparison, terminal = reports[names[0]], reports[names[1]]
    internal_scores, terminal_scores = read_csv('reset_history_scores.csv'), read_csv('reset_history_terminal_scores.csv')
    if len(internal_scores) != 1224 or len(terminal_scores) != terminal['n_score_rows'] or len(terminal_scores) != 1065:
        raise RuntimeError('Score count mismatch')
    fits = []
    for name, candidate in comparison['candidates'].items():
        for split, fit in [('all53', candidate['all_data_fit'])]+[(f['split'], f['fit']) for f in candidate['folds']]:
            if not fit['converged'] or not all(a['success'] for a in fit['multistart']):
                raise RuntimeError('Shared fit failed')
            fits.append({'candidate': name, 'split': split, 'successful_starts': len(fit['multistart'])})
        for fold in candidate['folds']:
            training = set(fold['fit']['training_cycle_ids'])
            for target in fold['target_outward_fits']:
                if target['cycle_id'] in training or not target['converged']:
                    raise RuntimeError('Target leakage or failed local fit')
            for pred in fold['target_predictions'].values():
                if training.intersection(pred['cycle_ids']):
                    raise RuntimeError('Target leakage')
            for mapping in fold['handoff_maps'].values():
                if any(not set(row['training_cycle_ids']).issubset(training) for row in mapping):
                    raise RuntimeError('Handoff mapping training leakage')
    data, selection, lookup = assemble()
    np.testing.assert_allclose(data['weights'].sum(axis=1), 1.)
    if selection != comparison['selection']:
        raise RuntimeError('Data selection did not reproduce')
    for j in range(len(selection)):
        apex = np.argmax(data['vapp'][j]); ix = np.arange(data['vapp'].shape[1]); vh = data['vh'][j]
        for mask, mass in [((ix <= apex) & (vh <= .3), .25), ((ix <= apex) & (vh > .3), .25), (ix > apex, .5)]:
            np.testing.assert_allclose(data['weights'][j, mask].sum(), mass)
    diagnostics = terminal['diagnostics']
    if len(diagnostics) != 18 or not all(d['refinement_passed'] and d['max_kcl_relative_error'] < 1e-6 for d in diagnostics):
        raise RuntimeError('Terminal refinement or KCL check failed')
    extended_checks, extended_path = verify_extended_relaxation(comparison, terminal, data, lookup, terminal_scores)
    order, reentry = reports[names[3]], reports[names[4]]
    if len(order['cycles']) != 254 or len(reentry['traces']) != 254:
        raise RuntimeError('Raw trace coverage mismatch')
    raw_groups = {gate: {'ordering_reversal_ids': group['strong_conservative_reversal_ids'],
                         'persistent_drop_then_rise_ids': reentry['groups'][gate]['drop_then_rise_trace_ids']}
                  for gate, group in order['groups'].items()}
    log = ROOT.parent / 'verification/2026-09-20/reset-history-tests-final.log'
    tests = re.findall(r'Ran (\d+) tests in ([\d.]+)s\s+OK', log.read_text())
    if not tests:
        raise RuntimeError('Passing test log missing')
    audit = {'scope': 'Saved RESET evidence, target separation, source hashes and numerical sensitivity; not physical validation',
             'source_checks': source_checks, 'shared_fits': fits,
             'n_internal_score_rows': len(internal_scores), 'n_terminal_score_rows': len(terminal_scores),
             'n_terminal_diagnostic_groups': len(diagnostics),
             'maximum_substep_refinement_difference_decade': max(d['maximum_refinement_difference_decade'] for d in diagnostics),
             'maximum_saved_kcl_relative_error': max(d['max_kcl_relative_error'] for d in diagnostics),
             'saved_unconverged_athermal_substeps': sum(d['unconverged_athermal_steps'] for d in diagnostics),
             'extended_iteration_checks': extended_checks,
             'extended_iteration_source_sha256': digest(extended_path),
             'raw_trace_groups': raw_groups, 'test_count': int(tests[-1][0]), 'test_seconds': float(tests[-1][1]),
             'test_log': str(log.relative_to(ROOT.parent)), 'test_log_sha256': digest(log),
             'verifier_source_sha256': digest(Path(__file__)),
             'limitations': ['The saved 80-iteration terminal run has capped field substeps, retained in the source output.',
                             'The separate 240-iteration verification changes only the cap, without changing parameters or equations.',
                             'A matching current curve does not identify a microscopic state or mechanism.']}
    out = ROOT / 'analysis/reset_history_numerical_audit.json'
    out.write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
    artifacts = [ROOT / 'analysis' / name for name in names]+[out, Path(__file__), extended_path,
                 ROOT / 'vulcan2d/reset_history.py', ROOT / 'tests/test_reset_history.py']
    artifacts += [ROOT / 'analysis' / name for name in [
        'compare_reset_histories.py', 'check_reset_history_terminal.py', 'profile_reset_history_transport.py',
        'check_reset_order_constraint.py', 'check_negative_reentry.py', 'plot_reset_history_evidence.py',
        'RESET_HISTORY_DESIGN_2026-09-21.md', 'RESET_HISTORY_RESULTS_2026-09-21.md',
        'reset_history_scores.csv', 'reset_history_terminal_scores.csv',
        'reset_order_constraint.csv', 'reset_order_constraint_points.csv',
        'figures/30_reset_history_and_order.png', 'figures/30_reset_history_and_order.pdf']]
    artifacts += sorted({ROOT / p for r in reports.values() for p in r['input_sha256']})
    manifest = {'date': '2026-09-21', 'scope': 'Complete RESET history stage; active research goal unachieved',
                'production_model_changed': False,
                'artifacts': {str(p.relative_to(ROOT)): {'sha256': digest(p), 'bytes': p.stat().st_size}
                              for p in dict.fromkeys(artifacts)}}
    (ROOT / 'analysis/reset_history_artifact_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('Source checks:', source_checks)
    print('Raw trace groups:', raw_groups)
    print('Saved cap flags:', audit['saved_unconverged_athermal_substeps'], 'Tests:', audit['test_count'])


if __name__ == '__main__':
    run()
