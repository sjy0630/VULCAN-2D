"""Check source identity, complete grouped splits, forward scores and numerics."""
from pathlib import Path
import hashlib
import json
import platform
import re
import sys
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'analysis'))
from compare_barrier_states import assemble, simulate, metrics, HEIGHTS, FOLDS


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    paths = [ROOT / 'analysis' / f for f in ('barrier_state_comparison.json', 'barrier_state_sensitivity.json')]
    main, sensitivity = [json.loads(p.read_text()) for p in paths]
    provenance = []
    for path, report in zip(paths, (main, sensitivity)):
        bad = [name for name, h in report['input_sha256'].items() if digest(ROOT / name) != h]
        if bad:
            raise RuntimeError(path.name+' inputs changed: '+str(bad))
        provenance.append({'report':path.name, 'verified_input_count':len(report['input_sha256'])})
    if main['status'] != 'complete':
        raise RuntimeError('Incomplete main comparison')
    va, observed, zeros, branches, lookup = assemble()
    np.testing.assert_array_equal(main['voltage_window_V'], va)
    np.testing.assert_array_equal(main['observed_current_A'], observed)
    if main['branches'] != branches or len(branches) != 202:
        raise RuntimeError('Branch metadata/coverage differs')
    expected = {'power', 'pf_like'} | {f'barrier_{h:g}' for h in HEIGHTS}
    ids = np.array([b['cycle_id'] for b in branches]); tested, shared_fits, starts, local_fits = [], [], [], []
    max_rescore, max_kcl = 0., 0.
    for label, split in main['splits'].items():
        training = np.array(split['training_branch_indices']); targets = np.array(split['target_branch_indices'])
        if set(split['fits']) != expected:
            raise RuntimeError('Missing profile/control in '+label)
        if label == 'all_descriptive':
            np.testing.assert_array_equal(training, np.arange(202))
        else:
            lo, hi = map(int, label.split('_')[1:])
            if (lo, hi) not in FOLDS:
                raise RuntimeError('Unknown fold')
            np.testing.assert_array_equal(training, np.flatnonzero((ids < lo) | (ids > hi)))
            np.testing.assert_array_equal(targets, np.flatnonzero((ids >= lo) & (ids <= hi)))
            if set(ids[training]) & set(ids[targets]):
                raise RuntimeError('Same cycle entered training and test')
            tested.extend(targets.tolist())
        best = min((k for k in expected if k.startswith('barrier_')), key=lambda k:split['fits'][k]['rmse_decade'])
        if best != split['selected_barrier_profile_key']:
            raise RuntimeError('Selected barrier does not follow training-only rule')
        for key, fit in split['fits'].items():
            if len(fit['local_parameters']) != len(training) or not fit['converged']:
                raise RuntimeError('Training state mapping or convergence failed')
            out = simulate(va, fit['shared_parameter'], fit['local_parameters'], fit['candidate'], lookup, fit['height_ev'])
            score = metrics(observed[training], out['current_a'], zeros[training], va, lookup)
            difference = abs(score['rmse_decade']-fit['rmse_decade'])
            max_rescore = max(max_rescore, difference); max_kcl = max(max_kcl, out['max_kcl_relative_error'])
            if difference > 1e-10:
                raise RuntimeError('Recomputed training score differs')
            shared_fits.append(fit); starts.extend(fit['multistart'])
        for candidate, saved in split['predictions'].items():
            fit = split['fits'][saved['source_fit_key']]; local = saved['local_identification']
            if local['shared_parameter'] != fit['shared_parameter'] or len(local['local_parameters']) != len(targets):
                raise RuntimeError('Target states/shared parameters mismatched')
            if not local['converged']:
                raise RuntimeError('Target identification did not converge')
            if label != 'all_descriptive':
                local_fits.append(local)
            out = simulate(va, fit['shared_parameter'], local['local_parameters'], candidate, lookup, fit['height_ev'])
            np.testing.assert_allclose(saved['predicted_current_A'], out['current_a'], rtol=2e-12, atol=0)
            cap = np.interp(va, lookup.voltage, lookup.current)
            if np.any(out['current_a'] > cap*(1+1e-9)):
                raise RuntimeError('Prediction exceeds passive load capacity')
            for key, mask in (('all_voltage_score', np.ones(len(va), bool)), ('higher_voltage_score', va > .16+1e-10)):
                score = metrics(observed[targets][:, mask], out['current_a'][:, mask], zeros[targets], va[mask], lookup)
                for metric in ('n_points', 'rmse_decade', 'bias_decade', 'n_high_signal_capacity_compatible',
                               'high_signal_capacity_compatible_rmse_decade', 'maximum_absolute_error_decade'):
                    if abs(score[metric]-saved[key][metric]) > 1e-10:
                        raise RuntimeError('Target score changed')
            max_kcl = max(max_kcl, out['max_kcl_relative_error'])
    if sorted(tested) != list(range(202)) or len(shared_fits) != 72 or len(local_fits) != 15:
        raise RuntimeError('Incomplete grouped coverage or fit set')
    refinement = sensitivity['all_training_profile_refinements']
    if len(refinement) != 60 or len(sensitivity['target_prediction_refinements']) != 5:
        raise RuntimeError('Missing numerical refinements')
    max_quad = max(r['quadrature_maximum_difference_decade'] for r in refinement)
    max_grid = max(r['combined_grid_quadrature_maximum_difference_decade'] for r in refinement)
    max_target = max(r['maximum_difference_decade'] for r in sensitivity['target_prediction_refinements'])
    if max_quad > 1e-5 or max_grid > 5e-4 or max_target > 5e-4 or max_kcl > 1e-8:
        raise RuntimeError('Numerical audit tolerance failed')
    log = ROOT.parent / 'verification/2026-09-20/barrier-state-tests-final.log'
    tests = re.findall(r'Ran (\d+) tests in ([\d.]+)s\s+OK', log.read_text())
    if not tests:
        raise RuntimeError('Full passing test log missing')
    audit = {'scope':'Constitutive low-bias comparison and identifiability audit; no mechanism or complete-cycle validation claim',
             'provenance':provenance, 'n_curves':101, 'n_branches':202, 'n_observed_points':observed.size,
             'all_cycles_retained_and_tested_once':True, 'same_cycle_never_split_across_train_test':True,
             'barrier_profile_selected_by_training_scores_only':True,
             'n_shared_fits':len(shared_fits), 'n_shared_fit_starts':len(starts),
             'n_converged_shared_fit_starts':sum(a['success'] for a in starts), 'n_target_local_fits':len(local_fits),
             'maximum_rescore_difference_decade':max_rescore, 'maximum_kcl_relative_error':max_kcl,
             'maximum_quadrature_refinement_decade':max_quad, 'maximum_combined_refinement_decade':max_grid,
             'maximum_target_prediction_refinement_decade':max_target,
             'test_count':int(tests[-1][0]), 'test_seconds':float(tests[-1][1]), 'test_log_sha256':digest(log),
             'runtime':{'python':platform.python_version(), 'numpy':np.__version__, 'scipy':scipy.__version__, 'platform':platform.platform()},
             'verifier_source_sha256':digest(Path(__file__))}
    audit_path = ROOT / 'analysis/barrier_state_validation_audit.json'
    audit_path.write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
    files = paths+[audit_path, Path(__file__), ROOT / 'tests/test_barrier_state.py']
    files += [ROOT / p for r in (main, sensitivity) for p in r['input_sha256']]
    files += [ROOT / 'analysis' / p for p in ('BARRIER_STATE_RESULTS_2026-09-21.md', 'plot_barrier_state_evidence.py',
                                            'barrier_transport_literature_metadata.json',
                                            'figures/34_barrier_state_evidence.png', 'figures/34_barrier_state_evidence.pdf')]
    logs = [log, ROOT.parent / 'verification/2026-09-20/barrier-state-serialization-failure.log',
            ROOT.parent / 'verification/2026-09-20/barrier-state-pre-serialization-fix.py.txt']
    manifest = {'scope':audit['scope'], 'date':'2026-09-21', 'production_model_changed':False,
                'artifacts':{str(p.relative_to(ROOT)):{'sha256':digest(p), 'bytes':p.stat().st_size} for p in dict.fromkeys(files)},
                'external_workspace_logs':{str(p.relative_to(ROOT.parent)):{'sha256':digest(p), 'bytes':p.stat().st_size} for p in logs}}
    (ROOT / 'analysis/barrier_state_artifact_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    run()
