"""Exactly three frozen surviving bare-budget candidates, no optimization."""
from pathlib import Path
import hashlib
import json
import signal
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from evaluate_bare_entry import OPTIONS, make_problem, predicted_feature


def main():
    started = time.monotonic()
    source = HERE/'bare_entry_budget_audit.json'
    barrier_path = HERE/'bare_reset_barrier_audit.json'
    constraints_path = ROOT/'analysis/constrained_shared_2026-11-03/feature_constraints.json'
    budget = json.loads(source.read_text())
    barrier = json.loads(barrier_path.read_text())
    constraints = json.loads(constraints_path.read_text())
    survivors = [r['gamma_c_ev'] for r in barrier['rows'] if not r['rejected_by_barrier']]
    candidates = [r['candidate'] for r in budget['cases']
                  if 'candidate' in r and r['gamma_c_ev'] in survivors]
    assert len(candidates) == 3
    selected = [name for name in constraints['proposed_success_criteria_before_next_fit']['primary_feature_ids']
                if constraints['features'][name]['topology'] == '1r']
    assert len(selected) == 7
    destination = HERE/'bare_three_candidates_forward.json'
    report = dict(status='running_three_frozen_bare_candidates_no_refit', options=OPTIONS,
                  candidates_source=str(source.relative_to(ROOT)),
                  selected_gamma_c_ev=survivors, selected_topology='1r', beta=1., A_RESET=0.,
                  maximum_evaluate_calls=3, evaluate_calls=0, rows=[])
    arrays = {}

    def save():
        report['elapsed_s'] = time.monotonic()-started
        destination.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        np.savez_compressed(destination.with_suffix('.npz'), **arrays)

    def timeout(*_):
        raise TimeoutError('30-second wall-clock forward-check budget exhausted')

    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 30.)
    save()
    problem = make_problem('1r', integrator_options=OPTIONS, mode='full')
    for index, candidate in enumerate(candidates):
        q = np.asarray(candidate['bare_six_coordinates'], float)
        row = dict(candidate_index=index, gamma_c_ev=candidate['gamma_c_ev'], q=q.tolist(),
                   status='running', source_candidate=candidate)
        report['rows'].append(row)
        save()
        run_started = time.monotonic()
        try:
            report['evaluate_calls'] += 1
            result = problem.evaluate(q, mode='full')
            row['forward_elapsed_s'] = time.monotonic()-run_started
            metrics = {}
            for phase in ('set', 'reset'):
                v = problem.waveforms['1r', phase]
                observed = problem.raw_training['1r', phase]
                predicted = abs(np.array([result['predictions']['1r', cid, phase]['current_a']
                                          for cid in problem.train_ids['1r']]))
                eligible = np.isfinite(observed) & (observed >= (1e-11 if phase == 'set' else 1e-12))
                eligible &= (abs(v)[None, :] > 1e-12)
                outward = np.arange(len(v)) <= np.argmax(abs(v))
                masks = dict(all_eligible=eligible, outward=eligible & outward,
                             returning=eligible & ~outward)
                if phase == 'set':
                    nonlimited = observed < .995e-3
                    masks.update(nonlimited=eligible & nonlimited, limited=eligible & ~nonlimited,
                                 outward_nonlimited=eligible & outward & nonlimited,
                                 returning_nonlimited=eligible & ~outward & nonlimited)
                scores = {}
                for name, mask in masks.items():
                    if np.any(predicted[mask] <= 0):
                        raise ValueError('Nonpositive prediction at eligible observation')
                    error = np.log10(predicted[mask])-np.log10(observed[mask])
                    scores[name] = dict(n=int(mask.sum()), raw_rmse_decade=float(np.sqrt(np.mean(error**2))),
                                        mean_signed_log_error_decade=float(np.mean(error)))
                metrics[phase] = scores
            features = {}
            for identifier in selected:
                definition = constraints['features'][identifier]
                phase = definition['phase']
                values, failures = [], []
                for cid in problem.train_ids['1r']:
                    try:
                        value, _ = predicted_feature(identifier, definition, problem.waveforms['1r', phase],
                                                     result['predictions']['1r', cid, phase]['current_a'])
                        if not np.isfinite(value):
                            raise ValueError('Nonfinite feature')
                        values.append(float(value))
                    except ValueError as exc:
                        failures.append(dict(record=cid, reason=str(exc)))
                lower, upper = definition['proposed_empirical_acceptance_interval']
                median = float(np.median(values)) if values else None
                features[identifier] = dict(model_median=median, fixed_interval=[lower, upper],
                                            passed=bool(not failures and lower <= median <= upper), failures=failures)
            prefix = f'candidate_{index}_'
            arrays[prefix+'q'] = q
            arrays[prefix+'residual'] = result['residual']
            for phase in ('set', 'reset'):
                trace = result['predictions']['1r', problem.train_ids['1r'][0], phase]
                arrays[prefix+phase+'_command_v'] = problem.waveforms['1r', phase]
                for key, value in trace.items():
                    if isinstance(value, np.ndarray):
                        arrays[prefix+phase+'_'+key] = value
            compliance = next((name for name in selected
                               if name.startswith('1r.set.') and 'current_threshold_A' in constraints['features'][name]), None)
            assert compliance is not None
            vset = problem.waveforms['1r', 'set']
            iset = abs(result['predictions']['1r', problem.train_ids['1r'][0], 'set']['current_a'])
            outward_set = np.arange(len(vset)) <= np.argmax(abs(vset))
            hits = np.flatnonzero(outward_set & (iset >= constraints['features'][compliance]['current_threshold_A']))
            n_passed = sum(r['passed'] for r in features.values())
            row.update(status='completed_frozen_forward_not_refitted', details=result['details'],
                       phase_metrics=metrics, n_primary_features=7, n_primary_passed=n_passed,
                       all_seven_primary_gates_passed=n_passed == 7, primary_features=features,
                       SET_compliance=dict(feature_id=compliance, threshold_a=constraints['features'][compliance]['current_threshold_A'],
                           reached_anywhere_on_outward=bool(len(hits)),
                           first_reached_command_v=float(vset[hits[0]]) if len(hits) else None,
                           midpoint_feature_and_target=features[compliance], reached_target_window=features[compliance]['passed']),
                       RESET_peak=features['1r.reset.outward.peak_current'],
                       RESET_tail=features['1r.reset.return.I_0p2V'])
        except Exception as exc:
            row.update(status='forward_or_scoring_failure_retained', error_type=type(exc).__name__,
                       error=str(exc), elapsed_s=time.monotonic()-run_started,
                       all_seven_primary_gates_passed=False,
                       interpretation='Numerical/scoring failure is not a physical impossibility claim.')
            if isinstance(exc, TimeoutError):
                report['status'] = 'budget_exhausted_partial_results_preserved'
                save()
                break
        save()
    signal.setitimer(signal.ITIMER_REAL, 0.)
    if report['status'] != 'budget_exhausted_partial_results_preserved':
        report['status'] = 'completed_three_existing_candidates_no_optimization'
    report['checkpoint_metadata'] = problem.checkpoint_metadata()
    report['source_sha256'] = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (source,barrier_path,constraints_path,Path(__file__),HERE/'evaluate_bare_entry.py')}
    report['evidence_scope'] = ('Exactly three fixed prior candidate evaluations. Each integrates one complete prepared bare SET/RESET pair, '
        'scored against all14 bare training records with the original seven feature gates and masks. Same integrator options as '
        'the first bare forward check. No optimization, new parameter selection, holdout access, or common-model/transfer improvement claim.')
    save()
    print(json.dumps({k:report[k] for k in ('status','elapsed_s','evaluate_calls')},indent=2))
    print(json.dumps([{k:r.get(k) for k in ('gamma_c_ev','status','forward_elapsed_s','n_primary_passed','SET_compliance','RESET_peak','RESET_tail','phase_metrics','error')}
                      for r in report['rows']],indent=2),flush=True)


if __name__ == '__main__':
    main()
