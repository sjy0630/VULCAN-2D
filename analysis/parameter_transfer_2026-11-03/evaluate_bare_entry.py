"""One frozen bare-group trajectory from a necessary-budget seed; no refit."""
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'analysis/spectral_dynamic_2026-11-03'))
from group_problem import make_problem
from evaluate_initiation import predicted_feature

Q = np.array([-.9, -3.00877392865045, -2.866666666666667,
              1.0994411894842853, 1.145063347980644, 13.325042306571167])
OPTIONS = dict(method='Radau', state_rtol=1e-8, state_atol=1e-10,
               current_rtol=1e-3, state_refine_atol=2e-5,
               max_step_v=.05, max_refinements=4)


def main():
    started = time.monotonic()
    problem = make_problem('1r', integrator_options=OPTIONS, mode='full')
    destination = HERE / 'bare_entry_forward.json'
    report = dict(status='running_frozen_bare_entry_no_optimization', q=Q.tolist(),
                  options=OPTIONS, selected_topology='1r', beta=1., A_RESET=0.)
    destination.write_text(json.dumps(report, indent=2)+'\n')
    result = problem.evaluate(Q, mode='full')
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
            masks.update(nonlimited=eligible & (observed < .995e-3),
                         limited=eligible & (observed >= .995e-3))
        rows = {}
        for name, mask in masks.items():
            if np.any(predicted[mask] <= 0):
                raise ValueError('Nonpositive prediction at eligible observation')
            error = np.log10(predicted[mask])-np.log10(observed[mask])
            rows[name] = dict(n=int(mask.sum()), raw_rmse_decade=float(np.sqrt(np.mean(error**2))),
                              mean_signed_log_error_decade=float(np.mean(error)))
        metrics[phase] = rows
    constraint_path = ROOT/'analysis/constrained_shared_2026-11-03/feature_constraints.json'
    constraints = json.loads(constraint_path.read_text())
    selected = [name for name in constraints['proposed_success_criteria_before_next_fit']['primary_feature_ids']
                if constraints['features'][name]['topology'] == '1r']
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
                                    passed=not failures and lower <= median <= upper,
                                    failures=failures)
    arrays = dict(q=Q, residual=result['residual'])
    # Only one genuinely integrated pair; records share it by declared preparation.
    for phase in ('set', 'reset'):
        trace = result['predictions']['1r', problem.train_ids['1r'][0], phase]
        arrays[phase+'_command_v'] = problem.waveforms['1r', phase]
        for key, value in trace.items():
            if isinstance(value, np.ndarray):
                arrays[phase+'_'+key] = value
    np.savez_compressed(destination.with_suffix('.npz'), **arrays)
    report.update(status='completed_frozen_bare_entry_no_refit_no_holdout', elapsed_s=time.monotonic()-started,
                  details=result['details'], phase_metrics=metrics,
                  n_primary_features=len(selected), n_primary_passed=sum(row['passed'] for row in features.values()),
                  primary_features=features, checkpoint_metadata=problem.checkpoint_metadata(),
                  evidence_scope='One full SET/RESET bare pair from the budget seed, scored against all14 bare training records; no separate per-record state or parameter fit. Not a joint model or a parameter-transfer result.',
                  source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (Path(__file__), constraint_path, HERE/'BARE_ENTRY_REVIEW.md')})
    destination.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key:report[key] for key in ('status','elapsed_s','phase_metrics','n_primary_passed','n_primary_features')},indent=2),flush=True)


if __name__ == '__main__':
    main()
