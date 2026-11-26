"""Bounded joint trial of one inherited near-closed RESET initiation term.

This is a conditional common-path-rate trial, not microscopic validation.
All fits use original training records, unchanged four-branch weights, fixed
spectral shape and fixed transistor data. No test records select parameters.
"""
from pathlib import Path
import hashlib
import json
import sys
import time
import traceback

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from joint_problem import SpectralJointProblem
from prefix_periodic import evaluate_prefix_checked
from vulcan2d.shared_spectral import SpectralShape, SpectralLaw
from vulcan2d.shared_spectral_ivp import SpectralTable, TabulatedSpectralLaw

HERE = Path(__file__).resolve().parent
RESULT = HERE/'initiation_joint_fit.json'
MAX_CALLS_PER_START = 40
MAX_SECONDS = 1200.
# Reuse the two declared compromise preparations/SET points. RESET k=1 is a
# preregistered decade above the too-slow diagnostic, with A=5 from the finite
# low-resistance holding / peak-opening threshold budget. It is not data proof.
SEEDS = [
    [-1.5228787452803376, -3.8239087409443187, -4.698970004336019,
     .17609125905568124, 0., 11.890816212940916, 9.346648212110011, 5.],
    [-1.3010299956639813, -3.3010299956639813, -4.301029995663981,
     -.09691001300805639, 0., 12.401639095481679, 9.857438932307765, 5.]]
LOWER = np.array([-4., -7., -8., -4., -4., 0., 0., 0.])
UPPER = np.array([0., np.log10(.001*(1-1e-8)), -2., 1.5, 1.5, 30., 30., 8.])
OPTIONS = {'method': 'Radau', 'state_rtol': 1e-8, 'state_atol': 1e-10,
           'current_rtol': 1e-3, 'state_refine_atol': 2e-5,
           'max_step_v': .05, 'max_refinements': 4}


class BudgetReached(RuntimeError):
    pass


def write(output):
    temporary = RESULT.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    temporary.replace(RESULT)


def save_predictions(path, predictions):
    arrays = {}
    for (top, cid, phase), trace in predictions.items():
        for key, value in trace.items():
            if isinstance(value, np.ndarray):
                arrays[f'{top}_{cid}_{phase}_{key}'] = value
    np.savez_compressed(path, **arrays)


def main():
    shape = SpectralShape.from_json()
    table = SpectralTable.load(HERE/'transport_g010_v0025.npz', shape)

    def factory(q, ron):
        if abs(ron-shape.ron_ohm) > 1e-8:
            raise ValueError('Shared training Ron differs from frozen shape')
        exact = SpectralLaw(shape, *10**np.asarray(q[:5]), reset_initiation_log=float(q[7]))
        return TabulatedSpectralLaw(exact, table)

    problem = SpectralJointProblem(factory, include_reset_initiation=True,
                                    integrator_options=OPTIONS)
    started = time.monotonic()
    output = {'status': 'running_bounded_conditional_joint_training',
              'seeds_predeclared': SEEDS, 'max_actual_calls_per_start': MAX_CALLS_PER_START,
              'max_total_optimization_seconds': MAX_SECONDS,
              'bounds_are_algorithmic_not_physical_confidence': {'lower': LOWER.tolist(), 'upper': UPPER.tolist()},
              'integrator_options': OPTIONS, 'objective': problem.checkpoint_metadata(),
              'scope': ['Fixed lognormal shape and measured transistor mapping; shared material across circuits.',
                        'Common voltage-path rate is an explicit extra protocol assumption.',
                        'General SET state-order conflict is not fixed by a RESET-only term; this trial quantifies compromise.',
                        'Prefix periodicity is a checked local numerical shortcut; final full32 audit required.',
                        'No holdout arrays loaded; all invalid trials and budget terminations retained.'],
              'runs': [], 'best_training': None}
    write(output)
    for index, seed in enumerate(SEEDS):
        if time.monotonic()-started >= MAX_SECONDS:
            output['runs'].append({'start': index, 'status': 'not_started_global_wall_budget'})
            break
        row = {'start': index, 'seed': seed, 'status': 'running', 'actual_calls': 0,
               'cache_hits': 0, 'calls': [], 'best': None}
        output['runs'].append(row)
        cache = {}

        def objective(q):
            key = tuple(float(x) for x in q)
            if key in cache:
                row['cache_hits'] += 1
                return cache[key].copy()
            if row['actual_calls'] >= MAX_CALLS_PER_START:
                raise BudgetReached('predeclared actual objective-call limit')
            if time.monotonic()-started >= MAX_SECONDS:
                raise BudgetReached('predeclared total optimization wall-clock limit')
            row['actual_calls'] += 1
            begin = time.monotonic()
            call = {'call': row['actual_calls'], 'q': list(key), 'status': 'running'}
            row['calls'].append(call)
            write(output)
            try:
                result = evaluate_prefix_checked(problem, q, max_prefix=12)
                residual = result['residual']
                cost = float(np.dot(residual, residual)/2.)
                call.update(status='valid', cost=cost, details=result['details'])
                candidate = {'start': index, 'call': row['actual_calls'], 'q': list(key),
                             'cost': cost, 'details': result['details']}
                if row['best'] is None or cost < row['best']['cost']:
                    row['best'] = candidate
                if output['best_training'] is None or cost < output['best_training']['cost']:
                    output['best_training'] = candidate
                    save_predictions(HERE/'initiation_best_training.npz', result['predictions'])
            except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
                residual = np.full(problem.n_residuals, np.sqrt(2e4/problem.n_residuals))
                call.update(status='invalid_preserved', failure_type=type(exc).__name__,
                            failure_message=str(exc), penalty_cost=1e4)
            call['elapsed_s'] = time.monotonic()-begin
            output['elapsed_s'] = time.monotonic()-started
            cache[key] = residual.copy()
            write(output)
            print(json.dumps({k: v for k, v in call.items() if k not in ('details',)}), flush=True)
            return residual

        try:
            fit = least_squares(objective, seed, bounds=(LOWER, UPPER),
                                x_scale=np.array([.5, .5, .5, .7, .7, 3., 3., 2.]),
                                diff_step=2e-3, max_nfev=40, ftol=1e-6, xtol=1e-6, gtol=1e-5)
            row.update(status='optimizer_terminated', optimizer_success=bool(fit.success),
                       optimizer_message=str(fit.message), optimizer_nfev=fit.nfev,
                       optimality=float(fit.optimality), terminal_q=fit.x.tolist())
        except BudgetReached as exc:
            row.update(status='budget_terminated_no_convergence_claim', reason=str(exc))
        except Exception as exc:
            row.update(status='unexpected_failure_preserved', failure_type=type(exc).__name__,
                       failure_message=str(exc), traceback=traceback.format_exc())
        output['elapsed_s'] = time.monotonic()-started
        write(output)
    output['status'] = 'bounded_training_trial_finished_requires_full32_validation'
    output['elapsed_s'] = time.monotonic()-started
    sources = [Path(__file__), HERE/'joint_problem.py', HERE/'prefix_periodic.py',
               ROOT/'vulcan2d/shared_spectral.py', ROOT/'vulcan2d/shared_spectral_ivp.py',
               HERE/'transport_g010_v0025.npz']
    output['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    write(output)
    print(json.dumps({'status': output['status'], 'elapsed_s': output['elapsed_s'],
                      'best': output['best_training']}), flush=True)


if __name__ == '__main__':
    main()
