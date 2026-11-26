"""Prepared bounded joint-history calibration; never executes on import.

The sole execution switch is --run-authorized, after source review. The first
three phases calibrate; RESET5/SET6 are training continuation checks only.
No finite failure penalty, state reset, clipping, hidden retry, or extra seed.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
from scipy.optimize import minimize
from joint_forward import (Precision, PhaseFailure, load_joint_kernel,
                           with_parameters, simulate_phase)
from joint_validation import phase_integrity, compare_phases

HERE = Path(__file__).resolve().parent
RESULT = HERE / 'bounded_joint_calibration.json'
PROGRESS = HERE / 'bounded_joint_calibration.progress.json'
CALIBRATION = ('set4', 'reset4', 'set5')
CONTINUATION = ('reset5', 'set6')
ALL_PHASES = CALIBRATION + CONTINUATION
OLD_KAPPA = 77.70013824898767
OLD_J = .0041089272077381635
OLD_RATIO = OLD_KAPPA / OLD_J
OLD_B = .5811330029022057
OLD_U0 = 9.860489172652638
NECESSARY_KAPPA_MIN = 14648.824351938409
NECESSARY_RATIO_MIN = 426403.0838962568
BMAX = math.log(10.)
LOW = np.array([math.log10(OLD_KAPPA / 10.), math.log10(OLD_RATIO / 10.), 0., 2.])
HIGH = np.array([math.log10(100. * NECESSARY_KAPPA_MIN),
                 math.log10(100. * NECESSARY_RATIO_MIN), BMAX, 16.])
CONTROL = Precision('control', 1e-8, 1e-9, .01)
SEARCH = Precision('search', 2e-6, 2e-7, .04)
NOMINAL = Precision('nominal', 1e-7, 1e-8, .02)
REFINED = Precision('refined', 1e-8, 1e-9, .01)
MAX_OBJECTIVE_REQUESTS = 85


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    return None if isinstance(value, float) and not math.isfinite(value) else value


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def hash_paths(paths):
    paths = sorted({str(Path(p).resolve()) for p in paths})
    with ThreadPoolExecutor(max_workers=4) as pool:
        return dict(zip(paths, pool.map(hash_file, paths)))


class Artifacts:
    """Serialize once, hash those bytes, and never reread generated artifacts."""
    def __init__(self):
        self.io_wall_s = 0.
        self.catalog = {}

    def save(self, path, producer, *, replace=False, catalog=True):
        started = time.monotonic()
        try:
            payload = producer()
            record = {'path': str(path), 'sha256': hashlib.sha256(payload).hexdigest(),
                      'bytes': len(payload)}
            with Path(path).open('wb' if replace else 'xb') as handle:
                handle.write(payload)
            if catalog:
                self.catalog[str(path)] = record
            return record
        finally:
            self.io_wall_s += time.monotonic() - started

    def json(self, path, value, *, replace=False, catalog=True):
        # Capture before serialization: later mutable summaries cannot change
        # an already-saved request or silently alter its catalog hash.
        snapshot = deepcopy(value)
        return self.save(path, lambda: (json.dumps(clean(snapshot), indent=2,
                         ensure_ascii=False, allow_nan=False) + '\n').encode(),
                         replace=replace, catalog=catalog)

    def npz(self, path, arrays):
        def produce():
            stream = BytesIO()
            np.savez_compressed(stream, **arrays)
            return stream.getvalue()
        return self.save(path, produce)

    def progress(self, message):
        started = time.monotonic()
        try:
            print(message, flush=True)
        finally:
            self.io_wall_s += time.monotonic() - started


class BudgetExceeded(TimeoutError):
    pass


class StopBlock(RuntimeError):
    pass


class NumericalBudget:
    """No numerical timer survives an artifact write or hash operation."""
    limits = {'setup': 60., 'new_optimizer': 120., 'old_optimizer': 120., 'final': 100.}

    def __init__(self):
        self.used = {name: 0. for name in self.limits}
        self.calls = []
        self.stopped = {}
        self.active = None

    def fit_used(self):
        return sum(self.used[name] for name in ('setup', 'new_optimizer', 'old_optimizer'))

    def _ensure(self, block, elapsed=0.):
        if block in self.stopped:
            raise BudgetExceeded(self.stopped[block])
        reason = None
        if self.used[block] + elapsed >= self.limits[block]:
            reason = f'{block} numerical budget {self.limits[block]:g} s exhausted'
        elif block != 'final' and self.fit_used() + elapsed >= 300.:
            reason = 'Total fitting numerical budget 300 s exhausted'
        if reason:
            self.stopped[block] = reason
            raise BudgetExceeded(reason)

    def check(self):
        if self.active is None:
            raise RuntimeError('Budget check outside an active numerical call')
        elapsed = time.monotonic() - self.active['started']
        block = self.active['block']
        if elapsed >= self.active['limit_s']:
            reason = f"Per-phase {self.active['limit_s']:g} s limit exhausted"
            self.stopped[block] = reason
            raise BudgetExceeded(reason)
        self._ensure(block, elapsed)

    def run(self, block, label, function, *, limit_s):
        if self.active is not None:
            raise RuntimeError('Nested numerical timers are prohibited')
        self._ensure(block)
        self.active = {'block': block, 'label': label, 'started': time.monotonic(), 'limit_s': limit_s}
        status = 'completed'
        try:
            self.check()
            answer = function(self.check)
            self.check()
            return answer
        except Exception:
            status = 'failed_or_budget_stopped'
            raise
        finally:
            elapsed = time.monotonic() - self.active['started']
            self.used[block] += elapsed
            self.calls.append({'block': block, 'label': label, 'elapsed_s': elapsed,
                               'per_call_limit_s': limit_s, 'status': status})
            self.active = None

    def charge_optimizer_overhead(self, block, elapsed):
        # SciPy work between objective calls is numerical, not artifact I/O.
        if elapsed > 0.:
            self.used[block] += elapsed
            self.calls.append({'block': block, 'label': 'optimizer_overhead',
                               'elapsed_s': elapsed, 'status': 'completed'})
        self._ensure(block)

    def snapshot(self):
        return {'used_s': dict(self.used), 'fit_total_s': self.fit_used(),
                'limits_s': dict(self.limits), 'stopped_blocks': dict(self.stopped),
                'calls': deepcopy(self.calls)}


def normalized(kappa, ratio, b, u0):
    physical_coordinates = np.array([math.log10(kappa), math.log10(ratio), b, u0])
    return (physical_coordinates - LOW) / (HIGH - LOW)


def parameters(x):
    x = np.asarray(x, dtype=float)
    if x.shape != (4,) or not np.all(np.isfinite(x)) or np.any((x < 0.) | (x > 1.)):
        raise ValueError('Normalized proposal outside [0,1]^4; not clipped')
    z = LOW + x * (HIGH - LOW)
    kappa, ratio = 10.**z[0], 10.**z[1]
    return {'kappa_per_v2': float(kappa), 'ratio_kappa_over_J': float(ratio),
            'J_positive_a': float(kappa / ratio), 'b': float(z[2]), 'initial_u': float(z[3])}


def inward_simplex(x):
    x = np.asarray(x, dtype=float)
    simplex = np.repeat(x[None, :], 5, axis=0)
    for j in range(4):
        simplex[j + 1, j] += .04 if x[j] + .04 <= 1. else -.04
    if np.any((simplex < 0.) | (simplex > 1.)):
        raise ValueError('Declared inward simplex is outside normalized domain')
    return simplex


def configuration():
    return {
        'contract': 'EXPERIMENT_PLAN.md',
        'calibration_phases': list(CALIBRATION), 'continuation_phases': list(CONTINUATION),
        'objective': 'mean of SET4, RESET4, SET5 original-mask MSE; all1170 also reported',
        'variables': ['log10(kappa)', 'log10(kappa/J)', 'b', 'u0'],
        'normalized_domain': [[0., 1.]] * 4,
        'coordinate_lower': LOW.tolist(), 'coordinate_upper': HIGH.tolist(),
        'necessary_bound_seed_scales': {'kappa': NECESSARY_KAPPA_MIN, 'ratio': NECESSARY_RATIO_MIN},
        'old_control': {'kappa': OLD_KAPPA, 'J': OLD_J, 'ratio': OLD_RATIO, 'b': OLD_B, 'u0': OLD_U0},
        'initial_u_is_not_measured': True, 'one_initial_state_for_whole_history': True,
        'seed_coordinates': [
            [NECESSARY_KAPPA_MIN, NECESSARY_RATIO_MIN, BMAX, OLD_U0],
            [4 * NECESSARY_KAPPA_MIN, NECESSARY_RATIO_MIN, BMAX, OLD_U0],
            [NECESSARY_KAPPA_MIN, 4 * NECESSARY_RATIO_MIN, BMAX, OLD_U0],
            [NECESSARY_KAPPA_MIN, NECESSARY_RATIO_MIN, BMAX / 2, OLD_U0]],
        'max_objective_requests': 85, 'optimizer': 'Nelder-Mead', 'maxfev_per_optimizer': 40,
        'simplex_normalized_step': .04, 'adaptive': False, 'xatol': 1e-3, 'fatol': 1e-6,
        'control_precision': asdict(CONTROL), 'search_precision': asdict(SEARCH),
        'final_precisions': [asdict(NOMINAL), asdict(REFINED)],
        'control_methods': ['DOP853', 'Radau'], 'calibration_and_final_method': 'Radau',
        'control_score_absolute_tolerance_decade': 1e-4,
        'method_and_precision_tolerances': {'log_current_decade': 1e-4, 'state_u': 1e-3},
        'numerical_block_seconds': dict(NumericalBudget.limits), 'fit_total_seconds': 300.,
        'setup_search_per_phase_seconds': 12., 'final_per_phase_seconds': 20.,
        'explicit_phase_data': {'initial_command_voltage_v': 0., 'nearzero_segment_budget': 1e-10,
                                'circuit_mode': 'fast'},
        'final_collect_rhs': True, 'final_independent_native_bisections': 48,
        'failure_objective': 'nonfinite infinity internally; JSON null with explicit failed status; never finite penalty',
        'cache_key_includes_method_and_precision': True,
        'continuation_excluded_from_all_candidate_selection': True,
        'continuation_is_not_original_holdout_or_independent_prediction': True,
        'original_global_contract': {'pairs': 46, 'residuals': 27023, 'aggregate_feature_gates': 20},
    }


class Runner:
    def __init__(self, kernel, data, audit, writer, report):
        self.kernel, self.data, self.audit = kernel, data, audit
        self.writer, self.report = writer, report
        self.budget = NumericalBudget()
        self.objective_count = 0
        self.cache = {}
        self.successful = []
        self.new_seeds = []

    def snapshot(self):
        self.report.update(objective_requests=self.objective_count,
                           timing=self.budget.snapshot(), artifact_IO_wall_s=self.writer.io_wall_s,
                           artifact_catalog=deepcopy(self.writer.catalog))
        self.writer.json(PROGRESS, self.report, replace=True, catalog=False)

    def request(self, x, *, label, block, precision, method='Radau', objective=True,
                phases=CALIBRATION, independent=False):
        number = len(self.report['requests']) + 1
        row = {'request_id': number, 'label': label, 'block': block, 'method': method,
               'precision': asdict(precision), 'x_normalized': np.asarray(x).tolist(),
               'phases': list(phases), 'objective_request': objective, 'status': 'requested',
               'finite_objective_available': False, 'objective_mean_phase_MSE': None,
               'phase_results': [], 'handoffs': [], 'physical_rejection': False}
        if objective:
            self.objective_count += 1
            row['objective_request_index'] = self.objective_count
        self.report['requests'].append(row)
        record_path = HERE / 'requests' / f'{number:03d}_{label}.json'
        self.writer.json(record_path, row)
        chain = {'record': row, 'results': {}}
        key = None
        try:
            if objective and self.objective_count > MAX_OBJECTIVE_REQUESTS:
                raise StopBlock('Objective request cap exhausted; no model call')
            try:
                p = parameters(x)
            except ValueError as exc:
                row.update(status='out_of_domain_request', error=str(exc))
                return chain
            row['parameters'] = p
            self.budget._ensure(block)
            key = (tuple(float(t) for t in x), method, tuple(asdict(precision).values()), tuple(phases))
            # Final validation is never answered by a search cache.
            if objective and key in self.cache:
                original = self.cache[key]
                available = original['record']['finite_objective_available']
                row.update(status='completed_cache_request' if available else 'failed_cache_request',
                           cache_source_request=original['record']['request_id'],
                           finite_objective_available=available,
                           objective_mean_phase_MSE=original['record']['objective_mean_phase_MSE'],
                           scores=deepcopy(original['record'].get('scores')),
                           handoffs=deepcopy(original['record']['handoffs']),
                           phase_results=deepcopy(original['record']['phase_results']))
                chain['results'] = original['results']
                return chain
            candidate = with_parameters(self.kernel, kappa_per_v2=p['kappa_per_v2'], J_positive_a=p['J_positive_a'])
            incoming = p['initial_u']
            incoming_command = 0.
            previous_phase = None
            for phase in phases:
                saved_result = None
                phase_data = dict(self.data[phase])
                phase_data['initial_command_voltage_v'] = incoming_command
                phase_row = {'phase': phase, 'initial_u': incoming,
                             'initial_command_voltage_v': incoming_command, 'status': 'started'}
                row['phase_results'].append(phase_row)
                phase_label = f'{number:03d}_{label}_{phase}'
                try:
                    def calculate(check):
                        nonlocal saved_result
                        saved_result = simulate_phase(candidate, phase_data, incoming, p['b'],
                            precision, method, check, collect_rhs=independent)
                        validation = phase_integrity(candidate, saved_result, phase_data, incoming,
                            p['b'], independent=independent, check_budget=check)
                        if not validation['passed'] or (independent and not validation['complete_integrity_passed']):
                            raise ValueError('Phase integrity did not pass; complete result preserved')
                        s, a = saved_result['summary'], saved_result['arrays']
                        state_error = max(abs(s['initial_u'] - incoming),
                                          abs(float(a['state_u'][0]) - incoming))
                        command_error = max(abs(s['initial_command_voltage_v'] - incoming_command),
                                            abs(float(a['command_voltage_v'][0]) - incoming_command))
                        handoff = {'from_phase': previous_phase, 'to_phase': phase,
                                   'source': 'single_whole_history_initial_state' if previous_phase is None else 'actual_previous_phase_terminal',
                                   'expected_initial_u': incoming,
                                   'reported_initial_u': s['initial_u'],
                                   'native_initial_u': float(a['state_u'][0]),
                                   'expected_initial_command_voltage_v': incoming_command,
                                   'reported_initial_command_voltage_v': s['initial_command_voltage_v'],
                                   'native_initial_command_voltage_v': float(a['command_voltage_v'][0]),
                                   'state_error_u': state_error, 'command_error_v': command_error,
                                   'passed': state_error <= 1e-12 and command_error <= 1e-12}
                        row['handoffs'].append(handoff)
                        if not handoff['passed']:
                            raise ValueError('Actual state/command handoff audit failed')
                        return saved_result
                    result = self.budget.run(block, phase_label, calculate,
                                             limit_s=20. if block == 'final' else 12.)
                    incoming = float(result['summary']['terminal_u'])
                    incoming_command = float(result['summary']['final_command_voltage_v'])
                    previous_phase = phase
                    phase_row.update(status='completed', terminal_u=incoming,
                                     terminal_command_voltage_v=incoming_command,
                                     metrics=deepcopy(result['summary']['metrics']))
                    chain['results'][phase] = result
                except PhaseFailure as exc:
                    saved_result = {'summary': exc.summary, 'arrays': exc.arrays}
                    phase_row.update(status='failed_phase', error_type=type(exc).__name__, error=str(exc))
                    raise
                except Exception as exc:
                    phase_row.update(status='failed_phase_or_validation', error_type=type(exc).__name__, error=str(exc))
                    raise
                finally:
                    if saved_result is not None:
                        phase_row['artifacts'] = {
                            'arrays': self.writer.npz(HERE / 'phase_trials' / f'{phase_label}.npz', saved_result['arrays']),
                            'summary': self.writer.json(HERE / 'phase_trials' / f'{phase_label}.json', saved_result['summary'])}
            # Scoring is recomputed only for the declared three calibration phases.
            def score(check):
                check()
                metrics = {name: deepcopy(chain['results'][name]['summary']['metrics']) for name in CALIBRATION}
                ns = [metrics[name]['all_eligible']['n'] for name in CALIBRATION]
                rmse = [metrics[name]['all_eligible']['RMSE_decade'] for name in CALIBRATION]
                if ns != [500, 170, 500] or not all(math.isfinite(t) for t in rmse):
                    raise ValueError('No valid original 1170-point calibration score')
                mse = sum(t*t for t in rmse) / 3.
                point_mse = sum(n*t*t for n, t in zip(ns, rmse)) / 1170.
                return {'mean_phase_MSE_decade2': mse, 'root_mean_phase_MSE_decade': math.sqrt(mse),
                        'all1170_RMSE_decade': math.sqrt(point_mse), 'phase_metrics': metrics,
                        'calibration_only': True}
            scores = self.budget.run(block, f'{number:03d}_score', score,
                                     limit_s=20. if block == 'final' else 12.)
            row.update(status='completed_chain', scores=scores,
                       objective_mean_phase_MSE=scores['mean_phase_MSE_decade2'],
                       finite_objective_available=bool(objective), terminal_u=incoming)
            if objective:
                self.cache[key] = chain
                self.successful.append(chain)
            if independent:
                row['continuation_metrics'] = {name: deepcopy(chain['results'][name]['summary']['metrics'])
                                               for name in CONTINUATION}
                row['continuation_used_for_selection'] = False
                merged = {f'{name}__{key}': value for name, answer in chain['results'].items()
                          for key, value in answer['arrays'].items()}
                row['complete_five_phase_arrays'] = self.writer.npz(HERE / f'{label}_chain.npz', merged)
            r = [scores['phase_metrics'][name]['all_eligible']['RMSE_decade'] for name in CALIBRATION]
            self.writer.progress(f"Completed {label}: SET4/RESET4/SET5 RMSE={r[0]:.6f}/{r[1]:.6f}/{r[2]:.6f}; "
                                 f"mean-phase MSE={scores['mean_phase_MSE_decade2']:.6g}; objective requests={self.objective_count}")
            return chain
        except (BudgetExceeded, StopBlock) as exc:
            row.update(status='block_stopped', error_type=type(exc).__name__, error=str(exc))
            raise StopBlock(str(exc)) from exc
        except PhaseFailure as exc:
            timed_out = (block in self.budget.stopped or exc.summary.get('failure_scope') == 'caller_budget_exhausted')
            row.update(status='block_stopped' if timed_out else 'numerically_unresolved',
                       error_type=type(exc).__name__, error=str(exc),
                       failure_scope=exc.summary.get('failure_scope'))
            if timed_out:
                self.budget.stopped.setdefault(block, str(exc))
                raise StopBlock(str(exc)) from exc
            return chain
        except Exception as exc:
            row.update(status='failed_without_finite_objective', error_type=type(exc).__name__, error=str(exc))
            return chain
        finally:
            if (objective and key is not None and row['status'] in
                    ('numerically_unresolved', 'failed_without_finite_objective')):
                # Exact failed duplicates have no finite score and no retry.
                self.cache[key] = chain
            self.writer.json(record_path, row, replace=True)
            self.snapshot()

    def optimizer(self, name, x0):
        record = {'block': name, 'initial_x': np.asarray(x0).tolist(),
                  'initial_simplex': inward_simplex(x0).tolist(), 'maxfev': 40,
                  'status': 'running', 'requests_before': self.objective_count}
        self.report['optimizers'].append(record)
        last_return = time.monotonic()
        local_requests = 0

        def objective(x):
            nonlocal last_return, local_requests
            now = time.monotonic()
            try:
                try:
                    self.budget.charge_optimizer_overhead(name, now - last_return)
                except BudgetExceeded:
                    # This callback is still a request. request() records it,
                    # then the stopped block is enforced before any solve.
                    pass
                local_requests += 1
                if local_requests > 40:
                    raise StopBlock('Optimizer maxfev request cap exhausted')
                chain = self.request(x, label=f'{name}_{local_requests:02d}', block=name,
                                     precision=SEARCH)
                value = chain['record']['objective_mean_phase_MSE']
                return float(value) if chain['record']['finite_objective_available'] else math.inf
            finally:
                last_return = time.monotonic()
        try:
            result = minimize(objective, np.asarray(x0, dtype=float), method='Nelder-Mead',
                options={'initial_simplex': np.array(record['initial_simplex']), 'maxfev': 40,
                         'adaptive': False, 'xatol': 1e-3, 'fatol': 1e-6, 'disp': False})
            self.budget.charge_optimizer_overhead(name, time.monotonic() - last_return)
            record.update(status='completed_bounded_optimizer', scipy_success=bool(result.success),
                          scipy_message=str(result.message), scipy_nfev=int(result.nfev),
                          scipy_nit=int(result.nit), scipy_x=result.x.tolist(), scipy_fun=float(result.fun),
                          global_optimum_claim=False)
        except (StopBlock, BudgetExceeded) as exc:
            record.update(status='stopped_no_retry', error=str(exc))
        except Exception as exc:
            record.update(status='optimizer_failed_no_retry', error_type=type(exc).__name__, error=str(exc))
        finally:
            record.update(requests_after=self.objective_count, objective_requests_in_block=local_requests)
            self.snapshot()

    def execute(self):
        xold = normalized(OLD_KAPPA, OLD_RATIO, OLD_B, OLD_U0)
        self.report['status'] = 'running_control_regression'
        try:
            dop = self.request(xold, label='control_DOP853', block='setup', precision=CONTROL,
                               method='DOP853', objective=False)
            if dop['record']['status'] != 'completed_chain':
                raise StopBlock('Old DOP853 actual-history control failed; no optimization')
            rad = self.request(xold, label='control_Radau', block='setup', precision=CONTROL)
            if rad['record']['status'] != 'completed_chain':
                raise StopBlock('Old Radau actual-history control failed; no optimization')
            def controls(check):
                checks = {}; reference = self.audit['saved_baseline']['phase_scores']
                for name in CALIBRATION:
                    check()
                    comparison = compare_phases(dop['results'][name], rad['results'][name])
                    observed_scores = {method: chain['results'][name]['summary']['metrics']['all_eligible']['RMSE_decade']
                                       for method, chain in [('DOP853', dop), ('Radau', rad)]}
                    score_errors = {method: abs(value-reference[name]['RMSE_decade']) for method,value in observed_scores.items()}
                    checks[name] = {'cross_method': comparison, 'old_summary_RMSE': reference[name]['RMSE_decade'],
                                    'actual_continuous_control_RMSE': observed_scores,
                                    'absolute_reference_score_errors': score_errors,
                                    'passed': comparison['passed'] and max(score_errors.values()) <= 1e-4}
                return {'phases': checks, 'passed': all(item['passed'] for item in checks.values()),
                        'old_NPZ_read': False, 'actual_state_handoffs_required': True}
            self.report['control_regression'] = self.budget.run('setup', 'control_regression_comparison',
                                                                controls, limit_s=12.)
            if not self.report['control_regression']['passed']:
                raise StopBlock('Old continuous controls did not pass method/score regression')
        except (StopBlock, BudgetExceeded) as exc:
            self.report.update(status='stopped_before_calibration', stop_reason=str(exc))
            return

        self.report['status'] = 'running_predeclared_seeds'
        for index, seed in enumerate(configuration()['seed_coordinates'], 1):
            try:
                chain = self.request(normalized(*seed), label=f'new_seed_{index}', block='setup', precision=SEARCH)
                if chain['record']['finite_objective_available']:
                    self.new_seeds.append(chain)
            except StopBlock as exc:
                self.report['setup_stop_reason'] = str(exc)
                break
        self.report['status'] = 'running_bounded_optimizers'
        if self.new_seeds:
            seed = min(self.new_seeds, key=lambda item: item['record']['objective_mean_phase_MSE'])
            self.report['new_optimizer_seed_request'] = seed['record']['request_id']
            self.optimizer('new_optimizer', seed['record']['x_normalized'])
        else:
            self.report['optimizers'].append({'block': 'new_optimizer', 'status': 'skipped_no_completed_new_seed'})
        self.optimizer('old_optimizer', xold)
        if not self.successful:
            self.report['status'] = 'no_completed_calibration_candidate'
            return
        best = min(self.successful, key=lambda item: item['record']['objective_mean_phase_MSE'])
        self.report['selected_candidate'] = deepcopy(best['record'])
        self.report['selected_candidate']['selection_rule'] = 'Minimum completed finite three-phase MSE only; no continuation score'
        self.report['status'] = 'running_fixed_final_validation'
        finals = {}
        for precision in (NOMINAL, REFINED):
            try:
                chain = self.request(best['record']['x_normalized'], label=f'final_{precision.name}',
                                     block='final', precision=precision, objective=False,
                                     phases=ALL_PHASES, independent=True)
            except StopBlock as exc:
                self.report.update(status='final_numerical_budget_stopped', stop_reason=str(exc))
                return
            finals[precision.name] = chain
            self.report.setdefault('final_chains', {})[precision.name] = deepcopy(chain['record'])
            if chain['record']['status'] != 'completed_chain':
                self.report['status'] = 'final_validation_incomplete_no_retry'
                return
        def compare_final(check):
            comparisons = {}
            first_nominal = finals['nominal']['results']['set4']['summary']
            first_refined = finals['refined']['results']['set4']['summary']
            selected = self.report['selected_candidate']['parameters']
            same_initial = (first_nominal['initial_u'] == first_refined['initial_u'] == selected['initial_u'])
            same_parameters = all(
                chain['record']['parameters'] == selected and
                all(result['summary']['parameters'] == first_nominal['parameters'] and
                    result['summary']['b'] == selected['b'] for result in chain['results'].values())
                for chain in finals.values())
            independent_checks = {}
            handoff_checks = {}
            for precision, chain in finals.items():
                independent_checks[precision] = {}
                handoff_checks[precision] = (len(chain['record']['handoffs']) == 5 and
                    all(item['passed'] for item in chain['record']['handoffs']))
                for name, result in chain['results'].items():
                    audit = result['summary']['phase_integrity']
                    independent_checks[precision][name] = bool(audit['independent_requested'] and
                        audit['passed'] and audit['complete_integrity_passed'] and
                        result['summary']['independent_native_readout']['passed'])
            for name in ALL_PHASES:
                check()
                comparisons[name] = compare_phases(finals['nominal']['results'][name], finals['refined']['results'][name])
            all_independent = all(len(checks) == 5 and all(checks.values()) for checks in independent_checks.values())
            return {'phases': comparisons, 'same_selected_parameters': same_parameters,
                    'same_overall_initial_u': same_initial, 'overall_initial_u': first_nominal['initial_u'],
                    'independent_requested_and_passed_by_phase': independent_checks,
                    'actual_handoffs_passed_by_precision': handoff_checks,
                    'passed': (same_initial and same_parameters and all_independent and
                               all(handoff_checks.values()) and all(item['passed'] for item in comparisons.values()))}
        try:
            self.report['final_precision_comparison'] = self.budget.run('final', 'compare_final_precisions',
                                                                       compare_final, limit_s=20.)
        except BudgetExceeded as exc:
            self.report.update(status='final_numerical_budget_stopped', stop_reason=str(exc))
            return
        self.report['numerical_validation_passed'] = self.report['final_precision_comparison']['passed']
        self.report['status'] = ('completed_bounded_joint_diagnostic' if self.report['numerical_validation_passed']
                                 else 'completed_results_failed_precision_comparison')


def load_inputs(writer):
    started = time.monotonic()
    audit = json.loads((HERE / 'DATA_PROTOCOL_AUDIT.json').read_text())
    kernel = load_joint_kernel()
    paths = [HERE / name for name in ('EXPERIMENT_PLAN.md', 'DATA_PROTOCOL_AUDIT.json',
             'training_phases.npz', 'joint_forward.py', 'joint_validation.py', 'bounded_joint_calibration.py')]
    hashes = hash_paths(paths)
    if hashes[str(HERE / 'training_phases.npz')] != audit['small_cache']['sha256']:
        raise ValueError('Training cache hash disagrees with predeclared data audit')
    data = {}
    expected_keys = {f'{name}_{key}' for name in ALL_PHASES
                     for key in ('command_voltage_v', 'observed_current_a', 'scoring_mask')}
    with np.load(HERE / 'training_phases.npz', allow_pickle=False) as archive:
        if set(archive.files) != expected_keys:
            raise ValueError('Cache must contain exactly the five declared phases and masks')
        for name in ALL_PHASES:
            phase = 'reset' if name.startswith('reset') else 'set'
            item = {key: archive[f'{name}_{key}'].copy() for key in
                    ('command_voltage_v', 'observed_current_a', 'scoring_mask')}
            v, observed, mask = item['command_voltage_v'], item['observed_current_a'], item['scoring_mask']
            original = np.isfinite(abs(observed)) & (abs(observed) >= (1e-11 if phase == 'set' else 1e-12)) & (abs(v) > 1e-12)
            expected = (503, 500) if phase == 'set' else (173, 170)
            if mask.dtype != np.bool_ or not np.array_equal(mask, original) or (len(v), int(mask.sum())) != expected:
                raise ValueError('Original native mask/count mismatch: ' + name)
            if v[0] != 0. or v[-1] != 0. or int((mask & (abs(v) <= .2 + 1e-12)).sum()) != 20:
                raise ValueError('Native phase endpoints/low20 count mismatch: ' + name)
            item.update(initial_command_voltage_v=0., nearzero_segment_budget=1e-10,
                        circuit_mode='fast', phase=phase, record_id=int(name.removeprefix(phase)))
            data[name] = item
    p = kernel.parameters
    if p.kappa_per_v2 != OLD_KAPPA or p.J_positive_a != OLD_J or p.lambda_per_v != .4470398752554136:
        raise ValueError('Loaded frozen kernel parameters disagree with declared old control')
    if kernel.ron != 58.463033350725695 or kernel.gc != .03127031375421752:
        raise ValueError('Loaded frozen Ron/Gamma_c disagree with the declared material')
    merged_hashes = dict(kernel.source_sha256)
    merged_hashes.update(hashes)
    return kernel, data, audit, merged_hashes, time.monotonic() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-authorized', action='store_true')
    args = parser.parse_args()
    if not args.run_authorized:
        parser.error('Prepared only: reviewed execution requires --run-authorized')
    if RESULT.exists():
        parser.error('Terminal result already exists; this bounded experiment will not rerun')
    started_path = HERE / 'RUN_STARTED.json'
    if started_path.exists():
        parser.error('A prior run was started; no automatic restart or retry is allowed')
    if not (HERE / 'EXPERIMENT_PLAN.md').is_file():
        parser.error('Missing frozen EXPERIMENT_PLAN.md')
    writer = Artifacts()
    writer.json(started_path, {'status': 'authorized_single_attempt_started',
                              'configuration': configuration()})
    for directory in ('requests', 'phase_trials'):
        (HERE / directory).mkdir(exist_ok=True)
    report = {'status': 'loading_inputs', 'configuration': configuration(), 'requests': [], 'optimizers': [],
              'numerical_validation_passed': False, 'full_unified_model_achieved': False,
              'full_joint_contract_evaluated': False, 'original_holdout_accessed': False,
              'bare_data_accessed': False, 'continuation_used_for_selection': False,
              'new_state_inversions': 0, 'old_trajectory_NPZ_read': False}
    runner = None
    try:
        kernel, data, audit, hashes, loading = load_inputs(writer)
        report.update(source_sha256=hashes, input_loading_and_hash_wall_s=loading,
                      fixed_material=deepcopy(kernel._joint_fixed_material),
                      saved_baseline=deepcopy(audit['saved_baseline']))
        writer.json(HERE / 'bounded_joint_calibration.plan.json',
                    {'configuration': configuration(), 'source_sha256': hashes})
        runner = Runner(kernel, data, audit, writer, report)
        runner.execute()
        # Hash-only provenance check, outside every numerical budget.
        hash_started = time.monotonic()
        actual = hash_paths(hashes)
        report['final_source_hash_wall_s'] = time.monotonic() - hash_started
        report['source_files_unchanged'] = actual == hashes
        if not report['source_files_unchanged']:
            report['status'] = 'source_changed_result_not_validated'
            report['numerical_validation_passed'] = False
    except Exception as exc:
        report.update(status='stopped_with_preserved_exception', error_type=type(exc).__name__, error=str(exc))
    finally:
        if runner is not None:
            report.update(objective_requests=runner.objective_count, timing=runner.budget.snapshot())
        report.update(artifact_catalog=deepcopy(writer.catalog), artifact_IO_wall_s=writer.io_wall_s,
                      terminal_result=True, automatic_retry_permitted=False)
        writer.json(RESULT, report)
        writer.progress('Terminal bounded joint diagnostic: ' + report['status'])


if __name__ == '__main__':
    main()
