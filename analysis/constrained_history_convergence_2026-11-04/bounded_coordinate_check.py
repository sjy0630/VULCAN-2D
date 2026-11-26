"""Prepared finite coordinate-grid check; no model work on import.

Execute only after root's source freeze, using --run-authorized. Calibration
is SET4/RESET4/SET5 only. The three phase upper bounds are modeling goals, never
noise intervals. The same four parameters and one whole-history state apply.
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

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / 'joint_history_calibration_2026-11-04'
if str(PREVIOUS) not in sys.path:
    sys.path.insert(0, str(PREVIOUS))
from joint_forward import Precision, PhaseFailure, load_joint_kernel, with_parameters, simulate_phase
from joint_validation import phase_integrity, compare_phases

PHASES = ('set4', 'reset4', 'set5')
SET_PHASES = ('set4', 'set5')
RESULT = HERE / 'bounded_coordinate_check.json'
PROGRESS = HERE / 'bounded_coordinate_check.progress.json'
OLD_REQUEST = PREVIOUS / 'requests/002_control_Radau.json'
SEED_REQUEST = PREVIOUS / 'requests/057_old_optimizer_11.json'
DATA = PREVIOUS / 'training_phases.npz'
OLD_KAPPA, OLD_J = 77.70013824898767, .0041089272077381635
LOW = np.array([math.log10(OLD_KAPPA / 10.), math.log10((OLD_KAPPA / OLD_J) / 10.), 0., 2.])
HIGH = np.array([math.log10(100. * 14648.824351938409),
                 math.log10(100. * 426403.0838962568), math.log(10.), 16.])
NOMINAL = Precision('nominal', 1e-7, 1e-8, .02)
REFINED = Precision('refined', 1e-8, 1e-9, .01)
GRID_UNIT, STEP_UNITS, IMPROVEMENT = .005, (4, 2, 1), 1e-4
MAX_REQUESTS = 96


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
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def hash_paths(paths):
    paths = sorted({str(Path(p).resolve()) for p in paths})
    with ThreadPoolExecutor(max_workers=4) as pool:
        return dict(zip(paths, pool.map(hash_file, paths)))


class Artifacts:
    def __init__(self):
        self.io_wall_s = 0.
        self.catalog = {}

    def save(self, path, producer, *, replace=False, catalog=True):
        started = time.monotonic()
        try:
            payload = producer()
            item = {'path': str(path), 'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}
            with Path(path).open('wb' if replace else 'xb') as handle:
                handle.write(payload)
            if catalog:
                self.catalog[str(path)] = item
            return item
        finally:
            self.io_wall_s += time.monotonic() - started

    def json(self, path, value, *, replace=False, catalog=True):
        snapshot = deepcopy(value)
        return self.save(path, lambda: (json.dumps(clean(snapshot), ensure_ascii=False,
                         indent=2, allow_nan=False) + '\n').encode(), replace=replace, catalog=catalog)

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


class Budget:
    limits = {'verification': 90., 'search': 180., 'final': 90.}

    def __init__(self):
        self.used = {name: 0. for name in self.limits}
        self.stopped, self.calls, self.active = {}, [], None

    def ensure(self, block, elapsed=0.):
        if block in self.stopped:
            raise BudgetExceeded(self.stopped[block])
        if self.used[block] + elapsed >= self.limits[block]:
            self.stopped[block] = f'{block} numerical budget {self.limits[block]:g} s exhausted'
            raise BudgetExceeded(self.stopped[block])

    def check(self):
        if self.active is None:
            raise RuntimeError('Budget callback outside numerical operation')
        elapsed = time.monotonic() - self.active['started']
        block = self.active['block']
        if elapsed >= self.active['limit_s']:
            self.stopped[block] = f"Per-phase {self.active['limit_s']:g} s budget exhausted"
            raise BudgetExceeded(self.stopped[block])
        self.ensure(block, elapsed)

    def run(self, block, label, function, limit_s):
        if self.active is not None:
            raise RuntimeError('Nested numerical timers are prohibited')
        self.ensure(block)
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

    def snapshot(self):
        return {'used_s': dict(self.used), 'total_numerical_s': sum(self.used.values()),
                'limits_s': dict(self.limits), 'stopped_blocks': dict(self.stopped), 'calls': deepcopy(self.calls)}


def decode(x):
    x = np.asarray(x, dtype=float)
    if x.shape != (4,) or not np.all(np.isfinite(x)) or np.any((x < 0.) | (x > 1.)):
        raise ValueError('Outside normalized declared domain; no boundary clipping')
    z = LOW + x * (HIGH - LOW)
    kappa, ratio = 10.**z[0], 10.**z[1]
    return {'kappa_per_v2': float(kappa), 'ratio_kappa_over_J': float(ratio),
            'J_positive_a': float(kappa / ratio), 'b': float(z[2]), 'initial_u': float(z[3])}


def cache_key(parameters, precision):
    return (tuple(parameters[key] for key in ('kappa_per_v2', 'ratio_kappa_over_J', 'J_positive_a', 'b', 'initial_u')),
            precision.rtol, precision.atol, precision.max_step, 'Radau', PHASES)


def configuration():
    return {'contract': 'EXPERIMENT_PLAN.md', 'phases': list(PHASES),
            'variables': ['log10(kappa)', 'log10(kappa/J)', 'b', 'u0'],
            'coordinate_lower': LOW.tolist(), 'coordinate_upper': HIGH.tolist(),
            'normalized_domain': [[0., 1.]] * 4, 'one_whole_history_initial_state': True,
            'u0_not_measured': True, 'objective': 'mean three original-mask phase MSE; retain all1170 RMSE',
            'modeling_goals': 'All three phase RMSEs <= same-precision actual old control; SET subset also reported',
            'constraints_are_not_noise_or_confidence_intervals': True,
            'nominal': asdict(NOMINAL), 'refined': asdict(REFINED), 'method': 'Radau',
            'fixed_material': {'lambda_per_v': .4470398752554136, 'Ron_ohm': 58.463033350725695,
                               'gamma_c_ev': .03127031375421752, 'recovery_order': 2, 'parallel_path_a_s': 0.},
            'coordinate_grid_unit': GRID_UNIT, 'step_units': list(STEP_UNITS),
            'normalized_step_sizes': [.02, .01, .005], 'minimum_MSE_improvement_decade2': IMPROVEMENT,
            'direction_order': [[axis, sign] for axis in range(4) for sign in (-1, 1)],
            'max_search_requests': MAX_REQUESTS, 'complete_round_requests': 8,
            'budget_seconds': dict(Budget.limits), 'verification_phase_seconds': 12.,
            'search_phase_seconds': 12., 'final_phase_seconds': 20.,
            'nearzero_segment_budget': 1e-10, 'initial_command_voltage_v': 0., 'circuit_mode': 'fast',
            'independent_native_bisections': 48, 'no_solver_fallback_or_automatic_retry': True,
            'preflight_rule': 'All three seed RMSEs below same-precision old in both precisions; each gap exceeds sum of observed old/seed nominal-refined RMSE differences',
            'preflight_difference_is_not_confidence_interval': True,
            'unknown_direction_policy': 'Full requested round may accept best completed feasible improvement, flagged uncertified; without such improvement stop unresolved, do not shrink',
            'incomplete_round_policy': 'Keep preceding incumbent; never select a point from a budget-interrupted round',
            'grid_stop_scope': 'Only declared-domain feasible coordinate directions at final incumbent and step .005; not continuous stationarity, KKT, global optimum, or diagonal search',
            'extension_model_or_score_accessed': False, 'original_holdout_accessed': False,
            'original_contract_unchanged': {'pairs': 46, 'residuals': 27023, 'aggregate_feature_gates': 20}}


class Study:
    def __init__(self, kernel, data, controls, writer, report):
        self.kernel, self.data, self.controls = kernel, data, controls
        self.writer, self.report = writer, report
        self.budget = Budget()
        self.search_requests = 0
        self.cache = {}
        self.old = {}

    def snapshot(self):
        self.report.update(search_requests=self.search_requests, timing=self.budget.snapshot(),
                           artifact_IO_wall_s=self.writer.io_wall_s, artifact_catalog=deepcopy(self.writer.catalog))
        self.writer.json(PROGRESS, self.report, replace=True, catalog=False)

    def modeling_goals(self, scores, precision):
        if precision.name not in self.old:
            return None
        targets = self.old[precision.name]['record']['scores']['phase_metrics']
        rows = {}
        for name in PHASES:
            actual = scores['phase_metrics'][name]['all_eligible']['RMSE_decade']
            limit = targets[name]['all_eligible']['RMSE_decade']
            rows[name] = {'actual_RMSE_decade': actual, 'old_control_RMSE_decade': limit,
                          'margin_old_minus_actual_decade': limit-actual, 'passed': actual <= limit}
        return {'passed': all(row['passed'] for row in rows.values()),
                'all_three_goals_passed': all(row['passed'] for row in rows.values()),
                'SET_subset_passed': all(rows[name]['passed'] for name in SET_PHASES), 'phases': rows,
                'scope': 'Modeling objective only; no noise or physical/statistical confidence interpretation'}

    def request(self, x, label, block, precision, *, fixed_parameters=None,
                independent=False, search=False, lattice_offset=None):
        number = len(self.report['requests']) + 1
        if search:
            self.search_requests += 1
        row = {'request_id': number, 'label': label, 'block': block, 'status': 'requested',
               'x_normalized': np.asarray(x).tolist(), 'precision': asdict(precision),
               'method': 'Radau', 'search_request': search,
               'search_request_index': self.search_requests if search else None,
               'lattice_offset': None if lattice_offset is None else np.asarray(lattice_offset).tolist(),
               'finite_score_available': False, 'objective_mean_phase_MSE': None,
               'phase_results': [], 'handoffs': [], 'physical_rejection': False,
               'neighborhood_classification': 'not_yet_resolved'}
        self.report['requests'].append(row)
        path = HERE / 'requests' / f'{number:03d}_{label}.json'
        self.writer.json(path, row)
        result = {'record': row, 'results': {}}
        key = None
        try:
            if search and self.search_requests > MAX_REQUESTS:
                raise StopBlock('Search request limit exceeded before any model execution')
            try:
                decoded = decode(x)
            except ValueError as exc:
                row.update(status='outside_declared_parameter_domain', error=str(exc),
                           neighborhood_classification='geometrically_outside_domain')
                return result
            p = dict(decoded if fixed_parameters is None else fixed_parameters)
            if fixed_parameters is not None:
                if set(p) != set(decoded) or any(not math.isclose(p[k], decoded[k], rel_tol=1e-13, abs_tol=1e-13) for k in decoded):
                    raise ValueError('Fixed source parameters disagree with their normalized coordinates')
            row['parameters'] = p
            self.budget.ensure(block)
            key = cache_key(p, precision)
            if search and key in self.cache:
                prior = self.cache[key]
                for name in ('finite_score_available', 'objective_mean_phase_MSE', 'scores',
                             'modeling_goals', 'phase_results', 'handoffs', 'neighborhood_classification'):
                    if name in prior['record']:
                        row[name] = deepcopy(prior['record'][name])
                row.update(status='cached_completed' if row['finite_score_available'] else 'cached_numerically_unresolved',
                           cache_source_request=prior['record']['request_id'])
                if row['finite_score_available']:
                    row['modeling_goals'] = self.modeling_goals(row['scores'], precision)
                    row['neighborhood_classification'] = ('completed_modeling_goal_feasible'
                        if (row['modeling_goals'] or {}).get('passed') else 'completed_modeling_goal_infeasible')
                result['results'] = prior['results']
                return result
            candidate = with_parameters(self.kernel, kappa_per_v2=p['kappa_per_v2'], J_positive_a=p['J_positive_a'])
            incoming, command, previous = p['initial_u'], 0., None
            for name in PHASES:
                saved = None
                phase_data = dict(self.data[name], initial_command_voltage_v=command)
                phase_record = {'phase': name, 'status': 'started', 'initial_u': incoming,
                                'initial_command_voltage_v': command}
                row['phase_results'].append(phase_record)
                phase_label = f'{number:03d}_{label}_{name}'
                try:
                    def calculate(check):
                        nonlocal saved
                        saved = simulate_phase(candidate, phase_data, incoming, p['b'], precision,
                                               'Radau', check, collect_rhs=independent)
                        audit = phase_integrity(candidate, saved, phase_data, incoming, p['b'],
                                                independent=independent, check_budget=check)
                        if not audit['passed'] or (independent and not audit['complete_integrity_passed']):
                            raise ValueError('Phase integrity failed; partial/full arrays preserved')
                        s, a = saved['summary'], saved['arrays']
                        state_error = max(abs(s['initial_u']-incoming), abs(float(a['state_u'][0])-incoming))
                        command_error = max(abs(s['initial_command_voltage_v']-command), abs(float(a['command_voltage_v'][0])-command))
                        join = {'from_phase': previous, 'to_phase': name,
                                'source': 'whole_history_initial_state' if previous is None else 'actual_previous_terminal',
                                'expected_initial_u': incoming, 'actual_initial_u': s['initial_u'],
                                'expected_initial_command_voltage_v': command,
                                'actual_initial_command_voltage_v': s['initial_command_voltage_v'],
                                'state_error_u': state_error, 'command_error_v': command_error,
                                'passed': state_error <= 1e-12 and command_error <= 1e-12}
                        row['handoffs'].append(join)
                        if not join['passed']:
                            raise ValueError('Actual state/command handoff failed')
                        return saved
                    answer = self.budget.run(block, phase_label, calculate, 20. if block == 'final' else 12.)
                    incoming = float(answer['summary']['terminal_u'])
                    command = float(answer['summary']['final_command_voltage_v'])
                    previous = name
                    phase_record.update(status='completed', terminal_u=incoming,
                                        terminal_command_voltage_v=command,
                                        metrics=deepcopy(answer['summary']['metrics']))
                    result['results'][name] = answer
                except PhaseFailure as exc:
                    saved = {'summary': exc.summary, 'arrays': exc.arrays}
                    phase_record.update(status='failed_phase', error_type=type(exc).__name__, error=str(exc))
                    raise
                except Exception as exc:
                    phase_record.update(status='failed_phase_or_validation', error_type=type(exc).__name__, error=str(exc))
                    raise
                finally:
                    if saved is not None:
                        phase_record['artifacts'] = {
                            'arrays': self.writer.npz(HERE / 'phase_trials' / f'{phase_label}.npz', saved['arrays']),
                            'summary': self.writer.json(HERE / 'phase_trials' / f'{phase_label}.json', saved['summary'])}
            def score(check):
                check()
                metrics = {name: deepcopy(result['results'][name]['summary']['metrics']) for name in PHASES}
                n = [metrics[name]['all_eligible']['n'] for name in PHASES]
                e = [metrics[name]['all_eligible']['RMSE_decade'] for name in PHASES]
                if n != [500, 170, 500] or not all(math.isfinite(t) for t in e):
                    raise ValueError('Original 1170-point scoring invalid')
                return {'mean_phase_MSE_decade2': sum(t*t for t in e)/3.,
                        'all1170_RMSE_decade': math.sqrt(sum(a*b*b for a,b in zip(n,e))/1170.),
                        'phase_metrics': metrics}
            scores = self.budget.run(block, f'{number:03d}_score', score, 20. if block == 'final' else 12.)
            feasible = self.modeling_goals(scores, precision)
            row.update(status='completed_chain', scores=scores, finite_score_available=True,
                       objective_mean_phase_MSE=scores['mean_phase_MSE_decade2'],
                       modeling_goals=feasible, terminal_u=incoming,
                       neighborhood_classification='completed_modeling_goal_feasible' if feasible is not None and feasible['passed']
                       else 'completed_modeling_goal_infeasible' if feasible is not None else 'completed_control')
            self.cache[key] = result
            if independent:
                merged = {f'{name}__{k}': v for name, answer in result['results'].items() for k,v in answer['arrays'].items()}
                row['complete_three_phase_arrays'] = self.writer.npz(HERE / f'{label}_chain.npz', merged)
            rmse = [scores['phase_metrics'][name]['all_eligible']['RMSE_decade'] for name in PHASES]
            self.writer.progress(f"Completed {label}: RMSE {rmse[0]:.6f}/{rmse[1]:.6f}/{rmse[2]:.6f}; "
                                 f"mean MSE {scores['mean_phase_MSE_decade2']:.6g}; search requests {self.search_requests}")
            return result
        except (BudgetExceeded, StopBlock) as exc:
            row.update(status='block_stopped', error_type=type(exc).__name__, error=str(exc),
                       neighborhood_classification='numerically_unresolved')
            raise StopBlock(str(exc)) from exc
        except PhaseFailure as exc:
            timeout = block in self.budget.stopped or exc.summary.get('failure_scope') == 'caller_budget_exhausted'
            row.update(status='block_stopped' if timeout else 'numerically_unresolved', error_type=type(exc).__name__,
                       error=str(exc), failure_scope=exc.summary.get('failure_scope'),
                       neighborhood_classification='numerically_unresolved')
            if timeout:
                self.budget.stopped.setdefault(block, str(exc))
                raise StopBlock(str(exc)) from exc
            return result
        except Exception as exc:
            row.update(status='numerically_unresolved', error_type=type(exc).__name__, error=str(exc),
                       neighborhood_classification='numerically_unresolved')
            return result
        finally:
            if key is not None and row['status'] == 'numerically_unresolved':
                self.cache[key] = result
            self.writer.json(path, row, replace=True)
            self.snapshot()

    def compare_pair(self, pair, block, label):
        def calculate(check):
            nominal, refined = pair['nominal'], pair['refined']
            a, z = nominal['record'], refined['record']
            phases = {}
            independent = {}
            same_parameters = a['parameters'] == z['parameters']
            initial = a['parameters']['initial_u']
            same_initial = all(chain['results']['set4']['summary']['initial_u'] == initial for chain in pair.values())
            joins = all(len(chain['record']['handoffs']) == 3 and all(h['passed'] for h in chain['record']['handoffs']) for chain in pair.values())
            for name in PHASES:
                check()
                phases[name] = compare_phases(nominal['results'][name], refined['results'][name])
            for precision, chain in pair.items():
                independent[precision] = {}
                for name, answer in chain['results'].items():
                    s = answer['summary'];audit = s['phase_integrity']
                    independent[precision][name] = bool(audit['independent_requested'] and audit['passed'] and
                        audit['complete_integrity_passed'] and s['independent_native_readout']['passed'])
            return {'same_parameters': same_parameters, 'same_overall_initial_u': same_initial,
                    'actual_handoffs_passed': joins, 'phases': phases,
                    'independent_requested_and_passed': independent,
                    'passed': bool(same_parameters and same_initial and joins and all(p['passed'] for p in phases.values()) and
                                   all(len(p) == 3 and all(p.values()) for p in independent.values()))}
        return self.budget.run(block, label, calculate, 20. if block == 'final' else 12.)

    def preflight(self):
        pairs = {}
        for name in ('old', 'request57'):
            source = self.controls[name]
            pair = {}
            for precision in (NOMINAL, REFINED):
                answer = self.request(source['x_normalized'], f'verify_{name}_{precision.name}', 'verification',
                                      precision, fixed_parameters=source['parameters'], independent=True)
                pair[precision.name] = answer
                self.report.setdefault('verification_chains', {}).setdefault(name, {})[precision.name] = deepcopy(answer['record'])
                if answer['record']['status'] != 'completed_chain':
                    self.report.update(status='preflight_numerically_unresolved', stop_reason=f'{name}/{precision.name} did not complete')
                    return None
            pairs[name] = pair
            comparison = self.compare_pair(pair, 'verification', f'compare_{name}_precisions')
            self.report.setdefault('verification_precision_comparisons', {})[name] = comparison
            if not comparison['passed']:
                self.report.update(status='preflight_precision_validation_failed', stop_reason=name)
                return None
            if name == 'old':
                self.old = pair

        def confirm(check):
            rows = {}
            for phase in PHASES:
                check()
                e = {name: {precision: pair[precision]['record']['scores']['phase_metrics'][phase]['all_eligible']['RMSE_decade']
                            for precision in ('nominal', 'refined')} for name,pair in pairs.items()}
                difference = abs(e['old']['nominal']-e['old']['refined']) + abs(e['request57']['nominal']-e['request57']['refined'])
                margins = {precision: e['old'][precision]-e['request57'][precision] for precision in ('nominal', 'refined')}
                rows[phase] = {'RMSE_decade': e, 'observed_numerical_difference_sum_decade': difference,
                               'old_minus_request57_margin_decade': margins,
                               'passed': all(gap > difference for gap in margins.values())}
            return {'phases': rows, 'passed': all(item['passed'] for item in rows.values()),
                    'difference_scope': 'Observed two-precision numerical change only, not an uncertainty/confidence interval'}
        self.report['request57_improvement_confirmation'] = self.budget.run('verification', 'confirm_request57_improvement', confirm, 12.)
        if not self.report['request57_improvement_confirmation']['passed']:
            self.report['status'] = 'request57_improvement_not_confirmed_search_not_started'
            return None
        return pairs['request57']

    def coordinate_search(self, seed):
        origin = np.asarray(self.controls['request57']['x_normalized'], dtype=float)
        offset = np.zeros(4, dtype=int)
        incumbent = seed['nominal']
        self.report['search_outcome'] = {'status': 'running', 'grid_stopped': False}
        for step in STEP_UNITS:
            while True:
                if self.search_requests + 8 > MAX_REQUESTS:
                    self.report['search_outcome'].update(status='request_budget_exhausted', incumbent_request=incumbent['record']['request_id'])
                    return incumbent
                round_id = len(self.report['rounds']) + 1
                round_row = {'round_id': round_id, 'step_units': step, 'normalized_step': step*GRID_UNIT,
                             'center_offset': offset.tolist(), 'center_x': (origin+GRID_UNIT*offset).tolist(),
                             'center_request_id': incumbent['record']['request_id'],
                             'center_MSE': incumbent['record']['objective_mean_phase_MSE'],
                             'requests': [], 'status': 'running', 'completed_all_eight_requests': False,
                             'selected_from_incomplete_round': False}
                self.report['rounds'].append(round_row)
                trials = []
                for axis in range(4):
                    for sign in (-1, 1):
                        proposal = offset.copy();proposal[axis] += sign*step
                        x = origin+GRID_UNIT*proposal
                        try:
                            trial = self.request(x, f'round{round_id:02d}_axis{axis}_{"minus" if sign < 0 else "plus"}',
                                                 'search', NOMINAL, search=True, lattice_offset=proposal)
                        except StopBlock as exc:
                            round_row.update(status='incomplete_budget_stopped_keep_previous_incumbent', stop_reason=str(exc))
                            round_row['requests'].append(self.report['requests'][-1]['request_id'])
                            self.report['search_outcome'].update(status='numerical_budget_exhausted',
                                incumbent_request=incumbent['record']['request_id'], incomplete_round=round_id)
                            self.snapshot()
                            return incumbent
                        round_row['requests'].append(trial['record']['request_id'])
                        trials.append((proposal, trial))
                round_row['completed_all_eight_requests'] = True

                def judge(check):
                    check()
                    unknown = [t['record']['request_id'] for _,t in trials
                               if t['record']['neighborhood_classification'] == 'numerically_unresolved']
                    known = [(off,t) for off,t in trials if t['record']['finite_score_available'] and
                             (t['record'].get('modeling_goals') or {}).get('passed')]
                    improving = [(off,t) for off,t in known if
                        incumbent['record']['objective_mean_phase_MSE']-t['record']['objective_mean_phase_MSE'] >= IMPROVEMENT]
                    best = min(improving, key=lambda item: item[1]['record']['objective_mean_phase_MSE']) if improving else None
                    return unknown, best
                try:
                    unknown, best = self.budget.run('search', f'round{round_id:02d}_decision', judge, 12.)
                except BudgetExceeded as exc:
                    round_row.update(status='decision_budget_stopped_keep_previous_incumbent', stop_reason=str(exc))
                    self.report['search_outcome'].update(status='numerical_budget_exhausted', incumbent_request=incumbent['record']['request_id'])
                    self.snapshot()
                    return incumbent
                round_row.update(unresolved_requests=unknown, full_resolved_neighborhood=not unknown)
                if best is not None:
                    offset, incumbent = best
                    round_row.update(status='accepted_best_completed_feasible_improvement',
                                     selected_request=incumbent['record']['request_id'],
                                     selected_MSE=incumbent['record']['objective_mean_phase_MSE'],
                                     full_neighborhood_best_certified=not unknown)
                    self.snapshot()
                    continue
                if unknown:
                    round_row['status'] = 'unresolved_neighborhood_no_acceptable_completed_improvement'
                    self.report['search_outcome'].update(status='unresolved_neighborhood', incumbent_request=incumbent['record']['request_id'])
                    self.snapshot()
                    return incumbent
                round_row['status'] = 'resolved_neighborhood_no_acceptable_improvement'
                if step == 1:
                    self.report['search_outcome'].update(status='coordinate_grid_stopped', grid_stopped=True,
                        incumbent_request=incumbent['record']['request_id'], final_round=round_id,
                        final_center_offset=offset.tolist(), normalized_step=GRID_UNIT,
                        scope=configuration()['grid_stop_scope'])
                    self.snapshot()
                    return incumbent
                self.snapshot()
                break
        raise RuntimeError('Search left declared finite grid policy without a stop reason')

    def execute(self):
        self.report['status'] = 'running_fixed_preflight'
        try:
            seed = self.preflight()
        except (BudgetExceeded, StopBlock) as exc:
            self.report.update(status='preflight_budget_stopped', stop_reason=str(exc))
            return
        if seed is None:
            return
        self.report['status'] = 'running_finite_coordinate_check'
        incumbent = self.coordinate_search(seed)
        self.report['selected_incumbent'] = deepcopy(incumbent['record'])
        self.report['selected_incumbent']['selection_scope'] = 'Complete requested rounds only; original three-phase objective with all three phase non-degradation goals'
        self.report['status'] = 'running_fixed_final_validation'
        pair = {}
        for precision in (NOMINAL, REFINED):
            try:
                result = self.request(incumbent['record']['x_normalized'], f'final_{precision.name}', 'final', precision,
                                      fixed_parameters=incumbent['record']['parameters'], independent=True)
            except StopBlock as exc:
                self.report.update(status='final_budget_stopped', stop_reason=str(exc))
                return
            pair[precision.name] = result
            self.report.setdefault('final_chains', {})[precision.name] = deepcopy(result['record'])
            if result['record']['status'] != 'completed_chain':
                self.report['status'] = 'final_numerically_unresolved'
                return
        try:
            comparison = self.compare_pair(pair, 'final', 'compare_final_precisions')
            self.report['final_precision_comparison'] = comparison
            def final_metrics(check):
                check()
                goals = {precision: self.modeling_goals(answer['record']['scores'], NOMINAL if precision == 'nominal' else REFINED)
                         for precision,answer in pair.items()}
                improvements = {}
                margins = {}
                for precision,answer in pair.items():
                    score = answer['record']['scores']['mean_phase_MSE_decade2']
                    improvements[precision] = {
                        'MSE_old_minus_final': self.old[precision]['record']['scores']['mean_phase_MSE_decade2']-score,
                        'MSE_request57_minus_final': seed[precision]['record']['scores']['mean_phase_MSE_decade2']-score}
                for name in PHASES:
                    old_e = {pr:self.old[pr]['record']['scores']['phase_metrics'][name]['all_eligible']['RMSE_decade'] for pr in pair}
                    new_e = {pr:pair[pr]['record']['scores']['phase_metrics'][name]['all_eligible']['RMSE_decade'] for pr in pair}
                    num_change = abs(old_e['nominal']-old_e['refined'])+abs(new_e['nominal']-new_e['refined'])
                    gaps = {pr:old_e[pr]-new_e[pr] for pr in pair}
                    margins[name] = {'old_minus_final_RMSE_decade': gaps,
                        'observed_numerical_difference_sum_decade': num_change,
                        'both_precision_improvements_exceed_observed_difference': all(g > num_change for g in gaps.values())}
                mean_mse_comparisons = {}
                final_mse = {pr:pair[pr]['record']['scores']['mean_phase_MSE_decade2'] for pr in pair}
                for label, control in (('old', self.old), ('request57', seed)):
                    control_mse = {pr:control[pr]['record']['scores']['mean_phase_MSE_decade2'] for pr in pair}
                    numerical_difference = (abs(control_mse['nominal']-control_mse['refined']) +
                                            abs(final_mse['nominal']-final_mse['refined']))
                    gaps = {pr:control_mse[pr]-final_mse[pr] for pr in pair}
                    mean_mse_comparisons[label] = {
                        'control_minus_final_MSE_decade2': gaps,
                        'observed_two_precision_difference_sum_decade2': numerical_difference,
                        'both_precision_gain_exceeds_observed_difference': all(g > numerical_difference for g in gaps.values()),
                        'both_precision_no_loss': all(g >= 0. for g in gaps.values()),
                        'scope': 'Observed numerical difference only; lack of further gain is not physical failure'}
                return {'modeling_goals_by_precision': goals,
                        'all_three_goals_passed_both_precisions': all(g['all_three_goals_passed'] for g in goals.values()),
                        'SET_subset_passed_both_precisions': all(g['SET_subset_passed'] for g in goals.values()),
                        'MSE_improvements': improvements, 'phase_margins_relative_to_numerical_change': margins,
                        'mean_MSE_comparison_to_fixed_controls': mean_mse_comparisons,
                        'noise_or_confidence_interpretation': False}
            self.report['final_comparison_to_fixed_controls'] = self.budget.run('final', 'final_control_comparison', final_metrics, 20.)
        except BudgetExceeded as exc:
            self.report.update(status='final_budget_stopped', stop_reason=str(exc))
            return
        self.report['numerical_validation_passed'] = comparison['passed']
        self.report['modeling_goals_passed'] = self.report['final_comparison_to_fixed_controls']['all_three_goals_passed_both_precisions']
        self.report['all_three_goals_passed'] = self.report['modeling_goals_passed']
        self.report['status'] = ('completed_finite_coordinate_diagnostic' if comparison['passed'] and self.report['modeling_goals_passed']
                                 else 'completed_results_failed_final_validation_or_modeling_goal')


def load_inputs():
    started = time.monotonic()
    # These two source requests contain only SET4/RESET4/SET5. Do not read the
    # previous full result, any extension phase, or any old simulated NPZ.
    controls = {'old': json.loads(OLD_REQUEST.read_text()), 'request57': json.loads(SEED_REQUEST.read_text())}
    for label, row in controls.items():
        if row['status'] != 'completed_chain' or row['phases'] != list(PHASES) or not row['finite_objective_available']:
            raise ValueError('Invalid fixed three-phase source request: ' + label)
        decoded = decode(row['x_normalized'])
        if set(row['parameters']) != set(decoded) or any(not math.isclose(row['parameters'][k], decoded[k], rel_tol=1e-13, abs_tol=1e-13) for k in decoded):
            raise ValueError('Source coordinate/parameter mismatch: ' + label)
    data, decoded_arrays = {}, []
    with np.load(DATA, allow_pickle=False) as archive:
        for name in PHASES:
            phase = 'reset' if name.startswith('reset') else 'set'
            item = {}
            for key in ('command_voltage_v', 'observed_current_a', 'scoring_mask'):
                full_key = f'{name}_{key}'
                item[key] = archive[full_key].copy();decoded_arrays.append(full_key)
            v, obs, mask = item['command_voltage_v'], item['observed_current_a'], item['scoring_mask']
            original = np.isfinite(abs(obs)) & (abs(obs) >= (1e-11 if phase == 'set' else 1e-12)) & (abs(v) > 1e-12)
            if mask.dtype != np.bool_ or not np.array_equal(mask, original) or (len(v), int(mask.sum())) != ((503,500) if phase == 'set' else (173,170)):
                raise ValueError('Original phase mask/count mismatch: ' + name)
            if v[0] != 0. or v[-1] != 0. or int((mask & (abs(v) <= .2+1e-12)).sum()) != 20:
                raise ValueError('Endpoint/low20 mismatch: ' + name)
            item.update(initial_command_voltage_v=0., nearzero_segment_budget=1e-10, circuit_mode='fast',
                        phase=phase, record_id=int(name.removeprefix(phase)))
            data[name] = item
    kernel = load_joint_kernel()
    if kernel.parameters.lambda_per_v != .4470398752554136 or kernel.ron != 58.463033350725695 or kernel.gc != .03127031375421752:
        raise ValueError('Frozen material settings changed')
    hashes = dict(kernel.source_sha256)
    hashes.update(hash_paths([Path(__file__), HERE/'EXPERIMENT_PLAN.md', OLD_REQUEST, SEED_REQUEST, DATA,
                              PREVIOUS/'joint_forward.py', PREVIOUS/'joint_validation.py']))
    return kernel, data, controls, hashes, decoded_arrays, time.monotonic()-started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-authorized', action='store_true')
    args = parser.parse_args()
    if not args.run_authorized:
        parser.error('Source review and explicit --run-authorized are required')
    if RESULT.exists() or (HERE/'RUN_STARTED.json').exists():
        parser.error('Prior attempt/result exists; no automatic rerun')
    if not (HERE/'EXPERIMENT_PLAN.md').is_file():
        parser.error('Missing frozen EXPERIMENT_PLAN.md')
    writer = Artifacts()
    writer.json(HERE/'RUN_STARTED.json', {'status':'authorized_single_attempt_started', 'configuration':configuration()})
    for name in ('requests','phase_trials'):
        (HERE/name).mkdir(exist_ok=True)
    report = {'status':'loading_inputs','configuration':configuration(),'requests':[],'rounds':[],
              'numerical_validation_passed':False,'modeling_goals_passed':False,
              'full_unified_model_achieved':False,'full_joint_contract_evaluated':False,
              'extension_model_or_scores_loaded':False,'original_holdout_accessed':False,
              'old_model_trajectory_NPZ_read':False,'new_state_inversions':0}
    study = None
    try:
        kernel,data,controls,hashes,decoded,loading = load_inputs()
        report.update(source_sha256=hashes,decoded_input_arrays=decoded,input_loading_and_hash_wall_s=loading,
                      fixed_controls={label:{'source':str(OLD_REQUEST if label=='old' else SEED_REQUEST),
                          'parameters':deepcopy(row['parameters']), 'x_normalized':deepcopy(row['x_normalized']),
                          'previous_three_phase_scores':deepcopy(row['scores'])} for label,row in controls.items()})
        writer.json(HERE/'bounded_coordinate_check.plan.json',{'configuration':configuration(),'source_sha256':hashes})
        study = Study(kernel,data,controls,writer,report)
        study.execute()
        started = time.monotonic()
        report['source_files_unchanged'] = hash_paths(hashes) == hashes
        report['final_source_hash_wall_s'] = time.monotonic()-started
        if not report['source_files_unchanged']:
            report.update(status='source_changed_result_not_validated',numerical_validation_passed=False)
    except Exception as exc:
        report.update(status='stopped_with_preserved_exception',error_type=type(exc).__name__,error=str(exc),
                      numerical_validation_passed=False)
    finally:
        if study is not None:
            report.update(search_requests=study.search_requests,timing=study.budget.snapshot())
        report.update(artifact_catalog=deepcopy(writer.catalog),artifact_IO_wall_s=writer.io_wall_s,
                      terminal_result=True,automatic_retry_permitted=False)
        writer.json(RESULT,report)
        writer.progress('Terminal coordinate diagnostic: '+report['status'])


if __name__ == '__main__':
    main()
