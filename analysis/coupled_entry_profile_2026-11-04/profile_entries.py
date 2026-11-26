"""Three segment-entry q/a profiles; frozen q1 material and original histories.

Prepared implementation only. Import performs no data/table load or model run.
Only three declared segment entries may change; no phase/cycle reanchoring.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import time
import traceback
from types import MappingProxyType

sys.dont_write_bytecode = True
import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
Q1_STAGE = ROOT / 'analysis/coupled_linear_recovery_2026-11-04'
CAL_STAGE = ROOT / 'analysis/coupled_training_calibration_2026-11-04'
PARENT_REPORT = Q1_STAGE / 'rate_controls_run/RESULT.json'
Q1_SHA = 'd5ff9715939315415fda4e772b4673aa7bf30465fc6c19bb7f4d01647d86aa1c'
CAL_SHA = '9352cf8ce4dd2081f8c32658dfd07c254207bc27ac4971cebecbff12f3dad58a'
FRACTIONS = (0., .5, .99)
PREFIX_LIMIT_V = .25
MAX_NFEV = 30
MAX_PREFIX_EVALUATIONS = 900
FIT_BUDGET_S = 300.
TOTAL_BUDGET_S = 600.
FD_RELATIVE_TO_MAX_ONE = 1e-4
DOMAIN_U_GUARD = 1e-7
# Descriptive identifiability diagnostics only: never fitting/selection gates.
NEAR_COST_MSE_DIFFERENCE_DECADE2 = 1e-4
DISTINCT_ENTRY_ABS_LOG_Q = math.log(2.)
DISTINCT_ENTRY_ABS_A_OVER_GREF = .1
FINAL_PRECISIONS = (
    dict(name='nominal', rtol=1e-7, atol_u=1e-9, atol_x=1e-13, max_step=.02),
    dict(name='refined', rtol=1e-9, atol_u=1e-11, atol_x=1e-15, max_step=.01),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


for _path, _expected in ((Q1_STAGE / 'q1_material.py', Q1_SHA),
                         (CAL_STAGE / 'bounded_calibration.py', CAL_SHA)):
    if sha(_path) != _expected:
        raise ValueError(f'Frozen dependency changed: {_path}')
for _directory in (Q1_STAGE, CAL_STAGE):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))
q1 = importlib.import_module('q1_material')
cal = importlib.import_module('bounded_calibration')
tp, fa, artifacts, kernel = cal.tp, cal.fa, cal.artifacts, cal.tp.kernel
if (Path(q1.__file__).resolve() != Q1_STAGE / 'q1_material.py'
        or Path(cal.__file__).resolve() != CAL_STAGE / 'bounded_calibration.py'
        or q1.training is not tp or q1.kernel is not kernel):
    raise RuntimeError('Frozen module identities differ')


def require(condition, message):
    if not condition:
        raise ValueError(message)


class BudgetExceeded(RuntimeError):
    pass


class PrefixLimitExceeded(RuntimeError):
    pass


class NumericalBudget:
    """Optimizer, prefix, verification and final numerics; explicit I/O pauses."""
    def __init__(self):
        self.total = self.fit = 0.
        self.started = self.kind = None

    def total_used(self):
        return self.total + (time.monotonic() - self.started if self.started is not None else 0.)

    def fit_used(self):
        return self.fit + (time.monotonic() - self.started
                           if self.started is not None and self.kind == 'fit' else 0.)

    def can_start(self, kind):
        return self.total_used() < TOTAL_BUDGET_S and (kind != 'fit' or self.fit_used() < FIT_BUDGET_S)

    def check(self):
        if self.total_used() >= TOTAL_BUDGET_S:
            raise BudgetExceeded('Predeclared 600 s total numerical budget exhausted')
        if self.kind == 'fit' and self.fit_used() >= FIT_BUDGET_S:
            raise BudgetExceeded('Predeclared 300 s prefix/profile numerical budget exhausted')

    def _accumulate(self):
        elapsed = time.monotonic() - self.started
        self.total += elapsed
        if self.kind == 'fit':
            self.fit += elapsed
        self.started = None

    @contextmanager
    def section(self, kind):
        if self.kind is not None:
            require(self.kind == kind and self.started is not None, 'Invalid nested numerical section')
            self.check()
            yield
            self.check()
            return
        require(kind in ('fit', 'final'), 'Explicit numerical budget kind required')
        self.kind, self.started = kind, time.monotonic()
        try:
            self.check()
            yield
            self.check()
        finally:
            self._accumulate()
            self.kind = None

    @contextmanager
    def io(self):
        active = self.started is not None
        if active:
            self._accumulate()
        try:
            yield
        finally:
            if active:
                self.started = time.monotonic()


def _parent_and_freeze(freeze_path):
    parent = json.loads(PARENT_REPORT.read_text())
    require(parent.get('status') == 'finite_rate_controls_and_final_checks_finished'
            and parent.get('sources_unchanged') is True
            and parent.get('numerical_final_candidate_verified') is True,
            'Parent q1 run is not terminal and numerically verified')
    require(parent['source_sha256_before'] == parent['source_sha256_after'], 'Parent source guard mismatch')
    selected = parent['selected_coarse_candidate']
    parameters = dict(selected['parameters'])
    entries = {key: tuple(value) for key, value in parent['fixed_segment_entries'].items()}
    require(set(entries) == {'bare1', 'bare5', 'series4'}, 'Three original segment entries required')
    rows = [row for row in parent['requests'] if row['kind'] == 'final']
    require(len(rows) == 2 and all(row['status'] == 'complete_valid'
            and row['parameters'] == parameters for row in rows), 'Parent final candidates differ')
    refined = [row for row in rows if row['request_id'] == 6 and row['precision']['name'] == 'refined']
    require(len(refined) == 1, 'Expected verified q1 refined request 6')
    review = json.loads((Q1_STAGE / 'SAVED_FORWARD_REVIEW.json').read_text())
    require(review.get('execution_selection_and_numerical_review_passed') is True,
            'Independent saved forward numerical review missing')
    freeze = json.loads(freeze_path.read_text())['files_sha256']
    required = dict(parent['source_sha256_before'])

    def include(path):
        path = str(Path(path).resolve())
        digest = sha(path)
        if path in required:
            require(required[path] == digest, 'Original frozen input changed: ' + path)
        required[path] = digest

    for path in (PARENT_REPORT, Q1_STAGE / 'SOURCE_FREEZE.json',
        Q1_STAGE / 'SAVED_FORWARD_REVIEW.json', Q1_STAGE / 'SAVED_FORWARD_REVIEW.md',
        Q1_STAGE / 'EARLY_SET_BOUND.json', Q1_STAGE / 'EARLY_SET_BOUND.md',
        Path(__file__), HERE / 'PROFILE_PLAN.md', HERE / 'STATIC_REVIEW.md',
        HERE / 'ENTRY_INFERENCE_SCOPE.md'):
        include(path)
    for row in rows:
        prefix = Path(row['artifact_prefix']).resolve()
        require(prefix.is_relative_to((Q1_STAGE / 'rate_controls_run').resolve()),
                'Parent final artifact outside original run')
        for path in (prefix.with_suffix('.json'), prefix.with_suffix('.npz'), row['declaration_path']):
            include(path)
    for path, digest in required.items():
        require(freeze.get(path) == digest, 'New root freeze misses exact input: ' + path)
    for path, digest in freeze.items():
        require(Path(path).is_absolute() and sha(path) == digest, 'Frozen input changed: ' + path)
    return parent, parameters, entries, refined[0], dict(freeze)


def _make_prefix(problem, segment):
    name, top, ids = segment
    key = (top, ids[0], 'set')
    full = problem.data[key]
    v = np.asarray(full['command_voltage_v'])
    apex = int(np.argmax(abs(v)))
    indices = np.flatnonzero((np.arange(len(v)) <= apex) & (v > 0) & (v <= PREFIX_LIMIT_V))
    require(len(indices) > 0, 'No original positive SET prefix: ' + name)
    stop = int(indices[-1]) + 1
    require(full['initial_command_voltage_v'] == 0 and v[0] == 0
            and np.all(v[:stop] >= 0) and np.all(np.diff(v[:stop]) >= 0)
            and np.all(v[:stop] <= PREFIX_LIMIT_V), 'Prefix is not the original monotone SET from zero')
    data = dict(full)
    for field, value in tuple(data.items()):
        if isinstance(value, np.ndarray) and value.shape == v.shape:
            data[field] = value[:stop].copy()
    mask = np.asarray(data['scoring_mask'], dtype=bool)
    require(np.all(~mask | ((data['command_voltage_v'] > 0)
            & (data['command_voltage_v'] <= PREFIX_LIMIT_V))) and mask.sum() >= 2,
            'At least two original eligible prefix samples are required')
    j = int(np.flatnonzero(mask)[0])
    gref = float(abs(data['observed_current_a'][j]) / abs(data['command_voltage_v'][j]))
    require(math.isfinite(gref) and gref > 0, 'Invalid original first-sample conductance scale')
    circuit = kernel.BareCircuit(.001) if top == '1r' else kernel.SeriesCircuit()
    return dict(segment=name, key=key, data=data, circuit=circuit, stop=stop, gref=gref,
        first_eligible_native_index=j, n_eligible=int(mask.sum()),
        endpoint_v=float(v[stop - 1]), native_indices=list(range(stop)))


def _precisions(prefix, A):
    atol_x = min(1e-14, 1e-6 * prefix['gref'] / A)
    require(math.isfinite(atol_x) and atol_x > 0, 'Dynamic numerical atol_x is not positive finite')
    fit = dict(name='prefix_fit', rtol=1e-8, atol_u=1e-10, atol_x=atol_x, max_step=.01)
    refined = dict(name='prefix_refined', rtol=1e-10, atol_u=1e-12, atol_x=atol_x / 100., max_step=.005)
    require(refined['atol_x'] > 0, 'Refined numerical atol_x underflow')
    return fit, refined


def _entry_bounds(material, prefix):
    """Conservative prefix gain/loss envelopes inside the unchanged table."""
    p, r = material.parameters, material.readout
    require(p.lambda_per_v >= 0, 'This declared monotone voltage envelope requires lambda >= 0')
    v = prefix['data']['command_voltage_v']
    length = float(np.sum(abs(np.diff(np.r_[0., v]))))
    vmax = float(v[-1])
    # Passive circuits give 0 <= Vh <= Vcommand. For positive SET chi=0.
    # q' >= -Rmax*q; q' <= Cmax*(1-q), no hypothetical timing required.
    current_cap = .001 if prefix['key'][0] == '1r' else float(r.ti_positive[-1])
    loss_budget = p.kappa_positive_per_v2 * vmax * math.exp(p.lambda_per_v * vmax) * length
    capture_budget = p.kappa_positive_per_v2 * vmax * (current_cap / p.J_positive_a) * length
    u_low = max(0., math.log1p(r.gamma_c_ev * math.exp(-float(r.lg[-1]))))
    u_high = min(80., math.log1p(r.gamma_c_ev * math.exp(-float(r.lg[0]))))
    guarded_low, guarded_high = u_low + DOMAIN_U_GUARD, u_high - DOMAIN_U_GUARD
    log_q_low = -guarded_high + loss_budget
    one_minus_q_ceiling = -math.expm1(-guarded_low) * math.exp(capture_budget)
    require(0 < one_minus_q_ceiling < 1, 'No finite positive upper entry bound under capture envelope')
    log_q_high = math.log1p(-one_minus_q_ceiling)
    require(math.isfinite(log_q_low + log_q_high) and log_q_low < log_q_high < 0,
            'No nonempty entry interval within the finite table after prefix budgets')
    upper_a_scaled = p.a_max_s / prefix['gref']
    require(math.isfinite(upper_a_scaled) and upper_a_scaled > 0, 'Invalid scaled a bound')
    # Make the declared parameter bound safe under its own round-trip; this
    # adjusts the search interval, never a solved physical state.
    scaled_a_endpoint_rounded_inward = upper_a_scaled * prefix['gref'] > p.a_max_s
    if scaled_a_endpoint_rounded_inward:
        upper_a_scaled = math.nextafter(upper_a_scaled, 0.)
    require(upper_a_scaled * prefix['gref'] <= p.a_max_s, 'Scaled a bound round-trip exceeds A')
    for lq in (log_q_low, log_q_high):
        material.validate_state((math.exp(lq), 0.))
    lower, upper = np.array([log_q_low, 0.]), np.array([log_q_high, upper_a_scaled])
    details = dict(variables=['natural_log_q0', 'a0_over_gref'], lower=lower, upper=upper,
        table_u_interval=[u_low, u_high], numerical_inward_u_guard=DOMAIN_U_GUARD,
        positive_prefix_loss_budget=loss_budget, positive_prefix_capture_budget=capture_budget,
        actual_current_upper_a=current_cap, prefix_path_length_v=length,
        q0_interval=[math.exp(log_q_low), math.exp(log_q_high)], a0_interval_s=[0., p.a_max_s],
        scaled_a_endpoint_rounded_inward=scaled_a_endpoint_rounded_inward,
        voltage_path_not_physical_time=True, table_domain_extended=False, state_clipping=False,
        envelope_is_conservative_not_interval_arithmetic=True)
    return lower, upper, details


def _state(z, prefix):
    return (math.exp(float(z[0])), float(z[1]) * prefix['gref'])


def _jacobian_diagnostics(jacobian):
    """Saved optimizer arithmetic only; no new derivatives or model calls."""
    jacobian = np.asarray(jacobian, dtype=float)
    require(jacobian.ndim == 2 and jacobian.shape[1] == 2
            and np.all(np.isfinite(jacobian)), 'Invalid returned two-column optimizer Jacobian')
    norms = np.linalg.norm(jacobian, axis=0)
    directions = [jacobian[:, j] / norms[j] if norms[j] > 0 else None for j in range(2)]
    alignment = (float(np.dot(directions[0], directions[1]))
                 if all(value is not None for value in directions) else None)
    _, singular, right_vectors = np.linalg.svd(jacobian, full_matrices=False)
    rank_threshold = float(max(jacobian.shape) * np.finfo(float).eps * singular[0])
    return dict(source='returned_optimizer_jacobian_no_new_model_or_difference_calls',
        coordinates=['natural_log_q0', 'a0_over_gref'], residual_unit='log10 current decade',
        matrix=jacobian, column_norms=norms, normalized_column_directions=directions,
        normalized_column_inner_product=alignment,
        alignment_undefined_reason='at_least_one_zero_sensitivity_column' if alignment is None else None,
        alignment_is_statistical_correlation=False, singular_values=singular,
        svd_right_vectors_rows=right_vectors,
        svd_right_vector_sign_is_arbitrary=True,
        singular_value_ratio_small_over_large=float(singular[-1] / singular[0]) if singular[0] > 0 else None,
        numerical_rank=int(np.sum(singular > rank_threshold)), numerical_rank_threshold=rank_threshold,
        rank_and_condition_depend_on_declared_coordinate_scaling=True,
        statistical_confidence_or_identifiability_established=False)


def _ending_pair_diagnostics(endings, n_residuals, gref):
    """Compare already qualified endings without changing minimum-cost ranking."""
    pairs = []
    for left_index, left in enumerate(endings):
        for right in endings[left_index + 1:]:
            lq, la = map(float, left['ending_state'])
            rq, ra = map(float, right['ending_state'])
            dcost = float(right['selection_cost'] - left['selection_cost'])
            dlogq = math.log(rq) - math.log(lq)
            dascaled = (ra - la) / gref
            dmse = 2. * dcost / n_residuals
            near = abs(dmse) <= NEAR_COST_MSE_DIFFERENCE_DECADE2
            distinct = (abs(dlogq) >= DISTINCT_ENTRY_ABS_LOG_Q
                        or abs(dascaled) >= DISTINCT_ENTRY_ABS_A_OVER_GREF)
            pairs.append(dict(left_start_id=left['start_id'], right_start_id=right['start_id'],
                left_state_q_a_s=[lq, la], right_state_q_a_s=[rq, ra],
                left_cost=left['selection_cost'], right_cost=right['selection_cost'],
                delta_cost_right_minus_left=dcost, absolute_cost_difference=abs(dcost),
                delta_mean_squared_log_residual_decade2=dmse,
                absolute_mean_squared_log_residual_difference_decade2=abs(dmse),
                delta_ln_q_right_minus_left=dlogq, delta_a_s_right_minus_left=ra - la,
                delta_a_over_gref_right_minus_left=dascaled,
                near_equal_cost_descriptive=near, distinct_entries_descriptive=distinct,
                near_equal_cost_distinct_alternative=bool(near and distinct)))
    return dict(n_qualified_ending_points=len(endings), n_pairs=len(pairs), pairs=pairs,
        multistart_degeneracy_check_available=len(endings) >= 2,
        unavailable_reason='fewer_than_two_qualified_endings' if len(endings) < 2 else None,
        near_cost_definition='absolute difference of 2*prefix_cost/n_eligible <= threshold',
        near_cost_mse_difference_threshold_decade2=NEAR_COST_MSE_DIFFERENCE_DECADE2,
        distinct_entry_definition='abs(delta_ln_q)>=ln(2) OR abs(delta_a/gref)>=0.1',
        distinct_entry_abs_log_q_threshold=DISTINCT_ENTRY_ABS_LOG_Q,
        distinct_entry_abs_a_over_gref_threshold=DISTINCT_ENTRY_ABS_A_OVER_GREF,
        thresholds_are_descriptive_not_statistical_confidence_bands=True,
        used_for_selection_or_acceptance=False,
        near_cost_distinct_alternatives_present=any(row['near_equal_cost_distinct_alternative'] for row in pairs),
        entry_identifiability_remains_unresolved=True,
        absence_of_flagged_alternatives_does_not_prove_unique_entry=True)


def _verify_prefix(result, material, prefix, state, budget, *, alternate_roots):
    a = result['arrays']
    rows = []

    def check(name, passed, **values):
        rows.append(dict(name=name, passed=bool(passed), **values))

    check('complete_prefix', result['summary']['status'] == 'complete'
          and len(result['residual']) == prefix['n_eligible'])
    check('original_prefix_commands', np.array_equal(a['command_voltage_v'], prefix['data']['command_voltage_v']))
    check('original_prefix_mask', np.array_equal(a['original_scoring_mask'], prefix['data']['scoring_mask']))
    check('requested_entry', (result['summary']['initial_state_q'], result['summary']['initial_state_a_s']) == state)
    q, u, hard_a = a['state_q'], a['state_u'], a['state_a_s']
    valid = bool(np.all(np.isfinite(q + u + hard_a))
        and np.all((q > 0) & (q < 1) & (u > 0) & (u <= 80)
                   & (hard_a >= 0) & (hard_a <= material.parameters.a_max_s)))
    check('finite_state_domain_current_A', valid)
    if valid:
        lg = np.log(material.readout.gamma_c_ev / np.expm1(u))
        check('frozen_Gamma_table', np.all(lg >= material.readout.lg[0]) and np.all(lg <= material.readout.lg[-1]))
        check('q_u_identity', np.allclose(q, np.exp(-u), rtol=1e-13, atol=0.))
    scale = 1e-18 + 1e-7 * abs(a['current_a'])
    check('KCL', np.all(abs(a['KCL_residual_a']) <= scale))
    check('KVL', np.all(abs(a['KVL_residual_v']) <= 1e-9))
    check('passivity', all(np.all(np.isfinite(a[key])) and np.all(a[key] * a['local_voltage_v'] >= 0)
          for key in ('current_a', 'soft_current_a', 'hard_current_a')))
    check('total_branch_sum', np.all(abs(a['current_a'] - a['soft_current_a'] - a['hard_current_a']) <= scale))
    check('same_a_in_readout', np.all(abs(a['hard_current_a'] - hard_a * a['local_voltage_v']) <= scale))
    # Explicitly audit the same passive circuit bounds used by _entry_bounds.
    # Reuse the existing native KVL/KCL tolerances; no new model evaluation.
    current_cap = .001 if prefix['key'][0] == '1r' else float(material.readout.ti_positive[-1])
    check('prefix_local_voltage_between_zero_and_command',
          np.all(a['local_voltage_v'] >= -1e-9)
          and np.all(a['local_voltage_v'] <= a['command_voltage_v'] + 1e-9),
          minimum_local_voltage_v=float(np.min(a['local_voltage_v'])),
          maximum_local_minus_command_v=float(np.max(a['local_voltage_v'] - a['command_voltage_v'])),
          tolerance_v=1e-9)
    check('prefix_nonnegative_current_within_declared_circuit_cap',
          np.all(a['current_a'] >= -scale) and np.all(a['current_a'] <= current_cap + scale),
          current_cap_a=current_cap, minimum_current_a=float(np.min(a['current_a'])),
          maximum_current_a=float(np.max(a['current_a'])),
          tolerance_definition='1e-18 + 1e-7*abs(current_a)')
    if prefix['key'][0] == '1r':
        check('actual_bare_1mA_compliance', np.all(abs(a['current_a']) <= .001 + scale))
    roots = 0
    if alternate_roots:
        for j in (prefix['first_eligible_native_index'], len(q) - 1):
            budget.check()
            point = prefix['circuit'].solve(float(a['command_voltage_v'][j]), material,
                (float(q[j]), float(hard_a[j])), reference=True)
            di = abs(point['current_a'] - a['current_a'][j])
            dv = max(abs(point[key] - a[key][j]) for key in
                     ('local_voltage_v', 'transistor_voltage_v', 'compliance_voltage_v'))
            check('alternate_root', di <= scale[j] and dv <= 1e-9, index=j, delta_i_a=di, delta_v_v=dv)
            roots += 1
    return dict(passed=all(row['passed'] for row in rows), checks=rows, alternate_root_checks=roots)


def _compare_prefix(left, right, A):
    a, b = left['arrays'], right['arrays']
    mask = a['original_scoring_mask']
    require(np.array_equal(mask, b['original_scoring_mask'])
            and np.array_equal(a['command_voltage_v'], b['command_voltage_v']),
            'Prefix comparison changed original commands/mask')
    di = float(np.max(abs(np.log10(abs(a['current_a'][mask])) - np.log10(abs(b['current_a'][mask])))))
    du = float(np.max(abs(a['state_u'] - b['state_u'])))
    dx = float(np.max(abs(a['state_a_s'] - b['state_a_s'])) / A)
    return dict(passed=di <= 3e-4 and du <= 1e-3 and dx <= 1e-5,
                max_logI_decade=di, max_abs_u=du, max_abs_a_over_A=dx, current_A_s=A)


def run(freeze_path, output):
    parent, parameters, old_entries, parent_refined, frozen = _parent_and_freeze(freeze_path)
    freeze_digest = sha(freeze_path)
    output.mkdir(exist_ok=False)
    prefix_dir, start_dir, final_dir = output / 'prefix_calls', output / 'starts', output / 'final'
    for directory in (prefix_dir, start_dir, final_dir):
        directory.mkdir()
    budget = NumericalBudget()
    report = dict(status='preparing', fixed_recovery_order=1, fixed_material_parameters=parameters,
        original_entries={key: list(value) for key, value in old_entries.items()},
        source_sha256_before=frozen, freeze_sha256_before=freeze_digest,
        parent_refined_request_id=6, old_entry_control_artifact_prefix=parent_refined['artifact_prefix'],
        fixed_fractions=FRACTIONS, max_nfev_per_start=MAX_NFEV, max_prefix_evaluations=MAX_PREFIX_EVALUATIONS,
        prefix_fit_budget_s=FIT_BUDGET_S, total_numerical_budget_s=TOTAL_BUDGET_S,
        maximum_final_full_evaluations=2, maximum_final_phases=184, final_precisions=FINAL_PRECISIONS,
        no_holdout_access=True, prefix_points_are_calibration_data=True,
        only_segment_entries_estimated=True, n_segment_entries=3, n_estimated_entry_coordinates=6,
        no_cycle_or_phase_reanchoring=True,
        no_new_physical_parameter=True, no_parameter_penalty_on_failure=True,
        no_state_clipping=True, artifact_IO_and_initial_loading_excluded=True,
        prefix_calls=[], n_actual_prefix_evaluations=0, baseline_regressions=[],
        segments={}, starts=[], final_requests=[], final_comparison=None)

    def write(path, value):
        with budget.io():
            artifacts.write(path, value)

    def persist():
        report.update(prefix_fit_numerical_wall_s=budget.fit_used(), total_numerical_wall_s=budget.total_used())
        write(output / 'RESULT.json', report)

    def save_prefix(call, phase_result, verification):
        name = f"{call['call_id']:04d}_{call['segment']}_{call['role']}"
        call['artifact_prefix'] = str(prefix_dir / name)
        with budget.io():
            arrays = {} if phase_result is None else dict(phase_result['arrays'], residual=phase_result['residual'])
            np.savez_compressed(prefix_dir / (name + '.npz'), **arrays)
            artifacts.write(prefix_dir / (name + '.json'), dict(call=call,
                phase_summary=None if phase_result is None else phase_result['summary'],
                numerical_verification=verification))

    persist()
    selected_entries = dict(old_entries)
    final_results = {}
    try:
        problem = tp.load_training_problem()
        contract = fa.load_feature_contract()
        report['data_contract'] = problem.checkpoint_metadata()
        control = json.loads(Path(parent_refined['artifact_prefix']).with_suffix('.json').read_text())
        require(control['summary']['status'] == 'complete_original_training'
                and control['summary']['n_completed_phases'] == 92,
                'Saved old-entry control is incomplete')
        report['old_entry_control'] = dict(artifact_prefix=parent_refined['artifact_prefix'],
            score=control['score'], gates=cal._gate_summary(control['features']),
            prior_refined_precision=parent_refined['precision'],
            retained_as_frozen_full_native_control=True, new_control_full_ODE_run=False)
        write(output / 'CONTROL_REFERENCE.json', report['old_entry_control'])
        material = q1.load_q1_material(kernel.Parameters(**parameters, enable_a=True),
            candidate_source_id=str(PARENT_REPORT), candidate_source_sha256=sha(PARENT_REPORT))
        for path, digest in material.source_sha256.items():
            require(frozen.get(path) == digest, 'Run freeze misses material input: ' + path)
        material = replace(material, source_sha256=MappingProxyType(dict(frozen)))
        require(material.metadata()['fixed_recovery_order'] == 1, 'Fixed q1 material changed order')
        prefixes = [_make_prefix(problem, segment) for segment in tp.SEGMENTS]
        # Decode only the three saved first-SET prefixes, not any heldout record.
        saved_prefixes = {}
        with np.load(Path(parent_refined['artifact_prefix']).with_suffix('.npz'), allow_pickle=False) as archive:
            for prefix in prefixes:
                top, cid, phase = prefix['key']
                key = f'{top}_cycle_{cid}_{phase}__'
                saved_prefixes[prefix['segment']] = dict(arrays={name: archive[key + name][:prefix['stop']].copy()
                    for name in ('command_voltage_v', 'original_scoring_mask', 'current_a', 'state_u', 'state_a_s')})

        def evaluate_prefix(prefix, state, precision, role, start_id=None, *, alternate_roots=False):
            call = dict(call_id=len(report['prefix_calls']) + 1, segment=prefix['segment'], role=role,
                start_id=start_id, initial_state_q=state[0], initial_state_a_s=state[1],
                precision=precision, status='declared', model_evaluation_performed=False)
            report['prefix_calls'].append(call)
            result = verification = None
            try:
                if report['n_actual_prefix_evaluations'] >= MAX_PREFIX_EVALUATIONS:
                    raise PrefixLimitExceeded('Predeclared 900 actual prefix evaluations exhausted')
                if not budget.can_start('fit'):
                    raise BudgetExceeded('No prefix fitting numerical budget remains')
                with budget.section('fit'):
                    material.validate_state(state)
                    call['model_evaluation_performed'] = True
                    report['n_actual_prefix_evaluations'] += 1
                    result = kernel.simulate_phase(material, prefix['circuit'], prefix['data'], state,
                        kernel.Precision(**precision), 'Radau', budget.check, collect_rhs=False)
                    verification = _verify_prefix(result, material, prefix, state, budget,
                                                  alternate_roots=alternate_roots)
                    require(verification['passed'], 'Prefix native numerical checks failed')
                    cost = .5 * float(np.dot(result['residual'], result['residual']))
                    require(math.isfinite(cost), 'Nonfinite original prefix log objective')
                    call.update(status='complete_valid', cost=cost,
                        rmse_decade=float(np.sqrt(np.mean(result['residual'] ** 2))),
                        n_residuals=len(result['residual']))
            except Exception as exc:
                if isinstance(exc, kernel.PhaseFailure):
                    result = dict(summary=exc.summary, arrays=exc.arrays, residual=np.empty(0))
                call.update(status='retained_failure', failure_type=type(exc).__name__, failure=str(exc),
                    traceback=traceback.format_exc(), no_finite_penalty_returned=True)
                save_prefix(call, result, verification)
                persist()
                raise
            save_prefix(call, result, verification)
            return result, call

        # All three regressions precede any fit; failure prevents all fitting/final runs.
        for prefix in prefixes:
            name = prefix['segment']
            fit_precision, refined_precision = _precisions(prefix, parameters['a_max_s'])
            report['segments'][name] = dict(key=prefix['key'], prefix_native_indices=prefix['native_indices'],
                endpoint_command_v=prefix['endpoint_v'], n_eligible=prefix['n_eligible'], gref_s=prefix['gref'],
                first_eligible_native_index=prefix['first_eligible_native_index'],
                fit_precision=fit_precision, refined_precision=refined_precision,
                original_entry=old_entries[name], selected_entry=old_entries[name],
                entry_source='old_entry_fallback_until_a_valid_fit_is_selected',
                ending_point_identifiability_diagnostics=_ending_pair_diagnostics(
                    [], prefix['n_eligible'], prefix['gref']))
            regression = dict(segment=name, passed=False)
            try:
                with budget.section('fit'):
                    result, call = evaluate_prefix(prefix, old_entries[name], refined_precision,
                        'old_entry_regression', alternate_roots=True)
                    comparison = _compare_prefix(result, saved_prefixes[name], parameters['a_max_s'])
                    regression.update(passed=comparison['passed'], call_id=call['call_id'], comparison=comparison)
            except Exception as exc:
                regression.update(failure_type=type(exc).__name__, failure=str(exc))
            report['baseline_regressions'].append(regression)
            persist()
        report['all_prefix_regressions_passed'] = (len(report['baseline_regressions']) == 3
            and all(row['passed'] for row in report['baseline_regressions']))
        if not report['all_prefix_regressions_passed']:
            raise RuntimeError('Old-entry prefix regression failed; no fit or full forward evaluation authorized')

        for prefix in prefixes:
            name = prefix['segment']
            segment_report = report['segments'][name]
            fit_precision, refined_precision = _precisions(prefix, parameters['a_max_s'])
            try:
                with budget.section('fit'):
                    lower, upper, bound_details = _entry_bounds(material, prefix)
                segment_report['entry_bounds'] = bound_details
            except Exception as exc:
                segment_report.update(entry_source=('old_entry_no_profile_budget' if isinstance(exc, BudgetExceeded)
                    else 'old_entry_no_valid_search_bounds'),
                    bounds_failure_type=type(exc).__name__, bounds_failure=str(exc))
                persist()
                continue
            valid_ends = []
            for fraction in FRACTIONS:
                start_id = len(report['starts']) + 1
                start = dict(start_id=start_id, segment=name, fraction=fraction,
                    status='declared', eligible_for_entry_selection=False,
                    fraction_is_physical_current_decomposition=False,
                    objective_calls=0, jacobian_perturbation_calls=0, jacobian_base_extra_calls=0,
                    exact_base_residual_reuse=0, finite_difference_steps=[])
                report['starts'].append(start)
                proposed_state = (old_entries[name][0] * (1 - fraction), fraction * prefix['gref'])
                proposed_z = np.array([math.log(proposed_state[0]), proposed_state[1] / prefix['gref']])
                x0 = np.minimum(upper, np.maximum(lower, proposed_z))
                start.update(proposed_initial_state=proposed_state, proposed_z=proposed_z,
                    projected_z=x0, projected_initial_state=_state(x0, prefix),
                    initial_parameter_projection_used=bool(np.any(x0 != proposed_z)),
                    lower=lower, upper=upper, projection_applies_to_initial_guess_only=True)
                write(start_dir / f'{start_id:02d}_{name}_declaration.json', start)
                if not budget.can_start('fit') or report['n_actual_prefix_evaluations'] >= MAX_PREFIX_EVALUATIONS:
                    start['status'] = 'not_run_prefix_budget_or_count_limit'
                    write(start_dir / f'{start_id:02d}_{name}_result.json', start)
                    persist()
                    continue
                cached_z = cached_residual = None

                def residual_at(z, role):
                    z = np.asarray(z, dtype=float)
                    require(z.shape == (2,) and np.all(np.isfinite(z))
                            and np.all(z >= lower) and np.all(z <= upper),
                            'Optimizer parameter trial outside declared entry bounds')
                    if 'first_evaluated_z' not in start:
                        start['first_evaluated_z'] = z.copy()
                        start['first_evaluated_state'] = _state(z, prefix)
                        start['optimizer_initial_interior_adjustment'] = bool(np.any(z != x0))
                    result, _ = evaluate_prefix(prefix, _state(z, prefix), fit_precision, role, start_id)
                    return result['residual'].copy()

                def objective(z):
                    nonlocal cached_z, cached_residual
                    start['objective_calls'] += 1
                    value = residual_at(z, 'optimizer_objective')
                    cached_z, cached_residual = np.asarray(z).copy(), value.copy()
                    return value

                def jacobian(z):
                    nonlocal cached_z, cached_residual
                    z = np.asarray(z, dtype=float)
                    budget.check()
                    if cached_z is not None and np.array_equal(z, cached_z):
                        base_residual = cached_residual.copy()
                        start['exact_base_residual_reuse'] += 1
                    else:
                        start['jacobian_base_extra_calls'] += 1
                        base_residual = residual_at(z, 'jacobian_base')
                    columns = []
                    for axis in range(2):
                        desired = FD_RELATIVE_TO_MAX_ONE * max(1., abs(float(z[axis])))
                        plus, minus = float(upper[axis] - z[axis]), float(z[axis] - lower[axis])
                        if plus >= desired:
                            delta = desired
                        elif minus >= desired:
                            delta = -desired
                        elif plus >= minus and plus > 0:
                            delta = .5 * plus
                        elif minus > 0:
                            delta = -.5 * minus
                        else:
                            raise ValueError('No feasible nonzero finite-difference parameter step')
                        trial = z.copy()
                        trial[axis] += delta
                        actual = float(trial[axis] - z[axis])
                        require(actual != 0, 'Finite-difference parameter step rounded to zero')
                        start['jacobian_perturbation_calls'] += 1
                        start['finite_difference_steps'].append(dict(axis=axis, base_z=z.copy(),
                            requested_abs_step=desired, actual_step=actual))
                        shifted = residual_at(trial, 'jacobian_perturbation')
                        columns.append((shifted - base_residual) / actual)
                    result = np.column_stack(columns)
                    require(np.all(np.isfinite(result)), 'Nonfinite finite-difference Jacobian')
                    return result

                try:
                    with budget.section('fit'):
                        optimizer = least_squares(objective, x0, jac=jacobian, bounds=(lower, upper),
                            method='trf', loss='linear', x_scale=np.ones(2),
                            max_nfev=MAX_NFEV, ftol=1e-6, xtol=1e-6, gtol=1e-6)
                        start.update(optimizer_status=int(optimizer.status), optimizer_success=bool(optimizer.success),
                            optimizer_message=str(optimizer.message), scipy_nfev=int(optimizer.nfev),
                            scipy_njev=int(optimizer.njev), optimizer_cost=float(optimizer.cost),
                            optimizer_optimality=float(optimizer.optimality), optimizer_active_mask=optimizer.active_mask,
                            ending_z=optimizer.x, ending_state=_state(optimizer.x, prefix),
                            converged_claim=False, max_nfev_end_is_not_convergence=optimizer.status == 0)
                        nominal, ncall = evaluate_prefix(prefix, _state(optimizer.x, prefix), fit_precision,
                            'endpoint_fit_precision', start_id, alternate_roots=True)
                        refined, rcall = evaluate_prefix(prefix, _state(optimizer.x, prefix), refined_precision,
                            'endpoint_refined', start_id, alternate_roots=True)
                        comparison = _compare_prefix(nominal, refined, parameters['a_max_s'])
                        start.update(endpoint_comparison=comparison, endpoint_call_ids=[ncall['call_id'], rcall['call_id']],
                            selection_cost=rcall['cost'], selection_rmse_decade=rcall['rmse_decade'])
                        jacobian_report = _jacobian_diagnostics(optimizer.jac)
                        start['jacobian_diagnostics'] = jacobian_report
                        start['jacobian_singular_values'] = jacobian_report['singular_values']
                        start['jacobian_condition_ratio_small_over_large'] = (
                            jacobian_report['singular_value_ratio_small_over_large'])
                        start['identifiability_established'] = False
                        require(comparison['passed'], 'Ending entry fails independent prefix precision comparison')
                    start.update(status='valid_numerically_checked_ending_point', eligible_for_entry_selection=True)
                    valid_ends.append(start)
                except Exception as exc:
                    start.update(status='start_terminated_with_retained_failure', eligible_for_entry_selection=False,
                        failure_type=type(exc).__name__, failure=str(exc), traceback=traceback.format_exc(),
                        best_intermediate_point_not_selected=True, no_finite_penalty_returned=True)
                write(start_dir / f'{start_id:02d}_{name}_result.json', start)
                persist()
            segment_report['ending_point_identifiability_diagnostics'] = _ending_pair_diagnostics(
                valid_ends, prefix['n_eligible'], prefix['gref'])
            if valid_ends:
                best = min(valid_ends, key=lambda row: (row['selection_cost'], row['start_id']))
                selected_entries[name] = tuple(best['ending_state'])
                segment_report.update(entry_source='minimum_refined_prefix_cost_among_valid_ending_points',
                    selected_start_id=best['start_id'], selected_entry=selected_entries[name],
                    selected_prefix_cost=best['selection_cost'], selected_prefix_rmse_decade=best['selection_rmse_decade'])
            else:
                segment_report.update(entry_source='old_entry_no_valid_fitted_ending_point',
                    selected_entry=old_entries[name], fallback_is_not_fit_success=True)
            persist()

        report['selected_entries'] = {key: list(value) for key, value in selected_entries.items()}
        entry_file = output / 'SELECTED_ENTRIES.json'
        write(entry_file, dict(fixed_material_parameters=parameters, fixed_recovery_order=1,
            initial_states=report['selected_entries'], segment_choices=report['segments'],
            only_three_segment_entries=True, prefix_points_are_calibration_data=True))
        final_material = replace(material, candidate_source_id=str(entry_file), candidate_source_sha256=sha(entry_file))
        for number, precision in enumerate(FINAL_PRECISIONS, start=1):
            row = dict(request_id=number, precision=precision, status='declared', eligible=False,
                initial_states=report['selected_entries'], material_parameters=parameters,
                full_evaluation_performed=False, starts_again_at_command_zero=True)
            report['final_requests'].append(row)
            if not budget.can_start('final'):
                row['status'] = 'not_run_total_budget'
                persist()
                continue
            result = gates = verification = None
            label = 'entry_profile_' + precision['name']
            row['artifact_prefix'] = str(final_dir / label)
            try:
                with budget.section('final'):
                    row['full_evaluation_performed'] = True
                    result = problem.evaluate(final_material, selected_entries, kernel.Precision(**precision),
                                              'Radau', budget.check, collect_rhs=False)
                    gates = fa.evaluate_features(problem, result['predictions'], contract)
                    verification = cal.verify_native(result, final_material, budget, alternate_roots=True)
                    require(gates['n_primary'] == 20, 'Final run omitted original feature gates')
                    require(verification['passed'], 'Final original-history native checks failed')
                row.update(status='complete_valid', eligible=True)
                final_results[precision['name']] = result
            except Exception as exc:
                if hasattr(exc, 'result'):
                    result = exc.result
                row.update(status='retained_failure', failure_type=type(exc).__name__,
                    failure=str(exc), traceback=traceback.format_exc(), no_finite_penalty_returned=True)
            if result is not None:
                with budget.io():
                    artifacts.save_case(final_dir, label, result, gates, verification)
                row['summary'] = result['summary']
                if result['score'] is not None:
                    row['score'] = result['score']
                if gates is not None:
                    row['gates'] = cal._gate_summary(gates)
                if verification is not None:
                    row['numerical_verification_passed'] = verification['passed']
                    row['alternate_root_checks'] = verification['alternate_root_checks']
            write(final_dir / (label + '_request.json'), row)
            persist()
        if len(final_results) == 2 and budget.can_start('final'):
            with budget.section('final'):
                report['final_comparison'] = cal.compare_final(final_results['nominal'], final_results['refined'],
                                                               parameters['a_max_s'])
                new_score, old_score = final_results['refined']['score'], control['score']
                report['vs_saved_old_entry_control'] = dict(
                    original_cost_before=old_score['verified_cost'], original_cost_after=new_score['verified_cost'],
                    original_cost_change=new_score['verified_cost'] - old_score['verified_cost'],
                    phase_rmse_decade_before={key: math.sqrt(value['weighted_total_mse'])
                        for key, value in old_score['blocks'].items()},
                    phase_rmse_decade_after={key: math.sqrt(value['weighted_total_mse'])
                        for key, value in new_score['blocks'].items()},
                    comparison_uses_same_original_training_and_weights=True,
                    exact_full_run_tolerances_differ_and_are_recorded=True,
                    prefix_is_calibration_not_holdout=True)
        report['status'] = 'bounded_entry_profile_and_full_history_checks_finished'
    except Exception as exc:
        report.update(status='stopped_with_retained_failure', failure_type=type(exc).__name__,
                      failure=str(exc), traceback=traceback.format_exc())
    finally:
        after, errors = {}, {}
        for path in frozen:
            try:
                after[path] = sha(path)
            except Exception as exc:
                after[path] = None
                errors[path] = f'{type(exc).__name__}: {exc}'
        try:
            after_freeze = sha(freeze_path)
        except Exception:
            after_freeze = None
        report.update(source_sha256_after=after, source_guard_errors=errors, freeze_sha256_after=after_freeze,
            sources_unchanged=after == frozen and after_freeze == freeze_digest)
        report['prefix_call_count_contract_respected'] = report['n_actual_prefix_evaluations'] <= MAX_PREFIX_EVALUATIONS
        report['n_final_full_evaluations'] = sum(row['full_evaluation_performed'] for row in report['final_requests'])
        report['n_final_completed_phases'] = sum(row.get('summary', {}).get('n_completed_phases', 0)
                                               for row in report['final_requests'])
        report['full_evaluation_count_contract_respected'] = (report['n_final_full_evaluations'] <= 2
            and report['n_final_completed_phases'] <= 184)
        report['n_fitted_segment_entries_selected'] = sum(
            row.get('entry_source') == 'minimum_refined_prefix_cost_among_valid_ending_points'
            for row in report['segments'].values())
        report['numerical_final_candidate_verified'] = (report['sources_unchanged']
            and report.get('all_prefix_regressions_passed', False)
            and report['prefix_call_count_contract_respected'] and report['full_evaluation_count_contract_respected']
            and len(report['final_requests']) == 2 and all(row['eligible']
            and row.get('alternate_root_checks') == 184 for row in report['final_requests'])
            and bool(report['final_comparison']) and report['final_comparison']['passed'])
        report.update(full_material_optimization_performed=False, microscopic_mechanism_identified=False,
            initial_states_uniquely_identified=False, heldout_prediction_evaluated=False,
            prefix_fit_is_extrapolation=False, gate_count_is_accuracy=False,
            numerical_budget_automatically_extended=False, tolerances_automatically_relaxed=False)
        persist()
        print(json.dumps({key: report[key] for key in ('status', 'n_actual_prefix_evaluations',
            'n_fitted_segment_entries_selected', 'n_final_full_evaluations',
            'numerical_final_candidate_verified', 'total_numerical_wall_s')}), flush=True)
    return 0 if report['numerical_final_candidate_verified'] else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, default=HERE / 'SOURCE_FREEZE.json')
    parser.add_argument('--output', type=Path, default=HERE / 'entry_profile_run')
    args = parser.parse_args(argv)
    return run(args.freeze.resolve(), args.output.resolve())


if __name__ == '__main__':
    raise SystemExit(main())
