"""At most four predeclared q1 controls on the original complete training task.

Preparation only: import loads no observations/tables and runs no experiment.
The first case preserves all selected q2 numeric parameters. Remaining cases
must be explicitly finalized by the root agent after the capacity audit;
this driver does not generate or adapt their parameters. Entries stay fixed.
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
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CALIBRATION = ROOT / 'analysis/coupled_training_calibration_2026-11-04'
PARENT_REPORT = CALIBRATION / 'calibration_run/RESULT.json'
SEED_PATH = HERE / 'SEED_DERIVATION.json'
CALIBRATION_SHA256 = '9352cf8ce4dd2081f8c32658dfd07c254207bc27ac4971cebecbff12f3dad58a'
Q1_SHA256 = 'd5ff9715939315415fda4e772b4673aa7bf30465fc6c19bb7f4d01647d86aa1c'
PARAMETER_NAMES = ('lambda_per_v', 'kappa_positive_per_v2', 'J_positive_a',
                   'b', 'a_max_s', 'kappa_a_per_v2', 'rho')
RATE_NAME = 'kappa_positive_per_v2'
SEARCH_LIMIT_S = 600.
TOTAL_LIMIT_S = 900.
MAX_EVALUATIONS = 6
MAX_PHASE_CALLS = 552
COARSE = dict(name='coarse', rtol=1e-6, atol_u=1e-8, atol_x=1e-10, max_step=.05)
FINAL_PRECISIONS = (
    dict(name='nominal', rtol=1e-7, atol_u=1e-9, atol_x=1e-11, max_step=.02),
    dict(name='refined', rtol=1e-9, atol_u=1e-11, atol_x=1e-13, max_step=.01),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


for _path, _digest in ((CALIBRATION / 'bounded_calibration.py', CALIBRATION_SHA256),
                       (HERE / 'q1_material.py', Q1_SHA256)):
    if sha(_path) != _digest:
        raise ValueError(f'Prepared dependency changed: {_path}')
if str(CALIBRATION) not in sys.path:
    sys.path.insert(0, str(CALIBRATION))
cal = importlib.import_module('bounded_calibration')
import q1_material as q1

tp, fa, artifacts = cal.tp, cal.fa, cal.artifacts
if (Path(cal.__file__).resolve() != (CALIBRATION / 'bounded_calibration.py').resolve()
        or Path(q1.__file__).resolve() != (HERE / 'q1_material.py').resolve()
        or tp is not q1.training or tp.kernel is not q1.kernel):
    raise RuntimeError('Prepared modules do not share the exact frozen kernel identity')


class BudgetExceeded(RuntimeError):
    pass


class NumericalBudget:
    """Count numerical work only; no extension or reset between candidates."""
    def __init__(self):
        self.total = self.search = 0.
        self.started = self.kind = None

    def total_used(self):
        return self.total + (time.monotonic() - self.started if self.started is not None else 0.)

    def search_used(self):
        return self.search + (time.monotonic() - self.started if self.kind == 'search' else 0.)

    def can_start(self, kind):
        return self.total_used() < TOTAL_LIMIT_S and (kind != 'search' or self.search_used() < SEARCH_LIMIT_S)

    def check(self):
        if self.total_used() >= TOTAL_LIMIT_S:
            raise BudgetExceeded('Predeclared 900 s total numerical budget exhausted')
        if self.kind == 'search' and self.search_used() >= SEARCH_LIMIT_S:
            raise BudgetExceeded('Predeclared 600 s coarse-control budget exhausted')

    @contextmanager
    def section(self, kind):
        if self.started is not None or kind not in ('search', 'final'):
            raise RuntimeError('Numerical sections must be explicit and nonnested')
        self.kind, self.started = kind, time.monotonic()
        try:
            self.check()
            yield
            self.check()
        finally:
            elapsed = time.monotonic() - self.started
            self.total += elapsed
            if kind == 'search':
                self.search += elapsed
            self.kind = self.started = None


def budget_cause(exc):
    seen = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, BudgetExceeded):
            return True
        seen.add(id(exc))
        exc = exc.__cause__ if exc.__cause__ is not None else exc.__context__
    return False


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _load_declarations(declaration_path):
    parent = json.loads(PARENT_REPORT.read_text())
    require(parent.get('status') == 'bounded_search_and_final_checks_finished'
            and parent.get('sources_unchanged') is True
            and parent.get('numerical_final_candidate_verified') is True
            and parent.get('n_completed_rounds') == 4,
            'Parent q2 calibration is not the verified terminal four-round run')
    center = parent['last_accepted_center']
    require(center['request_id'] == 26, 'Expected the predeclared q2 selected request 26')
    parameters = {key: float(value) for key, value in center['parameters'].items()}
    require(set(parameters) == set(PARAMETER_NAMES), 'Parent parameter keys changed')
    require(parameters[RATE_NAME] == 57.67798793605963, 'Parent same-number kappa changed')
    entries = {key: tuple(map(float, value)) for key, value in parent['fixed_segment_entries'].items()}
    require(entries == artifacts.ENTRIES, 'Original three segment entries changed')
    selected = [row for row in parent['requests'] if row['request_id'] == 26]
    require(len(selected) == 1 and selected[0]['eligible_for_selection']
            and selected[0]['parameters'] == parameters, 'Selected q2 source inconsistent')
    final = [row for row in parent['requests'] if row['kind'] == 'final']
    require(len(final) == 2 and {row['precision']['name'] for row in final} == {'nominal', 'refined'}
            and all(row['status'] == 'complete_valid' and row['eligible_for_selection']
                    and row['parameters'] == parameters for row in final),
            'Parent final verification did not use the same selected parameters')
    seed = json.loads(SEED_PATH.read_text())
    require(seed.get('status') == 'completed_reference_seed_declaration_no_forward_evaluation',
            'Analytic q1 seed declaration is not complete')
    reference = float(seed['q1_kappa_reference'])
    require(math.isfinite(reference) and reference > 0
            and reference == float(seed['kappa1_reference_per_v2']), 'Invalid q1 kappa reference')
    seed_parameters = dict(parameters, **{RATE_NAME: reference})
    require(seed['seed_parameters'] == seed_parameters, 'Seed changed parameters beyond kappa')
    require({key: tuple(value) for key, value in seed['initial_states'].items()} == entries,
            'Seed changed original segment entries')
    # The earlier rate-only seed proposals are NOT an execution declaration.
    # A root-authored final declaration is required after the capacity audit.
    declaration = json.loads(declaration_path.read_text())
    require(declaration.get('status') == 'final_predeclared_q1_controls_no_forward_evaluation'
            and declaration.get('parent_selected_request_id') == 26
            and declaration.get('parent_result_sha256') == sha(PARENT_REPORT)
            and declaration.get('fixed_recovery_order') == 1,
            'Final root-reviewed q1 declaration missing or inconsistent')
    require({key: tuple(value) for key, value in declaration['initial_states'].items()} == entries,
            'Final declaration changed segment entries')
    allowed = declaration['allowed_changed_parameters']
    require(isinstance(allowed, list) and len(set(allowed)) == len(allowed)
            and set(allowed) <= set(PARAMETER_NAMES), 'Invalid explicit changed-parameter scope')
    declarations = declaration['candidates']
    require(isinstance(declarations, list) and 1 <= len(declarations) <= 4,
            'Final declaration must contain one to four explicit controls')
    labels = []
    for index, row in enumerate(declarations):
        candidate = row['parameters']
        require(set(candidate) == set(PARAMETER_NAMES)
                and all(math.isfinite(float(value)) for value in candidate.values()),
                'Candidate must supply all seven finite numeric parameters')
        tp.kernel.Parameters(**candidate, enable_a=True)
        require(candidate[RATE_NAME] > 0, 'Candidate kappa must be positive')
        changed = {key for key in PARAMETER_NAMES if candidate[key] != parameters[key]}
        require(changed <= set(allowed)
                and set(row['changed_parameters']) == changed
                and len(row['changed_parameters']) == len(changed),
                'Candidate changes do not match its explicit declaration')
        require(row['fixed_recovery_order'] == 1 and row['forward_evaluated'] is False
                and {key: tuple(value) for key, value in row['initial_states'].items()} == entries
                and row['only_recovery_order_changed_from_selected_q2'] is (not bool(changed)),
                'Candidate changed undeclared states/order or mislabels the comparison')
        label = row['label']
        require(isinstance(label, str) and label.isascii() and label.isidentifier()
                and label not in labels and isinstance(row['role'], str) and bool(row['role']),
                'Unique safe label and explicit comparison role required')
        labels.append(label)
        if index == 0:
            require(candidate == parameters and row['only_recovery_order_changed_from_selected_q2'],
                    'First control must preserve every selected q2 numeric parameter')
    return parent, seed, declaration, parameters, entries, declarations, selected + final


def _verify_freeze(freeze_path, declaration_path, parent, seed, declaration, parent_rows):
    manifest = json.loads(freeze_path.read_text())
    files = manifest['files_sha256']
    require(isinstance(files, dict) and files, 'Empty new freeze manifest')
    require(parent['source_sha256_before'] == parent['source_sha256_after'], 'Parent source guard failed')
    required = dict(parent['source_sha256_before'])

    def include(path):
        path = str(Path(path).resolve())
        digest = sha(path)
        if path in required:
            require(required[path] == digest, 'Previously frozen parent input changed: ' + path)
        required[path] = digest

    for path in (
        CALIBRATION / 'SOURCE_FREEZE.json', PARENT_REPORT, SEED_PATH, declaration_path,
        HERE / 'q1_material.py', HERE / 'algebra_selfcheck.py', HERE / 'ALGEBRA_VALIDATION.json',
        HERE / 'STRUCTURAL_CONTROL.md', HERE / 'derive_seed.py', HERE / 'SEED_PLAN.md',
        HERE / 'FORWARD_PLAN.md', Path(__file__), fa.CONTRACT_PATH, fa.CONTRACT_SOURCE,
    ):
        include(path)
    for row in parent_rows:
        prefix = Path(row['artifact_prefix']).resolve()
        require(prefix.is_relative_to((CALIBRATION / 'calibration_run').resolve()),
                'Parent artifact prefix outside frozen calibration run')
        for path in (prefix.with_suffix('.json'), prefix.with_suffix('.npz'),
                     Path(row['declaration_path'])):
            include(path)
    seed_sources = seed.get('input_sha256_before')
    require(isinstance(seed_sources, dict) and seed_sources and seed.get('inputs_unchanged') is True
            and seed_sources == seed.get('input_sha256_after'), 'Seed input provenance guard missing/failed')
    declared_sources = declaration.get('input_sha256')
    require(isinstance(declared_sources, dict) and declared_sources
            and declared_sources.get(str(SEED_PATH.resolve())) == sha(SEED_PATH),
            'Final declaration must freeze the analytic seed and all derivation/capacity inputs')
    for source_map in (seed_sources, declared_sources):
        for path, digest in source_map.items():
            if path in required:
                require(required[path] == digest, 'Declared/seed/parent input digests conflict: ' + path)
            required[path] = digest
    for path, digest in required.items():
        require(files.get(path) == digest, 'New root freeze misses exact required input: ' + path)
    for path, digest in files.items():
        require(Path(path).is_absolute() and sha(path) == digest, 'Frozen input changed: ' + path)
    require(files[str(HERE / 'q1_material.py')] == Q1_SHA256, 'Q1 implementation is not the reviewed version')
    require(files[str(CALIBRATION / 'bounded_calibration.py')] == CALIBRATION_SHA256,
            'Candidate-aware numerical verifier changed')
    return dict(files)


def run(freeze_path, declaration_path, output):
    parent, seed, declaration, parameters, entries, declarations, parent_rows = _load_declarations(declaration_path)
    frozen = _verify_freeze(freeze_path, declaration_path, parent, seed, declaration, parent_rows)
    freeze_digest = sha(freeze_path)
    output.mkdir(exist_ok=False)
    cases, requests = output / 'cases', output / 'requests'
    cases.mkdir(); requests.mkdir()
    budget = NumericalBudget()
    report = dict(status='preparing', fixed_recovery_order=1,
        source_sha256_before=frozen, freeze_path=str(freeze_path), freeze_sha256_before=freeze_digest,
        parent_selected_request_id=26, parent_selected_parameters=parameters,
        fixed_segment_entries={key: list(value) for key, value in entries.items()},
        analytic_kappa_reference=seed['q1_kappa_reference'], declared_candidates=declarations,
        candidate_order_is_predeclared=True, maximum_full_evaluations=MAX_EVALUATIONS,
        maximum_phase_calls=MAX_PHASE_CALLS, requested_coarse_candidates=len(declarations),
        final_declaration_path=str(declaration_path), final_declaration_sha256=sha(declaration_path),
        allowed_changed_parameters=declaration['allowed_changed_parameters'],
        coarse_precision=COARSE, final_precisions=FINAL_PRECISIONS,
        search_budget_s=SEARCH_LIMIT_S, total_budget_s=TOTAL_LIMIT_S, final_reserved_s=300,
        artifact_IO_and_initial_loading_excluded=True,
        inherited_inline_source_guard_time_conservatively_included=True,
        no_retry=True, no_early_feature_screen=True, no_periodic_prediction_reuse=True,
        no_new_initial_state_selection=True, no_state_clipping=True,
        no_residual_penalty_for_failure=True, no_holdout_access=True,
        failed_candidate_is_not_physical_family_rejection=True,
        requests=[], full_evaluations_started=0, selected_coarse_candidate=None, final_comparison=None)

    def persist():
        report.update(search_numerical_wall_s=budget.search_used(), total_numerical_wall_s=budget.total_used())
        artifacts.write(output / 'RESULT.json', report)

    persist()
    coarse_rows, final_rows, final_results = [], [], {}
    selected = None
    try:
        problem = tp.load_training_problem()
        contract = fa.load_feature_contract()
        report['data_contract'] = problem.checkpoint_metadata()
        base = q1.load_q1_material(tp.kernel.Parameters(**parameters, enable_a=True),
            candidate_source_id=str(PARENT_REPORT), candidate_source_sha256=sha(PARENT_REPORT))
        require(base.metadata()['fixed_recovery_order'] == 1, 'Q1 material metadata lost its order')
        # Preserve all inherited inputs and add the complete root-reviewed run provenance.
        sources = dict(base.source_sha256)
        for path, digest in sources.items():
            require(frozen.get(path) == digest, 'Frozen run misses material input: ' + path)
        sources.update(frozen)
        base = replace(base, source_sha256=MappingProxyType(sources))

        def request(number, label, candidate, precision, kind, context):
            require(any(candidate == item['parameters'] for item in declarations),
                    'Candidate is absent from the frozen final declaration')
            row = dict(request_id=number, label=label, parameters=dict(candidate),
                precision=dict(precision), kind=kind, context=context, fixed_recovery_order=1,
                status='declared', eligible_for_selection=False, model_evaluation_performed=False,
                exact_parameter_tuple_hex=[float(candidate[key]).hex() for key in PARAMETER_NAMES])
            report['requests'].append(row)
            declaration = requests / f'{number:03d}_{label}.json'
            artifacts.write(declaration, row)
            row.update(declaration_path=str(declaration), declaration_sha256=sha(declaration))
            if not budget.can_start(kind):
                row.update(status='not_run_budget', budget_exhausted=True)
                artifacts.write(requests / f'{number:03d}_{label}_result.json', row)
                persist()
                return row, None
            require(report['full_evaluations_started'] < MAX_EVALUATIONS, 'Evaluation count would exceed six')
            material = replace(base, parameters=tp.kernel.Parameters(**candidate, enable_a=True),
                candidate_source_id=str(declaration), candidate_source_sha256=row['declaration_sha256'])
            result = gates = verification = None
            prefix = f'{number:03d}_{label}'
            row.update(status='running', artifact_prefix=str(cases / prefix))
            persist()
            print(json.dumps(dict(start=number, label=label, kappa=candidate[RATE_NAME])), flush=True)
            before_total, before_search = budget.total_used(), budget.search_used()
            try:
                with budget.section(kind):
                    row['model_evaluation_performed'] = True
                    report['full_evaluations_started'] += 1
                    result = problem.evaluate(material, entries, tp.kernel.Precision(**precision),
                                              'Radau', budget.check, collect_rhs=False)
                    gates = fa.evaluate_features(problem, result['predictions'], contract)
                    verification = cal.verify_native(result, material, budget, alternate_roots=kind == 'final')
                    require(gates['n_primary'] == 20, 'Original twenty gates were not all evaluated')
                    require(math.isfinite(result['score']['verified_cost']), 'Full original cost is nonfinite')
                row.update(status='complete_valid' if verification['passed'] else 'complete_failed_numerical_checks',
                    eligible_for_selection=bool(verification['passed']), budget_exhausted=False)
            except Exception as exc:
                if hasattr(exc, 'result'):
                    result = exc.result
                row.update(status='retained_failure', eligible_for_selection=False,
                    failure_type=type(exc).__name__, failure=str(exc), traceback=traceback.format_exc(),
                    budget_exhausted=bool(budget_cause(exc) or not budget.can_start(kind)),
                    not_a_physical_family_rejection=True)
            row.update(numerical_wall_s=budget.total_used() - before_total,
                       search_numerical_wall_s=budget.search_used() - before_search)
            if result is not None:
                artifacts.save_case(cases, prefix, result, gates, verification)
                row['summary'] = result['summary']
                row['completed_phase_count'] = result['summary'].get('n_completed_phases', 0)
                if result['score'] is not None:
                    row['cost'] = result['score']['verified_cost']
                    row['original_block_costs'] = {name: .5 * value['weighted_total_mse']
                        for name, value in result['score']['blocks'].items()}
                    row['weighted_phase_rmse_decade'] = {name: math.sqrt(value['weighted_total_mse'])
                        for name, value in result['score']['blocks'].items()}
                if gates is not None:
                    row['gates'] = cal._gate_summary(gates)
                if verification is not None:
                    row['numerical_checks_passed'] = verification['passed']
                    row['alternate_root_checks'] = verification['alternate_root_checks']
            artifacts.write(requests / f'{number:03d}_{label}_result.json', row)
            persist()
            print(json.dumps(dict(done=number, status=row['status'], cost=row.get('cost'),
                                  failed_gates=row.get('gates', {}).get('n_failed'))), flush=True)
            return row, result if kind == 'final' else None

        for number, declaration in enumerate(declarations, start=1):
            row, _ = request(number, declaration['label'], declaration['parameters'], COARSE, 'search',
                             {key: value for key, value in declaration.items() if key != 'parameters'})
            coarse_rows.append(row)
        eligible = [row for row in coarse_rows if row['eligible_for_selection']]
        if eligible:
            selected = min(eligible, key=lambda row: (row['cost'], row['request_id']))
            report['selected_coarse_candidate'] = dict(request_id=selected['request_id'],
                parameters=selected['parameters'], coarse_cost=selected['cost'], gates=selected['gates'],
                selection_rule='minimum original cost among completed native-valid candidates; first ID breaks ties')
            for number, precision in enumerate(FINAL_PRECISIONS, start=5):
                row, result = request(number, 'selected_' + precision['name'], selected['parameters'],
                    precision, 'final', dict(selected_coarse_request_id=selected['request_id'],
                        independent_full_rerun=True, coarse_trajectory_reused=False))
                final_rows.append(row)
                if row['eligible_for_selection'] and result is not None:
                    final_results[precision['name']] = result
            if len(final_results) == 2 and budget.can_start('final'):
                with budget.section('final'):
                    report['final_comparison'] = cal.compare_final(final_results['nominal'],
                        final_results['refined'], selected['parameters']['a_max_s'])
            report['status'] = 'finite_rate_controls_and_final_checks_finished'
        else:
            report['status'] = 'no_valid_complete_coarse_candidate_selected'
    except Exception as exc:
        report.update(status='stopped_with_retained_failure', failure_type=type(exc).__name__,
                      failure=str(exc), traceback=traceback.format_exc())
    finally:
        after = {}
        source_errors = {}
        for path in frozen:
            try:
                after[path] = sha(path)
            except Exception as exc:
                after[path] = None
                source_errors[path] = f'{type(exc).__name__}: {exc}'
        report['source_sha256_after'] = after
        report['source_guard_errors'] = source_errors
        try:
            report['freeze_sha256_after'] = sha(freeze_path)
        except Exception as exc:
            report['freeze_sha256_after'] = None
            report['freeze_read_error'] = f'{type(exc).__name__}: {exc}'
        report['sources_unchanged'] = after == frozen and report['freeze_sha256_after'] == freeze_digest
        report['n_coarse_declarations_processed'] = len(coarse_rows)
        report['all_declared_coarse_candidates_attempted'] = len(coarse_rows) == len(declarations) and all(
            row['model_evaluation_performed'] for row in coarse_rows)
        report['all_four_coarse_candidates_attempted'] = (len(declarations) == 4
            and report['all_declared_coarse_candidates_attempted'])
        report['n_valid_complete_coarse_candidates'] = sum(row['eligible_for_selection'] for row in coarse_rows)
        report['selection_does_not_cover_unrun_candidates'] = not report['all_declared_coarse_candidates_attempted']
        report['n_actual_model_evaluations'] = sum(row['model_evaluation_performed'] for row in report['requests'])
        report['n_completed_actual_phases'] = sum(row.get('completed_phase_count', 0) for row in report['requests'])
        report['max_evaluation_contract_respected'] = (report['n_actual_model_evaluations'] <= MAX_EVALUATIONS
            and report['n_completed_actual_phases'] <= MAX_PHASE_CALLS)
        report['final_two_precisions_valid'] = (len(final_rows) == 2
            and all(row['eligible_for_selection'] for row in final_rows)
            and all(row.get('alternate_root_checks') == 184 for row in final_rows)
            and bool(report['final_comparison']) and report['final_comparison']['passed'])
        report['numerical_final_candidate_verified'] = (report['sources_unchanged']
            and report['max_evaluation_contract_respected'] and report['final_two_precisions_valid'])
        report.update(full_q1_parameter_optimization_performed=False, optimization_converged=False,
            q1_family_physically_rejected=False, q2_family_physically_rejected=False,
            unified_physical_mechanism_established=False, gate_count_is_accuracy=False,
            rate_rescaled_controls_are_order_only_comparisons=False,
            numerical_budget_was_extended=False, heldout_prediction_evaluated=False)
        persist()
        print(json.dumps({key: report[key] for key in ('status', 'n_actual_model_evaluations',
            'n_completed_actual_phases', 'all_declared_coarse_candidates_attempted',
            'numerical_final_candidate_verified', 'total_numerical_wall_s')}), flush=True)
    return 0 if report['all_declared_coarse_candidates_attempted'] and report['numerical_final_candidate_verified'] else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, default=HERE / 'SOURCE_FREEZE.json')
    parser.add_argument('--declaration', type=Path, default=HERE / 'FORWARD_DECLARATION.json')
    parser.add_argument('--output', type=Path, default=HERE / 'rate_controls_run')
    args = parser.parse_args(argv)
    return run(args.freeze.resolve(), args.declaration.resolve(), args.output.resolve())


if __name__ == '__main__':
    raise SystemExit(main())
