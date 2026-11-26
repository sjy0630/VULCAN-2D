"""Prepared matched m1 counterfactual; no execution on import and no refit.

Run only after root accepts finalized m2 numerical validation and its lower
same-cycle score. This program uses one algebraically mapped parameter triple.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import signal
import sys
import time

sys.dont_write_bytecode = True
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
M1_SOURCE = ROOT/'analysis/reversible_family_2026-11-03/forward_kernel.py'
QREF_SOURCE = ROOT/'analysis/reversible_capture_2026-11-03/reversible_budget.json'
M1_SHA256 = '4382a56449b0b736a963356b1257873f49e0cf3125f74d6f2ec6f9aa865f0f17'
QREF_SHA256 = '7ddc84910aab62012361e5e058bf3e4c116e33dc3a96fc4763259abb86668414'
QREF_EXPECTED = .05490380671259502


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(item) for item in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def dump(path, value, exclusive=False):
    with path.open('x' if exclusive else 'w') as handle:
        json.dump(clean(value), handle, indent=2, allow_nan=False, ensure_ascii=False)
        handle.write('\n')


def load_named(path, name):
    """Unique names prevent aliasing m1/m2 dataclasses or the forward_kernel cache."""
    if name in sys.modules:
        raise RuntimeError('Refuse repeated loading or implicit control rerun')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # Required for dataclasses with future annotations.
    spec.loader.exec_module(module)
    return module


def main():
    dest = HERE/'matched_linear_control.json'
    planpath = HERE/'matched_linear_control.plan.json'
    output_paths = [dest, planpath, dest.with_suffix('.md')]
    output_paths += [HERE/f'matched_linear_{name}.{extension}'
                     for name in ('nominal', 'refined') for extension in ('json', 'npz')]
    if any(path.exists() for path in output_paths):
        raise FileExistsError('Preserve every existing control attempt; no overwrite or retry')
    fitpath = HERE/'finite_family_fit.json'
    fit = json.loads(fitpath.read_text())
    if fit.get('status') != 'completed_bounded_attempt_no_global_or_full_model_claim':
        raise ValueError('m2 fit has not finalized successfully')
    if fit.get('numerical_validation_passed') is not True or fit.get('lower_than_old_same_conditioned_score') is not True:
        raise ValueError('Control requires validated m2 with lower same-cycle score')
    if fit['plan'].get('fixed_relaxation_order_m') != 2:
        raise ValueError('Source must be the fixed-m2 family')
    if not all(fit.get('source_hashes_still_match', {}).values()):
        raise ValueError('m2 fit reported changed inputs')
    for path, digest in fit['source_sha256'].items():
        if sha(path) != digest:
            raise ValueError(f'Finalized m2 source changed: {path}')
    finalpaths = [HERE/f'final_{name}.json' for name in ('nominal', 'refined')]
    finals = [json.loads(path.read_text()) for path in finalpaths]
    if any(item.get('status') != 'completed' or item.get('fixed_recovery_order') != 2 for item in finals):
        raise ValueError('Require two completed fixed-m2 final summaries')
    if finals[0]['parameters'] != finals[1]['parameters']:
        raise ValueError('Final m2 precisions used different physical parameter triples')
    for item in finals:
        corresponding = next(row for row in fit['final_verifications']
                             if row['precision']['name'] == item['precision']['name'])
        if corresponding['parameters'] != item['parameters'] or corresponding['metrics'] != item['metrics']:
            raise ValueError('Standalone m2 final summary differs from finalized fit report')
    m2_parameters = finals[-1]['parameters']
    m2_rmse = finals[-1]['metrics']['all500']['RMSE_decade']
    if not m2_rmse == fit['verified_conditional_RMSE_decade'] < fit['plan']['prior_conditional_500point_RMSE_decade']:
        raise ValueError('m2 success gate is inconsistent with final summary')
    if sha(M1_SOURCE) != M1_SHA256 or sha(QREF_SOURCE) != QREF_SHA256:
        raise ValueError('Frozen m1 kernel or observed-return reference changed')
    reference = json.loads(QREF_SOURCE.read_text())
    qrow = next(row for row in reference['seed_stability'] if row['command_v'] == 2.)
    qref = float(qrow['q'])
    if qref != QREF_EXPECTED:
        raise ValueError('Observed return2 reference differs from preselected qref')
    m1 = load_named(M1_SOURCE, 'vulcan_frozen_m1_matched_control')
    m2 = load_named(HERE/'forward_kernel.py', 'vulcan_fixed_m2_readout_control')
    kernel = m1.ForwardKernel.load()
    m2_kernel = m2.ForwardKernel.load()
    # No physical root/ODE is evaluated by these identity checks.
    scalar_equalities = {name: getattr(kernel, name) == getattr(m2_kernel, name)
                         for name in ('ron', 'gc', 'anchor_u', 'vmin', 'log_scale')}
    array_names = ('lg', 'lv', 'li', 'tx', 'ti', 'voltage', 'observed_signed', 'observed_abs',
                   'path', 'mask', 'outward')
    array_equalities = {name: bool(np.array_equal(getattr(kernel, name), getattr(m2_kernel, name)))
                        for name in array_names}
    if not all(scalar_equalities.values()) or not all(array_equalities.values()):
        raise ValueError('m1/m2 readout, circuit, anchor, observations or native mask differ')
    with np.load(HERE/'final_refined.npz') as data:
        for key, value in (('command_voltage_v', kernel.voltage), ('observed_current_a', kernel.observed_signed),
                           ('original_scoring_mask', kernel.mask), ('path_voltage_s', kernel.path)):
            np.testing.assert_array_equal(data[key], value)
    p2 = m2_parameters
    p1 = dict(lambda_per_v=p2['lambda_per_v'], J_positive_a=p2['J_positive_a']*qref,
              kappa_per_v2=p2['kappa_per_v2']*qref)
    parameters = m1.Parameters(**p1)
    precisions = [m1.Precision(**item) for item in fit['plan']['verification_precisions']]
    if [item.name for item in precisions] != ['nominal', 'refined']:
        raise ValueError('Require the two predeclared nominal/refined precisions')
    capture_ratio = (p1['kappa_per_v2']/p1['J_positive_a'])/(p2['kappa_per_v2']/p2['J_positive_a'])
    recovery_ratio = p1['kappa_per_v2']/(p2['kappa_per_v2']*qref)
    if not math.isclose(capture_ratio, 1., rel_tol=1e-14, abs_tol=0.) or recovery_ratio != 1.:
        raise ValueError('Algebraic coefficient matching failed')
    plan = dict(
        scope='One matched linear-recovery counterfactual of finalized m2; no refit, seed scan, '
              'RESET, bare history, holdout or aggregate verification.',
        fixed_control_order_m=1, source_order_m=2, m2_parameters=p2, mapped_m1_parameters=p1,
        mapping='lambda1=lambda2, kappa1=kappa2*qref, J1=J2*qref',
        qref=qref, qref_source=str(QREF_SOURCE), qref_source_sha256=QREF_SHA256,
        qref_definition='Existing observed-cycle4 return2 inverse-state q, not fitted m2 state or measured occupation',
        matching='kappa1/J1=kappa2/J2 at every state; recovery and net vector field equal '
                 'at q=qref for every local v/I. No claim of matching state derivatives.',
        matched_capture_coefficient_ratio=capture_ratio, matched_reference_recovery_ratio=recovery_ratio,
        trajectory='Same .2V anchor; full original500 scores, continuous apex, two precisions, same KCL/readout.',
        scalar_identity_checks=scalar_equalities, array_identity_checks=array_equalities,
        verification_precisions=fit['plan']['verification_precisions'],
        precision_limits=fit['plan']['precision_limits'], legacy_limits=fit['plan']['legacy_limits'],
        compute_budget_s=120,
        budget_start='After input loading/hash/identity checks and plan saving; before circuit validation and ODEs.',
        max_physical_parameter_triples=1, max_forward_attempts=2, max_ODE_segments=6,
        no_parameter_optimization=True, no_adaptive_recovery=True,
        failure_scope='Preserve numerical domain/resolution/solver/time failures; never fabricate physical rejection.',
        source_m2_RMSE_decade=m2_rmse,
        interpretation='Controlled finite comparison, not rejection of all m1 fits or proof of microscopic cooperativity.')
    dump(planpath, plan, exclusive=True)
    sources = dict(kernel.source_sha256)
    sources.update(m2_kernel.source_sha256)
    for path in (fitpath, *finalpaths, HERE/'final_refined.npz', QREF_SOURCE, planpath, Path(__file__)):
        sources[str(path)] = sha(path)
    report = dict(status='running_matched_linear_control', plan=plan, source_sha256=sources, runs=[])
    dump(dest, report, exclusive=True)
    started = time.monotonic()
    def budget(*_):
        if time.monotonic()-started >= 120.:
            raise TimeoutError('120second matched-control total budget exhausted; no retry')
    def alarm(*_):
        raise TimeoutError('120second matched-control total budget exhausted; no retry')
    previous_handler = signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, 120.)
    completed = []
    try:
        report['circuit_validation'] = kernel.validate_circuit(check_budget=budget)
        modes = report['circuit_validation']['modes']
        mode = ('fast' if modes['fast']['passed'] else
                'strict_bisection42' if modes['strict_bisection42']['passed'] else None)
        if mode is None:
            raise RuntimeError('Neither control circuit mode passed; no ODE run')
        report['selected_circuit_mode'] = mode
        for precision in precisions:
            row = dict(precision=precision.name, status='running')
            report['runs'].append(row)
            arrays = {}
            try:
                budget()
                result = kernel.evaluate(parameters, precision, circuit_mode=mode,
                                         check_budget=budget, collect_rhs=True)
                arrays = result['arrays']
                row['summary'] = result['summary']
                audit, legacy_rows = kernel.audit_native_legacy(result, check_budget=budget)
                row['native_legacy_audit'] = audit
                arrays['native_legacy_audit'] = legacy_rows
                row['status'] = 'completed'
                completed.append((row, arrays))
            except m1.KernelFailure as exc:
                row.update(status='numerical_failure_preserved', summary=exc.summary,
                           failure_scope=exc.summary['failure_scope'], physical_rejection=False)
                arrays = exc.arrays
                if exc.summary['failure_scope'] == 'caller_budget_exhausted':
                    raise TimeoutError(str(exc)) from exc
            except Exception as exc:
                row.update(status='numerical_or_audit_failure_preserved', error_type=type(exc).__name__,
                           error=str(exc), physical_rejection=False)
                if isinstance(exc, TimeoutError):
                    raise
            finally:
                np.savez_compressed(HERE/f'matched_linear_{precision.name}.npz', **arrays)
                dump(HERE/f'matched_linear_{precision.name}.json', row)
                dump(dest, report)
        if len(completed) == 2:
            first, second = [item[1] for item in completed]
            differences = dict(max_log_current_change_decade=float(max(abs(np.log10(
                first['current_a'][kernel.mask]/second['current_a'][kernel.mask])))),
                max_state_u_change=float(max(abs(first['state_u']-second['state_u']))),
                max_state_q_change=float(max(abs(first['state_q']-second['state_q']))))
            differences['passed'] = all(differences[key] <= limit for key, limit in plan['precision_limits'].items())
            report['precision_comparison'] = differences
            audits_ok = all(not row['native_legacy_audit']['failures'] and all(
                row['native_legacy_audit'][key] <= limit for key, limit in plan['legacy_limits'].items())
                for row, _ in completed)
            continuity_ok = all(row['summary']['max_native_KCL_residual_a'] <= 1e-13
                and row['summary']['max_rhs_KCL_residual_a'] <= 1e-13
                and abs(row['summary']['apex_state_continuity_error']) <= 1e-12
                and abs(row['summary']['anchor_outward_join_error']) <= 1e-12 for row, _ in completed)
            report['numerical_validation_passed'] = bool(differences['passed'] and audits_ok and continuity_ok)
            report['native_legacy_audits_passed'] = audits_ok
            report['continuity_and_KCL_passed'] = continuity_ok
            score = completed[-1][0]['summary']['metrics']['all500']['RMSE_decade']
            report['control_candidate_RMSE_decade'] = score
            if report['numerical_validation_passed']:
                report['verified_matched_linear_RMSE_decade'] = score
                report['matched_control_minus_m2_RMSE_decade'] = score-m2_rmse
        report['status'] = 'completed_matched_control_attempts_no_refit'
    except Exception as exc:
        report.update(status='partial_or_failed_matched_control_preserved',
                      error_type=type(exc).__name__, error=str(exc), physical_rejection=False)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        signal.signal(signal.SIGALRM, previous_handler)
        report['compute_elapsed_s'] = time.monotonic()-started
        report['source_files_unchanged'] = {path: sha(path) == digest for path, digest in sources.items()}
        if not all(report['source_files_unchanged'].values()):
            report['status'] = 'invalid_source_changed_during_matched_control'
            report['numerical_validation_passed'] = False
            report.pop('verified_matched_linear_RMSE_decade', None)
            report.pop('matched_control_minus_m2_RMSE_decade', None)
        dump(dest, report)
    lines = ['# 匹配线性恢复反事实', '', f'状态：{report["status"]}。', '',
             '以原观测返扫2 V的qref固定匹配κ1=κ2qref、J1=J2qref、λ1=λ2。'
             '捕获系数在所有状态相同，qref处净率相同；没有重拟合或种子扫描。', '',
             '读出、电路、原生数据/掩码、条件锚均逐项检查一致。两精度、原500点、'
             '顶点连续性和legacy审计保留；q是谱宽坐标，不是测得的占据率。', '',
             '| 精度 | 状态 | 全500 RMSE/dec |', '|---|---|---:|']
    for row in report['runs']:
        score = row.get('summary', {}).get('metrics', {}).get('all500', {}).get('RMSE_decade')
        lines.append(f'| {row["precision"]} | {row["status"]} | '+('—' if score is None else f'{score:.9f}')+' |')
    lines.extend(['', f'已验证m2同条件RMSE：{m2_rmse:.9f} decade。'])
    if 'verified_matched_linear_RMSE_decade' in report:
        lines.append(f'通过数值审核的匹配m1：{report["verified_matched_linear_RMSE_decade"]:.9f} decade。')
    lines.extend(['', '这是一个固定映射的反事实，不是所有m1模型的全局最优比较；'
                  '失败不造物理反证，差异不证明微观二体反应或完整统一模型。'])
    dest.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({key: report.get(key) for key in ('status', 'compute_elapsed_s',
        'numerical_validation_passed', 'verified_matched_linear_RMSE_decade',
        'matched_control_minus_m2_RMSE_decade')}, indent=2))


if __name__ == '__main__':
    main()
