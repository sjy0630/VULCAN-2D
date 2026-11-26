"""Prepared saved-corner barrier for the sole mapped m1 control; no new physics solves."""
from pathlib import Path
import hashlib
import json
import math
import signal
import sys
import time

sys.dont_write_bytecode = True
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT/'analysis/reversible_capture_2026-11-03'
CACHE = Path('/private/tmp/vulcan-runtime-cache')
INPUT_HASHES = {
    OLD/'initial_barrier.json': 'fda98bde2b03ca5849341b3b87860c7a94ac45a9edd9143fb2ffa12e60257139',
    OLD/'initial_barrier.npz': 'd44323a203b1b803c8b6d05583fe7c7d5e3a5f568d032100093e2522905a0bda',
    HERE/'matched_linear_control.plan.json': 'bfbe5cae79977db0a5fdeab6d826e29ff402a12e102c4d879cfd4081ba5e2fea',
    HERE/'matched_linear_control.json': '7fcc3b7f244e03443d72ac3eeb065bd80335ad3f674cfeb2cd41c1cd45ea78b5',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj, exclusive=False):
    with path.open('x' if exclusive else 'w') as handle:
        json.dump(obj, handle, indent=2, allow_nan=False)
        handle.write('\n')


def main():
    dest = HERE/'matched_linear_barrier.json'
    planpath = HERE/'matched_linear_barrier.plan.json'
    if any(path.exists() for path in (dest, planpath, dest.with_suffix('.npz'), dest.with_suffix('.md'))):
        raise FileExistsError('Preserve existing diagnostic; no overwrite or automatic retry')
    for path, digest in INPUT_HASHES.items():
        if sha(path) != digest:
            raise ValueError(f'Predeclared source changed: {path}')
    original = json.loads((OLD/'initial_barrier.json').read_text())
    control_plan = json.loads((HERE/'matched_linear_control.plan.json').read_text())
    control = json.loads((HERE/'matched_linear_control.json').read_text())
    if original['status'] != 'completed_fixedq_prefix_and_500point_error_floor_no_ODE_or_fit':
        raise ValueError('Require completed saved-corner source')
    if control.get('plan') != control_plan or control_plan['fixed_control_order_m'] != 1:
        raise ValueError('Actual control report/plan or mapped model order mismatch')
    if not control.get('source_files_unchanged') or not all(control['source_files_unchanged'].values()):
        raise ValueError('Actual mapped-control sources were not unchanged')
    required_scalar = ('ron', 'gc', 'anchor_u', 'vmin', 'log_scale')
    required_arrays = ('lg', 'lv', 'li', 'tx', 'ti', 'voltage', 'observed_signed',
                       'observed_abs', 'path', 'mask', 'outward')
    if not all(control_plan['scalar_identity_checks'].get(key) is True for key in required_scalar):
        raise ValueError('Mapped control did not verify fixed readout/anchor scalars')
    if not all(control_plan['array_identity_checks'].get(key) is True for key in required_arrays):
        raise ValueError('Mapped control did not verify fixed spectrum/circuit/data arrays')
    # Validate both prior source chains, but never import a circuit or material module.
    sources = {str(path): digest for path, digest in INPUT_HASHES.items()}
    for report in (original, control):
        for path, digest in report['source_sha256'].items():
            if sha(path) != digest:
                raise ValueError(f'Inherited source hash no longer matches: {path}')
            if path in sources and sources[path] != digest:
                raise ValueError(f'Two saved source chains disagree: {path}')
            sources[path] = digest
    common_required = [CACHE/name for name in ('native_observations.npz', 'transport_g010_v0025.npz',
        'fig2b_transistor_lookup.json', 'bookoutput2_reverse_transistor_lookup.json')]
    shared_identity = {}
    for path in common_required:
        key = str(path)
        shared_identity[key] = original['source_sha256'][key] == control['source_sha256'][key]
    if not all(shared_identity.values()):
        raise ValueError('Old corner table/circuit/native data differ from mapped control')
    # Read the sole actual mapping; do not reconstruct or select another triple.
    parameters = control_plan['mapped_m1_parameters']
    lam, jref, kappa = [float(parameters[key]) for key in
        ('lambda_per_v', 'J_positive_a', 'kappa_per_v2')]
    if not all(math.isfinite(value) for value in (lam, jref, kappa)) or lam < 0. or jref <= 0. or kappa < 0.:
        raise ValueError('Invalid fixed mapped parameters')
    for run in control['runs']:
        if run['summary']['parameters'] != parameters:
            raise ValueError('Actual m1 attempt did not use the declared unique mapping')
    anchor_u = float(control['runs'][0]['summary']['conditional_anchor_u'])
    if any(run['summary']['conditional_anchor_u'] != anchor_u for run in control['runs']):
        raise ValueError('Control attempts used inconsistent anchors')
    fixed_source = ROOT/'analysis/reset_shape_2026-11-03/soft_field_mobility_probe.json'
    frozen_fixed = json.loads(fixed_source.read_text())['plan']
    for key in ('Ron_ohm', 'gamma_c_ev'):
        if original['plan']['seed'][key] != frozen_fixed[key]:
            raise ValueError('Old corner/material amplitude or state coordinate differs')
    if anchor_u != frozen_fixed['anchor_u']:
        raise ValueError('Mapped control anchor differs from original fixed anchor')
    expected_cases = {(0., 'nominal'), (0., 'refined'), (.003, 'nominal'), (.003, 'refined')}
    if {(case['material_envelope'], case['grid']['name']) for case in original['cases']} != expected_cases:
        raise ValueError('Require exactly original four fixed cases')
    nominal_qbar = math.exp(-anchor_u)
    for case in original['cases']:
        if not 0. < case['qbar'] < 1.:
            raise ValueError('Invalid original barrier state')
        if case['material_envelope'] == 0. and not math.isclose(case['qbar'], nominal_qbar, rel_tol=1e-14, abs_tol=0.):
            raise ValueError('Nominal qbar does not match the actual control anchor')
        if case['qbar'] < nominal_qbar:
            raise ValueError('Saved envelope qbar does not bound the same anchor')
    with np.load(OLD/'initial_barrier.npz') as archive:
        saved = {key: archive[key].copy() for key in archive.files}
    voltage = saved['command_voltage_v']
    observations = abs(saved['observed_current_a'])
    mask = saved['original_scoring_mask']
    outward = saved['outward_branch_mask']
    np.testing.assert_array_equal(mask, np.isfinite(observations) & (observations >= 1e-11) & (voltage > 1e-12))
    if (len(voltage), int(mask.sum()), int((mask & outward).sum())) != (503, 500, 250):
        raise ValueError('Original native/scoring counts changed')
    plan = dict(
        scope='One saved-corner necessary barrier for the actual mapped m1 control only; '
              'not a retry/continuation of its failed finite-table trajectory.',
        mapped_parameters=parameters, parameter_source='matched_linear_control.plan.json:mapped_m1_parameters',
        cases='Exactly original2grids x2material envelopes; same per-case qbar',
        shared_readout_circuit_native_hashes=shared_identity, conditional_anchor_u=anchor_u,
        capture_upper='vmax*Imax*(1-qbar)/J1', loss_lower='vmin*exp(lambda1*vmin)*qbar',
        saved_corner_fields=dict(Imax='strong_node_current_upper_a[1:]',
            vmax='weak_node_voltage_upper_v[1:]', vmin='strong_node_voltage_lower_v[:-1]'),
        rate_guard='1e-12*(new_capture_upper+new_loss_lower)+1e-18',
        prefix='Only contiguous finite upper<0 cells from.2V. Stop at first unknown/nonnegative cell. '
               'Intersect with old certified prefix; no new voltage nodes.',
        score='Reuse old certified_native_mask and optimistic_current_upper_a on the prefix intersection; '
              'sum max(log10(Iobs/Iceil),0)^2 and divide by original500. Other points contribute0 optimistically.',
        no_prefix_policy='If no positive-length prefix is certified, floor is null and no physical conclusion is drawn.',
        domain_scope='Necessary condition for an admissible continuous trajectory on the certified prefix; '
                     'no behavior beyond the saved table or failed4.707V trajectory is invented.',
        comparisons='Existing m2 achieved conditional score and old sinh conditional score only; not new acceptance gates.',
        compute_budget_s=10, budget_start='After source/data/identity validation and plan saving, before four-case arithmetic.',
        counts=dict(cases=4, new_KCL_roots=0, new_inversions=0, new_ODE=0,
                    new_fits=0, parameter_scans=0, new_spectral_quadrature=0),
        stop_policy='Preserve failures; no new root, inverse, grid, seed, fit, model domain or retry.')
    dump(planpath, plan, exclusive=True)
    for path in (planpath, HERE/'MATCHED_BARRIER_PLAN.md', Path(__file__)):
        sources[str(path)] = sha(path)
    report = dict(status='running_mapped_m1_saved_corner_barrier', plan=plan,
        source_sha256=sources, cases=[],
        source_m2_RMSE_decade=control_plan['source_m2_RMSE_decade'],
        prior_sinh_RMSE_decade=original['previous_achieved_conditional_500point_RMSE_decade'])
    dump(dest, report, exclusive=True)
    arrays = {key: saved[key] for key in ('command_voltage_v', 'observed_current_a',
              'original_scoring_mask', 'outward_branch_mask')}
    def timeout(*_):
        raise TimeoutError('10second saved-corner arithmetic budget exhausted; no retry')
    previous_handler = signal.signal(signal.SIGALRM, timeout)
    begun = time.monotonic()
    signal.setitimer(signal.ITIMER_REAL, 10.)
    try:
        for case in original['cases']:
            prefix = ('eta0' if case['material_envelope'] == 0. else 'eta003')+'_'+case['grid']['name']
            nodes = saved[prefix+'_command_nodes']
            imax = saved[prefix+'_strong_node_current_upper_a'][1:]
            vmax = saved[prefix+'_weak_node_voltage_upper_v'][1:]
            vmin = saved[prefix+'_strong_node_voltage_lower_v'][:-1]
            qbar = case['qbar']
            if not (len(nodes) == len(imax)+1 == len(vmax)+1 == len(vmin)+1
                    and np.all(vmin > 0.) and np.all(imax > 0.) and np.all(vmax >= vmin)):
                raise ValueError('Invalid saved rectangle coordinates')
            capture = vmax*imax*(1.-qbar)/jref
            loss = vmin*np.exp(lam*vmin)*qbar
            guard = 1e-12*(capture+loss)+1e-18
            upper = capture-loss+guard
            certified = np.isfinite(upper) & (upper < 0.)
            first = np.flatnonzero(~certified)
            available_count = int(first[0]) if len(first) else len(upper)
            count = min(available_count, int(case['contiguous_negative_cells']))
            end = float(nodes[count])
            use = (saved[prefix+'_certified_native_mask'] & mask & outward
                   & (voltage >= .2) & (voltage <= end)) if count else np.zeros_like(mask)
            ceilings = saved[prefix+'_optimistic_current_upper_a']
            deficits = np.zeros(len(voltage))
            floor, sse = None, None
            if count:
                if not (np.all(np.isfinite(ceilings[use])) and np.all(ceilings[use] > 0.)):
                    raise ValueError('Required ceiling unavailable in saved native domain')
                deficits[use] = np.maximum(np.log10(observations[use]/ceilings[use]), 0.)
                sse = float(np.dot(deficits, deficits))
                floor = math.sqrt(sse/500.)
            report['cases'].append(dict(material_envelope=case['material_envelope'], grid=case['grid'],
                qbar=qbar, certified_positive_length_prefix=bool(count),
                saved_corner_contiguous_negative_cells=available_count,
                used_prefix_cells=count, original_prefix_cells=case['contiguous_negative_cells'],
                certified_prefix_start_v=.2, used_prefix_end_v=end,
                original_prefix_end_v=case['certified_prefix_end_v'], prefix_extended=False,
                certified_original_native_points=int(use.sum()), original_denominator=500,
                minimum_negative_used_prefix_margin_v=float(min(-upper[:count])) if count else None,
                optimistic_squared_log_residual_sum=sse, necessary_500point_RMSE_floor_decade=floor,
                floor_exceeds_source_m2_score=bool(floor > report['source_m2_RMSE_decade']) if floor is not None else None,
                floor_exceeds_prior_sinh_score=bool(floor > report['prior_sinh_RMSE_decade']) if floor is not None else None,
                interpretation='Conditional necessary bound, not actual completed control RMSE' if count else
                               'No certified positive-length prefix; no conclusion, stop without extension'))
            arrays.update({prefix+'_command_nodes':nodes, prefix+'_Imax_a':imax,
                prefix+'_vmax_v':vmax, prefix+'_vmin_v':vmin, prefix+'_capture_upper':capture,
                prefix+'_loss_lower':loss, prefix+'_rounding_guard':guard, prefix+'_rate_upper':upper,
                prefix+'_negative_cells':certified, prefix+'_used_native_mask':use,
                prefix+'_saved_current_upper_a':ceilings, prefix+'_minimum_positive_log_residual':deficits})
        report['status'] = 'completed_mapped_m1_saved_corner_barrier_no_new_roots_or_ODE'
    except Exception as exc:
        report.update(status='failed_or_partial_saved_corner_diagnostic_preserved',
                      error_type=type(exc).__name__, error=str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        signal.signal(signal.SIGALRM, previous_handler)
        report['compute_elapsed_s'] = time.monotonic()-begun
        report['source_files_unchanged'] = {path: sha(path) == digest for path, digest in sources.items()}
        if not all(report['source_files_unchanged'].values()):
            report['status'] = 'invalid_source_changed_during_diagnostic'
        dump(dest, report)
        np.savez_compressed(dest.with_suffix('.npz'), **arrays)
    lines = ['# 唯一映射m1：保存角点的初始屏障', '',
             f'状态：{report["status"]}；纯保存数组算术 {report["compute_elapsed_s"]:.6f} s。', '',
             '参数直接读取实际matched control计划，未重选λ/J/κ。旧谱、电路、锚、原生数据和来源哈希已核对；'
             '没有重试失败轨迹、扩表、求根、反演、ODE或拟合。', '',
             '| 包络 | 网格 | 正长度前缀 | 使用终点/V | 原评分点 | 原500必要RMSE下界/dec |',
             '|---|---|---|---:|---:|---:|']
    for case in report['cases']:
        floor = case['necessary_500point_RMSE_floor_decade']
        lines.append(f'| {case["material_envelope"]} | {case["grid"]["name"]} | '
            f'{case["certified_positive_length_prefix"]} | {case["used_prefix_end_v"]:.5f} | '
            f'{case["certified_original_native_points"]} | '+('不下结论' if floor is None else f'{floor:.9f}')+' |')
    lines.extend(['', '只用新连续负率前缀与旧认证原生点的交集；其它原500点误差乐观设零。'
                  '无正长度前缀时不下物理结论，也不扩大检查。', '',
                  '下界不是实际完成轨迹的RMSE，更不把4.707 V处数值表域失败改写成物理反证。'
                  '只约束这一固定映射及相同读出/条件初态；不排除整个m1家族，不证明微观协作。'])
    if 'error' in report:
        lines.extend(['', report['error_type']+': '+report['error']])
    dest.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=report['status'], compute_elapsed_s=report['compute_elapsed_s'],
                         cases=report['cases']), indent=2))


if __name__ == '__main__':
    main()
