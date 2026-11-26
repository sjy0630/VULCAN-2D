"""Saved-array-only bound for forward terminal transfer, not local trapping.

No new KCL solve, inversion, quadrature, ODE, fitting, or parameter search.
"""
from pathlib import Path
import ast
import hashlib
import json
import math
import signal
import sys
import time

sys.dont_write_bytecode = True
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CACHE = Path('/private/tmp/vulcan-runtime-cache')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value, exclusive=False):
    with path.open('x' if exclusive else 'w') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def main():
    dest = HERE/'forward_transfer_barrier.json'
    planpath = HERE/'forward_transfer_barrier.plan.json'
    if any(path.exists() for path in (dest, planpath, dest.with_suffix('.npz'), dest.with_suffix('.md'))):
        raise FileExistsError('Preserve the existing attempt; no overwrite or automatic retry')
    original_path = HERE/'initial_barrier.json'
    original = json.loads(original_path.read_text())
    if original['status'] != 'completed_fixedq_prefix_and_500point_error_floor_no_ODE_or_fit':
        raise ValueError('Require completed original fixed-q barrier')
    with np.load(original_path.with_suffix('.npz')) as data:
        saved = {key: data[key].copy() for key in data.files}
    table = CACHE/'transport_g010_v0025.npz'
    if sha(table) != original['source_sha256'][str(table)]:
        raise ValueError('Transport table differs from original barrier source')
    with np.load(table) as data:
        metadata = json.loads(str(data['metadata']))
    temperature = float(metadata['shape']['temperature_k'])
    shape_path = ROOT/'analysis/constrained_shared_2026-11-03/lognormal_spectral_refinement.json'
    shape = json.loads(shape_path.read_text())['parameters']
    if temperature != float(shape['temperature_k']):
        raise ValueError('Table and frozen spectral source temperatures differ')
    spectral_path = ROOT/'vulcan2d/shared_spectral.py'
    # Read the existing exact literal without importing/executing the physics module.
    syntax = ast.parse(spectral_path.read_text())
    kb = next(ast.literal_eval(node.value) for node in syntax.body
              if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                 and target.id == 'KB_EV_PER_K' for target in node.targets))
    thermal_v = float(kb)*temperature  # Energies in eV: kBT/e numerically volts.
    if not math.isfinite(thermal_v) or thermal_v <= 0.:
        raise ValueError('Invalid frozen thermal voltage')
    v = saved['command_voltage_v']
    obs = abs(saved['observed_current_a'])
    mask = saved['original_scoring_mask']
    outward = saved['outward_branch_mask']
    np.testing.assert_array_equal(mask, np.isfinite(obs) & (obs >= 1e-11) & (v > 1e-12))
    if (len(v), int(mask.sum()), int((mask & outward).sum()), len(original['cases'])) != (503, 500, 250, 4):
        raise ValueError('Original sample mask/case count changed')
    plan = dict(
        scope='Single saved-array algebraic check of the same fixed seed with forward '
              'terminal transfer replacing net I in the proposed rate; not a trap-capture model.',
        identity='Iplus=I/[1-exp(-v/VT)] for v>0, equal-temperature reservoirs, '
                 'same elastic transmission kernel; Iminus=Iplus*exp(-v/VT).',
        temperature_k=temperature, KB_EV_PER_K=float(kb), thermal_voltage_v=thermal_v,
        temperature_source=str(table)+':metadata.shape.temperature_k',
        temperature_crosscheck_source=str(shape_path)+':parameters.temperature_k',
        boltzmann_source=str(spectral_path)+':KB_EV_PER_K literal',
        seed=original['plan']['seed'], cases='Exactly original two grids x two material envelopes',
        fixed_qbar='Unchanged per-case qbar from initial_barrier.json',
        upper='capture_upper_new=capture_upper_old/[-expm1(-vmin/VT)]; '
              'vmin=strong_node_voltage_lower_v[:-1]; emission_lower unchanged.',
        guard='1e-12*(capture_upper_new+emission_lower)+1e-18',
        prefix='Start at same .2V; stop at first upper>=0 or nonfinite. '
               'Never extend past originally saved certified prefix.',
        scoring='Use intersection of new prefix and saved native certified mask; '
                'reuse saved current ceilings and original500 denominator; all other errors zero.',
        no_recalibration='Keep lambda/J exactly fixed; this isolates one proposed flux correction. '
                         'The two old return zero-affinity calibrations are not silently reimposed.',
        counts=dict(cases=4, new_KCL_roots=0, new_inverse_problems=0,
                    new_quadrature=0, new_ODE=0, new_fits=0, new_parameter_scans=0),
        compute_budget_s=10,
        budget_start='After input loads, hashes and plan write; before four-case arithmetic.',
        stop_policy='One attempt only; preserve failures; no new temperature, seed, rate or table evaluation.',
        caveat='Forward terminal transfer is not reservoir incident population or local defect capture.')
    paths = [original_path, original_path.with_suffix('.npz'), table, shape_path,
             spectral_path, HERE/'reversible_budget.json', Path(__file__)]
    hashes = {str(path): sha(path) for path in paths}
    dump(planpath, plan, exclusive=True)
    hashes[str(planpath)] = sha(planpath)
    report = dict(status='running_saved_array_forward_transfer_barrier', plan=plan,
                  source_sha256=hashes, cases=[],
                  previous_achieved_conditional_500point_RMSE_decade=
                      original['previous_achieved_conditional_500point_RMSE_decade'])
    dump(dest, report, exclusive=True)
    arrays = {key: saved[key] for key in ('command_voltage_v', 'observed_current_a',
              'original_scoring_mask', 'outward_branch_mask')}
    def timeout(*_):
        raise TimeoutError('10second saved-array-only budget exhausted; preserve failure')
    previous_handler = signal.signal(signal.SIGALRM, timeout)
    begun = time.monotonic()
    signal.setitimer(signal.ITIMER_REAL, 10.)
    try:
        for case in original['cases']:
            prefix = ('eta0' if case['material_envelope'] == 0 else 'eta003')+'_'+case['grid']['name']
            nodes = saved[prefix+'_command_nodes']
            low_v = saved[prefix+'_strong_node_voltage_lower_v'][:-1]
            capture = saved[prefix+'_capture_upper']
            emission = saved[prefix+'_emission_lower']
            if not (np.all(np.isfinite(low_v)) and np.all(low_v > 0.)):
                raise ValueError('Forward-transfer correction requires strictly positive saved lower voltages')
            correction = 1./(-np.expm1(-low_v/thermal_v))
            capture_new = capture*correction
            guard = 1e-12*(capture_new+emission)+1e-18
            upper = capture_new-emission+guard
            good = np.isfinite(upper) & (upper < 0.)
            stops = np.flatnonzero(~good)
            available_count = int(stops[0]) if len(stops) else len(upper)
            count = min(available_count, case['contiguous_negative_cells'])
            end = float(nodes[count])
            old_mask = saved[prefix+'_certified_native_mask']
            use = old_mask & mask & outward & (v >= .2) & (v <= end)
            ceilings = saved[prefix+'_optimistic_current_upper_a']
            if not (np.all(np.isfinite(ceilings[use])) and np.all(ceilings[use] > 0.)):
                raise ValueError('Required current ceiling absent from saved certified mask')
            deficits = np.zeros(len(v))
            deficits[use] = np.maximum(np.log10(obs[use]/ceilings[use]), 0.)
            sse = float(np.dot(deficits, deficits))
            floor = math.sqrt(sse/500.)
            report['cases'].append(dict(
                material_envelope=case['material_envelope'], grid=case['grid'], qbar=case['qbar'],
                correction_minimum=float(min(correction)), correction_maximum=float(max(correction)),
                original_prefix_end_v=case['certified_prefix_end_v'], certified_prefix_end_v=end,
                contiguous_negative_cells=count, original_contiguous_negative_cells=case['contiguous_negative_cells'],
                prefix_extended=False, native_saved_ceiling_intersection_only=True,
                certified_original_native_points=int(use.sum()), eligible_total_unchanged=500,
                minimum_negative_prefix_margin_v=float(min(-upper[:count])) if count else None,
                optimistic_squared_log_residual_sum=sse,
                optimistic_500point_RMSE_floor_decade=floor,
                original_net_current_500point_floor_decade=case['optimistic_500point_RMSE_floor_decade'],
                floor_exceeds_previous_achieved_conditional_RMSE=
                    floor > report['previous_achieved_conditional_500point_RMSE_decade'],
                correction_never_reduces_capture=bool(np.all(correction >= 1.)),
                scope='Fixed seed and saved initial state only; all kappa>=0; '
                      'not a numerical trajectory or local-trap rejection'))
            arrays.update({prefix+'_command_nodes':nodes, prefix+'_vmin_v':low_v,
                prefix+'_forward_transfer_factor':correction, prefix+'_capture_upper':capture_new,
                prefix+'_emission_lower':emission, prefix+'_rounding_guard':guard,
                prefix+'_rate_upper':upper, prefix+'_negative_certified_cells':good,
                prefix+'_certified_native_mask':use, prefix+'_optimistic_current_upper_a':ceilings,
                prefix+'_minimum_positive_log_residual':deficits})
        report['status'] = 'completed_saved_array_forward_transfer_barrier_no_new_roots_ODE_or_fit'
    except Exception as exc:
        report.update(status='failed_or_partial_saved_array_forward_transfer_barrier_preserved',
                      error_type=type(exc).__name__, error=str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        signal.signal(signal.SIGALRM, previous_handler)
        report['compute_elapsed_s'] = time.monotonic()-begun
        report['source_files_unchanged'] = {path: sha(path) == value for path, value in hashes.items()}
        if not all(report['source_files_unchanged'].values()):
            report['status'] = 'invalid_source_changed_during_algebraic_check'
        dump(dest, report)
        np.savez_compressed(dest.with_suffix('.npz'), **arrays)
    lines = ['# 正向端到端事件流修正：保存数组屏障复算', '',
             f'状态：{report["status"]}；计算 {report["compute_elapsed_s"]:.6f} s。', '',
             f'温度保持现有谱表 {temperature:g} K，V_T={thermal_v:.17g} V。'
             '两端同温、同弹性传输核下，I₊=I/[1−exp(−v/V_T)]；这不是局部陷阱捕获通量。', '',
             '只把保存格点的捕获上界乘 C(vmin)，排出项不变，重算数值缓冲。'
             'λ/J、qbar、谱和晶体管保持原值；无新求根、反演、谱积分、ODE、拟合或扫描。', '',
             '| 包络 | 网格 | 最大 C | 旧/新前缀终点 V | 约束点 | 原500 RMSE 下界/dec |',
             '|---|---|---:|---|---:|---:|']
    for case in report['cases']:
        lines.append(f'| {case["material_envelope"]} | {case["grid"]["name"]} | '
            f'{case["correction_maximum"]:.12g} | {case["original_prefix_end_v"]:.5f} / '
            f'{case["certified_prefix_end_v"]:.5f} | {case["certified_original_native_points"]} | '
            f'{case["optimistic_500point_RMSE_floor_decade"]:.9f} |')
    lines.extend(['', '新前缀只与原保存的原生认证点取交集；电流上界直接复用，未扩展保存域。'
                  '其余原500点评分误差乐观设零，总分母不变。', '',
                  '若下界仍超过旧条件结果，此检查只排除固定种子中这一个端到端通量替换作为减误差补救。'
                  '它不排除未建模的局部注入/陷阱动力学，也不允许把 q 称为测得占据率。'])
    if 'error' in report:
        lines.extend(['', report['error_type']+': '+report['error']])
    dest.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=report['status'], compute_elapsed_s=report['compute_elapsed_s'],
                         temperature_k=temperature, thermal_voltage_v=thermal_v,
                         cases=report['cases']), indent=2))


if __name__ == '__main__':
    main()
