"""Synthetic history/scoring validation plus at most10 material-current probes.

Never invokes a real trajectory solver or optimization. Hard90-second budget.
"""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import signal
import time
from unittest.mock import patch

import numpy as np

from segmented_joint_problem import (ROOT, TABLE_PATH, TRAIN, SEGMENTS,
                                     SegmentedSpectralJointProblem, make_problem)
from joint_problem import SpectralJointProblem
from ordered_objective import DATA, OrderedJointProblem
from vulcan2d.shared_spectral import SpectralShape
from vulcan2d.shared_spectral_ivp import SpectralTable

OUT = Path(__file__).resolve().parent


def main():
    started = time.monotonic()

    def timeout(signum, frame):
        raise TimeoutError('90-second validation budget exceeded')

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(90)
    calibration = json.loads((OUT/'RON_CALIBRATION_SCOPE.json').read_text())
    ron = calibration['predeclared_Ron_sensitivity_points'][1]['Ron_ohm']
    q = np.array([-1., -3., -3., 0., 0., .25, .75, 1.25])
    expected_keys = {f'{top}_cycle_{cid}_{phase}_{suffix}'
                     for top, ids in TRAIN.items() for cid in ids
                     for phase in ['set', 'reset'] for suffix in ['voltage', 'current']}
    assert len(expected_keys) == 184
    loads, calls = [], []
    original_load = np.load

    class GuardedTrainingArchive:
        def __init__(self, archive):
            self.archive, self.keys_read = archive, []
            loads.append(self.keys_read)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.archive.close()

        def __getitem__(self, key):
            if key not in expected_keys:
                raise AssertionError('Attempted non-training observation access: '+str(key))
            self.keys_read.append(key)
            return self.archive[key]

    def guarded_load(path, *args, **kwargs):
        archive = original_load(path, *args, **kwargs)
        return GuardedTrainingArchive(archive) if Path(path).resolve() == DATA.resolve() else archive

    def synthetic_simulator(command, initial_u, law, circuit, **options):
        # Artificial trace for wiring only; it is deliberately not a physical
        # SET/RESET integration or a claim about fit accuracy.
        phase = 'reset' if np.min(command) < 0 else 'set'
        increment = .011 if phase == 'reset' else .01
        end = float(initial_u + increment)
        calls.append({'initial_u': float(initial_u), 'final_u': end,
                      'phase': phase, 'law_id': id(law), 'exact_law_id': id(law.exact_law),
                      'circuit_id': id(circuit), 'Ron_ohm': law.shape.ron_ohm,
                      'initial_command_v': options['initial_command_voltage_v']})
        current = np.sign(command) * (1e-8 * (1 + abs(command)) * (1 + .1*initial_u))
        return {'current_a': current, 'state_u': np.linspace(initial_u, end, len(command)),
                'final_u': end, 'validation_only_synthetic_trace': True}

    with patch('numpy.load', side_effect=guarded_load):
        problem = make_problem(ron, simulate_fn=synthetic_simulator)
        # Original loader/scorer is the reference; its old history evaluation
        # is NEVER called. Its factory is also never called here.
        baseline = SpectralJointProblem(problem.law_factory, mode='full',
                                         simulate_fn=synthetic_simulator)
    assert len(loads) == 2
    assert all(len(k) == 184 and set(k) == expected_keys for k in loads)
    assert not calls
    assert problem.n_parameters == 8 and problem.n_residuals == baseline.n_residuals == 27023
    assert problem.training_arrays_sha256 == baseline.training_arrays_sha256
    assert SegmentedSpectralJointProblem.score_predictions is OrderedJointProblem.score_predictions
    group_checks = []
    for key, groups in problem.groups.items():
        assert np.array_equal(problem.raw_training[key], baseline.raw_training[key])
        assert np.array_equal(problem.log_observed[key], baseline.log_observed[key])
        for a, b in zip(groups, baseline.groups[key]):
            assert a.keys() == b.keys()
            for field in a:
                assert np.array_equal(a[field], b[field]) if isinstance(a[field], np.ndarray) else a[field] == b[field]
            group_checks.append({'topology': key[0], 'phase': key[1], 'name': a['name'],
                                 'n': a['n'], 'normalization': float(a['normalization']),
                                 'mask_sha256': hashlib.sha256(a['mask'].tobytes()).hexdigest()})
    frozen_paths = [Path(p) for p in problem.checkpoint_metadata()['source_sha256'] if Path(p).resolve() != Path(__file__).resolve()]
    frozen_before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen_paths}
    factory = problem.law_factory
    factory_calls, created_laws = [], []

    def counted_factory(vector, actual_ron):
        factory_calls.append((np.asarray(vector).copy(), actual_ron))
        law = factory(vector, actual_ron)
        created_laws.append(law)
        return law

    problem.law_factory = counted_factory
    result = problem.evaluate(q)
    assert len(factory_calls) == 1 and len(calls) == 92
    assert len({r['law_id'] for r in calls}) == len({r['exact_law_id'] for r in calls}) == 1
    assert all(r['Ron_ohm'] == ron and r['initial_command_v'] == 0. for r in calls)
    assert all(r['phase'] == ('set' if j % 2 == 0 else 'reset') for j, r in enumerate(calls))
    expected_offsets = [0, 6, 28]
    for (name, topology, labels, initial_index), offset in zip(SEGMENTS, expected_offsets):
        assert calls[offset]['initial_u'] == q[initial_index]
        end = offset + 2*len(labels)
        for j in range(offset+1, end):
            assert calls[j]['initial_u'] == calls[j-1]['final_u']
    assert calls[6]['initial_u'] != calls[5]['final_u'], 'Must not bridge missing bare cycle4'
    assert calls[28]['initial_u'] != calls[27]['final_u'], 'Independent series specimen history must start at its own state'
    assert len(result['predictions']) == len({id(v) for v in result['predictions'].values()}) == 92
    original_residual, original_blocks = baseline.score_predictions(result['predictions'])
    assert np.array_equal(result['residual'], original_residual)
    assert result['details']['blocks'] == original_blocks
    assert not result['details']['scatter_statistic_subtracted_from_cost']
    assert 'model_dependent_cost_after_subtracting_fixed_scatter' not in result['details']
    assert np.isclose(result['details']['verified_cost'], np.dot(original_residual, original_residual)/2., rtol=0, atol=0)
    # Fail before invoking any simulator on invalid/old interface requests.
    rejects = []
    for name, action in [
        ('periodic_evaluate', lambda: problem.evaluate(q, mode='periodic')),
        ('periodic_factory', lambda: make_problem(ron, mode='periodic')),
        ('periodic_comparison', lambda: problem.compare_modes(q)),
        ('periodic_export', lambda: problem.export_training_evaluation(q, OUT/'SHOULD_NOT_EXIST.json', compare_modes=True)),
        ('old_seven_coordinate_vector', lambda: problem.evaluate(q[:7])),
        ('negative_segment_initial', lambda: problem.evaluate(np.r_[q[:6], -1., q[7]])),
    ]:
        try:
            action()
        except ValueError:
            rejects.append(name)
        else:
            raise AssertionError('Invalid interface request accepted: '+name)
    assert len(calls) == 92 and not (OUT/'SHOULD_NOT_EXIST.json').exists()
    problem.law_factory = factory

    # At most10 actual material-current points, with no trajectory evaluation.
    base_shape = SpectralShape(**problem.base_spectral_shape)
    scenario_shape = SpectralShape(**problem.scenario_spectral_shape)
    assert {k for k in asdict(base_shape) if asdict(base_shape)[k] != asdict(scenario_shape)[k]} == {'ron_ohm'}
    factor = base_shape.ron_ohm / ron
    original_current = SpectralShape.current_at_log_gamma
    actual_current_calls, probes = [], []

    def tracked_current(shape, voltage, log_gamma, **kwargs):
        count = int(np.broadcast(np.asarray(voltage), np.asarray(log_gamma)).size)
        actual_current_calls.extend([1]*count)
        assert len(actual_current_calls) <= 10
        return original_current(shape, voltage, log_gamma, **kwargs)

    with patch.object(SpectralShape, 'current_at_log_gamma', tracked_current):
        for voltage, gamma in [(.2, 1e-5), (-.2, 1e-5), (2., .03), (-2., .03)]:
            old = base_shape.current_at_gamma(voltage, gamma, order=32)
            new = scenario_shape.current_at_gamma(voltage, gamma, order=32)
            error = abs(new/(old*factor)-1)
            assert error < 1e-12
            probes.append({'voltage_v': voltage, 'same_gamma_ev': gamma,
                           'old_Ron_current_a': old, 'new_Ron_current_a': new,
                           'expected_factor': factor, 'relative_scaling_error': error,
                           'kind': 'actual_frozen_shape_spectral_quadrature_at_both_Ron'})
        # Outside the cached gamma range, the adapter must fall back to the
        # actual NEW law. This adds two material evaluations, for a total10.
        law = created_laws[0]
        voltage, gamma = -.35, 1e7
        u = math.log1p(law.exact_law.gamma_c_ev/gamma)
        assert law._profile(u, voltage) is None
        old = base_shape.current_at_gamma(voltage, gamma, order=32)
        new = law.current(voltage, u)
        error = abs(new/(old*factor)-1)
        assert error < 1e-12
        probes.append({'voltage_v': voltage, 'same_gamma_ev': gamma,
                       'old_Ron_current_a': old, 'new_Ron_current_a': new,
                       'expected_factor': factor, 'relative_scaling_error': error,
                       'kind': 'table_outside_gamma_fallback_broad_limit'})
        for voltage in [.2, -.2]:
            assert law.current(voltage, 0.) == voltage/ron
    assert len(actual_current_calls) == 10
    original_table = SpectralTable.load(TABLE_PATH, base_shape)
    shift = math.log(factor)
    assert np.array_equal(original_table.log_gamma, problem.transport_table.log_gamma)
    assert np.array_equal(original_table.log_voltage, problem.transport_table.log_voltage)
    assert np.array_equal(original_table.log_current + shift, problem.transport_table.log_current)
    assert original_table.metadata == problem.transport_table.metadata['source_table_metadata_unmodified']
    metadata = problem.checkpoint_metadata()
    assert metadata['actual_transport_Ron_ohm'] == ron
    assert metadata['ron_from_training_ohm'] == metadata['legacy_training_median_Ron_ohm'] == base_shape.ron_ohm
    assert metadata['n_parameters'] == 8 and not metadata['include_reset_initiation']
    assert all(frozen_before[str(p)] == hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen_paths)
    report = {
        'status': 'passed_synthetic_segmented_wiring_and_limited_amplitude_identity_validation',
        'real_ODE_trajectories_run': 0, 'optimizations_run': 0,
        'validation_q8_not_a_fit_seed': q.tolist(),
        'external_fixed_Ron_scenario_ohm': ron,
        'synthetic_phase_calls': len(calls), 'synthetic_pair_calls': len(calls)//2,
        'material_factory_calls_during_evaluation': len(factory_calls),
        'unique_tabulated_material_objects': len({r['law_id'] for r in calls}),
        'unique_underlying_SpectralLaw_objects': len({r['exact_law_id'] for r in calls}),
        'segment_phase_offsets': expected_offsets,
        'synthetic_history_segments': result['details']['history_segments'],
        'n_residuals': len(result['residual']),
        'original_masks_weights_raw_arrays_and_log_arrays_exactly_equal': True,
        'same_predictions_original_scorer_residual_bitwise_equal': True,
        'legacy_scatter_not_subtracted': True,
        'group_checks': group_checks,
        'training_access': {'loader_instances': len(loads), 'arrays_per_loader': [len(k) for k in loads],
                            'allowed_unique_array_count': len(expected_keys),
                            'all_accesses_training_only': True,
                            'allowed_keys_sha256': hashlib.sha256('\n'.join(sorted(expected_keys)).encode()).hexdigest()},
        'rejected_invalid_interfaces_before_any_new_phase_call': rejects,
        'actual_material_current_point_evaluations': len(actual_current_calls),
        'actual_quadrature_points': 8,
        'broad_limit_fallback_points': 2,
        'spectral_amplitude_probes': probes,
        'u_zero_positive_and_negative_Ron_paths_checked_algebraically': True,
        'table_shift_all_entries_exactly_equal_to_declared_expression': True,
        'original_frozen_sources_unchanged': True,
        'metadata': metadata,
        'scope': [
            'Synthetic traces validate object identity, record ordering, initial-state routing and scorer reuse only; they are not physical trajectories or fit results.',
            'All92 phases are called freshly; no record prediction is reused and no missing cycle is fabricated.',
            'Ten material-current evaluations test the exact1/Ron identity, with the same spectrum/Gamma at old/newRon. They do not establish new spectral accuracy, series load accuracy, ODE accuracy or joint model adequacy.',
            'External transistor code/data and all prior sources remain unchanged; actual46-pair training propagation has not been performed.',
        ],
        'validation_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_sha256': {**metadata['source_sha256'], str(OUT/'RON_CALIBRATION_SCOPE.json'): hashlib.sha256((OUT/'RON_CALIBRATION_SCOPE.json').read_bytes()).hexdigest()},
        'elapsed_s': time.monotonic()-started,
    }
    assert report['elapsed_s'] < 90
    signal.alarm(0)
    (OUT/'validate_segmented_joint_problem.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    lines = ['# 连续分段联合spectral接口：实现与有限验证', '',
             '**接口已通过合成历史/评分验证：同一个材料对象用于92个相位，裸14圈与串联32圈均逐圈调用；真实46对动力学尚未运行。**', '',
             '## 接口约定', '',
             '`make_problem(ron_ohm)` 把Ron作为外部固定场景；q8依次为5个log10材料参数及 `u_裸1、u_裸5、u_串联4`。五个材料参数在一次评估的全部圈中保持相同，三个初态仅表示已观测段的边界状态，不是三套材料定律。beta=1、A_RESET=0固定。',
             '裸历史为1–3及5–15两段；第4圈实际测过但缺失，故第5圈起态显式给出，不能沿第3圈终态直接接上。串联4–35连续传播；两个拓扑都使用同一个Ron、同一材料对象。',
             '只支持full；周期复用、compare_modes和带周期比较的导出在调用模拟器前即拒绝。未修改外部晶体管、原模型源文件或旧结果。', '',
             '## 合成验证结果', '',
             f'- 材料工厂调用1次；外层tabulated对象及内层SpectralLaw对象各1个，供92次相位调用共用。',
             '- 三段入口位于相位偏移0、6、28，对应q[5]、q[6]、q[7]；段内SET→RESET及跨圈状态均取上一相位终态。',
             '- 92份预测各自生成，不复用同一pair；缺第4圈与独立串联初态之间均不错误接续。',
             '- 184个训练数组访问被白名单逐项检查；原接口与新接口各加载一次，均没有保留集访问。',
             '- 原27023维残差、掩码、记录顺序、分组权重/归一化及原始观测数组完全一致；给同一组合成预测，原评分器输出逐位相同。',
             '- 跨记录预测不同后，不再把旧散布常数从cost中扣除；它只作为旧统计量保存。', '',
             '## Ron缓存与实际谱幅度验证', '',
             f'本次验证场景Ron={ron:.12f} Ω，旧训练中位Ron={base_shape.ron_ohm:.12f} Ω。metadata分别保存旧校准值与实际输运值，Ron不进入q8。',
             '先用旧shape载入旧表并核对metadata，再仅用dataclasses.replace改变Ron；新表复制原坐标，两个极性同时做自然对数变换 `logI新=logI旧+ln(R旧/R新)`。保存原表hash和原metadata，旧文件不写回。',
             '实际谱检查使用相同Gamma与谱形状，在±0.2 V、Gamma=1e−5 eV，以及±2 V、Gamma=0.03 eV处分别计算旧/新Ron，共8个积分点；另用2个表外宽谱点检查fallback确实使用新Ron，总计10个材料电流点。正负电压u=0路径另作代数检查，不新增谱点。',
             f'最大幅度比相对误差为{max(p["relative_scaling_error"] for p in probes):.3g}。这是1/Ron幅度恒等式验证，不是新的谱积分精度认证，也没有验证真实串联负载或46对轨迹。', '',
             f'验证用时{report["elapsed_s"]:.3f}秒，设置90秒上限；真实ODE次数0、优化次数0。全部来源hash、组掩码hash、段状态链及测试范围见同名JSON。']
    (OUT/'validate_segmented_joint_problem.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'elapsed_s': report['elapsed_s'], 'synthetic_phase_calls': len(calls),
                      'n_residuals': len(result['residual']), 'actual_material_current_points': len(actual_current_calls),
                      'max_relative_amplitude_error': max(p['relative_scaling_error'] for p in probes),
                      'real_ODE_trajectories': 0}, indent=2))


if __name__ == '__main__':
    main()
