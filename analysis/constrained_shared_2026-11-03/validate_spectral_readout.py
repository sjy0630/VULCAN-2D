"""Bounded local refit and independent validation of a FROZEN readout only.

The calibration objective retains the original observed-current deembedding.
After parameter freeze, predictions use command voltage and the measured
transistor law ONLY. An independent Voigt/Fermi energy integral verifies the
Hermite/digamma readout and selected scalar circuit roots. No kinetics, state
reset, holdout evaluation or complete model claim occurs here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, least_squares
from scipy.special import expit, voigt_profile

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from vulcan2d.transistor import measured_transistor_mapping
from vulcan2d.shared_gap import LookupGapCircuit
from spectral_readout import spectral_current

DATA = ROOT/'analysis/shared_hbn_2026-11-03/data/native_observations.npz'
SCREEN = OUT/'spectral_readout_screen.json'
RESULT = OUT/'spectral_readout_validation.json'
IDS = list(range(4, 36))
T_K = 300.
THERMAL = 8.617333262145e-5*T_K


def write(value):
    temporary = RESULT.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(RESULT)


def quantile(values, levels=(0., .5, 1.)):
    return np.quantile(values, levels).tolist()


def current(q, v, ron, order):
    return spectral_current(v, 10.**q[0], q[1], q[3], q[2], ron,
                            temperature_k=T_K, order=order)


def load_groups(mapping, old):
    groups = {}
    with np.load(DATA) as raw:
        for phase in ('set', 'reset'):
            volts = [raw[f'1t1r_cycle_{c}_{phase}_voltage'] for c in IDS]
            assert all(np.array_equal(v, volts[0]) for v in volts)
            v = volts[0]
            observations = abs(np.stack([raw[f'1t1r_cycle_{c}_{phase}_current'] for c in IDS]))
            apex = int(np.argmax(abs(v)))
            index = np.arange(len(v))
            use = ((index > apex) & (v >= .06)) if phase == 'set' else (
                (index <= apex) & (abs(v) >= .06) & (abs(v) <= .30))
            v, observed = v[use].copy(), observations[:, use].copy()
            if np.any(observed <= 0.):
                raise ValueError('Selected raw current must be positive without a floor')
            geometric = 10.**np.mean(np.log10(observed), axis=0)
            vt = np.array([mapping.voltage_for_current(float(i), 1.1, side=phase) for i in geometric])
            vh = np.sign(v)*(abs(v)-vt)
            if np.any(abs(v)-vt <= 0.):
                raise ValueError('Nonpassive deembedded calibration input')
            group = {'command': v, 'observed': observed, 'geometric': geometric,
                     'hbn_observed_current_deembedding': vh, 'native_indices': np.flatnonzero(use)}
            old_group = old['data'][phase]
            assert np.array_equal(v, np.array(old_group['command']))
            assert np.array_equal(observed, np.array(old_group['observed']))
            assert np.array_equal(geometric, np.array(old_group['current']))
            assert np.array_equal(vh, np.array(old_group['hbn']))
            groups[phase] = group
    return groups


def conditional(q, groups, ron, order):
    residuals, predictions = [], {}
    for phase, d in groups.items():
        pred = abs(current(q, d['hbn_observed_current_deembedding'], ron, order))
        if np.any(pred <= 0.):
            raise ValueError('Nonpositive spectral current')
        error = np.log10(pred/d['geometric'])
        residuals.append(error/np.sqrt(len(error)))
        predictions[phase] = pred
    return np.concatenate(residuals), predictions


def independent_integral(q, voltage, ron):
    """No Hermite nodes or digamma: Gaussian/Lorentz convolution is a Voigt.

    Gaussian-averaged transmission = pi*gamma*Voigt(E-center,w,gamma).
    Fermi tails outside the integration interval contribute <2 exp(-40) times
    pi*gamma/Ron. Independent adaptive energy quadrature handles both signs.
    """
    gamma, center, width = 10.**q[0], q[1]+q[2]*voltage, q[3]
    if voltage == 0.:
        return 0., 0.
    lower, upper = min(0., voltage)-40*THERMAL, max(0., voltage)+40*THERMAL
    coefficient = np.pi*gamma/ron
    def integrand(energy):
        occupation = expit((voltage-energy)/THERMAL)-expit(-energy/THERMAL)
        return coefficient*voigt_profile(energy-center, width, gamma)*occupation
    points = sorted({lower, upper, *[x for x in (0., voltage, center)
                                      if lower < x < upper]})
    total = error = 0.
    for a, b in zip(points[:-1], points[1:]):
        value, estimate = quad(integrand, a, b, epsabs=1e-16, epsrel=2e-11, limit=200)
        total += value
        error += estimate
    return total, error+coefficient*2*np.exp(-40.)


def solve_command(q, command, phase, ron, circuit, order):
    """Autonomous voltage-node bisection: no observation argument exists."""
    v = np.asarray(command)
    sign, magnitude = np.sign(v), abs(v)
    knots, currents = circuit._branches[phase]
    if np.max(magnitude) > knots[-1]:
        raise ValueError('Requested command exceeds transistor voltage domain')
    low, high = np.zeros_like(magnitude), magnitude.copy()
    for _ in range(56):
        vt = .5*(low+high)
        ih = abs(current(q, sign*(magnitude-vt), ron, order))
        it = np.interp(vt, knots, currents)
        move_right = ih > it
        low = np.where(move_right, vt, low)
        high = np.where(move_right, high, vt)
    vt = .5*(low+high)
    vh = sign*(magnitude-vt)
    ih = current(q, vh, ron, order)
    # Canonical direct law is independently called, rather than reusing interp.
    it = np.array([circuit.mapping.drain_current(float(s*x), 1.1)
                   for s, x in zip(sign, vt)])
    return {'predicted_signed_current_A': ih, 'hbn_voltage_V': vh,
            'transistor_voltage_V': sign*vt, 'canonical_transistor_current_A': it,
            'KCL_residual_A': ih-it, 'KVL_residual_V': v-vh-sign*vt}


def errors(pred, group):
    mean_errors = np.log10(abs(pred)/group['geometric'])
    raw_errors = np.log10(abs(pred)[None, :]/group['observed'])
    scatter = np.log10(group['observed']/group['geometric'][None, :])
    mean_mse, raw_mse, scatter_mse = np.mean(mean_errors**2), np.mean(raw_errors**2), np.mean(scatter**2)
    q10, q90 = np.quantile(group['observed'], [.1, .9], axis=0)
    return {'geometric_mean_RMSE_decade': float(np.sqrt(mean_mse)),
            'all_32_raw_records_RMSE_decade': float(np.sqrt(raw_mse)),
            'raw_record_RMSE_min_median_max': quantile(np.sqrt(np.mean(raw_errors**2, axis=1))),
            'raw_scatter_about_geometric_mean_RMS_decade': float(np.sqrt(scatter_mse)),
            'raw_MSE_minus_mean_MSE_minus_scatter_MSE': float(raw_mse-mean_mse-scatter_mse),
            'maximum_geometric_mean_abs_error_decade': float(np.max(abs(mean_errors))),
            'fraction_points_inside_raw_Q10_Q90': float(np.mean((abs(pred) >= q10) & (abs(pred) <= q90))),
            'per_record_RMSE_decade': np.sqrt(np.mean(raw_errors**2, axis=1)).tolist()}


def feature_rows(groups, circuit_predictions):
    rows = []
    for phase, d in groups.items():
        v = abs(d['command'])
        pred = abs(circuit_predictions[phase]['predicted_signed_current_A'])
        targets = (.06, .1, .2, .3, 1., 2., 4., 5.) if phase == 'set' else (.06, .1, .2, .3)
        for target in targets:
            ix = int(np.argmin(abs(v-target)))
            assert abs(v[ix]-target) < 1e-12
            values = d['observed'][:, ix]
            low, med, high = np.quantile(values, [.1, .5, .9])
            rows.append({'phase': phase, 'feature': f'current_at_abs_command_{target}V',
                'predicted_A': float(pred[ix]), 'geometric_mean_observed_A': float(d['geometric'][ix]),
                'raw_Q10_median_Q90_A': [float(low), float(med), float(high)],
                'inside_empirical_Q10_Q90': bool(low <= pred[ix] <= high)})
        pairs = ((.1, .2), (1., 2.), (2., 4.)) if phase == 'set' else ((.1, .2),)
        for first, second in pairs:
            a, b = [int(np.argmin(abs(v-target))) for target in (first, second)]
            denom = np.log(second/first)
            values = np.log(d['observed'][:, b]/d['observed'][:, a])/denom
            low, med, high = np.quantile(values, [.1, .5, .9])
            predicted = float(np.log(pred[b]/pred[a])/denom)
            rows.append({'phase': phase, 'feature': f'alpha_{first}V_to_{second}V',
                'predicted': predicted,
                'geometric_mean_observed': float(np.log(d['geometric'][b]/d['geometric'][a])/denom),
                'raw_Q10_median_Q90': [float(low), float(med), float(high)],
                'inside_empirical_Q10_Q90': bool(low <= predicted <= high)})
    return rows


def main():
    started = time.monotonic()
    old = json.loads(SCREEN.read_text())
    selected = next(f for f in old['families']['gaussian_ensemble']['fits'] if f['start'] == 1)
    initial = np.array(selected['q'])
    ron = selected['parameters']['ron_ohm']
    mapping = measured_transistor_mapping()
    groups = load_groups(mapping, old)
    circuit = LookupGapCircuit(mapping, 1.1)
    output = {'status': 'running_bounded_512_node_local_refit',
              'training_ids': IDS, 'original_start': 1, 'original_q': initial.tolist(),
              'fitted_scope': {'set': 'return command 0.06 through 5 V',
                               'reset': 'outward command -0.06 through -0.30 V'},
              'max_nfev': 60, 'checkpoints': []}
    calls, best = 0, float('inf')
    def objective(q):
        nonlocal calls, best
        residual = conditional(q, groups, ron, 512)[0]
        cost = float(residual@residual/2.)
        calls += 1
        if cost < best:
            best = cost
            output['best_seen'] = {'q': q.tolist(), 'cost': cost, 'actual_function_calls': calls}
        if calls == 1 or calls % 5 == 0:
            output['checkpoints'].append({'call': calls, 'cost': cost, 'best_cost': best,
                                          'elapsed_s': time.monotonic()-started})
            write(output)
        return residual
    initial_costs = {}
    for order in (512, 1024):
        residual, _ = conditional(initial, groups, ron, order)
        initial_costs[str(order)] = float(residual@residual/2.)
    fit = least_squares(objective, initial, bounds=([-7., -2., .001, 0.], [.7, 2., .999, 1.5]),
                        max_nfev=60, x_scale='jac', diff_step=1e-5,
                        ftol=1e-10, xtol=1e-10, gtol=1e-8)
    fitted = fit.x.copy()
    fit_end = time.monotonic()
    output.update(status='parameters_frozen_validating_independent_integrals_and_circuits',
                  selected_q=fitted.tolist(),
                  parameters={'gamma_ev': 10.**fitted[0], 'epsilon_ev': fitted[1],
                              'eta': fitted[2], 'width_ev': fitted[3], 'ron_ohm': ron,
                              'temperature_k': T_K},
                  local_refit={'nfev': fit.nfev, 'njev': fit.njev, 'actual_function_calls': calls,
                               'success': bool(fit.success), 'message': fit.message,
                               'optimality': float(fit.optimality), 'active_mask': fit.active_mask.tolist(),
                               'original_cost_by_order': initial_costs,
                               'elapsed_s': fit_end-started})
    write(output)
    conditional_predictions, conditional_metrics, conditional_costs = {}, {}, {}
    for order in (512, 1024):
        residual, predictions = conditional(fitted, groups, ron, order)
        conditional_predictions[order] = predictions
        conditional_metrics[str(order)] = {phase: errors(pred, groups[phase]) for phase, pred in predictions.items()}
        conditional_costs[str(order)] = float(residual@residual/2.)
    output['conditional_observed_current_deembedding'] = {
        'cost_by_order': conditional_costs, 'metrics': conditional_metrics,
        'maximum_512_to_1024_current_change_decade': max(float(np.max(abs(np.log10(
            conditional_predictions[512][p]/conditional_predictions[1024][p])))) for p in groups)}
    circuit_results = {order: {phase: solve_command(fitted, d['command'], phase, ron, circuit, order)
                              for phase, d in groups.items()} for order in (512, 1024)}
    autonomous, integral_checks, roots = {}, {}, []
    for phase, d in groups.items():
        result = circuit_results[1024][phase]
        prediction = result['predicted_signed_current_A']
        autonomous[phase] = {
            **errors(prediction, d), 'n_command_points': len(d['command']),
            'n_raw_current_points': int(d['observed'].size),
            'max_abs_KCL_residual_A': float(np.max(abs(result['KCL_residual_A']))),
            'max_relative_KCL_residual': float(np.max(abs(result['KCL_residual_A']/prediction))),
            'max_abs_KVL_residual_V': float(np.max(abs(result['KVL_residual_V']))),
            'all_passive_hbn_and_transistor': bool(np.all(prediction*result['hbn_voltage_V'] >= 0.) and
                                                  np.all(prediction*result['transistor_voltage_V'] >= 0.)),
            'maximum_512_to_1024_current_change_decade': float(np.max(abs(np.log10(
                abs(circuit_results[512][phase]['predicted_signed_current_A']/prediction))))),
            'command_voltage_V': d['command'].tolist(),
            **{key: value.tolist() for key, value in result.items()}}
        points = np.concatenate([d['hbn_observed_current_deembedding'], result['hbn_voltage_V']])
        integral = np.array([independent_integral(fitted, float(v), ron) for v in points])
        hermite = current(fitted, points, ron, 1024)
        discrepancy = abs(np.log10(abs(hermite/integral[:, 0])))
        integral_checks[phase] = {'n_verified_internal_voltages': len(points),
            'max_Hermite1024_vs_independent_Voigt_integral_decade': float(max(discrepancy)),
            'max_quadrature_error_estimate_A': float(max(integral[:, 1])),
            'internal_voltage_V': points.tolist(), 'independent_current_A': integral[:, 0].tolist(),
            'independent_quadrature_error_estimate_A': integral[:, 1].tolist(),
            'per_point_current_change_decade': discrepancy.tolist()}
        desired = (.06, .1, .2, .3, .5, 1., 2., 3., 4., 5.) if phase == 'set' else (.06, .1, .2, .3)
        for target in desired:
            ix = int(np.argmin(abs(abs(d['command'])-target)))
            command = float(d['command'][ix])
            sign, mag = np.sign(command), abs(command)
            def equation(vt):
                ih = abs(independent_integral(fitted, sign*(mag-vt), ron)[0])
                it = abs(mapping.drain_current(sign*vt, 1.1))
                return ih-it
            vt = brentq(equation, 0., mag, xtol=1e-13, rtol=1e-13)
            independent_i = independent_integral(fitted, sign*(mag-vt), ron)[0]
            roots.append({'phase': phase, 'command_V': command,
                'independent_Vt_V': float(sign*vt), 'independent_current_A': float(independent_i),
                'current_difference_decade': float(abs(np.log10(abs(independent_i/prediction[ix])))),
                'Vt_difference_V': float(sign*vt-result['transistor_voltage_V'][ix]),
                'independent_KCL_residual_A': float(equation(vt))})
        write({**output, 'independent_integrals_partial': integral_checks,
               'autonomous_circuit_partial': autonomous})
    output['autonomous_command_only_circuit'] = autonomous
    output['independent_Voigt_Fermi_energy_integrals'] = integral_checks
    output['independent_scalar_KCL_roots'] = roots
    output['key_features'] = feature_rows(groups, circuit_results[1024])
    checks = {
        '512_to_1024_current_change_below_1e_minus_5_decade': all(
            row['maximum_512_to_1024_current_change_decade'] < 1e-5 for row in autonomous.values()),
        'independent_energy_integral_difference_below_1e_minus_7_decade': all(
            row['max_Hermite1024_vs_independent_Voigt_integral_decade'] < 1e-7
            for row in integral_checks.values()),
        'independent_root_current_difference_below_1e_minus_7_decade': all(
            row['current_difference_decade'] < 1e-7 for row in roots),
        'canonical_transistor_KCL_below_1e_minus_13_A': all(
            row['max_abs_KCL_residual_A'] < 1e-13 for row in autonomous.values()),
        'all_operating_points_passive': all(row['all_passive_hbn_and_transistor'] for row in autonomous.values()),
        'raw_MSE_decomposition_residual_below_1e_minus_12': all(
            abs(row['raw_MSE_minus_mean_MSE_minus_scatter_MSE']) < 1e-12 for row in autonomous.values()),
    }
    output['selected_range_numerical_checks'] = {'all_passed': all(checks.values()), 'checks': checks,
                                                'scope': 'selected frozen parameters and evaluated voltages only'}
    output['aggregate_scores'] = {
        'phase_balanced_geometric_mean_RMSE_decade': float(np.sqrt(np.mean([
            row['geometric_mean_RMSE_decade']**2 for row in autonomous.values()]))),
        'phase_balanced_raw_record_RMSE_decade': float(np.sqrt(np.mean([
            row['all_32_raw_records_RMSE_decade']**2 for row in autonomous.values()]))),
        'pooled_all_raw_points_RMSE_decade': float(np.sqrt(sum(
            row['all_32_raw_records_RMSE_decade']**2*row['n_raw_current_points'] for row in autonomous.values())/
            sum(row['n_raw_current_points'] for row in autonomous.values()))),
        'n_empirical_key_feature_checks': len(output['key_features']),
        'n_inside_empirical_Q10_Q90': sum(row['inside_empirical_Q10_Q90'] for row in output['key_features'])}
    output['limitations'] = [
        'Local refit from Gaussian start 1 only; no global optimality or microscopic mechanism claim.',
        'Calibration still uses observed-current deembedding and equal phase weights; it is not autonomous-circuit least squares.',
        'Autonomous validation freezes all parameters and supplies only command voltages to series KCL. Observations enter scores only.',
        'One shared frozen effective LRS is used across the selected SET-return and low-bias RESET-outward segments and across all training records.',
        'Gamma_c and u are not separately identified: only this frozen gamma is estimated. No state dynamics, per-cycle hidden-state reset or chronology prediction is implemented.',
        'Only SET return >=0.06 V and RESET outward 0.06–0.30 V are validated. No high-bias RESET plateau, SET outward, bare HRS or full-device fit is certified.',
        '300 K, fixed source voltage reference, and measured-transistor mapping remain explicit assumptions; same-cell transistor identity is unresolved.',
        'Ron is the unchanged inherited training-only reference. Its finite-voltage estimate is not a direct zero-bias conductance measurement.',
        'Raw scatter is retained. Q10–Q90 bands are empirical training ranges, not confidence intervals or independent test coverage.',
        'Numerical verification applies only to the selected frozen parameter set and sampled voltage range; earlier high-width quadrature failures are not cleared.',
    ]
    output['elapsed_s'] = time.monotonic()-started
    output['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__), SCREEN, OUT/'spectral_readout.py', DATA,
                  ROOT/'vulcan2d/transistor.py', ROOT/'vulcan2d/shared_gap.py',
                  ROOT/'vulcan2d/data/fig2b_transistor_lookup.json',
                  ROOT/'vulcan2d/data/bookoutput2_reverse_transistor_lookup.json')}
    output['original_screen_core_source_hashes_match'] = {
        name: output['source_sha256'][name] == digest for name, digest in old['source_sha256'].items()
        if name in output['source_sha256']}
    output['status'] = 'completed_selected_frozen_readout_validation_not_a_switching_model'
    write(output)
    lines = ['# 冻结谱读出：数值加密与自主串联电路验证', '',
             '只验证一个共同冻结低阻态；没有状态动力学或完整开关预测。', '',
             '## 范围与复拟合', '',
             '训练第 4–35 圈；SET 返扫命令 0.06–5 V，RESET 外扫命令 −0.06 至 −0.30 V。',
             '从原 Gaussian start 1 出发，512 节点局部复拟合；原筛选和源码未修改。',
             f'实际 nfev={fit.nfev}，函数调用 {calls} 次，终止：{fit.message}，optimality={fit.optimality:.9g}。',
             f'原候选 512 节点代价 {initial_costs["512"]:.12g} → {conditional_costs["512"]:.12g}。',
             '此代价仍是观测电流去嵌入后的条件拟合目标，不是自主端口拟合目标。',
             '', '冻结参数：', '', '```json', json.dumps(output['parameters'], indent=2), '```', '',
             '## 自主电路结果', '',
             '参数冻结后只输入命令电压，以 h-BN 谱电流与独立晶体管电流相等求节点电压。',
             '预测函数没有观测电流入口；观测仅在完成求解后用于评分。', '',
             '| 区间 | 几何平均曲线 RMSE / decade | 全部原始记录 RMSE / decade | 原始散布 RMS / decade |',
             '| --- | ---: | ---: | ---: |']
    for phase, row in autonomous.items():
        lines.append(f'| {phase} | {row["geometric_mean_RMSE_decade"]:.9g} | '
                     f'{row["all_32_raw_records_RMSE_decade"]:.9g} | '
                     f'{row["raw_scatter_about_geometric_mean_RMS_decade"]:.9g} |')
    lines += ['', '原记录 MSE = 几何平均偏差 MSE + 记录散布 MSE，JSON 记录了数值恒等式残差。',
              '散布不是已证明的不可约测量噪声，冻结态模型也未解释周期之间的变化。', '',
              '## 独立数值核验', '']
    for phase, row in autonomous.items():
        lines.append(f'- {phase}：自主电流 512→1024 最大变化 '
                     f'{row["maximum_512_to_1024_current_change_decade"]:.9g} decade；'
                     f'1024 节点对独立 Voigt/Fermi 能量积分最大差 '
                     f'{integral_checks[phase]["max_Hermite1024_vs_independent_Voigt_integral_decade"]:.9g} decade；'
                     f'KCL 最大绝对残差 {row["max_abs_KCL_residual_A"]:.9g} A。')
    lines += [f'- 另用独立能量积分与标量 Brent 求根验证 {len(roots)} 个命令点；最大电流差 '
              f'{max(r["current_difference_decade"] for r in roots):.9g} decade。',
              '- Voigt 积分不使用 Hermite 节点或 digamma，与被检实现独立；所有节点均保持无源。', '',
              '## 关键特征', '',
              '| 区间 / 特征 | 自主预测 | 原记录 Q10 / 中位 / Q90 | 在经验区间内 |',
              '| --- | ---: | --- | --- |']
    for row in output['key_features']:
        value = row.get('predicted_A', row.get('predicted'))
        levels = row.get('raw_Q10_median_Q90_A', row.get('raw_Q10_median_Q90'))
        lines.append(f'| {row["phase"]} / {row["feature"]} | {value:.9g} | '+
                     ' / '.join(f'{x:.9g}' for x in levels)+
                     f' | {"是" if row["inside_empirical_Q10_Q90"] else "否"} |')
    lines += ['', '电流单位 A；指数无量纲。经验区间是预先固定训练统计，不能等同独立检验或置信区间。', '',
              '## 结论边界', '',
              f'共 {len(output["key_features"])} 项显示检查，'
              f'{output["aggregate_scores"]["n_inside_empirical_Q10_Q90"]} 项位于训练经验区间；应逐项看失败，不能由总误差抵消。',
              '有限温度 Gaussian 谱窗在这一冻结参数点的数值精度与串联电路闭合被核验；',
              '本次仍未通过关键 SET 形状要求：高压过早饱和，2–4 V 指数明显低于训练区间。',
              '没有检验高压 RESET 平台、完整 SET 外扫、裸 HRS 或连续历史。',
              'gamma_c 与 u 未分离，未建立状态方程，也没有把谱宽解释为已测缺陷态或原子通道。',
              '正向 1.1 V 晶体管来自插值，正负资料是否同一 cell 仍未确认；300 K 是声明假设。',
              '此前宽分布起点的数值失败仍保留；本验证不把其他参数区域一并认证。', '',
              f'本次总耗时 {output["elapsed_s"]:.3f} s，其中复拟合及初始化 {fit_end-started:.3f} s。', '']
    (OUT/'spectral_readout_validation.md').write_text('\n'.join(lines))
    print(json.dumps({'status': output['status'], 'parameters': output['parameters'],
                      'aggregate_scores': output['aggregate_scores'], 'elapsed_s': output['elapsed_s'],
                      'phase_scores': {p: {k: v for k, v in r.items() if 'RMSE' in k and not isinstance(v, list)}
                                       for p, r in autonomous.items()}}, indent=2), flush=True)


if __name__ == '__main__':
    main()
