"""Same-size lognormal alternative to the failed Gaussian frozen readout.

epsilon=m*exp(sigma*Z), Z~N(0,1), m>0, sigma>=0. Only the normalized
energy ensemble changes. gamma, eta, Ron, T and the signed single-level
Landauer formula are shared across polarities. gamma(u)=gamma_c/expm1(u)
retains a monotone state order because the ensemble does not depend on u.
No offset, polarity amplitude, kinetic state or production-model change.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.integrate import quad_vec
from scipy.optimize import least_squares
from scipy.special import digamma, ndtr, ndtri, roots_hermitenorm
from functools import lru_cache

from validate_spectral_readout import (ROOT, DATA, SCREEN, IDS, T_K, THERMAL,
    load_groups, errors, feature_rows, measured_transistor_mapping, LookupGapCircuit)

OUT = Path(__file__).resolve().parent
RESULT = OUT/'lognormal_spectral_screen.json'


@lru_cache(maxsize=8)
def rule(order):
    x, w = roots_hermitenorm(order)
    return x, w/np.sum(w)


def readout(q, voltage, ron, order):
    gamma, median, eta, sigma = 10.**q[0], 10.**q[1], q[2], q[3]
    v = np.asarray(voltage, dtype=float)
    flat = v.reshape(-1)
    if sigma == 0.:
        z, weight = np.array([0.]), np.array([1.])
    else:
        z, weight = rule(order)
    levels = median*np.exp(sigma*z)
    a = .5+(gamma+1j*(levels[None, :]+eta*flat[:, None]))/(2*np.pi*THERMAL)
    b = .5+(gamma+1j*(levels[None, :]-(1-eta)*flat[:, None]))/(2*np.pi*THERMAL)
    current = gamma/ron*((digamma(a)-digamma(b)).imag@weight)
    current[flat == 0.] = 0.
    if not np.all(np.isfinite(current)) or np.any(current*flat < -1e-30):
        raise FloatingPointError('Invalid signed lognormal spectral current')
    return float(current[0]) if v.ndim == 0 else current.reshape(v.shape)


def conditional(q, groups, ron, order):
    residuals, predictions = [], {}
    for phase, d in groups.items():
        pred = abs(readout(q, d['hbn_observed_current_deembedding'], ron, order))
        if np.any(pred <= 0.):
            raise FloatingPointError('Nonpositive selected current')
        error = np.log10(pred/d['geometric'])
        residuals.append(error/np.sqrt(len(error)))
        predictions[phase] = pred
    return np.concatenate(residuals), predictions


def independent_z_integral(q, voltage, ron):
    """Adaptive normal-coordinate integral, independent of Hermite nodes.

    This shares the analytically verified single-level digamma formula; it
    checks ensemble quadrature, not that formula a second independent way.
    |Z|>10 mass times |V|/Ron bounds the omitted tails.
    """
    v = np.asarray(voltage)
    gamma, median, eta, sigma = 10.**q[0], 10.**q[1], q[2], q[3]
    def f(z):
        level = median*np.exp(sigma*z)
        a = .5+(gamma+1j*(level+eta*v))/(2*np.pi*THERMAL)
        b = .5+(gamma+1j*(level-(1-eta)*v))/(2*np.pi*THERMAL)
        return gamma/ron*(digamma(a)-digamma(b)).imag*np.exp(-z*z/2)/np.sqrt(2*np.pi)
    result, error = quad_vec(f, -10., 10., epsabs=1e-15, epsrel=1e-10, norm='max', limit=500)
    tail = 2*ndtr(-10.)*float(np.max(abs(v)))/ron
    return result, float(error+tail)


def solve_command(q, command, phase, ron, circuit, order):
    """Command-only KCL prediction; observations cannot enter this function."""
    v = np.asarray(command)
    sign, mag = np.sign(v), abs(v)
    knots, transistor = circuit._branches[phase]
    if np.max(mag) > knots[-1]:
        raise ValueError('Unmeasured transistor voltage range')
    low, high = np.zeros_like(mag), mag.copy()
    for _ in range(56):
        mid = (low+high)/2.
        ih = abs(readout(q, sign*(mag-mid), ron, order))
        it = np.interp(mid, knots, transistor)
        right = ih > it
        low, high = np.where(right, mid, low), np.where(right, high, mid)
    vt = (low+high)/2.
    vh = sign*(mag-vt)
    ih = readout(q, vh, ron, order)
    it = np.array([circuit.mapping.drain_current(float(s*t), 1.1) for s, t in zip(sign, vt)])
    return {'predicted_signed_current_A': ih, 'hbn_voltage_V': vh,
            'transistor_voltage_V': sign*vt, 'KCL_residual_A': ih-it,
            'KVL_residual_V': v-vh-sign*vt}


def write(output):
    temporary = RESULT.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    temporary.replace(RESULT)


def main():
    begin = time.monotonic()
    old = json.loads(SCREEN.read_text())
    gaussian = next(x for x in old['families']['gaussian_ensemble']['fits'] if x['start'] == 1)
    ron = gaussian['parameters']['ron_ohm']
    mapping = measured_transistor_mapping()
    circuit = LookupGapCircuit(mapping, 1.1)
    groups = load_groups(mapping, old)
    # Three declared starts; no adaptive additions after seeing failures.
    mean, width = gaussian['parameters']['epsilon_ev'], gaussian['parameters']['width_ev']
    matched_sigma = np.sqrt(np.log1p((width/mean)**2))
    matched_median = mean/np.sqrt(1+(width/mean)**2)
    starts = [[gaussian['q'][0], np.log10(matched_median), gaussian['parameters']['eta'], matched_sigma],
              [-2.9, np.log10(.65), .4, .65], [-2.8, 0., .35, 1.]]
    output = {'status': 'running_three_preset_lognormal_starts', 'training_ids': IDS,
              'definition': 'epsilon=m*exp(sigma*Z), Z standard normal; no offset, m>0, sigma>=0',
              'q_order': ['log10_gamma_eV', 'log10_median_energy_eV', 'eta', 'sigma_log_energy'],
              'preset_starts': starts, 'max_nfev_per_start': 80, 'fitting_order': 128,
              'bounds': {'lower': [-7., np.log10(.02), .001, 0.],
                         'upper': [.7, np.log10(5.), .999, 2.]},
              'ron_ohm': ron, 'temperature_k': T_K, 'fits': [], 'checkpoints': []}
    write(output)
    for index, seed in enumerate(starts):
        calls = 0
        def objective(q):
            nonlocal calls
            residual = conditional(q, groups, ron, 128)[0]
            calls += 1
            if calls % 10 == 0:
                output['checkpoints'].append({'start': index, 'function_calls': calls,
                    'cost': float(residual@residual/2.), 'elapsed_s': time.monotonic()-begin})
                write(output)
            return residual
        result = least_squares(objective, seed,
            bounds=(output['bounds']['lower'], output['bounds']['upper']),
            max_nfev=80, diff_step=1e-4, x_scale='jac', ftol=1e-9, xtol=1e-9, gtol=1e-8)
        q = result.x
        metrics, predictions, costs = {}, {}, {}
        for order in (128, 512, 1024):
            residual, prediction = conditional(q, groups, ron, order)
            predictions[order] = prediction
            costs[str(order)] = float(residual@residual/2.)
            metrics[str(order)] = {p: errors(pred, groups[p]) for p, pred in prediction.items()}
        changes = {}
        for a, b in ((128, 512), (512, 1024)):
            changes[f'{a}_to_{b}'] = max(float(np.max(abs(np.log10(predictions[a][p]/predictions[b][p])))) for p in groups)
        row = {'start': index, 'q': q.tolist(), 'cost_by_order': costs,
               'conditional_metrics': metrics, 'max_refinement_change_decade': changes,
               '512_to_1024_passes_1e_minus_4_decade': changes['512_to_1024'] < 1e-4,
               'success': bool(result.success), 'message': result.message,
               'optimality': float(result.optimality), 'nfev': result.nfev,
               'actual_function_calls': calls, 'active_mask': result.active_mask.tolist()}
        output['fits'].append(row)
        write(output)
        print(json.dumps({k: row[k] for k in ('start', 'q', 'cost_by_order', 'max_refinement_change_decade', 'nfev')}), flush=True)
    eligible = [row for row in output['fits'] if row['512_to_1024_passes_1e_minus_4_decade']]
    selected = min(eligible or output['fits'], key=lambda row: row['cost_by_order']['1024'])
    q = np.array(selected['q'])
    output.update(status='selected_parameters_frozen_validating_commands',
        selected_start=selected['start'], selected_has_512_to_1024_certificate=bool(eligible),
        selection_rule='lowest 1024-node conditional cost among 512-to-1024-certified starts; if none, display best without certificate',
        selected_parameters={'gamma_ev': 10.**q[0], 'median_energy_ev': 10.**q[1],
                             'eta': q[2], 'sigma_log_energy': q[3], 'ron_ohm': ron, 'temperature_k': T_K})
    write(output)
    predictions = {order: {p: solve_command(q, d['command'], p, ron, circuit, order)
                           for p, d in groups.items()} for order in (512, 1024)}
    autonomous, independent = {}, {}
    for phase, d in groups.items():
        result = predictions[1024][phase]
        pred = result['predicted_signed_current_A']
        points = np.concatenate([d['hbn_observed_current_deembedding'], result['hbn_voltage_V']])
        reference, estimate = independent_z_integral(q, points, ron)
        value = readout(q, points, ron, 1024)
        independent[phase] = {'n_verified_voltages': len(points),
            'max_1024_vs_adaptive_normal_integral_current_difference_decade': float(np.max(abs(np.log10(abs(value/reference))))),
            'adaptive_integral_maxnorm_error_estimate_A': estimate,
            'internal_voltage_V': points.tolist(), 'independent_current_A': reference.tolist()}
        autonomous[phase] = {**errors(pred, d),
            'n_raw_current_points': int(d['observed'].size),
            'max_abs_KCL_residual_A': float(np.max(abs(result['KCL_residual_A']))),
            'max_abs_KVL_residual_V': float(np.max(abs(result['KVL_residual_V']))),
            'all_passive': bool(np.all(pred*result['hbn_voltage_V'] >= 0.) and np.all(pred*result['transistor_voltage_V'] >= 0.)),
            '512_to_1024_current_change_decade': float(np.max(abs(np.log10(abs(
                predictions[512][phase]['predicted_signed_current_A']/pred))))),
            'command_voltage_V': d['command'].tolist(),
            **{key: val.tolist() for key, val in result.items()}}
    output['autonomous_command_only_circuit'] = autonomous
    output['independent_ensemble_quadrature'] = independent
    output['key_features'] = feature_rows(groups, predictions[1024])
    levels = [.001, .01, .1, .5, .9, .99, .999]
    median, sigma = 10.**q[1], q[3]
    output['energy_distribution_diagnostics'] = {
        'quantile_probabilities': levels,
        'quantile_energies_ev': (median*np.exp(sigma*ndtri(levels))).tolist(),
        'mean_energy_ev': float(median*np.exp(sigma**2/2.)),
        'fraction_above_energy_ev': {str(e): float(ndtr(-np.log(e/median)/sigma)) if sigma > 0. else float(median > e)
                                     for e in (2., 5., 10.)},
        'interpretation': 'These are fitted candidate distribution diagnostics, not measured DOS or band-edge assignments.'}
    output['aggregate'] = {
        'phase_balanced_geomean_RMSE_decade': float(np.sqrt(np.mean([r['geometric_mean_RMSE_decade']**2 for r in autonomous.values()]))),
        'phase_balanced_raw_records_RMSE_decade': float(np.sqrt(np.mean([r['all_32_raw_records_RMSE_decade']**2 for r in autonomous.values()]))),
        'n_key_features': len(output['key_features']),
        'n_key_features_inside_training_Q10_Q90': sum(f['inside_empirical_Q10_Q90'] for f in output['key_features']),
        'independent_quadrature_passes_1e_minus_5_decade': all(
            r['max_1024_vs_adaptive_normal_integral_current_difference_decade'] < 1e-5 for r in independent.values())}
    output['scope'] = [
        'Same four readout parameters as Gaussian: gamma, median energy, log-energy sigma, eta. Ron and temperature fixed.',
        'Same training curves, current floors (none needed in selected windows), phase weights, transistor mapping and frozen-state hypothesis as the Gaussian screen.',
        'Only SET return command >=0.06 V and RESET outward abs(command) 0.06–0.30 V; no other branches or held-out data are scored.',
        'Positive lognormal energy spectrum is a candidate shape motivated by the Gaussian early-saturation failure, not independently measured electronic DOS.',
        'Distribution does not change with state. Thus gamma(u)=gamma_c/expm1(u) has the same monotone |I|-versus-u direction and exact Ohmic limit.',
        'Coarse optimization success does not certify quadrature or global optimality. All three starts and failed refinements are retained.',
        'No state dynamics, additional offset, polarity amplitude or per-cycle state was introduced; failing features are not repaired by adding parameters.',
        'Adaptive-normal integration is independent of Hermite nodes but shares the single-level digamma identity; this is an ensemble-quadrature check.',
        'Gamma_c and u remain unidentifiable separately from this frozen readout. No continuous cycle simulation or full unified-device claim.'
    ]
    output['elapsed_s'] = time.monotonic()-begin
    output['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__), OUT/'validate_spectral_readout.py', SCREEN, DATA,
                  OUT/'spectral_readout.py', ROOT/'vulcan2d/transistor.py', ROOT/'vulcan2d/shared_gap.py',
                  ROOT/'vulcan2d/data/fig2b_transistor_lookup.json',
                  ROOT/'vulcan2d/data/bookoutput2_reverse_transistor_lookup.json')}
    output['status'] = 'completed_bounded_lognormal_frozen_readout_screen_not_a_dynamic_model'
    write(output)
    lines = ['# 正偏斜谱分布：同参数数目的冻结读出替代筛选', '',
        '## 数学定义与边界', '',
        r'令 $Z\sim N(0,1)$，$\epsilon=m\exp(\sigma Z)$，$m>0,\sigma\ge0$；对同一有限温度 Lorentz-Landauer 电流作归一化期望。',
        'σ=0 是中心能量 m 的单谱线；m 是中位数，均值为 m exp(σ²/2)，没有另加能量 offset。',
        '正权归一平均保留 I·V≥0、0<∂I/∂V≤1/Ron、γ→∞ 的精确 Ohmic 极限。',
        '分布与状态无关，固定 ε/η 时每条谱线随 γ 增大而电流幅度不减，故 γ(u)=γc/expm1(u) 保留 SET 闭合增导方向。',
        '0<η<1 的全能窗极限仍为 |I|=πγ/Ron；高能尾改变进入平台的过程，没有增加独立电流幅度。',
        '这是候选能量分布形状，不是独立测得的 DOS。', '',
        '## 固定比较范围', '',
        '与 Gaussian 相同的第 4–35 圈、SET 返扫 0.06–5 V、RESET 外扫 −0.06 至 −0.30 V；',
        '相同观测电流去嵌入拟合目标及相位等权，Ron、300 K、1T 映射固定；四个参数未增加。',
        '仅三个预设起点、128 节点拟合每个最多 80 nfev，之后作 512/1024 加密；不追加起点或参数。', '',
        '| 起点 | 128 代价 | 512 代价 | 1024 代价 | 512→1024 最大差/decade | nfev |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in output['fits']:
        lines.append(f'| {r["start"]} | {r["cost_by_order"]["128"]:.9g} | {r["cost_by_order"]["512"]:.9g} | '
                     f'{r["cost_by_order"]["1024"]:.9g} | {r["max_refinement_change_decade"]["512_to_1024"]:.9g} | {r["nfev"]} |')
    lines += ['', f'选择起点 {selected["start"]}；512→1024 检查资格：{bool(eligible)}。完整停止原因、边界及检查点保留在 JSON。',
              '', '冻结参数：', '', '```json', json.dumps(output['selected_parameters'], indent=2), '```', '',
              '## 冻结参数后的自主电路', '',
              '只输入命令电压求 KCL；观测电流不参与预测。以下两种误差不能混称。', '',
              '| 区间 | 几何平均 RMSE/decade | 原始32圈 RMSE/decade | 自主512→1024最大差/decade | 独立积分差/decade |',
              '| --- | ---: | ---: | ---: | ---: |']
    for phase, r in autonomous.items():
        lines.append(f'| {phase} | {r["geometric_mean_RMSE_decade"]:.9g} | {r["all_32_raw_records_RMSE_decade"]:.9g} | '
                     f'{r["512_to_1024_current_change_decade"]:.9g} | '
                     f'{independent[phase]["max_1024_vs_adaptive_normal_integral_current_difference_decade"]:.9g} |')
    lines += ['', '## 全部原静态特征门槛', '',
              '| 区间 / 特征 | 预测 | 原始 Q10 / 中位 / Q90 | 通过经验区间 |',
              '| --- | ---: | --- | --- |']
    for f in output['key_features']:
        value = f.get('predicted_A', f.get('predicted'))
        band = f.get('raw_Q10_median_Q90_A', f.get('raw_Q10_median_Q90'))
        lines.append(f'| {f["phase"]} / {f["feature"]} | {value:.9g} | '+
                     ' / '.join(f'{x:.9g}' for x in band)+f' | {f["inside_empirical_Q10_Q90"]} |')
    lines += ['', '电流单位 A；经验范围不是置信区间或独立预测检验。', '',
              '## 能谱范围与结论限制', '', '```json',
              json.dumps(output['energy_distribution_diagnostics'], indent=2), '```', '',
              f'16 个显示特征中 {output["aggregate"]["n_key_features_inside_training_Q10_Q90"]} 个位于原训练范围。',
              '若关键特征仍失败，当前形式保留为失败结果，不通过追加 offset、极性幅度或新状态救回。',
              '这里只比较两个同参数数目的谱分布形状；没有证明分布唯一、没有独立确认缺陷谱或输运机制。',
              '没有完整状态轨迹、RESET 高压平台、SET 外扫或裸 HRS 预测，不能声称统一模型完成。',
              f'总耗时 {output["elapsed_s"]:.3f} s。', '']
    (OUT/'LOGNORMAL_SPECTRAL_SCREEN.md').write_text('\n'.join(lines))
    print(json.dumps({'status': output['status'], 'selected_start': selected['start'],
                      'parameters': output['selected_parameters'], 'aggregate': output['aggregate'],
                      'elapsed_s': output['elapsed_s']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
