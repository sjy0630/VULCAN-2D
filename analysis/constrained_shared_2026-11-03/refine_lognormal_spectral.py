"""Repair ensemble quadrature and locally refit the unchanged four parameters.

Fixed-energy composite Legendre integrates the full single-level digamma
current against the lognormal density. The omitted high-energy tail is bounded,
not renormalized. This selected finite-gamma backend is NOT a production
wide-gamma/Ohmic-limit integrator. Earlier Hermite failures remain untouched.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.special import digamma, ndtr, ndtri, roots_legendre

from lognormal_spectral_screen import independent_z_integral
from validate_spectral_readout import (ROOT, DATA, SCREEN, IDS, T_K, THERMAL,
    load_groups, errors, feature_rows, measured_transistor_mapping, LookupGapCircuit)

OUT = Path(__file__).resolve().parent
ORIGINAL = OUT/'lognormal_spectral_screen.json'
RESULT = OUT/'lognormal_spectral_refinement.json'
PANELS = np.concatenate([np.arange(21)*.25, [10., 20., 50., 100.]])


@lru_cache(maxsize=8)
def energy_rule(order):
    x, w = roots_legendre(order)
    energies, weights = [], []
    for low, high in zip(PANELS[:-1], PANELS[1:]):
        energies.extend(.5*(low+high)+.5*(high-low)*x)
        weights.extend(.5*(high-low)*w)
    return np.array(energies), np.array(weights)


def density(q, energy):
    median, sigma = 10.**q[1], q[3]
    if sigma <= 0.:
        raise ValueError('Selected fixed-energy backend requires positive finite sigma')
    return np.exp(-.5*(np.log(energy/median)/sigma)**2)/(energy*sigma*np.sqrt(2*np.pi))


def readout(q, voltage, ron, order=32):
    gamma, eta = 10.**q[0], q[2]
    v = np.asarray(voltage, dtype=float)
    flat = v.reshape(-1)
    energies, weights = energy_rule(order)
    probabilistic = weights*density(q, energies)  # Never renormalize the omitted tail.
    a = .5+(gamma+1j*(energies[None, :]+eta*flat[:, None]))/(2*np.pi*THERMAL)
    b = .5+(gamma+1j*(energies[None, :]-(1-eta)*flat[:, None]))/(2*np.pi*THERMAL)
    result = gamma/ron*((digamma(a)-digamma(b)).imag@probabilistic)
    result[flat == 0.] = 0.
    if not np.all(np.isfinite(result)) or np.any(result*flat < 0.):
        raise FloatingPointError('Invalid passive current')
    return float(result[0]) if v.ndim == 0 else result.reshape(v.shape)


def tail_diagnostics(q, voltages, ron, order):
    gamma, median, sigma = 10.**q[0], 10.**q[1], q[3]
    v = abs(np.asarray(voltages))
    distance = 100.-v-40.*THERMAL
    if np.any(distance <= 0.):
        raise ValueError('High-energy tail bound outside evaluated voltage domain')
    mass = float(ndtr(-np.log(100./median)/sigma))
    # Split each electron-energy integral into a window expanded by 40 kT and
    # its two Fermi tails. Inside: T<=gamma^2/distance^2; outside: weighted
    # integral <=2 exp(-40) pi gamma. Integrate this bound over eps>100.
    bound = mass*(gamma**2*v/(ron*distance**2)+2*np.exp(-40.)*np.pi*gamma/ron)
    energies, weights = energy_rule(order)
    quadrature_mass = float(weights@density(q, energies))
    return {'omitted_energy_probability_above_100eV': mass,
            'exact_probability_below_100eV': 1.-mass,
            'quadrature_probability_below_100eV': quadrature_mass,
            'probability_quadrature_error': quadrature_mass-(1.-mass),
            'max_omitted_current_bound_A': float(max(bound)),
            'per_voltage_omitted_current_bound_A': bound.tolist()}


def conditional(q, groups, ron, order):
    residuals, predictions = [], {}
    for phase, d in groups.items():
        pred = abs(readout(q, d['hbn_observed_current_deembedding'], ron, order))
        if np.any(pred <= 0.):
            raise FloatingPointError('Nonpositive readout current')
        error = np.log10(pred/d['geometric'])
        residuals.append(error/np.sqrt(len(error)))
        predictions[phase] = pred
    return np.concatenate(residuals), predictions


def solve_command(q, command, phase, ron, circuit, order):
    """Autonomous KCL; no observation input."""
    v = np.asarray(command)
    sign, mag = np.sign(v), abs(v)
    knots, transistor = circuit._branches[phase]
    if np.max(mag) > knots[-1]:
        raise ValueError('Transistor extrapolation required')
    low, high = np.zeros_like(mag), mag.copy()
    for _ in range(56):
        vt = .5*(low+high)
        ih = abs(readout(q, sign*(mag-vt), ron, order))
        it = np.interp(vt, knots, transistor)
        move = ih > it
        low, high = np.where(move, vt, low), np.where(move, high, vt)
    vt = .5*(low+high)
    vh = sign*(mag-vt)
    ih = readout(q, vh, ron, order)
    it = np.array([circuit.mapping.drain_current(float(s*t), 1.1) for s, t in zip(sign, vt)])
    return {'predicted_signed_current_A': ih, 'hbn_voltage_V': vh, 'transistor_voltage_V': sign*vt,
            'KCL_residual_A': ih-it, 'KVL_residual_V': v-vh-sign*vt}


def write(output):
    temporary = RESULT.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    temporary.replace(RESULT)


def main():
    began = time.monotonic()
    original = json.loads(ORIGINAL.read_text())
    old = json.loads(SCREEN.read_text())
    source_fit = next(r for r in original['fits'] if r['start'] == original['selected_start'])
    initial = np.array(source_fit['q'])
    ron = original['ron_ohm']
    mapping = measured_transistor_mapping()
    circuit = LookupGapCircuit(mapping, 1.1)
    groups = load_groups(mapping, old)
    # Algorithmic neighborhood, NOT an empirical or physical confidence range.
    radius = np.array([.30, .25, .10, .25])
    lower = np.maximum(initial-radius, [-7., np.log10(.02), .001, .1])
    upper = np.minimum(initial+radius, [.7, np.log10(5.), .999, 2.])
    output = {'status': 'checking_fixed_energy_quadrature_before_local_refit', 'training_ids': IDS,
        'initial_q': initial.tolist(), 'source_start': original['selected_start'],
        'parameter_order': original['q_order'], 'max_nfev': 40,
        'algorithmic_local_bounds_not_physical_confidence': {'lower': lower.tolist(), 'upper': upper.tolist(),
            'radius': radius.tolist(), 'purpose': 'keep a positive broad lognormal density resolved by a fixed energy mesh'},
        'energy_panels_eV': PANELS.tolist(), 'checkpoints': []}
    initial_checks = {}
    initial_residual, _ = conditional(initial, groups, ron, 32)
    for phase, d in groups.items():
        v = d['hbn_observed_current_deembedding']
        independent, estimate = independent_z_integral(initial, v, ron)
        values = {n: readout(initial, v, ron, n) for n in (16, 32, 64)}
        initial_checks[phase] = {
            '16_to_32_max_change_decade': float(max(abs(np.log10(abs(values[16]/values[32]))))),
            '32_to_64_max_change_decade': float(max(abs(np.log10(abs(values[32]/values[64]))))),
            '64_vs_adaptive_Z_max_change_decade': float(max(abs(np.log10(abs(values[64]/independent))))),
            'adaptive_Z_maxnorm_error_estimate_A': estimate,
            'tail': tail_diagnostics(initial, v, ron, 64)}
    output['initial_quadrature_checks'] = initial_checks
    if any(row['64_vs_adaptive_Z_max_change_decade'] > 1e-5 or row['32_to_64_max_change_decade'] > 1e-5
           for row in initial_checks.values()):
        output['status'] = 'fixed_energy_backend_failed_pre_fit_check_no_refit'
        write(output)
        print(json.dumps(output, indent=2))
        return
    calls = 0
    def objective(q):
        nonlocal calls
        residual = conditional(q, groups, ron, 32)[0]
        calls += 1
        if calls == 1 or calls % 5 == 0:
            output['checkpoints'].append({'call': calls, 'cost': float(residual@residual/2),
                                          'q': q.tolist(), 'elapsed_s': time.monotonic()-began})
            write(output)
        return residual
    fit = least_squares(objective, initial, bounds=(lower, upper), max_nfev=40,
                        diff_step=1e-5, x_scale='jac', ftol=1e-10, xtol=1e-10, gtol=1e-8)
    q = fit.x.copy()
    fitted_at = time.monotonic()
    output.update(status='parameters_frozen_independently_validating_fixed_energy_readout',
        selected_q=q.tolist(), parameters={'gamma_ev': 10.**q[0], 'median_energy_ev': 10.**q[1],
            'eta': q[2], 'sigma_log_energy': q[3], 'ron_ohm': ron, 'temperature_k': T_K},
        local_refit={'nfev': fit.nfev, 'njev': fit.njev, 'actual_function_calls': calls,
            'success': bool(fit.success), 'message': fit.message, 'optimality': float(fit.optimality),
            'active_mask': fit.active_mask.tolist(),
            'initial_accurate_conditional_cost': float(initial_residual@initial_residual/2),
            'elapsed_s': fitted_at-began})
    write(output)
    conditional_report = {}
    for order in (32, 64):
        residual, pred = conditional(q, groups, ron, order)
        conditional_report[str(order)] = {'cost': float(residual@residual/2),
            'metrics': {p: errors(i, groups[p]) for p, i in pred.items()}}
    output['conditional_observed_current_deembedding'] = conditional_report
    predictions = {n: {p: solve_command(q, d['command'], p, ron, circuit, n) for p, d in groups.items()}
                   for n in (32, 64)}
    autonomous, independent = {}, {}
    for phase, d in groups.items():
        result = predictions[64][phase]
        pred = result['predicted_signed_current_A']
        points = np.concatenate([d['hbn_observed_current_deembedding'], result['hbn_voltage_V']])
        reference, estimate = independent_z_integral(q, points, ron)
        v32, v64 = readout(q, points, ron, 32), readout(q, points, ron, 64)
        independent[phase] = {'n_verified_internal_voltages': len(points),
            'max_32_to_64_current_change_decade': float(max(abs(np.log10(abs(v32/v64))))),
            'max_64_vs_adaptive_Z_current_change_decade': float(max(abs(np.log10(abs(v64/reference))))),
            'adaptive_Z_maxnorm_error_estimate_A': estimate,
            'tail': tail_diagnostics(q, points, ron, 64),
            'internal_voltage_V': points.tolist(), 'independent_current_A': reference.tolist()}
        autonomous[phase] = {**errors(pred, d), 'n_raw_current_points': int(d['observed'].size),
            'max_abs_KCL_residual_A': float(max(abs(result['KCL_residual_A']))),
            'max_abs_KVL_residual_V': float(max(abs(result['KVL_residual_V']))),
            'all_passive': bool(np.all(pred*result['hbn_voltage_V'] >= 0.) and np.all(pred*result['transistor_voltage_V'] >= 0.)),
            'command_voltage_V': d['command'].tolist(),
            'max_32_to_64_autonomous_current_change_decade': float(max(abs(np.log10(abs(
                predictions[32][phase]['predicted_signed_current_A']/pred))))),
            **{key: value.tolist() for key, value in result.items()}}
    output['independent_quadrature'] = independent
    output['autonomous_command_only_circuit'] = autonomous
    output['key_features'] = feature_rows(groups, predictions[64])
    checks = {'quadrature_32_to_64_below_1e_minus_5_decade': all(r['max_32_to_64_current_change_decade'] < 1e-5 for r in independent.values()),
        'independent_adaptive_Z_difference_below_1e_minus_5_decade': all(r['max_64_vs_adaptive_Z_current_change_decade'] < 1e-5 for r in independent.values()),
        'KCL_below_1e_minus_13_A': all(r['max_abs_KCL_residual_A'] < 1e-13 for r in autonomous.values()),
        'all_circuit_points_passive': all(r['all_passive'] for r in autonomous.values())}
    output['numerical_checks'] = {'all_passed': all(checks.values()), 'checks': checks,
                                   'scope': 'selected fitted finite-gamma point and selected static voltage windows only'}
    output['aggregate'] = {
        'phase_balanced_geomean_RMSE_decade': float(np.sqrt(np.mean([r['geometric_mean_RMSE_decade']**2 for r in autonomous.values()]))),
        'phase_balanced_raw_records_RMSE_decade': float(np.sqrt(np.mean([r['all_32_raw_records_RMSE_decade']**2 for r in autonomous.values()]))),
        'n_features': len(output['key_features']),
        'n_features_inside_original_empirical_Q10_Q90': sum(f['inside_empirical_Q10_Q90'] for f in output['key_features'])}
    m, sigma = 10.**q[1], q[3]
    levels = [.001, .01, .1, .5, .9, .99, .999]
    output['energy_distribution'] = {'probabilities': levels, 'quantile_energy_eV': (m*np.exp(sigma*ndtri(levels))).tolist(),
        'mean_energy_eV': float(m*np.exp(sigma*sigma/2)),
        'fraction_above_energy_eV': {str(e): float(ndtr(-np.log(e/m)/sigma)) for e in (2., 5., 10., 100.)}}
    output['limitations'] = [
        'Same four-parameter lognormal spectral family; no added offset, polarity amplitude, state, temperature fit or transistor remapping.',
        'This local refit corrects quadrature, not model capacity by adding degrees of freedom. All earlier Hermite failures remain unchanged.',
        'Algorithmic local bounds are a numerical search neighborhood, not measured physical uncertainty. Only this basin was refined.',
        'Conditional calibration still deembeds observed geometric-mean current. Autonomous command-only KCL follows after all parameters are frozen.',
        'Adaptive-Z validation shares the exact single-level digamma identity but uses an independent coordinate, quadrature scheme and untruncated energy support.',
        'No probability renormalization hides the energy cutoff. A finite-gamma high-energy current tail bound is reported.',
        'The fixed-energy cutoff is not a general Ohmic-limit implementation at gamma->infinity. Certification is restricted to the fitted finite gamma and evaluated range.',
        'SET return 0.06–5 V and RESET outward 0.06–0.30 V only, cycles 4–35. No extra windows or holdouts select the parameters.',
        'Frozen gamma is not a dynamic prediction; gamma_c and u remain unseparated. No measured DOS, microscopic species or full unified model claim.',
        'Passing numerical checks does not imply passing feature gates. Every failed original feature remains listed.'
    ]
    output['elapsed_s'] = time.monotonic()-began
    output['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__), ORIGINAL, OUT/'lognormal_spectral_screen.py', OUT/'validate_spectral_readout.py',
                  SCREEN, DATA, ROOT/'vulcan2d/transistor.py', ROOT/'vulcan2d/shared_gap.py',
                  ROOT/'vulcan2d/data/fig2b_transistor_lookup.json', ROOT/'vulcan2d/data/bookoutput2_reverse_transistor_lookup.json')}
    output['status'] = 'completed_fixed_energy_quadrature_and_local_refit_not_dynamic_model'
    write(output)
    lines = ['# Lognormal 谱读出：修复积分精度后的局部复拟合', '',
        '原 Hermite 失败结果完整保留。本次不增加参数，只更换积分坐标并在原选中点附近局部复拟合。', '',
        '## 数值方法与边界', '',
        '直接在能量 ε 上积分 lognormal 密度乘单谱线 digamma 电流：0–5 eV 每 0.25 eV 一段，',
        '另有 5–10、10–20、20–50、50–100 eV 段；拟合每段 32 节点，检查每段 64 节点。',
        '不把截断概率重新归一化。ε>100 eV 的电流余项由下式显式上界控制：', '',
        r'$$P(\epsilon>100)\left[\frac{\gamma^2|V|}{R_{on}(100-|V|-40t)^2}+\frac{2\pi\gamma e^{-40}}{R_{on}}\right],\quad t=k_BT/e.$$','',
        '独立参照在 Z 坐标自适应积分到 ±10，包含能量大于 100 eV 的区域；该参照不使用 Legendre 能量节点。',
        '以上尾界用于所选有限 γ，不能把此有限能量网格当作 γ→∞ 的通用 Ohmic 极限实现。', '',
        '## 局部复拟合', '',
        f'nfev={fit.nfev}，实际函数调用={calls}，终止={fit.message}，optimality={fit.optimality:.9g}。',
        '局部边界是算法搜索邻域，不是实测置信区间；四个参数维持不变，全部边界记录在 JSON。',
        f'准确条件代价 {output["local_refit"]["initial_accurate_conditional_cost"]:.12g} → {conditional_report["64"]["cost"]:.12g}。',
        '', '```json', json.dumps(output['parameters'], indent=2), '```', '',
        '## 冻结后自主 KCL 与独立积分检查', '',
        '预测只输入命令电压，观测电流只在结果评分中使用。拟合本身仍是观测均流去嵌入条件目标。', '',
        '| 区间 | 几何平均RMSE/decade | 原始32圈RMSE/decade | 32→64最大差/decade | 独立Z积分最大差/decade | 尾电流界/A |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for phase, r in autonomous.items():
        c = independent[phase]
        lines.append(f'| {phase} | {r["geometric_mean_RMSE_decade"]:.9g} | {r["all_32_raw_records_RMSE_decade"]:.9g} | '
                     f'{c["max_32_to_64_current_change_decade"]:.9g} | {c["max_64_vs_adaptive_Z_current_change_decade"]:.9g} | '
                     f'{c["tail"]["max_omitted_current_bound_A"]:.9g} |')
    lines += ['', f'所选范围数值检查全部通过：{output["numerical_checks"]["all_passed"]}。这不等于形状门槛通过。', '',
        '## 所有原特征门槛', '', '| 区间 / 特征 | 预测 | 原始 Q10 / 中位 / Q90 | 通过 |',
        '| --- | ---: | --- | --- |']
    for f in output['key_features']:
        value = f.get('predicted_A', f.get('predicted'))
        band = f.get('raw_Q10_median_Q90_A', f.get('raw_Q10_median_Q90'))
        lines.append(f'| {f["phase"]} / {f["feature"]} | {value:.9g} | '+
                     ' / '.join(f'{x:.9g}' for x in band)+f' | {f["inside_empirical_Q10_Q90"]} |')
    lines += ['', '电流单位 A。经验 Q10–Q90 不是置信区间或未见数据检验。', '',
        f'原 16 项显示特征中 {output["aggregate"]["n_features_inside_original_empirical_Q10_Q90"]} 项通过。',
        '数值错误与物理/形状不足分别记录；不以减少的总 RMSE 覆盖失败特征。', '',
        '## 能谱范围与限制', '', '```json', json.dumps(output['energy_distribution'], indent=2), '```', '',
        '长尾是候选分布的拟合性质，不是独立观测到的缺陷能谱；高能尾是否有真实物理依据尚待判断。',
        '只验证原有静态区间；没有状态方程、完整 RESET 平台或裸 HRS 预测，不能称为统一动态模型。',
        f'总耗时 {output["elapsed_s"]:.3f} s，包含预检与局部拟合 {fitted_at-began:.3f} s。', '']
    (OUT/'lognormal_spectral_refinement.md').write_text('\n'.join(lines))
    print(json.dumps({'parameters': output['parameters'], 'local_refit': output['local_refit'],
        'numerical_checks': output['numerical_checks'], 'aggregate': output['aggregate'],
        'elapsed_s': output['elapsed_s']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
