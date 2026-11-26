"""Audit saved endpoints and arithmetic only; no model import or new solver."""
from pathlib import Path
from decimal import Decimal, localcontext
from collections import Counter
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / 'zero_crossing_compatibility_2026-11-04'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    result_path = HERE / 'population_result.json'
    result = json.loads(result_path.read_text())
    native = json.loads((HERE / 'audit.json').read_text())
    prior = json.loads((PREVIOUS / 'compatibility_result.json').read_text())
    errors, hashes, bridge_reviews, high_precision_pass_disagreements = [], [], [], []
    check_count = 0

    def check(label, condition):
        nonlocal check_count
        check_count += 1
        if not condition:
            errors.append(label)

    def equal(label, actual, expected, rel=1e-12, atol=1e-15):
        check(label, math.isclose(actual, expected, rel_tol=rel, abs_tol=atol))

    def linear(x, xs, ys):
        if xs[0] == xs[1]:
            check('Exact-node inverse remains at same node', x == xs[0] and ys[0] == ys[1])
            return ys[0]
        check('Saved forward interpolation stays in bracket', xs[0] <= x <= xs[1])
        return ys[0] + (ys[1] - ys[0]) * (x - xs[0]) / (xs[1] - xs[0])

    def power(x, y, m):
        a = m - 1.0
        lr = math.log1p((y - x) / x)
        return lr if a == 0 else math.exp(a * math.log(x)) * math.expm1(a * lr) / a

    def mixed(x, y, e):
        return y - x if e == 0 else math.log1p(e * (y - x) / (1 + e * x)) / e

    for path, expected in result['source_sha256'].items():
        actual = sha(path)
        check('Actual source hash: ' + path, actual == expected)
        hashes.append({'path': path, 'sha256': actual, 'matched': actual == expected})
    print('Actual source hashes checked', flush=True)
    check('15 actual sources', len(hashes) == 15)
    check('Execution completed', result['status'] == 'completed_finite_population_screen')
    check('31 planned and present', result['predeclared_bridges'] == len(result['bridges']) == 31)
    check('Exact reset training range', [b['reset_record'] for b in result['bridges']] == list(range(4, 35)))
    check('Exact audit training range', [b['reset_record'] for b in native['bridges']] == list(range(4, 35)))
    check('Independent native audit complete', native['anomaly_count'] == 0 and native['complete_bridges'] == 31)
    check('No new ODE/fits/b', all(result[k] == 0 for k in ('new_ODEs', 'new_fits', 'new_b_trials')))
    check('No state reanchor/holdout/joint scoring', all(result[k] is False for k in (
        'state_reanchoring', 'original_holdout_accessed', 'full_joint_contract_evaluated')))
    check('No numerical failures', result['failures'] == [])
    check('Numerical budget respected', 0 <= result['numerical_wall_s'] <= 10)
    check('Frozen parameters identical', result['parameters'] == prior['parameters'])
    check('Frozen readout identical', result['Ron_ohm'] == prior['Ron_ohm'] and result['gamma_c_ev'] == prior['gamma_c_ev'])
    upper, gc = result['shared_decision_upper_deltaZ'], result['gamma_c_ev']
    equal('Shared upper identical', upper, prior['necessary_budget']['decision_upper_deltaZ'])
    parameters = result['parameters']
    equal('Shared upper analytic arithmetic', upper, .5 * parameters['kappa_per_v2'] * .2**2
          * math.exp(.2 * parameters['lambda_per_v']) * 101 * (1 + 1e-6))
    provenance = Counter()
    total_circuit_records = 0
    maxima = {'current_log_error_decade': 0., 'KCL_a': 0., 'KVL_v': 0., 'u_algorithm_difference': 0.,
              'KCL_reconstruction_difference_a': 0., 'KCL_roundoff_allowance_fraction': 0.}
    caps, floors = [], []

    for bridge, raw in zip(result['bridges'], native['bridges']):
        k = bridge['reset_record']
        check(f'{k}: identity', bridge['bridge_id'] == raw['bridge_id'] and bridge['set_record'] == k + 1)
        check(f'{k}: native path audit', raw['all_required_checks_passed'] and raw['complete'])
        path = raw['joined_command_values_v']
        check(f'{k}: path direction', path[0] == -.2 and path[-1] == .2 and all(a <= b for a, b in zip(path, path[1:])))
        equal(f'{k}: saved path length', raw['total_command_path_distance_v'], .4)
        equal(f'{k}: reconstructed path length', sum(abs(b - a) for a, b in zip(path, path[1:])), .4)
        check(f'{k}: zero path duplicates', sum(a == b for a, b in zip(path, path[1:])) == 2)
        check(f'{k}: two valid endpoints', len(bridge['readpoints']) == 2 and all(p['status'] == 'valid' for p in bridge['readpoints']))
        for index, (p, phase) in enumerate(zip(bridge['readpoints'], ('reset', 'set'))):
            tag = f'{k} {phase} '
            original = raw[phase + '_read']
            check(tag + 'original mask', original['retained_by_original_mask'] and original['current_magnitude_a'] >= original['original_floor_a'])
            check(tag + 'data identity', p['native_index'] == original['native_index'] == (161 if index == 0 else 10)
                  and p['command_voltage_v'] == original['command_v'] == (-.2 if index == 0 else .2)
                  and p['observed_current_magnitude_a'] == original['current_magnitude_a'])
            provenance[p['provenance']] += 1
            if k == 4:
                check(tag + 'reuse provenance', p['provenance'] == 'reused_frozen_two_readpoint_result')
                check(tag + 'all reused values exact', all(p[key] == value for key, value in prior['readpoints'][index].items()))
            else:
                check(tag + 'new provenance', p['provenance'] == 'new_conditional_inversion')
            check(tag + 'execution monotonic and uniqueness flags', all(p[key] is True for key in (
                'transistor_monotonic_checked', 'material_slice_monotonic_checked', 'unique_target')))
            current, command = p['observed_current_magnitude_a'], p['command_voltage_v']
            drop, local = p['transistor_drop_magnitude_v'], p['local_voltage_magnitude_v']
            tx, ti = p['transistor_bracket_voltages'], p['transistor_bracket_currents']
            gx, gy = p['material_bracket_log_gamma'], p['material_bracket_log_current']
            lg = p['direct_log_gamma']
            check(tag + 'passive conditional split', drop >= 0 and 0 < local <= abs(command))
            equal(tag + 'conditional KVL', drop + local, abs(command))
            equal(tag + 'transistor forward arithmetic', linear(drop, tx, ti), current, atol=1e-24)
            equal(tag + 'material forward arithmetic', linear(lg, gx, gy), math.log(current))
            equal(tag + 'gamma mapping', math.exp(lg), p['direct_gamma_ev'])
            equal(tag + 'u mapping', math.log1p(gc * math.exp(-lg)), p['direct_u'])
            equal(tag + 'Z mapping', math.exp(p['direct_u']), p['direct_Z'])
            low, high = p['bisect_log_gamma_bracket']
            residuals, ub = p['bisect_log_current_residual_bracket'], p['bisect_u_bracket']
            check(tag + 'bisect residual signs and step count', low <= high and residuals[0] <= 0 <= residuals[1] and p['bisect_iterations'] == 48)
            for j, value in enumerate((low, high)):
                equal(tag + f'bracket residual {j}', linear(value, gx, gy) - math.log(current), residuals[j], rel=0, atol=1e-13)
            equal(tag + 'u lower direction', ub[0], math.log1p(gc * math.exp(-high)))
            equal(tag + 'u upper direction', ub[1], math.log1p(gc * math.exp(-low)))
            equal(tag + 'bisect u midpoint', p['bisect_u'], math.log1p(gc * math.exp(-(low + high) / 2)))
            difference = abs(p['direct_u'] - p['bisect_u'])
            check(tag + 'inverse agreement', difference <= 1e-8)
            maxima['u_algorithm_difference'] = max(maxima['u_algorithm_difference'], difference)
            pu, pz = p['numerical_padded_u_bracket'], p['numerical_padded_Z_bracket']
            equal(tag + 'padding lower', pu[0], min(p['direct_u'], ub[0]) - 1e-8)
            equal(tag + 'padding upper', pu[1], max(p['direct_u'], ub[1]) + 1e-8)
            check(tag + 'padded domain', 0 < pu[0] <= p['direct_u'] <= pu[1])
            for j in range(2):
                equal(tag + f'Z padded mapping {j}', pz[j], math.exp(pu[j]))
            check(tag + 'circuit modes', set(p['circuit_rechecks']) == {'fast', 'strict_bisection42'})
            slope = (ti[1] - ti[0]) / (tx[1] - tx[0])
            for mode, c in p['circuit_rechecks'].items():
                total_circuit_records += 1
                cur, lv, tv = c['current_a'], c['local_voltage_v'], c['transistor_voltage_v']
                err = abs(math.log10(abs(cur) / current))
                check(tag + mode + ' signs', cur * command > 0 and lv * cur > 0 and tv * command >= 0)
                check(tag + mode + ' original tolerances', err <= 5e-7 and abs(c['KCL_residual_a']) <= 1e-13 and abs(c['KVL_residual_v']) <= 1e-12)
                equal(tag + mode + ' log error', err, c['observed_current_log_error_decade'], rel=0, atol=1e-16)
                equal(tag + mode + ' KVL arithmetic', command - lv - tv, c['KVL_residual_v'], rel=0, atol=1e-17)
                reconstructed = cur - math.copysign(linear(abs(tv), tx, ti), command)
                kcl_difference = abs(reconstructed - c['KCL_residual_a'])
                allowance = slope * (2 * math.ulp(abs(command)) + 2 * math.ulp(abs(lv))) + 8 * math.ulp(abs(cur))
                check(tag + mode + ' KCL arithmetic with saved voltage subtraction', kcl_difference <= allowance)
                maxima['current_log_error_decade'] = max(maxima['current_log_error_decade'], err)
                maxima['KCL_a'] = max(maxima['KCL_a'], abs(c['KCL_residual_a']))
                maxima['KVL_v'] = max(maxima['KVL_v'], abs(c['KVL_residual_v']))
                maxima['KCL_reconstruction_difference_a'] = max(maxima['KCL_reconstruction_difference_a'], kcl_difference)
                maxima['KCL_roundoff_allowance_fraction'] = max(maxima['KCL_roundoff_allowance_fraction'], kcl_difference / allowance)

        minus, plus = bridge['readpoints']
        x, y = minus['numerical_padded_Z_bracket'][1], plus['numerical_padded_Z_bracket'][0]
        equal(f'{k}: negative upper', x, bridge['negative_Z_upper'])
        equal(f'{k}: positive lower', y, bridge['positive_Z_lower'])
        equal(f'{k}: delta lower', y - x, bridge['required_deltaZ_lower'])
        equal(f'{k}: margin', y - x - upper, bridge['margin_over_upper'])
        equal(f'{k}: ratio', (y - x) / upper, bridge['budget_ratio'])
        check(f'{k}: decision', bridge['status'] == ('conditional_violation' if y - x > upper else 'necessary_condition_not_excluding'))
        limits = bridge['structure_limits']
        review = {'reset_record': k, 'status': bridge['status'], 'required_deltaZ_lower': y - x, 'budget_ratio': bridge['budget_ratio']}
        if y <= x:
            check(f'{k}: no positive growth', limits == {'status': 'no_positive_growth_requirement', 'power_cap': None, 'mixed_floor': None})
        else:
            check(f'{k}: positive growth', limits['status'] == 'positive_growth_requirement' and x > 1)
            p, e = limits['power_cap'], limits['mixed_floor']
            equal(f'{k}: m1 saved', p['m1_requirement'], power(x, y, 1))
            equal(f'{k}: m2 saved', p['m2_requirement'], power(x, y, 2))
            equal(f'{k}: epsilon0 saved', e['epsilon0_requirement'], mixed(x, y, 0))
            equal(f'{k}: epsilon1 saved', e['epsilon1_requirement'], mixed(x, y, 1))
            check(f'{k}: actual endpoint statuses', p['status'] == 'conditional_m_upper' and e['status'] == 'conditional_epsilon_lower')
            check(f'{k}: actual power endpoint eligibility', p['m1_requirement'] <= upper < p['m2_requirement'])
            check(f'{k}: actual mixed endpoint eligibility', e['epsilon1_requirement'] <= upper < e['epsilon0_requirement'])
            check(f'{k}: m bracket direction', 1 <= p['pass_lo'] < p['fail_hi'] <= 2 and p['conservative_upper_m'] == p['fail_hi'])
            check(f'{k}: epsilon bracket direction', 0 <= e['fail_lo'] < e['pass_hi'] <= 1 and e['conservative_lower_epsilon'] == e['fail_lo'])
            check(f'{k}: algebraic step cap', 0 < p['algebraic_bisection_iterations'] <= 64 and 0 < e['algebraic_bisection_iterations'] <= 64)
            for endpoint in ('pass_lo', 'fail_hi'):
                equal(f'{k}: m {endpoint} saved arithmetic', power(x, y, p[endpoint]), p['F_' + endpoint])
            for endpoint in ('fail_lo', 'pass_hi'):
                equal(f'{k}: epsilon {endpoint} saved arithmetic', mixed(x, y, e[endpoint]), e['F_' + endpoint])
            check(f'{k}: saved power numerical classification', p['F_pass_lo'] <= upper < p['F_fail_hi'])
            check(f'{k}: saved mixed numerical classification', e['F_pass_hi'] <= upper < e['F_fail_lo'])
            with localcontext() as context:
                context.prec = 60
                xd, yd, ud = Decimal.from_float(x), Decimal.from_float(y), Decimal.from_float(upper)
                residuals = {}
                for endpoint in ('pass_lo', 'fail_hi'):
                    a = Decimal.from_float(p[endpoint]) - 1
                    residuals['m_' + endpoint] = float(((a * yd.ln()).exp() - (a * xd.ln()).exp()) / a - ud)
                for endpoint in ('fail_lo', 'pass_hi'):
                    a = Decimal.from_float(e[endpoint])
                    residuals['epsilon_' + endpoint] = float(((1 + a * yd).ln() - (1 + a * xd).ln()) / a - ud)
            check(f'{k}: reported m upper retains conservative side at Decimal60', residuals['m_fail_hi'] > 0)
            check(f'{k}: reported epsilon lower retains conservative side at Decimal60', residuals['epsilon_fail_lo'] > 0)
            # Adjacent floating endpoints are not an interval-arithmetic certificate.
            if residuals['m_pass_lo'] > 0 or residuals['epsilon_pass_hi'] > 0:
                high_precision_pass_disagreements.append({'reset_record': k, **residuals})
            caps.append({'reset_record': k, 'upper_m': p['conservative_upper_m']})
            floors.append({'reset_record': k, 'lower_epsilon': e['conservative_lower_epsilon']})
            review.update(conservative_upper_m=p['conservative_upper_m'], conservative_lower_epsilon=e['conservative_lower_epsilon'], decimal60_endpoint_residuals=residuals)
        bridge_reviews.append(review)

    counts = Counter(b['status'] for b in result['bridges'])
    for status in ('conditional_violation', 'necessary_condition_not_excluding', 'threshold_ineligible', 'numerically_unresolved'):
        check('Count ' + status, counts[status] == result['counts'][status])
    check('31 valid denominator', result['counts']['valid'] == result['counts']['predeclared'] == 31)
    check('24 versus 7', counts['conditional_violation'] == 24 and counts['necessary_condition_not_excluding'] == 7)
    check('All 7 unexcluded require negative growth', all(b['required_deltaZ_lower'] < 0 for b in result['bridges'] if b['status'] == 'necessary_condition_not_excluding'))
    for name, low, high in [('early', 4, 13), ('middle', 14, 23), ('late', 24, 34)]:
        selected = [b for b in result['bridges'] if low <= b['reset_record'] <= high]
        group = result['chronological_groups'][name]
        check(name + ' predeclared limits', group['reset_labels'] == [low, high] and group['n'] == len(selected))
        for status in ('conditional_violation', 'necessary_condition_not_excluding', 'threshold_ineligible', 'numerically_unresolved'):
            check(name + ' ' + status, group[status] == sum(b['status'] == status for b in selected))
    check('60 new attempts and completions', result['new_readpoint_attempts'] == result['new_readpoint_completed'] == provenance['new_conditional_inversion'] == 60)
    check('2 reused', result['reused_readpoints'] == provenance['reused_frozen_two_readpoint_result'] == 2)
    check('120 new circuits, 124 total saved including 4 reused', result['completed_new_circuit_rechecks'] == 120 and total_circuit_records == 124)
    shared = result['shared_structure_necessary_limits']
    check('Global m minimum', shared['most_restrictive_m'] == min(caps, key=lambda q: q['upper_m']))
    check('Global epsilon maximum', shared['most_restrictive_epsilon'] == max(floors, key=lambda q: q['lower_epsilon']))
    check('Not parameter estimates', shared['not_parameter_estimates_or_sufficient_conditions'] is True)
    ratios = [b['budget_ratio'] for b in result['bridges'] if b['status'] == 'conditional_violation']
    review = {
        'status': 'passed' if not errors else 'failed', 'checks_completed': check_count, 'errors': errors,
        'audit_source_sha256': sha(__file__), 'population_result_sha256': sha(result_path),
        'actual_source_hash_checks': hashes, 'actual_source_count': len(hashes),
        'counts': result['counts'], 'chronological_groups': result['chronological_groups'],
        'ledger': {'new_readpoint_attempts': 60, 'new_readpoint_completed': 60, 'reused_readpoints': 2,
                   'new_circuit_rechecks': 120, 'reused_saved_circuit_records': 4, 'total_saved_circuit_records': total_circuit_records},
        'maxima': maxima, 'violation_budget_ratio_range': [min(ratios), max(ratios)],
        'shared_structure_necessary_limits': shared, 'bridge_reviews': bridge_reviews,
        'adjacent_floating_pass_labels_not_strict_at_decimal60': high_precision_pass_disagreements,
        'all_reported_necessary_bound_sides_checked_at_decimal60': True,
        'numerical_execution_wall_s': result['numerical_wall_s'],
        'new_audit_roots': 0, 'new_audit_inversions': 0, 'new_audit_bisections': 0, 'new_audit_ODEs': 0,
        'audit_scope': 'Saved JSON arithmetic, 60-digit substitution at already saved algebraic endpoints, and actual-source byte hashes only. No physical kernel import, archive decoding, root/threshold search, inversion, ODE, fit or new b trial.',
        'limits': [
            '31 chronological bridges belong to one training history; no independent-sample significance or binomial confidence interval is claimed.',
            '7 necessary-condition-not-excluding bridges all have negative required deltaZ; this upper growth bound is uninformative for them and does not explain their curves.',
            '24 conditional violations concern the frozen readout/rates/positive parameters, original b domain and command-path clock without waiting evolution.',
            'Reported m and epsilon values are necessary bounds, not fitted parameters, identified microscopic orders, or sufficient continuous trajectories; old SET accuracy is not guaranteed.',
            'Floating padding, budget allowance and adjacent threshold endpoints are not a rigorous interval certificate or experimental error range.',
            'Whole-table monotonicity and native-index identity are supported by recorded execution/audit checks and verified byte hashes, not independently re-decoded historical arrays.',
        ],
    }
    destination = HERE / ('ACTUAL_RESULT_REVIEW.json' if not errors else 'ACTUAL_RESULT_REVIEW_FAILURE.json')
    with destination.open('x') as handle:
        json.dump(review, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    if errors:
        raise AssertionError(errors)
    m_bound, e_bound = shared['most_restrictive_m'], shared['most_restrictive_epsilon']
    unresolved_ids = [b['reset_record'] for b in result['bridges'] if b['status'] == 'necessary_condition_not_excluding']
    text = f'''# 31 条训练过零桥：实际结果保存算术审计

结论：**通过**。{check_count} 项保存算术/身份检查完成，15 项实际来源哈希相符。审计没有调用物理内核、解码历史数组或重新求根、反演、二分阈值、积分轨迹。

## 分母与执行账本

预定且有效桥均为 **31**：RESET4–34→SET5–35。24 桥违反条件状态增长预算，7 桥未被排除；阈值不合格及数值未定均为 0。早/中/晚固定分段分别为 **9/10、5/10、10/11**，矛盾覆盖训练历史，不仅是原先的单例。这些桥不是独立器件或独立实验，不能由 24/31 推导独立样本显著性或二项置信区间。

执行账本核对为 60 个新读点尝试且全部完成、120 次新电路回代；另有 2 个冻结读点原样复用，附带 4 条旧电路记录。保存记录合计 62 个读点、124 条电路记录，不能把后者写成 124 次新求解。原数值用时 {result['numerical_wall_s']:.9f} s，满足 10 s 预算。0 次 ODE、拟合、b 扫描、状态重锚或原联合合同重评分。

## 保存端点与必要界

62 个读点的保存晶体管/材料夹点前向代入、Γ→u→Z 映射、二分端点与符号、数值垫宽方向均通过。电路保存量满足原电流、KCL、KVL 与被动性要求；KCL 重构计入保存 Vcmd−Vh 相减经晶体管斜率传播的浮点舍入，没有放宽原物理容差。

每桥使用 **正端 Z 下界−负端 Z 上界** 与相同 U={upper:.12f} 比较。24 个违例的预算比范围为 **{min(ratios):.9f}–{max(ratios):.9f}**；这是状态增量预算比，不是电流误差。

7 个未排除桥的 RESET 标签为 {unresolved_ids}，其所需 ΔZ 下界全部为负。这一增长上界对它们没有排除力；不能称模型已成功解释这 7 桥。

## 恢复结构的保存代数边界

在相同固定前因子、读出、b 域及路径协议下，所有 24 个正增长桥的 m=1 端点未排除而 m=2 端点违反；ε=0 违反而 ε=1 未排除。检查了全部保存端点函数值与二分账本，未重新搜索阈值。m 使用 fail-hi 作上限，ε 使用 fail-lo 作下限，保守方向正确。

跨桥共享参数的必要条件由 **RESET{m_bound['reset_record']}→SET{m_bound['reset_record']+1}** 最严格：

- 单幂次恢复：m ≤ {m_bound['upper_m']:.15f}。
- 混合恢复：ε ≥ {e_bound['lower_epsilon']:.15f}。

这些是条件必要界，不是参数估计、微观反应阶数鉴定或轨迹充分条件，也不保证原 SET 精度。两种结构是分别筛选的备选形式，不应把两个边界联合解释成已验证的同一模型。

以 60 位十进制对既存端点做独立解析代入，所有最终报告的 m 上限及 ε 下限仍位于保守侧。{len(high_precision_pass_disagreements)} 桥的相邻“pass”浮点端点在高精度下略高于 U，差异约 10⁻¹⁴–10⁻¹⁵；这些未用于报告的必要界端点不影响结论，但说明相邻浮点夹点不能冒充严格区间证书。详细残差保存在 JSON；没有为此重跑模型或二分。

数值垫宽和裕度不代表实验噪声、仪器下限或置信区间。该结果不排除改变读出、κ/λ、恢复律、b 域或真实时间演化后的模型，也不否定所有统一 1R/1T1R 模型。

实际结果 SHA-256：`{review['population_result_sha256']}`。
'''
    with (HERE / 'ACTUAL_RESULT_REVIEW.md').open('x') as handle:
        handle.write(text)
    print(json.dumps({'status': review['status'], 'checks': check_count, 'sources': len(hashes),
                      'ratio_range': review['violation_budget_ratio_range'], 'shared_bounds': shared}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
