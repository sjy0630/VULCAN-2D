"""Training-only audit of the transistor premise of the SET resistance test.

No material parameters are fitted, no holdout arrays are loaded, and no circuit
mapping is promoted to a replacement. Alternative interpolants are assumptions,
not confidence intervals. All scalar outputs retain floating-point precision.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vulcan2d.transistor import TransistorLookup

OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'vulcan2d/data/fig2b_transistor_lookup.json'
DATA = ROOT / 'analysis/shared_hbn_2026-11-03/data/native_observations.npz'
IDS = list(range(4, 36))


def q(values):
    return np.quantile(np.asarray(values), [0., .5, 1.]).tolist()


def branches(raw):
    v, i = np.array(raw['drain_voltage_v']), np.array(raw['drain_current_a'])
    first, last = int(np.argmax(v)), len(v)-1-int(np.argmax(v[::-1]))
    up_v, up_i = v[:first+1], i[:first+1]
    down_i = np.interp(up_v, v[last:][::-1], i[last:][::-1])
    return up_v, up_i, down_i


class Map:
    def __init__(self, payload, branch='mean', voltage='linear', gate='linear', vg=1.1):
        self.gates = np.array([r['gate_voltage_v'] for r in payload['curves']])
        self.voltage, self.gate, self.vg = voltage, gate, vg
        self.curves = []
        for row in payload['curves']:
            v, up, down = branches(row)
            i = {'mean': .5*(up+down), 'up': up, 'down': down}[branch].copy()
            i = np.maximum(i, 0.)
            i[0] = 0.
            i = np.maximum.accumulate(i)
            self.curves.append((v, i, PchipInterpolator(v, i)))
        self.vmax = min(c[0][-1] for c in self.curves)

    def current(self, voltage, vg=None):
        v = np.asarray(voltage)
        g = self.vg if vg is None else vg
        values = np.stack([np.interp(v, x, y) if self.voltage == 'linear' else spline(v)
                           for x, y, spline in self.curves])
        low = int(np.clip(np.searchsorted(self.gates, g, side='right')-1, 0, len(self.gates)-2))
        w = (g-self.gates[low])/(self.gates[low+1]-self.gates[low])
        a, b = values[low], values[low+1]
        if self.gate == 'linear':
            return (1.-w)*a+w*b
        if self.gate == 'sqrt':
            return ((1.-w)*np.sqrt(a)+w*np.sqrt(b))**2
        if self.gate == 'log':
            return np.where(v == 0., 0., np.exp((1.-w)*np.log(np.maximum(a, 1e-300))+
                                               w*np.log(np.maximum(b, 1e-300))))
        return PchipInterpolator(self.gates, values, axis=0)(g)

    def inverse(self, current, scale=1., vg=None):
        """Smallest inverse; preserve the other endpoint for exact plateaus."""
        target = np.asarray(current)/scale
        maximum = self.current(self.vmax, vg)
        valid = (target >= 0.) & (target <= maximum)
        low = np.zeros_like(target)
        high = np.full_like(target, self.vmax)
        for _ in range(60):
            mid = (low+high)/2.
            smaller = self.current(mid, vg) < target
            low = np.where(smaller, mid, low)
            high = np.where(smaller, high, mid)
        left = (low+high)/2.
        low = left.copy()
        high = np.full_like(target, self.vmax)
        for _ in range(60):
            mid = (low+high)/2.
            larger = self.current(mid, vg) > target
            high = np.where(larger, mid, high)
            low = np.where(larger, low, mid)
        right = (low+high)/2.
        return np.where(valid, left, np.nan), np.where(valid, right, np.nan)


def contrast(mapping, currents, volts, scale=1., vg=None):
    left, right = mapping.inverse(currents, scale, vg)
    rh_low, rh_high = (volts-right)/currents, (volts-left)/currents
    ratio = rh_low[:, 1]/rh_high[:, 0]
    return {'ratio': ratio, 'vt_low': left, 'vt_high': right,
            'rh_low': rh_low, 'rh_high': rh_high}


def summary(mapping, currents, volts):
    result = contrast(mapping, currents, volts)
    ratio = result['ratio']
    return {'n_positive_Rh_increase': int(np.sum(ratio > 1.)),
            'ratio_min_median_max': q(ratio),
            'transistor_V2_min_median_max': q(result['vt_low'][:, 0]),
            'transistor_V4_min_median_max': q(result['vt_low'][:, 1]),
            'maximum_inverse_interval_width_V': float(np.max(result['vt_high']-result['vt_low'])),
            'per_cycle_ratio': ratio.tolist()}


def closest_root(function, start, end):
    """First continuous crossing from the unmodified mapping toward the bound."""
    samples = np.linspace(start, end, 201)
    previous, prev_y = samples[0], function(samples[0])
    for value in samples[1:]:
        y = function(value)
        if np.isfinite(y) and np.isfinite(prev_y) and prev_y*y <= 0.:
            return float(brentq(function, min(previous, value), max(previous, value), xtol=1e-13))
        previous, prev_y = value, y
    return None


def main():
    payload = json.loads(SOURCE.read_text())
    records = []
    with np.load(DATA) as raw:
        for cycle in IDS:
            voltage = raw[f'1t1r_cycle_{cycle}_set_voltage'].copy()
            current = abs(raw[f'1t1r_cycle_{cycle}_set_current'].copy())
            apex, last = int(np.argmax(voltage)), len(voltage)-1-int(np.argmax(voltage[::-1]))
            idx = [int(np.argmin(abs(voltage[:apex+1]-v))) for v in (2., 4.)]
            assert max(abs(voltage[idx]-[2., 4.])) < 1e-10
            records.append({'cycle': cycle, 'v': voltage, 'i': current, 'apex': apex, 'last': last, 'idx': idx})
    pairs = np.stack([r['i'][r['idx']] for r in records])
    volts = np.stack([r['v'][r['idx']] for r in records])
    baseline = Map(payload)
    check = TransistorLookup.from_payload(payload)
    test_v = np.linspace(0., 5., 501)
    assert np.max(abs(baseline.current(test_v)-[check.current(v, 1.1) for v in test_v])) < 1e-18
    base = contrast(baseline, pairs, volts)
    alt, inverse_spread = {}, []
    for branch in ('mean', 'up', 'down'):
        for voltage in ('linear', 'pchip'):
            for gate in ('linear', 'sqrt', 'log', 'pchip'):
                m = Map(payload, branch, voltage, gate)
                dense = m.current(np.linspace(0., 5., 10001))
                assert np.min(np.diff(dense)) >= -1e-15
                label = f'{branch}__Vds_{voltage}__Vg_{gate}'
                alt[label] = summary(m, pairs, volts)
                inverse_spread.append(m.inverse(pairs)[0])
    inverse_spread = np.stack(inverse_spread)
    all_v = np.concatenate([r['v'] for r in records])
    all_i = np.concatenate([r['i'] for r in records])
    positive = all_v > 0.
    scale_passive = float(np.max(all_i[positive]/baseline.current(all_v[positive])))
    required, high_voltage = [], []
    for k, row in enumerate(records):
        p, vv = pairs[k:k+1], volts[k:k+1]
        def scaled_difference(s):
            z = contrast(baseline, p, vv, s)
            return float(z['rh_low'][0, 1]-z['rh_high'][0, 0])
        scale_root = closest_root(scaled_difference, 1., max(scale_passive, .001))
        def gated_difference(g):
            z = contrast(baseline, p, vv, vg=g)
            return float(z['rh_low'][0, 1]-z['rh_high'][0, 0])
        gate_root = closest_root(gated_difference, 1.1, 1.0)
        i2, i4 = p[0]
        vt2, vt4 = base['vt_low'][k]
        ratio_i = i4/i2
        extra = float(i4*(base['rh_low'][k, 1]-base['rh_high'][k, 0]))
        extra_best = max(0., extra-ratio_i*vt2)
        # If both voltage errors can move adversarially, the bounded minimax
        # solution decreases Vt2 and increases Vt4; Vt2 cannot fall below zero.
        minimax = max(extra/(1.+ratio_i), extra_best)
        required.append({'cycle': row['cycle'], 'command_V': vv[0].tolist(),
            'current_A': p[0].tolist(), 'baseline_transistor_V': [float(vt2), float(vt4)],
            'baseline_Rh_ohm': base['rh_low'][k].tolist(),
            'Rh4_over_Rh2': float(base['ratio'][k]),
            'required_transistor_V4_if_V2_fixed': float(vt4+extra),
            'extra_transistor_V4_if_V2_fixed': extra,
            'minimum_extra_V4_allowing_V2_to_zero': extra_best,
            'minimum_joint_Linf_voltage_change_V': minimax,
            'closest_current_scale_removing_two_point_increase': scale_root,
            'closest_effective_gate_V_removing_two_point_increase': gate_root})
        v, ii = row['v'][:row['apex']+1], row['i'][:row['apex']+1]
        use = (v >= 2.) & (v <= 4.)
        x, y = v[use], ii[use]
        vt = baseline.inverse(y)[0]
        rh = (x-vt)/y
        slope_i = float(np.polyfit(x, y, 1)[0])
        slope_rh = float(np.polyfit(x, rh, 1)[0])
        noise = {}
        for window in (11, 31):
            residual = y-savgol_filter(y, window, 2)
            noise[str(window)] = {'residual_MAD_A': float(np.median(abs(residual-np.median(residual)))),
                                  'residual_RMS_A': float(np.sqrt(np.mean(residual**2)))}
        down = np.interp(x, row['v'][row['last']:][::-1], row['i'][row['last']:][::-1])
        # Window contrast is a robustness diagnostic, not modified source data.
        neighborhood = []
        for target in (2., 4.):
            near = abs(v-target) <= .10000000001
            tv = baseline.inverse(ii[near])[0]
            neighborhood.append(float(np.median((v[near]-tv)/ii[near])))
        high_voltage.append({'cycle': row['cycle'], 'n_points_2V_to_4V': int(sum(use)),
            'OLS_current_slope_A_per_V': slope_i, 'OLS_Rh_slope_ohm_per_V': slope_rh,
            'adjacent_current_decreases': int(np.sum(np.diff(y) < 0.)),
            'adjacent_Rh_increases': int(np.sum(np.diff(rh) > 0.)),
            'highpass_residual_diagnostics_not_measurement_uncertainty': noise,
            'return_over_outward_current_ratio_min_median_max': q(down/y),
            'endpoint_100mV_neighborhood_median_Rh_ratio': neighborhood[1]/neighborhood[0]})
    roots = np.array([r['closest_current_scale_removing_two_point_increase'] for r in required])
    gates = np.array([r['closest_effective_gate_V_removing_two_point_increase'] for r in required])
    common_scale, common_gate = float(min(roots)), float(min(gates))
    gate_current_margin = baseline.current(all_v[positive], common_gate)-all_i[positive]
    scale_current_margin = common_scale*baseline.current(all_v[positive])-all_i[positive]
    core_inverse = np.array([[check.voltage_for_current(float(i), 1.1) for i in row] for row in pairs])
    core_scale_residual = []
    for pair, s in zip(pairs, roots):
        drop = np.array([check.voltage_for_current(float(i/s), 1.1) for i in pair])
        resistance = (np.array([2., 4.])-drop)/pair
        core_scale_residual.append(float(resistance[1]-resistance[0]))
    assert np.max(abs(core_inverse-base['vt_low'])) < 1e-12
    assert np.max(np.abs(core_scale_residual)) < 1e-6
    source_diagnostics = []
    for row in payload['curves']:
        v, up, down = branches(row)
        mean = .5*(up+down)
        use = (v >= .2) & (v <= 2.5) & (mean > 1e-12)
        monotone = np.maximum.accumulate(np.maximum(mean, 0.))
        source_diagnostics.append({'gate_V': row['gate_voltage_v'], 'n_source_points': len(row['drain_voltage_v']),
            'Vds_grid_step_min_median_max': q(np.diff(v)),
            'up_down_relative_difference_0p2_to_2p5V_min_median_max': q(abs(up[use]-down[use])/mean[use]),
            'max_monotonization_increment_A': float(np.max(monotone-mean)),
            'raw_mean_negative_increment_count': int(np.sum(np.diff(mean) < 0.))})
    # Neighboring gates constrain only a broad monotone envelope, not Vg error.
    neighbor = {str(g): summary(Map(payload, vg=g), pairs, volts) for g in (1., 1.5)}
    source_up, source_down = Map(payload, 'up'), Map(payload, 'down')
    source_voltage_hyst = abs(source_up.inverse(pairs)[0]-source_down.inverse(pairs)[0])
    # Even arbitrary monotone Vds interpolation confines each inverse to its
    # neighboring measured voltage knots; this is broader than linear/PCHIP.
    knots = np.unique(np.concatenate([r[0] for r in baseline.curves]))
    knot_i = baseline.current(knots)
    bracket_hi = np.searchsorted(knot_i, pairs, side='left')
    bracket_lo = np.maximum(bracket_hi-1, 0)
    knot_vlo, knot_vhi = knots[bracket_lo], knots[bracket_hi]
    knot_ratio_lower = ((volts[:, 1]-knot_vhi[:, 1])/pairs[:, 1]) / (
                       (volts[:, 0]-knot_vlo[:, 0])/pairs[:, 0])
    source_hysteresis_at_target_gate = abs(source_up.current(test_v)-source_down.current(test_v))
    source_mean = baseline.current(test_v)
    source_use = (test_v >= .2) & (test_v <= 2.5)
    high_aggregate = {
        'n_positive_OLS_current_slope': sum(r['OLS_current_slope_A_per_V'] > 0. for r in high_voltage),
        'n_positive_OLS_Rh_slope': sum(r['OLS_Rh_slope_ohm_per_V'] > 0. for r in high_voltage),
        'OLS_current_slope_A_per_V_min_median_max': q([r['OLS_current_slope_A_per_V'] for r in high_voltage]),
        'OLS_Rh_slope_ohm_per_V_min_median_max': q([r['OLS_Rh_slope_ohm_per_V'] for r in high_voltage]),
        'endpoint_100mV_neighborhood_median_Rh_ratio_min_median_max': q([
            r['endpoint_100mV_neighborhood_median_Rh_ratio'] for r in high_voltage]),
        'return_outward_current_median_ratio_across_cycles_min_median_max': q([
            r['return_over_outward_current_ratio_min_median_max'][1] for r in high_voltage]),
        'highpass_RMS_11point_A_min_median_max': q([
            r['highpass_residual_diagnostics_not_measurement_uncertainty']['11']['residual_RMS_A']
            for r in high_voltage]),
    }
    output = {
        'status': 'training_only_circuit_premise_audit_no_material_fit_or_mapping_replacement',
        'training_cycle_ids': IDS, 'loaded_arrays': 'SET voltage/current only, cycles 4 through 35',
        'source': {key: payload[key] for key in payload if key != 'curves'},
        'gate_grid_V': baseline.gates.tolist(),
        'direct_measurement_at_1p1V': bool(np.any(baseline.gates == 1.1)),
        'baseline': summary(baseline, pairs, volts),
        'alternative_interpolation_assumptions_not_confidence_intervals': alt,
        'neighbor_gate_monotonic_envelope_not_allowed_gate_changes': neighbor,
        'monotone_Vds_interpolation_only_conservative_bound': {
            'gate_interpolation': 'fixed linear current at 1.1 V; raw mean branch',
            'inverse_voltage_brackets_V': np.stack([knot_vlo, knot_vhi], axis=-1).tolist(),
            'Rh_ratio_lower_bound_min_median_max': q(knot_ratio_lower),
            'n_strict_positive_Rh_increase_under_any_monotone_Vds_interpolation': int(sum(knot_ratio_lower > 1.)),
        },
        'per_cycle_constraints': required,
        'minimum_changes_summary': {
            'extra_V4_if_V2_fixed_min_median_max': q([r['extra_transistor_V4_if_V2_fixed'] for r in required]),
            'minimum_joint_Linf_V_change_min_median_max': q([r['minimum_joint_Linf_voltage_change_V'] for r in required]),
            'current_scale_root_min_median_max': q(roots),
            'effective_gate_root_V_min_median_max': q(gates),
            'common_current_scale_removing_all_32_two_point_increases': common_scale,
            'common_effective_gate_V_removing_all_32_two_point_increases': common_gate,
            'minimum_current_scale_for_passivity_at_every_positive_command_training_point': scale_passive,
            'common_scale_all_positive_command_points_passive': bool(np.all(scale_current_margin >= 0.)),
            'common_gate_all_positive_command_points_passive': bool(np.all(gate_current_margin >= 0.)),
            'common_gate_minimum_transistor_current_headroom_A': float(min(gate_current_margin)),
            'at_common_scale_Rh4_over_Rh2_min_median_max': q(contrast(baseline, pairs, volts, common_scale)['ratio']),
            'at_common_gate_Rh4_over_Rh2_min_median_max': q(contrast(baseline, pairs, volts, vg=common_gate)['ratio']),
            'independent_source_sweep_inverse_voltage_span_V2_min_median_max': q(source_voltage_hyst[:, 0]),
            'independent_source_sweep_inverse_voltage_span_V4_min_median_max': q(source_voltage_hyst[:, 1]),
            'all_tested_interpolants_V2_inverse_span_min_median_max': q(np.ptp(inverse_spread[:, :, 0], axis=0)),
            'all_tested_interpolants_V4_inverse_span_min_median_max': q(np.ptp(inverse_spread[:, :, 1], axis=0)),
            'linear_1p1V_source_up_down_relative_difference_0p2_to_2p5V_min_median_max': q(
                source_hysteresis_at_target_gate[source_use]/source_mean[source_use]),
        },
        'source_curve_diagnostics': source_diagnostics,
        'high_voltage_diagnostics': high_voltage,
        'high_voltage_aggregate': high_aggregate,
        'canonical_implementation_verification': {
            'max_baseline_inverse_voltage_difference_V': float(np.max(abs(core_inverse-base['vt_low']))),
            'max_abs_Rh_difference_at_scale_roots_ohm': float(np.max(np.abs(core_scale_residual))),
        },
        'interpretation': [
            'The original transport law plus nonincreasing SET gap requires Rh nonincreasing when I rises. The deembedded contradiction is conditional on transistor mapping and quasi-static voltage interpretation.',
            'Raw currents and source sweeps are not rounded; highpass residuals and window contrasts are diagnostics, not estimated instrument noise or confidence limits.',
            'Source up/down, Vds interpolation, and Vg interpolation changes are evaluated separately without altering core code.',
            'Vg=1.1 V is not directly sampled. Linear, square-root-current, log-current and PCHIP gate interpolants are sensitivity choices; their agreement does not close the unsampled-gate or cross-cell provenance gap.',
            'The nearby source gate curves give a wide monotone envelope; querying Vg=1.0 or 1.5 is not a claim that those were the target applied gate voltages.',
            'A common series resistance cannot remove the Rh difference: Vt_new=Vt+I*Rs gives Rh_new=Rh-Rs, leaving Rh4-Rh2 unchanged.',
            'Current-scale and effective-gate roots quantify how much mapping change would erase only the two-point constraint. They are not fitted transistor parameters and are not evidence for applying such changes.',
            'A current scale different from one produces the same fractional discrepancy against the stored standalone transistor law at every sampled gate. Sweep hysteresis measures repeat-path differences only, not transistor-to-transistor spread.',
            'No replicate device population, direct 1.1-V gate sweep, or same-cell transistor characterization is provided here; therefore no empirical confidence interval for cross-cell current_scale or gate_offset can be assigned.',
            'Removal of the two-point constraint does not imply joint fit, whole-branch monotonicity, RESET agreement, or a validated microscopic mechanism.',
        ],
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), SOURCE, DATA, ROOT/'vulcan2d/transistor.py')},
    }
    path = OUT/'circuit_constraint_audit.json'
    path.write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    info = output['minimum_changes_summary']
    rows = ['| 分支 / VDS 插值 / VG 插值 | 仍上升的圈数 | Rh4/Rh2 最小 / 中位 / 最大 |',
            '| --- | ---: | --- |']
    for name, value in alt.items():
        rows.append(f"| {name} | {value['n_positive_Rh_increase']} | " +
                    ' / '.join(f'{x:.9g}' for x in value['ratio_min_median_max'])+' |')
    text = f'''# 训练数据中的电路去嵌入前提审计

仅使用 1T1R 第 4–35 圈 SET 和独立 Fig.2b 晶体管原始曲线。未读取留出数据，
未拟合材料参数，未修改电路映射或核心。JSON 保留原始浮点精度；下文为显示缩写。

## 结论

默认映射下 **32/32 圈** Rh 从命令电压 2 V 到 4 V 上升，比例最小/中位/最大为
{output['baseline']['ratio_min_median_max']}。24 种原始扫描分支与插值组合均保留该趋势。
这表明该现象不由常见插值或独立 1T 的扫描迟滞单独造成；**仍不能解除同器件映射假设**。

维持 2 V 点的晶体管压降不变时，消除上升需要在 4 V 点额外增加的晶体管压降为
{info['extra_V4_if_V2_fixed_min_median_max']} V（最小/中位/最大）。
令 r=I4/I2，有 ΔVt4−rΔVt2 ≥ I4(Rh4−Rh2)。因此，若允许两点误差分别朝最有利方向变化，
最小共同绝对误差界为 max[D/(1+r), D−r Vt2]，D=I4(Rh4−Rh2)；对应
{info['minimum_joint_Linf_V_change_min_median_max']} V。它是数学界，不是测量误差。

## 哪些映射变化能消除这一条必要条件

- 每圈所需 current_scale 的根最小/中位/最大：{info['current_scale_root_min_median_max']}。
  共用比例 {common_scale} 能消除全部 32 圈的两点上升；相当于较独立 1T 插值整体降低
  {(1-common_scale)*100:.6f}% 电流。全训练点无源串联可行性只要求比例不低于 {scale_passive}，
  因此不能仅凭电路可行性排除这个变化。但原始独立 1T 测量并未支持这项整体缩放。
- 在原线性栅压插值下，每圈临界有效栅压最小/中位/最大为
  {info['effective_gate_root_V_min_median_max']} V；共用 {common_gate} V 消除全部两点上升。
  相比标称 1.1 V，偏移 {(common_gate-1.1)*1000:.6f} mV。这不是观测到的栅压误差。
  两种共同映射变化均通过所有正命令电压训练点的无源电路可行性检查，仍未证明材料模型能重现整条曲线。
- 增加共同串联电阻没有作用：Rh′=Rh−Rs，Rh4−Rh2 严格不变。

## 独立 1T 数据究竟约束了什么

源栅压网格为 {baseline.gates.tolist()} V，**没有 VG=1.1 V 的实测曲线**。
当前线性电流插值给 1.0/1.5 V 曲线 80%/20% 权重。
独立去/回扫产生的反演电压跨度：2 V 点 {info['independent_source_sweep_inverse_voltage_span_V2_min_median_max']} V；
4 V 点 {info['independent_source_sweep_inverse_voltage_span_V4_min_median_max']} V。
24 种插值/分支总跨度分别为 {info['all_tested_interpolants_V2_inverse_span_min_median_max']} V 和
{info['all_tested_interpolants_V4_inverse_span_min_median_max']} V。
这些是敏感性范围，不是置信区间。相邻实测栅压给出的单调包络甚至允许去嵌入趋势改变，
所以不能声称源数据在未测 1.1 V 处完全排除了另一种曲率；也不能把相邻栅压直接替换为真实栅压。
固定 1.1 V 线性栅压插值后，即使允许源 VDS 网格间任意单调插值，利用相邻电压节点得到
Rh 比保守下界最小/中位/最大 {q(knot_ratio_lower)}，仍有 {int(sum(knot_ratio_lower > 1.))}/32 圈上升。

'''+'\n'.join(rows)+f'''

## 高压散点、斜率与迟滞

JSON 的 high_voltage_diagnostics 保留每圈 2–4 V 全部点、原始相邻增减次数、
电流/Rh 线性斜率、11/31 点平滑后的高频残差、同电压去/回扫比以及 ±0.1 V 邻域中位数端点对比。
这些检验不修改输入数据，平滑残差不能等同仪器噪声。原始 1T 曲线的网格步长、
去/回扫差和单调化改变量另列 source_curve_diagnostics。
32 圈的电流和 Rh 线性斜率全部为正；Rh 斜率最小/中位/最大为
{high_aggregate['OLS_Rh_slope_ohm_per_V_min_median_max']} Ω/V。
±0.1 V 邻域检验的 Rh 比为
{high_aggregate['endpoint_100mV_neighborhood_median_Rh_ratio_min_median_max']}，全部保留上升。
11 点平滑高频残差 RMS 为 {high_aggregate['highpass_RMS_11point_A_min_median_max']} A；
各圈高压区回扫/去程电流的中位比范围为
{high_aggregate['return_outward_current_median_ratio_across_cycles_min_median_max']}。

## 判据适用边界

这项审计支持继续将“固定独立晶体管映射下的高压 Rh 上升”作为结构筛选条件。
但直接 1.1 V 数据、同一 cell 的晶体管标定和器件间差异统计均缺失，无法为 current_scale
或有效栅压指定有测量依据的置信范围。所需变化明确可量化，却不能仅凭现有独立 1T 数据
称其物理不可能。不得为救材料模型自动应用这些比例或栅压变化。
消除两点矛盾也不等于整条 SET/RESET、连续历史或裸 1R 已得到解释。
'''
    (OUT/'circuit_constraint_audit.md').write_text(text)
    print(json.dumps({'file': str(path), 'baseline': output['baseline'],
                      'minimum_changes_summary': info}, indent=2))


if __name__ == '__main__':
    main()
