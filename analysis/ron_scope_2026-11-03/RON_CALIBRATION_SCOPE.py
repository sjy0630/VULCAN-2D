"""Training-only upper constraints on one shared spectral conductance ceiling.

Exactly two declared read bands; no parameter scan, ODE, fit, or new seed.
Earlier audit products are read-only and remain frozen.
"""
from pathlib import Path
import hashlib
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = ROOT / 'analysis/shared_hbn_2026-11-03/data/native_observations.npz'
CONTROL_SOURCE = ROOT / 'analysis/parameter_transfer_2026-11-03/bare_entry_forward.json'
TRAIN = [1, 2, 3, *range(5, 16)]


def main():
    control_ron = json.loads(CONTROL_SOURCE.read_text())['checkpoint_metadata']['frozen_spectral_shape']['ron_ohm']
    samples = {'SET_return_50mV': [], 'RESET_outward_50_to_200mV': []}
    digest = hashlib.sha256()
    loaded_keys = []
    with np.load(DATA) as z:
        for label in TRAIN:
            for phase in ['set', 'reset']:
                a = []
                for suffix in ['voltage', 'current']:
                    key = f'1r_cycle_{label}_{phase}_{suffix}'
                    x = np.array(z[key], copy=True)
                    loaded_keys.append(key)
                    digest.update(key.encode()); digest.update(x.tobytes())
                    a.append(x)
                v, signed = a
                y = abs(signed)
                apex = int(np.argmax(abs(v)))
                ids = np.arange(len(v))
                if phase == 'set':
                    mask = (ids > apex) & np.isclose(abs(v), .05, rtol=0, atol=1e-8)
                    group = 'SET_return_50mV'
                    assert sum(mask) == 1
                    assert np.all(y[mask] < .995e-3), 'Declared SET read band must be below compliance flag'
                else:
                    mask = (ids <= apex) & (abs(v) >= .05) & (abs(v) <= .200001)
                    group = 'RESET_outward_50_to_200mV'
                    assert sum(mask) == 7
                assert np.all(np.isfinite(y[mask])) and np.all(y[mask] > 0)
                assert np.all(y[mask] >= (1e-11 if phase == 'set' else 1e-12))
                for k in np.flatnonzero(mask):
                    samples[group].append({'label': label, 'phase': phase,
                                           'raw_index_zero_based': int(k),
                                           'voltage_v': float(v[k]), 'raw_current_a': float(signed[k]),
                                           'abs_current_a': float(y[k]),
                                           'nominal_Ron_upper_ohm': float(abs(v[k]) / y[k])})
    groups = []
    for name, rows in [*samples.items(), ('combined_bare_read_bands', sum(samples.values(), []))]:
        witness = min(rows, key=lambda r: r['nominal_Ron_upper_ohm'])
        nominal = witness['nominal_Ron_upper_ohm']
        groups.append({'name': name, 'n_read_points': len(rows),
                       'nominal_Ron_upper_ohm': nominal, 'binding_observation': witness,
                       'upper_by_fractional_envelope_expansion': [
                           {'expansion': epsilon, 'Ron_upper_ohm': (1 + epsilon) * nominal}
                           for epsilon in [0.0, .05, .10]]})
    nominal_joint = groups[-1]['nominal_Ron_upper_ohm']
    declared = [
        {'role': 'frozen_old_control', 'Ron_ohm': control_ron,
         'definition': 'Original fixed value, retained as an explicit violating control rather than a newly accepted calibration.'},
        {'role': 'predeclared_nominal_upper_margin', 'Ron_ohm': .95 * nominal_joint,
         'definition': '0.95 times the nominal combined read-band upper bound; diagnostic margin, not a fitted optimum.'},
        {'role': 'predeclared_half_margin_sensitivity', 'Ron_ohm': .5 * .95 * nominal_joint,
         'definition': 'Half the preceding diagnostic value; broad sensitivity only, not a data-identified lower bound.'},
    ]
    report = {
        'status': 'training_only_data_derived_Ron_upper_scope_no_scan_no_ODE_no_fit',
        'training_labels': TRAIN,
        'history': 'Same physical device, numeric order and same-label SET-before-RESET user-confirmed; omitted measured cycles4/23 remain unavailable. No history bridge is needed for this pointwise bound.',
        'fixed_family_inequality': '|F(V,u)| <= |V| / Ron, all states and both polarities.',
        'necessary_upper': 'If Iobs <= (1+epsilon)|V|/Ron is required, then Ron <= (1+epsilon) min_read_points(|V|/Iobs).',
        'shared_parameter_policy': 'One common Ron for 1R and1T1R; no topology-specific, polarity-specific, or per-cycle Ron fitting.',
        'group_bounds': groups,
        'raw_selected_read_points': samples,
        'predeclared_Ron_sensitivity_points': declared,
        'not_identified': 'This envelope supplies an upper bound only. It supplies no positive lower bound; passing it does not establish feasibility of the spectral shape or dynamics.',
        'epsilon_scope': '0/5/10 percent are declared observation-envelope sensitivities, not confidence intervals, noise estimates, fitted error bars, or new primary gates.',
        'selection_exclusions': ['Zero-command points excluded', 'RESET0.025V points excluded',
                                 'SET return readings at0.1V and above excluded to avoid1mA compliance',
                                 'No holdout observations used', 'All retained points pass original phase scoring floors'],
        'downstream_cached_budget_requirements': [
            'These three points are fixed before the next diagnostic; do not select an optimization seed from this report.',
            'At fixed spectral shape and gamma, material current scales exactly as1/Ron; a frozen material-current table may be rescaled without new spectral quadrature.',
            'Series terminal currents do not scale linearly because the measured transistor shares the voltage. Re-solve the series load relation and rederive state/endpoint inversions for each common Ron.',
            'Use the same changed Ron in both circuit topologies; recalculate affected budget inputs before comparing rates. Do not transplant old terminal arrays as unchanged physics.',
            'Correcting an amplitude ceiling alone does not ensure peak timing, RESET tail, SET transition, state-history continuity, or joint-parameter transfer will improve.',
        ],
        'scope': [
            'The combined bound combines two bare read bands only; no new1T1R observations enter this calibration.',
            'Current magnitudes preserve the stored phase convention; this is a same-point envelope constraint, not a cross-polarity conductance equality.',
            'The nominal extreme is sensitive to acquisition offsets and other uncalibrated effects. The report does not assign all scatter to intrinsic material changes.',
            'No modification of the frozen transport, prior diagnostics, fitting masks, or parameter files.',
        ],
        'loaded_array_keys': loaded_keys,
        'loaded_training_array_sha256': digest.hexdigest(),
        'control_source_sha256': hashlib.sha256(CONTROL_SOURCE.read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUT / 'RON_CALIBRATION_SCOPE.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    lines = ['# 共同Ron的训练数据上界与预声明敏感性', '',
             '**按现有观测口径，冻结spectral族的共同Ron名义上必须不大于61.5400 Ω；即使允许10%包络宽容，上界仍只有67.6940 Ω。数据在这里仅给上界，不识别下界。**',
             '这是新阶段的范围说明，旧93.48008755 Ω及其全部结果保持冻结。只读取原训练14标签1、2、3、5–15；没有扫描、ODE、重新拟合或优化起点选择。', '',
             '## 上界与读点选择', '',
             '同一spectral材料理论对任意状态满足 `|F(V,u)|≤|V|/Ron`。若要求观测允许落在放宽包络 `Iobs≤(1+ε)|V|/Ron` 内，则必要条件是', '',
             '`Ron ≤ (1+ε) min(|V|/Iobs)`。', '',
             '使用SET返扫50 mV的14点，以及RESET外扫50–200 mV的98点。SET这14点均低于既有0.995 mA限流标记；RESET没有施加裸SET的1 mA限流。排除零命令点、RESET25 mV首点及容易受SET限流影响的较高返扫电压。所有保留点均高于原评分门槛。', '',
             '| 读点集合 | 点数 | 名义Ron上界/Ω | 5%宽容上界/Ω | 10%宽容上界/Ω |',
             '|---|---:|---:|---:|---:|']
    for row in groups:
        values = [x['Ron_upper_ohm'] for x in row['upper_by_fractional_envelope_expansion']]
        lines.append(f'| {row["name"]} | {row["n_read_points"]} | {values[0]:.9f} | {values[1]:.9f} | {values[2]:.9f} |')
    lines += ['', '名义联合上界由标签14的SET返扫点决定：V=+0.0500000007450581 V，I=0.812479236629 mA，V/I=61.540035106 Ω。RESET单独上界同样来自标签14：V=−0.0750000029802322 V，|I|=1.185802626424 mA，|V|/|I|=63.248302297 Ω。',
              '“联合”在此只表示合并两组裸器件读点，没有加入新的1T1R约束。5%/10%宽容是预声明敏感性，不能当成仪器不确定度、置信区间或新的主拟合门槛。', '',
              '## 预声明三个共同Ron诊断点', '',
              '| 角色 | Ron/Ω | 定义 |', '|---|---:|---|',
              f'| 旧结果控制 | {declared[0]["Ron_ohm"]:.12f} | 原93.48008755 Ω，保留为超出上述包络的控制点 |',
              f'| 名义上界留5%余量 | {declared[1]["Ron_ohm"]:.12f} | 0.95×61.540035106 Ω |',
              f'| 再减半的敏感性点 | {declared[2]["Ron_ohm"]:.12f} | 0.5×上一诊断点 |',
              '', '后两点仅用于随后固定的缓存预算敏感性比较，均不是优化起点或最优物理参数。0.95的余量与再减半是事先声明的诊断选择，不能将29.2315 Ω称为数据确定的下界，也不能称这些点已通过完整模型。',
              '仅由上述不等式，允许的Ron没有被识别出正的下界；更小Ron虽不违反这条天花板条件，却可能违背其他幅度、曲线形状或动力学约束。', '',
              '## 两接法使用同一个Ron，后续缓存需如何重算', '',
              '每个诊断点必须将同一个Ron用于1R与1T1R，不按圈、极性或拓扑分别调Ron。固定谱形状和Gamma时，材料电流严格随1/Ron缩放，因此可以重用材料谱缓存；但串联后的端口电流与材料电压受晶体管分压影响，不能把原series端口电流数组简单乘一个比例。',
              '后续需重解串联负载关系，并重新反演受影响的状态锚和端点预算，再比较SET速率条件。旧Ron下的端点/缓存预算不能原封不动迁移过去。',
              '同一器件与编号顺序已获确认，缺4/23确实测过但未提供；本篇是逐点必要上界，不依赖穿过缺圈的连续状态重建。测量零点、量程及其他未标定因素仍可能影响幅度，不能将全部散布自动归因于本征物理。',
              '**这项校准只修正可能不合适的固定导电上限。它不保证SET跃迁、RESET峰值/尾部、连续循环或两接法参数迁移会随之改善。**']
    (OUT / 'RON_CALIBRATION_SCOPE.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'group_bounds': groups, 'predeclared_points': declared}, indent=2))


if __name__ == '__main__':
    main()
