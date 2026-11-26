"""Model-independent raw boundary audit, original 14 bare training labels only.

No workbook, holdout array, material model, inferred chronology, ODE, or fit.
Numeric thresholds describe existing scoring masks, not calibrated noise.
"""
from pathlib import Path
import hashlib
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = ROOT / 'analysis/shared_hbn_2026-11-03/data/native_observations.npz'
TRAIN = [1, 2, 3, *range(5, 16)]
FLOOR = {'set': 1e-11, 'reset': 1e-12}
LOW_MAX = 0.200001  # Includes stored float32-origin 0.200000002980232 V.


def summary(values):
    values = np.asarray(values, float)
    return {'n': len(values), 'min': float(np.min(values)),
            'q10': float(np.quantile(values, .1)), 'median': float(np.median(values)),
            'q90': float(np.quantile(values, .9)), 'max': float(np.max(values))}


def main():
    raw, boundaries, loaded_keys = {}, {}, []
    digest = hashlib.sha256()
    with np.load(DATA) as z:
        for label in TRAIN:
            for phase in ['set', 'reset']:
                prefix = f'1r_cycle_{label}_{phase}'
                arrays = []
                for suffix in ['voltage', 'current']:
                    key = prefix + '_' + suffix
                    # Only these 56 named arrays are decompressed/accessed.
                    a = np.array(z[key], copy=True)
                    loaded_keys.append(key)
                    digest.update(key.encode()); digest.update(a.tobytes())
                    arrays.append(a)
                v, i = arrays
                assert np.all(np.isfinite(v)) and np.all(np.isfinite(i))
                apex = int(np.argmax(abs(v)))
                raw[label, phase] = (v, i)
                for branch, region in [('start', np.arange(len(v)) <= apex),
                                       ('end', np.arange(len(v)) > apex)]:
                    ids = np.flatnonzero(region & (abs(v) <= LOW_MAX))
                    samples = []
                    for k in ids:
                        samples.append({
                            'raw_index_zero_based': int(k),
                            'voltage_v': float(v[k]), 'abs_voltage_v': float(abs(v[k])),
                            'raw_current_a': float(i[k]), 'abs_current_a': float(abs(i[k])),
                            'zero_voltage': bool(v[k] == 0),
                            'below_existing_phase_scoring_floor': bool(abs(i[k]) < FLOOR[phase]),
                            'raw_current_negative': bool(i[k] < 0),
                            'raw_IV_product_negative': bool(v[k] * i[k] < 0),
                            'set_at_or_above_existing_compliance_flag': bool(phase == 'set' and abs(i[k]) >= .995e-3),
                        })
                    boundaries[label, phase, branch] = samples
    common = None
    for samples in boundaries.values():
        s = {x['abs_voltage_v'] for x in samples}
        common = s if common is None else common.intersection(s)
    common = sorted(common)
    assert len(common) == 5 and common[0] == 0.0
    points = common[1:]

    def at(label, phase, branch, voltage):
        return next(x for x in boundaries[label, phase, branch] if x['abs_voltage_v'] == voltage)

    scatter = []
    for phase, branch in [('set', 'start'), ('reset', 'end'), ('set', 'end'), ('reset', 'start')]:
        for voltage in points:
            samples = [at(cid, phase, branch, voltage) for cid in TRAIN]
            scatter.append({
                'phase': phase, 'branch': branch, 'abs_voltage_v': voltage,
                'raw_current_a': summary([x['raw_current_a'] for x in samples]),
                'abs_current_a': summary([x['abs_current_a'] for x in samples]),
                'below_existing_scoring_floor_labels': [cid for cid, x in zip(TRAIN, samples) if x['below_existing_phase_scoring_floor']],
                'set_compliance_flag_labels': [cid for cid, x in zip(TRAIN, samples) if x['set_at_or_above_existing_compliance_flag']],
            })

    def compare(label_a, phase_a, branch_a, label_b, phase_b, branch_b):
        readings = []
        for voltage in points:
            a, b = at(label_a, phase_a, branch_a, voltage), at(label_b, phase_b, branch_b, voltage)
            readings.append({
                'abs_voltage_v': voltage,
                'first_raw_current_a': a['raw_current_a'], 'second_raw_current_a': b['raw_current_a'],
                'second_over_first_magnitude_ratio': b['abs_current_a'] / a['abs_current_a'] if a['abs_current_a'] > 0 else None,
                'first_below_scoring_floor': a['below_existing_phase_scoring_floor'],
                'second_below_scoring_floor': b['below_existing_phase_scoring_floor'],
                'both_at_or_above_existing_scoring_floors': not (a['below_existing_phase_scoring_floor'] or b['below_existing_phase_scoring_floor']),
                'any_SET_compliance_flag': a['set_at_or_above_existing_compliance_flag'] or b['set_at_or_above_existing_compliance_flag'],
                'interpretation': 'Cross-polarity terminal-current magnitude ratio only; not equality or ratio of intrinsic conductance/state.',
            })
        return {'first_label': label_a, 'first_phase': phase_a, 'first_branch': branch_a,
                'second_label': label_b, 'second_phase': phase_b, 'second_branch': branch_b,
                'readings': readings}

    lrs = [compare(cid, 'set', 'end', cid, 'reset', 'start') for cid in TRAIN]
    hrs_same_label = [compare(cid, 'set', 'start', cid, 'reset', 'end') for cid in TRAIN]
    adjacent = [compare(cid, 'reset', 'end', cid+1, 'set', 'start') for cid in TRAIN if cid+1 in TRAIN]
    assert len(adjacent) == 12
    gap = compare(3, 'reset', 'end', 5, 'set', 'start')
    gap['interpretation'] = 'Cycle4 was measured but not supplied, as confirmed by the user. Its waveform/state effect is unknown; do not connect3->5 as adjacent cycles.'
    zeros = []
    for phase, branch in [('set', 'start'), ('set', 'end'), ('reset', 'start'), ('reset', 'end')]:
        samples = [at(cid, phase, branch, 0.0) for cid in TRAIN]
        zeros.append({'phase': phase, 'branch': branch,
                      'raw_current_a': summary([x['raw_current_a'] for x in samples]),
                      'per_label_raw_current_a': {str(cid): x['raw_current_a'] for cid, x in zip(TRAIN, samples)},
                      'interpretation': 'Recorded zero-command current; not conductance, not a measured noise distribution, not proof of state discontinuity.'})
    sign_summary = []
    for phase in ['set', 'reset']:
        v = np.concatenate([raw[cid, phase][0] for cid in TRAIN])
        i = np.concatenate([raw[cid, phase][1] for cid in TRAIN])
        sign_summary.append({'phase': phase, 'n_all_loaded_points': len(i),
                             'negative_current_count': int(sum(i < 0)), 'zero_current_count': int(sum(i == 0)),
                             'positive_current_count': int(sum(i > 0)),
                             'negative_IV_product_count': int(sum(v*i < 0))})
    report = {
        'status': 'raw_boundary_observation_audit_no_model_no_inferred_chronology',
        'training_labels_only': TRAIN,
        'user_confirmed_fact': 'Same physical device; numeric order corresponds to cycles; same-label SET precedes RESET; missing4/23 were measured but their data were not supplied.',
        'comparison_order_confirmed': True,
        'unconfirmed_history': ['Waveforms/state effects of omitted cycles4/23', 'Wait durations, intervening operations, wiring/sign conventions, and instrument noise calibration'],
        'common_abs_voltage_points_v_including_zero': common,
        'boundary_samples': [{'label': label, 'phase': phase, 'branch': branch, 'samples': samples}
                             for (label, phase, branch), samples in boundaries.items()],
        'cross_label_scatter': scatter,
        'sign_summary': sign_summary,
        'zero_command_observations': zeros,
        'conditional_same_label_SET_end_to_RESET_start': lrs,
        'same_label_HRS_start_end_observation_only': hrs_same_label,
        'conditional_numeric_adjacent_RESET_end_to_SET_start': adjacent,
        'numeric_gap_3_to_5_not_a_continuous_link': gap,
        'existing_scoring_floors_a_not_noise_calibration': FLOOR,
        'scope': [
            'Only original14 training-label arrays accessed; no holdout arrays or cloud workbook opened.',
            'All stored signs preserved. RESET negative command with positive stored current differs from the signed passive-current convention; raw polarity metadata is unresolved.',
            'No current division by signed voltage is used to declare equal conductance across polarities. Rectification, state evolution on read segments, and acquisition effects are not separated by this audit.',
            'Across-label min/quantiles/max summarize empirical scatter including possible cycle evolution; they are not confidence intervals or pure instrument noise.',
            'Existing10pA SET/1pA RESET masks are analysis inclusion thresholds, not independently measured detection limits. Below-threshold observations remain visible and ratios are flagged.',
            'SET return>=0.995mA is the existing compliance flag. Comparisons affected by it do not compare equal material voltages.',
            'Zero-command observations and RESET-only0.025V points are preserved but not turned into conductance or a phase-boundary state.',
            'Same-label and numeric-adjacent ordering is now user-confirmed. Magnitude comparisons still do not prove identical state; no state reset, interpolation across missing records, or silent carry-over is introduced.',
        ],
        'loaded_array_keys': loaded_keys,
        'loaded_training_array_sha256': digest.hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'RAW_BOUNDARY_AUDIT.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    lines = ['# 裸1R边界原始读数审计', '',
             '**用户已确认：同一器件、数字编号顺序对应、同号先SET后RESET；缺4/23是实际测过但未提供的数据。3→5隔着未知第4圈，不能当相邻循环直接连接。**',
             '只读原训练14标签：1、2、3、5–15的缓存；未访问保留集数组，未打开云端工作簿，也未计算物理模型。', '',
             '## 实际读点、符号与测量门槛', '',
             '两种扫描在起点/末端共同具有 |V|=0、0.05、0.10、0.15、0.20 V；存储的非零值分别为 '+', '.join(f'{v:.17g}' for v in points)+' V。无需插值。RESET另有0.025、0.075、0.125、0.175 V，不能与不存在的SET同电压点直接配对。',
             '14条RESET在负电压下仍存正电流，SET也全为正电流；未发现负号零星翻转。此缓存与带符号被动电流约定不同，不能据此判断负电阻。所有原始符号见JSON。',
             '现有评分门槛为SET10 pA、RESET1 pA，**不是已标定仪器噪声或检出限**；尚不能声称某个值低于真实噪声。表中明确标出低于分析门槛的记录，不删除它们。', '',
             '## 各边界跨标签散布', '',
             '这是可能包含循环演化的实测散布，不是仪器噪声估计或置信区间。电流列为幅值；跨极性不据此定义同一电导。', '',
             '| 边界 | 电压幅值/V | 电流单位 | min | Q10 | median | Q90 | max | 低于门槛数 | SET限流标记数 |',
             '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in scatter:
        a = row['abs_current_a']
        name = {'set_start': 'SET起点', 'set_end': 'SET末端', 'reset_start': 'RESET起点', 'reset_end': 'RESET末端'}[row['phase']+'_'+row['branch']]
        scale, unit = (1e12, 'pA') if (row['phase'], row['branch']) in [('set', 'start'), ('reset', 'end')] else (1e6, 'µA')
        lines.append(f'| {name} | {row["abs_voltage_v"]:.2f} | {unit} | {a["min"]*scale:.6g} | {a["q10"]*scale:.6g} | {a["median"]*scale:.6g} | {a["q90"]*scale:.6g} | {a["max"]*scale:.6g} | {len(row["below_existing_scoring_floor_labels"])} | {len(row["set_compliance_flag_labels"])} |')
    lines += ['', '## 已确认同号SET→RESET：低阻边界', '',
              '同号先SET后RESET的关系已确认；以下是这两个相接扫描的读数比较，尚不等于直接测得相同隐藏状态。0.05 V的SET末端14条均未触发限流标记；0.10 V已有7/14、0.15与0.20 V各12/14触发。故优先列0.05 V，不把高电压处限流差异解释成状态跳变。', '',
              '| 标签 | SET末端+0.05 V / µA | RESET起点−0.05 V / µA | RESET/SET幅度比 |',
              '|---:|---:|---:|---:|']
    for row in lrs:
        r = row['readings'][0]
        lines.append(f'| {row["first_label"]} | {r["first_raw_current_a"]*1e6:.6g} | {r["second_raw_current_a"]*1e6:.6g} | {r["second_over_first_magnitude_ratio"]:.6g} |')
    ratios = [x['readings'][0]['second_over_first_magnitude_ratio'] for x in lrs]
    lines += ['', f'14个幅度比范围{min(ratios):.6g}–{max(ratios):.6g}，中位数{np.median(ratios):.6g}。时序已由用户确认；读数接近本身仍不能证明相同隐藏状态或无整流。', '',
              '## 已确认数字相邻RESET→下一SET：高阻边界', '',
              '下表仅列训练范围内差1的12对数字标签，数字顺序与循环顺序对应已获确认。列0.20 V是为了减少低于现有门槛的记录；这个有限读电压仍不能保证读出不改变状态，也不能消除整流或未知等待的作用。幅度比为下一SET(+V)/前一RESET(−V)。', '',
              '| 标签连接 | RESET末端−0.20 V / pA | 下一SET起点+0.20 V / pA | SET/RESET幅度比 | 门槛标记 |',
              '|---|---:|---:|---:|---|']
    for row in adjacent:
        r = row['readings'][-1]
        flag = '均高于各自分析门槛' if r['both_at_or_above_existing_scoring_floors'] else 'SET低于10 pA门槛'
        lines.append(f'| {row["first_label"]}→{row["second_label"]} | {r["first_raw_current_a"]*1e12:.6g} | {r["second_raw_current_a"]*1e12:.6g} | {r["second_over_first_magnitude_ratio"]:.6g} | {flag} |')
    lines += ['', '3→5之间实际测过第4圈但数据未提供，因此不作为相邻连续连接；其纯读数对照单独保存在JSON。标签15之后不访问保留集。各对0.05/0.10/0.15 V原读数、门槛标记及同号HRS起末端对照均在JSON内。',
              '即便读数比明显偏离1，跨极性、等待/读出过程和仪器信息未解决前，也不能把它直接解释成记忆丢失或重新初始化。', '',
              '## 零命令电压与RESET独有首点', '',
              '| 边界 | 零命令电流min/A | median/A | max/A |', '|---|---:|---:|---:|']
    for row in zeros:
        a = row['raw_current_a']
        lines.append(f'| {row["phase"]} {row["branch"]} | {a["min"]:.6g} | {a["median"]:.6g} | {a["max"]:.6g} |')
    lines += ['', 'SET末端与RESET起点的零命令电流可达µA量级；RESET的−0.025 V首点也与后续低压读数不宜混为一条理想零偏直线。它们需要仪器符号、量程、稳定等待及位移电流等原始元数据解释，本篇不选择其中任何原因，也不把零命令点除以电压。',
              '编号顺序、同号SET后RESET以及缺号确实发生过测量已经确认，不再列为待确认事项。仍缺失的是第4/23圈具体波形、扫描间操作/等待时长、当前符号以及仪器低电流/零点元数据；特别是不能仅凭编号顺序重建缺失循环的状态作用。']
    (OUT / 'RAW_BOUNDARY_AUDIT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'loaded_training_labels': TRAIN, 'n_arrays_accessed': len(loaded_keys),
                      'common_abs_voltage_v': common, 'conditional_adjacent_pairs': len(adjacent),
                      'LRS_0p05V_ratio': summary(ratios), 'zero_observations': zeros}, indent=2))


if __name__ == '__main__':
    main()
