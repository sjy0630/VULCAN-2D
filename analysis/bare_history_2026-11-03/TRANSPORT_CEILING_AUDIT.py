"""Necessary frozen-transport ceiling audit using bare training data only.

No ODE, optimization, new calibration, holdout arrays, or parameter changes.
Nominal strict comparisons and 5%/10% envelope sensitivity are separate.
"""
from pathlib import Path
import hashlib
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = ROOT / 'analysis/shared_hbn_2026-11-03/data/native_observations.npz'
FIXED_SOURCE = ROOT / 'analysis/parameter_transfer_2026-11-03/bare_entry_forward.json'
TRAIN = [1, 2, 3, *range(5, 16)]
FLOOR = {'set': 1e-11, 'reset': 1e-12}
EXPANSIONS = [0.0, 0.05, 0.10]


def main():
    frozen = json.loads(FIXED_SOURCE.read_text())['checkpoint_metadata']['frozen_spectral_shape']
    ron = frozen['ron_ohm']
    assert ron == 93.4800875524324
    digest = hashlib.sha256()
    loaded_keys, blocks = [], []
    with np.load(DATA) as z:
        for phase in ['set', 'reset']:
            vv, yy = [], []
            for label in TRAIN:
                pair = []
                for suffix in ['voltage', 'current']:
                    key = f'1r_cycle_{label}_{phase}_{suffix}'
                    a = np.array(z[key], copy=True)
                    loaded_keys.append(key)
                    digest.update(key.encode()); digest.update(a.tobytes())
                    pair.append(a)
                vv.append(pair[0]); yy.append(pair[1])
            v = np.array(vv); signed = np.array(yy); y = abs(signed)
            assert np.all(v == v[0]) and np.all(np.isfinite(y))
            original = (abs(v) > 1e-12) & (y >= FLOOR[phase])
            audit = original & (abs(v) >= 0.05)
            material_ceiling = abs(v) / ron
            upper = np.minimum(material_ceiling, .001) if phase == 'set' else material_ceiling
            ratio = np.divide(y, upper, out=np.zeros_like(y), where=upper > 0)
            apex = int(np.argmax(abs(v[0])))
            outward = np.broadcast_to(np.arange(v.shape[1]) <= apex, v.shape)
            returning = ~outward
            low = (abs(v) >= .05) & (abs(v) <= .200001)
            at50 = (abs(v) >= .05) & (abs(v) <= .050001)

            def witness(mask):
                if not np.any(mask):
                    return None
                r, k = np.unravel_index(np.argmax(np.where(mask, ratio, -np.inf)), ratio.shape)
                return {'label': TRAIN[r], 'raw_index_zero_based': int(k),
                        'branch': 'outward' if outward[r, k] else 'return',
                        'voltage_v': float(v[r, k]), 'raw_current_a': float(signed[r, k]),
                        'abs_current_a': float(y[r, k]), 'nominal_upper_a': float(upper[r, k]),
                        'observed_over_nominal_upper': float(ratio[r, k]),
                        'fractional_excess': float(ratio[r, k] - 1),
                        'absolute_excess_a': float(y[r, k] - upper[r, k])}

            def scope_summary(name, region):
                denom = original & region
                mask = audit & region
                ndenom, naudited = int(sum(denom.ravel())), int(sum(mask.ravel()))
                rows = []
                for expansion in EXPANSIONS:
                    exceed = mask & (ratio > 1 + expansion)
                    error = np.where(mask, np.maximum(np.log10(np.maximum(ratio, 1e-300) / (1 + expansion)), 0), 0)
                    rows.append({
                        'ceiling_fractional_expansion': expansion,
                        'n_exceeding': int(sum(exceed.ravel())),
                        'exceeding_labels': [label for label, row in zip(TRAIN, exceed) if np.any(row)],
                        'raw_log_RMSE_necessary_lower_bound_decade': float(np.sqrt(np.sum(error**2) / ndenom)) if ndenom else None,
                        'original_scoring_denominator_count': ndenom,
                        'audited_numerator_point_count': naudited,
                        'meaning': 'Pointwise ceiling-only lower bound; unaudited original-mask points contribute zero, so this remains conservative.',
                    })
                return {'name': name, 'n_original_eligible_in_scope': ndenom,
                        'n_audited_absV_ge_0p05': naudited, 'max_ratio_witness': witness(mask),
                        'sensitivity': rows}

            scopes = [scope_summary('all', np.ones(v.shape, bool)),
                      scope_summary('outward', outward), scope_summary('return', returning),
                      scope_summary('outward_0p05_to_0p2V', outward & low),
                      scope_summary('return_0p05_to_0p2V', returning & low),
                      scope_summary('return_0p05V', returning & at50)]
            per_label = []
            for r, label in enumerate(TRAIN):
                per_label.append({'label': label,
                                  'n_audited': int(sum(audit[r])),
                                  'max_ratio': float(np.max(ratio[r, audit[r]])),
                                  'n_exceeding_by_expansion': {str(t): int(sum(audit[r] & (ratio[r] > 1+t))) for t in EXPANSIONS},
                                  'outward_low_n_exceeding_by_expansion': {str(t): int(sum(audit[r] & outward[r] & low[r] & (ratio[r] > 1+t))) for t in EXPANSIONS},
                                  'return_50mV_ratio': float(ratio[r, np.flatnonzero(returning[r] & at50[r])[0]])})
            exceeding_points = []
            for r, k in np.argwhere(audit & (ratio > 1)):
                exceeding_points.append({'label': TRAIN[r], 'raw_index_zero_based': int(k),
                                         'branch': 'outward' if outward[r, k] else 'return',
                                         'voltage_v': float(v[r, k]), 'raw_current_a': float(signed[r, k]),
                                         'nominal_upper_a': float(upper[r, k]), 'ratio': float(ratio[r, k]),
                                         'active_nominal_ceiling': 'compliance' if phase == 'set' and material_ceiling[r, k] >= .001 else 'material',
                                         'exceeds_5pct': bool(ratio[r, k] > 1.05), 'exceeds_10pct': bool(ratio[r, k] > 1.10)})
            block = {'phase': phase, 'original_phase_scoring_floor_a': FLOOR[phase],
                     'scopes': scopes, 'per_label': per_label, 'all_nominal_exceeding_points': exceeding_points}
            if phase == 'set':
                for kind, mask in [('compliance', material_ceiling >= .001), ('material', material_ceiling < .001)]:
                    mask &= audit & (ratio > 1)
                    block[kind + '_ceiling_exceedances'] = {'count': int(sum(mask.ravel())), 'max_ratio_witness': witness(mask)}
            blocks.append(block)
    report = {
        'status': 'completed_fixed_transport_ceiling_necessary_condition_audit',
        'training_labels_only': TRAIN,
        'confirmed_history': 'Same physical device; numeric order corresponds to cycles; same-label SET precedes RESET; missing4/23 were measured but their data were not supplied.',
        'missing_history_policy': 'Cycle3->5 cannot be joined as adjacent; this ceiling is pointwise and does not need the missing waveform.',
        'fixed_Ron_ohm': ron,
        'Ron_source': 'Existing bare-training RESET0.05..0.2V slope-resistance median used by OrderedJointProblem; not the minimum resistance or a measured universal conductance ceiling.',
        'material_inequality': '|F(V,u)| <= |V|/Ron for every state in the fixed shared_spectral family.',
        'terminal_inequality_SET': '|I| <= min(1e-3 A, |Vcommand|/Ron)',
        'terminal_inequality_RESET': '|I| <= |Vcommand|/Ron',
        'nominal_tolerance': 'No added numerical or measurement tolerance; microscopic compliance overshoots are counted separately.',
        'sensitivity': 'Uniform5%/10% expansion of the nominal terminal envelope is diagnostic only, not a new fitting gate, uncertainty calibration, or modification of the physical model.',
        'raw_RMSE_lower_bound_formula': 'sqrt(sum_audited max(log10(Iobs/U)-log10(1+epsilon),0)^2 / N_original_branch_mask). All original eligible points at |V|<0.05V contribute zero.',
        'blocks': blocks,
        'scope': [
            'Only original14 bare-training labels and |V|>=0.05V points passing original phase masks enter excess counts; no holdout arrays accessed.',
            'No slope fitting, Ron change, spectrum calculation, ODE, optimization, or chronology reconstruction.',
            'The lower bounds are raw branch log-RMSE, not the separately weighted optimization objective. epsilon>0 refers to a relaxed envelope diagnostic.',
            'This condition ignores all dynamical restrictions and therefore cannot certify model accuracy or explain all residual error.',
            'For a universal fixed Ron, no hidden-state history can exceed the ceiling. Same-device cycles make the maximum-versus-median calibration distinction relevant, but do not by themselves prove which measured changes are intrinsic physics versus acquisition effects.',
            'Stored current polarity and zero-command anomalies remain documented in RAW_BOUNDARY_AUDIT; magnitudes here do not identify rectification or noise.',
        ],
        'loaded_array_keys': loaded_keys, 'loaded_training_array_sha256': digest.hexdigest(),
        'fixed_source_sha256': hashlib.sha256(FIXED_SOURCE.read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUT / 'TRANSPORT_CEILING_AUDIT.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    lines = ['# 固定输运电导上限：裸训练数据必要条件审计', '',
             '**当前Ron=93.48008755 Ω作为整个固定谱家族的电导上限，与部分裸循环的观测幅度冲突；该冲突在5%和10%宽松敏感性下仍存在。它独立于状态动力学。**',
             '只读原训练14标签1、2、3、5–15。用户现已确认同一器件、编号顺序对应、同号先SET后RESET，且缺4/23是实际测过但未提供的数据。未跨缺圈重建历史，未读保留集。', '',
             '## 上界为何与动力学无关', '',
             '当前冻结shared_spectral族对所有状态严格满足 `|F(V,u)|≤|V|/Ron`。裸RESET端口因此同样受限；裸SET还受1 mA限流，取 `min(1 mA, |Vcommand|/Ron)`。任何初态、速率或循环历史都不能突破这个固定上限。',
             'Ron来自原训练RESET0.05–0.2 V电阻估计的**中位数**，不是所有循环的最小电阻，也未被独立测定为普适最大电导的倒数。本篇不更改或重新拟合Ron。', '',
             '## 超界计数', '',
             '严格列不增加容差；5%/10%列仅将观测比较包络分别放宽，属于敏感性诊断，不是新主门槛或已知仪器不确定度。只计原评分门槛合格且电压幅值≥0.05 V的点。', '',
             '| 范围 | 审计点数 | 严格超界 | 超过5% | 超过10% | 最大相对超出 |',
             '|---|---:|---:|---:|---:|---:|']
    for block in blocks:
        names = ['all', 'outward_0p05_to_0p2V'] if block['phase'] == 'reset' else ['all', 'return_0p05V']
        for row in block['scopes']:
            if row['name'] not in names:
                continue
            w = row['max_ratio_witness']; counts = [x['n_exceeding'] for x in row['sensitivity']]
            lines.append(f'| {block["phase"]} {row["name"]} | {row["n_audited_absV_ge_0p05"]} | {counts[0]} | {counts[1]} | {counts[2]} | {100*w["fractional_excess"]:.4f}% |')
    lines += ['', 'SET严格超界143点中，136点只是1 mA限流附近的微小超调，最大0.0010991%；另外7点位于返扫0.05 V，是真正超过当前材料包络的读数，最大51.9013%。这两类不混作同等强度的物理证据。',
              '最大SET例为标签14返扫+0.050000000745 V：实测0.812479 mA，上界0.534873 mA。最大RESET例为标签3外扫−0.625 V：实测9.973462 mA，上界6.685916 mA，超出49.1712%。',
              'RESET0.05–0.2 V外扫共98点，严格超界49点；5%/10%放宽后仍各42点，来自标签3、6、7、9、14、15。标签8的7点严格超界但不足5%。该电压带返扫未见超界。', '',
              '## 仅由上界造成的原分支误差下界', '',
              '对每个观测，最有利预测不能超过包络，所以最小可能对数误差至少为 `max(log10(Iobs/U),0)`。将这些项平方求和，仍用原分支完整评分点数作分母。RESET的原评分包含±0.025 V，本篇未审计这些点，令其贡献为0，因此下界保守。', '',
              '| 原分支 | 原分母点数 | 严格下界/decade | 放宽5%包络下界 | 放宽10%包络下界 |',
              '|---|---:|---:|---:|---:|']
    for block in blocks:
        for row in block['scopes']:
            if row['name'] not in ['all', 'outward', 'return']:
                continue
            b = row['sensitivity']
            lines.append(f'| {block["phase"]} {row["name"]} | {row["n_original_eligible_in_scope"]} | {b[0]["raw_log_RMSE_necessary_lower_bound_decade"]:.9g} | {b[1]["raw_log_RMSE_necessary_lower_bound_decade"]:.9g} | {b[2]["raw_log_RMSE_necessary_lower_bound_decade"]:.9g} |')
    lines += ['', '这些是原始分支RMSE下界，不是分组加权目标函数，也不是预测性能估计。全RESET严格下界约0.022 decade，远小于此前失败前向的3.18 decade；电导上限校准问题确实存在，但不能用它解释掉所有动力学失败。', '',
              '## 同一器件循环确认后的含义', '',
              '同一材料理论允许同一器件的状态跨圈变化；但如果Ron被固定为所有状态共享的上限，其数值必须能够容纳需要解释的工作范围。把一组循环的中位低阻值直接当作绝对最大电导，会先天排除其中更高电流的循环，随后再调状态方程无法补救这个幅度限制。',
              '观测超界说明“这个固定包络＋当前读数口径”不一致，尚不能证明全部散布都来自材料本征变化。零点、量程、限流与接线/符号等测量因素仍待区分；5%/10%敏感性也不是误差标定。',
              '本篇仅定位必须澄清的上限校准问题；没有自动降低Ron、增加独立逐圈参数、改变已有门槛或重新拟合。完整逐点索引、分标签计数和来源hash见同名JSON。']
    (OUT / 'TRANSPORT_CEILING_AUDIT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'fixed_Ron_ohm': ron,
                      'blocks': [{'phase': b['phase'], 'all': b['scopes'][0], 'focused': b['scopes'][3 if b['phase']=='reset' else 5]} for b in blocks]}, indent=2))


if __name__ == '__main__':
    main()
