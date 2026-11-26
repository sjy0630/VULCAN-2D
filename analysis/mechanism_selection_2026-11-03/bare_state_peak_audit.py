"""Prior-phase bare read-state features and limited peak-current comparisons.

Original14 training records only. Descriptive slopes are observed-curve
summaries, not per-cycle material fits. No ODE, spectrum, or parameter search.
"""
from pathlib import Path
import csv
import hashlib
import json

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = ROOT/'analysis/shared_hbn_2026-11-03/data/native_observations.npz'
IDENTITY = ROOT/'analysis/bare_history_2026-11-03/identity_confirmation.json'
TRAIN = [1, 2, 3, *range(5, 16)]


def association(x, y):
    return {'n': len(x), 'pearson_r': float(np.corrcoef(x, y)[0, 1]),
            'spearman_rho': float(spearmanr(x, y).statistic),
            'p_values_not_used': True, 'scope': 'Same14-record association; no independent mechanism test.'}


def metrics(predicted_log, observed_log):
    error = np.asarray(predicted_log) - observed_log
    worst = int(np.argmax(abs(error)))
    return {'n': len(error), 'RMSE_decade': float(np.sqrt(np.mean(error**2))),
            'MAE_decade': float(np.mean(abs(error))),
            'median_absolute_error_decade': float(np.median(abs(error))),
            'median_absolute_factor': float(10.**np.median(abs(error))),
            'maximum_absolute_factor': float(10.**np.max(abs(error))),
            'worst_label': TRAIN[worst],
            'mean_signed_error_decade': float(np.mean(error))}


def main():
    identity = json.loads(IDENTITY.read_text())
    assert identity['confirmed']['bare_physical_device_count'] == 1
    assert identity['confirmed']['same_label_SET_then_RESET']
    features, provenance, loaded_keys = [], [], []
    digest = hashlib.sha256()
    with np.load(DATA) as z:
        for cid in TRAIN:
            raw = {}
            for phase in ['set', 'reset']:
                values = []
                for suffix in ['voltage', 'current']:
                    key = f'1r_cycle_{cid}_{phase}_{suffix}'
                    a = np.array(z[key], copy=True)
                    loaded_keys.append(key)
                    digest.update(key.encode()); digest.update(a.tobytes())
                    values.append(a)
                raw[phase] = values
            sv, signed_set = raw['set']; si = abs(signed_set)
            rv, signed_reset = raw['reset']; ri = abs(signed_reset)
            sa, ra = int(np.argmax(abs(sv))), int(np.argmax(abs(rv)))
            sj = int(np.flatnonzero((np.arange(len(sv)) > sa) & np.isclose(abs(sv), .05, atol=1e-8, rtol=0))[0])
            assert 1e-11 <= si[sj] < .995e-3
            gset = float(si[sj]/abs(sv[sj]))
            low = (np.arange(len(rv)) <= ra) & (abs(rv) >= .05) & (abs(rv) <= .200001)
            assert sum(low) == 7 and np.all(ri[low] >= 1e-12)
            x, y = abs(rv[low]), ri[low]
            g0 = float(np.dot(x, y)/np.dot(x, x))
            g1 = float(np.dot(x-x.mean(), y-y.mean()) / np.dot(x-x.mean(), x-x.mean()))
            intercept = float(y.mean()-g1*x.mean())
            assert g0 > 0 and g1 > 0
            peak = int(np.argmax(ri[:ra+1]))
            peak_current = float(ri[peak]); peak_v = float(abs(rv[peak]))
            peak_support = np.flatnonzero(ri[:ra+1] >= .99*peak_current)
            ranked = np.sort(ri[:ra+1])
            tail = int(np.flatnonzero((np.arange(len(rv)) > ra) & np.isclose(abs(rv), .2, atol=1e-8, rtol=0))[0])
            switch = int(np.flatnonzero(si[:sa+1] >= .995e-3)[0])
            assert switch > 0
            row = {
                'label': cid,
                'SET_return_50mV_actual_voltage_v': float(sv[sj]),
                'SET_return_50mV_current_a': float(si[sj]),
                'SET_return_50mV_conductance_s': gset,
                'SET_return_50mV_resistance_ohm': 1./gset,
                'RESET_50to200mV_origin_slope_s': g0,
                'RESET_50to200mV_origin_resistance_ohm': 1./g0,
                'RESET_50to200mV_intercept_slope_s': g1,
                'RESET_50to200mV_intercept_resistance_ohm': 1./g1,
                'RESET_50to200mV_fitted_intercept_a': intercept,
                'RESET_origin_slope_over_SET_conductance': g0/gset,
                'RESET_intercept_slope_over_origin_slope': g1/g0,
                'RESET_peak_current_a': peak_current,
                'RESET_peak_abs_voltage_v': peak_v,
                'RESET_peak_native_index_zero_based': peak,
                'RESET_peak_is_outward_endpoint': peak == ra,
                'RESET_peak_top_two_fractional_gap': float((peak_current-ranked[-2])/peak_current),
                'RESET_points_within_one_percent_of_peak': int(len(peak_support)),
                'RESET_near_peak_support_min_abs_voltage_v': float(min(abs(rv[peak_support]))),
                'RESET_near_peak_support_max_abs_voltage_v': float(max(abs(rv[peak_support]))),
                'RESET_peak_above_1mA': peak_current > .001,
                'RESET_peak_above_10mA_not_a_clipping_claim': peak_current > .010,
                'SET_switch_threshold_a': .995e-3,
                'SET_switch_last_below_voltage_v': float(sv[switch-1]),
                'SET_switch_first_crossing_voltage_v': float(sv[switch]),
                'SET_switch_bracket_midpoint_v': float((sv[switch-1]+sv[switch])/2),
                'SET_switch_bracket_width_v': float(sv[switch]-sv[switch-1]),
                'RESET_return_0p2V_actual_voltage_v': float(rv[tail]),
                'RESET_return_0p2V_current_a': float(ri[tail]),
                'RESET_return_0p2V_below_existing_1pA_floor': bool(ri[tail] < 1e-12),
                'conditional_Ipeak_over_Gset_Vpeak': peak_current/(gset*peak_v),
            }
            features.append(row)
            provenance.append({
                'label': cid, 'SET_return_50mV_index': sj,
                'RESET_low_voltage_indices': np.flatnonzero(low).tolist(),
                'RESET_low_abs_voltage_v': x.tolist(), 'RESET_low_raw_current_a': signed_reset[low].tolist(),
                'RESET_origin_fit_current_rmse_a': float(np.sqrt(np.mean((y-g0*x)**2))),
                'RESET_intercept_fit_current_rmse_a': float(np.sqrt(np.mean((y-(g1*x+intercept))**2))),
                'RESET_peak_neighborhood': [
                    {'index': k, 'voltage_v': float(rv[k]), 'raw_current_a': float(signed_reset[k])}
                    for k in range(max(0, peak-1), min(ra, peak+1)+1)],
                'RESET_return_0p2V_index': tail, 'SET_switch_crossing_index': switch,
                'signed_current_counts': {phase: {'negative': int(sum(raw[phase][1] < 0)), 'positive': int(sum(raw[phase][1] > 0))}
                                          for phase in ['set', 'reset']},
            })
    g = np.array([r['SET_return_50mV_conductance_s'] for r in features])
    gr = np.array([r['RESET_50to200mV_origin_slope_s'] for r in features])
    gf = np.array([r['RESET_50to200mV_intercept_slope_s'] for r in features])
    peak_i = np.array([r['RESET_peak_current_a'] for r in features])
    peak_v = np.array([r['RESET_peak_abs_voltage_v'] for r in features])
    x, y = np.log10(g), np.log10(peak_i)
    alpha, intercept = np.polyfit(x, y, 1)
    predicted = {key: np.zeros(len(TRAIN)) for key in ['constant_log_mean_LOO', 'one_variable_logG_linear_LOO', 'Gset_times_training_median_Vpeak_LOO']}
    folds = []
    for j, cid in enumerate(TRAIN):
        train = np.arange(len(TRAIN)) != j
        slope_j, intercept_j = np.polyfit(x[train], y[train], 1)
        median_v = float(np.median(peak_v[train]))
        predicted['constant_log_mean_LOO'][j] = float(np.mean(y[train]))
        predicted['one_variable_logG_linear_LOO'][j] = slope_j*x[j]+intercept_j
        predicted['Gset_times_training_median_Vpeak_LOO'][j] = x[j]+np.log10(median_v)
        folds.append({'left_out_label': cid, 'fit_labels': [c for c in TRAIN if c != cid],
                      'available_prior_phase_G_SET_s': float(g[j]), 'observed_RESET_peak_a': float(peak_i[j]),
                      'linear_logG_slope': float(slope_j), 'linear_logG_intercept_logA': float(intercept_j),
                      'training_median_Vpeak_v': median_v,
                      'predicted_RESET_peak_a': {k: float(10.**v[j]) for k, v in predicted.items()},
                      'signed_error_decade': {k: float(v[j]-y[j]) for k, v in predicted.items()}})
    comparisons = {k: metrics(v, y) for k, v in predicted.items()}
    baseline_mse = comparisons['constant_log_mean_LOO']['RMSE_decade']**2
    for result in comparisons.values():
        result['squared_log_error_reduction_vs_LOO_constant'] = 1-result['RMSE_decade']**2/baseline_mse
    conditional_log = x+np.log10(peak_v)
    conditional = metrics(conditional_log, y)
    conditional.update(uses_same_RESET_phase_Vpeak=True,
                       prediction_claim_allowed=False,
                       meaning='Conditional amplitude decomposition with observed outcome-phase Vpeak; never compared as a prospective predictor.')
    report = {
        'status': 'bare_training_prior_phase_feature_and_LOO_diagnostic_no_mechanism_claim',
        'training_labels_only': TRAIN,
        'confirmed_pairing': 'Same-label SET precedes RESET on the same physical device; no missing-cycle history is reconstructed.',
        'predeclared_comparisons': {
            'constant': 'For each left-out cycle, geometric mean peak current of the other13 cycles, minimizing their log squared error.',
            'one_variable': 'Fit log10(Ipeak/A)=a log10(G_SET/S)+b on13 other cycles; use only the left-out SET50mV read as its predictor.',
            'fixed_peak_voltage': 'Predict Ipeak=G_SET times median Vpeak of the other13 cycles; no multiplicative coefficient fitted.',
            'conditional_decomposition': 'Ipeak versus G_SET times its own observed RESET Vpeak; descriptive only, not prior-phase prediction.',
        },
        'feature_definitions': {
            'SET_G': '|I|/|V| at the single native nonlimited SET return50mV sample.',
            'RESET_origin_slope': 'sum(|V| |I|)/sum(V^2) over outward50..200mV; seven native points.',
            'RESET_intercept_slope': 'OLS slope/intercept of |I| on |V| over the same seven points; descriptive observations, not a cycle-specific physical model.',
            'RESET_peak': 'Largest current magnitude on the outward branch, first native index on ties; endpoint maxima would be retained.',
            'SET_switch': 'First outward crossing0.995mA; midpoint of previous/native-crossing voltages, with the full native bracket retained.',
            'RESET_tail': 'Native return−0.2V current magnitude; no interpolation or noise subtraction.',
        },
        'features': features, 'raw_feature_provenance': provenance,
        'same_data_associations': {
            'logG_SET_vs_logRESET_origin_slope': association(np.log10(g), np.log10(gr)),
            'logG_SET_vs_logRESET_intercept_slope': association(np.log10(g), np.log10(gf)),
            'logG_SET_vs_logIpeak': association(x, y),
            'logG_SET_vs_logVpeak': association(x, np.log10(peak_v)),
            'full_sample_logG_to_logIpeak_linear_description': {
                'slope': float(alpha), 'intercept_logA': float(intercept),
                'RMSE_decade_in_sample': metrics(alpha*x+intercept, y)['RMSE_decade'],
                'scope': 'Two global descriptive coefficients on the same14 records; not independent validation or per-cycle physics parameters.'},
        },
        'leave_one_record_out_comparisons': comparisons,
        'leave_one_record_out_folds': folds,
        'conditional_Gset_Vpeak_decomposition': conditional,
        'instrument_and_sampling_observations': {
            'SET_50mV_all_below_existing_compliance_flag': True,
            'RESET_peak_endpoint_labels': [r['label'] for r in features if r['RESET_peak_is_outward_endpoint']],
            'RESET_peak_above_1mA_labels': [r['label'] for r in features if r['RESET_peak_above_1mA']],
            'RESET_peak_above_10mA_labels': [r['label'] for r in features if r['RESET_peak_above_10mA_not_a_clipping_claim']],
            'scope': 'No RESET compliance or range metadata was supplied. Above10mA observations do not justify inventing a10mA limit, dropping cycles or labeling peaks as artifacts. Native peak neighborhood/near-peak ambiguity is preserved.',
            'zero_and_0p025V_RESET_points': 'Excluded from descriptive low-bias slopes; not used to infer offset correction.',
            'raw_sign_scope': 'RESET currents are stored positive at negative voltage; all signs retained in provenance, comparisons use magnitudes rather than imposing a signed-Ohmic convention.',
        },
        'scope': [
            'Only original14 training labels accessed; no holdout or new electrical model simulation.',
            'LOO uses13 other cycles which may occur later in time. It is internal cross-validation, not a chronological future-cycle test or independent external validation.',
            'Cycles of one device are not established IID samples; n=14 is small. No p-value, confidence interval, feature search, outlier deletion, or mechanism selection by statistical significance.',
            'Within-cycle prior-phase G_SET is valid input before RESET; its strong low-voltage association supports carrying a conductance-related state, but does not identify its microscopic origin or the minimum dynamical state dimension.',
            'Measured Vpeak is future RESET information. Its conditional decomposition residual must never be described as achieved prior-phase prediction accuracy.',
            'RESET slope fits and fold-wise statistical regressions are measurement summaries/prediction diagnostics, not per-cycle material-parameter fitting.',
        ],
        'loaded_array_keys': loaded_keys,
        'loaded_training_array_sha256': digest.hexdigest(),
        'identity_confirmation_sha256': hashlib.sha256(IDENTITY.read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUT/'BARE_STATE_PEAK_AUDIT.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    with (OUT/'BARE_STATE_PEAK_FEATURES.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(features[0]))
        writer.writeheader(); writer.writerows(features)
    lines = ['# 裸器件SET读态与随后RESET：特征及留一诊断', '',
             '**SET末端低压电导与随后RESET低压电流尺度高度相关。加入它可降低峰电流的留一RMSE，但中位误差反而增大；单一读电导没有解释全部峰值差异，14条同器件数据不能据此指定微观机制。**',
             '只用训练标签1、2、3、5–15。用户已确认同号先SET后RESET；未读保留集、未补缺圈、未运行ODE或拟合逐圈材料参数。', '',
             '## 观测特征', '',
             'G_SET取返扫+50 mV的|I|/|V|，14点均未限流。RESET低压斜率使用外扫50–200 mV的7点，分别作过零和带截距直线摘要；它们是数据特征，不是每圈新增的物理参数。SET切换为首次达到0.995 mA的原生电压括号中点，完整括号留在JSON/CSV，不把中点当精确切换阈值。', '',
             '| 标签 | G_SET/mS | R_SET/Ω | RESET过零G/mS | RESET带截距G/mS | 截距/µA | 峰/mA | 峰电压/V | SET切换中点/V | RESET尾0.2V/pA |',
             '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in features:
        lines.append(f'| {r["label"]} | {r["SET_return_50mV_conductance_s"]*1e3:.5g} | {r["SET_return_50mV_resistance_ohm"]:.5g} | {r["RESET_50to200mV_origin_slope_s"]*1e3:.5g} | {r["RESET_50to200mV_intercept_slope_s"]*1e3:.5g} | {r["RESET_50to200mV_fitted_intercept_a"]*1e6:.5g} | {r["RESET_peak_current_a"]*1e3:.5g} | {r["RESET_peak_abs_voltage_v"]:.3f} | {r["SET_switch_bracket_midpoint_v"]:.3f} | {r["RESET_return_0p2V_current_a"]*1e12:.5g} |')
    a = report['same_data_associations']
    lines += ['', '## 同数据关联：能保留什么结论', '',
              f'log G_SET与log RESET过零斜率的Pearson r={a["logG_SET_vs_logRESET_origin_slope"]["pearson_r"]:.6f}；改为带截距斜率后r={a["logG_SET_vs_logRESET_intercept_slope"]["pearson_r"]:.6f}。这个接近关系支持把SET留下的导电状态传入随后RESET，不支持在两相位间任意重新指定低阻尺度。',
              f'log G_SET与log RESET峰电流的Pearson r={a["logG_SET_vs_logIpeak"]["pearson_r"]:.6f}，Spearman rho={a["logG_SET_vs_logIpeak"]["spearman_rho"]:.6f}。关联减弱说明峰值不仅是低压读数的固定倍数；没有从此判断额外状态的微观身份或最低维数。', '',
              '## 用前相位信息的训练内留一比较', '',
              '三种比较在计算结果前固定：常数为其余13圈峰电流log均值；单变量回归为其余13圈的log G_SET→log Ipeak线性回归；固定峰电压比较为G_SET乘其余13圈峰电压中位数。留出的这一圈只能提供SET读电导，不能提供它自己的RESET峰电压。', '',
              '| 留一方法 | RMSE/decade | MAE/decade | 中位倍数误差 | 最大倍数误差 | 相对常数平方log误差减少 |',
              '|---|---:|---:|---:|---:|---:|']
    for key, r in comparisons.items():
        lines.append(f'| {key} | {r["RMSE_decade"]:.6f} | {r["MAE_decade"]:.6f} | {r["median_absolute_factor"]:.4f} | {r["maximum_absolute_factor"]:.4f} | {r["squared_log_error_reduction_vs_LOO_constant"]*100:.2f}% |')
    lines += ['', '改进取决于指标：平方log误差下降38.5%或46.6%，但中位倍数误差从常数的1.14增至1.26或1.41；不能概括为所有循环都预测得更好。',
              '留一训练折仍含时间上更晚的循环；这些是同一器件14条内的交叉验证，不是独立保留集或真正未来循环预测。不能因为其中一个结果较好就把它认定为最优物理模型。每折用到的13个标签、系数及预测都保存在JSON。', '',
              '## G_SET×同圈Vpeak只是条件分解', '',
              f'使用同圈实测RESET峰电压时，G_SET×Vpeak的log RMSE为{conditional["RMSE_decade"]:.6f} decade。这直接用了待解释相位的结果，不能与前面留一预测混称准确率。14圈Ipeak/(G_SET×Vpeak)范围{min(r["conditional_Ipeak_over_Gset_Vpeak"] for r in features):.6f}–{max(r["conditional_Ipeak_over_Gset_Vpeak"] for r in features):.6f}，也并非严格等于1。',
              '这项分解只显示低阻尺度、峰电压位置与峰前非线性/状态演化需要分别约束；不能从乘积关系证明某种缺陷、细丝或界面机制。', '',
              '## 仪器与峰位采样限制', '',
              '14条RESET峰均在扫描内部，并全部超过1 mA；标签3、10分别达到10.6723、14.3129 mA。因此不能把裸SET的1 mA限流套给RESET，也不能凭常见量程擅设10 mA上限。当前没有RESET仪器限流/量程元数据，没有据此删除异常点或判为仪器伪影。',
              '峰值取原生25 mV采样点；峰邻点、前两高点差及处于最高值1%内的电压范围均已保存，用来显示近峰平坦/波动可能影响峰位判定。这些描述不提供新误差条或检出限。',
              '零电压和RESET25 mV首点不进入低压斜率；未扣零点，也没有把带截距拟合出的截距直接解释为仪器偏置。RESET负电压下原电流存正值，报告使用幅值，不强制正负读点是相同欧姆电导。',
              '较稳妥的下一步约束是：任何最小扩展都应保留同圈SET读态→RESET低压尺度的传递，再解释峰位和峰前电流变化。此报告没有选定扩展方程或启动拟合。']
    (OUT/'BARE_STATE_PEAK_AUDIT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'n': len(features), 'associations': a,
                      'LOO': comparisons, 'conditional_not_prediction': conditional}, indent=2))


if __name__ == '__main__':
    main()
