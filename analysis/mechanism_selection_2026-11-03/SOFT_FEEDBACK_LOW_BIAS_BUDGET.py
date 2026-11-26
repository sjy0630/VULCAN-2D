"""Predeclared soft-only low-bias necessary-condition entry check.

Stops before beta/rate optimization when observed state bands cannot follow
monotone SET. Includes a native zero/low-voltage inventory without correction.
No ODE, new spectrum, physical-parameter fit, alternate read-point selection,
or holdout access. A two-point descriptive common-state fit quantifies cost.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import signal
import sys
import time

import numpy as np
from scipy.optimize import brentq, minimize_scalar

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RON_DIR = ROOT/'analysis/ron_scope_2026-11-03'
sys.path.insert(0, str(RON_DIR))
from ron_scaling_review import CachedRonTransport
from vulcan2d.transistor import TransistorLookup, measured_transistor_mapping

RON = 58.463033350725695
GC_REFERENCE_RON = 93.48008755
GC = .05*RON/GC_REFERENCE_RON
PLANNED_BETA = [1., 2., 4., 6.]
OBSERVATION_FRACTIONAL_BAND = .05
TABLE_FRACTIONAL_ENVELOPE = .003


def cached_transport(runtime_cache_dir):
    if runtime_cache_dir is None:
        return CachedRonTransport(), []
    cache = Path(runtime_cache_dir)
    table_path = cache/'transport_g010_v0025.npz'
    positive_path = cache/'fig2b_transistor_lookup.json'
    reverse_path = cache/'bookoutput2_reverse_transistor_lookup.json'
    transport = CachedRonTransport.__new__(CachedRonTransport)
    transport.table_path = table_path
    with np.load(table_path) as z:
        transport.lg = z['log_gamma'].copy()
        transport.lv = z['log_voltage'].copy()
        transport.li = z['log_current'].copy()
        transport.r0 = json.loads(str(z['metadata']))['shape']['ron_ohm']
    transport.mapping = measured_transistor_mapping(
        positive_lookup=TransistorLookup.from_json(positive_path),
        reverse_lookup=TransistorLookup.from_json(reverse_path))
    transport.knots = {}
    for phase in ['set', 'reset']:
        lookup = transport.mapping._active_lookup(phase)
        x = np.unique(np.concatenate([a['drain_voltage_v'] for a in lookup.curves()]))
        transport.knots[phase] = (x, np.array([transport.mapping.current(float(v), 1.1, phase) for v in x]))
    return transport, [positive_path, reverse_path]


def main(runtime_cache_dir=None):
    started = time.monotonic()

    def timeout(signum, frame):
        raise TimeoutError('90-second cached arithmetic budget exceeded')

    signal.signal(signal.SIGALRM, timeout); signal.alarm(90)
    data_path = (ROOT/'analysis/shared_hbn_2026-11-03/data/native_observations.npz'
                 if runtime_cache_dir is None else Path(runtime_cache_dir)/'native_observations.npz')
    budget_dir = RON_DIR if runtime_cache_dir is None else Path(runtime_cache_dir)
    json_path, npz_path = budget_dir/'RON_SCALING_REVIEW.json', budget_dir/'RON_SCALING_REVIEW.npz'
    normalization_path = budget_dir/'validate_segmented_joint_problem.json'
    normalization_source = json.loads(normalization_path.read_text())
    source = json.loads(json_path.read_text())
    ron_index = next(j for j, row in enumerate(source['rows']) if row['Ron_ohm'] == RON)
    saved = source['rows'][ron_index]
    with np.load(npz_path) as z:
        late_commands = z['soft_set_command_left_v'].copy()
        late_current = z[f'ron_{ron_index}_soft_set_current_lower_a'].copy()
        assert float(z[f'ron_{ron_index}_Ron_ohm']) == RON
    assert late_current.shape == late_commands.shape == (60,)
    keys = ['1t1r_cycle_4_set_voltage', '1t1r_cycle_4_set_current']
    loaded_digest = hashlib.sha256()
    with np.load(data_path) as z:
        arrays = []
        for key in keys:
            a = np.array(z[key], copy=True)
            loaded_digest.update(key.encode()); loaded_digest.update(a.tobytes())
            arrays.append(a)
    voltage, signed_current = arrays
    current = abs(signed_current)
    apex = int(np.argmax(abs(voltage)))
    outward = np.arange(len(voltage)) <= apex

    def native_index(target):
        ids = np.flatnonzero(outward & np.isclose(voltage, target, atol=1e-10, rtol=0))
        return int(ids[0]) if len(ids) == 1 else None

    inventory = []
    for k in np.flatnonzero(outward & (voltage >= 0) & (voltage <= .2000000001)):
        inventory.append({'raw_index_zero_based': int(k), 'command_voltage_v': float(voltage[k]),
                          'raw_current_a': float(signed_current[k]), 'abs_current_a': float(current[k]),
                          'existing_SET_scoring_eligible': bool(voltage[k] > 1e-12 and current[k] >= 1e-11),
                          'zero_command_not_inverted': bool(voltage[k] == 0.)})
    alternatives = [{'requested_voltage_v': value, 'native_index_zero_based': native_index(value),
                     'available': native_index(value) is not None,
                     'used_in_budget_or_state_inversion': False} for value in [.3, .5]]
    i1, i2 = native_index(.1), native_index(.2)
    usable = i1 is not None and i2 is not None
    if usable:
        usable = bool(all(np.isfinite(signed_current[j]) and signed_current[j] > 0 and current[j] >= 1e-11 for j in [i1, i2]))
    transport, lookup_paths = cached_transport(runtime_cache_dir)
    expected_table_hash = source['source_sha256']['analysis/spectral_dynamic_2026-11-03/transport_g010_v0025.npz']
    assert hashlib.sha256(transport.table_path.read_bytes()).hexdigest() == expected_table_hash
    cache_verification = {'transport_table': expected_table_hash}
    if runtime_cache_dir is not None:
        expected_raw = '0aedbe51f41b4f4355e1de4d26048c6790a9d454b8b7d56f51bd76a4671465a8'
        assert hashlib.sha256(data_path.read_bytes()).hexdigest() == expected_raw
        cache_verification['native_observations_from_frozen_data_inventory'] = expected_raw
        for path in lookup_paths:
            origin = str(ROOT/'vulcan2d/data'/path.name)
            expected = normalization_source['source_sha256'][origin]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
            cache_verification[path.name] = expected
    bands = []
    if usable:
        for j in [i1, i2]:
            endpoints = []
            for side, observed_factor, material_factor in [
                ('gamma_lower', 1-OBSERVATION_FRACTIONAL_BAND, 1+TABLE_FRACTIONAL_ENVELOPE),
                ('gamma_upper', 1+OBSERVATION_FRACTIONAL_BAND, 1-TABLE_FRACTIONAL_ENVELOPE),
            ]:
                observed = float(current[j]*observed_factor)
                transistor_v = float(transport.mapping.voltage_for_current(observed, 1.1, 'set'))
                local_v = float(voltage[j]-transistor_v)
                assert local_v > 0
                # T voltage uses the actual endpoint current, not its divided
                # table-current surrogate. Ftrue=(1+/-eta)Ftable envelopes
                # give the indicated conservative Gamma endpoint direction.
                table_target = observed/material_factor
                gamma = transport.inverse(local_v, table_target, RON, 'set')
                u = math.log1p(GC/gamma)
                reconstructed = float(transport.material(local_v, gamma, RON, 'set'))
                endpoints.append({'side': side, 'allowed_observed_current_a': observed,
                                  'transistor_voltage_v': transistor_v,
                                  'local_material_voltage_v': local_v,
                                  'material_envelope_factor': material_factor,
                                  'table_current_inversion_target_a': table_target,
                                  'gamma_ev': gamma, 'u': u,
                                  'table_inverse_relative_residual': reconstructed/table_target-1})
            bands.append({'command_voltage_v': float(voltage[j]), 'native_index_zero_based': j,
                          'raw_observed_current_a': float(signed_current[j]),
                          'raw_observed_current_band_a': [float(current[j]*(1-.05)), float(current[j]*(1+.05))],
                          'gamma_band_ev': [endpoints[0]['gamma_ev'], endpoints[1]['gamma_ev']],
                          'u_band': [endpoints[1]['u'], endpoints[0]['u']],
                          'endpoints': endpoints})
    delta = bands[0]['u_band'][1]-bands[1]['u_band'][0] if usable else None
    monotone_failure = bool(usable and delta < 0)
    # Do not turn negative Delta into a negative rate upper bound or silently
    # reverse the prescribed Gamma interval to obtain a spurious calculation.
    if usable and not monotone_failure:
        raise RuntimeError('Upstream state-order condition changed; this stopped-entry script must not silently substitute a new beta optimization')
    relaxation = None
    if usable:
        def gamma_endpoint(j, observation_factor, material_factor):
            endpoint_i = float(current[j]*observation_factor)
            local = float(voltage[j]-transport.mapping.voltage_for_current(endpoint_i, 1.1, 'set'))
            return transport.inverse(local, endpoint_i/material_factor, RON, 'set')

        def contact_function(delta_read):
            lower_01 = gamma_endpoint(i1, 1-delta_read, 1+TABLE_FRACTIONAL_ENVELOPE)
            upper_02 = gamma_endpoint(i2, 1+delta_read, 1-TABLE_FRACTIONAL_ENVELOPE)
            return math.log(lower_01/upper_02)

        assert contact_function(0.) > 0 and contact_function(.95) < 0
        minimum_delta = brentq(contact_function, 0., .95, xtol=1e-13, rtol=1e-13)
        contact_gamma = gamma_endpoint(i1, 1-minimum_delta, 1+TABLE_FRACTIONAL_ENVELOPE)
        # The independent zero-error Gamma values are reversed. For any
        # monotone increasing I(Gamma), minimizing the two separable log
        # errors subject to Gamma1<=Gamma2 puts the optimum at Gamma1=Gamma2.
        free_gamma = [gamma_endpoint(j, 1., 1.) for j in [i1, i2]]
        assert free_gamma[0] > free_gamma[1]
        left, right = np.log(min(free_gamma)), np.log(max(free_gamma))
        cuts = [float(left), *[float(x) for x in transport.lg if left < x < right], float(right)]
        commands = np.array([voltage[i1], voltage[i2]])
        observed = np.array([current[i1], current[i2]])
        normalized_group = next(g for g in normalization_source['group_checks']
                                if g['topology'] == '1t1r' and g['phase'] == 'set')
        n_phase = normalized_group['n']
        assert n_phase == 16000 and math.isclose(normalized_group['normalization'], math.sqrt(n_phase))

        def descriptive_fit(eta):
            memo = {}

            def evaluate(log_gamma):
                key = float(log_gamma)
                if key not in memo:
                    gamma = math.exp(key)
                    nominal_i, nominal_vh = transport.series(commands, gamma, RON, 'set')
                    if eta:
                        # F(Ron)/(1/(1+/-eta)) equals the declared material
                        # envelope. Re-solve transistor division at each of
                        # the envelope currents; these virtual Rons do not
                        # change the physical fixed-Ron scenario.
                        lower_i, _ = transport.series(commands, gamma, RON/(1-eta), 'set')
                        upper_i, _ = transport.series(commands, gamma, RON/(1+eta), 'set')
                        closest = np.minimum(np.maximum(observed, lower_i), upper_i)
                    else:
                        lower_i = upper_i = closest = nominal_i
                    errors = np.log10(closest/observed)
                    memo[key] = (float(np.mean(errors**2)), nominal_i, nominal_vh,
                                 lower_i, upper_i, closest, errors)
                return memo[key]

            candidates = [(evaluate(c)[0], c, 'native_logGamma_partition_boundary') for c in cuts]
            for lo, hi in zip(cuts[:-1], cuts[1:]):
                result = minimize_scalar(lambda c: evaluate(c)[0], bounds=(lo, hi),
                                         method='bounded', options={'xatol': 1e-12})
                assert result.success
                candidates.append((float(result.fun), float(result.x), 'bounded_within_native_logGamma_piece'))
            value, best_log_gamma, kind = min(candidates)
            _, predicted_i, local_vh, low_i, high_i, closest, errors = evaluate(best_log_gamma)
            sse = float(np.dot(errors, errors))
            return {'material_table_fractional_envelope': eta,
                    'descriptive_common_gamma_ev_not_a_seed': math.exp(best_log_gamma),
                    'nominal_KCL_predicted_current_a': predicted_i.tolist(),
                    'nominal_KCL_material_voltage_v': local_vh.tolist(),
                    'lower_envelope_KCL_current_a': low_i.tolist(),
                    'upper_envelope_KCL_current_a': high_i.tolist(),
                    'closest_envelope_current_a': closest.tolist(),
                    'per_point_signed_log_error_decade': errors.tolist(),
                    'per_point_absolute_log_error_decade': abs(errors).tolist(),
                    'two_point_RMSE_decade': math.sqrt(value),
                    'two_point_squared_log_error_sum': sse,
                    'original_series_SET_denominator': n_phase,
                    'contribution_to_original_series_SET_MSE': sse/n_phase,
                    'contribution_to_total_half_SSE_objective': sse/(2*n_phase),
                    'sqrt_phase_MSE_contribution_decade': math.sqrt(sse/n_phase),
                    'one_dimensional_objective_evaluations': len(memo), 'selected_numeric_candidate_type': kind,
                    'scope': 'Optimistic two-point common-state numerical consistency fit. It neither follows actual dynamics nor estimates physical parameters for a future candidate; envelope errors may vary independently between points.'}

        relaxation = {
            'minimum_symmetric_linear_relative_observation_band': minimum_delta,
            'table_fractional_envelope_for_contact': TABLE_FRACTIONAL_ENVELOPE,
            'contact_gamma_ev': contact_gamma,
            'contact_function_logGamma_ratio_residual': contact_function(minimum_delta),
            'contact_endpoint_observation_currents_a': [float(current[i1]*(1-minimum_delta)), float(current[i2]*(1+minimum_delta))],
            'free_nominal_gamma_at_each_observation_ev': free_gamma,
            'native_logGamma_partition_boundaries': cuts,
            'nominal_table_two_point_fit': descriptive_fit(0.),
            'table_envelope_relaxed_two_point_fit': descriptive_fit(TABLE_FRACTIONAL_ENVELOPE),
            'normalization_source': str(normalization_path),
            'normalization_source_sha256': hashlib.sha256(normalization_path.read_bytes()).hexdigest(),
            'contract_scope': 'The original20 primary-feature contract does not demand each cycle4 0.1/0.2V sample agree within5%. Failure of this added sensitivity band is not a ban on forward candidates that tolerate residuals at these points.',
        }
    jzero = native_index(0.)
    zero = None if jzero is None else {
        'native_index_zero_based': jzero, 'raw_current_a': float(signed_current[jzero]),
        'is_nonzero': bool(signed_current[jzero] != 0.),
        'relative_to_raw_0p1V_current': float(current[jzero]/current[i1]) if usable else None,
        'relative_to_raw_0p2V_current': float(current[jzero]/current[i2]) if usable else None,
        'subtracted_from_any_observation': False,
        'used_for_state_inversion': False,
    }
    low_nodes = [] if not usable else voltage[i1:i2+1].tolist()
    plan = {
        'beta_values': PLANNED_BETA, 'Ron_ohm': RON,
        'gamma_c_ev': GC, 'gamma_c_definition': '.05*Ron/93.48008755 eV',
        'parallel_ohmic_channel_amplitude_a': 0.,
        'low_path_native_command_nodes_v': low_nodes,
        'planned_gamma_partition_count': 64,
        'planned_JS_lower_a': 1e-7,
        'planned_early_path_length_v': 1.8,
        'late_native_cached_left_node_count': 60,
        'planned_joint_constraint': 'SameJS must satisfy early k_lower <= min(low-bias k_upper, late-return k_upper).',
    }
    report = {
        'status': 'stopped_before_beta_budget_due_to_conditional_low_bias_state_order_failure' if monotone_failure else 'unusable_predeclared_low_bias_read',
        'predeclared_plan': plan,
        'actual_beta_budget_cases_computed': 0,
        'actual_JS_optimizations': 0,
        'actual_descriptive_two_point_scalar_consistency_diagnostics': 2 if relaxation else 0,
        'actual_gamma_path_partitions_computed': 0,
        'actual_ODE_trajectories': 0, 'actual_new_spectral_quadrature_points': 0,
        'native_read_eligibility_passed': usable,
        'existing_SET_analysis_floor_a_not_instrument_noise_limit': 1e-11,
        'read_current_half_width_fraction': .05,
        'five_percent_is_new_sensitivity_not_original_primary_gate': True,
        'table_current_envelope_half_width_fraction': .003,
        'table_error_scope': {'historical_sampled_max_error_decade': .000989,
                             'equivalent_upward_relative_error': 10**.000989-1,
                             'chosen_relative_envelope': .003,
                             'why_not_0p0015': '0.000989 decade corresponds to about0.228%, larger than0.15%.',
                             'qualification': 'A declared0.3% envelope around the existing interpolated table, motivated by past sampled error; not a newly certified global interval bound.'},
        'endpoint_state_bands': bands,
        'maximum_allowed_monotone_SET_drop_delta_u': delta,
        'minimum_required_u_increase': -delta if monotone_failure else None,
        'gamma_0p1_lower_over_gamma_0p2_upper': bands[0]['gamma_band_ev'][0]/bands[1]['gamma_band_ev'][1] if usable else None,
        'monotone_SET_state_order_incompatible': monotone_failure,
        'reason_no_rate_optimization': 'For SET,du/dpath=-k[drive]+<=0. The two allowed state bands require u(0.2)>u(0.1). NegativeDelta invalidates the proposed low-bias rate interval before beta orJS is varied.',
        'four_beta_values_are_unexecuted_plan_not_a_four_case_scan': True,
        'quantified_band_relaxation_and_original_cost_impact': relaxation,
        'cycle4_native_outward_zero_to_0p2_inventory': inventory,
        'zero_command_observation': zero,
        'alternate_native_positions_inventory_only': alternatives,
        'unused_later_budget_cached_inputs': {
            'source_Ron_ohm': saved['Ron_ohm'],
            'anchor_reinversions': saved['anchor_reinversions'],
            'saved_anchor_u_fields_refer_to_source_gc_not_current_plan': True,
            'late_command_left_v': late_commands.tolist(),
            'late_cached_nominal_KCL_current_a': late_current.tolist(),
            'status': 'Loaded for provenance only; no later budget evaluated after upstream contradiction.',
        },
        'interpretation': [
            'Conditional failure of fixed spectral shape + a=0 series limit + interpreting low-bias observed current as pure passive conduction + monotone SET.',
            'Nonzero raw zero-command current means the observed low-bias signal cannot automatically be treated as a static passive conduction current. Bias/offset, displacement current or settling are possible unseparated contributions, not established explanations.',
            'No zero subtraction, current calibration, time constant, alternate read point, or new state initialization is introduced.',
            'This is not an empirical exclusion of beta>1, all parallel-channel extensions, or all material physics. A nonzero parallel channel or an explicit measurement/readout model is outside this a=0 diagnostic.',
            'The logical state-order condition precedes the four planned beta calculations. They were not run and no continuous-JS minimum or finite-grid result is reported.',
            'The added5% per-point band is not an original primary fitting gate; this diagnostic does not prohibit a forward trajectory with finite low-bias residuals or require adding an offset term.',
            'Nozero-voltage state inversion is meaningful for the passive material lawF(0,u)=0; the zero sample is inventory only.',
        ],
        'loaded_observation_array_keys': keys,
        'loaded_cycle4_arrays_sha256': loaded_digest.hexdigest(),
        'runtime_cache_dir': str(runtime_cache_dir) if runtime_cache_dir else None,
        'runtime_inputs_verified_against_frozen_source_checksums': cache_verification,
        'inherited_code_source_hashes_not_cloud_reread': {
            key: source['source_sha256'][key] for key in ['analysis/ron_scope_2026-11-03/ron_scaling_review.py', 'vulcan2d/transistor.py']},
        'source_sha256': {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [json_path, npz_path, normalization_path, data_path,
                                    transport.table_path,
                                    *(lookup_paths if lookup_paths else [ROOT/'vulcan2d/data/fig2b_transistor_lookup.json',
                                    ROOT/'vulcan2d/data/bookoutput2_reverse_transistor_lookup.json'])]},
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'elapsed_s': time.monotonic()-started,
    }
    assert report['elapsed_s'] < 90
    signal.alarm(0)
    (OUT/'SOFT_FEEDBACK_LOW_BIAS_BUDGET.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    if not usable:
        (OUT/'SOFT_FEEDBACK_LOW_BIAS_BUDGET.md').write_text('# Soft低偏预算入口检查\n\n预声明0.1/0.2 V原生读点不合格，停止；没有换点或开展β预算。详情见JSON。\n')
        return
    lines = ['# Soft反馈低偏预算：在状态次序检查处停止', '',
             '**实际计算了低偏读点的去嵌入、状态带及两点误差影响，尚未计算任何β/JS预算。固定谱、a=0纯被动导电读出和单调SET不能把这两个点同时限制在新增±5%诊断带内；这不是原20项主特征合同的失败判决，也不禁止允许低偏残差的forward。**',
             f'预声明计划为beta=[1,2,4,6]、Ron={RON:.12f} Ω、gc={GC:.12g} eV、series的并联欧姆通道a=0。只读第4圈SET的缓存数组和已存在的Ron场景锚/60点KCL缓存，没有保留集、新谱、ODE或拟合。', '',
             '## 原生读点与状态带', '',
             '0.1 V和0.2 V的原始电流分别为0.49979、0.83748 nA，都高于既有SET10 pA分析门槛。该门槛不是仪器噪声标定，也不保证观测电流全是静态导电分量。',
             '对每端采用原始电流±5%敏感性带，这是新增诊断而非原20项主特征的逐点硬门槛；固定表另取±0.3%幅度包络。旧采样最大误差0.000989 decade约为0.228%，不能用0.15%覆盖。0.3%仍是声明的敏感性包络，不是本次重新证明的全域误差界。',
             '端点去嵌入时晶体管压降使用允许的实际端口电流；只有材料表的反演目标除以1.003或0.997，不能把缩放后的代理电流拿去算晶体管压降。', '',
             '| 原生命令/V | 观测/nA | Gamma下界/eV | Gamma上界/eV | u下界 | u上界 |',
             '|---:|---:|---:|---:|---:|---:|']
    for row in bands:
        lines.append(f'| {row["command_voltage_v"]:.2f} | {row["raw_observed_current_a"]*1e9:.5f} | {row["gamma_band_ev"][0]:.10g} | {row["gamma_band_ev"][1]:.10g} | {row["u_band"][0]:.9f} | {row["u_band"][1]:.9f} |')
    lines += ['', f'最有利的允许SET闭合量Delta=u_0.1,max−u_0.2,min={delta:.9f}<0；换言之，两带至少要求u增加{-delta:.9f}。Gamma的0.1 V下界仍是0.2 V上界的{report["gamma_0p1_lower_over_gamma_0p2_upper"]:.6f}倍，带完全反序。',
              '既有SET律为du/d路径=−k[drive]₊≤0，不能产生这种状态增加。此矛盾先于β和JS选择：不能把负Delta当作速率上界，不能反转64格Gamma区间来继续算，也不能称“4个β扫描失败”。',
              '**实际β预算数=0，JS优化数=0，64格状态分片数=0。** 原2 V目标/5→2 V返扫60点缓存只作溯源保存，未在上游失败后继续运算。', '',
              '## 需要放宽多少，以及它对原目标函数有多大影响', '',
              f'使用完整晶体管去嵌入、材料反演及±0.3%材料包络，令两Gamma带刚好相接的最小对称线性电流半宽为δ={relaxation["minimum_symmetric_linear_relative_observation_band"]*100:.6f}%。这直接求解端口读数容许带，不是把Gamma比当成电流误差。',
              '独立反演的两个最优Gamma反序，因此在单调Gamma1≤Gamma2约束下，两点乐观误差最小值落在共同Gamma边界。以下仅用一个共同Gamma作数值一致性摘要；每次预测都重解晶体管分压，并分别在原表和±0.3%材料包络下求最有利误差。没有拟合动力学、实际循环初态或新优化seed。', '',
              '| 两点诊断 | 点1绝对log误差 | 点2绝对log误差 | 两点RMSE/decade | 原seriesSET的MSE贡献 | 原总cost贡献 |',
              '|---|---:|---:|---:|---:|---:|']
    for label, key in [('原表名义', 'nominal_table_two_point_fit'), ('0.3%材料包络放宽', 'table_envelope_relaxed_two_point_fit')]:
        fit = relaxation[key]
        lines.append(f'| {label} | {fit["per_point_absolute_log_error_decade"][0]:.9g} | {fit["per_point_absolute_log_error_decade"][1]:.9g} | {fit["two_point_RMSE_decade"]:.9g} | {fit["contribution_to_original_series_SET_MSE"]:.9g} | {fit["contribution_to_total_half_SSE_objective"]:.9g} |')
    lines += ['', '原seriesSET组有16000个评分点，归一化为1/√16000；两点平方log误差和为SSE时，phase MSE贡献=SSE/16000，总目标cost贡献=SSE/(2×16000)。保留这两个局部残差可能只占原目标很小的一部分，不能将新增5%局部诊断失败冒充原完整拟合合同失败。',
              '包络诊断允许两点误差在各自材料包络内独立取最有利值，且未施加真实动力学，因此仅是乐观描述性最小，不是已实现的预测，也不是全域数值区间证明。', '',
              '## 零命令点与0.02–0.2 V原生观测', '',
              '| 原生索引 | 命令/V | 原始电流/nA | 原评分纳入 |', '|---:|---:|---:|---|']
    for row in inventory:
        lines.append(f'| {row["raw_index_zero_based"]} | {row["command_voltage_v"]:.2f} | {row["raw_current_a"]*1e9:.5f} | {"是" if row["existing_SET_scoring_eligible"] else "否（零电压）"} |')
    lines += ['', f'0 V原始电流为{zero["raw_current_a"]*1e9:.5f} nA，约相当于0.1 V读数的{zero["relative_to_raw_0p1V_current"]*100:.2f}%、0.2 V读数的{zero["relative_to_raw_0p2V_current"]*100:.2f}%。纯被动静态导电律在0 V给F(0,u)=0；这个零点不能拿来反演状态。',
              '低偏偏置/零点、位移电流或未稳定过程都可能使“观测电流=静态导电电流”不成立，但本数据尚未将它们分开。本篇没有自行减零点、校准电流、假设时间常数或补充测量模型。',
              '0.3 V与0.5 V的原生位置已仅作inventory登记，未换成替代预算点，也未反演其状态。', '',
              '## 结论范围', '',
              '被否决的是“当前固定谱形状＋a=0的series极限＋这两个原始低偏点视为纯导电读出＋单调SET”的联合条件。它不等于实验证明beta>1错误，也不排除所有双通道扩展；并联通道非零、读出中存在额外电流、或输运形状需要调整都超出本次单一入口检查。',
              '本轮没有计算β联合必要速率交集，也没有新起点可报告。这个局部诊断不禁止任何允许两点有有限残差的forward候选，不强制加入offset项，更不把局部观测解释问题替换成全模型成功/失败。',
              f'缓存计算用时{report["elapsed_s"]:.3f}秒；上限90秒。详细电流带、晶体管压降、反演残差及来源hash见同名JSON。']
    (OUT/'SOFT_FEEDBACK_LOW_BIAS_BUDGET.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status': report['status'], 'elapsed_s': report['elapsed_s'],
                      'actual_beta_budget_cases_computed': 0, 'maximum_SET_drop_delta': delta,
                      'minimum_observation_delta': relaxation['minimum_symmetric_linear_relative_observation_band'],
                      'nominal_two_point_fit': relaxation['nominal_table_two_point_fit'],
                      'envelope_two_point_fit': relaxation['table_envelope_relaxed_two_point_fit'],
                      'zero_command_observation': zero}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-cache-dir', type=Path,
                        help='Verified local copies of table, rawnpz and two transistor lookup JSON files')
    main(parser.parse_args().runtime_cache_dir)
