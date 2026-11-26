"""Training-only shape and passive-loss necessary conditions; no new model fit.

Reads the verified local native cache and a completed trial's frozen background.
The predeclared high-signal window is descriptive, never a replacement score mask.
"""
from pathlib import Path
import hashlib
import json
import signal
import time

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PLAN = HERE / 'RESET_SHAPE_PLAN.json'
TRIAL = ROOT / 'analysis/mechanism_selection_2026-11-03/field_rupture_trial.npz'
TRIAL_JSON = TRIAL.with_suffix('.json')
IDS = [1, 2, 3, *range(5, 16)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def delta_contact(currents, background, gs, gb, v, set_same_band):
    """Pointwise necessary interval contact; never clip negative excess."""
    currents = np.asarray(currents)
    gs = np.asarray(gs)
    if np.any(currents <= background) or np.any(gs <= gb):
        return dict(status='invalid_nominal_nonpositive_excess', delta=None)
    limits = [float(np.min(1-background/currents))]
    if set_same_band:
        limits.append(float(np.min(1-gb/gs)))
    limit = min(limits)

    def gap(d):
        g_low = (1-d)*gs-gb if set_same_band else gs-gb
        g_high = (1+d)*gs-gb if set_same_band else gs-gb
        i_low = (1-d)*currents-background
        i_high = (1+d)*currents-background
        assert np.all(i_low > 0) and np.all(g_low > 0)
        lo = i_low/(v*g_high)
        hi = i_high/(v*g_low)
        return float(max(lo)-min(hi))

    if gap(0.) <= 0:
        return dict(status='contact', delta=0., positive_excess_domain_upper_delta=limit)
    end = limit*(1-1e-10)
    if end <= 0 or gap(end) > 0:
        return dict(status='contact_not_reached_before_lower_excess_nonpositive', delta=None,
                    positive_excess_domain_upper_delta=limit,
                    scope='No clipping, log of nonpositive excess, or claim of impossibility beyond this diagnostic domain')
    d = float(brentq(gap, 0., end, xtol=2e-14))
    if background == 0 and gb == 0:
        ratios = currents/(v*gs)
        ratio = float(max(ratios)/min(ratios))
        x = np.sqrt(ratio) if set_same_band else ratio
        analytic = (x-1)/(x+1)
        assert abs(d-analytic) < 1e-11
    return dict(status='contact', delta=d, contact_interval_gap=gap(d),
                positive_excess_domain_upper_delta=limit)


def normalized_span(currents, background, gs, gb, v, members):
    if len(members) < 2:
        return dict(n=len(members), records=[IDS[j] for j in members], status='fewer_than_two_records')
    i = currents[members]
    g = gs[members]
    r = (i-background)/(v*(g-gb))
    assert np.all(r > 0)
    low, high = int(np.argmin(r)), int(np.argmax(r))
    return dict(n=len(members), records=[IDS[j] for j in members], status='computed',
                normalized_min=float(r[low]), normalized_max=float(r[high]),
                min_record=IDS[members[low]], max_record=IDS[members[high]],
                normalized_ratio_max_over_min=float(r[high]/r[low]),
                log10_span_decade=float(np.log10(r[high]/r[low])),
                SET_exact_delta=delta_contact(i, background, g, gb, v, False),
                SET_same_band_delta=delta_contact(i, background, g, gb, v, True))


def loss_features(r, select, peak, voltage):
    js = np.flatnonzero(select)
    if len(js) == 0:
        return dict(n=0)
    pairs = [j for j in range(1, len(r)) if select[j-1] and select[j]]
    rises = [j for j in pairs if r[j] > r[j-1]*(1+1e-12)]
    above = [int(j) for j in js if r[j] > 1+1e-12]
    best_recovery = None
    minimum = int(js[0])
    for j in js[1:]:
        ratio = float(r[j]/r[minimum])
        if best_recovery is None or ratio > best_recovery['factor']:
            best_recovery = dict(from_index=minimum, to_index=int(j),
                                 from_voltage_v=float(voltage[minimum]), to_voltage_v=float(voltage[j]), factor=ratio)
        if r[j] < r[minimum]:
            minimum = int(j)
    largest = max(rises, key=lambda j:r[j]/r[j-1]) if rises else None
    return dict(n=int(len(js)), first_index=int(js[0]), last_index=int(js[-1]),
                min_normalized=float(min(r[js])), max_normalized=float(max(r[js])),
                max_normalized_index=int(js[np.argmax(r[js])]),
                n_above_initial_amplitude=int(len(above)), above_initial_indices=above,
                adjacent_native_pairs_compared=len(pairs), local_increase_count=len(rises),
                local_increase_end_indices=rises,
                largest_adjacent_increase_factor=None if largest is None else float(r[largest]/r[largest-1]),
                largest_adjacent_increase_end_index=largest,
                maximum_ordered_recovery=best_recovery,
                peak_normalized=float(r[peak]) if select[peak] else None)


def shape_summary(nodes, subset):
    rows = [n for n in nodes if n[subset].get('status') == 'computed']
    all14 = [n for n in rows if n[subset]['n'] == len(IDS)]
    def summarize(rs):
        if not rs:
            return dict(n_nodes=0)
        worst = max(rs, key=lambda x:x[subset]['normalized_ratio_max_over_min'])
        d0 = [x[subset]['SET_exact_delta']['delta'] for x in rs if x[subset]['SET_exact_delta']['delta'] is not None]
        d1 = [x[subset]['SET_same_band_delta']['delta'] for x in rs if x[subset]['SET_same_band_delta']['delta'] is not None]
        return dict(n_nodes=len(rs), maximum_span_node_index=worst['native_index'],
                    maximum_span_branch=worst['branch'], maximum_span_abs_voltage_v=worst['abs_voltage_v'],
                    maximum_span=worst[subset],
                    maximum_computable_SET_exact_delta=max(d0) if d0 else None,
                    maximum_computable_SET_same_band_delta=max(d1) if d1 else None,
                    contact_not_reached_count_SET_exact=len(rs)-len(d0),
                    contact_not_reached_count_SET_same_band=len(rs)-len(d1))
    return dict(any_at_least_two=summarize(rows), all14=summarize(all14))


def scenario(name, bg, gb, obs, gs, v, mask, apex, peaks, high_min, bg_max):
    a0 = gs-gb
    assert np.all(a0 > 0)
    excess = obs-bg[None, :]
    valid = mask & (excess > 0)
    robust = valid & (obs >= high_min) & (bg[None, :] <= bg_max*obs)
    normalized = np.zeros_like(obs)
    nonzero = v > 0
    normalized[:, nonzero] = excess[:, nonzero]/(v[None, nonzero]*a0[:, None])
    features = []
    nodes = []
    for j, cid in enumerate(IDS):
        prepeak = np.arange(len(v)) <= peaks[j]
        features.append(dict(record=cid, SET_read_conductance_s=float(gs[j]),
            background_SET_read_conductance_s=float(gb), initial_path_amplitude_s=float(a0[j]),
            RESET_peak_native_index=int(peaks[j]), RESET_peak_abs_voltage_v=float(v[peaks[j]]),
            RESET_peak_current_a=float(obs[j, peaks[j]]),
            raw_peak_I_over_vGset=float(obs[j, peaks[j]]/(v[peaks[j]]*gs[j])),
            full_path_positive_only=loss_features(normalized[j], valid[j], peaks[j], v),
            full_path_high_signal=loss_features(normalized[j], robust[j], peaks[j], v),
            outward_prepeak_positive_only=loss_features(normalized[j], valid[j]&prepeak, peaks[j], v),
            outward_prepeak_high_signal=loss_features(normalized[j], robust[j]&prepeak, peaks[j], v),
            invalid_nonpositive_excess_count=int(np.sum(mask[j]&~valid[j])),
            high_signal_point_count=int(np.sum(robust[j]))))
    for k in np.flatnonzero(nonzero):
        members = np.flatnonzero(valid[:, k]).tolist()
        strong = np.flatnonzero(robust[:, k]).tolist()
        nodes.append(dict(native_index=int(k), branch='outward' if k <= apex else 'return',
            abs_voltage_v=float(v[k]), original_eligible_count=int(sum(mask[:, k])),
            nonpositive_excess_records=[IDS[j] for j in np.flatnonzero(mask[:, k]&~valid[:, k])],
            positive_only=normalized_span(obs[:, k], bg[k], gs, gb, v[k], members),
            high_signal=normalized_span(obs[:, k], bg[k], gs, gb, v[k], strong)))
    branch = []
    counters = dict(total_native_nonzero_pairs=0, original_both_eligible=0,
                    original_eligible_both_positive=0, high_signal_usable=0,
                    nonpositive_excess_among_original_eligible=0,
                    low_signal_among_original_positive=0,
                    excessive_background_among_original_positive=0)
    for j, cid in enumerate(IDS):
        pairrows = []
        for k in range(1, apex+1):
            ret = len(v)-1-k
            assert v[k] == v[ret] and ret > apex
            eligible = bool(mask[j,k] and mask[j,ret])
            positive = bool(excess[j,k] > 0 and excess[j,ret] > 0)
            strong = bool(robust[j,k] and robust[j,ret])
            low = bool(obs[j,k] < high_min or obs[j,ret] < high_min)
            highbg = bool(bg[k] > bg_max*obs[j,k] or bg[ret] > bg_max*obs[j,ret])
            product = float(np.log10(excess[j,k]/v[k])+np.log10(excess[j,ret]/v[ret])) if strong else None
            pairrows.append(dict(outward_index=k, return_index=ret, abs_voltage_v=float(v[k]),
                original_both_eligible=eligible, both_excess_positive=positive,
                both_high_signal_usable=strong, low_signal=low, excessive_background=highbg,
                log10_branch_product_s2=product))
            counters['total_native_nonzero_pairs'] += 1
            counters['original_both_eligible'] += int(eligible)
            counters['original_eligible_both_positive'] += int(eligible and positive)
            counters['high_signal_usable'] += int(strong)
            counters['nonpositive_excess_among_original_eligible'] += int(eligible and not positive)
            counters['low_signal_among_original_positive'] += int(eligible and positive and low)
            counters['excessive_background_among_original_positive'] += int(eligible and positive and highbg)
        logs = [p['log10_branch_product_s2'] for p in pairrows if p['both_high_signal_usable']]
        branch.append(dict(record=cid, native_pair_count=len(pairrows), high_signal_usable_pairs=len(logs),
            status='range_computed' if len(logs) >= 2 else 'not_testable_fewer_than_two_high_signal_pairs',
            branch_product_log10_span_decade=float(max(logs)-min(logs)) if len(logs) >= 2 else None,
            pairs=pairrows))
    counters['unusable_fraction_all_native_pairs'] = 1-counters['high_signal_usable']/counters['total_native_nonzero_pairs']
    counters['unusable_fraction_original_both_eligible'] = 1-counters['high_signal_usable']/counters['original_both_eligible']
    counters['reason_counts_overlap'] = True
    return dict(name=name, background_abs_current_a=bg.tolist(), background_SET_read_conductance_s=float(gb),
        original_point_count=int(mask.sum()), positive_excess_count=int(valid.sum()),
        nonpositive_excess_count=int(np.sum(mask&~valid)),
        nonpositive_return_excess_count=int(np.sum((mask&~valid)[:,apex+1:])),
        high_signal_point_count=int(robust.sum()),
        per_record_features=features,
        normalized_path_values_including_negative=[[float(x) if vk > 0 else None for x,vk in zip(row,v)] for row in normalized],
        invalid_or_low_signal_values_not_used_for_logs=True,
        fixed_voltage_common_shape=nodes,
        shape_span_summary=dict(positive_only=shape_summary(nodes,'positive_only'), high_signal=shape_summary(nodes,'high_signal')),
        branch_product_summary=counters, branch_product_by_record=branch)


def arbitrary_background(obs, gs, v, mask, apex, high_min):
    results=[]
    for delta in (0., .05, .1, .2):
        nodes=[]
        for k in np.flatnonzero(v > 0):
            members=np.flatnonzero(mask[:,k])
            currents=obs[members,k]
            demands=(1-delta)*currents-v[k]*(1+delta)*gs[members]
            imax=int(np.argmax(demands)); imin=int(np.argmin(currents))
            low=max(0.,float(demands[imax])); high=float((1+delta)*currents[imin])
            nodes.append(dict(native_index=int(k), branch='outward' if k <= apex else 'return',
                abs_voltage_v=float(v[k]), n_records=len(members), records=[IDS[j] for j in members],
                lower_common_background_a=low, upper_common_background_a=high,
                lower_over_upper=low/high, violates=low > high*(1+1e-12),
                lower_witness_record=IDS[members[imax]], upper_witness_record=IDS[members[imin]],
                both_witnesses_high_signal=bool(currents[imax] >= high_min and currents[imin] >= high_min),
                every_record_high_signal=bool(np.all(currents >= high_min))))
        failed=[n for n in nodes if n['violates']]
        worst=max(nodes,key=lambda n:n['lower_over_upper'])
        results.append(dict(delta=delta,n_nodes=len(nodes),violating_nodes=len(failed),
            violating_outward_nodes=sum(n['branch']=='outward' for n in failed),
            violating_return_nodes=sum(n['branch']=='return' for n in failed),
            violating_nodes_with_both_witnesses_high_signal=sum(n['both_witnesses_high_signal'] for n in failed),
            violating_nodes_with_all_records_high_signal=sum(n['every_record_high_signal'] for n in failed),
            worst_node=worst,nodes=nodes))
    return dict(scenarios=results,
        scope='Loose necessary bound for passive pure loss with a0<=G_SET and one nonnegative B(v) shared by cycles. No Gamma fitting; per-node pass is not full-trajectory feasibility; excludes no cycle/state-dependent background.')


def static_background_difference(obs, gs, v, mask, apex, high_min):
    results=[]
    for delta in (0., .05, .1, .2):
        pairs=[]
        for j,cid in enumerate(IDS):
            for k in range(1,apex+1):
                ret=len(v)-1-k
                assert v[k]==v[ret]
                io=float(obs[j,k]); ir=float(obs[j,ret]); vg=float(v[k]*gs[j])
                eligible=bool(mask[j,k] and mask[j,ret])
                low=(1-delta)*io-(1+delta)*ir
                high=(1+delta)*io-(1-delta)*ir
                bound=(1+delta)*vg
                pairs.append(dict(record=cid,outward_index=k,return_index=ret,abs_voltage_v=float(v[k]),
                    original_both_eligible=eligible,outward_current_a=io,return_current_a=ir,
                    nominal_difference_over_vGset=(io-ir)/vg,
                    allowed_difference_lower_a=low,allowed_difference_upper_a=high,
                    pure_loss_difference_upper_a=bound,
                    violates_upper=eligible and low>bound*(1+1e-12),
                    violates_lower=eligible and high < -1e-12*vg,
                    lower_difference_over_allowed_upper=low/bound,
                    outward_high_signal=io>=high_min,return_high_signal=ir>=high_min))
        eligible=[p for p in pairs if p['original_both_eligible']]
        upper=[p for p in eligible if p['violates_upper']]
        lower=[p for p in eligible if p['violates_lower']]
        results.append(dict(delta=delta,total_native_pairs=len(pairs),original_eligible_pairs=len(eligible),
            upper_violations=len(upper),lower_violations=len(lower),
            upper_violation_records=sorted(set(p['record'] for p in upper)),
            lower_violation_records=sorted(set(p['record'] for p in lower)),
            upper_violations_with_outward_high_signal=sum(p['outward_high_signal'] for p in upper),
            worst_upper_witness=max(eligible,key=lambda p:p['lower_difference_over_allowed_upper']),
            most_negative_nominal_witness=min(eligible,key=lambda p:p['nominal_difference_over_vGset']),
            per_record=[dict(record=cid,n_eligible=sum(p['record']==cid for p in eligible),
                upper_violations=sum(p['record']==cid for p in upper),lower_violations=sum(p['record']==cid for p in lower)) for cid in IDS],
            pairs=pairs))
    return dict(scenarios=results,
        scope='Same-v subtraction cancels any background static within each record, even if it differs by cycle. Requires passive pure loss with a0<=G_SET, no growth from SET read. Original mask retained; no log or requirement of high return current. Relative bands are sensitivity allowances.')


def main():
    started=time.monotonic()
    plan=json.loads(PLAN.read_text())
    def timeout(signum,frame):
        raise TimeoutError('Declared 60-second diagnostic budget exceeded; do not restart automatically')
    signal.signal(signal.SIGALRM,timeout)
    signal.alarm(plan['compute_budget_s'])
    source=Path(plan['runtime_native_cache'])
    assert sha(source) == plan['expected_native_cache_sha256']
    observations=[]; gs=[]; loaded=[]; reads=[]; digest=hashlib.sha256(); voltage=None
    with np.load(source) as z:
        for cid in IDS:
            phase_data={}
            for phase in ('set','reset'):
                values=[]
                for suffix in ('voltage','current'):
                    key=f'1r_cycle_{cid}_{phase}_{suffix}'
                    a=z[key].copy(); values.append(a); loaded.append(key)
                    digest.update(key.encode());digest.update(a.tobytes())
                phase_data[phase]=values
            sv,si=phase_data['set']; rv,ri=phase_data['reset']
            ap=int(np.argmax(abs(sv)))
            sj=int(np.flatnonzero((np.arange(len(sv))>ap)&np.isclose(abs(sv),.05,atol=1e-8,rtol=0))[0])
            assert 0 < abs(si[sj]) < .995e-3
            gs.append(abs(float(si[sj]/sv[sj])))
            reads.append(dict(record=cid,native_index=sj,raw_voltage_v=float(sv[sj]),raw_current_a=float(si[sj])))
            if voltage is None:voltage=rv
            else:np.testing.assert_array_equal(voltage,rv)
            observations.append(abs(ri))
    obs=np.array(observations); gs=np.array(gs); v=abs(voltage)
    apex=int(np.argmax(v)); mask=np.isfinite(obs)&(obs>=1e-12)&(v[None,:]>1e-12)
    assert int(mask.sum())==2799 and apex==100 and len(v)==202
    peaks=np.argmax(obs[:,:apex+1],axis=1)
    with np.load(TRIAL) as z:
        np.testing.assert_array_equal(z['record_ids'],IDS)
        np.testing.assert_array_equal(z['observed_reset_abs_current_a'],obs)
        np.testing.assert_array_equal(z['original_reset_mask'],mask)
        np.testing.assert_array_equal(z['prior_SET_conductance_s'],gs)
        bg=z['prior_SET_conditioned_training_leak_abs_current'].copy()
        gb_all=gs-z['prior_SET_conditioned_training_a0']
        assert np.all(gb_all==gb_all[0]) and np.all(z['prior_SET_conditioned_training_initial_state_clipped']==0)
        gb=float(gb_all[0])
        parameters=z['prior_SET_conditioned_training_parameters'].copy()
    trial=json.loads(TRIAL_JSON.read_text())
    assert digest.hexdigest()==trial['loaded_training_arrays_sha256']
    high_min=plan['robust_diagnostic_window']['observed_current_min_a']
    bg_max=plan['robust_diagnostic_window']['background_fraction_max']
    scenarios=[scenario('zero_background',np.zeros_like(v),0.,obs,gs,v,mask,apex,peaks,high_min,bg_max),
               scenario('frozen_prior_SET_conditioned_trial_background',bg,gb,obs,gs,v,mask,apex,peaks,high_min,bg_max)]
    result=dict(status='completed_training_only_necessary_shape_diagnostic_no_new_physical_fit',
        predeclared_plan=plan, training_ids=IDS, original_RESET_mask_count=int(mask.sum()),
        original_outward_mask_count=int(mask[:,:apex+1].sum()),original_return_mask_count=int(mask[:,apex+1:].sum()),
        original_mask_counts_by_record=mask.sum(axis=1).tolist(),
        fixed_trial_parameters=parameters.tolist(),native_abs_voltage_v=v.tolist(),SET_read_provenance=reads,
        scenarios=scenarios,arbitrary_common_background_relaxation=arbitrary_background(obs,gs,v,mask,apex,high_min),
        static_within_record_background_difference_bound=static_background_difference(obs,gs,v,mask,apex,high_min),
        numerical_checks=dict(training_native_arrays_match_frozen_trial=True,training_array_hash_match=True,
            original2799_mask_exact=True,zero_background_delta_contacts_match_analytic=True,
            no_negative_excess_clipping=True,all_normalized_values_finite=True),
        scope=[
            'Only original14 training IDs accessed. No held-out array loaded, no scoring denominator or original feature gate changed.',
            'Normalized a0 is conditional on the measured preceding SET50mV read; this does not predict SET formation or full history.',
            'A pure-loss channel must stay below a0 and decline along acquisition path. Raw local increases and values above1 remain in outputs.',
            'The10uA and1% window is predeclared robustness screening, not calibrated instrument resolution or new fitting acceptance.',
            'Relative tolerances are deterministic sensitivity allowances, not measurement noise estimates, confidence intervals, or fitted physical parameters.',
            'Per-node interval overlap is only an optimistic necessary condition: the same SET-read perturbation must serve all voltages, and one common monotone shape must work across nodes.',
            'Branch-product requires positive resolvable excess on both branches; absence of such points leaves this identity untested, not failed.',
            'Common nonnegative voltage-only hazard additionally assumes equal path clock on outward and return, symmetric triangle and no wait. Timing remains unknown.',
            'Fixed full-training background is a conditional scenario, not independently established truth. No choice of arbitrary free background was fitted.',
            'Failures do not identify a microscopic replacement mechanism or exclude state-dependent backgrounds, non-ohmic high channels, nonmonotone readout, or protocol variation.'],
        actual_new_ODE_calls=0,actual_new_physical_parameter_fits=0,actual_new_spectral_quadrature_points=0,
        loaded_native_array_keys=loaded,loaded_training_array_sha256=digest.hexdigest(),
        source_sha256={str(p):sha(p) for p in (source,TRIAL,TRIAL_JSON,PLAN)},script_sha256=sha(Path(__file__)),
        elapsed_s=time.monotonic()-started)
    (HERE/'RESET_SHAPE_DIAGNOSTIC.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    write_markdown(result)
    signal.alarm(0)
    print(json.dumps(dict(status=result['status'],elapsed_s=result['elapsed_s'],
        scenarios=[dict(name=s['name'],nonpositive_excess=s['nonpositive_excess_count'],
            high_signal_points=s['high_signal_point_count'],branch_product=s['branch_product_summary'],
            all14_high_signal_span=s['shape_span_summary']['high_signal']['all14']) for s in scenarios],
        arbitrary_common_background=[{k:v for k,v in x.items() if k!='nodes'} for x in result['arbitrary_common_background_relaxation']['scenarios']],
        static_background_difference=[{k:v for k,v in x.items() if k not in ('pairs','per_record')} for x in result['static_within_record_background_difference_bound']['scenarios']]),indent=2))


def write_markdown(r):
    lines=['# 训练14圈 RESET 共同纯损失形状诊断','',
        '只读标签1、2、3、5–15；原2799个RESET评分点（外扫1400、返扫1399）及原门槛保持。没有新物理拟合、ODE、谱扫描或保留集访问。',
        '', '## 预声明的数学条件与范围','',
        '令 v=|V|，Q=(I−B)/v，a₀=G_SET−G_B,SET，r=Q/a₀。被动纯损失要求 r≤1，且沿实际采集顺序非增；共用衰减形状还要求各圈的r在同一分支/电压重合。所有r>1、局部上升及负扣背景结果均保留，不截零、不对负数取log。',
        '零背景与上一试验条件模型的全训练冻结背景分别报告。后者只作为固定情景，不是独立确证的真实背景。前相位SET读数作为条件输入，不代表已经预测形成过程。',
        '稳健窗口预先固定为观测电流≥10 µA、B/I≤1%、扣背景后正；这是诊断窗口，不是新的评分掩码，也不是已校准仪器分辨率。无测量误差标定；下列相对容差仅为灵敏度比较。','',
        '## 峰前幅度与局部增加','',
        '| 情景 | 圈 | 峰电压/V | Ipeak/(vpeak GSET) | 高信号峰前r最大值 | r>1点数 | 邻接增加次数/可比对数 | 最大邻接增加倍数 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in r['scenarios']:
        for f in s['per_record_features']:
            a=f['outward_prepeak_high_signal']
            lines.append(f'| {s["name"]} | {f["record"]} | {f["RESET_peak_abs_voltage_v"]:.3f} | {f["raw_peak_I_over_vGset"]:.6f} | {a.get("max_normalized",float("nan")):.6f} | {a.get("n_above_initial_amplitude",0)} | {a.get("local_increase_count",0)}/{a.get("adjacent_native_pairs_compared",0)} | {a.get("largest_adjacent_increase_factor")} |')
    lines += ['', '完整沿路径的增加、最大顺序恢复、全部原生r值和不可用点均在JSON；峰位只用于描述已观测峰前窗口，不是预测特征。','',
              '## 固定电压下共同形状需要多少容差','',
              '设正归一化读数跨度 R=rmax/rmin。零背景、SET读数固定时，最小对称原电流相对半带为δ=(R−1)/(R+1)；RESET与SET原读数各自允许同一半带时，δ=(√R−1)/(√R+1)。非零背景先对原读数区间减固定背景，再求归一化区间接触；不把δ直接套到扣背景余量。若下界先变非正，明确标不可计算，不截零。',
              '每一节点的交集只是乐观必要条件：实际同一SET读数扰动必须对所有电压一致，且要存在一条共同单调形状。逐点可行不代表整条轨迹可行。','',
              '| 情景 | 全14圈同时高信号的节点数 | 最大跨度所在分支、电压/V | 最大R | 该点SET固定最小δ | 该点SET同带最小δ |',
              '|---|---:|---|---:|---:|---:|']
    for s in r['scenarios']:
        q=s['shape_span_summary']['high_signal']['all14']
        if q['n_nodes']:
            w=q['maximum_span'];d0=w['SET_exact_delta']['delta'];d1=w['SET_same_band_delta']['delta']
            lines.append(f'| {s["name"]} | {q["n_nodes"]} | {q["maximum_span_branch"]}, {q["maximum_span_abs_voltage_v"]:.3f} | {w["normalized_ratio_max_over_min"]:.6f} | {d0:.6%} | {d1:.6%} |')
        else:lines.append(f'| {s["name"]} | 0 | 不可用 | — | — | — |')
    lines += ['', 'JSON逐节点保留全部正值集合和高信号集合的成员、跨度、所需容差。部分圈子集不能冒充全部14圈共同检验；仅正但低信号的跨度只能作条件性描述。','',
              '## 同圈双分支乘积：先看是否能评价','',
              '对于固定背景、共同电压危险率λ(v)≥0、同路径时钟、无等待的对称三角扫描，Qout(v)Qret(v)=a₀² exp[−2H(M)]应与v无关。仅在两支同时通过高信号窗口且扣背景为正时才计算log乘积。','',
              '| 情景 | 原生非零电压配对 | 原评分两支均可用 | 其内两支扣背景均正 | 双支高信号可用 | 原生配对不可用比例 |',
              '|---|---:|---:|---:|---:|---:|']
    for s in r['scenarios']:
        b=s['branch_product_summary']
        lines.append(f'| {s["name"]} | {b["total_native_nonzero_pairs"]} | {b["original_both_eligible"]} | {b["original_eligible_both_positive"]} | {b["high_signal_usable"]} | {b["unusable_fraction_all_native_pairs"]:.2%} |')
    if all(s['branch_product_summary']['high_signal_usable']==0 for s in r['scenarios']):
        lines += ['', '**两情景均没有双支同时满足稳健窗口的配对，因而本轮没有可靠地检验分支乘积恒等式；不能把“无可用配对”写成恒等式被实验否定。**']
    lines += ['', '固定背景高于/等于观测的点直接标无效；不会以自由拟合背景挽救负余量。低信号与背景比例问题的原因计数可能重叠，JSON列出每一配对。','',
              '## 任意共同非负背景的松弛必要界','',
              '若前SET读电导确实给出a₀≤G_SET，且RESET全过程只损失被动通道，并有同圈间共享的B(v)≥0，则每个节点必须满足 L=max(0,max_j[I_j−vG_j])≤U=min_j I_j。因为G_SET已含背景，这只是保守上界，不需要扫描Γ。',
              '灵敏度带同时施加在原RESET与SET读数上：Lδ=max(0,max_j[(1−δ)I_j−v(1+δ)G_j])，Uδ=min_j[(1+δ)I_j]。允许的原评分点全部保留；未按此界改评分。','',
              '| 相对半带δ | 违反节点/200 | 外扫/返扫 | 两个见证电流均≥10µA的违反节点 | 全14圈均≥10µA的违反节点 | 最大L/U | 其电压/V、分支 |',
              '|---:|---:|---:|---:|---:|---:|---|']
    for x in r['arbitrary_common_background_relaxation']['scenarios']:
        w=x['worst_node']
        lines.append(f'| {x["delta"]:.0%} | {x["violating_nodes"]}/{x["n_nodes"]} | {x["violating_outward_nodes"]}/{x["violating_return_nodes"]} | {x["violating_nodes_with_both_witnesses_high_signal"]} | {x["violating_nodes_with_all_records_high_signal"]} | {w["lower_over_upper"]:.6g} | {w["abs_voltage_v"]:.3f}, {w["branch"]} |')
    lines += ['', '此界失败仅约束上述条件组合；不排除随圈/状态演化的背景，也不确认额外通道、离子、缺陷或接触机制。通过此松弛界同样不证明完整动力学可行。',
              '', '## 同圈背景相消的分支差界','',
              '若同一圈的B(v)不随分支/状态改变，即使不同圈背景不同，同电压相减后仍有0≤Iout−Iret≤vG_SET：因为aout≥aret≥0且aout≤a₀≤G_SET。它不依赖Γ或谱形，也不要求两支采用同一电压危险率；仍要求前SET读点至RESET期间没有通道增长。',
              '原电流相对带给出差值区间[(1−δ)Iout−(1+δ)Iret,(1+δ)Iout−(1−δ)Iret]，必须与[0,v(1+δ)G_SET]相交。这里只做线性差分、不取log，故不要求返扫也达到10µA；原两支评分有效条件仍保留，不假装已知pA量级的仪器误差。',
              '', '| 相对半带δ | 原可用配对 | 超过上界配对 | 低于零下界配对 | 上界违反圈 | 最强见证L差/(vG上界) | 见证圈、电压/V |',
              '|---:|---:|---:|---:|---|---:|---|']
    for x in r['static_within_record_background_difference_bound']['scenarios']:
        w=x['worst_upper_witness']
        lines.append(f'| {x["delta"]:.0%} | {x["original_eligible_pairs"]} | {x["upper_violations"]} | {x["lower_violations"]} | {x["upper_violation_records"]} | {w["lower_difference_over_allowed_upper"]:.6f} | {w["record"]}, {w["abs_voltage_v"]:.3f} |')
    lines += ['', '该界失败约束“同圈静态背景＋被动纯损失＋SET读态上界”这一组合；不能据此确定是哪一假设失效，更不能宣称所有两通道模型失败。SET读数误差不能变成逐电压重新初始化的物理参数。',
              '', '## 可复现性','',
              f'原始运行缓存SHA256：`{r["source_sha256"][r["predeclared_plan"]["runtime_native_cache"]]}`。只加载56个训练数组，逐数组联合哈希与冻结试验完全一致；原2799掩码、观测与SET电导也逐项一致。',
              f'计算用时 {r["elapsed_s"]:.3f} s，预算60 s。新ODE=0、新物理拟合=0、新谱积分=0；冻结旧产物未修改。']
    (HERE/'RESET_SHAPE_DIAGNOSTIC.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
