"""One bounded record10 soft-state budget; no kinetic fit, ODE, or domain scan."""
from pathlib import Path
import hashlib
import json
import math
import signal
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from vulcan2d.shared_spectral import SpectralShape


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started=time.monotonic()
    plan_path=HERE/'soft_state_budget.plan.json'
    plan=json.loads(plan_path.read_text())
    destination=HERE/'soft_state_budget.json'
    if destination.exists():
        raise FileExistsError('Preserve existing completed or failed result; no automatic rerun')
    report=dict(status='running',plan=plan,inverse_results=[],actual_inverse_problems=0,
                actual_new_ODE_calls=0,actual_new_physical_fits=0,actual_direct_spectrum_endpoints=0)
    old_failure=HERE/'pre_reference_path_fix/soft_state_budget.json'
    if old_failure.exists():
        old=json.loads(old_failure.read_text())
        assert old['actual_inverse_problems']==0 and old['actual_direct_spectrum_endpoints']==0
        report['prior_input_failure']=dict(source=str(old_failure.relative_to(ROOT)),sha256=sha(old_failure),
            error=old['error'],elapsed_s=old['elapsed_s_including_inputs'],actual_inverse_problems=0,
            interpretation='Malformed reference path only; original failed result preserved, no physical point consumed')
    def timeout(*_):
        raise TimeoutError('Declared60s total budget exceeded; preserve output, do not retry automatically')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,plan['compute_budget_seconds'])
    try:
        native=Path(plan['native_cache']);table=Path(plan['table_cache'])
        assert sha(native)==plan['expected_native_sha256']
        assert sha(table)==plan['expected_table_sha256']
        reference=ROOT/plan['actual_series_reference'];reference_json=reference.with_suffix('.json')
        reference_payload=json.loads(reference_json.read_text())
        ron=plan['Ron_ohm'];gc=plan['gamma_c_ev'];envelope=plan['material_table_fractional_envelope']
        assert reference_payload['plan']['Ron_ohm']==ron
        assert reference_payload['plan']['gamma_c_ev']==gc
        with np.load(table,allow_pickle=False) as z:
            lg=z['log_gamma'].copy();lv=z['log_voltage'].copy();li=z['log_current'].copy()
            metadata=json.loads(str(z['metadata'].item()))
        scale=math.log(metadata['shape']['ron_ohm']/ron)
        loaded=[];digest=hashlib.sha256();data={}
        with np.load(native,allow_pickle=False) as z:
            for phase in ('set','reset'):
                for suffix in ('voltage','current'):
                    key=f'1r_cycle_10_{phase}_{suffix}'
                    data[phase+'_'+suffix]=z[key].copy();loaded.append(key)
                    digest.update(key.encode());digest.update(data[phase+'_'+suffix].tobytes())
        sv=data['set_voltage'];si=data['set_current'];rv=data['reset_voltage'];ri=abs(data['reset_current'])
        ap=int(np.argmax(abs(sv))); sj=int(np.flatnonzero((np.arange(len(sv))>ap)&np.isclose(abs(sv),.05,rtol=0,atol=1e-8))[0])
        assert 0<abs(si[sj])<.995e-3
        gset=abs(float(si[sj]/sv[sj]));ap=int(np.argmax(abs(rv)))
        out=int(np.flatnonzero((np.arange(len(rv))<=ap)&np.isclose(abs(rv),plan['outward_witness_abs_voltage_v'],rtol=0,atol=3e-7))[0])
        targets=plan['return_anchor_abs_voltages_v']
        returns=[int(np.flatnonzero((np.arange(len(rv))>ap)&np.isclose(abs(rv),v,rtol=0,atol=3e-7))[0]) for v in targets]
        assert np.all(np.diff(returns)>0) and all(j>out for j in returns)
        assert all(ri[j]>=1e-12 for j in [out,*returns])
        assert abs(float(rv[out]))==abs(float(rv[returns[targets.index(.975)]]))
        with np.load(reference,allow_pickle=False) as z:
            series_u=z['refined_state_u'].copy();series_v=z['command_voltage_v'].copy()
            series_vh=z['refined_local_voltage_v'].copy();series_i=z['refined_current_a'].copy()
        series_gamma=gc/np.expm1(series_u)
        assert np.all(np.isfinite(series_gamma)) and np.all(series_gamma>0)
        series_j=int(np.argmax(series_gamma))

        def column(v):
            x=math.log(v)
            if not lv[0]<=x<=lv[-1]:raise ValueError('Voltage outside existing table support')
            j=int(np.clip(np.searchsorted(lv,x)-1,0,len(lv)-2))
            w=(x-lv[j])/(lv[j+1]-lv[j])
            c=(1-w)*li[1,:,j]+w*li[1,:,j+1]+scale
            assert np.all(np.diff(c)>0)
            return c

        def forward(v,gamma):
            x=math.log(gamma)
            if not lg[0]<=x<=lg[-1]:raise ValueError('Gamma outside existing table support')
            return math.exp(float(np.interp(x,lg,column(v))))

        def inverse(label,v,target,kind):
            report['actual_inverse_problems']+=1
            assert report['actual_inverse_problems']<=plan['maximum_inverse_problems']
            c=column(v)
            row=dict(label=label,abs_voltage_v=v,table_target_current_a=target,bound_kind=kind,
                     table_current_supported_a=[math.exp(c[0]),math.exp(c[-1])],
                     analytic_passive_current_ceiling_a=v/ron)
            if target<=0:
                row.update(status='zero_target_no_finite_outward_u_upper',Gamma_bound_ev=0.,u_bound=None)
            elif not c[0]<=math.log(target)<=c[-1]:
                row.update(status='target_outside_original_table_no_extrapolation',Gamma_bound_ev=None,u_bound=None,
                           exceeds_analytic_current_ceiling=target>v/ron)
            else:
                x=float(np.interp(math.log(target),c,lg));g=math.exp(x)
                row.update(status='inverted_within_original_table',Gamma_bound_ev=g,u_bound=math.log1p(gc/g),
                    log_Gamma_bound=x,table_log10_current_residual=math.log10(forward(v,g)/target),
                    Gamma_over_gamma_c=g/gc,Gamma_over_median_spectral_energy=g/metadata['shape']['median_energy_ev'],
                    Gamma_over_thermal_energy=g/(8.617333262145e-5*metadata['shape']['temperature_k']),
                    log10_Gamma_distance_to_lower_table_edge=(x-lg[0])/math.log(10.),
                    log10_Gamma_distance_to_upper_table_edge=(lg[-1]-x)/math.log(10.))
            report['inverse_results'].append(row)
            return row

        witness_v=abs(float(rv[out]));io=float(ri[out]);same_return=returns[targets.index(.975)]
        ref_current=forward(witness_v,float(max(series_gamma)))
        report.update(input_provenance=dict(loaded_native_array_keys=loaded,loaded_record10_arrays_sha256=digest.hexdigest(),
            SET_read=dict(native_index=sj,raw_voltage_v=float(sv[sj]),raw_current_a=float(si[sj]),G_SET_s=gset),
            outward=dict(native_index=out,raw_voltage_v=float(rv[out]),raw_current_a=float(data['reset_current'][out])),
            return_anchors=[dict(native_index=j,raw_voltage_v=float(rv[j]),raw_current_a=float(data['reset_current'][j])) for j in returns]),
            fixed_spectrum=dict(table_metadata=metadata,Ron_ohm=ron,gamma_c_ev=gc,
                                table_Gamma_domain_ev=[math.exp(lg[0]),math.exp(lg[-1])],
                                table_voltage_domain_v=[math.exp(lv[0]),math.exp(lv[-1])]),
            actual_series_reference=dict(source=str(reference.relative_to(ROOT)),reference_kind='Existing completed seriescycle4 SET linear-field probe, not a fit produced by this task',
                native_point_count=len(series_u),u_range=[float(min(series_u)),float(max(series_u))],
                Gamma_range_ev=[float(min(series_gamma)),float(max(series_gamma))],
                terminal_u=float(series_u[-1]),total_u_decrease=float(series_u[0]-series_u[-1]),
                maximum_Gamma_native_index=series_j,maximum_Gamma_command_v=float(series_v[series_j]),
                maximum_Gamma_local_v=float(series_vh[series_j]),maximum_Gamma_terminal_current_a=float(series_i[series_j]),
                same_bare_negative_voltage_v=-witness_v,
                bare_negative_soft_current_at_maximum_series_Gamma_a=ref_current,
                interpretation='State/amplitude scale comparison under the same law. Bare and series devices are not paired and are not assigned equal initial states. This is not parameter-transfer validation.'))
        scenarios=[]
        for delta in plan['relative_current_half_bands']:
            key='nominal' if delta==0 else '20percent'
            ilo=(1-delta)*io;gupper=(1+delta)*gset
            lower=max(0.,ilo-witness_v*gupper)
            invout=inverse(key+'_outward',witness_v,lower/(1+envelope),'Gamma_lower_and_u_upper')
            rows=[]
            for target,j in zip(targets,returns):
                upper=(1+delta)*float(ri[j])
                invret=inverse(key+f'_return_{target:g}V',abs(float(rv[j])),upper/(1-envelope),'Gamma_upper_and_u_lower')
                valid=invout['u_bound'] is not None and invret['u_bound'] is not None
                change=invret['u_bound']-invout['u_bound'] if valid else None
                rows.append(dict(target_abs_voltage_v=target,native_index=j,current_upper_a=upper,
                    inverse=invret,necessary_signed_delta_u_lower=change,
                    positive_u_increase_required=bool(change is not None and change>0),
                    minimum_Gamma_out_over_return=invout['Gamma_bound_ev']/invret['Gamma_bound_ev'] if valid else None,
                    command_path_voltage_between_nodes_v=2*abs(float(rv[ap]))-witness_v-abs(float(rv[j])),
                    clock_scope='Known voltage-path separation only, not elapsed seconds or a physical rate',
                    initial_state_scope='Return bound is conditional from observation, never used as a predictive initialization'))
            same=rows[targets.index(.975)]
            scenarios.append(dict(relative_current_half_band=delta,outward_soft_current_lower_a=lower,
                outward_soft_fraction_lower=lower/ilo,
                same_voltage_required_background_difference_lower_a=max(0.,ilo-(1+delta)*float(ri[same_return])-witness_v*gupper),
                outward_inverse=invout,return_bounds=rows,
                required_bare_outward_Gamma_lower_over_actual_series_max=invout['Gamma_bound_ev']/max(series_gamma) if invout['Gamma_bound_ev'] else None,
                minimum_bare_soft_amplitude_over_max_series_state_at_same_bare_voltage=lower/ref_current,
                same_voltage_necessary_delta_u_lower=same['necessary_signed_delta_u_lower'],
                tail0p2_necessary_delta_u_lower=rows[-1]['necessary_signed_delta_u_lower']))
        report['scenarios']=scenarios
        broad=any(s['outward_inverse']['Gamma_bound_ev'] is not None and
                  (s['outward_inverse']['Gamma_bound_ev']>gc or s['outward_inverse']['Gamma_bound_ev']>10*max(series_gamma)) for s in scenarios)
        report['broad_Gamma_triggered']=broad
        checks=[]
        if broad:
            selected=[scenarios[0]['outward_inverse'],scenarios[1]['outward_inverse'],
                      scenarios[0]['return_bounds'][-1]['inverse'],scenarios[1]['return_bounds'][-1]['inverse']]
            selected=[x for x in selected if x['status']=='inverted_within_original_table']
            assert len(selected)<=4
            shape=SpectralShape(**{**metadata['shape'],'ron_ohm':ron})
            vv=-np.array([x['abs_voltage_v'] for x in selected]);gg=np.array([x['log_Gamma_bound'] for x in selected])
            d32=abs(shape.current_at_log_gamma(vv,gg,order=32));d64=abs(shape.current_at_log_gamma(vv,gg,order=64))
            for row,c32,c64 in zip(selected,d32,d64):
                ratio=float(c64/row['table_target_current_a'])
                checks.append(dict(label=row['label'],Gamma_ev=row['Gamma_bound_ev'],abs_voltage_v=row['abs_voltage_v'],
                    table_target_current_a=row['table_target_current_a'],direct_order32_current_a=float(c32),direct_order64_current_a=float(c64),
                    order32_vs64_log10_difference=float(math.log10(c32/c64)),
                    direct64_vs_table_log10_error=float(math.log10(ratio)),direct64_over_table=ratio,
                    within_declared_table_fractional_envelope=bool(1-envelope<=ratio<=1+envelope),
                    conservative_bound_direction_at_direct64=bool(ratio<=1+envelope) if row['bound_kind']=='Gamma_lower_and_u_upper' else bool(ratio>=1-envelope)))
            report['actual_direct_spectrum_endpoints']=len(selected)
            report['actual_direct_spectrum_forward_evaluations']=2*len(selected)
            report['direct_quadrature_metadata']={str(order):shape.quadrature_metadata(order) for order in (32,64)}
        report['direct_spectrum_checks']=checks
        report['source_sha256']={str(p):sha(p) for p in (native,table,reference,reference_json,plan_path,Path(__file__),ROOT/'vulcan2d/shared_spectral.py')}
        report['all_inverse_targets_inside_original_table']=all(r['status']=='inverted_within_original_table' for r in report['inverse_results'])
        report['status']='completed_conditional_soft_state_budget' if report['all_inverse_targets_inside_original_table'] else 'completed_with_unresolved_domain_bounds'
        report['scope']=[
            'One record10 bounded necessary-budget diagnostic. New physical fits0, ODE0, holdout0; original scoring masks and gates unchanged.',
            'Assumes passive Ohmic a cannot grow after prior SETread. This is a model hypothesis, not an observed fact. Dynamic soft background is permitted.',
            'Gamma inferred from total return current is an upper bound only, never an observed soft state. No assumed50% or90% background fraction.',
            'Delta20% raw-current bands and0.3% table envelope are sensitivity assumptions, not measured noise or a new global numerical certificate.',
            'Positive delta_u lower bounds require sufficient soft weakening only under this model decomposition and fixed spectrum. They do not identify species or prove this decomposition.',
            'Order32/64 checks validate selected numerical amplitudes, not broad-state physical adequacy or full-history prediction.',
            'State change per elapsed time is unknown. Large amplitude/state excursion does not by itself measure a fast or slow physical time constant.']
    except Exception as exc:
        report.update(status='partial_or_failed_budget_preserved',error_type=type(exc).__name__,error=str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        report['elapsed_s_including_inputs']=time.monotonic()-started
        report['cumulative_elapsed_s_including_preserved_failure']=report['elapsed_s_including_inputs']+report.get('prior_input_failure',{}).get('elapsed_s',0.)
    destination.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    write_report(report)
    print(json.dumps(dict(status=report['status'],elapsed_s=report['elapsed_s_including_inputs'],
        actual_inverse_problems=report['actual_inverse_problems'],actual_direct_spectrum_endpoints=report['actual_direct_spectrum_endpoints'],
        series=report.get('actual_series_reference'),scenarios=report.get('scenarios'),direct=report.get('direct_spectrum_checks'),
        error=report.get('error')),indent=2))


def write_report(r):
    lines=['# 圈10：纯损失假设下的软状态幅度预算','',
        '这是已执行的必要条件诊断，不是完整统一模型或新拟合。只使用圈10的前SET50mV读点、已固定0.975V外扫见证、同电压返扫及已有六个返扫锚；没有新增圈或保留集。',
        '',f'运行状态：`{r["status"]}`。实际反演{r["actual_inverse_problems"]}/16，直接谱检查{r["actual_direct_spectrum_endpoints"]}/4端点（各32/64阶），新ODE=0、拟合=0。']
    if r['status']=='partial_or_failed_budget_preserved':
        lines += ['',f'失败已保留：{r.get("error_type")}: {r.get("error")}。不自动重试或覆盖。']
    elif 'scenarios' in r:
        lines += ['', '## 假设、反演和状态含义','',
            '固定共同Ron=58.46303335Ω、Γc及既有谱。假设a≥0为欧姆通道，且从前SET读点起不能增长，故a≤G_SET；不假定软背景在两支相等。外扫必须有B_out≥max(0,Iout−vG_SET)，返扫只能推出B_ret≤Iret。',
            '原读数±δ带下，外扫下界为max[0,(1−δ)Iout−v(1+δ)G_SET]；返扫上界为(1+δ)Iret。沿用0.3%表包络，分别反演下界/(1+0.003)和上界/(1−0.003)。这些带是敏感性假设，不是仪器噪声或新验收门槛。',
            '由于u=log(1+Γc/Γ)随Γ递减，Γout下界给uout上界，Γret上界给uret下界；于是Δu≥u(Γret,U)−u(Γout,L)。总返扫电流反演给出的只是Γ上界，不能称已测软状态。',
            '', '## 外扫必须提供多大的软电流','',
            '| 原电流半带 | B_out下界/mA | 软份额必要下界 | Γout下界/eV | uout上界 | Γ下界/实际series最大Γ |',
            '|---:|---:|---:|---:|---:|---:|']
        for s in r['scenarios']:
            x=s['outward_inverse']
            lines.append(f'| {s["relative_current_half_band"]:.0%} | {s["outward_soft_current_lower_a"]*1e3:.9g} | {s["outward_soft_fraction_lower"]:.5%} | {x.get("Gamma_bound_ev")} | {x.get("u_bound")} | {s["required_bare_outward_Gamma_lower_over_actual_series_max"]} |')
        ref=r['actual_series_reference']
        lines += ['',f'实际对照是已完成的1T1R第4圈SET线性局部场探针：u范围{ref["u_range"]}，Γ范围{ref["Gamma_range_ev"]} eV。把其中最大Γ仅代入同一裸器件−0.975V输运律，软电流为{ref["bare_negative_soft_current_at_maximum_series_Gamma_a"]:.9g} A。此项只比较状态/幅度尺度；不是要求两块器件同初态，也不是跨器件预测验证。',
            '', '## 到后续返扫锚的必要变化','',
            '| 原电流半带 | 返扫电压幅值/V | Γret上界/eV | uret下界 | Δu必要下界 | Γout下界/Γret上界 |',
            '|---:|---:|---:|---:|---:|---:|']
        for s in r['scenarios']:
            for x in s['return_bounds']:
                inv=x['inverse']
                lines.append(f'| {s["relative_current_half_band"]:.0%} | {x["target_abs_voltage_v"]:.3f} | {inv.get("Gamma_bound_ev")} | {inv.get("u_bound")} | {x["necessary_signed_delta_u_lower"]} | {x["minimum_Gamma_out_over_return"]} |')
        lines += ['', '全部返扫节点在外扫见证之后；路径电压距离仅记作已知采集顺序，不换算秒数。正Δu下界意味着在这些假设下必须充分削弱软输运。它不要求a增长，也不能证明软支路实际承担了该电流。',
            '', '## 宽Γ域与数值复核','',
            f'原Γ表域：{r["fixed_spectrum"]["table_Gamma_domain_ev"]} eV；所有反演是否在原域内：{r["all_inverse_targets_inside_original_table"]}。宽Γ触发条件是否满足：{r["broad_Gamma_triggered"]}。未扩大表域、未外推、未在直接谱上重新求根。',
            '', '| 预定端点 | 64阶/表电流 | 32→64阶log差 | 64阶与表log差 | 在0.3%包络内 |',
            '|---|---:|---:|---:|---|']
        for x in r['direct_spectrum_checks']:
            lines.append(f'| {x["label"]} | {x["direct64_over_table"]:.9g} | {x["order32_vs64_log10_difference"]:.3g} | {x["direct64_vs_table_log10_error"]:.3g} | {x["within_declared_table_fractional_envelope"]} |')
        lines += ['', 'Γ的宽度相对Γc、谱能量、热能及到原表边界的距离都保存在JSON。直接积分一致性只说明这些端点的数值读出可信，不证明固定谱可以物理地跨越全部状态。',
            '', '## 对“慢调制器”的含义','',
            '在a纯损失这一明确假设下，u如果仅缓慢改变断裂率，而软通道始终保持原series探针中的微小幅度，就必须面对上述外扫软电流缺口。候选需实际跨越相应状态和输运幅度，且随后满足返扫上界；不能只修改λ(u)却让Fsoft一直保持很小。',
            '这里没有真实扫描时钟，因此不能由状态变化大就断言物理时间尺度一定快，更不能识别物种、局部温度或细丝。若允许a增长、非欧姆硬路径或SET读态上界失效，本预算前提改变，不能据此排除那些模型。',
            '', '## 来源及停机状态','',
            'JSON记录四个圈10原始数组名与联合哈希、已验证缓存哈希、谱实现、冻结series参考及预声明文件哈希。所有结果是同训练资料下的条件预算，不是保留集验证。']
    lines += ['',f'总用时（含输入及核验）{r["elapsed_s_including_inputs"]:.6f} s，声明预算60 s。']
    if 'prior_input_failure' in r:
        lines += ['',f'首次输入路径字段混入说明文字，反演0次即停止；原脚本、计划及失败JSON/MD保存在pre_reference_path_fix/。只修正路径编码后执行同一16问题，累计用时{r["cumulative_elapsed_s_including_preserved_failure"]:.6f} s，没有增加物理点。']
    (HERE/'soft_state_budget.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
