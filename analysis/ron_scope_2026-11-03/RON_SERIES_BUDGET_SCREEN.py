"""Closed 19 original physical gamma_c x3 predeclared Ron cached screen.

No new spectrum, circuit solve, ODE, fit, or parameter point. Reuse the
re-inverted anchors/KCL arrays and complete continuous-JS extremum enumerator.
"""
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
PREVIOUS=ROOT/'analysis/parameter_transfer_2026-11-03'
sys.path.insert(0,str(PREVIOUS))
from series_uncapped_threshold_scope import scan_threshold_interval


def main():
    started=time.monotonic()
    scaling_path=HERE/'RON_SCALING_REVIEW.json'
    arrays_path=HERE/'RON_SCALING_REVIEW.npz'
    calibration_path=HERE/'RON_CALIBRATION_SCOPE.json'
    old_source_path=PREVIOUS/'SERIES_ENTRY_REVIEW.json'
    old_uncapped_path=PREVIOUS/'SERIES_UNCAPPED_THRESHOLD_SCOPE.json'
    algorithm_path=PREVIOUS/'series_uncapped_threshold_scope.py'
    scaling=json.loads(scaling_path.read_text())
    calibration=json.loads(calibration_path.read_text())
    original=json.loads(old_source_path.read_text())
    old_uncapped=json.loads(old_uncapped_path.read_text())
    with np.load(arrays_path) as z:
        commands=z['soft_set_command_left_v'].copy()
        currents=[z[f'ron_{i}_soft_set_current_lower_a'].copy() for i in range(3)]
        ron_array=[float(z[f'ron_{i}_Ron_ohm']) for i in range(3)]
    gcs=np.array([r['gamma_c_ev'] for r in original['rows']])
    expected=np.unique(np.r_[np.logspace(-4,0,17),.03,.05])
    assert np.array_equal(gcs,expected) and len(gcs)==19
    assert np.allclose(commands,np.linspace(2,5,61)[:-1],rtol=0,atol=1e-14)
    assert all(x.shape==(60,) and np.all(x>0) for x in currents)
    rons=[r['Ron_ohm'] for r in calibration['predeclared_Ron_sensitivity_points']]
    assert rons==ron_array==[r['Ron_ohm'] for r in scaling['rows']]
    cap=original['transistor_current_upper_at_command2V_a']
    source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (scaling_path,arrays_path,calibration_path,old_source_path,old_uncapped_path,algorithm_path,Path(__file__))}
    report=dict(status='running_closed_57_case_cached_budget_screen',original_physical_gamma_c_ev=gcs.tolist(),
        predeclared_Ron_ohm=rons,JS_lower_a=1e-7,beta=1.,A_RESET=0.,cases=[],
        source_sha256=source_hashes,
        method='series_uncapped_threshold_scope.scan_threshold_interval: all positive-part breakpoints and stationary roots after linear-curvature splits; independent bounded cross-check; k-box constraints intersected before separate extremum enumeration.',
        scope=[
            'Exactly the original19 physical gamma_c values x3 predeclared Ron values. No mapped19, added gc/Ron point, or expansion after outcomes.',
            'Anchors and60-node series current bounds were separately re-inverted/re-solved for each Ron in RON_SCALING_REVIEW; they are reused as frozen inputs here.',
            'JS >=1e-7 A; no bare1mA ceiling in independent-series audit. Upper limit is min(initial anchored2V activation, cap-based target2V arrival). Equality is closure only, not an activating seed.',
            'Positive finite k without an algorithm box and original [1e-4,10^1.5] per-command-V k box are checked separately.',
            'Any overlap is only a necessary endpoint-budget compatibility result, never a selected seed, verified trajectory, microscopic mechanism, or joint 1R/1T1R solution.',
            'No continuous-gc global exclusion. Omitted JS<1e-7 and other Ron values remain outside this closed domain.',
            'Same generous0.2V observation anchor,2V outward path allowance,5V Q10/1.05 and2V Q90 return bounds, constant command-path-rate interpretation and fixed numerical spectrum/T mapping assumptions.',
            'No new spectral quadrature, KCL solve, ODE, fit, holdout read, or change to old outputs. Old exact-cache versus table-recomputed baseline offsets are separated.',
            'Full57-case domain is closed after this one audit; numerical failures are retained and not converted into physical exclusions.'])
    def timeout(*_):
        raise TimeoutError('60-second cached-computation budget exhausted')
    compute_started=time.monotonic()
    signal.signal(signal.SIGALRM,timeout)
    signal.setitimer(signal.ITIMER_REAL,60.)
    stopped=False
    try:
        for ri,ron in enumerate(rons):
            source=scaling['rows'][ri]
            anchors=source['anchor_reinversions']
            g0=anchors['initial_0p2']['rederived_gamma_ev']
            target=anchors['outward_2V_lower']['rederived_gamma_ev']
            g5=anchors['return_5V_lower']['rederived_gamma_ev']
            g2=anchors['return_2V_upper']['rederived_gamma_ev']
            initial_i2=source['initial_terminal_at_2V_a']
            for gc in gcs:
                row=dict(Ron_index=ri,Ron_ohm=ron,gamma_c_ev=float(gc),status='running')
                report['cases'].append(row)
                try:
                    B=math.log1p(gc/g0);A=math.log1p(gc/target)
                    u2=math.log1p(gc/g2);du=math.log1p(gc/g5)-u2
                    activation=initial_i2*math.exp(B);arrival=cap*math.exp(A)
                    upper=min(activation,arrival)
                    d=u2+np.log(currents[ri]/cap)
                    result=scan_threshold_interval(B,A,du,d,cap,1e-7,upper)
                    row.update(status='completed_closed_interval_extremum_enumeration',
                        anchored_u=B,target_u=A,return_allowed_du=du,
                        initial_activation_JS_upper_a=activation,target_reach_JS_upper_a=arrival,
                        physical_JS_upper_closure_a=upper,limiting_upper='initial_activation' if activation<=arrival else 'target_reach',
                        scan=result)
                except Exception as exc:
                    row.update(status='numerical_failure_retained',error_type=type(exc).__name__,error=str(exc))
                    if isinstance(exc,TimeoutError):
                        stopped=True
                        raise
    except TimeoutError:
        stopped=True
    finally:
        signal.setitimer(signal.ITIMER_REAL,0.)
        report['compute_elapsed_s']=time.monotonic()-compute_started
    successful=[r for r in report['cases'] if r['status']=='completed_closed_interval_extremum_enumeration']
    report['n_attempted']=len(report['cases']);report['n_completed']=len(successful)
    report['n_numerical_failures']=len(report['cases'])-len(successful)
    report['n_overlap_without_k_box']=sum(r['scan']['any_budget_overlap_without_k_box'] for r in successful)
    report['n_overlap_with_original_k_box']=sum(r['scan']['any_budget_overlap_with_original_k_box'] for r in successful)
    report['selected_seeds']=[]
    report['Ron_summaries']=[]
    for ri,ron in enumerate(rons):
        group=[r for r in successful if r['Ron_index']==ri]
        if not group:
            report['Ron_summaries'].append(dict(Ron_ohm=ron,status='no_successful_cases'))
            continue
        best=min(group,key=lambda r:r['scan']['without_algorithm_k_box']['ratio_infimum_over_interval_closure'])
        opt=best['scan']['without_algorithm_k_box']
        boxed=[r for r in group if r['scan']['with_original_algorithm_k_box'] is not None]
        boxed_best=min(boxed,key=lambda r:r['scan']['with_original_algorithm_k_box']['ratio_infimum_over_interval_closure']) if boxed else None
        report['Ron_summaries'].append(dict(Ron_ohm=ron,n_completed=len(group),
            n_overlap_without_k_box=sum(r['scan']['any_budget_overlap_without_k_box'] for r in group),
            n_overlap_with_original_k_box=sum(r['scan']['any_budget_overlap_with_original_k_box'] for r in group),
            best_gamma_c_ev=best['gamma_c_ev'],minimum_ratio_without_k_box=opt['ratio_infimum_over_interval_closure'],
            best_JS_closure_a=opt['best_JS_a'],best_k_lower_per_v=opt['best_k_lower_per_v'],best_k_upper_per_v=opt['best_k_upper_per_v'],
            best_gamma_c_with_k_box_ev=boxed_best['gamma_c_ev'] if boxed_best else None,
            minimum_ratio_with_original_k_box=boxed_best['scan']['with_original_algorithm_k_box']['ratio_infimum_over_interval_closure'] if boxed_best else None))
    if len(successful)==57:
        report['status']=('closed_57_case_domain_no_necessary_overlap_no_expansion'
                          if report['n_overlap_without_k_box']==0 else 'closed_57_case_domain_contains_necessary_overlap_only')
    else:
        report['status']='closed_bounded_attempt_incomplete_no_domain_expansion'
    report['timed_out']=stopped
    control=next((r for r in report['Ron_summaries'] if r['Ron_ohm']==rons[0]),None)
    report['old_Ron_control_comparison']=dict(old_exact_cache_minimum_ratio=old_uncapped['minimum_ratio'],
        old_exact_cache_best_gamma_c_ev=old_uncapped['best_gamma_c_ev'],
        table_recomputed_minimum_ratio=control.get('minimum_ratio_without_k_box') if control else None,
        table_minus_old_exact_cache=(control['minimum_ratio_without_k_box']-old_uncapped['minimum_ratio']) if control and 'minimum_ratio_without_k_box' in control else None,
        interpretation='Same Ron, different cached numerical baseline: this offset is not a Ron effect or a physical/model improvement.')
    report['total_main_elapsed_s']=time.monotonic()-started
    report['timer_scope']='60-second computation alarm starts after all input file loads/hashes; excludes process/module startup and input-file waiting.'
    out=HERE/'RON_SERIES_BUDGET_SCREEN.json'
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# 封闭19gc×3Ron的series必要预算筛查','',
           '固定原19个物理gc与已声明的3个Ron；复用各Ron重反演的状态锚和重新解KCL得到的60节点电流数组。没有增加映射gc点、重解电路、ODE或拟合。',
           '', 'JS下界固定1e-7 A，去掉独立series不适用的裸1 mA上界，只保留初锚激活及晶体管电流上界给出的目标到达限制；上端等号只作最有利闭包，不是可激活seed。',
           '', '| Ron/Ω | 完成gc数 | 不限k最小比值 | 最佳原gc/eV | JS闭包上界/mA | 不限k相容数 | 原k盒相容数 |',
           '|---:|---:|---:|---:|---:|---:|---:|']
    for s in report['Ron_summaries']:
        if 'minimum_ratio_without_k_box' in s:
            lines.append(f'| {s["Ron_ohm"]:.9g} | {s["n_completed"]} | {s["minimum_ratio_without_k_box"]:.10g} | {s["best_gamma_c_ev"]:.9g} | {s["best_JS_closure_a"]*1e3:.9g} | {s["n_overlap_without_k_box"]} | {s["n_overlap_with_original_k_box"]} |')
    if len(successful)==57 and report['n_overlap_without_k_box']==0:
        lines+=['','57项均无SET必要速率交集；不限k的算法范围仍然冲突。此次声明域到此封闭，不扩展参数域、不挑新点、不进入ODE。']
    else:
        lines+=['',f'完成{len(successful)}/57项，数值失败{report["n_numerical_failures"]}项。必要预算交集不代表完整轨迹相容，失败/未知不能当作物理排除。']
    comparison=report['old_Ron_control_comparison']
    lines+=['',f'旧Ron的原精确缓存对照为{comparison["old_exact_cache_minimum_ratio"]:.12g}；本次重反演/表缓存对照为{comparison["table_recomputed_minimum_ratio"]}，差值{comparison["table_minus_old_exact_cache"]}。该差值为数值基线偏移，不能记作降低Ron的收益。',
            '', '范围仍是离散19gc×3Ron、JS≥1e-7和既有条件端点预算；没有证明连续gc/Ron域无解，也不是完整联合模型结论。算法逐段枚举正部折点与全部驻点，并独立处理原k盒交集。',
            f'计算阶段{report["compute_elapsed_s"]:.3f}s（限时60s，不含启动/输入文件等待）；完整闭包极值、有限/无限物理速率上界、数值失败和来源hash见JSON。']
    out.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:report[k] for k in ('status','compute_elapsed_s','n_completed','n_numerical_failures','n_overlap_without_k_box','n_overlap_with_original_k_box','Ron_summaries','old_Ron_control_comparison')},indent=2))


if __name__=='__main__':main()
