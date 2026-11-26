"""Two preselected single-cycle SET forwards, including reverse initialization.

No optimization, additional seeds, holdouts, RESET or continuous history.
"""
from pathlib import Path
import hashlib
import json
import math
import signal
import sys
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OLD=ROOT/'analysis/mechanism_selection_2026-11-03'
CACHE=Path('/private/tmp/vulcan-runtime-cache')
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(OLD))
from SOFT_FEEDBACK_LOW_BIAS_BUDGET import cached_transport


class DomainError(ValueError):
    pass


class NumericalResolutionError(ValueError):
    pass


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    budget_path=HERE/'soft_beta_relaxed_budget.json'
    budget=json.loads(budget_path.read_text())
    assert budget['status']=='completed_20_fixed_conditional_budgets_no_ODE_no_fit'
    ron=budget['plan']['Ron_ohm'];gc=budget['plan']['gamma_c_ev']
    anchor=budget['anchors_u']['initial_0p2']
    prior_path=HERE/'pre_circuit_domain_fix/soft_beta_conditional_forward.json'
    prior=json.loads(prior_path.read_text())
    assert len(prior['runs'])==0 and prior['compute_elapsed_s']<1.
    spent=float(prior['compute_elapsed_s'])
    seeds=[]
    for beta,fraction in [(4.,.95),(6.,.8)]:
        original=next(r for r in budget['rows'] if r['beta']==beta and r['ceiling_fraction']==fraction)
        case=original['cases'][0]
        rate=math.sqrt(case['k_lower_per_V']*case['k_upper_per_V']) if beta==4 else 2*case['k_lower_per_V']
        seeds.append(dict(name=f'beta{int(beta)}',beta=beta,ceiling_fraction=fraction,JS_A=original['JS_A'],
                          k_per_v=rate,selection='sqrt(nominal_k_lower*nominal_k_upper)' if beta==4 else '2*nominal_k_lower',
                          original_nominal_k_lower=case['k_lower_per_V'],original_nominal_k_upper=case['k_upper_per_V']))
    plan=dict(scope='Two frozen single-cycle1T1R SET conditional checks; no fit, holdout, RESET,32-cycle history or completed1R/1T1R model.',
              seeds=seeds,Ron_ohm=ron,gamma_c_ev=gc,gate_voltage_v=1.1,parallel_path_a=0.,
              cycle=4,conditioned_anchor_command_v=.2,conditioned_anchor_gamma_ev=budget['anchors_gamma_ev']['initial_0p2'],conditioned_anchor_u=anchor,
              state_law='du/ds=-k*max(log(I_T(A,u)/JS)+beta*u,0); Gamma=gc/expm1(u); I_T from series KCL',
              backward_initialization='Integrate the same signed du/dA from A=.2 down to0. At exactly zero command,I=0 and drive=0. No freezing0->.2.',
              forward='Outward path s:0->5,A=s; return s:5->10,A=10-s. Continue state at apex; duplicated native endpoints do not add dwell.',
              state_domain=[0.,80.],domain_policy='No state projection. Gamma and every positive material voltage must remain inside existing table support; zero voltage is the exact passive endpoint. Stop and preserve any domain failure.',
              precision_levels=[dict(name='nominal',rtol=1e-7,atol=1e-8,max_step=.02),dict(name='refined',rtol=1e-8,atol=1e-9,max_step=.01)],
              method='DOP853',max_runs=4,max_ODE_segments=12,
              rate_selection='Exactly the two specified rates, no adaptive seeds,rate scans,initial-state changes or optimization.',
              KCL_method='Scalar brentq in transistor-drop voltage; compare to original42-step material-voltage bisection before any ODE. On comparison failure use original42-step method.',
              KCL_brent_xtol_v=1e-15,KCL_brent_rtol=1e-13,KCL_brent_maxiter=100,
              KCL_precheck_commands_v=[.02,.1,.2,1.,2.,5.],
              KCL_precheck_states=[anchor,budget['anchors_u']['outward_2V_lower'],budget['anchors_u']['return_5V_lower'],budget['anchors_u']['return_2V_upper'],15.,25.],
              KCL_precheck_acceptance=dict(max_abs_log10_current_difference=5e-7,max_local_voltage_difference_v=1e-9,max_abs_KCL_residual_a=1e-13),
              scoring='Every original cycle4 SET point with V!=0,finite absI>=1e-11; raw log10 residuals, no floor replacement or low-bias exclusion.',
              reported_groups=['all500','outward','return','both_branches_abs_command<=.2','outward_abs_command<=.2','all_remaining_abs_command>.2'],
              feature_scope='Single-cycle predictions beside original32-record aggregate bands for context only; no aggregate gate-passing claim.',
              compute_budget_s=120,budget_start='After all input loads and hashes, but before KCL comparison and every ODE. Timeout stops all remaining work and retains completed segments/runs.',
              numerical_refinement='Both prescribed precisions start independently from the same .2V anchor through backward solve; no extra repeat beyond these four runs.',
              stop_policy='All failures retained; timeout forbids restart or extensions.',
              numerical_interface_correction=dict(previous_attempt=str(prior_path),previous_sha256=sha(prior_path),
                  previous_compute_s=spent,previous_ODE_runs=0,
                  correction='Use original log-voltage bounds so log(5) is accepted at the table endpoint despite exp(log(5)) rounding below5. Bound transistor drop independently and require a bracketed KCL root; no extrapolation.',
                  low_voltage_resolution_policy='A positive local voltage below table minimum is numerical_unknown, not evidence of physical candidate failure.'),
              remaining_compute_budget_s=120.-spent)
    plan_path=HERE/'soft_beta_conditional_forward.plan.json'
    plan_path.write_text(json.dumps(plan,indent=2,allow_nan=False)+'\n')
    t,lookup_paths=cached_transport(CACHE)
    for p in (t.table_path,*lookup_paths,CACHE/'RON_SCALING_REVIEW.json',CACHE/'RON_SCALING_REVIEW.npz'):
        assert sha(p)==budget['source_sha256'][str(p)]
    source_paths=[budget_path,HERE/'soft_beta_relaxed_budget.plan.json',HERE/'soft_beta_relaxed_budget.npz',
                  t.table_path,*lookup_paths,CACHE/'RON_SCALING_REVIEW.json',CACHE/'RON_SCALING_REVIEW.npz',
                  OLD/'SOFT_FEEDBACK_LOW_BIAS_BUDGET.py',ROOT/'analysis/ron_scope_2026-11-03/ron_scaling_review.py',
                  ROOT/'vulcan2d/transistor.py',plan_path,Path(__file__),prior_path]
    sources={str(p):sha(p) for p in source_paths}
    keys=['1t1r_cycle_4_set_voltage','1t1r_cycle_4_set_current'];digest=hashlib.sha256()
    with np.load(CACHE/'native_observations.npz') as z:
        raw=[]
        for key in keys:
            value=z[key].copy();raw.append(value);digest.update(key.encode());digest.update(value.tobytes())
    voltage,signed_observed=raw;observed=abs(signed_observed);apex=int(np.argmax(voltage))
    assert len(voltage)==503 and voltage[0]==voltage[-1]==0 and max(voltage)==5.
    outward=np.arange(len(voltage))<=apex
    path=np.where(outward,voltage,10.-voltage)
    assert np.all(np.diff(path)>=0)
    mask=np.isfinite(observed)&(observed>=1e-11)&(abs(voltage)>1e-12)
    assert mask.sum()==500
    constraints_path=ROOT/'analysis/constrained_shared_2026-11-03/feature_constraints.json'
    constraints=json.loads(constraints_path.read_text());sources[str(constraints_path)]=sha(constraints_path)
    original_features={key:constraints['features'][key] for key in constraints['proposed_success_criteria_before_next_fit']['primary_feature_ids']
                       if constraints['features'][key]['topology']=='1t1r' and constraints['features'][key]['phase']=='set'}
    report=dict(status='running_predeclared_two_seed_forward',plan=plan,source_sha256=sources,
                loaded_observation_keys=keys,loaded_cycle4_arrays_sha256=digest.hexdigest(),runs=[],
                raw_counts=dict(native_points=len(voltage),eligible_points=int(mask.sum()),zero_voltage_points=int((voltage==0).sum())))
    destination=HERE/'soft_beta_conditional_forward.json'
    destination.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    arrays=dict(command_voltage_v=voltage,path_voltage_s=path,observed_current_a=signed_observed,original_scoring_mask=mask)
    lg,lv,li=t.lg,t.lv,t.li[0];log_factor=math.log(t.r0/ron);vmin=math.exp(lv[0]);vmax=math.exp(lv[-1])
    tx,ti=t.knots['set'];gamma_min=math.exp(lg[0]);gamma_max=math.exp(lg[-1])
    assert np.all(np.diff(tx)>0) and tx[0]==0 and ti[0]==0
    def validate_state(u):
        if not math.isfinite(u) or not 0<=u<=80:
            raise DomainError(f'State outside[0,80]:{u!r}')
        if u==0:
            raise DomainError('Gamma=+inf at u=0 lies outside existing finite table; no projection')
        gamma=gc/math.expm1(u)
        if not gamma_min<=gamma<=gamma_max:
            raise DomainError(f'Gamma outside table:{gamma!r};u={u!r}')
        return gamma
    def material_scalar(v,logg):
        if v==0:return 0.
        if v<0:raise DomainError(f'Negative material voltage:{v!r}')
        logv=math.log(v)
        if logv<lv[0]:
            raise NumericalResolutionError(f'Positive material voltage below finite table resolution:{v!r}')
        if logv>lv[-1]:
            raise DomainError(f'Positive material voltage outside table:{v!r}')
        ix=int(np.searchsorted(lg,logg)-1);iy=int(np.searchsorted(lv,logv)-1)
        ix=min(max(ix,0),len(lg)-2);iy=min(max(iy,0),len(lv)-2)
        fx=(logg-lg[ix])/(lg[ix+1]-lg[ix]);fy=(logv-lv[iy])/(lv[iy+1]-lv[iy])
        value=(1-fx)*((1-fy)*li[ix,iy]+fy*li[ix,iy+1])+fx*((1-fy)*li[ix+1,iy]+fy*li[ix+1,iy+1])
        return math.exp(value+log_factor)
    def fast_series(command,u):
        gamma=validate_state(float(u));A=float(command)
        if A<0 or (A>0 and math.log(A)>lv[-1]):raise DomainError(f'Command outside material domain:{A!r}')
        if A==0:return 0.,0.,0.
        logg=math.log(gamma)
        def residual(vt):
            return float(np.interp(vt,tx,ti))-material_scalar(A-vt,logg)
        vtmax=min(A,float(tx[-1]))
        if residual(0.)>0 or residual(vtmax)<0:raise DomainError('KCL root unavailable inside transistor/material domains')
        vt=brentq(residual,0.,vtmax,xtol=plan['KCL_brent_xtol_v'],rtol=plan['KCL_brent_rtol'],maxiter=plan['KCL_brent_maxiter'])
        vh=A-vt;current=material_scalar(vh,logg)
        return current,vh,current-float(np.interp(vt,tx,ti))
    def old_series(command,u):
        gamma=validate_state(float(u));A=float(command)
        if A==0:return 0.,0.,0.
        if A<0 or math.log(A)>lv[-1]:raise DomainError('Command outside finite table domain')
        if math.log(A)<lv[0]:raise NumericalResolutionError('Command below finite material table resolution')
        current,vh=t.series(A,gamma,ron)
        current,vh=float(current),float(vh)
        if math.log(vh)<lv[0]:raise NumericalResolutionError('Original KCL solver fell below material table resolution')
        if math.log(vh)>lv[-1] or not 0<=A-vh<=tx[-1]:raise DomainError('Original KCL solver left material/transistor table domain')
        return current,vh,current-float(np.interp(A-vh,tx,ti))
    def timeout(*_):raise TimeoutError('120-second total computation budget exhausted; no restart')
    started=time.monotonic();signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,120.-spent)
    try:
        comparison=[]
        for A in plan['KCL_precheck_commands_v']:
            for u in plan['KCL_precheck_states']:
                fast=fast_series(A,u);old=old_series(A,u)
                comparison.append(dict(command_v=A,u=u,fast_current_a=fast[0],old_current_a=old[0],
                    abs_log_current_difference=abs(math.log10(fast[0]/old[0])),local_voltage_difference_v=abs(fast[1]-old[1]),abs_KCL_residual_a=abs(fast[2])))
        limits=plan['KCL_precheck_acceptance']
        check=dict(n_points=len(comparison),max_abs_log10_current_difference=max(x['abs_log_current_difference'] for x in comparison),
                   max_local_voltage_difference_v=max(x['local_voltage_difference_v'] for x in comparison),
                   max_abs_KCL_residual_a=max(x['abs_KCL_residual_a'] for x in comparison),points=comparison)
        passed=all(check[key]<=limit for key,limit in limits.items())
        check['passed']=passed;check['selected_solver']='scalar_brent' if passed else 'original42_step_bisection'
        report['KCL_precheck']=check;series=fast_series if passed else old_series
        for seed in seeds:
            for precision in plan['precision_levels']:
                run=dict(seed=seed,precision=precision,status='running',segments=[])
                report['runs'].append(run);prefix=seed['name']+'_'+precision['name'];clock=time.monotonic()
                last_eval={};calls=0;max_kcl=0.
                def rate(A,u):
                    nonlocal calls,max_kcl
                    calls+=1;last_eval.update(command_v=float(A),u=float(u))
                    current,vh,kcl=series(A,u);max_kcl=max(max_kcl,abs(kcl))
                    if current==0:return 0.
                    return -seed['k_per_v']*max(math.log(current/seed['JS_A'])+seed['beta']*u,0.)
                def integrate(label,span,u0,command_function):
                    segment=dict(name=label,span=list(span),initial_u=float(u0),status='running');run['segments'].append(segment)
                    solution=solve_ivp(lambda s,y:[rate(command_function(s),float(y[0]))],span,[u0],method='DOP853',
                                       rtol=precision['rtol'],atol=precision['atol'],max_step=precision['max_step'],dense_output=True)
                    segment.update(success=bool(solution.success),message=str(solution.message),nfev=int(solution.nfev),
                                   accepted_steps=len(solution.t)-1,final_u=float(solution.y[0,-1]),final_path=float(solution.t[-1]),
                                   status='completed' if solution.success else 'solver_failed')
                    arrays[prefix+'_'+label+'_solver_coordinate']=solution.t;arrays[prefix+'_'+label+'_solver_state_u']=solution.y[0]
                    if not solution.success:raise RuntimeError('ODE segment failed: '+solution.message)
                    return solution
                try:
                    back=integrate('backward_0p2_to_zero',(.2,0.),anchor,lambda A:A)
                    u0=float(back.y[0,-1]);run['derived_initial_u']=u0
                    out=integrate('outward_zero_to5',(0.,5.),u0,lambda s:s)
                    ret=integrate('return5_to_zero',(5.,10.),float(out.y[0,-1]),lambda s:10.-s)
                    state=np.empty_like(voltage);state[outward]=out.sol(path[outward])[0];state[~outward]=ret.sol(path[~outward])[0]
                    if np.any(np.diff(state)>5e-7):raise RuntimeError('Forward state unexpectedly increases beyond solver tolerance')
                    evaluated=np.array([series(A,u) for A,u in zip(voltage,state)])
                    current,local,kcl=evaluated.T
                    if np.any(~np.isfinite(current)) or np.any(current[mask]<=0):raise RuntimeError('Invalid eligible current')
                    original_i,original_local=t.series(voltage,np.array([validate_state(u) for u in state]),ron)
                    native_compare=float(np.max(abs(np.log10(current[mask]/original_i[mask]))))
                    groups=dict(all500=mask,outward=mask&outward,return_=mask&~outward,
                                low_both_branches=mask&(voltage<=.2+1e-12),low_outward=mask&outward&(voltage<=.2+1e-12),
                                remaining_above_0p2=mask&(voltage>.2+1e-12))
                    scored={}
                    for name,m in groups.items():
                        errors=np.log10(current[m])-np.log10(observed[m])
                        scored[name]=dict(n=int(m.sum()),raw_RMSE_decade=float(np.sqrt(np.mean(errors**2))),
                            mean_signed_log_error_decade=float(np.mean(errors)),median_absolute_factor=float(10**np.median(abs(errors))))
                    def point(value,branch):
                        m=outward if branch=='outward' else ~outward
                        j=int(np.argmin(np.where(m,abs(voltage-value),np.inf)));assert abs(voltage[j]-value)<1e-10
                        return j
                    features={}
                    for name,f in original_features.items():
                        if 'magnitude_V' in f:
                            j=point(f['magnitude_V'],'return');value=float(current[j]);indices=[j]
                        else:
                            a,b=f['window_abs_V'];ja,jb=point(a,'return'),point(b,'return');indices=[ja,jb]
                            value=float(math.log(current[jb]/current[ja])/math.log(voltage[jb]/voltage[ja]))
                        low,high=f['proposed_empirical_acceptance_interval']
                        features[name]=dict(single_cycle_prediction=value,original32_aggregate_band=[low,high],
                            single_prediction_inside_reference_band=low<=value<=high,native_indices=indices,
                            scope='Reference comparison only; no32-record aggregate gate evaluated.')
                    janchor=point(.2,'outward')
                    run.update(status='completed_conditional_single_cycle_forward',metrics=scored,reference_features=features,
                        terminal_u=float(state[-1]),state_min=float(min(state)),state_max=float(max(state)),
                        forward_anchor_u=float(state[janchor]),forward_anchor_state_error=float(state[janchor]-anchor),
                        forward_anchor_current_a=float(current[janchor]),forward_anchor_current_over_observed=float(current[janchor]/observed[janchor]),
                        max_native_KCL_residual_a=float(max(abs(kcl))),max_rhs_KCL_residual_a=max_kcl,
                        native_current_vs_original_KCL_max_log10_difference=native_compare,
                        original_KCL_native_local_voltage_max_difference_v=float(max(abs(local-original_local))),
                        low_outward_points=[dict(native_index=int(j),command_v=float(voltage[j]),observed_a=float(observed[j]),predicted_a=float(current[j]),state_u=float(state[j])) for j in np.flatnonzero(groups['low_outward'])],
                        n_rhs_calls=calls)
                    arrays[prefix+'_state_u']=state;arrays[prefix+'_current_a']=current;arrays[prefix+'_local_voltage_v']=local;arrays[prefix+'_KCL_residual_a']=kcl
                except Exception as exc:
                    run.update(status='failed_preserved',error_type=type(exc).__name__,error=str(exc),last_rhs_evaluation=last_eval,n_rhs_calls=calls,max_rhs_KCL_residual_a=max_kcl)
                    if isinstance(exc,NumericalResolutionError):run['failure_scope']='numerical_unknown_below_table_resolution_not_physical_rejection'
                    if run['segments'] and run['segments'][-1]['status']=='running':run['segments'][-1]['status']='exception_before_segment_completion'
                    if isinstance(exc,TimeoutError):raise
                finally:run['elapsed_s']=time.monotonic()-clock
        report['status']='completed_two_fixed_seeds_with_prescribed_refinements'
        report['precision_comparisons']={}
        for seed in seeds:
            a,b=[seed['name']+'_'+x for x in ('nominal','refined')]
            if a+'_current_a' in arrays and b+'_current_a' in arrays:
                report['precision_comparisons'][seed['name']]=dict(
                    max_native_current_log10_change=float(np.max(abs(np.log10(arrays[a+'_current_a'][mask]/arrays[b+'_current_a'][mask])))),
                    max_native_state_change=float(np.max(abs(arrays[a+'_state_u']-arrays[b+'_state_u']))),
                    terminal_state_change=float(arrays[a+'_state_u'][-1]-arrays[b+'_state_u'][-1]))
    except Exception as exc:
        report.update(status='partial_or_failed_preserved_no_restart',error_type=type(exc).__name__,error=str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0.);report['compute_elapsed_s']=time.monotonic()-started
    report['prior_interface_failure_compute_s']=spent
    report['cumulative_compute_elapsed_s']=report['compute_elapsed_s']+spent
    report['n_completed_runs']=sum(x['status']=='completed_conditional_single_cycle_forward' for x in report['runs'])
    report['n_failed_runs']=sum(x['status']=='failed_preserved' for x in report['runs'])
    report['scope']=['All500 original eligible points are scored, including low bias and the conditioning anchor; this is conditional accuracy, not independent prediction.',
                     'The same equation propagates0->5->0 after backward initialization; no frozen0->.2 segment or phase-state reset.',
                     'Singlecycle4 only. No RESET,32-cyclehistory,bare1R path formation,shared unification or microscopic identification is established.',
                     'JS is the reference of a log drive; physical state-dependent threshold is JS*exp(-beta*u), not an imposed compliance.',
                     'Backward reconstruction uses the measured .2V anchor. It does not identify the actual pre-SET state or validate a causal initialization procedure.']
    destination.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(destination.with_suffix('.npz'),**arrays)
    lines=['# 两个预选 soft SET 候选：完整单圈条件前向检查','',
           '固定共同谱、Ron/gc/JS与第4圈外扫0.2V锚。先用同一有符号物理ODE从0.2V反积到0V求初态，再完整0→5→0；返扫沿累计路径继续演化，没有冻结低压段。每候选只做指定名义与加密精度，无拟合、额外seed、RESET或跨圈历史。',
           '', '| 候选 | 精度 | 状态 | 全500点RMSE/dec | 外扫低压10点 | 两支低压20点 | 其余480点 |',
           '|---|---|---|---:|---:|---:|---:|']
    for run in report['runs']:
        m=run.get('metrics')
        values=[f'{m[g]["raw_RMSE_decade"]:.7g}' for g in ('all500','low_outward','low_both_branches','remaining_above_0p2')] if m else ['—']*4
        lines.append('| '+' | '.join([run['seed']['name'],run['precision']['name'],run['status'],*values])+' |')
    lines+=['',f'完成{report["n_completed_runs"]}个完整运行，失败{report["n_failed_runs"]}个；含首次接口检查的总计算{report["cumulative_compute_elapsed_s"]:.3f}s，预算120s。状态：{report["status"]}。',
            '', '首次接口检查因exp(log5)向下舍入而误拒5V（0.0356s、0个ODE）；原源码/计划/结果完整保留于pre_circuit_domain_fix。修正为原生log域比较，并独立约束晶体管压降及夹根。物理参数、谱域与种子均未改变，已从总预算扣除首次耗时。',
            '', '低压组定义为0<|V命令|≤0.2V；原始非零电压、|I|≥1e−11A及有限观测的mask保持500点。没有减零点，没有替换或删去低偏残差；条件锚也纳入评分，故不是独立测试误差。',
            '', 'JSON保存反向/外扫/返扫求解器状态、初末态、0.2V锚重合误差、KCL旧新法比较、加密差异和逐点低压残差；NPZ保存所有完整原生轨迹及已完成求解段。越界或求解失败原样保留，不投影或继续扩域。',
            '', '六个原SET返扫特征仅将此单圈值与原32圈聚合参考带并列，不声明聚合门槛通过。Backward初态来自本圈已测0.2V，因此仅检查指定锚附近存在的完整条件轨迹，不是识别真实初态，也不是前向因果初始化验证。']
    if 'KCL_precheck' in report:
        c=report['KCL_precheck'];lines+=['',f'KCL预检{c["n_points"]}点；新旧电流最大log差{c["max_abs_log10_current_difference"]:.3g}dec，局部电压最大差{c["max_local_voltage_difference_v"]:.3g}V，KCL残差{c["max_abs_KCL_residual_a"]:.3g}A；选择{c["selected_solver"]}。']
    for seed in seeds:
        lines+=['',f'候选{seed["name"]}：β={seed["beta"]:g}，JS={seed["JS_A"]:.12g} A，k={seed["k_per_v"]:.12g} V⁻¹。JS是log驱动参照值，有效阈值为JS exp(−βu)，不能当作实际限流。']
    if report.get('precision_comparisons'):
        for name,c in report['precision_comparisons'].items():lines+=['',f'{name}加密：原生电流最大改变{c["max_native_current_log10_change"]:.3g}dec，状态最大改变{c["max_native_state_change"]:.3g}，末态差{c["terminal_state_change"]:.3g}。']
    destination.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=report['status'],compute_elapsed_s=report['compute_elapsed_s'],completed=report['n_completed_runs'],failed=report['n_failed_runs'],
        runs=[{k:x[k] for k in ('seed','precision','status','metrics','error','derived_initial_u','terminal_u','forward_anchor_state_error') if k in x} for x in report['runs']],
        precision_comparisons=report.get('precision_comparisons')),indent=2))


if __name__=='__main__':main()
