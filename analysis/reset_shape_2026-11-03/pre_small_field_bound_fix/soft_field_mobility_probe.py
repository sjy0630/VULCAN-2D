"""One fixed SET local-field-mobility probe, screened before any ODE.

No fitting, rate scan, additional exponent, RESET change or prior-file edit.
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
CACHE=Path('/private/tmp/vulcan-runtime-cache')
OLD=ROOT/'analysis/mechanism_selection_2026-11-03'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(OLD))
from SOFT_FEEDBACK_LOW_BIAS_BUDGET import cached_transport


class DomainError(ValueError):
    pass


class NumericalUnknown(ValueError):
    pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    old_json=HERE/'soft_beta_conditional_forward.json'
    old=json.loads(old_json.read_text())
    original_files=[HERE/('soft_beta_conditional_forward.'+suffix) for suffix in ('py','plan.json','json','npz','md')]
    preserved={str(p):sha(p) for p in original_files}
    budget_path=HERE/'soft_beta_relaxed_budget.json'
    budget=json.loads(budget_path.read_text())
    base=next(x for x in old['plan']['seeds'] if x['name']=='beta6')
    beta=base['beta'];js=base['JS_A'];kappa=base['k_per_v']
    assert beta==6. and base['ceiling_fraction']==.8 and kappa==.8900744371038326
    ron=old['plan']['Ron_ohm'];gc=old['plan']['gamma_c_ev'];anchor=budget['anchors_u']['initial_0p2']
    plan=dict(scope='One fixed local-field mobility SET candidate, conditional cycle4 only; no complete unified model or microscopic mechanism claim.',
        beta=beta,ceiling_fraction=.8,JS_A=js,kappa_per_v2=kappa,
        kappa_selection='Exactly the previous beta6 k numeric value, now multiplying local Vh; its unit is V^-2. No new fitted coefficient or scan.',
        Ron_ohm=ron,gamma_c_ev=gc,gate_voltage_v=1.1,parallel_path_a=0.,
        law='du/ds=-kappa*abs(Vh)*positive(log(IT/JS)+beta*u); Gamma=gc/expm1(u), signed positive SET current from series KCL',
        early_state_cells=64,late_state_cells=64,late_left_command_cells=60,early_path_v=1.8,
        early_cell_upper='Vh(A=2,u_high)*positive(logIT(A=2,u_low)-logJS+beta*u_high)',
        early_kappa_lower='sum(du/Emax)/1.8; any Emax<=0 means impossible cell crossing',
        late_command_lower='min_cells[Vh(Aleft,u_low)*positive(logIT(Aleft,u_high)-logJS+beta*u_low)]',
        late_kappa_upper='(u5_max-u2_min)/sum(dA*late_command_lower); zero integral yields infinity, not curve success',
        screen='Only if the specified kappa is inside this necessary interval may the two declared precisions run. Otherwise stop without tuning.',
        backward='Same negative signed du/dA integrated.2->0 from the existing cycle4.2V anchor; then complete0->5->0 with no low-bias freezing.',
        anchor_u=anchor,anchor_gamma_ev=budget['anchors_gamma_ev']['initial_0p2'],
        forward_path='s:0->5,A=s; s:5->10,A=10-s; duplicate native endpoints add no dwell.',
        precisions=[dict(name='nominal',rtol=1e-7,atol=1e-8,max_step=.02),dict(name='refined',rtol=1e-8,atol=1e-9,max_step=.01)],
        ODE_method='DOP853',max_physical_seeds=1,max_solver_attempts=2,max_ODE_segments=6,
        state_domain=[0.,80.],domain_policy='No state or voltage projection; every material current is evaluated within the existing table. Unknown subtable positive voltage stops unless the inactive-rate proof applies.',
        inactive_proof='For0<A<vmin:IT<=F(vmin,u). Only if log(F(vmin,u)/JS)+beta*u<=0 return exact state derivative0. Multiplication by nonnegative Vh preserves inactivity. Never assign an out-of-domain current.',
        KCL_method='Scalar brentq in VT in[0,min(A,lookup_max)] with bracket check; compare against original42-step bisection at36fixed points before ODE.',
        KCL_xtol_v=1e-15,KCL_rtol=1e-13,KCL_maxiter=100,
        KCL_comparison_commands_v=[.02,.1,.2,1.,2.,5.],
        KCL_comparison_states=[anchor,budget['anchors_u']['outward_2V_lower'],budget['anchors_u']['return_5V_lower'],budget['anchors_u']['return_2V_upper'],15.,25.],
        KCL_comparison_limits=dict(current_log10=5e-7,local_voltage_v=1e-9,residual_a=1e-13),
        scoring='Original500cycle4SET points with finite absI>=1e-11,V!=0; no current floors,baseline subtraction or low-point exclusions.',
        groups=['all500','outward250','return250','outward_low10','both_branches_low20','remaining480'],
        reference_features='Original six SET-return feature bands are comparison context for one trajectory, not32-record aggregate gates.',
        endpoint_checks=['outward2V>=original_lower','return5V>=original_lower','return2V<=original_upper'],
        compute_budget_s=120,budget_scope='Independent120s budget after input preparation/plan/hash loading; includes necessary screen,KCL,up to2ODE attempts,scoring,precision comparison and arithmetic audit.',
        no_extensions='No extra seed,exponent,kappa,initial-state adjustment,fit,RESET change or domain expansion; preserve failures and stop on timeout.')
    plan_path=HERE/'soft_field_mobility_probe.plan.json'
    plan_path.write_text(json.dumps(plan,indent=2,allow_nan=False)+'\n')
    t,lookups=cached_transport(CACHE)
    for p in (t.table_path,*lookups,CACHE/'RON_SCALING_REVIEW.json',CACHE/'RON_SCALING_REVIEW.npz'):
        assert sha(p)==budget['source_sha256'][str(p)]
    with np.load(old_json.with_suffix('.npz')) as z:
        voltage=z['command_voltage_v'].copy();path=z['path_voltage_s'].copy()
        signed_obs=z['observed_current_a'].copy();mask=z['original_scoring_mask'].copy()
    obs=abs(signed_obs);outward=np.arange(len(voltage))<=np.argmax(voltage)
    np.testing.assert_array_equal(mask,np.isfinite(obs)&(obs>=1e-11)&(abs(voltage)>1e-12));assert mask.sum()==500
    with np.load(CACHE/'RON_SCALING_REVIEW.npz') as z:commands=z['soft_set_command_left_v'].copy()
    assert len(commands)==60 and abs(commands[0]-2)<1e-12 and abs(commands[-1]-4.95)<1e-10
    source=json.loads((CACHE/'RON_SCALING_REVIEW.json').read_text())
    endpoints=next(r for r in source['rows'] if r['Ron_ohm']==ron)['anchor_reinversions']
    features=next(r for r in old['runs'] if r['status']=='completed_conditional_single_cycle_forward')['reference_features']
    sources={str(p):sha(p) for p in [budget_path,old_json,old_json.with_suffix('.npz'),t.table_path,*lookups,
        CACHE/'RON_SCALING_REVIEW.json',CACHE/'RON_SCALING_REVIEW.npz',OLD/'SOFT_FEEDBACK_LOW_BIAS_BUDGET.py',
        ROOT/'analysis/ron_scope_2026-11-03/ron_scaling_review.py',ROOT/'vulcan2d/transistor.py',plan_path,Path(__file__)]}
    report=dict(status='running_predeclared_field_probe',plan=plan,source_sha256=sources,
                original_two_seed_files_sha256=preserved,runs=[])
    dest=HERE/'soft_field_mobility_probe.json'
    dest.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    arrays=dict(command_voltage_v=voltage,path_voltage_s=path,observed_current_a=signed_obs,original_scoring_mask=mask)
    lg,lv,li=t.lg,t.lv,t.li[0];vmin=math.exp(lv[0]);logfactor=math.log(t.r0/ron);tx,ti=t.knots['set']
    def gamma_of(u):
        if not math.isfinite(u) or not 0<u<=80:raise DomainError(f'State outside finite-table admissible part of[0,80]:{u!r}')
        g=gc/math.expm1(u)
        if not lg[0]<=math.log(g)<=lg[-1]:raise DomainError(f'Gamma outside table at u={u!r}')
        return g
    def material(v,logg):
        if v==0:return 0.
        if v<0:raise DomainError('Negative material voltage')
        logv=math.log(v)
        if logv<lv[0]:raise NumericalUnknown(f'Material voltage below table resolution:{v!r}')
        if logv>lv[-1]:raise DomainError('Material voltage above table domain')
        ix=min(max(int(np.searchsorted(lg,logg)-1),0),len(lg)-2)
        iy=min(max(int(np.searchsorted(lv,logv)-1),0),len(lv)-2)
        fx=(logg-lg[ix])/(lg[ix+1]-lg[ix]);fy=(logv-lv[iy])/(lv[iy+1]-lv[iy])
        x=(1-fx)*((1-fy)*li[ix,iy]+fy*li[ix,iy+1])+fx*((1-fy)*li[ix+1,iy]+fy*li[ix+1,iy+1])
        return math.exp(x+logfactor)
    def fast(A,u):
        g=gamma_of(float(u));A=float(A)
        if A==0:return 0.,0.,0.
        if A<0 or math.log(A)>lv[-1]:raise DomainError('Command outside material domain')
        logg=math.log(g)
        def residual(vt):return float(np.interp(vt,tx,ti))-material(A-vt,logg)
        upper=min(A,float(tx[-1]))
        if residual(0)>0 or residual(upper)<0:raise DomainError('No KCL root inside both table domains')
        vt=brentq(residual,0.,upper,xtol=1e-15,rtol=1e-13,maxiter=100)
        vh=A-vt;i=material(vh,logg)
        return i,vh,i-float(np.interp(vt,tx,ti))
    def old_series(A,u):
        g=gamma_of(float(u));A=float(A)
        if A==0:return 0.,0.,0.
        if A<0 or math.log(A)>lv[-1]:raise DomainError('Command outside material domain')
        if math.log(A)<lv[0]:raise NumericalUnknown('Command below table resolution')
        i,vh=t.series(A,g,ron);i,vh=float(i),float(vh)
        if math.log(vh)<lv[0]:raise NumericalUnknown('KCL material voltage below table resolution')
        if math.log(vh)>lv[-1] or not 0<=A-vh<=tx[-1]:raise DomainError('KCL outside table domain')
        return i,vh,i-float(np.interp(A-vh,tx,ti))
    def timeout(*_):raise TimeoutError('120-second field-probe budget exhausted; no restart')
    start=time.monotonic();signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,120.)
    try:
        start_u=budget['anchors_u']['initial_0p2'];target_u=budget['anchors_u']['outward_2V_lower']
        lo=budget['anchors_u']['return_2V_upper'];hi=budget['anchors_u']['return_5V_lower']
        early=np.linspace(target_u,start_u,65);late=np.linspace(lo,hi,65)
        ie,ve=t.series(np.full(65,2.),gc/np.expm1(early),ron)
        il,vl=t.series(commands[:,None],gc/np.expm1(late)[None,:],ron)
        assert np.all(np.diff(ie)<=1e-14) and np.all(np.diff(ve)>=-1e-12)
        assert np.all(np.diff(il,axis=0)>=-1e-14) and np.all(np.diff(vl,axis=0)>=-1e-12)
        assert np.all(np.diff(il,axis=1)<=1e-14) and np.all(np.diff(vl,axis=1)>=-1e-12)
        early_upper=ve[1:]*np.maximum(np.log(ie[:-1])-math.log(js)+beta*early[1:],0.)
        blocked=bool(np.any(early_upper<=0));lower=math.inf if blocked else float(np.sum(np.diff(early)/early_upper)/1.8)
        late_cells=vl[:,:-1]*np.maximum(np.log(il[:,1:])-math.log(js)+beta*late[:-1][None,:],0.)
        late_lower=np.min(late_cells,axis=1);integral=float(.05*np.sum(late_lower))
        upper=(hi-lo)/integral if integral>0 else math.inf
        allowed=bool(not blocked and lower<=kappa<=upper)
        report['necessary_screen']=dict(kappa_lower_per_v2=lower if math.isfinite(lower) else None,
            kappa_upper_per_v2=upper if math.isfinite(upper) else None,early_blocked=blocked,late_upper_unbounded=integral==0,
            fixed_kappa_per_v2=kappa,fixed_kappa_inside_necessary_interval=allowed,
            early_min_cell_upper_v=float(min(early_upper)),late_integral_lower_v2=integral,
            start_u=start_u,target_u=target_u,return_state_interval=[lo,hi],
            max_budget_KCL_residual_a=float(max(np.max(abs(ie-np.interp(2.-ve,tx,ti))),np.max(abs(il-np.interp(commands[:,None]-vl,tx,ti))))),
            scope='Fixed-table conditional necessary inequalities only; not sufficient for actual arrival, full curve or original primary feature acceptance.')
        arrays.update(early_u=early,early_current_a=ie,early_local_voltage_v=ve,early_cell_upper=early_upper,
                      late_u=late,late_command_left_v=commands,late_current_a=il,late_local_voltage_v=vl,
                      late_cell_lower=late_cells,late_command_lower=late_lower)
        print(json.dumps(dict(stage='necessary_screen',**report['necessary_screen'])),flush=True)
        if not allowed:
            report['status']='stopped_fixed_kappa_excluded_by_necessary_budget_no_ODE'
        else:
            comparisons=[]
            for A in plan['KCL_comparison_commands_v']:
                for u in plan['KCL_comparison_states']:
                    x,y=fast(A,u),old_series(A,u)
                    comparisons.append([abs(math.log10(x[0]/y[0])),abs(x[1]-y[1]),abs(x[2])])
            maximum=np.max(comparisons,axis=0);limits=plan['KCL_comparison_limits']
            passed=bool(maximum[0]<=limits['current_log10'] and maximum[1]<=limits['local_voltage_v'] and maximum[2]<=limits['residual_a'])
            report['KCL_comparison']=dict(n_points=36,max_log10_current_difference=float(maximum[0]),max_local_voltage_difference_v=float(maximum[1]),
                                          max_abs_KCL_residual_a=float(maximum[2]),passed=passed,selected='scalar_brent' if passed else 'original42_bisection')
            series=fast if passed else old_series
            for precision in plan['precisions']:
                run=dict(precision=precision,status='running',segments=[]);report['runs'].append(run)
                prefix=precision['name'];clock=time.monotonic();last={};cert=[];calls=0;rhs_kcl=0.
                print(json.dumps(dict(stage='ODE_start',precision=prefix)),flush=True)
                def rate(A,u):
                    nonlocal calls,rhs_kcl
                    calls+=1;last.update(command_v=float(A),u=float(u))
                    if 0<A<vmin:
                        g=gamma_of(u);imax=material(vmin,math.log(g));drive=math.log(imax/js)+beta*u
                        cert.append(dict(command_v=float(A),u=float(u),upper_current_a=imax,upper_log_drive=drive,
                                         inactive=drive<=0,out_of_domain_current_evaluated=False))
                        if drive<=0:return 0.
                        raise NumericalUnknown('Subtable positive command has active upper drive; cannot continue')
                    i,vh,kcl=series(A,u);rhs_kcl=max(rhs_kcl,abs(kcl))
                    return 0. if i==0 else -kappa*abs(vh)*max(math.log(i/js)+beta*u,0.)
                def integrate(name,span,u0,command):
                    row=dict(name=name,span=list(span),initial_u=float(u0),status='running');run['segments'].append(row)
                    sol=solve_ivp(lambda s,y:[rate(command(s),float(y[0]))],span,[u0],method='DOP853',
                                  rtol=precision['rtol'],atol=precision['atol'],max_step=precision['max_step'],dense_output=True)
                    row.update(status='completed' if sol.success else 'solver_failed',success=bool(sol.success),message=str(sol.message),
                               nfev=int(sol.nfev),accepted_steps=len(sol.t)-1,final_u=float(sol.y[0,-1]))
                    arrays[prefix+'_'+name+'_coordinate']=sol.t;arrays[prefix+'_'+name+'_state_u']=sol.y[0]
                    if not sol.success:raise RuntimeError(sol.message)
                    return sol
                try:
                    back=integrate('backward',(.2,0.),anchor,lambda A:A);u0=float(back.y[0,-1])
                    run['derived_initial_u']=u0
                    forward=integrate('outward',(0.,5.),u0,lambda s:s)
                    reverse=integrate('return',(5.,10.),float(forward.y[0,-1]),lambda s:10.-s)
                    state=np.empty_like(voltage);state[outward]=forward.sol(path[outward])[0];state[~outward]=reverse.sol(path[~outward])[0]
                    assert np.max(np.diff(state))<=5e-7
                    values=np.array([series(A,u) for A,u in zip(voltage,state)]);current,local,kcl=values.T
                    assert np.all(np.isfinite(current)) and np.all(current>=0) and np.all(current[mask]>0)
                    groups=dict(all500=mask,outward250=mask&outward,return250=mask&~outward,
                        outward_low10=mask&outward&(voltage<=.2+1e-12),both_branches_low20=mask&(voltage<=.2+1e-12),remaining480=mask&(voltage>.2+1e-12))
                    scores={}
                    for name,m in groups.items():
                        e=np.log10(current[m]/obs[m]);scores[name]=dict(n=int(m.sum()),raw_RMSE_decade=float(np.sqrt(np.mean(e*e))),
                            mean_signed_log_error_decade=float(np.mean(e)),median_absolute_factor=float(10**np.median(abs(e))))
                    refs={}
                    for name,f in features.items():
                        jj=f['native_indices']
                        val=float(current[jj[0]]) if len(jj)==1 else float(math.log(current[jj[1]]/current[jj[0]])/math.log(voltage[jj[1]]/voltage[jj[0]]))
                        a,b=f['original32_aggregate_band']
                        refs[name]=dict(single_cycle_prediction=val,original32_aggregate_band=[a,b],single_prediction_inside_reference_band=a<=val<=b,
                                        native_indices=jj,scope='Single cycle beside reference band; not a32-record aggregate gate test.')
                    endpoint_rows=[]
                    for name,A,isout,kind in [('outward_2V_lower',2.,True,'at_least'),('return_5V_lower',5.,False,'at_least'),('return_2V_upper',2.,False,'at_most')]:
                        branch=outward if isout else ~outward;j=int(np.argmin(np.where(branch,abs(voltage-A),np.inf)));assert voltage[j]==A
                        bound=endpoints[name]['current_a'];value=float(current[j])
                        endpoint_rows.append(dict(name=name,native_index=j,predicted_current_a=value,bound_current_a=bound,direction=kind,
                            actual_u=float(state[j]),bound_u=float(math.log1p(gc/endpoints[name]['rederived_gamma_ev'])),
                            condition_satisfied=value>=bound if kind=='at_least' else value<=bound))
                    ja=int(np.argmin(np.where(outward,abs(voltage-.2),np.inf)))
                    check_i,check_v=t.series(voltage,np.array([gamma_of(float(u)) for u in state]),ron)
                    run.update(status='completed_fixed_single_seed_conditional_forward',metrics=scores,reference_features=refs,
                        actual_budget_endpoints=endpoint_rows,all_three_endpoint_conditions_satisfied=all(x['condition_satisfied'] for x in endpoint_rows),
                        terminal_u=float(state[-1]),forward_anchor_state_error=float(state[ja]-anchor),forward_anchor_current_over_observed=float(current[ja]/obs[ja]),
                        max_native_KCL_residual_a=float(max(abs(kcl))),max_rhs_KCL_residual_a=rhs_kcl,
                        max_native_log_current_difference_vs_original_KCL=float(max(abs(np.log10(current[mask]/check_i[mask])))),
                        max_native_local_voltage_difference_vs_original_KCL=float(max(abs(local-check_v))),
                        low_outward_points=[dict(native_index=int(j),command_v=float(voltage[j]),observed_a=float(obs[j]),predicted_a=float(current[j]),state_u=float(state[j])) for j in np.flatnonzero(groups['outward_low10'])])
                    arrays[prefix+'_current_a']=current;arrays[prefix+'_state_u']=state;arrays[prefix+'_local_voltage_v']=local;arrays[prefix+'_KCL_residual_a']=kcl
                    # Pure arithmetic checks independent of the metric loop.
                    rmse=float(np.linalg.norm(np.log10(current[mask])-np.log10(obs[mask]))/math.sqrt(500))
                    np.testing.assert_allclose(rmse,scores['all500']['raw_RMSE_decade'],rtol=1e-12)
                    run['arithmetic_audit_passed']=True
                except Exception as exc:
                    run.update(status='failed_preserved',error_type=type(exc).__name__,error=str(exc),last_rhs_evaluation=last,
                        failure_scope='numerical_unknown_not_physical_rejection' if isinstance(exc,NumericalUnknown) else 'specified_candidate_or_solver_failure')
                    if run['segments'] and run['segments'][-1]['status']=='running':run['segments'][-1]['status']='exception_before_completion'
                    if isinstance(exc,TimeoutError):raise
                finally:
                    run.update(elapsed_s=time.monotonic()-clock,n_rhs_calls=calls,
                        inactive_endpoint_certificate=dict(test_count=len(cert),inactive_count=sum(x['inactive'] for x in cert),
                            max_upper_log_drive=max((x['upper_log_drive'] for x in cert),default=None),
                            state_range=[min(x['u'] for x in cert),max(x['u'] for x in cert)] if cert else None,evaluations=cert))
            if all(r['status']=='completed_fixed_single_seed_conditional_forward' for r in report['runs']):
                report['precision_comparison']=dict(max_native_current_log10_change=float(max(abs(np.log10(arrays['nominal_current_a'][mask]/arrays['refined_current_a'][mask])))),
                    max_native_state_change=float(max(abs(arrays['nominal_state_u']-arrays['refined_state_u']))),
                    terminal_state_change=float(arrays['nominal_state_u'][-1]-arrays['refined_state_u'][-1]))
            report['status']='completed_fixed_field_probe_with_all_failures_retained'
        unchanged={p:sha(p)==h for p,h in preserved.items()};assert all(unchanged.values())
        report['original_two_seed_files_unchanged']=unchanged
    except Exception as exc:
        report.update(status='partial_or_failed_field_probe_preserved_no_restart',error_type=type(exc).__name__,error=str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0.);report['compute_elapsed_s']=time.monotonic()-start
    report['n_completed_runs']=sum(x['status']=='completed_fixed_single_seed_conditional_forward' for x in report['runs'])
    report['n_failed_runs']=sum(x['status']=='failed_preserved' for x in report['runs'])
    dest.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');np.savez_compressed(dest.with_suffix('.npz'),**arrays)
    lines=['# 单一局部场迁移率 SET 候选','',
        '固定β=6、JS/gc/Ron及既有0.2V锚；只把原SET速率乘局部材料电压|Vh|。κ数值固定0.8900744371038326、单位V⁻²，没有拟合、率扫描或新增指数。这是待检候选，尚无微观机制证明，也未修改RESET或检验1R/1T1R全统一。',
        '',f'状态：{report["status"]}；计算{report["compute_elapsed_s"]:.6f}s，独立预算120s。']
    if 'necessary_screen' in report:
        screen=report['necessary_screen'];lines+=['',f'64格早界κ≥{screen["kappa_lower_per_v2"]}；60左节点晚界κ≤{screen["kappa_upper_per_v2"] if screen["kappa_upper_per_v2"] is not None else "∞"}。指定κ未被必要区间排除：{screen["fixed_kappa_inside_necessary_interval"]}。必要界不是充分条件，实际端点另列。']
    lines+=['', '| 精度 | 状态 | 全500点RMSE | 外扫低压10点 | 其余480点 | 返扫250点 |', '|---|---|---:|---:|---:|---:|']
    for run in report['runs']:
        m=run.get('metrics');values=[f'{m[k]["raw_RMSE_decade"]:.8f}' for k in ('all500','outward_low10','remaining480','return250')] if m else ['—']*4
        lines.append('| '+' | '.join([run['precision']['name'],run['status'],*values])+' |')
    for run in report['runs']:
        if 'actual_budget_endpoints' not in run:continue
        lines+=['',f'## {run["precision"]["name"]}实际端点','', '| 条件 | 实际电流/µA | 原界/µA | 实际u | 满足 |','|---|---:|---:|---:|---|']
        for e in run['actual_budget_endpoints']:lines.append(f'| {e["name"]} | {e["predicted_current_a"]*1e6:.8f} | {e["bound_current_a"]*1e6:.8f} | {e["actual_u"]:.9f} | {e["condition_satisfied"]} |')
        lines+=['', '| 原SET返扫特征 | 此单圈值 | 原32圈聚合参考带 | 此值在带内 |','|---|---:|---|---|']
        for name,f in run['reference_features'].items():lines.append(f'| {name} | {f["single_cycle_prediction"]:.9g} | {f["original32_aggregate_band"]} | {f["single_prediction_inside_reference_band"]} |')
    if 'precision_comparison' in report:
        c=report['precision_comparison'];lines+=['',f'加密后原生电流最大log变化{c["max_native_current_log10_change"]:.6g}dec，状态最大变化{c["max_native_state_change"]:.6g}，末态差{c["terminal_state_change"]:.6g}。']
    lines+=['', '所有500个原始有效SET点包含低偏与条件锚；未减零点或修改分析门槛。初态由同圈0.2V观测反积得到，故不是独立预测或因果初态验证。六个单圈特征与原32圈带的比较不等于聚合通过。',
        '', '原两seed的源码/计划/JSON/NPZ/MD在执行前后均校验sha保持不变。新计划、全部预算节点、两精度轨迹、失败状态、近零inactive证书和实到端点均保存于本候选同名前缀文件。未追加候选、κ值或初态。']
    dest.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=report['status'],compute_elapsed_s=report['compute_elapsed_s'],necessary_screen=report.get('necessary_screen'),
        completed=report['n_completed_runs'],failed=report['n_failed_runs'],precision_comparison=report.get('precision_comparison'),
        runs=[{k:r[k] for k in ('precision','status','metrics','actual_budget_endpoints','error') if k in r} for r in report['runs']]),indent=2))


if __name__=='__main__':main()
