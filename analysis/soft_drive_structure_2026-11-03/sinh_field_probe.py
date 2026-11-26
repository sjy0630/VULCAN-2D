"""One approved, data-informed fixed sinh field-mobility conditional probe.

Vf=.5V is an additional structural scale, not a measured microscopic length
or activation barrier. No scale/rate scan or fitting; preserve every failure.
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
PREVIOUS=ROOT/'analysis/reset_shape_2026-11-03'
CACHE=Path('/private/tmp/vulcan-runtime-cache')
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'analysis/mechanism_selection_2026-11-03'))
from SOFT_FEEDBACK_LOW_BIAS_BUDGET import cached_transport


class DomainError(ValueError):pass
class NumericalUnknown(ValueError):pass


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    base_path=PREVIOUS/'soft_field_mobility_probe.json';base=json.loads(base_path.read_text())
    budget_path=PREVIOUS/'soft_beta_relaxed_budget.json';budget=json.loads(budget_path.read_text())
    beta,js,ron,gc=[base['plan'][key] for key in ('beta','JS_A','Ron_ohm','gamma_c_ev')]
    assert beta==6. and js==1938.5461216120304
    vf=.5;anchor=budget['anchors_u']['initial_0p2']
    guarded=[PREVIOUS/(name+'.'+ext) for name in ('soft_beta_conditional_forward','soft_field_mobility_probe') for ext in ('py','plan.json','json','npz','md')]
    guarded+=[HERE/('drive_trace_diagnostic.'+ext) for ext in ('py','plan.json','json','npz','md')]
    guards={str(p):sha(p) for p in guarded}
    plan=dict(scope='One exploratory data-informed fixed-scale1T1Rcycle4SET probe; no holdout,RESET,bare1R or unified-history validation.',
        mobility='M(v)=Vf*sinh(v/Vf),smallfield limit M(v)/v->1',Vf_v=vf,
        scale_interpretation='One added fixed structural scale selected after the existing residual diagnostic; not a measured hopping length,activation barrier or independently validated microscopic parameter.',
        beta=beta,JS_A=js,Ron_ohm=ron,gamma_c_ev=gc,parallel_path_a=0.,gate_voltage_v=1.1,
        state_law='du/ds=-kappa*M(abs(Vh))*positive(log(IT/JS)+beta*u);Gamma=gc/expm1(u)',
        kappa_rule='Exactly2*the recomputed64-cell necessary early lower bound; unitsV^-2. No adjustment after actual trajectories.',
        early_state_cells=64,early_cell_upper='M(Vh(2,u_high))*positive(logIT(2,u_low)-logJS+beta*u_high)',
        early_kappa_lower='sum(du/Emax)/1.8; any nonpositive Emax prevents crossing and stops the probe',
        late_state_cells=64,late_left_command_cells=60,
        late_lower='min_j[M(Vh(Aleft,u_low_j))*positive(logIT(Aleft,u_high_j)-logJS+beta*u_low_j)]',
        late_upper='(u5_max-u2_min)/sum(.05*late_lower);zero integral means infinity only',
        screen_policy='Run only if the unique derived kappa lies in the necessary interval; no tuning when excluded or when actual endpoints fail.',
        anchor_command_v=.2,anchor_u=anchor,anchor_gamma_ev=budget['anchors_gamma_ev']['initial_0p2'],
        initialization='Reverse integrate the same signed physical ODE.2->0; then full0->5->0,return uses cumulative path s=5->10. No frozen0->.2.',
        precisions=[dict(name='nominal',rtol=1e-7,atol=1e-8,max_step=.02),dict(name='refined',rtol=1e-8,atol=1e-9,max_step=.01)],
        method='DOP853',max_physical_seeds=1,max_solver_attempts=2,max_ODE_segments=6,
        domain='u in[0,80],Gamma and nonzero material voltage inside existing table; no state/voltage projection or extrapolated current.',
        exact_inactive='For0<A<vmin,IT<=F(vmin,u);if this gives log(F(vmin,u)/JS)+beta*u<=0,rate is exactly0.',
        bounded_smallfield='Otherwise IT<=A/Ron,Vh<=A,u<=80 give B=kappa*Dmax*Vf^2*(cosh(vmin/Vf)-1)=2*kappa*Dmax*Vf^2*sinh(vmin/(2Vf))^2. Dmax=max(log(vmin/(RonJS))+80beta,0). Use the stable sinh-squared form; only B<1e-12 plus valid state/Gamma permits numerical RHS omission. No out-of-domain current is assigned.',
        bound_scope='Per monotone0<->vmin physical segment,not per RHS retry and not a global stability-adjusted trajectory error bound.',
        KCL='Scalar brentq inVT in[0,min(A,Tmax)],strict bracket. Compare36 fixed points against original42-step bisection; use original if comparison fails.',
        KCL_tolerances=dict(xtol_v=1e-15,rtol=1e-13,maxiter=100,comparison_logI=5e-7,comparison_Vh_v=1e-9,comparison_KCL_a=1e-13),
        scoring='Original500 finite nonzero-voltage cycle4SET absI>=1e-11 points; raw log residual, no floors,baseline correction or point exclusions.',
        groups=['all500','outward250','return250','outward_low10','both_branches_low20','remaining480','outward0p5_to1p5_50'],
        outward_current_thresholds_a=[1e-6,1e-5],
        actual_endpoint_checks=['outward2V>=original lower','return5V>=original lower','return2V<=original upper'],
        feature_scope='All original6SETreturn features compared with32-record reference bands; a single-cycle value is not an aggregate gate evaluation.',
        compute_budget_s=120,budget_start='After all input loads/hash guard and plan saving; includes necessary screen,all ODEs,scoring and arithmetic audit.',
        stopping_policy='Preserve failures; no extra scale,rate,seed,parameter fitting or numerical recovery beyond these2planned attempts. No final-goal claim.')
    plan_path=HERE/'sinh_field_probe.plan.json';plan_path.write_text(json.dumps(plan,indent=2,allow_nan=False)+'\n')
    t,lookups=cached_transport(CACHE)
    for p in (t.table_path,*lookups,CACHE/'RON_SCALING_REVIEW.json',CACHE/'RON_SCALING_REVIEW.npz'):
        assert sha(p)==budget['source_sha256'][str(p)]
    with np.load(base_path.with_suffix('.npz')) as z:
        v=z['command_voltage_v'].copy();s=z['path_voltage_s'].copy();signed_obs=z['observed_current_a'].copy();mask=z['original_scoring_mask'].copy()
    obs=abs(signed_obs);out=np.arange(len(v))<=np.argmax(v)
    np.testing.assert_array_equal(mask,np.isfinite(obs)&(obs>=1e-11)&(v>1e-12));assert mask.sum()==500
    with np.load(CACHE/'RON_SCALING_REVIEW.npz') as z:commands=z['soft_set_command_left_v'].copy()
    assert len(commands)==60 and abs(commands[0]-2)<1e-12 and abs(commands[-1]-4.95)<1e-10
    anchors=next(x for x in json.loads((CACHE/'RON_SCALING_REVIEW.json').read_text())['rows'] if x['Ron_ohm']==ron)['anchor_reinversions']
    feature_ref=next(x for x in base['runs'] if x['precision']['name']=='refined' and 'metrics' in x)['reference_features']
    sources={str(p):sha(p) for p in [base_path,base_path.with_suffix('.npz'),budget_path,t.table_path,*lookups,CACHE/'RON_SCALING_REVIEW.json',
        CACHE/'RON_SCALING_REVIEW.npz',ROOT/'analysis/mechanism_selection_2026-11-03/SOFT_FEEDBACK_LOW_BIAS_BUDGET.py',
        ROOT/'analysis/ron_scope_2026-11-03/ron_scaling_review.py',ROOT/'vulcan2d/transistor.py',plan_path,Path(__file__)]}
    report=dict(status='running_one_fixed_sinh_probe',plan=plan,source_sha256=sources,frozen_baseline_hashes=guards,runs=[])
    dest=HERE/'sinh_field_probe.json';dest.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    arrays=dict(command_voltage_v=v,path_voltage_s=s,observed_current_a=signed_obs,original_scoring_mask=mask)
    lg,lv,li=t.lg,t.lv,t.li[0];tx,ti=t.knots['set'];vmin=math.exp(lv[0]);lf=math.log(t.r0/ron)
    def gamma(u):
        if not math.isfinite(u) or not 0<u<=80:raise DomainError(f'State outside admissible finite-table subset of[0,80]:{u!r}')
        g=gc/math.expm1(u)
        if not lg[0]<=math.log(g)<=lg[-1]:raise DomainError('Gamma outside table')
        return g
    def material(V,logg):
        if V==0:return 0.
        if V<0 or math.log(V)>lv[-1]:raise DomainError('Material voltage outside table')
        y=math.log(V)
        if y<lv[0]:raise NumericalUnknown('Positive material voltage below table resolution')
        ix=min(max(int(np.searchsorted(lg,logg)-1),0),len(lg)-2);iy=min(max(int(np.searchsorted(lv,y)-1),0),len(lv)-2)
        a=(logg-lg[ix])/(lg[ix+1]-lg[ix]);b=(y-lv[iy])/(lv[iy+1]-lv[iy])
        return math.exp((1-a)*((1-b)*li[ix,iy]+b*li[ix,iy+1])+a*((1-b)*li[ix+1,iy]+b*li[ix+1,iy+1])+lf)
    def fast(A,u):
        logg=math.log(gamma(float(u)));A=float(A)
        if A==0:return 0.,0.,0.
        if A<0 or math.log(A)>lv[-1]:raise DomainError('Command outside table domain')
        def f(Vt):return float(np.interp(Vt,tx,ti))-material(A-Vt,logg)
        top=min(A,float(tx[-1]))
        if f(0)>0 or f(top)<0:raise DomainError('KCL root not bracketed in actual table domains')
        vt=brentq(f,0.,top,xtol=1e-15,rtol=1e-13,maxiter=100);vh=A-vt;i=material(vh,logg)
        return i,vh,i-float(np.interp(vt,tx,ti))
    def legacy(A,u):
        g=gamma(float(u));A=float(A)
        if A==0:return 0.,0.,0.
        if A<0 or math.log(A)>lv[-1]:raise DomainError('Command outside table')
        if math.log(A)<lv[0]:raise NumericalUnknown('Positive command below table resolution')
        i,vh=t.series(A,g,ron);i,vh=float(i),float(vh)
        if math.log(vh)<lv[0]:raise NumericalUnknown('Legacy material voltage below table resolution')
        if math.log(vh)>lv[-1] or not 0<=A-vh<=tx[-1]:raise DomainError('Legacy root outside material/transistor domains')
        return i,vh,i-float(np.interp(A-vh,tx,ti))
    def M(V):return vf*np.sinh(np.asarray(V)/vf)
    def timeout(*_):raise TimeoutError('120second fixed sinh probe budget exhausted; no restart')
    begun=time.monotonic();signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,120.)
    try:
        passivity=float(np.max(li-(lv[None,:]-math.log(t.r0))));assert passivity<=0
        report['table_passivity_max_log_ratio']=passivity
        early=np.linspace(budget['anchors_u']['outward_2V_lower'],anchor,65)
        lo,hi=budget['anchors_u']['return_2V_upper'],budget['anchors_u']['return_5V_lower'];late=np.linspace(lo,hi,65)
        ie,ve=t.series(np.full(65,2.),gc/np.expm1(early),ron)
        il,vl=t.series(commands[:,None],gc/np.expm1(late)[None,:],ron)
        assert np.all(np.diff(ie)<=1e-14) and np.all(np.diff(ve)>=-1e-12)
        assert np.all(np.diff(il,axis=0)>=-1e-14) and np.all(np.diff(vl,axis=0)>=-1e-12)
        upper_cells=M(ve[1:])*np.maximum(np.log(ie[:-1])-math.log(js)+beta*early[1:],0.)
        blocked=bool(np.any(upper_cells<=0));lower=math.inf if blocked else float(np.sum(np.diff(early)/upper_cells)/1.8)
        late_cells=M(vl[:,:-1])*np.maximum(np.log(il[:,1:])-math.log(js)+beta*late[:-1][None,:],0.)
        late_lower=np.min(late_cells,axis=1);integral=float(.05*np.sum(late_lower));upper=(hi-lo)/integral if integral>0 else math.inf
        kappa=2*lower;allowed=bool(not blocked and math.isfinite(kappa) and lower<=kappa<=upper)
        report['necessary_screen']=dict(kappa_lower_per_v2=lower if math.isfinite(lower) else None,kappa_upper_per_v2=upper if math.isfinite(upper) else None,
            derived_unique_kappa_per_v2=kappa if math.isfinite(kappa) else None,early_blocked=blocked,late_upper_unbounded=integral==0,
            late_integral_lower_v2=integral,fixed_rule_kappa_inside_necessary_interval=allowed,
            max_budget_KCL_residual_a=float(max(np.max(abs(ie-np.interp(2.-ve,tx,ti))),np.max(abs(il-np.interp(commands[:,None]-vl,tx,ti))))))
        arrays.update(early_u=early,early_current_a=ie,early_local_voltage_v=ve,early_cell_upper=upper_cells,
            late_u=late,late_command_left_v=commands,late_current_a=il,late_local_voltage_v=vl,late_cell_lower=late_cells,late_command_lower=late_lower)
        print(json.dumps(dict(stage='necessary_screen',**report['necessary_screen'])),flush=True)
        if not allowed:report['status']='stopped_sinh_rule_excluded_by_necessary_interval_no_ODE'
        else:
            Dmax=max(math.log(vmin/(ron*js))+beta*80,0.)
            B=2*kappa*Dmax*vf**2*math.sinh(vmin/(2*vf))**2
            report['smallfield_bound']=dict(vmin_v=vmin,Dmax=Dmax,local_integral_upper_per_monotone_segment=B,below_declared_1e_minus12=B<1e-12,
                stable_expression='2*kappa*Dmax*Vf^2*sinh(vmin/(2Vf))^2',scope='Local omission bound,not a full-trajectory error certificate')
            comparison=[]
            for A in (.02,.1,.2,1.,2.,5.):
                for u in (anchor,early[0],hi,lo,15.,25.):
                    a,b=fast(A,u),legacy(A,u);comparison.append([abs(math.log10(a[0]/b[0])),abs(a[1]-b[1]),abs(a[2])])
            maxima=np.max(comparison,axis=0);passed=bool(maxima[0]<=5e-7 and maxima[1]<=1e-9 and maxima[2]<=1e-13)
            report['KCL_comparison']=dict(n_points=36,max_log_current_difference=float(maxima[0]),max_local_voltage_difference_v=float(maxima[1]),
                max_abs_KCL_residual_a=float(maxima[2]),passed=passed,selected='scalar_brent' if passed else 'legacy42_bisection')
            series=fast if passed else legacy
            for precision in plan['precisions']:
                run=dict(precision=precision,status='running',segments=[]);report['runs'].append(run);prefix=precision['name']
                clock=time.monotonic();last={};near=[];which=[None];calls=0;rhs_kcl=0.
                print(json.dumps(dict(stage='ODE_start',precision=prefix)),flush=True)
                def rate(A,u):
                    nonlocal calls,rhs_kcl
                    calls+=1;last.update(command_v=float(A),u=float(u))
                    if 0<A<vmin:
                        g=gamma(u);imax=material(vmin,math.log(g));d=math.log(imax/js)+beta*u
                        exact=d<=0;approx=not exact and B<1e-12
                        near.append(dict(segment=which[0],command_v=float(A),u=float(u),upper_log_drive=d,
                            exact_inactive=exact,bounded_numerical_omission=approx,local_integral_bound=B,out_of_domain_current_evaluated=False))
                        if exact or approx:return 0.
                        raise NumericalUnknown('Subtable positive command not covered by inactive or bounded omission certificate')
                    i,vh,kcl=series(A,u);rhs_kcl=max(rhs_kcl,abs(kcl))
                    return 0. if i==0 else -kappa*vf*math.sinh(abs(vh)/vf)*max(math.log(i/js)+beta*u,0.)
                def integrate(name,span,u0,command):
                    which[0]=name;row=dict(name=name,span=list(span),initial_u=float(u0),status='running');run['segments'].append(row)
                    sol=solve_ivp(lambda q,y:[rate(command(q),float(y[0]))],span,[u0],method='DOP853',rtol=precision['rtol'],atol=precision['atol'],max_step=precision['max_step'],dense_output=True)
                    row.update(status='completed' if sol.success else 'solver_failed',success=bool(sol.success),message=str(sol.message),
                        nfev=int(sol.nfev),accepted_steps=len(sol.t)-1,final_u=float(sol.y[0,-1]))
                    arrays[prefix+'_'+name+'_coordinate']=sol.t;arrays[prefix+'_'+name+'_state_u']=sol.y[0]
                    if not sol.success:raise RuntimeError(sol.message)
                    return sol
                try:
                    back=integrate('backward',(.2,0.),anchor,lambda A:A);u0=float(back.y[0,-1]);run['derived_initial_u']=u0
                    outward=integrate('outward',(0.,5.),u0,lambda q:q)
                    returned=integrate('return',(5.,10.),float(outward.y[0,-1]),lambda q:10.-q)
                    u=np.empty_like(v);u[out]=outward.sol(s[out])[0];u[~out]=returned.sol(s[~out])[0]
                    assert np.max(np.diff(u))<=5e-7
                    values=np.array([series(A,state) for A,state in zip(v,u)]);cur,vh,kcl=values.T
                    assert np.all(np.isfinite(cur)) and np.all(cur>=0) and np.all(cur[mask]>0)
                    groups=dict(all500=mask,outward250=mask&out,return250=mask&~out,outward_low10=mask&out&(v<=.2+1e-12),
                        both_branches_low20=mask&(v<=.2+1e-12),remaining480=mask&(v>.2+1e-12),outward0p5_to1p5_50=mask&out&(v>.5+1e-12)&(v<=1.5+1e-12))
                    metrics={}
                    for name,m in groups.items():
                        e=np.log10(cur[m]/obs[m]);metrics[name]=dict(n=int(m.sum()),RMSE_decade=float(np.sqrt(np.mean(e*e))),
                            mean_signed_log_error_decade=float(np.mean(e)),median_absolute_factor=float(10**np.median(abs(e))))
                    refs={}
                    for name,f in feature_ref.items():
                        jj=f['native_indices'];value=float(cur[jj[0]]) if len(jj)==1 else float(math.log(cur[jj[1]]/cur[jj[0]])/math.log(v[jj[1]]/v[jj[0]]))
                        l,h=f['original32_aggregate_band'];refs[name]=dict(single_cycle_prediction=value,original32_aggregate_band=[l,h],native_indices=jj,
                            single_prediction_inside_reference_band=l<=value<=h,scope='Reference comparison only,not32-record aggregate success')
                    actual=[]
                    for name,A,branch,kind in [('outward_2V_lower',2.,out,'at_least'),('return_5V_lower',5.,~out,'at_least'),('return_2V_upper',2.,~out,'at_most')]:
                        j=int(np.argmin(np.where(branch,abs(v-A),np.inf)));assert v[j]==A
                        limit=anchors[name]['current_a'];value=float(cur[j]);actual.append(dict(name=name,native_index=j,predicted_current_a=value,
                            bound_current_a=limit,direction=kind,actual_u=float(u[j]),bound_u=float(math.log1p(gc/anchors[name]['rederived_gamma_ev'])),
                            condition_satisfied=value>=limit if kind=='at_least' else value<=limit))
                    crossings=[]
                    for threshold in plan['outward_current_thresholds_a']:
                        item=dict(threshold_a=threshold)
                        for name,trace in [('observed',obs),('predicted',cur)]:
                            hits=np.flatnonzero(out&mask&(trace>=threshold))
                            item[name]=dict(reached=False) if not len(hits) else dict(reached=True,last_below_v=float(v[hits[0]-1]),first_at_or_above_v=float(v[hits[0]]))
                        crossings.append(item)
                    ja=int(np.argmin(np.where(out,abs(v-.2),np.inf)))
                    i0,v0=t.series(v,np.array([gamma(float(x)) for x in u]),ron)
                    run.update(status='completed_fixed_sinh_single_cycle',metrics=metrics,reference_features=refs,actual_budget_endpoints=actual,
                        all_three_actual_endpoints_satisfied=all(x['condition_satisfied'] for x in actual),outward_current_crossings=crossings,
                        terminal_u=float(u[-1]),forward_anchor_state_error=float(u[ja]-anchor),forward_anchor_current_over_observed=float(cur[ja]/obs[ja]),
                        max_native_KCL_residual_a=float(max(abs(kcl))),max_rhs_KCL_residual_a=rhs_kcl,
                        max_native_log_current_difference_vs_legacy=float(max(abs(np.log10(cur[mask]/i0[mask])))),
                        max_native_local_voltage_difference_vs_legacy=float(max(abs(vh-v0))))
                    arrays[prefix+'_current_a']=cur;arrays[prefix+'_state_u']=u;arrays[prefix+'_local_voltage_v']=vh;arrays[prefix+'_KCL_residual_a']=kcl
                    np.testing.assert_allclose(np.linalg.norm(np.log10(cur[mask])-np.log10(obs[mask]))/math.sqrt(500),metrics['all500']['RMSE_decade'],rtol=1e-12)
                    run['arithmetic_audit_passed']=True
                except Exception as exc:
                    run.update(status='failed_preserved',error_type=type(exc).__name__,error=str(exc),last_rhs_evaluation=last,
                        failure_scope='numerical_unknown_not_physical_rejection' if isinstance(exc,NumericalUnknown) else 'specified_candidate_or_solver_failure')
                    if run['segments'] and run['segments'][-1]['status']=='running':run['segments'][-1]['status']='exception_before_completion'
                    if isinstance(exc,TimeoutError):raise
                finally:
                    segments=sorted({x['segment'] for x in near if x['bounded_numerical_omission']})
                    run.update(elapsed_s=time.monotonic()-clock,n_rhs_calls=calls,smallfield_certificate=dict(RHS_test_count=len(near),
                        exact_inactive_count=sum(x['exact_inactive'] for x in near),bounded_omission_count=sum(x['bounded_numerical_omission'] for x in near),
                        omission_physical_segments=segments,local_integral_bound_per_crossing=B,sum_local_bounds=len(segments)*B,
                        scope='RHS retries do not multiply physical path;local bound not a global error certificate.',evaluations=near))
            if 'nominal_current_a' in arrays and 'refined_current_a' in arrays:
                report['precision_comparison']=dict(max_native_log_current_change=float(max(abs(np.log10(arrays['nominal_current_a'][mask]/arrays['refined_current_a'][mask])))),
                    max_native_state_change=float(max(abs(arrays['nominal_state_u']-arrays['refined_state_u']))),terminal_state_change=float(arrays['nominal_state_u'][-1]-arrays['refined_state_u'][-1]))
            report['status']='completed_single_fixed_sinh_scale_probe_all_failures_retained'
        unchanged={p:sha(p)==h for p,h in guards.items()};assert all(unchanged.values());report['frozen_baseline_files_unchanged']=unchanged
    except Exception as exc:report.update(status='partial_or_failed_sinh_probe_preserved_no_restart',error_type=type(exc).__name__,error=str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0.);report['compute_elapsed_s']=time.monotonic()-begun
    report['n_completed_runs']=sum(x['status']=='completed_fixed_sinh_single_cycle' for x in report['runs']);report['n_failed_runs']=sum(x['status']=='failed_preserved' for x in report['runs'])
    dest.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');np.savez_compressed(dest.with_suffix('.npz'),**arrays)
    lines=['# 单一固定sinh场迁移率：探索性条件SET探针','',
        'M(v)=Vf sinh(v/Vf)，Vf固定0.5V，低场归一化M(v)/v→1。这新增一个固定结构尺度，是看到既有外扫残差后的探索性选择，不是测得的跳跃长度/势垒，也不能称“没有新参数”。没有拟合、尺度/速率扫描或保留集选择。',
        '',f'状态：{report["status"]}；计算{report["compute_elapsed_s"]:.6f}s，预声明上限120s。']
    if 'necessary_screen' in report:
        b=report['necessary_screen'];lines+=['',f'64格必要κ下界={b["kappa_lower_per_v2"]}；唯一κ=2κlo={b["derived_unique_kappa_per_v2"]} V⁻²；晚κ上界={b["kappa_upper_per_v2"] if b["kappa_upper_per_v2"] is not None else "∞"}。必要区间未排除不代表实际到达。']
    lines+=['', '| 精度 | 状态 | 全500 RMSE | 外扫低10 | 其余480 | 返扫250 | 外扫(.5,1.5]50点 |','|---|---|---:|---:|---:|---:|---:|']
    for r in report['runs']:
        m=r.get('metrics');cells=[f'{m[k]["RMSE_decade"]:.8f}' for k in ('all500','outward_low10','remaining480','return250','outward0p5_to1p5_50')] if m else ['—']*5
        lines.append('| '+' | '.join([r['precision']['name'],r['status'],*cells])+' |')
        if m:
            lines+=['',f'### {r["precision"]["name"]}实际端点','', '| 端点 | 实际/µA | 原界/µA | 实际u | 满足 |','|---|---:|---:|---:|---|']
            for x in r['actual_budget_endpoints']:lines.append(f'| {x["name"]} | {x["predicted_current_a"]*1e6:.8f} | {x["bound_current_a"]*1e6:.8f} | {x["actual_u"]:.9g} | {x["condition_satisfied"]} |')
            lines+=['', '| 返扫参考特征 | 单圈预测 | 原32圈聚合带 | 单值在带内 |','|---|---:|---|---|']
            for name,f in r['reference_features'].items():lines.append(f'| {name} | {f["single_cycle_prediction"]:.9g} | {f["original32_aggregate_band"]} | {f["single_prediction_inside_reference_band"]} |')
            for x in r['outward_current_crossings']:lines+=['',f'{x["threshold_a"]:g}A原生首次跨越：观测{x["observed"]}；预测{x["predicted"]}。']
    if 'precision_comparison' in report:
        c=report['precision_comparison'];lines+=['',f'两精度原生电流最大log差{c["max_native_log_current_change"]:.6g}dec，全部状态最大差{c["max_native_state_change"]:.6g}，末态差{c["terminal_state_change"]:.6g}。']
    if 'smallfield_bound' in report:
        b=report['smallfield_bound'];lines+=['',f'近零使用稳定表达式2κDmaxVf²sinh²(vmin/(2Vf))，单调微段局部遗漏上界{b["local_integral_upper_per_monotone_segment"]:.9g}。精确inactive与有界数值忽略分开记录，未赋域外电流、未夹状态/电压，局部界不冒充全程误差界。']
    lines+=['', '同一0.2V条件锚用于反求初态并纳入500点评分；未删低压点。这不是独立预测或真实初态识别。三个预算端点与六个单圈参考都不是32圈聚合门槛通过。正迁移率不移动D=0停止面，本探针不声明修复返扫形状、RESET、bare1R历史或完整统一模型。',
        '', '所有既有基线及驱动诊断文件在运行前后hash核查不变；本候选只有固定Vf=.5V与由既定规则产生的一项κ，失败也保留，没有结果出来后的调参或追加种子。']
    dest.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=report['status'],compute_elapsed_s=report['compute_elapsed_s'],necessary_screen=report.get('necessary_screen'),
        completed=report['n_completed_runs'],failed=report['n_failed_runs'],precision_comparison=report.get('precision_comparison'),
        runs=[{k:x[k] for k in ('precision','status','metrics','actual_budget_endpoints','outward_current_crossings','error') if k in x} for x in report['runs']]),indent=2))


if __name__=='__main__':main()
