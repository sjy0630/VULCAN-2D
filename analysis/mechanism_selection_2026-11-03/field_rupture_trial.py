"""Predeclared conditional bare-RESET field-rupture trial, not a joint model.

Fourteen original training cycles only. a0 is conditioned on prior SET read
conductance (or a training-fold-only constant); all three fitted parameters are
global within a fit. Exact command-path solution, no ODE or SET prediction.
"""
from pathlib import Path
import hashlib
import json
import math
import signal
import sys
import time

import numpy as np
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
SPECTRAL=ROOT/'analysis/spectral_dynamic_2026-11-03'
sys.path.insert(0,str(SPECTRAL))
from evaluate_initiation import predicted_feature

IDS=[1,2,3,5,6,7,8,9,10,11,12,13,14,15]
NAMES=['Vr_v','log10_kappa_per_v2','log10_Gamma_leak_ev']
LOW=np.array([0.,-1.,-10.]);HIGH=np.array([1.,4.,-4.])
STARTS=np.array([[.5,1.,-6.3],[.5,2.,-6.3],[.5,3.,-6.3]])
RON=58.463033350725695


def dose(voltage,vr):
    v=abs(np.asarray(voltage));apex=int(np.argmax(v));maximum=float(v[apex])
    result=.5*np.maximum(v-vr,0.)**2
    result[apex+1:]=max(maximum-vr,0.)**2-.5*np.maximum(v[apex+1:]-vr,0.)**2
    return result


class Background:
    """Only interpolate existing fixed-shape material currents; scale byR0/R."""
    def __init__(self,table_path,shape_path,voltage):
        with np.load(table_path) as z:
            self.lg=z['log_gamma'].copy();self.lv=z['log_voltage'].copy();self.li=z['log_current'].copy()
        payload=json.loads(shape_path.read_text());self.shape=payload.get('parameters',payload)
        self.log_factor=math.log(self.shape['ron_ohm']/RON)
        self.v=abs(np.asarray(voltage));self.logv=np.full(len(self.v),-np.inf)
        np.log(self.v,out=self.logv,where=self.v>0)
        self.iy=np.clip(np.searchsorted(self.lv,np.log(np.maximum(self.v,np.exp(self.lv[0]))))-1,0,len(self.lv)-2)
        self.fv=(np.log(np.maximum(self.v,np.exp(self.lv[0])))-self.lv[self.iy])/(self.lv[self.iy+1]-self.lv[self.iy])
        self.js=int(np.searchsorted(self.lv,math.log(.05))-1)
        self.fs=(math.log(.05)-self.lv[self.js])/(self.lv[self.js+1]-self.lv[self.js])
        assert self.lg[0]<=-10.*math.log(10) and self.lg[-1]>=-4.*math.log(10)
        assert np.all(self.v<=math.exp(self.lv[-1])+1e-12)

    def evaluate(self,log10_gamma):
        x=float(log10_gamma)*math.log(10.)
        j=int(np.clip(np.searchsorted(self.lg,x)-1,0,len(self.lg)-2))
        f=(x-self.lg[j])/(self.lg[j+1]-self.lg[j])
        profile=(1-f)*self.li[1,j]+f*self.li[1,j+1]
        log_i=(1-self.fv)*profile[self.iy]+self.fv*profile[self.iy+1]+self.log_factor
        log_i=np.where(self.v>0,log_i,-np.inf)
        positive=(1-f)*self.li[0,j]+f*self.li[0,j+1]
        log_set=(1-self.fs)*positive[self.js]+self.fs*positive[self.js+1]+self.log_factor
        return log_i,math.exp(float(log_set))/.05


def analytic_checks(voltage):
    v=abs(voltage);maximum=float(max(v));ap=int(np.argmax(v))
    s=np.where(np.arange(len(v))<=ap,v,2*maximum-v)
    rows=[]
    for vr in (0.,.5,1.):
        # Independent integration of a triangular command's positive-part
        # drive. Include all kink positions; trapezoids are then exact for
        # piecewise-linear drive, without using the closed-dose formulas.
        points=np.unique(np.r_[s,0.,maximum,2*maximum,vr,2*maximum-vr])
        triangular=np.minimum(points,2*maximum-points)
        drive=np.maximum(triangular-vr,0.)
        integrated=np.r_[0.,np.cumsum(.5*(drive[:-1]+drive[1:])*np.diff(points))]
        independent=np.interp(s,points,integrated)
        exact=dose(voltage,vr)
        err=float(np.max(abs(independent-exact)))
        assert err<1e-12 and np.min(np.diff(exact))>=-1e-12
        for logk in (-1.,1.,2.,3.,4.):
            k=10.**logk;a=np.exp(-k*exact)
            assert np.all(np.isfinite(a)) and np.all(a>=0) and np.max(np.diff(a))<=1e-12
            outward_at_peak=.5*max(maximum-vr,0.)**2
            return_at_peak=max(maximum-vr,0.)**2-.5*max(maximum-vr,0.)**2
            assert outward_at_peak==return_at_peak
        rows.append(dict(Vr_v=vr,max_dose_difference_vs_independent_integral_v2=err,
                         dose_nondecreasing=True,path_conductance_continuous_nondecreasing_loss=True))
    return dict(checks=rows,optional_ODE_check_performed=False,
        path_only_outward_peak_formula='min(Vmax,(Vr+sqrt(Vr^2+4/kappa))/2); not the total-current peak with background',
        return_path_property='d(a*v)/dv=a*(1+kappa*v*(v-Vr)_+)>0; hence path current decreases along the return scan',
        independent_review='field_law_check subagent checked dose units/continuity, path-only peak and LOO design without fitting')


def metrics(predicted,observed,mask,outward,ids):
    if predicted.shape!=observed.shape or np.any(~np.isfinite(predicted)) or np.any(predicted<0):
        raise ValueError('Negative, nonfinite or wrongly shaped prediction retained as failure')
    groups={}
    for name,m in {'all_eligible':mask,'outward':mask&outward[None,:],
                   'return':mask&~outward[None,:]}.items():
        if np.any(predicted[m]<=0):raise ValueError('Nonpositive prediction at eligible observation')
        e=np.log10(predicted[m])-np.log10(observed[m])
        groups[name]=dict(n=int(m.sum()),raw_RMSE_decade=float(np.sqrt(np.mean(e*e))),
            median_absolute_factor_error=float(10.**np.median(abs(e))),
            factor_from_RMSE=float(10.**np.sqrt(np.mean(e*e))),mean_signed_log_error_decade=float(np.mean(e)))
    groups['per_record']=[dict(record=cid,n=int(m.sum()),raw_RMSE_decade=float(np.sqrt(np.mean((np.log10(p[m])-np.log10(o[m]))**2))))
                          for cid,p,o,m in zip(ids,predicted,observed,mask)]
    return groups


def feature_report(predicted,voltage,features,selected,ids):
    result={}
    for name in selected:
        definition=features[name];low,high=definition['proposed_empirical_acceptance_interval']
        rows=[]
        for cid,current in zip(ids,predicted):
            try:
                value,detail=predicted_feature(name,definition,voltage,current)
                if not np.isfinite(value):raise ValueError('Nonfinite feature')
                rows.append(dict(record=cid,value=float(value),valid=True,**detail))
            except ValueError as exc:rows.append(dict(record=cid,value=None,valid=False,error=str(exc)))
        values=[x['value'] for x in rows if x['valid']]
        median=float(np.median(values)) if len(values)==len(rows) else None
        result[name]=dict(model_median=median,original_aggregate_interval=[low,high],
            aggregate_gate_passed=bool(median is not None and low<=median<=high),record_values=rows,
            n_record_values_inside_aggregate_interval=sum(low<=x<=high for x in values),
            per_record_count_scope='Descriptive only; original gate is the across-record median, not fourteen separate hard gates.')
    return dict(n_original_aggregate_gates=4,n_original_aggregate_gates_passed=sum(x['aggregate_gate_passed'] for x in result.values()),features=result)


def main():
    main_started=time.monotonic()
    raw_path=ROOT/'analysis/shared_hbn_2026-11-03/data/native_observations.npz'
    table_path=SPECTRAL/'transport_g010_v0025.npz'
    shape_path=ROOT/'analysis/constrained_shared_2026-11-03/lognormal_spectral_refinement.json'
    constraints_path=ROOT/'analysis/constrained_shared_2026-11-03/feature_constraints.json'
    calibration_path=ROOT/'analysis/ron_scope_2026-11-03/RON_CALIBRATION_SCOPE.json'
    constraints=json.loads(constraints_path.read_text());calibration=json.loads(calibration_path.read_text())
    assert calibration['predeclared_Ron_sensitivity_points'][1]['Ron_ohm']==RON
    selected=[k for k in constraints['proposed_success_criteria_before_next_fit']['primary_feature_ids']
              if constraints['features'][k]['topology']=='1r' and constraints['features'][k]['phase']=='reset']
    assert len(selected)==4
    observed=[];set_g=[];set_reads=[];loaded_keys=[];loaded_hash=hashlib.sha256();voltage=None
    with np.load(raw_path) as z:
        for cid in IDS:
            for phase in ('set','reset'):
                prefix=f'1r_cycle_{cid}_{phase}';v=z[prefix+'_voltage'].copy();i=z[prefix+'_current'].copy()
                for suffix,value in (('voltage',v),('current',i)):
                    key=prefix+'_'+suffix;loaded_keys.append(key);loaded_hash.update(key.encode());loaded_hash.update(value.tobytes())
                if phase=='set':
                    ap=int(np.argmax(abs(v)));j=int(np.argmin(np.where(np.arange(len(v))>ap,abs(abs(v)-.05),np.inf)))
                    assert abs(abs(v[j])-.05)<1e-7 and 0<abs(i[j])<.995e-3
                    g=abs(float(i[j]/v[j]));set_g.append(g)
                    set_reads.append(dict(record=cid,voltage_v=float(v[j]),current_a=float(i[j]),conductance_s=g,native_index=j))
                else:
                    if voltage is None:voltage=v
                    else:assert np.array_equal(v,voltage)
                    observed.append(abs(i))
    observed=np.array(observed);set_g=np.array(set_g)
    assert voltage[0]==voltage[-1]==0 and abs(min(voltage))==2.5 and max(voltage)==0
    mask=np.isfinite(observed)&(observed>=1e-12)&(abs(voltage)[None,:]>1e-12)
    assert int(mask.sum())==2799
    outward=np.arange(len(voltage))<=np.argmax(abs(voltage))
    obs_log=np.zeros_like(observed);np.log10(observed,out=obs_log,where=mask)
    background=Background(table_path,shape_path,voltage)
    checks=analytic_checks(voltage)
    plan=dict(model='I_h=Fsoft(V,Gamma_leak;frozen shape,common Ron)+aV',Ron_ohm=RON,
        decay='da/ds=-kappa*(abs(Vh)-Vr)_+*a; a=a0*exp(-kappa*D), s in command-path volts',
        background='One group-common Gamma_leak, fixed during RESET; effective background transport approximation, no species assigned',
        parameter_names=NAMES,lower=LOW.tolist(),upper=HIGH.tolist(),fixed_starts=STARTS.tolist(),
        max_nfev_per_start=100,least_squares_method='trf',ftol=1e-8,xtol=1e-8,gtol=1e-8,
        fitting_objective='Unweighted original raw log10 RESET residuals on observed absI>=1e-12 and V!=0; no residual floor, denominator change, branch rescaling or robust loss',
        training_ids=IDS,n_original_RESET_eligible_points=int(mask.sum()),models=['prior_SET_conditioned','fold_training_SET_geomean_constant'],
        conditional_a0='max(own preceding SETreturn50mV observed conductance-Fsoft(+.05,Gamma_leak)/.05,0)',
        constant_a0='max(geomean(fit-fold training SET conductances)-Fsoft(+.05,Gamma_leak)/.05,0); exclude test SET from this constant',
        evaluations='For each model one full-training fit plus14LOO refits, each starts from all3 fixed starts; no full-data initialization for LOO',
        maximum_fit_calls=90,compute_budget_s=120,domain_expansion=False,
        failure_policy='Retain numerical/solver stops; incomplete folds are not dropped from a claimed full LOO metric.')
    plan_path=HERE/'field_rupture_trial.plan.json'
    plan_path.write_text(json.dumps(plan,indent=2,allow_nan=False)+'\n')
    source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (table_path,shape_path,constraints_path,calibration_path,Path(__file__),plan_path,SPECTRAL/'evaluate_initiation.py')}
    report=dict(status='running_predeclared_conditional_RESET_trial',predeclared_plan=plan,
        prior_SET_reads=set_reads,analytic_solution_checks=checks,models={},fit_calls=0,
        loaded_native_array_keys=loaded_keys,loaded_training_arrays_sha256=loaded_hash.hexdigest(),
        source_sha256=source_hashes,
        scope=['This predicts RESET conditional on preceding measured SET read conductance; it does not predict SET, path formation or series behavior.',
               'LOO is internal conditional prediction among original14 training cycles, not the original heldout set or chronological future validation. Other training cycles may be chronologically later.',
               'a0 is not a free per-RESET fitted parameter. Gamma_leak,Vr,kappa are common within each fit and independently refit using13RESET records perLOO fold.',
               'No waiting or seconds inferred; kappa has unitsV^-2. Bare local voltage equals commanded voltage in this submodel.',
               'Field-driven decay continues even at low current whenever |V|>Vr, so it has no positive-current-threshold tail barrier. Below Vr the path freezes.',
               'Original shape, material table and common Ron fixed; numerical table approximation remains. No ODE or new quadrature.',
               'Parameters hitting bounds or solver evaluation limits are reported, not treated as physical identification.',
               'This is not a completed unified1R/1T1R model, continuous full-history fit, or proof of a microscopic species.'])
    destination=HERE/'field_rupture_trial.json'
    destination.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    arrays=dict(record_ids=np.array(IDS),command_voltage_v=voltage,observed_reset_abs_current_a=observed,
                original_reset_mask=mask,prior_SET_conductance_s=set_g)

    def predict(theta,g_input):
        vr,logk,loggamma=np.asarray(theta,float);k=10.**logk;D=dose(voltage,vr)
        log_leak,leak_g=background.evaluate(loggamma)
        a0=np.maximum(np.asarray(g_input)-leak_g,0.)
        with np.errstate(divide='ignore',under='ignore'):
            log_a0=np.log(a0)
            log_path=log_a0[:,None]+background.logv[None,:]-k*D[None,:]
            log_i=np.logaddexp(log_leak[None,:],log_path)
            a=a0[:,None]*np.exp(-k*D[None,:])
            current=np.exp(log_i)
        if np.any(~np.isfinite(current)) or np.any(current<0) or np.any(~np.isfinite(a)) or np.any(a<0):
            raise ValueError('Negative/nonfinite current or path conductance')
        if np.min(np.diff(D)) < -1e-12 or np.max(np.diff(a,axis=1)) > 1e-12:
            raise ValueError('Analytic path violates monotone dose/loss')
        return dict(abs_current=current,log10_current=log_i/math.log(10.),a=a,a0=a0,D=D,
                    leak_abs_current=np.exp(log_leak),leak_set_conductance_s=leak_g,
                    initial_state_clipped=a0==0.,initial_read_total_g=a0+leak_g)

    def fit_group(model_name,train_indices,excluded):
        fold=dict(excluded_record=excluded,fit_records=[IDS[i] for i in train_indices],starts=[],status='running')
        constant=float(np.exp(np.mean(np.log(set_g[train_indices]))))
        ginput=set_g[train_indices] if model_name=='prior_SET_conditioned' else np.full(len(train_indices),constant)
        fold['fit_SET_geomean_conductance_s']=constant
        fold['constant_excludes_test_SET']=excluded is not None
        train_mask=mask[train_indices];observed_log=obs_log[train_indices]
        def residual(theta):
            p=predict(theta,ginput)
            r=p['log10_current'][train_mask]-observed_log[train_mask]
            if np.any(~np.isfinite(r)):raise ValueError('Nonfinite eligible raw log residual')
            return r
        for start_index,start in enumerate(STARTS):
            trial=dict(start_index=start_index,start=start.tolist(),status='running')
            fold['starts'].append(trial);report['fit_calls']+=1;clock=time.monotonic()
            try:
                solution=least_squares(residual,start,bounds=(LOW,HIGH),method='trf',
                    max_nfev=100,ftol=1e-8,xtol=1e-8,gtol=1e-8)
                if not np.all(np.isfinite(solution.x)) or not np.isfinite(solution.cost):
                    raise ValueError('Nonfinite fitted parameters or cost')
                distance=np.minimum(solution.x-LOW,HIGH-solution.x)
                trial.update(status='finite_bounded_result',parameters=solution.x.tolist(),
                    sum_squared_raw_log_residual=float(2*solution.cost),n_residuals=len(solution.fun),
                    raw_RMSE_decade=float(np.sqrt(2*solution.cost/len(solution.fun))),
                    nfev=int(solution.nfev),njev=int(solution.njev) if solution.njev is not None else None,
                    solver_status=int(solution.status),solver_message=str(solution.message),solver_success=bool(solution.success),
                    optimality=float(solution.optimality),active_mask=solution.active_mask.tolist(),
                    distance_to_parameter_boundary=distance.tolist(),
                    within_1e_5_of_any_parameter_boundary=bool(np.any(distance<=1e-5)),elapsed_s=time.monotonic()-clock)
            except Exception as exc:
                trial.update(status='failure_retained',error_type=type(exc).__name__,error=str(exc),elapsed_s=time.monotonic()-clock)
                if isinstance(exc,TimeoutError):raise
        valid=[r for r in fold['starts'] if r['status']=='finite_bounded_result']
        if valid:
            best=min(valid,key=lambda x:x['sum_squared_raw_log_residual'])
            fold.update(status='three_starts_attempted_best_training_objective_retained',
                best_start_index=best['start_index'],parameters=best['parameters'],best_solver_success=best['solver_success'],
                selection_uses_excluded_RESET=False,best_sum_squared_raw_log_residual=best['sum_squared_raw_log_residual'])
        else:fold['status']='all_three_starts_failed'
        return fold

    def store_prediction(prefix,theta,ginput):
        pred=predict(theta,ginput)
        arrays[prefix+'_parameters']=np.array(theta)
        for key in ('abs_current','a','a0','D','leak_abs_current','initial_state_clipped','initial_read_total_g'):
            arrays[prefix+'_'+key]=np.asarray(pred[key])
        return pred

    def timeout(*_):raise TimeoutError('120-second computation budget exhausted')
    compute_started=time.monotonic();signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,120.)
    try:
        all_indices=np.arange(len(IDS))
        for model_name in plan['models']:
            model=dict(status='running',full_training_fit=None,LOO_folds=[])
            report['models'][model_name]=model
            full=fit_group(model_name,all_indices,None);model['full_training_fit']=full
            if 'parameters' in full:
                ginput=set_g if model_name=='prior_SET_conditioned' else np.full(len(IDS),full['fit_SET_geomean_conductance_s'])
                pred=store_prediction(model_name+'_training',full['parameters'],ginput)
                model['training_metrics']=metrics(pred['abs_current'],observed,mask,outward,IDS)
                model['training_feature_gates']=feature_report(pred['abs_current'],voltage,constraints['features'],selected,IDS)
                model['training_initial_state_clipping']=dict(n=int(np.sum(pred['initial_state_clipped'])),
                    records=[IDS[i] for i in np.flatnonzero(pred['initial_state_clipped'])],
                    max_initial_read_conductance_error_s=float(np.max(abs(pred['initial_read_total_g']-ginput))))
            loo_pred=np.full_like(observed,np.nan);loo_a=np.full_like(observed,np.nan);loo_D=np.full_like(observed,np.nan)
            loo_leak=np.full_like(observed,np.nan);loo_theta=np.full((len(IDS),3),np.nan);loo_a0=np.full(len(IDS),np.nan)
            for test_index,cid in enumerate(IDS):
                train_indices=all_indices[all_indices!=test_index]
                fold=fit_group(model_name,train_indices,cid);model['LOO_folds'].append(fold)
                if 'parameters' not in fold:continue
                ginput=np.array([set_g[test_index] if model_name=='prior_SET_conditioned' else fold['fit_SET_geomean_conductance_s']])
                p=predict(fold['parameters'],ginput)
                loo_pred[test_index]=p['abs_current'][0];loo_a[test_index]=p['a'][0];loo_D[test_index]=p['D']
                loo_leak[test_index]=p['leak_abs_current'];loo_theta[test_index]=fold['parameters'];loo_a0[test_index]=p['a0'][0]
                fold['test_conditioning']=dict(used_test_SET=model_name=='prior_SET_conditioned',
                    input_conductance_s=float(ginput[0]),a0_s=float(p['a0'][0]),
                    clipped_to_zero=bool(p['initial_state_clipped'][0]),
                    initial_read_total_g_s=float(p['initial_read_total_g'][0]),
                    initial_read_error_vs_conditioning_g_s=float(p['initial_read_total_g'][0]-ginput[0]),
                    test_RESET_used_only_for_final_scoring=True)
                fold['excluded_record_metrics']=metrics(p['abs_current'],observed[test_index:test_index+1],mask[test_index:test_index+1],outward,[cid])
            arrays[model_name+'_LOO_abs_current']=loo_pred;arrays[model_name+'_LOO_a']=loo_a
            arrays[model_name+'_LOO_D']=loo_D;arrays[model_name+'_LOO_leak_abs_current']=loo_leak
            arrays[model_name+'_LOO_parameters']=loo_theta;arrays[model_name+'_LOO_a0']=loo_a0
            complete=np.all(np.isfinite(loo_pred)) and len(model['LOO_folds'])==14
            model['n_completed_LOO_predictions']=int(np.sum(np.all(np.isfinite(loo_pred),axis=1)))
            if complete:
                model['LOO_metrics']=metrics(loo_pred,observed,mask,outward,IDS)
                model['LOO_feature_gates']=feature_report(loo_pred,voltage,constraints['features'],selected,IDS)
                model['status']='completed_training_and14_internal_conditional_LOO_folds'
            else:model['status']='incomplete_LOO_no_partial_metric_disguised_as_full'
        report['status']='completed_predeclared_conditional_RESET_trial'
    except Exception as exc:
        report.update(status='partial_or_failed_trial_preserved',error_type=type(exc).__name__,error=str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0.);report['compute_elapsed_s']=time.monotonic()-compute_started
    all_runs=[s for m in report['models'].values() for f in ([m['full_training_fit']] if m.get('full_training_fit') else [])+m['LOO_folds'] for s in f['starts']]
    report['solver_summary']=dict(n_recorded_start_runs=len(all_runs),
        n_finite_results=sum(x['status']=='finite_bounded_result' for x in all_runs),
        n_evaluation_limit_stops=sum(x.get('solver_status')==0 for x in all_runs),
        n_solver_success=sum(x.get('solver_success',False) for x in all_runs),
        n_results_near_parameter_boundary=sum(x.get('within_1e_5_of_any_parameter_boundary',False) for x in all_runs))
    if all('LOO_metrics' in m for m in report['models'].values()) and len(report['models'])==2:
        c=report['models']['prior_SET_conditioned'];b=report['models']['fold_training_SET_geomean_constant']
        report['comparison']=dict(LOO_RMSE_conditioned_minus_constant={g:c['LOO_metrics'][g]['raw_RMSE_decade']-b['LOO_metrics'][g]['raw_RMSE_decade'] for g in ('all_eligible','outward','return')},
            interpretation='Difference measures the value of prior SET conditioning in this restricted RESET law, not a test of full state formation or unified topology transfer.')
    report['main_elapsed_including_input_output_preparation_s']=time.monotonic()-main_started
    report['budget_scope']='120-second alarm covers fitting and scoring only after all input loads and predeclared plan/source hashes; startup/module/file waiting is separately included in main elapsed where applicable.'
    destination.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(destination.with_suffix('.npz'),**arrays)
    lines=['# 场驱动局域高导路径RESET条件预测试验','',
        '只用原14条训练裸RESET及各自前相位SET返扫50mV读数。Ron固定58.46303335Ω，背景谱形固定，Γ_leak为全组共同参数且本段RESET内不变。这里只检验RESET，未预测SET、a的形成、完整跨圈历史或1T1R。',
        '', '`I_h=Fsoft(V,Γ_leak)+aV`；`a=a0 exp(-κD)`，外扫`D=0.5(|V|−Vr)_+²`，返扫`D=(Vmax−Vr)_+²−0.5(|V|−Vr)_+²`。κ单位V⁻²，不是秒率；无等待假设。只要|V|>Vr，低电流下也会继续损失，不存在旧电流阈值屏障。',
        '', '条件模型a0来自本圈前SET电导减共同背景；对照使用该折其他训练SET电导几何均值后再减背景。两者都仅拟合同3个全局参数，各折均从预声明3起点重新拟合，每起点最多100次评估。',
        '', '| 模型 | 评分 | 全RESET RMSE/dec | 外扫 | 返扫 | 全点中位倍数误差 | 原聚合门槛 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for name,m in report['models'].items():
        for split,key in (('训练','training'),('内部LOO','LOO')):
            if key+'_metrics' not in m:continue
            v=m[key+'_metrics'];f=m[key+'_feature_gates']
            lines.append(f'| {name} | {split} | {v["all_eligible"]["raw_RMSE_decade"]:.6g} | {v["outward"]["raw_RMSE_decade"]:.6g} | {v["return"]["raw_RMSE_decade"]:.6g} | {v["all_eligible"]["median_absolute_factor_error"]:.6g} | {f["n_original_aggregate_gates_passed"]}/4 |')
    lines+=['',f'原始mask共{int(mask.sum())}点，所有完整汇总保持该分母，不平均14个折的RMSE。中位倍数误差定义为10^median(|log10预测/实测|)。原4项门槛按14条预测特征的中位数判断；JSON另列逐条值，但不把其分母混入聚合通过数。',
        '', 'LOO是原训练集内部、给定前SET读态的条件预测，可使用时间上较后的其他训练圈，不是原保留集，也不是因果时序外推。任何误差降低仅支持此子模型中的前态信息价值，不能称统一模型已成功。',
        '',f'独立积分正部驱动验证解析D连续/非减，a非负连续/非增；未运行ODE。计算{report["compute_elapsed_s"]:.3f}s，上限120s。状态：{report["status"]}；已记录{report["solver_summary"]["n_recorded_start_runs"]}个起点，评估上限停止{report["solver_summary"]["n_evaluation_limit_stops"]}个，近参数边界结果{report["solver_summary"]["n_results_near_parameter_boundary"]}个。',
        '每折参数、三个起点目标值、求解器停止原因、边界距离、a0截零和输入读态误差均保留于JSON；全原生电流/a/D/背景轨迹见NPZ。负值/非有限或失败不会被静默删掉。']
    destination.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=report['status'],compute_elapsed_s=report['compute_elapsed_s'],solver_summary=report['solver_summary'],
        models={k:{j:v[j] for j in ('training_metrics','LOO_metrics') if j in v} for k,v in report['models'].items()},
        parameters={k:v['full_training_fit'].get('parameters') for k,v in report['models'].items() if v.get('full_training_fit')},
        gates={k:{j:v[j]['n_original_aggregate_gates_passed'] for j in ('training_feature_gates','LOO_feature_gates') if j in v} for k,v in report['models'].items()}),indent=2))


if __name__=='__main__':main()
