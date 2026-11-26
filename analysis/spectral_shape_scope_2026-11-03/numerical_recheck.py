"""One bounded numerical recheck of the original21 unresolved shape points.

No new shape, core edit, ODE, or overwrite of the frozen47-point screen.
Uses existing64/128 energy rules or the existing independent adaptive-Z
integrator.120s limit; unfinished/uncertified points remain numerical unknowns.
"""
from pathlib import Path
import sys,json,math,time,hashlib
import numpy as np
from scipy.optimize import brentq

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(OUT))
sys.path.insert(0,str(ROOT/'analysis/constrained_shared_2026-11-03'))
from vulcan2d.shared_spectral import SpectralShape
from vulcan2d.transistor import measured_transistor_mapping
from lognormal_spectral_screen import independent_z_integral
from shape_feasibility_screen import monotone_intervals,DATA,TRAIN_R

SOURCE=OUT/'shape_feasibility_screen.json'
MAX_SECONDS=120.


class Deadline(Exception):pass


def main():
    started=time.monotonic();original=json.loads(SOURCE.read_text());source_hash=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    unresolved=[x for x in original['results'] if x['status'] in ('invalid','numerically_invalid_refinement')]
    assert len(unresolved)==21
    mapping=measured_transistor_mapping();bare=[];loaded=hashlib.sha256()
    with np.load(DATA) as archive:
        for cid in TRAIN_R:
            prefix=f'1r_cycle_{cid}_set';v=archive[prefix+'_voltage'].copy();ii=archive[prefix+'_current'].copy()
            for label,a in [('voltage',v),('stored_current',ii)]:loaded.update((prefix+label).encode());loaded.update(a.tobytes())
            ap=int(np.argmax(v));bare.append((cid,v[:ap+1],abs(ii[:ap+1])))
    all_bare_voltage=np.unique(np.concatenate([x[1] for x in bare]))
    report={'status':'running_original21_numerical_recheck','max_wall_seconds':MAX_SECONDS,
        'original_screen_sha256':source_hash,'original_unresolved_names':[x['name'] for x in unresolved],
        'new_shape_points':0,'core_modified':False,'results':[],
        'scope':'Existing64/128 energy quadrature or existing independent adaptive-Z only; no altered physical conditions or gates.'}

    def check():
        if time.monotonic()-started>MAX_SECONDS:raise Deadline('120s numerical recheck budget reached')

    def write():
        report['elapsed_s']=time.monotonic()-started
        tmp=OUT/'numerical_recheck.json.tmp';tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(OUT/'numerical_recheck.json')

    for prior in unresolved:
        row={'name':prior['name'],'q_shape':prior['q_shape'],'original_error':prior.get('error'),'status':'started'}
        report['results'].append(row)
        try:
            check();qshape=np.array(prior['q_shape']);fixed=original['fixed']
            shape=SpectralShape(median_energy_ev=10**qshape[0],eta=qshape[1],sigma_log_energy=qshape[2],
                ron_ohm=fixed['ron_ohm'],temperature_k=fixed['temperature_k'])
            mass={};orders=[]
            for order in (64,128):
                try:mass[str(order)]=shape.quadrature_metadata(order);orders.append(order)
                except Exception as exc:mass[str(order)]={'error':type(exc).__name__+': '+str(exc)}
            row['probability_rule_checks']=mass
            selected=orders[0] if orders else 'adaptive_Z'

            def z_current(voltage,loggamma):
                check();q=np.array([loggamma/math.log(10.),*qshape])
                value,error=independent_z_integral(q,np.asarray(voltage,dtype=float),shape.ron_ohm)
                return value,float(error)

            def evaluate(backend):
                probes=[];cache={}
                def current(voltage,loggamma):
                    check();v=np.asarray(voltage,dtype=float);key=(v.shape,v.tobytes(),float(loggamma))
                    if key not in cache:
                        value=(z_current(v,loggamma)[0] if backend=='adaptive_Z' else
                               shape.current_at_log_gamma(v,loggamma,order=backend))
                        if not np.all(np.isfinite(value)) or np.any(value*v<0):raise FloatingPointError('Readout sign/domain failed')
                        cache[key]=value
                    return cache[key]
                def invert(command,i,phase):
                    local=float(np.sign(command)*(abs(command)-mapping.voltage_for_current(i,1.1,phase)))
                    if command*local<=0:raise ValueError('No passive internal voltage')
                    lg=brentq(lambda x:math.log(abs(float(current(local,x)))/i),-40.,20.,xtol=2e-10)
                    probes.append((local,lg,i));return math.exp(lg)
                intervals={}
                for target in original['targets']:
                    intervals[target['key']]=[invert(target['command_v'],i,target['phase']) for i in target['current_Q10_Q90_a']]
                ls=[intervals[f'set_{v:g}'] for v in [.3,.2,.1,.06]];lr=[intervals[f'reset_{v:g}'] for v in [.06,.1,.2,.3]]
                lsok,_=monotone_intervals(ls,True);lrok,_=monotone_intervals(lr,False)
                flo=max(x[0] for x in ls+lr);fhi=min(x[1] for x in ls+lr)
                low={'set_directional_interval_feasible':lsok,'reset_directional_interval_feasible':lrok,
                    'common_frozen_gamma_feasible':flo<=fhi,'common_frozen_gamma_intersection_ev':[flo,fhi],
                    'scope':'Frozen condition reported separately; not required for finite-rate classification.'}
                def terminal(a,gamma):
                    lg=math.log(gamma);vh=brentq(lambda v:float(current(v,lg))-mapping.current(a-v,1.1,'set'),0.,a,xtol=1e-11)
                    return float(current(vh,lg)),float(vh)
                scenarios=[]
                for expansion in (0.,.01,.05):
                    high={}
                    for v in (5.,4.,2.):
                        t=next(x for x in original['targets'] if x['key']==f'set_{v:g}')
                        if expansion and v in (5.,4.):
                            band=[t['current_Q10_Q90_a'][0]/(1+expansion),t['current_Q10_Q90_a'][1]*(1+expansion)]
                            high[v]=[invert(t['command_v'],i,'set') for i in band]
                        else:high[v]=intervals[t['key']]
                    highok,_=monotone_intervals([high[v] for v in (5.,4.,2.)],True)
                    pathok,_=monotone_intervals([high[v] for v in (5.,4.,2.)]+ls,True)
                    lo=min(high[v][0] for v in (5.,4.));hi=max(high[v][1] for v in (5.,4.))
                    _,vmin=terminal(4.,hi)
                    lowcurve=np.asarray(current(all_bare_voltage,math.log(lo)));highcurve=np.asarray(current(all_bare_voltage,math.log(hi)))
                    alphas=[];crossings=[]
                    for cid,v,i in bare:
                        indices=np.searchsorted(all_bare_voltage,v);a=lowcurve[indices];b=highcurve[indices]
                        before=np.flatnonzero(i<a);above=np.flatnonzero((i>=b)&(v>.2));pair=None
                        for k in above:
                            earlier=before[before<k]
                            if len(earlier):pair=int(earlier[-1]),int(k);break
                        if pair is None:crossings.append({'cycle':cid,'status':'no_bracket'});continue
                        left,right=pair;ordered=vmin>v[right];alpha=1/float(v[right]-v[left]) if ordered and highok else None
                        if alpha is not None:alphas.append(alpha)
                        crossings.append({'cycle':cid,'status':'bracket','start_v':float(v[left]),'end_v':float(v[right]),
                            'voltage_ordered':bool(ordered),'alpha_lower':alpha})
                    bound=max(alphas) if alphas else None
                    scenarios.append({'current_expansion_fraction':expansion,'high_directional_feasible':highok,
                        'selected_SET_path_directional_feasible':pathok,'gamma_band_ev':[lo,hi],'soft_local_min_v':vmin,
                        'n_valid_time_comparisons':len(alphas),'max_necessary_alpha':bound,
                        'common_path_not_rejected_by_selected_conditions':bool(pathok and lrok and ((not alphas) or bound<=1+1e-5)),
                        'also_passes_conditional_low_frozen_state':bool(pathok and lrok and ((not alphas) or bound<=1+1e-5) and flo<=fhi),
                        'bare_crossings':crossings})
                # Existing independent Z integration audits every inverted
                # endpoint when it is not already the primary backend.
                differences=[];relative_errors=[];coarse_refine=[]
                for v,lg,i in probes:
                    check();value=float(current(v,lg))
                    independent,error=z_current(v,lg)
                    independent=float(independent)
                    if independent*v<=0:raise FloatingPointError('Independent signed current failed')
                    differences.append(abs(math.log10(abs(value/independent))))
                    relative_errors.append(error/abs(independent))
                    if backend==64 and 128 in orders:
                        finer=float(shape.current_at_log_gamma(v,lg,order=128))
                        coarse_refine.append(abs(math.log10(abs(value/finer))))
                numerical={'backend':backend,'n_endpoints_checked':len(probes),
                    'max_vs_existing_adaptive_Z_log_difference_decade':max(differences),
                    'max_adaptive_Z_relative_error_estimate':max(relative_errors),
                    'order64_to128_max_change_decade':max(coarse_refine) if coarse_refine else None,
                    'scope':'Finite selected endpoint oracle states; adaptive error estimates, not a rigorous all-state certificate.'}
                passed=(max(differences)<=1e-5 and max(relative_errors)<=1e-5 and (not coarse_refine or max(coarse_refine)<=1e-5))
                return {'low_bias':low,'scenarios':scenarios,'numerical':numerical,'numerical_passed':passed}

            try:
                result=evaluate(selected)
                if not result['numerical_passed'] and selected==64:
                    row['first_attempt']=result['numerical'];result=evaluate(128 if 128 in orders else 'adaptive_Z')
            except Deadline:raise
            except Exception as exc:
                if selected=='adaptive_Z':raise
                row['first_attempt_error']=type(exc).__name__+': '+str(exc)
                result=evaluate('adaptive_Z')
            row.update(result,status='resolved_existing_integration' if result['numerical_passed'] else 'numerical_unknown_after_recheck')
        except Deadline as exc:
            row.update(status='numerical_unknown_budget',error=str(exc));report['status']='wall_budget_terminated';write();break
        except Exception as exc:
            row.update(status='numerical_unknown_after_recheck',error=type(exc).__name__+': '+str(exc))
        write();print(json.dumps({'name':row['name'],'status':row['status'],
            'backend':row.get('numerical',{}).get('backend'),
            'candidate_by_scenario':[x['common_path_not_rejected_by_selected_conditions'] for x in row.get('scenarios',[])]}),flush=True)
    else:report['status']='completed_original21_recheck'
    resolved=[x for x in report['results'] if x['status']=='resolved_existing_integration']
    candidates=[]
    for row in resolved:
        for case in row['scenarios']:
            if case['common_path_not_rejected_by_selected_conditions']:
                candidates.append({'name':row['name'],'q_shape':row['q_shape'],'expansion':case['current_expansion_fraction'],
                    'low_frozen_also_passes':row['low_bias']['common_frozen_gamma_feasible'],
                    'scope':'Only not rejected by selected necessary conditions; not a dynamic result.'})
    report['summary']={'n_original_unknown':21,'n_attempted':len(report['results']),'n_resolved':len(resolved),
        'n_still_numerically_unknown':21-len(resolved),'not_rejected_candidates':candidates}
    report['original_screen_unchanged']=hashlib.sha256(SOURCE.read_bytes()).hexdigest()==source_hash
    report['loaded_training_arrays_sha256']=loaded.hexdigest()
    report['source_sha256']={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in
        [Path(__file__),SOURCE,ROOT/'vulcan2d/shared_spectral.py',ROOT/'analysis/constrained_shared_2026-11-03/lognormal_spectral_screen.py']}
    write()
    lines=['# 原21个数值未知点：一次有上限的积分复核','',
       '原47点结果保持冻结，没有新增形状、修改core、运行ODE或更改验收门槛。只使用已有64/128阶规则和既有独立Z积分，120秒上限。','',
       f'状态：{report["status"]}；已尝试{len(report["results"])}点；数值可用{len(resolved)}点；仍未知{21-len(resolved)}点。',
       f'原结果哈希保持不变：{report["original_screen_unchanged"]}。','',
       '| 点 | 状态 | 使用已有积分 | 所选必要条件未否决情景 |','|---|---|---|---|']
    for row in report['results']:
        eligible=[s['current_expansion_fraction'] for s in row.get('scenarios',[]) if s['common_path_not_rejected_by_selected_conditions']]
        lines.append(f'| {row["name"]} | {row["status"]} | {row.get("numerical",{}).get("backend", "未完成")} | {eligible if row["status"]=="resolved_existing_integration" else "数值未知"} |')
    lines+=['','通过数值复核只恢复了这些既有形状点的可计算性。未知点不计物理失败；未被必要条件否决也不等于通过四相动态验证。',
       'adaptive_Z后端的证据是已有自适应积分误差估计及已核对的单能级公式；它不自动认证生产全状态插值域，后续若要实际模型仍须单独核验。',
       '此次复核到此停止，不扩张形状域或引入新积分器。详细误差、失败和原始结果哈希在JSON。']
    (OUT/'numerical_recheck.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report['summary']),flush=True)


if __name__=='__main__':main()
