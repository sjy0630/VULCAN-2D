"""Bounded training-only screen of EXISTING common lognormal shape parameters.

At most47 predeclared shapes /300s, no ODE or new material parameter.
Gamma inversions are oracle necessary-condition diagnostics, not predictions.
Low-window freezing is a separately marked stronger condition, never silently
required to reject the larger finite-rate family. Original gates are unchanged.
"""
from pathlib import Path
import sys,json,math,time,hashlib,itertools
import numpy as np
from scipy.optimize import brentq
from scipy.stats import qmc

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from vulcan2d.shared_spectral import SpectralShape
from vulcan2d.transistor import measured_transistor_mapping

OUT=Path(__file__).resolve().parent
BASE=ROOT/'analysis/constrained_shared_2026-11-03/lognormal_spectral_refinement.json'
GLOBAL=ROOT/'analysis/constrained_shared_2026-11-03/lognormal_spectral_screen.json'
DATA=ROOT/'analysis/shared_hbn_2026-11-03/data/native_observations.npz'
TRAIN_R=[1,2,3,5,6,7,8,9,10,11,12,13,14,15]
TRAIN_T=list(range(4,36))
MAX_SECONDS=300.
MAX_SHAPES=47


class Deadline(Exception):pass


def monotone_intervals(intervals,increasing=True):
    if not len(intervals):return False,[]
    path=[];last=-math.inf if increasing else math.inf
    for lo,hi in intervals:
        chosen=max(lo,last) if increasing else min(hi,last)
        if chosen>hi or chosen<lo:return False,path
        path.append(chosen);last=chosen
    return True,path


def main():
    start=time.monotonic();base=json.loads(BASE.read_text());global_=json.loads(GLOBAL.read_text())
    p=base['parameters'];mapping=measured_transistor_mapping()
    local_lo=np.array(base['algorithmic_local_bounds_not_physical_confidence']['lower'])[1:]
    local_hi=np.array(base['algorithmic_local_bounds_not_physical_confidence']['upper'])[1:]
    global_lo=np.array(global_['bounds']['lower'])[1:];global_hi=np.array(global_['bounds']['upper'])[1:]
    center=np.array(base['selected_q'])[1:];points=[('selected_baseline',center)]
    for dim in range(3):
        for side,edge in [('lo',local_lo),('hi',local_hi)]:
            q=center.copy();q[dim]=edge[dim];points.append((f'local_axis_{dim}_{side}',q))
    for bits in itertools.product((0,1),repeat=3):
        points.append(('local_corner_'+''.join(map(str,bits)),np.where(bits,local_hi,local_lo)))
    for index,x in enumerate(qmc.Sobol(3,scramble=True,seed=20261103).random_base2(5)):
        points.append((f'global_stratum_{index:02d}',global_lo+x*(global_hi-global_lo)))
    assert len(points)==MAX_SHAPES
    native={};bare=[];loaded=hashlib.sha256()
    with np.load(DATA) as archive:
        for phase in ('set','reset'):
            vs=[];ys=[]
            for cid in TRAIN_T:
                prefix=f'1t1r_cycle_{cid}_{phase}';v=archive[prefix+'_voltage'].copy();i=archive[prefix+'_current'].copy()
                for suffix,a in [('voltage',v),('stored_current',i)]:loaded.update((prefix+suffix).encode());loaded.update(a.tobytes())
                vs.append(v);ys.append(abs(i))
            if not all(np.array_equal(vs[0],x) for x in vs):raise ValueError('Different native grids')
            native[phase]=(vs[0],np.array(ys))
        for cid in TRAIN_R:
            prefix=f'1r_cycle_{cid}_set';v=archive[prefix+'_voltage'].copy();i=archive[prefix+'_current'].copy()
            for suffix,a in [('voltage',v),('stored_current',i)]:loaded.update((prefix+suffix).encode());loaded.update(a.tobytes())
            ap=int(np.argmax(v));bare.append((cid,v[:ap+1],abs(i[:ap+1])))

    targets=[]
    for phase,volts in [('set',[5.,4.,2.,.3,.2,.1,.06]),('reset',[.06,.1,.2,.3])]:
        v,y=native[phase];index=np.arange(len(v));ap=int(np.argmax(abs(v)))
        branch=index>ap if phase=='set' else index<=ap
        for value in volts:
            j=int(np.argmin(np.where(branch,abs(abs(v)-value),np.inf)))
            targets.append({'key':f'{phase}_{value:g}','phase':phase,'command_v':float(v[j]),
                'native_index':j,'current_Q10_Q90_a':np.quantile(y[:,j],[.1,.9]).tolist()})
    baseline_protocol={'definition':'same physical F/G; all shape parameters shared across both topologies',
        'time_scope':'General alpha=1 compatibility only; unknown sweep speeds are not inferred or fitted.',
        'single_state_assumptions':'monotone gamma during SET/RESET; same-state time comparison additionally requires local-voltage-monotone, locally Lipschitz closure and quasi-static circuit voltages.'}
    report={'status':'running_predeclared_shape_screen','max_shapes':MAX_SHAPES,'max_wall_seconds':MAX_SECONDS,
        'shape_parameter_order':['log10_median_energy_eV','eta','sigma_log_energy'],
        'predeclared_shapes':[{'name':name,'q_shape':q.tolist()} for name,q in points],
        'domains_algorithmic_not_physical_confidence':{'local_lower':local_lo.tolist(),'local_upper':local_hi.tolist(),
            'global_lower':global_lo.tolist(),'global_upper':global_hi.tolist()},
        'fixed':{'ron_ohm':p['ron_ohm'],'temperature_k':p['temperature_k'],'transistor':'unchanged nominal signed mapping'},
        'training_ids':{'1r':TRAIN_R,'1t1r':TRAIN_T},'targets':targets,'assumptions':baseline_protocol,'results':[],
        'loaded_training_arrays_sha256':loaded.hexdigest()}

    def write():
        report['elapsed_s']=time.monotonic()-start
        tmp=OUT/'shape_feasibility_screen.json.tmp';tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(OUT/'shape_feasibility_screen.json')

    def check():
        if time.monotonic()-start>MAX_SECONDS:raise Deadline('300s wall limit; remaining shapes are not evaluated')

    for name,q in points:
        row={'name':name,'q_shape':q.tolist(),'status':'started'};report['results'].append(row)
        try:
            check();t0=time.monotonic()
            shape=SpectralShape(median_energy_ev=10**q[0],eta=q[1],sigma_log_energy=q[2],ron_ohm=p['ron_ohm'],temperature_k=p['temperature_k'])
            probes=[]
            def invert(command,current,phase):
                check();local=np.sign(command)*(abs(command)-mapping.voltage_for_current(current,1.1,phase))
                if command*local<=0:raise ValueError('No passive local voltage for target band')
                lg=brentq(lambda x:math.log(abs(shape.current_at_log_gamma(local,x))/current),-40.,20.,xtol=2e-10)
                probes.append((float(local),lg,float(current)));return math.exp(lg)
            base_intervals={}
            for target in targets:
                base_intervals[target['key']]=[invert(target['command_v'],i,target['phase']) for i in target['current_Q10_Q90_a']]
            low_set=[base_intervals[f'set_{x:g}'] for x in [.3,.2,.1,.06]]
            low_reset=[base_intervals[f'reset_{x:g}'] for x in [.06,.1,.2,.3]]
            low_set_ok,_=monotone_intervals(low_set,True);low_reset_ok,_=monotone_intervals(low_reset,False)
            frozen_lo=max(x[0] for x in low_set+low_reset);frozen_hi=min(x[1] for x in low_set+low_reset)
            row['low_bias']={'set_directional_interval_feasible':low_set_ok,'reset_directional_interval_feasible':low_reset_ok,
                'common_frozen_gamma_feasible':frozen_lo<=frozen_hi,'common_frozen_gamma_intersection_ev':[frozen_lo,frozen_hi],
                'freezing_scope':'An additional conditional diagnostic, not required to classify a finite-rate direction candidate.',
                'gamma_intervals_ev':{k:v for k,v in base_intervals.items() if abs(float(k.split('_')[1]))<=.3}}

            def terminal(command,gamma,order=32):
                check();vh=brentq(lambda v:shape.current_at_gamma(v,gamma,order=order)-mapping.current(command-v,1.1,'set'),0.,command,xtol=1e-11)
                return float(shape.current_at_gamma(vh,gamma,order=order)),float(vh)

            scenarios=[]
            for expansion in (0.,.01,.05):
                high={}
                for value in (5.,4.,2.):
                    target=next(x for x in targets if x['key']==f'set_{value:g}')
                    # Sensitivity affects only the5/4 endpoint bands.2V and all
                    # low-bias bands remain unchanged in every scenario.
                    if expansion and value in (5.,4.):
                        band=[target['current_Q10_Q90_a'][0]/(1+expansion),target['current_Q10_Q90_a'][1]*(1+expansion)]
                        high[value]=[invert(target['command_v'],i,'set') for i in band]
                    else:high[value]=base_intervals[target['key']]
                high_ok,_=monotone_intervals([high[x] for x in (5.,4.,2.)],True)
                complete_set_ok,_=monotone_intervals([high[x] for x in (5.,4.,2.)]+low_set,True)
                band=[min(high[x][0] for x in (5.,4.)),max(high[x][1] for x in (5.,4.))]
                _,vmin=terminal(4.,band[1]);crossings=[]
                for cid,v,i in bare:
                    check();a=shape.current_at_gamma(v,band[0]);b=shape.current_at_gamma(v,band[1])
                    below=np.flatnonzero(i<a);above=np.flatnonzero((i>=b)&(v>.2));pair=None
                    for k in above:
                        earlier=below[below<k]
                        if len(earlier):pair=(int(earlier[-1]),int(k));break
                    if pair is None:crossings.append({'cycle':cid,'status':'no_bracket'});continue
                    left,right=pair;dv=float(v[right]-v[left]);ordered=vmin>v[right]
                    crossings.append({'cycle':cid,'status':'bracket','start_v':float(v[left]),'end_v':float(v[right]),
                        'path_width_v':dv,'voltage_ordered':bool(ordered),'alpha_lower':1/dv if ordered and high_ok else None})
                alphas=[x['alpha_lower'] for x in crossings if x.get('alpha_lower') is not None]
                max_alpha=max(alphas) if alphas else None
                no_order_rejection=(not alphas) or max_alpha<=1.+1e-5
                scenarios.append({'current_expansion_fraction':expansion,'high_directional_feasible':high_ok,
                    'selected_SET_path_directional_feasible':complete_set_ok,'gamma_band_ev':band,'soft_local_min_v':vmin,
                    'n_valid_time_comparisons':len(alphas),'max_necessary_alpha':max_alpha,
                    'common_path_not_rejected_by_selected_conditions':bool(complete_set_ok and low_reset_ok and no_order_rejection),
                    'also_passes_conditional_low_frozen_state':bool(complete_set_ok and low_reset_ok and no_order_rejection and frozen_lo<=frozen_hi),
                    'bare_crossings':crossings})
            # Certify all inverted finite gamma points and the signed readout;
            # difficult narrow distributions fail loudly rather than silently
            # using an underresolved energy rule or renormalized cutoff.
            vv=np.array([x[0] for x in probes]);gg=np.array([x[1] for x in probes]);obs=np.array([x[2] for x in probes])
            coarse=abs(shape.current_at_log_gamma(vv,gg));fine=abs(shape.current_at_log_gamma(vv,gg,order=64))
            change=float(np.max(abs(np.log10(coarse/fine))))
            row.update(status='numerically_valid' if change<=1e-5 else 'numerically_invalid_refinement',
                scenarios=scenarios,numerical={'order32_to64_max_current_change_decade':change,'threshold_decade':1e-5,
                    'oracle_root_max_current_residual_decade':float(np.max(abs(np.log10(coarse/obs)))),
                    'quadrature_scope':'All inverted finite-state endpoints only, not certification of full future dynamic domain.'},elapsed_s=time.monotonic()-t0)
        except Deadline as exc:
            row.update(status='wall_budget_terminated',error=str(exc));report['status']='wall_budget_terminated';write();break
        except Exception as exc:
            row.update(status='invalid',error=type(exc).__name__+': '+str(exc))
        write()
        print(json.dumps({'name':name,'status':row['status'],
             'candidate_by_scenario':[s['common_path_not_rejected_by_selected_conditions'] for s in row.get('scenarios',[])]}),flush=True)
    else:report['status']='completed_predeclared_47_shapes'
    valid=[x for x in report['results'] if x['status']=='numerically_valid']
    candidates=[]
    for row in valid:
        for s in row['scenarios']:
            if s['common_path_not_rejected_by_selected_conditions']:
                candidates.append({'name':row['name'],'q_shape':row['q_shape'],'current_expansion_fraction':s['current_expansion_fraction'],
                    'low_common_frozen_also_passes':row['low_bias']['common_frozen_gamma_feasible'],'max_necessary_alpha':s['max_necessary_alpha'],
                    'scope':'Not rejected by selected necessary conditions only; no autonomous dynamic fit or unique microscopic explanation.'})
    report['summary']={'n_evaluated':len(report['results']),'n_numerically_valid':len(valid),
        'n_invalid':sum(x['status'] in ('invalid','numerically_invalid_refinement') for x in report['results']),
        'n_not_evaluated':len(points)-len(report['results']),'not_rejected_candidates':candidates}
    report['limitations']=[
        'Only existing lognormal median, log-width and voltage-division eta vary, commonly for1R and1T1R. No cell-specific parameter or extra state.',
        'Gamma is an interval oracle state here, not a free new material coefficient or predicted trajectory.',
        'Low-bias frozen-state intersection is reported separately from directional finite-rate feasibility.',
        'Failure of empirical interval conditions is conditional on those descriptive budgets, not a statistical population rejection.',
        '1%/5% endpoint expansions are explicit sensitivity scenarios, not changes to original acceptance gates.',
        'No fullRESET, raw-record RMSE, kinetics, test window or unseen-device generalization is validated.',
        'If a sampled shape removes this order comparison, it still must pass common dynamics, low-resistance readout and all four phase gates.',
        'A bounded discrete screen cannot prove the entire predeclared domain empty. No automatic extra points or enlarged domain follow.']
    report['source_sha256']={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in [Path(__file__),BASE,GLOBAL,ROOT/'vulcan2d/shared_spectral.py']}
    write()
    lines=['# 既有谱形参数的有限必要条件筛选','',
       '没有增加材料参数、没有运行ODE、没有修改core或正在运行的拟合。47个预声明点/300秒上限；只读取固定训练集。',
       '现选点、6个旧局部域轴点、8个局部角点、32个旧全局域Sobol点均在启动前登记。算法域不是物理置信区间。','',
       '低偏0.06–0.3V分别检查SET/RESET单向状态区间；“共同冻结读态”是另列的更强条件，不用于偷偷否决整个有限速率族。',
       '高压检查5→4→2V及到低偏的单向连接，并重新计算bare同态穿带界。1%/5%只扩张5/4V端口区间，原主gate不变。','',
       f'状态：{report["status"]}；已评估 {len(report["results"])} 点，数值有效 {len(valid)} 点。','',
       '| 未被所选必要条件否决的点 | [log10 median,eta,sigma] | 扩张 | 低偏共同冻结也可行 |','|---|---|---:|---|']
    for c in candidates:lines.append(f'| {c["name"]} | {c["q_shape"]} | {100*c["current_expansion_fraction"]:.0f}% | {c["low_common_frozen_also_passes"]} |')
    if not candidates:lines.append('| 本次离散点中没有 | — | — | — |')
    lines += ['', '这些结果仅表示在明确误差预算和名义电路下通过/未通过局部必要条件，不是统一动态模型成功或整个谱族无解。无效点、终止原因、完整端点与逐记录穿带结果均保留在JSON。',
        '真正进步仍需同一方程、连续状态、原始四相评分及既定特征门槛；不能用未知扫速、逐接法参数或逐点oracle状态代替预测。']
    (OUT/'shape_feasibility_screen.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report['summary']),flush=True)


if __name__=='__main__':main()
