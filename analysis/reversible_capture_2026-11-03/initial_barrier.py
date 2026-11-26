"""Fixed-q invariant prefix and optimistic original-500-point RMSE floor."""
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
OLD=ROOT/'analysis/soft_drive_structure_2026-11-03'
CACHE=Path('/private/tmp/vulcan-runtime-cache')
sys.path[:0]=[str(ROOT),str(ROOT/'analysis/mechanism_selection_2026-11-03')]
from SOFT_FEEDBACK_LOW_BIAS_BUDGET import cached_transport


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    dest=HERE/'initial_barrier.json'
    if dest.exists():raise FileExistsError('Preserve completed or failed diagnostic; no automatic rerun')
    budget=json.loads((HERE/'reversible_budget.json').read_text())
    assert budget['status']=='completed_one_seed_reversible_rate_budget_no_ODE_or_optimization'
    seed=budget['plan']['seed'];lam=seed['lambda_per_v'];jref=seed['J_positive_a']
    ron,gc=seed['Ron_ohm'],seed['gamma_c_ev']
    old=json.loads((OLD/'sinh_field_probe.json').read_text())
    oldrun=next(r for r in old['runs'] if r['precision']['name']=='refined')
    prior_rmse=oldrun['metrics']['all500']['RMSE_decade']
    with np.load(CACHE/'native_observations.npz') as z:
        v=z['1t1r_cycle_4_set_voltage'].copy();signed=z['1t1r_cycle_4_set_current'].copy()
    observed=abs(signed);out=np.arange(len(v))<=np.argmax(v)
    mask=np.isfinite(observed)&(observed>=1e-11)&(v>1e-12)
    assert len(v)==503 and mask.sum()==500 and (mask&out).sum()==250
    with np.load(OLD/'sinh_field_probe.npz') as z:
        np.testing.assert_array_equal(v,z['command_voltage_v'])
        np.testing.assert_array_equal(signed,z['observed_current_a'])
        np.testing.assert_array_equal(mask,z['original_scoring_mask'])
    assert sha(CACHE/'native_observations.npz')=='0aedbe51f41b4f4355e1de4d26048c6790a9d454b8b7d56f51bd76a4671465a8'
    plan={'scope':'Fixedconditionalcapture seed, allnonnegativekappa, initialqbar invariantprefix only; not a newtrajectory or original32cycleaggregate test',
        'seed':seed,'state_law':'qprime=kappa*R, R=v*((I/J)*(1-q)-exp(lambda*v)*q)',
        'qbar':'Sameenvelope q0_optimistic_upper fromcompletedreversiblebudget; trueconditionalq(.2)<=qbar, not measuredoccupancy',
        'command_range_v':[.2,2.],
        'grids':[{'name':'nominal','step_v':.01,'KCL_bisections':48},
                 {'name':'refined','step_v':.005,'KCL_bisections':52}],
        'material_relative_envelopes':[0.,.003],
        'envelope_scope':'Inherited numerical sensitivityaroundfixedphysical spectrum; not measurednoise or newglobalaccuracycertificate',
        'fixedq_rectangle_upper':'vmax*Imax*(1-qbar)/J-vmin*exp(lambda*vmin)*qbar+roundingguard, Imax=strong(Ahigh,qbar)Iupper,vmax=weak(Ahigh,qbar)Vupper,vmin=strong(Alow,qbar)Vlower',
        'prefix_rule':'OnlycontiguouscellsstartingA=.2withexpandedupper_R<0; stopbeforefirstnonnegative/unknowncell. Exactzero treatedunknown fornumeric caution.',
        'invariance':'Atqbar thevectorfieldisnonpositive oncertifiedprefix, soadmissiblecontinuoussolutionstartingq<=qbar staysq<=qbar forallkappa>=0. No claim beyondprefix oronreturn.',
        'current_ceiling':'Ateligible nativeoutward pointsinsideprefix, strongmaterialKCL(A,qbar) currentupper boundsI(A,q)<=Iceil.',
        'score_floor':'sqrt(sum_{certified originaleligiblepoints} max(log10(Iobs/Iceil),0)^2 /500); allother original500point contributions assigned0 optimistically',
        'representative_voltages_v':[1.,1.4],
        'numerical_guard_v':1e-12,'rate_guard':'1e-12*(captureupper+emissionlower)+1e-18V',
        'KCL_residual_limit_a':1e-12,
        'counts':{'native503':503,'original_eligible500':500,'outward250':250,'new_inverses':0,'new_ODE':0,'new_fit':0},
        'compute_budget_s':60,'budget_start':'Afterallinputs/sourcehashes/planwrite,beforeKCL/barrier/scorecalculations',
        'stop_policy':'Preserveallfailures; no prefixextension,gamma/seedtuning,extra grids,heldoutorODE',
        'comparison':'Prior.18223064samecycleconditionalRMSE comparedasexisting achievederror,notanewacceptancegate'}
    planpath=HERE/'initial_barrier.plan.json';planpath.write_text(json.dumps(plan,indent=2,allow_nan=False)+'\n')
    t,lookups=cached_transport(CACHE)
    paths=[HERE/'reversible_budget.json',OLD/'sinh_field_probe.json',OLD/'sinh_field_probe.npz',
           CACHE/'native_observations.npz',t.table_path,*lookups,planpath,Path(__file__),
           ROOT/'analysis/mechanism_selection_2026-11-03/SOFT_FEEDBACK_LOW_BIAS_BUDGET.py',
           ROOT/'analysis/ron_scope_2026-11-03/ron_scaling_review.py']
    sources={str(p):sha(p) for p in paths}
    r={'status':'running_fixedq_initial_barrier','plan':plan,'source_sha256':sources,'cases':[],
       'previous_achieved_conditional_500point_RMSE_decade':prior_rmse,
       'same_native_observations_and_mask_verified':True,'KCL_node_problems':0,'KCL_vector_batches':0,
       'new_inverse_problems':0,'new_ODE':0,'new_fit':0}
    arrays={'command_voltage_v':v,'observed_current_a':signed,'original_scoring_mask':mask,'outward_branch_mask':out}
    tx,ti=t.knots['set'];vmin_table=math.exp(t.lv[0])
    started=time.monotonic()
    def timeout(*_):raise TimeoutError('60secondfixedbarrierbudgetexhausted;preserveoutputs')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,60)
    try:
        def bracket(A,q,factor,steps):
            A=np.asarray(A,float);gamma=gc*q/(1-q)
            assert 0<q<1 and t.lg[0]<=math.log(gamma)<=t.lg[-1]
            assert np.all(A>=.2) and np.all(A<=2.)
            lo=np.zeros_like(A);hi=A.copy()
            for _ in range(steps):
                mid=.5*(lo+hi);assert np.all(mid>=vmin_table)
                f=t.material(mid,gamma,ron/factor,'set')-np.interp(A-mid,tx,ti)
                high=f>0;hi=np.where(high,mid,hi);lo=np.where(high,lo,mid)
            center=.5*(lo+hi)
            residual=t.material(center,gamma,ron/factor,'set')-np.interp(A-center,tx,ti)
            assert np.max(abs(residual))<=plan['KCL_residual_limit_a']
            vlo=np.maximum(0.,lo-plan['numerical_guard_v']);vhi=np.minimum(A,hi+plan['numerical_guard_v'])
            ilo=np.interp(A-vhi,tx,ti);ihi=np.interp(A-vlo,tx,ti)
            r['KCL_node_problems']+=int(A.size);r['KCL_vector_batches']+=1
            return {'Ilo':ilo,'Ihi':ihi,'vlo':vlo,'vhi':vhi,
                    'max_residual_a':float(np.max(abs(residual))),
                    'max_root_width_v':float(np.max(hi-lo))}
        for eta in plan['material_relative_envelopes']:
            oldcase=next(c for c in budget['cases'] if c['material_envelope']==eta and c['grid']['name']=='refined')
            qbar=oldcase['first_passage']['q0_optimistic_upper']
            for grid in plan['grids']:
                pre=('eta0' if eta==0 else 'eta003')+'_'+grid['name']
                nodes=np.linspace(.2,2.,round(1.8/grid['step_v'])+1)
                strong=bracket(nodes,qbar,1+eta,grid['KCL_bisections'])
                weak=bracket(nodes,qbar,1-eta,grid['KCL_bisections'])
                imax=strong['Ihi'][1:];vmax=weak['vhi'][1:];vmin=strong['vlo'][:-1]
                capture=vmax*imax*(1-qbar)/jref
                emission=vmin*np.exp(lam*vmin)*qbar
                guard=1e-12*(capture+emission)+1e-18
                upper=capture-emission+guard
                certified=upper<0
                first=np.flatnonzero(~certified)
                count=int(first[0]) if len(first) else len(upper)
                end=float(nodes[count])
                use=mask&out&(v>=.2)&(v<=end)
                indices=np.flatnonzero(use)
                ceilings=bracket(v[use],qbar,1+eta,grid['KCL_bisections'])
                iceil=ceilings['Ihi'];assert np.all(iceil>0)
                deficit=np.maximum(np.log10(observed[use]/iceil),0.)
                total_sse=float(np.dot(deficit,deficit));floor=math.sqrt(total_sse/500.)
                ceilingfull=np.full(len(v),np.nan);ceilingfull[use]=iceil
                deficitfull=np.zeros(len(v));deficitfull[use]=deficit
                reps=[]
                for target in plan['representative_voltages_v']:
                    hit=np.flatnonzero(out&mask&np.isclose(v,target,atol=1e-12,rtol=0))
                    item={'requested_command_v':target,'unique_native_point':len(hit)==1,
                          'inside_certified_prefix':bool(len(hit)==1 and use[hit[0]])}
                    if item['inside_certified_prefix']:
                        j=int(hit[0]);item.update(native_index=j,observed_current_a=float(observed[j]),
                            optimistic_model_current_upper_a=float(ceilingfull[j]),
                            minimum_log10_underprediction=float(deficitfull[j]),
                            observed_over_current_upper=float(observed[j]/ceilingfull[j]))
                    reps.append(item)
                arrays.update({pre+'_command_nodes':nodes,pre+'_capture_upper':capture,
                    pre+'_emission_lower':emission,pre+'_rounding_guard':guard,
                    pre+'_rate_upper':upper,pre+'_negative_certified_cells':certified,
                    pre+'_certified_native_mask':use,pre+'_optimistic_current_upper_a':ceilingfull,
                    pre+'_minimum_positive_log_residual':deficitfull,
                    pre+'_strong_node_current_upper_a':strong['Ihi'],pre+'_strong_node_voltage_lower_v':strong['vlo'],
                    pre+'_weak_node_voltage_upper_v':weak['vhi']})
                case={'material_envelope':eta,'grid':grid,'qbar':qbar,
                    'certified_prefix_start_v':.2,'certified_prefix_end_v':end,
                    'contiguous_negative_cells':count,'total_cells':len(upper),
                    'prefix_minimum_negative_margin_v':float((-upper[:count]).min()) if count else None,
                    'next_cell':None if count==len(upper) else {'left_command_v':float(nodes[count]),
                        'right_command_v':float(nodes[count+1]),'expanded_rate_upper_v':float(upper[count]),
                        'interpretation':'Unknownornonnegativeupper; notcertifiedpositive andprefixnotextended'},
                    'certified_original_native_points':int(use.sum()),'eligible_total_unchanged':500,
                    'optimistic_squared_log_residual_sum':total_sse,
                    'optimistic_500point_RMSE_floor_decade':floor,
                    'floor_over_previous_achieved_conditional_RMSE':floor/prior_rmse,
                    'floor_exceeds_previous_achieved_conditional_RMSE':floor>prior_rmse,
                    'representative_points':reps,
                    'maximum_KCL_residual_a':max(strong['max_residual_a'],weak['max_residual_a'],ceilings['max_residual_a']),
                    'maximum_root_bracket_width_v':max(strong['max_root_width_v'],weak['max_root_width_v'],ceilings['max_root_width_v']),
                    'scope':'Currentceilingandscorefloorforallkappa>=0underfixedconditionalqbar/spectrum/envelope. Noactualqtrajectory. Remainingoriginalpointsoptimisticallyzeroerror.'}
                r['cases'].append(case)
        r['status']='completed_fixedq_prefix_and_500point_error_floor_no_ODE_or_fit'
    except Exception as exc:r.update(status='failed_or_partial_barrier_preserved',error_type=type(exc).__name__,error=str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);r['compute_elapsed_s']=time.monotonic()-started
    dest.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(dest.with_suffix('.npz'),**arrays)
    report(r)
    print(json.dumps({k:r[k] for k in ('status','compute_elapsed_s','KCL_node_problems','KCL_vector_batches','cases')},indent=2))


def report(r):
    lines=['# 初始状态屏障与原500点误差下界','',
        '保持唯一两点条件种子的λ、J+及固定谱/晶体管。对qbar=q(.2V)的精确条件值或包络下最有利上界，检验实际净率在该边界是否持续非正。没有挑选κ，没有ODE、反演或拟合。',
        '', '每个电压格用强/弱材料KCL根区间和明确数值缓冲给R(A,qbar)上界。只认证从0.2V开始连续上界<0的前缀；第一次未知/非负上界处立即停止。κ≥0时，连续解从q≤qbar出发不能向上穿过这条边界。因此每个认证的原生外扫点I≤强材料I(A,qbar)。',
        '', '| 材料包络 | 网格 | 认证前缀终点/V | 原500点中受约束点数 | 乐观RMSE下界/dec | 相对旧0.182成绩 |',
        '|---|---|---:|---:|---:|---:|']
    for c in r['cases']:
        lines.append(f'| {c["material_envelope"]} | {c["grid"]["name"]} | {c["certified_prefix_end_v"]:.5f} | {c["certified_original_native_points"]} | {c["optimistic_500point_RMSE_floor_decade"]:.9f} | {c["floor_over_previous_achieved_conditional_RMSE"]:.5f} |')
    lines+=['', '评分下界仅累加认证区间内max(log10(Iobs/Iceil),0)²，其他所有原评分点的误差都乐观设为0，总分母仍为500。没有删去低压点、把分母改成认证点数或把这条电流上界冒充真实预测轨迹。',
        '', '| 材料包络 | 网格 | 命令/V | 实测/µA | 模型电流最高/µA | 必要低估倍数 |',
        '|---|---|---:|---:|---:|---:|']
    for c in r['cases']:
        for p in c['representative_points']:
            if p['inside_certified_prefix']:
                lines.append(f'| {c["material_envelope"]} | {c["grid"]["name"]} | {p["requested_command_v"]} | {p["observed_current_a"]*1e6:.9g} | {p["optimistic_model_current_upper_a"]*1e6:.9g} | {p["observed_over_current_upper"]:.9g} |')
            else:lines.append(f'| {c["material_envelope"]} | {c["grid"]["name"]} | {p["requested_command_v"]} | — | 不在认证前缀，不报告约束 | — |')
    lines+=['', 'qbar是固定读出下的条件状态界，不是测得的占据率。0.3%是材料表数值敏感性假设，不是仪器噪声或新的全域精度证书。该前缀不覆盖0→0.2V历史，也不延伸到返扫；连续性与局部唯一解是屏障论证前提。',
        '', f'旧对照是同一cycle4、相同原生数据与500点掩码的sinh条件结果RMSE={r["previous_achieved_conditional_500point_RMSE_decade"]:.9f}dec。比较它不是临时新增验收门槛；若新候选即使把其余误差设零仍更差，就没有理由为了降低这项既有误差再扫描κ。它仍只排除这一固定λ/J+和条件初态种子，不排除所有可逆捕获模型或完整聚合合同。',
        '',f'状态：{r["status"]}；计算{r["compute_elapsed_s"]:.6f}s（原预算60s）。KCL节点{r["KCL_node_problems"]}、向量批次{r["KCL_vector_batches"]}。没有新反演、ODE、参数拟合或保留集访问。']
    if 'error'in r:lines+=['',r['error_type']+': '+r['error']]
    (HERE/'initial_barrier.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
