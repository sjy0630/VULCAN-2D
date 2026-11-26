"""Frozen SET trajectory/drive diagnostic. No model calls, ODEs or fits."""
from pathlib import Path
import hashlib
import json
import math
import signal
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    plan_path=HERE/'drive_trace_diagnostic.plan.json';plan=json.loads(plan_path.read_text())
    source_paths={name:ROOT/(plan[name+'_source']+'.json') for name in ('constant','field')}
    reports={name:json.loads(p.read_text()) for name,p in source_paths.items()}
    archives={}
    for name,p in source_paths.items():
        with np.load(p.with_suffix('.npz')) as z:archives[name]={k:z[k].copy() for k in z.files}
    constant=reports['constant'];seed=next(x for x in constant['plan']['seeds'] if x['name']=='beta6')
    beta,js,coefficient=seed['beta'],seed['JS_A'],seed['k_per_v']
    assert beta==reports['field']['plan']['beta'] and js==reports['field']['plan']['JS_A']
    assert coefficient==reports['field']['plan']['kappa_per_v2']
    shared=('command_voltage_v','path_voltage_s','observed_current_a','original_scoring_mask')
    for key in shared:np.testing.assert_array_equal(archives['constant'][key],archives['field'][key])
    v=archives['field']['command_voltage_v'];path=archives['field']['path_voltage_s'];obs=abs(archives['field']['observed_current_a'])
    mask=archives['field']['original_scoring_mask'];out=np.arange(len(v))<=np.argmax(v)
    np.testing.assert_array_equal(mask,np.isfinite(obs)&(obs>=1e-11)&(v>1e-12));assert mask.sum()==500
    sources={str(p):sha(p) for p in [plan_path,Path(__file__),*[q for p in source_paths.values() for q in (p,p.with_suffix('.npz'))]]}
    result=dict(status='running_saved_trace_diagnostic',plan=plan,source_sha256=sources,beta=beta,JS_A=js,
                scope='The rate coefficients have different units:constant V^-1,field V^-2. Their equal numeric value is not an equal physical calibration.',models={})
    arrays={key:archives['field'][key] for key in shared}
    dest=HERE/'drive_trace_diagnostic.json';dest.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    def point(A,branch):
        allowed=out if branch=='outward' else ~out
        j=int(np.argmin(np.where(allowed,abs(v-A),np.inf)));assert abs(v[j]-A)<1e-10
        return j
    def coordinate(j):
        return dict(native_index=int(j),command_v=float(v[j]),path_s_v=float(path[j]))
    def crossing(current,threshold):
        ids=np.flatnonzero(out&mask);hits=ids[current[ids]>=threshold]
        if not len(hits):return dict(reached=False)
        j=int(hits[0]);before=max(0,j-1)
        return dict(reached=True,last_below=coordinate(before),first_at_or_above=coordinate(j),
                    lower_current_a=float(current[before]),upper_current_a=float(current[j]))
    def timeout(*_):raise TimeoutError('60second saved-array budget exceeded')
    began=time.monotonic();signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,60.)
    try:
        for name,z in archives.items():
            pre='beta6_refined' if name=='constant' else 'refined'
            nominal='beta6_nominal' if name=='constant' else 'nominal'
            current=z[pre+'_current_a'];u=z[pre+'_state_u'];vh=z[pre+'_local_voltage_v']
            ni=z[nominal+'_current_a'];nu=z[nominal+'_state_u']
            positive=(current>0)&(v>0)
            D=np.full(len(v),-np.inf);D[positive]=np.log(current[positive]/js)+beta*u[positive]
            Dn=np.full(len(v),-np.inf);Dn[positive]=np.log(ni[positive]/js)+beta*nu[positive]
            rate=-coefficient*(vh if name=='field' else 1.)*np.maximum(D,0.)
            rate[~positive]=0.
            threshold=js*np.exp(-beta*u)
            regimes=np.zeros(len(v),int)
            regimes[positive&(D<-.01)]=1;regimes[positive&(D>=-.01)&(D<=0)]=2
            regimes[positive&(D>0)&(D<=.01)]=3;regimes[positive&(D>.01)]=4
            errors=np.full(len(v),np.nan);errors[mask]=np.log10(current[mask]/obs[mask]);SSE=float(np.sum(errors[mask]**2))
            stored_run=next(x for x in reports[name]['runs'] if x['precision']['name']=='refined' and 'metrics' in x and (name=='field' or x['seed']['name']=='beta6'))
            np.testing.assert_allclose(np.sqrt(SSE/500),stored_run['metrics']['all500']['raw_RMSE_decade'],rtol=1e-12)
            rows=[];selected=[];slopes=[];regions=[]
            branch_summaries={}
            for branch,select in [('outward',out),('return',~out)]:
                ids=np.flatnonzero(select&mask)
                counts={label:int(np.sum(regimes[ids]==code)) for code,label in [(1,'inactive'),(2,'near_inactive'),(3,'near_active'),(4,'active')]}
                active=ids[D[ids]>0];sign_sensitive=ids[(D[ids]>0)!=(Dn[ids]>0)]
                branch_summaries[branch]=dict(n=len(ids),regime_counts=counts,
                    first_active_native=coordinate(active[0]) if len(active) else None,last_active_native=coordinate(active[-1]) if len(active) else None,
                    D_range=[float(min(D[ids])),float(max(D[ids]))],
                    max_nominal_refined_drive_change=float(np.max(abs(D[ids]-Dn[ids]))),
                    drive_sign_sensitive_to_precision=[coordinate(j) for j in sign_sensitive])
                adjacent=[]
                for a,b in zip(ids[:-1],ids[1:]):
                    if b!=a+1 or v[a]==v[b]:continue
                    alpha_p=float(math.log(current[b]/current[a])/math.log(v[b]/v[a]))
                    alpha_o=float(math.log(obs[b]/obs[a])/math.log(v[b]/v[a]))
                    adjacent.append((a,b,math.sqrt(v[a]*v[b]),alpha_p,alpha_o))
                slopes.extend(dict(branch=branch,left_index=int(a),right_index=int(b),geomean_command_v=c,
                    predicted_alpha=ap,observed_alpha=ao) for a,b,c,ap,ao in adjacent)
                edges=plan['voltage_region_edges_v']
                for low,high in zip(edges[:-1],edges[1:]):
                    m=select&mask&(v>low+1e-12)&(v<=high+1e-12);e=errors[m]
                    ja,jb=point(low,branch),point(high,branch)
                    start,end=(ja,jb) if branch=='outward' else (jb,ja)
                    slope_rows=[x for x in adjacent if low<x[2]<=high]
                    pa=np.array([x[3] for x in slope_rows]);oa=np.array([x[4] for x in slope_rows])
                    regions.append(dict(branch=branch,lower_exclusive_v=low,upper_inclusive_v=high,n=int(m.sum()),
                        RMSE_decade=float(np.sqrt(np.mean(e*e))),signed_log_bias=float(np.mean(e)),
                        fraction_total_squared_log_error=float(np.sum(e*e)/SSE),state_loss=float(u[start]-u[end]),
                        drive_range=[float(min(D[m])),float(max(D[m]))],
                        rate_magnitude_range_per_v=[float(min(-rate[m])),float(max(-rate[m]))],
                        n_positive_drive=int(np.sum(D[m]>0)),
                        adjacent_slope_count=len(slope_rows),
                        predicted_alpha_q10_median_q90=np.quantile(pa,[.1,.5,.9]).tolist() if len(pa) else None,
                        observed_alpha_q10_median_q90=np.quantile(oa,[.1,.5,.9]).tolist() if len(oa) else None))
                for low,high in plan['secant_windows_v']:
                    a,b=point(low,branch),point(high,branch)
                    rows.append(dict(branch=branch,window_v=[low,high],native_indices=[a,b],
                        predicted_alpha=float(math.log(current[b]/current[a])/math.log(high/low)),
                        observed_alpha=float(math.log(obs[b]/obs[a])/math.log(high/low))))
                for A in plan['selected_native_voltages_v']:
                    j=point(A,branch)
                    selected.append(dict(branch=branch,**coordinate(j),current_a=float(current[j]),observed_a=float(obs[j]),
                        local_voltage_v=float(vh[j]),u=float(u[j]),D=float(D[j]),
                        log_current_over_JS=float(math.log(current[j]/js)),beta_u=float(beta*u[j]),
                        effective_current_threshold_a=float(threshold[j]),state_rate_per_path_v=float(rate[j]),
                        signed_log_error_decade=float(errors[j])))
            assert sum(x['n'] for x in regions)==500
            np.testing.assert_allclose(sum(x['fraction_total_squared_log_error'] for x in regions),1.,rtol=1e-12)
            ret_ids=np.flatnonzero((~out)&mask);ret_all=np.flatnonzero(~out)
            initial_ret=int(ret_all[0]);final_ret=int(ret_all[-1]);return_loss=float(u[initial_ret]-u[final_ret])
            total_loss=float(u[0]-u[-1]);loss_fraction=[]
            for fraction in plan['return_loss_fractions']:
                hit=ret_all[(u[initial_ret]-u[ret_all])>=fraction*return_loss]
                j=int(hit[0]);loss_fraction.append(dict(fraction=fraction,**coordinate(j),u=float(u[j]),remaining_return_loss=float(u[j]-u[-1])))
            stop=[]
            for drive,label in [(D,'refined'),(Dn,'nominal')]:
                hits=[int(j) for j in ret_ids if np.all(drive[ret_ids[ret_ids>=j]]<=0)]
                if hits:
                    j=hits[0];prev=j-1
                    stop.append(dict(precision=label,first_native_permanent_nonpositive_drive=coordinate(j),
                        preceding_native=coordinate(prev),D_at_first=float(drive[j]),D_at_previous=float(drive[prev]),
                        state_at_first=float(u[j] if label=='refined' else nu[j])))
                else:stop.append(dict(precision=label,first_native_permanent_nonpositive_drive=None))
            eps=plan['state_freeze_tolerance_u']
            frozen=[int(j) for j in ret_all if np.max(abs(u[j:]-u[-1]))<=eps]
            jfreeze=frozen[0]
            node_state_exact=[int(j) for j in ret_all if np.array_equal(u[j:],np.full(len(u)-j,u[-1]))]
            return_nodes=[]
            for A in [5.,4.98,4.9,4.8,4.,3.,2.,1.,.2,0.]:
                j=point(A,'return');return_nodes.append(dict(**coordinate(j),u=float(u[j]),remaining_state_loss=float(u[j]-u[-1]),
                    D=float(D[j]) if positive[j] else None,rate_per_path_v=float(rate[j])))
            freezing=dict(total_return_state_loss=return_loss,total_SET_state_loss=total_loss,
                return_loss_fraction_of_total_SET_loss=return_loss/total_loss,
                initial_return_D=float(D[initial_ret]),initial_return_rate_per_v=float(rate[initial_ret]),
                loss_completion_native_points=loss_fraction,drive_stop_brackets=stop,
                first_native_tail_with_remaining_state_change_le_1e_minus6=coordinate(jfreeze),
                first_bitwise_constant_native_tail=coordinate(node_state_exact[0]) if node_state_exact else None,native_samples=return_nodes,
                n_nonzero_return_points_after_permanent_D_nonpositive=(int(np.sum(ret_ids>=stop[0]['first_native_permanent_nonpositive_drive']['native_index'])) if stop[0]['first_native_permanent_nonpositive_drive'] else 0))
            transitions=[dict(threshold_a=x,observed=crossing(obs,x),predicted=crossing(current,x)) for x in plan['outward_current_thresholds_a']]
            low_loss=float(u[0]-u[point(.2,'outward')])
            model=dict(all500_RMSE_decade=float(math.sqrt(SSE/500)),branch_summaries=branch_summaries,
                low_outward_zero_to_0p2_state_loss=low_loss,low_loss_fraction_of_total_SET_loss=low_loss/total_loss,
                residual_regions=regions,secant_slopes=rows,selected_native_samples=selected,
                outward_threshold_crossings=transitions,return_freezing=freezing,
                maximum_absolute_log_residual=dict(**coordinate(int(np.flatnonzero(mask)[np.argmax(abs(errors[mask]))])),
                    abs_error_decade=float(np.max(abs(errors[mask])))),
                original500_mask_preserved=True)
            result['models'][name]=model
            arrays.update({name+'_current_a':current,name+'_local_voltage_v':vh,name+'_state_u':u,
                name+'_drive':D,name+'_nominal_drive':Dn,name+'_state_rate_per_path_v':rate,
                name+'_effective_current_threshold_a':threshold,name+'_regime_code':regimes,
                name+'_log_residual_decade':errors,name+'_adjacent_slopes':np.array([[x['left_index'],x['right_index'],x['geomean_command_v'],x['predicted_alpha'],x['observed_alpha']] for x in slopes])})
        result['algebraic_implications']=[
            'For positive mobility m, u_prime=-c*m(A,u)*positive(D) has exactly the same stopping surface D(A,u)=0. Changing mobility can change the path and the state/voltage at first contact, but cannot change the surface itself.',
            'Once a return trajectory has D<=0 at state u_f and command decreases, monotone passive I_T(A,u_f) cannot increase D. The frozen state remains an admissible unique solution; positive mobility alone cannot restart it.',
            'Below the stopping command, predicted return is the fixed-state circuit curve I_T(A,u_f). Mobility affects it only through the inherited scalar u_f; it cannot independently tune its voltage-dependent slopes while preserving u_f and the same spectrum/transistor.',
            'If retaining the present u_f, a mobility-only change leaves every frozen-return native current unchanged. A different u_f could change amplitudes/slopes through the fixed spectral family, but compatibility must be tested, not presumed.',
            'D>0 at the first native outward point demonstrates native-grid activation, not a measured microscopic threshold or an exact onset below0.02V. The same0.2V condition makes pre-anchor behavior a reconstructed trajectory, not a predicted physical initial history.']
        result['suggested_finite_followup_not_executed']=dict(
            question='Can one frozen return state under the existing spectrum/transistor satisfy the independent voltage-dependent return constraints?',
            proposed_check='Use existing fixed return endpoints/bands to form state intervals by monotone circuit inversion, then intersect them without fitting kinetics. Distinguish the single-cycle observed curve from original32-record aggregate bands.',
            decision='If a common state is excluded under credible readout uncertainty, preserving frozen return plus changing only positive mobility cannot fix those constraints; examine the stopping condition, readout, or spectrum only after that finite test.',
            actual_new_inverse_calls=0,actual_new_ODE_runs=0,actual_new_fits=0)
        result['frozen_source_hashes_unchanged']={str(p):sha(p)==sources[str(p)] for p in source_paths.values()}
        assert all(result['frozen_source_hashes_unchanged'].values())
        result['status']='completed_saved_trace_drive_and_freezing_diagnostic_no_new_model'
    except Exception as exc:result.update(status='partial_or_failed_preserved',error_type=type(exc).__name__,error=str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0.);result['compute_elapsed_s']=time.monotonic()-began
    dest.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');np.savez_compressed(dest.with_suffix('.npz'),**arrays)
    lines=['# 已有β6 SET轨迹：驱动、外扫激活与返扫停止诊断','',
        '只读取已冻结的constant β6和local-field β6加密轨迹；没有新模型调用、ODE、扫描、拟合或保留集。两者使用同一cycle4、500点评分与0.2V条件锚。此处不重建或跨越用户确认的裸器件历史分段。',
        '', 'D=ln(I/JS)+βu；constant速率为−k[D]₊，field速率为−κ|Vh|[D]₊。相同数值系数的单位不同，不能当作同一物理标定。|D|≤0.01只作近停止的描述性邻域，精确符号另行保留。',
        '', '| 模型 | 全500 RMSE | 0→0.2V状态损失 | 占全SET状态损失 | 返扫总状态损失 | 返扫占全损失 |',
        '|---|---:|---:|---:|---:|---:|']
    for name,m in result['models'].items():
        f=m['return_freezing'];lines.append(f'| {name} | {m["all500_RMSE_decade"]:.7f} | {m["low_outward_zero_to_0p2_state_loss"]:.9g} | {m["low_loss_fraction_of_total_SET_loss"]*100:.4f}% | {f["total_return_state_loss"]:.9g} | {f["return_loss_fraction_of_total_SET_loss"]*100:.6f}% |')
    for name,m in result['models'].items():
        f=m['return_freezing'];lines+=['',f'## {name}返扫停止','',f'5V初始D={f["initial_return_D"]:.9g}；原生非零返扫点中，{f["n_nonzero_return_points_after_permanent_D_nonpositive"]}/250点处于已永久非正驱动之后。']
        for x in f['drive_stop_brackets']:
            if x['first_native_permanent_nonpositive_drive']:
                a=x['preceding_native'];b=x['first_native_permanent_nonpositive_drive'];lines.append(f'{x["precision"]}原生停止括区：A从{a["command_v"]:g}到{b["command_v"]:g}V，D从{x["D_at_previous"]:.9g}到{x["D_at_first"]:.9g}；不宣称连续精确事件位置。')
        lines+=['', '| 返扫命令/V | u | 尚余状态损失 | D |', '|---:|---:|---:|---:|']
        for x in f['native_samples']:lines.append(f'| {x["command_v"]:g} | {x["u"]:.10g} | {x["remaining_state_loss"]:.6g} | {x["D"]} |')
        lines+=['',f'## {name}全部电压区间残差','', '| 分支 | 电压区间/V | n | RMSE/dec | 有符号log偏差 | 占总平方log误差 | 状态损失 |', '|---|---|---:|---:|---:|---:|---:|']
        for x in m['residual_regions']:lines.append(f'| {x["branch"]} | ({x["lower_exclusive_v"]:g},{x["upper_inclusive_v"]:g}] | {x["n"]} | {x["RMSE_decade"]:.6g} | {x["signed_log_bias"]:.6g} | {x["fraction_total_squared_log_error"]*100:.4f}% | {x["state_loss"]:.7g} |')
        lines+=['',f'## {name}固定端点局部斜率','', '| 分支 | 电压窗/V | 预测α | 实测α |','|---|---|---:|---:|']
        for x in m['secant_slopes']:lines.append(f'| {x["branch"]} | {x["window_v"]} | {x["predicted_alpha"]:.6g} | {x["observed_alpha"]:.6g} |')
    lines+=['', '## 对只修改迁移率的代数限制','',
        '正迁移率只乘在[D]₊前，不移动D=0停止面；它可以改变到达停止面的历史与最终u，不能在相同冻结态下独立修正不同电压的返扫斜率。返扫一旦D≤0，命令降低、同一u下被动电流降低，使D继续非增，故正迁移率不能自行重新激活。',
        '', '停止之后，返扫就是固定u的材料谱与晶体管串联曲线。若要声称改迁移率可以修复这些区间，必须先证明改变继承u后同一个静态状态能兼容各返扫约束；不能忽略停止面而继续加速率参数。',
        '', '建议下一有限检查：对既有返扫端点/带做单调电路反演形成状态允许区间并求交，不拟合动力学；单圈实测与原32圈聚合带分开。该建议尚未执行，本轮反演次数、ODE数与拟合数都为0。',
        '',f'纯算术计算{result["compute_elapsed_s"]:.4f}s。所有500点及每一电压区间均保留；局部相邻斜率不平滑、不截断，完整数组在NPZ。近停止邻域、1e−6冻结容差与原生括区是诊断定义，不是实测误差条或微观阈值。']
    dest.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=result['status'],compute_elapsed_s=result['compute_elapsed_s'],models={name:dict(
        RMSE=m['all500_RMSE_decade'],low_state_loss=m['low_outward_zero_to_0p2_state_loss'],return_freezing=m['return_freezing'],
        largest_regions=sorted(m['residual_regions'],key=lambda x:x['fraction_total_squared_log_error'],reverse=True)[:3]) for name,m in result['models'].items()}),indent=2))


if __name__=='__main__':main()
