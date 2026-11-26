"""One conditional reversible-capture seed: bounded rates, no ODE or search."""
from pathlib import Path
import hashlib
import json
import math
import signal
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OLD = ROOT / 'analysis/soft_drive_structure_2026-11-03'
RATE = ROOT / 'analysis/soft_rate_compatibility_2026-11-03'
CACHE = Path('/private/tmp/vulcan-runtime-cache')
sys.path[:0] = [str(ROOT), str(ROOT / 'analysis/mechanism_selection_2026-11-03')]
from SOFT_FEEDBACK_LOW_BIAS_BUDGET import cached_transport


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    destination = HERE / 'reversible_budget.json'
    if destination.exists():
        raise FileExistsError('Preserve completed or failed output; no automatic rerun')
    inversion = json.loads((OLD / 'DRIVE_STRUCTURE_INVERSION.json').read_text())
    old_probe = json.loads((OLD / 'sinh_field_probe.json').read_text())
    rate = json.loads((RATE / 'rate_compatibility.json').read_text())
    branch = json.loads((RATE / 'branch_order_bound.json').read_text())
    saved = {x['command_v']: x for x in inversion['observations']
             if x['branch'] == 'return' and x['command_v'] in (2., 4.)}
    ron, gc = (old_probe['plan'][k] for k in ('Ron_ohm', 'gamma_c_ev'))
    q2, q4 = [math.exp(-saved[A]['u']) for A in (2., 4.)]
    ell2, ell4 = [math.log(saved[A]['current_a']) + math.log((1-q)/q)
                  for A, q in ((2., q2), (4., q4))]
    lam = (ell4-ell2)/(saved[4.]['material_voltage_v']-saved[2.]['material_voltage_v'])
    log_j = ell2-lam*saved[2.]['material_voltage_v']
    jref = math.exp(log_j)
    plan = {
        'scope': 'One softonly seriescycle4 reversiblecapture seed; conditional training component, not complete1R/1T1R model or32cycleaggregate verification',
        'state_coordinate': 'q=exp(-u)=Gamma/(Gamma+gc); Gamma=gc*q/(1-q)',
        'rate_per_kappa': 'R(A,q)=v*((I/J)*(1-q)-exp(lambda*v)*q), I/v fromsame passive series circuit',
        'kappa_unit': 'V^-2', 'path_clock': 's=cumulative commandvoltage path, not identified seconds',
        'seed': {'lambda_per_v': lam, 'J_positive_a': jref, 'log_J_positive_a': log_j,
                 'Ron_ohm': ron, 'gamma_c_ev': gc, 'parallel_path_a_s': 0.,
                 'selection': 'Exactly two saved cycle4return2/4 inverse points, analytic zero-affinity alignment; no rounding or adaptive seed selection',
                 'interpretation': 'Two-point conditional algebraic calibration, not independent prediction, optimization, or measured activation barrier'},
        'new_inverse_problems': 0, 'new_ODE': 0, 'new_optimizations': 0,
        'maximum_physical_seeds': 1,
        'material_relative_envelopes': [0., .003],
        'envelope_scope': 'Inherited conditional numerical envelope around fixed spectrum, not measurednoise or a newglobal accuracy certificate; no recalibration oflambda/J',
        'grids': [
            {'name': 'nominal', 'q_cells': 64, 'command_step_v': .05, 'KCL_bisections': 48},
            {'name': 'refined', 'q_cells': 128, 'command_step_v': .025, 'KCL_bisections': 52}],
        'first_passage_command_range_v': [.2, 2.],
        'first_passage_q_range': 'q0upper=exp(-savedinitialu_lower),qtargetlower=exp(-savedout2u_upper) perenvelope; logarithmicqcells',
        'first_passage_bound': 'Ucell=max_over_all_A_rectangles upper_R; ifallpositive kappa>=sum(deltaq/Ucell)/1.8V. AnypositivewidthcellU<=0 blocksupcrossing. Backtracking/initialdecaydoesnotremoveanynecessarycrossing.',
        'rectangle_bounds': {
            'Imax': 'strongmaterialKCL(Ahigh,qhigh) currentupper',
            'Imin': 'weakmaterialKCL(Alow,qlow) currentlower',
            'vmax': 'weakmaterialKCL(Ahigh,qlow) voltageupper',
            'vmin': 'strongmaterialKCL(Alow,qhigh) voltagelower',
            'upper_R': 'vmax*Imax*(1-qlow)/J-vmin*exp(lambda*vmin)*qlow',
            'lower_R': 'vmin*Imin*(1-qhigh)/J-vmax*exp(lambda*vmax)*qhigh'},
        'sign_map_command_range_v': [.05, 5.],
        'sign_map_q_range': 'Sameq0upper toexp(-savedL2_return2_lowerstate), onlya finite diagnosticbox; doesnotcoverqbelowq0,nearzerocommandorallpossibletrajectories',
        'sign_policy': 'Positiveonlyifexpandedlower_R>0,negativeonlyifexpandedupper_R<0,elseunknown; upper_R>0aloneisnotpositive-rate evidence',
        'KCL_numerical_guard_v': 1e-12,
        'net_rate_rounding_guard': '1e-12*(positive_bound+negative_bound)+1e-18V, subtractedfromlower/addedtoupper',
        'numerical_scope': 'Analytical rectangle bounds plus rootbrackets anddeclaredfloating guards; not formal machine-interval proof or newtransport certificate',
        'KCL_residual_limit_a': 1e-12,
        'stability': 'Recompute two saved nodes only: B=(I/J)*(1-q)-exp(lambda*v)*q; R=v*B; Bq/exp(lambda*v)=chi-1/(1-q)+lambda*v_u atzeroaffinity; qdotJacobian=kappa*v*Bq. q=0/I=0 mayremainabsorbing.',
        'compute_budget_s': 60,
        'budget_start': 'Afterallinputs,sourcehashes,andplanwrite;beforestability/rectangle/budgetcalculations',
        'stop_policy': 'Preservefailures, noextra seed, newscale, inverse, adaptivegrid, ODE, or holdout'
    }
    HERE.mkdir(parents=True, exist_ok=True)
    plan_path = HERE / 'reversible_budget.plan.json'
    plan_path.write_text(json.dumps(plan, indent=2, allow_nan=False)+'\n')
    t, lookups = cached_transport(CACHE)
    assert sha(t.table_path) == '00d6d9ea0b1dfe855181e7016afc6dd5e23d169f83f19cfcedfc9d3b35f9e4e2'
    paths = [OLD/'DRIVE_STRUCTURE_INVERSION.json', OLD/'sinh_field_probe.json',
             RATE/'rate_compatibility.json', RATE/'branch_order_bound.json',
             t.table_path, *lookups, plan_path, Path(__file__),
             ROOT/'analysis/mechanism_selection_2026-11-03/SOFT_FEEDBACK_LOW_BIAS_BUDGET.py',
             ROOT/'analysis/ron_scope_2026-11-03/ron_scaling_review.py']
    sources = {str(p): sha(p) for p in paths}
    report = {'status': 'running_fixed_reversible_budget', 'plan': plan,
              'source_sha256': sources, 'cases': [], 'seed_stability': [],
              'new_inverse_problems': 0, 'new_ODE': 0, 'new_optimizations': 0,
              'KCL_node_problems': 0, 'KCL_vector_batches': 0}
    arrays = {}
    tx, ti = t.knots['set']
    table_vmin = math.exp(t.lv[0])
    started = time.monotonic()
    def timeout(*_):
        raise TimeoutError('Fixed60second reversible arithmetic budget exhausted; no automatic restart')
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 60)
    try:
        assert lam >= 0 and jref > 0
        for A in (2., 4.):
            x = saved[A]; q = math.exp(-x['u']); v = x['material_voltage_v']; i = x['current_a']
            chi = x['circuit_chi_minus_dlogI_du']
            vu = i*chi/x['g_t_s']
            va = x['g_t_s']/(x['g_h_s']+x['g_t_s'])
            affinity = math.log(i/jref)+math.log((1-q)/q)-lam*v
            normal = chi-1/(1-q)+lam*vu
            Bq = math.exp(lam*v)*normal
            report['seed_stability'].append({'command_v': A, 'q': q, 'local_voltage_v': v,
                'current_a': i, 'affinity': affinity,
                'R_per_kappa_v': v*((i/jref)*(1-q)-math.exp(lam*v)*q),
                'circuit_chi': chi, 'v_u': vu, 'v_A': va,
                'bracket_derivative_Bq_normalized_by_exp_lambda_v': normal,
                'qdot_jacobian_per_kappa': v*Bq,
                'local_linearized_class': 'repelling' if normal>0 else 'attracting' if normal<0 else 'neutral',
                'scope': 'Savedconditionalinverse-point localderivatives; notglobalstability or an actualtrajectory'})
        assert all(abs(x['affinity']) < 1e-12 for x in report['seed_stability'])

        def bracket_nodes(A, q, factor, steps):
            A,q=np.broadcast_arrays(np.asarray(A,float),np.asarray(q,float))
            assert np.all((q>0)&(q<1)) and np.all(A>=.05) and np.all(A<=5.)
            gamma=gc*q/(1-q)
            assert np.all(np.log(gamma)>=t.lg[0]) and np.all(np.log(gamma)<=t.lg[-1])
            lo=np.zeros_like(A);hi=A.copy()
            for _ in range(steps):
                mid=.5*(lo+hi)
                assert np.all(mid>=table_vmin)
                f=t.material(mid,gamma,ron/factor,'set')-np.interp(A-mid,tx,ti)
                high=f>0;hi=np.where(high,mid,hi);lo=np.where(high,lo,mid)
            center=.5*(lo+hi)
            residual=t.material(center,gamma,ron/factor,'set')-np.interp(A-center,tx,ti)
            max_residual=float(np.max(abs(residual)))
            assert max_residual<=plan['KCL_residual_limit_a']
            guard=plan['KCL_numerical_guard_v']
            vlo=np.maximum(0.,lo-guard);vhi=np.minimum(A,hi+guard)
            ilo=np.interp(A-vhi,tx,ti);ihi=np.interp(A-vlo,tx,ti)
            report['KCL_node_problems']+=int(A.size)
            report['KCL_vector_batches']+=1
            return {'Ilo':ilo,'Ihi':ihi,'vlo':vlo,'vhi':vhi,
                    'residual':max_residual,'root_width':float(np.max(hi-lo))}

        def rectangle_map(anodes,qnodes,eta,steps,prefix):
            strong=bracket_nodes(anodes[:,None],qnodes[None,:],1+eta,steps)
            weak=bracket_nodes(anodes[:,None],qnodes[None,:],1-eta,steps)
            Imax=strong['Ihi'][1:,1:];Imin=weak['Ilo'][:-1,:-1]
            vmax=weak['vhi'][1:,:-1];vmin=strong['vlo'][:-1,1:]
            qlo=qnodes[:-1][None,:];qhi=qnodes[1:][None,:]
            cupper=vmax*Imax*(1-qlo)/jref
            elower=vmin*np.exp(lam*vmin)*qlo
            clower=vmin*Imin*(1-qhi)/jref
            eupper=vmax*np.exp(lam*vmax)*qhi
            upper_guard=1e-12*(cupper+elower)+1e-18
            lower_guard=1e-12*(clower+eupper)+1e-18
            upper=cupper-elower+upper_guard
            lower=clower-eupper-lower_guard
            assert np.all(np.isfinite(lower)) and np.all(np.isfinite(upper)) and np.all(lower<=upper)
            arrays.update({prefix+'_command_nodes':anodes,prefix+'_q_nodes':qnodes,
                prefix+'_R_upper':upper,prefix+'_R_lower':lower,
                prefix+'_capture_upper':cupper,prefix+'_capture_lower':clower,
                prefix+'_emission_upper':eupper,prefix+'_emission_lower':elower,
                prefix+'_upper_rounding_guard':upper_guard,prefix+'_lower_rounding_guard':lower_guard,
                prefix+'_Imax':Imax,prefix+'_Imin':Imin,prefix+'_vmax':vmax,prefix+'_vmin':vmin})
            return lower,upper,{'maximum_node_KCL_residual_a':max(strong['residual'],weak['residual']),
                                'maximum_root_bracket_width_v':max(strong['root_width'],weak['root_width'])}

        for eta in plan['material_relative_envelopes']:
            old=next(c for c in rate['cases'] if c['envelope']==eta and c['grid']['name']=='refined')
            entry={x['name']:x for x in old['anchor_bounds']}
            q0=math.exp(-entry['initial_u_lower']['u'])
            target=math.exp(-entry['outward2_u_upper']['u'])
            old_return=next(c for c in branch['cases'] if c['material_envelope']==eta)
            qmapmax=math.exp(-old_return['L2_return2_lower_state'])
            assert 0<q0<target<qmapmax<1
            for grid in plan['grids']:
                prefix=('eta0' if eta==0 else 'eta003')+'_'+grid['name']
                da=grid['command_step_v'];nq=grid['q_cells'];steps=grid['KCL_bisections']
                anodes=np.linspace(.2,2.,round(1.8/da)+1)
                qnodes=np.geomspace(q0,target,nq+1)
                low,up,check=rectangle_map(anodes,qnodes,eta,steps,prefix+'_first')
                umax=up.max(axis=0);argmax=up.argmax(axis=0)
                blocked=np.flatnonzero(umax<=0)
                if len(blocked):
                    klower=None;integral=None
                else:
                    integral=float(np.sum(np.diff(qnodes)/umax))
                    klower=integral/1.8
                arrays[prefix+'_first_Ucell']=umax
                arrays[prefix+'_first_Ucell_argmax_command_cell']=argmax
                if not len(blocked):arrays[prefix+'_first_cell_passage_cost_per_kappa']=np.diff(qnodes)/umax
                fp={'q0_optimistic_upper':q0,'qtarget_optimistic_lower':target,
                    'command_path_available_v':1.8,'q_cells':nq,'command_cells':len(anodes)-1,
                    'barrier_cell_indices':blocked.tolist(),'upcrossing_excluded_by_rectangle_upper':bool(len(blocked)),
                    'kappa_first_passage_lower_per_v2':klower,'sum_deltaq_over_upper_inverse_v':integral,
                    'minimum_cell_upper_R_v':float(umax.min()),'maximum_cell_upper_R_v':float(umax.max()),
                    'scope':'Apositiveuppermeansnotexcludedonly; firstpassageboundallowsinitialdecayandbacktracking',
                    **check}
                amap=np.linspace(.05,5.,round(4.95/da)+1)
                qmap=np.geomspace(q0,qmapmax,nq+1)
                low,up,checkmap=rectangle_map(amap,qmap,eta,steps,prefix+'_sign')
                pos=low>0;neg=up<0;unknown=~(pos|neg)
                assert not np.any(pos&neg)
                arrays[prefix+'_sign_class']=np.where(pos,1,np.where(neg,-1,0)).astype(np.int8)
                sm={'q_range':[q0,qmapmax],'command_range_v':[.05,5.],
                    'positive_entire_rectangles':int(pos.sum()),'negative_entire_rectangles':int(neg.sum()),
                    'unknown_rectangles':int(unknown.sum()),'total_rectangles':int(up.size),
                    'minimum_positive_lower_margin_v':float(low[pos].min()) if pos.any() else None,
                    'minimum_negative_upper_margin_v':float((-up[neg]).min()) if neg.any() else None,
                    'not_a_trajectory':True,'not_exhaustive_state_or_voltage_domain':True,
                    **checkmap}
                report['cases'].append({'material_envelope':eta,'grid':grid,
                                        'first_passage':fp,'finite_box_sign_map':sm})
        report['status']='completed_one_seed_reversible_rate_budget_no_ODE_or_optimization'
    except Exception as exc:
        report.update(status='failed_or_partial_reversible_budget_preserved',error_type=type(exc).__name__,error=str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0.)
        report['compute_elapsed_s']=time.monotonic()-started
    destination.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(destination.with_suffix('.npz'),**arrays)
    write_report(report)
    print(json.dumps({k:report[k] for k in ('status','compute_elapsed_s','KCL_node_problems','KCL_vector_batches','seed_stability','cases')},indent=2))


def write_report(r):
    seed=r['plan']['seed']
    lines=['# 单一可逆捕获/发射种子的速率预算','',
        '只检验q′=κq v[(I/J+)(1−q)−exp(λv)q]。κq为一个常数率系数，未额外乘q；q=exp(−u)，Γ=Γc q/(1−q)。soft-only、同一固定谱与晶体管映射。',
        '',f'运行状态：{r["status"]}。λ={seed["lambda_per_v"]:.12g} V⁻¹，J+={seed["J_positive_a"]:.12g} A。',
        '参数来自同一cycle4已存返扫2/4V点的零亲和势代数对齐，属于两点条件校准，不是独立预测或微观势垒识别。本轮没有新反演、ODE、优化、保留集或额外种子。',
        '', '## 局部亲和势与稳定性','',
        '| 命令/V | q | 亲和势 | Bq/exp(λv) | 局部类别 |','|---|---:|---:|---:|---|']
    for x in r['seed_stability']:
        lines.append(f'| {x["command_v"]} | {x["q"]:.10g} | {x["affinity"]:.3g} | {x["bracket_derivative_Bq_normalized_by_exp_lambda_v"]:.10g} | {x["local_linearized_class"]} |')
    lines+=['', '4V排斥、2V吸引不能称为两处稳定停止态。有限速率能否经过排斥区尚未计算。这个候选允许q双向变化，但并未证明能重现返扫或裸器件多次恢复。',
        '', '## 可逆首达必要下界','',
        '每个A/q矩形按无源电路的单调性分别包住I、v、捕获及发射，再相减。取每个q格全部A格的最大净率上界Ucell；上穿所有q格至少需要ΣΔq/(κq Ucell)的命令路径。允许q先衰减、回退和多次穿越。任何正宽度格Ucell≤0都是上穿屏障，不能省略或夹成正数。',
        '', '| 材料数值包络 | 网格 | q0上界 | q目标下界 | κq必要下界/V⁻² | 被屏障排除 |','|---|---|---:|---:|---:|---|']
    for c in r['cases']:
        f=c['first_passage']
        lines.append(f'| {c["material_envelope"]} | {c["grid"]["name"]} | {f["q0_optimistic_upper"]:.10g} | {f["qtarget_optimistic_lower"]:.10g} | {f["kappa_first_passage_lower_per_v2"]} | {f["upcrossing_excluded_by_rectangle_upper"]} |')
    lines+=['', '有正上界只代表此必要检查未排除，不代表实际净率为正或目标可达。这里没有继承旧单调u的out2→ret4损失上界，也没有据此选择或拟合κq。',
        '', '## 有限矩形符号图','',
        '| 材料包络 | 网格 | 全格正 | 全格负 | 未知 | 总格数 |','|---|---|---:|---:|---:|---:|']
    for c in r['cases']:
        m=c['finite_box_sign_map']
        lines.append(f'| {c["material_envelope"]} | {c["grid"]["name"]} | {m["positive_entire_rectangles"]} | {m["negative_entire_rectangles"]} | {m["unknown_rectangles"]} | {m["total_rectangles"]} |')
    lines+=['', '分类由扩宽后的L>0或U<0给出，跨零者全部保留未知；不是将点网格符号冒充整个矩形证明。图只覆盖声明的0.05–5V及有限q范围，不是完整轨迹、相图或全状态域保证。',
        '', '0.3%是继承的固定材料表数值敏感性假设，不是仪器噪声或新全域精度证书。强/弱材料分别闭合KCL，端点取最有利q0上界和目标下界。根括区额外扩宽±1e−12V，净率含明确舍入缓冲；这些不冒充形式化机器区间证明。',
        '', f'计算{r["compute_elapsed_s"]:.6f}s，原预算60s；KCL节点问题{r["KCL_node_problems"]}，向量批次{r["KCL_vector_batches"]}。原500点评分及原20项聚合特征均未改变、也未在本预算中重新评分。',
        '', '零场率为零是有效路径律假设，不能声称已预测0.1V下小时级保持。q=0且I=0仍可吸收；后续实际仿真必须继承有限段入口态，不能每圈注入种子。准确统一1R/1T1R的目标仍未完成。']
    if 'error'in r:lines+=['',r['error_type']+': '+r['error']]
    (HERE/'reversible_budget.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
