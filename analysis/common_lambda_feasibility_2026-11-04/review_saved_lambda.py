"""Review saved lambda points only; no new trial points, roots, model imports."""
from pathlib import Path
from io import BytesIO
from collections import Counter
from datetime import datetime, timezone
import hashlib,json,math,time
import numpy as np
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'empirical_integral_2026-11-04'
POS=('set4_outward_0p2_to_0p8','set4_outward_0p8_to_1p6')
NEG=('reset4_outward_0p2_to_0p8','reset4_outward_0p8_to_1p6')
checks=[]; inputs={}; started=time.monotonic()
def sha(b): return hashlib.sha256(b).hexdigest()
def read(p,purpose):
    p=Path(p);b=p.read_bytes();key=str(p)
    inputs.setdefault(key,{'sha256':sha(b),'bytes':len(b),'purposes':[]})
    if purpose not in inputs[key]['purposes']:inputs[key]['purposes'].append(purpose)
    return b
def js(p,purpose):return json.loads(read(p,purpose))
def check(name,ok,**detail):checks.append({'name':name,'passed':bool(ok),**detail})
def eq(name,a,b):check(name,np.array_equal(a,b))
def near(name,a,b,rtol=1e-10,atol=2e-11):
    a,b=np.asarray(a),np.asarray(b);same=a.shape==b.shape
    check(name,same and np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.allclose(a,b,rtol=rtol,atol=atol),
          max_abs_difference=float(np.max(abs(a-b))) if same and a.size else 0.,rtol=rtol,atol=atol)
r=js(HERE/'SAVED_LAMBDA_FEASIBILITY.json','terminal result and complete function traces')
old=js(OLD/'empirical_integral_diagnostic.json','saved lambda0, window B/y and array identities')
bd=js(OLD/'LAMBDA_NECESSARY_BOUND.json','predeclared boundary brackets')
theory=js(HERE/'H_MONOTONICITY_CHECK.json','independently reviewed uniqueness premises')
read(HERE/'H_MONOTONICITY_REVIEW.md','independent theorem review provenance')
freeze=js(HERE/'SOURCE_FREEZE.json','run freeze manifest')
check('manifest hash',inputs[str(HERE/'SOURCE_FREEZE.json')]['sha256']==r['source_freeze']['manifest_sha256'])
for path,h in r['source_freeze']['source_sha256'].items():check('frozen '+path,sha(read(path,'source/input hash'))==h)
check('reviewed runner',r['source_freeze']['source_sha256'][str(HERE/'saved_lambda_feasibility.py')]=='041f7e5cbcd73694102a98e3b1ea79609208eced5d21b341099d561fb1f4a00e')
check('terminal complete',r['status']=='completed_finite_saved_node_diagnostic' and r['sources_unchanged'])
for k in ('new_ODEs','new_state_inversions','new_static_circuit_roots','new_nonlinear_forward_fits','new_quadrature_nodes','full_unified_model_achieved','unconditional_mechanism_rejection'):check('scope '+k,r[k] in (0,False))
windows={};decoded=[]
with np.load(BytesIO(read(OLD/'empirical_integral_arrays.npz','144 saved v/F/weight members')),allow_pickle=False) as a:
    for row in old['window_variants']:
        pre=row['array_prefix'];keys=[pre+k for k in ('local_voltage_v','F','weights_path_v')];decoded+=keys
        v=abs(a[keys[0]]);mass=a[keys[1]]*a[keys[2]]
        check(row['variant_id']+' positive saved support',np.all(np.isfinite(v)) and np.all(np.isfinite(mass)) and np.all(v>0) and np.all(mass>0))
        windows[(row['interpolant'],row['gauss_order'],row['window_id'])]={'v':v,'mass':mass,'B':row['B_a_v2'],'y':row['delta_u']}
check('144 unique NPZ members',len(decoded)==len(set(decoded))==144)
bounds={(x['interpolant'],x['gauss_order']):x for x in bd['configurations']}
theories={(x['interpolant'],x['gauss_order']):x for x in theory['configurations']}
totals=Counter();findings=[];closures=[];saved_lambdas={};nsums=nsolves=0
for c in r['configurations']:
    key=(c['interpolant'],c['gauss_order']);name=f'{key[0]}_g{key[1]}'
    local={w:z for (i,n,w),z in windows.items() if (i,n)==key};boundary=bounds[key]['pairs']['reset_outward']
    trace={x['lambda_per_v']:x for x in c['H_evaluation_trace']}
    check(name+' complete 74 H points',c['status']=='completed_finite_saved_node_diagnostic' and len(trace)==len(c['H_evaluation_trace'])==74)
    requests=[]
    for root in c['boundary_roots'].values():
        for e in root['evaluations']:requests.extend((w,e['lambda']) for w in NEG)
    for x in c['H_evaluation_trace']:requests.extend((w,x['lambda_per_v']) for w in POS+NEG)
    for x in c['candidates']:requests.extend((w,x['lambda_per_v']) for w in sorted(local))
    allowed=set(requests);cache={};saved_lambdas[name]=sorted({lam for w,lam in allowed})
    def integral(w,lam):
        global nsums
        assert (w,lam) in allowed,'No new lambda request is allowed in this review'
        if (w,lam) not in cache:
            z=local[w];A=math.fsum((z['mass']*np.exp((lam-old['fixed_material']['lambda_per_v'])*z['v'])).tolist())
            cache[(w,lam)]={'A':A,'log_A':math.log(A),'B':z['B'],'y':z['y']};nsums+=1
        return cache[(w,lam)]
    count=Counter({'new_saved_window_sum_evaluations':len(allowed),'saved_sum_cache_hits':len(requests)-len(allowed),
        'saved_node_terms_reweighted':sum(len(local[w]['v']) for w,lam in allowed),'H_parameter_point_evaluations':len(trace),
        'linear_identification_attempts':2*len(trace),'two_by_two_linear_solve_attempts':2*len(trace),'two_by_two_linear_solves_returned':2*len(trace),
        'candidate_checks':len(c['candidates']),'candidate_window_closure_pairs':12*len(c['candidates']),
        'parameter_scalar_root_problems_attempted':2+len(c['H_root_attempts']),'parameter_scalar_roots_completed':2+len(c['H_root_attempts'])})
    for kind,root in c['boundary_roots'].items():
        label=name+' R='+kind;history=root['evaluations']
        ex=[boundary['strict_necessary_lambda_lower_bound_jensen_per_v'],boundary['lambda_above_this_suffices_for_R_greater_than_S_per_v']] if kind=='S' else [boundary['T_crossing_lambda_lower_bound_jensen_per_v'],boundary['strict_necessary_lambda_upper_bound_jensen_per_v']]
        eq(label+' initial bounds',root['initial_bracket'],[ex[0]-1e-10,ex[1]+1e-10])
        near(label+' threshold',root['threshold'],boundary['S_B_high_over_low'] if kind=='S' else boundary['T_y_high_over_low'])
        eq(label+' endpoint identities',[e['lambda'] for e in history[:2]],root['initial_bracket'])
        lo,hi=root['initial_bracket'];fl,fh=[e['f'] for e in history[:2]];check(label+' signs',fl<=0<=fh)
        for j,e in enumerate(history):
            pair=[integral(w,e['lambda']) for w in NEG];f=pair[1]['log_A']-pair[0]['log_A']-math.log(root['threshold'])
            near(label+' function '+str(j),e['f'],f,atol=2e-14)
            if j>=2:
                eq(label+' saved midpoint '+str(j),e['lambda'],(lo+hi)/2)
                if e['f']==0:lo=hi=e['lambda'];fl=fh=0.
                elif e['f']<0:lo,fl=e['lambda'],e['f']
                else:hi,fh=e['lambda'],e['f']
        eq(label+' final bracket',root['bracket'],[lo,hi]);eq(label+' final f',root['endpoint_values'],[fl,fh])
        check(label+' finite budget',root['completed'] and len(history)-2<=48 and root['numerical_wall_s']<2)
        reason=root['stopping_reason']
        check(label+' stopping state',(reason=='iteration_limit_48' and len(history)-2==48) or (reason=='floating_exact_zero_midpoint' and history[-1]['f']==0 and lo==hi) or (reason=='floating_midpoint_stagnation' and (lo+hi)/2 in (lo,hi)))
        count['reset_boundary_scalar_function_evaluations']+=len(history);count['reset_boundary_bisection_steps']+=len(history)-2
    L=c['boundary_roots']['S']['bracket'][1];U=c['boundary_roots']['T']['bracket'][0]
    eq(name+' inner interval',c['conservative_open_positive_interval'],[L,U])
    eq(name+' 31 declared grid',[g['lambda_per_v'] for g in c['grid']],[L+(U-L)*j/32 for j in range(1,32)])
    eq(name+' grid counts',c['grid_plan'],{'planned':31,'attempted':31,'completed':31})
    recomputed={}
    for lam,x in trace.items():
        co={}
        for side,ids,names in [('positive',POS,('kappa','r')),('negative',NEG,('alpha','beta'))]:
            s=x[side];rs=[integral(w,lam) for w in ids];X=np.array([[z['A'],-z['B']] for z in rs]);y=np.array([z['y'] for z in rs]);norm=np.linalg.norm(X,axis=0);N=X/norm
            sv=np.linalg.svd(N,compute_uv=False);rank=int(np.sum(sv>2*np.finfo(float).eps*sv[0]));coef=np.linalg.solve(N,y)/norm;nsolves+=1;co[side]=coef
            label=f'{name} {lam.hex()} {side}'
            near(label+' X',s['raw_X_A_minus_B'],X);near(label+' y',s['y'],y)
            for sw,z,w in zip(s['window_integrals'],rs,ids):
                check(label+' identity '+w,sw['window_id']==w and sw['lambda_per_v']==lam)
                near(label+' integral '+w,[sw[k] for k in ('A','log_A','B','y')],[z[k] for k in ('A','log_A','B','y')])
            check(label+' rank',rank==s['numerical_rank']==2)
            near(label+' condition',s['column_normalized_condition_2'],sv[0]/sv[-1])
            near(label+' coefficients',[s['coefficients'][k] for k in names],coef)
            check(label+' positive',s['available'] and s['all_positive'] and np.all(coef>0))
        k,rr=co['positive'];al,be=co['negative'];H=math.log(al)+math.log(be)-math.log(k)-math.log(rr);bf=.5*math.log(al/k);bg=.5*math.log(rr/be)
        near(name+' H '+lam.hex(),x['H'],H);near(name+' b '+lam.hex(),[x['b_F'],x['b_G']],[bf,bg])
        R=integral(NEG[1],lam)['A']/integral(NEG[0],lam)['A']
        check(name+' strict RESET interior '+lam.hex(),boundary['S_B_high_over_low']<R<boundary['T_y_high_over_low'])
        recomputed[lam]={'positive':co['positive'],'negative':co['negative'],'H':H,'bf':bf,'bg':bg}
    grid=c['grid'];check(name+' all grid valid',all(g['status']=='completed' and g['all_positive'] and L<g['lambda_per_v']<U for g in grid))
    check(name+' decreasing grid H',np.all(np.diff([g['H'] for g in grid])<0))
    brackets=[(a['lambda_per_v'],b['lambda_per_v']) for a,b in zip(grid,grid[1:]) if a['H']*b['H']<0]
    check(name+' one H bracket',len(brackets)==c['H_sign_change_brackets_found']==len(c['H_root_attempts'])==1)
    check(name+' no omitted holes/brackets',c['nonpositive_adjacent_pairs_not_bridged']==[] and c['H_sign_change_brackets_not_attempted_due_to_cap']==[])
    for root in c['H_root_attempts']:
        label=name+' Hroot';lo,hi=root['initial_bracket'];eq(label+' grid bracket',root['initial_bracket'],brackets[root['bracket_index']])
        fl,fh=trace[lo]['H'],trace[hi]['H'];eq(label+' initial H',root['verified_cached_endpoint_H'],[fl,fh]);check(label+' opposite signs',fl>0>fh)
        for j,e in enumerate(root['evaluations']):
            eq(label+' saved midpoint '+str(j),e['lambda'],(lo+hi)/2);eq(label+' trace H '+str(j),e['H'],trace[e['lambda']]['H']);check(label+' positive '+str(j),e['all_positive'])
            if e['H']==0:lo=hi=e['lambda'];fl=fh=0.
            elif math.copysign(1,e['H'])==math.copysign(1,fl):lo,fl=e['lambda'],e['H']
            else:hi,fh=e['lambda'],e['H']
        eq(label+' final bracket',root['bracket'],[lo,hi]);eq(label+' final H',root['final_endpoint_H'],[fl,fh]);eq(label+' retained bracket',root['last_retained_bracket'],[lo,hi])
        eq(label+' candidate endpoint',root['candidate_lambda'],lo if abs(fl)<=abs(fh) else hi)
        check(label+' H residual',abs(root['candidate_H'])<=1e-8)
        check(label+' stagnation',root['stopping_reason']=='floating_midpoint_stagnation' and (lo+hi)/2 in (lo,hi))
        check(label+' budget',root['completed'] and len(root['evaluations'])<=48 and root['numerical_wall_s']<2)
        count['H_bisection_steps']+=len(root['evaluations'])
    for a in c['candidates']:
        lam=a['lambda_per_v'];x=recomputed[lam];k,rr=x['positive'];al,be=x['negative'];b=(x['bf']+x['bg'])/2;project=np.array([k*math.exp(2*b),rr*math.exp(-2*b)])
        check(name+' candidate complete',a['status']=='completed' and all(z['status']=='completed' for z in a['window_plan']))
        near(name+' direct tuple',[a['direct_common_tuple'][k] for k in ('lambda_per_v','kappa','J_a','b')],[lam,k,k/rr,b])
        near(name+' projected coefficients',[a['common_b_negative_coefficients'][k] for k in ('alpha','beta')],project)
        near(name+' b projection',a['b_common_midpoint'],b)
        check(name+' b domain fails',x['bf']<0 and x['bg']<0 and not a['both_b_estimates_in_0_to_ln10'] and not a['b_common_in_0_to_ln10'] and not a['eligible_for_continuous_forward_check'])
        check(name+' product relation passes',a['numeric_shared_relation_satisfied'] and abs(a['H'])<=1e-8 and abs(a['b_F']-a['b_G'])<=5e-9)
        cc=[]
        for typ,listkey in [('separate','separate_polarity_coefficients_closure'),('common_b','forced_common_b_closure')]:
            check(name+' '+typ+' 12 windows',len(a[listkey])==12)
            for row in a[listkey]:
                w=row['window_id'];z=integral(w,lam);coef=(project if typ=='common_b' else np.array([al,be])) if w.startswith('reset') else np.array([k,rr]);first,second=coef*[z['A'],z['B']];pred=first-second;E=z['y']-pred;den=abs(z['y'])+abs(first)+abs(second)
                expected={'A':z['A'],'B':z['B'],'delta_u':z['y'],'recovery_term':first,'capture_term':second,'predicted_delta_u':pred,'E':E,'normalization_denominator':den,'normalized_E':E/den}
                for metric,value in expected.items():near(name+' '+typ+' '+w+' '+metric,row[metric],value)
                entry={'configuration':name,'type':typ,'window_id':w,'identification':w in POS+NEG,'delta_u':z['y'],'predicted_delta_u':float(pred),'E':float(E),'normalized_E':float(E/den),'opposite_sign':bool(z['y']*pred<0)}
                closures.append(entry);cc.append(entry)
            for used,lab in [(True,'identification_windows'),(False,'remaining_eight_training_windows')]:
                maximum=max(abs(z['normalized_E']) for z in cc if z['type']==typ and z['identification']==used)
                near(name+' '+typ+' closure maximum '+lab,a['closure_max_abs_normalized_E'][typ][lab],maximum)
        remain=[z for z in cc if z['type']=='common_b' and not z['identification']]
        check(name+' three remaining direction errors',sum(z['opposite_sign'] for z in remain)==3)
        findings.append({'configuration':name,'lambda_per_v':lam,'b':float(b),'kappa':float(k),'J_a':float(k/rr),'H_saved':a['H'],'b_domain_passed':False,'eligible_for_continuous_forward_check':False,'remaining_common_max_abs_normalized_E':max(abs(z['normalized_E']) for z in remain),'remaining_sign_error_windows':[z['window_id'] for z in remain if z['opposite_sign']]})
    check(name+' uniqueness consistency',theories[key]['premises_satisfied'] and theories[key]['H_derivative_upper_v']<0 and len(c['candidates'])==1)
    for metric,value in count.items():check(name+' ledger '+metric,c['work_counts'].get(metric,0)==value,expected=value,actual=c['work_counts'].get(metric,0))
    check(name+' configuration budget',c['numerical_wall_s']<15);totals.update(count)
    check(name+' all saved sum requests checked',len(cache)==len(allowed))
for metric,value in totals.items():check('total ledger '+metric,r['work_counts'].get(metric,0)==value,expected=value,actual=r['work_counts'].get(metric,0))
check('592 systems resolved',nsolves==592);check('2014 saved sums rechecked',nsums==2014)
check('12 saved roots',totals['parameter_scalar_roots_completed']==12)
check('399 R evaluations 296 H contexts',totals['reset_boundary_scalar_function_evaluations']==399 and totals['H_parameter_point_evaluations']==296)
check('total scalar evaluations',r['parameter_scalar_function_evaluations']==695)
near('numerical time sum',r['numerical_wall_s'],sum(c['numerical_wall_s'] for c in r['configurations']))
check('total budget',r['numerical_wall_s']<60)
for path,z in inputs.items():check('unchanged '+path,sha(Path(path).read_bytes())==z['sha256'])
failed=[z for z in checks if not z['passed']]
report={'status':'passed_saved_lambda_review' if not failed else 'failed_saved_lambda_review','created_utc':datetime.now(timezone.utc).isoformat(),'script_sha256':sha(Path(__file__).read_bytes()),
'check_count':len(checks),'passed_count':len(checks)-len(failed),'failed_count':len(failed),'failures':failed,'new_lambda_trial_points':0,'new_parameter_root_searches':0,'new_model_imports':0,'new_model_calls':0,'new_inversions':0,'new_ODEs':0,
'saved_window_sums_rechecked':nsums,'saved_linear_systems_resolved':nsolves,'saved_root_traces_checked':12,'inputs_read':inputs,'decoded_npz_members':decoded,'saved_lambda_values_by_configuration':saved_lambdas,'reconstructed_ledger':dict(totals),'candidate_findings':findings,'closure_rechecks':closures,
'policy':'Arithmetic is restricted to exact lambda values already saved in boundary/H/candidate records; compensated sums and 592 saved 2x2 systems are recomputed, not new searches.',
'floating_bracket_caveat':'Saved sign decisions/midpoints are checked. Last-bit residual signs can depend on equivalent summation order; the saved machine-precision interval is not a rigorous real-arithmetic enclosure.',
'interpretation':'All four finite-sum configurations have one shared-product root consistent with monotonicity, but b<0 outside the predeclared [0,ln10] family. Even retaining negative b diagnostically, three of eight remaining windows predict delta_u of opposite sign. No candidate passes forward eligibility. This does not exclude all shared physical models or certify unknown noise, clocks, readout or between-sample trajectories.',
'wall_s':time.monotonic()-started,'checks':checks}
out=HERE/'SAVED_LAMBDA_REVIEW.json';md=HERE/'SAVED_LAMBDA_REVIEW.md'
if out.exists() or md.exists():raise FileExistsError('Review files exist; refusing overwrite')
out.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
lines=['# 共同 λ 保存结果复核','',f'**{report["passed_count"]}/{report["check_count"]} 项检查通过，{report["failed_count"]} 项失败。**','',
'只在已记录的 λ 上复算保存正和、线性系统和闭合；没有新增 λ 试点、根搜索、模型调用、反演或 ODE。',
'核对8个R边界、4个H根的全部保存函数迹/二分决策/终点，四配置各31个内部点，592次列归一识别，2014个唯一保存窗口求和请求及账本。四候选的独立系数与共同b两套12窗闭合均已复算。','',
'| 配置 | 乘积根 λ / V⁻¹ | 共同 b | 剩余八窗最大归一化闭合差 | 前向资格 |','|---|---:|---:|---:|---|']
for z in findings:lines.append(f'| {z["configuration"]} | {z["lambda_per_v"]:.10f} | {z["b"]:.9f} | {z["remaining_common_max_abs_normalized_E"]:.9f} | 不满足 |')
lines+=['','**四个根均满足数值乘积关系，但 b 全部为负，超出事先限定的 [0,ln10]。** 结果与已复核的H单调性及唯一性理论一致；本次没有符合预定前向资格的候选。b范围是本次模型族的预设约束，不能称作独立测得的物理定律。','',
'即使保留负b作诊断，其余八窗仍不闭合。四配置都在RESET4两个返扫窗和SET5低压返扫窗出现方向相反。PCHIP/Gauss16具体为：','',
'| 窗口 | 重建 Δu | 共同模型积分预测 Δu |','|---|---:|---:|']
for z in closures:
    if z['configuration']=='pchip_g16' and z['type']=='common_b' and z['opposite_sign']:lines.append(f'| {z["window_id"]} | {z["delta_u"]:.9g} | {z["predicted_delta_u"]:.9g} |')
lines+=['','归一化闭合差为 E/(|Δu|+|恢复项|+|捕获项|)，不是电流百分比误差或RMSE。未知测量噪声、时钟、读出和点间轨迹仍未解决；结论不能推广成所有共同物理模型不可能，也不代替完整1R/1T1R验收。',
'浮点括区的保存符号决策已核对；末位符号可能受求和次序影响，机器精度括宽不构成严格数学区间证书。','',
f'实际读取{len(inputs)}个唯一输入文件；NPZ仅解码48窗口变体的局部电压/F/权重，共144个成员。所有输入在复核末尾哈希未变。',
f'结果JSON SHA256：{inputs[str(HERE/"SAVED_LAMBDA_FEASIBILITY.json")]["sha256"]}',
f'来源NPZ SHA256：{inputs[str(OLD/"empirical_integral_arrays.npz")]["sha256"]}',
'逐文件用途/哈希、保存λ清单、账本、闭合及全部检查见 SAVED_LAMBDA_REVIEW.json。','']
if failed:lines+=['失败项：']+['- '+z['name'] for z in failed]
md.write_text('\n'.join(lines))
print(json.dumps({k:report[k] for k in ('status','check_count','passed_count','failed_count','saved_window_sums_rechecked','saved_linear_systems_resolved','wall_s')},indent=2))

