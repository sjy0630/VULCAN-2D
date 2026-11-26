"""Independent arithmetic audit of saved integral artifacts; no model imports.

No new roots, inversions, ODEs, parameter searches, or material evaluations.
Only saved two-column systems are resolved; transistor lookup interpolation at
saved root outputs independently checks KCL. Frozen inputs are read-only.
"""
from collections import Counter
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import hashlib
import json
import math
import time
import numpy as np

HERE = Path(__file__).resolve().parent
EXPECTED_RUNNER = '6ca987d6db74fc8af96ace80288a69a0db910dbfcbf79ecd200fdc95017e7a16'
checks, inputs, decoded = [], {}, {}
started = time.monotonic()

def sha(data): return hashlib.sha256(data).hexdigest()
def read(path, purpose):
    path = Path(path); data = path.read_bytes(); key = str(path)
    inputs.setdefault(key, {'sha256':sha(data),'bytes':len(data),'purposes':[]})
    if purpose not in inputs[key]['purposes']: inputs[key]['purposes'].append(purpose)
    return data
def jread(path,purpose): return json.loads(read(path,purpose))
def archive(path,purpose,keys=None):
    with np.load(BytesIO(read(path,purpose)),allow_pickle=False) as a:
        names = a.files if keys is None else keys
        decoded[str(path)] = list(names)
        return {k:a[k].copy() for k in names}
def check(name, ok, **details):
    checks.append({'name':name,'passed':bool(ok),**details})
def close(name,a,b,rtol=2e-12,atol=1e-15):
    a,b=np.asarray(a),np.asarray(b)
    ok = a.shape==b.shape and np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.allclose(a,b,rtol=rtol,atol=atol)
    diff=float(np.max(abs(a-b))) if a.shape==b.shape and a.size else 0.
    check(name,ok,max_absolute_difference=diff,relative_tolerance=rtol,absolute_tolerance=atol)
def exact(name,a,b): check(name,np.array_equal(a,b))

r = jread(HERE/'empirical_integral_diagnostic.json','terminal saved diagnostic and ledgers')
freeze = jread(HERE/'SOURCE_FREEZE.json','pre-execution frozen hashes')
audit = jread(HERE/'WINDOW_DATA_AUDIT.json','fixed native identities and masks')
data = archive(HERE/'windows_input.npz','saved input identities and native endpoints')
arrays = archive(HERE/'empirical_integral_arrays.npz','all saved integration and root/material-trace arrays')
for path,digest in r['source_sha256'].items():
    actual=sha(read(path,'hash-only unchanged frozen source check'))
    check('source:'+path,actual==digest,expected_sha256=digest,actual_sha256=actual)
for path,digest in freeze['source_sha256'].items():
    check('freeze_agrees_with_result:'+path,r['source_sha256'].get(path)==digest)
check('runner_final_reviewed_sha',inputs[str(HERE/'empirical_integral_diagnostic.py')]['sha256']==EXPECTED_RUNNER)
check('saved_npz_hash',inputs[str(HERE/'empirical_integral_arrays.npz')]['sha256']==r['all_numerical_arrays']['sha256'])
check('terminal_complete',r['terminal_result'] and r['status']=='completed_conditional_interpolation_diagnostic' and r['source_files_unchanged'])
for key in ('new_ODEs','new_nonlinear_forward_fits','new_b_scans','initial_state_reanchoring','extension_accessed','original_holdout_accessed','full_unified_model_achieved','unconditional_mechanism_rejection','experimental_noise_certificate','true_between_sample_trajectory_certificate'):
    check('claim_scope:'+key,r[key] in (0,False))

# Only domains are decoded from the material archive, not its log-current table.
cache=Path('/private/tmp/vulcan-runtime-cache')
domain=archive(cache/'transport_g010_v0025.npz','material coordinate domains only',['log_gamma','log_voltage'])
lg,lv=domain['log_gamma'],domain['log_voltage']
gc=r['fixed_material']['Gamma_c_ev']; lam=r['fixed_material']['lambda_per_v']

def transistor_table(filename):
    payload=jread(cache/filename,'frozen transistor lookup arithmetic at saved voltages')
    curves=[]
    for raw in payload['curves']:
        v=np.asarray(raw['drain_voltage_v']); i=np.asarray(raw['drain_current_a'])
        apex=int(np.argmax(v)); last=len(v)-1-int(np.argmax(v[::-1])); up=v[:apex+1]
        vals=i[:apex+1] if apex==len(v)-1 else .5*(i[:apex+1]+np.interp(up,v[last:][::-1],i[last:][::-1]))
        vals=np.maximum(vals,0.); vals[0]=0.; vals=np.maximum.accumulate(vals)
        curves.append((float(raw['gate_voltage_v']),up,vals))
    curves.sort(key=lambda x:x[0]); gates=np.array([c[0] for c in curves])
    x=np.unique(np.concatenate([c[1] for c in curves])); h=int(np.searchsorted(gates,1.1,side='right')); lo=h-1
    low=np.interp(x,curves[lo][1],curves[lo][2]); high=np.interp(x,curves[h][1],curves[h][2])
    y=low+(1.1-gates[lo])/(gates[h]-gates[lo])*(high-low)
    return x,y
transistors={1:transistor_table('fig2b_transistor_lookup.json'),-1:transistor_table('bookoutput2_reverse_transistor_lookup.json')}

roots=arrays['static_root_trace']; material=arrays['inverse_material_trace']
ledger=r['work_ledger']; counts=ledger['counts']; contexts={c['context_id']:c for c in ledger['contexts']}
expected_counts={'conditional_readpoint_inversions_attempted':426,'conditional_readpoint_inversions_completed':426,
    'quadrature_window_variants_attempted':48,'quadrature_window_variants_completed':48,
    'quadrature_nodes_attempted':20160,'quadrature_nodes_completed':20160,
    'static_root_attempts':21108,'static_root_returns':21108,'static_root_validated':21108,
    'inverse_material_evaluations_attempted':22152,'inverse_material_evaluations_returned':22152,
    'inverse_material_bisection_iterations_attempted':20448,'inverse_material_bisection_iterations_returned':20448,
    'merged_branch_PCHIP_constructions':6,'new_linear_identification_attempts':8,
    'new_2x2_linear_solves_attempted':8,'new_2x2_linear_solves_returned':8,'new_linear_identifications_completed':8}
for key,n in expected_counts.items():check('count:'+key,counts.get(key)==n)
check('no_ledger_errors',ledger['errors']==[])
exact('root_shape',roots.shape,(21108,12)); exact('material_shape',material.shape,(22152,8))
check('root_finite_and_returned_validated',np.all(np.isfinite(roots)) and np.all(roots[:,5:7]==1.))
check('material_finite_and_returned',np.all(np.isfinite(material)) and np.all(material[:,6]==1.))
exact('context_ids',sorted(contexts),list(range(474)))

# Independent domain, passivity, KVL and transistor-side KCL arithmetic at all saved roots.
cmd,u,current,local,vt,kcl,kvl=(roots[:,j] for j in (3,4,7,8,9,10,11))
loggamma=np.log(gc)-np.log(np.expm1(u))
check('all_root_state_and_material_domains',np.all((u>0)&(loggamma>=lg[0])&(loggamma<=lg[-1])&(abs(local)>=math.exp(lv[0]))&(abs(local)<=math.exp(lv[-1]))))
check('all_root_sign_and_passivity',np.all((cmd*current>0)&(cmd*local>0)&(cmd*vt>=0)&(current*local>0)&(current*vt>=0)))
close('all_root_KVL_direct',cmd-local-vt,kvl,rtol=0.,atol=1e-15)
check('all_root_KVL_limit',np.max(abs(kvl))<=1e-12 and np.max(abs(cmd-local-vt))<=1e-12)
recomputed_kcl=np.empty(len(roots))
for sign,(tx,ti) in transistors.items():
    select=np.sign(cmd)==sign
    check('transistor_domain:'+str(sign),np.all((abs(vt[select])>=tx[0])&(abs(vt[select])<=tx[-1])))
    recomputed_kcl[select]=current[select]-sign*np.interp(abs(vt[select]),tx,ti)
close('all_root_KCL_independent_lookup',recomputed_kcl,kcl,rtol=1e-5,atol=2e-18)
check('all_root_KCL_limit',np.max(abs(kcl))<=1e-13 and np.max(abs(recomputed_kcl))<=1e-13)
check('material_trace_domains',np.all((material[:,3]>=math.exp(lv[0]))&(material[:,3]<=math.exp(lv[-1]))&(material[:,4]>=lg[0])&(material[:,4]<=lg[-1])&(material[:,7]>0)))

points={(p['phase'],p['native_index']):p for p in r['readpoints']}
check('426_unique_readpoints',len(points)==len(r['readpoints'])==426)
inverse_amplifications=[]
for (phase,index),p in points.items():
    label=p['label']; inv=p['inverse']; cid=p['context_id']; tr=roots[roots[:,0]==cid]; mt=material[material[:,0]==cid]
    check(label+':completed_identity',p['status']=='completed' and contexts[cid]['label']==label and p['original_scoring_eligible'])
    exact(label+':original_command_current',[p['command_voltage_v'],p['observed_current_original_column_a'],p['path_voltage_distance_v']],
          [data[phase+'_command_voltage_v'][index],data[phase+'_observed_current_a'][index],data[phase+'_path_voltage_s'][index]])
    exact(label+':inverse_context_rows',tr[:,1:3],[[-1,0],[-1,1]])
    exact(label+':52_saved_material_calls',mt[:,1],np.arange(1,53))
    exact(label+':material_iteration_kinds',mt[:,2],np.r_[np.zeros(2),np.ones(48),np.full(2,2)])
    sign=1 if p['command_voltage_v']>0 else -1
    exact(label+':material_polarity',mt[:,5],np.full(52,sign))
    close(label+':saved_material_local',mt[:,3],np.full(52,inv['local_voltage_magnitude_v']),atol=1e-15)
    close(label+':u_gamma_map',inv['direct_u'],math.log1p(gc*math.exp(-inv['direct_log_gamma'])),atol=1e-14)
    close(label+':gamma_and_Z',[inv['direct_gamma_ev'],inv['direct_Z']],[math.exp(inv['direct_log_gamma']),math.exp(inv['direct_u'])])
    check(label+':saved_inverse_agreement',abs(inv['direct_u']-inv['bisect_u'])<=1e-8)
    target=math.log(inv['observed_current_magnitude_a']); low,high=float(lg[0]),float(lg[-1]); trace_ok=True
    trace_ok &= mt[0,4]==low and mt[1,4]==high and math.log(mt[0,7])<=target<=math.log(mt[1,7])
    for m in mt[2:50]:
        midpoint=(low+high)/2; trace_ok &= math.isclose(m[4],midpoint,abs_tol=1e-14,rel_tol=0.)
        if math.log(m[7])-target>=0: high=midpoint
        else: low=midpoint
    check(label+':saved_bisection_trace_arithmetic',trace_ok)
    close(label+':saved_final_bracket',inv['bisect_log_gamma_bracket'],[low,high],atol=1e-14)
    close(label+':final_bracket_material_positions',mt[-2:,4],[low,high],atol=1e-14)
    close(label+':bracket_residuals',inv['bisect_log_current_residual_bracket'],np.log(mt[-2:,7])-target,atol=1e-14)
    for mode,k in [('fast',0),('strict_bisection42',1)]:
        pt=inv['circuit_rechecks'][mode]
        close(label+':saved_circuit_'+mode,tr[k,3:5],[pt['command_voltage_v'],inv['direct_u']],atol=0.)
        close(label+':saved_output_'+mode,tr[k,7:],[pt[x] for x in ('current_a','local_voltage_v','transistor_voltage_v','KCL_residual_a','KVL_residual_v')],atol=0.)
        error=abs(math.log10(abs(pt['current_a'])/inv['observed_current_magnitude_a']))
        check(label+':observed_backcheck_'+mode,error<=5e-7)
    check(label+':two_root_agreement',abs(math.log10(abs(tr[0,7]/tr[1,7])))<=5e-7 and abs(tr[0,8]-tr[1,8])<=1e-9)
    sens=inv['inverse_sensitivity']; check(label+':chi_not_selection',not sens['used_for_point_selection'] and not sens['measurement_uncertainty_interval'])
    for j,cell in enumerate(sens.get('incident_cells',[])):
        if cell['defined']:
            chi=math.log(10)*(-math.expm1(-inv['direct_u']))*(1+cell['L_v']*inv['observed_current_magnitude_a']*cell['dVT_dI_v_per_a']/inv['local_voltage_magnitude_v'])/cell['L_Gamma']
            close(label+':chi_formula_'+str(j),cell['chi_abs_du_per_decade'],chi)
            inverse_amplifications.append(chi)

windows={w['window_id']:w for w in audit['windows']}; integrated={}; node_total=strict_total=0
for row in r['window_variants']:
    name=row['variant_id']; pre=row['array_prefix']; phase=row['phase']; order=row['gauss_order']; idx=np.asarray(row['native_indices']); a={k[len(pre):]:v for k,v in arrays.items() if k.startswith(pre)}
    cid=row['context_id']; tr=roots[roots[:,0]==cid]; fast=tr[tr[:,2]==0]; strict=tr[tr[:,2]==1]
    check(name+':complete',row['status']=='completed' and np.all(a['completed_mask']) and np.all(a['strict_completed_mask']))
    exact(name+':fixed_native_indices',idx,data[windows[row['window_id']]['index_key']]); exact(name+':saved_native_indices',a['native_indices'],idx)
    ns=data[phase+'_path_voltage_s'][idx]; nu=np.array([points[(phase,int(j))]['inverse']['direct_u'] for j in idx])
    close(name+':native_path',a['native_path_v'],ns,atol=0.); close(name+':native_u',a['native_inverse_u'],nu,atol=0.)
    dep=idx if row['interpolant']=='linear' else np.unique(np.concatenate([data[w['index_key']] for w in windows.values() if w['phase']==phase and w['branch']==row['branch']]))
    exact(name+':interpolation_dependencies',row['interpolation_dependency_indices'],dep)
    check(name+':dependency_count',len(dep)==(len(idx) if row['interpolant']=='linear' else 71))
    n=(len(idx)-1)*order; node_total+=n; strict_total+=len(strict)
    check(name+':node_counts',n==row['node_count']==row['completed_nodes']==row['planned_nodes']==len(fast))
    exact(name+':node_order',fast[:,1],np.arange(n)); exact(name+':native_left',a['native_interval_left_index'],np.repeat(idx[:-1],order)); exact(name+':within_interval',a['within_interval_node_index'],np.tile(np.arange(order),len(idx)-1))
    close(name+':root_commands_and_u',fast[:,3:5],np.column_stack((a['command_voltage_v'],a['state_u'])),atol=0.)
    close(name+':root_outputs',fast[:,7:],np.column_stack([a[k] for k in ('current_a','local_voltage_v','transistor_voltage_v','KCL_residual_a','KVL_residual_v')]),atol=0.)
    node_s=a['node_path_v'].reshape((-1,order)); wt=a['weights_path_v'].reshape((-1,order)); h=np.diff(ns)
    check(name+':node_geometry',np.all(np.diff(a['node_path_v'])>0)&np.all(node_s>ns[:-1,None])&np.all(node_s<ns[1:,None])&np.all(wt>0))
    close(name+':weights_sum_each_interval',wt.sum(axis=1),h,rtol=0.,atol=2e-16)
    t=(2*node_s-ns[1:,None]-ns[:-1,None])/h[:,None]
    moments=True
    for degree in range(2*order):
        expected=2/(degree+1) if degree%2==0 else 0.
        moments &= np.all(abs(np.sum((2*wt/h[:,None])*t**degree,axis=1)-expected)<=2e-12)
    check(name+':saved_gauss_moments_through_2n_minus_1',moments)
    close(name+':command_interpolation',a['command_voltage_v'],np.interp(a['node_path_v'],ns,data[phase+'_command_voltage_v'][idx]),atol=1e-14)
    us=a['state_u'].reshape((-1,order)); check(name+':interpolated_u_in_local_endpoint_range',np.all(us>=np.minimum(nu[:-1],nu[1:])[:,None]-1e-12) and np.all(us<=np.maximum(nu[:-1],nu[1:])[:,None]+1e-12))
    if row['interpolant']=='linear':close(name+':linear_u',a['state_u'],np.interp(a['node_path_v'],ns,nu),atol=1e-13)
    f=abs(a['local_voltage_v'])*np.exp(lam*abs(a['local_voltage_v'])-a['state_u']); g=abs(a['local_voltage_v'])*abs(a['current_a'])*np.expm1(a['state_u'])
    close(name+':F_saved_algebra',a['F'],f); close(name+':G_saved_algebra',a['G'],g)
    A=math.fsum((a['weights_path_v']*a['F']).tolist()); B=math.fsum((a['weights_path_v']*a['G']).tolist()); delta=float(nu[-1]-nu[0])
    close(name+':integrals_fsum',[row['A_v2'],row['B_a_v2'],row['delta_u']],[A,B,delta])
    integrated[(row['interpolant'],order,row['window_id'])]=(A,B,delta)
    selected=np.array([0,n//2-1,n//2,n-1]) if order==16 else np.array([],dtype=int)
    exact(name+':predeclared_strict_nodes',a['strict_selected_node_indices'],selected); exact(name+':strict_ledger_indices',strict[:,1],selected)
    if len(selected):
        close(name+':strict_root_inputs',strict[:,3:5],fast[selected,3:5],atol=0.)
        close(name+':strict_saved_outputs',strict[:,7:],np.column_stack([a['strict_'+k] for k in ('current_a','local_voltage_v','transistor_voltage_v','KCL_residual_a','KVL_residual_v')]),atol=0.)
        check(name+':strict_comparison',np.max(abs(np.log10(abs(fast[selected,7]/strict[:,7]))))<=5e-7 and np.max(abs(fast[selected,8]-strict[:,8]))<=1e-9)
check('quadrature_total_20160',node_total==20160); check('strict_total_96',strict_total==96)

coefficients=[]; closure_rows=[]
for group in r['coefficient_diagnostics']:
    interp,order=group['interpolant'],group['gauss_order']; name=f'{interp}_g{order}'; estimates={}
    for side in ('positive','negative'):
        saved=group[side]; ids=saved['window_ids']; ab=np.array([integrated[(interp,order,w)] for w in ids]); X=ab[:,:2]*[1,-1]; y=ab[:,2]
        norms=np.linalg.norm(X,axis=0); N=X/norms; singular=np.linalg.svd(N,compute_uv=False); rank=int(np.sum(singular>max(N.shape)*np.finfo(float).eps*singular[0])); cond=singular[0]/singular[-1]
        coef=np.linalg.solve(N,y)/norms; estimates[side]=coef
        close(name+':'+side+':design',saved['raw_X_A_minus_B'],X); close(name+':'+side+':y',saved['y_delta_u'],y)
        close(name+':'+side+':norms',saved['column_l2_norms'],norms); close(name+':'+side+':normalized_design',saved['column_normalized_X'],N)
        close(name+':'+side+':singular_values',saved['column_normalized_singular_values'],singular); close(name+':'+side+':dimensionless_condition',saved['column_normalized_condition_2'],cond)
        check(name+':'+side+':rank',saved['numerical_rank']==rank==2)
        close(name+':'+side+':coefficients',[saved['coefficients'][k] for k in saved['coefficient_names']],coef,rtol=1e-10,atol=1e-12)
        check(name+':'+side+':positivity_flag',saved['all_coefficients_positive']==bool(np.all(coef>0)))
        coefficients.append({'variant':name,'side':side,'names':saved['coefficient_names'],'coefficients':coef.tolist(),'column_normalized_condition':float(cond),'rank':rank})
    check(name+':both_negative_reset_coefficients',np.all(estimates['negative']<0))
    check(name+':shared_ratios_undefined',group['shared_coefficient_check']['defined'] is False)
    for item in group['all_twelve_window_checks']:
        wid=item['window_id']; A,B,delta=integrated[(interp,order,wid)]; coef=estimates['negative' if wid.startswith('reset4') else 'positive']
        first,second=coef*[A,B]; pred=first-second; error=delta-pred; denom=abs(delta)+abs(first)+abs(second)
        expected={'delta_u':delta,'recovery_term':first,'capture_term':second,'predicted_delta_u':pred,'E':error,'absolute_E':abs(error),'normalization_denominator':denom,'normalized_E':error/denom}
        for k,v in expected.items():close(name+':'+wid+':closure_'+k,item['closure'][k],v,rtol=1e-10,atol=3e-13)
        closure_rows.append({'variant':name,'window_id':wid,'used_for_identification':item['used_for_identification'],'E':float(error),'normalized_E':float(error/denom)})
check('eight_linear_systems_rechecked',len(coefficients)==8)
check('32_nonidentification_window_checks',sum(not x['used_for_identification'] for x in closure_rows)==32)
for row in r['sensitivity_comparisons']:
    a,n=row['first']; b,m=row['second']; wid=row['window_id']; one=integrated[(a,n,wid)]; two=integrated[(b,m,wid)]
    close(str((row['kind'],a,n,b,m,wid))+':A_B_difference',[row['A_difference'],row['B_difference']],[one[0]-two[0],one[1]-two[1]],rtol=1e-7,atol=2e-16)
    check(str((row['kind'],a,n,b,m,wid))+':not_noise_bound',not row['true_trajectory_or_noise_error_bound'])

timing=r['numerical_timing']; grouped=Counter()
for call in timing['calls']:
    grouped[call['block']]+=call['elapsed_s']
    check('timing:'+call['label'],call['status']=='completed' and 0<=call['elapsed_s']<call['item_limit_s'])
for block,total in grouped.items():
    close('timing_sum:'+block,total,timing['used_s'][block]); check('block_budget:'+block,total<timing['limits_s'][block])
close('total_numerical_time',sum(grouped.values()),timing['total_numerical_s'])
check('total_budget_below_190s',timing['total_numerical_s']<190 and timing['stopped_blocks']=={})

# Hash all actually read inputs again. No source/result/array mutation is made.
for path,record in list(inputs.items()):check('unchanged_during_review:'+path,sha(Path(path).read_bytes())==record['sha256'])
failed=[c for c in checks if not c['passed']]
summary={'status':'passed_saved_arithmetic_review' if not failed else 'failed_saved_arithmetic_review',
    'created_utc':datetime.now(timezone.utc).isoformat(),'review_script_sha256':sha(Path(__file__).read_bytes()),
    'new_model_imports':0,'new_model_function_calls':0,'new_roots':0,'new_inversions':0,'new_ODEs':0,'new_parameter_searches':0,
    'saved_2x2_systems_resolved':8,'saved_material_trace_points_checked':22152,'saved_bisection_iterations_checked':20448,
    'saved_roots_checked':21108,'saved_unique_readpoints_checked':426,'window_variants_checked':48,'quadrature_nodes_checked':20160,
    'strict_quadrature_nodes_checked':96,'remaining_window_closures_checked':32,
    'check_count':len(checks),'passed_count':len(checks)-len(failed),'failed_count':len(failed),'failures':failed,
    'inputs_read':inputs,'decoded_archive_members':decoded,
    'material_log_current_table_decoded':False,'material_currents_recomputed':False,
    'KCL_review':'Independently reconstructed frozen transistor knots and interpolated at every saved transistor voltage; no circuit root rerun.',
    'PCHIP_review':'Checked 71-point dependency identities and saved node states within adjacent endpoint ranges; PCHIP polynomial not reconstructed.',
    'inverse_sensitivity_review':'Checked saved chi from saved incident slopes; material slopes themselves not reevaluated.',
    'not_read':'No progress JSON or historical fit result decoded; no bare/holdout/extension observation members decoded. Frozen source files and native archive fingerprinted only unless named above.',
    'root_max_abs_saved_KCL_a':float(np.max(abs(kcl))),'root_max_abs_recomputed_KCL_a':float(np.max(abs(recomputed_kcl))),
    'root_max_abs_KVL_v':float(np.max(abs(cmd-local-vt))),
    'inverse_chi_range_from_saved_slopes':[min(inverse_amplifications),max(inverse_amplifications)],
    'coefficient_rechecks':coefficients,'window_closure_rechecks':closure_rows,
    'negative_coefficients_in_all_four_variants':all(np.all(np.asarray(c['coefficients'])<0) for c in coefficients if c['side']=='negative'),
    'interpretation':'All four specified interpolants/quadrature variants yield negative alpha and beta for the two fixed RESET4 outward windows. This is conditional incompatibility with positive rates for these reconstructed paths at frozen readout/lambda/path clock; it is not unconditional rejection of shared 1R/1T1R dynamics, an experimental uncertainty certificate, or full-contract validation.',
    'arithmetic_review_wall_s':time.monotonic()-started,'checks':checks}
json_path=HERE/'SAVED_RESULT_REVIEW.json'; md_path=HERE/'SAVED_RESULT_REVIEW.md'
if json_path.exists() or md_path.exists():raise FileExistsError('Review outputs already exist; no overwrite')
json_path.write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
lines=['# 保存结果的独立算术复核','',f'**{summary["passed_count"]}/{summary["check_count"]} 项检查通过，{summary["failed_count"]} 项失败。**','',
    '只读取保存输入、输出、根求解账本和冻结表；未导入模型，未执行新反演、根求解、ODE 或参数搜索。',
    '重解的 8 个 2×2 系统完全来自已保存 A/B/Δu，是复核算术，不是新拟合尝试。','',
    '- 426 个唯一反演点：原索引/观测、保存的 48 步材料二分轨迹及两个电路回代互相一致。',
    '- 48 个窗口变体、20,160 个积分节点及 96 个严格复核节点：节点身份、域、被动性、权重矩、F/G、A/B 已核对。',
    '- 全部 21,108 个保存根输出：独立按冻结晶体管表插值核对 KCL，直接核对 KVL；没有重新求根。',
    '- 8 次列归一识别及 48 次窗口闭合复算，其中 32 次为未用于识别的窗口检查。',
    '- 全部实际读取文件在复核末尾重验哈希；冻结 runner SHA 与审查版本一致。','',
    '| 插值 / 阶数 | α | β | 负支列归一条件数 |','|---|---:|---:|---:|']
for c in coefficients:
    if c['side']=='negative':lines.append(f'| {c["variant"]} | {c["coefficients"][0]:.9g} | {c["coefficients"][1]:.9g} | {c["column_normalized_condition"]:.7g} |')
lines += ['', '**四种预定变体的 α、β 都是负数。** 因此在固定读出、lambda、路径时钟及这两种重建轨迹下，两个 RESET4 外扫窗口不能由该式的正恢复/捕获系数同时精确满足。',
    '这不是所有统一材料模型不可能的证明。未知测量噪声、实际扫描/等待时间、点间轨迹和固定读出误差未被此复核消除。Gauss 阶数差及 linear/PCHIP 差也不是实验误差界。',
    '本阶段仅检查三相位中的固定窗口，不能替代完整 46 对 / 92 相位 / 27023 残差 / 20 聚合特征验收。','',
    f'保存 KCL 最大绝对值 {summary["root_max_abs_saved_KCL_a"]:.6g} A；独立插值重算 {summary["root_max_abs_recomputed_KCL_a"]:.6g} A。',
    f'直接 KVL 最大绝对值 {summary["root_max_abs_KVL_v"]:.6g} V。',
    f'终态 JSON SHA256：`{inputs[str(HERE/"empirical_integral_diagnostic.json")]["sha256"]}`。',
    f'数组 NPZ SHA256：`{inputs[str(HERE/"empirical_integral_arrays.npz")]["sha256"]}`。',
    f'冻结 runner SHA256：`{EXPECTED_RUNNER}`。','',
    f'真实读取 {len(inputs)} 个唯一文件；两个本阶段 NPZ 全部解码，材料 NPZ 仅解码 log_gamma/log_voltage。逐文件用途、SHA、实际解码成员与全部检查见 JSON。',
    'PCHIP 只核对 71 点依赖和节点值域，没有重新构造插值多项式；反演 χ 只按保存斜率复算，没有重新计算材料导数。历史模型记录和总观测缓存仅作哈希核对。','']
if failed:lines+=['未通过检查：']+[f'- {c["name"]}' for c in failed]+['']
md_path.write_text('\n'.join(lines))
print(json.dumps({k:summary[k] for k in ('status','check_count','passed_count','failed_count','negative_coefficients_in_all_four_variants','arithmetic_review_wall_s')},indent=2))
