"""Predeclared24case necessary beta budgets; no ODE or parameter optimization."""
from pathlib import Path
import sys,time,json,math,hashlib
import numpy as np

START=time.monotonic();ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from vulcan2d.shared_spectral import SpectralShape
from vulcan2d.transistor import measured_transistor_mapping


def deadline():
    if time.monotonic()-START>120:raise TimeoutError('Declared120second cap')


def jump_lower(B,A,delta,beta,c):
    if B<=A:return np.zeros_like(c),np.zeros_like(c,dtype=bool)
    barrier=c+beta*A<=0
    with np.errstate(divide='ignore',invalid='ignore'):
        lower=np.where(barrier,np.inf,np.log((c+beta*B)/(c+beta*A))/(beta*delta))
    return lower,barrier


def main():
    olddir=ROOT/'analysis/conditional_clock_2026-11-03';initial=ROOT/'analysis/spectral_dynamic_2026-11-03/seed_screen.json'
    sources=[olddir/'budget_seeds.json',olddir/'same_record_budget_audit.json',initial,OUT/'SCREEN_PLAN.md']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    seed=json.loads(sources[0].read_text());same=json.loads(sources[1].read_text());base=json.loads(initial.read_text())
    shape=SpectralShape.from_json();mapping=measured_transistor_mapping();deadline()
    anchors=base['anchors'];bounds=base['finite_read_bounds']
    g5=seed['inputs']['g5_ev'];g2=seed['inputs']['g2_ev'];gr=anchors['bare_set_2V_high']['gamma_ev']
    gl=bounds['gamma_set_low_min_ev'];greset=bounds['gamma_reset_low_min_ev']
    gR0=seed['initial_gamma_ev']['bare_geomean_0p2V'];gT0=seed['initial_gamma_ev']['series_cycle4_0p2V']
    initial_i28=seed['inputs']['initial_bare_current_at_2p8V_a']
    command=np.linspace(2,5,61)[:-1];lo=np.zeros_like(command);hi=command.copy()
    for _ in range(36):
        mid=.5*(lo+hi);residual=shape.current_at_gamma(mid,g5)-np.array([mapping.current(float(v),1.1,'set') for v in command-mid])
        mask=residual>0;hi=np.where(mask,mid,hi);lo=np.where(mask,lo,mid);deadline()
    soft_i_lower=shape.current_at_gamma(lo,g5)
    bare_reset_v=np.linspace(0,.2,101)[:-1];bare_reset_i_lower=abs(shape.current_at_gamma(-bare_reset_v,greset))
    ga_lo=anchors['soft_reset_0p3V_low']['gamma_ev'];ga_hi=anchors['soft_reset_0p3V_high']['gamma_ev']
    gb_lo=anchors['soft_reset_1p4V_low']['gamma_ev'];gb_hi=anchors['soft_reset_1p4V_high']['gamma_ev']
    rgamma=np.geomspace(gb_lo,ga_hi,129);reset_i_upper=abs(shape.current_at_gamma(-1.4,rgamma[1:]))
    matrices={};cache={'soft_set_command_left_v':command,'soft_set_current_lower_a':soft_i_lower,
                      'bare_reset_voltage_left_v':bare_reset_v,'bare_reset_current_lower_a':bare_reset_i_lower,
                      'soft_reset_gamma_edges_ev':rgamma,'soft_reset_current_upper_a':reset_i_upper}
    for row in same['records']:
        if not row['endpoint_monotone_possible']:continue
        gamma=np.geomspace(row['gamma_2V_band_ev'][0],row['gamma_pre_band_ev'][1],129)
        voltage=np.array(row['lower_integral_voltage_nodes_v'])
        current=shape.current_at_gamma(voltage[:-1,None],gamma[None,:-1])
        matrices[row['record_id']]=(gamma,voltage,current)
        cache[f'record_{row["record_id"]}_gamma_edges_ev']=gamma
        cache[f'record_{row["record_id"]}_voltage_edges_v']=voltage
        cache[f'record_{row["record_id"]}_current_lower_nodes_a']=current
        deadline()
    np.savez_compressed(OUT/'fixed_spectrum_budget_arrays.npz',**cache)
    cases=[];record_diagnostics=[];candidate_seeds=[]
    for gc in [.2,.4,1.]:
        uL=math.log1p(gc/gl);uR0=math.log1p(gc/gR0);uT0=math.log1p(gc/gT0)
        u5=math.log1p(gc/g5);u2=math.log1p(gc/g2);duS=u5-u2
        B28=math.log1p(gc/gr);A28=math.log1p(gc/seed['outward_compliance_condition']['gamma_at_2p8V_0p995mA_lower_ev'])
        uRead=math.log1p(gc/greset);duR=math.log1p(gc/gb_hi)-math.log1p(gc/ga_lo)
        for beta in [.25,.5,.75,1.]:
            deadline()
            activation_max=initial_i28*math.exp(beta*uR0);low_read_max=.001*math.exp(beta*uL)
            jsmax=min(activation_max,low_read_max);jsupper=.98*jsmax
            js=np.geomspace(1e-7,jsupper,181) if jsupper>1e-7 else np.array([])
            c=np.log(.001/js)
            broad_lower,broad_barrier=jump_lower(B28,A28,.8,beta,c)
            legacy_lower,legacy_barrier=jump_lower(B28,uL,7.95,beta,c)
            all_jump=[];all_barrier=[]
            for row in same['records']:
                B=math.log1p(gc/row['gamma_before_upper_ev']);A=math.log1p(gc/row['gamma_after_lower_ev'])
                kl,barrier=jump_lower(B,A,row['path_bracket_v'],beta,c)
                all_jump.append(kl);all_barrier.append(barrier)
            weakest_jump=np.min(all_jump,axis=0);strongest_jump=np.max(all_jump,axis=0)
            set_lower=np.maximum.reduce([weakest_jump,broad_lower,legacy_lower])
            soft_integral=np.sum(np.maximum(np.log(soft_i_lower[:,None]/js[None,:])+beta*u2,0)*.05,axis=0)
            series_upper=np.divide(duS,soft_integral,out=np.full(len(js),np.inf),where=soft_integral>0)
            # Same-record diagnoses do not become new full-curve hard gates.
            diag=[]
            for row in same['records']:
                if not row['endpoint_monotone_possible']:
                    diag.append({'record_id':row['record_id'],'status':'5percent_endpoint_direction_incompatible_diagnostic'});continue
                gamma,voltage,current=matrices[row['record_id']]
                jlow=np.min(current*(1+gc/gamma[None,1:])**beta,axis=1);dv=np.diff(voltage)
                B=math.log1p(gc/row['gamma_pre_band_ev'][1]);A=math.log1p(gc/row['gamma_after_lower_ev'])
                uStartMax=math.log1p(gc/row['gamma_2V_band_ev'][0]);du=uStartMax-B
                integral=np.sum(np.maximum(np.log(jlow[:,None]/js[None,:]),0)*dv[:,None],axis=0)
                kup=np.divide(du,integral,out=np.full(len(js),np.inf),where=integral>0)
                klo,barrier=jump_lower(B,A,row['path_bracket_v'],beta,c)
                feasible=(~barrier)&(klo<=kup)
                ratio=np.divide(klo,kup,out=np.full(len(js),np.inf),where=kup>0)
                ratio[np.isinf(kup)&np.isfinite(klo)]=0.
                D=float(np.sum(dv*np.log(jlow/.001)));length=float(sum(dv));c0=math.log(.001/jsmax)
                applicable=(length*c0+D>0 and c0+beta*B>0 and B>A>=0 and du>0)
                fullbound=(B-A)/(du*row['path_bracket_v'])*min(length,(length*c0+D)/(c0+beta*B)) if applicable else None
                best=int(np.argmin(ratio)) if len(ratio) else None
                diag.append({'record_id':row['record_id'],'status':'finite_rate_overlap_exists' if np.any(feasible) else 'no_finite_grid_rate_overlap_diagnostic',
                             'finite_JS_overlap_count':int(sum(feasible)),
                             'minimum_finite_grid_rate_ratio':float(ratio[best]) if best is not None and math.isfinite(ratio[best]) else None,
                             'best_JS_a':float(js[best]) if best is not None else None,
                             'all_JS_under_declared_main_ceiling_ratio_lower_bound':fullbound,
                             'full_threshold_formula_applicable':applicable,'D_beta_left_integral':D,'c0':c0,
                             'common_ceiling_scope':'Uses declared common initial/read budget ceiling, not a measured read resistance for every record.'})
            record_diagnostics.append({'gamma_c_ev':gc,'beta':beta,'records':diag,
                                       'endpoint_direction_incompatible_ids':[x['record_id'] for x in diag if 'endpoint_direction' in x['status']],
                                       'records_with_finite_rate_overlap':sum(x['status']=='finite_rate_overlap_exists' for x in diag)})
            jrmax=float(np.max(reset_i_upper*(1+gc/rgamma[:-1])**beta))
            jrbase=np.geomspace(1e-6,.99*jrmax,181) if .99*jrmax>1e-6 else np.array([])
            jr=np.unique(np.r_[jrbase,max(bare_reset_i_lower)*np.linspace(.8,.999,121)])
            jr=jr[(jr>1e-6)&(jr<.99*jrmax)]
            reset_series_lower=duR/(1.1*np.log(jrmax/jr))
            reset_integral=np.sum(np.maximum(np.log(np.maximum(bare_reset_i_lower[:,None],1e-300)/jr[None,:]),0)*.002,axis=0)
            reset_bare_upper=np.divide(uRead,reset_integral,out=np.full(len(jr),np.inf),where=reset_integral>0)
            for alpha in [20.,50.]:
                set_upper=alpha*series_upper;reset_lower=alpha*reset_series_lower
                sok=np.isfinite(set_lower)&np.isfinite(set_upper)&(set_lower>0)&(set_lower<=set_upper)
                rok=np.isfinite(reset_lower)&np.isfinite(reset_bare_upper)&(reset_lower>0)&(reset_lower<=reset_bare_upper)
                sratio=np.divide(set_lower,set_upper,out=np.full(len(js),np.inf),where=set_upper>0)
                sratio[np.isinf(set_upper)]=np.inf
                rratio=np.divide(reset_lower,reset_bare_upper,out=np.full(len(jr),np.inf),where=reset_bare_upper>0)
                rratio[np.isinf(reset_bare_upper)]=np.inf
                si=int(np.argmin(sratio)) if len(sratio) else None;ri=int(np.argmin(rratio)) if len(rratio) else None
                case={'gamma_c_ev':gc,'beta':beta,'alpha':alpha,'JS_activation_ceiling_a':activation_max,
                      'JS_common_low_read_ceiling_a':low_read_max,'JS_ceiling_a':jsmax,
                      'JS_grid_count':len(js),'JS_grid_min_max_a':[float(js[0]),float(js[-1])] if len(js) else None,
                      'JR_maximum_soft_drive_a':jrmax,'JR_grid_count':len(jr),
                      'SET_compatible_threshold_count':int(sum(sok)),'RESET_compatible_threshold_count':int(sum(rok)),
                      'RESET_unbounded_upper_threshold_count':int(sum(np.isinf(reset_bare_upper))),
                      'SET_all_14_jump_barrier_threshold_count':int(sum(np.all(all_barrier,axis=0))),
                      'SET_2p8V_barrier_threshold_count':int(sum(broad_barrier)),
                      'minimum_SET_required_over_allowed':float(sratio[si]) if si is not None and math.isfinite(sratio[si]) else None,
                      'minimum_RESET_required_over_allowed':float(rratio[ri]) if ri is not None and math.isfinite(rratio[ri]) else None,
                      'best_SET_JS_a':float(js[si]) if si is not None else None,
                      'best_RESET_JR_a':float(jr[ri]) if ri is not None else None,
                      'status':'necessary_main_budget_overlap' if np.any(sok) and np.any(rok) else 'no_joint_finite_main_budget_seed',
                      'same_record_5percent_diagnostic_overlap_records':sum(x['status']=='finite_rate_overlap_exists' for x in diag)}
                if np.any(sok) and np.any(rok):
                    ks=math.sqrt(set_lower[si]*set_upper[si]) if math.isfinite(set_upper[si]) else set_lower[si]*1.5
                    kr=math.sqrt(reset_lower[ri]*reset_bare_upper[ri]) if math.isfinite(reset_bare_upper[ri]) else reset_lower[ri]*1.5
                    candidate={'gamma_c_ev':gc,'beta':beta,'alpha':alpha,'JS_a':float(js[si]),'JR_a':float(jr[ri]),
                               'kS_bare_per_v':float(ks),'kR_bare_per_v':float(kr),
                               'kS_series_per_v':float(ks/alpha),'kR_series_per_v':float(kr/alpha),'uR0':uR0,'uT0':uT0,
                               'scope':'Necessary main budgets only; per-record conditions and full trajectories not certified.'}
                    candidate_seeds.append(candidate);case['candidate_seed']=candidate
                cases.append(case)
    report={'status':'completed_predeclared_finite_beta_budget_screen_no_ODE','elapsed_s':time.monotonic()-START,
            'domain':{'gamma_c_ev':[.2,.4,1.],'beta':[.25,.5,.75,1.],'alpha':[20.,50.],'reset_initiation_log':0},
            'cases':cases,'same_record_diagnostics':record_diagnostics,'candidate_seeds':candidate_seeds,
            'scope':['Main budgets use declared aggregate5%high-voltage and finite-read conditions; not measured pointwise hard gates for every curve.',
                     'Common bareSET read resistance187.68ohm is aQ90budget, not the measured resistance of each record; bare1/11 can exceed it.',
                     'Per-record5%condition failures are retained as diagnostics and are not silently imposed on full-curve scoring.',
                     'Beta is shared by both polarities and topologies. Alpha is one conditional protocol ratio shared by both polarities, not measured.',
                     'Within-record constant command-path rate is assumed; unknown nonuniform dwell/scan timing can invalidate this interpretation.',
                     'No threshold limit seeds, continuous gamma_c/beta domain claims, ODE, optimizer, holdout use, or new spectral shape.',
                     'Changing beta does not remove monotone-voltage comparison or fixed-spectrum monotone-state Fermi-window conditions.'],
            'numerical':{'same_record_gamma_partitions':128,'soft_SET_left_intervals':60,'bare_RESET_left_intervals':100,
                         'soft_RESET_gamma_partitions':128,'fixed_spectrum_matrices_reused_across_all_beta':True},
            'source_sha256':hashes,'source_files_unchanged':all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items()),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    deadline()
    (OUT/'finite_beta_budget_screen.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# 有限beta反馈必要预算筛','',
           '24个预先声明情景；固定谱、同beta作用SET/RESET、同alpha作用两极性。无ODE、无优化。主预算与逐条5%诊断分别报告，公共Q90读预算不冒充每条实测点。','',
           '| gamma_c/eV | beta | alpha | SET阈值通过数 | RESET阈值通过数 | SET最小下/上比 | RESET最小下/上比 |','|---:|---:|---:|---:|---:|---:|---:|']
    for r in cases:lines.append(f'| {r["gamma_c_ev"]} | {r["beta"]} | {r["alpha"]:g} | {r["SET_compatible_threshold_count"]} | {r["RESET_compatible_threshold_count"]} | {r["minimum_SET_required_over_allowed"]} | {r["minimum_RESET_required_over_allowed"]} |')
    lines+=['',f'有限主预算候选数：{len(candidate_seeds)}。'+('没有种子，按计划停止。' if not candidate_seeds else '这些候选仍未通过完整轨迹与物理验证。'),
            '', 'JS上界保留实际初态2.8V激活与Icc*exp(beta*uL)低阻可达性；c可负，逐点检查c+beta*A的stop barrier。RESET已按相同beta重算，不复用beta1结论。',
            '逐条slow/pre-switch诊断及c0全阈值保守界保存在JSON。二者受同一记录常数k_per_commandV限制，未知自适应停留不能由电压点距补出来。',
            '改变beta没有自动解除跨拓扑同态电压比较或单向态冻结谱的两点Fermi界。结果仅限当前有限域、声明预算与原模型类。',
            f'计算耗时{report["elapsed_s"]:.3f}s；旧源产物哈希保持不变：{report["source_files_unchanged"]}。']
    (OUT/'finite_beta_budget_screen.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'elapsed_s':report['elapsed_s'],'candidate_seed_count':len(candidate_seeds),'cases':cases},indent=2))


if __name__=='__main__':main()
