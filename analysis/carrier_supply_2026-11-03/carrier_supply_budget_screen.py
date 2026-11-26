"""Six fixed carrier-supply budget cases; no ODE, fits, or domain expansion."""
from pathlib import Path
import sys,time,json,math,hashlib
import numpy as np

START=time.monotonic();ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from vulcan2d.shared_spectral import SpectralShape

ISTAR=.001


def deadline():
    if time.monotonic()-START>120:raise TimeoutError('Declared120second cap')


def old_jump(B,A,delta,c):
    if B<=A:return np.zeros_like(c)
    with np.errstate(divide='ignore',invalid='ignore'):
        return np.where(c+A>0,np.log((c+B)/(c+A))/delta,np.inf)


def interval_statistics(lower,upper,threshold):
    compatible=np.isfinite(lower)&(lower>0)&(lower<=upper)
    bounded=compatible&np.isfinite(upper)
    unbounded=compatible&np.isinf(upper)
    ratio=np.divide(lower,upper,out=np.full(len(lower),np.inf),where=(upper>0)&np.isfinite(upper))
    ratio[unbounded]=0.
    best=int(np.argmin(ratio)) if len(ratio) else None
    return {'compatible_count':int(sum(compatible)),'bounded_compatible_count':int(sum(bounded)),
            'compatible_unbounded_upper_count':int(sum(unbounded)),
            'minimum_required_over_allowed':float(ratio[best]) if best is not None and math.isfinite(ratio[best]) else None,
            'best_threshold_a':float(threshold[best]) if best is not None else None,
            'best_index':best},bounded


def main():
    beta_dir=ROOT/'analysis/beta_feedback_2026-11-03';clock_dir=ROOT/'analysis/conditional_clock_2026-11-03'
    sources=[beta_dir/'finite_beta_budget_screen.json',beta_dir/'fixed_spectrum_budget_arrays.npz',
             clock_dir/'budget_seeds.json',clock_dir/'same_record_budget_audit.json',
             ROOT/'analysis/spectral_dynamic_2026-11-03/seed_screen.json',OUT/'SCREEN_PLAN.md']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    beta_report=json.loads(sources[0].read_text());seed=json.loads(sources[2].read_text())
    same=json.loads(sources[3].read_text());base=json.loads(sources[4].read_text())
    with np.load(sources[1]) as z:cache={k:z[k].copy() for k in z.files}
    shape=SpectralShape.from_json();anchors=base['anchors'];bounds=base['finite_read_bounds'];deadline()
    soft_i=cache['soft_set_current_lower_a'];bare_reset_i=cache['bare_reset_current_lower_a']
    reset_gamma=cache['soft_reset_gamma_edges_ev'];reset_i=cache['soft_reset_current_upper_a']
    g5=seed['inputs']['g5_ev'];g2=seed['inputs']['g2_ev'];gr=anchors['bare_set_2V_high']['gamma_ev']
    gl=bounds['gamma_set_low_min_ev'];greset=bounds['gamma_reset_low_min_ev']
    gbhi=anchors['soft_reset_1p4V_high']['gamma_ev'];galo=anchors['soft_reset_0p3V_low']['gamma_ev']
    gR0=seed['initial_gamma_ev']['bare_geomean_0p2V'];gT0=seed['initial_gamma_ev']['series_cycle4_0p2V']
    jump_rows=[];gamma_edges=[];voltage=[]
    for row in same['records']:
        gpre=row['gamma_before_upper_ev'];gafter=row['gamma_after_lower_ev'];assert gafter>gpre>0
        gamma_edges.append(np.geomspace(gpre,gafter,129));voltage.append(row['after_voltage_v']);jump_rows.append(row)
    gamma_edges=np.array(gamma_edges);voltage=np.array(voltage)
    # One new fixed spectral matrix. Current compliance belongs only to SET.
    jump_unlimited_i=shape.current_at_gamma(voltage[:,None],gamma_edges[:,1:])
    jump_i_upper=np.minimum(ISTAR,jump_unlimited_i)
    np.savez_compressed(OUT/'jump_upper_fixed_spectrum.npz',gamma_edges_ev=gamma_edges,
                        after_command_voltage_v=voltage,unlimited_current_upper_a=jump_unlimited_i,
                        SET_compliance_current_upper_a=jump_i_upper,
                        record_ids=np.array([r['record_id'] for r in jump_rows]))
    deadline();cases=[];diagnostic_groups=[];selected=[]
    for gc in [.2,.4,1.]:
        prior=next(c for c in beta_report['cases'] if c['gamma_c_ev']==gc and c['beta']==1 and c['alpha']==20)
        jsmax=prior['JS_ceiling_a'];js=np.geomspace(1e-7,.98*jsmax,181);c=np.log(ISTAR/js)
        uR0=math.log1p(gc/gR0);uT0=math.log1p(gc/gT0)
        uL=math.log1p(gc/gl);u5=math.log1p(gc/g5);u2=math.log1p(gc/g2);duS=u5-u2
        B28=math.log1p(gc/gr);A28=math.log1p(gc/seed['outward_compliance_condition']['gamma_at_2p8V_0p995mA_lower_ev'])
        upper_u=np.log1p(gc/gamma_edges[:,:-1]);lower_u=np.log1p(gc/gamma_edges[:,1:]);delta_u=upper_u-lower_u
        h_upper=(jump_i_upper[:,:,None]/ISTAR)*np.maximum(np.log(jump_i_upper[:,:,None]/js[None,None,:])+upper_u[:,:,None],0)
        piece_time=np.divide(delta_u[:,:,None],h_upper,out=np.full_like(h_upper,np.inf),where=h_upper>0)
        jump_lower=np.sum(piece_time,axis=1)/np.array([r['path_bracket_v'] for r in jump_rows])[:,None]
        stop=np.any(h_upper<=0,axis=1)
        weakest_jump=np.min(jump_lower,axis=0);weakest_ids=np.argmin(jump_lower,axis=0)
        broad=old_jump(B28,A28,.8,c);legacy=old_jump(B28,uL,7.95,c)
        set_lower=np.maximum.reduce([weakest_jump,broad,legacy])
        soft_integral=np.sum(.05*(soft_i[:,None]/ISTAR)*np.maximum(np.log(soft_i[:,None]/js[None,:])+u2,0),axis=0)
        series_upper=np.divide(duS,soft_integral,out=np.full(len(js),np.inf),where=soft_integral>0)
        # Per-record slow diagnostics are separate from the main aggregate budget.
        diagnoses=[]
        for index,row in enumerate(same['records']):
            if not row['endpoint_monotone_possible']:
                diagnoses.append({'record_id':row['record_id'],'status':'5percent_endpoint_direction_incompatible_diagnostic'});continue
            prefix=f'record_{row["record_id"]}_'
            ge=cache[prefix+'gamma_edges_ev'];ve=cache[prefix+'voltage_edges_v'];fi=cache[prefix+'current_lower_nodes_a']
            u_lower=np.log1p(gc/ge[1:]);h_lower=(fi[:,:,None]/ISTAR)*np.maximum(np.log(fi[:,:,None]/js[None,None,:])+u_lower[None,:,None],0)
            global_h_lower=np.min(h_lower,axis=1)
            integral=np.sum(np.diff(ve)[:,None]*global_h_lower,axis=0)
            du=math.log1p(gc/row['gamma_2V_band_ev'][0])-math.log1p(gc/row['gamma_pre_band_ev'][1])
            slow_upper=np.divide(du,integral,out=np.full(len(js),np.inf),where=integral>0)
            stats,_=interval_statistics(jump_lower[index],slow_upper,js)
            stats.update(record_id=row['record_id'],status='necessary_interval_overlap_diagnostic' if stats['compatible_count'] else 'no_finite_grid_interval_overlap_diagnostic')
            diagnoses.append(stats)
        diagnostic_groups.append({'gamma_c_ev':gc,'records':diagnoses,
                                  'scope':'Record-specific5%conditions are diagnostic, not additional full-curve gates.'})
        # RESET is NOT capped at I*. This is an actual-current multiplier.
        reset_u_upper=np.log1p(gc/reset_gamma[:-1]);jrmax=float(np.max(reset_i*np.exp(reset_u_upper)))
        jr=np.unique(np.r_[np.geomspace(1e-6,.99*jrmax,181),max(bare_reset_i)*np.linspace(.8,.999,121)])
        jr=jr[(jr>1e-6)&(jr<.99*jrmax)]
        reset_h_max=np.max((reset_i[:,None]/ISTAR)*np.maximum(np.log(reset_i[:,None]/jr[None,:])+reset_u_upper[:,None],0),axis=0)
        duR=math.log1p(gc/gbhi)-math.log1p(gc/galo);uRead=math.log1p(gc/greset)
        reset_series_lower=np.divide(duR,1.1*reset_h_max,out=np.full(len(jr),np.inf),where=reset_h_max>0)
        reset_integral=np.sum(.002*(bare_reset_i[:,None]/ISTAR)*np.maximum(np.log(np.maximum(bare_reset_i[:,None],1e-300)/jr[None,:]),0),axis=0)
        reset_bare_upper=np.divide(uRead,reset_integral,out=np.full(len(jr),np.inf),where=reset_integral>0)
        for alpha in [20.,50.]:
            set_upper=alpha*series_upper;reset_lower=alpha*reset_series_lower
            ss,sbounded=interval_statistics(set_lower,set_upper,js);rs,rbounded=interval_statistics(reset_lower,reset_bare_upper,jr)
            if not(ss['compatible_count'] and rs['compatible_count']):status='necessary_main_intervals_do_not_intersect'
            elif np.any(sbounded) and np.any(rbounded):status='bounded_necessary_main_intervals_intersect'
            else:status='necessary_intervals_compatible_upper_unconstrained_not_selected'
            item={'gamma_c_ev':gc,'alpha':alpha,'beta':1.,'Istar_a':ISTAR,'reset_initiation_log':0,
                  'JS_activation_ceiling_a':prior['JS_activation_ceiling_a'],'JS_common_low_read_ceiling_a':prior['JS_common_low_read_ceiling_a'],
                  'JS_ceiling_a':jsmax,'JS_grid_min_max_a':[float(js[0]),float(js[-1])],'JS_grid_count':len(js),
                  'JR_ceiling_a':jrmax,'JR_grid_count':len(jr),'SET':ss,'RESET':rs,
                  'SET_all_records_jump_stop_threshold_count':int(sum(np.all(stop,axis=0))),
                  'RESET_zero_max_drive_threshold_count':int(sum(reset_h_max<=0)),
                  'status':status}
            best=ss['best_index']
            if best is not None:
                native=jump_rows[int(weakest_ids[best])]
                B=math.log1p(gc/native['gamma_before_upper_ev']);A=math.log1p(gc/native['gamma_after_lower_ev'])
                item['best_SET_details']={'weakest_record_id':native['record_id'],
                      'partitioned_jump_k_lower':float(weakest_jump[best]),
                      'old_whole_jump_Icc_k_lower':float(old_jump(B,A,native['path_bracket_v'],c[best:best+1])[0]),
                      'used_bare_k_lower':float(set_lower[best]),
                      'bare_k_upper':float(set_upper[best]) if math.isfinite(set_upper[best]) else None}
            if status=='bounded_necessary_main_intervals_intersect':
                si=int(np.flatnonzero(sbounded)[0]);ri=int(np.flatnonzero(rbounded)[0])
                ks=math.sqrt(set_lower[si]*set_upper[si]);kr=math.sqrt(reset_lower[ri]*reset_bare_upper[ri])
                candidate={'gamma_c_ev':gc,'alpha':alpha,'JS_a':float(js[si]),'JR_a':float(jr[ri]),
                           'kS_bare_per_v':ks,'kR_bare_per_v':kr,'kS_series_per_v':ks/alpha,'kR_series_per_v':kr/alpha,
                           'uR0':uR0,'uT0':uT0,'scope':'Bounded necessary budgets only; no trajectory or mechanism validation.'}
                item['selected_conditional_seed']=candidate;selected.append(candidate)
            cases.append(item)
        deadline()
    report={'status':'completed_six_case_carrier_supply_screen_no_ODE','elapsed_s':time.monotonic()-START,
            'candidate_law':'du/ds=sign*[k*(abs(I)/1mA)*max(log(abs(I)/J)+u,0)]; sharedbeta1,A_RESET0',
            'cases':cases,'selected_conditional_seeds':selected,'same_record_diagnostics':diagnostic_groups,
            'jump_numerical_scope':{'gamma_intervals':128,'fixed_F_matrix_evaluated_once':True,
                                    'SET_upper_current_capped_at_Icc':True,'RESET_current_capped':False},
            'scope':['Current-supply attempt-frequency factor is a candidate effective hypothesis, not an established mechanism.',
                     'Main SET uses weakest14native jump bound; individual5%slow diagnostics are not silently promoted to full-curve gates.',
                     'Infinite upper rate means unbounded necessary interval, not failure; finite rates mathematically exist if lower finite. This screen does not select such underconstrained cases for ODE.',
                     'Common Q90read budget is conditional and not each record measured read resistance.',
                     'Within-record constant command-path rate, ideal1mASET compliance, pre-compliance actualV=command and continuous state are assumed.',
                     'Both polarities use the samealpha. Neitheralpha nor physical sweep times are measured here.',
                     'Positive current multiplier preserves same-state voltage ordering and positiveSET state direction; earlier Fermi/order limitations remain.',
                     'Only6predeclared cases and finite threshold grids; no continuous domain theorem, fitting, ODE, extra parameter, or enlarged search.'],
            'source_sha256':hashes,'source_files_unchanged':all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items()),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    deadline();(OUT/'carrier_supply_budget_screen.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# 共同载流子供给因子的必要预算结果','',
           'SET/RESET共同乘|I|/1mA，beta1、A_RESET0，不新增拟合参数。固定6情景，无ODE。裸RESET没有1mA限流。','',
           '| gamma_c/eV | alpha | SET相容阈值 | RESET相容/有限上界阈值 | SET最小下/上比 | RESET未约束上界相容数 |','|---:|---:|---:|---:|---:|---:|']
    for r in cases:lines.append(f'| {r["gamma_c_ev"]} | {r["alpha"]:g} | {r["SET"]["compatible_count"]} | {r["RESET"]["compatible_count"]}/{r["RESET"]["bounded_compatible_count"]} | {r["SET"]["minimum_required_over_allowed"]:.7g} | {r["RESET"]["compatible_unbounded_upper_count"]} |')
    lines+=['',f'已选有限上下界条件种子：{len(selected)}。'+('无种子，按计划停止；不扩大域。' if not selected else '仅必要预算，不是轨迹验证。'),
            '每条jump保留128个状态片的电流上界并积分delta_u/h_upper，不把低电流阶段当成整段1mA。完整与旧Icc-only界的比较见JSON。',
            '无穷上界只表示该预算未约束，有限速率仍可在数学上存在；报告把“无交集”与“相容但上界未约束、未选择”分开，不能把后者当作失败或成功拟合。',
            '公共Q90读预算、逐条5%诊断、实际时间/常数路径率条件均保留。该正电流因子不自动解除单向状态读出或同态电压排序问题。',
            f'用时{report["elapsed_s"]:.3f}s；旧源哈希保持不变：{report["source_files_unchanged"]}。']
    (OUT/'carrier_supply_budget_screen.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'elapsed_s':report['elapsed_s'],'selected_seed_count':len(selected),'cases':cases},indent=2))


if __name__=='__main__':main()
