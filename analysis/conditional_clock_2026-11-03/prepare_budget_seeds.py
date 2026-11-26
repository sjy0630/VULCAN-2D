"""Bounded finite-threshold A_RESET=0 seed preparation, never an ODE fit."""
from pathlib import Path
import sys, json, math, time, hashlib
import numpy as np

START=time.monotonic()
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from vulcan2d.shared_spectral import SpectralShape
from vulcan2d.transistor import measured_transistor_mapping

OLD=ROOT/'analysis/spectral_dynamic_2026-11-03'
def deadline():
    if time.monotonic()-START>120: raise TimeoutError('Declared120second preparation cap')


def main():
    base=json.loads((OLD/'seed_screen.json').read_text())
    prior=json.loads((OLD/'finite_rate_order_audit.json').read_text())
    sensitivities=json.loads((OLD/'finite_rate_order_sensitivity.json').read_text())
    sensitivity=next(c for c in sensitivities['sensitivity_cases'] if c['fractional_symmetric_current_expansion']==.05)
    anchors=base['anchors'];bounds=base['finite_read_bounds']
    shape=SpectralShape.from_json();mapping=measured_transistor_mapping();deadline()
    g5=next(e['gamma_ev'] for e in sensitivity['endpoint_bands'] if e['command_v']==5 and e['side']=='lower')
    g2=anchors['soft_set_return_2V_high']['gamma_ev'];gr=anchors['bare_set_2V_high']['gamma_ev']
    gl=bounds['gamma_set_low_min_ev'];greset=bounds['gamma_reset_low_min_ev']
    ga_lo=anchors['soft_reset_0p3V_low']['gamma_ev'];ga_hi=anchors['soft_reset_0p3V_high']['gamma_ev']
    gb_lo=anchors['soft_reset_1p4V_low']['gamma_ev'];gb_hi=anchors['soft_reset_1p4V_high']['gamma_ev']
    gR0=anchors['bare_set_0p2V_geomean']['gamma_ev'];gT0=bounds['cycle4_set_initial_0p2V_gamma_ev']
    # Same60left intervals as existing budget; vector bisection saves scalar roots.
    command=np.linspace(2,5,61)[:-1];lo=np.zeros_like(command);hi=command.copy()
    for _ in range(36):
        middle=.5*(lo+hi)
        current=shape.current_at_gamma(middle,g5)
        residual=current-np.array([mapping.current(float(v),1.1,'set') for v in command-middle])
        positive=residual>0;hi=np.where(positive,middle,hi);lo=np.where(positive,lo,middle)
        deadline()
    soft_i_lower=shape.current_at_gamma(lo,g5)
    bare_v=np.linspace(0,.2,101)[:-1]
    bare_i_lower=abs(shape.current_at_gamma(-bare_v,greset))
    gamma_grid=np.geomspace(gb_lo,ga_hi,129)
    soft_reset_i_upper=abs(shape.current_at_gamma(-1.4,gamma_grid[1:]))
    initial_i28=float(shape.current_at_gamma(2.8,gR0))
    # Actual outward compliance arrival, distinct from a merely positive drive.
    # Only declared bare training records are read; 5% upward pre-current is
    # deliberately favorable to feasibility. The post target is0.995mA.
    bracket_rows=[];loaded=hashlib.sha256()
    with np.load(ROOT/'analysis/shared_hbn_2026-11-03/data/native_observations.npz') as z:
        for cid in base['training_ids']['1r']:
            prefix=f'1r_cycle_{cid}_set'
            v=z[prefix+'_voltage'].copy();i=abs(z[prefix+'_current'].copy())
            for value in (v,i):loaded.update(value.tobytes())
            apex=int(np.argmax(v));hits=np.flatnonzero(i[:apex+1]>=.995e-3)
            if not len(hits) or hits[0]==0:continue
            after=int(hits[0]);before=after-1
            bracket_rows.append({'record_id':cid,'before_index':before,'after_index':after,
                                 'before_voltage_v':float(v[before]),'after_voltage_v':float(v[after]),
                                 'before_current_a':float(i[before]),'before_current_upper_a':float(i[before]*1.05),
                                 'after_observed_current_a':float(i[after]),'after_required_current_a':.995e-3,
                                 'path_bracket_v':float(v[after]-v[before])})
    inverse_v=np.array([2.8]+[x[k] for x in bracket_rows for k in ('before_voltage_v','after_voltage_v')])
    inverse_i=np.array([.995e-3]+[x[k] for x in bracket_rows for k in ('before_current_upper_a','after_required_current_a')])
    lglo=np.full(len(inverse_v),-32.);lghi=np.full(len(inverse_v),15.)
    for _ in range(43):
        middle=.5*(lglo+lghi);fi=shape.current_at_log_gamma(inverse_v,middle)
        high=fi>inverse_i;lghi=np.where(high,middle,lghi);lglo=np.where(high,lglo,middle);deadline()
    # Favor feasibility: gamma_before upper, gamma_after lower.
    gamma_cross28=float(math.exp(lglo[0]))
    for j,row in enumerate(bracket_rows):
        row['gamma_before_upper_ev']=float(math.exp(lghi[1+2*j]))
        row['gamma_after_lower_ev']=float(math.exp(lglo[2+2*j]))
    groups=[];all_candidates=[]
    for gc in [.2,.4,1.]:
        deadline()
        uR=math.log1p(gc/gr);uL=math.log1p(gc/gl)
        u5=math.log1p(gc/g5);u2=math.log1p(gc/g2);duS=u5-u2
        uRead=math.log1p(gc/greset);duR=math.log1p(gc/gb_hi)-math.log1p(gc/ga_lo)
        uR0=math.log1p(gc/gR0);uT0=math.log1p(gc/gT0)
        uAfter28=math.log1p(gc/gamma_cross28)
        jsmax=min(.001*(1-1e-10),initial_i28*math.exp(uR0))
        jrmax=float(max(soft_reset_i_upper*(1+gc/gamma_grid[:-1])))
        # Finite thresholds only. Neither J->0 nor infinite rate bounds seed a trial.
        jsgrid=np.geomspace(1e-7,.98*jsmax,181)
        jrgrid=np.unique(np.r_[np.geomspace(1e-6,.99*jrmax,181),
                              max(bare_i_lower)*np.linspace(.80,.999,121)])
        jrgrid=jrgrid[(jrgrid>1e-6)&(jrgrid<.99*jrmax)]
        for alpha in [20.,50.]:
            sr=[];rr=[];diagnostic_set=[]
            for js in jsgrid:
                c=math.log(.001/js)
                legacy_lower=math.log((uR+c)/(uL+c))/7.95
                arrival28_lower=max(0.,math.log((uR+c)/(uAfter28+c))/.8)
                bracket_lowers=[max(0.,math.log((math.log1p(gc/x['gamma_before_upper_ev'])+c)
                                              /(math.log1p(gc/x['gamma_after_lower_ev'])+c))/x['path_bracket_v'])
                                for x in bracket_rows]
                weakest_bracket_lower=min(bracket_lowers) if bracket_lowers else 0.
                all_bracket_lower=max(bracket_lowers) if bracket_lowers else 0.
                # Even the weakest observed narrow bracket is included here;
                # requiring all records would be stronger, and is also recorded.
                lower=max(legacy_lower,arrival28_lower,weakest_bracket_lower)
                integral=float(np.maximum(np.log(soft_i_lower/js)+u2,0).sum()*.05)
                upper=alpha*duS/integral if integral>0 else math.inf
                diagnostic_set.append({'JS_a':float(js),'legacy_lower':legacy_lower,
                                       'arrival_2p8V_lower':arrival28_lower,'weakest_native_bracket_lower':weakest_bracket_lower,
                                       'all_native_brackets_lower':all_bracket_lower,'used_lower':lower,
                                       'alpha_times_series_upper':upper if math.isfinite(upper) else None,
                                       'ratio_legacy':legacy_lower/upper,'ratio_arrival_2p8V':arrival28_lower/upper,
                                       'ratio_weakest_native_bracket':weakest_bracket_lower/upper,'ratio_used':lower/upper})
                if 0<lower<upper<math.inf:
                    sr.append((upper/lower,float(js),lower,upper))
            for jr in jrgrid:
                lower=alpha*duR/(1.1*math.log(jrmax/jr))
                integral=float(np.maximum(np.log(np.maximum(bare_i_lower,1e-300)/jr),0).sum()*.002)
                upper=uRead/integral if integral>0 else math.inf
                if 0<lower<upper<math.inf:
                    rr.append((upper/lower,float(jr),lower,upper))
            summary={'alpha':alpha,'gamma_c_ev':gc,'finite_JS_budget_count':len(sr),'finite_JR_budget_count':len(rr),
                     'JS_actual_initial_activation_upper_a':jsmax,'JR_soft_maximum_drive_upper_a':jrmax,
                     'minimum_SET_ratios':{key:min(x[key] for x in diagnostic_set) for key in
                                          ['ratio_legacy','ratio_arrival_2p8V','ratio_weakest_native_bracket','ratio_used']},
                     'best_SET_point_by_combined_ratio':min(diagnostic_set,key=lambda x:x['ratio_used'])}
            if sr and rr:
                # Prefer broad finite SET interval; RESET picks a moderate rate
                # with >=20% finite rate slack when available, not its divergent end.
                s=max(sr,key=lambda x:x[0])
                reasonable=[r for r in rr if r[0]>=1.2]
                r=min(reasonable or rr,key=lambda x:math.sqrt(x[2]*x[3]))
                ks=math.sqrt(s[2]*s[3]);kr=math.sqrt(r[2]*r[3])
                q=[math.log10(x) for x in [gc,s[1],r[1],ks,kr]]+[uR0,uT0]
                row={'status':'finite_necessary_budget_seed_not_trajectory_validated','alpha':alpha,'gamma_c_ev':gc,
                     'JS_a':s[1],'JR_a':r[1],'kS_bare_per_v':ks,'kR_bare_per_v':kr,
                     'kS_series_per_v':ks/alpha,'kR_series_per_v':kr/alpha,
                     'uR0':uR0,'uT0':uT0,'q_bare_reference':q,
                     'set_k_bare_allowed_interval':[s[2],s[3]],'reset_k_bare_allowed_interval':[r[2],r[3]],
                     'set_rate_interval_ratio':s[0],'reset_rate_interval_ratio':r[0],
                     'initial_SET_drive_at_2p8V':math.log(initial_i28/s[1])+uR0,
                     'bare_RESET_closed_state_drive_at_0p5V':math.log((.5/shape.ron_ohm)/r[1]),
                     'bare_RESET_drive_at_observed_5mA_u0':math.log(.005/r[1]),
                     'bare_RESET_closed_state_threshold_voltage_v':shape.ron_ohm*r[1],
                     'bare_RESET_ohmic_current_at_0p5V_a':.5/shape.ron_ohm,
                     'original_working_kR_upper_31p6_exceeded':kr>10**1.5,
                     'general_alpha_margin':'nominal20boundary_no_strict_margin' if alpha==20 else 'above_saved_approx20_condition'}
                summary['candidate']=row;all_candidates.append(row)
            groups.append(summary)
    selected=[]
    for alpha in [20.,50.]:
        candidates=[r for r in all_candidates if r['alpha']==alpha]
        candidates.sort(key=lambda r:(r['original_working_kR_upper_31p6_exceeded'],r['kR_bare_per_v']))
        selected.extend(candidates[:2])
    report={'status':'completed_bounded_finite_budget_seed_preparation_no_ODE','elapsed_s':time.monotonic()-START,
            'parameter_order':['log10_gamma_c_ev','log10_JS_a','log10_JR_a','log10_kS_bare_per_v','log10_kR_bare_per_v','uR0','uT0'],
            'alpha_definition':'r_series/r_bare = k_bare/k_series for both SET and RESET; scenarios only, not measured protocol',
            'spectral_shape_file':'analysis/constrained_shared_2026-11-03/lognormal_spectral_refinement.json',
            'reset_initiation_log':0.,'high_endpoint_sensitivity_fraction':.05,
            'finite_grid':{'gamma_c_ev':[.2,.4,1.],'alpha':[20.,50.],
                           'JS_rule':'181logpoints from1e-7A to0.98*min(Icc,actual anchored initial2.8V activation ceiling)',
                           'JR_rule':'181finite logpoints from1e-6A to0.99*soft drive ceiling plus121points at0.80..0.999 of the bare conservative maximum current; only finite rate intervals retained',
                           'scope':'No finite seed found in this declared grid; not a theorem over every positive threshold.'},
            'general_alpha_stored_bound':sensitivity['general_alpha_lower'],
            'alpha20_scope':'Boundary scenario: stored20.000019 is float voltage-grid precision, but no strict high-order margin is claimed.',
            'initial_gamma_ev':{'bare_geomean_0p2V':gR0,'series_cycle4_0p2V':gT0},
            'initial_state_scope':'One initial anchor per acquisition sequence; never per-cycle reset. Unknown bare chronology remains conditional.',
            'outward_compliance_condition':{'gamma_at_2p8V_0p995mA_lower_ev':gamma_cross28,
                    'before_2V_anchor':anchors['bare_set_2V_high'],'global_path_budget_v':.8,
                    'native_brackets':bracket_rows,'native_before_current_expansion_fraction':.05,
                    'native_bracket_rule':'Only weakest of all recorded bracket rate bounds is required, favoring feasibility; all-record maximum also saved.',
                    'loaded_bare_training_arrays_sha256':loaded.hexdigest()},
            'inputs':{'g5_ev':g5,'g2_ev':g2,'gamma_bare_set_read_lower_ev':gl,'gamma_bare_reset_read_lower_ev':greset,
                      'initial_bare_current_at_2p8V_a':initial_i28},
            'groups':groups,'selected_seeds':selected,
            'scope':['No new spectral shape, material parameter family, ODE, holdout access, or limit-threshold seed.',
                     'Budget compatibility is necessary only; RESET bounds deliberately permit all low-read drift and may be loose.',
                     'Actual SET outward arrival at0.995mA by2.8V and weakest native narrow bracket are enforced separately from mere positive SET initiation.',
                     '0.5V/5mA checks only positive RESET initiation; no claim of correct observed peak/onset or sufficient trajectory.',
                     'Current low-bias direction violations remain unresolved by a clock ratio alone.',
                     'Shared 5% endpoint sensitivity is diagnostic, not a relaxation of original fitting gates.'],
            'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [OLD/'seed_screen.json',OLD/'finite_rate_order_audit.json',OLD/'finite_rate_order_sensitivity.json',Path(__file__)]}}
    deadline()
    (OUT/'budget_seeds.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# 条件时钟的有限预算种子（非轨迹解）','',
           '固定原谱、A_RESET=0；alpha=20/50只是条件情景。使用已有5%高压端点敏感性，不改原验收门槛。无ODE，无J→0或无限速率种子。','',
           '| alpha | gamma_c/eV | JS/A | JR/A | kS bare/V | kR bare/V | kS series/V | kR series/V |','|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in selected:lines.append('| '+' | '.join(f'{r[k]:.7g}' for k in ['alpha','gamma_c_ev','JS_a','JR_a','kS_bare_per_v','kR_bare_per_v','kS_series_per_v','kR_series_per_v'])+' |')
    if not selected:lines+=['','**没有有限预算种子通过。** 旧7.95V速率预算不足以保证外扫实际到达限流；本版同时要求2→2.8V和观测窄括区到达0.995mA，不能用正启动力代替。']
    lines+=['','| alpha | gamma_c/eV | 旧7.95V最小比 | 2→2.8V最小比 | 最宽容原生括区最小比 |','|---:|---:|---:|---:|---:|']
    for g in groups:
        r=g['minimum_SET_ratios'];lines.append(f'| {g["alpha"]:g} | {g["gamma_c_ev"]:g} | {r["ratio_legacy"]:.6g} | {r["ratio_arrival_2p8V"]:.6g} | {r["ratio_weakest_native_bracket"]:.6g} |')
    lines+=['','表中比值是所需bare速率下界/允许bare速率上界；>1表示该有限阈值网格无交集。原生括区甚至只使用14条中最宽容的下界，前点电流再向上放宽5%；并未强加逐条最大值。',
            'alpha=20贴近通用约20下界，无严格裕度；不能声称全部必要条件已有宽裕。初态由裸0.2V、series第4圈0.2V原锚固定一次。',
            '只有SET与RESET预算均通过时才生成候选并附0.5V/约5mA启动力；本轮SET全部失败，因此没有提交RESET候选。时钟因子也不能解除单支路的返扫方向违例。',
            f'固定有限网格完成用时{report["elapsed_s"]:.3f}s；共生成{len(all_candidates)}个预算候选，按较小RESET参考速率优先每情景保留最多2个。']
    (OUT/'budget_seeds.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'elapsed_s':report['elapsed_s'],'selected_seeds':selected},indent=2))


if __name__=='__main__':main()
