"""Independent endpoint-budget sensitivity of the conditional crossing bound.

Only training data; adds no material parameter and changes no original gate,
seed evidence, rate audit, or core. The 1/5/10% bands are diagnostic scenarios.
"""
from pathlib import Path
import sys,json,math,hashlib
import numpy as np
from scipy.optimize import brentq

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(OUT))
from vulcan2d.shared_spectral import SpectralShape
from vulcan2d.transistor import measured_transistor_mapping
from seed_screen import profile_ratio,DATA,TRAIN,SHAPE

SEED=OUT/'seed_screen.json';ORDER=OUT/'finite_rate_order_audit.json'


def main():
    seed=json.loads(SEED.read_text());prior=json.loads(ORDER.read_text())
    shape=SpectralShape.from_json(SHAPE);mapping=measured_transistor_mapping()
    source=prior['same_state_general_comparison']['soft_endpoint_sources']
    ranges={a:{'lower':next(x['current_a'] for x in source if x['command_v']==a and x['quantile']==.1),
               'upper':next(x['current_a'] for x in source if x['command_v']==a and x['quantile']==.9)} for a in (5.,4.)}
    bare=[];loaded=hashlib.sha256()
    with np.load(DATA) as archive:
        for cid in TRAIN['1r']:
            prefix=f'1r_cycle_{cid}_set';v=archive[prefix+'_voltage'].copy();original=archive[prefix+'_current'].copy()
            for suffix,a in [('voltage',v),('stored_current',original)]:loaded.update((prefix+suffix).encode());loaded.update(a.tobytes())
            ap=int(np.argmax(v));bare.append((cid,v[:ap+1],abs(original[:ap+1])))

    def inverse(a,i):
        local=a-mapping.voltage_for_current(i,1.1,'set')
        gamma=math.exp(brentq(lambda lg:math.log(shape.current_at_log_gamma(local,lg)/i),-32.,5.,xtol=1e-11))
        return gamma,local

    def forward(a,g):
        vh=brentq(lambda v:shape.current_at_gamma(v,g)-mapping.current(a-v,1.1,'set'),0.,a,xtol=1e-12)
        return float(shape.current_at_gamma(vh,g)),float(vh)

    # Independent command-only KCL minimax calculation of original band gap.
    g5lo,_=inverse(5.,ranges[5.]['lower']);g4hi,_=inverse(4.,ranges[4.]['upper'])
    def signed_errors(lg):
        g=math.exp(lg)
        return math.log(forward(5.,g)[0]/ranges[5.]['lower']),math.log(forward(4.,g)[0]/ranges[4.]['upper'])
    if g5lo>g4hi:
        common=brentq(lambda lg:sum(signed_errors(lg)),math.log(g4hi),math.log(g5lo),xtol=1e-12)
        errors=signed_errors(common);needed=math.expm1(max(abs(x) for x in errors))
    else:common=math.log(g5lo);errors=(0.,0.);needed=0.

    def case(fraction):
        endpoints=[]
        for a in (5.,4.):
            for side in ('lower','upper'):
                current=ranges[a][side]/(1+fraction) if side=='lower' else ranges[a][side]*(1+fraction)
                gamma,local=inverse(a,current)
                endpoints.append({'command_v':a,'side':side,'current_a':current,'gamma_ev':gamma,'local_v':local})
        g5min=next(x['gamma_ev'] for x in endpoints if x['command_v']==5 and x['side']=='lower')
        g4max=next(x['gamma_ev'] for x in endpoints if x['command_v']==4 and x['side']=='upper')
        feasible=g5min<=g4max
        lo=min(x['gamma_ev'] for x in endpoints);hi=max(x['gamma_ev'] for x in endpoints)
        current,vmin=forward(4.,hi)
        # Check the forward point via independent tighter energy quadrature.
        vmin_fine=brentq(lambda v:shape.current_at_gamma(v,hi,order=64)-mapping.current(4-v,1.1,'set'),0.,4.,xtol=1e-12)
        crossings=[]
        for cid,v,i in bare:
            ilow=shape.current_at_gamma(v,lo);ihigh=shape.current_at_gamma(v,hi)
            before=np.flatnonzero(i<ilow);above=np.flatnonzero((i>=ihigh)&(v>.2))
            last=first=None
            for k in above:
                possible=before[before<k]
                if len(possible):last=int(possible[-1]);first=int(k);break
            if first is None:
                crossings.append({'cycle':cid,'status':'no_bracket'});continue
            delta=float(v[first]-v[last]);ordered=vmin>v[first]
            crossings.append({'cycle':cid,'status':'bracket','last_below_v':float(v[last]),'first_above_v':float(v[first]),
                'last_below_current_a':float(i[last]),'first_above_current_a':float(i[first]),
                'last_below_is_below_nominal_compliance':bool(i[last]<.995e-3),
                'path_width_v':delta,'voltage_order_margin_v':float(vmin-v[first]),'voltage_order_passed':bool(ordered),
                'alpha_lower_if_conditions_hold':1/delta if ordered and feasible else None})
        valid=[x['alpha_lower_if_conditions_hold'] for x in crossings if x.get('alpha_lower_if_conditions_hold') is not None]
        return {'fractional_symmetric_current_expansion':fraction,'endpoint_bands':endpoints,
                'endpoint_monotone_gamma_feasible':bool(feasible),'gamma_band_ev':[lo,hi],
                'gamma_band_definition':'Convex interval hull of the two expanded endpoint gamma intervals, not their set union.',
                'soft_internal_voltage_min_v':vmin,'soft_voltage_order32_to64_change_v':abs(vmin-vmin_fine),
                'bare_crossings':crossings,'n_applicable_crossing_comparisons':len(valid),
                'general_alpha_lower':max(valid) if valid else None,
                'min_individual_alpha_lower':min(valid) if valid else None,
                'comparison_scope':'Nonvacuous only when endpoints allow monotone gamma. One valid training crossing is enough for a common clock ratio; all14 are reported.'}

    cases=[case(x) for x in (0.,.01,.05,.10)]
    # Re-profile the special logG SET bound consistently with the expanded 5V
    # endpoint; the 2V/RESET budgets and actual first-activation screen stay fixed.
    g2=seed['anchors']['soft_set_return_2V_high']['gamma_ev']
    gr=seed['anchors']['bare_set_2V_high']['gamma_ev'];gl=seed['finite_read_bounds']['gamma_set_low_min_ev']
    for result in cases:
        g5=next(x['gamma_ev'] for x in result['endpoint_bands'] if x['command_v']==5 and x['side']=='lower')
        current_lower=np.array([forward(float(a),g5)[0] for a in np.linspace(2,5,61)[:-1]])
        intersections=[];all_rows=[]
        for row in prior['profiled_rate_and_clock_rows']:
            gc=row['gamma_c_ev'];ur=math.log1p(gc/gr);ul=math.log1p(gc/gl);u5=math.log1p(gc/g5);u2=math.log1p(gc/g2);delta=u5-u2
            upper=min(.001*(1-1e-10),row['true_initial_activation_JS_upper_a'])
            def rates(logjs):
                js=10**logjs;c=math.log(.001/js)
                kl=math.log((ur+c)/(ul+c))/7.95
                integral=float(np.maximum(np.log(current_lower/js)+u2,0).sum()*.05)
                ku=delta/integral if integral else math.inf
                return {'ratio':kl/ku,'k_lower':kl,'k_upper':ku if math.isfinite(ku) else None}
            prof=profile_ratio(rates,math.log10(upper),(ur-ul)*3/(7.95*delta))
            low=prof['ratio'];rr=row['reset_profile_unchanged']['ratio'];high=math.inf if rr==0 else 1/rr
            general=result['general_alpha_lower']
            final_low=max(low,general) if general is not None else None
            valid=final_low is not None and final_low<=high
            detail={'gamma_c_ev':gc,'logG_alpha_lower':low,'logG_alpha_upper':high if math.isfinite(high) else None,
                'intersected_alpha_lower':final_low,'intersected_alpha_upper':high if math.isfinite(high) else None,
                'general_and_logG_intersection_nonempty':bool(valid)}
            all_rows.append(detail)
            if valid:intersections.append(detail)
        result['special_logG_Areset0_clock_rows']=all_rows
        result['special_logG_Areset0_intersections']=intersections
        result['common_path_alpha1_passes_general_comparison']=bool(result['general_alpha_lower'] is not None and result['general_alpha_lower']<=1)

    # Locate where the voltage-order premise first disappears; report this
    # only as a sensitivity threshold, and recompute crossing bands there.
    max_bare=max(x['first_above_v'] for x in cases[-1]['bare_crossings'] if x['status']=='bracket')
    min_bare=min(x['first_above_v'] for x in cases[-1]['bare_crossings'] if x['status']=='bracket')
    def vmin_at(p):
        g,_=inverse(5.,ranges[5.]['upper']*(1+p))
        return forward(4.,g)[1]
    breaks={}
    for label,target in [('first_record_voltage_order_loss',max_bare),('last_record_voltage_order_loss',min_bare)]:
        if vmin_at(0)>target and vmin_at(.5)<target:
            p=brentq(lambda x:vmin_at(x)-target,0.,.5,xtol=2e-10)
            breaks[label]={'approximate_current_expansion_fraction':p,'target_bare_upper_voltage_v':target,
                           'scope':'Crossing bracket identity verified again immediately above threshold.',
                           'check_immediately_above':case(p+1e-7)}

    report={'status':'training_only_endpoint_budget_sensitivity_no_ODE',
        'expansion_definition':'Multiplicatively symmetric: [I_Q10/(1+p), I_Q90*(1+p)]. Diagnostic only; original gates unchanged.',
        'original_endpoint_minimum_expansion':{'fraction':needed,'percent':100*needed,'common_gamma_ev':math.exp(common),
            'terminal_KCL_log_errors':list(errors),'scope':'Independent minimax of 5V lower and4V upper original empirical current bands.'},
        'sensitivity_cases':cases,'voltage_order_sensitivity_breaks':breaks,
        'rate_interpretation':[
            'alpha = r_series/r_bare = k_path,bare/k_path,series only under common physical material rates and constant/equivalent scan clocks.',
            'The general comparison requires fixed selected readout, one monotone scalar SET state, local-voltage-monotone closure and bounded actual local voltages.',
            'The scalar flow additionally needs local Lipschitz/uniqueness in state uniformly over the relevant voltage range, or an explicit positive integrable inverse-rate bound with no zero-rate dwell. Monotonic voltage dependence alone is insufficient.',
            'Counterexample if uniqueness is dropped: h(u)=sqrt(abs(u-u0)) admits finite crossing as well as arbitrary dwell at u0. The current positive-part logG is locally Lipschitz on this positive-state band.',
            'Bare before-band inference requires actual voltage equal to command; its selected current is well below nominal compliance. After-band inference needs only actual voltage no greater than command. Both remain quasi-static circuit assumptions.',
            'A_RESET init*exp(-u) changes negative-polarity kinetics only and cannot remove the general SET same-state comparison.',
            'The old A_RESET=0 RESET profile interval cannot be used as an upper bound for A_RESET>0. Those RESET budgets must be rederived.',
            'Nonempty profiled alpha intervals are necessary-budget compatibility, not fitted physical rates or proof of a complete trajectory.',
            'A common path-rate assumption alpha=1 differs from unknown actual sweep-rate ratio. No elapsed time is reconstructed from voltage spacing.',
            'If voltage ordering fails, this comparison gives no alpha bound for that record; it does not establish that alpha=1 is physically correct.'],
        'loaded_training_arrays_sha256':loaded.hexdigest(),
        'source_sha256':{str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in [Path(__file__),ORDER,SEED,SHAPE,ROOT/'vulcan2d/shared_spectral.py']}}
    (OUT/'finite_rate_order_sensitivity.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# 同态穿带时间界：端口误差预算敏感性','',
      '本文件不改原始验收 gate；1%、5%、10% 是显式敏感性情景。放宽按乘法对称 `[Q10/(1+p),Q90(1+p)]` 定义。','',
      f'独立自主 KCL 检查：原 5/4V 区间需最少 {100*needed:.6g}% 扩张才能允许单向 gamma；不能用原本不相容的精确端点前提作空推理。','',
      '| 扩张 | 端点单向可行 | gamma带/eV | soft局部V最低值 | 适用bare记录 | alpha必要下界 |','|---:|---|---|---:|---:|---:|']
    for r in cases:lines.append(f'| {100*r["fractional_symmetric_current_expansion"]:.0f}% | {r["endpoint_monotone_gamma_feasible"]} | {r["gamma_band_ev"]} | {r["soft_internal_voltage_min_v"]:.7g} | {r["n_applicable_crossing_comparisons"]} | {r["general_alpha_lower"]} |')
    lines += ['', 'alpha=r_series/r_bare；本例裸穿带电压间隔约0.05V、soft高压返扫范围1V。仅在共同单状态与局部电压排序等条件成立时，对应 soft 留带时间不长于 bare 穿带时间。',
      '状态带取两端 gamma 区间的区间包络（convex hull），不是不相交区间的集合并。只要同一条单调轨迹两端满足已放宽区间，全段必留在此包络中。',
      '非空的端点单向区间使 1/5/10% 情景不再依赖不可能的原端点前提。数值下界约20中的微小偏差来自原始浮点电压列，不是物理精度。','',
      '一般时间比较还需对状态局部 Lipschitz/唯一流，或直接保证正速率的倒数可积且无零速率任意停留。仅 h≥0 且对电压单调不够：h(u)=sqrt(|u−u0|) 允许非唯一的任意停留与有限穿越。当前 positive-part logG 在所用正状态带内满足局部 Lipschitz。',
      'bare 穿带前的小 gamma 推断需 Vactual=Vcommand；所选点均低于名义限流。穿带后的大 gamma 推断只需 Vactual≤Vcommand。这里仍采用已声明的准静态电路，不能替代实际电压时间测量。','',
      '## 与具体 logG 速率预算的交集','',
      '每种情景重新计算正向 logG 的5→2V预算（只放宽5V端，2V不变）；负向沿用原 A_RESET=0 条件预算。共同alpha=1若小于一般下界则不通过。',
      '重新profile后仍可能在较大gamma_c、alpha≥约20处留下必要预算交集；这说明未知真实扫速不能被直接当作已排除，也不证明这些速率比能拟合完整曲线。','']
    for r in cases:
        ok=r['special_logG_Areset0_intersections']
        lines += [f'{100*r["fractional_symmetric_current_expansion"]:.0f}% 情景：交集采样点 {len(ok)} 个；gamma_c范围 '+
                  (f'[{min(x["gamma_c_ev"] for x in ok):.7g},{max(x["gamma_c_ev"] for x in ok):.7g}] eV。' if ok else '无。')]
    lines += ['', '## 20倍比较何时不再适用','']
    for label,row in breaks.items():lines.append(f'- {label}：端口上界约放宽 {100*row["approximate_current_expansion_fraction"]:.6g}% 时，soft最低局部电压降至某裸穿带电压上限；越过后该记录不再支持这项电压排序比较。')
    lines += ['', '超过这一敏感性阈值意味着论证前提失效，不意味着模型已成功。详细逐记录括区与区间交集见JSON。',
      'A_RESET项只改负向开启，不改SET，因此不能解除这条一般SET比较。加入A后，原A=0的RESET速率上界不再能直接套用，必须重新推导；但一般SET时间条件仍原样存在。',
      '实际时间协议、额外状态、非单调局部动力学或电路映射变化均可能改变条件；本结果没有宣布全部物理模型无解。']
    (OUT/'finite_rate_order_sensitivity.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'needed_endpoint_expansion_percent':100*needed,
        'cases':[{k:r[k] for k in ('fractional_symmetric_current_expansion','endpoint_monotone_gamma_feasible','soft_internal_voltage_min_v','n_applicable_crossing_comparisons','general_alpha_lower')} for r in cases],
        'breaks':{k:100*v['approximate_current_expansion_fraction'] for k,v in breaks.items()}},indent=2))


if __name__=='__main__':main()
