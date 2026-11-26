"""Training-only activation, relative-clock and same-state ordering audit.

Reads the preserved seed screen; never modifies that evidence or any core.
No ODE, optimizer fit to test data, or inferred measurement timestamps.
"""
from pathlib import Path
import sys,json,math,hashlib
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from vulcan2d.shared_spectral import SpectralShape
from vulcan2d.transistor import measured_transistor_mapping
from seed_screen import profile_ratio, TRAIN, DATA, SHAPE

OUT=Path(__file__).resolve().parent
SOURCE=OUT/'seed_screen.json'


def main():
    old=json.loads(SOURCE.read_text());shape=SpectralShape.from_json(SHAPE);mapping=measured_transistor_mapping()
    anchors=old['anchors'];loaded=hashlib.sha256();raw={}
    with np.load(DATA) as archive:
        for top,ids in TRAIN.items():
            vs=[];ys=[]
            for cid in ids:
                prefix=f'{top}_cycle_{cid}_set'
                v=archive[prefix+'_voltage'].copy();i=archive[prefix+'_current'].copy()
                for suffix,a in [('voltage',v),('stored_current',i)]:loaded.update((prefix+suffix).encode());loaded.update(a.tobytes())
                vs.append(v);ys.append(abs(i))
            raw[top]=(vs,np.array(ys))

    def inverse_gamma(v,i):
        return math.exp(brentq(lambda lg:math.log(abs(shape.current_at_log_gamma(v,lg))/i),-32.,15.,xtol=1e-11))

    def terminal(a,gamma):
        vh=brentq(lambda v:shape.current_at_gamma(v,gamma)-mapping.current(a-v,1.1,'set'),0.,a,xtol=2e-12)
        return float(shape.current_at_gamma(vh,gamma)),float(vh)

    # Initial gamma cannot exceed the later low-bias gamma under one-way SET.
    vs,ys=raw['1r'];v=vs[0];ap=int(np.argmax(v));ind=np.arange(len(v))
    j=int(np.argmin(np.where(ind<=ap,abs(v-.2),np.inf)))
    initial_current_upper=float(np.quantile(ys[:,j],.9))
    initial_gamma_upper=inverse_gamma(float(v[j]),initial_current_upper)
    gamma_floor=1e-12
    grid=np.geomspace(gamma_floor,initial_gamma_upper,1025)
    trigger_current_upper=shape.current_at_gamma(2.8,grid[1:])
    trigger_refined=shape.current_at_gamma(2.8,grid[1:],order=64)
    # Small-gamma omitted interval: normalized Lorentz convolution is a Cauchy
    # average of a [0,1], L-Lipschitz Fermi window; bound E min(L|X|,1).
    e,w,_,tail,_=shape._energy_rule(64)
    thermal=shape.thermal_ev;eta=shape.eta
    spectral_fraction=float(w@(expit(-(e-(1-eta)*2.8)/thermal)-expit(-(e+eta*2.8)/thermal)))+tail
    lipschitz=1/(2*thermal)
    z=lipschitz*gamma_floor
    cauchy_error=(z/math.pi)*math.log1p(1/(z*z))+2/math.pi*math.atan(z)
    tiny_normalized_current_upper=math.pi/shape.ron_ohm*(spectral_fraction+cauchy_error)

    g5=anchors['soft_set_return_5V_low']['gamma_ev'];g2=anchors['soft_set_return_2V_high']['gamma_ev']
    gr=anchors['bare_set_2V_high']['gamma_ev'];gl=old['finite_read_bounds']['gamma_set_low_min_ev']
    voltage_grid=np.linspace(2.,5.,61)
    soft_current_lower=np.array([terminal(float(a),g5)[0] for a in voltage_grid[:-1]])
    rows=[]
    for source_row in old['rate_scan']:
        gc=source_row['gamma_c_ev']
        jstart=max(float(np.max(trigger_current_upper*(1+gc/grid[:-1]))),
                   (gc+gamma_floor)*tiny_normalized_current_upper)
        upper=min(.001*(1-1e-10),jstart)
        ur=math.log1p(gc/gr);ul=math.log1p(gc/gl);u5=math.log1p(gc/g5);u2=math.log1p(gc/g2);delta=u5-u2
        def rates(logjs):
            js=10**logjs;c=math.log(.001/js)
            lower=math.log((ur+c)/(ul+c))/7.95
            integral=float(np.maximum(np.log(soft_current_lower/js)+u2,0).sum()*.05)
            ku=delta/integral if integral>0 else math.inf
            return {'ratio':lower/ku,'k_lower':lower,'k_upper':ku if math.isfinite(ku) else None,'drive_integral_lower':integral}
        chosen=profile_ratio(rates,math.log10(upper),(ur-ul)*3/(7.95*delta))
        sr=chosen['ratio'];rr=source_row['reset']['ratio']
        alpha_upper=math.inf if rr==0 else 1/rr
        rows.append({'gamma_c_ev':gc,'original_set_profile':source_row['set'],
                     'true_initial_activation_JS_upper_a':jstart,'new_set_profile':chosen,
                     'reset_profile_unchanged':source_row['reset'],
                     'shared_path_rates_possible':sr<=1 and rr<=1,
                     'relative_clock_alpha_lower':sr,'relative_clock_alpha_upper':None if math.isinf(alpha_upper) else alpha_upper,
                     'relative_clock_interval_nonempty':sr<=alpha_upper})

    # A same-state comparison window independent of the special log-rate form.
    vs,ys=raw['1t1r'];v=vs[0];ap=int(np.argmax(v));ind=np.arange(len(v))
    endpoints=[]
    for value in (5.,4.):
        j=int(np.argmin(np.where(ind>ap,abs(v-value),np.inf)))
        for quantile in (.1,.9):
            current=float(np.quantile(ys[:,j],quantile));local=float(v[j]-mapping.voltage_for_current(current,1.1,'set'))
            endpoints.append({'command_v':float(v[j]),'quantile':quantile,'current_a':current,
                              'local_v':local,'gamma_ev':inverse_gamma(local,current)})
    band_lo=min(r['gamma_ev'] for r in endpoints);band_hi=max(r['gamma_ev'] for r in endpoints)
    _,soft_local_min=terminal(4.,band_hi)
    crossings=[]
    for cid,v,current in zip(TRAIN['1r'],raw['1r'][0],raw['1r'][1]):
        apex=int(np.argmax(v));vv=v[:apex+1];ii=current[:apex+1]
        lower_curve=shape.current_at_gamma(vv,band_lo);upper_curve=shape.current_at_gamma(vv,band_hi)
        before=np.flatnonzero(ii<lower_curve)
        above=np.flatnonzero(ii>=upper_curve)
        above=above[(above>0)&(vv[above]>.2)]
        first=None;last=None
        for k in above:
            earlier=before[before<k]
            if len(earlier):first=int(k);last=int(earlier[-1]);break
        if first is None:
            crossings.append({'cycle':cid,'status':'no_observed_bracket'});continue
        delta=float(vv[first]-vv[last]);ceiling=float(vv[first])
        crossings.append({'cycle':cid,'status':'conditional_band_crossing_bracket',
            'last_below_band_index':last,'first_above_band_index':first,
            'last_below_command_v':float(vv[last]),'first_above_command_v':ceiling,
            'last_below_current_a':float(ii[last]),'first_above_current_a':float(ii[first]),
            'bracket_path_v':delta,'soft_local_min_exceeds_bare_ceiling':soft_local_min>ceiling,
            'constant_scan_alpha_lower':1/delta if delta>0 and soft_local_min>ceiling else None})

    eligible=[r for r in rows if r['relative_clock_interval_nonempty']]
    sr_pass=[r['gamma_c_ev'] for r in rows if r['new_set_profile']['ratio']<=1]
    clock_ranges=[{'gamma_c_ev':r['gamma_c_ev'],'alpha_lower':r['relative_clock_alpha_lower'],
                   'alpha_upper':r['relative_clock_alpha_upper']} for r in eligible]
    diagnostic=[]
    for r in old['diagnostic_compromises']:
        ls=r['set_rates_at_candidate']['ratio'];lr=r['reset_rates_at_candidate']['ratio']
        diagnostic.append({'name':r['name'],'alpha_lower':ls,'alpha_upper':1/lr,'interval_nonempty':ls<=1/lr})
    report={'status':'training_only_conditional_finite_rate_order_audit_no_ODE',
      'training_ids_read':TRAIN,'phases_read':['set'],'preserved_seed_screen_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
      'activation_screen':{'voltage_ceiling_v':2.8,'initial_gamma_upper_ev':initial_gamma_upper,
        'initial_bound_source':'bare SET outward0.2V Q90 current; one-way SET makes earlier gamma no larger',
        'initial_0p2V_current_q90_a':initial_current_upper,'gamma_partition_intervals':1024,'gamma_floor_ev':gamma_floor,
        'below_floor_Cauchy_window_error_bound':cauchy_error,'spectral_fraction_at_2p8V':spectral_fraction,
        'current_order32_to64_max_change_decade':float(np.max(abs(np.log10(trigger_current_upper/trigger_refined)))),
        'formula':'JS <= max_gamma<=gamma_initial_max F(2.8V,gamma)*(1+gamma_c/gamma); necessary first activation, not sufficient 1mA forming'},
      'profiled_rate_and_clock_rows':rows,'diagnostic_fixed_q_clock_intervals':diagnostic,
      'relative_clock_definition':'alpha=k_path,bare/k_path,series=r_series/r_bare for common physical rates and constant scan rates; same alpha for SET and RESET here',
      'relative_clock_formula':'alpha >= L_SET_bare/U_SET_series; alpha <= U_RESET_bare/L_RESET_series. Thresholds optimized independently subject to first-activation constraint.',
      'same_state_general_comparison':{'gamma_band_from_soft_5V_4V_empirical_endpoint_union_ev':[band_lo,band_hi],
        'soft_endpoint_sources':endpoints,'soft_high_return_command_path_v':1.,'soft_internal_voltage_lower_v':soft_local_min,
        'bare_training_crossing_brackets':crossings,
        'time_inequality':'If both traverse/stay in the same gamma band, monotone closure grows with local voltage, and soft local V exceeds all bare crossing V, then Delta_t_soft(5->4) <= Delta_t_bare_cross <= Delta_t_bare_bracket.',
        'constant_scan_conversion':'r_series/r_bare >= 1V / Delta_V_bare_bracket; actual timestamps are unavailable.',
        'scope':[
            'The band is the union of endpoint empirical gamma ranges, not proof that a uniquely reconstructed physical state lies there.',
            'A model merely staying somewhere in this band is a weaker condition than fitting both endpoint bands, which have a small direction tension.',
            'Uses monotone state and a scalar Markov closure magnitude nondecreasing with actual local V at fixed state; nonmonotone rates, extra state, or different voltage calibration can escape.',
            'A bracket assumes quasi-static voltage mapping with no unrecorded local-voltage excursion and no earlier nonmonotone state excursion.',
            'Elapsed times or effective sweep rates cannot be inferred from voltage-point spacing. No protocol difference is fitted or asserted.',
            'The argument concerns traversal of the same state band; it does not compare arbitrary total state changes in disjoint intervals.']},
      'summary':{'any_common_path_rate_row':any(r['shared_path_rates_possible'] for r in rows),
        'new_SET_possible_gamma_c_range_ev':([min(sr_pass),max(sr_pass)] if sr_pass else None),
        'profiled_common_relative_clock_rows':clock_ranges},
      'loaded_training_arrays_sha256':loaded.hexdigest(),
      'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),SOURCE,SHAPE,ROOT/'vulcan2d/shared_spectral.py']}}
    (OUT/'finite_rate_order_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# 有限速率与状态排序：训练条件审核','',
      '没有运行 ODE，没有更改 core 或原 seed 证据。所有结论明确区分共同路径率、共同物理时间律及状态/电压条件。','',
      '## 1. 原 ratio=0 的来源','',
      '原 SET 阈值上界采用全谱面积，仅保证某个足够大电压可以启动。部分 JS 因而高于裸初态在 2.8V 的实际可激活上界，soft 超驱动下界积分可变成零，造成 ratio=0。',
      '本次对所有允许裸初态使用实际 F(2.8V,gamma) 的单调分片上界，并对 gamma→0 尾段给 Cauchy 平滑误差界；不是只检查某一个名义初态。',
      f'加入启动约束后，采样 SET 可交叠范围：{report["summary"]["new_SET_possible_gamma_c_range_ev"]} eV。共同路径率仍有可行采样点：{report["summary"]["any_common_path_rate_row"]}。','',
      '## 2. 一个共同扫速比能否缓解两极性','',
      '定义 alpha=r_series/r_bare=k_path,bare/k_path,series。SET 要求 alpha≥L_bare/U_series；RESET 要求 alpha≤U_bare/L_series。',
      '这两个方向相反，但不能只凭方向相反就断言区间为空。允许重新选择原有 JS/JR 后，结果如下；没有据此填写实际扫速。','',
      '| gamma_c/eV | alpha 下界 | alpha 上界 |','|---:|---:|---:|']
    for r in clock_ranges:lines.append(f'| {r["gamma_c_ev"]:.7g} | {r["alpha_lower"]:.7g} | {r["alpha_upper"] if r["alpha_upper"] is not None else "无此上界"} |')
    if not clock_ranges:lines.append('| 没有采样可行点 | — | — |')
    lines += ['', '上述是必要预算允许区间，不是充分条件或真实轨迹。固定两个 diagnostic q 的时钟区间仍需单独查看 JSON，不可把重新 profile 的区间移植给它们。','',
      '## 3. 与具体 logG 无关的同态比较','',
      f'固定谱形下，soft 5→4V 两端经验 gamma 区间的联合覆盖为 [{band_lo:.8g},{band_hi:.8g}] eV。若 soft 全段留在这个范围，局部电压至少 {soft_local_min:.6g}V。',
      '裸从更窄谱状态进入更宽谱状态时，必须穿过同一 gamma 带。若其穿带实际电压处处更低，共同一状态、随局部电压单调增强的闭合率给出：',
      '`soft 留在此带的时间 ≤ bare 穿过此带的时间 ≤ bare 观测包围区间的时间`。',
      '这是同一状态区间的比较，不是把 bare 总状态落差与 soft 总落差直接相除。恒速条件下变为 alpha≥1V/裸电压包围区间。','',
      '| bare记录 | 包围区间/V | alpha必要下界 | 电压排序成立 |','|---:|---|---:|---|']
    for r in crossings:
        if r['status']=='conditional_band_crossing_bracket':lines.append(f'| {r["cycle"]} | {r["last_below_command_v"]:.4g}→{r["first_above_command_v"]:.4g} | {r["constant_scan_alpha_lower"]} | {r["soft_local_min_exceeds_bare_ceiling"]} |')
    lines += ['', '原始记录没有时间，电压点距不是扫速证据。该结果只提出必须核验的相对时间要求。soft 状态带、裸单调穿带、准静态电压上界、同一标量 Markov 率均是显式前提。',
      '若某个假设不成立，应报告实际逃逸来源；不能由此直接宣布全部一状态物理模型无解。速率参数和协议是否可共享，还须由真实连续候选与原始时间/接线信息核对。']
    (OUT/'finite_rate_order_audit.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'summary':report['summary'],'diagnostic_clock':diagnostic,'soft_gamma_band':[band_lo,band_hi],
                      'soft_vmin':soft_local_min,'bare_alpha_bounds':[r.get('constant_scan_alpha_lower') for r in crossings]},indent=2))


if __name__=='__main__':main()
