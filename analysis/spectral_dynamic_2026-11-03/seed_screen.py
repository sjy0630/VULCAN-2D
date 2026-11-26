"""Training-only necessary rate screen for the fixed-shape beta=1 trial.

No ODE, holdout arrays, fitted extra state, or production edit. Diagnostic
compromises are deliberately NOT labelled feasible seeds. Endpoint empirical
bands are descriptive budgets, not confidence intervals or new hard gates.
"""
from pathlib import Path
import sys, json, math, hashlib
import numpy as np
from scipy.optimize import brentq, minimize_scalar

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from vulcan2d.shared_spectral import SpectralShape
from vulcan2d.transistor import measured_transistor_mapping

OUT=Path(__file__).resolve().parent
DATA=ROOT/'analysis/shared_hbn_2026-11-03/data/native_observations.npz'
SHAPE=ROOT/'analysis/constrained_shared_2026-11-03/lognormal_spectral_refinement.json'
TRAIN={'1r':[1,2,3,5,6,7,8,9,10,11,12,13,14,15],'1t1r':list(range(4,36))}
PARAMETERS=['log10_gamma_c_eV','log10_JS_A','log10_JR_A','log10_kS_per_V','log10_kR_per_V','uR0','uT0']
# Registered diagnostic points, not solutions selected after an ODE search.
DIAGNOSTICS=[dict(name='faster_closure_compromise',gamma_c=.03,JS=1.5e-4,JR=2e-5,kS=1.5,kR=.1),
             dict(name='higher_threshold_compromise',gamma_c=.05,JS=5e-4,JR=5e-5,kS=.8,kR=.1)]


def profile_ratio(function, upper_log10, zero_limit):
    """Grid plus local refinement; includes J->0 limit, not a global theorem."""
    grid=np.linspace(-12.,upper_log10-1e-8,301)
    values=np.array([function(x)['ratio'] for x in grid])
    index=int(np.argmin(values));best=function(float(grid[index]));best['threshold_log10']=float(grid[index])
    if 0<index<len(grid)-1:
        fit=minimize_scalar(lambda x:function(x)['ratio'],bounds=(grid[index-1],grid[index+1]),method='bounded')
        row=function(float(fit.x));row['threshold_log10']=float(fit.x)
        if row['ratio']<best['ratio']:best=row
    if zero_limit<best['ratio']:
        best={'ratio':float(zero_limit),'threshold_log10':None,'k_lower':None,'k_upper':None,'threshold_limit':'J approaches zero'}
    return best


def main():
    shape=SpectralShape.from_json(SHAPE);mapping=measured_transistor_mapping()
    loaded=hashlib.sha256();raw={}
    with np.load(DATA) as archive:
        for top,ids in TRAIN.items():
            for phase in ('set','reset'):
                curves=[];wave=None
                for cid in ids:
                    prefix=f'{top}_cycle_{cid}_{phase}'
                    v=archive[prefix+'_voltage'].copy();stored=archive[prefix+'_current'].copy()
                    for suffix,a in [('voltage',v),('stored_current',stored)]:loaded.update((prefix+suffix).encode());loaded.update(a.tobytes())
                    if wave is not None and not np.array_equal(wave,v):raise ValueError('Different native waveforms')
                    wave=v;curves.append(abs(stored))
                raw[top,phase]=(wave,np.array(curves))

    def invert(v,i):
        if not 0<i<abs(v)/shape.ron_ohm:raise ValueError('Read point beyond shared Ohmic ceiling')
        lg=brentq(lambda x:math.log(abs(shape.current_at_log_gamma(v,x))/i),-32.,15.,xtol=2e-11)
        return math.exp(lg)

    def anchor(top,phase,branch,voltage,quantile):
        v,y=raw[top,phase];index=np.arange(len(v));apex=int(np.argmax(abs(v)))
        mask=index<=apex if branch=='outward' else index>apex
        j=int(np.argmin(np.where(mask,abs(abs(v)-voltage),np.inf)))
        current=float(np.quantile(y[:,j],quantile)) if isinstance(quantile,float) else float(np.exp(np.mean(np.log(y[:,j]))))
        local=float(v[j])
        if top=='1t1r':local=float(np.sign(local)*(abs(local)-mapping.voltage_for_current(current,1.1,phase)))
        return {'topology':top,'phase':phase,'branch':branch,'command_v':float(v[j]),'quantile':quantile,
                'current_a':current,'local_v':local,'gamma_ev':invert(local,current),'native_index':j}

    anchors={
        'soft_set_return_5V_low':anchor('1t1r','set','return',5.,.1),
        'soft_set_return_2V_high':anchor('1t1r','set','return',2.,.9),
        'bare_set_2V_high':anchor('1r','set','outward',2.,.9),
        'bare_set_0p2V_geomean':anchor('1r','set','outward',.2,'geomean'),
        'soft_reset_0p3V_low':anchor('1t1r','reset','outward',.3,.1),
        'soft_reset_0p3V_high':anchor('1t1r','reset','outward',.3,.9),
        'soft_reset_1p4V_low':anchor('1t1r','reset','outward',1.4,.1),
        'soft_reset_1p4V_high':anchor('1t1r','reset','outward',1.4,.9)}
    # Selected cycle-4 input anchors its one initial state; it is not reset each cycle.
    v,y=raw['1t1r','set'];ap=int(np.argmax(v));j=int(np.argmin(np.where(np.arange(len(v))<=ap,abs(v-.2),np.inf)))
    first_i=float(y[0,j]);first_v=float(v[j]-mapping.voltage_for_current(first_i,1.1,'set'))
    first_gamma=invert(first_v,first_i)
    # Conservative declared finite read budgets, not exact u=0/frozen assumptions.
    gamma_set_low=invert(.05,.05/187.68)
    gamma_reset_low=invert(-.2,.2/207.751)
    g5=anchors['soft_set_return_5V_low']['gamma_ev'];g2=anchors['soft_set_return_2V_high']['gamma_ev']
    gr=anchors['bare_set_2V_high']['gamma_ev']
    ga_lo=anchors['soft_reset_0p3V_low']['gamma_ev'];ga_hi=anchors['soft_reset_0p3V_high']['gamma_ev']
    gb_lo=anchors['soft_reset_1p4V_low']['gamma_ev'];gb_hi=anchors['soft_reset_1p4V_high']['gamma_ev']

    def terminal(command,gamma):
        local=brentq(lambda vh:shape.current_at_gamma(vh,gamma)-mapping.current(command-vh,1.1,'set'),0.,command,xtol=2e-12)
        return shape.current_at_gamma(local,gamma)

    # Conservative left Riemann sums: current is monotone in positive magnitude.
    set_v=np.linspace(2.,5.,61);soft_i_lower=np.array([terminal(v,g5) for v in set_v[:-1]])
    reset_v=np.linspace(0.,.2,101);bare_i_lower=abs(shape.current_at_gamma(-reset_v[:-1],gamma_reset_low))
    # Rigorous monotonic-product upper bound on J over each gamma interval:
    # f(V,g)<=f(V,g_hi), 1+gc/g<=1+gc/g_lo. Local |Vh|<=1.4V.
    gamma_grid=np.geomspace(gb_lo,ga_hi,129)
    soft_reset_i_upper=abs(shape.current_at_gamma(-1.4,gamma_grid[1:]))
    refined_upper=abs(shape.current_at_gamma(-1.4,gamma_grid[1:],order=64))
    integration_change=float(np.max(abs(np.log10(soft_reset_i_upper/refined_upper))))

    def budgets(gc):
        uR=math.log1p(gc/gr);uL=math.log1p(gc/gamma_set_low)
        u5=math.log1p(gc/g5);u2=math.log1p(gc/g2);set_delta=u5-u2
        bare_reset_allowance=math.log1p(gc/gamma_reset_low)
        soft_reset_required=math.log1p(gc/gb_hi)-math.log1p(gc/ga_lo)
        jR_max=float(np.max(soft_reset_i_upper*(1+gc/gamma_grid[:-1])))
        # Full spectral-area trigger bound deliberately favors feasibility; the
        # actual 2.8V constraint can only be stricter. No frozen HRS is assumed.
        jS_max=min(.001*(1-1e-10),math.pi*(gr+gc)/shape.ron_ohm)
        def set_rates(logjs):
            js=10**logjs;c=math.log(.001/js)
            lower=math.log((uR+c)/(uL+c))/7.95
            integral=float(np.maximum(np.log(soft_i_lower/js)+u2,0.).sum()*.05)
            upper=set_delta/integral if integral>0 else math.inf
            return {'ratio':lower/upper,'k_lower':lower,'k_upper':upper if math.isfinite(upper) else None,'drive_integral_lower':integral}
        def reset_rates(logjr):
            jr=10**logjr
            if jr>=jR_max:return {'ratio':math.inf,'k_lower':None,'k_upper':None}
            lower=soft_reset_required/(1.1*math.log(jR_max/jr))
            integral=float(np.maximum(np.log(np.maximum(bare_i_lower,1e-300)/jr),0.).sum()*.002)
            upper=bare_reset_allowance/integral if integral>0 else math.inf
            return {'ratio':lower/upper,'k_lower':lower,'k_upper':upper if math.isfinite(upper) else None,'bare_drive_integral_lower':integral}
        set_limit=(uR-uL)*3./(7.95*set_delta)
        reset_limit=soft_reset_required*.198/(1.1*bare_reset_allowance)
        return {'gamma_c_ev':gc,'set':profile_ratio(set_rates,math.log10(jS_max),set_limit),
                'reset':profile_ratio(reset_rates,math.log10(jR_max),reset_limit),
                'set_threshold_upper_a':jS_max,'reset_threshold_upper_a':jR_max,
                'u_bare_2V_lower':uR,'u_bare_low_read_upper':uL,'soft_set_allowed_state_drop':set_delta,
                'bare_reset_allowed_state_opening':bare_reset_allowance,'soft_reset_required_state_opening':soft_reset_required},set_rates,reset_rates

    scan=[]
    for gc in np.logspace(-6.,2.,161):
        row,_,_=budgets(float(gc));row['both_rate_intervals_overlap']=bool(row['set']['ratio']<=1 and row['reset']['ratio']<=1);scan.append(row)
    diagnostics=[]
    for candidate in DIAGNOSTICS:
        gc=candidate['gamma_c'];row,sfun,rfun=budgets(gc)
        uR0=math.log1p(gc/anchors['bare_set_0p2V_geomean']['gamma_ev']);uT0=math.log1p(gc/first_gamma)
        q=[math.log10(candidate[x]) for x in ('gamma_c','JS','JR','kS','kR')]+[uR0,uT0]
        initial_gamma=anchors['bare_set_0p2V_geomean']['gamma_ev']
        current_threshold=candidate['JS']*math.exp(-uR0)
        initial_activation=(brentq(lambda vv:shape.current_at_gamma(vv,initial_gamma)-current_threshold,0.,5.)
                            if shape.current_at_gamma(5.,initial_gamma)>current_threshold else None)
        row.update(name=candidate['name'],q=q,status='diagnostic_compromise_not_feasible_seed',
                   candidate_parameters=candidate,set_rates_at_candidate=sfun(q[1]),reset_rates_at_candidate=rfun(q[2]),
                   frozen_initial_SET_activation_v=initial_activation,
                   activation_scope='First mathematical drive onset at initial state, not first 1mA compliance or a prediction of forming voltage.')
        diagnostics.append(row)
    report={'status':'completed_training_only_necessary_rate_screen_no_ODE',
        'parameter_order':PARAMETERS,'beta':1.,'training_ids':TRAIN,'anchors':anchors,
        'finite_read_bounds':{'gamma_set_low_min_ev':gamma_set_low,'gamma_reset_low_min_ev':gamma_reset_low,
            'bare_set_read_v':.05,'bare_set_read_R_upper_ohm':187.68,'bare_reset_read_v':-.2,'bare_reset_R_upper_ohm':207.751,
            'cycle4_set_initial_0p2V_gamma_ev':first_gamma},
        'rate_scan':scan,'any_sampled_joint_interval_overlap':any(r['both_rate_intervals_overlap'] for r in scan),
        'necessary_rate_formulas':{
            'set_lower':'kS >= ln[(u_bare_2V + c)/(u_bare_low + c)]/7.95, c=ln(Icc/JS)>0; allows 1mA at every remaining step.',
            'set_upper':'kS <= (u_soft_5V_max - u_soft_2V_min) / integral_2^5 [ln(I_terminal(V,gamma_5V_min)/JS)+u_soft_2V_min]+ dV.',
            'reset_lower':'kR >= (u_soft_1p4V_min - u_soft_0p3V_max)/(1.1 ln(JR_max/JR)); if JR>=JR_max the required opening is impossible.',
            'reset_upper':'kR <= u_bare_0p2V_max / integral_0^.2 [ln(I_bare(-V,gamma_bare_min)/JR)]+ dV; generously starts from u=0, allowing finite drift.',
            'trigger_upper':'JS < min(Icc, pi*(gamma_bare_2V_max+gamma_c)/Ron) is a deliberately loose full-window bound. Actual 2.675V forming remains an autonomous-trajectory gate.'},
        'eligible_seeds':[], 'diagnostic_compromises':diagnostics,
        'working_bounds_diagnostic_only':{'lower':[-4.,-7.,-8.,-4.,-4.,0.,0.],
            'upper':[0.,math.log10(.001*(1-1e-8)),-2.,1.5,1.5,30.,30.],
            'scope':'Algorithmic diagnostic box, not physical priors or a promise of feasible full trajectories.'},
        'numerical_checks':{'soft_reset_product_bound_order32_to64_max_change_decade':integration_change,
            'set_lower_integral_rule':'60 left subintervals over 2..5V','bare_reset_lower_integral_rule':'100 left subintervals over 0..0.2V',
            'soft_reset_upper_rule':'128 monotone gamma-product intervals; max |Vh|=1.4V; ignores transistor drop to favor feasibility'},
        'scope':[
            'No bare or soft branch is assumed exactly frozen. Bare RESET is allowed its entire u=0 to low-read upper-state budget.',
            'The SET bound lets bare current equal 1mA throughout remaining 7.95 command-path volts, much faster than actual preforming data permit.',
            'The RESET lower rate uses a generous whole-window maximum drive, and its upper rate uses a conservative low-conductance current curve.',
            'Endpoint Q10-Q90 and low-read bands define this conditional rate diagnostic. They are not confidence intervals or automatically added full-fit acceptance gates.',
            'A no-overlap scan is not a proof over every positive parameter or every allowed curve approximation; finite-domain threshold profiling and J->0 limits are recorded.',
            'Shared path rates are an extra protocol assumption. Unknown physical sweep-rate differences must not be invented as fitting factors.',
            'Compromise q values are supplied only to obtain actual continuous diagnostic trajectories; they are not pre-screened successful seeds.',
            'No test current/voltage arrays were accessed.'],
        'loaded_training_arrays_sha256':loaded.hexdigest(),
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),SHAPE,ROOT/'vulcan2d/shared_spectral.py']}}
    (OUT/'seed_screen.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    set_pass=[r['gamma_c_ev'] for r in scan if r['set']['ratio']<=1]
    reset_pass=[r['gamma_c_ev'] for r in scan if r['reset']['ratio']<=1]
    lines=['# β=1 共享谱动力学：训练必要条件与诊断起点','',
      '这是条件速率筛选，没有运行 ODE。没有把 bare 低压区当成精确冻结，也没有读取留出数组。',
      '两个起点均明确标记为 diagnostic compromise，不能宣称是可行 seed。','',
      '## 保守速率预算','',
      'SET：bare 在 2 V 的高阻状态取允许最导电端；之后剩余 7.95 V 路径全部允许 1 mA，给最快闭合的 kS 下界。',
      'soft 5→2 V 仅用端点允许 γ 范围，利用单向闭合和被动电路给沿途电流下界，再给 kS 上界。',
      'RESET：允许 bare 从 u=0 开始、到 −0.2 V 用尽全部低阻读点状态预算；它是有限非零漂移条件。',
      'soft 0.3→1.4 V 的最低必要状态打开量给 kR 下界；最大驱动忽略晶体管压降并对全部允许 γ 作单调乘积上界。','',
      '公式：令 c=ln(Icc/JS)>0，kS≥ln[(uR+c)/(uL+c)]/7.95；soft 状态落差除以正向最低超驱动积分给 kS 上界。',
      'RESET 的 kR 下界为最少状态打开量除以 1.1V×最大 log 超驱动；bare 的 kR 上界为全部容许状态打开量除以低导电状态对应的最小驱动积分。',
      '阈值全谱面积上界刻意放宽首次触发；实际约2.675V达到限流仍须完整轨迹检查。初始驱动启动电压与首次限流电压不是一回事。','',
      f'本次 log 网格中 SET 速率区间可交叠的 γc 范围：{[min(set_pass),max(set_pass)] if set_pass else None} eV。',
      f'RESET 可交叠范围：{[min(reset_pass),max(reset_pass)] if reset_pass else None} eV。',
      f'同一 γc 同时通过：{report["any_sampled_joint_interval_overlap"]}。这是有范围、有端点预算的必要条件证据，不是整个材料族无解定理。','',
      '## 可运行的诊断起点','']
    for row in diagnostics:
        lines += [f'### {row["name"]}', '',f'`q = {row["q"]}`','',
                  f'在该 γc 下 profile 后的 kS 下/上界最小比 = {row["set"]["ratio"]:.6g}；kR 比 = {row["reset"]["ratio"]:.6g}。',
                  '大于 1 表示本条件预算没有共同速率；实际完整轨迹仍可用于展示偏差位置，不能把这些 q 冒充通过筛选。','']
    lines += ['## 边界','',
      '端点经验区间不是置信区间，局部不相容不能替代完整原始数据评分。真实时间律共享不自动要求路径率共享；协议未知不能任意填速率比。',
      '若考虑旧 GapLaw 已存在的 near-closed RESET initiation 增量，可另查 lnJR+Ainit exp(−u) 是否解除早期低阻保持与软开启的冲突；本文件没有实现或拟合该项。',
      '所有原始散布、无效候选和数值域限制保留。详细公式、数值和来源哈希在同名 JSON。']
    (OUT/'seed_screen.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'joint_overlap':report['any_sampled_joint_interval_overlap'],'set_gc_range':([min(set_pass),max(set_pass)] if set_pass else None),
                     'reset_gc_range':([min(reset_pass),max(reset_pass)] if reset_pass else None),
                     'diagnostic_q':[x['q'] for x in diagnostics]},indent=2))


if __name__=='__main__':main()
