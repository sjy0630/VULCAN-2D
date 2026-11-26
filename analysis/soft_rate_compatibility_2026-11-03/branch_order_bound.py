"""Three-anchor necessary condition for a monotone scalar SET speed."""
from pathlib import Path
import hashlib,json,math,signal,sys,time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OLD=ROOT/'analysis/soft_drive_structure_2026-11-03'
sys.path[:0]=[str(ROOT),str(ROOT/'analysis/mechanism_selection_2026-11-03')]
from SOFT_FEEDBACK_LOW_BIAS_BUDGET import cached_transport


def main():
    dest=HERE/'branch_order_bound.json'
    if dest.exists():raise FileExistsError('Preserve previous result; no automatic rerun')
    previous=json.loads((HERE/'rate_compatibility.json').read_text())
    source=json.loads((OLD/'sinh_field_probe.json').read_text())
    refs=source['runs'][1]['reference_features']
    ilo,ihi=refs['1t1r.set.return.I_2V']['original32_aggregate_band']
    ron,gc=source['plan']['Ron_ohm'],source['plan']['gamma_c_ev']
    plan={'scope':'Conditional singlecurve threeanchor relaxation, not proof oforiginal32cycleaggregate infeasibility',
          'hypothesis':'uprime=-r(A,u); same nonnegative r onoutwardandreturn, nondecreasing inbothAandu; samevoltagepathclock, noextra dwell,jumps,hiddenstates',
          'proof':'Out4->5->ret4 has2Vpath withA>=4 andu>=uret4,loss>=2r(4,uret4). Ret4->2 has2Vpath withA<=4 andu<=uret4,loss<=2r(4,uret4). Thus2uret4<=uout4+uret2<=H+U2,necessary2L<=H+U2.',
          'softlaw_coverage':'Forr=kappa*M(Vh)*positive(logI/JS+beta*u), nonnegative nondecreasingM and beta>=supchi suffice. Lorentzpositive-spectrumchi<=2/(1-exp(-u)); u>=return2lowerstatebound throughoutgives a global sufficientbeta floor onthispathdomain. AnyJS/kappa; no claim for smallerbeta.',
          'Ron_ohm':ron,'gamma_c_ev':gc,'return2_current_band_a':[ilo,ihi],
          'numerical_material_envelopes':[0.,.003],'envelope_scope':'Inheritedconditional(1-e)Ftable<=Ftrue<=(1+e)Ftable, not measurednoise or newglobalcertificate',
          'maximum_new_inverse_problems':4,'inverses':'Perenvelope invertIlo/Fstrong forU2;Ihi/Fweak forL2. H/L reusedfrompreviouscompletedbudget.',
          'maximum_terminal_tradeoff_forward_roots':4,'new_parameter_fits':0,'new_ODE':0,
          'physical_time_scope':'Unknownclock/dwellnotfilled; equalcommandpathis notassertedequalphysicaltime',
          'compute_budget_s':30,'stop_policy':'Noadaptiveparameter expansion,holdout ornewmapping; preservefailure'}
    planpath=HERE/'branch_order_bound.plan.json';planpath.write_text(json.dumps(plan,indent=2)+'\n')
    t,lookups=cached_transport(Path('/private/tmp/vulcan-runtime-cache'))
    paths=[HERE/'rate_compatibility.json',OLD/'sinh_field_probe.json',t.table_path,*lookups,planpath,Path(__file__)]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    r={'status':'running','plan':plan,'source_sha256':hashes,'cases':[],'actual_inverse_problems':0,'actual_terminal_tradeoff_roots':0}
    started=time.monotonic()
    def timeout(*_):raise TimeoutError('30second fixedbound budgetexhausted')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,30)
    try:
        for eta in plan['numerical_material_envelopes']:
            old=next(c for c in previous['cases'] if c['envelope']==eta and c['grid']['name']=='refined')
            L,H=old['late_state_interval'];states=[]
            for label,I,factor in [('U2',ilo,1+eta),('L2',ihi,1-eta)]:
                local=2-float(t.mapping.voltage_for_current(I,1.1,'set'))
                gamma=t.inverse(local,I/factor,ron,'set');u=math.log1p(gc/gamma)
                states.append({'name':label,'current_a':I,'local_voltage_v':local,'amplitude_factor':factor,'gamma_ev':gamma,'u':u})
                r['actual_inverse_problems']+=1
            U2,L2=[s['u'] for s in states]
            assert 0<L2<=U2<L<=H
            margin=2*L-H-U2
            beta_floor=2/(-math.expm1(-L2))
            # Most favorable material/circuit corners for satisfying currentconstraints.
            required_uret2_min=2*L-H
            current2max,voltage2=t.series(2.,gc/math.expm1(required_uret2_min),ron/(1+eta))
            allowed_uret4_max=(H+U2)/2
            current4min,voltage4=t.series(4.,gc/math.expm1(allowed_uret4_max),ron/(1-eta))
            r['actual_terminal_tradeoff_roots']+=2
            r['cases'].append({'material_envelope':eta,'H_out2_upper_state':H,'L_return4_lower_state':L,
                'return2_inverse_bounds':states,'U2_return2_upper_state':U2,'L2_return2_lower_state':L2,
                'necessary_2L_le_H_plus_U2':2*L<=H+U2,'state_inequality_violation':margin,
                'sufficient_beta_for_fullpath_monotone_rate':beta_floor,'beta4_and6_both_covered':4>=beta_floor,
                'holding_out2_and_return4':{'required_return2_state_min':required_uret2_min,
                    'most_favorable_return2_current_max_a':float(current2max),'original_return2_lower_a':ilo,
                    'current_shortfall_fraction':1-float(current2max)/ilo,
                    'most_favorable_material_voltage_v':float(voltage2)},
                'holding_out2_and_return2':{'allowed_return4_state_max':allowed_uret4_max,
                    'most_favorable_return4_current_min_a':float(current4min),
                    'original_return4_upper_a':previous['plan']['return4_current_upper_a'],
                    'current_excess_fraction':float(current4min)/previous['plan']['return4_current_upper_a']-1,
                    'most_favorable_material_voltage_v':float(voltage4)}})
        r['status']='completed_conditional_branch_order_bound_no_ODE_or_fit'
    except Exception as exc:r.update(status='failed_or_partial_preserved',error_type=type(exc).__name__,error=str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);r['compute_elapsed_s']=time.monotonic()-started
    dest.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    lines=['# 三个条件读点对单状态单调速率的约束','',
           '这里不要求返扫提前冻结，也不固定JS或κ；要求同一非负速率r(A,u)对命令幅值A及u都非减。相等2V路径不可冒充已知相等物理时间。',
           '', 'out4→5→ret4全段A≥4、u≥u_ret4，状态损失≥2r(4,u_ret4)；ret4→2全段A≤4、u≤u_ret4，损失≤2r(4,u_ret4)。故2u_ret4≤u_out4+u_ret2。结合out2后的单调u及三个条件参考，必须2L≤H+U2。',
           '', '| 材料包络 | H | L | U2 | 2L−H−U2 | 足够保证全段速率单调的β下界 |',
           '|---|---:|---:|---:|---:|---:|']
    for c in r['cases']:
        lines.append(f'| {c["material_envelope"]} | {c["H_out2_upper_state"]:.9f} | {c["L_return4_lower_state"]:.9f} | {c["U2_return2_upper_state"]:.9f} | {c["state_inequality_violation"]:.9f} | {c["sufficient_beta_for_fullpath_monotone_rate"]:.9f} |')
    lines+=['', '正Lorentz谱读出给χ≤2/(1−exp(−u))；全段u≥返回2V带上界电流对应的最宽松L2，因此上述β阈值是整个相关状态域的充分界，不是少量采样导数判断。无源晶体管降低χ，且Vh随A/u均非减；非负非减M保持该方向。',
            '', '| 材料包络 | 满足out2和ret4时，ret2最高/µA | 原ret2最低/µA | 满足out2和ret2时，ret4最低/µA | 原ret4最高/µA |',
            '|---|---:|---:|---:|---:|']
    for c in r['cases']:
        a=c['holding_out2_and_return4'];b=c['holding_out2_and_return2']
        lines.append(f'| {c["material_envelope"]} | {a["most_favorable_return2_current_max_a"]*1e6:.7f} | {ilo*1e6:.7f} | {b["most_favorable_return4_current_min_a"]*1e6:.7f} | {b["original_return4_upper_a"]*1e6:.7f} |')
    lines+=['', '这些是满足另两个条件时的必要终端电流代价，不是全500点RMSE下界。所有参考仍是当前单圈条件检验，原32圈聚合验收不能未经论证替换为每圈均过此三点。谱形、晶体管映射、单调只减状态与路径时钟仍是条件；低β、状态依赖且不保序的速率、非单调场响应、双向演化或额外记忆可能改变前提。',
            '',f'运行状态{r["status"]}，新增反演{r["actual_inverse_problems"]}/4、终端根{r["actual_terminal_tradeoff_roots"]}/4，计算{r["compute_elapsed_s"]:.6f}s。无新ODE或拟合。']
    if 'error'in r:lines+=['',r['error_type']+': '+r['error']]
    dest.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k not in ('plan','source_sha256')},indent=2))


if __name__=='__main__':main()
