"""Twenty predeclared conditional soft-SET budgets; no low-bias hard gate.

Tests only an existing .2->2 V arrival and 5->2 V return envelope. Uses
one fixed gamma_c/Ron and the same J for both inequalities. No ODE/fit,
physical rate identification, or claim of passing the original contract.
"""
from pathlib import Path
import hashlib
import json
import math
import signal
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OLD=ROOT/'analysis/mechanism_selection_2026-11-03'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(OLD))
from SOFT_FEEDBACK_LOW_BIAS_BUDGET import cached_transport

CACHE=Path('/private/tmp/vulcan-runtime-cache')
RON=58.463033350725695
GC=.05*RON/93.4800875524324
BETAS=[1.,2.,4.,6.]
FRACTIONS=[.2,.5,.8,.95,.99]
NSTATE=64


def main():
    plan={'betas':BETAS,'JS_ceiling_fractions':FRACTIONS,'n_parameter_pairs':20,
      'gamma_c_ev':GC,'Ron_ohm':RON,'state_cells':NSTATE,'late_command_cells':60,
      'envelopes':[0.,.003],
      'early_path_V':1.8,'early_drive':'log(I_T(A,u)/JS)+beta*u',
      'early_upper_per_cell':'log(I_T(2,u_low)*(1+eta)/JS)+beta*u_high',
      'late_lower_per_command':'min_cells[log(I_T(Aleft,u_high)*(1-eta)/JS)+beta*u_low]',
      'early_budget':'sum(du_cell/positive_upper_drive)/1.8',
      'late_budget':'(u5_max-u2_min)/sum(dA*positive_lower_drive)',
      'zero_policy':'nonpositive early upper -> cannot cross; zero late integral -> unbounded, not accepted',
      'scope':'Existing conditional anchors only. Extra cycle4 .1/.2V 5% gate omitted, not silently relaxed. No complete low-bias or original primary-feature acceptance claim.',
      'selection_policy':'Report all 20 pairs, no adaptive extra points or optimization.',
      'budget_s':60,'ODE_runs':0,'parameter_optimizations':0}
    (HERE/'soft_beta_relaxed_budget.plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    src=CACHE/'RON_SCALING_REVIEW.json'
    source=json.loads(src.read_text())
    record=next(r for r in source['rows'] if r['Ron_ohm']==RON)
    anchor=record['anchor_reinversions']
    gamma={k:v['rederived_gamma_ev'] for k,v in anchor.items()}
    u={k:math.log1p(GC/g) for k,g in gamma.items()}
    start,target=u['initial_0p2'],u['outward_2V_lower']
    lo,hi=u['return_2V_upper'],u['return_5V_lower']
    assert start>target and hi>lo
    with np.load(CACHE/'RON_SCALING_REVIEW.npz') as z:
        commands=z['soft_set_command_left_v'].copy()
    assert len(commands)==60 and abs(commands[0]-2)<1e-12 and abs(commands[-1]-4.95)<1e-10
    t,_=cached_transport(CACHE)
    def g_of(x): return GC/np.expm1(x)
    early=np.linspace(target,start,NSTATE+1)
    late=np.linspace(lo,hi,NSTATE+1)
    started=time.monotonic()
    def timeout(*_):raise TimeoutError('60 second bounded budget exceeded; no restart')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,60.)
    try:
        ie,ve=t.series(np.full(NSTATE+1,2.),g_of(early),RON)
        il,vl=t.series(commands[:,None],g_of(late[1:])[None,:],RON)
        assert np.all(np.diff(ie)<=1e-14) and np.all(np.diff(il,axis=0)>=-1e-14)
        current_target=float(ie[0]);current_initial=float(ie[-1])
        rows=[]
        for beta in BETAS:
            ceiling=min(math.log(current_initial)+beta*start,math.log(current_target)+beta*target)
            for fraction in FRACTIONS:
                logj=ceiling+math.log(fraction)
                cases=[]
                for eta in plan['envelopes']:
                    dmax=np.log(ie[:-1]*(1+eta))-logj+beta*early[1:]
                    early_block=bool(np.any(dmax<=0.))
                    klow=math.inf if early_block else float(np.sum(np.diff(early)/dmax)/1.8)
                    lower=np.min(np.log(il*(1-eta))-logj+beta*late[:-1][None,:],axis=1)
                    integral=float(.05*np.maximum(lower,0.).sum())
                    khigh=(hi-lo)/integral if integral>0 else math.inf
                    cases.append({'current_fractional_envelope':eta,
                        'early_cannot_cross_cell':early_block,
                        'k_lower_per_V':None if not np.isfinite(klow) else klow,
                        'k_upper_per_V':None if not np.isfinite(khigh) else khigh,
                        'late_upper_unbounded':integral==0.,'late_integral_lower':integral,
                        'lower_over_upper':None if early_block else klow/khigh,
                        'necessary_intervals_not_excluded':bool(not early_block and klow<=khigh),
                        'minimum_early_cell_upper_drive':float(min(dmax)),
                        'late_drive_lower_range':[float(min(lower)),float(max(lower))]})
                rows.append({'beta':beta,'ceiling_fraction':fraction,'JS_A':math.exp(logj),
                    'JS_reference_at_u_target_A':math.exp(logj-beta*target),
                    'JS_interpretation':'JS is the reference in the log-drive; JS*exp(-beta*u) is the state-dependent current threshold, not an actual imposed compliance',
                    'cases':cases})
        residual_early=ie-np.interp(2.-ve,*t.knots['set'])
        residual_late=il-np.interp(commands[:,None]-vl,*t.knots['set'])
        out={'status':'completed_20_fixed_conditional_budgets_no_ODE_no_fit',
            'plan':plan,'anchors_gamma_ev':gamma,'anchors_u':u,
            'current_at_command2_initial_A':current_initial,'current_at_command2_target_A':current_target,
            'rows':rows,'elapsed_s':time.monotonic()-started,
            'max_KCL_abs_residual_A':float(max(np.max(abs(residual_early)),np.max(abs(residual_late)))),
            'scope':['All conditions are necessary relaxations using old declared conditional anchors, not original primary-gate success.',
                     'No low-bias 5% acceptance condition. Its actual residual must still be included in any subsequent full forward score.',
                     'No original holdout, new transport shape, actual state propagation or parameter fitting.',
                     'The .3% terminal current envelope is a sensitivity allowance, not a new global quadrature certificate.',
                     'Finite same-JS grid is not a continuous-JS or beta impossibility proof.',
                     'Counterpart RESET uses opposite beta feedback and is untested here; bare hard-path formation and known history remain required.'],
            'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [src,CACHE/'RON_SCALING_REVIEW.npz',CACHE/'transport_g010_v0025.npz',
                 CACHE/'fig2b_transistor_lookup.json',CACHE/'bookoutput2_reverse_transistor_lookup.json',
                 OLD/'SOFT_FEEDBACK_LOW_BIAS_BUDGET.py',Path(__file__)]}}
        (HERE/'soft_beta_relaxed_budget.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
        np.savez_compressed(HERE/'soft_beta_relaxed_budget.npz',early_u=early,early_command2_current_A=ie,
            late_u=late,late_command_left_V=commands,late_current_lower_A=il,
            early_local_V=ve,late_local_V=vl)
        lines=['# Soft SET：不新增低偏硬门槛的有限 β 必要筛查','',
            '只检查既有0.2→2 V到达和5→2 V返扫条件；没有从0 V开始传播，也未计算RESET。原0.1/0.2 V局部5%冲突保留为残差诊断，不冒充原验收门槛。',
            '',f'固定Ron={RON:.8g} Ω、gc={GC:.8g} eV，4个β×5个共同JS，恰20项；两种数值敏感性口径不是新增物理参数点。64状态分片、60返扫路径节点给出宽松必要上下界。',
            '', '| β | JS/上界 | JS/A | target态有效阈值/µA | 名义k下界 | 名义k上界 | 必要区间未排除 |',
            '|---:|---:|---:|---:|---:|---:|---|']
        for row in rows:
            c=row['cases'][0]
            lower='不可穿越' if c['early_cannot_cross_cell'] else f'{c["k_lower_per_V"]:.5g}'
            upper='无有限界' if c['late_upper_unbounded'] else f'{c["k_upper_per_V"]:.5g}'
            lines.append(f'| {row["beta"]:g} | {row["ceiling_fraction"]:g} | {row["JS_A"]:.6g} | {row["JS_reference_at_u_target_A"]*1e6:.5g} | {lower} | {upper} | {c["necessary_intervals_not_excluded"]} |')
        lines+=['','“未排除”只取得进一步前向检验资格，不等于曲线通过；返扫上界无穷也不表示返扫已解释。JS的大数值不能当作实际电流，需同时查看target态有效阈值。',
            '',f'最大缓存KCL残差{out["max_KCL_abs_residual_A"]:.3g} A；计算{out["elapsed_s"]:.3f} s；没有优化、ODE、保留集或域扩展。所有边界和数值敏感性结果保存在JSON。']
        (HERE/'soft_beta_relaxed_budget.md').write_text('\n'.join(lines)+'\n')
        print(json.dumps({'status':out['status'],'not_excluded_by_beta':{str(b):sum(r['cases'][0]['necessary_intervals_not_excluded'] for r in rows if r['beta']==b) for b in BETAS},'elapsed_s':out['elapsed_s']}))
    finally:signal.setitimer(signal.ITIMER_REAL,0.)

if __name__=='__main__':main()
