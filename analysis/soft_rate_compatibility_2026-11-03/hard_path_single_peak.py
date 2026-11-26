"""Bounded interval-only hard-path unimodality screen, with frozen soft cap.

No inverse, direct quadrature, ODE, rate fit, or held-out data access.
"""
from pathlib import Path
import hashlib
import itertools
import json
import math
import signal
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def interval_single_peak(lower,upper):
    lower=np.asarray(lower,float);upper=np.asarray(upper,float)
    assert lower.ndim==1 and len(lower)>0 and np.all(lower<=upper)
    prefix=np.maximum.accumulate(lower)
    suffix=np.maximum.accumulate(lower[::-1])[::-1]
    prefix_ok=np.logical_and.accumulate(prefix<=upper)
    suffix_ok=np.logical_and.accumulate((suffix<=upper)[::-1])[::-1]
    peaks=np.flatnonzero(prefix_ok&suffix_ok)
    trace=None
    if len(peaks):
        p=int(peaks[0]);trace=np.r_[prefix[:p],max(prefix[p],suffix[p]),suffix[p+1:]]
        assert np.all(trace>=lower) and np.all(trace<=upper)
        assert np.all(np.diff(trace[:p+1])>=0) and np.all(np.diff(trace[p:])<=0)
    return peaks,trace


def is_unimodal(values):
    values=np.asarray(values)
    return any(np.all(np.diff(values[:p+1])>=0) and np.all(np.diff(values[p:])<=0) for p in range(len(values)))


def synthetic_controls():
    options=[(a,b) for a in (0,1,2) for b in (0,1,2) if a<=b]
    n=0;valid=0
    for intervals in itertools.product(options,repeat=3):
        lower,upper=np.array(intervals).T
        actual=bool(len(interval_single_peak(lower,upper)[0]))
        brute=any(is_unimodal(seq) for seq in itertools.product(*[range(int(a),int(b)+1) for a,b in intervals]))
        assert actual==brute
        n+=1;valid+=int(actual)
    cases=[('rising',[0,1,2,3],[0,1,2,3],True),
           ('falling',[4,3,2,1],[4,3,2,1],True),
           ('plateau_peak',[1,3,3,2,1],[1,3,3,2,1],True),
           ('forced_valley',[3,1,2,0],[3,1,2,0],False),
           ('wide_middle_allows_peak',[3,0,2,1],[4,4,3,2],True),
           ('two_distinct_valleys',[5,1,4,2,3],[5,1,4,2,3],False)]
    for _,lower,upper,expected in cases:
        assert bool(len(interval_single_peak(lower,upper)[0]))==expected
    return dict(exhaustive_three_node_interval_cases=n,exhaustive_feasible_cases=valid,
                independent_brute_force_agrees=True,named_cases=[dict(name=x[0],expected_feasible=x[3]) for x in cases])


def valley_witness(lower,upper,current,indices,voltage,record,high_min,high_only):
    n=len(lower);eligible=current>=high_min if high_only else np.ones(n,dtype=bool)
    before=[None]*n;after=[None]*n;best=None
    for j in range(n):
        before[j]=best
        if eligible[j] and (best is None or lower[j]>lower[best]):best=j
    best=None
    for j in range(n-1,-1,-1):
        after[j]=best
        if eligible[j] and (best is None or lower[j]>=lower[best]):best=j
    candidates=[]
    for j in range(1,n-1):
        i,k=before[j],after[j]
        if not eligible[j] or i is None or k is None:continue
        if lower[i]>upper[j] and lower[k]>upper[j]:
            ratio=float(min(lower[i],lower[k])/upper[j])
            candidates.append(dict(record=record,eligible_array_positions=[i,j,k],
                native_indices=[int(indices[x]) for x in (i,j,k)],
                signed_command_voltage_v=[float(voltage[indices[x]]) for x in (i,j,k)],
                observed_abs_current_a=[float(current[x]) for x in (i,j,k)],
                a_interval_lower_s=[float(lower[x]) for x in (i,j,k)],
                a_interval_upper_s=[float(upper[x]) for x in (i,j,k)],
                minimum_side_lower_over_middle_upper=ratio,
                minimum_side_lower_minus_middle_upper_s=float(min(lower[i],lower[k])-upper[j]),
                all_three_raw_currents_high_signal=bool(np.all(current[[i,j,k]]>=high_min))))
    if not candidates:return dict(count=0,largest_relative=None,largest_absolute=None)
    return dict(count=len(candidates),
        largest_relative=max(candidates,key=lambda x:(x['minimum_side_lower_over_middle_upper'],-x['native_indices'][1])),
        largest_absolute=max(candidates,key=lambda x:(x['minimum_side_lower_minus_middle_upper_s'],-x['native_indices'][1])))


def main():
    started=time.monotonic();dest=HERE/'hard_path_single_peak.json'
    if dest.exists():raise FileExistsError('Existing result preserved; no automatic rerun')
    plan_path=HERE/'hard_path_single_peak.plan.json';plan=json.loads(plan_path.read_text())
    r=dict(status='running',plan=plan,scenarios=[],actual_new_ODE_calls=0,actual_new_inverse_problems=0,
           actual_new_physical_fits=0,actual_new_direct_spectral_quadrature=0)
    def timeout(*_):raise TimeoutError('Declared120s total budget exceeded; preserve result')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,plan['maximum_total_runtime_seconds'])
    try:
        native=Path(plan['native_cache']);table=Path(plan['table_cache']);ref=ROOT/plan['reference_series_npz']
        assert sha(native)==plan['expected_native_sha256'] and sha(table)==plan['expected_table_sha256']
        with np.load(ref,allow_pickle=False) as z:
            actual_gamma=plan['gamma_c_ev']/np.expm1(z['refined_state_u'])
            assert math.isclose(float(max(actual_gamma)),plan['Gamma_cap_ev'],rel_tol=2e-14)
        ids=plan['training_ids'];obs=[];loaded=[];digest=hashlib.sha256();voltage=None
        with np.load(native,allow_pickle=False) as z:
            for cid in ids:
                values=[]
                for suffix in ('voltage','current'):
                    key=f'1r_cycle_{cid}_reset_{suffix}';a=z[key].copy()
                    values.append(a);loaded.append(key);digest.update(key.encode());digest.update(a.tobytes())
                if voltage is None:voltage=values[0]
                else:np.testing.assert_array_equal(voltage,values[0])
                obs.append(abs(values[1]))
        assert digest.hexdigest()=='d7a32109f603d63ce3c8069473de57b65c12eb6ece6c252291752689d78af377'
        obs=np.array(obs);v=abs(voltage);apex=int(np.argmax(v))
        mask=np.isfinite(obs)&(obs>=1e-12)&(v[None,:]>1e-12)
        assert mask.sum()==2799 and mask[:,:apex+1].sum()==1400 and mask[:,apex+1:].sum()==1399
        with np.load(table,allow_pickle=False) as z:
            lg=z['log_gamma'].copy();lv=z['log_voltage'].copy();li=z['log_current'].copy()
            metadata=json.loads(str(z['metadata'].item()))
        x=math.log(plan['Gamma_cap_ev']);assert lg[0]<=x<=lg[-1]
        j=int(np.clip(np.searchsorted(lg,x)-1,0,len(lg)-2));w=(x-lg[j])/(lg[j+1]-lg[j])
        profile=(1-w)*li[1,j]+w*li[1,j+1]+math.log(metadata['shape']['ron_ohm']/plan['Ron_ohm'])
        nonzero=v>0;assert np.all(np.log(v[nonzero])>=lv[0]) and np.all(np.log(v[nonzero])<=lv[-1])
        nominal_soft=np.zeros_like(v);nominal_soft[nonzero]=np.exp(np.interp(np.log(v[nonzero]),lv,profile))
        soft=(1+plan['material_table_fractional_upper_allowance'])*nominal_soft
        controls=synthetic_controls();r['synthetic_controls']=controls
        high_min=plan['high_signal_witness_threshold_a']
        for delta in plan['relative_raw_current_half_bands']:
            records=[]
            for row,cid,eligible in zip(obs,ids,mask):
                indices=np.flatnonzero(eligible);vv=v[indices];current=row[indices]
                low=np.maximum(0.,((1-delta)*current-soft[indices])/vv)
                upper=(1+delta)*current/vv
                peaks,trace=interval_single_peak(low,upper)
                any_witness=valley_witness(low,upper,current,indices,voltage,cid,high_min,False)
                high_witness=valley_witness(low,upper,current,indices,voltage,cid,high_min,True)
                assert bool(len(peaks)) == (any_witness['count']==0)
                records.append(dict(record=cid,n_original_eligible=len(indices),single_peak_feasible=bool(len(peaks)),
                    n_possible_peak_indices=len(peaks),possible_peak_native_indices=indices[peaks].tolist(),
                    native_indices=indices.tolist(),a_lower_s=low.tolist(),a_upper_s=upper.tolist(),
                    feasible_sequence_not_a_fit=trace.tolist() if trace is not None else None,
                    all_point_valley_witnesses=any_witness,high_signal_valley_witnesses=high_witness,
                    interpretation='All interval and peak choices are optimistic necessary relaxations, not a shared u trajectory or estimated physical state'))
            infeasible=[x for x in records if not x['single_peak_feasible']]
            high=[x['high_signal_valley_witnesses']['largest_relative'] for x in infeasible if x['high_signal_valley_witnesses']['count']]
            alls=[x['all_point_valley_witnesses']['largest_relative'] for x in infeasible]
            r['scenarios'].append(dict(relative_raw_current_half_band=delta,n_records=len(records),
                n_original_eligible=sum(x['n_original_eligible'] for x in records),
                feasible_records=[x['record'] for x in records if x['single_peak_feasible']],
                infeasible_records=[x['record'] for x in infeasible],
                infeasible_records_with_high_signal_witness=[x['record'] for x in infeasible if x['high_signal_valley_witnesses']['count']],
                infeasible_records_with_only_low_signal_witness=[x['record'] for x in infeasible if not x['high_signal_valley_witnesses']['count']],
                strongest_high_signal_witness=max(high,key=lambda x:x['minimum_side_lower_over_middle_upper']) if high else None,
                strongest_any_current_witness=max(alls,key=lambda x:x['minimum_side_lower_over_middle_upper']) if alls else None,
                records=records))
        r.update(status='completed_exact_interval_single_peak_screen',original_RESET_mask_count=int(mask.sum()),
            original_outward_count=int(mask[:,:apex+1].sum()),original_return_count=int(mask[:,apex+1:].sum()),
            original_counts_by_record=mask.sum(axis=1).tolist(),signed_command_voltage_v=voltage.tolist(),
            nominal_soft_current_at_cap_a=nominal_soft.tolist(),soft_current_upper_including_numerical_allowance_a=soft.tolist(),
            source_sha256={str(p):sha(p) for p in (native,table,ref,plan_path,Path(__file__))},
            loaded_native_array_keys=loaded,loaded_training_RESET_arrays_sha256=digest.hexdigest(),
            same_native_array_identity_as_frozen_six_point_return_diagnostic=True,
            interpretation=[
                'Gamma_cap is a predeclared bounded-soft hypothesis, not an empirical bound on the bare device.',
                'The cap may vary freely per point from0 toBcap in this necessary relaxation; no equality subtraction or measured state decomposition is assumed.',
                'Every peak position and every interval choice is allowed. Failure therefore cannot be fixed by choosing different common rate constants inside this component class.',
                'Passing does not supply a shared u trajectory, shared rates, SET formation, cross-topology transfer, or complete unified-model accuracy.',
                'Failure excludes only monotone RESET u, this shared-direction-envelope birth/death structure, and the bounded-soft cap together. It excludes no arbitrary bidirectional channel, stochastic model, contact readout, or wider soft-state scenario.',
                'Raw20% bands and the fixed0.3% numerical soft upper allowance are sensitivity assumptions, not calibrated noise.',
                'All original2799 scoring points retained.10uA is evidence grading for three-node witnesses, not a new score mask.',
                'No measured time clock is introduced; native ordering is used, including duplicate apex samples as a permissive state relaxation.'])
    except Exception as exc:
        r.update(status='partial_or_failed_screen_preserved',error_type=type(exc).__name__,error=str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);r['elapsed_s_including_inputs']=time.monotonic()-started
    dest.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');write_report(r)
    print(json.dumps(dict(status=r['status'],elapsed_s=r['elapsed_s_including_inputs'],
        controls=r.get('synthetic_controls'),error=r.get('error'),
        scenarios=[{k:v for k,v in x.items() if k!='records'} for x in r['scenarios']]),indent=2))


def write_report(r):
    lines=['# 有界soft下的硬通道单峰必要条件','',
        f'状态：`{r["status"]}`。原训练14圈RESET、原2799评分点；新ODE=0、反演=0、物理拟合=0、直接谱积分=0。',
        '', '## 模型和预声明边界','',
        '本检验针对a′=|Vh|[kn exp(σb)q(A−a)−kd exp(−σb)(1−q)a]，q=exp(−u)。在RESET场方向固定、u非减时，瞬时平衡a_eq非增，因此a只允许始终升/降，或先升后降一次，平台允许。',
        'Γcap固定为既有series软状态最大值0.001821372451496755 eV，仅代表“soft保持有限幅度”的待检场景，不是裸1R实测上界。使用同一固定谱和Ron=58.46303335Ω；负极性缓存电流乘既定1.003数值上包络，不重新积分或扩大表域。',
        '在每个原生节点独立允许B从0到Bcap，得到a区间[max(0,((1−δ)I−Bcap)/v),(1+δ)I/v]。δ只取预声明0及20%，不是已测噪声；没有把扣背景结果当真实a。',
        '', '## 区间检验','',
        '每个节点均可作峰。峰前非减的最小序列是L的前缀最大值，峰后非增的最小序列是L的后缀最大值；逐项不超过U且峰处兼容即为可行。程序为可行记录保存一条构造序列，但它不代表物理拟合。',
        '失败证书为i<j<k且L_i>U_j、L_k>U_j。即便任意挑选允许背景和状态，也必须先降后升。强见证要求三个原始电流都≥10µA，低信号证据单独降级；原评分点未删除。']
    if r['status']=='partial_or_failed_screen_preserved':
        lines+=['',f'运行错误已保留：{r.get("error_type")}: {r.get("error")}。未调整参数或自动重试。']
    else:
        lines+=['', '| 原电流半带 | 可行圈 | 不可行圈 | 有三点高信号证书的不可行圈 | 仅低信号证书的圈 |',
                '|---:|---|---|---|---|']
        for s in r['scenarios']:
            lines.append(f'| {s["relative_raw_current_half_band"]:.0%} | {s["feasible_records"]} | {s["infeasible_records"]} | {s["infeasible_records_with_high_signal_witness"]} | {s["infeasible_records_with_only_low_signal_witness"]} |')
        lines+=['', '## 最强高信号谷证书','',
                '| 半带 | 圈 | 原生索引i,j,k | 电压/V | 原始电流/mA | L_i,U_j,L_k /mS | min(L_i,L_k)/U_j |',
                '|---:|---:|---|---|---|---|---:|']
        for s in r['scenarios']:
            w=s['strongest_high_signal_witness']
            if w is None:continue
            amps=[w['a_interval_lower_s'][0]*1e3,w['a_interval_upper_s'][1]*1e3,w['a_interval_lower_s'][2]*1e3]
            currents=[x*1e3 for x in w['observed_abs_current_a']]
            lines.append(f'| {s["relative_raw_current_half_band"]:.0%} | {w["record"]} | {w["native_indices"]} | {[round(x,6) for x in w["signed_command_voltage_v"]]} | {[round(x,6) for x in currents]} | {[round(x,6) for x in amps]} | {w["minimum_side_lower_over_middle_upper"]:.7g} |')
        lines+=['', '所有圈的区间、允许峰索引和高/低信号证书保存在JSON；最大相对谷与最大绝对电导间隔分别保留，没有按单个展示见证删点。',
                '', '## 范围及校验','',
                '若失败，排除的仅是“本竞争生成/移除结构＋RESET u非减＋上述小soft上限”组合，不能写成所有双向通道模型失败。若通过，也只是允许逐点自由选背景、自由选峰的乐观必要条件，尚未存在共同u轨迹、速率或完整SET/RESET历史。',
                '裸器件材料电压使用原生命令电压；未伪造晶体管去嵌入、等待时间或秒率。缺圈未连接，保留集未读取。',
                f'独立枚举了{r["synthetic_controls"]["exhaustive_three_node_interval_cases"]}组三节点整数区间及其中所有整数序列，全部与区间算法一致；另验证6个较长的单调、平台、峰、谷及宽区间控制。',
                '原RESET数组联合哈希与冻结六点返扫诊断一致；28个读取键、原2799掩码计数、缓存/脚本/计划/series参照哈希均已保存。']
    lines+=['',f'总耗时（含输入）{r["elapsed_s_including_inputs"]:.6f} s，预声明预算120 s。未调Γcap、未追加宽容带、未改旧冻结文件。']
    (HERE/'hard_path_single_peak.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
