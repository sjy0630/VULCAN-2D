"""Post-hoc pairwise Fermi-window constraints; no spectra, fitting or ODE.

Only 1.1V training labels 4--35 and 0.9/1.3V labels >=4 are parsed numerically.
Current multiplicative 5% endpoints are jointly re-deembedded.
"""
from pathlib import Path
import csv, json, math, sys, time, hashlib, collections
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from vulcan2d.transistor import measured_transistor_mapping
KT=8.617333262145e-5*300


def included(g,c):
    return (g=='1.1' and 4<=c<=35) or (g in ('0.9','1.3') and c>=4)


def main():
    start=time.monotonic();mapping=measured_transistor_mapping();data={};zeros={};sha=hashlib.sha256()
    native=ROOT/'analysis/cross_polarity_physics_native_points.csv'
    cycles=ROOT/'analysis/cross_polarity_physics_cycles.csv'
    with cycles.open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            g=r['gate_V'];cid=int(r['cycle_id'])
            if not included(g,cid) or r['lookup_branch']!='mean' or r['window']!='primary':continue
            zeros[g,cid]=float(r['set_zero_bias_abs_current_max_A'])
    with native.open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            g=r['gate_V'];cid=int(r['cycle_id'])
            if not included(g,cid) or r['lookup_branch']!='mean' or r['phase']!='set':continue
            n=int(r['raw_index']);v=float(r['applied_abs_voltage_V']);i=float(r['observed_abs_current_A'])
            data.setdefault((g,cid),[]).append((n,v,i));sha.update(f'{g}:{cid}:{n}:{v:.17g}:{i:.17g}'.encode())
    knots={}
    for g in ('0.9','1.1','1.3'):
        x=np.unique(np.concatenate([c['drain_voltage_v'] for c in mapping.lookup.curves()]))
        y=np.array([mapping.current(v,float(g),'set') for v in x]);knots[g]=(x,y)
    def inv(i,g):
        x,y=knots[g];i=np.asarray(i);k=np.searchsorted(y,i,side='left');kk=np.clip(k,1,len(y)-1);d=y[kk]-y[kk-1]
        valid=(k>0)&(k<len(y))&(d>0)&np.isfinite(i)
        ans=x[kk-1]+(i-y[kk-1])*np.divide(x[kk]-x[kk-1],d,out=np.zeros_like(d),where=d>0)
        return np.where(valid,ans,np.nan)
    record_outputs=[];inverse_checks=[]
    policies=[('all_native_pairs',0,0.),('span_ge_3indices_and_0p05V',3,.05),('span_ge_3indices_and_0p10V',3,.10)]
    for (g,cid),pts in sorted(data.items()):
        if time.monotonic()-start>180:raise RuntimeError('Three-minute computation limit exceeded; no partial aggregate emitted')
        a=np.array(sorted(pts));raw=a[:,0].astype(int);v=a[:,1];i=a[:,2]
        assert len(raw)==len(set(raw.tolist()))
        vt=inv(i,g);vh=v-vt;cap=np.interp(v,*knots[g])
        cond=(inv(i*1.01,g)-inv(i*.99,g))/(.02*np.maximum(vh,1e-300))
        guard=np.isfinite(vh)&(v>0)&(i>0)&(vh>=.1-1e-12)&(vh<=.3+1e-12)&(vt/np.maximum(v,1e-300)<=.25)&(i/np.maximum(cap,1e-300)<=.5)&np.isfinite(cond)&(cond<=1)
        ilow=i/1.05;ihigh=i*1.05
        vhmax=v-inv(ilow,g);vhmin=v-inv(ihigh,g)
        for floor_name,threshold in [('original_positive',0.),('high_signal',max(100*zeros[g,cid],1e-10))]:
            keep=np.flatnonzero(guard&(i>threshold));rr=raw[keep];vv=v[keep];ii=i[keep]
            row=dict(gate_v=float(g),record_id=cid,floor_policy=floor_name,current_floor_a=threshold,
                full_scan_zero_reference_a=zeros[g,cid],cached_set_points=len(pts),n_guarded_points=int(sum(guard)),n_retained_points=len(keep),
                retained_raw_indices=rr.tolist(),retained_port_voltage_v=vv.tolist(),retained_current_a=ii.tolist(),
                retained_nominal_material_voltage_v=vh[keep].tolist(),retained_5percent_material_min_v=vhmin[keep].tolist(),
                retained_5percent_material_max_v=vhmax[keep].tolist(),pair_policy_results=[])
            for name,minraw,minspan in policies:
                answer=dict(pair_policy=name,min_raw_span=minraw,min_port_span_v=minspan)
                if len(keep)<5:
                    answer.update(status='insufficient_points',tested_pairs=0)
                    row['pair_policy_results'].append(answer);continue
                early,later=np.triu_indices(len(keep),1);e=keep[early];l=keep[later]
                span=(raw[l]-raw[e]>=minraw)&(v[e]-v[l]>=minspan-1e-12)
                # Uniform high/low ordering for every admitted current combination.
                ordered=np.isfinite(vhmin[e])&np.isfinite(vhmax[l])&(vhmin[e]>vhmax[l])&(vhmin[l]>0)
                valid=span&ordered;e=e[valid];l=l[valid]
                answer.update(tested_pairs=len(e),candidate_pairs_after_span=int(sum(span)),excluded_nonuniform_voltage_order=int(sum(span&~ordered)))
                if not len(e):
                    answer.update(status='insufficient_pair_span');row['pair_policy_results'].append(answer);continue
                hi=vhmax[e];lo=vhmin[l]
                logratio=np.log(ilow[e]/ihigh[l]);logbound=(hi-lo)/KT+np.log(-np.expm1(-hi/KT))-np.log(-np.expm1(-lo/KT))
                excess=(logratio-logbound)/np.log(10);k=int(np.argmax(excess));ee=int(e[k]);ll=int(l[k]);gap=float(excess[k])
                answer.update(status='conditional_violation' if gap>1e-10 else 'no_pair_violation',violation_pairs=int(sum(excess>1e-10)),maximum_excess_decade=gap,
                    strongest_pair=dict(earlier_raw_index=int(raw[ee]),later_raw_index=int(raw[ll]),raw_index_span=int(raw[ll]-raw[ee]),
                    port_voltage_span_v=float(v[ee]-v[ll]),nominal_material_span_v=float(vh[ee]-vh[ll]),
                    earlier_port_voltage_v=float(v[ee]),later_port_voltage_v=float(v[ll]),
                    earlier_observed_current_a=float(i[ee]),later_observed_current_a=float(i[ll]),
                    earlier_nominal_material_voltage_v=float(vh[ee]),later_nominal_material_voltage_v=float(vh[ll]),
                    earlier_most_favorable_current_a=float(ilow[ee]),later_most_favorable_current_a=float(ihigh[ll]),
                    earlier_most_favorable_material_voltage_v=float(hi[k]),later_most_favorable_material_voltage_v=float(lo[k]),
                    current_ratio_lower=float(np.exp(logratio[k])),fermi_bound_upper=float(np.exp(logbound[k])),ratio_over_bound=float(np.exp(logratio[k]-logbound[k]))))
                row['pair_policy_results'].append(answer)
            record_outputs.append(row)
        # Check two deterministic retained points against forward mapping for all three I values.
        chosen=np.flatnonzero(guard)
        if len(chosen):
            for j in (chosen[0],chosen[-1]):
                for target in (ilow[j],i[j],ihigh[j]):
                    tv=float(inv(np.array([target]),g)[0]);inverse_checks.append(abs(mapping.current(tv,float(g),'set')/target-1))
    groups=[]
    for g in (.9,1.1,1.3):
        for floor in ('original_positive','high_signal'):
            records=[r for r in record_outputs if r['gate_v']==g and r['floor_policy']==floor]
            for name,_,_ in policies:
                pairs=[(r,next(a for a in r['pair_policy_results'] if a['pair_policy']==name)) for r in records]
                counts=dict(collections.Counter(a['status'] for _,a in pairs));bad=[(r,a) for r,a in pairs if a['status']=='conditional_violation']
                group=dict(gate_v=g,floor_policy=floor,pair_policy=name,n_records=len(records),counts=counts,tested_pairs=sum(a['tested_pairs'] for _,a in pairs),
                    violating_record_ids=[r['record_id'] for r,_ in bad])
                if bad:
                    r,a=max(bad,key=lambda pair:pair[1]['maximum_excess_decade'])
                    group['strongest_violation']=dict(record_id=r['record_id'],**a)
                    group['excess_decade_min_median_max']=np.quantile([a['maximum_excess_decade'] for _,a in bad],[0,.5,1]).tolist()
                groups.append(group)
    report=dict(status='posthoc_algebraic_pair_screen_no_fit_no_spectral_inverse_no_ODE',temperature_k=300,kbt_ev=KT,
        formula='exp((Vhi-Vlo)/kBT)*(1-exp(-Vhi/kBT))/(1-exp(-Vlo/kBT)), uniform eta in [0,1]',
        scope='1.1V training record labels4--35 only;0.9/1.3V labels>=4. Gate and label filters run before numeric observation parsing.',
        current_budget='[I/1.05,I*1.05]; jointly re-deembedded endpoints; most favorable Ie/1.05 and Il*1.05',
        guards=dict(nominal_material_voltage_window_v=[.1,.3],transistor_voltage_fraction_max=.25,capacity_fraction_max=.5,inverse_condition_max=1,minimum_native_points=5),
        assumptions=['Quasi-static conduction and valid fixed measured T mapping.','Fixed equal300K reservoirs, nonnegative frozen transmission and fixed linear rigid voltage shift with eta in[0,1].','At lower comparison voltage later state current is no smaller than earlier state.'],
        limitations=['Post-hoc multiple-pair screening; neither5% budget nor counts constitute statistical significance or confidence intervals.',
            'No-pair-violation is a necessary-condition pass on retained low-bias points only, not validation of kinetics or complete curves.',
            'Current floor is sensitivity choice; full-scan zero reference is not noise standard deviation.',
            'Excluded/insufficient coverage is not a pass. Pair counts are correlated, not independent samples.',
            'Only source cache original port I/V reused; old deembedding fields ignored. Original cloud XLSX not reopened.',
            'No1.1 holdout currents parsed into numerical arrays; source CSV is streamed only to select authorized gates and labels.',
            'Crossgate labels are not assumed to identify the same cell, cycle chronology, or sweep timing.'],
        selected_port_observation_sha256=sha.hexdigest(),maximum_forward_inverse_relative_residual=max(inverse_checks),
        groups=groups,records=record_outputs,elapsed_s=time.monotonic()-start)
    report['source_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT/'fermi_window_training_crossgate.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# 训练及跨栅压 SET 返扫：有限温度全谱类成对必要界','',
        '纯代数及当前晶体管映射检验，无谱态反演、拟合或 ODE。只将 1.1 V 编号4–35训练数据，以及0.9/1.3 V编号≥4的原生端口观测纳入数值数组。','',
        '名义内部窗口 .10–.30 V；沿用三项 T 保护阈值。高信号为 I>max(100×完整扫描最大零偏参考,1e-10 A)。5% 为 [I/1.05,I×1.05]，每次电流变化联合去嵌入。任意 η∈[0,1]、300 K。','',
        '主跨度筛选：原生 raw_index 间隔≥3 且端口电压跨度≥.05 V；同时报告全部点对与≥.10 V跨度敏感性。要求整个电流区间的高低内部电压顺序一致。','',
        '| Gate | 电流门槛 | 跨度 | 条件违例 | 未见点对违例 | 点不足 | 跨度不足 | 检查点对数 |','|---:|---|---|---:|---:|---:|---:|---:|']
    for g in groups:
        c=g['counts'];lines.append(f'| {g["gate_v"]} | {g["floor_policy"]} | {g["pair_policy"]} | {c.get("conditional_violation",0)} | {c.get("no_pair_violation",0)} | {c.get("insufficient_points",0)} | {c.get("insufficient_pair_span",0)} | {g["tested_pairs"]} |')
    lines+=['','## 高信号且至少 .05 V / 3 原生索引跨度的最强见证','',
        '| Gate/编号 | raw_index | 端口V早→晚 | 电流A早→晚 | 5%最小电流比 | 5%最大上界 | 超界倍数 |','|---|---|---|---|---:|---:|---:|']
    for g in groups:
        if g['floor_policy']!='high_signal' or g['pair_policy']!='span_ge_3indices_and_0p05V':continue
        a=g.get('strongest_violation')
        if not a:continue
        p=a['strongest_pair'];lines.append(f'| {g["gate_v"]}/{a["record_id"]} | {p["earlier_raw_index"]}→{p["later_raw_index"]} | {p["earlier_port_voltage_v"]:.5g}→{p["later_port_voltage_v"]:.5g} | {p["earlier_observed_current_a"]:.8g}→{p["later_observed_current_a"]:.8g} | {p["current_ratio_lower"]:.8g} | {p["fermi_bound_upper"]:.8g} | {p["ratio_over_bound"]:.8g} |')
    lines+=['','## 解释边界','',
        '多点后验筛查，不是统计显著性；点对互相关，5%不是测得噪声置信区间。未见违例只表示保留低压点对没有否决此必要条件，不能证明整个模型或动力学有效。违反的是冻结非负谱/固定线性刚性移位/返扫传输不减/等温300K/当前有效准静态去嵌入的前提交集，不能唯一归因于正向状态回退。','',
        '逐记录保留点、实际电流/电压、5%联合去嵌入端点、最强点对和缺覆盖原因见JSON。没有使用旧缓存内部电压；未重新打开原工作簿；跨栅压不配对，不假定同器件或扫速。','',f'计算耗时 {report["elapsed_s"]:.3f}s；独立正向回代最大相对残差 {report["maximum_forward_inverse_relative_residual"]:.3g}。']
    (OUT/'fermi_window_training_crossgate.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(elapsed_s=report['elapsed_s'],forward_residual=report['maximum_forward_inverse_relative_residual'],groups=groups),indent=2),flush=True)

if __name__=='__main__':main()
