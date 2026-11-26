"""Same bare record slow-pre-switch versus abrupt-switch beta1 budgets.

No ODE, series observations, alpha fit, new shape, or altered seed report.
"""
from pathlib import Path
import sys,json,math,time,hashlib,signal
import numpy as np

START=time.monotonic()
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from vulcan2d.shared_spectral import SpectralShape


def deadline():
    if time.monotonic()-START>120:raise TimeoutError('Declared120second cap')


def main():
    source=OUT/'budget_seeds.json';source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    old=json.loads(source.read_text());shape=SpectralShape.from_json()
    rows=old['outward_compliance_condition']['native_brackets']
    records=[];loaded=hashlib.sha256();inverse_v=[];inverse_i=[]
    with np.load(ROOT/'analysis/shared_hbn_2026-11-03/data/native_observations.npz') as z:
        for row in rows:
            cid=row['record_id'];prefix=f'1r_cycle_{cid}_set'
            v=z[prefix+'_voltage'].copy();i=abs(z[prefix+'_current'].copy())
            for a in (v,i):loaded.update(a.tobytes())
            apex=int(np.argmax(v));j=int(np.argmin(abs(v[:apex+1]-2.)))
            assert j<row['before_index'] and abs(v[j]-2)<1e-5 and i[j]>0
            record=dict(row);record.update(low_index=j,low_voltage_v=float(v[j]),low_current_a=float(i[j]),
                                         low_current_band_a=[float(i[j]/1.05),float(i[j]*1.05)],
                                         pre_current_band_a=[row['before_current_a']/1.05,row['before_current_upper_a']])
            records.append(record)
            inverse_v.extend([v[j],v[j],row['before_voltage_v']])
            inverse_i.extend([i[j]/1.05,i[j]*1.05,row['before_current_a']/1.05])
    vv=np.array(inverse_v);ii=np.array(inverse_i);lo=np.full(len(vv),-32.);hi=np.full(len(vv),15.)
    for _ in range(43):
        middle=.5*(lo+hi);value=shape.current_at_log_gamma(vv,middle)
        positive=value>ii;hi=np.where(positive,middle,hi);lo=np.where(positive,lo,middle);deadline()
    # Preserve outward numerical brackets in the direction favorable to feasibility.
    for j,row in enumerate(records):
        row['gamma_2V_band_ev']=[float(math.exp(lo[3*j])),float(math.exp(hi[3*j+1]))]
        row['gamma_pre_band_ev']=[float(math.exp(lo[3*j+2])),row['gamma_before_upper_ev']]
        row['endpoint_monotone_possible']=row['gamma_2V_band_ev'][0]<=row['gamma_pre_band_ev'][1]
        row['budgets']=[]
    gc_results=[]
    for row in records:
        if not row['endpoint_monotone_possible']:continue
        glo=row['gamma_2V_band_ev'][0];ghi=row['gamma_pre_band_ev'][1]
        gamma=np.geomspace(glo,ghi,129)
        count=max(1,int(math.ceil((row['before_voltage_v']-row['low_voltage_v'])/.05)))
        voltage=np.linspace(row['low_voltage_v'],row['before_voltage_v'],count+1)
        dv=np.diff(voltage)
        current=shape.current_at_gamma(voltage[:-1,None],gamma[None,:-1])
        row['lower_integral_voltage_nodes_v']=voltage.tolist()
        row['gamma_product_partition_intervals']=128
        for gc in [.2,.4,1.]:
            # F increases in gamma; exp(u)=1+gc/gamma decreases. For each
            # partition this product is a lower bound; then minimize the cells.
            jlower=np.min(current*(1+gc/gamma[None,1:]),axis=1)
            row['budgets'].append({'gamma_c_ev':gc,'normalized_drive_J_lower_a':jlower.tolist()})
        deadline()
    for gc in [.2,.4,1.]:
        boundrow=next(g for g in old['groups'] if g['gamma_c_ev']==gc)
        jsmax=boundrow['JS_actual_initial_activation_upper_a']
        js=np.geomspace(1e-7,.98*jsmax,181);c=np.log(.001/js)
        all_lower=[];all_upper=[];record_summaries=[]
        for row in records:
            if not row['endpoint_monotone_possible']:
                record_summaries.append({'record_id':row['record_id'],'status':'endpoint_monotone_incompatible'})
                continue
            g2lo=row['gamma_2V_band_ev'][0];gprehi=row['gamma_pre_band_ev'][1]
            u2max=math.log1p(gc/g2lo);upremin=math.log1p(gc/gprehi)
            uaftermax=math.log1p(gc/row['gamma_after_lower_ev'])
            delta=max(0.,u2max-upremin)
            item=next(b for b in row['budgets'] if b['gamma_c_ev']==gc)
            jlow=np.array(item['normalized_drive_J_lower_a']);dv=np.diff(row['lower_integral_voltage_nodes_v'])
            integral=np.sum(np.maximum(np.log(jlow[:,None]/js[None,:]),0)*dv[:,None],axis=0)
            upper=np.divide(delta,integral,out=np.full(len(js),np.inf),where=integral>0)
            lower=np.maximum(0.,np.log((upremin+c)/(uaftermax+c))/row['path_bracket_v'])
            ratio=lower/upper;best=int(np.argmin(ratio));passes=lower<=upper
            item.update(allowed_slow_state_drop=delta,pre_jump_u_lower=upremin,post_jump_u_upper=uaftermax)
            summary={'record_id':row['record_id'],'status':'no_finite_grid_rate_overlap' if not np.any(passes) else 'necessary_rate_overlap_exists',
                     'finite_JS_overlap_count':int(sum(passes)), 'minimum_lower_over_upper':float(ratio[best]),
                     'best_JS_a':float(js[best]),'jump_k_lower_per_v':float(lower[best]),
                     'slow_k_upper_per_v':float(upper[best]) if math.isfinite(upper[best]) else None,
                     'slow_integral_lower_at_best':float(integral[best]),'allowed_slow_state_drop':delta}
            record_summaries.append(summary);all_lower.append(lower);all_upper.append(upper)
        if all_lower:
            joint_lower=np.max(all_lower,axis=0);joint_upper=np.min(all_upper,axis=0);joint_ratio=joint_lower/joint_upper
            j=int(np.argmin(joint_ratio))
            joint={'minimum_joint_lower_over_upper':float(joint_ratio[j]),'best_JS_a':float(js[j]),
                   'k_lower_per_v':float(joint_lower[j]),'k_upper_per_v':float(joint_upper[j]),
                   'finite_JS_overlap_count':int(sum(joint_lower<=joint_upper))}
        else:joint=None
        gc_results.append({'gamma_c_ev':gc,'JS_grid_min_max_a':[float(js[0]),float(js[-1])],'JS_grid_count':len(js),
                           'record_summaries':record_summaries,'joint_record_interval':joint})
    report={'status':'completed_same_record_necessary_budget_audit_no_ODE','elapsed_s':time.monotonic()-START,
            'gamma_c_ev':[.2,.4,1.],'current_budget_fraction':.05,'Icc_a':.001,'arrival_target_a':.000995,
            'records':records,'gamma_c_results':gc_results,
            'scope':['No series data or alpha is used in either rate bound; each contradiction is within one bare record.',
                     'Beta1 threshold law, fixed spectral shape, continuous monotone SET state, ideal1mA compliance and pre-compliance actualV=command are assumed.',
                     '5% bands are a declared sensitivity, not confidence intervals; gates are not changed.',
                     'Result is finite original JS grid only, not proof over all positive thresholds or all material dynamics.',
                     'Per-record feasibility need not imply a common parameter fit; joint interval is separately shown.',
                     'No per-cycle resetting, ODE, held-out arrays, extra parameter family, alpha enlargement or source report overwrite.'],
            'numerical':{'inverse_log_gamma_bisections':43,'gamma_product_partitions':128,'voltage_left_interval_max_v':.05},
            'provenance':{'budget_seeds_sha256_before':source_hash,'budget_seeds_unchanged':source_hash==hashlib.sha256(source.read_bytes()).hexdigest(),
                          'bare_training_arrays_sha256':loaded.hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    deadline()
    (OUT/'same_record_budget_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# 同一裸记录：缓变段与骤变段的速率相容性','',
           '固定原beta1谱模型、理想1mA限流、三个gamma_c与原181点有限JS网格。两段均来自同一条裸SET外扫，不使用series数据或alpha。2V/限流前点各用5%乘性电流带。','',
           '| gamma_c/eV | 记录内无交集 | 端点方向不相容 | 最宽容记录的最小下/上界比 | 联合下/上界最小比 |','|---:|---:|---:|---:|---:|']
    for g in gc_results:
        r=g['record_summaries'];valid=[x for x in r if 'minimum_lower_over_upper' in x]
        lines.append(f'| {g["gamma_c_ev"]} | {sum(x["status"]=="no_finite_grid_rate_overlap" for x in r)}/{len(r)} | {sum(x["status"]=="endpoint_monotone_incompatible" for x in r)} | {min(x["minimum_lower_over_upper"] for x in valid):.6g} | {g["joint_record_interval"]["minimum_joint_lower_over_upper"]:.6g} |')
    lines+=['','比值>1表示在本有限网格内所需跳变速率大于缓变段允许速率。不能用未知扫速比解决同一记录内部的这两个要求。',
            '缓变驱动下界：对每个gamma分片取F(V,gamma_lo)*(1+gamma_c/gamma_hi)，再取所有分片最小值；沿电压使用左Riemann正超驱动积分。这个保守积分除允许状态落差给kS上界。',
            '骤变下界仍是ln[(u_pre_lower+c)/(u_after_upper+c)]/DeltaV，c=ln(Icc/JS)>0。完整每条记录电流带、反演区间、积分下界和最佳有限阈值见JSON。',
            '当前结论只针对声明数据带、固定谱、原beta1状态律和理想准静态限流。过冲、真实时序、不同输运或其他动力学须单独核实。',
            f'用时{report["elapsed_s"]:.3f}s；原budget_seeds.json哈希不变：{report["provenance"]["budget_seeds_unchanged"]}。']
    (OUT/'same_record_budget_audit.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'elapsed_s':report['elapsed_s'],'summary':[{'gamma_c_ev':g['gamma_c_ev'],'joint':g['joint_record_interval'],
              'per_record_min_ratios':[x.get('minimum_lower_over_upper') for x in g['record_summaries']]} for g in gc_results]},indent=2))


if __name__=='__main__':main()
