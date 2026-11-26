"""Pure rescoring of saved5% gamma intervals; no spectral inverse or fit."""
from pathlib import Path
import json,hashlib,time,math
import numpy as np

OUT=Path(__file__).resolve().parent
SOURCE=OUT/'cross_gate_lowbias_order.json'

def score(record,strong):
 original=next(r for r in record['results'] if r['floor']=='all_positive' and r['budget_fraction']==.05)
 raw=np.array(record['retained_raw_indices'],int);current=np.array(record['retained_observed_abs_current_a'])
 threshold=max(100*record['zero_reference_a'],1e-10 if record['phase']=='set' else 1e-11) if strong else 0.
 keep=current>threshold
 result={'gate_v':record['gate_v'],'record_id':record['record_id'],'phase':record['phase'],
  'sensitivity_floor_a':threshold,'full_scan_zero_reference_a':record['zero_reference_a'],
  'original_window_points':len(raw),'retained_points':int(sum(keep)),'removed_by_current_floor':int(sum(~keep)),
  'kept_raw_indices':raw[keep].tolist()}
 if sum(keep)<5:
  result.update(status='insufficient_points',reason='Fewer than5 original guarded points remain; no filling or new inverse.')
  return result
 if original['status'] not in ('interval_order_feasible','conditional_order_violation'):
  result.update(status='saved_inverse_unavailable',reason=original['status']);return result
 lookup={int(k):j for j,k in enumerate(original['raw_indices'])}
 if any(int(k) not in lookup for k in raw[keep]):
  result.update(status='saved_index_mismatch');return result
 rows=np.array([lookup[int(k)] for k in raw[keep]],int)
 lower=np.log10(np.array(original['gamma_lower_ev'])[rows]);upper=np.log10(np.array(original['gamma_upper_ev'])[rows])
 best_index=0;worst=(0.,0,0)
 for j in range(len(lower)):
  if record['phase']=='set':
   if lower[j]>lower[best_index]:best_index=j
   gap=lower[best_index]-upper[j]
  else:
   if upper[j]<upper[best_index]:best_index=j
   gap=lower[j]-upper[best_index]
  if gap>worst[0]:worst=(float(gap),best_index,j)
 gap,first,last=worst;result.update(status='conditional_order_violation' if gap>0 else 'interval_order_feasible',maximum_interval_gap_decade=gap)
 if gap>0:
  positions=np.flatnonzero(keep);witness=[]
  for which,j in [('earlier',first),('later',last)]:
   k=positions[j];i=float(current[k]);zero=record['zero_reference_a']
   witness.append({'position':which,'raw_index':int(raw[k]),'port_abs_voltage_v':record['retained_port_v'][k],
    'observed_abs_current_a':i,'nominal_material_abs_voltage_v':record['retained_fresh_material_abs_voltage_v'][k],
    'current_over_full_scan_zero_reference':i/zero if zero>0 else None,
    'gamma_5percent_interval_ev':[float(10**lower[j]),float(10**upper[j])]})
  result['maximum_violation_witness_pair']=witness
 return result

def main():
 start=time.monotonic();d=json.loads(SOURCE.read_text());selected=[r for r in d['records'] if r['record_id']>=4]
 report={'status':'saved_arrays_only_rescore_no_inverse_no_fit','input_record_count':len(d['records']),
  'selected_record_count':len(selected),'excluded_label_1_to_3_count':len(d['records'])-len(selected),
  'budget_fraction':.05,'window_and_T_guards':'Exactly inherited from original lowbias audit; only record/current masks change.',
  'scenarios':{},'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
  'source_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  'scope':['IDs>=4 is startup-label sensitivity, not proof of common chronology across gate groups.',
   '100 times maximum full-scan zero reference is an explicit sensitivity floor, not a measured noise standard deviation or confidence threshold.',
   'SET1e-10A andRESET1e-11A are declared sensitivity cutoffs, not claimed instrument detection limits.',
   'Existing5% current interval roots and numerical padding are reused unchanged; no new deembedding, inverse, fit, or1.1 current read.',
   'Coverage loss is retained explicitly. Removing a violating point does not validate the omitted segment.',
   'Conditional violation constrains fixed spectral readout + current T mapping + scalar one-way state; it does not identify a unique mechanism or exclude all shared equations.']}
 for name,strong in [('records_ge4',False),('records_ge4_high_current_floor',True)]:
  records=[score(r,strong) for r in selected];groups=[]
  for g in (.9,1.3):
   for ph in ('set','reset'):
    group=[r for r in records if r['gate_v']==g and r['phase']==ph];bad=[r for r in group if r['status']=='conditional_order_violation']
    counts={status:sum(r['status']==status for r in group) for status in ('conditional_order_violation','interval_order_feasible','insufficient_points','saved_inverse_unavailable','saved_index_mismatch')}
    row={'gate_v':g,'phase':ph,'n_records':len(group),'counts':counts,
     'retained_points':sum(r['retained_points'] for r in group),'removed_by_current_floor':sum(r['removed_by_current_floor'] for r in group),
     'violation_record_ids':[r['record_id'] for r in bad],
     'insufficient_record_ids':[r['record_id'] for r in group if r['status']=='insufficient_points']}
    if bad:
     gaps=[r['maximum_interval_gap_decade'] for r in bad];worst=max(bad,key=lambda r:r['maximum_interval_gap_decade'])
     row.update(gap_min_median_max_decade=np.quantile(gaps,[0,.5,1]).tolist(),
      maximum_violation={'record_id':worst['record_id'],'gap_decade':worst['maximum_interval_gap_decade'],'witness_pair':worst['maximum_violation_witness_pair']})
    groups.append(row)
  report['scenarios'][name]={'definition':'ID>=4; '+('I>max(100*full_scan_zero_ref,1e-10A SET or1e-11A RESET)' if strong else 'retain all original positive-current guarded points'),
    'groups':groups,'records':records}
 report['elapsed_s']=time.monotonic()-start
 (OUT/'cross_gate_lowbias_robustness.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 lines=['# 跨栅压低偏检验：启动编号与电流下限敏感性','',
  '只重评分已保存的5% gamma区间；原402条分析、主拟合门槛、反演结果不变。未新增拟合或反演。', '',
  'A：保留编号≥4；B：在A上加I>max(100×完整扫描最大零偏参考，SET 1e-10 A / RESET 1e-11 A)。这些是明确的敏感性条件，零偏参考不是噪声标准差，固定电流门槛也不是已测仪器检测限。','',
  '| 情景 | gate | 支路 | 条件违例 | 方向可行 | 覆盖不足 | 数值/索引未知 |','|---|---:|---|---:|---:|---:|---:|']
 for name,scenario in report['scenarios'].items():
  for g in scenario['groups']:
   c=g['counts'];lines.append(f'| {"B" if name.endswith("floor") else "A"} | {g["gate_v"]} | {g["phase"]} | {c["conditional_order_violation"]} | {c["interval_order_feasible"]} | {c["insufficient_points"]} | {c["saved_inverse_unavailable"]+c["saved_index_mismatch"]} |')
 lines += ['','## 各组最大违例的两点电流','', '| 情景 | gate/支路 | 编号 | gamma区间缺口/decade | 早→晚端口电流/A | raw_index |','|---|---|---:|---:|---|---|']
 for name,scenario in report['scenarios'].items():
  for g in scenario['groups']:
   w=g.get('maximum_violation')
   if not w:continue
   a,b=w['witness_pair'];lines.append(f'| {"B" if name.endswith("floor") else "A"} | {g["gate_v"]}/{g["phase"]} | {w["record_id"]} | {w["gap_decade"]:.6g} | {a["observed_abs_current_a"]:.6g} → {b["observed_abs_current_a"]:.6g} | {a["raw_index"]} → {b["raw_index"]} |')
 lines += ['','每个见证点的原端口电压、重新去嵌入电压、5% gamma区间、相对零偏参考倍数和逐记录覆盖原因都保存在JSON。', '',
  'B情景减少违例时，必须同时检查覆盖不足是否增加；筛掉的曲线段不能据此宣称已通过。剩余违例仍是固定谱形/固定T映射/单向状态三项前提的条件约束，不等于所有共享物理模型无解。', '',f'402条中保留{len(selected)}条；纯数组重评分耗时{report["elapsed_s"]:.3f}s。']
 (OUT/'cross_gate_lowbias_robustness.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({'elapsed_s':report['elapsed_s'],'groups':{k:v['groups'] for k,v in report['scenarios'].items()}},indent=2))

if __name__=='__main__':main()
