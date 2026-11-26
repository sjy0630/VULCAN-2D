"""Bounded, no-fit 0.9/1.3 V single-branch spectral ordering check."""
from pathlib import Path
import csv,json,math,sys,time,hashlib,collections
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from vulcan2d.transistor import measured_transistor_mapping
from vulcan2d.shared_spectral import SpectralShape


def main():
 start=time.monotonic();shape=SpectralShape.from_json();mapping=measured_transistor_mapping()
 native=ROOT/'analysis/cross_polarity_physics_native_points.csv';cycles=ROOT/'analysis/cross_polarity_physics_cycles.csv'
 zeros={};records={};arrays_hash=hashlib.sha256()
 with cycles.open(encoding='utf-8-sig',newline='') as f:
  for r in csv.DictReader(f):
   if r['gate_V'] not in ('0.9','1.3') or r['lookup_branch']!='mean' or r['window']!='primary':continue
   for phase in ('set','reset'):zeros[r['gate_V'],int(r['cycle_id']),phase]=float(r[phase+'_zero_bias_abs_current_max_A'])
 with native.open(encoding='utf-8-sig',newline='') as f:
  for r in csv.DictReader(f):
   if r['gate_V'] not in ('0.9','1.3') or r['lookup_branch']!='mean':continue
   key=r['gate_V'],int(r['cycle_id']),r['phase'];v=float(r['applied_abs_voltage_V']);i=float(r['observed_abs_current_A']);n=int(r['raw_index'])
   records.setdefault(key,[]).append((n,v,i));arrays_hash.update(f'{key}:{n}:{v:.17g}:{i:.17g}'.encode())
 report={'scope':'0.9/1.3 only; no fit, no1.1 residuals or holdout selection; original cached port I/V re-deembedded with current mapping.',
  'branches':{'set':'low-bias return selected by original extractor; raw_index increases in physical record order','reset':'low-bias initial outward selected by original extractor'},
  'window':'Nominal freshly deembedded |Vh| in[.10,.30]V; same nominal mask for all current-budget scenarios.',
  'guards':{'T_voltage_fraction_max':.25,'capacity_fraction_max':.5,'inverse_condition_max':1.,'root_log10_padding':1e-6},
  'floor_policies':['all_positive','greater_than_3_full_scan_zero_reference'],
  'current_budgets':[0,.01,.05], 'records':[]}
 knots={}
 for g in ('0.9','1.3'):
  for phase in ('set','reset'):
   look=mapping._active_lookup(phase);v=np.unique(np.concatenate([c['drain_voltage_v'] for c in look.curves()]));y=np.array([mapping.current(x,float(g),phase) for x in v]);knots[g,phase]=(v,y)
 def inverse_t(values,g,phase):
  x,y=knots[g,phase];values=np.asarray(values);j=np.searchsorted(y,values,side='left');valid=(j>0)&(j<len(y));jj=np.clip(j,1,len(y)-1);den=y[jj]-y[jj-1];valid &= den>0
  return np.where(valid,x[jj-1]+(values-y[jj-1])*np.divide(x[jj]-x[jj-1],den,out=np.zeros_like(den),where=den>0),np.nan)
 used=[];index=0
 for key,points in sorted(records.items()):
  g,cid,phase=key;points=sorted(points);a=np.array(points);n=a[:,0].astype(int);v=a[:,1];i=a[:,2];vt=inverse_t(i,g,phase);vh=v-vt
  capacity=np.interp(v,*knots[g,phase]);cond=(inverse_t(i*1.01,g,phase)-inverse_t(i*.99,g,phase))/(.02*np.maximum(vh,1e-300))
  # Raw low-current points remain in the all-positive result; floor is a separately labelled sensitivity.
  reasons=collections.Counter();keep=[]
  for k in range(len(v)):
   why=[]
   if not np.isfinite(vh[k]) or i[k]<=0 or v[k]<=0:why.append('invalid_or_zero_port')
   elif vh[k]<=0:why.append('nonpositive_material_voltage')
   elif not .1-1e-12<=vh[k]<=.3+1e-12:why.append('outside_nominal_internal_window')
   else:
    if vt[k]/v[k]>.25:why.append('transistor_voltage_fraction_guard')
    if i[k]/capacity[k]>.5:why.append('transistor_capacity_guard')
    if not np.isfinite(cond[k]) or cond[k]>1:why.append('inverse_condition_guard')
   if why:reasons.update(why)
   else:keep.append(k)
  row={'gate_v':float(g),'record_id':cid,'phase':phase,'cached_native_points':len(v),'zero_reference_a':zeros[key],
       'nominal_exclusion_reasons':dict(reasons),'retained_raw_indices':n[keep].tolist(),
       'retained_port_v':v[keep].tolist(),'retained_observed_abs_current_a':i[keep].tolist(),
       'retained_fresh_material_abs_voltage_v':vh[keep].tolist(),'root_ids':[]}
  for k in keep:
   ids=[]
   for factor in [1.,1/1.01,1.01,1/1.05,1.05]:
    target=i[k]*factor;local=float(v[k]-inverse_t(np.array([target]),g,phase)[0]);ids.append(index)
    used.append((local if phase=='set' else -local,target));index+=1
   row['root_ids'].append(ids)
  report['records'].append(row)
 print(json.dumps({'stage':'port_inventory','records':len(records),'roots':len(used),'elapsed_s':time.monotonic()-start}),flush=True)
 used=np.array(used);voltage=used[:,0];target=used[:,1]
 valid=np.isfinite(voltage)&(abs(voltage)>0)&(target>0)&(target<abs(voltage)/shape.ron_ohm)
 with np.load(OUT/'transport_g010_v0025.npz',allow_pickle=False) as z:lg=z['log_gamma'];lv=z['log_voltage'];li=z['log_current']
 x=np.full(len(target),np.nan)
 for j in np.flatnonzero(valid):
  pos=int(np.clip(np.searchsorted(lv,math.log(abs(voltage[j])))-1,0,len(lv)-2));f=(math.log(abs(voltage[j]))-lv[pos])/(lv[pos+1]-lv[pos]);profile=(1-f)*li[0 if voltage[j]>0 else 1,:,pos]+f*li[0 if voltage[j]>0 else 1,:,pos+1]
  x[j]=np.interp(math.log(target[j]),profile,lg)
 ids=np.flatnonzero(valid);xx=x[ids];vv=voltage[ids];tt=target[ids]
 def residual(q,v=vv,t=tt,order=32):return np.log(abs(shape.current_at_log_gamma(v,q,order=order))/t)
 y=residual(xx);oldx=xx.copy();oldy=y.copy();xx=xx+.001;y=residual(xx)
 for iteration in range(4):
  active=abs(y)>1e-10
  if not np.any(active):break
  step=np.divide(y*(xx-oldx),y-oldy,out=np.zeros_like(y),where=abs(y-oldy)>1e-20)
  new=np.where(active,xx-np.clip(step,-2,2),xx);oldx,oldy=xx,y;xx=new;y=residual(xx)
  print(json.dumps({'stage':'exact_inverse','iteration':iteration,'max_log_current_residual':float(max(abs(y))),'elapsed_s':time.monotonic()-start}),flush=True)
 x[ids]=xx;pad=1e-6*np.log(10);low=residual(xx-pad);high=residual(xx+pad)
 certified=np.zeros(len(target),bool);certified[ids]=(low<=0)&(high>=0)&np.isfinite(y)
 report['numerical']={'root_count':len(target),'certified_root_count':int(sum(certified)),
  'max_exact_log_current_residual':float(max(abs(y))), 'exact_root_padding_decade':1e-6,
  'table_role':'Initial root guess only; exact spectral evaluations bracket every certified inverse.'}
 # Independent order64 checks at deterministic evenly spaced used roots.
 sample=ids[np.unique(np.linspace(0,len(ids)-1,min(128,len(ids))).astype(int))]
 delta=np.log10(abs(shape.current_at_log_gamma(voltage[sample],x[sample],order=64))/target[sample]);report['numerical']['max_order64_current_residual_decade']=float(max(abs(delta)))
 counts=collections.Counter()
 for row in report['records']:
  rids=np.array(row.pop('root_ids'),int);curr=np.array(row['retained_observed_abs_current_a']);raw=np.array(row['retained_raw_indices']);answers=[]
  for floor in report['floor_policies']:
   mask=np.ones(len(curr),bool) if floor=='all_positive' else curr>3*row['zero_reference_a']
   for p,lcol,hcol in [(0,0,0),(.01,1,2),(.05,3,4)]:
    keep=np.flatnonzero(mask);r={'floor':floor,'budget_fraction':p,'n_points':len(keep),'floor_removed_points':int(sum(~mask))}
    if len(keep)<5:r.update(status='insufficient_points',reason='Requires5 native retained points; no interpolation or cross-gap data invented.')
    else:
     roots=rids[keep];bad=~(certified[roots[:,lcol]]&certified[roots[:,hcol]])
     if np.any(bad):r.update(status='numerical_or_domain_unknown',unknown_raw_indices=raw[keep][bad].tolist())
     else:
      lower=x[roots[:,lcol]]/np.log(10)-1e-6;upper=x[roots[:,hcol]]/np.log(10)+1e-6
      # Interval isotonic feasibility; diagnostics are necessary, never kinetics fits.
      if row['phase']=='set':cum=np.maximum.accumulate(lower);violation=cum-upper
      else:cum=np.minimum.accumulate(upper);violation=lower-cum
      worst=int(np.argmax(violation));gap=max(0.,float(violation[worst]));r.update(status='interval_order_feasible' if gap==0 else 'conditional_order_violation',
       maximum_interval_gap_decade=gap,worst_later_raw_index=int(raw[keep][worst]),
       gamma_lower_ev=np.power(10,lower).tolist(),gamma_upper_ev=np.power(10,upper).tolist(),raw_indices=raw[keep].tolist(),
       native_index_gaps=int(np.sum(np.diff(raw[keep])!=1)))
    answers.append(r);counts[str((row['gate_v'],row['phase'],floor,p,r['status']))]+=1
  row['results']=answers
 report['counts']=dict(counts);report['elapsed_s']=time.monotonic()-start
 report['provenance']={'cache_native_sha256':hashlib.sha256(native.read_bytes()).hexdigest(),'cache_cycles_sha256':hashlib.sha256(cycles.read_bytes()).hexdigest(),'selected_observation_fields_sha256':arrays_hash.hexdigest(),
  'source_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_fields_used':['gate_V','cycle_id','phase','raw_index','applied_abs_voltage_V','observed_abs_current_A'],
  'old_deembedding_reused':False,'raw_workbooks':'Cloud placeholders; this run uses archived observation fields, not a fresh workbook extraction.'}
 report['limitations']=['A monotone gamma condition depends on fixed spectral readout and fixed T mapping; it is not mechanism identification.',
 'Curve IDs are file labels, not confirmed cross-gate same-cell or chronological pairs. No SET-to-RESET equality is assumed.',
 'No shared scan speed or physical time is assumed; only within-record order.',
 '1%/5% bands and3zero floor are sensitivity scenarios, not noise confidence intervals. Nominal small violations alone are not physical refutation.',
 'Subset feasibility does not certify the whole curve. Exclusions and insufficient coverage remain explicit; skipped gaps are not filled.',
 '0.9/1.3 data were explored historically and are additional constraints, not a pristine prospective holdout.',
 'Source CSV contains1.1 rows which are skipped before numeric observation processing; no1.1 holdout currents contribute to this audit.']
 (OUT/'cross_gate_lowbias_order.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 lines=['# 0.9/1.3 V低偏单支路状态方向检验','',report['scope'],'',
 '重新使用当前统一晶体管映射去嵌入缓存中的原始端口I/V，未读取旧内部电压作为观测，未拟合。窗口为名义内部|Vh|=0.10–0.30V；保护阈值沿用原低偏诊断。SET返回按原raw_index递增要求gamma不减，RESET起始要求gamma不增。','',
 '| gate | 支路 | floor | 电流预算 | 可判点不足 | 数值/域未知 | 区间方向可行 | 条件违例 |','|---:|---|---|---:|---:|---:|---:|---:|']
 for gate in (.9,1.3):
  for phase in ('set','reset'):
   for floor in report['floor_policies']:
    for p in (0,.01,.05):
     vals=[counts[str((gate,phase,floor,p,s))] for s in ('insufficient_points','numerical_or_domain_unknown','interval_order_feasible','conditional_order_violation')]
     lines.append(f'| {gate} | {phase} | {floor} | {p:.0%} | '+' | '.join(map(str,vals))+' |')
 lines+=['','逐记录raw_index、保留点、去嵌入电压、每种预算的gamma区间及失效理由见JSON。正向/负向读出采用各自符号；不同gate不配对，不假设扫速相同。','',
 '可行只表示存在满足方向的区间选择，不能证明完整动力学。违例只限制固定谱形+当前T映射+单向状态；尤其原值的小违例可由噪声或查表误差产生，须看1%/5%与datafloor敏感性。','',
 '缓存来源及配对限制见 CROSS_GATE_DATA_CONSTRAINT_PLAN.md；原始xlsx为云端占位，本轮不声称重新核对原表。','',
 f'精确反演认证 {sum(certified)}/{len(target)}，64阶抽查最大电流残差 {max(abs(delta)):.3g} decade；耗时 {report["elapsed_s"]:.2f}s。']
 (OUT/'cross_gate_lowbias_order.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'stage':'complete','elapsed_s':report['elapsed_s'],'counts':dict(counts)}),flush=True)

if __name__=='__main__':main()
