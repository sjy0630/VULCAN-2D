"""Pure algebra over saved conservative J bounds; no new simulation or grid."""
from pathlib import Path
import json,math,hashlib,time

OUT=Path(__file__).resolve().parent


def main():
    start=time.monotonic();source=OUT/'same_record_budget_audit.json'
    d=json.loads(source.read_text());records=[]
    for row in d['records']:
        for b in row['budgets']:
            voltage=row['lower_integral_voltage_nodes_v']
            length=voltage[-1]-voltage[0]
            D=sum((y-x)*math.log(j/.001) for x,y,j in zip(voltage[:-1],voltage[1:],b['normalized_drive_J_lower_a']))
            B=b['pre_jump_u_lower'];A=b['post_jump_u_upper'];du=b['allowed_slow_state_drop'];delta=row['path_bracket_v']
            applicable=D>0 and B>A>=0 and du>0 and B>0
            bound=(B-A)/(du*delta)*min(length,D/B) if applicable else None
            records.append({'gamma_c_ev':b['gamma_c_ev'],'record_id':row['record_id'],
                            'slow_path_length_v':length,'D_log_drive_left_integral_lower':D,
                            'slow_state_drop_upper':du,'jump_pre_u_lower_B':B,'jump_post_u_upper_A':A,
                            'jump_path_length_v':delta,'positive_D_and_state_order':applicable,
                            'all_JS_ratio_lower_bound':bound,
                            'excludes_every_0ltJSltIcc':bool(bound is not None and bound>1)})
    groups=[]
    for gc in [.2,.4,1.]:
        r=[x for x in records if x['gamma_c_ev']==gc]
        values=[x['all_JS_ratio_lower_bound'] for x in r if x['all_JS_ratio_lower_bound'] is not None]
        bad=[x['record_id'] for x in d['records'] if not x['endpoint_monotone_possible']]
        groups.append({'gamma_c_ev':gc,'input_records':len(d['records']),'records':len(r),'applicable_records':len(values),
                       'endpoint_direction_incompatible_ids':bad,
                       'records_excluding_all_JS':sum(x['excludes_every_0ltJSltIcc'] for x in r),
                       'minimum_all_JS_ratio_lower_bound':min(values) if values else None,
                       'maximum_all_JS_ratio_lower_bound':max(values) if values else None})
    out={'status':'pure_algebra_full_positive_subcompliance_threshold_bound',
         'threshold_domain':'0 < JS < Icc=1mA; c=ln(Icc/JS)>0; endpoint c=0 also satisfies the weak inequality',
         'formula':'ratio >= (B-A)/(slow_delta*jump_path)*min(slow_path,D/B), provided D>0,B>A>=0,slow_delta>0',
         'scope':['Only three fixed gamma_c values and14declared bare records; no continuous gamma_c exclusion.',
                  'No series/alpha information, threshold grid, new spectrum evaluation, ODE, fitting, or limit-threshold seed.',
                  'Conditional on original beta1 law, fixed spectral readout, ideal1mA compliance, continuous monotone state and declared5% endpoint bands.',
                  'D uses saved conservative product bounds and leftRiemann sum; individual log-drive nodes need not all be positive.'],
         'groups':groups,'records':records,'elapsed_s':time.monotonic()-start,
         'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
         'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'same_record_full_threshold_bound.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    lines=['# 同一裸记录的全JS必要界（不再依赖阈值网格）','',
           '以下只对现有三个gamma_c与14条裸训练记录，不扫描或外推连续gamma_c域。没有series数据或alpha。','',
           '令c=ln(Icc/JS)>0，缓变路径长度L，允许状态落差Δu，jump前态下界B、后态上界A，jump长度δ。保存的分片/左和给D=ΣΔV ln(J_lower/Icc)。只要D>0，即使某些单点log驱动为负，仍有ΣΔV[c+ln(J_lower/Icc)]_+≥Lc+D。','',
           '又ln[(B+c)/(A+c)]≥(B−A)/(B+c)，故所需jump速率/允许slow速率 ≥ (B−A)/(Δuδ) × min(L,D/B)。这里min来自c≥0的有理函数全域下确界，没有取J→0生成种子。','',
           '| gamma_c/eV | 速率界适用记录 | 全JS下界>1的记录 | 另有端点方向不相容 | 最宽容全JS比值下界 |','|---:|---:|---:|---:|---:|']
    for g in groups:lines.append(f'| {g["gamma_c_ev"]} | {g["applicable_records"]}/{g["input_records"]} | {g["records_excluding_all_JS"]}/{g["input_records"]} | {g["endpoint_direction_incompatible_ids"]} | {g["minimum_all_JS_ratio_lower_bound"]:.7g} |')
    lines+=['','若下界>1，则该记录在所有0<JS<Icc内都没有兼顾缓变/骤变的kS；不只是181点网格没找到。',
            '这仍是固定谱、beta1状态律、理想限流和5%条件数据带的限制；不能外推为所有材料动力学或所有统一物理模型无解。真实过冲/时间协议若破坏前提，要另核实，不能由此假称已识别机制。']
    (OUT/'same_record_full_threshold_bound.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'groups':groups,'elapsed_s':out['elapsed_s']},indent=2))


if __name__=='__main__':main()
