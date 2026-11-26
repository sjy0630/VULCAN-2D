"""Extract two existing SET witnesses and evaluate a proposed thermal bound.

This is a numerical conditional check, not a proof of the proposed bound.
No new fit, spectral inversion, point search, or 1.1 V observations are used.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from vulcan2d.transistor import measured_transistor_mapping

KB_EV_PER_K = 8.617333262145e-5
TEMPERATURE_K = 300.0
KT = KB_EV_PER_K * TEMPERATURE_K


def proposed_bound(hi, lo):
    assert hi > lo > 0.0
    return math.exp((hi-lo)/KT) * (-math.expm1(-hi/KT))/(-math.expm1(-lo/KT))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = OUT/'cross_gate_lowbias_robustness.json'
    native = ROOT/'analysis/cross_polarity_physics_native_points.csv'
    saved = json.loads(source.read_text())
    groups = saved['scenarios']['records_ge4_high_current_floor']['groups']
    selected = [g for g in groups if g['phase']=='set']
    mapping = measured_transistor_mapping()
    raw = {}
    with native.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            if row['lookup_branch']!='mean' or row['phase']!='set' or row['gate_V'] not in ('0.9','1.3'):
                continue
            key = (float(row['gate_V']), int(row['cycle_id']), int(row['raw_index']))
            raw[key] = (float(row['applied_abs_voltage_V']), float(row['observed_abs_current_A']))
    results = []
    max_inverse_residual = 0.0
    for group in selected:
        gate = group['gate_v']
        pair = group['maximum_violation']
        record = pair['record_id']
        points = []
        for old in pair['witness_pair']:
            key = gate, record, old['raw_index']
            port, obs = raw[key]
            assert port == old['port_abs_voltage_v']
            assert obs == old['observed_abs_current_a']
            conditions = {}
            for name, factor in [('lower',1/1.05),('nominal',1.0),('upper',1.05)]:
                current = obs*factor
                vt = mapping.voltage_for_current(current, gate, side='set')
                vh = port-vt
                residual = abs(mapping.current(vt, gate, 'set')/current-1)
                max_inverse_residual = max(max_inverse_residual, residual)
                conditions[name] = dict(current_a=current, transistor_voltage_v=vt,
                    material_voltage_v=vh, relative_inverse_residual=residual)
            assert abs(conditions['nominal']['material_voltage_v']-old['nominal_material_abs_voltage_v'])<1e-12
            points.append(dict(position=old['position'], raw_index=old['raw_index'],
                port_voltage_v=port, observed_current_a=obs, current_and_deembedding=conditions,
                achievable_material_voltage_interval_v=[conditions['upper']['material_voltage_v'],conditions['lower']['material_voltage_v']]))
        early, late = points
        # This corner simultaneously minimizes the current ratio and maximizes
        # the proposed bound, because Vh(I)=Vport-VT(I) is nonincreasing.
        scenarios = {}
        for name, e, l in [('nominal','nominal','nominal'),('most_favorable_5percent','lower','upper')]:
            ec = early['current_and_deembedding'][e]
            lc = late['current_and_deembedding'][l]
            ratio = ec['current_a']/lc['current_a']
            bound = proposed_bound(ec['material_voltage_v'],lc['material_voltage_v'])
            scenarios[name] = dict(earlier_current_choice=e,later_current_choice=l,
                earlier_current_a=ec['current_a'], later_current_a=lc['current_a'],
                earlier_material_voltage_v=ec['material_voltage_v'], later_material_voltage_v=lc['material_voltage_v'],
                current_ratio=ratio, proposed_bound=bound, ratio_over_bound=ratio/bound,
                logarithmic_excess_decade=math.log10(ratio/bound),
                conditional_violation=ratio>bound)
        # A bounded numerical cross-check of the corner monotonicity; no new points.
        best = (math.inf,None,None)
        for ie in range(31):
            fe=1/1.05+(1.05-1/1.05)*ie/30
            ei=early['observed_current_a']*fe
            ev=early['port_voltage_v']-mapping.voltage_for_current(ei,gate,'set')
            for il in range(31):
                fl=1/1.05+(1.05-1/1.05)*il/30
                li=late['observed_current_a']*fl
                lv=late['port_voltage_v']-mapping.voltage_for_current(li,gate,'set')
                excess=(ei/li)/proposed_bound(ev,lv)
                if excess<best[0]:best=(excess,fe,fl)
        assert abs(best[0]/scenarios['most_favorable_5percent']['ratio_over_bound']-1)<1e-12
        results.append(dict(gate_v=gate,phase='set',record_id=record,points=points,scenarios=scenarios,
            corner_grid_crosscheck=dict(grid_shape=[31,31],minimum_ratio_over_bound=best[0],earlier_factor=best[1],later_factor=best[2])))
    report=dict(status='numeric_evaluation_of_proposed_bound_pending_independent_theorem_review',
        proposed_bound='exp((Vhi-Vlo)/kBT)*(1-exp(-Vhi/kBT))/(1-exp(-Vlo/kBT))',
        temperature_k=TEMPERATURE_K,kbt_ev=KT,current_budget='[Iobserved/1.05, Iobserved*1.05], matching archived multiplicative 5% sensitivity',
        deembedding='Vh=Vport-measured_transistor_mapping().voltage_for_current(I,VG,set), recomputed for every perturbed current.',
        favorable_corner_reason='Lower early current increases Vh_hi and lowers Iearly/Ilate; higher late current decreases Vh_lo and lowers Iearly/Ilate. The proposed bound increases with Vhi and decreases with Vlo for 0<Vlo<Vhi.',
        state_assumption='Application across changing states additionally requires the later state to give at least the earlier-state current at the same lower voltage; monotone zero-bias conductance alone is insufficient.',
        limitations=['The proposed general Landauer inequality is not proved by this numerical script; independent derivation pending.',
            'No assertion of a measured 5% noise confidence interval is made.',
            'The transistor mapping is taken as fixed; device-to-reference variation is not absorbed into this current budget.',
            'Native port CSV observations independently checked against saved witnesses; original XLSX cloud placeholders were not reopened.',
            'These are two preselected witnesses, not a new broad scan or universal exclusion of unified models.'],
        provenance=dict(witness_json_sha256=digest(source),native_csv_sha256=digest(native),
            transistor_py_sha256=digest(ROOT/'vulcan2d/transistor.py'),
            positive_lookup_sha256=digest(ROOT/'vulcan2d/data/fig2b_transistor_lookup.json'),
            script_sha256=digest(Path(__file__))),
        maximum_relative_forward_check_residual=max_inverse_residual,results=results)
    (OUT/'finite_temperature_witness_values.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# 两个既有 SET 见证的有限温度候选界：数值输入核对','',
        '**这是对待独立证明公式的条件数值检查，不构成公式证明。** 未拟合、未谱反演、未扩展点搜索，未读 1.1 V 留出观测。','',
        '温度 300 K；kBT = '+str(KT)+' eV。5% 沿用此前乘法区间 `[I/1.05, I×1.05]`；每个电流端点均重新去嵌入。','',
        '| Gate/编号 | raw_index | 端口 V/V | 观测 I/A | 名义 Vh/V | 5% 可达 Vh 范围/V |','|---|---:|---:|---:|---:|---|']
    for result in results:
        for p in result['points']:
            lo,hi=p['achievable_material_voltage_interval_v']
            lines.append(f'| {result["gate_v"]}/{result["record_id"]} | {p["raw_index"]} | {p["port_voltage_v"]:.8g} | {p["observed_current_a"]:.9g} | {p["current_and_deembedding"]["nominal"]["material_voltage_v"]:.12g} | [{lo:.12g}, {hi:.12g}] |')
    lines+=['','| Gate/编号 | 情景 | 早/晚电流比 | 候选界 | 比值/候选界 | 条件违例 |','|---|---|---:|---:|---:|---|']
    for result in results:
        for name,r in result['scenarios'].items():
            lines.append(f'| {result["gate_v"]}/{result["record_id"]} | {name} | {r["current_ratio"]:.10g} | {r["proposed_bound"]:.10g} | {r["ratio_over_bound"]:.10g} | {r["conditional_violation"]} |')
    lines+=['','最有利角点为早点电流 I/1.05、晚点电流 I×1.05：它同时压低观测比值、抬高早点 Vh、降低晚点 Vh。候选界随早点 Vh 增大而增大、随晚点 Vh 增大而减小，因此该角点是全区间最有利组合。31×31 同一对点的电流区间检查与角点一致。','',
        '跨状态使用还需要明确：在相同的较低偏压，晚状态电流不低于早状态电流。仅声称零偏电导增大并不够。','',
        '0.9 V record12 在该条件计算下即使采用最有利 5% 组合仍超过候选上界；1.3 V record75 不违反该宽泛候选界，尽管它违反此前固定谱形的方向约束。','',
        '原 CSV 电流/端口电压与已有见证独立逐项核对一致；正向晶体管查表逆解重新二分，并做回代检查。未重新读取原 XLSX。电流预算不代表仪器置信区间，也不覆盖晶体管跨器件差异。']
    (OUT/'finite_temperature_witness_values.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'maximum_relative_forward_check_residual':max_inverse_residual,'results':results},indent=2))

if __name__=='__main__':main()
