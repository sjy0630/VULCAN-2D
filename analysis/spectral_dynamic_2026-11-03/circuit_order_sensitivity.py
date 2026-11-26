"""Training-only external-circuit sensitivity of the scalar SET order bound.

No material parameter changes, ODE fitting, test arrays, data correction, or
desktop-default changes. One positive scale multiplies both signed T branches.
"""
from pathlib import Path
from dataclasses import replace
import hashlib
import json
import math
import sys

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from vulcan2d.shared_spectral import SpectralShape
from vulcan2d.transistor import measured_transistor_mapping

DATA = ROOT/'analysis/shared_hbn_2026-11-03/data/native_observations.npz'
ORDER = OUT/'finite_rate_order_audit.json'
SHAPE = ROOT/'analysis/constrained_shared_2026-11-03/lognormal_spectral_refinement.json'
TRAIN = {'1r': [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
         '1t1r': list(range(4, 36))}


def main():
    shape = SpectralShape.from_json(SHAPE)
    reference = measured_transistor_mapping()
    prior = json.loads(ORDER.read_text())['same_state_general_comparison']
    loaded = hashlib.sha256()
    raw = {}
    with np.load(DATA) as archive:
        for topology, ids in TRAIN.items():
            for phase in (('set',) if topology == '1r' else ('set', 'reset')):
                rows = []
                for cid in ids:
                    prefix = f'{topology}_cycle_{cid}_{phase}'
                    v = archive[prefix+'_voltage'].copy()
                    i = archive[prefix+'_current'].copy()
                    for suffix, values in [('voltage', v), ('stored_current', i)]:
                        loaded.update((prefix+suffix).encode())
                        loaded.update(values.tobytes())
                    rows.append((cid, v, abs(i)))
                raw[topology, phase] = rows
    ranges = {}
    for command in (5., 4.):
        currents = []
        for _, v, i in raw['1t1r', 'set']:
            index = np.arange(len(v))
            j = int(np.argmin(np.where(index > np.argmax(v), abs(v-command), np.inf)))
            assert abs(v[j]-command) < 1e-10
            currents.append(i[j])
        q = np.quantile(currents, [.1, .9])
        ranges[command] = {'lower': float(q[0]), 'upper': float(q[1])}
        for side, quantile in [('lower', .1), ('upper', .9)]:
            old = next(x['current_a'] for x in prior['soft_endpoint_sources']
                       if x['command_v'] == command and x['quantile'] == quantile)
            assert abs(old-ranges[command][side]) < 1e-16

    # Check every training port on both signed branches, with exactly one c.
    passive = {}
    for phase in ('set', 'reset'):
        demand, fixed_readout_demand = [], []
        for cid, v, i in raw['1t1r', phase]:
            for command, current in zip(abs(v), i):
                if command == 0.:
                    continue  # noisy zero-command rows are not DC passivity constraints
                ref = reference.current(float(command), 1.1, phase)
                demand.append((float(current/ref), cid, float(command), float(current)))
                available = float(command-current*shape.ron_ohm)
                floor = (float(current/reference.current(available, 1.1, phase))
                         if available > 0 else math.inf)
                fixed_readout_demand.append((floor, cid, float(command), float(current)))
        greatest = max(demand)
        fixed = max(fixed_readout_demand)
        passive[phase] = {
            'minimum_scale_for_passivity': greatest[0],
            'binding_record_command_current': list(greatest[1:]),
            'minimum_scale_with_finite_Ron_limit': fixed[0],
            'finite_Ron_binding_record_command_current': list(fixed[1:]),
            'scope': 'Necessary port-voltage budget only, not a material/dynamic fit.'}
    passive_floor = max(x['minimum_scale_for_passivity'] for x in passive.values())
    ron_floor = max(x['minimum_scale_with_finite_Ron_limit'] for x in passive.values())
    print(json.dumps({'stage': 'training_passivity', 'by_phase': passive}), flush=True)

    bare_brackets = prior['bare_training_crossing_brackets']
    cache = {}

    def inverse(command, current, mapping, order=32):
        vh = command-mapping.voltage_for_current(current, 1.1, 'set')
        if vh <= current*shape.ron_ohm:
            raise ValueError('Endpoint exceeds the fixed-readout Ohmic maximum')
        lg = brentq(lambda lg: math.log(shape.current_at_log_gamma(vh, lg, order=order)/current),
                    -32., 15., xtol=1e-11)
        return math.exp(lg), vh

    def forward(command, gamma, mapping, order=32):
        root = brentq(lambda v: shape.current_at_gamma(v, gamma, order=order)
                      -mapping.current(command-v, 1.1, 'set'),
                      0., command, xtol=1e-12)
        return float(shape.current_at_gamma(root, gamma, order=order)), float(root)

    def case(p, scale, *, verify=False):
        key = p, float(scale), verify
        if key in cache:
            return cache[key]
        mapping = replace(reference, current_scale=float(scale))
        result = {'expansion_fraction': p, 'common_signed_current_scale': float(scale),
                  'all_training_signed_ports_passive': bool(scale >= passive_floor),
                  'all_training_ports_pass_fixed_Ron_budget': bool(scale >= ron_floor)}
        points = {}
        try:
            for command in (5., 4.):
                for side in ('lower', 'upper'):
                    current = ranges[command][side]*(1+p if side == 'upper' else 1/(1+p))
                    gamma, local = inverse(command, current, mapping)
                    points[command, side] = {'command_v': command, 'side': side,
                                            'current_a': current, 'gamma_ev': gamma,
                                            'local_v': local}
        except ValueError as error:
            result.update(status='endpoint_infeasible', reason=str(error))
            cache[key] = result
            return result
        lo = min(x['gamma_ev'] for x in points.values())
        hi = max(x['gamma_ev'] for x in points.values())
        viable_lo = points[5., 'lower']['gamma_ev']
        viable_hi = points[4., 'upper']['gamma_ev']
        endpoint_feasible = viable_lo <= viable_hi
        _, hull_vmin = forward(4., hi, mapping)
        _, tight_vmin = forward(4., viable_hi, mapping)
        result.update(status='evaluated', endpoint_bands=list(points.values()),
                      monotone_endpoint_feasible=bool(endpoint_feasible),
                      original_hull_gamma_band_ev=[lo, hi], hull_voltage_min_v=hull_vmin,
                      monotone_viable_gamma_band_ev=[viable_lo, viable_hi] if endpoint_feasible else None,
                      tight_voltage_min_v=tight_vmin,
                      tight_direct_voltage_v=points[4., 'upper']['local_v'],
                      tight_KCL_minus_direct_v=tight_vmin-points[4., 'upper']['local_v'])
        # Recheck every saved bracket under the new state bands; do not assume
        # a bracket remains valid when the deembedded state interval changes.
        brackets = []
        for record in bare_brackets:
            cid = record['cycle']
            v0, v1 = record['last_below_command_v'], record['first_above_command_v']
            i0, i1 = record['last_below_current_a'], record['first_above_current_a']
            row = {'cycle': cid, 'before_v': v0, 'after_v': v1,
                   'width_v': v1-v0, 'before_current_a': i0, 'after_current_a': i1}
            for label, lower, upper, vmin in [('hull', lo, hi, hull_vmin),
                                            ('tight', viable_lo, viable_hi, tight_vmin)]:
                pre = float(shape.current_at_gamma(v0, lower))
                post = float(shape.current_at_gamma(v1, upper))
                bracket_valid = bool(i0 < pre and i1 >= post and i0 < .995e-3)
                order_margin = float(vmin-v1)
                applies = bool(endpoint_feasible and bracket_valid and order_margin > 1e-8)
                row[label] = {'bracket_reconfirmed': bracket_valid,
                              'before_classification_margin_ratio': pre/i0,
                              'after_classification_margin_ratio': i1/post,
                              'voltage_order_margin_v': order_margin,
                              'comparison_applies': applies,
                              'alpha_lower': 1/(v1-v0) if applies else None}
            brackets.append(row)
        result['bare_brackets'] = brackets
        for label in ('hull', 'tight'):
            bounds = [r[label]['alpha_lower'] for r in brackets if r[label]['comparison_applies']]
            result[label+'_applicable_count'] = len(bounds)
            result[label+'_alpha_lower'] = max(bounds) if bounds else None
        if verify:
            fine = []
            for point in points.values():
                g64, _ = inverse(point['command_v'], point['current_a'], mapping, order=64)
                fine.append(abs(math.log10(g64/point['gamma_ev'])))
            _, vh64 = forward(4., hi, mapping, order=64)
            result['numerical_check'] = {
                'max_endpoint_gamma_order32_to64_decade': max(fine),
                'hull_KCL_order32_to64_change_v': abs(vh64-hull_vmin),
                'tight_KCL_vs_analytic_inverse_abs_v': abs(result['tight_KCL_minus_direct_v'])}
        cache[key] = result
        return result

    scales = [1., .98, .96, .94, .92, .90, .88, .86, .84, .82, .80,
              .7890279332985641, .75, .70, .65, .60, .55, .54,
              max(.53, ron_floor)*(1+1e-6)]
    rows = []
    for p in (.01, .05):
        for scale in scales:
            result = case(p, scale, verify=scale in (1., .90, .80))
            rows.append(result)
            print(json.dumps({'stage': 'case', **{k: result.get(k) for k in
                ('expansion_fraction', 'common_signed_current_scale', 'status',
                 'monotone_endpoint_feasible', 'hull_voltage_min_v', 'tight_voltage_min_v',
                 'hull_applicable_count', 'tight_applicable_count')}}), flush=True)

    thresholds = []
    for p in (.01, .05):
        current4 = ranges[4.]['upper']*(1+p)
        for target in sorted({r['first_above_command_v'] for r in bare_brackets}, reverse=True):
            # Exact endpoint/tight-band threshold from V_T,ref(I/c)>=4-Vbare.
            exact = current4/reference.current(4-target, 1.1, 'set')
            # Original, deliberately wider interval-hull threshold for comparison.
            valid = sorted([(r['common_signed_current_scale'], r['hull_voltage_min_v']-target)
                            for r in rows if r['expansion_fraction'] == p and r['status'] == 'evaluated'])
            hull_root = None
            for (left, y0), (right, y1) in zip(valid[:-1], valid[1:]):
                if y0*y1 <= 0:
                    hull_root = float(brentq(lambda c: case(p, c)['hull_voltage_min_v']-target,
                                            left, right, xtol=2e-10))
                    break
            check = case(p, exact, verify=True)
            thresholds.append({'expansion_fraction': p, 'bare_crossing_ceiling_v': target,
                               'affected_training_records': [r['cycle'] for r in bare_brackets
                                                             if r['first_above_command_v'] == target],
                               'tight_direct_scale_threshold': exact,
                               'tight_scale_reduction_fraction': 1-exact,
                               'hull_scale_threshold': hull_root,
                               'tight_threshold_case': check,
                               'hull_threshold_case': case(p, hull_root, verify=True) if hull_root else None})

    # Endpoint feasibility need not vary monotonically with c: the measured T
    # curve is piecewise nonlinear. Resolve sign changes rather than declaring
    # all scales below an ordering threshold to be viable.
    endpoint_feasibility = []
    for p in (.01, .05):
        def gap(scale):
            mapping = replace(reference, current_scale=float(scale))
            lower, _ = inverse(5., ranges[5.]['lower']/(1+p), mapping)
            upper, _ = inverse(4., ranges[4.]['upper']*(1+p), mapping)
            return math.log(lower/upper)
        grid = np.linspace(ron_floor*(1+1e-6), 1., 501)
        values = np.array([gap(x) for x in grid])
        roots = []
        for left, right, y0, y1 in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
            if y0*y1 < 0:
                roots.append(float(brentq(gap, left, right, xtol=2e-11)))
        boundaries = [float(grid[0]), *roots, 1.]
        intervals = [{'scale_interval': [a, b], 'midpoint_monotone_feasible': bool(gap((a+b)/2) <= 0)}
                     for a, b in zip(boundaries[:-1], boundaries[1:])]
        # Closest-from-one interval with feasible endpoints and no remaining
        # endpoint-based uniform voltage ordering, verified just inside it.
        all_loss = min(x['tight_direct_scale_threshold'] for x in thresholds if x['expansion_fraction'] == p)
        candidate_ceiling = max(min(x['scale_interval'][1], all_loss) for x in intervals
                                if x['midpoint_monotone_feasible'] and x['scale_interval'][0] < all_loss)
        check = case(p, candidate_ceiling-1e-7, verify=True)
        endpoint_feasibility.append({'expansion_fraction': p, 'scale_grid_points': len(grid),
                                     'located_gap_roots': roots, 'sampled_intervals': intervals,
                                     'max_sampled_log_gamma5min_over_gamma4max': float(max(values)),
                                     'closest_nonvacuous_all_order_loss_scale_boundary': candidate_ceiling,
                                     'check_just_below_boundary': check,
                                     'scope': '501-point scan with all observed sign changes refined; not a theorem excluding unsampled additional roots.'})

    paths = [Path(__file__), ORDER, SHAPE, ROOT/'vulcan2d/shared_spectral.py',
             ROOT/'vulcan2d/transistor.py', ROOT/'vulcan2d/data/fig2b_transistor_lookup.json',
             ROOT/'vulcan2d/data/bookoutput2_reverse_transistor_lookup.json']
    report = {
        'status': 'completed_training_only_circuit_sensitivity_no_fit',
        'material_parameters_changed': False,
        'data_or_desktop_defaults_changed': False,
        'scaling_convention': 'One c multiplies both positive and negative reference T currents; all gate mappings fixed.',
        'current_expansion': '[Q10/(1+p), Q90*(1+p)], p=1% or5%, sensitivity scenarios not confidence intervals.',
        'training_ids': TRAIN, 'training_arrays_sha256': loaded.hexdigest(),
        'unexpanded_endpoint_current_ranges_a': ranges,
        'passivity': passive, 'joint_passivity_floor': passive_floor,
        'joint_fixed_Ron_budget_floor': ron_floor,
        'positive_gate_samples_v': reference.lookup.gate_voltages.tolist(),
        'positive_has_direct_1p1V_gate': bool(np.any(np.isclose(reference.lookup.gate_voltages, 1.1))),
        'reverse_has_direct_1p1V_gate': bool(np.any(np.isclose(reference.reset_lookup.gate_voltages, 1.1))),
        'derivation': [
            'I_T(V_T)=c I_T,ref(V_T), hence V_h=U-V_T,ref(I/c).',
            'A 4V endpoint can have V_h<=Vbare only if c<=I4_max/I_T,ref(4-Vbare).',
            'For a monotone SET trajectory satisfying both endpoints, gamma(t) lies in [gamma5,min,gamma4,max]; nonempty only if gamma5,min<=gamma4,max.',
            'The smaller viable band implies V_h,min=KCL(4V,gamma4,max)=4V-V_T,ref(I4_max/c), independently checked against exact spectral KCL.',
            'The earlier full endpoint hull is wider and loses its sufficient voltage bound earlier; its failure alone does not erase the tighter endpoint-conditioned comparison.',
            'Loss of this voltage-order sufficient condition removes this particular proof, not every possible same-state integral bound or all model discrepancies.'
        ],
        'sensitivity_cases': rows, 'voltage_order_scale_thresholds': thresholds,
        'endpoint_feasibility_scan': endpoint_feasibility,
        'scope': [
            'No scale is estimated, calibrated, or assigned probability from these data.',
            'Positive Fig2b has no directly measured VG=1.1V curve; reverse data do include1.1V.',
            'Reference standalone versus array transistor identity and population variation are unconfirmed; positive/reverse same-cell correspondence is also unconfirmed.',
            'Changing c disagrees with the stored reference-transistor currents by1-c at every nonzero point; no measured cross-cell distribution licenses or rejects that change.',
            'Common signed passivity is only necessary; it neither validates RESET dynamics nor proves a complete joint trajectory.',
            'Bare before bracket inference uses Vh=Vcommand below nominal1mA compliance; after inference uses Vh<=Vcommand. No unrecorded overshoot is allowed in this DC comparison.',
            'No timestamps or sweep-rate differences are invented; A_RESET still does not alter SET.'
        ],
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (OUT/'circuit_order_sensitivity.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    lines = ['# 外电路映射对同态电压排序的影响', '',
             '固定 hBN 谱形、同一正负晶体管电流比例 c，仅做训练数据敏感性；没有拟合或更改默认模型。', '',
             '## 先区分两种边界', '',
             '旧计算取四个端点状态的完整区间包络，给出保守的电压下界。若同一条单调 SET 轨迹确实满足两端区间，可收紧为 `[γ5,min,γ4,max]`。后者的最低局部电压就是 4V 端口上界电流的去嵌入电压。', '',
             '`Vh,min = 4 − VT,ref(I4,max/c)`；故排序对某个裸记录失效的边界为 `c = I4,max / IT,ref(4−Vbare,max)`。数值谱反演与自主 KCL 再次核对这个直接公式。', '',
             '| 端口扩张 | 裸穿带上限/V | 旧包络失效 c | 更紧边界 c | 相对参考电流减小 | 端点单调可行 |',
             '|---:|---:|---:|---:|---:|---|']
    for r in thresholds:
        lines.append(f'| {100*r["expansion_fraction"]:.0f}% | {r["bare_crossing_ceiling_v"]:.4g} | '
                     f'{r["hull_scale_threshold"]:.6g} | {r["tight_direct_scale_threshold"]:.6g} | '
                     f'{100*r["tight_scale_reduction_fraction"]:.3g}% | '
                     f'{r["tight_threshold_case"].get("monotone_endpoint_feasible")} |')
    lines += ['', '下降穿过旧包络边界，仅表示旧的较松证明不再适用；更紧的单调端点界可能仍支持约20倍时钟条件。下降穿过更紧边界后，这一全段电压排序前提才不再由端点保证。两者都不等于模型已拟合成功。', '',
              '## 端点可行性不能漏检', '']
    for r in endpoint_feasibility:
        intervals = '; '.join(f'[{x["scale_interval"][0]:.8g},{x["scale_interval"][1]:.8g}]: '
                              + ('可行' if x['midpoint_monotone_feasible'] else '不可行')
                              for x in r['sampled_intervals'])
        lines += [f'{100*r["expansion_fraction"]:.0f}% 情景：{intervals}。', '',
                  f'保持端点单调可行、同时使全部已检查记录失去此项电压排序保证的最近边界约为 c={r["closest_nonvacuous_all_order_loss_scale_boundary"]:.8g}；已在边界下方复查。', '']
    lines += ['尤其1%情景存在中间的端点不相容区间；单看约0.85–0.88的电压交叉会漏掉这个约束，不能把该处当作可行模型。5%情景在相应电压交叉处仍允许单调端点。上述区间来自501点扫描及过零精化，没有拟合材料或时钟。', '',
              '## 共同正负支路的电压预算', '',
              f'所有训练点要求 c≥{passive_floor:.8g} 才满足无源电压预算；计入固定谱形的 Ron 极限后要求 c≥{ron_floor:.8g}。这是必要条件，不能当作允许 c 的统计置信区间。', '',
              '| 情景 | c | 单调端点可行 | 旧包络最低Vh | 更紧最低Vh | 旧/紧仍适用记录数 |',
              '|---:|---:|---|---:|---:|---:|']
    for r in rows:
        if r['common_signed_current_scale'] not in (1., .90, .88, .86, .84, .80, .7890279332985641, .60):
            continue
        if r['status'] == 'evaluated':
            lines.append(f'| {100*r["expansion_fraction"]:.0f}% | {r["common_signed_current_scale"]:.6g} | '
                         f'{r["monotone_endpoint_feasible"]} | {r["hull_voltage_min_v"]:.6g} | '
                         f'{r["tight_voltage_min_v"]:.6g} | {r["hull_applicable_count"]}/{r["tight_applicable_count"]} |')
        else:
            lines.append(f'| {100*r["expansion_fraction"]:.0f}% | {r["common_signed_current_scale"]:.6g} | 端口不可行 | — | — | — |')
    lines += ['', '## 能说明什么', '',
              '正向 Fig.2b 只有 VG=0.5、1.0、1.5、2.0、2.5、3.0V，没有直接测得的1.1V曲线；负向 lookup 包含1.1V。参考晶体管与阵列晶体管是否同一个 cell、正负数据是否同一个晶体管、器件间分布多大，都尚未确认。', '',
              '因此本结果不能给“电路映射错误的概率”，也不能用调 c 后的拟合反过来证明晶体管确有这一差异。但它量化了应优先独立核实的映射变化幅度；现有数据不足以把此瓶颈无条件归因于材料动力学。', '',
              'c 对正负支路共同作用，不能只改 SET。无源预算通过不代表 RESET 已验证。原始训练数组、参考曲线、core 和已有拟合文件均未修改。', '',
              '完整逐记录状态括区、32→64积分核对、共同正负无源条件和来源哈希见同名 JSON。']
    (OUT/'circuit_order_sensitivity.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'stage': 'complete', 'joint_passivity_floor': passive_floor,
                      'joint_Ron_floor': ron_floor, 'output': str(OUT/'circuit_order_sensitivity.md')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
