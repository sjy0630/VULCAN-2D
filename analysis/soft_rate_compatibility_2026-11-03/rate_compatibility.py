"""Finite necessary rate bounds; no kinetic integration or parameter fitting."""
from pathlib import Path
import hashlib
import json
import math
import signal
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OLD = ROOT / 'analysis/soft_drive_structure_2026-11-03'
CACHE = Path('/private/tmp/vulcan-runtime-cache')
sys.path[:0] = [str(ROOT), str(ROOT / 'analysis/mechanism_selection_2026-11-03')]
from SOFT_FEEDBACK_LOW_BIAS_BUDGET import cached_transport


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    result_path = HERE / 'rate_compatibility.json'
    if result_path.exists():
        raise FileExistsError('Preserve completed or failed arithmetic result; no automatic rerun')
    old = json.loads((OLD / 'sinh_field_probe.json').read_text())
    prior_inverse = json.loads((OLD / 'DRIVE_STRUCTURE_INVERSION.json').read_text())
    p = old['plan']
    ron, gc, beta, js = (p[k] for k in ('Ron_ohm', 'gamma_c_ev', 'beta', 'JS_A'))
    target = prior_inverse['pair_tradeoffs'][1]
    initial = next(x for x in prior_inverse['observations'] if x['branch'] == 'outward' and x['command_v'] == .2)
    plan = {
        'scope': 'Conditional seriescycle4 early arrival and subsequentout2->5->ret4 retention, not32cycle aggregate contract or fullsharedmodel',
        'state_law': 'du/ds=-kappa*M(absVh)*positive(log(I/JS)+beta*u)',
        'Ron_ohm': ron, 'gamma_c_ev': gc, 'beta': beta, 'JS_A': js,
        'initial_command_v': .2, 'initial_current_a': initial['current_a'],
        'outward2_current_lower_a': target['earlier_target_a'],
        'return4_current_upper_a': target['later_target_a'],
        'mobility_cases': ['constant1', 'linearVh', 'sinhVf0p5'],
        'new_mobility_scales': 0, 'new_parameter_fits': 0, 'new_ODE': 0,
        'numerical_material_relative_envelopes': [0., .003],
        'envelope_scope': 'Conditional pointwise bounds(1-e)Ftable<=Ftrue<=(1+e)Ftable; inherited sensitivity, not a newly proven global certificate',
        'grids': [{'name': 'nominal', 'early_state_cells': 64, 'late_command_step_v': .05},
                  {'name': 'refined', 'early_state_cells': 128, 'late_command_step_v': .025}],
        'max_anchor_inversions': 6,
        'early_upper': 'For eachu cell[x,y],sup rate/k<=M(Vh_upper(A2,y))*positive(logI_upper(A2,x)/JS+beta*y); k>=sum(du/upper)/1.8',
        'late_lower': 'u in[L,H]. EachleftAcell lower=M(Vh_lower(A,L))*positive(logI_lower(A,H)/JS+beta*L); sumout2->5 andret4->5; k<= (H-L)/sum',
        'envelope_circuit': 'I_upper=series(Ftable*(1+e)),Vh_lower fromsame; I_lower=series(Ftable*(1-e)),Vh_upper fromsame. Combine separate corners only for conservativebounds.',
        'universal_nondecreasing_M': 'DefineK=kappa*M(2V). EarlyVh<=A<=2 givesK>=constant_mobility_earlylower. Latecells withVh_lower(A,L)>=2 giveM>=M(2); omitallothercells. K<=(H-L)/sumselectedpositive_drive*deltaA.',
        'universal_edge_case': 'IfM(2)=0, allnonnegative nondecreasingM vanish before2and no positive state decrease is possible. Assumes finiteM, unique admissible trajectories; not state/history-dependentM.',
        'numerical_checks': ['same passivetransistor map', 'no outoftable evaluation', 'KCLresidual<=1e-12A', 'grid refinement compares bounds, not kinetic trajectories'],
        'compute_budget_s': 60, 'budget_start': 'Afterinputs/sourcehashes andplanwrite; beforeanchorinversion orbudgetarithmetic',
        'stop_policy': 'Preserve allfailures; no adaptiveparameter expansion, newcandidate, interpolation modification, orholdout access'
    }
    plan_path = HERE / 'rate_compatibility.plan.json'
    plan_path.write_text(json.dumps(plan, indent=2) + '\n')
    t, lookups = cached_transport(CACHE)
    assert sha(t.table_path) == '00d6d9ea0b1dfe855181e7016afc6dd5e23d169f83f19cfcedfc9d3b35f9e4e2'
    inputs = [OLD / 'sinh_field_probe.json', OLD / 'DRIVE_STRUCTURE_INVERSION.json',
              t.table_path, *lookups, plan_path, Path(__file__),
              ROOT / 'analysis/mechanism_selection_2026-11-03/SOFT_FEEDBACK_LOW_BIAS_BUDGET.py',
              ROOT / 'analysis/ron_scope_2026-11-03/ron_scaling_review.py']
    hashes = {str(q): sha(q) for q in inputs}
    r = {'status': 'running', 'plan': plan, 'source_sha256': hashes, 'cases': [], 'anchor_inverse_count': 0}
    arrays = {}
    started = time.monotonic()
    def timeout(*_):
        raise TimeoutError('Fixed60second arithmetic budget exhausted')
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 60)
    try:
        for eta in plan['numerical_material_relative_envelopes']:
            endpoints = []
            # Most optimistic initialstate and requireddecrease, widest lateband.
            for name, A, I, factor in [
                ('initial_u_lower', .2, plan['initial_current_a'], 1-eta),
                ('outward2_u_upper', 2., plan['outward2_current_lower_a'], 1+eta),
                ('return4_u_lower', 4., plan['return4_current_upper_a'], 1-eta)]:
                vt = float(t.mapping.voltage_for_current(I, 1.1, 'set'))
                v = A-vt
                gamma = t.inverse(v, I/factor, ron, 'set')
                u = math.log1p(gc/gamma)
                r['anchor_inverse_count'] += 1
                endpoints.append({'name': name, 'command_v': A, 'target_current_a': I, 'material_voltage_v': v,
                                  'material_amplitude_factor': factor, 'gamma_ev': float(gamma), 'u': u})
            u0, H, L = (x['u'] for x in endpoints)
            assert u0 > H >= L > 0
            for grid in plan['grids']:
                prefix = f'e{eta:g}_{grid["name"]}'
                state = np.linspace(H, u0, grid['early_state_cells']+1)
                early_g = gc/np.expm1(state)
                Iplus, Vminus = t.series(np.full_like(state, 2.), early_g, ron/(1+eta))
                Iminus, Vplus = t.series(np.full_like(state, 2.), early_g, ron/(1-eta))
                assert np.all(np.diff(Iplus) < 0) and np.all(np.diff(Vplus) > 0)
                upper_D = np.maximum(np.log(Iplus[:-1]/js) + beta*state[1:], 0.)
                dA = grid['late_command_step_v']
                aout = np.linspace(2., 5., round(3/dA)+1)[:-1]
                aret = np.linspace(4., 5., round(1/dA)+1)[:-1]
                a = np.concatenate([aout, aret])
                # true I>=I_minus(A,H); true Vh>=V_minus(A,L).
                Ilow, _ = t.series(a, np.full_like(a, gc/math.expm1(H)), ron/(1-eta))
                Ihi_at_L, Vlow = t.series(a, np.full_like(a, gc/math.expm1(L)), ron/(1+eta))
                lower_D = np.maximum(np.log(Ilow/js) + beta*L, 0.)
                tx, ti = t.knots['set']
                kcl = max(np.max(abs(Iplus-np.interp(2-Vminus,tx,ti))),
                          np.max(abs(Iminus-np.interp(2-Vplus,tx,ti))),
                          np.max(abs(Ihi_at_L-np.interp(a-Vlow,tx,ti))))
                assert kcl <= 1e-12
                record = {'envelope': eta, 'grid': grid, 'anchor_bounds': endpoints,
                          'late_state_interval': [L,H], 'maximum_late_state_loss': H-L,
                          'maximum_budget_KCL_residual_a': float(kcl), 'mobility_bounds': []}
                arrays.update({prefix+'_early_state': state, prefix+'_early_upper_drive': upper_D,
                               prefix+'_early_local_voltage_upper': Vplus, prefix+'_early_current_upper': Iplus,
                               prefix+'_late_command_left': a, prefix+'_late_current_lower': Ilow,
                               prefix+'_late_local_voltage_lower': Vlow, prefix+'_late_drive_lower': lower_D})
                for name, mobility, unit in [
                    ('constant1', lambda x: np.ones_like(x), 'V^-1'),
                    ('linearVh', lambda x: x, 'V^-2'),
                    ('sinhVf0p5', lambda x: .5*np.sinh(x/.5), 'V^-2')]:
                    ceiling = mobility(Vplus[1:])*upper_D
                    assert np.all(ceiling > 0)
                    lower = float(np.sum(np.diff(state)/ceiling)/1.8)
                    floor = mobility(Vlow)*lower_D
                    integral = float(dA*np.sum(floor))
                    upper = (H-L)/integral if integral>0 else None
                    record['mobility_bounds'].append({'name':name,'rate_unit':unit,'early_rate_lower':lower,
                        'late_rate_upper':upper,'late_integral_lower':integral,
                        'rate_intervals_disjoint':upper is not None and lower>upper,
                        'lower_over_upper':lower/upper if upper else None})
                    arrays[prefix+'_'+name+'_late_integrand_lower'] = floor
                chosen = Vlow >= 2.
                drive_integral = float(dA*np.sum(lower_D[chosen]))
                Klo = record['mobility_bounds'][0]['early_rate_lower']
                Khi = (H-L)/drive_integral if drive_integral>0 else None
                record['all_nondecreasing_voltage_only_mobility'] = {
                    'scaled_rate': 'K=kappa*M(2V)', 'unit': 'V^-1',
                    'early_scaled_rate_lower': Klo, 'late_scaled_rate_upper': Khi,
                    'selected_left_cells': int(chosen.sum()), 'total_left_cells': len(a),
                    'selected_command_ranges_outward_and_return':
                        [[float(x[m].min()),float(x[m].max())] if m.any() else None for x,m in
                         [(aout,chosen[:len(aout)]),(aret,chosen[len(aout):])]],
                    'selected_drive_integral_lower_v': drive_integral,
                    'rate_intervals_disjoint': Khi is not None and Klo>Khi,
                    'lower_over_upper': Klo/Khi if Khi else None}
                arrays[prefix+'_universal_selected_cells'] = chosen
                r['cases'].append(record)
        assert r['anchor_inverse_count'] == 6
        r['status'] = 'completed_bounded_early_late_rate_screen_no_ODE_no_fit'
    except Exception as exc:
        r.update(status='failed_or_partial_preserved',error_type=type(exc).__name__,error=str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        r['compute_elapsed_s'] = time.monotonic()-started
    result_path.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(result_path.with_suffix('.npz'),**arrays)
    lines=['# 早期到达与后续返扫的速率兼容性','',
           '固定β6、JS、Ron、谱和晶体管映射；仅检验既有单圈条件锚，不是32圈聚合否定。',
           '三种已试迁移率，无新尺度、ODE或拟合；0.3%材料包络是条件敏感性假设，不是新测量误差。','',
           f'状态：{r["status"]}；{r["anchor_inverse_count"]}次锚反演，计算{r["compute_elapsed_s"]:.6f}s。','',
           '| 材料包络 | 网格 | M | 早期κ下界 | 后期κ上界 | 下/上比 | 不相交 |',
           '|---|---|---|---:|---:|---:|---|']
    for case in r['cases']:
        for b in case['mobility_bounds']:
            lines.append(f'| {case["envelope"]} | {case["grid"]["name"]} | {b["name"]} | {b["early_rate_lower"]:.9g} | {b["late_rate_upper"]} | {b["lower_over_upper"]} | {b["rate_intervals_disjoint"]} |')
    lines += ['', '## 任意非负、非递减的电压迁移率','',
              '早期Vh≤A≤2V，故M(Vh)≤M(2V)。后期仅保留Vh的保守下界≥2V的左节点格，故M(Vh)≥M(2V)。对K=κM(2V)比较同一必要上下界；M(2V)=0时早期不能发生所需变化。',
              '', '| 材料包络 | 网格 | K下界 | K上界 | 下/上比 | 后期选中格 |',
              '|---|---|---:|---:|---:|---:|']
    for c in r['cases']:
        b=c['all_nondecreasing_voltage_only_mobility']
        lines.append(f'| {c["envelope"]} | {c["grid"]["name"]} | {b["early_scaled_rate_lower"]:.9g} | {b["late_scaled_rate_upper"]} | {b["lower_over_upper"]} | {b["selected_left_cells"]}/{b["total_left_cells"]} |')
    lines += ['', '这些必要界使用不同端点的保守乘积，不把点采样最小值当连续全局下界。外扫4→5和返扫5→4都计入路径。一般化不覆盖随状态/历史变化的迁移率、非单调场响应、修改反馈停止面或新的读出/电路假设。',
              '', '不相交只说明这些冻结条件下不存在同时满足所用到达/返扫条件的速率；不抹去之前0.182dec条件RMSE改善，也不宣称整个材料理论或原聚合合同无解。']
    if 'error' in r: lines += ['',r['error_type']+': '+r['error']]
    result_path.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':r['status'],'compute_elapsed_s':r['compute_elapsed_s'],
        'cases':[{'envelope':c['envelope'],'grid':c['grid']['name'],'bounds':c['mobility_bounds'],
                  'universal':c['all_nondecreasing_voltage_only_mobility']} for c in r['cases']]},indent=2))


if __name__=='__main__':
    main()
