"""One authorized, saved-model arithmetic self-check; no model imports/calls."""
from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent
PHASES = ('set4', 'reset4', 'set5')
PRECISIONS = ('nominal', 'refined')
SAMPLING = ('native', 'every_other_keep_endpoints')
WINDOWS = (('outward_low', 'outward', .2, .8),
           ('outward_high', 'outward', .8, 1.6),
           ('return_high', 'return', 1.6, .8),
           ('return_low', 'return', .8, .2))
RESULT = HERE / 'SELF_CHECK_RESULT.json'
SEVERE_CONDITION = 1e8


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def json_write(path, data):
    payload = (json.dumps(clean(data), indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode()
    with path.open('xb') as f:
        f.write(payload)
    return {'path': str(path), 'sha256': sha(payload), 'bytes': len(payload)}


def closure(delta, A, B, coefficients):
    first, second = coefficients[0]*A, coefficients[1]*B
    error = delta - (first-second)
    denominator = abs(first)+abs(second)+abs(delta)
    return {
        'delta_u': float(delta), 'recovery_integral': float(first),
        'capture_integral': float(second), 'predicted_delta_u': float(first-second),
        'E': float(error), 'absolute_E': float(abs(error)),
        'normalization_denominator': float(denominator),
        'denominator_zero': bool(denominator == 0.),
        'normalized_E': float(error/denominator) if denominator != 0. else None,
        'absolute_normalized_E': float(abs(error/denominator)) if denominator != 0. else None,
    }


def trap(values, coordinate):
    return float(np.sum(.5*(values[:-1]+values[1:])*np.diff(coordinate)))


def identify(rows, truth, names):
    X = np.array([[r['A_v2'], -r['B_a_v2']] for r in rows], dtype=float)
    y = np.array([r['delta_u'] for r in rows], dtype=float)
    true = np.array(truth, dtype=float)
    scaled = X*true[None, :]
    column_norms = np.linalg.norm(X, axis=0)
    singular = np.linalg.svd(scaled, compute_uv=False)
    tolerance = max(scaled.shape)*np.finfo(float).eps*singular[0]
    rank = int(np.count_nonzero(singular > tolerance))
    condition = float(singular[0]/singular[-1]) if singular[-1] > 0. else math.inf
    norm_scaled = X/column_norms[None, :] if np.all(column_norms > 0.) else None
    unit_condition = float(np.linalg.cond(norm_scaled)) if norm_scaled is not None else math.inf
    out = {
        'window_ids': [r['id'] for r in rows], 'coefficient_names': names,
        'raw_design_A_minus_B': X, 'delta_u_vector': y,
        'raw_column_l2_norms': column_norms,
        'raw_column_units': ['V^2', 'A V^2'],
        'true_parameter_column_scaling': true,
        'dimensionless_scaled_design': scaled, 'scaled_singular_values': singular,
        'rank_tolerance': float(tolerance), 'scaled_numerical_rank': rank,
        'scaled_condition_2': condition, 'unit_column_condition_2': unit_condition,
        'severe_condition_threshold': SEVERE_CONDITION,
        'severely_ill_conditioned': bool(condition > SEVERE_CONDITION),
        'known_coefficients': dict(zip(names, truth)),
        'identification_available': False,
    }
    if rank != 2 or not np.all(np.isfinite(scaled)):
        out.update(status='rank_deficient_or_nonfinite_no_unique_identification')
        return out
    try:
        relative = np.linalg.solve(scaled, y)
        estimate = true*relative
        if not np.all(np.isfinite(estimate)):
            raise ValueError('Nonfinite 2x2 identified coefficients')
        positive = bool(np.all(estimate > 0.))
        out.update(
            identification_available=True,
            status=('diagnostic_only_ill_conditioned' if condition > SEVERE_CONDITION else
                    'finite_identification_positive' if positive else 'finite_identification_nonpositive'),
            identified_coefficients=dict(zip(names, estimate)),
            coefficient_ratios_to_known=dict(zip(names, relative)),
            relative_coefficient_errors=dict(zip(names, relative-1.)),
            coefficients_all_positive=positive,
            calibration_linear_residual=X@estimate-y,
        )
    except np.linalg.LinAlgError as exc:
        out.update(status='linear_solve_failure', error=str(exc))
    return out


def make_report(result):
    def fmt(x, digits=6):
        return '未定义' if x is None else f'{x:.{digits}g}'
    lines = [
        '# 已知模型的积分离散自检', '',
        '**本结果只检验保存模型轨迹上的离散积分诊断，没有检验实验机制。**', '',
        '只使用父阶段最终 nominal/refined 的 SET4、RESET4、SET5 原生预测数组。',
        '没有反演、求根、ODE、优化或模型函数调用；未使用实验观察电流列及后续相位。', '',
        'F/G 按路径距离作分段线性梯形积分；“隔点”保留每个固定窗口的两个端点。',
        '它不是电路一致 u 插值。两个求解精度同采样的比较只反映保存求解差异，',
        '原生/隔点差异不能作为实验轨迹或测量误差证书。', '',
        '## 已知系数的积分闭合', '',
        '| 保存求解精度 | 积分采样 | 12 窗口最大绝对闭合差 | 最大归一化绝对闭合差 |',
        '|---|---|---:|---:|',
    ]
    for group in result['groups']:
        m = group['known_closure_summary']
        lines.append(f"| {group['precision']} | {group['sampling']} | {fmt(m['max_absolute_E'])} | {fmt(m['max_absolute_normalized_E'])} |")
    lines += ['', 'E=Δu−(c₁A−c₂B)，归一化分母为 |c₁A|+|c₂B|+|Δu|。', '',
              '## 两个外扫窗口识别出的系数', '',
              '以下百分数为相对已知模型参数的偏差，不是实验拟合误差。', '',
              '| 精度 / 采样 | κ 偏差 | r 偏差 | α 偏差 | β 偏差 | αβ/(κr) |',
              '|---|---:|---:|---:|---:|---:|']
    for g in result['groups']:
        pos, neg = g['identification']['positive'], g['identification']['negative']
        errors = []
        for obj, name in [(pos,'kappa'),(pos,'r'),(neg,'alpha'),(neg,'beta')]:
            value = obj.get('relative_coefficient_errors', {}).get(name)
            errors.append('未识别' if value is None else f'{100*value:.6g}%')
        lines.append('| '+g['precision']+' / '+g['sampling']+' | '+' | '.join(errors)+' | '+fmt(g['shared_coefficient_check'].get('identified_product_ratio'))+' |')
    lines += ['', '正支只用 SET4 的两个外扫区间，负支只用 RESET4 的两个外扫区间。',
              '其余 8 个固定区间的闭合结果、列尺度、奇异值、条件数、正系数检查、',
              'b_F/b_G 与共同倍率域都在 JSON 中逐项保留；没有更换病态窗口。', '',
              '## 自检能说明什么', '',
              '这批数组由已知共同动力学生成。离散识别出现的系数或乘积偏差，',
              '首先说明采样/梯形积分和条件数会给积分诊断引入偏差，不能据此否定',
              '生成这些数组的模型。较小闭合误差也不能认证未观测的实验点间轨迹。', '',
              '没有依据任意阈值宣布物理通过或失败。原方程的连续积分恒等式仍然成立；',
              '将本方法用于实验前，必须处理插值假设、反演病态性及误差传播。', '',
              f"数值算术时间 {result['arithmetic_wall_s']:.6f} s；来源加载/哈希另计。", '',
              '来源哈希、逐窗口原索引与采样索引、全部算术量和失败状态见 SELF_CHECK_RESULT.json。',
              '该自检不替代完整 1R/1T1R 联合验收。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-authorized', action='store_true')
    args = parser.parse_args()
    if not args.run_authorized:
        parser.error('Explicit authorized single execution required')
    if RESULT.exists() or (HERE/'SELF_CHECK_STARTED.json').exists():
        parser.error('This self-check was already started; no automatic rerun')
    started = time.monotonic()
    script_hash = sha(Path(__file__).read_bytes())
    plan_hash = sha((HERE/'SELF_CHECK_PLAN.md').read_bytes())
    json_write(HERE/'SELF_CHECK_STARTED.json', {'status':'authorized_single_arithmetic_attempt',
        'script_sha256':script_hash, 'plan_sha256':plan_hash, 'new_model_evaluations':0})
    sources = {}
    result = {'status':'loading_saved_sources', 'new_model_function_calls':0, 'new_roots':0,
              'new_inversions':0, 'new_ODEs':0, 'new_optimizations':0,
              'experimental_current_column_used':False, 'extension_accessed':False,
              'original_holdout_accessed':False, 'experimental_mechanism_test':False,
              'script_sha256':script_hash, 'plan_sha256':plan_hash}

    def source(path, expected=None):
        payload = Path(path).read_bytes()
        record = {'sha256':sha(payload), 'bytes':len(payload)}
        if expected is not None:
            record['matches_parent_record'] = record['sha256'] == expected
            if not record['matches_parent_record']:
                raise ValueError('Source hash mismatch: '+str(path))
        sources[str(path)] = record
        return payload

    try:
        parent_path = STAGE/'bounded_coordinate_check.json'
        parent = json.loads(source(parent_path))
        if parent['status'] != 'completed_finite_coordinate_diagnostic' or not parent['terminal_result']:
            raise ValueError('Parent result is not the declared completed terminal stage')
        records = {r['label']:r for r in parent['requests'] if r['label'] in ('final_nominal','final_refined')}
        if set(records) != {'final_nominal','final_refined'}:
            raise ValueError('Both declared final precision chains required')
        if records['final_nominal']['parameters'] != records['final_refined']['parameters']:
            raise ValueError('Precision chains differ in known parameters')
        arrays, summaries = {}, {}
        allowed = ('command_voltage_v','path_voltage_s','state_u','current_a',
                   'local_voltage_v','original_scoring_mask')
        known = None
        for precision in PRECISIONS:
            record = records['final_'+precision]
            if record['status'] != 'completed_chain' or [r['phase'] for r in record['phase_results']] != list(PHASES):
                raise ValueError('Final chain must contain precisely the three declared phases')
            previous = record['parameters']['initial_u']
            for row in record['phase_results']:
                name = row['phase']
                meta = row['artifacts']
                summary = json.loads(source(meta['summary']['path'],meta['summary']['sha256']))
                payload = source(meta['arrays']['path'],meta['arrays']['sha256'])
                with np.load(BytesIO(payload), allow_pickle=False) as archive:
                    a = {k:archive[k].copy() for k in allowed}
                summaries[(precision,name)], arrays[(precision,name)] = summary, a
                p = summary['parameters']; b = summary['b']
                truth = {'lambda':p['lambda_per_v'], 'kappa':p['kappa_per_v2'],
                         'r':p['kappa_per_v2']/p['J_positive_a'], 'b':b,
                         'alpha':p['kappa_per_v2']*math.exp(2*b),
                         'beta':p['kappa_per_v2']/p['J_positive_a']*math.exp(-2*b)}
                if known is None: known = truth
                if truth != known:
                    raise ValueError('Saved phase known dynamics differ')
                if summary['status'] != 'completed' or not summary['phase_integrity']['complete_integrity_passed']:
                    raise ValueError('Saved final phase was not fully validated')
                if float(a['state_u'][0]) != previous or summary['initial_u'] != previous:
                    raise ValueError('Saved chain lost actual state continuity')
                previous = float(a['state_u'][-1])
                if previous != summary['terminal_u']:
                    raise ValueError('Saved native terminal differs from summary')
                shapes = {a[k].shape for k in allowed}
                if len(shapes) != 1 or a['state_u'].ndim != 1:
                    raise ValueError('Saved native arrays inconsistent')
                if a['original_scoring_mask'].dtype != np.bool_:
                    raise ValueError('Original mask must be boolean')
                for k in allowed[:-1]:
                    if not np.all(np.isfinite(a[k])):
                        raise ValueError('Nonfinite saved array: '+k)
                if np.any(a['state_u'] <= 0.) or np.any(np.diff(a['path_voltage_s']) < 0.):
                    raise ValueError('Invalid saved state or path order')
        for name in PHASES:
            a,z=arrays[('nominal',name)],arrays[('refined',name)]
            for key in ('command_voltage_v','path_voltage_s','original_scoring_mask'):
                if not np.array_equal(a[key],z[key]):
                    raise ValueError('Precision chains use different native sampling: '+key)
        result.update(known_parameters=known, source_sha256=sources,
                      input_loading_and_hash_wall_s=time.monotonic()-started)
        compute_start=time.monotonic()
        groups=[]
        for precision in PRECISIONS:
            for sampling in SAMPLING:
                rows=[];unique_by_phase={}
                for name in PHASES:
                    a=arrays[(precision,name)]
                    command=abs(a['command_voltage_v']); pivot=int(np.argmax(command))
                    s,u=a['path_voltage_s'],a['state_u']
                    local,current=abs(a['local_voltage_v']),abs(a['current_a'])
                    F=local*np.exp(known['lambda']*local-u)
                    G=local*current*np.expm1(u)
                    unique_by_phase[name]=set()
                    for wid,branch,first,last in WINDOWS:
                        branch_mask=np.arange(len(command))<=pivot if branch=='outward' else np.arange(len(command))>pivot
                        first_idx=np.flatnonzero(branch_mask&np.isclose(command,first,rtol=0.,atol=1e-12))
                        last_idx=np.flatnonzero(branch_mask&np.isclose(command,last,rtol=0.,atol=1e-12))
                        if len(first_idx)!=1 or len(last_idx)!=1 or first_idx[0]>=last_idx[0]:
                            raise ValueError('Fixed endpoint absent, nonunique or nonchronological: '+name+wid)
                        native=np.arange(first_idx[0],last_idx[0]+1,dtype=int)
                        if not np.all(branch_mask[native]) or not np.all(a['original_scoring_mask'][native]):
                            raise ValueError('Fixed window contains a different branch or non-scored native point')
                        if not np.all(np.diff(s[native])>0.):
                            raise ValueError('Fixed window path is not strictly increasing')
                        unique_by_phase[name].update(native.tolist())
                        use=native.copy() if sampling=='native' else native[::2].copy()
                        if use[-1]!=native[-1]:use=np.r_[use,native[-1]]
                        A,B=trap(F[use],s[use]),trap(G[use],s[use])
                        delta=float(u[native[-1]]-u[native[0]])
                        coefficients=(known['alpha'],known['beta']) if name=='reset4' else (known['kappa'],known['r'])
                        if not all(math.isfinite(x) for x in (A,B,delta)) or A<=0. or B<=0.:
                            raise ValueError('Nonpositive/nonfinite fixed-window integrals')
                        rows.append({'id':name+'__'+wid,'phase':name,'branch':branch,
                            'command_abs_start_v':first,'command_abs_stop_v':last,
                            'native_indices':native,'used_indices':use,'native_count':len(native),
                            'used_count':len(use),'path_start_v':float(s[use[0]]),'path_stop_v':float(s[use[-1]]),
                            'A_v2':A,'B_a_v2':B,'delta_u':delta,
                            'known_coefficient_closure':closure(delta,A,B,coefficients)})
                by_id={row['id']:row for row in rows}
                pos=identify([by_id['set4__outward_low'],by_id['set4__outward_high']],
                    (known['kappa'],known['r']),['kappa','r'])
                neg=identify([by_id['reset4__outward_low'],by_id['reset4__outward_high']],
                    (known['alpha'],known['beta']),['alpha','beta'])
                shared={'available':False,'known_product_ratio':1.,'known_b':known['b']}
                if pos['identification_available'] and neg['identification_available']:
                    pc,nc=pos['identified_coefficients'],neg['identified_coefficients']
                    k,r,al,be=pc['kappa'],pc['r'],nc['alpha'],nc['beta']
                    product_denominator=k*r
                    shared.update(available=True,all_four_coefficients_positive=bool(min(k,r,al,be)>0.),
                        identified_product_ratio=float(al*be/product_denominator) if product_denominator!=0. else None,
                        product_denominator_zero=bool(product_denominator==0.))
                    if min(k,r,al,be)>0.:
                        tf,tg=al/k,r/be
                        bf,bg=.5*math.log(tf),.5*math.log(tg)
                        shared.update(recovery_multiplier=tf,capture_multiplier=tg,b_F=bf,b_G=bg,
                            b_F_minus_b_G=bf-bg,b_F_error_from_known=bf-known['b'],
                            b_G_error_from_known=bg-known['b'],
                            both_multipliers_in_declared_domain=bool(1.<=tf<=100. and 1.<=tg<=100.))
                    else:
                        shared.update(log_b_values_defined=False,interpretation='Nonpositive discrete coefficients; no physical b inferred')
                identification_ids={'set4__outward_low','set4__outward_high','reset4__outward_low','reset4__outward_high'}
                checks=[]
                for row in rows:
                    group=neg if row['phase']=='reset4' else pos
                    check={'window_id':row['id'],'used_for_identification':row['id'] in identification_ids,
                           'identified_coefficients_available':group['identification_available']}
                    if group['identification_available']:
                        names=group['coefficient_names'];coeff=group['identified_coefficients']
                        check['closure']=closure(row['delta_u'],row['A_v2'],row['B_a_v2'],[coeff[n] for n in names])
                    checks.append(check)
                norms=[row['known_coefficient_closure']['absolute_normalized_E'] for row in rows]
                group={'precision':precision,'sampling':sampling,'windows':rows,
                    'unique_native_window_points_by_phase':{k:len(v) for k,v in unique_by_phase.items()},
                    'known_closure_summary':{'max_absolute_E':max(row['known_coefficient_closure']['absolute_E'] for row in rows),
                        'max_absolute_normalized_E':max(v for v in norms if v is not None),
                        'zero_normalization_denominator_windows':[row['id'] for row in rows if row['known_coefficient_closure']['denominator_zero']]},
                    'identification':{'positive':pos,'negative':neg},'shared_coefficient_check':shared,
                    'identified_coefficient_window_checks':checks,
                    'remaining_check_windows':8,
                    'interpretation':'Known-model discretization self-check only; no experimental mechanism conclusion'}
                groups.append(group)
        comparisons=[]
        for sampling in SAMPLING:
            nominal=next(g for g in groups if g['precision']=='nominal' and g['sampling']==sampling)
            refined=next(g for g in groups if g['precision']=='refined' and g['sampling']==sampling)
            diffs=[]
            for a,z in zip(nominal['windows'],refined['windows']):
                if a['id']!=z['id'] or not np.array_equal(a['used_indices'],z['used_indices']):
                    raise ValueError('Precision integral comparison samples differ')
                diffs.append({'window_id':a['id'],'A_nominal_minus_refined':a['A_v2']-z['A_v2'],
                    'B_nominal_minus_refined':a['B_a_v2']-z['B_a_v2'],
                    'E_nominal_minus_refined':a['known_coefficient_closure']['E']-z['known_coefficient_closure']['E']})
            comparisons.append({'sampling':sampling,'window_differences':diffs,
                'max_absolute_E_difference':max(abs(row['E_nominal_minus_refined']) for row in diffs),
                'interpretation':'Same native observation locations; solver-precision comparison, not sampling-error bound'})
        result.update(status='completed_known_model_discretization_selfcheck',groups=groups,
            precision_comparisons=comparisons,arithmetic_wall_s=time.monotonic()-compute_start,
            native_sampling_error_certificate=False,experimental_noise_certificate=False,
            new_physics_claim=False,source_sha256=sources)
        result_artifact=json_write(RESULT,result)
        report_payload=make_report(result).encode()
        with (HERE/'SELF_CHECK_REPORT.md').open('xb') as f:f.write(report_payload)
        json_write(HERE/'SELF_CHECK_ARTIFACTS.json',{
            'result':result_artifact,'report':{'path':str(HERE/'SELF_CHECK_REPORT.md'),'sha256':sha(report_payload)},
            'source_count':len(sources),'script_sha256':script_hash,'plan_sha256':plan_hash,
            'single_execution':True,'total_wall_s_before_final_artifact_write':time.monotonic()-started})
        print(json.dumps(clean({'status':result['status'],'arithmetic_wall_s':result['arithmetic_wall_s'],
            'groups':[{'precision':g['precision'],'sampling':g['sampling'],
                       'known_closure':g['known_closure_summary'],'shared_coefficients':g['shared_coefficient_check'],
                       'positive':g['identification']['positive'],'negative':g['identification']['negative']} for g in groups]}),indent=2),flush=True)
    except Exception as exc:
        result.update(status='failed_saved_selfcheck_no_retry',error_type=type(exc).__name__,error=str(exc),source_sha256=sources)
        if not RESULT.exists():json_write(RESULT,result)
        raise


if __name__ == '__main__':
    main()
