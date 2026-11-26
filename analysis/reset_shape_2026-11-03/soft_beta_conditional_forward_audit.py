"""Pure arithmetic audit of saved SET trajectories and original seed endpoints."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE=Path(__file__).resolve().parent
CACHE=Path('/private/tmp/vulcan-runtime-cache')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=HERE/'soft_beta_conditional_forward.json'
    report=json.loads(p.read_text())
    assert report['n_completed_runs']==4 and report['n_new_recovery_solver_attempts']==2
    assert len(report['runs'])==6 and report['n_failed_runs']==2
    assert report['cumulative_compute_elapsed_s']<120
    for name in ('soft_beta_conditional_forward.py','soft_beta_conditional_forward.plan.json'):
        source=HERE/name
        assert sha(source)==report['source_sha256'][str(source)]
    original_path=CACHE/'RON_SCALING_REVIEW.json'
    original=json.loads(original_path.read_text())
    ref=next(x for x in original['rows'] if x['Ron_ohm']==report['plan']['Ron_ohm'])['anchor_reinversions']
    complete=[r for r in report['runs'] if r['status']=='completed_conditional_single_cycle_forward']
    result=dict(scope='Saved-array arithmetic only, no new ODE, fitted rate, initial state or experimental uncertainty.',
                source_sha256={str(q):sha(q) for q in (p,p.with_suffix('.npz'),original_path,Path(__file__))},
                cumulative_compute_elapsed_s=report['cumulative_compute_elapsed_s'],n_physical_seeds=2,n_solver_attempts=6,
                n_preserved_numerical_unknown_attempts=2,n_completed_precision_trajectories=4,trajectories=[])
    with np.load(p.with_suffix('.npz')) as z, np.load(HERE/'pre_inactive_endpoint_fix/soft_beta_conditional_forward.npz') as before:
        v=z['command_voltage_v'];obs=abs(z['observed_current_a']);mask=z['original_scoring_mask']
        out=np.arange(len(v))<=np.argmax(v)
        np.testing.assert_array_equal(mask,np.isfinite(obs)&(obs>=1e-11)&(abs(v)>1e-12))
        assert mask.sum()==500 and (mask&out).sum()==250 and (mask&~out).sum()==250
        for prefix in ('beta4_refined','beta6_nominal'):
            for suffix in ('current_a','state_u','local_voltage_v','KCL_residual_a'):
                np.testing.assert_array_equal(z[prefix+'_'+suffix],before[prefix+'_'+suffix])
        result['two_previously_completed_trajectories_preserved_bitwise']=True
        for run in complete:
            prefix=run['seed']['name']+'_'+run['precision']['name']
            current,state=z[prefix+'_current_a'],z[prefix+'_state_u']
            assert np.all(np.isfinite(current)) and np.all(current>=0) and np.all(current[mask]>0)
            assert np.all(np.isfinite(state)) and np.all((state>=0)&(state<=80)) and max(np.diff(state))<=5e-7
            e=np.log10(current[mask])-np.log10(obs[mask]);rmse=float(np.sqrt(np.mean(e*e)))
            np.testing.assert_allclose(rmse,run['metrics']['all500']['raw_RMSE_decade'],rtol=1e-13)
            rows=[]
            for name,A,isout,direction in (('outward_2V_lower',2.,True,'at_least'),
                                          ('return_5V_lower',5.,False,'at_least'),
                                          ('return_2V_upper',2.,False,'at_most')):
                branch=out if isout else ~out
                j=int(np.argmin(np.where(branch,abs(v-A),np.inf)))
                assert v[j]==A
                bound=ref[name]['current_a'];actual=float(current[j])
                gamma=ref[name]['rederived_gamma_ev'];bound_u=float(np.log1p(report['plan']['gamma_c_ev']/gamma))
                rows.append(dict(endpoint=name,native_index=j,command_v=A,predicted_current_a=actual,
                                 original_conditional_bound_current_a=bound,direction=direction,
                                 actual_u=float(state[j]),original_conditional_bound_u=bound_u,
                                 predicted_current_over_bound=actual/bound,
                                 actual_endpoint_condition_satisfied=actual>=bound if direction=='at_least' else actual<=bound,
                                 scope='Original conditional seed endpoint only, not an original32-record aggregate curve gate.'))
            low=mask&out&(v<=.2+1e-12)
            low_sse=float(np.sum(np.log10(current[low]/obs[low])**2))
            all_sse=float(np.sum(e*e))
            cert=run.get('inactive_endpoint_certificate')
            if cert:
                assert cert['endpoint_test_count']==cert['inactive_trigger_count']
                assert cert['max_upper_log_drive'] is None or cert['max_upper_log_drive']<=0
                assert all(not x['out_of_domain_current_evaluated'] and x['exact_inactive_rate_proven'] for x in cert['evaluations'])
            result['trajectories'].append(dict(name=prefix,seed=run['seed'],metrics=run['metrics'],
                original_seed_endpoints=rows,all_three_endpoint_conditions_satisfied=all(x['actual_endpoint_condition_satisfied'] for x in rows),
                derived_initial_u=run['derived_initial_u'],terminal_u=run['terminal_u'],forward_anchor_state_error=run['forward_anchor_state_error'],
                low_outward_ten_point_fraction_of_total_squared_log_error=low_sse/all_sse,
                reference_features=run['reference_features'],inactive_endpoint_certificate=cert))
    result['precision_comparisons']=report['precision_comparisons']
    result['conclusion_scope']=['Endpoint feasibility is necessary, not a successful full-curve or unified-model fit.',
        'Low-bias errors remain in the500-point denominator, and non-low-bias residuals are also material.',
        'The initial state was reconstructed from the same measured0.2V anchor; no causally independent initial-state prediction is established.',
        'Two numerical_unknown attempts are retained and two exact-inactive endpoint recoveries are additional solver attempts, not new physical seeds.']
    q=HERE/'soft_beta_conditional_forward_audit.json';q.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# 实际 SET 到达条件与保存轨迹审计','',
           '本附录只复算已有NPZ；未增加积分、拟合或初态。两个既有完整轨迹与修复前数组逐项bitwise一致。原始mask保持500点，外扫/返扫各250点；所有低压残差均保留。',
           '', '| 候选/精度 | 外扫2V / µA；要求≥26.105524 | 返扫5V / µA；要求≥48.036762 | 返扫2V / µA；要求≤30.2394 | 三条件同时成立 |',
           '|---|---:|---:|---:|---|']
    for row in result['trajectories']:
        ends=row['original_seed_endpoints']
        cells=[f'{x["predicted_current_a"]*1e6:.7f}（u={x["actual_u"]:.8f}）' for x in ends]
        lines.append('| '+' | '.join([row['name'],*cells,str(row['all_three_endpoint_conditions_satisfied'])])+' |')
    lines+=['', '这些是产生候选所用的三个条件端点，必须与完整曲线误差分开。β4外扫2V略未到达，同时返扫2V超上界；β6能同时满足这三项，但不能据此声称原曲线验收、32圈聚合或1R/1T1R统一已成功。',
            '', '| 候选/精度 | 全500点RMSE | 外扫250点 | 返扫250点 | 外扫低压10点 | 其余480点 | 低压10点占总平方log误差 |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for row in result['trajectories']:
        m=row['metrics'];values=[f'{m[k]["raw_RMSE_decade"]:.7f}' for k in ('all500','outward','return_','low_outward','remaining_above_0p2')]
        lines.append('| '+' | '.join([row['name'],*values,f'{row["low_outward_ten_point_fraction_of_total_squared_log_error"]*100:.3f}%'])+' |')
    lines+=['', '主要结论是：返扫曲线已相对接近，但外扫尤其起始段仍不准确；即便单列0.2V以下的低偏误差，其余480点RMSE仍约0.64–0.66dec，因此问题不只在低偏两个读点。高β与当前k也不能仅凭预算可行就视为统一模型修复成功。',
            '', '## 精度与端点处理', '']
    for name,c in result['precision_comparisons'].items():
        lines.append(f'- {name}：加密后原生电流最大log变化{c["max_native_current_log10_change"]:.6g}dec，状态最大变化{c["max_native_state_change"]:.6g}，末态差{c["terminal_state_change"]:.6g}。')
    for row in result['trajectories']:
        c=row['inactive_endpoint_certificate']
        if c:
            lines+=['',f'{row["name"]}恢复运行的近零inactive证书：触发{c["inactive_trigger_count"]}次，最大驱动上界{c["max_upper_log_drive"]:.9g}≤0，状态范围{c["tested_state_range"]}，正命令电压范围{c["tested_command_range"]}V。只以I_T≤F(vmin,u)证明状态导数为0，未估计域外电流或夹变量。']
    lines+=['', f'保留6个求解尝试：4条完成轨迹、2条原numerical_unknown。只有2个固定物理seed。包含首次接口错误与两次恢复的累计计算{result["cumulative_compute_elapsed_s"]:.6f}s，小于120s。',
            '', '同一0.2V实测锚被用于反向求初态并纳入条件评分；未知真实历史、测量低压贡献、RESET动力学以及bare1R通道均未由此验证。JS仍是log驱动参照参数，不能解释为外加限流。']
    q.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='saved_trajectory_and_endpoint_audit_passed_no_new_integrations',cumulative_compute_s=result['cumulative_compute_elapsed_s'],
                         precision_comparisons=result['precision_comparisons'],
                         endpoint_checks={x['name']:x['all_three_endpoint_conditions_satisfied'] for x in result['trajectories']}),indent=2))


if __name__=='__main__':main()
