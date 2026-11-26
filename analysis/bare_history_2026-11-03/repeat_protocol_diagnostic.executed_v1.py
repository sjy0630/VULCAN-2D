"""Four existing bare candidates, three consecutive identical command pairs.

Hypothetical no-wait repeat protocol only. Simulation cycle numbers do not
identify experimental labels. No fitting, chronological score, or joint claim.
"""
from pathlib import Path
import hashlib
import json
import math
import signal
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREVIOUS = ROOT/'analysis/parameter_transfer_2026-11-03'
sys.path.insert(0, str(PREVIOUS))
from evaluate_bare_entry import OPTIONS, make_problem, predicted_feature


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gamma_record(law, u):
    log_gamma = law.exact_law.log_gamma(u)
    return dict(u=float(u), gamma_ev=math.exp(log_gamma) if math.isfinite(log_gamma) else None,
                gamma_class='finite' if math.isfinite(log_gamma) else 'positive_infinity_ohmic_limit')


def main():
    started = time.monotonic()
    first_path = PREVIOUS/'bare_entry_forward.json'
    three_path = PREVIOUS/'bare_three_candidates_forward.json'
    first = json.loads(first_path.read_text())
    three = json.loads(three_path.read_text())
    constraints_path = ROOT/'analysis/constrained_shared_2026-11-03/feature_constraints.json'
    constraints = json.loads(constraints_path.read_text())
    selected = [name for name in constraints['proposed_success_criteria_before_next_fit']['primary_feature_ids']
                if constraints['features'][name]['topology'] == '1r']
    assert len(selected) == 7
    assert first['options'] == three['options'] == OPTIONS
    candidates = [dict(identifier='previous_first', q=first['q'], source=first_path,
                       archive=first_path.with_suffix('.npz'), archive_prefix='', previous_row=first)]
    candidates += [dict(identifier=f'previous_survivor_{idx}', q=row['q'], source=three_path,
                        archive=three_path.with_suffix('.npz'), archive_prefix=f'candidate_{idx}_', previous_row=row)
                   for idx,row in enumerate(three['rows'])]
    assert len(candidates) == 4 and all(c['previous_row']['n_primary_features'] == 7 for c in candidates)
    output_path = HERE/'repeat_protocol_diagnostic.json'
    arrays = {}
    report = dict(status='running_hypothetical_identical_bare_repeats', options=OPTIONS,
                  maximum_pair_calls=12, pair_calls=0, completed_pairs=0, candidates=[],
                  protocol=dict(topology='1r', cycles_per_candidate=3, waiting_time_assumption='zero',
                      command='same original native bare SET waveform followed by same native RESET waveform each simulation cycle',
                      state_policy='initial_u assigned once per candidate; all phase and cycle boundaries inherit final_u',
                      experimental_label_mapping=None),
                  data_scope='Only original training loader and previously generated frozen trajectory files; no holdout arrays.',
                  evidence_scope=[
                      'User confirmed bare labels refer to repeated cycling of one device; actual order, same-number SET/RESET pairing, missing label4 and label23 meanings remain unconfirmed.',
                      'simulation_cycle 1..3 are hypothetical equal-command repetitions with no wait, not experimental labels or known chronology.',
                      'Seven original aggregate feature intervals are retained solely as diagnostics. No chronological experimental score or full joint-model claim.',
                      'Exactly four previously forwarded candidates; no parameter optimization, new points, new spectral tables, or manual state resets.',
                      'Gamma at u=0 is the model positive-infinity ohmic limit, represented by null plus gamma_class in JSON and +inf in NPZ.'])

    def save():
        report['elapsed_s'] = time.monotonic()-started
        output_path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        np.savez_compressed(output_path.with_suffix('.npz'), **arrays)

    def alarm(*_):
        raise TimeoutError('90-second repeat-protocol budget exhausted')

    signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, 90.)
    save()
    try:
        problem = make_problem('1r', integrator_options=OPTIONS, mode='full')
        # _pair starts at zero. The preceding RESET must really end at zero,
        # otherwise repeating it would silently invent an inter-cycle segment.
        assert problem.waveforms['1r','reset'][-1] == 0.
        report['command_boundary_verified_reset_ends_at_zero_v'] = True
        for ci, item in enumerate(candidates):
            q = np.asarray(item['q'], float)
            assert np.array_equal(q, np.asarray(item['previous_row']['q']))
            law = problem.joint.law_factory(problem.to_joint_q(q), problem.ron)
            initial_u = state = float(q[5])
            candidate = dict(identifier=item['identifier'], q=q.tolist(), gamma_c_ev=10.**q[0],
                             status='running', initial_state=gamma_record(law,state), simulation_cycles=[])
            report['candidates'].append(candidate)
            arrays[f'candidate_{ci}_q'] = q
            with np.load(item['archive']) as z:
                old = {phase:{key:z[item['archive_prefix']+phase+'_'+key].copy()
                              for key in ('current_a','state_u','command_v')}
                       for phase in ('set','reset')}
            for cycle in range(1,4):
                cycle_start = state
                row = dict(simulation_cycle=cycle, initial_state=gamma_record(law,state), status='running')
                candidate['simulation_cycles'].append(row)
                save()
                call_started = time.monotonic()
                try:
                    assert report['pair_calls'] < 12
                    report['pair_calls'] += 1
                    pair = problem.joint._pair(law, state, '1r')
                    row['pair_elapsed_s'] = time.monotonic()-call_started
                    report['completed_pairs'] += 1
                    assert float(pair[0]['state_u'][0]) == cycle_start
                    assert float(pair[0]['final_u']) == float(pair[1]['state_u'][0])
                    state = float(pair[-1]['final_u'])
                    row['final_state'] = gamma_record(law,state)
                    row['SET_final_RESET_initial_state'] = gamma_record(law,float(pair[0]['final_u']))
                    row['state_continuity_verified'] = True
                    for phase, trace in zip(('set','reset'),pair):
                        prefix = f'candidate_{ci}_simulation_cycle_{cycle}_{phase}_'
                        arrays[prefix+'command_v'] = problem.waveforms['1r',phase]
                        for key,value in trace.items():
                            if isinstance(value,np.ndarray):
                                arrays[prefix+key] = value
                        arrays[prefix+'initial_u'] = np.array(cycle_start if phase=='set' else pair[0]['final_u'])
                        arrays[prefix+'final_u'] = np.array(trace['final_u'])
                        arrays[prefix+'gamma_ev'] = np.array([math.exp(law.exact_law.log_gamma(u)) for u in trace['state_u']])
                    features = {}
                    for name in selected:
                        definition = constraints['features'][name]
                        phase = definition['phase']
                        trace = pair[0 if phase=='set' else 1]
                        lower,upper = definition['proposed_empirical_acceptance_interval']
                        try:
                            value, detail = predicted_feature(name,definition,problem.waveforms['1r',phase],trace['current_a'])
                            if not np.isfinite(value):
                                raise ValueError('Nonfinite feature')
                            features[name] = dict(value=float(value),fixed_aggregate_interval=[lower,upper],
                                                  passed=bool(lower<=value<=upper),detail=detail)
                        except ValueError as exc:
                            features[name] = dict(value=None,fixed_aggregate_interval=[lower,upper],
                                                  passed=False,failure=str(exc))
                    row['aggregate_feature_diagnostics'] = features
                    row['aggregate_features_passed'] = sum(x['passed'] for x in features.values())
                    row['aggregate_feature_count'] = 7
                    row['all_seven_aggregate_features_passed'] = row['aggregate_features_passed']==7
                    threshold_name = next(name for name in selected if 'current_threshold_A' in constraints['features'][name])
                    row['SET_threshold'] = features[threshold_name]
                    row['RESET_peak_current'] = features['1r.reset.outward.peak_current']
                    row['RESET_tail_current_at_return_0p2V'] = features['1r.reset.return.I_0p2V']
                    if cycle == 1:
                        comparison = {}
                        for phase,trace in zip(('set','reset'),pair):
                            current = abs(trace['current_a'])
                            old_i = abs(old[phase]['current_a'])
                            mask = (current>0)&(old_i>0)
                            comparison[phase] = dict(
                                command_grid_exactly_equal=bool(np.array_equal(problem.waveforms['1r',phase],old[phase]['command_v'])),
                                max_abs_current_difference_a=float(np.max(abs(trace['current_a']-old[phase]['current_a']))),
                                max_abs_log10_current_difference_nonzero=float(np.max(abs(np.log10(current[mask])-np.log10(old_i[mask])))),
                                max_abs_state_difference=float(np.max(abs(trace['state_u']-old[phase]['state_u']))),
                                arrays_bitwise_equal=bool(np.array_equal(trace['current_a'],old[phase]['current_a']) and np.array_equal(trace['state_u'],old[phase]['state_u'])))
                        row['first_cycle_reproduction_of_previous_forward'] = comparison
                    row['final_minus_once_prepared_u0'] = state-initial_u
                    row['status'] = 'completed_hypothetical_repeat'
                except Exception as exc:
                    row.update(status='numerical_or_diagnostic_failure_preserved',error_type=type(exc).__name__,error=str(exc),
                               interpretation='Failure is retained; no state is invented to continue this chain.')
                    candidate['status'] = 'incomplete_chain'
                    save()
                    if isinstance(exc,TimeoutError):
                        raise
                    break
                save()
            if candidate['status'] != 'incomplete_chain':
                candidate['status'] = 'completed_three_continuous_hypothetical_cycles'
                final = candidate['simulation_cycles'][-1]['final_state']
                candidate['final_state'] = final
                candidate['prepared_initial_state_self_restored_within_state_atol'] = abs(final['u']-initial_u)<=OPTIONS['state_atol']
                candidate['cycle_end_u_spread_last_two'] = abs(candidate['simulation_cycles'][-1]['final_state']['u']-candidate['simulation_cycles'][-2]['final_state']['u'])
            save()
        report['status'] = 'completed_twelve_hypothetical_pairs_no_refit'
    except Exception as exc:
        report.update(status='partial_or_failed_diagnostic_preserved',error_type=type(exc).__name__,error=str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0.)
        report['source_sha256'] = {str(p.relative_to(ROOT)):sha(p) for p in
            (first_path,first_path.with_suffix('.npz'),three_path,three_path.with_suffix('.npz'),constraints_path,
             PREVIOUS/'evaluate_bare_entry.py',PREVIOUS/'group_problem.py',Path(__file__))}
        save()
    lines = ['# 裸器件重复命令假设诊断','',
             '只对4个既有候选各连续运行3对原SET/RESET命令。每候选初态仅赋值一次，SET→RESET→下一SET全部继承；假设无等待。',
             'simulation_cycle不是实验标签。真实顺序、同号配对、缺4和23的含义尚未确认，因此不能把结果称为实际chronological评分。原7项聚合门槛只用于诊断。','',
             '| 既有候选gc/eV | simulation_cycle | 初u | 末u | SET阈值中点/V | RESET峰/A | RESET返扫0.2V/A | 聚合门槛 |',
             '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c in report['candidates']:
        for r in c['simulation_cycles']:
            if r['status']=='completed_hypothetical_repeat':
                lines.append(f'| {c["gamma_c_ev"]:.9g} | {r["simulation_cycle"]} | {r["initial_state"]["u"]:.7g} | {r["final_state"]["u"]:.7g} | {r["SET_threshold"]["value"]} | {r["RESET_peak_current"]["value"]:.7g} | {r["RESET_tail_current_at_return_0p2V"]["value"]:.7g} | {r["aggregate_features_passed"]}/7 |')
            else:
                lines.append(f'| {c["gamma_c_ev"]:.9g} | {r["simulation_cycle"]} | — | — | — | — | — | {r["status"]} |')
    restored = [c.get('prepared_initial_state_self_restored_within_state_atol') for c in report['candidates']]
    lines += ['',f'一次准备初态在三轮后按数值state_atol恢复：{restored}。这只是该无等待、同命令假设中的状态自维持检查，不能单凭它判断真实实验循环顺序或排除其他时序。',
              '初/终Γ、相位衔接状态、全部7项特征及首周期旧轨迹复现误差见JSON，所有原生采样轨迹见NPZ。u=0的Γ为+∞欧姆极限，JSON通过gamma_class明确标记。',
              f'调用{report["pair_calls"]}/12对，完成{report["completed_pairs"]}对；用时{report["elapsed_s"]:.3f}s。未优化、未取新点、未读取保留集、未改旧文件或模型主干。']
    output_path.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:report[k] for k in ('status','pair_calls','completed_pairs','elapsed_s')},indent=2))
    for c in report['candidates']:
        print(json.dumps(dict(gamma_c_ev=c['gamma_c_ev'],prepared_state_restored=c.get('prepared_initial_state_self_restored_within_state_atol'),
            cycles=[{k:r.get(k) for k in ('simulation_cycle','initial_state','final_state','aggregate_features_passed','SET_threshold','RESET_peak_current','RESET_tail_current_at_return_0p2V','first_cycle_reproduction_of_previous_forward','error')}
                    for r in c['simulation_cycles']]),indent=2))


if __name__=='__main__':
    main()
