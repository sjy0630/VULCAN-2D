"""One bounded training-history static screen, no ODE, fit or re-anchoring."""
from pathlib import Path
from dataclasses import asdict
import argparse
import copy
import hashlib
import json
import math
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent/'zero_crossing_compatibility_2026-11-04'
FROZEN = {
    'check_compatibility.py': '576b529060a5a8d3c73ee8221e81627b01c8b69ecf2eb0d9ec22c74cac22f831',
    'minimal_static_loader.py': '72e2c018a68d9f2e66dd5279b67e428993e800203b25df4e9d2c18c787f2e778',
    'compatibility_result.json': 'b2ad8ef97df45b8f984af63841bd225f007d428aa6c733476d8c4899f8f7e227',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def power_requirement(x, y, m):
    a = m-1.
    log_ratio = math.log1p((y-x)/x)
    return log_ratio if a == 0. else math.exp(a*math.log(x))*math.expm1(a*log_ratio)/a


def mixed_requirement(x, y, epsilon):
    return y-x if epsilon == 0. else math.log1p(epsilon*(y-x)/(1.+epsilon*x))/epsilon


def necessary_structure_limits(x, y, upper):
    if y <= x:
        return {'status': 'no_positive_growth_requirement', 'power_cap': None, 'mixed_floor': None}
    power = {'m1_requirement': power_requirement(x,y,1.), 'm2_requirement': power_requirement(x,y,2.)}
    if power['m1_requirement'] > upper:
        power.update(status='entire_m_1_to_2_domain_insufficient', conservative_upper_m=None)
    elif power['m2_requirement'] <= upper:
        power.update(status='m2_not_excluded_by_growth_bound', conservative_upper_m=None)
    else:
        low, high, iterations = 1., 2., 0
        for _ in range(64):
            middle = .5*(low+high)
            if middle == low or middle == high:
                break
            if power_requirement(x,y,middle) <= upper:
                low = middle
            else:
                high = middle
            iterations += 1
        power.update(status='conditional_m_upper', pass_lo=low, fail_hi=high,
            F_pass_lo=power_requirement(x,y,low), F_fail_hi=power_requirement(x,y,high),
            conservative_upper_m=high, algebraic_bisection_iterations=iterations)
    mixed = {'epsilon0_requirement': mixed_requirement(x,y,0.), 'epsilon1_requirement': mixed_requirement(x,y,1.)}
    if mixed['epsilon0_requirement'] <= upper:
        mixed.update(status='epsilon0_not_excluded', conservative_lower_epsilon=0.)
    elif mixed['epsilon1_requirement'] > upper:
        mixed.update(status='epsilon_0_to_1_domain_insufficient', conservative_lower_epsilon=None)
    else:
        low, high, iterations = 0., 1., 0
        for _ in range(64):
            middle = .5*(low+high)
            if middle == low or middle == high:
                break
            if mixed_requirement(x,y,middle) > upper:
                low = middle
            else:
                high = middle
            iterations += 1
        mixed.update(status='conditional_epsilon_lower', fail_lo=low, pass_hi=high,
            F_fail_lo=mixed_requirement(x,y,low), F_pass_hi=mixed_requirement(x,y,high),
            conservative_lower_epsilon=low, algebraic_bisection_iterations=iterations)
    return {'status': 'positive_growth_requirement', 'power_cap': power, 'mixed_floor': mixed}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-authorized',action='store_true')
    args=parser.parse_args()
    if not args.run_authorized:
        parser.error('Explicit finite-plan execution required')
    destination=HERE/'population_result.json'
    if destination.exists():
        raise FileExistsError('No repeat numerical screening in this stage')
    load_started=time.monotonic()
    for name,digest in FROZEN.items():
        assert sha(PREVIOUS/name)==digest, name
    sys.path.insert(0,str(PREVIOUS))
    from check_compatibility import invert_readpoint
    from minimal_static_loader import load_minimal
    kernel=load_minimal()
    prior=json.loads((PREVIOUS/'compatibility_result.json').read_text())
    audit=json.loads((HERE/'audit.json').read_text())
    assert audit['anomaly_count']==0 and audit['complete_bridges']==31
    assert audit['source_sha256'][str(kernel.cache/'native_observations.npz')]==kernel.source_sha256[str(kernel.cache/'native_observations.npz')]
    assert [row['reset_record'] for row in audit['bridges']]==list(range(4,35))
    for row in audit['bridges']:
        assert row['set_record']==row['reset_record']+1
        assert row['all_required_checks_passed']
        assert math.isclose(row['total_command_path_distance_v'],.4,abs_tol=1e-14)
    upper=prior['necessary_budget']['decision_upper_deltaZ']
    assert asdict(kernel.parameters)==prior['parameters']
    assert kernel.ron==prior['Ron_ohm'] and kernel.gc==prior['gamma_c_ev']
    expected_upper=.5*kernel.parameters.kappa_per_v2*.2**2*math.exp(.2*kernel.parameters.lambda_per_v)*(100.+1.)*(1.+1e-6)
    assert math.isclose(expected_upper,upper,rel_tol=1e-14)
    sources=dict(kernel.source_sha256)
    for p in [PREVIOUS/name for name in FROZEN]+[HERE/'audit.json',HERE/'EXPERIMENT_PLAN.md',Path(__file__)]:
        sources[str(p)]=sha(p)
    result={'status':'started','source_sha256':sources,'input_IO_wall_s':time.monotonic()-load_started,
        'parameters':asdict(kernel.parameters),'Ron_ohm':kernel.ron,'gamma_c_ev':kernel.gc,
        'shared_decision_upper_deltaZ':upper,'predeclared_bridges':31,
        'readpoint_data_source':'Independent audit extracts; native cache hash verified without re-decoding unrelated fields.',
        'bridges':[],'new_readpoint_attempts':0,'new_readpoint_completed':0,'reused_readpoints':0,
        'new_ODEs':0,'new_fits':0,'new_b_trials':0,'state_reanchoring':False,'original_holdout_accessed':False,
        'full_joint_contract_evaluated':False,'failures':[]}
    started=time.monotonic()
    def check_budget():
        if time.monotonic()-started>10.:
            raise TimeoutError('10 s numerical budget exhausted')
    try:
        for row in audit['bridges']:
            check_budget()
            k=row['reset_record']
            output={'reset_record':k,'set_record':k+1,'bridge_id':row['bridge_id'],'status':'started','readpoints':[]}
            result['bridges'].append(output)
            for index,phase in enumerate(['reset','set']):
                read=row[phase+'_read']
                if not read['retained_by_original_mask']:
                    output['readpoints'].append({'status':'threshold_ineligible','data':read})
                    continue
                if k==4:
                    point=copy.deepcopy(prior['readpoints'][index])
                    assert point['command_voltage_v']==read['command_v']
                    assert point['observed_current_magnitude_a']==read['current_magnitude_a']
                    result['reused_readpoints']+=1
                    provenance='reused_frozen_two_readpoint_result'
                else:
                    result['new_readpoint_attempts']+=1
                    try:
                        point=invert_readpoint(kernel,f'{phase}_{k if phase=="reset" else k+1}_0p2',
                            read['command_v'],read['current_magnitude_a'],check_budget)
                    except TimeoutError:
                        raise
                    except Exception as exc:
                        error={'status':'numerically_unresolved','data':read,'error':repr(exc)}
                        output['readpoints'].append(error)
                        result['failures'].append({'bridge':k,'phase':phase,**error})
                        continue
                    result['new_readpoint_completed']+=1
                    provenance='new_conditional_inversion'
                output['readpoints'].append({'status':'valid','native_index':read['native_index'],
                    'provenance':provenance,**point})
            if any(p['status']=='threshold_ineligible' for p in output['readpoints']):
                output['status']='threshold_ineligible'
            elif any(p['status']!='valid' for p in output['readpoints']):
                output['status']='numerically_unresolved'
            else:
                minus,plus=output['readpoints']
                x=minus['numerical_padded_Z_bracket'][1]
                y=plus['numerical_padded_Z_bracket'][0]
                output.update(negative_Z_upper=x,positive_Z_lower=y,required_deltaZ_lower=y-x,
                    margin_over_upper=y-x-upper,budget_ratio=(y-x)/upper,
                    status='conditional_violation' if y-x>upper else 'necessary_condition_not_excluding',
                    structure_limits=necessary_structure_limits(x,y,upper))
        check_budget()
        statuses=['conditional_violation','necessary_condition_not_excluding','threshold_ineligible','numerically_unresolved']
        result['counts']={status:sum(r['status']==status for r in result['bridges']) for status in statuses}
        result['counts']['predeclared']=31
        result['counts']['valid']=result['counts']['conditional_violation']+result['counts']['necessary_condition_not_excluding']
        result['chronological_groups']={}
        for name,low,high in [('early',4,13),('middle',14,23),('late',24,34)]:
            selected=[r for r in result['bridges'] if low<=r['reset_record']<=high]
            result['chronological_groups'][name]={'reset_labels':[low,high],'n':len(selected),
                **{status:sum(r['status']==status for r in selected) for status in statuses}}
        caps=[];floors=[]
        for r in result['bridges']:
            limits=r.get('structure_limits',{})
            cap=limits.get('power_cap');floor=limits.get('mixed_floor')
            if cap and cap.get('conservative_upper_m') is not None:
                caps.append({'reset_record':r['reset_record'],'upper_m':cap['conservative_upper_m']})
            if floor and floor.get('conservative_lower_epsilon') is not None:
                floors.append({'reset_record':r['reset_record'],'lower_epsilon':floor['conservative_lower_epsilon']})
        result['shared_structure_necessary_limits']={
            'most_restrictive_m':min(caps,key=lambda x:x['upper_m']) if caps else None,
            'most_restrictive_epsilon':max(floors,key=lambda x:x['lower_epsilon']) if floors else None,
            'not_parameter_estimates_or_sufficient_conditions':True}
        assert result['new_readpoint_attempts']<=60 and result['reused_readpoints']==2
        result['completed_new_circuit_rechecks']=2*result['new_readpoint_completed']
        result['status']='completed_with_unresolved_points' if result['failures'] else 'completed_finite_population_screen'
    except Exception as exc:
        result['status']='stopped_incomplete'
        result['error']=repr(exc)
        raise
    finally:
        result['numerical_wall_s']=time.monotonic()-started
        payload=(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
        with destination.open('xb') as handle:
            handle.write(payload)
        print(json.dumps({'status':result['status'],'counts':result.get('counts'),
            'numerical_wall_s':result['numerical_wall_s'],'result_sha256':hashlib.sha256(payload).hexdigest()},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
