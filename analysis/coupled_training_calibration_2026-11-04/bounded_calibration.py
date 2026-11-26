"""Four bounded coordinate rounds on complete original coupled training histories.

Executable preparation only; import does not load data or run an experiment.
Each new candidate uses all 46 pairs / 92 phases / 27023 original residuals.
Scalar-positive parameters and all three entry states remain frozen. This is
a limited four-coordinate search, not full optimization or convergence.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
NATIVE=ROOT/'analysis/coupled_native_training_2026-11-04'
if str(NATIVE) not in sys.path:
    sys.path.insert(0,str(NATIVE))
import training_problem_v2 as tp
import feature_audit as fa
import run_native_baseline_v2 as artifacts

if artifacts.tp is not tp or artifacts.fa is not fa:
    raise RuntimeError('Inert artifact helper imports do not share the frozen module identities')

FROZEN_NATIVE={
    'training_problem_v2.py':'de4539d573e3b0c13fbfbf8a9d89b28d29a2baa8d3458ef3145bb0e9f32480f7',
    'coupled_forward_v2.py':'c6873e8cec07965a3637bb695c942fab2fd75d699f0413ae1a2b0332097c2d2e',
    'feature_audit.py':'ad45f4192175a23996f8dab3742984cdd0e9615e25565394212dabe5e421bda4',
    'run_native_baseline_v2.py':'b5a8ebda898cbac07d2fd7f01bee58729c9f5c43042c82fa51819d9754691c86',
    'SEED_DERIVATION.json':'7b4ffd1933a7e4bd02ce605194bee37a9b6be15945a9c2b9e19cc41f96874b6f',
}
COORDINATES=('a_max_s','kappa_a_per_v2','rho','b')
FROZEN_POSITIVE=('lambda_per_v','kappa_positive_per_v2','J_positive_a')
PHYSICAL_BOX=((1e-5,.2),(1e-4,1e3),(1e-3,1e3),(0.,math.log(10.)))
INITIAL_STEPS=(.5,.5,.5,.4)
COARSE=dict(name='coarse',rtol=1e-6,atol_u=1e-8,atol_x=1e-10,max_step=.05)
FINAL_PRECISIONS=tuple(dict(value) for value in artifacts.PRECISIONS)
N_ROUNDS=4
MIN_IMPROVEMENT=1e-4
SEARCH_LIMIT_S=900.
TOTAL_LIMIT_S=1200.
MAX_EVALUATIONS=35


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class BudgetExceeded(RuntimeError):
    pass


class NumericalBudget:
    """Outer loading/artifact I/O is excluded; numerical sections are counted."""
    def __init__(self):
        self.total=0.
        self.search=0.
        self.started=None
        self.kind=None

    def total_used(self):
        return self.total+(time.monotonic()-self.started if self.started is not None else 0.)

    def search_used(self):
        return self.search+(time.monotonic()-self.started if self.kind=='search' else 0.)

    def can_start(self,kind):
        return self.total_used()<TOTAL_LIMIT_S and (kind!='search' or self.search_used()<SEARCH_LIMIT_S)

    def check(self):
        if self.total_used()>=TOTAL_LIMIT_S:
            raise BudgetExceeded('Predeclared 1200 s total numerical budget exhausted')
        if self.kind=='search' and self.search_used()>=SEARCH_LIMIT_S:
            raise BudgetExceeded('Predeclared 900 s baseline/search budget exhausted; final reserve retained')

    @contextmanager
    def section(self,kind):
        if self.started is not None or kind not in ('search','final'):
            raise RuntimeError('Numerical budget sections must be explicit and nonnested')
        self.kind=kind
        self.started=time.monotonic()
        try:
            self.check()
            yield
            self.check()
        finally:
            elapsed=time.monotonic()-self.started
            self.total+=elapsed
            if kind=='search':
                self.search+=elapsed
            self.started=None
            self.kind=None


def _budget_cause(exc):
    seen=set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc,BudgetExceeded):
            return True
        seen.add(id(exc))
        exc=exc.__cause__ if exc.__cause__ is not None else exc.__context__
    return False


def _verify_freeze(path):
    manifest=json.loads(path.read_text())
    files=manifest['files_sha256']
    required={str((NATIVE/name).resolve()):digest for name,digest in FROZEN_NATIVE.items()}
    required[str(Path(__file__).resolve())]=sha(Path(__file__))
    required[str(fa.CONTRACT_PATH.resolve())]=sha(fa.CONTRACT_PATH)
    required[str(fa.CONTRACT_SOURCE.resolve())]=sha(fa.CONTRACT_SOURCE)
    for filename,digest in required.items():
        if files.get(filename)!=digest:
            raise ValueError('Root freeze does not include the exact prepared source: '+filename)
    for filename,digest in files.items():
        if sha(filename)!=digest:
            raise ValueError('Root frozen source changed: '+filename)
    for module,name in ((tp,'training_problem_v2.py'),(fa,'feature_audit.py'),
                        (artifacts,'run_native_baseline_v2.py'),(tp.kernel,'coupled_forward_v2.py')):
        if Path(module.__file__).resolve()!=(NATIVE/name).resolve():
            raise ValueError('Unexpected imported module identity: '+name)
    return files


def _seed():
    payload=json.loads((NATIVE/'SEED_DERIVATION.json').read_text())
    fixed=payload['fixed_scalar_material']
    parameters={key:float(fixed[key]) for key in FROZEN_POSITIVE}
    parameters['b']=float(fixed['b'])
    parameters.update({key:float(payload['new_parameters'][key]) for key in COORDINATES[:3]})
    names={'bare_cycles1_to3':'bare1','bare_cycles5_to15':'bare5','series_cycles4_to35':'series4'}
    entries={names[row['segment']]:(float(row['initial_q']),float(row['initial_a_s']))
             for row in payload['segment_entries']}
    if parameters!=artifacts.PARAMETERS or entries!=artifacts.ENTRIES:
        raise ValueError('Frozen seed and prior baseline declarations differ')
    _validate_parameters(parameters,parameters)
    return parameters,entries


def _validate_parameters(parameters,seed):
    if set(parameters)!=set(seed) or any(parameters[key]!=seed[key] for key in FROZEN_POSITIVE):
        raise ValueError('A candidate changed a frozen positive-state parameter')
    for name,(low,high) in zip(COORDINATES,PHYSICAL_BOX):
        if not math.isfinite(parameters[name]) or not low<=parameters[name]<=high:
            raise ValueError('Candidate outside predeclared exploration box: '+name)


def _parameter_key(parameters):
    # Exact values, including all frozen scalars. No decimal rounding or
    # distance-based aliasing. Cache is used only at the common coarse precision.
    return tuple(float(parameters[name]) for name in FROZEN_POSITIVE+COORDINATES)


def _coordinate_values(parameters):
    return [math.log10(parameters[name]) if axis<3 else parameters[name]
            for axis,name in enumerate(COORDINATES)]


def _neighbor(center,axis,direction,step):
    coordinate=_coordinate_values(center)
    raw=coordinate[axis]+direction*step
    low,high=PHYSICAL_BOX[axis]
    lower,upper=(math.log10(low),math.log10(high)) if axis<3 else (low,high)
    bounded=min(upper,max(lower,raw))
    parameters=dict(center)
    if bounded==lower:
        parameters[COORDINATES[axis]]=low
    elif bounded==upper:
        parameters[COORDINATES[axis]]=high
    elif bounded!=coordinate[axis]:
        parameters[COORDINATES[axis]]=10.**bounded if axis<3 else bounded
    return parameters,dict(axis=axis,coordinate=COORDINATES[axis],direction=direction,
        center_coordinate=coordinate[axis],step=step,raw_trial_coordinate=raw,
        bounded_trial_coordinate=bounded,parameter_box_projection_used=bounded!=raw,
        physical_box=[low,high],state_clipping_used=False)


def verify_native(result,material,budget,*,alternate_roots):
    """All native states/ports/joins; final stages add two alternate roots/phase."""
    rows=[]
    roots=0
    A=material.parameters.a_max_s
    def check(name,passed,**values):
        rows.append(dict(name=name,passed=bool(passed),**values))
    check('complete_original_scope',result['summary'].get('n_completed_phases')==92
        and len(result['predictions'])==92 and len(result['residual'])==27023)
    for segment in result['segments']:
        budget.check()
        previous=(segment['initial_state_q'],segment['initial_state_a_s'])
        for phase in segment['result']['phases']:
            s=phase['summary']
            incoming=(s['initial_state_q'],s['initial_state_a_s'])
            check('exact_two_state_handoff',incoming==previous,segment=segment['segment_id'],
                  record=s['record_id'],phase=s['phase'])
            previous=(s['final_state_q'],s['final_state_a_s'])
    for key,phase in result['predictions'].items():
        budget.check()
        top,cid,name=key
        arrays=phase['arrays']
        q,u,a=arrays['state_q'],arrays['state_u'],arrays['state_a_s']
        valid=bool(np.all(np.isfinite(q+u+a)) and np.all((q>0)&(q<1)&(u>0)&(u<=80)&(a>=0)&(a<=A)))
        check('finite_state_domain_current_A',valid,key=key,a_max_s=A)
        if valid:
            lg=np.log(material.readout.gamma_c_ev/np.expm1(u))
            check('frozen_Gamma_table_domain',np.all(lg>=material.readout.lg[0]) and np.all(lg<=material.readout.lg[-1]),key=key)
            check('q_u_coordinate_identity',np.allclose(q,np.exp(-u),rtol=1e-13,atol=0.),key=key)
        scale=1e-18+1e-7*abs(arrays['current_a'])
        check('passivity',all(np.all(np.isfinite(arrays[k])) and np.all(arrays[k]*arrays['local_voltage_v']>=0)
              for k in ('current_a','soft_current_a','hard_current_a')),key=key)
        check('KCL',np.all(abs(arrays['KCL_residual_a'])<=scale),key=key)
        check('KVL',np.all(abs(arrays['KVL_residual_v'])<=1e-9),key=key)
        check('branch_sum',np.all(abs(arrays['current_a']-arrays['soft_current_a']-arrays['hard_current_a'])<=scale),key=key)
        check('same_a_in_readout',np.all(abs(arrays['hard_current_a']-a*arrays['local_voltage_v'])<=scale),key=key)
        if top=='1r' and name=='set':
            check('bare_SET_actual_1mA_compliance',np.all(abs(arrays['current_a'])<=.001+scale),key=key)
        if alternate_roots:
            circuit=tp.kernel.SeriesCircuit() if top=='1t1r' else tp.kernel.BareCircuit(.001 if name=='set' else None)
            voltage=arrays['command_voltage_v']
            apex=int(np.argmax(abs(voltage)))
            return_ids=np.arange(apex+1,len(voltage))
            return_index=int(return_ids[np.argmin(abs(abs(voltage[return_ids])-.2))])
            for j in (int(np.argmax(abs(arrays['current_a']))),return_index):
                budget.check()
                point=circuit.solve(float(voltage[j]),material,(float(q[j]),float(a[j])),reference=True)
                roots+=1
                di=abs(point['current_a']-arrays['current_a'][j])
                dv=max(abs(point[k]-arrays[k][j]) for k in ('local_voltage_v','transistor_voltage_v','compliance_voltage_v'))
                check('alternate_total_current_root',di<=scale[j] and dv<=1e-9,key=key,index=j,
                      delta_current_a=di,delta_voltage_v=dv)
    return dict(passed=all(row['passed'] for row in rows),checks=rows,
        alternate_root_checks=roots,current_candidate_a_max_s=A,
        scope='native conservation, domain and continuous handoffs; final alternate roots share frozen tables',
        independently_refined_precision_checked=False)


def _midpoint_states(result):
    rows=[]
    for key,phase in result['predictions'].items():
        arrays=phase['arrays']
        if not {'state_q','state_u','state_a_s','command_voltage_v','path_voltage_s'}<=set(arrays):
            continue
        path=arrays['path_voltage_s']
        j=int(np.argmin(abs(path-.5*path[-1])))
        apex=int(np.argmax(abs(arrays['command_voltage_v'])))
        rows.append(dict(key=key,selection='nearest native half command-path coordinate; first tie',
            midpoint_native_index=j,midpoint_path_v=float(path[j]),
            midpoint_command_v=float(arrays['command_voltage_v'][j]),
            midpoint_q=float(arrays['state_q'][j]),midpoint_u=float(arrays['state_u'][j]),
            midpoint_a_s=float(arrays['state_a_s'][j]),apex_native_index=apex,
            apex_q=float(arrays['state_q'][apex]),apex_a_s=float(arrays['state_a_s'][apex]),
            final_q=float(arrays['state_q'][-1]),final_a_s=float(arrays['state_a_s'][-1])))
    return rows


def _gate_summary(gates):
    return dict(n_primary=gates['n_primary'],n_passed=gates['n_passed'],
        n_failed=len(gates['failed_ids']),failed_ids=gates['failed_ids'],
        count_is_not_accuracy_or_mechanism_quality=True,
        features={name:{key:value[key] for key in ('model_median','acceptance_interval','passed','undefined_records')}
                  for name,value in gates['features'].items()})


def compare_final(left,right,A):
    rows=[]
    for key,phase in left['predictions'].items():
        a,b=phase['arrays'],right['predictions'][key]['arrays']
        mask=a['original_scoring_mask']
        if not np.array_equal(mask,b['original_scoring_mask']) or not np.array_equal(a['command_voltage_v'],b['command_voltage_v']):
            raise ValueError('Final comparison changed original masks or native commands')
        logdiff=float(np.max(abs(np.log10(abs(a['current_a'][mask]))-np.log10(abs(b['current_a'][mask])))))
        udiff=float(np.max(abs(a['state_u']-b['state_u'])))
        xdiff=float(np.max(abs(a['state_a_s']-b['state_a_s']))/A)
        rows.append(dict(key=key,max_logI_decade=logdiff,max_abs_u=udiff,max_abs_x=xdiff,
            passed=logdiff<=3e-4 and udiff<=1e-3 and xdiff<=1e-5))
    return dict(passed=len(rows)==92 and all(row['passed'] for row in rows),phases=rows,
        current_candidate_a_max_s=A,max_logI_decade=max(row['max_logI_decade'] for row in rows),
        max_abs_u=max(row['max_abs_u'] for row in rows),max_abs_x=max(row['max_abs_x'] for row in rows))


def _pareto(cache):
    rows=[row for row in cache.values() if row.get('eligible_for_selection')]
    answer=[]
    for row in rows:
        cost,gates=row['cost'],row['gates']['n_failed']
        dominated=any(other['cost']<=cost and other['gates']['n_failed']<=gates
            and (other['cost']<cost or other['gates']['n_failed']<gates) for other in rows)
        if not dominated:
            answer.append(dict(request_id=row['request_id'],cost=cost,failed_gate_count=gates,
                parameters=row['parameters'],artifact_prefix=row['artifact_prefix']))
    return dict(candidates=answer,criteria=['original_cost','original_failed_gate_count'],
        diagnostic_only=True,used_for_selection=False,gate_count_is_not_accuracy=True,
        undefined_features_keep_original_failed_gate_status=True)


def run(freeze_path,output):
    frozen=_verify_freeze(freeze_path)
    parameters,entries=_seed()
    output.mkdir(exist_ok=False)
    cases=output/'cases'
    requests=output/'requests'
    cases.mkdir();requests.mkdir()
    budget=NumericalBudget()
    report=dict(status='preparing',source_sha256_before=frozen,freeze_sha256=sha(freeze_path),
        seed_parameters=parameters,fixed_segment_entries={k:list(v) for k,v in entries.items()},
        frozen_positive_parameters={key:parameters[key] for key in FROZEN_POSITIVE},
        tuned_coordinates=list(COORDINATES),coordinate_types=['log10','log10','log10','linear'],
        physical_exploration_box={name:list(bounds) for name,bounds in zip(COORDINATES,PHYSICAL_BOX)},
        box_is_exploration_assumption_not_measurement_limit=True,
        initial_steps=list(INITIAL_STEPS),minimum_cost_improvement=MIN_IMPROVEMENT,
        rounds_requested=N_ROUNDS,neighbors_per_round=8,maximum_candidate_requests=35,
        maximum_full_evaluations=35,maximum_phase_calls=3220,
        coarse_precision=COARSE,final_precisions=FINAL_PRECISIONS,
        search_budget_s=SEARCH_LIMIT_S,total_budget_s=TOTAL_LIMIT_S,final_reserved_s=300,
        artifact_IO_and_initial_loading_excluded=True,
        inherited_inline_source_guard_time_conservatively_included=True,
        no_holdout_access=True,no_new_initial_state_selection=True,no_periodic_shortcut=True,
        no_residual_penalty_for_failure=True,no_state_clipping=True,
        requests=[],rounds=[],full_evaluations_started=0,final_comparison=None)
    def persist():
        report.update(search_numerical_wall_s=budget.search_used(),total_numerical_wall_s=budget.total_used())
        artifacts.write(output/'RESULT.json',report)
    persist()
    cache={}
    accepted=None
    final_results={}
    final_rows=[]
    try:
        problem=tp.load_training_problem()
        contract=fa.load_feature_contract()
        report['data_contract']=problem.checkpoint_metadata()
        base=tp.kernel.load_material(tp.kernel.Parameters(**parameters,enable_a=True),
            candidate_source_id=str(NATIVE/'SEED_DERIVATION.json'),
            candidate_source_sha256=FROZEN_NATIVE['SEED_DERIVATION.json'])

        def request(request_id,label,candidate,precision,kind,context):
            _validate_parameters(candidate,parameters)
            row=dict(request_id=request_id,label=label,parameters=dict(candidate),precision=dict(precision),
                kind=kind,context=context,status='declared',eligible_for_selection=False,
                exact_parameter_tuple_hex=[float(value).hex() for value in _parameter_key(candidate)])
            report['requests'].append(row)
            path=requests/f'{request_id:03d}_{label}.json'
            artifacts.write(path,row)
            row['declaration_path']=str(path)
            row['declaration_sha256']=sha(path)
            key=_parameter_key(candidate)
            if kind=='search' and key in cache:
                previous=cache[key]
                row.update(status='exact_tuple_cache_hit',cache_source_request_id=previous['request_id'],
                    eligible_for_selection=previous['eligible_for_selection'],
                    cost=previous.get('cost'),gates=previous.get('gates'),
                    original_block_costs=previous.get('original_block_costs'),
                    weighted_phase_mse=previous.get('weighted_phase_mse'),
                    artifact_prefix=previous.get('artifact_prefix'),model_evaluation_performed=False,
                    cached_source_status=previous['status'],budget_exhausted=False)
                artifacts.write(requests/f'{request_id:03d}_{label}_result.json',row)
                persist()
                print(json.dumps(dict(cache=request_id,source=previous['request_id'])),flush=True)
                return row,None
            if not budget.can_start(kind):
                row.update(status='not_run_budget',budget_exhausted=True,model_evaluation_performed=False)
                artifacts.write(requests/f'{request_id:03d}_{label}_result.json',row)
                persist()
                return row,None
            if report['full_evaluations_started']>=MAX_EVALUATIONS:
                raise RuntimeError('Predeclared maximum full evaluation count exceeded')
            material=replace(base,parameters=tp.kernel.Parameters(**candidate,enable_a=True),
                candidate_source_id=str(path),candidate_source_sha256=row['declaration_sha256'])
            result=gates=verification=midpoints=None
            prefix=f'{request_id:03d}_{label}'
            row.update(status='running',artifact_prefix=str(cases/prefix),model_evaluation_performed=False)
            persist()
            print(json.dumps(dict(start=request_id,label=label,parameters=candidate)),flush=True)
            before_total,before_search=budget.total_used(),budget.search_used()
            try:
                with budget.section(kind):
                    row['model_evaluation_performed']=True
                    report['full_evaluations_started']+=1
                    result=problem.evaluate(material,entries,tp.kernel.Precision(**precision),
                                            'Radau',budget.check,collect_rhs=False)
                    gates=fa.evaluate_features(problem,result['predictions'],contract)
                    verification=verify_native(result,material,budget,alternate_roots=kind=='final')
                    midpoints=_midpoint_states(result)
                row.update(status='complete_valid' if verification['passed'] else 'complete_failed_numerical_checks',
                    eligible_for_selection=bool(verification['passed']),budget_exhausted=False)
            except Exception as exc:
                if hasattr(exc,'result'):
                    result=exc.result
                exhausted=_budget_cause(exc) or not budget.can_start(kind)
                row.update(status='retained_failure',eligible_for_selection=False,
                    failure_type=type(exc).__name__,failure=str(exc),traceback=traceback.format_exc(),
                    budget_exhausted=bool(exhausted))
            row.update(numerical_wall_s=budget.total_used()-before_total,
                       search_numerical_wall_s=budget.search_used()-before_search)
            if result is not None:
                artifacts.save_case(cases,prefix,result,gates,verification)
                row['summary']=result['summary']
                row['completed_phase_count']=result['summary'].get('n_completed_phases',0)
                if result['score'] is not None:
                    row['cost']=result['score']['verified_cost']
                    row['original_block_costs']={name:.5*value['weighted_total_mse']
                                               for name,value in result['score']['blocks'].items()}
                    row['weighted_phase_mse']={name:value['weighted_total_mse']
                                              for name,value in result['score']['blocks'].items()}
                    row['plain_allpoint_RMSE_decade']=result['score']['plain_allpoint_RMSE_decade']
                if gates is not None:
                    row['gates']=_gate_summary(gates)
                if verification is not None:
                    row['numerical_checks_passed']=verification['passed']
                    row['alternate_root_checks']=verification['alternate_root_checks']
                if midpoints is None:
                    midpoints=_midpoint_states(result)
                midpoint_path=cases/(prefix+'_midpoint_states.json')
                artifacts.write(midpoint_path,dict(native_midpoint_states=midpoints))
                row['midpoint_state_path']=str(midpoint_path)
            if row['eligible_for_selection'] and ('cost' not in row or 'gates' not in row
                    or not math.isfinite(row['cost']) or row['gates']['n_primary']!=20):
                raise RuntimeError('Eligible full candidate lacks original cost/all twenty feature gates')
            if kind=='search':
                cache[key]=row
            artifacts.write(requests/f'{request_id:03d}_{label}_result.json',row)
            persist()
            print(json.dumps(dict(done=request_id,status=row['status'],cost=row.get('cost'),
                                  failed_gates=row.get('gates',{}).get('n_failed'))),flush=True)
            return row,result if kind=='final' else None

        baseline,_=request(1,'baseline_coarse',parameters,COARSE,'search',dict(role='original enabled seed'))
        if not baseline['eligible_for_selection']:
            report['status']='stopped_without_valid_coarse_seed'
        else:
            accepted=baseline
            steps=list(INITIAL_STEPS)
            for round_index in range(N_ROUNDS):
                if not budget.can_start('search'):
                    report['search_stop']='900_second_search_limit_before_next_round'
                    break
                center=accepted
                center_parameters=dict(center['parameters'])
                planned=[]
                for axis in range(4):
                    for direction in (-1,1):
                        candidate,detail=_neighbor(center_parameters,axis,direction,steps[axis])
                        _validate_parameters(candidate,parameters)
                        request_id=2+8*round_index+len(planned)
                        planned.append(dict(request_id=request_id,parameters=candidate,detail=detail))
                round_row=dict(round=round_index+1,status='running',center_request_id=center['request_id'],
                    fixed_center_parameters=center_parameters,center_cost=center['cost'],steps=list(steps),
                    planned_neighbors=planned,request_ids=[],accepted_request_id=None)
                report['rounds'].append(round_row)
                persist()
                candidates=[]
                aborted=False
                for plan in planned:
                    detail=plan['detail']
                    label=f"round{round_index+1:02d}_axis{detail['axis']}_{'minus' if detail['direction']<0 else 'plus'}"
                    row,_=request(plan['request_id'],label,plan['parameters'],COARSE,'search',
                                  dict(round=round_index+1,center_request_id=center['request_id'],**detail))
                    candidates.append(row)
                    round_row['request_ids'].append(row['request_id'])
                    if row.get('budget_exhausted'):
                        aborted=True
                        break
                if aborted or len(candidates)!=8:
                    round_row.update(status='aborted_incomplete_round_no_center_acceptance',
                        accepted_request_id=None,center_after_request_id=center['request_id'],
                        not_run_request_ids=[p['request_id'] for p in planned if p['request_id'] not in round_row['request_ids']],
                        partial_neighbor_improvements_not_accepted=True)
                    report['search_stop']='900_second_limit_inside_round'
                    persist()
                    break
                eligible=[row for row in candidates if row['eligible_for_selection']]
                best=min(eligible,key=lambda row:(row['cost'],row['request_id'])) if eligible else None
                improvement=center['cost']-best['cost'] if best is not None else None
                improved=best is not None and improvement>=MIN_IMPROVEMENT
                if improved:
                    accepted=best
                else:
                    steps=[step/2. for step in steps]
                round_row.update(status='completed_all_eight_neighbor_requests',
                    n_eligible_neighbors=len(eligible),best_neighbor_request_id=best['request_id'] if best else None,
                    candidate_cost_improvement=improvement,improvement_accepted=bool(improved),
                    accepted_request_id=accepted['request_id'] if improved else None,
                    center_after_request_id=accepted['request_id'],steps_for_next_round=list(steps))
                report['pareto_diagnostic']=_pareto(cache)
                persist()
                print(json.dumps(dict(round=round_index+1,accepted_center=accepted['request_id'],
                                      cost=accepted['cost'],improved=bool(improved))),flush=True)
            report['last_accepted_center']=dict(request_id=accepted['request_id'],
                parameters=accepted['parameters'],coarse_cost=accepted['cost'],gates=accepted['gates'])
            report['pareto_diagnostic']=_pareto(cache)
            for number,precision in enumerate(FINAL_PRECISIONS,start=34):
                row,result=request(number,'best_'+precision['name'],accepted['parameters'],precision,'final',
                    dict(role='same last accepted center, independent requested precision',
                         selected_coarse_request_id=accepted['request_id'],coarse_cache_not_used=True))
                final_rows.append(row)
                if row['eligible_for_selection'] and result is not None:
                    final_results[precision['name']]=result
            if len(final_results)==2 and budget.can_start('final'):
                with budget.section('final'):
                    report['final_comparison']=compare_final(final_results['nominal'],final_results['refined'],
                                                            accepted['parameters']['a_max_s'])
            report['status']='bounded_search_and_final_checks_finished'
    except Exception as exc:
        report.update(status='stopped_with_retained_failure',failure_type=type(exc).__name__,
                      failure=str(exc),traceback=traceback.format_exc())
    finally:
        report['pareto_diagnostic']=_pareto(cache)
        report['source_sha256_after']={path:sha(path) for path in frozen}
        report['sources_unchanged']=report['source_sha256_after']==frozen
        report['n_completed_rounds']=sum(row['status']=='completed_all_eight_neighbor_requests' for row in report['rounds'])
        report['all_four_rounds_completed']=report['n_completed_rounds']==N_ROUNDS
        report['n_actual_model_evaluations']=sum(row.get('model_evaluation_performed',False) for row in report['requests'])
        report['n_completed_actual_phases']=sum(row.get('completed_phase_count',0) for row in report['requests']
                                               if row.get('model_evaluation_performed'))
        report['max_evaluation_contract_respected']=report['n_actual_model_evaluations']<=35
        report['final_two_precisions_valid']=(len(final_rows)==2 and all(row.get('eligible_for_selection') for row in final_rows)
            and bool(report.get('final_comparison',{})) and report['final_comparison']['passed'])
        report['numerical_final_candidate_verified']=(report['sources_unchanged'] and report['final_two_precisions_valid'])
        report['optimization_converged']=False
        report['full_parameter_optimization_performed']=False
        report['unified_physical_mechanism_established']=False
        report['feature_gate_count_is_accuracy']=False
        report['reserved_final_budget_was_automatically_extended']=False
        persist()
        print(json.dumps({key:report[key] for key in ('status','n_completed_rounds','n_actual_model_evaluations',
            'n_completed_actual_phases','numerical_final_candidate_verified','total_numerical_wall_s')}),flush=True)
    return 0 if report['all_four_rounds_completed'] and report['numerical_final_candidate_verified'] else 2


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',type=Path,default=HERE/'SOURCE_FREEZE.json')
    parser.add_argument('--output',type=Path,default=HERE/'calibration_run')
    args=parser.parse_args(argv)
    return run(args.freeze.resolve(),args.output.resolve())


if __name__=='__main__':
    raise SystemExit(main())
