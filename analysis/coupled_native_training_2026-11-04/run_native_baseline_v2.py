"""V2 endpoint-fixed four-case original-training baseline; no fit or implicit retry."""
from pathlib import Path
from dataclasses import replace
from contextlib import contextmanager
import hashlib
import json
import math
import time
import traceback
import numpy as np
import training_problem_v2 as tp
import feature_audit as fa

HERE=Path(__file__).resolve().parent
PARAMETERS=dict(lambda_per_v=.4470398752554136,J_positive_a=.0037505935471625162,
    kappa_positive_per_v2=57.67798793605963,b=.769297378470313,
    a_max_s=.013206747378601957,kappa_a_per_v2=3.60746202417315,rho=.3361287809489076)
ENTRIES={'bare1':(4.298219923945171e-5,0.),'bare5':(.00025057076016083026,0.),
         'series4':(7.755279151592979e-5,0.)}
PRECISIONS=(dict(name='nominal',rtol=1e-7,atol_u=1e-9,atol_x=1e-11,max_step=.02),
            dict(name='refined',rtol=1e-9,atol_u=1e-11,atol_x=1e-13,max_step=.01))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not math.isfinite(x):return str(x)
    return x

def write(p,x):p.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n')

class BudgetExceeded(RuntimeError):pass

class Budget:
    def __init__(self):self.elapsed=0.;self.start=None
    def used(self):return self.elapsed+(time.monotonic()-self.start if self.start is not None else 0.)
    def check(self):
        if self.used()>900:raise BudgetExceeded('Predeclared 900 s numerical budget exhausted')
    @contextmanager
    def section(self):
        assert self.start is None
        self.start=time.monotonic()
        try:self.check();yield;self.check()
        finally:self.elapsed+=time.monotonic()-self.start;self.start=None

def save_case(out,label,result,gates=None,verification=None):
    arrays={'weighted_residual':result['residual'],'raw_residual':result['raw_residual']}
    phases=[]
    for (top,cid,phase),value in result['predictions'].items():
        prefix=f'{top}_cycle_{cid}_{phase}__'
        arrays.update({prefix+k:v for k,v in value['arrays'].items()})
        arrays[prefix+'residual']=value['residual']
        phases.append(dict(topology=top,record_id=cid,phase=phase,summary=value['summary']))
    segments=[]
    for segment in result['segments']:
        segments.append({k:v for k,v in segment.items() if k!='result'})
        segments[-1]['history_summary']=segment.get('result',{}).get('summary')
    np.savez_compressed(out/(label+'.npz'),**arrays)
    write(out/(label+'.json'),dict(summary=result['summary'],score=result['score'],
        segments=segments,phases=phases,features=gates,numerical_verification=verification))

def verify_case(result,material,budget):
    rows=[];roots=0
    def check(name,passed,**values):rows.append(dict(name=name,passed=bool(passed),**values))
    for segment in result['segments']:
        previous=(segment['initial_state_q'],segment['initial_state_a_s'])
        history=segment['result']
        for phase in history['phases']:
            s=phase['summary'];incoming=(s['initial_state_q'],s['initial_state_a_s'])
            check('exact_state_handoff',incoming==previous,segment=segment['segment_id'],record=s['record_id'],phase=s['phase'])
            previous=(s['final_state_q'],s['final_state_a_s'])
    for key,phase in result['predictions'].items():
        budget.check();top,cid,name=key;a=phase['arrays']
        q,u,h=a['state_q'],a['state_u'],a['state_a_s']
        check('finite_state_domain',np.all(np.isfinite(q+u+h)) and np.all((q>0)&(q<1)&(h>=0)&(h<=PARAMETERS['a_max_s'])),key=key)
        check('passivity',all(np.all(a[k]*a['local_voltage_v']>=0) for k in ('current_a','soft_current_a','hard_current_a')),key=key)
        scale=1e-18+1e-7*abs(a['current_a'])
        check('KCL',np.all(abs(a['KCL_residual_a'])<=scale),key=key)
        check('KVL',np.max(abs(a['KVL_residual_v']))<=1e-9,key=key)
        check('branch_sum',np.all(abs(a['current_a']-a['soft_current_a']-a['hard_current_a'])<=scale),key=key)
        circuit=tp.kernel.SeriesCircuit() if top=='1t1r' else tp.kernel.BareCircuit(.001 if name=='set' else None)
        if circuit.compliance_a is not None:check('actual_compliance',np.all(abs(a['current_a'])<=.001+scale),key=key)
        if not material.parameters.enable_a:check('disabled_a_zero',np.all(h==0),key=key)
        v=a['command_voltage_v'];turn=int(np.argmax(abs(v)));return_ids=np.arange(turn+1,len(v))
        jreturn=int(return_ids[np.argmin(abs(abs(v[return_ids])-.2))])
        for j in (int(np.argmax(abs(a['current_a']))),jreturn):
            budget.check()
            point=circuit.solve(float(v[j]),material,(float(q[j]),float(h[j])),reference=True);roots+=1
            di=abs(point['current_a']-a['current_a'][j]);dv=max(abs(point[k]-a[k][j]) for k in ('local_voltage_v','transistor_voltage_v','compliance_voltage_v'))
            check('alternate_root',di<=scale[j] and dv<=1e-9,key=key,index=j,delta_current_a=di,delta_voltage_v=dv)
    return dict(passed=all(r['passed'] for r in rows),checks=rows,alternate_root_checks=roots,
        scope='native conservation and state joins; alternate root algorithm shares frozen material tables')

def compare(left,right):
    rows=[]
    for key,phase in left['predictions'].items():
        a=phase['arrays'];b=right['predictions'][key]['arrays'];mask=a['original_scoring_mask']
        assert np.array_equal(mask,b['original_scoring_mask']) and np.array_equal(a['command_voltage_v'],b['command_voltage_v'])
        logdiff=float(np.max(abs(np.log10(abs(a['current_a'][mask]))-np.log10(abs(b['current_a'][mask])))))
        udiff=float(np.max(abs(a['state_u']-b['state_u'])))
        xdiff=float(np.max(abs(a['state_a_s']-b['state_a_s']))/PARAMETERS['a_max_s'])
        rows.append(dict(key=key,max_logI_decade=logdiff,max_abs_u=udiff,max_abs_x=xdiff,
            passed=logdiff<=3e-4 and udiff<=1e-3 and xdiff<=1e-5))
    return dict(passed=all(r['passed'] for r in rows),phases=rows,
        max_logI_decade=max(r['max_logI_decade'] for r in rows),
        max_abs_u=max(r['max_abs_u'] for r in rows),max_abs_x=max(r['max_abs_x'] for r in rows))

def main():
    freeze=json.loads((HERE/'SOURCE_FREEZE_V2.json').read_text())
    for path,digest in freeze['files_sha256'].items():
        if sha(path)!=digest:raise RuntimeError('Prelaunch frozen source mismatch: '+path)
    out=HERE/'baseline_run_v2';out.mkdir(exist_ok=False)
    budget=Budget();report=dict(status='preparing',parameters=PARAMETERS,initial_states=ENTRIES,
        no_optimization=True,no_holdout_access=True,maximum_cases=4,maximum_phases=368,
        cases=[],comparisons={},numerical_budget_s=900,source_sha256_before=freeze['files_sha256'])
    results={}
    write(out/'RESULT.json',report)
    try:
        problem=tp.load_training_problem()
        report['data_contract']=problem.checkpoint_metadata()
        with budget.section():
            report['score_arithmetic_check']=tp.compare_legacy_scorer(problem)
            report['feature_arithmetic_check']=fa.audit_observed_contract(problem)
        assert report['score_arithmetic_check']['passed'] and report['feature_arithmetic_check']['passed']
        source=HERE/'SEED_DERIVATION.json'
        enabled=tp.kernel.load_material(tp.kernel.Parameters(**PARAMETERS,enable_a=True),
            candidate_source_id=str(source),candidate_source_sha256=sha(source))
        disabled=replace(enabled,parameters=tp.kernel.Parameters(**PARAMETERS,enable_a=False))
        for family,material in (('disabled',disabled),('enabled',enabled)):
            for precision in PRECISIONS:
                label=family+'_'+precision['name'];report['active_case']=label
                write(out/'RESULT.json',report);print(json.dumps({'start':label}),flush=True)
                row=dict(label=label,status='running');report['cases'].append(row)
                result=None
                try:
                    with budget.section():
                        result=problem.evaluate(material,ENTRIES,tp.kernel.Precision(**precision),'Radau',budget.check,collect_rhs=False)
                        gates=fa.evaluate_features(problem,result['predictions'])
                        verification=verify_case(result,material,budget)
                    save_case(out,label,result,gates,verification);results[label]=result
                    row.update(status='complete',summary=result['summary'],score=result['score'],
                        n_feature_gates_passed=gates['n_passed'],failed_feature_ids=gates['failed_ids'],
                        numerical_verification_passed=verification['passed'],alternate_root_checks=verification['alternate_root_checks'])
                    print(json.dumps({'done':label,'cost':result['score']['verified_cost'],'gates':gates['n_passed']}),flush=True)
                except Exception as exc:
                    row.update(status='retained_failure',failure_type=type(exc).__name__,failure=str(exc),traceback=traceback.format_exc())
                    if hasattr(exc,'result'):result=exc.result
                    if result is not None:save_case(out,label+'_partial',result);row['summary']=result['summary']
                    print(json.dumps({'failure':label,'type':type(exc).__name__,'error':str(exc)}),flush=True)
                    if budget.used()>900:raise BudgetExceeded('Budget exhausted; remaining declared cases not run') from exc
                write(out/'RESULT.json',report)
            if all(family+'_'+p['name'] in results for p in PRECISIONS):
                with budget.section():report['comparisons'][family]=compare(results[family+'_nominal'],results[family+'_refined'])
        report['status']='completed_declared_cases'
    except Exception as exc:
        report.update(status='stopped_with_failure',failure_type=type(exc).__name__,failure=str(exc),traceback=traceback.format_exc())
    finally:
        report['numerical_wall_s']=budget.used()
        report['source_sha256_after']={path:sha(path) for path in report['source_sha256_before']}
        report['sources_unchanged']=report['source_sha256_after']==report['source_sha256_before']
        report['n_completed_cases']=sum(c['status']=='complete' for c in report['cases'])
        report['n_completed_phases']=sum(c.get('summary',{}).get('n_completed_phases',0) for c in report['cases'])
        report['full_scope_completed']=report['n_completed_cases']==4 and report['n_completed_phases']==368
        report['numerically_verified']=(report['full_scope_completed'] and report['sources_unchanged']
            and all(c.get('numerical_verification_passed',False) for c in report['cases'])
            and len(report['comparisons'])==2 and all(c['passed'] for c in report['comparisons'].values()))
        report['unified_physical_model_achieved']=False
        report['goal_status']='active'
        write(out/'RESULT.json',report)
        print(json.dumps({k:report[k] for k in ('status','n_completed_cases','n_completed_phases','numerically_verified','numerical_wall_s')}),flush=True)
    return 0 if report['numerically_verified'] else 2

if __name__=='__main__':raise SystemExit(main())
