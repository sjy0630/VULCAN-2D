"""Prepared finite conditional empirical-integral diagnostic; explicit run only.

No ODE, nonlinear forward fit, b scan, state anchoring, extrapolation, point
deletion, or automatic retry. All four interpolation/quadrature variants remain.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from io import BytesIO
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
import numpy as np
from scipy.interpolate import PchipInterpolator

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent
STATIC = ANALYSIS/'zero_crossing_compatibility_2026-11-04'
SIGNED = ANALYSIS/'signed_soft_history_2026-11-04'
for directory in (STATIC, SIGNED):
    if str(directory) not in sys.path:sys.path.insert(0,str(directory))
from minimal_static_loader import load_minimal
from check_compatibility import invert_readpoint

PHASES=('set4','reset4','set5')
INTERPOLANTS=('linear','pchip')
ORDERS=(8,16)
WINDOW_SUFFIXES=('outward_0p2_to_0p8','outward_0p8_to_1p6',
                 'return_1p6_to_0p8','return_0p8_to_0p2')
RESULT=HERE/'empirical_integral_diagnostic.json'
PROGRESS=HERE/'empirical_integral_diagnostic.progress.json'
ARRAYS=HERE/'empirical_integral_arrays.npz'
CACHE_SHA='a08b1e322fae5316d9caf52cf9a4d3a000f46c11fd2716e4cac8591f1f599f52'
LIMITS={'KCL_a':1e-13,'KVL_v':1e-12,'root_log_current_decade':5e-7,
        'root_local_voltage_v':1e-9,'severe_column_normalized_condition':1e8}


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def digest(payload):return hashlib.sha256(payload).hexdigest()


def failure_evidence(exc):
    """Snapshot existing traceback locals only; never rerun a failed operation."""
    scalar_keys={'target','command','observed','sign','drop','local','y','j','fraction',
                 'lg_direct','low','high','rlow','rhigh','middle','residual','u','u_bisect','log_error'}
    frames=[];raw_frames=[];tb=exc.__traceback__
    while tb is not None:
        frame=tb.tb_frame;local=frame.f_locals
        row={'file':frame.f_code.co_filename,'function':frame.f_code.co_name,'line':tb.tb_lineno}
        if Path(frame.f_code.co_filename).name=='check_compatibility.py':
            snapshot={}
            for key in scalar_keys:
                value=local.get(key)
                if isinstance(value,(int,float,np.integer,np.floating)):
                    snapshot[key]={'value':float(value) if math.isfinite(float(value)) else None,
                                   'finite':math.isfinite(float(value))}
            for key in ('x','y','lg','lv','log_current_slice'):
                value=local.get(key)
                if isinstance(value,np.ndarray):
                    flat=value.ravel();ends=[float(flat[0]),float(flat[-1])] if flat.size else []
                    snapshot[key+'_array']={'shape':list(value.shape),'size':int(value.size),
                        'first_last':clean(ends),'first_last_finite':[math.isfinite(v) for v in ends]}
            for key in ('tj','gj'):
                value=local.get(key)
                if isinstance(value,(list,tuple)) and len(value)<=2:snapshot[key]=clean(value)
            row['existing_local_snapshot']=snapshot
        frames.append(row);raw_frames.append((frame.f_code.co_name,local));tb=tb.tb_next
    formatted=traceback.extract_tb(exc.__traceback__)
    for row,entry in zip(frames,formatted):row['source_line']=entry.line
    stage='unknown_from_available_evidence'
    caller=next((loc for name,loc in reversed(raw_frames) if name=='invert_readpoint'),None)
    if isinstance(exc,BudgetStop):stage='numerical_time_budget'
    elif any(name=='inverse_piecewise' for name,_ in raw_frames) and caller is not None:
        stage='transistor_piecewise_inverse' if 'drop' not in caller else 'material_log_gamma_piecewise_inverse'
    elif caller is not None:
        last_source=frames[-1].get('source_line') or ''
        if 'kernel.vmin <= local <= abs(command)' in last_source:stage='local_voltage_domain_or_passivity_precheck'
        elif 'lv[0] <= y <= lv[-1]' in last_source:stage='local_log_voltage_table_precheck'
        elif 'abs(u - u_bisect)' in last_source:stage='direct_vs_bisection_state_check'
        elif 'log_error <=' in last_source:stage='circuit_current_backcheck'
        elif 'KCL_residual' in last_source or 'KVL_residual' in last_source:stage='circuit_conservation_backcheck'
        else:stage='inside_frozen_readpoint_inverse_more_specific_cause_unresolved'
    message=str(exc)
    cause=('table_target_outside_domain' if 'Target outside frozen table' in message else
           'nonunique_table_plateau_target' if 'Nonunique plateau target' in message else
           'numerical_time_budget' if isinstance(exc,BudgetStop) else 'see_saved_traceback_without_physical_exclusion')
    return {'identified_stage':stage,'cause_class':cause,'traceback':frames,
            'recomputed_for_classification':False,'physical_mechanism_rejection':False}


class Writer:
    def __init__(self):self.io_s=0.;self.catalog={}
    def write(self,path,producer,replace=False,catalog=True):
        start=time.monotonic()
        try:
            payload=producer()
            with Path(path).open('wb' if replace else 'xb') as handle:handle.write(payload)
            record={'path':str(path),'sha256':digest(payload),'bytes':len(payload)}
            if catalog:self.catalog[str(path)]=record
            return record
        finally:self.io_s+=time.monotonic()-start
    def json(self,path,data,replace=False,catalog=True):
        snapshot=deepcopy(data)
        return self.write(path,lambda:(json.dumps(clean(snapshot),indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode(),replace,catalog)
    def npz(self,path,arrays,replace=False):
        def produce():
            stream=BytesIO();np.savez_compressed(stream,**arrays);return stream.getvalue()
        return self.write(path,produce,replace)
    def progress(self,message):
        start=time.monotonic()
        try:print(message,flush=True)
        finally:self.io_s+=time.monotonic()-start


class BudgetStop(TimeoutError):pass


class Budgets:
    limits={'inverse':60.,'quadrature':120.,'algebra':10.}
    def __init__(self):
        self.used={k:0. for k in self.limits};self.calls=[];self.stopped={};self.active=None
    def check(self):
        if self.active is None:raise RuntimeError('Numerical check outside timed item')
        block=self.active['block'];elapsed=time.monotonic()-self.active['start']
        if block in self.stopped:raise BudgetStop(self.stopped[block])
        if elapsed>=self.active['item_limit']:
            raise BudgetStop('Per-item numerical time limit exhausted')
        if self.used[block]+elapsed>=self.limits[block]:
            self.stopped[block]='Block numerical budget exhausted'
            raise BudgetStop(self.stopped[block])
    def run(self,block,label,function,item_limit):
        if self.active is not None:raise RuntimeError('Nested numerical timing prohibited')
        if block in self.stopped or self.used[block]>=self.limits[block]:
            self.stopped.setdefault(block,'Block numerical budget exhausted')
            raise BudgetStop(self.stopped[block])
        self.active={'block':block,'label':label,'start':time.monotonic(),'item_limit':item_limit}
        status='completed'
        try:
            self.check();answer=function(self.check);self.check();return answer
        except Exception:
            status='failed_or_timed_out';raise
        finally:
            elapsed=time.monotonic()-self.active['start'];self.used[block]+=elapsed
            self.calls.append({'block':block,'label':label,'elapsed_s':elapsed,
                               'item_limit_s':item_limit,'status':status})
            if self.used[block]>=self.limits[block]:self.stopped.setdefault(block,'Block numerical budget exhausted')
            self.active=None
    def snapshot(self):return {'limits_s':dict(self.limits),'used_s':dict(self.used),
        'total_numerical_s':sum(self.used.values()),'stopped_blocks':dict(self.stopped),'calls':deepcopy(self.calls)}


def validate_point(kernel,command,u,point):
    kernel.gamma(float(u))
    values=[point[k] for k in ('current_a','local_voltage_v','transistor_voltage_v','KCL_residual_a','KVL_residual_v')]
    if not all(math.isfinite(float(x)) for x in values):raise ValueError('Nonfinite static circuit point')
    current,local,transistor,kcl,kvl=values
    if command==0.:
        if (current,local,transistor)!=(0.,0.,0.):raise ValueError('Nonzero port at zero command')
    else:
        if current*command<=0. or local*command<=0. or transistor*command<0.:raise ValueError('Static polarity failure')
        top=kernel.base.tx[-1] if command>0. else kernel.negative_tx[-1]
        if not kernel.vmin<=abs(local)<=abs(command)+1e-12 or abs(transistor)>top+1e-12:
            raise ValueError('Static local or transistor voltage outside finite domains')
    if current*local<0. or current*transistor<0.:raise ValueError('Static passivity failure')
    if abs(kcl)>LIMITS['KCL_a'] or abs(kvl)>LIMITS['KVL_v'] or abs(command-local-transistor)>LIMITS['KVL_v']:
        raise ValueError('Static KCL/KVL failure')


class Ledger:
    """Count readpoint inverses, material evaluations and static solve invocations.

    A root call means one circuit.solve invocation, not an internal function
    evaluation. Returned/validated counts remain distinct from item success.
    """
    root_columns=['context_id','node_index','mode_code','command_v','state_u','returned',
                  'validated','current_a','local_voltage_v','transistor_voltage_v','KCL_a','KVL_v']
    material_columns=['context_id','call_index','kind_code','local_v','log_gamma','sign','returned','current_a']
    def __init__(self):self.counts=Counter();self.contexts=[];self.roots=[];self.material=[];self.errors=[]
    def context(self,kind,label,**fields):
        row={'context_id':len(self.contexts),'kind':kind,'label':label,**fields}
        self.contexts.append(row);return row['context_id']
    def snapshot(self):return {'counts':dict(self.counts),'contexts':deepcopy(self.contexts),
        'root_trace_columns':self.root_columns,'root_mode_codes':{'fast':0,'strict_bisection42':1},
        'material_trace_columns':self.material_columns,
        'material_kind_codes':{'initial_bracket':0,'bisection_iteration':1,'final_bracket_check':2},
        'errors':deepcopy(self.errors),
        'count_scope':'Static root counts are circuit.solve invocations; internal circuit root residual evaluations are not new root solves. Material trace covers independent inverse material calls outside those roots.'}
    def arrays(self):return {'static_root_trace':np.asarray(self.roots,dtype=float).reshape((-1,len(self.root_columns))),
                            'inverse_material_trace':np.asarray(self.material,dtype=float).reshape((-1,len(self.material_columns)))}


class CountedCircuit:
    def __init__(self,kernel,circuit,ledger,context,kind,mode,check,node=-1):
        self.kernel,self.circuit,self.ledger=kernel,circuit,ledger
        self.context,self.kind,self.mode,self.check,self.node=context,kind,mode,check,node
    def solve(self,command,u):
        self.check();ledger=self.ledger;key=self.kind+'_'+('fast' if self.mode=='fast' else 'strict42')
        ledger.counts['static_root_attempts']+=1;ledger.counts[key+'_attempts']+=1
        row=[self.context,self.node,0 if self.mode=='fast' else 1,float(command),float(u),0.,0.,np.nan,np.nan,np.nan,np.nan,np.nan]
        ledger.roots.append(row)
        try:
            point=self.circuit.solve(float(command),float(u))
            row[5]=1.;row[7:]=[float(point[k]) for k in ('current_a','local_voltage_v','transistor_voltage_v','KCL_residual_a','KVL_residual_v')]
            ledger.counts['static_root_returns']+=1;ledger.counts[key+'_returns']+=1
            validate_point(self.kernel,float(command),float(u),point)
            row[6]=1.;ledger.counts['static_root_validated']+=1;ledger.counts[key+'_validated']+=1
            self.check();return point
        except Exception as exc:
            ledger.errors.append({'kind':'static_root_call_exception','context_id':self.context,'node_index':self.node,
                                  'mode':self.mode,'returned':bool(row[5]),'validated':bool(row[6]),
                                  'error_type':type(exc).__name__,'error':str(exc)})
            if not row[5]:ledger.counts['static_root_failures_before_return']+=1
            elif not row[6]:ledger.counts['static_root_validation_failures']+=1
            raise


class InverseProxy:
    """Read-only delegation to frozen kernel; adds accounting, never changes values."""
    def __init__(self,kernel,ledger,context,check):
        self.core,self.ledger,self.context,self.check=kernel,ledger,context,check
        self.material_call=0
    def __getattr__(self,key):return getattr(self.core,key)
    def make_circuit(self,mode):
        return CountedCircuit(self.core,self.core.make_circuit(mode=mode),self.ledger,self.context,'inverse',mode,self.check)
    def _material_abs(self,local,log_gamma,sign):
        self.check();self.material_call+=1;number=self.material_call
        kind=0 if number<=2 else 1 if number<=50 else 2
        row=[self.context,number,kind,float(local),float(log_gamma),int(sign),0.,np.nan]
        self.ledger.material.append(row);self.ledger.counts['inverse_material_evaluations_attempted']+=1
        if kind==1:self.ledger.counts['inverse_material_bisection_iterations_attempted']+=1
        try:
            answer=self.core._material_abs(local,log_gamma,sign)
            row[6:]=[1.,float(answer)];self.ledger.counts['inverse_material_evaluations_returned']+=1
            if kind==1:self.ledger.counts['inverse_material_bisection_iterations_returned']+=1
            self.check();return answer
        except Exception as exc:
            self.ledger.errors.append({'kind':'inverse_material_call_exception','context_id':self.context,
                'call_index':number,'returned':bool(row[6]),'error_type':type(exc).__name__,'error':str(exc)})
            raise


def incident_intervals(grid,value):
    if not math.isfinite(value) or value<grid[0] or value>grid[-1]:
        raise ValueError('Sensitivity coordinate outside frozen grid')
    matches=np.flatnonzero(grid==value)
    if len(matches):
        j=int(matches[0]);intervals=[]
        if j>0:intervals.append((j-1,j))
        if j<len(grid)-1:intervals.append((j,j+1))
        return intervals,{'exact_knot':True,'knot_index':j,'domain_boundary':j in (0,len(grid)-1)}
    j=int(np.searchsorted(grid,value))
    return [(j-1,j)],{'exact_knot':False,'knot_index':None,'domain_boundary':False}


def inverse_sensitivity(kernel,answer):
    """Incident-cell derivative arithmetic only; never used to reject a point."""
    command=answer['command_voltage_v'];sign=1 if command>0. else -1
    local=answer['local_voltage_magnitude_v'];observed=answer['observed_current_magnitude_a'];u=answer['direct_u']
    g=answer['direct_log_gamma'];vlog=math.log(local)
    lg,lv=kernel.lg,kernel.lv
    gs,gmark=incident_intervals(lg,g);vs,vmark=incident_intervals(lv,vlog)
    tx,ti=(kernel.base.tx,kernel.base.ti) if sign>0 else (kernel.negative_tx,kernel.negative_ti)
    tj=answer['transistor_bracket_indices'];tmark={'exact_knot':tj[0]==tj[1],
        'knot_index':tj[0] if tj[0]==tj[1] else None,
        'domain_boundary':tj[0]==tj[1] and tj[0] in (0,len(tx)-1)}
    if tj[0]==tj[1]:
        t_segments=[];j=tj[0]
        if j>0:t_segments.append((j-1,j))
        if j<len(tx)-1:t_segments.append((j,j+1))
    else:t_segments=[tuple(tj)]
    spectrum=kernel.base.li if sign>0 else kernel.li_negative
    rows=[]
    for ga,gb in gs:
        for va,vb in vs:
            tg=(g-lg[ga])/(lg[gb]-lg[ga]);tv=(vlog-lv[va])/(lv[vb]-lv[va])
            Lg=((1-tv)*(spectrum[gb,va]-spectrum[ga,va])+tv*(spectrum[gb,vb]-spectrum[ga,vb]))/(lg[gb]-lg[ga])
            Lv=((1-tg)*(spectrum[ga,vb]-spectrum[ga,va])+tg*(spectrum[gb,vb]-spectrum[gb,va]))/(lv[vb]-lv[va])
            for ta,tb in t_segments:
                di=float(ti[tb]-ti[ta]);dv=float(tx[tb]-tx[ta]);slope=dv/di if di>0. else math.inf
                row={'gamma_cell':[ga,gb],'voltage_cell':[va,vb],'transistor_segment':[ta,tb],
                     'L_Gamma':float(Lg),'L_v':float(Lv),'dVT_dI_v_per_a':slope,'defined':False}
                factor=1.+float(Lv)*observed*slope/local
                if Lg>0. and math.isfinite(slope) and math.isfinite(factor) and factor>0.:
                    chi=math.log(10.)*(-math.expm1(-u))*factor/float(Lg)
                    if math.isfinite(chi):row.update(defined=True,chi_abs_du_per_decade=chi)
                if not row['defined']:row['reason']='Nonpositive/undefined incident slope or circuit inverse factor'
                rows.append(row)
    values=[r['chi_abs_du_per_decade'] for r in rows if r['defined']]
    return {'status':'all_incident_slopes_defined' if len(values)==len(rows) else 'some_incident_slopes_undefined',
        'gamma_marker':gmark,'local_voltage_marker':vmark,'transistor_marker':tmark,
        'incident_cells':rows,'finite_chi_min':min(values) if values else None,
        'finite_chi_max':max(values) if values else None,'all_incident_values_defined':len(values)==len(rows),
        'used_for_point_selection':False,'measurement_uncertainty_interval':False,
        'interpretation':'Local fixed-circuit inverse amplification per log10 current; no measurement-noise certificate.'}


def closure(delta,A,B,coeff):
    first,second=coeff[0]*A,coeff[1]*B;error=delta-(first-second)
    denominator=abs(delta)+abs(first)+abs(second)
    return {'delta_u':delta,'recovery_term':first,'capture_term':second,'predicted_delta_u':first-second,
            'E':error,'absolute_E':abs(error),'normalization_denominator':denominator,
            'denominator_zero':denominator==0.,'normalized_E':error/denominator if denominator!=0. else None}


def identify(rows,names,reference,ledger,check,result):
    check();result.update(window_ids=[r['window_id'] for r in rows],coefficient_names=names,available=False)
    if any(r.get('status')!='completed' for r in rows):
        result['status']='unattempted_missing_complete_fixed_window';return result
    ledger.counts['new_linear_identification_attempts']+=1;result['status']='identification_attempted'
    X=np.array([[r['A_v2'],-r['B_a_v2']] for r in rows]);y=np.array([r['delta_u'] for r in rows])
    result.update(raw_X_A_minus_B=X,y_delta_u=y,column_units=['V^2','A V^2'])
    norms=np.linalg.norm(X,axis=0)
    result['column_l2_norms']=norms
    if np.any(norms<=0.) or not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        result['status']='nonfinite_or_zero_design_column';return result
    normalized=X/norms[None,:];result['column_normalized_X']=normalized
    singular=np.linalg.svd(normalized,compute_uv=False)
    tolerance=max(normalized.shape)*np.finfo(float).eps*singular[0]
    rank=int(np.count_nonzero(singular>tolerance));cond=float(singular[0]/singular[-1]) if singular[-1]>0 else math.inf
    result.update(column_normalized_X=normalized,column_normalized_singular_values=singular,
        rank_tolerance=tolerance,numerical_rank=rank,column_normalized_condition_2=cond,
        fixed_old_reference_coefficients=reference,
        severely_ill_conditioned=cond>LIMITS['severe_column_normalized_condition'])
    result['fixed_old_reference_scaled_condition_2']=float(np.linalg.cond(X*np.asarray(reference)[None,:]))
    if rank!=2:result['status']='rank_deficient_no_unique_identification';return result
    check();ledger.counts['new_2x2_linear_solves_attempted']+=1
    try:
        coeff=np.linalg.solve(normalized,y)/norms
        ledger.counts['new_2x2_linear_solves_returned']+=1
        if not np.all(np.isfinite(coeff)):raise ValueError('Nonfinite linear coefficients')
        result.update(status='identified_diagnostic_only',available=True,
            coefficients=dict(zip(names,coeff)),all_coefficients_positive=bool(np.all(coeff>0.)),
            identification_window_linear_residual=X@coeff-y,
            exact_two_window_closure_is_not_evidence=True)
    except np.linalg.LinAlgError as exc:result.update(status='linear_solve_failure',error=str(exc))
    check();return result


def load_inputs():
    start=time.monotonic();sources={}
    freeze_path=HERE/'SOURCE_FREEZE.json'
    freeze_payload=freeze_path.read_bytes();freeze=json.loads(freeze_payload)
    frozen=freeze['source_sha256']
    required={str(HERE/name) for name in ('empirical_integral_diagnostic.py','EXPERIMENT_PLAN.md','WINDOW_DATA_AUDIT.json','windows_input.npz')}
    required.update(str(p) for p in (STATIC/'minimal_static_loader.py',STATIC/'check_compatibility.py',SIGNED/'signed_forward_kernel.py'))
    if not required.issubset(frozen):raise ValueError('SOURCE_FREEZE lacks required explicit inputs/sources')
    for path,expected in frozen.items():
        actual=digest(Path(path).read_bytes())
        if actual!=expected:raise ValueError('Frozen source mismatch: '+path)
        sources[path]=actual
    sources[str(freeze_path)]=digest(freeze_payload)
    audit=json.loads((HERE/'WINDOW_DATA_AUDIT.json').read_text())
    payload=(HERE/'windows_input.npz').read_bytes()
    if digest(payload)!=CACHE_SHA or audit['cache']['sha256']!=CACHE_SHA:raise ValueError('Wrong declared window input cache')
    with np.load(BytesIO(payload),allow_pickle=False) as archive:
        if set(archive.files)!=set(audit['cache']['keys']) or len(archive.files)!=36:raise ValueError('Unexpected window-cache key set')
        data={key:archive[key].copy() for key in archive.files}
    expected_windows=[phase+'_'+suffix for phase in PHASES for suffix in WINDOW_SUFFIXES]
    if [w['window_id'] for w in audit['windows']]!=expected_windows:raise ValueError('Fixed window order changed')
    for phase in PHASES:
        v,i,m=(data[phase+'_'+key] for key in ('command_voltage_v','observed_current_a','scoring_mask'))
        floor=1e-12 if phase=='reset4' else 1e-11
        original=np.isfinite(abs(i))&(abs(i)>=floor)&(abs(v)>1e-12)
        if m.dtype!=np.bool_ or not np.array_equal(m,original):raise ValueError('Original scoring mask changed')
        path=np.cumsum(abs(np.diff(np.r_[0.,v])))
        if not np.array_equal(data[phase+'_path_voltage_s'],path):raise ValueError('Declared path clock mismatch')
        if not np.array_equal(data[phase+'_native_index'],np.arange(len(v))):raise ValueError('Native index mismatch')
        if not np.array_equal(data[phase+'_outward_mask'],np.arange(len(v))<=np.argmax(abs(v))):raise ValueError('Outward mask mismatch')
        union=[]
        for window in [w for w in audit['windows'] if w['phase']==phase]:
            idx=data[window['index_key']]
            if not np.array_equal(idx,np.asarray(window['native_indices'])) or not np.all(m[idx]) or not np.all(np.diff(path[idx])>0.):
                raise ValueError('Window identity, eligibility or chronology mismatch')
            union.extend(idx.tolist())
        if not np.array_equal(np.unique(union),data[phase+'_window_unique_native_indices']) or len(np.unique(union))!=142:
            raise ValueError('Window union identity mismatch')
    kernel=load_minimal()
    for arr in (kernel.lg,kernel.lv,kernel.base.li,kernel.li_negative,kernel.base.tx,kernel.base.ti,kernel.negative_tx,kernel.negative_ti):arr.flags.writeable=False
    if (kernel.parameters.lambda_per_v,kernel.ron,kernel.gc)!=(.4470398752554136,58.463033350725695,.03127031375421752):
        raise ValueError('Fixed material constants mismatch')
    sources.update(kernel.source_sha256)
    return kernel,data,audit,sources,time.monotonic()-start


class Runner:
    def __init__(self,kernel,data,audit,writer,report):
        self.kernel,self.data,self.audit,self.writer,self.report=kernel,data,audit,writer,report
        self.budget=Budgets();self.ledger=Ledger();self.arrays={};self.points={};self.pchips={};self.gauss={}
        for phase in PHASES:
            for index in data[phase+'_window_unique_native_indices']:
                j=int(index);label=f'{phase}:{j}'
                row={'label':label,'phase':phase,'native_index':j,'status':'unattempted',
                     'command_voltage_v':float(data[phase+'_command_voltage_v'][j]),
                     'observed_current_original_column_a':float(data[phase+'_observed_current_a'][j]),
                     'observed_current_magnitude_a':float(abs(data[phase+'_observed_current_a'][j])),
                     'path_voltage_distance_v':float(data[phase+'_path_voltage_s'][j]),
                     'original_scoring_eligible':bool(data[phase+'_scoring_mask'][j])}
                self.points[(phase,j)]=row
        self.report['readpoints']=list(self.points.values());self.report['window_variants']=[]
    def snapshot(self):
        self.report.update(numerical_timing=self.budget.snapshot(),work_ledger=self.ledger.snapshot(),artifact_IO_wall_s=self.writer.io_s)
        self.writer.json(PROGRESS,self.report,replace=True,catalog=False)
    def inversions(self):
        self.report['status']='running_all_fixed_readpoint_inversions'
        for ordinal,row in enumerate(self.points.values()):
            if 'inverse' in self.budget.stopped:
                row['reason_unattempted']=self.budget.stopped['inverse'];continue
            context=self.ledger.context('inverse',row['label'],phase=row['phase'],native_index=row['native_index'])
            row['context_id']=context;row['status']='attempted';self.ledger.counts['conditional_readpoint_inversions_attempted']+=1
            raw=None
            try:
                def calculate(check):
                    nonlocal raw
                    proxy=InverseProxy(self.kernel,self.ledger,context,check)
                    raw=invert_readpoint(proxy,row['label'],row['command_voltage_v'],row['observed_current_magnitude_a'],check)
                    fast,strict=(raw['circuit_rechecks'][mode] for mode in ('fast','strict_bisection42'))
                    ld=abs(math.log10(abs(fast['current_a']/strict['current_a'])))
                    vd=abs(fast['local_voltage_v']-strict['local_voltage_v'])
                    if ld>LIMITS['root_log_current_decade'] or vd>LIMITS['root_local_voltage_v']:raise ValueError('Inverse two-circuit comparison failed')
                    raw['two_circuit_comparison']={'max_log_current_difference_decade':ld,'local_voltage_difference_v':vd}
                    try:raw['inverse_sensitivity']=inverse_sensitivity(self.kernel,raw)
                    except Exception as exc:raw['inverse_sensitivity']={'status':'undefined_preserved','error_type':type(exc).__name__,
                        'error':str(exc),'used_for_point_selection':False,'measurement_uncertainty_interval':False}
                    return raw
                answer=self.budget.run('inverse',row['label'],calculate,2.)
                row.update(status='completed',inverse=answer)
                self.ledger.counts['conditional_readpoint_inversions_completed']+=1
            except Exception as exc:
                row.update(status='timeout_unresolved' if isinstance(exc,BudgetStop) else 'inverse_failed_unresolved',
                    error_type=type(exc).__name__,error=str(exc),partial_inverse=raw,
                    failure_evidence=failure_evidence(exc),physical_mechanism_rejection=False)
                self.ledger.counts['conditional_readpoint_inversions_failed']+=1
            if (ordinal+1)%32==0 or row['status']!='completed':self.snapshot()
        completed=sum(row['status']=='completed' for row in self.points.values())
        self.writer.progress(f'Conditional inverse stage: {completed}/426 complete; all planned point statuses retained')
        self.snapshot()
    def branch_indices(self,phase,branch):
        windows=[w for w in self.audit['windows'] if w['phase']==phase and w['branch']==branch]
        return np.unique(np.concatenate([self.data[w['index_key']] for w in windows]))
    def quadrature(self):
        self.report['status']='running_fixed_interpolant_quadrature'
        for interpolant in INTERPOLANTS:
            for order in ORDERS:
                for window in self.audit['windows']:
                    wid,phase=window['window_id'],window['phase'];indices=self.data[window['index_key']]
                    label=f'{interpolant}_g{order}_{wid}'
                    dependent=indices if interpolant=='linear' else self.branch_indices(phase,window['branch'])
                    missing=[int(j) for j in dependent if self.points[(phase,int(j))]['status']!='completed']
                    row={'variant_id':label,'window_id':wid,'phase':phase,'branch':window['branch'],
                         'interpolant':interpolant,'gauss_order':order,'status':'unattempted',
                         'native_indices':indices.tolist(),'interpolation_dependency_indices':dependent.tolist(),
                         'planned_nodes':(len(indices)-1)*order,'new_node_initial_states_for_ODE':False}
                    self.report['window_variants'].append(row)
                    if missing:
                        row.update(status='unresolved_missing_inverse_dependency',missing_native_indices=missing);continue
                    if 'quadrature' in self.budget.stopped:
                        row['reason_unattempted']=self.budget.stopped['quadrature'];continue
                    context=self.ledger.context('quadrature',label,window_id=wid,interpolant=interpolant,gauss_order=order)
                    row['context_id']=context;row['status']='attempted'
                    self.ledger.counts['quadrature_window_variants_attempted']+=1
                    native_s=self.data[phase+'_path_voltage_s'][indices]
                    native_u=np.array([self.points[(phase,int(j))]['inverse']['direct_u'] for j in indices])
                    row['delta_u']=float(native_u[-1]-native_u[0])
                    out={};self.arrays.update({label+'__native_indices':indices.copy(),label+'__native_path_v':native_s.copy(),label+'__native_inverse_u':native_u.copy()})
                    try:
                        def calculate(check):
                            check()
                            if order not in self.gauss:self.gauss[order]=np.polynomial.legendre.leggauss(order)
                            x,w=self.gauss[order];h=np.diff(native_s)
                            node_s=(native_s[:-1,None]+.5*h[:,None]*(1+x[None,:])).ravel()
                            weights=(.5*h[:,None]*w[None,:]).ravel()
                            command=np.interp(node_s,native_s,self.data[phase+'_command_voltage_v'][indices])
                            if interpolant=='linear':u=np.interp(node_s,native_s,native_u)
                            else:
                                key=(phase,window['branch'])
                                if key not in self.pchips:
                                    branch_s=self.data[phase+'_path_voltage_s'][dependent]
                                    branch_u=np.array([self.points[(phase,int(j))]['inverse']['direct_u'] for j in dependent])
                                    if len(branch_s)!=71:raise ValueError('PCHIP must use the complete declared 71-point branch')
                                    self.pchips[key]=PchipInterpolator(branch_s,branch_u,extrapolate=False)
                                    self.ledger.counts['merged_branch_PCHIP_constructions']+=1
                                u=np.asarray(self.pchips[key](node_s))
                            if not np.all(np.isfinite(u)):raise ValueError('Nonfinite declared interpolated state')
                            n=len(node_s);selected=np.array([0,n//2-1,n//2,n-1],dtype=int) if order==16 else np.empty(0,dtype=int)
                            out.update(node_path_v=node_s,weights_path_v=weights,command_voltage_v=command,state_u=u,
                                native_interval_left_index=np.repeat(indices[:-1],order),within_interval_node_index=np.tile(np.arange(order),len(indices)-1),
                                completed_mask=np.zeros(n,dtype=bool),strict_selected_node_indices=selected,
                                strict_completed_mask=np.zeros(len(selected),dtype=bool))
                            for key in ('current_a','local_voltage_v','transistor_voltage_v','KCL_residual_a','KVL_residual_v','F','G'):
                                out[key]=np.full(n,np.nan)
                            for key in ('current_a','local_voltage_v','transistor_voltage_v','KCL_residual_a','KVL_residual_v'):
                                out['strict_'+key]=np.full(len(selected),np.nan)
                            strict_positions={int(j):k for k,j in enumerate(selected)}
                            fast=CountedCircuit(self.kernel,self.kernel.make_circuit(mode='fast'),self.ledger,context,'quadrature','fast',check)
                            strict=CountedCircuit(self.kernel,self.kernel.make_circuit(mode='strict_bisection42'),self.ledger,context,'quadrature','strict_bisection42',check)
                            differences=[]
                            for j in range(n):
                                check();self.ledger.counts['quadrature_nodes_attempted']+=1;fast.node=j
                                point=fast.solve(float(command[j]),float(u[j]))
                                for key in ('current_a','local_voltage_v','transistor_voltage_v','KCL_residual_a','KVL_residual_v'):out[key][j]=point[key]
                                if j in strict_positions:
                                    strict.node=j;other=strict.solve(float(command[j]),float(u[j]));k=strict_positions[j]
                                    for key in ('current_a','local_voltage_v','transistor_voltage_v','KCL_residual_a','KVL_residual_v'):out['strict_'+key][k]=other[key]
                                    ld=abs(math.log10(abs(point['current_a']/other['current_a'])));vd=abs(point['local_voltage_v']-other['local_voltage_v'])
                                    differences.append({'node_index':j,'log_current_difference_decade':ld,'local_voltage_difference_v':vd})
                                    if ld>LIMITS['root_log_current_decade'] or vd>LIMITS['root_local_voltage_v']:raise ValueError('Gauss16 strict42 static comparison failed')
                                    out['strict_completed_mask'][k]=True
                                local,current=abs(point['local_voltage_v']),abs(point['current_a'])
                                f=local*math.exp(self.kernel.parameters.lambda_per_v*local-float(u[j]));g=local*current*math.expm1(float(u[j]))
                                if not math.isfinite(f) or not math.isfinite(g) or min(f,g)<=0.:raise ValueError('Invalid quadrature F/G')
                                out['F'][j],out['G'][j],out['completed_mask'][j]=f,g,True
                                self.ledger.counts['quadrature_nodes_completed']+=1
                            A=float(np.dot(weights,out['F']));B=float(np.dot(weights,out['G']))
                            if not math.isfinite(A) or not math.isfinite(B) or min(A,B)<=0.:raise ValueError('Invalid complete A/B integrals')
                            check();return {'A_v2':A,'B_a_v2':B,'node_count':n,
                                'strict_comparisons':differences,'interpolation_rule':'piecewise_linear_u' if interpolant=='linear' else 'PCHIP_on_merged_71_native_node_branch',
                                'numerical_quadrature_error_certificate':False,'true_experimental_trajectory_certificate':False}
                        values=self.budget.run('quadrature',label,calculate,5.)
                        row.update(status='completed',**values);self.ledger.counts['quadrature_window_variants_completed']+=1
                    except Exception as exc:
                        row.update(status='timeout_unresolved' if isinstance(exc,BudgetStop) else 'quadrature_failed_unresolved',
                            error_type=type(exc).__name__,error=str(exc),failure_evidence=failure_evidence(exc),
                            partial_integrals_not_used_for_identification=True)
                        self.ledger.counts['quadrature_window_variants_failed']+=1
                    finally:
                        for key,value in out.items():self.arrays[label+'__'+key]=value
                        row['completed_nodes']=int(out.get('completed_mask',np.empty(0,dtype=bool)).sum())
                        row['array_prefix']=label+'__'
                    self.snapshot()
                complete=sum(r['status']=='completed' for r in self.report['window_variants'] if r['interpolant']==interpolant and r['gauss_order']==order)
                self.writer.progress(f'Quadrature {interpolant}/Gauss{order}: {complete}/12 complete fixed windows')
    def algebra(self):
        self.report['status']='running_fixed_two_window_linear_identification'
        reference=(self.kernel.parameters.kappa_per_v2,self.kernel.parameters.kappa_per_v2/self.kernel.parameters.J_positive_a)
        positive_ids=['set4_'+suffix for suffix in WINDOW_SUFFIXES[:2]]
        negative_ids=['reset4_'+suffix for suffix in WINDOW_SUFFIXES[:2]]
        groups=[];self.report['coefficient_diagnostics']=groups
        for interpolant in INTERPOLANTS:
            for order in ORDERS:
                groups.append({'interpolant':interpolant,'gauss_order':order,
                    'positive':{'status':'unattempted','available':False,'window_ids':positive_ids,'coefficient_names':['kappa','r']},
                    'negative':{'status':'unattempted','available':False,'window_ids':negative_ids,'coefficient_names':['alpha','beta']},
                    'shared_coefficient_check':{'status':'unattempted','defined':False,'not_a_physical_pass_fail_test':True},
                    'all_twelve_window_checks':[{'window_id':w['window_id'],'status':'unattempted',
                        'used_for_identification':w['window_id'] in positive_ids+negative_ids} for w in self.audit['windows']],
                    'remaining_check_windows':8,'errors_in_variables_and_correlated_endpoints':True,
                    'exact_two_window_closure_is_not_mechanism_evidence':True})
        for group in groups:
            interp,order=group['interpolant'],group['gauss_order']
            rows={r['window_id']:r for r in self.report['window_variants'] if r['interpolant']==interp and r['gauss_order']==order}
            for side,ids,names in [('positive',positive_ids,['kappa','r']),('negative',negative_ids,['alpha','beta'])]:
                record=group[side]
                if 'algebra' in self.budget.stopped:
                    record['reason_unattempted']=self.budget.stopped['algebra'];continue
                try:
                    def calculate_identification(check):
                        return identify([rows[k] for k in ids],names,reference,self.ledger,check,record)
                    self.budget.run('algebra',f'{interp}_g{order}_{side}',calculate_identification,10.)
                    if record['available']:self.ledger.counts['new_linear_identifications_completed']+=1
                except Exception as exc:
                    record.update(status='timeout_unresolved' if isinstance(exc,BudgetStop) else 'linear_diagnostic_failed_unresolved',
                        available=False,error_type=type(exc).__name__,error=str(exc),failure_evidence=failure_evidence(exc))
                    self.report.setdefault('algebra_failures',[]).append({'group':[interp,order],'side':side,
                        'error_type':type(exc).__name__,'error':str(exc),'already_recorded_diagnostics_retained':True})
            shared=group['shared_coefficient_check']
            if 'algebra' in self.budget.stopped:
                shared['reason_unattempted']=self.budget.stopped['algebra']
                for item in group['all_twelve_window_checks']:item['reason_unattempted']=self.budget.stopped['algebra']
                continue
            try:
                def calculate_checks(check):
                    check();positive,negative=group['positive'],group['negative']
                    shared.update(status='computed_diagnostic',
                        identification_conditions={side:{key:group[side].get(key) for key in
                            ('status','numerical_rank','column_normalized_condition_2','severely_ill_conditioned')} for side in ('positive','negative')},
                        any_identification_severely_ill_conditioned=any(group[side].get('severely_ill_conditioned',False) for side in ('positive','negative')))
                    if positive['available'] and negative['available']:
                        p,n=positive['coefficients'],negative['coefficients'];k,r,al,be=p['kappa'],p['r'],n['alpha'],n['beta']
                        denominator=k*r;numerator=al*be
                        valid=min(k,r,al,be)>0. and denominator>0. and all(math.isfinite(float(x)) for x in (k,r,al,be,denominator,numerator))
                        if valid:
                            product=numerator/denominator;tf=al/k;tg=r/be
                            if all(math.isfinite(float(x)) and x>0. for x in (product,tf,tg)):
                                bf,bg=.5*math.log(tf),.5*math.log(tg)
                                shared.update(defined=True,product_alpha_beta_over_kappa_r=product,
                                    recovery_multiplier=tf,capture_multiplier=tg,b_F=bf,b_G=bg,b_F_minus_b_G=bf-bg,
                                    both_multipliers_in_1_to_100=bool(1.<=tf<=100. and 1.<=tg<=100.))
                        if not shared['defined']:shared['reason']='Nonpositive coefficient or undefined/overflowing ratio; raw estimates retained without NNLS'
                    else:shared['reason']='A required fixed two-window identification was unavailable'
                    for item in group['all_twelve_window_checks']:
                        check();row=rows[item['window_id']];ident=negative if row['phase']=='reset4' else positive
                        item.update(status='computed_diagnostic',window_complete=row['status']=='completed',
                            identified_coefficients_available=ident['available'],
                            identification_severely_ill_conditioned=ident.get('severely_ill_conditioned'))
                        if row['status']=='completed' and ident['available']:
                            coeff=[ident['coefficients'][name] for name in ident['coefficient_names']]
                            item['closure']=closure(row['delta_u'],row['A_v2'],row['B_a_v2'],coeff)
                self.budget.run('algebra',f'{interp}_g{order}_shared_and_remaining_windows',calculate_checks,10.)
            except Exception as exc:
                shared.update(status='timeout_or_failure_partial_diagnostic_preserved',error_type=type(exc).__name__,error=str(exc))
                self.report.setdefault('algebra_failures',[]).append({'group':[interp,order],'side':'shared_and_remaining_windows',
                    'error_type':type(exc).__name__,'error':str(exc),'already_recorded_diagnostics_retained':True})
        comparisons=[];self.report['sensitivity_comparisons']=comparisons
        pairs=[('quadrature_order',interp,8,interp,16) for interp in INTERPOLANTS]
        pairs += [('interpolation_assumption','linear',order,'pchip',order) for order in ORDERS]
        lookup={(r['interpolant'],r['gauss_order'],r['window_id']):r for r in self.report['window_variants']}
        for kind,a,n,z,m in pairs:
            for window in self.audit['windows']:
                comparisons.append({'kind':kind,'window_id':window['window_id'],'first':[a,n],'second':[z,m],
                    'status':'unattempted','true_trajectory_or_noise_error_bound':False})
        if 'algebra' in self.budget.stopped:
            for row in comparisons:row['reason_unattempted']=self.budget.stopped['algebra']
        else:
            try:
                def calculate_sensitivity(check):
                    for row in comparisons:
                        check();a,n=row['first'];z,m=row['second'];wid=row['window_id']
                        one,two=lookup[(a,n,wid)],lookup[(z,m,wid)]
                        row.update(status='computed_diagnostic',both_complete=one['status']==two['status']=='completed')
                        if row['both_complete']:
                            row.update(A_difference=one['A_v2']-two['A_v2'],B_difference=one['B_a_v2']-two['B_a_v2'],
                                A_relative_difference=(one['A_v2']-two['A_v2'])/max(abs(one['A_v2']),abs(two['A_v2'])),
                                B_relative_difference=(one['B_a_v2']-two['B_a_v2'])/max(abs(one['B_a_v2']),abs(two['B_a_v2'])))
                self.budget.run('algebra','fixed_quadrature_and_interpolation_sensitivity',calculate_sensitivity,10.)
            except Exception as exc:self.report.setdefault('algebra_failures',[]).append({'side':'sensitivity_comparisons',
                'error_type':type(exc).__name__,'error':str(exc),'already_recorded_diagnostics_retained':True})
        for group in groups:
            for side in ('positive','negative'):
                if group[side]['status']=='unattempted':group[side].setdefault('reason_unattempted','Algebra execution did not reach this predeclared slot')
            for item in group['all_twelve_window_checks']:
                if item['status']=='unattempted':item.setdefault('reason_unattempted','Algebra execution did not reach this predeclared check')
    def execute(self):
        self.inversions();self.quadrature();self.algebra()
        self.arrays.update(self.ledger.arrays())
        self.report['all_numerical_arrays']=self.writer.npz(ARRAYS,self.arrays)
        completed_points=sum(p['status']=='completed' for p in self.points.values())
        completed_variants=sum(v['status']=='completed' for v in self.report['window_variants'])
        self.report['completeness']={'planned_readpoints':426,'attempted_readpoints':self.ledger.counts['conditional_readpoint_inversions_attempted'],
            'completed_readpoints':completed_points,'unattempted_readpoints':sum(p['status']=='unattempted' for p in self.points.values()),
            'planned_window_variants':48,'attempted_window_variants':self.ledger.counts['quadrature_window_variants_attempted'],
            'completed_window_variants':completed_variants,
            'planned_quadrature_nodes':20160,'completed_quadrature_nodes':self.ledger.counts['quadrature_nodes_completed'],
            'planned_static_root_calls':21108,'planned_material_bisection_iterations':20448,
            'planned_two_coefficient_identifications':8,'completed_two_coefficient_identifications':self.ledger.counts['new_linear_identifications_completed']}
        full=completed_points==426 and completed_variants==48 and not self.report.get('algebra_failures') and 'algebra' not in self.budget.stopped
        if full:
            expected={'static_root_attempts':21108,'static_root_returns':21108,'static_root_validated':21108,
                      'inverse_material_bisection_iterations_attempted':20448,'inverse_material_bisection_iterations_returned':20448,
                      'inverse_material_evaluations_attempted':22152,'inverse_material_evaluations_returned':22152,
                      'quadrature_nodes_attempted':20160,'quadrature_nodes_completed':20160,'new_linear_identification_attempts':8}
            self.report['full_success_work_count_checks']={k:{'actual':self.ledger.counts[k],'expected':v,'passed':self.ledger.counts[k]==v} for k,v in expected.items()}
            if not all(x['passed'] for x in self.report['full_success_work_count_checks'].values()):
                self.report['status']='work_count_mismatch_preserved';return
        self.report['status']='completed_conditional_interpolation_diagnostic' if full else 'completed_partial_conditional_diagnostic_unresolved_items_retained'
        self.snapshot()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-authorized',action='store_true');args=parser.parse_args()
    if not args.run_authorized:parser.error('Prepared only; explicit reviewed execution required')
    if RESULT.exists() or (HERE/'RUN_STARTED.json').exists():parser.error('Prior attempt exists; no automatic restart or overwrite')
    if not (HERE/'SOURCE_FREEZE.json').is_file() or not (HERE/'EXPERIMENT_PLAN.md').is_file():parser.error('Plan and SOURCE_FREEZE required before sole execution')
    writer=Writer();writer.json(HERE/'RUN_STARTED.json',{'status':'authorized_single_attempt_started',
        'budgets_s':Budgets.limits,'new_ODEs':0,'new_nonlinear_forward_fits':0,'new_b_scans':0})
    report={'status':'loading_frozen_sources','new_ODEs':0,'new_nonlinear_forward_fits':0,'new_b_scans':0,
        'initial_state_reanchoring':False,'extension_accessed':False,'original_holdout_accessed':False,
        'full_unified_model_achieved':False,'unconditional_mechanism_rejection':False,
        'experimental_noise_certificate':False,'true_between_sample_trajectory_certificate':False,
        'limits':LIMITS,'input_scope':'Three training phases; twelve predeclared windows, 426 unique native points. Full1170 score contract unchanged.',
        'claim_scope':'Conditional on frozen a=0 readout, lambda, common path clock and the stated interpolation. Not a physical mechanism pass/fail certificate.'}
    runner=None
    try:
        kernel,data,audit,sources,loading=load_inputs()
        report.update(source_sha256=sources,input_loading_and_hash_wall_s=loading,
            fixed_material={'lambda_per_v':kernel.parameters.lambda_per_v,'Ron_ohm':kernel.ron,'Gamma_c_ev':kernel.gc,'parallel_path_a_s':0.},
            raw_data_audit_counts=deepcopy(audit['counts']))
        runner=Runner(kernel,data,audit,writer,report);runner.execute()
        started=time.monotonic();actual={path:digest(Path(path).read_bytes()) for path in sources}
        report.update(final_source_hash_wall_s=time.monotonic()-started,source_files_unchanged=actual==sources)
        if actual!=sources:report['status']='source_changed_results_not_frozen'
    except Exception as exc:
        report.update(status='stopped_with_preserved_exception',error_type=type(exc).__name__,error=str(exc))
    finally:
        if runner is not None:
            report.update(numerical_timing=runner.budget.snapshot(),work_ledger=runner.ledger.snapshot())
            if 'all_numerical_arrays' not in report:
                runner.arrays.update(runner.ledger.arrays())
                if not ARRAYS.exists():report['all_numerical_arrays']=writer.npz(ARRAYS,runner.arrays)
        report.update(artifact_IO_wall_s=writer.io_s,artifact_catalog=deepcopy(writer.catalog),
                      terminal_result=True,automatic_retry_permitted=False)
        writer.json(RESULT,report)
        writer.progress('Terminal conditional integral diagnostic: '+report['status'])


if __name__=='__main__':main()
