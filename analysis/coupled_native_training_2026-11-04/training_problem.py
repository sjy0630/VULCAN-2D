"""Original training records with actual continuous coupled q/a histories.

Exactly three declared segment entries, 46 pairs, 92 phases and the original
27023 weighted residual entries. Import performs no data load or experiment.
The archived objective is used only by the explicitly called arithmetic
reference check; production scoring never imports the old package initializer.
"""
from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import MappingProxyType, SimpleNamespace

sys.dont_write_bytecode = True
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
KERNEL_PATH = ROOT/'analysis/coupled_state_kernel_2026-11-04/coupled_forward.py'
KERNEL_SHA256 = '1288ae379fbefff679bebd3ad110a10cee344f970d8feb6a8dff7838b58a26d7'
ORDERED_PATH = ROOT/'analysis/constrained_shared_2026-11-03/ordered_objective.py'
ORDERED_SHA256 = 'd398c68145418c718539d3b7abbef05ce5ccddb02195afbab7b9f2d099a97f55'
IDENTITY_PATH = ROOT/'analysis/bare_history_2026-11-03/identity_confirmation.json'
IDENTITY_SHA256 = 'dc05cac07e3e7e50986df5e2aa065a2ea7e9628693fa16d7cb37d585268d0189'
DATA_PATH = Path('/private/tmp/vulcan-runtime-cache/native_observations.npz')
DATA_SHA256 = '0aedbe51f41b4f4355e1de4d26048c6790a9d454b8b7d56f51bd76a4671465a8'
TRAINING_ARRAYS_SHA256 = '7a1d7ade35abf7116ce5b77dfcdf89d3fa205a90b65043062be7bd1638712b5e'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _check_source(path, expected):
    if sha(path) != expected:
        raise ValueError(f'Frozen source/data hash changed: {path}')


# Expose the exact module used by callers to construct the shared material;
# no second import alias with incompatible dataclass identities is created.
_check_source(KERNEL_PATH,KERNEL_SHA256)
if str(KERNEL_PATH.parent) not in sys.path:
    sys.path.insert(0,str(KERNEL_PATH.parent))
kernel = importlib.import_module('coupled_forward')
if Path(kernel.__file__).resolve() != KERNEL_PATH.resolve():
    raise RuntimeError('coupled_forward module name is occupied by another source')

TRAIN = MappingProxyType({
    '1r': (1,2,3,5,6,7,8,9,10,11,12,13,14,15),
    '1t1r': tuple(range(4,36)),
})
SEGMENTS = (
    ('bare1','1r',(1,2,3)),
    ('bare5','1r',tuple(range(5,16))),
    ('series4','1t1r',tuple(range(4,36))),
)
PHASES = ('set','reset')
EXPECTED_GROUPS = {
    ('1r','set','nonlimited_outward'): (730,'e617327c53abe5bbfa56c54d13a50327ccb633cc4054839dc18c9b873bf98575'),
    ('1r','set','nonlimited_return'): (27,'1b3784cb1f80ee793111b4ded5e5bb0c80af56379ece680e2902cc06feb237d6'),
    ('1r','set','limited'): (2027,'4d93e9a061434f0f9c60a9af86454144ad4e55ac983f2a14f3d4f39da5f27562'),
    ('1r','reset','all_eligible'): (2799,'00087c77c4e42444b6a1d890d9230ed30bdabc8fb25b4d2eb60a646f9bd22312'),
    ('1t1r','set','all_eligible'): (16000,'00ce8751eae53accf8f2782a5b5dbfa14e42780c38b7500b264a4ff96368f5f2'),
    ('1t1r','reset','all_eligible'): (5440,'ab20addb5d44150775ae4bad8ce36915c95cc89c1ffae70d131a8ce3a34d2a07'),
}


def _frozen(value):
    array = np.asarray(value)
    return np.frombuffer(array.tobytes(),dtype=array.dtype).reshape(array.shape)


def _budget(callback):
    if callback is not None:
        callback()


def prediction_current(prediction):
    """Accept a coupled phase result, a current trace, or a native array."""
    if isinstance(prediction,Mapping):
        if 'arrays' in prediction:
            prediction = prediction['arrays']
        if isinstance(prediction,Mapping):
            prediction = prediction['current_a']
    return np.asarray(prediction)


class TrainingFailure(RuntimeError):
    """Preserve every completed/failed phase; never score a partial full task."""
    def __init__(self, message, result):
        super().__init__(message)
        self.result = result


@dataclass(frozen=True)
class TrainingProblem:
    data: object
    waveforms: object
    raw_training: object
    log_observed: object
    scoring_masks: object
    groups: object
    source_sha256: object
    decoded_keys: tuple
    training_arrays_sha256: str
    scatter_constant_statistic: float
    n_residuals: int = 27023

    @property
    def train_ids(self):
        return TRAIN

    @property
    def phase_keys(self):
        return tuple((top,cid,phase) for _,top,ids in SEGMENTS for cid in ids for phase in PHASES)

    def checkpoint_metadata(self):
        rows=[]
        offset=0
        for (top,phase),groups in self.groups.items():
            for group in groups:
                n=group['n']
                rows.append(dict(topology=top,phase=phase,name=group['name'],n=n,
                    normalization=float(group['normalization']),mask_sha256=group['mask_sha256'],
                    residual_start=offset,residual_stop=offset+n))
                offset+=n
        return dict(status='original_training_adapter_no_fit',train_ids={k:list(v) for k,v in TRAIN.items()},
            source_sha256=dict(self.source_sha256),training_arrays_sha256=self.training_arrays_sha256,
            loaded_data_path=str(DATA_PATH),decoded_array_keys=list(self.decoded_keys),
            n_decoded_arrays=len(self.decoded_keys),holdout_arrays_decoded=False,
            n_training_pairs=46,n_training_phases=92,n_residuals=self.n_residuals,
            segment_entries=[dict(entry_key=name,topology=top,record_ids=list(ids)) for name,top,ids in SEGMENTS],
            objective='four phase groups equal weight; bare SET three subgroups each one third',
            cost_definition='0.5 * dot(weighted_residual, weighted_residual); no extra /4',
            original_mask_threshold_a={'set':1e-11,'reset':1e-12},
            original_voltage_mask='abs(command_voltage)>1e-12',
            bare_set_observation_limit_threshold_a=.995e-3,
            bare_set_actual_circuit_compliance_a=.001,
            reset_compliance_a=None,series_gate_v=1.1,group_contract=rows,
            legacy_observation_scatter_statistic=self.scatter_constant_statistic,
            scatter_subtracted_from_objective=False,periodic_prediction_reuse=False,
            missing_measured_bare_cycles=[4,23],missing_cycles_simulated_or_bridged=False,
            timing_policy='native voltage-path clock; no measured seconds or unrecorded waits inferred')

    def _score(self, predictions):
        if not isinstance(predictions,Mapping) or set(predictions) != set(self.phase_keys):
            raise ValueError('Scoring requires exactly the original 92 training phase keys')
        weighted_parts,raw_parts,blocks,record_parts = [],[],{},{}
        for top,ids in TRAIN.items():
            for phase in PHASES:
                key=top,phase
                predicted=[]
                for cid in ids:
                    source=predictions[top,cid,phase]
                    trace=source.get('arrays',source) if isinstance(source,Mapping) else None
                    if isinstance(trace,Mapping) and 'command_voltage_v' in trace:
                        if not np.array_equal(np.asarray(trace['command_voltage_v']),self.data[top,cid,phase]['command_voltage_v']):
                            raise ValueError('Prediction native waveform differs from its own original record')
                    predicted.append(prediction_current(source))
                predicted=abs(np.array(predicted))
                if predicted.shape != self.raw_training[key].shape:
                    raise ValueError('Prediction shape differs from the original record matrix')
                group_rows=[]
                phase_mse=0.
                for group in self.groups[key]:
                    selected=group['mask']
                    if np.any(~np.isfinite(predicted[selected])) or np.any(predicted[selected]<=0):
                        raise ValueError('Invalid predicted current at an original eligible observation')
                    errors=np.log10(predicted[selected])-self.log_observed[key][selected]
                    block=errors/group['normalization']
                    weighted_parts.append(block)
                    raw_parts.append(errors)
                    weighted=float(np.dot(block,block))
                    phase_mse+=weighted
                    group_rows.append(dict(name=group['name'],n=group['n'],
                        raw_rmse_decade=float(np.sqrt(np.mean(errors**2))),weighted_group_mse=weighted,
                        first_record_point_weight=group['first_record_point_weight'],
                        remaining_records_point_weight=group['remaining_records_point_weight']))
                blocks[f'{top}_{phase}']=dict(weighted_total_mse=phase_mse,groups=group_rows)
                for row,cid in enumerate(ids):
                    selected=self.scoring_masks[key][row]
                    errors=np.log10(predicted[row,selected])-self.log_observed[key][row,selected]
                    outward=self.data[top,cid,phase]['outward_mask'][selected]
                    metrics=dict(n=int(len(errors)),plain_RMSE_decade=float(np.sqrt(np.mean(errors**2))),
                        mean_signed_log_error_decade=float(np.mean(errors)),
                        outward_n=int(outward.sum()),return_n=int((~outward).sum()),
                        outward_RMSE_decade=float(np.sqrt(np.mean(errors[outward]**2))) if np.any(outward) else None,
                        return_RMSE_decade=float(np.sqrt(np.mean(errors[~outward]**2))) if np.any(~outward) else None)
                    record_parts.setdefault((top,cid),{})[phase]=(metrics,errors)
        weighted=np.concatenate(weighted_parts)
        raw=np.concatenate(raw_parts)
        if len(weighted)!=self.n_residuals or len(raw)!=self.n_residuals or not np.all(np.isfinite(weighted)):
            raise ValueError('Original fixed-dimensional weighted residual contract failed')
        per_record=[]
        for top,ids in TRAIN.items():
            for cid in ids:
                pieces=record_parts[top,cid]
                all_errors=np.concatenate([pieces[phase][1] for phase in PHASES])
                per_record.append(dict(topology=top,record_id=cid,
                    **{phase:pieces[phase][0] for phase in PHASES},
                    all_eligible=dict(n=len(all_errors),plain_RMSE_decade=float(np.sqrt(np.mean(all_errors**2))))))
        sse=float(np.dot(weighted,weighted))
        details=dict(n_residuals=len(weighted),verified_cost=sse/2.,weighted_norm_squared=sse,
            balanced_fourphase_RMSE_decade=float(np.sqrt(sse/4.)),
            plain_allpoint_RMSE_decade=float(np.sqrt(np.mean(raw**2))),
            blocks=blocks,per_record=per_record,cost_includes_record_scatter=True,
            scatter_subtracted_from_cost=False,normalization='unchanged original ordered_objective')
        return weighted,raw,details

    def score_predictions(self, predictions):
        """Pure original scorer for arbitrary complete native predictions."""
        weighted,_,details=self._score(predictions)
        return weighted,details

    def evaluate(self, material, initial_states, precision, method, check_budget,
                 collect_rhs=False):
        """Run all three histories using one explicit frozen material object."""
        _check_source(KERNEL_PATH,KERNEL_SHA256)
        if not isinstance(material,kernel.CoupledMaterial):
            raise TypeError('Construct one material with training_problem.kernel')
        if material.source_sha256.get(str(KERNEL_PATH.resolve())) != KERNEL_SHA256:
            raise ValueError('Material provenance does not match the frozen coupled kernel')
        if not isinstance(initial_states,Mapping) or set(initial_states)!={name for name,_,_ in SEGMENTS}:
            raise ValueError('Only explicit segment entries bare1, bare5 and series4 are accepted')
        entries={name:material.validate_state(initial_states[name]) for name,_,_ in SEGMENTS}
        if not isinstance(precision,kernel.Precision) or method not in ('DOP853','Radau'):
            raise ValueError('Explicit kernel Precision and DOP853/Radau required')
        if check_budget is not None and not callable(check_budget):
            raise TypeError('check_budget must be callable or None')
        circuits={('1r','set'):kernel.BareCircuit(compliance_a=.001),
                  ('1r','reset'):kernel.BareCircuit(compliance_a=None)}
        series=kernel.SeriesCircuit(compliance_a=None)
        circuits.update({('1t1r',phase):series for phase in PHASES})
        predictions,segments={},[]
        result=dict(summary=dict(status='running',checkpoint_metadata=self.checkpoint_metadata(),
            material=material.metadata(),initial_states={k:list(v) for k,v in entries.items()},
            one_shared_material_object=True,n_segment_entries=3,n_initial_state_coordinates=6,
            n_active_initial_state_coordinates=6 if material.parameters.enable_a else 3,
            initial_states_selected_by_adapter=False,periodic_prediction_reuse=False,
            n_completed_phases=0,n_attempted_phases=0),predictions=predictions,segments=segments,
            residual=np.empty(0),raw_residual=np.empty(0),score=None)
        active_segment=None
        try:
            for name,top,ids in SEGMENTS:
                active_segment=name
                _budget(check_budget)
                keys=[(top,cid,phase) for cid in ids for phase in PHASES]
                phases=[dict(data=self.data[key],circuit=circuits[top,key[2]]) for key in keys]
                segment=dict(segment_id=name,topology=top,record_ids=list(ids),
                    initial_state_q=entries[name][0],initial_state_a_s=entries[name][1],
                    entry_source='explicit_unknown_state_after_missing_bare4' if name=='bare5'
                    else 'explicit_segment_preparation_state',status='running')
                segments.append(segment)
                try:
                    history=kernel.simulate_history(material,phases,entries[name],precision,method,
                                                    check_budget,collect_rhs=collect_rhs)
                except kernel.HistoryFailure as exc:
                    segment.update(status='failed_history_retained',result=exc.result)
                    for key,phase_result in zip(keys,exc.result['phases']):
                        predictions[key]=phase_result
                    raise
                segment.update(status='complete',result=history,
                    final_state_q=history['summary']['final_state_q'],
                    final_state_a_s=history['summary']['final_state_a_s'])
                if len(history['phases'])!=len(keys):
                    raise RuntimeError('Kernel returned an incomplete segment without failure')
                for key,phase_result in zip(keys,history['phases']):
                    predictions[key]=phase_result
                result['summary']['n_completed_phases']=len(predictions)
                result['summary']['n_attempted_phases']=len(predictions)
            _budget(check_budget)
            weighted,raw,score=self._score(predictions)
            result.update(residual=weighted,raw_residual=raw,score=score)
            result['summary'].update(status='complete_original_training',n_completed_phases=92,
                n_attempted_phases=92,n_integrated_pairs=46,n_bare_pairs=14,n_series_pairs=32,
                n_residuals=len(weighted),verified_cost=score['verified_cost'],
                missing_bare4_bridged=False,all_92_phases_independently_propagated=True,
                state_reset_only_at_declared_segment_entries=True,
                all_material_parameters_shared_across_segments=True)
            return result
        except Exception as exc:
            result['summary'].update(status='incomplete_training_failure_retained',
                failed_segment=active_segment,error_type=type(exc).__name__,error=str(exc),
                n_attempted_phases=len(predictions),
                n_completed_phases=sum(p.get('summary',{}).get('status')=='complete' for p in predictions.values()),
                full_objective_available=False,partial_score_reported_as_full=False,
                interpretation='Preserved input/numerical/domain failure; not evidence of physical rejection')
            raise TrainingFailure(str(exc),result) from exc


def load_training_problem(*, check_budget=None):
    """Decode only the 184 named voltage/current arrays of the original train split."""
    if check_budget is not None and not callable(check_budget):
        raise TypeError('check_budget must be callable or None')
    sources={KERNEL_PATH:KERNEL_SHA256,ORDERED_PATH:ORDERED_SHA256,
             IDENTITY_PATH:IDENTITY_SHA256,DATA_PATH:DATA_SHA256}
    for path,digest in sources.items():
        _budget(check_budget)
        _check_source(path,digest)
    identity=json.loads(IDENTITY_PATH.read_text())['confirmed']
    if not (identity['bare_physical_device_count']==1 and identity['numeric_chronological_order']
            and identity['same_label_SET_then_RESET'] and identity['missing_measured_cycles']==[4,23]):
        raise ValueError('Frozen user-confirmed chronology does not support these segments')
    data,waveforms,raw_training,log_observed,masks,groups={},{},{},{},{},{}
    decoded=[]
    digest=hashlib.sha256()
    scatter=0.
    segment_for={(top,cid):name for name,top,ids in SEGMENTS for cid in ids}
    with np.load(DATA_PATH,allow_pickle=False) as archive:
        for top,ids in TRAIN.items():
            for phase in PHASES:
                vv,ii=[],[]
                for cid in ids:
                    _budget(check_budget)
                    prefix=f'{top}_cycle_{cid}_{phase}_'
                    vkey,ikey=prefix+'voltage',prefix+'current'
                    vv.append(archive[vkey].copy())
                    ii.append(archive[ikey].copy())
                    decoded.extend((vkey,ikey))
                vv=np.array(vv)
                signed=np.array(ii)
                observed=abs(signed)
                if vv.ndim!=2 or signed.shape!=vv.shape or not np.all(np.isfinite(vv)):
                    raise ValueError('Original native training voltage/current shapes changed')
                if not np.all(vv==vv[0]) or vv[0,0]!=0 or vv[0,-1]!=0:
                    raise ValueError('Frozen original training waveform equality/zero endpoints changed')
                key=top,phase
                v=vv[0]
                digest.update(str(key).encode())
                digest.update(vv.tobytes())
                digest.update(observed.tobytes())
                mask=np.isfinite(observed)&(abs(v)[None,:]>1e-12)
                mask &= observed >= (1e-11 if phase=='set' else 1e-12)
                logs=np.zeros_like(observed)
                logs[mask]=np.log10(observed[mask])
                outward=np.arange(len(v))<=np.argmax(abs(v))
                if key==('1r','set'):
                    named=[('nonlimited_outward',mask&(observed<.995e-3)&outward),
                           ('nonlimited_return',mask&(observed<.995e-3)&~outward),
                           ('limited',mask&(observed>=.995e-3))]
                else:
                    named=[('all_eligible',mask)]
                group_rows=[]
                coverage=np.zeros_like(mask,dtype=int)
                for name,selected in named:
                    n=int(selected.sum())
                    mask_hash=hashlib.sha256(selected.tobytes()).hexdigest()
                    if (n,mask_hash)!=EXPECTED_GROUPS[top,phase,name]:
                        raise ValueError(f'Original objective mask changed: {(top,phase,name)}')
                    coverage+=selected
                    count=selected.sum(axis=0)
                    mean=np.where(selected,logs,0.).sum(axis=0)/np.maximum(count,1)
                    observed_scatter=float(np.sum(np.where(selected,(logs-mean)**2,0.))/n)
                    scatter+=observed_scatter/(2.*len(named))
                    group_rows.append(MappingProxyType(dict(name=name,mask=_frozen(selected),n=n,
                        normalization=np.sqrt(n*len(named)),mask_sha256=mask_hash,
                        observation_scatter_mse=observed_scatter,
                        first_record_point_weight=float(selected[0].sum()/n),
                        remaining_records_point_weight=float(selected[1:].sum()/n))))
                if not np.array_equal(coverage,mask.astype(int)):
                    raise ValueError('Original subgroups are not an exact disjoint mask partition')
                waveforms[key]=_frozen(v)
                raw_training[key]=_frozen(observed)
                log_observed[key]=_frozen(logs)
                masks[key]=_frozen(mask)
                groups[key]=tuple(group_rows)
                for row,cid in enumerate(ids):
                    data[top,cid,phase]=MappingProxyType(dict(command_voltage_v=_frozen(vv[row]),
                        observed_current_a=_frozen(signed[row]),scoring_mask=_frozen(mask[row]),
                        initial_command_voltage_v=0.,record_id=cid,phase=phase,topology=top,
                        segment_id=segment_for[top,cid],outward_mask=_frozen(outward),
                        observed_limited_mask=_frozen(observed[row]>=.995e-3)))
    if len(decoded)!=184 or len(set(decoded))!=184 or len(data)!=92:
        raise ValueError('Training-only decode count differs from 46 pairs/92 phases/184 arrays')
    if digest.hexdigest()!=TRAINING_ARRAYS_SHA256:
        raise ValueError('Decoded original training-array digest changed')
    n=sum(group['n'] for phase_groups in groups.values() for group in phase_groups)
    if n!=27023:
        raise ValueError('Original objective must contain exactly 27023 residual entries')
    source_hashes={str(path.resolve()):value for path,value in sources.items()}
    source_hashes[str(Path(__file__).resolve())]=sha(Path(__file__))
    return TrainingProblem(*[MappingProxyType(value) for value in (
        data,waveforms,raw_training,log_observed,masks,groups,source_hashes)],
        tuple(decoded),digest.hexdigest(),float(scatter),n)


def compare_legacy_scorer(problem, predictions=None):
    """Explicit source-only arithmetic reference; no model or old package import.

    If predictions is omitted, make one deterministic record/node-varying
    prediction collection using already decoded TRAINING observations only.
    This is a scorer test fixture, not a candidate simulation or fit.
    """
    if not isinstance(problem,TrainingProblem):
        raise TypeError('Loaded TrainingProblem required')
    _check_source(ORDERED_PATH,ORDERED_SHA256)
    generated=predictions is None
    if generated:
        predictions={}
        for top_index,(top,ids) in enumerate(TRAIN.items()):
            for record_index,cid in enumerate(ids):
                for phase_index,phase in enumerate(PHASES):
                    observed=abs(problem.data[top,cid,phase]['observed_current_a'])
                    node=np.arange(len(observed),dtype=float)
                    offset=(.07*(top_index+1)+.011*(record_index+1)-.023*phase_index
                            +.03*np.cos(node/7.)+.005*(node%5))
                    predictions[top,cid,phase]=observed*np.power(10.,offset)
    flattened={key:prediction_current(value) for key,value in predictions.items()}
    tree=ast.parse(ORDERED_PATH.read_text(),filename=str(ORDERED_PATH))
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='OrderedJointProblem')
    method=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='score_predictions')
    isolated=ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[]))
    namespace={'np':np,'TRAIN':{top:list(ids) for top,ids in TRAIN.items()}}
    exec(compile(isolated,str(ORDERED_PATH),'exec'),namespace)
    fake=SimpleNamespace(raw_training=problem.raw_training,log_observed=problem.log_observed,
                         groups=problem.groups,n_residuals=problem.n_residuals)
    old,old_blocks=namespace['score_predictions'](fake,flattened)
    new,new_details=problem.score_predictions(predictions)
    equal=np.array_equal(old,new)
    blocks_equal=old_blocks==new_details['blocks']
    return dict(status='pure_arithmetic_original_scorer_comparison',passed=bool(equal and blocks_equal),
        residual_bitwise_equal=bool(equal),group_details_exactly_equal=bool(blocks_equal),
        n_residuals=len(new),max_abs_weighted_difference=float(np.max(abs(old-new))),
        reference_cost=float(np.dot(old,old)/2.),new_cost=new_details['verified_cost'],
        generated_record_and_node_varying_fixture=generated,
        frozen_reference_source=str(ORDERED_PATH),frozen_reference_sha256=ORDERED_SHA256,
        old_package_imported=False,model_evaluations=0,holdout_arrays_decoded=False)
