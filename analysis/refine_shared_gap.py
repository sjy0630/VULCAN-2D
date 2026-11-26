"""Accurate, all-parameter joint recalibration of the unchanged GapLaw.

Uses only the declared training records. Log-mean sufficient statistics
minimize the model-dependent part of the original per-record squared error;
record-to-record variance is reported separately, never subtracted from scores.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vulcan2d.shared_gap import GapLaw, BareGapCircuit, LookupGapCircuit
from vulcan2d.shared_gap_ivp import simulate_gap_ivp
from vulcan2d.transistor import measured_transistor_mapping

OLD=ROOT/'analysis/shared_hbn_2026-11-03'
OUT=ROOT/'analysis/shared_gap_refinement_2026-11-03'
DATA=OLD/'data/native_observations.npz'
NAMES=['log10_v0','log10_i0','log10_Icrit_set','log10_Icrit_reset',
       'log10_kset','log10_kreset','log10_escale','initial_u_1r','initial_u_1t1r',
       'reset_initiation_drive_V']
LOWER=np.array([-2.3,-9.,-6.,-7.,-4.,-4.,-2.7,.01,.01,0.])
UPPER=np.array([.3,-.5,-2.,-.5,5.,5.,.3,29.9,29.9,1.5])
SCALES=np.array([1.,1.,1.,1.,2.,2.,1.,5.,5.,.3])


def law_from_q(q,ron):
    v0,i0,ics,icr,ks,kr,es=10.**np.asarray(q[:7])
    return GapLaw(ron_ohm=ron,v0_v=v0,i0_a=i0,u_max=30.,kset_per_v=ks,kreset_per_v=kr,
                  eth_set_v=v0*np.arcsinh(ics/i0),eth_reset_v=v0*np.arcsinh(icr/i0),
                  escale_v=es,reset_initiation_drive_v=float(q[9]))


class JointProblem:
    def __init__(self, *, state_rtol=1e-6,state_atol=1e-8,current_rtol=5e-4):
        old=json.loads((OLD/'gap_joint_fit.json').read_text())
        self.ron=old['shared_ron_ohm']
        self.split=old['split']
        self.waveforms={};self.groups={};self.raw_training={}
        with np.load(DATA) as raw:
            for topology,split in self.split.items():
                for phase in ('set','reset'):
                    records=split['train']
                    vv=np.array([raw[f'{topology}_cycle_{cid}_{phase}_voltage'] for cid in records])
                    ii=np.abs(np.array([raw[f'{topology}_cycle_{cid}_{phase}_current'] for cid in records]))
                    if not np.all(vv==vv[0]): raise ValueError('Training waveforms differ')
                    v=vv[0]; key=(topology,phase)
                    self.waveforms[key]=v;self.raw_training[key]=ii
                    valid=np.isfinite(ii)&(abs(v)[None,:]>1e-12)&(ii>=(1e-11 if phase=='set' else 1e-12))
                    if key==('1r','set'):
                        up=np.arange(len(v))<=np.argmax(abs(v))
                        masks=[valid&(ii<.995e-3)&up, valid&(ii<.995e-3)&~up,valid&(ii>=.995e-3)]
                    else:masks=[valid]
                    groups=[]
                    logs=np.log10(np.maximum(ii,1e-300))
                    for mask in masks:
                        count=mask.sum(axis=0);indices=np.flatnonzero(count)
                        if not len(indices):raise ValueError('Empty training group')
                        mean=(np.where(mask,logs,0.).sum(axis=0)[indices]/count[indices])
                        weight=np.sqrt(count[indices]/mask.sum()/len(masks))
                        scatter=float(np.sum(np.where(mask,(logs-np.divide(np.where(mask,logs,0.).sum(axis=0),count,
                            out=np.zeros_like(count,dtype=float),where=count>0)[None,:])**2,0.))/mask.sum())
                        groups.append((indices,mean,weight,scatter))
                    self.groups[key]=groups
        transistor=LookupGapCircuit(measured_transistor_mapping(),1.1)
        self.circuits={'1r':(BareGapCircuit(.001),BareGapCircuit()),'1t1r':(transistor,transistor)}
        self.options=dict(method='Radau',state_rtol=state_rtol,state_atol=state_atol,current_rtol=current_rtol)

    def trajectory(self,q,topology,repeats=1):
        law=law_from_q(q,self.ron); u=float(q[7 if topology=='1r' else 8]);traces=[]
        for repeat in range(repeats):
            current=[]
            for phase,port in zip(('set','reset'),self.circuits[topology]):
                result=simulate_gap_ivp(self.waveforms[topology,phase],u,law,port,**self.options)
                u=result['final_u'];current.append(result)
            traces.append(current)
        return traces

    def residual(self,q,return_details=False):
        pieces=[];detail={}
        for topology in self.circuits:
            n=1 if topology=='1r' else 2
            runs=self.trajectory(q,topology,n)
            for r,phases in enumerate(runs):
                for phase,result in zip(('set','reset'),phases):
                    pred=np.abs(result['current_a']); errors=[]
                    for indices,mean,weight,scatter in self.groups[topology,phase]:
                        if np.any(pred[indices]<=0.) or not np.all(np.isfinite(pred[indices])):
                            raise ValueError('Invalid model current at scored input')
                        errors.append((np.log10(pred[indices])-mean)*weight/np.sqrt(n))
                    block=np.concatenate(errors);pieces.append(block)
                    detail[f'{topology}_{r}_{phase}']={'weighted_bias_mse':float(np.dot(block,block)),
                        'final_u':result['final_u'],'numerical_error_estimate':result['metadata']['current_relative_error_estimate']}
        residual=np.concatenate(pieces)
        return (residual,detail) if return_details else residual


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--max-nfev',type=int,default=100)
    parser.add_argument('--starts',type=int,default=1)
    parser.add_argument('--diff-step',type=float,default=1e-3)
    parser.add_argument('--output',default='accurate_fit.json')
    args=parser.parse_args();OUT.mkdir(exist_ok=True)
    problem=JointProblem()
    previous=json.loads((OLD/'gap_joint_fit.json').read_text())
    initial=np.asarray(previous['fits'][previous['selected_start']]['q'])
    old_base=json.loads((OLD/'gap_base_fit.json').read_text())
    seeds=[initial,np.r_[old_base['fits'][old_base['selected_start']]['q'],.05]]
    calls=0;invalid=[];best=np.inf;t0=time.monotonic();results=[]
    initial_error,initial_detail=problem.residual(initial,True)
    size=len(initial_error)
    def residual(q):
        nonlocal calls,best
        calls+=1
        try: error=problem.residual(q)
        except (ValueError,RuntimeError,OverflowError,FloatingPointError) as exc:
            invalid.append({'call':calls,'error':str(exc),'q':q.tolist()})
            error=np.full(size,100./np.sqrt(size))
        norm=float(np.linalg.norm(error));best=min(best,norm)
        if calls%25==1:
            print(json.dumps({'calls':calls,'seconds':round(time.monotonic()-t0,1),'best_norm':best,
                              'current_norm':norm,'invalid_trials':len(invalid)}),flush=True)
        return error
    for index,seed in enumerate(seeds[:args.starts]):
        result=least_squares(residual,np.clip(seed,LOWER+1e-7,UPPER-1e-7),bounds=(LOWER,UPPER),
                             x_scale=SCALES,diff_step=args.diff_step,max_nfev=args.max_nfev,
                             ftol=2e-7,xtol=2e-7,gtol=2e-6)
        try:
            checked,detail=problem.residual(result.x,True)
            valid=True;cost=float(np.dot(checked,checked)/2);reason=None
        except (ValueError,RuntimeError,OverflowError,FloatingPointError) as exc:
            valid=False;cost=None;detail={};reason=str(exc)
        row={'start':index,'q':result.x.tolist(),'success':bool(result.success),'message':str(result.message),
             'nfev':int(result.nfev),'njev':int(result.njev),'optimality':float(result.optimality),
             'valid':valid,'verified_bias_cost':cost,'validation_error':reason,'blocks':detail,
             'singular_values_jacobian':np.linalg.svd(result.jac,compute_uv=False).tolist(),
             'active_bounds':result.active_mask.tolist()}
        results.append(row)
        chosen=min((r for r in results if r['valid']),key=lambda r:r['verified_bias_cost'],default=None)
        output={'status':'unchanged_material_law_recalibrated' if chosen else 'no_valid_fit',
                'parameter_names':NAMES,'shared_ron_ohm':problem.ron,'split':problem.split,'settings':vars(args),
                'integrator_options':problem.options,'fits':results,'selected_start':chosen['start'] if chosen else None,
                'selected_law':asdict(law_from_q(chosen['q'],problem.ron)) if chosen else None,
                'initial_bias_cost':float(np.dot(initial_error,initial_error)/2),'initial_blocks':initial_detail,
                'n_actual_evaluations':calls,'invalid_trials':invalid,'elapsed_s':time.monotonic()-t0,
                'loss':'four equal phases, three equal bareSET groups, two continuous 1T1R pairs; weighted training log means with per-record eligibility',
                'loss_note':'record variance is a model-independent constant; final published RMSE must include it',
                'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    [Path(__file__),ROOT/'vulcan2d/shared_gap.py',ROOT/'vulcan2d/shared_gap_ivp.py',DATA,OLD/'gap_joint_fit.json']}}
        (OUT/args.output).write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'completed':row}),flush=True)


if __name__=='__main__':main()
