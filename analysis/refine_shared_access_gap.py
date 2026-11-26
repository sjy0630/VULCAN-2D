"""One nested, shared access-voltage correction motivated by training evidence."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import least_squares, brentq

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
from refine_shared_gap import JointProblem, law_from_q, LOWER, UPPER, SCALES, NAMES, OUT, OLD, DATA
from vulcan2d.shared_access_gap import AccessGapLaw
from vulcan2d.shared_gap_ivp import simulate_gap_ivp


def access_law(q,ron):
    return AccessGapLaw(**asdict(law_from_q(q[:10],ron)),access_voltage_v=10.**float(q[10]))


class AccessProblem(JointProblem):
    def trajectory(self,q,topology,repeats=1):
        law=access_law(q,self.ron);u=float(q[7 if topology=='1r' else 8]);traces=[]
        for repeat in range(repeats):
            phases=[]
            for phase,port in zip(('set','reset'),self.circuits[topology]):
                result=simulate_gap_ivp(self.waveforms[topology,phase],u,law,port,**self.options)
                u=result['final_u'];phases.append(result)
            traces.append(phases)
        return traces


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--max-nfev',type=int,default=100)
    parser.add_argument('--starts',type=int,default=2)
    parser.add_argument('--diff-step',type=float,default=1e-3)
    parser.add_argument('--output',default='access_fit.json')
    args=parser.parse_args();OUT.mkdir(exist_ok=True)
    problem=AccessProblem()
    baseline=json.loads((OUT/'accurate_fit.json').read_text())
    base=np.array(baseline['fits'][baseline['selected_start']]['q'])
    law=law_from_q(base,problem.ron)
    seeds=[]
    # For each declared access amplitude, preserve the training bare forming
    # drive estimate in the initial seed; this is not a test-data adjustment.
    for b in (.1,1.):
        q=np.r_[base,np.log10(b)]
        target=2.675/q[7]
        j=brentq(lambda logj: law.v0_v*np.arcsinh(10.**logj/law.i0_a)+b*(10.**logj/1e-3)**3-target,
                  -12.,-1.)
        q[2]=j
        seeds.append(q)
    lower,upper=np.r_[LOWER,-6.],np.r_[UPPER,2.]
    scales=np.r_[SCALES,2.]
    calls=0;invalid=[];best=np.inf;t0=time.monotonic();fits=[]
    initial=problem.residual(seeds[0]);size=len(initial)
    def residual(q):
        nonlocal calls,best
        calls+=1
        try:error=problem.residual(q)
        except (ValueError,RuntimeError,OverflowError,FloatingPointError) as exc:
            invalid.append({'call':calls,'error':str(exc),'q':q.tolist()})
            error=np.full(size,100./np.sqrt(size))
        norm=float(np.linalg.norm(error));best=min(best,norm)
        if calls%25==1:print(json.dumps({'calls':calls,'seconds':round(time.monotonic()-t0,1),
            'best_norm':best,'current_norm':norm,'invalid_trials':len(invalid)}),flush=True)
        return error
    for index,seed in enumerate(seeds[:args.starts]):
        result=least_squares(residual,np.clip(seed,lower+1e-7,upper-1e-7),bounds=(lower,upper),
            x_scale=scales,diff_step=args.diff_step,max_nfev=args.max_nfev,
            ftol=2e-7,xtol=2e-7,gtol=2e-6)
        try:
            checked,detail=problem.residual(result.x,True)
            valid=True;cost=float(np.dot(checked,checked)/2.);reason=None
        except (ValueError,RuntimeError,OverflowError,FloatingPointError) as exc:
            valid=False;cost=None;detail={};reason=str(exc)
        row={'start':index,'q':result.x.tolist(),'success':bool(result.success),'message':str(result.message),
             'nfev':int(result.nfev),'njev':int(result.njev),'optimality':float(result.optimality),
             'valid':valid,'verified_bias_cost':cost,'validation_error':reason,'blocks':detail,
             'singular_values_jacobian':np.linalg.svd(result.jac,compute_uv=False).tolist(),
             'active_bounds':result.active_mask.tolist()}
        fits.append(row)
        chosen=min((r for r in fits if r['valid']),key=lambda r:r['verified_bias_cost'],default=None)
        report={'status':'shared_access_voltage_candidate_fitted' if chosen else 'no_valid_fit',
            'parameter_names':NAMES+['log10_access_voltage_V'],'shared_ron_ohm':problem.ron,
            'split':problem.split,'settings':vars(args),'integrator_options':problem.options,'fits':fits,
            'selected_start':chosen['start'] if chosen else None,
            'selected_law':asdict(access_law(chosen['q'],problem.ron)) if chosen else None,
            'n_actual_evaluations':calls,'invalid_trials':invalid,'elapsed_s':time.monotonic()-t0,
            'hypothesis':'one shared passive current-dependent series drop; cubic functional form and exp(3u) remain effective assumptions',
            'loss':'same training log-mean sufficient statistics and four-phase/group weighting as accurate baseline',
            'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [Path(__file__),ROOT/'analysis/refine_shared_gap.py',ROOT/'vulcan2d/shared_gap.py',
                 ROOT/'vulcan2d/shared_gap_ivp.py',ROOT/'vulcan2d/shared_access_gap.py',DATA,OUT/'accurate_fit.json']}}
        (OUT/args.output).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'completed':row}),flush=True)


if __name__=='__main__':main()
