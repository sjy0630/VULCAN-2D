"""Train-only block calibration to avoid SET-switch Jacobian trapping RESET.

The material family is unchanged. RESET-only common parameters are optimized
while the numerically sharp bare SET switching parameters remain frozen.
All four phase/group losses and two continuously propagated pairs remain in
the objective. This is an optimizer intervention, not a new mechanism.
"""
from pathlib import Path
from dataclasses import asdict
import argparse
import hashlib
import json
import time
import numpy as np
from scipy.optimize import least_squares
from refine_shared_gap import JointProblem, OUT, NAMES, LOWER, UPPER, SCALES, law_from_q, DATA, ROOT

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--max-nfev',type=int,default=70)
    args=parser.parse_args()
    source=OUT/'accurate_fit.json'
    original=json.loads(source.read_text())
    q=np.array(original['fits'][original['selected_start']]['q'])
    p=JointProblem(state_rtol=2e-8,state_atol=2e-10,current_rtol=2e-5)
    indices=np.array([3,5,9])
    base=q.copy(); count=0; invalid=[]; start=time.monotonic()
    initial=p.residual(q); size=len(initial)
    best=(float(initial@initial/2),q.copy())
    def fun(x):
        nonlocal count,best
        trial=base.copy();trial[indices]=x;count+=1
        try:r=p.residual(trial)
        except (RuntimeError,ValueError,FloatingPointError,OverflowError) as exc:
            invalid.append({'q':trial.tolist(),'error':str(exc)})
            r=np.full(size,100./np.sqrt(size))
        cost=float(r@r/2)
        if cost<best[0]:best=(cost,trial.copy())
        if count%10==1: print(json.dumps({'calls':count,'seconds':time.monotonic()-start,'best_cost':best[0],'cost':cost}),flush=True)
        return r
    result=least_squares(fun,q[indices],bounds=(LOWER[indices],UPPER[indices]),
        x_scale=SCALES[indices],diff_step=1e-4,max_nfev=args.max_nfev,
        ftol=1e-8,xtol=1e-8,gtol=2e-6)
    final=base.copy();final[indices]=result.x
    checked,detail=p.residual(final,True)
    row={'start':0,'q':final.tolist(),'success':bool(result.success),'message':str(result.message),
         'nfev':int(result.nfev),'njev':int(result.njev),'optimality':float(result.optimality),
         'valid':True,'verified_bias_cost':float(checked@checked/2),'blocks':detail,
         'active_bounds':result.active_mask.tolist(),'optimized_indices':indices.tolist()}
    report={'status':'same_family_reset_block_recalibration','parameter_names':NAMES,
        'shared_ron_ohm':p.ron,'split':p.split,'settings':vars(args),'integrator_options':p.options,
        'fits':[row],'selected_start':0,'selected_law':asdict(law_from_q(final,p.ron)),
        'initial_bias_cost':float(initial@initial/2),'invalid_trials':invalid,
        'n_actual_evaluations':count,'elapsed_s':time.monotonic()-start,
        'loss':'unchanged four-phase two-pair training surrogate; common RESET parameters only optimized',
        'limitations':['No claim of global optimum; all-parameter and chronological limitations remain.',
                        'No new microscopic model or per-device parameter was introduced.'],
        'sources':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in
            [Path(__file__),source,ROOT/'analysis/refine_shared_gap.py',ROOT/'vulcan2d/shared_gap.py',
             ROOT/'vulcan2d/shared_gap_ivp.py',DATA]}}
    (OUT/'block_fit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(row),flush=True)

if __name__=='__main__':main()
