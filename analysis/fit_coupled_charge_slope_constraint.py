"""Post-hoc shape/current tradeoff; avoids rejecting a class from one optimum.

Slope residuals reuse the same observed positive points. Their penalty is an
explicit diagnostic choice, not independent evidence or a noise calibration.
No primary held-out forecast is overwritten.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'analysis'))
from compare_parallel_paths import assemble,VOLTAGE
from compare_side_charge_readout import scores
from compare_coupled_charge import LO,HI,amplitude
from vulcan2d.coupled_charge import stationary


def run():
    source=ROOT/'analysis/coupled_charge_comparison.json';main=json.loads(source.read_text())
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for p,h in main['input_sha256'].items():
        if digest(ROOT/p)!=h:raise RuntimeError('Input changed: '+p)
    ids,y,w,_=assemble();n,m=y.shape;sw=np.sqrt(w/n);slope_operator=np.zeros_like(w)
    for i,weight in enumerate(w):
        use=(VOLTAGE>=2)&(VOLTAGE<=3)&(weight>0);x=np.log10(VOLTAGE[use]);x-=x.mean()
        slope_operator[i,use]=x/(x@x)
    measured_slopes=np.sum(slope_operator*y,axis=1);span=np.log10(3/2)
    lower=np.r_[LO,np.full(n,-.6)];upper=np.r_[HI,np.full(n,.6)]
    base=main['splits']['all_descriptive']['fit'];starts=[a['parameters'] for a in base['multistart'][:2]]
    files=[Path(__file__),source]+[ROOT/p for p in main['input_sha256']]
    result={'scope':'Post-hoc current/positive-slope Pareto check, all data consumed; no new held-out prediction',
            'penalty_definition':'Add sqrt(lambda/n)*log10(3/2)*(predicted-observed local log-log slope) for each pair; same data reused.',
            'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in dict.fromkeys(files)},'fits':[]}
    for penalty in (1.,10.,100.):
        cache={}
        def evaluate(x):
            if 'x' not in cache or not np.array_equal(cache['x'],x):
                raw=stationary(VOLTAGE[None,:],x[:6],x[6:,None]);shape=raw['log10_abs_current_scale']
                a,inside=amplitude(y,shape,w);grad=raw['derivatives'];sj=np.sum(slope_operator[:,:,None]*grad,axis=1)
                profiled=grad-inside[:,None,None]*np.sum(w[:,:,None]*grad,axis=1)[:,None,:]
                cache.update(x=x.copy(),prediction=shape+a[:,None],amplitude=a,grad=profiled,
                             slope=np.sum(slope_operator*shape,axis=1),slope_grad=sj)
            return cache
        scale=np.sqrt(penalty/n)*span
        def residual(x):
            c=evaluate(x)
            return np.r_[((c['prediction']-y)*sw).ravel(),scale*(c['slope']-measured_slopes)]
        def jacobian(x):
            c=evaluate(x);j=np.zeros((n*m+n,6+n))
            j[:n*m,:6]=(c['grad'][...,:6]*sw[:,:,None]).reshape(n*m,6)
            j[np.arange(n*m),np.repeat(np.arange(n)+6,m)]=(c['grad'][...,6]*sw).ravel()
            j[n*m:,:6]=c['slope_grad'][:,:6]*scale
            j[n*m+np.arange(n),6+np.arange(n)]=c['slope_grad'][:,6]*scale
            return j
        attempts=[]
        for initial in starts:
            out=least_squares(residual,np.clip(initial,lower+1e-8,upper-1e-8),jac=jacobian,bounds=(lower,upper),
                              x_scale=np.r_[[.15,.25,.25,.3,3.,3.],np.full(n,.1)],max_nfev=500,
                              ftol=3e-8,xtol=3e-8,gtol=3e-8)
            attempts.append({'parameters':out.x.tolist(),'converged':bool(out.success),'nfev':out.nfev,
                             'penalized_cost':float(out.cost),'optimality':float(out.optimality)})
        best=min(attempts,key=lambda a:a['penalized_cost']);c=evaluate(np.array(best['parameters']))
        row={'penalty':penalty,'shared_parameters':best['parameters'][:6],
             'states':np.column_stack((c['amplitude'],best['parameters'][6:])).tolist(),
             'converged':best['converged'],'multistart':attempts,
             'unpenalized_scores':scores(y,c['prediction'],w,ids),
             'predicted_slopes':c['slope'].tolist(),'observed_slopes':measured_slopes.tolist(),
             'slope_rmse':float(np.sqrt(np.mean((c['slope']-measured_slopes)**2))),
             'predicted_log10_A':c['prediction'].tolist()}
        result['fits'].append(row)
        (ROOT/'analysis/coupled_charge_slope_tradeoff.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print('penalty',penalty,'scores',row['unpenalized_scores'],'slope',np.quantile(c['slope'],[.1,.5,.9]),
              'converged',best['converged'],flush=True)


if __name__=='__main__':run()
