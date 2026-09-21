"""Grouped all53 experiment for a reciprocal four-state charge network."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'analysis'))
from compare_parallel_paths import assemble, VOLTAGE, FOLDS
from compare_side_charge_readout import scores
from vulcan2d.coupled_charge import stationary, predict, charging

LO = np.array([-.5,.01,.01,0.,-8.,-8.])
HI = np.array([.5,.99,.99,1.,8.,8.])
STARTS = ([.08,.12,.7,.3,1.,0.], [.10,.08,.12,.5,-2.,0.],
          [.02,.05,.1,.8,3.,-4.], [.15,.7,.1,.7,-3.,3.])


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def amplitude(y, shape, weights):
    result = np.sum(weights*(y-shape), axis=1)/weights.sum(axis=1)
    return np.clip(result, -12., -2.), (result > -12.) & (result < -2.)


def fit(v, y, w, *, fixed_shared=None, max_nfev=320):
    n,m = y.shape; offset = 6 if fixed_shared is None else 0
    lower = np.r_[LO[:offset], np.full(n,-.6)]; upper = np.r_[HI[:offset], np.full(n,.6)]
    sw = np.sqrt(w/(n*w.sum(axis=1)[:,None])); cache = {}
    def evaluate(x):
        if 'x' not in cache or not np.array_equal(x, cache['x']):
            shared = x[:offset] if offset else np.asarray(fixed_shared)
            result = stationary(v[None,:], shared, x[offset:,None])
            shape = result['log10_abs_current_scale']; a, inside = amplitude(y,shape,w)
            grad = result['derivatives'].copy()
            grad -= inside[:,None,None]*np.sum((w/w.sum(axis=1)[:,None])[:,:,None]*grad,axis=1)[:,None,:]
            cache.update(x=x.copy(),shared=shared,shape=shape,amplitude=a,grad=grad)
        return cache
    def residual(x):
        c=evaluate(x)
        return ((c['shape']+c['amplitude'][:,None]-y)*sw).ravel()
    def jacobian(x):
        c=evaluate(x); result=np.zeros((n*m,n+offset))
        if offset:result[:,:offset]=(c['grad'][...,:6]*sw[:,:,None]).reshape(n*m,6)
        result[np.arange(n*m),np.repeat(np.arange(n)+offset,m)]=(c['grad'][...,6]*sw).ravel()
        return result
    attempts=[]
    for shared in ([fixed_shared] if fixed_shared is not None else STARTS):
        grid=np.linspace(-.599,.599,41)
        shape=stationary(v[None,:],shared,grid[:,None])['log10_abs_current_scale']
        energy=[]
        for obs,weight in zip(y,w):
            a,_=amplitude(obs[None,:],shape,np.broadcast_to(weight,shape.shape))
            cost=np.sum(weight*(shape+a[:,None]-obs)**2,axis=1)
            energy.append(grid[np.argmin(cost)])
        initial=np.r_[np.asarray(shared)[:offset],energy]
        beginning=time.perf_counter()
        out=least_squares(residual,np.clip(initial,lower+1e-8,upper-1e-8),jac=jacobian,bounds=(lower,upper),
                          x_scale=np.r_[np.array([.15,.25,.25,.3,3.,3.])[:offset],np.full(n,.1)],
                          max_nfev=max_nfev,ftol=3e-8,xtol=3e-8,gtol=3e-8)
        attempts.append({'parameters':out.x.tolist(),'rmse_decade':float(np.sqrt(2*out.cost)),
                         'converged':bool(out.success),'nfev':out.nfev,'message':out.message,
                         'optimality':float(out.optimality),'seconds':time.perf_counter()-beginning})
    best=min(attempts,key=lambda a:a['rmse_decade']);c=evaluate(np.array(best['parameters']))
    return {'shared_parameters':c['shared'].tolist(),
            'states':np.column_stack((c['amplitude'],best['parameters'][offset:])).tolist(),
            'rmse_decade':best['rmse_decade'],'converged':best['converged'],'multistart':attempts,
            'conditional_mutual_energy_ev':float(charging(c['shared'])[0]),
            'shared_boundary_indices':np.flatnonzero(np.minimum(c['shared']-LO,HI-c['shared'])/(HI-LO)<1e-5).tolist()}


def run(quick=False):
    previous=ROOT/'analysis/side_charge_readout.json';archive=json.loads(previous.read_text())
    for p,h in archive['input_sha256'].items():
        if digest(ROOT/p)!=h:raise RuntimeError('Archived source changed: '+p)
    ids,y,w,selection=assemble()
    files=[Path(__file__),ROOT/'vulcan2d/coupled_charge.py',ROOT/'tests/test_coupled_charge.py',
           ROOT/'analysis/COUPLED_CHARGE_NETWORK_DESIGN_2026-09-21.md',previous]+[ROOT/p for p in archive['input_sha256']]
    hashes={str(p.relative_to(ROOT)):digest(p) for p in dict.fromkeys(files)}
    path=ROOT/'analysis'/('coupled_charge_comparison_quick.json' if quick else 'coupled_charge_comparison.json')
    result={'scope':'Four-state reciprocal charge readout; retained structural energy has no evolution law yet',
            'status':'running','input_sha256':hashes,'ids':ids.tolist(),'voltage_V':VOLTAGE.tolist(),
            'observed_log10_current_A':y.tolist(),'weights':w.tolist(),'selection':selection,'splits':{},
            'shared_parameter_order':['E_d_eV','x_d','x_t','U_fraction','ln_Gamma_L_over_R','ln_Gamma_t_over_R'],
            'state_order':['log10_current_scale_A','E_t_eV'],'quick_without_holdouts':quick}
    if path.exists():
        result=json.loads(path.read_text())
        if result['input_sha256']!=hashes:raise RuntimeError('Preserve existing results before source changes')
    def save():path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    splits=[('all_descriptive',np.ones(len(ids),bool),np.ones(len(ids),bool))]
    if not quick:
        splits += [(f'fold_{lo}_{hi}',(ids<lo)|(ids>hi),(ids>=lo)&(ids<=hi)) for lo,hi in FOLDS]
        splits += [('chronological',ids<=35,ids>35)]
    for label,train,target in splits:
        row=result['splits'].setdefault(label,{'training_ids':ids[train].tolist(),'target_ids':ids[target].tolist(),'targets':{}})
        if 'fit' not in row:
            print('Fitting',label,flush=True)
            row['fit']=fit(VOLTAGE,y[train],w[train]);save()
            print(label,row['fit']['rmse_decade'],row['fit']['shared_parameters'],
                  [(a['nfev'],a['rmse_decade'],a['converged']) for a in row['fit']['multistart']],flush=True)
        found=row['fit'];prediction=predict(VOLTAGE,found['shared_parameters'],found['states'])
        row['training_scores']=scores(y[train],prediction,w[train],ids[train])
        if label=='all_descriptive':row['predicted_log10_A']=prediction.tolist()
        else:
            for protocol,mask in [('SET_only',VOLTAGE>0),('joint_descriptive',np.ones(len(VOLTAGE),bool))]:
                if protocol in row['targets']:continue
                local=fit(VOLTAGE[mask],y[target][:,mask],w[target][:,mask],fixed_shared=found['shared_parameters'])
                pred=predict(VOLTAGE,found['shared_parameters'],local['states'])
                row['targets'][protocol]={'identification':local,'predicted_log10_A':pred.tolist(),
                                         'scores':scores(y[target],pred,w[target],ids[target])};save()
                print(label,protocol,row['targets'][protocol]['scores'],flush=True)
        save()
    if not quick:
        for protocol in ['SET_only','joint_descriptive']:
            pool=np.empty_like(y)
            for label,row in result['splits'].items():
                if label.startswith('fold_'):pool[np.array(row['target_ids'])-1]=row['targets'][protocol]['predicted_log10_A']
            result[protocol+'_five_fold_scores']=scores(y,pool,w,ids)
    result['status']='complete';save();print(path,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--quick',action='store_true')
    run(parser.parse_args().quick)
