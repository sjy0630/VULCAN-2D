"""Explicit supplemental fast-side start, kept separate from three primary starts."""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
from scipy.optimize import least_squares
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'analysis'))
from fit_retained_transport import digest
from compare_activated_configuration import assemble, lookups, scores
from check_retained_charge_contact import ratio34
from vulcan2d.retained_transport_parameters import pack, unpack, LO, HI, SCALE, NAMES
from vulcan2d.retained_charge_configuration import trajectory


def run(split):
    parent = ROOT/'analysis'/f'retained_charge_configuration_{split}_single_contact.json'
    source = json.loads(parent.read_text()); hashes = dict(source['input_sha256'])
    for name, sha in hashes.items(): assert digest(ROOT/name) == sha, name
    for path in [parent, Path(__file__), ROOT/'analysis/fit_retained_transport.py',
                 ROOT/'vulcan2d/retained_transport_parameters.py', ROOT/'analysis/RETAINED_SIDE_RATE_PROFILE_DESIGN_2026-09-21.md']:
        hashes[str(path.relative_to(ROOT))] = digest(path)
    initial = pack(source['shared_parameters'],source['parameters']); initial[13] = 6.
    initial = np.clip(initial, LO+1e-12, HI-1e-12)
    d = assemble(); lookup = lookups(); train = source['training_ids']; end = train[-1]*676
    mask = d['weights'][:end] > 0; weight = np.sqrt(d['weights'][:end][mask]/len(train))
    observed = np.log10(d['observed'][:end][mask]); settings = source['settings']
    output = ROOT/'analysis'/f'retained_transport_{split}_fast_start.json'
    if output.exists():
        prior = json.loads(output.read_text()); assert prior['input_sha256'] == hashes
        if prior['status'] == 'complete': return
    result = {'status':'running','scope':'Exploratory supplemental fast-side initialization, no event prescribed',
              'split':split,'training_ids':train,'target_ids':source['target_ids'],'settings':settings,
              'input_sha256':hashes,'parameter_names':list(NAMES),'bounds':[LO.tolist(),HI.tolist()],
              'parent_result':str(parent.relative_to(ROOT)),'initial':initial.tolist(),'starts':[],
              'initial_condition':'q=0, zero-bias conditional-equilibrium m'}
    def save():output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    save(); calls=[0]; worst=[0.]; start=time.perf_counter()
    def objective(x):
        s,p=unpack(x);r=trajectory(d['voltage'][:end],s,p,*lookup,**settings)
        worst[0]=max(worst[0],r['maximum_kcl_relative_error']);assert worst[0]<1e-8
        y=r['current_A'][mask];assert np.all(np.isfinite(y)&(y>0))
        residual=weight*(np.log10(y)-observed);calls[0]+=1
        if calls[0]%100==0:print(split,'fast',calls[0],float(np.linalg.norm(residual)),flush=True)
        return residual
    fit=least_squares(objective,initial,bounds=(LO,HI),x_scale=SCALE,diff_step=1e-5,
                      max_nfev=180,ftol=2e-7,xtol=2e-7,gtol=2e-7)
    s,p=unpack(fit.x)
    row={'index':0,'initial':initial.tolist(),'coordinates':fit.x.tolist(),'shared_parameters':s.tolist(),'parameters':p.tolist(),
         'rmse_decade':float(np.linalg.norm(fit.fun)),'converged':bool(fit.success),'status':int(fit.status),'message':fit.message,
         'nfev':fit.nfev,'actual_evaluations':calls[0],'seconds':time.perf_counter()-start,'optimality':float(fit.optimality),
         'active_bound_indices':np.flatnonzero(fit.active_mask!=0).tolist(),'maximum_kcl_relative_error':worst[0]}
    result['starts']=[row];save();print('FINISH',split,row,flush=True)
    r=trajectory(d['voltage'],s,p,*lookup,**settings,diagnostics=True)
    result.update(status='complete',selected_start=0,selected_converged=bool(fit.success),coordinates=fit.x.tolist(),
        shared_parameters=s.tolist(),parameters=p.tolist(),training_scores=scores(d,r['current_A'],train),
        target_scores=scores(d,r['current_A'],source['target_ids']),full_scores=scores(d,r['current_A']),
        maximum_kcl_relative_error=r['maximum_kcl_relative_error'],multiple_root_point_count=int(np.count_nonzero(r['root_count']>1)),
        SET_return_4_over_3=ratio34(d,r['current_A'],2),RESET_outward_4_over_3=ratio34(d,r['current_A'],4))
    np.savez_compressed(output.with_suffix('.npz'),**{k:d[k] for k in ('voltage','observed','cycle','window','weights','low_signal')},
        current=r['current_A'],internal_voltage=r['internal_voltage_V'],transistor_voltage=r['transistor_voltage_V'],
        states=r['states'],root_count=r['root_count'],quadrature_weights=r['quadrature_weights'])
    result['prediction_sha256']=digest(output.with_suffix('.npz'));save()
    print('COMPLETE',split,result['training_scores']['rmse_decade'],result['target_scores']['rmse_decade'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--split',required=True,choices=['all_descriptive','chronological'])
    run(parser.parse_args().split)
