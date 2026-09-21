"""Charging-energy profiles and conditional dielectric assumptions.

These are post-fit descriptive identifiability checks, not confidence intervals
or new held-cycle validation scores.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'analysis'))
from compare_contact_networks import assemble_pairs
from vulcan2d.contact_network import (ChargingAssumptions,fit_shared_shape,domain,
    initial_points,log10_shape)


def profile_u(pairs):
    name='finite_right';lo,hi=domain(name);vs=[];ys=[];groups=[];weights=[]
    for k,p in enumerate(pairs):
        for v,y in [(p.positive_voltage_v,p.positive_log10_current_a),(p.negative_voltage_v,p.negative_log10_current_a)]:
            vs.append(v);ys.append(y);groups.extend([k]*len(v));weights.extend([1/np.sqrt(2*len(pairs)*len(v))]*len(v))
    v=np.concatenate(vs);y=np.concatenate(ys);g=np.array(groups);w=np.array(weights);w2=w*w;den=np.bincount(g,weights=w2)
    results=[]
    for u in (0.,.05,.1,.2,.35,.5,.7,1.,1.5,2.,3.):
        def residual(pars):
            r=y-log10_shape(v,name,[*pars,u]);a=np.bincount(g,weights=r*w2)/den
            return (r-a[g])*w
        fits=[least_squares(residual,p[:4],bounds=(lo[:4],hi[:4]),x_scale=hi[:4]-lo[:4],
            max_nfev=500,ftol=2e-8,xtol=2e-8,gtol=2e-8) for p in initial_points(name)]
        good=[f for f in fits if f.success]
        if not good:raise RuntimeError('Fixed U profile failed')
        f=min(good,key=lambda f:f.cost)
        results.append({'U_eV':u,'parameters':[*f.x.tolist(),u],'rmse_decade':float(np.sqrt(2*f.cost)),
                        'n_converged_starts':len(good),'all_start_rmse_decade':[float(np.sqrt(2*s.cost)) for s in fits]})
    return results


def run():
    pairs,_=assemble_pairs();groups={}
    for tag,selected in [('all53',pairs),('cycles4to53',[p for p in pairs if p.cycle_id>=4])]:
        values=[]
        for eps in (2.,3.1,8.):
            for name in ('coulomb_left','coulomb_right'):
                f=fit_shared_shape(selected,name,ChargingAssumptions(dielectric_r=eps))
                values.append({'epsilon_assumption':eps,'candidate':name,'fit':f})
                print(tag,eps,name,round(f['profiled_pair_branch_equal_rmse_decade'],5),flush=True)
        groups[tag]={'dielectric_sensitivity':values,'right_slow_U_profile':profile_u(selected)}
    files=[Path(__file__),ROOT/'analysis/compare_contact_networks.py',ROOT/'vulcan2d/contact_network.py']
    result={'groups':groups,'limitations':['Descriptive refits of previously used data; not a new validation score.',
        'Point-charge dielectric estimate and minimum separation are assumptions, not measured interaction energies.',
        'Flat profiles or equivalent parametrizations prevent interpreting an optimizer value as a measured U.',
        'Negative differential slopes are diagnostic outputs; no monotone circuit solver was applied to such candidates.'],
        'input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    out=ROOT/'analysis/contact_charging_sensitivity.json';out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(out)


if __name__=='__main__':run()
