"""Training-only conditional frozen-LRS shape screen, NOT cycle prediction.

Uses training geometric-mean SET return and low-bias RESET outward curves.
Deembedding uses observed mean current and is explicitly diagnostic. No
state kinetics, bare-HRS fit, test observations or per-cycle states are used.
"""
from pathlib import Path
from dataclasses import asdict
import sys,json,time,hashlib
import numpy as np
from scipy.optimize import least_squares,brentq

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from vulcan2d.transistor import measured_transistor_mapping
from spectral_readout import spectral_current

OUT=Path(__file__).parent
DATA=ROOT/'analysis/shared_hbn_2026-11-03/data/native_observations.npz'
REFERENCE=ROOT/'analysis/shared_gap_refinement_2026-11-03/block_fit.json'

def build_data():
    ref=json.loads(REFERENCE.read_text()); ron=ref['shared_ron_ohm'];mapping=measured_transistor_mapping()
    groups={}
    with np.load(DATA) as raw:
        for phase in ['set','reset']:
            ids=ref['split']['1t1r']['train']
            v=raw[f'1t1r_cycle_{ids[0]}_{phase}_voltage']
            observations=np.abs(np.array([raw[f'1t1r_cycle_{cid}_{phase}_current'] for cid in ids]))
            assert all(np.array_equal(v,raw[f'1t1r_cycle_{cid}_{phase}_voltage']) for cid in ids)
            geometric=10**np.mean(np.log10(np.maximum(observations,1e-300)),axis=0)
            apex=int(np.argmax(abs(v))); index=np.arange(len(v))
            use=((index>apex)&(v>=.06)) if phase=='set' else ((index<=apex)&(abs(v)>=.06)&(abs(v)<=.30))
            vv=v[use]; ii=geometric[use];side=phase
            transistor=np.array([mapping.voltage_for_current(float(i),1.1,side=side) for i in ii])
            hbn=np.sign(vv)*(abs(vv)-transistor)
            if np.any(abs(vv)-transistor<=0):raise ValueError('Invalid deembedded mean point')
            groups[phase]={'command':vv,'current':ii,'hbn':hbn,'observed':observations[:,use]}
    return groups,ron,mapping,ref['split']['1t1r']['train']

def evaluate(q,groups,ron,order=128):
    gamma,epsilon,eta,width=10**q[0],q[1],q[2],q[3]
    residual=[];metrics={}
    for phase,d in groups.items():
        pred=np.abs(spectral_current(d['hbn'],gamma,epsilon,width,eta,ron,order=order))
        if np.any(pred<=0):raise ValueError('Nonpositive readout prediction')
        errors=np.log10(pred/d['current'])
        residual.append(errors/np.sqrt(len(errors)))
        metrics[phase]={'mean_curve_rmse_decade':float(np.sqrt(np.mean(errors**2))),
                        'max_abs_error_decade':float(np.max(abs(errors))),
                        'predicted_at_observed_internal_voltage':pred.tolist()}
    return np.concatenate(residual),metrics

def main():
    groups,ron,mapping,train=build_data();t0=time.monotonic();results={}
    for family in ['single_level','gaussian_ensemble']:
        rows=[]
        seeds=[[-3.,.12,.85,.0],[-2.8,.5,.25,.25],[-2.8,.6,.5,.6],[-3.,-.1,.7,.3]]
        for index,seed in enumerate(seeds):
            active=[0,1,2] if family=='single_level' else [0,1,2,3]
            initial=np.array(seed)[active]
            lower=np.array([-7.,-2.,.001,0.])[active]
            upper=np.array([.7,2.,.999,1.5])[active]
            def unpack(x):
                q=np.zeros(4);q[active]=x;return q
            def fun(x):return evaluate(unpack(x),groups,ron)[0]
            result=least_squares(fun,initial,bounds=(lower,upper),x_scale='jac',max_nfev=160,
                                 diff_step=1e-4,ftol=1e-8,xtol=1e-8,gtol=1e-7)
            q=unpack(result.x);error,metrics=evaluate(q,groups,ron)
            fine,fine_metrics=evaluate(q,groups,ron,order=512)
            max_refine=max(float(np.max(abs(np.log10(np.array(metrics[p]['predicted_at_observed_internal_voltage'])/
                                np.array(fine_metrics[p]['predicted_at_observed_internal_voltage']))))) for p in groups)
            row={'start':index,'q':q.tolist(),'parameters':{'gamma_ev':10**q[0],'epsilon_ev':q[1],
                'eta':q[2],'width_ev':q[3],'ron_ohm':ron,'temperature_k':300.},
                'cost':float(error@error/2),'fine_cost':float(fine@fine/2),'metrics':fine_metrics,
                'quadrature_max_current_change_decade':max_refine,'success':bool(result.success),
                'message':str(result.message),'nfev':result.nfev,'optimality':float(result.optimality)}
            rows.append(row)
            print(json.dumps({k:v for k,v in row.items() if k!='metrics'}),flush=True)
            results[family]={'fits':rows,'selected_start':min(range(len(rows)),key=lambda j:rows[j]['fine_cost'])}
            output={'status':'conditional_frozen_readout_shape_screen_not_dynamic_prediction',
                'training_ids':train,'families':results,'data':{p:{k:v.tolist() for k,v in d.items()} for p,d in groups.items()},
                'assumptions':['One frozen effective LRS shared by SET return>=60mV and RESET outward60–300mV.',
                    'Default measured transistor mapping and300K temperature are conditional assumptions.',
                    'Geometric means deembedded using observed current are a diagnostic; full native/circuit propagation remains required.',
                    'OneGaussian energywidth is a candidate disorder assumption, not measured DOS.',
                    'No kinetics, bareHRS, fullRESET or test data are fitted; passing is not a unified model success.'],
                'elapsed_s':time.monotonic()-t0,'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    [Path(__file__),OUT/'spectral_readout.py',DATA,REFERENCE,ROOT/'vulcan2d/transistor.py']}}
            (OUT/'spectral_readout_screen.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
