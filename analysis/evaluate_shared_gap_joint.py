"""Full-grid, no-refit evaluation of the common 1R/1T1R gap candidate."""
from __future__ import annotations
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
from fit_shared_gap_joint import FOLDER, DATA, load_data, material, pair
from vulcan2d.shared_gap import BareGapCircuit, LookupGapCircuit, simulate_gap
from vulcan2d.transistor import measured_transistor_mapping


def score(v, obs, pred, phase, selection=None):
    floor=1e-11 if phase=='set' else 1e-12
    obs,pred=np.abs(obs),np.abs(pred)
    use=(abs(v)>1e-12)&np.isfinite(obs)&(obs>=floor)
    if selection is not None: use &= selection
    invalid=use & (~np.isfinite(pred)|(pred<=0))
    eligible=use & ~invalid
    errors=np.log10(pred[eligible]/obs[eligible])
    return {'n':int(np.count_nonzero(use)), 'invalid_predictions':int(np.count_nonzero(invalid)),
            'rmse_decade':float(np.sqrt(np.mean(errors**2))) if len(errors) and not np.any(invalid) else None,
            'median_abs_error_decade':float(np.median(abs(errors))) if len(errors) else None}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--step',type=float,default=.001)
    args=parser.parse_args()
    report=json.loads((FOLDER/'gap_joint_fit.json').read_text())
    if report['selected_start'] is None: raise RuntimeError('No valid fitted candidate')
    fit=report['fits'][report['selected_start']]
    data,split,ron=load_data()
    law=material(fit['q'],ron)
    q=fit['q']
    trans=LookupGapCircuit(measured_transistor_mapping(),1.1)
    circuits={'1r':(BareGapCircuit(1e-3),BareGapCircuit()),'1t1r':(trans,trans)}
    predictions={}
    state_history=[]
    raw=np.load(DATA)
    # Bare records have unknown continuous chronology: common initial state
    # per record, with SET-to-RESET propagation within the conditional pair.
    voltage=[data['1r','train',p][0] for p in ('set','reset')]
    bare=pair(law,q[7],voltage,circuits['1r'],args.step)
    for cid in split['1r']['train']+split['1r']['test']:
        for phase,result in zip(('set','reset'),bare): predictions['1r',cid,phase]=result
    # One cycle4 initial state; never reset at train/test or cycle boundaries.
    u=q[8]
    cache={}
    for cid in range(4,54):
        voltage=[raw[f'1t1r_cycle_{cid}_{p}_voltage'] for p in ('set','reset')]
        key=(u,voltage[0].tobytes(),voltage[1].tobytes())
        if key not in cache: cache[key]=pair(law,u,voltage,circuits['1t1r'],args.step)
        results=cache[key]
        state_history.append({'cycle':cid,'initial_u':u,'after_set_u':results[0]['final_u'],
                              'after_reset_u':results[1]['final_u']})
        for phase,result in zip(('set','reset'),results): predictions['1t1r',cid,phase]=result
        u=results[-1]['final_u']
    metrics={}
    arrays={}
    for top,groups in split.items():
        metrics[top]={}
        for group,ids in groups.items():
            metrics[top][group]={}
            for phase in ('set','reset'):
                v,obs=data[top,group,phase]
                pred=np.array([predictions[top,cid,phase]['current_a'] for cid in ids])
                vv=np.broadcast_to(v,obs.shape)
                values={'all_eligible':score(vv,obs,pred,phase)}
                if top=='1r' and phase=='set':
                    values['nonlimited']=score(vv,obs,pred,phase,obs<.995e-3)
                    values['limited']=score(vv,obs,pred,phase,obs>=.995e-3)
                if top=='1r' and phase=='reset':
                    values['without_initial_25mV']=score(vv,obs,pred,phase,
                        ~(np.isclose(vv,-.025,atol=1e-6)&(np.arange(len(v))<=np.argmax(abs(v)))))
                    if 24 in ids:
                        keep=np.broadcast_to(np.array(ids)[:,None]!=24,obs.shape)
                        values['without_pair24']=score(vv,obs,pred,phase,keep)
                median=10.**np.median(np.log10(np.maximum(obs,1e-15)),axis=0)
                median_pred=np.median(abs(pred),axis=0)
                values['median_curve']=score(v,median,median_pred,phase)
                metrics[top][group][phase]=values
                arrays[f'{top}_{group}_{phase}_voltage']=v
                arrays[f'{top}_{group}_{phase}_observed']=obs
                arrays[f'{top}_{group}_{phase}_predicted']=pred
    voltage=data['1r','train','set'][0]
    cfbare=simulate_gap(voltage,q[7],law,BareGapCircuit(1e-3),max_step_v=args.step)
    cft=simulate_gap(voltage,q[7],law,trans,max_step_v=args.step)
    def readout(result):
        state=result['final_u']
        return {'final_u':state,'intrinsic_read_R_at_0p05V_ohm':.05/law.current(.05,state),
                'intrinsic_small_signal_R_ohm':ron+law.v0_v*state*np.exp(state)/law.i0_a,
                'max_current_A':float(np.max(abs(result['current_a'])))}
    cf={'protocol':'identical initial u and 0 to5 to0 waveform; only external circuit differs',
        '1r':readout(cfbare),'1t1r':readout(cft)}
    for phase,result in zip(('set','reset'),bare):
        for key in ('current_a','state_u','hbn_voltage_v','transistor_voltage_v','compliance_drop_v'):
            arrays[f'1r_model_{phase}_{key}']=result[key]
    for cid in (4,35,36,53):
        for phase in ('set','reset'):
            for key in ('current_a','state_u','hbn_voltage_v','transistor_voltage_v'):
                arrays[f'1t1r_cycle{cid}_{phase}_{key}']=predictions['1t1r',cid,phase][key]
    output={'status':'single_family_joint_calibration_evaluated_not_unique_mechanism',
            'shared_material':asdict(law),'initial_u':{'1r_records':q[7],'1t1r_cycle4':q[8]},
            'split':split,'metrics':metrics,'continuous_1t1r_history':state_history,
            'same_initial_state_circuit_counterfactual':cf,'selected_training_start':fit['start'],
            'optimizer':{key:fit[key] for key in ('success','message','optimality','nfev')},
            'evaluation_max_step_v':args.step,
            'limitations':report['limitations']+[
                'Ron is anchored by training low-bias resistance, not independently predicted.',
                'The microscopic gap and physical time are not identified.',
                'The primary prediction is current magnitude; measured RESET sign is not an independent validation.',
                'Final numerical acceptance depends on separate independent validation.',
                'Bare pair24 mismatch and initial -25mV RESET anomaly are reported, not silently removed.'],
            'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (Path(__file__),ROOT/'vulcan2d/shared_gap.py',DATA,FOLDER/'gap_joint_fit.json')}}
    (FOLDER/'gap_joint_evaluation.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(FOLDER/'gap_joint_evaluation.npz',**arrays)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for ax,(top,phase) in zip(axes.flat,[('1r','set'),('1r','reset'),('1t1r','set'),('1t1r','reset')]):
        for group,color in [('train','#728899'),('test','#e4a75e')]:
            v,obs=data[top,group,phase]
            quant=np.quantile(np.log10(np.maximum(obs,1e-15)),[.1,.5,.9],axis=0)
            ax.fill_between(v,10**quant[0],10**quant[2],color=color,alpha=.2)
            ax.semilogy(v,10**quant[1],color=color,lw=1.4,label=f'{group} median (10-90%)')
        for cid,color,label,ls in ([(split[top]['train'][0],'#bb2222','shared model','-')] if top=='1r'
                else [(4,'#8855aa','model cycle 4','--'),(53,'#bb2222','model cycle 53','-')]):
            r=predictions[top,cid,phase]
            ax.semilogy(v,np.maximum(abs(r['current_a']),1e-15),color=color,lw=1.7,ls=ls,label=label)
        ax.set(title=f'{top.upper()} {phase.upper()}',xlabel='Command voltage (V)',ylabel='|Current| (A)')
        ax.set_ylim(1e-12,2e-2 if top=='1r' and phase=='reset' else 2e-3)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('One shared material law: full-grid joint comparison\nVoltage-path candidate; not a transient or microscopic validation',fontsize=13)
    fig.savefig(FOLDER/'gap_joint_comparison.png',dpi=170)
    fig.savefig(FOLDER/'gap_joint_comparison.pdf')
    print(json.dumps({'metrics':metrics,'counterfactual':cf,'history_first_last':[state_history[0],state_history[-1]]},indent=2),flush=True)


if __name__=='__main__': main()
