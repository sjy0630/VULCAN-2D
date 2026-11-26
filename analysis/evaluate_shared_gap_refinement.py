"""Full native-grid, no-refit evaluation of accurate/shared-access candidates.

Raw record scatter remains in every primary RMSE. The train/test division is
fixed before this evaluation, but these experiments have historical exposure
in project development and are not an untouched prospective validation set.
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vulcan2d.shared_gap import GapLaw, BareGapCircuit, LookupGapCircuit
from vulcan2d.shared_access_gap import AccessGapLaw
from vulcan2d.shared_gap_ivp import simulate_gap_ivp
from vulcan2d.transistor import measured_transistor_mapping

FOLDER=ROOT/'analysis/shared_gap_refinement_2026-11-03'
DATA=ROOT/'analysis/shared_hbn_2026-11-03/data/native_observations.npz'
FLOORS={'set':1e-11,'reset':1e-12}
OPTIONS={'method':'Radau','state_rtol':1e-7,'state_atol':1e-9,'current_rtol':1e-4}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score(voltage,observation,prediction,phase,selection=None):
    v,obs,pred=np.asarray(voltage),abs(np.asarray(observation)),abs(np.asarray(prediction))
    mask=(abs(v)>1e-12)&np.isfinite(obs)&(obs>=FLOORS[phase])
    if selection is not None:mask &= selection
    invalid=mask&(~np.isfinite(pred)|(pred<=0.))
    good=mask&~invalid
    error=np.log10(pred[good]/obs[good])
    usable=len(error)>0 and not np.any(invalid)
    return {'n_eligible':int(mask.sum()),'n_invalid_predictions':int(invalid.sum()),
            'rmse_decade':float(np.sqrt(np.mean(error**2))) if usable else None,
            'mae_decade':float(np.mean(abs(error))) if usable else None,
            'signed_mean_error_decade':float(np.mean(error)) if usable else None,
            'median_abs_error_decade':float(np.median(abs(error))) if usable else None,
            'p90_abs_error_decade':float(np.quantile(abs(error),.9)) if usable else None,
            'sum_squared_log_error':float(np.sum(error**2)) if usable else None}


def peak(voltage,current):
    index=int(np.argmax(abs(current)))
    return {'magnitude_a':float(abs(current[index])),'command_voltage_v':float(voltage[index]),'sample_index':index}


def combine(scores):
    total=sum(x['n_eligible'] for x in scores)
    invalid=sum(x['n_invalid_predictions'] for x in scores)
    squares=[x['sum_squared_log_error'] for x in scores]
    return {'n_eligible':total,'n_invalid_predictions':invalid,
            'pooled_raw_rmse_decade':float(np.sqrt(sum(squares)/total))
             if total and not invalid and all(x is not None for x in squares) else None}


def plot_candidate(label,raw,predictions,split,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for ax,(top,phase) in zip(axes.flat,[('1r','set'),('1r','reset'),('1t1r','set'),('1t1r','reset')]):
        for group,color in [('train','#416b89'),('test','#bd7b32')]:
            ids=split[top][group]
            vv=np.array([raw[top,cid,phase][0] for cid in ids])
            obs=np.array([abs(raw[top,cid,phase][1]) for cid in ids])
            for v,current in zip(vv,obs):
                use=(abs(v)>1e-12)&(current>0.)
                ax.semilogy(v[use],current[use],color=color,alpha=.07,lw=.5)
            med=10.**np.median(np.log10(np.maximum(obs,1e-300)),axis=0)
            use=(abs(vv[0])>1e-12)&(med>0.)
            ax.semilogy(vv[0][use],med[use],color=color,lw=1.5,label=f'{group} median')
        choices=[(split[top]['train'][0],'shared model','-')] if top=='1r' else [(4,'model cycle 4','--'),(53,'model cycle 53','-')]
        for cid,legend,style in choices:
            v=raw[top,cid,phase][0];current=abs(predictions[top,cid,phase]['current_a'])
            use=(abs(v)>1e-12)&(current>0.)
            ax.semilogy(v[use],current[use],color='#a32442',lw=1.8,ls=style,label=legend)
        ax.set(title=f'{top.upper()} {phase.upper()}',xlabel='Command voltage (V)',ylabel='|Current| (A)')
        ax.set_ylim(1e-12,2e-2 if (top,phase)==('1r','reset') else 2e-3)
        ax.grid(alpha=.18);ax.legend(fontsize=8)
    fig.suptitle(f'{label}: one shared material law, frozen-parameter evaluation\nRaw records retained; adaptive integration; historical test exposure',fontsize=12)
    fig.savefig(path.with_suffix('.png'),dpi=180)
    fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)


def comparison_if_ready():
    candidates={label:FOLDER/f'{label}_evaluation.json' for label in ('accurate','block','access','access_profile')}
    paths={label:path for label,path in candidates.items() if path.exists()}
    if len(paths)<2:return
    reports={label:json.loads(path.read_text()) for label,path in paths.items()}
    result={'status':'frozen_parameter_raw_score_comparison','sources':{k:sha(p) for k,p in paths.items()},
            'historical_test_exposure':True,'comparisons':{}}
    for reference,candidate in [('accurate','block'),('accurate','access'),('block','access'),
                                ('accurate','access_profile'),('block','access_profile'),('access','access_profile')]:
        if reference not in reports or candidate not in reports:continue
        changes={}
        for top in ('1r','1t1r'):
            changes[top]={}
            for group in ('train','test'):
                changes[top][group]={}
                for phase in ('set','reset'):
                    a=reports[reference]['metrics'][top][group][phase]['all_eligible']['rmse_decade']
                    b=reports[candidate]['metrics'][top][group][phase]['all_eligible']['rmse_decade']
                    changes[top][group][phase]={'reference_rmse_decade':a,'candidate_rmse_decade':b,
                        'delta_candidate_minus_reference_decade':b-a if a is not None and b is not None else None}
        result['comparisons'][candidate+'_vs_'+reference]=changes
    (FOLDER/'refinement_evaluation_comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=('accurate','block','access','access_profile'),default=None)
    parser.add_argument('--fit',default=None)
    parser.add_argument('--label',default=None)
    args=parser.parse_args()
    fitpath=FOLDER/(args.fit or f'{args.model or "accurate"}_fit.json')
    fitted=json.loads(fitpath.read_text())
    fit_hash=sha(fitpath)
    label=args.label or args.model or fitpath.stem.removesuffix('_fit')
    if label=='access' and len(fitted['fits'])<int(fitted['settings'].get('starts',2)):
        raise RuntimeError('Access fitting starts are incomplete; wait for the final fit file')
    if fitted.get('selected_start') is None:raise RuntimeError('No selected valid fit')
    selected=next(row for row in fitted['fits'] if row['start']==fitted['selected_start'])
    if not selected['valid']:raise RuntimeError('Selected fit is invalid')
    cls=AccessGapLaw if 'access_voltage_v' in fitted['selected_law'] else GapLaw
    law=cls(**fitted['selected_law']);q=selected['q'];split=fitted['split']
    options=dict(OPTIONS)
    for key in ('state_rtol','state_atol','current_rtol'):
        options[key]=min(options[key],fitted.get('integrator_options',{}).get(key,options[key]))
    source_check={}
    for file,expected in fitted['sources'].items():
        if file.startswith('vulcan2d/') or file==str(DATA.relative_to(ROOT)):
            source_check[file]=sha(ROOT/file)==expected
    if not all(source_check.values()):raise RuntimeError('Core model/data source drift since fitting')
    raw={}
    with np.load(DATA) as data:
        for top,groups in split.items():
            for cid in groups['train']+groups['test']:
                for phase in ('set','reset'):
                    raw[top,cid,phase]=(data[f'{top}_cycle_{cid}_{phase}_voltage'].copy(),
                                       data[f'{top}_cycle_{cid}_{phase}_current'].copy())
    trans=LookupGapCircuit(measured_transistor_mapping(),1.1)
    ports={'1r':(BareGapCircuit(.001),BareGapCircuit()),'1t1r':(trans,trans)}
    started=time.monotonic();predictions={};histories={'1r':[],'1t1r':[]};cache={}
    all_diagnostics=[]
    def run_pair(top,cid,u,command):
        results=[];initial=u
        for phase,port in zip(('set','reset'),ports[top]):
            voltage=raw[top,cid,phase][0]
            result=simulate_gap_ivp(voltage,u,law,port,initial_command_voltage_v=command,**options)
            all_diagnostics.append(result['metadata'])
            results.append(result)
            u,command=result['final_u'],float(voltage[-1])
            predictions[top,cid,phase]=result
        history={'cycle':cid,'initial_u':initial,'after_set_u':results[0]['final_u'],'after_reset_u':u,
                 'set_reset_handoff_error_u':results[1]['metadata']['initial_u']-results[0]['final_u']}
        return u,command,history,results
    for cid in split['1r']['train']+split['1r']['test']:
        voltage=[raw['1r',cid,p][0] for p in ('set','reset')]
        key=(voltage[0].tobytes(),voltage[1].tobytes())
        if key in cache:
            u,command,hist,results=cache[key]
            hist=dict(hist,cycle=cid)
            for phase,result in zip(('set','reset'),results):predictions['1r',cid,phase]=result
        else:
            u,command,hist,results=run_pair('1r',cid,float(q[7]),0.)
            cache[key]=(u,command,hist,results)
        histories['1r'].append(hist)
    u,command=float(q[8]),0.
    for cid in range(4,54):
        previous=u
        u,command,hist,_=run_pair('1t1r',cid,u,command)
        hist['previous_reset_to_set_error_u']=hist['initial_u']-previous
        histories['1t1r'].append(hist)
        if cid in (4,35,36,53):
            print(json.dumps({'candidate':label,'cycle':cid,'after_reset_u':u,'elapsed_s':round(time.monotonic()-started,2)}),flush=True)
    arrays={};metrics={};peaks={};per_record={}
    for top,groups in split.items():
        metrics[top]={};peaks[top]={};per_record[top]={}
        for group,ids in groups.items():
            metrics[top][group]={};peaks[top][group]={}
            for phase in ('set','reset'):
                v=np.array([raw[top,cid,phase][0] for cid in ids])
                obs=np.array([raw[top,cid,phase][1] for cid in ids])
                pred=np.array([predictions[top,cid,phase]['current_a'] for cid in ids])
                if not np.all(v==v[0]):raise RuntimeError('Median evaluation requires matching native voltage grids')
                outward=np.arange(v.shape[1])[None,:]<=np.argmax(abs(v),axis=1)[:,None]
                values={'all_eligible':score(v,obs,pred,phase),
                        'outbound':score(v,obs,pred,phase,outward),
                        'return':score(v,obs,pred,phase,~outward)}
                if (top,phase)==('1r','set'):
                    free=abs(obs)<.995e-3
                    values.update(nonlimited=score(v,obs,pred,phase,free),
                                  nonlimited_outbound=score(v,obs,pred,phase,free&outward),
                                  nonlimited_return=score(v,obs,pred,phase,free&~outward),
                                  limited=score(v,obs,pred,phase,~free))
                if (top,phase)==('1r','reset'):
                    values['without_initial_25mV']=score(v,obs,pred,phase,~(np.isclose(v,-.025,atol=1e-6)&outward))
                    if 24 in ids:
                        values['without_pair24']=score(v,obs,pred,phase,np.array(ids)[:,None]!=24)
                med=10.**np.median(np.log10(np.maximum(abs(obs),1e-300)),axis=0)
                med_pred=10.**np.median(np.log10(np.maximum(abs(pred),1e-300)),axis=0)
                values['median_curve']=score(v[0],med,med_pred,phase)
                valid=(abs(v)>1e-12)&np.isfinite(obs)&(abs(obs)>=FLOORS[phase])
                logs=np.log10(np.maximum(abs(obs),1e-300));count=valid.sum(axis=0)
                mean=np.divide(np.where(valid,logs,0.).sum(axis=0),count,out=np.zeros_like(count,dtype=float),where=count>0)
                values['raw_record_log_scatter_rms_decade']=float(np.sqrt(np.sum(np.where(valid,(logs-mean)**2,0.))/valid.sum()))
                metrics[top][group][phase]=values
                rows=[]
                for index,cid in enumerate(ids):
                    observed_peak,predicted_peak=peak(v[index],obs[index]),peak(v[index],pred[index])
                    rows.append({'cycle':cid,'observed':observed_peak,'predicted':predicted_peak,
                                 'peak_magnitude_ratio_predicted_over_observed':predicted_peak['magnitude_a']/observed_peak['magnitude_a']})
                    per_record[top].setdefault(str(cid),{})[phase]=score(v[index],obs[index],pred[index],phase)
                    prefix=f'{top}_cycle{cid}_{phase}'
                    arrays[prefix+'_voltage']=v[index]
                    arrays[prefix+'_observed_current_a']=obs[index]
                    for key in ('current_a','state_u','hbn_voltage_v','transistor_voltage_v','supply_voltage_v','compliance_drop_v','compliance_limited'):
                        arrays[prefix+'_'+key]=predictions[top,cid,phase][key]
                peaks[top][group][phase]={'records':rows,
                    'observed_peak_current_median_a':float(np.median([r['observed']['magnitude_a'] for r in rows])),
                    'predicted_peak_current_median_a':float(np.median([r['predicted']['magnitude_a'] for r in rows])),
                    'observed_peak_voltage_median_v':float(np.median([r['observed']['command_voltage_v'] for r in rows])),
                    'predicted_peak_voltage_median_v':float(np.median([r['predicted']['command_voltage_v'] for r in rows]))}
            metrics[top][group]['pooled_two_phase_raw']=combine([metrics[top][group][p]['all_eligible'] for p in ('set','reset')])
    voltage=raw['1r',split['1r']['train'][0],'set'][0]
    counterfactual={'protocol':'Identical initial u and native bare SET command path; only external circuit changes.',
                    'initial_u':float(q[7]),'observational_validation':False,'circuits':{}}
    for top,port in [('1r',BareGapCircuit(.001)),('1t1r',trans)]:
        trace=simulate_gap_ivp(voltage,float(q[7]),law,port,**options)
        state=trace['final_u']
        counterfactual['circuits'][top]={'final_u':state,'maximum_current_a':float(max(abs(trace['current_a']))),
            'intrinsic_read_R_at_0p05V_ohm':.05/law.current(.05,state),
            'intrinsic_small_signal_R_ohm':law.ron_ohm+law.v0_v*state*np.exp(state)/law.i0_a}
        for key in ('current_a','state_u','hbn_voltage_v','transistor_voltage_v'):
            arrays[f'same_initial_{top}_{key}']=trace[key]
    handoff=max(abs(row['set_reset_handoff_error_u']) for hist in histories.values() for row in hist)
    boundary={row['cycle']:row for row in histories['1t1r'] if row['cycle'] in (35,36)}
    kvl=max(float(np.max(abs(r['supply_voltage_v']-r['hbn_voltage_v']-r['transistor_voltage_v']))) for r in predictions.values())
    sources=[Path(__file__),fitpath,DATA,ROOT/'vulcan2d/shared_gap.py',ROOT/'vulcan2d/shared_gap_ivp.py']
    if cls is AccessGapLaw:sources.append(ROOT/'vulcan2d/shared_access_gap.py')
    report={'status':'full_grid_no_refit_evaluation_completed','candidate':label,'fit_file':str(fitpath.relative_to(ROOT)),
        'shared_material':asdict(law),'initial_u':{'1r_each_conditional_pair':float(q[7]),'1t1r_cycle4':float(q[8])},
        'selected_start':fitted['selected_start'],'calibration_status':fitted['status'],
        'optimizer':{key:selected.get(key) for key in ('success','message','nfev','njev','optimality','verified_bias_cost','optimized_indices','active_bounds','profiled_indices','profile_offset_u')},
        'optimizer_interpretation':'Termination flag and training bias cost do not establish global optimality or physical correctness.',
        'split':split,'observation_floor_a':FLOORS,'evaluation_integrator_options':options,
        'metrics':metrics,'per_record_scores':per_record,'peak_currents_and_voltages':peaks,
        'state_history':histories,'continuity_checks':{'max_abs_set_reset_handoff_error_u':handoff,
            'cycle35_after_reset_u':boundary[35]['after_reset_u'],'cycle36_initial_u':boundary[36]['initial_u'],
            'train_test_boundary_handoff_error_u':boundary[36]['initial_u']-boundary[35]['after_reset_u']},
        'numerical_checks':{'max_abs_kvl_residual_v':kvl,
            'all_current_refinement_checks_passed':all(d['current_tolerance_refinement_passed'] for d in all_diagnostics),
            'max_current_relative_error_estimate':max(d['current_relative_error_estimate'] for d in all_diagnostics),
            'max_state_difference_last_refinement':max(d['max_state_difference_last_refinement'] for d in all_diagnostics),
            'n_integrated_phases':len(all_diagnostics)},
        'same_initial_state_circuit_counterfactual':counterfactual,
        'elapsed_s':time.monotonic()-started,'source_matches_fit':source_check,
        'fit_file_unchanged_during_evaluation':sha(fitpath)==fit_hash,
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources},
        'limitations':['All primary RMSE values compare every eligible raw record directly; experimental scatter is not removed.',
            'Test records were excluded from these numerical fits, but project development has historical exposure to them; this is retrospective, not prospective validation.',
            'Bare-record chronology is unknown. Each conditional SET/RESET pair uses the same initial state; pair24 and initial RESET -25mV sensitivities are separately reported.',
            '1T1R state is propagated continuously from cycle4 through53, including35→36; cycles1–3 remain excluded from primary normal-operation scores.',
            'Voltage-path dynamics are not physical time; the effective gap/access voltage does not identify atomic species or a unique mechanism.',
            'Current tolerance refinement estimates output accuracy. It is not a rigorous bound on the full hidden-state trajectory.',
            'Same-initial circuit comparison is a model counterfactual, not a paired same-cell experiment.']}
    if not report['fit_file_unchanged_during_evaluation']:raise RuntimeError('Fit file changed during evaluation; refusing mixed provenance')
    report['core_sources_unchanged_during_evaluation']=all(sha(ROOT/file)==fitted['sources'][file] for file in source_check)
    if not report['core_sources_unchanged_during_evaluation']:
        raise RuntimeError('Core source changed during evaluation; refusing mixed provenance')
    prefix=FOLDER/f'{label}_evaluation'
    prefix.with_suffix('.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(prefix.with_suffix('.npz'),**arrays)
    plot_candidate(label,raw,predictions,split,FOLDER/f'{label}_comparison')
    comparison_if_ready()
    summary={top:{group:{phase:metrics[top][group][phase]['all_eligible']['rmse_decade'] for phase in ('set','reset')}
                   for group in ('train','test')} for top in ('1r','1t1r')}
    print(json.dumps({'candidate':label,'raw_rmse_decade':summary,'continuity':report['continuity_checks'],
                      'output':str(prefix.with_suffix('.json'))},indent=2),flush=True)


if __name__=='__main__':main()
