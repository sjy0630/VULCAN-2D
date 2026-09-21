"""Joint paired-branch comparison under CONTACT_NETWORK_DESIGN_2026-09-20.md."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'analysis'))
from compare_occupation_transport import extract_curves
from compare_transport_candidates import extract_points,write_csv
from vulcan2d.contact_network import Pair,NAMES,fit_shared_shape,log10_shape,profile_amplitude

FOLDS=((1,11),(12,22),(23,33),(34,43),(44,53))


def assemble_pairs(signal='all_positive'):
    positive,_,_=extract_curves('linear_current',3.)
    native=extract_points();pairs=[];selections=[]
    for c in positive:
        rows=[r for r in native if r['branch']=='reset_initial' and r['cycle_id']==c.cycle_id]
        valid=[r for r in rows if r['valid'] and .1<=r['hbn_voltage_V']<=.3]
        selected=[r for r in valid if signal=='all_positive' or r['current_A']>10*r['zero_reference_A']]
        enough=len(selected)>=5 and sum(c.voltage_v<=.3)>=3
        selections.append({'cycle_id':c.cycle_id,'signal':signal,'positive_points':len(c.voltage_v),
            'positive_anchor_points':int(sum(c.voltage_v<=.3)),
            'negative_numerical_valid_points':len(valid),'negative_selected_points':len(selected),
            'negative_n_points_above_10x_zero':sum(r['current_A']>10*r['zero_reference_A'] for r in selected),
            'maximum_negative_zero_reference_fraction':max((r['zero_reference_A']/r['current_A'] for r in selected),default=None),
            'included':enough,'reason':'' if enough else 'insufficient_common_branch_coverage'})
        if enough:
            pairs.append(Pair(c.cycle_id,c.voltage_v,c.log10_current_a,
                    -np.array([r['hbn_voltage_V'] for r in selected]),
                    np.log10([r['current_A'] for r in selected])))
    return pairs,selections


def score(pair,fit,amplitude,protocol):
    errors={}
    for label,v,y in [('positive',pair.positive_voltage_v,pair.positive_log10_current_a),
                      ('negative',pair.negative_voltage_v,pair.negative_log10_current_a)]:
        e=amplitude+log10_shape(v,fit['name'],fit['parameters'])-y
        errors[label]={'rmse_decade':float(np.sqrt(np.mean(e**2))),'bias_decade':float(np.mean(e)),
                       'maximum_abs_error_decade':float(np.max(np.abs(e))),'n_points':len(e)}
    return {'cycle_id':pair.cycle_id,'candidate':fit['name'],'protocol':protocol,
            'parameters':fit['parameters'],'parameter_at_boundary':fit['parameter_at_boundary'],
            'training_cycle_ids':fit['training_cycle_ids'],'log10_reference_amplitude':amplitude,
            'rmse_decade':float(np.sqrt(np.mean([e['rmse_decade']**2 for e in errors.values()]))),
            **{branch+'_'+k:value for branch,e in errors.items() for k,value in e.items()}}


def predict_targets(training,targets,name):
    if {p.cycle_id for p in training}&{p.cycle_id for p in targets}:raise ValueError('Target pair in training')
    fit=fit_shared_shape(training,name);rows=[]
    for target in targets:
        rows.append(score(target,fit,fit['log10_reference_amplitude_center'],'five_fold_autonomous_branch'))
        amp=profile_amplitude(target,name,fit['parameters'],positive_anchor=.3)
        rows.append(score(target,fit,amp,'five_fold_positive_low_anchor'))
    return fit,rows


def summarize(rows):
    if not rows:return {'n_pairs':0}
    result={'n_pairs':len(rows),'cycle_ids':[r['cycle_id'] for r in rows],
            'equal_pair_branch_rmse_decade':float(np.sqrt(np.mean([r['rmse_decade']**2 for r in rows]))),
            'median_pair_rmse_decade':float(np.median([r['rmse_decade'] for r in rows])),
            'max_pair_rmse_decade':float(max(r['rmse_decade'] for r in rows))}
    for branch in ('positive','negative'):
        result[branch+'_rmse_decade']=float(np.sqrt(np.mean([r[branch+'_rmse_decade']**2 for r in rows])))
        result[branch+'_bias_decade']=float(np.mean([r[branch+'_bias_decade'] for r in rows]))
    return result


def run(quick=False):
    scores=[];selections=[];groups={}
    data={signal:assemble_pairs(signal) for signal in ('all_positive','signal_guard')}
    for signal,(ps,sel) in data.items():selections.extend(sel)
    for label,ps in [('all53',data['all_positive'][0]),
                     ('cycles4to53',[p for p in data['all_positive'][0] if p.cycle_id>=4]),
                     ('signal_guard',data['signal_guard'][0])]:
        groups[label]={'eligible_cycle_ids':[p.cycle_id for p in ps],'candidates':{}}
        for name in NAMES:
            full=fit_shared_shape(ps,name);early=[p for p in ps if p.cycle_id<=35]
            prior=fit_shared_shape(early,name);rows=[];folds=[]
            for p in ps:
                amp=profile_amplitude(p,name,full['parameters'])
                rows.append(score(p,full,amp,'retrospective_shared_amplitude'))
                if p.cycle_id>35:
                    rows.append(score(p,prior,prior['log10_reference_amplitude_center'],'chronological_autonomous_branch'))
                    amp=profile_amplitude(p,name,prior['parameters'],positive_anchor=.3)
                    rows.append(score(p,prior,amp,'chronological_positive_low_anchor'))
            if label=='all53' and not quick:
                for lo,hi in FOLDS:
                    fit,r=predict_targets([p for p in ps if not lo<=p.cycle_id<=hi],
                                          [p for p in ps if lo<=p.cycle_id<=hi],name)
                    folds.append({'held_range':[lo,hi],'fit':fit});rows.extend(r)
            for r in rows:r['selection']=label
            scores.extend(rows)
            groups[label]['candidates'][name]={'all_data_fit':full,'early_fit':prior,'fold_fits':folds,
                'evaluation':{protocol:summarize([r for r in rows if r['protocol']==protocol]) for protocol in sorted({r['protocol'] for r in rows})},
                'cycle_groups':{tag:summarize([r for r in rows if r['protocol']=='retrospective_shared_amplitude' and test(r['cycle_id'])])
                    for tag,test in [('first3',lambda i:i<=3),('later50',lambda i:i>=4)]}}
            print(label,name,'N=',len(ps),'RMSE=',round(full['profiled_pair_branch_equal_rmse_decade'],4),
                  'params=',np.round(full['parameters'],4).tolist(),flush=True)
    files=[Path(__file__),ROOT/'analysis/CONTACT_NETWORK_DESIGN_2026-09-20.md',ROOT/'vulcan2d/contact_network.py',
        ROOT/'analysis/compare_occupation_transport.py',ROOT/'analysis/compare_transport_candidates.py',
        ROOT/'analysis/diagnose_cross_polarity_physics.py',ROOT/'analysis/audit_transistor_gate_interpolation.py',
        ROOT/'1T1M写入.xlsx',ROOT/'1T1M擦除.xlsx',ROOT/'vulcan2d/data/fig2b_transistor_lookup.json',
        ROOT/'vulcan2d/data/bookoutput2_reverse_transistor_lookup.json']
    report={'design':'CONTACT_NETWORK_DESIGN_2026-09-20.md','quick_without_five_fold':quick,'groups':groups,
        'limitations':['Quasi-stationary branch hypotheses, not an autonomous switching cycle.',
            'Internal coordinates and eligibility use measured currents and measured transistor lookups.',
            'One shared amplitude per pair assumes the same channel scale across scans; it is tested, not established.',
            'Two-site linear electrostatics and single-electron occupancy are effective hypotheses, not Poisson self-consistency or measured Coulomb blockade.',
            'Rate modulation is an exponential approximation and may produce negative differential conductance; diagnostics are reported.',
            'All datasets have informed model development; no external blinded validation is claimed.'],
        'input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    tag='_quick' if quick else '';out=ROOT/'analysis'/f'contact_network_comparison{tag}.json'
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    write_csv(ROOT/'analysis'/f'contact_network_scores{tag}.csv',scores)
    write_csv(ROOT/'analysis'/f'contact_network_selection{tag}.csv',selections)
    print(out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--quick',action='store_true')
    run(parser.parse_args().quick)
