"""What paired-state information is actually recoverable from the preceding SET?

Shared shapes are held-pair-fold fits. Target SET can initialize one energy and
amplitude; target RESET never enters that forecast. Joint target refits below
are explicitly descriptive capacity checks, not forecasts or proof of state
evolution. A separate-amplitude benchmark exposes the handoff constraint.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys
import numpy as np
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'analysis'))
from compare_contact_networks import assemble_pairs,score,summarize
from compare_transport_candidates import write_csv
from vulcan2d.contact_network import log10_shape,profile_amplitude,current,domain


def adjust_energies(pair,source,indices,positive_only):
    name=source['name'];base=np.array(source['parameters']);lo,hi=domain(name)
    vp,vn=pair.positive_voltage_v,pair.negative_voltage_v
    yp,yn=pair.positive_log10_current_a,pair.negative_log10_current_a
    def residual(values):
        p=base.copy();p[indices]=values
        rp=yp-log10_shape(vp,name,p)
        if positive_only:return (rp-rp.mean())/np.sqrt(len(rp))
        rn=yn-log10_shape(vn,name,p);a=(rp.mean()+rn.mean())/2
        return np.r_[(rp-a)/np.sqrt(2*len(rp)),(rn-a)/np.sqrt(2*len(rn))]
    guesses=[base[indices]]
    for delta in (-.15,.15,.3):guesses.append(np.clip(base[indices]+delta,lo[indices]+1e-8,hi[indices]-1e-8))
    fits=[least_squares(residual,g,bounds=(lo[indices],hi[indices]),max_nfev=300,
                        ftol=1e-9,gtol=1e-9,xtol=1e-9) for g in guesses]
    good=[f for f in fits if f.success]
    if not good:raise RuntimeError('Energy adjustment failed')
    f=min(good,key=lambda f:f.cost);p=base.copy();p[indices]=f.x
    amp=float(np.mean(yp-log10_shape(vp,name,p))) if positive_only else profile_amplitude(pair,name,p)
    result=dict(source);result['parameters']=p.tolist()
    result['parameter_at_boundary']=bool(np.any(np.minimum(p-lo,hi-p)/(hi-lo)<1e-5))
    return result,amp


def run():
    report=json.loads((ROOT/'analysis/contact_network_comparison.json').read_text())
    pairs,_=assemble_pairs();rows=[];details=[]
    # A geometry-bounded finite-U candidate is a stronger check than an
    # infinite-repulsion fit. Choice follows the saved post-fit audit.
    name='coulomb_right'
    folds=report['groups']['all53']['candidates'][name]['fold_fits']
    for pair in pairs:
        source=next(f['fit'] for f in folds if f['held_range'][0]<=pair.cycle_id<=f['held_range'][1])
        assert pair.cycle_id not in source['training_cycle_ids']
        baseamp=float(np.mean(pair.positive_log10_current_a-log10_shape(pair.positive_voltage_v,name,source['parameters'])))
        rows.append(score(pair,source,baseamp,'target_SET_all_points_amplitude_only'))
        records={}
        for label,indices,positive_only in [('target_SET_E1_forecast',[0],True),
                                           ('joint_target_E1_descriptive',[0],False),
                                           ('joint_target_E1_E2_descriptive',[0,1],False)]:
            fitted,amp=adjust_energies(pair,source,indices,positive_only)
            rows.append(score(pair,fitted,amp,label));records[label]={'parameters':fitted['parameters'],'amplitude':amp}
        ap=baseamp
        an=float(np.mean(pair.negative_log10_current_a-log10_shape(pair.negative_voltage_v,name,source['parameters'])))
        pos=ap+log10_shape(pair.positive_voltage_v,name,source['parameters'])-pair.positive_log10_current_a
        neg=an+log10_shape(pair.negative_voltage_v,name,source['parameters'])-pair.negative_log10_current_a
        r=score(pair,source,ap,'independent_target_amplitudes_descriptive')
        r.update(negative_rmse_decade=float(np.sqrt(np.mean(neg**2))),negative_bias_decade=float(neg.mean()),
                 negative_maximum_abs_error_decade=float(np.max(np.abs(neg))),
                 rmse_decade=float(np.sqrt((np.mean(pos**2)+np.mean(neg**2))/2)))
        rows.append(r)
        details.append({'cycle_id':pair.cycle_id,'source_parameters':source['parameters'],
                        'target_fits':records,'independent_amplitude_shift_RESET_minus_SET_decade':an-ap})
    # A post-fit check of observed positive points below the .15-V training
    # window prevents energy shifts from hiding incorrect low-bias behavior.
    low_source=ROOT/'analysis/cross_polarity_physics_eta_grid.csv'
    with low_source.open(encoding='utf-8-sig') as handle:
        low_rows=[r for r in csv.DictReader(handle) if r['gate_V']=='1.1' and r['lookup_branch']=='mean'
                  and r['status']=='valid' and .1-1e-10<=float(r['hbn_voltage_V'])<.15-1e-10]
    for row in rows:
        obs=[r for r in low_rows if int(r['cycle_id'])==row['cycle_id']]
        v=np.array([float(r['hbn_voltage_V']) for r in obs]);y=np.log10([float(r['set_current_A']) for r in obs])
        err=row['log10_reference_amplitude']+log10_shape(v,name,row['parameters'])-y
        row['positive_0p10_0p14_holdout_rmse_decade']=float(np.sqrt(np.mean(err**2)))
    groups={}
    for tag,test in [('all53',lambda i:True),('first3',lambda i:i<=3),('cycles4to53',lambda i:i>=4)]:
        groups[tag]={p:summarize([r for r in rows if r['protocol']==p and test(r['cycle_id'])]) for p in sorted({r['protocol'] for r in rows})}
        for protocol,summary in groups[tag].items():
            summary['positive_0p10_0p14_holdout_rmse_decade']=float(np.sqrt(np.mean([
                r['positive_0p10_0p14_holdout_rmse_decade']**2 for r in rows if r['protocol']==protocol and test(r['cycle_id'])])))
    files=[Path(__file__),ROOT/'analysis/contact_network_comparison.json',ROOT/'analysis/compare_contact_networks.py',ROOT/'vulcan2d/contact_network.py',low_source]
    result={'scope':'Target-excluded shared-shape information audit; observed SET-based forecast versus explicitly joint descriptive refits',
        'candidate':name,'groups':groups,'details':details,
        'limitations':['A one-energy SET-conditioned forecast uses the entire target SET return, so it is not autonomous or a voltage extrapolation.',
            'Joint target energy fits and separate target amplitudes consume RESET data and are descriptive only.',
            'The extra .10-.14-V positive check uses observed interpolated points disjoint from the .15-3-V positive fitting window; it is a subsequent internal check, not new external data.',
            'An independent branch amplitude gap may reflect polarity, structural change, sampling or measurement effects; it is not itself proof of relaxation.',
            'Only two energies are allowed to change in the capacity checks; failure does not exclude every possible static network.'],
        'input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    out=ROOT/'analysis/contact_state_information.json';out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    write_csv(ROOT/'analysis/contact_state_information_scores.csv',rows)
    print(out)
    for tag,g in groups.items():print(tag,[(p,round(s['positive_rmse_decade'],4),round(s['negative_rmse_decade'],4)) for p,s in g.items()])


if __name__=='__main__':run()
