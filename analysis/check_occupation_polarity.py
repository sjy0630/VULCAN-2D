"""Test the frozen full-bias single-level polarity prediction without eta fits."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vulcan2d.occupation_transport import current


def run():
    src=ROOT/'analysis/occupation_cycle_scores.csv'
    obs=ROOT/'analysis/cross_polarity_physics_eta_grid.csv'
    with src.open() as f:scores=list(csv.DictReader(f))
    with obs.open(encoding='utf-8-sig') as f:observed=list(csv.DictReader(f))
    names=('full_bias_level','full_bias_level_background')
    fits={(r['candidate'],int(r['cycle_id'])):ast.literal_eval(r['parameters']) for r in scores
        if r['candidate'] in names and r['method']=='linear_current' and float(r['upper_V'])==3.
        and r['protocol']=='loco_autonomous_branch'}
    rows=[]
    for r in observed:
        if float(r['gate_V'])!=1.1 or r['lookup_branch']!='mean' or r['status']!='valid':continue
        v=float(r['hbn_voltage_V']);cid=int(r['cycle_id'])
        if not .1-1e-12<=v<=.3+1e-12:continue
        for name in names:
            pars=fits[name,cid]
            ratio=abs(float(current(-v,0.,name,pars)))/float(current(v,0.,name,pars))
            observed_ratio=float(r['eta'])
            rows.append({'candidate':name,'cycle_id':cid,'hbn_voltage_V':v,'parameters':pars,
                         'predicted_negative_positive_ratio':ratio,'observed_reset_set_ratio':observed_ratio,
                         'error_decade':float(np.log10(ratio/observed_ratio)),
                         'both_reads_above_10x_zero':max(float(r['set_zero_reference_fraction']),float(r['reset_zero_reference_fraction']))<.1})
    result={}
    for name in names:
        result[name]={}
        for label,test in [('all53',lambda r:True),('cycles4to53',lambda r:r['cycle_id']>=4),
                           ('within_positive_fit_window',lambda r:r['hbn_voltage_V']>=.15-1e-12),
                           ('signal_guard',lambda r:r['both_reads_above_10x_zero'])]:
            selected=[r for r in rows if r['candidate']==name and test(r)]
            cycles=sorted({r['cycle_id'] for r in selected})
            errors=[np.mean([r['error_decade']**2 for r in selected if r['cycle_id']==cid]) for cid in cycles]
            result[name][label]={'n_cycles':len(cycles),'cycle_ids':cycles,'n_points':len(selected),
                 'equal_cycle_rmse_log_ratio_decade':float(np.sqrt(np.mean(errors))),
                 'predicted_ratio_min_max':[min(r['predicted_negative_positive_ratio'] for r in selected),max(r['predicted_negative_positive_ratio'] for r in selected)],
                 'observed_ratio_q10_median_q90':np.quantile([r['observed_reset_set_ratio'] for r in selected],[.1,.5,.9]).tolist(),
                 'fraction_observed_ratios_below_one':float(np.mean([r['observed_reset_set_ratio']<1 for r in selected]))}
    files=[Path(__file__),src,obs,ROOT/'vulcan2d/occupation_transport.py']
    report={'scope':'Target-excluded positive-branch shape predicts reverse/forward current ratio without a fitted polarity multiplier',
            'groups':result,'points':rows,
            'limitations':['Observed ratio is RESET initial / paired SET return, across two scans; it is not a same-instant polarity reversal.',
                'Frozen structure and voltage-independent channel rates are the hypotheses under test. Failure may imply state change or a more complex static transport network.',
                'Main window .10-.30 V extends .05 V below the positive-branch fitting window; observed branch currents are interpolation-only, with no extrapolation.',
                'Original measured-lookup guards apply; zero-reference sensitivity is diagnostic, not a noise distribution.',
                'No independently fitted reverse amplitude, energy or lever arm is permitted in this check.'],
            'input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    out=ROOT/'analysis/occupation_polarity_check.json';out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(out);print(json.dumps(result,indent=2))


if __name__=='__main__':run()
