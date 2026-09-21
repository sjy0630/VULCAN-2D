"""Post-comparison diagnostics: actual series solves and parameter profiles.

The terminal predictions reuse saved fits; no new fit is selected by terminal
errors. Profile tolerances describe sensitivity, not confidence intervals.
"""
from pathlib import Path
import ast
import csv
import hashlib
import json
import sys
import numpy as np
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'analysis'))
from compare_occupation_transport import extract_curves, subset, summarize
from compare_transport_candidates import write_csv
from audit_transistor_gate_interpolation import gate_interpolate
from vulcan2d.transistor import load_default_lookup
from vulcan2d.occupation_transport import (domain,initial_points,log10_shape,series_current,
    current,profile_amplitude,fit_shared_shape)


def profile_gamma(curves):
    name='occupation_background';bounds=domain(name)
    v=np.concatenate([c.voltage_v for c in curves]);y=np.concatenate([c.log10_current_a for c in curves])
    counts=np.array([len(c.voltage_v) for c in curves]);g=np.repeat(np.arange(len(curves)),counts)
    weights=1/np.sqrt(len(curves)*counts[g]);rows=[]
    for loggamma in np.linspace(-6,-1,11):
        def residual(p):
            r=y-log10_shape(v,name,[*p,loggamma]);means=np.bincount(g,weights=r)/counts
            return (r-means[g])*weights
        fits=[least_squares(residual,p[:2],bounds=(bounds[0][:2],bounds[1][:2]),
                x_scale=bounds[1][:2]-bounds[0][:2],max_nfev=300,ftol=1e-9,xtol=1e-9,gtol=1e-9)
              for p in initial_points(name)]
        good=[f for f in fits if f.success]
        if not good:raise RuntimeError('Gamma profile did not converge')
        f=min(good,key=lambda f:f.cost)
        rows.append({'log10_gamma':float(loggamma),'v0_V':float(f.x[0]),'width_V':float(f.x[1]),
                     'rmse_decade':float(np.sqrt(2*f.cost))})
    return rows


def run():
    with (ROOT/'analysis/occupation_cycle_scores.csv').open() as f:records=list(csv.DictReader(f))
    with (ROOT/'analysis/occupation_native_points.csv').open() as f:points=list(csv.DictReader(f))
    groups={};terminal=[];profiles={}
    tr=load_default_lookup();gates=tr.gate_voltages
    grid=np.unique(np.concatenate([c['drain_voltage_v'] for c in tr.curves()]))
    matrix=np.array([[tr.current(v,g) for v in grid] for g in gates])
    it=gate_interpolate(gates,matrix,1.1,'linear_current')
    for upper in (3.,4.2):
        selected={cid:[] for cid in range(1,54)}
        for p in points:
            if p['method']=='linear_current' and float(p['upper_V'])==upper and not p['reason']:
                selected[int(p['cycle_id'])].append(p)
        rows=[]
        for r in records:
            if r['method']!='linear_current' or float(r['upper_V'])!=upper:continue
            ps=selected[int(r['cycle_id'])]
            if 'low_anchor' in r['protocol'] or 'extrapolation' in r['protocol']:
                ps=[p for p in ps if float(p['hbn_V'])>1.2]
            va=np.array([float(p['applied_V']) for p in ps]);actual=np.array([float(p['current_A']) for p in ps])
            pars=ast.literal_eval(r['parameters']);amp=float(r['log10_reference_amplitude'])
            pred,vh=series_current(va,grid,it,amp,r['candidate'],pars)
            err=np.log10(pred/actual)
            balance=np.max(np.abs(pred-np.interp(va-vh,grid,it))/np.maximum(pred,1e-30))
            item={'cycle_id':int(r['cycle_id']),'candidate':r['candidate'],'protocol':r['protocol'],
                  'upper_V':upper,'parameters':pars,'parameter_at_boundary':r['parameter_at_boundary']=='True',
                  'log10_reference_amplitude':amp,'rmse_decade':float(np.sqrt(np.mean(err**2))),
                  'bias_decade':float(np.mean(err)),'n_scored':len(err),
                  'maximum_relative_current_balance_error':float(balance)}
            rows.append(item)
        groups[str(upper)]={}
        for name in sorted({r['candidate'] for r in rows}):
            groups[str(upper)][name]={p:summarize([r for r in rows if r['candidate']==name and r['protocol']==p])
                                    for p in sorted({r['protocol'] for r in rows})}
        terminal.extend(rows)
        curves,_,_=extract_curves('linear_current',upper)
        profiles[str(upper)]={'full':profile_gamma(curves),'low':profile_gamma([subset(c,True) for c in curves])}
    files=[Path(__file__),ROOT/'vulcan2d/occupation_transport.py',ROOT/'analysis/occupation_comparison.json',
           ROOT/'analysis/occupation_cycle_scores.csv',ROOT/'analysis/occupation_native_points.csv']
    result={'scope':'Post-comparison check; fixed saved fits with no terminal-current refitting',
            'terminal_groups':groups,'background_profiles':profiles,
            'maximum_relative_current_balance_error':max(r['maximum_relative_current_balance_error'] for r in terminal),
            'limitations':['Masks are selected using observed internal voltage and measured signal; forecasts are conditional on this scoring domain.',
                'Only frozen SET-return branches are solved; no switching-state history is generated.',
                'Target low readings enter only explicitly conditioned protocols; the terminal solver itself receives no target current.',
                'Profile loss is not a statistical likelihood or confidence interval.'],
            'input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    out=ROOT/'analysis/occupation_circuit_profiles.json'
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    write_csv(ROOT/'analysis/occupation_terminal_scores.csv',terminal)
    print(out)
    for n,g in groups['3.0'].items():print(n,{p:round(s['equal_cycle_rmse_decade'],5) for p,s in g.items()})


if __name__=='__main__':run()
