"""Execute the saved SET-return occupation discrimination design.

All predictions here use de-embedded voltage coordinates derived from measured
current. Even an amplitude-autonomous branch score is not a terminal-voltage
autonomous circuit simulation. No production switching equations are changed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'analysis'))
from audit_transistor_gate_interpolation import METHODS, gate_interpolate, invert
from diagnose_cross_polarity_physics import read_pairs
from compare_transport_candidates import write_csv
from vulcan2d.transistor import load_default_lookup
from vulcan2d.transport_candidates import Curve
from vulcan2d.occupation_transport import NAMES, fit_shared_shape, log10_shape, profile_amplitude

SPLIT = 1.2


def extract_curves(method, upper, pairs=None, gate=1.1):
    source = load_default_lookup()
    gates = source.gate_voltages
    grid = np.unique(np.concatenate([c['drain_voltage_v'] for c in source.curves()]))
    matrix = np.array([[source.current(v,g) for v in grid] for g in gates])
    id_curve = gate_interpolate(gates,matrix,gate,method)
    curves, selections, points = [], [], []
    for pair in read_pairs(str(gate)) if pairs is None else pairs:
        cid = pair['cycle_id']; raw = pair['set']
        vs, currents = np.abs(raw['voltage']),np.abs(raw['current'])
        start = len(vs)-1-int(np.argmax(vs[::-1]))
        zero = float(np.max(currents[vs<=1e-12]))
        vv, yy, reasons = [], [], {}
        for k in range(start,len(vs)):
            va, i = float(vs[k]),float(currents[k]); reason = ''; vh = None
            if not np.isfinite(va) or not np.isfinite(i):reason='nonfinite'
            elif va<=0 or i<=0:reason='nonpositive'
            elif i<=10*zero:reason='signal_guard'
            else:
                try:
                    vd = invert(grid,id_curve,i); vh = va-vd
                    cap = float(np.interp(va,grid,id_curve))
                    if vh<=0:reason='nonpositive_internal_voltage'
                    elif vd/va>.5 or i/cap>.7:reason='transistor_guard'
                    elif not .15<=vh<=upper:reason='outside_window'
                except ValueError:reason='lookup_domain'
            reasons[reason or 'selected'] = reasons.get(reason or 'selected',0)+1
            points.append({'method':method,'upper_V':upper,'cycle_id':cid,'raw_index':k,
                           'applied_V':va,'current_A':i,'hbn_V':vh,'reason':reason})
            if not reason:vv.append(vh);yy.append(np.log10(i))
        v,y = np.asarray(vv),np.asarray(yy)
        nl,nh = int(sum(v<=SPLIT)),int(sum(v>SPLIT))
        accepted = nl>=5 and nh>=5
        selections.append({'method':method,'upper_V':upper,'cycle_id':cid,'low_n':nl,'high_n':nh,
                           'accepted':accepted,'reasons':reasons})
        if accepted:
            order=np.argsort(v);curves.append(Curve(cid,v[order],y[order]))
    return curves,selections,points


def subset(c, low):
    mask = c.voltage_v<=SPLIT if low else c.voltage_v>SPLIT
    return Curve(c.cycle_id,c.voltage_v[mask],c.log10_current_a[mask])


def score(c, fit, amplitude, protocol):
    error = amplitude+log10_shape(c.voltage_v,fit['name'],fit['parameters'])-c.log10_current_a
    return {'cycle_id':c.cycle_id,'candidate':fit['name'],'protocol':protocol,
            'parameters':fit['parameters'],'parameter_at_boundary':fit['parameter_at_boundary'],
            'log10_reference_amplitude':amplitude,'n_scored':len(error),
            'rmse_decade':float(np.sqrt(np.mean(error**2))),'bias_decade':float(np.mean(error)),
            'max_abs_error_decade':float(np.max(np.abs(error))),
            'training_cycle_ids':fit['training_cycle_ids']}


def predict_target(training, target, name, protocol):
    if target.cycle_id in {c.cycle_id for c in training}:
        raise ValueError('Target cycle leaked into training')
    if protocol=='loco_voltage_extrapolation':
        training=[subset(c,True) for c in training]
    fit=fit_shared_shape(training,name)
    if protocol=='loco_autonomous_branch':
        amp=fit['log10_reference_amplitude_center'];evaluation=target
    else:
        amp=profile_amplitude(subset(target,True),name,fit['parameters']);evaluation=subset(target,False)
    return score(evaluation,fit,amp,protocol)


def evaluate_loco(curves,name):
    rows=[]
    for target in curves:
        training=[c for c in curves if c.cycle_id!=target.cycle_id]
        fit=fit_shared_shape(training,name)
        rows.append(score(target,fit,fit['log10_reference_amplitude_center'],'loco_autonomous_branch'))
        amp=profile_amplitude(subset(target,True),name,fit['parameters'])
        rows.append(score(subset(target,False),fit,amp,'loco_full_shape_low_anchor'))
        rows.append(predict_target(training,target,name,'loco_voltage_extrapolation'))
    return rows


def summarize(rows):
    if not rows:return {'n_cycles':0}
    err=np.array([r['rmse_decade'] for r in rows])
    return {'n_cycles':len(rows),'target_cycle_ids':[r['cycle_id'] for r in rows],
            'equal_cycle_rmse_decade':float(np.sqrt(np.mean(err**2))),
            'median_cycle_rmse_decade':float(np.median(err)),
            'cycle_rmse_q10_q90':np.quantile(err,[.1,.9]).tolist(),
            'maximum_cycle_rmse_decade':float(err.max()),
            'mean_signed_bias_decade':float(np.mean([r['bias_decade'] for r in rows])),
            'parameters_q10_median_q90':np.quantile([r['parameters'] for r in rows],[.1,.5,.9],axis=0).tolist(),
            'n_boundary_fits':sum(r['parameter_at_boundary'] for r in rows)}


def run(quick=False):
    pairs=read_pairs('1.1');groups={};scores=[];selections=[];native=[]
    for method in METHODS:
        groups[method]={}
        for upper in (3.,4.2):
            curves,selection,points=extract_curves(method,upper,pairs)
            selections.extend(selection);native.extend(points)
            result={'eligible_cycle_ids':[c.cycle_id for c in curves],'candidates':{}}
            early=[c for c in curves if c.cycle_id<=35]
            for name in NAMES:
                full=fit_shared_shape(curves,name)
                low=fit_shared_shape([subset(c,True) for c in curves],name)
                prior=fit_shared_shape(early,name)
                prior_low=fit_shared_shape([subset(c,True) for c in early],name)
                rows=[]
                for c in curves:
                    a=profile_amplitude(c,name,full['parameters'])
                    rows.append(score(c,full,a,'retrospective_profiled'))
                    if c.cycle_id>35:
                        rows.append(score(c,prior,prior['log10_reference_amplitude_center'],'chronological_autonomous_branch'))
                        a=profile_amplitude(subset(c,True),name,prior['parameters'])
                        rows.append(score(subset(c,False),prior,a,'chronological_full_shape_low_anchor'))
                        a=profile_amplitude(subset(c,True),name,prior_low['parameters'])
                        rows.append(score(subset(c,False),prior_low,a,'chronological_voltage_extrapolation'))
                if method=='linear_current' and not quick:rows.extend(evaluate_loco(curves,name))
                for r in rows:r.update(method=method,upper_V=upper)
                scores.extend(rows)
                result['candidates'][name]={'all_data_fit':full,'low_voltage_fit':low,
                    'early_full_fit':prior,'early_low_fit':prior_low,
                    'evaluation':{p:summarize([r for r in rows if r['protocol']==p]) for p in sorted({r['protocol'] for r in rows})}}
                print(method,upper,name,'n=',len(curves),'fit=',round(full['profiled_equal_cycle_rmse_decade'],4),
                      'pars=',np.round(full['parameters'],4).tolist(),flush=True)
            groups[method][str(upper)]=result
    files=[Path(__file__),ROOT/'analysis/OCCUPATION_COMPARISON_DESIGN_2026-09-20.md',
           ROOT/'vulcan2d/occupation_transport.py',ROOT/'vulcan2d/transport_candidates.py',
           ROOT/'analysis/audit_transistor_gate_interpolation.py',ROOT/'analysis/diagnose_cross_polarity_physics.py',
           ROOT/'1T1M写入.xlsx',ROOT/'vulcan2d/data/fig2b_transistor_lookup.json']
    report={'design':'OCCUPATION_COMPARISON_DESIGN_2026-09-20.md','quick_without_loco':quick,
            'branch':'SET return, 1.1 V gate','voltage_split_V':SPLIT,'groups':groups,
            'input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            'limitations':['Retrospective internal evaluation on previously used data.',
                'De-embedded independent coordinate and eligibility use measured currents; not a terminal-autonomous cycle score.',
                'Per-cycle amplitude profiles do not identify channel number and rates separately.',
                'Single-level energy and coupling are conditional on symmetric bias and assumed 300 K.',
                'Four interpolation alternatives do not bound all transistor mismatch.',
                'Local Jacobian and optimization spread are diagnostics, not statistical confidence intervals.']}
    tag='_quick' if quick else ''
    out=ROOT/'analysis'/f'occupation_comparison{tag}.json'
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    write_csv(ROOT/'analysis'/f'occupation_cycle_scores{tag}.csv',scores)
    write_csv(ROOT/'analysis'/f'occupation_selection{tag}.csv',selections)
    write_csv(ROOT/'analysis'/f'occupation_native_points{tag}.csv',native)
    print(out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--quick',action='store_true')
    run(parser.parse_args().quick)
