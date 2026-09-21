"""Exploratory fixed-shape transfer from gate 1.1 V to existing other gates."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'analysis'))
from compare_occupation_transport import extract_curves,subset,score,summarize
from compare_transport_candidates import write_csv
from diagnose_cross_polarity_physics import read_pairs,DATASETS
from audit_transistor_gate_interpolation import METHODS,gate_interpolate
from vulcan2d.transistor import load_default_lookup
from vulcan2d.transport_candidates import Curve,fit_shared_shape as fit_power
from vulcan2d.occupation_transport import NAMES,fit_shared_shape,profile_amplitude,series_current


def run():
    report=json.loads((ROOT/'analysis/occupation_comparison.json').read_text())
    tr=load_default_lookup();grid=np.unique(np.concatenate([c['drain_voltage_v'] for c in tr.curves()]))
    matrix=np.array([[tr.current(v,g) for v in grid] for g in tr.gate_voltages])
    groups={};scores=[];selection=[]
    for gate in (.9,1.3):
        pairs=read_pairs(str(gate));groups[str(gate)]={}
        for method in METHODS:
            curves,sel,points=extract_curves(method,3.,pairs,gate)
            for row in sel:row['target_gate_V']=gate
            selection.extend(sel)
            valid={c.cycle_id:[] for c in curves}
            for p in points:
                if p['cycle_id'] in valid and not p['reason'] and p['hbn_V']>1.2:valid[p['cycle_id']].append(p)
            it=gate_interpolate(tr.gate_voltages,matrix,gate,method)
            result={'n_total_cycles':len(pairs),'n_eligible_cycles':len(curves),
                    'eligible_cycle_ids':[c.cycle_id for c in curves],
                    'excluded_cycle_ids':[s['cycle_id'] for s in sel if not s['accepted']],
                    'candidates':{}}
            for name in NAMES:
                # Source fit sees only the 1.1-V dataset. Numeric cycle labels
                # repeat between files; they do not denote shared observations.
                source=report['groups'][method]['3.0']['candidates'][name]['all_data_fit']
                local=fit_shared_shape(curves,name)
                rows=[]
                for c in curves:
                    a=profile_amplitude(subset(c,True),name,source['parameters'])
                    rows.append(score(subset(c,False),source,a,'cross_gate_low_anchor_internal'))
                    ps=valid[c.cycle_id];va=np.array([p['applied_V'] for p in ps]);actual=np.array([p['current_A'] for p in ps])
                    pred,vh=series_current(va,grid,it,a,name,source['parameters'])
                    error=np.log10(pred/actual)
                    rows.append({'cycle_id':c.cycle_id,'candidate':name,'protocol':'cross_gate_low_anchor_terminal',
                        'parameters':source['parameters'],'parameter_at_boundary':source['parameter_at_boundary'],
                        'rmse_decade':float(np.sqrt(np.mean(error**2))),'bias_decade':float(np.mean(error)),
                        'n_scored':len(error),'log10_reference_amplitude':a})
                    a=profile_amplitude(c,name,local['parameters'])
                    rows.append(score(c,local,a,'target_gate_retrospective_profiled'))
                for row in rows:row.update(target_gate_V=gate,source_gate_V=1.1,method=method)
                scores.extend(rows)
                result['candidates'][name]={'source_parameters':source['parameters'],'target_gate_descriptive_fit':local,
                    'evaluation':{p:summarize([r for r in rows if r['protocol']==p]) for p in sorted({r['protocol'] for r in rows})}}
            windows=[]
            for lo,hi in ((.15,.4),(.4,.8),(.8,1.2),(1.2,2),(2,3)):
                cs=[];exponents=[]
                for c in curves:
                    mask=(c.voltage_v>=lo)&(c.voltage_v<=hi)
                    if sum(mask)<5:continue
                    part=Curve(c.cycle_id,c.voltage_v[mask],c.log10_current_a[mask]);cs.append(part)
                    x=np.log10(part.voltage_v);y=part.log10_current_a
                    exponents.append(float(np.sum((x-x.mean())*(y-y.mean()))/np.sum((x-x.mean())**2)))
                windows.append({'window_V':[lo,hi],'n_cycles':len(cs),'cycle_ids':[c.cycle_id for c in cs],
                    'shared_exponent':fit_power(cs,'power')['parameter'] if len(cs)>=2 else None,
                    'n_local_exponent_below_one':sum(x<1 for x in exponents),
                    'local_exponent_q10_median_q90':np.quantile(exponents,[.1,.5,.9]).tolist() if exponents else None})
            result['local_shape_windows']=windows;groups[str(gate)][method]=result
            print(gate,method,len(curves),'/',len(pairs),[(n,round(g['evaluation']['cross_gate_low_anchor_internal']['equal_cycle_rmse_decade'],4)) for n,g in result['candidates'].items()],flush=True)
    files=[Path(__file__),ROOT/'analysis/occupation_comparison.json',ROOT/'analysis/compare_occupation_transport.py',
           ROOT/'analysis/audit_transistor_gate_interpolation.py',ROOT/'vulcan2d/occupation_transport.py',
           ROOT/'vulcan2d/data/fig2b_transistor_lookup.json']+[ROOT/DATASETS[g][0] for g in ('0.9','1.3')]
    final={'scope':'Exploratory cross-gate shape transfer; registered after 1.1-V candidate comparison',
        'groups':groups,'input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'limitations':['Target <=1.2-V data set amplitude; target high data only score fixed source shapes.',
            'Eligibility uses measured currents and observed internal voltage; high-current excluded regimes are not validated.',
            'Different gates may form different slow structural states. Failure of fixed-shape transfer is not a disproof of all localized-state transport.',
            'Target-gate descriptive fits use all selected target data and are not predictions.',
            'Numeric cycle IDs are local to each dataset; this check uses SET only and requires no assumed SET/RESET pairing.',
            'All positive gates are interpolated; four variants do not bound every possible transistor error.']}
    out=ROOT/'analysis/occupation_cross_gate.json';out.write_text(json.dumps(final,indent=2,allow_nan=False)+'\n')
    write_csv(ROOT/'analysis/occupation_cross_gate_scores.csv',scores)
    write_csv(ROOT/'analysis/occupation_cross_gate_selection.csv',selection)
    print(out)


if __name__=='__main__':run()
