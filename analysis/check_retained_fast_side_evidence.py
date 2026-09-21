"""Resolve the predeclared ellm=6 comparison; keep more extreme failures separate."""
from pathlib import Path
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'analysis'))
from fit_retained_transport import digest
from compare_activated_configuration import assemble,lookups,scores
from vulcan2d.retained_charge_configuration import trajectory


def run():
    partial_path=ROOT/'analysis/retained_stiff_profile_refinement.json';partial=json.loads(partial_path.read_text())
    partial_arrays=np.load(partial_path.with_suffix('.npz'));assert digest(partial_path.with_suffix('.npz'))==partial['arrays_sha256']
    assert partial['candidates']['all_descriptive_6']['ellm']==6.
    assert partial['candidates']['all_descriptive_6']['two_successive_checks_pass']
    coarse_path=ROOT/'analysis/retained_side_rate_refinement.json';coarse=json.loads(coarse_path.read_text())
    coarse_arrays=np.load(coarse_path.with_suffix('.npz'));assert digest(coarse_path.with_suffix('.npz'))==coarse['arrays_sha256']
    d=assemble();lookup=lookups();mask=d['weights']>0;w=d['weights'][mask]
    rows={'all_descriptive':partial['candidates']['all_descriptive_6']}
    key='all_descriptive_6_step0.000625_current_A';arrays={'all_descriptive_current_A':partial_arrays[key]}
    source_path=ROOT/'analysis/retained_charge_configuration_chronological_single_contact.json';parent=json.loads(source_path.read_text())
    p=np.array(parent['parameters']);p[8]=6.;previous=coarse_arrays['chronological_6_factor4_current_A'];records=[]
    def feature(y,c,wi):
        m=(d['cycle']==c)&(d['window']==wi)&(abs(d['voltage'])>=.1-1e-10)
        return float(10**np.mean(np.log10(y[m])))
    for step in (.00125,.000625):
        settings=dict(parent['settings'],panels=24,table_step=.00125,max_step=step)
        r=trajectory(d['voltage'],parent['shared_parameters'],p,*lookup,**settings)
        err=np.log10(r['current_A'][mask]/previous[mask]);maximum=float(abs(err).max());weighted=float(np.sqrt(np.dot(w,err**2)/w.sum()))
        norm=float(np.max(abs(r['states'].sum(axis=1)-1)))
        row={'settings':settings,'maximum_log_change_from_previous':maximum,'weighted_log_change_from_previous':weighted,
             'maximum_kcl_relative_error':r['maximum_kcl_relative_error'],'maximum_probability_sum_error':norm,
             'passes_successive_tolerances':maximum<.005 and weighted<.001 and norm<2e-11,
             'training_scores':scores(d,r['current_A'],parent['training_ids']),
             'cycle_features':{str(c):{name:feature(r['current_A'],c,wi) for name,wi in [('SET_return',2),('RESET_outward',4)]} for c in (3,4)}}
        records.append(row);previous=r['current_A'];print(step,maximum,weighted,norm,row['passes_successive_tolerances'],flush=True)
    rows['chronological']={'ellm':6.,'refinements':records,'two_successive_checks_pass':all(r['passes_successive_tolerances'] for r in records)}
    arrays['chronological_current_A']=previous
    out={'scope':'Predeclared ellm=6 no-refit comparison only; ellm=9/12 have unresolved fine-step checks',
         'candidates':rows,'all_ellm6_checks_pass':all(r['two_successive_checks_pass'] for r in rows.values()),
         'original_stiff_sweep_completed':False,
         'original_stiff_sweep_failure':'Assertion at all_descriptive ellm12 max_step=.0003125; original log retained. Does not affect independently validated selected fits.',
         'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in [partial_path,partial_path.with_suffix('.npz'),coarse_path,coarse_path.with_suffix('.npz'),source_path,Path(__file__)]}}
    destination=ROOT/'analysis/retained_fast_side_evidence.npz';np.savez_compressed(destination,**arrays);out['arrays_sha256']=digest(destination)
    (ROOT/'analysis/retained_fast_side_evidence.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    partial_arrays.close();coarse_arrays.close()


if __name__=='__main__':run()
