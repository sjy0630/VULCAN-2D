"""Resolve fast-side path-step errors; retain coarse failures verbatim."""
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
    source_path=ROOT/'analysis/retained_side_rate_refinement.json';source=json.loads(source_path.read_text())
    stored=np.load(source_path.with_suffix('.npz'));assert digest(source_path.with_suffix('.npz'))==source['arrays_sha256']
    d=assemble();lookup=lookups();mask=d['weights']>0;w=d['weights'][mask];rows={};arrays={}
    def features(y):
        return {str(c):{name:float(10**np.mean(np.log10(y[(d['cycle']==c)&(d['window']==wi)&(abs(d['voltage'])>=.1-1e-10)])))
                       for name,wi in [('SET_return',2),('RESET_outward',4)]} for c in (3,4)}
    for name,entry in source['candidates'].items():
        if entry['all_primary_refinements_pass']:continue
        split=name.rsplit('_',1)[0]
        parent=json.loads((ROOT/'analysis'/f'retained_charge_configuration_{split}_single_contact.json').read_text())
        p=np.array(parent['parameters']);p[8]=entry['ellm'];previous=stored[f'{name}_factor4_current_A']
        refinements=[];successive=0
        for step in (.00125,.000625,.0003125):
            settings=dict(parent['settings'],panels=24,table_step=.00125,max_step=step)
            r=trajectory(d['voltage'],parent['shared_parameters'],p,*lookup,**settings)
            err=np.log10(r['current_A'][mask]/previous[mask]);maximum=float(abs(err).max());weighted=float(np.sqrt(np.dot(w,err**2)/w.sum()))
            passed=maximum<.005 and weighted<.001
            successive=successive+1 if passed else 0
            record={'settings':settings,'maximum_log_change_from_previous':maximum,'weighted_log_change_from_previous':weighted,
                    'passes_successive_tolerances':passed,'maximum_kcl_relative_error':r['maximum_kcl_relative_error'],
                    'maximum_probability_sum_error':float(np.max(abs(r['states'].sum(axis=1)-1))),
                    'cycle_features':features(r['current_A']),'training_scores':scores(d,r['current_A'],parent['training_ids'])}
            assert r['maximum_kcl_relative_error']<1e-8 and record['maximum_probability_sum_error']<2e-11
            refinements.append(record);arrays[f'{name}_step{step}_current_A']=r['current_A'];previous=r['current_A']
            print(name,step,maximum,weighted,passed,flush=True)
            if successive>=2:break
        rows[name]={'ellm':entry['ellm'],'refinements':refinements,'two_successive_checks_pass':successive>=2}
        destination=ROOT/'analysis/retained_stiff_profile_refinement.npz';np.savez_compressed(destination,**arrays)
        result={'scope':'Successive path-step convergence after original fast-profile failures; coarse records retained',
                'candidates':rows,'arrays_sha256':digest(destination),
                'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in [source_path,source_path.with_suffix('.npz'),Path(__file__)]}}
        (ROOT/'analysis/retained_stiff_profile_refinement.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    stored.close()


if __name__=='__main__':run()
