"""Check every side-rate profile case, without hiding stiff-limit failures."""
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
    path=ROOT/'analysis/retained_side_rate_profile.json';source=json.loads(path.read_text());saved=np.load(path.with_suffix('.npz'))
    for name,sha in source['input_sha256'].items():assert digest(ROOT/name)==sha,name
    assert digest(path.with_suffix('.npz'))==source['arrays_sha256']
    d=assemble();lookup=lookups();mask=d['weights']>0;weight=d['weights'][mask];rows={};arrays={}
    def feature(y,c,wi):
        m=(d['cycle']==c)&(d['window']==wi)&(abs(d['voltage'])>=.1-1e-10)
        return float(10**np.mean(np.log10(y[m])))
    for split,entries in source['candidates'].items():
        parent=json.loads((ROOT/'analysis'/f'retained_charge_configuration_{split}_single_contact.json').read_text())
        for i,entry in enumerate(entries):
            name=f'{split}_{i}';baseline=saved[f'{name}_current_A'];p=np.array(parent['parameters']);p[8]=entry['ellm']
            refinements=[];previous=baseline
            for factor in (2,4):
                settings=dict(parent['settings'],panels=16 if factor==2 else 24,table_step=.005/factor,max_step=.01/factor)
                r=trajectory(d['voltage'],parent['shared_parameters'],p,*lookup,**settings)
                err=np.log10(r['current_A'][mask]/baseline[mask]);step=np.log10(r['current_A'][mask]/previous[mask])
                maximum=float(abs(err).max());weighted=float(np.sqrt(np.dot(weight,err**2)/weight.sum()))
                record={'settings':settings,'maximum_log_change_from_primary':maximum,'weighted_log_change_from_primary':weighted,
                        'maximum_log_change_from_previous':float(abs(step).max()),
                        'weighted_log_change_from_previous':float(np.sqrt(np.dot(weight,step**2)/weight.sum())),
                        'passes_primary_tolerances':maximum<.005 and weighted<.001,
                        'maximum_kcl_relative_error':r['maximum_kcl_relative_error'],
                        'training_scores':scores(d,r['current_A'],parent['training_ids']),
                        'cycle_features':{str(c):{label:feature(r['current_A'],c,wi) for label,wi in [('SET_return',2),('RESET_outward',4)]} for c in (3,4)}}
                assert r['maximum_kcl_relative_error']<1e-8
                refinements.append(record);arrays[f'{name}_factor{factor}_current_A']=r['current_A'];previous=r['current_A']
            rows[name]={'ellm':entry['ellm'],'refinements':refinements,'all_primary_refinements_pass':all(r['passes_primary_tolerances'] for r in refinements)}
            print(name,entry['ellm'],[r['maximum_log_change_from_primary'] for r in refinements],rows[name]['all_primary_refinements_pass'],flush=True)
    destination=ROOT/'analysis/retained_side_rate_refinement.npz';np.savez_compressed(destination,**arrays)
    result={'scope':'Full side-rate profile numerical audit; failures retained','candidates':rows,'arrays_sha256':digest(destination),
            'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in [path,path.with_suffix('.npz'),Path(__file__)]}}
    (ROOT/'analysis/retained_side_rate_refinement.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':run()
