"""Direct training-only tail features used in the current-floor no-go lemma.

No latent-state inversion, candidate fitting, test observations, or source
changes. Plateau comparison threshold is the predeclared training Q10.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
FEATURES=OUT/'feature_constraints.json'
DATA=ROOT/'analysis/shared_hbn_2026-11-03/data/native_observations.npz'


def main():
    features=json.loads(FEATURES.read_text())
    ids=features['training_ids']['1r']
    records=[];digest=hashlib.sha256()
    with np.load(DATA) as raw:
        for cid in ids:
            v=raw[f'1r_cycle_{cid}_reset_voltage']
            i=raw[f'1r_cycle_{cid}_reset_current']
            digest.update(v.tobytes());digest.update(i.tobytes())
            apex=int(np.argmax(abs(v)))
            row={'record':cid,'outward_tail':{}}
            for value in (2.,2.5):
                index=int(np.argmin(abs(abs(v[:apex+1])-value)))
                if abs(abs(v[index])-value)>1e-6:raise ValueError('Tail voltage missing on native grid')
                row['outward_tail'][str(value)]={'index':index,'voltage_v':float(v[index]),'current_magnitude_a':float(abs(i[index]))}
            row['peak_current_a']=float(max(abs(i[:apex+1])))
            records.append(row)
    summary={}
    for value in (2.,2.5):
        tail=np.array([r['outward_tail'][str(value)]['current_magnitude_a'] for r in records])
        summary[str(value)]={'current_q10_q50_q90_a':np.quantile(tail,[.1,.5,.9]).tolist(),
                            'n_below_1p9035uA':int(sum(tail<1.9035e-6)),
                            'n_below_2p67851uA':int(sum(tail<2.67851e-6))}
    report={'status':'direct_training_feature_not_prediction','training_ids':ids,'test_arrays_loaded':False,
            'training_array_sha256':digest.hexdigest(),'feature_constraints_sha256':hashlib.sha256(FEATURES.read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'records':records,'summary':summary,
            'lemma_scope':'A universal hard floor G_reset=0 for i<=I_floor cannot let a passive one-state model cross downward through I_floor during increasing |command voltage|. The result assumes no other evolving state and finite positive differential resistance.'}
    (OUT/'tail_constraint.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
