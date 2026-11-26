"""Frozen baseline coefficient profile; training loss only, no test selection."""
from pathlib import Path
import json
import sys
import hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'analysis'))
from refine_shared_access_gap import AccessProblem, OUT

if __name__=='__main__':
    source=OUT/'accurate_fit.json'
    fit=json.loads(source.read_text())
    base=fit['fits'][fit['selected_start']]['q']
    problem=AccessProblem()
    rows=[]
    for b in (1e-6,1e-5,1e-4,1e-3,1e-2,.1,1.):
        q=np.r_[base,np.log10(b)]
        try:
            r,d=problem.residual(q,True)
            row={'B':b,'q':q.tolist(),'bias_cost':float(r@r/2),'blocks':d}
        except (ValueError,RuntimeError,FloatingPointError,OverflowError) as exc:
            row={'B':b,'error':str(exc)}
        rows.append(row)
        print(json.dumps(row),flush=True)
    (OUT/'access_seed_profile.json').write_text(json.dumps({
        'scope':'training-only frozen-parameter nested coefficient profile; not fitted or microscopic evidence',
        'baseline_bias_cost':fit['fits'][fit['selected_start']]['verified_bias_cost'],
        'rows':rows,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    },indent=2)+'\n')
