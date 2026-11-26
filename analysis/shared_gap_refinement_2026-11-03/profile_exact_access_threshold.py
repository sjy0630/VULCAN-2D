"""Three training-only seeds preserving the old effective SET threshold C."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.optimize import brentq

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'analysis'))
from refine_shared_access_gap import AccessProblem, access_law, law_from_q, OUT

source=OUT/'accurate_fit.json'
fit=json.loads(source.read_text())
base=np.asarray(fit['fits'][fit['selected_start']]['q'],dtype=float)
problem=AccessProblem()
original=law_from_q(base,problem.ron)
target=original.eth_set_v
rows=[]
started=time.monotonic()
for b in (.001,.01,.1):
    q=np.r_[base,np.log10(b)]
    q[2]=brentq(lambda logj: original.v0_v*np.arcsinh(10.**logj/original.i0_a)
                +b*(10.**logj/1e-3)**3-target,-12.,-1.,xtol=1e-13)
    row={'B_V':b,'q':q.tolist(),'target_C_V':target,
         'effective_C_V':original.v0_v*np.arcsinh(10.**q[2]/original.i0_a)
                         +b*(10.**q[2]/1e-3)**3}
    try:
        residual,detail=problem.residual(q,True)
        row.update(valid=True,bias_cost=float(residual@residual/2),blocks=detail,
                   law=asdict(access_law(q,problem.ron)))
    except (ValueError,RuntimeError,FloatingPointError,OverflowError) as exc:
        row.update(valid=False,error=f'{type(exc).__name__}: {exc}')
    rows.append(row)
    print(json.dumps({k:v for k,v in row.items() if k not in ('blocks','law','q')}),flush=True)
report={'status':'three_training_only_exact_effective_SET_threshold_seed_probes',
        'baseline_bias_cost':fit['fits'][fit['selected_start']]['verified_bias_cost'],
        'only_changes':'B and q[2] (Icrit_set) chosen to keep C=e_set+B*(Icrit_set/Iref)^3 fixed',
        'remaining_parameters_frozen':True,'rows':rows,'elapsed_s':time.monotonic()-started,
        'limitations':['Not a refit, not a holdout assessment, and not evidence of a microscopic mechanism.',
                       'C preserves the gap-plus-access activation coefficient; the small Ron*I contribution is not held constant.'],
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (Path(__file__),source,ROOT/'analysis/refine_shared_access_gap.py',
                                   ROOT/'analysis/refine_shared_gap.py',ROOT/'vulcan2d/shared_access_gap.py')}}
(OUT/'seed_exact_threshold_profile.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
