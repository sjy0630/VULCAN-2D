"""Controlled common RESET threshold/rate profile, with unchanged material law.

Lower the open-state threshold while preserving the threshold at the bare
post-SET state. This follows the training-only excess series RESET peak.
No test observations or topology-specific parameters are used.
"""
from pathlib import Path
from dataclasses import asdict
import sys
import json
import hashlib
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'analysis'))
from refine_shared_gap import JointProblem, OUT, law_from_q, NAMES, DATA

if __name__=='__main__':
    source=OUT/'accurate_fit.json'
    report=json.loads(source.read_text())
    base=np.array(report['fits'][report['selected_start']]['q'])
    p=JointProblem(state_rtol=2e-8,state_atol=2e-10,current_rtol=2e-5)
    original=law_from_q(base,p.ron)
    u_bare=p.trajectory(base,'1r')[0][0]['final_u']
    rows=[]; started=time.monotonic()
    for shift in (0.,-.25,-.5,-.75):
        for rate_shift in (0.,1.,2.):
            q=base.copy();q[3]+=shift;q[5]+=rate_shift
            partial=law_from_q(q,p.ron)
            q[9]+=(original.eth_reset_v-partial.eth_reset_v)*np.exp(u_bare)
            try:
                residual,detail=p.residual(q,True)
                row={'start':len(rows),'q':q.tolist(),'valid':True,
                    'verified_bias_cost':float(residual@residual/2),'blocks':detail,
                    'threshold_log10_shift':shift,'rate_log10_shift':rate_shift}
            except (RuntimeError,ValueError,OverflowError,FloatingPointError) as exc:
                row={'start':len(rows),'q':q.tolist(),'valid':False,'error':str(exc)}
            rows.append(row)
            selected=min((r for r in rows if r['valid']),key=lambda r:r['verified_bias_cost'])
            output={'status':'training_selected_controlled_reset_profile_not_global_optimization',
                'parameter_names':NAMES,'shared_ron_ohm':p.ron,'split':p.split,
                'settings':{'threshold_shifts':[0.,-.25,-.5,-.75],'rate_shifts':[0.,1.,2.]},
                'integrator_options':p.options,'fits':rows,'selected_start':selected['start'],
                'selected_law':asdict(law_from_q(selected['q'],p.ron)),
                'elapsed_s':time.monotonic()-started,'bare_post_set_u_reference':u_bare,
                'selection':'lowest unchanged four-phase/two-pair training surrogate; no test observations',
                'sources':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in
                    (Path(__file__),source,ROOT/'analysis/refine_shared_gap.py',ROOT/'vulcan2d/shared_gap.py',
                     ROOT/'vulcan2d/shared_gap_ivp.py',DATA)}}
            (OUT/'reset_profile_fit.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
            print(json.dumps({k:v for k,v in row.items() if k!='blocks'}),flush=True)
