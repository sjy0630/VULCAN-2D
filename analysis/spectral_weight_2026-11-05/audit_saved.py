"""Rebuild scores and independently check saved trajectories and capacity."""
import json
import math
from pathlib import Path
from dataclasses import fields
import numpy as np
from scipy.optimize import brentq
import check_weight as w


def main():
    H=w.HERE
    result=json.loads((H/'RESULT.json').read_text())
    problem=w.tp.load_training_problem()
    parent=w.q1.load_q1_material(w.k.Parameters(**result['parameters'],enable_a=True),
        candidate_source_id='independent_saved_audit',candidate_source_sha256=w.sha(__file__))
    values={f.name:getattr(parent,f.name) for f in fields(w.k.CoupledMaterial)}
    summary={}
    for label,case in result['cases'].items():
        if case['refined']['status']!='complete':
            summary[label]=dict(status='incomplete_no_full_score',details=case['refined']); continue
        saved=json.loads((H/(label+'_refined.json')).read_text())
        material=parent if label=='additive' else w.WeightMaterial(**values,
            drive_mode='total' if label=='weighted_total' else 'per_reference_weight')
        pred={}
        with np.load(H/(label+'_refined.npz')) as z:
            for key in problem.phase_keys:
                prefix='__'.join(map(str,key))+'__'
                pred[key]=dict(arrays={n[len(prefix):]:z[n].copy() for n in z.files if n.startswith(prefix)})
        residual,score=problem.score_predictions(pred)
        assert len(residual)==27023
        assert abs(score['verified_cost']-saved['score']['verified_cost'])<1e-12
        features=w.fa.evaluate_features(problem,pred)
        assert features==json.loads((H/(label+'_refined_features.json')).read_text())
        joins=0
        for entry,top,ids in w.tp.SEGMENTS:
            previous=None
            for cid in ids:
                for phase in w.tp.PHASES:
                    s=saved['phases']['__'.join(map(str,(top,cid,phase)))]
                    initial=(s['initial_state_q'],s['initial_state_a_s'])
                    if previous is not None:
                        assert initial==previous
                        joins+=1
                    else: assert list(initial)==result['initial_states'][entry]
                    previous=(s['final_state_q'],s['final_state_a_s'])
        spectrum=[]; roots=[]
        # First/late cases in both circuits, both polarities, local voltage extremes included.
        selected=[('1r',1,'set'),('1r',15,'reset'),('1t1r',4,'set'),('1t1r',35,'reset')]
        for key in selected:
            a=pred[key]['arrays']
            candidates=np.flatnonzero(abs(a['local_voltage_v'])>.01)
            inds=np.unique(candidates[np.linspace(0,len(candidates)-1,4,dtype=int)])
            for j in inds:
                q=float(a['state_q'][j]); scaled=float(a['state_a_s'][j]); v=float(a['local_voltage_v'][j])
                ref,error=w.direct(material,v,q)
                exact=ref+scaled*v if label=='additive' else material.weight((q,scaled))*ref
                spectrum.append(abs(math.log10(exact/a['current_a'][j])))
            # Series circuit reference solves using direct ensemble integral, no table.
            if key[0]=='1t1r':
                j=inds[len(inds)//2]; command=float(a['command_voltage_v'][j]); sign=np.sign(command)
                q=float(a['state_q'][j]); scaled=float(a['state_a_s'][j]); r=material.readout
                tx,ti=(r.tx_positive,r.ti_positive) if sign>0 else (r.tx_negative,r.ti_negative)
                def cur(v):
                    f=0. if v==0 else abs(w.direct(material,sign*v,q)[0])
                    return f+scaled*v if label=='additive' else material.weight((q,scaled))*f
                root=brentq(lambda vh:cur(vh)-np.interp(abs(command)-vh,tx,ti),0,abs(command),xtol=1e-13)
                expected=cur(root)
                roots.append(dict(key=list(key),command=command,local_direct=root,
                    local_table=abs(float(a['local_voltage_v'][j])),
                    current_error_decade=abs(math.log10(expected/abs(a['current_a'][j])))))
        summary[label]=dict(score_rebuilt=True,all_feature_records_exact=True,history_joins=joins,
            n_residuals=len(residual),n_spectral_samples=len(spectrum),max_trajectory_spectrum_error_decade=max(spectrum),
            independent_direct_roots=roots,
            passed=max(spectrum)<.01 and all(x['current_error_decade']<.01 for x in roots))
    p=parent.parameters
    cap=.001/(.001+p.J_positive_a)
    Wmax=1+p.a_max_s*parent.readout.ron_ohm
    # I(q,V) is monotone in q for this Lorentzian ensemble, so this is a conditional capacity bound.
    F,error=w.direct(parent,.05,cap)
    allowed_R=187.67999555827214
    capacity=dict(q_cap=cap,W_max=Wmax,V_read=.05,spectral_current_A=F,
        R_min_at_cap_ohm=.05/(Wmax*F),required_weight_for_acceptance=.05/(allowed_R*F),
        R_acceptance_upper_ohm=allowed_R,
        assumptions='Positive SET, I<=1mA, q entry<=q_cap, lambda>=0, original J; applies to total and I/W with W>=1. Not an arbitrary-rate or arbitrary-entry exclusion.')
    negative_ratio=Wmax/(parent.readout.ron_ohm*p.J_positive_a)*math.exp(-4*p.b)/(p.lambda_per_v*math.e)*(1-cap)/cap
    capacity.update(negative_boundary_capture_recovery_upper_ratio=negative_ratio,
        all_bare_phases_invariant_for_current_Wmax=negative_ratio<1,
        negative_proof='Use |I|<=Wmax*|V|/Ron and max(v exp(-lambda*v))=1/(lambda*e), chi=2b. Thus negative q-boundary also points inward.',
        weight_necessary_bound_scope='430.37 is conditional on retaining q<=qcap. Increasing Wmax may invalidate the negative-boundary invariant; it is not a global minimum channel count for all redesigned dynamics.')
    out=dict(status='saved_score_history_direct_spectrum_and_capacity_audit_complete',cases=summary,capacity=capacity,
        sources_unchanged=all(w.sha(p)==h for p,h in json.loads((H/'SOURCE_FREEZE.json').read_text()).items()))
    w.save('SAVED_AUDIT.json',out)
    print(json.dumps(out,indent=2))


if __name__=='__main__': main()
