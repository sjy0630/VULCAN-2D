"""Audit only the changed bare SET initial state; inherit proven unchanged runs."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np

import independent_access_audit as audit

OUT=Path(__file__).resolve().parent
NEW=OUT/'access_profile_fit.json'
OLD=OUT/'access_fit.json'
OLD_AUDIT=OUT/'access_final_independent_audit.json'


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started=time.monotonic()
    new=json.loads(NEW.read_text());old=json.loads(OLD.read_text())
    prior=json.loads(OLD_AUDIT.read_text())
    q=np.asarray(new['fits'][new['selected_start']]['q'],dtype=float)
    qold=np.asarray(old['fits'][old['selected_start']]['q'],dtype=float)
    if q.shape != qold.shape: raise ValueError('Parameter-vector shape changed')
    changed=np.flatnonzero(q != qold).tolist()
    if changed != [7]: raise ValueError(f'Expected only bare initial-state index7, found {changed}')
    if new['selected_law'] != old['selected_law'] or new['selected_law'] != prior['law']:
        raise ValueError('Material parameters changed; previous audit cannot be inherited')
    if not prior.get('numerical_passed') or not prior.get('sources_unchanged_during_audit'):
        raise ValueError('Previous selected-parameter audit is not valid for inheritance')
    inherited_core=['vulcan2d/shared_gap.py','vulcan2d/shared_access_gap.py',
                    'vulcan2d/shared_gap_ivp.py','vulcan2d/transistor.py',
                    'vulcan2d/data/fig2b_transistor_lookup.json',
                    'vulcan2d/data/bookoutput2_reverse_transistor_lookup.json',
                    str(audit.DATA.relative_to(audit.ROOT))]
    core_hashes={name:digest(audit.ROOT/name) for name in inherited_core}
    if any(value != prior['source_sha256'][name] for name,value in core_hashes.items()):
        raise ValueError('A core or input hash changed; full prior audit cannot be inherited')
    law=audit.AccessGapLaw(**new['selected_law'])
    mapping=audit.measured_transistor_mapping()
    with np.load(audit.DATA) as raw:
        ids=new['split']['1r']['train']
        voltage=raw[f'1r_cycle_{ids[0]}_set_voltage'].copy()
        observations=abs(np.array([raw[f'1r_cycle_{cid}_set_current'] for cid in ids]))
        if not all(np.array_equal(voltage,raw[f'1r_cycle_{cid}_set_voltage']) for cid in ids):
            raise ValueError('Bare training waveforms differ')
    options=dict(method='Radau',state_rtol=2e-8,state_atol=2e-10,current_rtol=5e-5,
                 max_step_v=.005,max_refinements=4)
    output={'status':'supplement_in_progress','old_initial_u':float(qold[7]),'new_initial_u':float(q[7]),
            'changed_parameter_indices':changed,'same_material_exactly':True,
            'canonical_material_sha256':hashlib.sha256(json.dumps(asdict(law),sort_keys=True,
                                          separators=(',',':'),allow_nan=False).encode()).hexdigest(),
            'same_1t1r_initial_exactly':bool(q[8]==qold[8]),'core_and_input_hashes':core_hashes,
            'production_options':options,'unchanged_accuracy_gates':audit.GATES,
            'prior_audit_file':str(OLD_AUDIT.relative_to(audit.ROOT)),
            'prior_audit_sha256':digest(OLD_AUDIT)}
    files=[Path(__file__),NEW,OLD,OLD_AUDIT,OUT/'independent_access_audit.py']
    hashes={str(p.relative_to(audit.ROOT)):digest(p) for p in files}
    try:
        try:
            trace=audit.simulate_gap_ivp(voltage,float(q[7]),law,audit.BareGapCircuit(.001),**options)
        except RuntimeError as exc:
            output['stricter_production_failure']=str(exc)
            used=dict(new['integrator_options'])
            output['actual_production_options_used']=used
            trace=audit.simulate_gap_ivp(voltage,float(q[7]),law,audit.BareGapCircuit(.001),**used)
        reference=audit.reference_phase(voltage,float(q[7]),law,audit.BareGapCircuit(.001),
                                         rtol=2e-13,atol=2e-15,max_step=.005)
        comparison=audit.compare(trace,reference)
        checks=audit.frozen_checks(voltage,trace,law,'1r','set',mapping)
        old_endpoint=prior['full_trajectory_audits']['1r_pair1_set']['final_u']
        endpoint_same=bool(trace['final_u']==old_endpoint==0.)
        inherited=['1r_pair1_reset','1t1r_pair1_set','1t1r_pair1_reset',
                   '1t1r_pair2_set','1t1r_pair2_reset']
        output.update(independent_integration=comparison,circuit_checks=checks,
            new_bare_set_training_score=audit.training_score(voltage,observations,trace['current_a'],'1r','set'),
            post_set_readout=audit.hard_readout(trace['final_u'],law),
            same_zero_bare_reset_initial_exactly=endpoint_same,
            inherited_primary_audits={key:prior['full_trajectory_audits'][key] for key in inherited},
            production_metadata=trace['metadata'])
        passed=bool(comparison['passed'] and checks['passed'] and endpoint_same
                    and all(row['independent_integration']['passed'] and row['circuit_checks']['passed']
                            for row in output['inherited_primary_audits'].values()))
        output.update(numerical_passed=passed,
                      status='supplement_passed_with_exact_primary_audit_inheritance' if passed else 'supplement_failed_gates')
    except Exception as exc:
        output.update(status='supplement_failed_or_incomplete',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        output.update(elapsed_s=time.monotonic()-started,source_sha256=hashes,
            source_files_unchanged_during_supplement=all(digest(p)==hashes[str(p.relative_to(audit.ROOT))] for p in files),
            limitations=['Only the bare SET initial state changed; no optimization or B=0 rerun was performed.',
                         'Inheritance covers primary trajectories, not a new B=0 bare initial-state ablation.',
                         'Exact material/code/input equality and the same zero bare RESET initial state justify the five unchanged branches.',
                         'Numerical agreement does not validate a microscopic mechanism or improve the unchanged 1T SET resistance trend.'])
        (OUT/'access_profile_initial_audit.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:output.get(k) for k in ('status','old_initial_u','new_initial_u','numerical_passed',
                                                 'same_zero_bare_reset_initial_exactly','independent_integration','elapsed_s')}),flush=True)


if __name__=='__main__': main()
