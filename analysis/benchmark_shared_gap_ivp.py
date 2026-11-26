"""Accuracy/speed audit for adaptive integration; no fitting or model changes."""
from __future__ import annotations
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vulcan2d.shared_gap import GapLaw, BareGapCircuit, LookupGapCircuit
from vulcan2d.shared_gap_ivp import simulate_gap_ivp
from vulcan2d.transistor import measured_transistor_mapping
from check_shared_gap_independent import independent_phase, compare_trace, digest, evaluate_rows

FOLDER=ROOT/'analysis/shared_hbn_2026-11-03'
OUTPUT=FOLDER/'gap_adaptive_benchmark.json'


def bounded_reference_phase(voltage, initial_u, law, circuit):
    """Independent literal 0/u_max event handling for a single-polarity phase.

    This reference keeps voltage interpolation and one solve_ivp call over
    the full outward/return phase. It does not reuse production segmentation,
    positive compliance invariant, or tolerance-refinement logic.
    """
    commands=np.r_[0.,voltage]
    path=np.r_[0.,np.cumsum(abs(np.diff(commands)))]
    unique,indices,inverse=np.unique(path,return_index=True,return_inverse=True)
    waveform=commands[indices]
    nonzero=voltage[voltage != 0.]
    sign=float(np.sign(nonzero[0]))
    if not np.all(np.sign(nonzero)==sign):
        raise ValueError('Bounded reference requires one polarity per phase')
    bound=0. if sign > 0. else law.u_max
    def rhs(s,y):
        u=float(np.clip(y[0],0.,law.u_max))
        v=float(np.interp(s,unique,waveform))
        # Smoothly extend the one-sided interior rate solely for event root
        # location. Evaluating the projected zero rate outside the boundary
        # can prevent Radau from crossing the root at tight tolerances.
        # Integration stops at the first exact physical-bound event.
        interior=min(law.u_max-1e-12,max(1e-12,u))
        return [law.derivative(interior,circuit.solve(v,law,interior).current_a)]
    def event(s,y):
        return float(y[0])-bound
    event.terminal=True
    event.direction=-1. if sign > 0. else 1.
    started=time.monotonic()
    sol=solve_ivp(rhs,(0.,float(unique[-1])),[initial_u],method='Radau',
                  rtol=1e-10,atol=1e-12,max_step=.005,events=event,dense_output=True)
    if not sol.success:
        raise RuntimeError(sol.message)
    observed=path[1:]
    states=np.full(len(voltage),bound,dtype=float)
    before=observed <= sol.t[-1]
    states[before]=sol.sol(observed[before])[0]
    if np.any(states < -1e-11) or np.any(states > law.u_max+1e-11):
        raise RuntimeError('Independent event reference left literal bounds')
    # Only roundoff at the root is normalized, not a physical-rate overshoot.
    states=np.clip(states,0.,law.u_max)
    return evaluate_rows(voltage,states,law,circuit),{
        'success':True,'method':'Radau with independent literal-bound terminal event',
        'rtol':1e-10,'atol':1e-12,'max_step_v':.005,
        'nfev':int(sol.nfev),'elapsed_s':time.monotonic()-started,
        'absorbing_event_occurred':bool(len(sol.t_events[0]))}


def main():
    fitpath=FOLDER/'gap_joint_fit.json'
    fitted=json.loads(fitpath.read_text())
    base=GapLaw(**fitted['selected_law'])
    q=next(row['q'] for row in fitted['fits'] if row['start']==fitted['selected_start'])
    cases={
        'selected':base,
        'nearby_transport_and_rates':replace(base,ron_ohm=base.ron_ohm*1.03,v0_v=base.v0_v*.98,
            i0_a=base.i0_a*1.08,kset_per_v=base.kset_per_v*.85,kreset_per_v=base.kreset_per_v*1.1,
            eth_set_v=base.eth_set_v*.99,eth_reset_v=base.eth_reset_v*1.02,escale_v=base.escale_v*1.1,
            reset_initiation_drive_v=base.reset_initiation_drive_v*.9),
        'nearby_closed_gap':replace(base,eth_set_v=base.eth_set_v-.01,
            eth_reset_v=base.eth_reset_v*.95,kreset_per_v=base.kreset_per_v*.8),
        'nearby_faster_rates':replace(base,kset_per_v=base.kset_per_v*2.,kreset_per_v=base.kreset_per_v*2.)}
    with np.load(FOLDER/'data/native_observations.npz') as data:
        voltages={top:[data[f'{top}_cycle_{cycle}_{phase}_voltage'].copy() for phase in ('set','reset')]
                  for top,cycle in [('1r',1),('1t1r',4)]}
    transistor=LookupGapCircuit(measured_transistor_mapping(),1.1)
    settings={'method':'Radau','state_rtol':1e-5,'state_atol':1e-7,'current_rtol':1e-3}
    report={'scope':'One full bare pair plus two consecutive series pairs per objective; four frozen parameter cases.',
            'settings':settings,'fitted_source':str(fitpath.relative_to(ROOT)),
            'input_fit_sha256':digest(fitpath),'source_sha256':{
                path:digest(ROOT/path) for path in ('vulcan2d/shared_gap.py','vulcan2d/shared_gap_ivp.py','analysis/benchmark_shared_gap_ivp.py')},
            'cases':{},'status':'in_progress',
            'accuracy_requirement':'Maximum sampled |delta log10 I| <= 1e-3 decade versus independently segmented-reference Radau integration.',
            'limitations':['Reference uses original independent_phase with max_step=.01, rtol1e-9, atol1e-11 and shares material/circuit law.',
                           'Four declared neighboring parameter cases are a bounded audit, not proof over the complete optimizer search domain.',
                           'Current refinement is an a posteriori numerical consistency estimate at output samples, not a rigorous error bound.']}
    for name,law in cases.items():
        block={'law':asdict(law),'reference':{},'adaptive':{},'comparisons':{},'failures':[]}
        report['cases'][name]=block
        reference={}
        ref_start=time.monotonic()
        for top,initial,pairs in [('1r',q[7],1),('1t1r',q[8],2)]:
            u=initial
            for pair in range(pairs):
                for phase,v in zip(('set','reset'),voltages[top]):
                    key=f'{top}_pair{pair+1}_{phase}'
                    port=transistor if top=='1t1r' else BareGapCircuit(.001 if phase=='set' else None)
                    try:
                        result,details=independent_phase(v,u,law,port,0.,'Radau')
                        if result is None:
                            old_failure=details
                            result,details=bounded_reference_phase(v,u,law,port)
                            details['original_reference_failure_preserved']=old_failure
                            details['recovery_reason']='Add literal 0/u_max terminal event after original projected-RHS reference leaves saved bounds.'
                        reference[key]=result;u=result['final_u']
                        block['reference'][key]={'final_u':u,'diagnostics':details}
                    except Exception as exc:
                        block['failures'].append({'method':'reference','phase':key,'error':repr(exc)})
                        break
        block['reference_total_s']=time.monotonic()-ref_start
        adaptive_start=time.monotonic()
        for top,initial,pairs in [('1r',q[7],1),('1t1r',q[8],2)]:
            u=initial
            for pair in range(pairs):
                for phase,v in zip(('set','reset'),voltages[top]):
                    key=f'{top}_pair{pair+1}_{phase}'
                    port=transistor if top=='1t1r' else BareGapCircuit(.001 if phase=='set' else None)
                    try:
                        result=simulate_gap_ivp(v,u,law,port,**settings)
                        u=result['final_u']
                        block['adaptive'][key]={'final_u':u,'metadata':result['metadata']}
                        if key in reference:
                            block['comparisons'][key]=compare_trace(result,reference[key],v)
                    except Exception as exc:
                        block['failures'].append({'method':'adaptive','phase':key,'error':repr(exc)})
                        break
        block['adaptive_objective_s']=time.monotonic()-adaptive_start
        errors=[r['max_abs_delta_log10_current_decade'] for r in block['comparisons'].values()]
        block['max_abs_log10_current_error']=max(errors) if errors else None
        block['passed']=not block['failures'] and len(errors)==6 and max(errors)<=1e-3
        OUTPUT.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'case':name,'adaptive_s':block['adaptive_objective_s'],
                          'max_log_error':block['max_abs_log10_current_error'],'passed':block['passed'],
                          'failures':block['failures']}),flush=True)
    report['status']='passed' if all(b['passed'] for b in report['cases'].values()) else 'failed_with_details'
    report['production_shared_gap_unchanged']=digest(ROOT/'vulcan2d/shared_gap.py')==fitted['source_sha256']['vulcan2d/shared_gap.py']
    OUTPUT.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
