"""Independent numerical and bounded physical audit of a selected shared law.

Only training observations are used for descriptive ablation scores. Reference
integration has its own triangular path construction and explicitly written
rate equation, using DOP853 rather than production Radau. It shares the circuit
operating-point implementation, which is separately checked against direct
current-space Brent roots and the independent algebraic voltage expression.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vulcan2d.shared_gap import GapLaw, GapPoint, BareGapCircuit, LookupGapCircuit
from vulcan2d.shared_access_gap import AccessGapLaw
from vulcan2d.shared_gap_ivp import simulate_gap_ivp
from vulcan2d.transistor import measured_transistor_mapping

OUT = Path(__file__).resolve().parent
DATA = ROOT/'analysis/shared_hbn_2026-11-03/data/native_observations.npz'
PRODUCTION_OPTIONS = dict(method='Radau', state_rtol=2e-8, state_atol=2e-10,
                          current_rtol=5e-5, max_step_v=.05, max_refinements=4)
REFERENCE_OPTIONS = dict(rtol=2e-10, atol=2e-12, max_step=.01)
GATES = dict(max_current_log10_difference=2e-4, max_state_difference=2e-5,
             max_endpoint_difference=2e-5, max_independent_root_current_relative_error=2e-8,
             max_kcl_residual_a=1e-10, max_voltage_closure_v=1e-8,
             max_power_decomposition_relative_error=1e-10)


def voltage_components(current, u, law):
    # Explicit independent algebra. u<=30 keeps exp(u) within normal range.
    ohmic = law.ron_ohm*current
    gap = u*law.v0_v*math.asinh(current*math.exp(u)/law.i0_a)
    access = u*getattr(law, 'access_voltage_v', 0.)*(current*math.exp(u)/1e-3)**3
    return ohmic, gap, access


def independent_current(command, u, law, top, phase, mapping):
    sign, target = (-1. if command < 0. else 1.), abs(float(command))
    if target == 0.:
        return 0., 0., 0.
    def vh(current):
        return sum(voltage_components(current, u, law))
    if top == '1r':
        if phase == 'set' and vh(1e-3) < target:
            magnitude = 1e-3
        else:
            magnitude = brentq(lambda current: vh(current)-target, 0., target/law.ron_ohm,
                               xtol=1e-27, rtol=2e-13, maxiter=300)
        return sign*magnitude, sign*vh(magnitude), 0.
    # During root bracketing, negative transistor voltage is extended by zero
    # current. The physical root always lies in the passive voltage interval.
    def residual(current):
        vt = max(0., target-vh(current))
        return current-abs(mapping.drain_current(sign*vt, 1.1))
    magnitude = brentq(residual, 0., target/law.ron_ohm,
                       xtol=1e-27, rtol=2e-13, maxiter=300)
    return sign*magnitude, sign*vh(magnitude), sign*(target-vh(magnitude))


def reference_phase(voltage, initial, law, circuit, *, rtol=2e-10, atol=2e-12, max_step=.01, method='DOP853'):
    """Independent DOP853 solution on each half of a one-polarity triangle.

    No production clock helper, positive-floor helper, event helper or rate
    implementation is called. Only physical [0,u_max] bounds are enforced;
    the compliance stopping state must emerge from the unchanged equation.
    """
    v = np.asarray(voltage)
    peak = int(np.argmax(abs(v)))
    last_peak = len(v)-1-int(np.argmax(abs(v[::-1])))
    if v[0] != 0. or v[-1] != 0. or np.any(np.diff(abs(v[:peak+1])) < 0.) or np.any(np.diff(abs(v[last_peak:])) > 0.):
        raise ValueError('Independent audit expects the recorded single-polarity triangular phase')
    sign = float(np.sign(v[peak]))
    maximum = abs(float(v[peak]))
    state = float(initial)
    states = np.empty(len(v))
    total_evaluations = 0
    for indices, outward in ((np.arange(peak+1), True), (np.arange(last_peak, len(v)), False)):
        start_state = state
        def rhs(s, y):
            nonlocal total_evaluations
            total_evaluations += 1
            if total_evaluations > 300000:
                raise RuntimeError('Independent explicit reference exceeded 300000 RHS calls; stiffness not certified')
            u = min(law.u_max, max(0., float(y[0])))
            command = sign*(s if outward else maximum-s)
            current = float(circuit.solve(command, law, u).current_a)
            if current == 0.:
                return [0.]
            # Continue the one-sided incoming rate solely for event location
            # when an internal RK stage crosses a physical bound. The event
            # uses unclipped y and terminates at the first hit; all subsequent
            # values are absorbing. Thus no post-bound physical evolution is
            # added, and the pre-bound differential equation is unchanged.
            drive = law.v0_v*math.asinh(abs(current)*math.exp(u)/law.i0_a)
            if current > 0.:
                threshold, rate, direction = law.eth_set_v, law.kset_per_v, -1.
            else:
                threshold = law.eth_reset_v+law.reset_initiation_drive_v*math.exp(-u)
                rate, direction = law.kreset_per_v, 1.
            return [direction*rate*math.sinh(max(drive-threshold, 0.)/law.escale_v)]
        bound = 0. if sign > 0. else law.u_max
        if state == bound:
            states[indices] = state
            continue
        def boundary(s, y):
            return float(y[0])-bound
        boundary.terminal = True
        boundary.direction = -1. if sign > 0. else 1.
        result = solve_ivp(rhs, (0., maximum), [state], method=method,
                           rtol=rtol, atol=atol, max_step=max_step,
                           dense_output=True, events=boundary)
        if not result.success:
            raise RuntimeError('Independent '+method+' failed: '+result.message)
        sampling = abs(v[indices]) if outward else maximum-abs(v[indices])
        event = len(result.t_events[0]) > 0
        in_range = sampling <= result.t[-1]
        states[indices[in_range]] = result.sol(sampling[in_range])[0]
        if event:
            states[indices[~in_range]] = bound
            state = bound
        else:
            state = float(result.y[0,-1])
        if np.min(states[indices]) < -1e-8 or np.max(states[indices]) > law.u_max+1e-8:
            raise RuntimeError('Independent dense output left physical state bounds')
        states[indices] = np.clip(states[indices], 0., law.u_max)
        if outward:
            states[peak:last_peak+1] = state
        if not (-1e-8 <= state <= law.u_max+1e-8):
            raise RuntimeError('Independent endpoint left physical bounds')
        state = min(law.u_max, max(0., state))
    points = [circuit.solve(float(command), law, float(u)) for command,u in zip(v,states)]
    output = {name: np.array([getattr(p,name) for p in points],
                             dtype=bool if name == 'compliance_limited' else float)
              for name in GapPoint.__dataclass_fields__}
    output.update(state_u=states, final_u=state, nfev=total_evaluations,
                  method='independently constructed '+method+' triangle',
                  metadata={'method':method,'state_rtol':rtol,'state_atol':atol,'max_step_v':max_step,
                            'source':'independent audit solver with one-sided bound-event continuation'})
    return output


def compare(production, reference):
    p, r = abs(production['current_a']), abs(reference['current_a'])
    use = (p > 0.) & (r > 0.)
    zero_mismatch = int(np.count_nonzero((p == 0.) != (r == 0.)))
    logerr = float(np.max(abs(np.log10(p[use]/r[use])))) if np.any(use) else 0.
    stateerr = float(np.max(abs(production['state_u']-reference['state_u'])))
    enderr = abs(production['final_u']-reference['final_u'])
    return {'max_current_log10_difference': logerr,
            'max_current_relative_difference': float(np.max(abs(p[use]/r[use]-1.))) if np.any(use) else 0.,
            'zero_current_mismatch_count': zero_mismatch,
            'max_state_difference': stateerr, 'endpoint_difference': enderr,
            'production_endpoint_u': production['final_u'], 'reference_endpoint_u': reference['final_u'],
            'independent_nfev': reference['nfev'],
            'passed': zero_mismatch == 0 and logerr <= GATES['max_current_log10_difference']
                and stateerr <= GATES['max_state_difference'] and enderr <= GATES['max_endpoint_difference']}


def fast_bare_set_reference(voltage, initial, law, method):
    """Resolve an extremely fast bare SET layer by the exact ds/du=1/G change.

    This audit-only fallback is restricted to a B=0 bare positive triangle
    whose compliance exceeds Icrit. The applied voltage keeps evolving as s;
    it is NOT frozen during the transformed layer and no state jump is imposed.
    Both integration methods must agree at all stored observations.
    """
    v=np.asarray(voltage,dtype=float)
    peak=int(np.argmax(v))
    maximum=float(v[peak])
    if (np.any(v<0.) or v[0] != 0. or v[-1] != 0.
            or getattr(law,'access_voltage_v',0.) != 0.):
        raise ValueError('Fast-layer fallback is restricted to the B=0 bare SET triangle')
    cap=1e-3
    if law.v0_v*math.asinh(cap/law.i0_a) <= law.eth_set_v:
        raise ValueError('Fast-layer fallback requires a finite-hit zero-gap state')
    circuit=BareGapCircuit(cap)
    nfev=0
    def rate(s,u):
        nonlocal nfev
        nfev+=1
        u=min(law.u_max,max(0.,float(u)))
        current=abs(circuit.solve(float(s),law,u).current_a)
        drive=law.v0_v*math.asinh(current*math.exp(u)/law.i0_a)
        return -law.kset_per_v*math.sinh(max(drive-law.eth_set_v,0.)/law.escale_v)
    def fast_event(s,y): return rate(s,y[0])+1e5
    fast_event.terminal=True
    fast_event.direction=-1.
    def zero_event(s,y): return float(y[0])
    zero_event.terminal=True
    zero_event.direction=-1.
    forward=solve_ivp(lambda s,y:[rate(s,y[0])],(0.,maximum),[initial],method=method,
                      rtol=2e-11,atol=2e-13,max_step=.005,
                      dense_output=True,events=(fast_event,zero_event))
    if not forward.success or not len(forward.t_events[0]):
        raise RuntimeError('Fast-layer entry was not independently resolved: '+forward.message)
    s_entry=float(forward.t[-1]);u_entry=float(forward.y[0,-1])
    def inverse_rhs(u,y):
        value=rate(float(y[0]),u)
        if value >= 0.:
            raise RuntimeError('Inverse fast-layer path encountered a nonnegative SET rate')
        return [1./value]
    def maximum_event(u,y): return float(y[0])-maximum
    maximum_event.terminal=True
    maximum_event.direction=1.
    inverse=solve_ivp(inverse_rhs,(u_entry,0.),[s_entry],method=method,
                      rtol=2e-11,atol=2e-13,max_step=.01,
                      dense_output=True,events=maximum_event)
    if not inverse.success or inverse.t[-1] != 0.:
        raise RuntimeError('Inverse fast-layer reference did not reach the physical zero bound')
    s_zero=float(inverse.y[0,-1])
    states=np.zeros(len(v))
    for j,s in enumerate(v[:peak+1]):
        if s <= s_entry:
            states[j]=float(forward.sol(float(s))[0])
        elif s < s_zero:
            states[j]=brentq(lambda u: float(inverse.sol(u)[0])-s,0.,u_entry,xtol=1e-12)
        # After the independently located first bound contact, SET is absorbing.
    points=[circuit.solve(float(command),law,float(u)) for command,u in zip(v,states)]
    result={name:np.array([getattr(p,name) for p in points],
                          dtype=bool if name=='compliance_limited' else float)
            for name in GapPoint.__dataclass_fields__}
    result.update(state_u=states,final_u=0.,nfev=nfev,
                  method=method+' with exact inverse fast-layer path',
                  metadata={'method':method,'source':'audit-only exact inverse-coordinate fast-layer reference',
                            'fast_rate_trigger_per_V':1e5,'entry_voltage_V':s_entry,'entry_u':u_entry,
                            'zero_bound_voltage_V':s_zero,'inverse_elapsed_voltage_V':s_zero-s_entry,
                            'state_rtol':2e-11,'state_atol':2e-13,
                            'physical_law_and_command_voltage_unchanged':True})
    return result


def frozen_checks(voltage, trace, law, top, phase, mapping):
    n = len(voltage)
    indices = np.unique(np.r_[np.linspace(0, n-1, 21, dtype=int),
                             np.argmax(abs(trace['current_a'])), np.argmin(trace['state_u']),
                             np.argmax(abs(voltage))])
    root_rows = []
    for j in indices:
        v, u, current = float(voltage[j]), float(trace['state_u'][j]), float(trace['current_a'][j])
        ref_i, ref_vh, ref_vt = independent_current(v, u, law, top, phase, mapping)
        root_rows.append({'index': int(j), 'command_V': v, 'u': u,
                          'production_current_A': current, 'reference_current_A': ref_i,
                          'relative_current_error': abs(current-ref_i)/max(abs(ref_i), 1e-30),
                          'hbn_voltage_error_V': abs(float(trace['hbn_voltage_v'][j])-ref_vh)})
    i, vh, vt = trace['current_a'], trace['hbn_voltage_v'], trace['transistor_voltage_v']
    kcl = max((abs(float(x)-mapping.drain_current(float(y), 1.1)) for x,y in zip(i,vt)), default=0.) if top == '1t1r' else 0.
    closure = float(np.max(abs(voltage-vh-vt-trace['compliance_drop_v'])))
    components = np.array([voltage_components(float(x), float(u), law) for x,u in zip(i, trace['state_u'])])
    powers = i[:,None]*components
    denom = np.maximum(abs(i*vh), 1e-30)
    power_error = float(np.max(abs(np.sum(powers,axis=1)-i*vh)/denom))
    monotone_error = float(max(0., np.max(np.diff(trace['state_u'])))) if phase == 'set' else float(max(0., np.max(-np.diff(trace['state_u']))))
    root_max = max(row['relative_current_error'] for row in root_rows)
    return {'independent_root_count': len(root_rows), 'root_points': root_rows,
            'max_independent_root_relative_current_error': root_max,
            'max_kcl_residual_A': kcl, 'max_voltage_closure_V': closure,
            'min_material_power_W': float(np.min(i*vh)),
            'min_transistor_power_W': float(np.min(i*vt)),
            'min_component_power_W': float(np.min(powers)),
            'max_power_decomposition_relative_error': power_error,
            'max_wrong_direction_state_change': monotone_error,
            'passed': root_max <= GATES['max_independent_root_current_relative_error']
                and kcl <= GATES['max_kcl_residual_a'] and closure <= GATES['max_voltage_closure_v']
                and power_error <= GATES['max_power_decomposition_relative_error']
                and np.min(powers) >= -1e-18 and np.min(i*vt) >= -1e-18 and monotone_error <= 2e-7}


def training_score(v, observations, current, top, phase):
    use = (abs(v)[None,:] > 1e-12) & (observations >= (1e-11 if phase == 'set' else 1e-12))
    error = np.log10(np.maximum(abs(current),1e-30))[None,:]-np.log10(np.maximum(observations,1e-30))
    def rmse(mask):
        return float(np.sqrt(np.mean(error[mask]**2))) if np.any(mask) else None
    median = 10.**np.median(np.log10(np.maximum(observations,1e-15)),axis=0)
    peak = int(np.argmax(abs(current)))
    measured_peak = int(np.argmax(median))
    result = {'training_record_point_rmse_decade': rmse(use),
              'predicted_peak_current_A': float(max(abs(current))),
              'predicted_peak_command_V': float(v[peak]),
              'training_median_peak_current_A': float(max(median)),
              'training_median_peak_command_V': float(v[measured_peak])}
    if top == '1r' and phase == 'set':
        result['nonlimited_rmse_decade'] = rmse(use & (observations < .995e-3))
        result['limited_rmse_decade'] = rmse(use & (observations >= .995e-3))
        result['peak_voltage_note'] = 'Compliance plateau peak position is not a switching threshold.'
    return result


def hard_readout(final_u, law):
    return {'state_u': final_u, 'intrinsic_R_at_50mV_ohm': .05/law.current(.05, final_u),
            'intrinsic_differential_R_at_zero_ohm': law.ron_ohm+law.v0_v*final_u*math.exp(final_u)/law.i0_a}


def run_trajectories(law, initial, waveforms, observations, mapping, do_reference, ablation_fallback=False):
    circuit = LookupGapCircuit(mapping, 1.1)
    reports, traces = {}, {}
    for top, repeats in (('1r',1),('1t1r',2)):
        production_state = reference_state = initial[top]
        for repeat in range(repeats):
            for phase in ('set','reset'):
                key = f'{top}_pair{repeat+1}_{phase}'
                v = waveforms[top,phase]
                load = circuit if top == '1t1r' else BareGapCircuit(1e-3 if phase == 'set' else None)
                print(json.dumps({'stage': key, 'reference': do_reference}), flush=True)
                fallback = None
                try:
                    trace = simulate_gap_ivp(v, production_state, law, load, **PRODUCTION_OPTIONS)
                except RuntimeError as exc:
                    if not ablation_fallback:
                        raise
                    # No production implementation is changed. A bound-event
                    # discontinuity in the B=0 counterfactual is resolved by
                    # the explicitly documented independent event solver.
                    ordinary_reference_failure=None
                    try:
                        trace = reference_phase(v,production_state,law,load,**REFERENCE_OPTIONS)
                        check = reference_phase(v,production_state,law,load,method='BDF',
                                                rtol=2e-10,atol=2e-12,max_step=.005)
                    except RuntimeError as ref_exc:
                        if not (top=='1r' and phase=='set' and getattr(law,'access_voltage_v',0.)==0.):
                            raise
                        ordinary_reference_failure=str(ref_exc)
                        trace=fast_bare_set_reference(v,production_state,law,'DOP853')
                        check=fast_bare_set_reference(v,production_state,law,'Radau')
                    comparison = compare(trace,check)
                    fallback={'production_failure':str(exc),'ordinary_reference_failure':ordinary_reference_failure,
                              'independent_method_comparison':comparison,
                              'reference_method':trace['method'],'comparison_method':check['method']}
                    if not comparison['passed']:
                        raise RuntimeError('B=0 independent fallback failed unchanged numerical gates: '+str(comparison))
                row = {'initial_u': production_state, 'final_u': trace['final_u'],
                       'production_metadata': trace['metadata'],
                       'training_score': training_score(v,observations[top,phase],trace['current_a'],top,phase),
                       'circuit_checks': frozen_checks(v,trace,law,top,phase,mapping)}
                production_state = trace['final_u']
                if fallback is not None:
                    row['independent_event_fallback']=fallback
                if phase == 'set':
                    row['post_set_readout'] = hard_readout(production_state,law)
                if do_reference:
                    reference = reference_phase(v, reference_state, law, load, **REFERENCE_OPTIONS)
                    reference_state = reference['final_u']
                    row['independent_integration'] = compare(trace,reference)
                reports[key], traces[key] = row, trace
    return reports, traces


def access_counterfactual(law, trace, voltage):
    if not isinstance(law,AccessGapLaw):
        return None
    without = replace(law, access_voltage_v=0.)
    apex = int(np.argmax(abs(voltage)))
    a,b = [int(np.argmin(abs(voltage[:apex+1]-v))) for v in (2.,4.)]
    u = float(trace['state_u'][a])
    currents = [float(trace['current_a'][j]) for j in (a,b)]
    def resistance(model):
        return [model.voltage(current,u)/current for current in currents]
    r, r0 = resistance(law), resistance(without)
    dynamic = []
    for j in (a,b):
        current, state = float(trace['current_a'][j]), float(trace['state_u'][j])
        parts = voltage_components(current,state,law)
        dynamic.append({'command_V':float(voltage[j]), 'u':state,'current_A':current,
                        'ohmic_voltage_V':parts[0],'gap_voltage_V':parts[1],'access_voltage_V':parts[2],
                        'intrinsic_resistance_ohm':float(trace['hbn_voltage_v'][j])/current})
    current_grid = np.geomspace(1e-9,1e-3,41)
    return {'status':'same_state_transport_counterfactual_not_independent_experimental_validation',
            'frozen_u_from_2V_prediction':u,'probe_currents_from_predicted_2V_4V_A':currents,
            'with_access_R_ohm':r,'without_access_R_ohm':r0,
            'with_access_R_ratio':r[1]/r[0],'without_access_R_ratio':r0[1]/r0[0],
            'dynamic_2V_4V_decomposition':dynamic,
            'actual_trajectory_R_4V_over_2V':dynamic[1]['intrinsic_resistance_ohm']/dynamic[0]['intrinsic_resistance_ohm'],
            'actual_trajectory_u_change_2V_to_4V':dynamic[1]['u']-dynamic[0]['u'],
            'actual_trajectory_current_ratio_4V_over_2V':dynamic[1]['current_A']/dynamic[0]['current_A'],
            'current_grid_A':current_grid.tolist(),
            'fixed_u_R_with_access_ohm':[law.voltage(float(i),u)/i for i in current_grid],
            'fixed_u_R_without_access_ohm':[without.voltage(float(i),u)/i for i in current_grid]}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--fit',default='accurate_fit.json')
    parser.add_argument('--output',default=None)
    parser.add_argument('--strict',action='store_true',help='Tighten both integrators; preserve original accuracy gates')
    parser.add_argument('--reference-strict',action='store_true',help='Tighten independent reference and cap production steps, retaining production state tolerances')
    args=parser.parse_args()
    if args.strict:
        PRODUCTION_OPTIONS.update(state_rtol=1e-11, state_atol=1e-13, max_step_v=.005)
        REFERENCE_OPTIONS.update(rtol=2e-13, atol=2e-15, max_step=.005)
    elif args.reference_strict:
        PRODUCTION_OPTIONS.update(max_step_v=.005)
        REFERENCE_OPTIONS.update(rtol=2e-13, atol=2e-15, max_step=.005)
    fit_path=Path(args.fit)
    if not fit_path.is_absolute(): fit_path=OUT/fit_path
    fit=json.loads(fit_path.read_text())
    if fit['selected_start'] is None: raise ValueError('No valid selected fitted model')
    selected=fit['fits'][fit['selected_start']]
    params=fit['selected_law']
    law=AccessGapLaw(**params) if 'access_voltage_v' in params else GapLaw(**params)
    initial={'1r':float(selected['q'][7]),'1t1r':float(selected['q'][8])}
    mapping=measured_transistor_mapping()
    observations,waveforms={},{}
    with np.load(DATA) as raw:
        for top in ('1r','1t1r'):
            ids=fit['split'][top]['train']
            for phase in ('set','reset'):
                vv=[raw[f'{top}_cycle_{cid}_{phase}_voltage'] for cid in ids]
                if not all(np.array_equal(v,vv[0]) for v in vv):
                    raise ValueError('The selected representative-waveform audit requires identical grids')
                waveforms[top,phase]=vv[0].copy()
                observations[top,phase]=abs(np.array([raw[f'{top}_cycle_{cid}_{phase}_current'] for cid in ids]))
    source_paths=[Path(__file__),fit_path,DATA,ROOT/'vulcan2d/shared_gap.py',ROOT/'vulcan2d/shared_access_gap.py',
                  ROOT/'vulcan2d/shared_gap_ivp.py',ROOT/'vulcan2d/transistor.py',
                  ROOT/'vulcan2d/data/fig2b_transistor_lookup.json',ROOT/'vulcan2d/data/bookoutput2_reverse_transistor_lookup.json']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    started=time.monotonic()
    output={'status':'independent_audit_in_progress','fit_source':str(fit_path.relative_to(ROOT)),
            'law':asdict(law),'initial_states':initial,'predeclared_gates':GATES,
            'production_options':PRODUCTION_OPTIONS,'reference_options':REFERENCE_OPTIONS,'source_sha256':hashes}
    name=args.output or fit_path.stem+'_independent_audit.json'
    destination=OUT/name
    try:
        reports,traces=run_trajectories(law,initial,waveforms,observations,mapping,True)
        output['full_trajectory_audits']=reports
        output['numerical_passed']=all(row['independent_integration']['passed'] and row['circuit_checks']['passed']
                                       for row in reports.values())
        output['same_state_access_counterfactual']=access_counterfactual(law,traces['1t1r_pair1_set'],waveforms['1t1r','set'])
        if isinstance(law,AccessGapLaw) and law.access_voltage_v > 0.:
            no_access=replace(law,access_voltage_v=0.)
            ablation,ablated=run_trajectories(no_access,initial,waveforms,observations,mapping,False,True)
            output['no_refit_B_zero_ablation']={'law':asdict(no_access),'initial_states':initial,
                'all_other_parameters_frozen':True,'full_trajectory_audits':ablation,
                'scope':'Each topology retains its fitted initial state; state is continuously recomputed, not copied from the access trajectory. Any production boundary failure is recorded and resolved only in this audit using independently compared DOP853/BDF event solutions.'}
        output['status']='completed_passed_numerical_gates' if output['numerical_passed'] else 'completed_with_failed_numerical_gates'
    except Exception as exc:
        output['status']='audit_failed_or_incomplete'
        output['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        output['elapsed_s']=time.monotonic()-started
        output['sources_unchanged_during_audit']=all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p.relative_to(ROOT))] for p in source_paths)
        output['limitations']=[
            'No microscopic species, heating law or contact mechanism is established by this audit.',
            'Reference integration independently constructs path and rates, but shares circuit.solve; algebraic current-space root checks separately validate sampled operating points.',
            'For bound-event localization the independent reference extends the incoming one-sided rate to clipped internal stages; the event uses the unprojected state, stops at first contact, and then enforces absorption.',
            'Numerical criteria cover this selected law, the full bare pair and two continuous 1T1R pairs; they do not certify arbitrary parameters or all 50 cycles.',
            'Training errors use raw record variability and eligibility masks; they are descriptive calibration scores, not a new holdout.',
            'The B=0 ablation is not reoptimized and is distinct from the separately refitted nested baseline.',
            'Bare same-label pairing, fixed transistor provenance and quasi-static voltage conventions remain conditional.',
            'The clock is commanded-voltage path, not seconds; transient compliance and capacitance are not simulated.']
        destination.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'output':str(destination),'status':output['status'],
                          'elapsed_s':output['elapsed_s'],'numerical_passed':output.get('numerical_passed')}),flush=True)


if __name__=='__main__': main()
