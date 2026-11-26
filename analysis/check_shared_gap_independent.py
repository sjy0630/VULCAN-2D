"""Frozen-parameter, independent integration validation of shared-gap candidate.

Uses the full measured command grids for one uninterrupted SET→RESET pair
in each topology. Radau and DOP853 integrate the voltage-path ODE with linear
command interpolation; neither reuses the production RK stepping code.
Independent voltage-space Brent roots additionally audit circuit inversion.
No parameters are fitted, and failures are preserved in the JSON output.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vulcan2d.shared_gap import GapLaw, BareGapCircuit, LookupGapCircuit, simulate_gap
from vulcan2d.transistor import measured_transistor_mapping

FOLDER = ROOT / 'analysis/shared_hbn_2026-11-03'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_rows(voltage, state, law, circuit):
    points = [circuit.solve(float(v), law, float(u)) for v, u in zip(voltage, state)]
    return {'state_u': np.asarray(state),
            'current_a': np.array([p.current_a for p in points]),
            'hbn_voltage_v': np.array([p.hbn_voltage_v for p in points]),
            'transistor_voltage_v': np.array([p.transistor_voltage_v for p in points]),
            'supply_voltage_v': np.array([p.supply_voltage_v for p in points]),
            'compliance_limited': np.array([p.compliance_limited for p in points]),
            'final_u': float(state[-1])}


def independent_phase(voltage, initial_u, law, circuit, initial_command, method):
    commands = np.r_[initial_command, voltage]
    path = np.r_[0., np.cumsum(np.abs(np.diff(commands)))]
    # Repeated voltages consume zero path. The interpolation uses unique path
    # values; every observation remains mapped back through `inverse`.
    unique_path, indices, inverse = np.unique(path, return_index=True, return_inverse=True)
    unique_command = commands[indices]
    outside_trials = 0
    def rhs(s, y):
        nonlocal outside_trials
        u = float(y[0])
        if not 0. <= u <= law.u_max:
            outside_trials += 1
        # Defines the projected ODE outside its feasible domain for the IVP
        # solver's trial evaluations only. Saved states must remain feasible.
        u = float(np.clip(u, 0., law.u_max))
        command = float(np.interp(s, unique_path, unique_command))
        return [law.derivative(u, circuit.solve(command, law, u).current_a)]
    start = time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        sol = solve_ivp(rhs, (0., float(unique_path[-1])), [float(initial_u)],
                        method=method, t_eval=unique_path, rtol=1e-9, atol=1e-11,
                        max_step=.01)
    details = {'success': bool(sol.success), 'message': str(sol.message),
               'nfev': int(sol.nfev), 'njev': int(sol.njev), 'nlu': int(sol.nlu),
               'elapsed_s': time.monotonic() - start,
               'outside_domain_trial_evaluations': outside_trials,
               'warnings': sorted({str(item.message) for item in caught}),
               'rtol': 1e-9, 'atol': 1e-11, 'max_step_path_v': .01,
               'path_length_v': float(unique_path[-1])}
    if not sol.success or len(sol.t) != len(unique_path):
        return None, details
    states = sol.y[0][inverse][1:]
    details['saved_min_u'] = float(np.min(states))
    details['saved_max_u'] = float(np.max(states))
    if np.any(states < 0.) or np.any(states > law.u_max):
        details.update(success=False, message='IVP saved state left physical interval; no clipping applied')
        return None, details
    return evaluate_rows(voltage, states, law, circuit), details


def compare_trace(trial, reference, voltage):
    delta_u = trial['state_u'] - reference['state_u']
    delta_i = trial['current_a'] - reference['current_a']
    valid = np.abs(voltage) > 1e-12
    delta_log = np.log10(np.maximum(np.abs(trial['current_a'][valid]), 1e-16)) - np.log10(np.maximum(np.abs(reference['current_a'][valid]), 1e-16))
    index = np.flatnonzero(valid)[int(np.argmax(np.abs(delta_log)))]
    return {'max_abs_delta_u': float(np.max(np.abs(delta_u))),
            'final_delta_u': float(delta_u[-1]),
            'max_abs_delta_current_a': float(np.max(np.abs(delta_i))),
            'rms_delta_current_a': float(np.sqrt(np.mean(delta_i**2))),
            'rms_delta_log10_current_decade': float(np.sqrt(np.mean(delta_log**2))),
            'max_abs_delta_log10_current_decade': float(np.max(np.abs(delta_log))),
            'largest_log_difference': {'sample_index': int(index),
                                       'command_voltage_v': float(voltage[index]),
                                       'reference_current_a': float(reference['current_a'][index]),
                                       'trial_current_a': float(trial['current_a'][index]),
                                       'reference_u': float(reference['state_u'][index]),
                                       'trial_u': float(trial['state_u'][index])}}


def independent_load(command, state, law, circuit):
    if isinstance(circuit, BareGapCircuit):
        current = law.current(command, state)
        if circuit.compliance_a is not None and abs(current) > circuit.compliance_a:
            current = float(np.copysign(circuit.compliance_a, command))
        return current, law.voltage(current, state), 0.
    sign, magnitude = np.sign(command), abs(command)
    if magnitude == 0.:
        return 0., 0., 0.
    side = 'reset' if command < 0. else 'set'
    mapping = circuit.mapping
    lookup = mapping._active_lookup(side)
    gate = mapping.effective_gate_voltage(circuit.gate_voltage, side)
    max_current = mapping.current_scale * lookup.current(lookup.vds_max, gate)
    max_terminal = lookup.vds_max + max_current * mapping.series_resistance_ohm
    low, high = max(0., magnitude - max_terminal), magnitude
    def residual(vh):
        vt = magnitude - vh
        it = abs(mapping.drain_current(sign * vt, circuit.gate_voltage))
        return law.current(vh, state) - it
    vh = brentq(residual, low, high, xtol=1e-13, rtol=1e-13)
    return sign * law.current(vh, state), sign * vh, sign * (magnitude - vh)


def load_audit(result, voltage, law, circuit):
    indices = np.unique(np.r_[np.linspace(0, len(voltage)-1, 31, dtype=int),
                              np.argmax(np.abs(result['current_a'])),
                              np.argmax(np.abs(voltage))])
    di, dvh, dvt, residual = [], [], [], []
    for index in indices:
        point = circuit.solve(float(voltage[index]), law, float(result['state_u'][index]))
        current, vh, vt = independent_load(float(voltage[index]), float(result['state_u'][index]), law, circuit)
        di.append(abs(current-point.current_a))
        dvh.append(abs(vh-point.hbn_voltage_v))
        dvt.append(abs(vt-point.transistor_voltage_v))
        residual.append(abs(point.supply_voltage_v-point.hbn_voltage_v-point.transistor_voltage_v))
    return {'success': True, 'n_sampled_points': len(indices),
            'max_abs_current_difference_a': float(max(di)),
            'max_abs_hbn_voltage_difference_v': float(max(dvh)),
            'max_abs_transistor_voltage_difference_v': float(max(dvt)),
            'max_abs_kvl_residual_v': float(max(residual))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fit', default='gap_joint_fit.json')
    parser.add_argument('--output', default='gap_independent_validation.json')
    args = parser.parse_args()
    fit_path, output_path = FOLDER / args.fit, FOLDER / args.output
    start = time.monotonic()
    fitted = json.loads(fit_path.read_text())
    selected = next(row for row in fitted['fits'] if row['start'] == fitted['selected_start'])
    law = GapLaw(**fitted['selected_law'])
    trans = LookupGapCircuit(measured_transistor_mapping(), 1.1)
    circuits = {'1r': (BareGapCircuit(1e-3), BareGapCircuit()), '1t1r': (trans, trans)}
    source_hash = digest(ROOT/'vulcan2d/shared_gap.py')
    report = {'status': 'in_progress', 'selected_start': fitted['selected_start'],
              'selected_q': selected['q'], 'selected_law': fitted['selected_law'],
              'input_fit_sha256': digest(fit_path), 'input_fit_file': str(fit_path.relative_to(ROOT)), 'source_sha256': source_hash,
              'source_matches_fit': source_hash == fitted['source_sha256']['vulcan2d/shared_gap.py'],
              'script_sha256': digest(Path(__file__)),
              'scope': 'First full SET→RESET pair in each topology; no state reset between phases; all original command points.',
              'voltage_clock': 'cumulative_absolute_command_voltage_path_v',
              'bounds_handling': 'RHS trial-state projection only; invalid saved states fail independently.',
              'topologies': {}, 'failures': []}
    if not report['source_matches_fit']:
        raise RuntimeError('Production source hash drifted after fitting')
    critical = law.i0_a*np.sinh(law.eth_set_v/law.v0_v)
    stop_u = max(0., float(np.log(critical/1e-3)))
    report['bare_set_analytical_stop'] = {'critical_current_a': float(critical),
                                          'compliance_a': .001, 'u_star': stop_u,
                                          'interpretation': 'Starting above u_star, positive-field evolution at ideal 1 mA compliance cannot cross below u_star; only a positive u_star enforces a finite gap.'}
    with np.load(FOLDER/'data/native_observations.npz') as data:
        for ti, (topology, ports) in enumerate(circuits.items()):
            source_cycle = fitted['split'][topology]['train'][0]
            voltages = [data[f'{topology}_cycle_{source_cycle}_{phase}_voltage'].copy()
                        for phase in ('set', 'reset')]
            initial = selected['q'][7+ti]
            block = {'source_cycle_voltage_grid': source_cycle, 'initial_u': initial,
                     'point_counts': list(map(len, voltages)), 'independent_integrators': {},
                     'rk4_refinement': {}, 'independent_circuit_checks': {}}
            report['topologies'][topology] = block
            traces = {}
            for method in ('Radau', 'DOP853'):
                state, command, results, diagnostics = initial, 0., [], []
                try:
                    for phase, voltage, port in zip(('set', 'reset'), voltages, ports):
                        result, detail = independent_phase(voltage, state, law, port, command, method)
                        detail['phase'] = phase
                        diagnostics.append(detail)
                        if result is None:
                            raise RuntimeError(detail['message'])
                        state, command = result['final_u'], float(voltage[-1])
                        results.append(result)
                    traces[method] = results
                    block['independent_integrators'][method] = {'success': True, 'phases': diagnostics,
                        'set_final_u': results[0]['final_u'], 'reset_final_u': results[1]['final_u'],
                        'set_min_u': float(np.min(results[0]['state_u']))}
                except Exception as exc:
                    block['independent_integrators'][method] = {'success': False, 'phases': diagnostics, 'error': repr(exc)}
                    report['failures'].append({'topology': topology, 'method': method, 'error': repr(exc)})
                output_path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
                print(json.dumps({'topology': topology, 'integrator': method, 'result': block['independent_integrators'][method]}), flush=True)
            if 'Radau' not in traces:
                continue
            reference = traces['Radau']
            if topology == '1r':
                block['reference_below_analytical_stop_u'] = max(0., stop_u - float(np.min(reference[0]['state_u'])))
            if 'DOP853' in traces:
                block['DOP853_vs_Radau'] = {phase: compare_trace(trial, ref, v)
                    for phase, trial, ref, v in zip(('set', 'reset'), traces['DOP853'], reference, voltages)}
            for phase, result, voltage, port in zip(('set', 'reset'), reference, voltages, ports):
                try:
                    block['independent_circuit_checks'][phase] = load_audit(result, voltage, law, port)
                except Exception as exc:
                    block['independent_circuit_checks'][phase] = {'success': False, 'error': repr(exc)}
                    report['failures'].append({'topology': topology, 'circuit_phase': phase, 'error': repr(exc)})
            for step in (.02, .005, .001):
                state, command, results = initial, 0., []
                try:
                    for voltage, port in zip(voltages, ports):
                        result = simulate_gap(voltage, state, law, port, max_step_v=step,
                                              initial_command_voltage_v=command)
                        state, command = result['final_u'], float(voltage[-1])
                        results.append(result)
                    entry = {'success': True, 'set_final_u': results[0]['final_u'],
                             'reset_final_u': results[1]['final_u'],
                             'phase_errors_vs_Radau': {phase: compare_trace(trial, ref, v)
                               for phase, trial, ref, v in zip(('set', 'reset'), results, reference, voltages)},
                             'metadata': [r['metadata'] for r in results]}
                    if topology == '1r':
                        entry['set_below_analytical_stop_u'] = max(0., stop_u-float(np.min(results[0]['state_u'])))
                    block['rk4_refinement'][str(step)] = entry
                except Exception as exc:
                    block['rk4_refinement'][str(step)] = {'success': False, 'error': repr(exc)}
                    report['failures'].append({'topology': topology, 'rk4_step': step, 'error': repr(exc)})
                output_path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    report['elapsed_s'] = time.monotonic()-start
    report['source_unchanged_at_end'] = digest(ROOT/'vulcan2d/shared_gap.py') == source_hash
    report['status'] = 'completed_with_preserved_failures' if report['failures'] else 'completed'
    summary = {'max_current_error_decade_by_rk4_step': {}}
    for step in ('.02', '.005', '.001'):
        entries = [entry['max_abs_delta_log10_current_decade']
                   for block in report['topologies'].values()
                   for entry in block['rk4_refinement'].get(str(float(step)), {}).get('phase_errors_vs_Radau', {}).values()]
        summary['max_current_error_decade_by_rk4_step'][str(float(step))] = max(entries) if entries else None
    cross = [entry['max_abs_delta_log10_current_decade']
             for block in report['topologies'].values()
             for entry in block.get('DOP853_vs_Radau', {}).values()]
    roots = [entry['max_abs_hbn_voltage_difference_v']
             for block in report['topologies'].values()
             for entry in block['independent_circuit_checks'].values() if entry['success']]
    summary['max_cross_integrator_current_difference_decade'] = max(cross) if cross else None
    summary['max_brent_voltage_difference_v'] = max(roots) if roots else None
    summary['bare_stopping_state_undershoot_by_step'] = {
        step: row['set_below_analytical_stop_u'] for step, row in
        report['topologies']['1r']['rk4_refinement'].items() if row['success']}
    summary['rk4_fine_current_accuracy_below_0p0001_decade'] = (
        summary['max_current_error_decade_by_rk4_step']['0.001'] is not None
        and summary['max_current_error_decade_by_rk4_step']['0.001'] < 1e-4)
    report['numerical_summary'] = summary
    report['numerical_conclusions'] = [
        f'The analytical bare SET stopping bound is u_star={stop_u:.12g}=max(0,log(Icrit_set/Icomp)); finite-gap lower bound: {stop_u > 0.}.',
        'The bare analytical stopping-state undershoot is reported for each RK4 step; agreement between only two coarse grids is insufficient when both violate this invariant.',
        f"Maximum current discrepancies versus Radau, in decade: {summary['max_current_error_decade_by_rk4_step']}.",
        f"Maximum Radau/DOP853 current difference: {summary['max_cross_integrator_current_difference_decade']} decade.",
        'Use refined traces for state/critical-current claims; a shared kernel and a small integration error do not establish physical explanatory accuracy.'
    ]
    report['limitations'] = ['One full pair per topology does not validate all 53 cycles or every parameter candidate.',
                             'Both independent IVP methods reuse the material law and prepared circuit; separate voltage-root checks isolate circuit inversion at sampled points.',
                             'Numerical agreement does not validate the physical mechanism or the voltage-path clock.',
                             'Parameters are frozen at the coarse-grid training selection; no refinement-dependent refit is performed.']
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(str(output_path), flush=True)


if __name__ == '__main__':
    main()
