"""Append an independently integrated second 1T1R pair to the frozen audit.

All three methods start at the same first-pair Radau RESET endpoint. Only
this additional pair is integrated; no previous audit or fitting is repeated.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vulcan2d.shared_gap import GapLaw, LookupGapCircuit, simulate_gap
from vulcan2d.transistor import measured_transistor_mapping
from check_shared_gap_independent import independent_phase, compare_trace, digest

FOLDER = ROOT/'analysis/shared_hbn_2026-11-03'
AUDIT = FOLDER/'gap_independent_validation.json'
FIT = FOLDER/'gap_joint_fit.json'


def main():
    started = time.monotonic()
    report = json.loads(AUDIT.read_text())
    fitted = json.loads(FIT.read_text())
    source_hash = digest(ROOT/'vulcan2d/shared_gap.py')
    if source_hash != report['source_sha256'] or digest(FIT) != report['input_fit_sha256']:
        raise RuntimeError('Frozen source or fitted parameter file changed since first-pair audit')
    first = report['topologies']['1t1r']['independent_integrators']['Radau']
    if not first['success']:
        raise RuntimeError('First-pair independent endpoint unavailable')
    initial = first['reset_final_u']
    law = GapLaw(**fitted['selected_law'])
    circuit = LookupGapCircuit(measured_transistor_mapping(), 1.1)
    with np.load(FOLDER/'data/native_observations.npz') as data:
        voltages = [data[f'1t1r_cycle_5_{phase}_voltage'].copy() for phase in ('set', 'reset')]
        for phase, v in zip(('set', 'reset'), voltages):
            if not np.array_equal(v, data[f'1t1r_cycle_4_{phase}_voltage']):
                raise RuntimeError('Second-pair voltage waveform differs from first')
    block = {'status': 'in_progress',
             'purpose': 'Check that repeated-cycle stability is shared by independent integrators, rather than a production RK fixed point.',
             'initial_u_all_methods': initial,
             'initial_u_source': 'First complete 1T1R pair, independent Radau RESET endpoint',
             'source_cycle_voltage_grid': 5,
             'all_methods_share_identical_initial_state': True,
             'set_reset_state_transfer_continuous': True,
             'script_sha256': digest(Path(__file__)),
             'source_sha256': source_hash, 'results': {}, 'comparisons': {}, 'failures': []}
    traces = {}
    for method in ('Radau', 'DOP853', 'RK4_0.001'):
        state, command, results, details = initial, 0., [], []
        try:
            for phase, voltage in zip(('set', 'reset'), voltages):
                if method.startswith('RK4'):
                    result = simulate_gap(voltage, state, law, circuit, max_step_v=.001,
                                          initial_command_voltage_v=command)
                    detail = {'success': True, 'phase': phase, 'metadata': result['metadata']}
                else:
                    result, detail = independent_phase(voltage, state, law, circuit, command, method)
                    detail['phase'] = phase
                    if result is None:
                        raise RuntimeError(detail['message'])
                results.append(result)
                details.append(detail)
                state, command = result['final_u'], float(voltage[-1])
            traces[method] = results
            block['results'][method] = {'success': True, 'phases': details,
                'set_final_u': results[0]['final_u'], 'reset_final_u': results[1]['final_u'],
                'signed_cycle_map_residual_u': state-initial,
                'absolute_cycle_map_residual_u': abs(state-initial)}
        except Exception as exc:
            block['results'][method] = {'success': False, 'phases': details, 'error': repr(exc)}
            block['failures'].append({'method': method, 'error': repr(exc)})
    if 'Radau' in traces:
        for method in ('DOP853', 'RK4_0.001'):
            if method in traces:
                block['comparisons'][method+'_vs_Radau'] = {
                    phase: compare_trace(trial, reference, v) for phase, trial, reference, v in
                    zip(('set', 'reset'), traces[method], traces['Radau'], voltages)}
    block['source_unchanged_at_end'] = source_hash == digest(ROOT/'vulcan2d/shared_gap.py')
    block['elapsed_s'] = time.monotonic()-started
    block['status'] = 'completed_with_preserved_failures' if block['failures'] else 'completed'
    block['limitations'] = [
        'This checks the deterministic map for the repeated measured voltage waveform and frozen parameters, not cycle-to-cycle experimental variation.',
        'Small cycle-map residual at this initial state is not a global stability or uniqueness proof.',
        'Independent solvers reuse the material and circuit law; their time/path integration algorithms are independent of production RK4.'
    ]
    report['continuous_second_pair'] = block
    AUDIT.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(block, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
