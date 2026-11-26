"""A preregistered compromise trajectory, not a claimed feasible fitted seed."""
from pathlib import Path
import hashlib
import json
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from joint_problem import SpectralJointProblem
from vulcan2d.shared_spectral import SpectralShape, SpectralLaw
from vulcan2d.shared_spectral_ivp import SpectralTable, TabulatedSpectralLaw, simulate_spectral_ivp

HERE = Path(__file__).resolve().parent
Q = [-1.52287874528, -3.82390874094, -4.69897000434,
     .17609125906, -1., 11.89081621275, 9.34664821192]


def main():
    started = time.monotonic()
    output = {'status': 'started_diagnostic_compromise_not_prefilter_passed_not_fitted',
              'q': Q, 'scope': 'Original training records, real continuous32 series cycles, no holdout',
              'selection': 'Compromise from seed-screen agent before observing any ODE outcome', 'phases': []}
    path = HERE/'diagnostic_compromise.json'

    def write():
        path.write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')

    shape = SpectralShape.from_json()
    table = SpectralTable.load(HERE/'transport_g010_v0025.npz', shape)

    def factory(q, ron):
        if abs(ron-shape.ron_ohm) > 1e-8:
            raise ValueError('Shared training Ron differs from frozen spectral shape')
        return TabulatedSpectralLaw(SpectralLaw(shape, *10**np.array(q[:5])), table)

    def simulate(command, initial_u, law, circuit, **options):
        begin = time.monotonic()
        result = simulate_spectral_ivp(command, initial_u, law, circuit, **options)
        row = {'phase_index': len(output['phases']), 'initial_u': float(initial_u),
               'final_u': result['final_u'], 'peak_current_A': float(max(abs(result['current_a']))),
               'command_min_V': float(min(command)), 'command_max_V': float(max(command)),
               'n_points': len(command), 'elapsed_s': time.monotonic()-begin,
               'metadata': result['metadata']}
        output['phases'].append(row)
        write()
        print(json.dumps({k: v for k, v in row.items() if k != 'metadata'}), flush=True)
        return result

    problem = SpectralJointProblem(factory, mode='full', simulate_fn=simulate,
        integrator_options={'method': 'Radau', 'state_rtol': 1e-8, 'state_atol': 1e-10,
            'current_rtol': 1e-3, 'state_refine_atol': 2e-5, 'max_step_v': .05, 'max_refinements': 4})
    write()
    try:
        result = problem.evaluate(Q, mode='full')
        output['status'] = 'completed_diagnostic_full32_not_fitted'
        output['details'] = result['details']
        arrays = {}
        for (top, cid, phase), trace in result['predictions'].items():
            for key, value in trace.items():
                if isinstance(value, np.ndarray):
                    arrays[f'{top}_{cid}_{phase}_{key}'] = value
        np.savez_compressed(HERE/'diagnostic_compromise.npz', **arrays)
    except Exception as exc:
        output['status'] = 'diagnostic_trajectory_failed_preserved'
        output['failure'] = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
        print(output['failure']['traceback'], flush=True)
    output['elapsed_s'] = time.monotonic()-started
    output['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__), HERE/'joint_problem.py', ROOT/'vulcan2d/shared_spectral.py',
                  ROOT/'vulcan2d/shared_spectral_ivp.py', HERE/'transport_g010_v0025.npz']}
    write()
    print(json.dumps({'status': output['status'], 'elapsed_s': output['elapsed_s'],
                      'cost': output.get('details', {}).get('verified_cost')}), flush=True)


if __name__ == '__main__':
    main()
