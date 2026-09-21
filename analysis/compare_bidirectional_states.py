"""Forward-circuit tests of minimum reversible state laws on complete traces.

This first stage is descriptive mechanism testing, not a SET-to-RESET forecast.
All candidate currents are generated from applied voltage; target currents are
used only as observations in fitting/scoring. Event selection is retrospective.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'analysis'))
from diagnose_cross_polarity_physics import read_pairs, GateLookup, DATASETS
from vulcan2d.transistor import load_reverse_lookup
from vulcan2d.bidirectional_state import NAMES, trajectory, parameter_bounds, log_rates, conditional_barriers

CASES = [('1.3', 23), ('0.9', 6), ('0.9', 14), ('1.3', 10), ('1.3', 22), ('1.3', 52),
         ('1.1', 1), ('1.1', 3), ('1.1', 4), ('1.1', 27), ('1.1', 53)]


def prepare(pair, lookup):
    va = np.abs(pair['reset']['voltage']); obs = np.abs(pair['reset']['current'])
    apex = int(np.argmax(va)); index = np.arange(len(va))
    valid = (va >= .08-1e-10) & np.isfinite(obs) & (obs > 0)
    weights = np.zeros(len(va))
    masks = [(index <= apex) & (va <= .3), (index <= apex) & (va > .3), index > apex]
    for mass, mask in zip((.25, .25, .5), masks):
        selected = mask & valid
        if not np.any(selected):
            raise RuntimeError('Missing branch window')
        weights[selected] = mass / sum(selected)
    capacity = np.interp(va, lookup.voltage, lookup.current)
    return {'vapp': va, 'observed': obs, 'weights': weights, 'masks': masks,
            'n_above_transistor_capacity_all_nonzero': int(sum((va > 1e-12) & (obs > capacity))),
            'n_above_transistor_capacity_scored': int(sum(valid & (obs > capacity))),
            'zero_reference_A': float(max(obs[va <= 1e-12]))}


def metrics(data, prediction):
    w = data['weights']; valid = w > 0
    if np.any(prediction[valid] <= 0) or np.any(~np.isfinite(prediction[valid])):
        raise RuntimeError('Invalid scored prediction')
    error = np.zeros(len(w)); error[valid] = np.log10(prediction[valid] / data['observed'][valid])
    return {'rmse_decade': float(np.sqrt(np.sum(w*error*error))),
            **{name+'_rmse_decade': float(np.sqrt(np.mean(error[valid & mask]**2)))
               for name, mask in zip(('low_outward', 'high_outward', 'return'), data['masks'])},
            'maximum_error_decade': float(np.max(np.abs(error[valid])))}


def fit_case(data, lookup, name, starts=4, max_nfev=180):
    lo, hi = parameter_bounds(name); va = data['vapp']; valid = data['weights'] > 0
    target = np.log10(data['observed'][valid]); sw = np.sqrt(data['weights'][valid])
    failed_integrations = 0
    def simulate(p, **kw):
        return trajectory(va, p, name, lookup.voltage, lookup.current, **kw)
    def residual(p):
        nonlocal failed_integrations
        try:
            pred = simulate(p)['current_a']
            if np.any(pred[valid] <= 0) or np.any(~np.isfinite(pred[valid])):
                raise RuntimeError('Nonpositive prediction')
            return (np.log10(pred[valid])-target)*sw
        except RuntimeError:
            failed_integrations += 1
            return np.full(sum(valid), 30./np.sqrt(sum(valid)))
    guesses = [
        [-9., 6., -5.5, 5., 6., .2, .3, .5, .02],
        [-8.5, 4., -5., 2., 10., -.2, -.5, .5, .1],
        [-9., 8., -5.8, 8., 15., .1, .4, .9, .02],
        [-10., 10., -4.8, 2., 8., 8., 0., -.1, .1, .3],
    ]
    # Each row is the same physical starting point for the nested controls.
    guesses[3] = [-10., 10., -4.8, 2., 8., 0., -.1, .1, .3]
    attempts = []
    for k, start in enumerate(guesses[:starts]):
        initial = np.array(start[:8] if name == 'single_hop' else start)
        initial = np.clip(initial, lo+1e-7, hi-1e-7)
        beginning = time.perf_counter()
        f = least_squares(residual, initial, bounds=(lo, hi), x_scale=hi-lo,
                          diff_step=5e-4, max_nfev=max_nfev, ftol=2e-5, xtol=2e-5, gtol=2e-5)
        record = {'start_index': k, 'success': bool(f.success), 'nfev': f.nfev,
                  'rmse_decade': float(np.sqrt(2*f.cost)), 'parameters': f.x.tolist(),
                  'seconds': time.perf_counter()-beginning, 'optimality': float(f.optimality),
                  'message': f.message}
        attempts.append(record)
        print(name, 'start', k, 'RMSE', record['rmse_decade'], 'converged', f.success,
              'seconds', round(record['seconds'], 1), flush=True)
    completed = [r for r in attempts if r['success'] and r['rmse_decade'] < 29.]
    best = min(completed or attempts, key=lambda r:r['rmse_decade'])
    p = np.array(best['parameters']); prediction = simulate(p, rtol=2e-8, atol=2e-10)
    training_precision = simulate(p)
    refinement = np.abs(np.log10(prediction['current_a'][valid] / training_precision['current_a'][valid]))
    independent = simulate(p, rtol=2e-8, atol=2e-10, method='Radau', max_rhs_calls=100000)
    difference = np.abs(np.log10(prediction['current_a'][valid] / independent['current_a'][valid]))
    grid = np.linspace(0, 1.7, 171); lb, lr = log_rates(grid, p, name)
    eb, er = conditional_barriers(grid, p, name)
    return {'parameters': p.tolist(), 'converged': best['success'], 'multistart': attempts,
            'failed_trial_integrations': failed_integrations,
            'parameter_boundary_indices': np.flatnonzero(np.minimum(p-lo, hi-p)/(hi-lo) < 1e-4).tolist(),
            'scores': metrics(data, prediction['current_a']),
            'maximum_solver_refinement_difference_decade': float(refinement.max()),
            'maximum_independent_solver_difference_decade': float(difference.max()),
            'max_kcl_relative_error': prediction['max_kcl_relative_error'],
            'numerical_check_passed': bool(refinement.max() < .002 and difference.max() < .002),
            'voltage_grid_V': grid.tolist(), 'log_build_rate_per_V': lb.tolist(), 'log_remove_rate_per_V': lr.tolist(),
            'conditional_attempt_per_path_per_V': 1e13,
            'conditional_minimum_build_barrier_eV': float(eb.min()),
            'conditional_minimum_remove_barrier_eV': float(er.min()),
            'predicted_current_A': prediction['current_a'].tolist(),
            'predicted_internal_voltage_V': prediction['internal_voltage_v'].tolist(),
            'predicted_log_state': prediction['log_state'].tolist()}


def run(only=None, starts=4, max_nfev=180, candidate=None):
    cases = CASES if only is None else [(only.split(':')[0], int(only.split(':')[1]))]
    sources = [Path(__file__), ROOT / 'vulcan2d/bidirectional_state.py', ROOT / 'vulcan2d/_bidirectional_core.cpp',
               ROOT / 'analysis/diagnose_cross_polarity_physics.py', ROOT / 'vulcan2d/transistor.py',
               ROOT / 'vulcan2d/data/bookoutput2_reverse_transistor_lookup.json',
               ROOT / 'analysis/BIDIRECTIONAL_STATE_DESIGN_2026-09-21.md']
    for gate, cycle in cases:
        pair = next(p for p in read_pairs(gate) if p['cycle_id'] == cycle)
        lookup = GateLookup(load_reverse_lookup(), float(gate)); data = prepare(pair, lookup)
        for name in NAMES if candidate is None else (candidate,):
            print('Starting', gate, cycle, name, flush=True)
            result = fit_case(data, lookup, name, starts, max_nfev)
            files = sources+[ROOT / DATASETS[gate][1]]
            report = {'scope': 'Descriptive full-trace fit through the forward circuit; not a held-out prediction or unified SET/RESET state',
                      'gate_V': float(gate), 'cycle_id': cycle, 'candidate': name,
                      'initial_log_state': 0., 'state_clock': 'Applied-voltage path, no repeated-voltage dwell',
                      'parameter_names': ['log10_AH_A', 'beta_per_sqrtV', 'log10_AC_A', 'alpha_per_V',
                                          'log_peak_rate_per_V', 'delta0_eV', 'effective_charge_e', 'saddle_fraction']
                                          +(['common_saddle_lowering_eV_per_V2'] if name == 'polarizable_saddle' else []),
                      'weighting': 'Each trace: 1/4 outward applied <=0.3V, 1/4 other outward, 1/2 return; positive currents at applied >=0.08V',
                      'selection': {k: data[k] for k in ('n_above_transistor_capacity_all_nonzero', 'n_above_transistor_capacity_scored', 'zero_reference_A')},
                      'applied_abs_voltage_V': data['vapp'].tolist(), 'observed_current_A': data['observed'].tolist(),
                      'weights': data['weights'].tolist(), 'fit': result,
                      'limitations': ['A single negative trace is allowed independent transport and kinetic parameters; no cross-scan initialization claim.',
                                      'Common H/C envelopes remain effective nuisance transport, not identified PF/hopping physics.',
                                      'Conditional barriers use nu/sweep_rate=1e13 per V solely as a stated example, not measured timing.',
                                      'The scalar state is not uniquely separable from transport and is not assigned a microscopic species.',
                                      'All observations including return are fitted; selected events are retrospective.',
                                      'Optimization convergence and multiple starts do not establish a global optimum.',
                                      'Rates are normalized to a peak over 0-1.7V; peak prior is exp(-12) to exp(18) per V, and lambda prior is 0 to 0.5 eV/V^2.',
                                      'Transistor capacity violations remain in scoring; their fit errors cannot be repaired by the passive device law.'],
                      'input_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
            out = ROOT / 'analysis' / f'bidirectional_{gate}_{cycle}_{name}.json'
            out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
            print('Finished', gate, cycle, name, result['scores'], 'numerics', result['numerical_check_passed'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--only'); parser.add_argument('--candidate', choices=NAMES)
    parser.add_argument('--starts', type=int, default=4); parser.add_argument('--max-nfev', type=int, default=180)
    args = parser.parse_args(); run(args.only, args.starts, args.max_nfev, args.candidate)
