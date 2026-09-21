"""Outward-conditioned stochastic bottleneck test for gate 1.3, trace 23.

The two outward jump locations define retrospective high/low configuration
labels; they are observations, not predictions. Transport and rates see only
the outward trace. The return is used for scoring after those fits.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from scipy.optimize import least_squares, minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'analysis'))
from diagnose_cross_polarity_physics import read_pairs, GateLookup
from vulcan2d.transistor import load_reverse_lookup
from vulcan2d.stochastic_bottleneck import (frozen_load, clock_grid, transition_log_probabilities,
                                           negative_log_likelihood, population, sample_paths, kinetic_parameters)
from vulcan2d.bidirectional_state import log_rates


def fit_transport_outward(va, observed, labels, lookup):
    valid = (va >= .08-1e-10) & (observed > 0)
    w = np.zeros(len(va))
    for state in (0, 1):
        choose = valid & (labels == state); w[choose] = .5/sum(choose)
    lo, hi = np.array([-14., 0., -12., .05]), np.array([-4., 15., -3., 20.])
    def residual(p):
        _, pred = frozen_load(va, p, labels, lookup.voltage, lookup.current)
        return np.log10(pred[valid]/observed[valid])*np.sqrt(w[valid])
    fits = []
    for start in ([-9., 6., -5., 4.], [-10., 10., -4., 1.], [-8., 3., -5., 10.]):
        f = least_squares(residual, start, bounds=(lo, hi), x_scale=hi-lo, max_nfev=300,
                          ftol=1e-9, xtol=1e-9, gtol=1e-9)
        fits.append({'parameters': f.x.tolist(), 'converged': bool(f.success), 'rmse_decade': float(np.sqrt(2*f.cost))})
    best = min(fits, key=lambda r:r['rmse_decade'])
    return {**best, 'multistart': fits, 'n_outward_points': len(va), 'n_scored_outward_points': int(sum(valid))}


def calibrate_outward_transport(vapp, observed, labels, lookup):
    """Discard every target return current before any transport initialization."""
    va = np.asarray(vapp); apex = int(np.argmax(va))
    if len(labels) != apex+1:
        raise ValueError('Exactly one label for each outward point required')
    return fit_transport_outward(va[:apex+1], np.asarray(observed)[:apex+1], labels, lookup)


def fit_rates_outward(grid, labels, candidate):
    bounds = [(-12., 18.), (-1., 1.)]
    if candidate != 'constant': bounds += [(-1., 1.)]
    if candidate == 'polarizable_saddle': bounds += [(0., .5)]
    starts = [[1., 0., .1, .02], [4., .2, .5, .02], [8., .4, 1., .1], [1., 0., -.3, 0.]]
    attempts = []
    def objective(p):
        return negative_log_likelihood(transition_log_probabilities(grid, p, candidate), labels)
    for seed in starts:
        x = np.array(seed[:len(bounds)])
        f = minimize(objective, x, method='L-BFGS-B', bounds=bounds,
                     options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-7, 'maxls': 40})
        attempts.append({'parameters': f.x.tolist(), 'negative_log_likelihood': float(f.fun),
                         'converged': bool(f.success), 'message': str(f.message), 'nit': f.nit})
    best = min(attempts, key=lambda r:r['negative_log_likelihood'])
    p = np.array(best['parameters']); lo, hi = np.array(bounds).T
    return {**best, 'multistart': attempts,
            'boundary_indices': np.flatnonzero(np.minimum(p-lo, hi-p)/(hi-lo) < 1e-4).tolist()}


def profile_rate_peak(grid, labels, start):
    profiles = []
    for fixed in (12., 18., 24., 30.):
        def objective(x):
            return negative_log_likelihood(transition_log_probabilities(grid, [fixed, *x], 'polarizable_saddle'), labels)
        attempts = []
        for seed in (start[1:], [.3, .4, .05], [.3, .1, .3]):
            f = minimize(objective, seed, method='L-BFGS-B', bounds=[(-1., 1.), (-1., 1.), (0., .5)],
                         options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-7, 'maxls': 40})
            attempts.append({'parameters': [fixed, *f.x.tolist()], 'converged': bool(f.success),
                             'negative_log_likelihood': float(f.fun)})
        best = min(attempts, key=lambda x:x['negative_log_likelihood'])
        profiles.append({**best, 'fixed_log_peak_rate_per_V': fixed, 'multistart': attempts})
    return profiles


def run():
    raw = read_pairs('1.3')[22]['reset']; va = np.abs(raw['voltage']); obs = np.abs(raw['current']); apex = int(np.argmax(va))
    events = json.loads((ROOT / 'analysis/negative_reentry_audit.json').read_text())
    trace = next(r for r in events['traces'] if r['gate_V'] == 1.3 and r['cycle_id'] == 23)
    if [r['kind'] for r in trace['events']] != ['drop', 'rise']:
        raise RuntimeError('The saved outward event assignment changed')
    drop, rise = [r['raw_index_after'] for r in trace['events']]
    z = np.ones(apex+1, dtype=int); z[drop:rise] = 0
    lookup = GateLookup(load_reverse_lookup(), 1.3)
    # Trim before fitting: target return currents/labels are never supplied.
    transport = calibrate_outward_transport(va, obs, z, lookup)
    grid_out = clock_grid(va[:apex+1], transport['parameters'], lookup.voltage, lookup.current, substeps=8)
    full_grid = clock_grid(va, transport['parameters'], lookup.voltage, lookup.current, substeps=8)
    refined_grid = clock_grid(va, transport['parameters'], lookup.voltage, lookup.current, substeps=16)
    v0, i0 = frozen_load(va, transport['parameters'], 0., lookup.voltage, lookup.current)
    v1, i1 = frozen_load(va, transport['parameters'], 1., lookup.voltage, lookup.current)
    full_labels = np.r_[z, np.ones(len(va)-apex-1, dtype=int)]
    return_mask = (np.arange(len(va)) > apex) & (va >= .08-1e-10) & (obs > 0)
    return_high_error = np.log10(i1[return_mask]/obs[return_mask])
    candidates = {}
    for name in ('constant', 'single_hop', 'polarizable_saddle'):
        fit = fit_rates_outward(grid_out, z, name); p = fit['parameters']
        logs = transition_log_probabilities(full_grid, p, name)
        refined = transition_log_probabilities(refined_grid, p, name)
        nll = negative_log_likelihood(logs[:apex], z)
        finer_nll = negative_log_likelihood(refined[:apex], z)
        return_logs = logs[apex:]; conditional_population = population(return_logs, initial_probability=1.)
        log_all_return_high = float(return_logs[:, 1, 1].sum())
        _, loss = log_rates(full_grid['on_voltage'][apex:], kinetic_parameters(p, name), 'polarizable_saddle')
        hazard = float(np.sum(np.exp(loss)*full_grid['path_increment'][apex:, None]))
        simulated = sample_paths(logs, n_paths=20000, seed=317)
        differences = np.diff(simulated.astype(int), axis=1)
        n_drop = (differences[:, :apex] < 0).sum(axis=1); n_rise = (differences[:, :apex] > 0).sum(axis=1)
        pair = (n_drop == 1) & (n_rise == 1) & (simulated[:, apex] == 1)
        with_return = pair & np.all(simulated[:, apex:] == 1, axis=1)
        drop_positions, rise_positions = [], []
        for row in np.flatnonzero(pair):
            drop_positions.append(float(va[np.flatnonzero(differences[row, :apex] < 0)[0]+1]))
            rise_positions.append(float(va[np.flatnonzero(differences[row, :apex] > 0)[0]+1]))
        candidates[name] = {'fit': fit, 'refined_outward_negative_log_likelihood': finer_nll,
                            'quadrature_nll_difference': abs(finer_nll-nll),
                            'maximum_transition_probability_refinement_difference': float(np.max(np.abs(np.exp(logs)-np.exp(refined)))),
                            'conditional_all_sampled_return_high_probability': float(np.exp(log_all_return_high)),
                            'conditional_uninterrupted_return_high_probability': float(np.exp(-hazard)),
                            'conditional_return_high_probability_by_point': conditional_population.tolist(),
                            'return_mean_current_A': ((1-conditional_population)*i0[apex:]+conditional_population*i1[apex:]).tolist(),
                            'sampling_seed': 317, 'n_sampled_paths': 20000,
                            'probability_sampled_one_drop_one_rise_outward': float(pair.mean()),
                            'probability_sampled_pair_and_high_return': float(with_return.mean()),
                            'drop_voltage_q10_median_q90_conditioned_on_pair': np.quantile(drop_positions, [.1, .5, .9]).tolist() if drop_positions else None,
                            'rise_voltage_q10_median_q90_conditioned_on_pair': np.quantile(rise_positions, [.1, .5, .9]).tolist() if rise_positions else None}
        print(name, 'outward NLL', finer_nll, 'return-high probability', np.exp(log_all_return_high),
              'paired path frequency', with_return.mean(), 'bounds', fit['boundary_indices'], flush=True)
    profiles = profile_rate_peak(grid_out, z, candidates['polarizable_saddle']['fit']['parameters'])
    for profile in profiles:
        transition = transition_log_probabilities(full_grid, profile['parameters'], 'polarizable_saddle')
        profile['conditional_all_sampled_return_high_probability'] = float(np.exp(transition[apex:, 1, 1].sum()))
        # If nu/r=1e13 /V, the fastest fitted rate corresponds to this barrier.
        # A negative result is retained as a failed conditional physical check.
        profile['conditional_minimum_barrier_eV_at_attempt_per_path_1e13'] = float(.025851999786435*(np.log(1e13)-profile['fixed_log_peak_rate_per_V']))
        print('Peak profile', profile['fixed_log_peak_rate_per_V'], profile['negative_log_likelihood'],
              profile['conditional_all_sampled_return_high_probability'], flush=True)
    paths = [Path(__file__), ROOT / 'vulcan2d/stochastic_bottleneck.py', ROOT / 'vulcan2d/bidirectional_state.py',
             ROOT / 'vulcan2d/_bidirectional_core.cpp', ROOT / 'vulcan2d/transistor.py',
             ROOT / 'analysis/diagnose_cross_polarity_physics.py', ROOT / 'analysis/negative_reentry_audit.json',
             ROOT / '1.3V栅压RESET.xlsx', ROOT / 'vulcan2d/data/bookoutput2_reverse_transistor_lookup.json']
    result = {'scope': 'Retrospectively selected two-jump trace; outward-conditioned stochastic mechanism probe, no microphysical identification',
              'gate_V': 1.3, 'cycle_id': 23, 'outward_apex_raw_index': apex,
              'outward_drop_raw_index': drop, 'outward_rise_raw_index': rise,
              'assigned_outward_binary_configuration': z.tolist(), 'applied_abs_voltage_V': va.tolist(),
              'observed_current_A': obs.tolist(), 'transport_fit_outward_only': transport,
              'frozen_low_current_A': i0.tolist(), 'frozen_high_current_A': i1.tolist(),
              'frozen_low_internal_voltage_V': v0.tolist(), 'frozen_high_internal_voltage_V': v1.tolist(),
              'held_return_if_high_rmse_decade': float(np.sqrt(np.mean(return_high_error**2))),
              'held_return_if_high_bias_decade': float(return_high_error.mean()), 'candidates': candidates,
              'polarizable_rate_peak_profiles': profiles,
              'limitations': ['Only two transitions inform kinetics; multiple parameter values and stochastic mechanisms remain indistinguishable.',
                              'Outward state labels are assigned using previously detected raw jumps; hidden subresolution jumps are not resolved.',
                              'Return probabilities refer to the observed high label; no high state is forced in the prediction.',
                              'Probability of high at sampled points differs from continuous survival between points.',
                              'The two frozen current levels are effective descriptions; a high-state transport mismatch can remain even when survival is plausible.',
                              'The rate law obeys an energy relation at a common internal voltage; evaluating rates on two circuit configurations does not prove detailed balance of the biased whole circuit.',
                              'The observed dataset informed this exploratory hypothesis; no external blind validation or generalization claim.',
                              'No microscopic species or absolute attempt rate, energy barrier, scan speed or dwell time is measured.'],
              'input_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (ROOT / 'analysis/stochastic_reentry_probe.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    run()
