"""Integration, grouped forecast scores, relaxation bounds and propagation."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'analysis'))
from compare_parallel_paths import assemble, VOLTAGE
from compare_side_charge_readout import scores
from vulcan2d.charge_spectrum import predict
from vulcan2d.coupled_charge_reduced import stationary_current
from vulcan2d.coupled_charge import VT, EDGES, log_rates
from vulcan2d.charge_moments import coefficients, propagate_means


def run():
    source = ROOT/'analysis/charge_spectrum_comparison.json'
    main = json.loads(source.read_text()); assert main['status'] == 'complete'
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for name, sha in main['input_sha256'].items():
        assert digest(ROOT/name) == sha
    ids, y, weights, _ = assemble()
    refinements, slopes, relaxation, direct = [], [], [], []
    pools = {p: np.empty_like(y) for p in ('SET_only', 'joint_descriptive')}
    for label, row in main['splits'].items():
        fit = row['fit']; shared = fit['shared_parameters']
        states = [('training', row['training_ids'], fit['states'])]
        states += [(p, row['target_ids'], t['identification']['states']) for p, t in row['targets'].items()]
        for scope, cycle_ids, local in states:
            primary = predict(VOLTAGE, shared, local)
            fine = predict(VOLTAGE, shared, local, panels=16)
            finer = predict(VOLTAGE, shared, local, panels=32)
            d1, d2 = float(np.max(np.abs(primary-fine))), float(np.max(np.abs(fine-finer)))
            assert d1 < 2e-7 and d2 < 1e-9
            refinements.append({'split': label, 'scope': scope, 'n_states': len(local),
                                'maximum_96_192_difference_decade': d1, 'maximum_192_384_difference_decade': d2})
            if label.startswith('fold_') and scope in pools:
                pools[scope][np.array(cycle_ids)-1] = finer
        p = np.array(shared[:5]); low = shared[0]-shared[5]
        logf = -np.logaddexp(0., low/VT)
        lower = np.exp(-np.logaddexp(0., -low/VT))/(1+np.sqrt(np.exp(logf)))
        row_relax = {'split': label, 'spectrum_minimum_energy_eV': low,
                     'slow_decay_relative_to_uncoupled_lower_bound': float(lower),
                     'bias_to_zero_slow_rate_ratio_upper_bound': float(1/lower),
                     'interpretation': 'All channel energies, trap states and positive constant side rates at the selected shared parameters only'}
        relaxation.append(row_relax)
        # Independent adaptive integral of the exact single-channel kernel.
        state = fit['states'][len(fit['states'])//2]
        for voltage in (-.3, -.1, .1, .3, 2., 3.):
            ref = predict([voltage], shared, state, panels=32)[0, 0]-state[0]
            def scaled_integrand(x):
                channel = p.copy(); channel[0] += shared[5]*x
                return 10**(stationary_current(voltage, channel, state[1])['log10_abs_current_scale']-ref)/2
            integral, error = quad(scaled_integrand, -1., 1., epsabs=1e-10, epsrel=1e-10, limit=160)
            delta = float(abs(np.log10(integral)))
            assert delta < 1e-8
            direct.append({'split': label, 'voltage_V': voltage,
                           'adaptive_vs_384_difference_decade': delta, 'adaptive_scaled_error_estimate': error})
        if label == 'all_descriptive':
            prediction = predict(VOLTAGE, shared, fit['states'], panels=32)
            for cid, obs, pred, w in zip(ids, y, prediction, weights):
                mask = (VOLTAGE >= 2) & (VOLTAGE <= 3) & (w > 0)
                x = np.log10(VOLTAGE[mask]); x -= x.mean()
                slopes.append({'cycle_id': int(cid), 'n_points': int(sum(mask)),
                               'observed_slope': float(obs[mask]@x/(x@x)),
                               'predicted_slope': float(pred[mask]@x/(x@x))})
    refined_scores = {p: scores(y, values, weights, ids) for p, values in pools.items()}
    rng = np.random.default_rng(21301)
    bound_checks, propagation = [], []
    for _ in range(100):
        p = [rng.uniform(-.5, .5), rng.uniform(.1, .9), rng.uniform(.1, .9), rng.uniform(0, .5), rng.uniform(-3, 3)]
        et, logt = rng.uniform(-.6, .6), rng.uniform(-3, 3)
        v = rng.uniform(-3, 3); c0 = coefficients(0., p, et, logt); cv = coefficients(v, p, et, logt)
        f = expit(-p[0]/VT); upper = (1+np.sqrt(f))/expit(p[0]/VT)
        ratio = cv['slow_decay_rate']/c0['slow_decay_rate']
        assert ratio <= upper*(1+1e-12)
        bound_checks.append(float(ratio/upper))
        initial = rng.dirichlet(np.ones(4)); dt = rng.uniform(.001, 2.)
        means = propagate_means([v], [dt], p, et, [initial[1]+initial[3], initial[2]+initial[3]], logt)[0]
        q = np.zeros((4, 4))
        for (a, b), rate in zip(EDGES, np.exp(log_rates(v, [*p, logt], et)[0])):
            q[b, a] += rate; q[a, a] -= rate
        exact = expm(q*dt)@initial
        delta = float(np.max(np.abs(means-[exact[1]+exact[3], exact[2]+exact[3]])))
        assert delta < 2e-11
        propagation.append(delta)
    files = [Path(__file__), source, ROOT/'vulcan2d/charge_moments.py', ROOT/'tests/test_charge_moments.py',
             ROOT/'analysis/CHARGE_RELAXATION_BOUND_2026-09-21.md']+[ROOT/p for p in main['input_sha256']]
    result = {'scope': 'Numerical and conditional relaxation audit; no calibrated retention time or native-terminal forecast',
              'quadrature_refinements': refinements, 'independent_adaptive_integrals': direct,
              'refined_384_point_five_fold_scores': refined_scores, 'positive_high_voltage_slopes': slopes,
              'conditional_relaxation_bounds': relaxation, 'random_relaxation_bound_checks': len(bound_checks),
              'maximum_random_rate_ratio_over_bound': max(bound_checks),
              'independent_four_state_propagation_checks': len(propagation),
              'maximum_mean_occupation_propagation_difference': max(propagation),
              'input_sha256': {str(p.relative_to(ROOT)): digest(p) for p in dict.fromkeys(files)}}
    (ROOT/'analysis/charge_spectrum_validation.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('Refinements', len(refinements), max(r['maximum_96_192_difference_decade'] for r in refinements), flush=True)
    print('Refined forecasts', refined_scores, flush=True)
    print('All-data conditional relaxation bound', relaxation[0], flush=True)


if __name__ == '__main__':
    run()
