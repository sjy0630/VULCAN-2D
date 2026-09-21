"""Post-hoc lever-arm profiles: can stronger equilibrium gating repair first3?

This explicitly uses target RESET for descriptive capacity, not prediction.
The primary grouped comparison remains archived and unchanged.
"""
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'analysis'))
from compare_parallel_paths import assemble, VOLTAGE
from compare_side_charge_readout import BOUNDS, STARTS, profile_supply, seed_energies, scores
from vulcan2d.side_charge import shape_and_gradients, predict


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_profile(y, weights, lever, baseline):
    n, m = y.shape; rootw = np.sqrt(weights/(n*weights.sum(axis=1)[:, None]))
    lower = np.r_[BOUNDS[0][:3], np.full(n, -.6)]
    upper = np.r_[BOUNDS[1][:3], np.full(n, .6)]
    cache = {}
    def evaluate(p):
        if 'p' not in cache or not np.array_equal(cache['p'], p):
            shared = np.r_[p[:3], lever]
            shape, grad, local, _ = shape_and_gradients(VOLTAGE, shared, p[3:])
            amp, inside = profile_supply(y, shape, weights)
            norm = weights/weights.sum(axis=1)[:, None]
            grad -= inside[:, None, None]*np.sum(norm[:, :, None]*grad, axis=1)[:, None, :]
            local -= inside[:, None]*np.sum(norm*local, axis=1)[:, None]
            cache.update(p=p.copy(), shape=shape, amp=amp, grad=grad, local=local)
        return cache
    def residual(p):
        c = evaluate(p)
        return ((c['shape']+c['amp'][:, None]-y)*rootw).ravel()
    def jacobian(p):
        c = evaluate(p); jac = np.zeros((n*m, n+3))
        jac[:, :3] = (c['grad'][:, :, :3]*rootw[:, :, None]).reshape(n*m, 3)
        jac[np.arange(n*m), np.repeat(np.arange(n)+3, m)] = (c['local']*rootw).ravel()
        return jac
    attempts = []
    for base in [baseline]+list(STARTS[1:3]):
        shared = np.r_[np.asarray(base)[:3], lever]
        initial = np.r_[shared[:3], seed_energies(VOLTAGE, y, weights, shared, 'equilibrium')]
        beginning = time.perf_counter()
        solution = least_squares(residual, np.clip(initial, lower+1e-8, upper-1e-8), jac=jacobian,
                                 bounds=(lower, upper), x_scale=np.r_[[2., .2, .5], np.full(n, .1)],
                                 max_nfev=900, ftol=2e-8, xtol=2e-8, gtol=2e-8)
        attempts.append({'parameters':solution.x.tolist(), 'rmse_decade':float(np.sqrt(2*solution.cost)),
                         'converged':bool(solution.success), 'nfev':solution.nfev,
                         'optimality':float(solution.optimality), 'seconds':time.perf_counter()-beginning})
    best = min(attempts, key=lambda a:a['rmse_decade']); c = evaluate(np.array(best['parameters']))
    return {'fixed_lever_arm':lever, 'shared_parameters':best['parameters'][:3]+[lever],
            'states':np.column_stack((c['amp'], best['parameters'][3:])).tolist(),
            'rmse_decade':best['rmse_decade'], 'converged':best['converged'], 'multistart':attempts}


def run():
    main_path = ROOT/'analysis/side_charge_readout.json'
    main = json.loads(main_path.read_text())
    for p, sha in main['input_sha256'].items():
        if digest(ROOT/p) != sha:
            raise RuntimeError('Input changed: '+p)
    ids, y, weights, _ = assemble()
    baseline = main['candidates']['equilibrium']['splits']['all_descriptive']['fit']['shared_parameters']
    output = ROOT/'analysis/side_charge_lever_profiles.json'
    files = [Path(__file__), main_path]+[ROOT/p for p in main['input_sha256']]
    result = {'scope':'Post-hoc descriptive lever profiles; target RESET is consumed; not a revision of grouped forecasts',
              'profiles':[], 'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in dict.fromkeys(files)}}
    for group, mask, values in [('all53', np.ones(len(ids), bool), (0., .25, .5, .75, .865, 1.)),
                                ('first3_only', ids <= 3, (.865, 1.))]:
        for lever in values:
            fitted = fit_profile(y[mask], weights[mask], lever, baseline)
            predicted = predict(VOLTAGE, fitted['shared_parameters'], fitted['states'])
            refined = predict(VOLTAGE, fitted['shared_parameters'], fitted['states'], order=48)
            row = {'group':group, 'fit':fitted, 'scores':scores(y[mask], predicted, weights[mask], ids[mask]),
                   'maximum_24_48_difference_decade':float(np.max(np.abs(predicted-refined))),
                   'predicted_log10_A':predicted.tolist()}
            result['profiles'].append(row)
            output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
            print(group, lever, fitted['rmse_decade'], fitted['converged'],
                  'first3',row['scores'].get('first3'), flush=True)


if __name__ == '__main__':
    run()
