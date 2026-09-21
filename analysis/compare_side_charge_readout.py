"""All53 constitutive charge-readout test and held-pair SET-conditioned RESET.

Internal coordinates inherit observed-current de-embedding. Native terminal
validation is separate, so these scores are not claimed as port forecasts.
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
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'analysis'))
from compare_parallel_paths import assemble, VOLTAGE, FOLDS
from vulcan2d.side_charge import shape_and_gradients, predict

BOUNDS = (np.array([.25, .005, 0., -1.]), np.array([6., 2., 2., 1.]))
STARTS = ([.8, .12, .5, .6], [2., .20, .8, .9], [4., .08, .5, .98], [5., .3, 1.2, -.6])


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile_supply(y, shape, weights):
    amp = np.sum(weights*(y-shape), axis=1)/weights.sum(axis=1)
    return np.clip(amp, -7., -1.), (amp > -7.) & (amp < -1.)


def seed_energies(v, y, weights, shared, mode):
    grid = np.linspace(-.599, .599, 65)
    shape = shape_and_gradients(v, shared, grid, mode=mode)[0]
    energies = []
    for observed, w in zip(y, weights):
        amps, _ = profile_supply(observed[None, :], shape, np.broadcast_to(w, shape.shape))
        loss = np.sum(w*(shape+amps[:, None]-observed)**2, axis=1)
        energies.append(grid[np.argmin(loss)])
    return energies


def fit(v, y, weights, mode, *, fixed_shared=None, max_nfev=450):
    y, weights = np.asarray(y), np.asarray(weights)
    n, m = y.shape
    root_weight = np.sqrt(weights/(weights.sum(axis=1)[:, None]*n))
    offset = (4 if mode == 'equilibrium' else 3) if fixed_shared is None else 0
    lower = np.r_[BOUNDS[0][:offset], np.full(n, -.6)]
    upper = np.r_[BOUNDS[1][:offset], np.full(n, .6)]
    scales = np.r_[np.array([2., .2, .5, .5])[:offset], np.full(n, .1)]
    cache = {}

    def evaluate(parameters):
        if 'p' not in cache or not np.array_equal(parameters, cache['p']):
            shared = np.array(fixed_shared) if not offset else np.r_[parameters[:offset], ([0.] if offset == 3 else [])]
            shape, grad, local, occupancy = shape_and_gradients(v, shared, parameters[offset:], mode=mode)
            amp, inside = profile_supply(y, shape, weights)
            norm = weights/weights.sum(axis=1)[:, None]
            # Differentiate the analytically profiled supply coefficient.
            grad -= inside[:, None, None]*np.sum(norm[:, :, None]*grad, axis=1)[:, None, :]
            local -= inside[:, None]*np.sum(norm*local, axis=1)[:, None]
            cache.update(p=parameters.copy(), shared=shared, amp=amp, shape=shape,
                         grad=grad, local=local, occupancy=occupancy)
        return cache

    def residual(parameters):
        c = evaluate(parameters)
        return ((c['amp'][:, None]+c['shape']-y)*root_weight).ravel()

    def jacobian(parameters):
        c = evaluate(parameters)
        matrix = np.zeros((n*m, offset+n))
        if offset:
            matrix[:, :offset] = (c['grad'][:, :, :offset]*root_weight[:, :, None]).reshape(n*m, offset)
        matrix[np.arange(n*m), np.repeat(np.arange(n)+offset, m)] = (c['local']*root_weight).ravel()
        return matrix

    starts = [fixed_shared] if fixed_shared is not None else STARTS
    attempts = []
    for shared in starts:
        energies = seed_energies(v, y, weights, shared, mode)
        initial = np.r_[np.asarray(shared)[:offset], energies]
        initial = np.clip(initial, lower+1e-8, upper-1e-8)
        beginning = time.perf_counter()
        result = least_squares(residual, initial, jac=jacobian, bounds=(lower, upper),
                               x_scale=scales, max_nfev=max_nfev, ftol=2e-8, xtol=2e-8, gtol=2e-8)
        attempts.append({'parameters':result.x.tolist(), 'rmse_decade':float(np.sqrt(2*result.cost)),
                         'converged':bool(result.success), 'nfev':result.nfev, 'message':result.message,
                         'optimality':float(result.optimality), 'seconds':time.perf_counter()-beginning})
    best = min(attempts, key=lambda a:a['rmse_decade']); c = evaluate(np.array(best['parameters']))
    states = np.column_stack((c['amp'], best['parameters'][offset:]))
    return {'mode':mode, 'shared_parameters':c['shared'].tolist(), 'states':states.tolist(),
            'rmse_decade':best['rmse_decade'], 'converged':best['converged'], 'multistart':attempts,
            'supply_boundary_count':int(np.sum((states[:, 0] <= -7+1e-5) | (states[:, 0] >= -1-1e-5))),
            'energy_boundary_count':int(np.sum(np.abs(states[:, 1]) >= .6-1e-5))}


def scores(y, predicted, weights, indices):
    error = predicted-y
    groups = {'all':np.ones(len(indices), bool), 'first3':indices <= 3, 'later':indices > 3}
    output = {}
    for group, selected in groups.items():
        if not np.any(selected):
            continue
        e, w = error[selected], weights[selected]
        row = {'n_pairs':int(np.sum(selected)), 'weighted_rmse_decade':float(np.sqrt(np.mean(np.sum(w*e*e, axis=1)/w.sum(axis=1))))}
        for name, mask in (('positive_low', (VOLTAGE > 0) & (VOLTAGE <= .3)),
                           ('positive_high', VOLTAGE > .3), ('negative', VOLTAGE < 0)):
            row[name+'_rmse_decade'] = float(np.sqrt(np.mean([np.mean(a[mask & (b > 0)]**2) for a, b in zip(e, w)])))
        output[group] = row
    return output


def run(quick=False, only=None):
    reference_path = ROOT/'analysis/parallel_path_comparison.json'
    reference = json.loads(reference_path.read_text())
    for p, sha in reference['input_sha256'].items():
        if digest(ROOT/p) != sha:
            raise RuntimeError('Inherited source changed: '+p)
    ids, y, weights, selection = assemble()
    if selection != reference['selection']:
        raise RuntimeError('Inherited data selection changed')
    paths = [Path(__file__), ROOT/'vulcan2d/side_charge.py', ROOT/'tests/test_side_charge.py',
             ROOT/'analysis/SIDE_CHARGE_READOUT_DESIGN_2026-09-21.md', reference_path,
             ROOT/'vulcan2d/barrier_state.py']+[ROOT/p for p in reference['input_sha256']]
    provenance = {str(p.relative_to(ROOT)):digest(p) for p in dict.fromkeys(paths)}
    suffix = ('_'+only if only else '')+('_quick' if quick else '')
    path = ROOT/'analysis'/f'side_charge_readout{suffix}.json'
    report = {'scope':'All53 equilibrium charge readout, internal constitutive comparison; no structure evolution or autonomous terminal claim',
              'status':'running', 'voltage_V':VOLTAGE.tolist(), 'ids':ids.tolist(), 'observed_log10_current_A':y.tolist(),
              'weights':weights.tolist(), 'selection':selection, 'input_sha256':provenance, 'candidates':{},
              'quick_without_holdouts':quick, 'shared_parameter_order':['width_nm', 'low_barrier_ev', 'height_difference_ev', 'signed_lever_arm'],
              'state_order':['log10_supply_S', 'addition_energy_ev']}
    if path.exists():
        report = json.loads(path.read_text())
        if report['input_sha256'] != provenance:
            raise RuntimeError('Preserve results before running changed sources')
    def save():
        path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    for mode in ([only] if only else ['frozen', 'equilibrium']):
        candidate = report['candidates'].setdefault(mode, {'splits':{}})
        splits = [('all_descriptive', np.ones(len(ids), bool), np.ones(len(ids), bool))]
        if not quick:
            splits += [(f'fold_{lo}_{hi}', (ids < lo) | (ids > hi), (ids >= lo) & (ids <= hi)) for lo, hi in FOLDS]
            splits += [('chronological', ids <= 35, ids > 35)]
        for label, train, target in splits:
            if label in candidate['splits']:
                continue
            print('Fitting', mode, label, 'training pairs', int(sum(train)), flush=True)
            found = fit(VOLTAGE, y[train], weights[train], mode)
            row = {'training_ids':ids[train].tolist(), 'target_ids':ids[target].tolist(), 'fit':found, 'targets':{}}
            predicted = predict(VOLTAGE, found['shared_parameters'], found['states'], mode=mode)
            row['training_scores'] = scores(y[train], predicted, weights[train], ids[train])
            if label == 'all_descriptive':
                row['all_descriptive_predicted_log10_A'] = predicted.tolist()
            else:
                for name, mask in (('SET_only', VOLTAGE > 0), ('joint_descriptive', np.ones(len(VOLTAGE), bool))):
                    local = fit(VOLTAGE[mask], y[target][:, mask], weights[target][:, mask], mode,
                                fixed_shared=found['shared_parameters'])
                    p = predict(VOLTAGE, found['shared_parameters'], local['states'], mode=mode)
                    row['targets'][name] = {'identification':local, 'predicted_log10_A':p.tolist(),
                                            'scores':scores(y[target], p, weights[target], ids[target])}
            candidate['splits'][label] = row; save()
            print(mode, label, 'training',found['rmse_decade'], 'shared',found['shared_parameters'],
                  'converged', found['converged'], 'starts',[(a['nfev'], round(a['rmse_decade'], 5), a['converged']) for a in found['multistart']], flush=True)
            if row['targets']:
                print('SET-only forecast', row['targets']['SET_only']['scores'], flush=True)
        if not quick:
            for protocol in ('SET_only', 'joint_descriptive'):
                predictions = np.empty_like(y)
                for label, row in candidate['splits'].items():
                    if label.startswith('fold_'):
                        predictions[np.array(row['target_ids'])-1] = row['targets'][protocol]['predicted_log10_A']
                candidate[protocol+'_five_fold_scores'] = scores(y, predictions, weights, ids)
    report['status'] = 'complete'; save()
    print(path, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--only', choices=['frozen', 'equilibrium'])
    args = parser.parse_args(); run(args.quick, args.only)
