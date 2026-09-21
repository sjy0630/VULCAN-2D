"""Bounded charge spectrum, whole-pair holdouts and zero-width control."""
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'analysis'))
from compare_parallel_paths import assemble, VOLTAGE, FOLDS
from compare_side_charge_readout import scores
from compare_coupled_charge import amplitude
from vulcan2d.charge_spectrum import shape_and_gradient, predict

LO = np.array([-.5, .01, .01, 0., -8., 0.])
HI = np.array([.5, .99, .99, 1., 8., .6])
STARTS = ([.08, .12, .7, .3, 1., .1], [.1, .08, .12, .5, -2., .2],
          [.02, .05, .1, .8, 3., .4], [.15, .7, .1, .7, -3., .55])


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit(v, y, w, *, fixed_shared=None, zero_width_fit=None, max_nfev=400):
    n, m = y.shape; offset = 6 if fixed_shared is None else 0
    lower = np.r_[LO[:offset], np.full(n, -.6)]; upper = np.r_[HI[:offset], np.full(n, .6)]
    normalized = w/w.sum(axis=1)[:, None]
    sw = np.sqrt(normalized/n); cache = {}
    def evaluate(x):
        if 'x' not in cache or not np.array_equal(x, cache['x']):
            shared = x[:offset] if offset else np.asarray(fixed_shared)
            out = shape_and_gradient(v, shared, x[offset:])
            shape = out['log10_abs_current_scale']; a, inside = amplitude(y, shape, w)
            grad = out['derivatives']
            grad = grad-inside[:, None, None]*np.sum(normalized[:, :, None]*grad, axis=1)[:, None, :]
            cache.update(x=x.copy(), shared=shared, shape=shape, amplitude=a, grad=grad)
        return cache
    def residual(x):
        c = evaluate(x)
        return ((c['shape']+c['amplitude'][:, None]-y)*sw).ravel()
    def jacobian(x):
        c = evaluate(x); j = np.zeros((n*m, offset+n))
        if offset: j[:, :offset] = (c['grad'][..., :6]*sw[:, :, None]).reshape(n*m, 6)
        j[np.arange(n*m), np.repeat(np.arange(n)+offset, m)] = (c['grad'][..., 6]*sw).ravel()
        return j
    starts = []
    for p in ([fixed_shared] if fixed_shared is not None else STARTS):
        grid = np.linspace(-.599, .599, 41)
        shape = shape_and_gradient(v, p, grid)['log10_abs_current_scale']
        energies = []
        for obs, weight in zip(y, w):
            a, _ = amplitude(obs[None, :], shape, np.broadcast_to(weight, shape.shape))
            energies.append(grid[np.argmin(np.sum(weight*(shape+a[:, None]-obs)**2, axis=1))])
        starts.append(('prespecified_grid', np.r_[np.asarray(p)[:offset], energies]))
    if zero_width_fit is not None:
        assert offset == 6 and len(zero_width_fit['states']) == n
        p = np.r_[zero_width_fit['shared_parameters'][:5], 0.]
        starts.append(('same_training_zero_width_fit', np.r_[p, np.array(zero_width_fit['states'])[:, 1]]))
    attempts = []
    for start_number, (source, initial) in enumerate(starts):
        beginning = time.perf_counter()
        out = least_squares(residual, np.clip(initial, lower+1e-9, upper-1e-9), jac=jacobian,
                            bounds=(lower, upper), x_scale=np.r_[[.15, .25, .25, .3, 3., .25][:offset], np.full(n, .1)],
                            max_nfev=max_nfev, ftol=3e-8, xtol=3e-8, gtol=3e-8)
        record = {'parameters': out.x.tolist(), 'source': source, 'rmse_decade': float(np.sqrt(2*out.cost)),
                  'converged': bool(out.success), 'nfev': out.nfev, 'message': out.message,
                  'optimality': float(out.optimality), 'seconds': time.perf_counter()-beginning}
        attempts.append(record)
        print(' start', start_number, record['rmse_decade'], record['converged'], out.nfev, flush=True)
    best = min(attempts, key=lambda a: a['rmse_decade']); c = evaluate(np.array(best['parameters']))
    return {'shared_parameters': c['shared'].tolist(),
            'states': np.column_stack((c['amplitude'], best['parameters'][offset:])).tolist(),
            'rmse_decade': best['rmse_decade'], 'converged': best['converged'], 'multistart': attempts,
            'shared_boundary_indices': np.flatnonzero(np.minimum(c['shared']-LO, HI-c['shared'])/(HI-LO) < 1e-5).tolist()}


def run():
    reference_path = ROOT/'analysis/coupled_charge_comparison.json'
    reference = json.loads(reference_path.read_text())
    for p, sha in reference['input_sha256'].items():
        assert digest(ROOT/p) == sha, 'Archived source changed: '+p
    ids, y, w, selection = assemble()
    files = [Path(__file__), ROOT/'vulcan2d/charge_spectrum.py', ROOT/'tests/test_charge_spectrum.py',
             ROOT/'analysis/CHARGE_SPECTRUM_DESIGN_2026-09-21.md', ROOT/'vulcan2d/coupled_charge_reduced.py',
             reference_path] + [ROOT/p for p in reference['input_sha256']]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in dict.fromkeys(files)}
    path = ROOT/'analysis/charge_spectrum_comparison.json'
    result = {'scope': 'Independent reciprocal charge channels with bounded uniform energy spectrum; no structural evolution law',
              'status': 'running', 'input_sha256': hashes, 'ids': ids.tolist(), 'voltage_V': VOLTAGE.tolist(),
              'observed_log10_current_A': y.tolist(), 'weights': w.tolist(), 'selection': selection,
              'quadrature': {'panels': 8, 'order': 12}, 'splits': {}}
    if path.exists():
        result = json.loads(path.read_text())
        assert result['input_sha256'] == hashes, 'Preserve existing results before source changes'
    def save(): path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for label, old in reference['splits'].items():
        train, target = np.array(old['training_ids'])-1, np.array(old['target_ids'])-1
        row = result['splits'].setdefault(label, {'training_ids': old['training_ids'], 'target_ids': old['target_ids'], 'targets': {}})
        if 'fit' not in row:
            print('Fitting', label, flush=True)
            row['fit'] = fit(VOLTAGE, y[train], w[train], zero_width_fit=old['fit']); save()
            print('Best', label, row['fit']['rmse_decade'], row['fit']['shared_parameters'], flush=True)
        found = row['fit']; pred = predict(VOLTAGE, found['shared_parameters'], found['states'])
        row['training_scores'] = scores(y[train], pred, w[train], ids[train])
        row['zero_width_training_control_rmse_decade'] = old['fit']['rmse_decade']
        if label == 'all_descriptive': row['predicted_log10_A'] = pred.tolist()
        else:
            assert not set(train) & set(target)
            for protocol, mask in [('SET_only', VOLTAGE > 0), ('joint_descriptive', np.ones(len(VOLTAGE), bool))]:
                if protocol in row['targets']: continue
                local = fit(VOLTAGE[mask], y[target][:, mask], w[target][:, mask], fixed_shared=found['shared_parameters'])
                pred = predict(VOLTAGE, found['shared_parameters'], local['states'])
                row['targets'][protocol] = {'identification': local, 'predicted_log10_A': pred.tolist(),
                                             'scores': scores(y[target], pred, w[target], ids[target])}; save()
                print(label, protocol, row['targets'][protocol]['scores'], flush=True)
        save()
    for protocol in ('SET_only', 'joint_descriptive'):
        pooled = np.empty_like(y)
        for label, row in result['splits'].items():
            if label.startswith('fold_'): pooled[np.array(row['target_ids'])-1] = row['targets'][protocol]['predicted_log10_A']
        result[protocol+'_five_fold_scores'] = scores(y, pooled, w, ids)
    result['status'] = 'complete'; save(); print(path, flush=True)


if __name__ == '__main__':
    run()
