"""Matched-training cross starts for nested saddle candidates.

Avoid mistaking a local optimization basin for evidence for the extra saddle
parameter. Original fixed-start results remain immutable separate artifacts.
No all-data parameters initialize the chronological fit.
"""
from pathlib import Path
import argparse
import json
import hashlib
import time
import numpy as np
from scipy.optimize import least_squares
from compare_activated_configuration import ROOT, assemble, lookups, scores
from vulcan2d.activated_configuration import trajectory, LO, HI


def run(split):
    d = assemble(); lookup = lookups(); original = {}
    for saddle in ('single', 'polarizable'):
        path = ROOT/'analysis'/f'activated_configuration_{split}_{saddle}.json'
        original[saddle] = (path, json.loads(path.read_text()))
        assert original[saddle][1]['status'] == 'complete'
    np.testing.assert_array_equal(original['single'][1]['shared_parameters'], original['polarizable'][1]['shared_parameters'])
    np.testing.assert_array_equal(original['single'][1]['training_ids'], original['polarizable'][1]['training_ids'])
    for saddle, other in [('single', 'polarizable'), ('polarizable', 'single')]:
        path, old = original[saddle]; result = json.loads(json.dumps(old))
        result['status'] = 'running'; output = path.with_name(path.stem+'_reconciled.json')
        if output.exists():
            existing = json.loads(output.read_text())
            if existing['status'] == 'complete': continue
        result['input_sha256'].update({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in (Path(__file__), original['single'][0], original['polarizable'][0])})
        train = old['training_ids']; end = int(np.flatnonzero(d['cycle'] == train[-1])[-1]+1)
        mask = d['weights'][:end] > 0; weights = np.sqrt(d['weights'][:end][mask]/len(train))
        y = np.log10(d['observed'][:end][mask]); settings = old['settings']; shared = old['shared_parameters']
        active = np.arange(7 if saddle == 'single' else 8)
        initial = np.asarray(original[other][1]['parameters'])
        def expand(x):
            p = np.zeros(8); p[active] = x; return p
        count = [0]; maximum_kcl = [0.]; started = time.perf_counter()
        def objective(x):
            r = trajectory(d['voltage'][:end], shared, expand(x), *lookup, **settings)
            count[0] += 1; maximum_kcl[0] = max(maximum_kcl[0], r['maximum_kcl_relative_error'])
            assert maximum_kcl[0] < 1e-6
            residual = weights*(np.log10(r['current_A'][mask])-y)
            if count[0] % 100 == 0: print(split, saddle, count[0], np.linalg.norm(residual), flush=True)
            return residual
        print('Matched-split nested start', split, saddle, flush=True)
        fitted = least_squares(objective, initial[active], bounds=(LO[active], HI[active]),
                               x_scale=np.array([1., .2, .2, .2, 10., .2, .1, .01])[active],
                               diff_step=1e-5, max_nfev=120, ftol=2e-7, xtol=2e-7, gtol=2e-7)
        row = {'index': len(old['starts']), 'initial': expand(initial[active]).tolist(),
               'initialization_source': str(original[other][0].relative_to(ROOT)),
               'parameters': expand(fitted.x).tolist(), 'rmse_decade': float(np.linalg.norm(fitted.fun)),
               'converged': bool(fitted.success), 'status': int(fitted.status), 'message': fitted.message,
               'nfev': fitted.nfev, 'actual_evaluations': count[0], 'seconds': time.perf_counter()-started,
               'maximum_kcl_relative_error': maximum_kcl[0], 'optimality': float(fitted.optimality),
               'active_bound_indices': active[np.abs(fitted.active_mask) > 0].tolist()}
        result['starts'].append(row); best = min(result['starts'], key=lambda r: r['rmse_decade'])
        result.update(selected_start=best['index'], parameters=best['parameters'],
                      reconciliation='One additional start from the other saddle candidate of the exact same training split')
        r = trajectory(d['voltage'], shared, best['parameters'], *lookup, **settings, diagnostics=True)
        result.update(training_scores=scores(d, r['current_A'], train), full_scores=scores(d, r['current_A']),
                      target_scores=scores(d, r['current_A'], old['target_ids']),
                      maximum_kcl_relative_error=r['maximum_kcl_relative_error'],
                      multiple_root_point_count=int(np.count_nonzero(r['root_count'] > 1)), status='complete')
        np.savez_compressed(output.with_suffix('.npz'), voltage=d['voltage'], observed=d['observed'], cycle=d['cycle'],
                            window=d['window'], weights=d['weights'], low_signal=d['low_signal'],
                            current=r['current_A'], internal_voltage=r['internal_voltage_V'],
                            states=r['states'], root_count=r['root_count'], quadrature_weights=r['quadrature_weights'])
        result['prediction_sha256'] = hashlib.sha256(output.with_suffix('.npz').read_bytes()).hexdigest()
        output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print('COMPLETE', output.name, result['training_scores']['rmse_decade'], result['target_scores']['rmse_decade'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--split', required=True, choices=['all_descriptive', 'chronological'])
    run(parser.parse_args().split)
