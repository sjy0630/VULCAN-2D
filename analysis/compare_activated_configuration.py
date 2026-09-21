"""Continuous native-voltage model test; target current never drives state.

This is an exploratory extension after earlier candidate failures. Chronology
is a held-out parameter-estimation split within already inspected data, not a
new independently collected validation set.
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
from diagnose_cross_polarity_physics import read_pairs, GateLookup, load_default_lookup, load_reverse_lookup
from vulcan2d.activated_configuration import trajectory, LO, HI

# Fixed starts; no target-state inversion or target-fit initialization.
STARTS = [
    [-2.08, .25, .15, -.12, -6., .12, .08, .004],
    [-2.12, .12, -.15, .08, -5., .08, .10, .003],
    [-2.08, .40, .12, -.25, -12., .15, .13, .006],
    [-2.08, .15, .02, .09, -8., .06, .04, .002],
]
SOURCES = ['vulcan2d/activated_configuration.py', 'vulcan2d/_activated_configuration_core.cpp',
           'analysis/compare_activated_configuration.py', 'analysis/diagnose_cross_polarity_physics.py',
           'analysis/charge_spectrum_expanded_prior.json', 'analysis/charge_spectrum_expanded_groups.json',
           'analysis/ACTIVATED_CONFIGURATION_DESIGN_2026-09-21.md',
           '1T1M写入.xlsx', '1T1M擦除.xlsx']


def assemble():
    pieces = {k: [] for k in ('voltage', 'observed', 'cycle', 'window', 'low_signal', 'raw_index')}
    labels = [f'{phase}_{direction}_{band}' for phase in ('set', 'reset')
              for direction in ('outward', 'return') for band in ('low', 'high')]
    for pair in read_pairs('1.1'):
        for phase_id, phase in enumerate(('set', 'reset')):
            raw = pair[phase]; v = np.asarray(raw['voltage']); y = np.abs(raw['current'])
            x = np.abs(v); index = np.arange(len(v)); valid = np.isfinite(y) & (y > 0)
            peak = np.max(x); apex = np.flatnonzero(np.isclose(x, peak, atol=1e-10, rtol=0))
            direction = np.where(index < apex[0], 0, 1)
            valid &= (x >= .08-1e-10) & ~np.isin(index, apex)
            window = np.where(valid, 4*phase_id+2*direction+(x > .30+1e-10), -1)
            zeros = y[(x == 0) & np.isfinite(y)]
            zero_scale = float(np.max(zeros)) if len(zeros) else 0.
            for key, data in [('voltage', v), ('observed', y), ('cycle', np.full(len(v), pair['cycle_id'])),
                              ('window', window), ('low_signal', y <= 3*zero_scale), ('raw_index', index)]:
                pieces[key].append(data)
    d = {key: np.concatenate(parts) for key, parts in pieces.items()}
    d['weights'] = np.zeros(len(d['voltage']))
    for cycle in np.unique(d['cycle']):
        for window in range(8):
            mask = (d['cycle'] == cycle) & (d['window'] == window)
            if not mask.any(): raise ValueError(f'Empty scoring window {cycle}/{window}')
            d['weights'][mask] = 1/(8*np.count_nonzero(mask))
    d['window_labels'] = labels
    return d


def lookups():
    tables = [GateLookup(f(), 1.1) for f in (load_default_lookup, load_reverse_lookup)]
    return tuple((t.voltage, t.current) for t in tables)


def scores(d, prediction, ids=None):
    mask = d['weights'] > 0
    if ids is not None: mask &= np.isin(d['cycle'], ids)
    error = np.zeros(len(mask)); error[mask] = np.log10(prediction[mask]/d['observed'][mask])
    def metric(select):
        m = mask & select; w = d['weights'][m]
        return float(np.sqrt(np.sum(w*error[m]**2)/w.sum())) if len(w) else None
    return {'rmse_decade': metric(np.ones(len(mask), bool)), 'point_count': int(mask.sum()),
            'first3_rmse': metric(d['cycle'] <= 3), 'later50_rmse': metric(d['cycle'] > 3),
            'low_signal_count': int(np.count_nonzero(mask & d['low_signal'])),
            'windows': {label: metric(d['window'] == i) for i, label in enumerate(d['window_labels'])},
            'per_cycle': {str(c): metric(d['cycle'] == c) for c in np.unique(d['cycle'][mask])}}


def baseline(split):
    if split == 'all_descriptive':
        return json.loads((ROOT/'analysis/charge_spectrum_expanded_prior.json').read_text())['fit']['shared_parameters']
    return json.loads((ROOT/'analysis/charge_spectrum_expanded_groups.json').read_text())['splits'][split]['fit']['shared_parameters']


def run(args):
    d = assemble(); lookup = lookups(); shared = baseline(args.split)
    train = np.arange(1, 54) if args.split == 'all_descriptive' else np.arange(1, 36)
    target = [] if args.split == 'all_descriptive' else list(range(36, 54))
    end = len(d['voltage']) if args.split == 'all_descriptive' else int(np.flatnonzero(d['cycle'] == 35)[-1]+1)
    mask = (d['weights'][:end] > 0) & np.isin(d['cycle'][:end], train)
    w = np.sqrt(d['weights'][:end][mask]/len(train)); y = np.log10(d['observed'][:end][mask])
    active = np.arange(8 if args.saddle == 'polarizable' else 7)
    settings = dict(panels=args.panels, order=8, table_step=args.table_step, max_step=args.max_step)
    hashes = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES}
    result = {'status': 'running', 'scope': 'Full native continuous fixed-DC protocol; exploratory candidate',
              'split': args.split, 'saddle': args.saddle, 'training_ids': train.tolist(), 'target_ids': target,
              'shared_parameters': shared, 'input_sha256': hashes, 'settings': settings,
              'bounds': [LO.tolist(), HI.tolist()], 'starts': [], 'initial_q1': 0.,
              'low_signal_rule': 'magnitude <= 3 times maximum zero-voltage magnitude within that scan',
              'score_rule': 'Eight equal-mass windows per pair; apex and |Vapp|<0.08 unscored; full sequence propagated'}
    path = ROOT/'analysis'/args.output
    if path.exists():
        result = json.loads(path.read_text())
        assert result['input_sha256'] == hashes and result['settings'] == settings
        assert result['split'] == args.split and result['saddle'] == args.saddle
    def save(): path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    def expand(x):
        p = np.zeros(8); p[active] = x; return p
    for index, start in enumerate(STARTS[:args.starts]):
        if index < len(result['starts']): continue
        counter = [0]; worst_kcl = [0.]; start_time = time.perf_counter()
        def objective(x):
            r = trajectory(d['voltage'][:end], shared, expand(x), *lookup, **settings)
            worst_kcl[0] = max(worst_kcl[0], r['maximum_kcl_relative_error'])
            if r['maximum_kcl_relative_error'] > 1e-6: raise RuntimeError('Load residual exceeds tolerance')
            current = r['current_A'][mask]
            if np.any(current <= 0) or np.any(~np.isfinite(current)): raise RuntimeError('Invalid prediction')
            residual = w*(np.log10(current)-y); counter[0] += 1
            if counter[0] % 100 == 0:
                print(args.split, args.saddle, index, counter[0], 'rmse', np.linalg.norm(residual), flush=True)
            return residual
        print('Start', args.split, args.saddle, index, flush=True)
        fit = least_squares(objective, np.asarray(start)[active], bounds=(LO[active], HI[active]),
                            x_scale=np.array([1., .2, .2, .2, 10., .2, .1, .01])[active],
                            diff_step=1e-5, max_nfev=args.max_nfev, ftol=2e-7, xtol=2e-7, gtol=2e-7)
        row = {'index': index, 'initial': expand(np.asarray(start)[active]).tolist(), 'parameters': expand(fit.x).tolist(),
               'rmse_decade': float(np.linalg.norm(fit.fun)), 'converged': bool(fit.success),
               'status': int(fit.status), 'message': fit.message, 'nfev': fit.nfev,
               'actual_evaluations': counter[0], 'seconds': time.perf_counter()-start_time,
               'maximum_kcl_relative_error': worst_kcl[0], 'optimality': float(fit.optimality),
               'active_bound_indices': active[np.abs(fit.active_mask) > 0].tolist()}
        result['starts'].append(row); save(); print('Finished', row, flush=True)
    best = min(result['starts'], key=lambda x: x['rmse_decade']); p = best['parameters']
    r = trajectory(d['voltage'], shared, p, *lookup, **settings, diagnostics=True)
    result.update(selected_start=best['index'], parameters=p, training_scores=scores(d, r['current_A'], train),
                  full_scores=scores(d, r['current_A']), target_scores=scores(d, r['current_A'], target),
                  maximum_kcl_relative_error=r['maximum_kcl_relative_error'],
                  multiple_root_point_count=int(np.count_nonzero(r['root_count'] > 1)),
                  status='complete')
    prediction_path = path.with_suffix('.npz')
    np.savez_compressed(prediction_path, voltage=d['voltage'], observed=d['observed'], cycle=d['cycle'],
                        window=d['window'], weights=d['weights'], low_signal=d['low_signal'],
                        current=r['current_A'], internal_voltage=r['internal_voltage_V'],
                        states=r['states'], root_count=r['root_count'], quadrature_weights=r['quadrature_weights'])
    result['prediction_sha256'] = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    save(); print('COMPLETE', path, result['training_scores'], result['target_scores'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['all_descriptive', 'chronological'], default='all_descriptive')
    parser.add_argument('--saddle', choices=['single', 'polarizable'], default='single')
    parser.add_argument('--starts', type=int, default=4); parser.add_argument('--max-nfev', type=int, default=250)
    parser.add_argument('--panels', type=int, default=8); parser.add_argument('--table-step', type=float, default=.005)
    parser.add_argument('--max-step', type=float, default=.01); parser.add_argument('--output', required=True)
    run(parser.parse_args())
