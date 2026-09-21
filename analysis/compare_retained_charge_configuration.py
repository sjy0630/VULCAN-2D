"""Same full native protocol, with retained joint charge/configuration states."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import sys
import numpy as np
from scipy.optimize import least_squares
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'analysis'))
from compare_activated_configuration import assemble, lookups, scores, baseline
from vulcan2d.retained_charge_configuration import trajectory, LO, HI


def starts(split):
    old = json.loads((ROOT/f'analysis/activated_configuration_{split}_single_reconciled.json').read_text())
    return [old['parameters']+[g] for g in (-3., 0., 6.)]+[
        [-2.1, .25, -.1, .3, -8., .1, .1, 0., -2.],
        [-2.1, .25, .3, -.1, -10., .1, .05, .004, 0.]]


def run(args):
    d = assemble(); lookup = lookups(); shared = baseline(args.split)
    train = np.arange(1, 54) if args.split == 'all_descriptive' else np.arange(1, 36)
    target = [] if args.split == 'all_descriptive' else list(range(36, 54))
    end = np.flatnonzero(d['cycle'] == train[-1])[-1]+1
    mask = d['weights'][:end] > 0; weight = np.sqrt(d['weights'][:end][mask]/len(train))
    observed = np.log10(d['observed'][:end][mask])
    active = np.array([0, 1, 2, 3, 4, 5, 6, 8] if args.saddle == 'single' else range(9))
    settings = dict(panels=args.panels, order=8, table_step=args.table_step, max_step=args.max_step)
    inputs = ['analysis/compare_retained_charge_configuration.py', 'analysis/compare_activated_configuration.py',
              'analysis/diagnose_cross_polarity_physics.py', 'vulcan2d/retained_charge_configuration.py',
              'vulcan2d/_retained_charge_configuration_core.cpp', 'vulcan2d/_activated_configuration_core.cpp',
              'vulcan2d/activated_configuration.py', 'vulcan2d/charge_spectrum.py',
              'vulcan2d/coupled_charge.py', 'vulcan2d/coupled_charge_reduced.py', 'vulcan2d/transistor.py',
              'vulcan2d/data/fig2b_transistor_lookup.json', 'vulcan2d/data/bookoutput2_reverse_transistor_lookup.json',
              'analysis/RETAINED_CHARGE_CONFIGURATION_DESIGN_2026-09-21.md',
              'analysis/RETAINED_CHARGE_IMPLEMENTATION_DESIGN_2026-09-21.md',
              'analysis/charge_spectrum_expanded_prior.json', 'analysis/charge_spectrum_expanded_groups.json',
              f'analysis/activated_configuration_{args.split}_single_reconciled.json',
              '1T1M写入.xlsx', '1T1M擦除.xlsx']
    hashes = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in inputs}
    path = ROOT/'analysis'/args.output
    result = {'status': 'running', 'scope': 'Conductor adiabatic, joint side-charge and configuration retained continuously',
              'split': args.split, 'saddle': args.saddle, 'training_ids': train.tolist(), 'target_ids': target,
              'shared_parameters': shared, 'settings': settings, 'input_sha256': hashes,
              'bounds': [LO.tolist(), HI.tolist()], 'initialization': 'q=0; m in q=0 zero-bias thermal marginal',
              'score_rule': 'Archived eight equal-mass native windows; all scans propagate continuously',
              'starts': []}
    if path.exists():
        result = json.loads(path.read_text())
        assert result['input_sha256'] == hashes and result['settings'] == settings
        assert result['split'] == args.split and result['saddle'] == args.saddle
    def save(): path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    def expand(x):
        p = np.zeros(9); p[active] = x; return p
    for index, initial in enumerate(starts(args.split)[:args.starts]):
        if index < len(result['starts']): continue
        initial = np.asarray(initial); calls = [0]; kcl = [0.]; started = time.perf_counter()
        def objective(x):
            r = trajectory(d['voltage'][:end], shared, expand(x), *lookup, **settings)
            kcl[0] = max(kcl[0], r['maximum_kcl_relative_error']); calls[0] += 1
            if kcl[0] > 1e-6: raise RuntimeError('Excessive load residual')
            current = r['current_A'][mask]
            if np.any(current <= 0) or np.any(~np.isfinite(current)): raise RuntimeError('Invalid current prediction')
            residual = weight*(np.log10(current)-observed)
            if calls[0] % 100 == 0: print(args.split, args.saddle, index, calls[0], np.linalg.norm(residual), flush=True)
            return residual
        print('Start', args.split, args.saddle, index, expand(initial[active]).tolist(), flush=True)
        fitted = least_squares(objective, initial[active], bounds=(LO[active], HI[active]),
                               x_scale=np.array([1., .2, .2, .2, 10., .2, .1, .01, 8.])[active],
                               max_nfev=args.max_nfev, ftol=2e-7, xtol=2e-7, gtol=2e-7)
        row = {'index': index, 'initial': expand(initial[active]).tolist(), 'parameters': expand(fitted.x).tolist(),
               'rmse_decade': float(np.linalg.norm(fitted.fun)), 'converged': bool(fitted.success),
               'status': int(fitted.status), 'message': fitted.message, 'nfev': fitted.nfev,
               'actual_evaluations': calls[0], 'seconds': time.perf_counter()-started,
               'maximum_kcl_relative_error': kcl[0], 'optimality': float(fitted.optimality),
               'active_bound_indices': active[np.abs(fitted.active_mask) > 0].tolist()}
        result['starts'].append(row); save(); print('Finished', row, flush=True)
    best = min(result['starts'], key=lambda row: row['rmse_decade'])
    r = trajectory(d['voltage'], shared, best['parameters'], *lookup, **settings, diagnostics=True)
    result.update(status='complete', parameters=best['parameters'], selected_start=best['index'],
                  training_scores=scores(d, r['current_A'], train), target_scores=scores(d, r['current_A'], target),
                  full_scores=scores(d, r['current_A']), maximum_kcl_relative_error=r['maximum_kcl_relative_error'],
                  multiple_root_point_count=int(np.count_nonzero(r['root_count'] > 1)),
                  maximum_relative_duration_rounding=r['maximum_relative_duration_rounding'],
                  distinct_substep_durations_V=r['distinct_substep_durations_V'].tolist())
    np.savez_compressed(path.with_suffix('.npz'), voltage=d['voltage'], observed=d['observed'], cycle=d['cycle'],
                        window=d['window'], weights=d['weights'], low_signal=d['low_signal'], current=r['current_A'],
                        internal_voltage=r['internal_voltage_V'], transistor_voltage=r['transistor_voltage_V'],
                        states=r['states'], root_count=r['root_count'], quadrature_weights=r['quadrature_weights'])
    result['prediction_sha256'] = hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest()
    save(); print('COMPLETE', path.name, result['training_scores'], result['target_scores'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['all_descriptive', 'chronological'], default='all_descriptive')
    parser.add_argument('--saddle', choices=['single', 'polarizable'], default='single')
    parser.add_argument('--starts', type=int, default=5); parser.add_argument('--max-nfev', type=int, default=180)
    parser.add_argument('--panels', type=int, default=8); parser.add_argument('--table-step', type=float, default=.005)
    parser.add_argument('--max-step', type=float, default=.01); parser.add_argument('--output', required=True)
    run(parser.parse_args())
