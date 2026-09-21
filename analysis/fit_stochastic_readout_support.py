"""Optimistic instantaneous state support; never labeled a path prediction."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import sys
import numpy as np
from scipy.optimize import least_squares
from scipy.special import log_expit
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'analysis'))
from compare_activated_configuration import assemble, lookups, scores
from vulcan2d.finite_readout_support import readout, nearest_configuration, LO, HI
from vulcan2d.coupled_charge import charging


def inherited(split):
    r = json.loads((ROOT/'analysis'/f'retained_charge_configuration_{split}_single_contact.json').read_text())
    s, p = r['shared_parameters'], r['parameters']
    return np.array([p[0]+log_expit(s[4])/np.log(10.), s[0], s[5], s[1], p[1], charging(np.r_[s[:5], 0.])[0]])


def initializations(split):
    return [inherited(split), [-4.4, .45, .4, .3, .12, .12], [-4.4, .65, .6, .7, .2, .05],
            [-4.4, .12, .05, .5, .15, .3], [-4.4, .9, .8, .5, .07, .5]]


def run(args):
    d = assemble(); lookup = lookups(); mask = d['weights'] > 0
    train_ids = list(range(1, 54 if args.split == 'all_descriptive' else 36))
    target_ids = [] if args.split == 'all_descriptive' else list(range(36, 54))
    train = mask & np.isin(d['cycle'], train_ids)
    voltage, inverse = np.unique(d['voltage'], return_inverse=True)
    observed = d['observed'][train]; weight = np.sqrt(d['weights'][train]/len(train_ids))
    settings = {'panels': 12, 'order': 8, 'table_step': .005}
    active = np.arange(5 if args.states == 'two' else 6)
    name = f'stochastic_readout_support_{args.split}_{args.states}'
    path = ROOT/'analysis'/f'{name}{"_reconciled" if args.nested else ""}.json'
    sources = ['analysis/fit_stochastic_readout_support.py', 'analysis/STOCHASTIC_READOUT_SUPPORT_DESIGN_2026-09-21.md',
               'analysis/compare_activated_configuration.py', 'analysis/diagnose_cross_polarity_physics.py',
               'vulcan2d/finite_readout_support.py', 'vulcan2d/_finite_readout_core.cpp', 'vulcan2d/_activated_configuration_core.cpp',
               'vulcan2d/charge_spectrum.py', 'vulcan2d/coupled_charge.py', 'vulcan2d/coupled_charge_reduced.py',
               'vulcan2d/transistor.py', 'vulcan2d/data/fig2b_transistor_lookup.json', 'vulcan2d/data/bookoutput2_reverse_transistor_lookup.json',
               f'analysis/retained_charge_configuration_{args.split}_single_contact.json', '1T1M写入.xlsx', '1T1M擦除.xlsx']
    starts = initializations(args.split); old_starts = []
    if args.nested:
        own = f'analysis/{name}.json'
        other = f'analysis/stochastic_readout_support_{args.split}_{"four" if args.states == "two" else "two"}.json'
        base, sibling = [json.loads((ROOT/file).read_text()) for file in (own, other)]
        assert base['status'] == sibling['status'] == 'complete'
        assert base['training_ids'] == sibling['training_ids'] == train_ids
        starts = [sibling['parameters']]; old_starts = base['starts']; sources += [own, other]
    hashes = {source: hashlib.sha256((ROOT/source).read_bytes()).hexdigest() for source in sources}
    result = {'status': 'running', 'scope': 'Best instantaneous state chosen using each observed current; optimistic support floor only',
              'split': args.split, 'states': args.states, 'training_ids': train_ids, 'target_ids': target_ids,
              'settings': settings, 'bounds': [LO.tolist(), HI.tolist()], 'input_sha256': hashes, 'starts': old_starts.copy()}
    if path.exists():
        result = json.loads(path.read_text()); assert result['input_sha256'] == hashes
        if result['status'] == 'complete': return
    def save(): path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    def expand(x):
        p = np.zeros(6); p[active] = x; return p
    for number, start in enumerate(starts):
        index = len(old_starts)+number
        if index < len(result['starts']): continue
        calls = [0]; worst_kcl = [0.]; begin = time.perf_counter()
        initial = np.clip(start, LO+1e-12, HI-1e-12)
        def objective(x):
            table = readout(voltage, expand(x), np.eye(4), *lookup, **settings)
            worst_kcl[0] = max(worst_kcl[0], table['maximum_kcl_relative_error']); assert worst_kcl[0] < 1e-7
            selected, error = nearest_configuration(observed, table['current_A'][:, inverse[train]])
            calls[0] += 1
            if calls[0] % 150 == 0: print(args.split, args.states, index, calls[0], np.linalg.norm(weight*error), flush=True)
            return weight*error
        fit = least_squares(objective, initial[active], bounds=(LO[active], HI[active]),
                            x_scale=np.array([1., .3, .3, .3, .3, .3])[active],
                            max_nfev=120 if args.nested else 250, ftol=2e-7, xtol=2e-7, gtol=2e-7)
        row = {'index': index, 'initial': expand(initial[active]).tolist(), 'parameters': expand(fit.x).tolist(),
               'rmse_decade': float(np.linalg.norm(fit.fun)), 'converged': bool(fit.success),
               'status': int(fit.status), 'message': fit.message, 'nfev': fit.nfev, 'actual_evaluations': calls[0],
               'seconds': time.perf_counter()-begin, 'maximum_kcl_relative_error': worst_kcl[0],
               'active_bound_indices': active[abs(fit.active_mask) > 0].tolist(), 'optimality': float(fit.optimality)}
        result['starts'].append(row); save(); print('Finished', row, flush=True)
    best = min(result['starts'], key=lambda row: row['rmse_decade']); p = best['parameters']
    table = readout(voltage, p, np.eye(4), *lookup, **settings)
    currents = table['current_A'][:, inverse]
    labels = np.full(len(mask), -1); current = np.zeros(len(mask))
    labels[mask], error = nearest_configuration(d['observed'][mask], currents[:, mask])
    current[mask] = currents[labels[mask], np.flatnonzero(mask)]
    result.update(status='complete', parameters=p, selected_start=best['index'],
                  training_support_scores=scores(d, current, train_ids), target_support_scores=scores(d, current, target_ids),
                  full_support_scores=scores(d, current), maximum_kcl_relative_error=table['maximum_kcl_relative_error'])
    np.savez_compressed(path.with_suffix('.npz'), voltage=d['voltage'], observed=d['observed'], cycle=d['cycle'],
                        weights=d['weights'], window=d['window'], low_signal=d['low_signal'],
                        current=current, oracle_state=labels, conditional_current=currents,
                        unique_voltage=voltage, voltage_inverse=inverse,
                        conditional_internal_voltage=table['internal_voltage_V'], conditional_transistor_voltage=table['transistor_voltage_V'])
    result['readout_sha256'] = hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest(); save()
    print('COMPLETE', path.name, result['training_support_scores']['rmse_decade'], result['target_support_scores']['rmse_decade'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--split', required=True, choices=['all_descriptive', 'chronological'])
    parser.add_argument('--states', required=True, choices=['two', 'four']); parser.add_argument('--nested', action='store_true')
    run(parser.parse_args())
