"""Joint readout refit for a declared four-domain support diagnostic."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
from scipy.optimize import least_squares
from fit_stochastic_readout_support import ROOT, initializations
from compare_activated_configuration import assemble, lookups, scores
from vulcan2d.finite_readout_support import readout, fractions, nearest_configuration, LO, HI


def run(split):
    d = assemble(); lookup = lookups(); mask = d['weights'] > 0
    train_ids = list(range(1, 54 if split == 'all_descriptive' else 36)); target_ids = [] if split == 'all_descriptive' else list(range(36, 54))
    train = mask & np.isin(d['cycle'], train_ids); weight = np.sqrt(d['weights'][train]/len(train_ids)); y = d['observed'][train]
    voltage, inverse = np.unique(d['voltage'], return_inverse=True); pop = fractions(4)
    source = ROOT/'analysis'/f'stochastic_readout_support_{split}_four_reconciled.json'
    base = json.loads(source.read_text()); assert base['status'] == 'complete'
    starts = [base['parameters'], *initializations(split)[1:]]; settings = base['settings']
    paths = [source, Path(__file__), ROOT/'analysis/STOCHASTIC_DOMAIN_REFIT_DESIGN_2026-09-21.md',
             ROOT/'analysis/fit_stochastic_readout_support.py', ROOT/'vulcan2d/finite_readout_support.py',
             ROOT/'vulcan2d/_finite_readout_core.cpp', ROOT/'vulcan2d/_activated_configuration_core.cpp',
             ROOT/'1T1M写入.xlsx', ROOT/'1T1M擦除.xlsx']
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    path = ROOT/'analysis'/f'stochastic_domain_support_{split}_N4.json'
    result = {'status': 'running', 'scope': 'Joint readout fit with observed-current Oracle over 35 configurations; no path likelihood',
              'domains': 4, 'configuration_count': len(pop), 'split': split,
              'training_ids': train_ids, 'target_ids': target_ids, 'settings': settings,
              'bounds': [LO.tolist(), HI.tolist()], 'input_sha256': hashes, 'starts': []}
    if path.exists():
        result = json.loads(path.read_text()); assert result['input_sha256'] == hashes
        if result['status'] == 'complete': return
    def save(): path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for index, initial in enumerate(starts):
        if index < len(result['starts']): continue
        calls = [0]; kcl = [0.]; begin = time.perf_counter()
        def objective(p):
            r = readout(voltage, p, pop, *lookup, **settings); calls[0] += 1
            kcl[0] = max(kcl[0], r['maximum_kcl_relative_error']); assert kcl[0] < 1e-7
            _, error = nearest_configuration(y, r['current_A'][:, inverse[train]])
            if calls[0] % 100 == 0: print(split, index, calls[0], np.linalg.norm(weight*error), flush=True)
            return weight*error
        fit = least_squares(objective, np.clip(initial, LO+1e-12, HI-1e-12), bounds=(LO, HI),
                            x_scale=[1., .3, .3, .3, .3, .3], max_nfev=250,
                            ftol=2e-7, xtol=2e-7, gtol=2e-7)
        row = {'index': index, 'initial': list(initial), 'parameters': fit.x.tolist(),
               'rmse_decade': float(np.linalg.norm(fit.fun)), 'converged': bool(fit.success),
               'status': int(fit.status), 'message': fit.message, 'nfev': fit.nfev,
               'actual_evaluations': calls[0], 'seconds': time.perf_counter()-begin,
               'maximum_kcl_relative_error': kcl[0], 'optimality': float(fit.optimality),
               'active_bound_indices': np.flatnonzero(abs(fit.active_mask) > 0).tolist()}
        result['starts'].append(row); save(); print('Finished', row, flush=True)
    best = min(result['starts'], key=lambda row: row['rmse_decade'])
    r = readout(voltage, best['parameters'], pop, *lookup, **settings)
    labels = np.full(len(mask), -1); current = np.zeros(len(mask))
    labels[mask], _ = nearest_configuration(d['observed'][mask], r['current_A'][:, inverse[mask]])
    current[mask] = r['current_A'][labels[mask], inverse[mask]]
    result.update(status='complete', parameters=best['parameters'], selected_start=best['index'],
                  training_support_scores=scores(d, current, train_ids), target_support_scores=scores(d, current, target_ids),
                  full_support_scores=scores(d, current), maximum_kcl_relative_error=r['maximum_kcl_relative_error'])
    np.savez_compressed(path.with_suffix('.npz'), current=current, oracle_index=labels, fractions=pop,
                        voltage=d['voltage'], observed=d['observed'], cycle=d['cycle'], weights=d['weights'], window=d['window'], low_signal=d['low_signal'],
                        unique_voltage=voltage, voltage_inverse=inverse, configuration_current=r['current_A'],
                        configuration_internal_voltage=r['internal_voltage_V'], configuration_transistor_voltage=r['transistor_voltage_V'])
    result['readout_sha256'] = hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest(); save()
    print('COMPLETE', split, result['training_support_scores']['rmse_decade'], result['target_support_scores']['rmse_decade'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--split', required=True, choices=['all_descriptive', 'chronological'])
    run(parser.parse_args().split)
