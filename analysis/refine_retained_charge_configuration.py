"""Exploratory matched-training nested, boundary and contact sensitivities."""
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
from vulcan2d.retained_charge_configuration import trajectory, LO, HI


def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(args):
    stem = f'retained_charge_configuration_{args.split}'
    def read(suffix):
        path = ROOT/'analysis'/f'{stem}_{suffix}.json'
        record = json.loads(path.read_text()); assert record['status'] == 'complete'
        for name, sha in record['input_sha256'].items(): assert digest(ROOT/name) == sha, name
        return path, record
    if args.mode == 'nested':
        source, old = read(args.saddle)
        other_source, other = read('polarizable' if args.saddle == 'single' else 'single')
        assert old['training_ids'] == other['training_ids']
        assert old['shared_parameters'] == other['shared_parameters']
        initials = [other['parameters']]
        output = source.with_name(source.stem+'_reconciled.json')
        extra_sources = [other_source]
    elif args.mode == 'wide':
        assert args.saddle == 'single'
        source, old = read('single_reconciled'); initials = [old['parameters']]
        output = ROOT/'analysis'/f'{stem}_single_wide.json'; extra_sources = []
    else:
        assert args.saddle == 'single'
        source, old = read('single_wide'); base = np.array(old['parameters'])
        original_gl = old['shared_parameters'][4]
        initials = [np.r_[base, original_gl]]
        for gl in (0., abs(original_gl)):
            p = base.copy(); p[0] += (log_expit(original_gl)-log_expit(gl))/np.log(10.)
            initials.append(np.r_[p, gl])
        output = ROOT/'analysis'/f'{stem}_single_contact.json'; extra_sources = []
    contact = args.mode == 'contact'
    low, high = LO.copy(), HI.copy()
    if args.mode != 'nested':
        high[1] = 1.2; low[2:4] = -1.2; high[2:4] = 1.2
    if contact: low = np.r_[low, -12.]; high = np.r_[high, 12.]
    active = np.array([0, 1, 2, 3, 4, 5, 6, 8] if args.saddle == 'single' else range(9))
    if contact: active = np.r_[active, 9]
    d = assemble(); lookup = lookups(); train = old['training_ids']
    end = np.flatnonzero(d['cycle'] == train[-1])[-1]+1
    mask = d['weights'][:end] > 0; weight = np.sqrt(d['weights'][:end][mask]/len(train))
    observed = np.log10(d['observed'][:end][mask]); settings = old['settings']
    hashes = dict(old['input_sha256'])
    for p in [source, *extra_sources, Path(__file__), ROOT/'analysis/RETAINED_CHARGE_SENSITIVITY_DESIGN_2026-09-21.md']:
        hashes[str(p.relative_to(ROOT))] = digest(p)
    result = json.loads(json.dumps(old))
    result.update(status='running', sensitivity=args.mode, input_sha256=hashes,
                  parent_result=str(source.relative_to(ROOT)),
                  bounds=[low[:9].tolist(), high[:9].tolist()],
                  parameter_scope='9 physical parameters plus contact ratio' if contact else '9 physical parameters',
                  starts=list(old['starts']) if args.mode == 'nested' else [])
    if contact: result['contact_ratio_bounds'] = [-12., 12.]
    if output.exists():
        result = json.loads(output.read_text()); assert result['input_sha256'] == hashes
        if result['status'] == 'complete': return
    offset = len(old['starts']) if args.mode == 'nested' else 0
    def save(): output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    def expand(x):
        p = np.zeros(len(low)); p[active] = x
        shared = np.array(old['shared_parameters'])
        if contact: shared[4] = p[9]
        return p[:9], shared
    for number, proposed in enumerate(initials):
        index = offset+number
        if index < len(result['starts']): continue
        initial = np.clip(proposed, low+1e-12, high-1e-12)
        calls = [0]; kcl = [0.]; started = time.perf_counter()
        def objective(x):
            p, shared = expand(x)
            r = trajectory(d['voltage'][:end], shared, p, *lookup, **settings)
            kcl[0] = max(kcl[0], r['maximum_kcl_relative_error']); calls[0] += 1
            assert kcl[0] < 1e-6
            predicted = r['current_A'][mask]
            assert np.all((predicted > 0) & np.isfinite(predicted))
            residual = weight*(np.log10(predicted)-observed)
            if calls[0] % 100 == 0: print(args.split, args.mode, args.saddle, index, calls[0], np.linalg.norm(residual), flush=True)
            return residual
        scale = np.array([1., .2, .2, .2, 10., .2, .1, .01, 8.]+([6.] if contact else []))
        fitted = least_squares(objective, initial[active], bounds=(low[active], high[active]),
                               x_scale=scale[active], max_nfev=120 if args.mode == 'nested' else 180,
                               ftol=2e-7, xtol=2e-7, gtol=2e-7)
        p, shared = expand(fitted.x)
        row = dict(index=index, proposed_initial=list(proposed), initial=expand(initial[active])[0].tolist(),
                   initial_shared_parameters=expand(initial[active])[1].tolist(),
                   parameters=p.tolist(), shared_parameters=shared.tolist(), rmse_decade=float(np.linalg.norm(fitted.fun)),
                   converged=bool(fitted.success), status=int(fitted.status), message=fitted.message,
                   nfev=fitted.nfev, actual_evaluations=calls[0], seconds=time.perf_counter()-started,
                   maximum_kcl_relative_error=kcl[0], optimality=float(fitted.optimality),
                   active_bound_indices=active[abs(fitted.active_mask) > 0].tolist())
        result['starts'].append(row); save(); print('Finished', row, flush=True)
    best = min(result['starts'], key=lambda row: row['rmse_decade'])
    shared = best.get('shared_parameters', old['shared_parameters']); p = best['parameters']
    r = trajectory(d['voltage'], shared, p, *lookup, **settings, diagnostics=True)
    result.update(status='complete', parameters=p, shared_parameters=shared, selected_start=best['index'],
                  training_scores=scores(d, r['current_A'], train), target_scores=scores(d, r['current_A'], old['target_ids']),
                  full_scores=scores(d, r['current_A']), maximum_kcl_relative_error=r['maximum_kcl_relative_error'],
                  multiple_root_point_count=int(np.count_nonzero(r['root_count'] > 1)),
                  maximum_relative_duration_rounding=r['maximum_relative_duration_rounding'],
                  distinct_substep_durations_V=r['distinct_substep_durations_V'].tolist())
    np.savez_compressed(output.with_suffix('.npz'), voltage=d['voltage'], observed=d['observed'], cycle=d['cycle'],
                        window=d['window'], weights=d['weights'], low_signal=d['low_signal'], current=r['current_A'],
                        internal_voltage=r['internal_voltage_V'], transistor_voltage=r['transistor_voltage_V'],
                        states=r['states'], root_count=r['root_count'], quadrature_weights=r['quadrature_weights'])
    result['prediction_sha256'] = digest(output.with_suffix('.npz')); save()
    print('COMPLETE', output.name, result['training_scores']['rmse_decade'], result['target_scores']['rmse_decade'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', required=True, choices=['all_descriptive', 'chronological'])
    parser.add_argument('--saddle', default='single', choices=['single', 'polarizable'])
    parser.add_argument('--mode', required=True, choices=['nested', 'wide', 'contact'])
    run(parser.parse_args())
