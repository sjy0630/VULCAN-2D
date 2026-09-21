"""Jointly fit retained charge and transport without prescribed cycle events."""
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
from compare_activated_configuration import assemble, lookups, scores
from check_retained_charge_contact import ratio34
from vulcan2d.retained_charge_configuration import trajectory
from vulcan2d.retained_transport_parameters import pack, unpack, LO, HI, SCALE, NAMES


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    parent = ROOT/'analysis'/f'retained_charge_configuration_{args.split}_single_contact.json'
    old = json.loads(parent.read_text()); assert old['status'] == 'complete'
    hashes = dict(old['input_sha256'])
    for name, expected in hashes.items(): assert digest(ROOT/name) == expected, name
    for path in [parent, Path(__file__), ROOT/'vulcan2d/retained_transport_parameters.py',
                 ROOT/'analysis/RETAINED_TRANSPORT_AUDIT_DESIGN_2026-09-21.md', ROOT/'tests/test_retained_transport_parameters.py']:
        hashes[str(path.relative_to(ROOT))] = digest(path)
    d = assemble(); lookup = lookups(); train = old['training_ids']; end = int(np.flatnonzero(d['cycle'] == train[-1])[-1]+1)
    mask = d['weights'][:end] > 0; weight = np.sqrt(d['weights'][:end][mask]/len(train))
    log_observed = np.log10(d['observed'][:end][mask]); settings = old['settings']
    original = pack(old['shared_parameters'], old['parameters'])
    initials = [original.copy() for _ in range(3)]
    initials[1][6] = 0.; initials[2][6] = -6.
    output = ROOT/'analysis'/f'retained_transport_{args.split}.json'
    result = dict(status='running', scope='Joint continuous fixed-contact fit; no imposed event or per-cycle state',
                  split=args.split, training_ids=train, target_ids=old['target_ids'], settings=settings,
                  input_sha256=hashes, parameter_names=list(NAMES), bounds=[LO.tolist(), HI.tolist()],
                  parent_result=str(parent.relative_to(ROOT)), starts=[], initial_condition='q=0, zero-bias conditional-equilibrium m',
                  previous_training_rmse=old['training_scores']['rmse_decade'], maximum_nfev_per_start=180)
    if output.exists():
        result = json.loads(output.read_text()); assert result['input_sha256'] == hashes and result['settings'] == settings
        if result['status'] == 'complete': return
    def save(): output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    save()
    for index, proposed in enumerate(initials):
        if index < len(result['starts']): continue
        initial = np.maximum(LO+1e-12, np.minimum(HI-1e-12, proposed)); calls = [0]; worst = [0.]; started = time.perf_counter()
        best_seen = [np.inf, initial.copy()]
        def objective(x):
            s, p = unpack(x); r = trajectory(d['voltage'][:end], s, p, *lookup, **settings)
            worst[0] = max(worst[0], r['maximum_kcl_relative_error']); assert worst[0] < 1e-8
            predicted = r['current_A'][mask]; assert np.all(np.isfinite(predicted) & (predicted > 0))
            residual = weight*(np.log10(predicted)-log_observed); metric = float(np.linalg.norm(residual)); calls[0] += 1
            if metric < best_seen[0]: best_seen[:] = [metric, x.copy()]
            if calls[0] % 100 == 0:
                print(args.split, index, calls[0], metric, 'best', best_seen[0], flush=True)
                output.with_name(output.stem+'_live.json').write_text(json.dumps({
                    'status': 'running', 'index': index, 'actual_evaluations': calls[0], 'best_training_rmse': best_seen[0],
                    'best_coordinates': best_seen[1].tolist(), 'seconds': time.perf_counter()-started}, indent=2)+'\n')
            return residual
        print('START', args.split, index, initial.tolist(), flush=True)
        fit = least_squares(objective, initial, bounds=(LO, HI), x_scale=SCALE, diff_step=1e-5,
                            max_nfev=180, ftol=2e-7, xtol=2e-7, gtol=2e-7)
        s, p = unpack(fit.x)
        row = dict(index=index, proposed_initial=proposed.tolist(), initial=initial.tolist(), coordinates=fit.x.tolist(),
                   shared_parameters=s.tolist(), parameters=p.tolist(), rmse_decade=float(np.linalg.norm(fit.fun)),
                   converged=bool(fit.success), status=int(fit.status), message=fit.message, nfev=fit.nfev,
                   actual_evaluations=calls[0], seconds=time.perf_counter()-started, optimality=float(fit.optimality),
                   active_bound_indices=np.flatnonzero(fit.active_mask != 0).tolist(), maximum_kcl_relative_error=worst[0])
        result['starts'].append(row); save(); print('FINISH', args.split, row, flush=True)
    best = min(result['starts'], key=lambda r: r['rmse_decade'])
    s, p = unpack(best['coordinates']); r = trajectory(d['voltage'], s, p, *lookup, **settings, diagnostics=True)
    result.update(status='complete', selected_start=best['index'], selected_converged=best['converged'],
                  coordinates=best['coordinates'], shared_parameters=s.tolist(), parameters=p.tolist(),
                  training_scores=scores(d, r['current_A'], train), target_scores=scores(d, r['current_A'], old['target_ids']),
                  full_scores=scores(d, r['current_A']), maximum_kcl_relative_error=r['maximum_kcl_relative_error'],
                  multiple_root_point_count=int(np.count_nonzero(r['root_count'] > 1)),
                  SET_return_4_over_3=ratio34(d, r['current_A'], 2), RESET_outward_4_over_3=ratio34(d, r['current_A'], 4))
    np.savez_compressed(output.with_suffix('.npz'), **{k:d[k] for k in ('voltage','observed','cycle','window','weights','low_signal')},
                        current=r['current_A'], internal_voltage=r['internal_voltage_V'], transistor_voltage=r['transistor_voltage_V'],
                        states=r['states'], root_count=r['root_count'], quadrature_weights=r['quadrature_weights'])
    result['prediction_sha256'] = digest(output.with_suffix('.npz')); save()
    output.with_name(output.stem+'_live.json').write_text(json.dumps({'status':'complete','selected_start':best['index']})+'\n')
    print('COMPLETE', args.split, result['training_scores']['rmse_decade'], result['target_scores']['rmse_decade'],
          result['SET_return_4_over_3'], result['RESET_outward_4_over_3'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--split', required=True, choices=['all_descriptive','chronological'])
    run(parser.parse_args())
