"""Joint amplitude/curvature test of a finite-temperature barrier state.

All curves are retained. Held-out target low-voltage points identify one local
state, then higher voltages test the constitutive law through a forward circuit.
This is voltage extrapolation, not chronological or autonomous cycle prediction.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'analysis'))
from diagnose_cross_polarity_physics import read_pairs, GateLookup
from vulcan2d.transistor import load_reverse_lookup
from vulcan2d.barrier_state import BarrierTable, forward_circuit, effective_internal

HEIGHTS = (.10, .20, .35, .50, .75, 1., 1.5, 2., 3., 4.)
FOLDS = ((1, 21), (22, 41), (42, 61), (62, 81), (82, 101))
TABLES = {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table_at(height):
    if height not in TABLES:
        TABLES[height] = BarrierTable(height)
    return TABLES[height]


def assemble():
    rows, currents, zeros = [], [], []
    pairs = read_pairs('1.3'); voltage = np.round(np.arange(.08, .30001, .01), 10)
    for pair in pairs:
        va = np.abs(pair['reset']['voltage']); obs = np.abs(pair['reset']['current'])
        apex = int(np.argmax(va)); zero = float(obs[va <= 1e-12].max())
        for phase in ('outward', 'return'):
            allowed = np.arange(apex+1) if phase == 'outward' else np.arange(apex+1, len(va))
            indices = [int(next(j for j in allowed if abs(va[j]-v) <= 1e-10)) for v in voltage]
            selected = obs[indices]
            if np.any(~np.isfinite(selected)) or np.any(selected <= 0):
                raise RuntimeError('Nonpositive/nonfinite observations require explicit handling')
            rows.append({'cycle_id':pair['cycle_id'], 'branch':phase, 'raw_indices':indices,
                         'zero_reference_A':zero, 'n_points_not_above_3zero':int(sum(selected <= 3*zero))})
            currents.append(selected); zeros.append(zero)
    lookup = GateLookup(load_reverse_lookup(), 1.3)
    current = np.array(currents); cap = np.interp(voltage, lookup.voltage, lookup.current)
    for row, obs in zip(rows, current):
        row['capacity_incompatible_point_indices'] = np.flatnonzero(obs > cap).tolist()
    return voltage, current, np.array(zeros), rows, lookup


def simulate(voltage, shared, local, candidate, lookup, height=None, table=None):
    local = np.asarray(local); va = np.broadcast_to(np.asarray(voltage), (len(local), len(voltage)))
    if candidate == 'barrier':
        physical = table_at(height) if table is None else table
        callback = lambda v:physical.internal(v, shared, local[:, None])
    else:
        callback = lambda v:effective_internal(v, shared, local[:, None], candidate)
    return forward_circuit(va, callback, lookup.voltage, lookup.current)


def bounds(candidate):
    if candidate == 'barrier':
        return (-7., -1.), (0., 6.)
    if candidate == 'power':
        return (.15, 12.), (-16., -2.)
    if candidate == 'pf_like':
        return (0., 35.), (-16., -2.)
    raise ValueError(candidate)


def fit(voltage, observed, candidate, lookup, height=None, fixed_shared=None, max_nfev=180):
    n, m = observed.shape; y = np.log10(observed); root_weight = 1/np.sqrt(n*m)
    global_bounds, local_bounds = bounds(candidate)
    offset = 0 if fixed_shared is not None else 1
    lo = np.r_[([global_bounds[0]] if offset else []), np.full(n, local_bounds[0])]
    hi = np.r_[([global_bounds[1]] if offset else []), np.full(n, local_bounds[1])]
    cache = {}; calls = 0
    def evaluate(p):
        nonlocal calls
        if 'p' not in cache or not np.array_equal(p, cache['p']):
            shared = float(p[0]) if offset else fixed_shared
            cache['out'] = simulate(voltage, shared, p[offset:], candidate, lookup, height)
            cache['p'] = p.copy(); calls += 1
        return cache['out']
    def residual(p):
        out = evaluate(p)
        return (np.log10(out['current_a'])-y).ravel()*root_weight
    def jacobian(p):
        out = evaluate(p); indices = np.arange(n*m)
        columns = np.repeat(np.arange(n)+offset, m)
        if offset:
            data = np.r_[out['dlog10_current_dshared'].ravel(), out['dlog10_current_dlocal'].ravel()]*root_weight
            return csr_matrix((data, (np.r_[indices, indices], np.r_[np.zeros(n*m, int), columns])), shape=(n*m, n+1))
        return csr_matrix((out['dlog10_current_dlocal'].ravel()*root_weight, (indices, columns)), shape=(n*m, n))
    guesses = ([fixed_shared] if fixed_shared is not None else
               [-4., -2.] if candidate == 'barrier' else [1., 4.] if candidate == 'power' else [3., 15.])
    attempts = []
    for guess in guesses:
        if candidate == 'barrier':
            model = table_at(height); at = .13
            transmission = model.interpolate(np.full(len(model.width), at), model.width)[0]
            relative = np.median(y-np.log10(voltage)[None, :], axis=1)-guess
            local = np.interp(relative, transmission[::-1], model.width[::-1])
        else:
            if candidate == 'power':
                shape = guess*np.log10(voltage/.2)
            else:
                shape = np.log10(voltage/.2)+guess*(np.sqrt(voltage)-np.sqrt(.2))/np.log(10.)
            local = np.mean(y-shape, axis=1)
        initial = np.r_[([guess] if offset else []), local]
        initial = np.clip(initial, lo+1e-7, hi-1e-7)
        beginning = time.perf_counter()
        solution = least_squares(residual, initial, jac=jacobian, bounds=(lo, hi),
                                 x_scale=np.full(len(initial), 2.), max_nfev=max_nfev,
                                 ftol=1e-8, xtol=1e-8, gtol=1e-8,
                                 tr_options={'atol':1e-10, 'btol':1e-10, 'maxiter':3000})
        attempts.append({'success':bool(solution.success), 'nfev':solution.nfev, 'rmse_decade':float(np.sqrt(2*solution.cost)),
                         'parameters':solution.x.tolist(), 'seconds':time.perf_counter()-beginning,
                         'message':solution.message, 'optimality':float(solution.optimality)})
    best = min(attempts, key=lambda r:r['rmse_decade']); p = np.array(best['parameters'])
    shared = float(p[0]) if offset else float(fixed_shared); local = p[offset:]
    out = simulate(voltage, shared, local, candidate, lookup, height)
    return {'candidate':candidate, 'height_ev':height, 'shared_parameter':shared,
            'local_parameters':local.tolist(), 'rmse_decade':best['rmse_decade'], 'converged':best['success'],
            'multistart':attempts, 'max_kcl_relative_error':out['max_kcl_relative_error'],
            'n_forward_calls':calls,
            'local_boundary_low_count':int(sum(local-local_bounds[0] < 1e-4)),
            'local_boundary_high_count':int(sum(local_bounds[1]-local < 1e-4)),
            'shared_near_boundary':bool(min(shared-global_bounds[0], global_bounds[1]-shared) < 1e-4)}


def metrics(observed, predicted, zeros, voltage, lookup):
    error = np.log10(predicted/observed)
    capacity = np.interp(voltage, lookup.voltage, lookup.current)
    high_signal = observed > 3*zeros[:, None]
    compatible = observed <= capacity[None, :]
    robust = high_signal & compatible
    return {'n_points':int(error.size), 'rmse_decade':float(np.sqrt(np.mean(error**2))),
            'bias_decade':float(np.mean(error)), 'per_branch_rmse_decade':np.sqrt(np.mean(error**2, axis=1)).tolist(),
            'n_high_signal_capacity_compatible':int(robust.sum()),
            'high_signal_capacity_compatible_rmse_decade':float(np.sqrt(np.mean(error[robust]**2))) if robust.any() else None,
            'maximum_absolute_error_decade':float(np.max(np.abs(error)))}


def run(stage='all'):
    va, obs, zeros, branches, lookup = assemble(); ids = np.array([r['cycle_id'] for r in branches])
    input_files = [Path(__file__), ROOT / 'vulcan2d/barrier_state.py', ROOT / 'analysis/BARRIER_STATE_DESIGN_2026-09-21.md',
                   ROOT / 'analysis/diagnose_cross_polarity_physics.py', ROOT / 'vulcan2d/transistor.py',
                   ROOT / '1.3V栅压.xlsx', ROOT / '1.3V栅压RESET.xlsx',
                   ROOT / 'vulcan2d/data/bookoutput2_reverse_transistor_lookup.json']
    inputs = {str(p.relative_to(ROOT)):digest(p) for p in input_files}
    target_path = ROOT / 'analysis/barrier_state_comparison.json'
    report = {'scope':'Frozen low-bias negative-branch constitutive comparison, not a width-dynamics or SET-to-RESET model',
              'status':'running', 'gate_V':1.3, 'height_profile_ev':list(HEIGHTS),
              'mass_ratio':.5, 'temperature_k':300., 'maximum_width_nm':6.,
              'voltage_window_V':va.tolist(), 'observed_current_A':obs.tolist(), 'branches':branches,
              'splits':{}, 'input_sha256':inputs,
              'limitations':['All data informed model development; no external blind validation.',
                             'A constant width on each low-bias branch is a testable frozen-state hypothesis, not an observed fact.',
                             'Width is conditional on effective mass and cannot identify an actual atomic gap.',
                             'The shared prefactor is not a measured count of quantum channels.',
                             'Target low-voltage observations identify a local state; this is constitutive voltage extrapolation, not autonomous or chronological prediction.',
                             'All scored data and capacity/noise flags are retained; the high-signal summary is only a secondary sensitivity.']}
    if target_path.exists():
        old = json.loads(target_path.read_text())
        if old['input_sha256'] != inputs:
            raise RuntimeError('Existing run uses changed code/data; preserve it under a distinct version before recomputing')
        report = old
    def save():
        target_path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    splits = [('all_descriptive', np.ones(len(ids), bool), np.ones(len(ids), bool))]
    if stage == 'all':
        splits += [(f'fold_{lo}_{hi}', (ids < lo) | (ids > hi), (ids >= lo) & (ids <= hi)) for lo, hi in FOLDS]
    for label, training, target in splits:
        record = report['splits'].setdefault(label, {'training_branch_indices':np.flatnonzero(training).tolist(),
                                                    'target_branch_indices':np.flatnonzero(target).tolist(), 'fits':{}})
        candidates = [('power', None), ('pf_like', None)]+[('barrier', h) for h in HEIGHTS]
        for candidate, height in candidates:
            name = candidate if height is None else f'barrier_{height:g}'
            if name in record['fits']:
                continue
            fitted = fit(va, obs[training], candidate, lookup, height)
            record['fits'][name] = fitted
            print(label, name, 'rmse', fitted['rmse_decade'], 'shared', fitted['shared_parameter'],
                  'bounds', fitted['local_boundary_low_count'], fitted['local_boundary_high_count'],
                  'converged', fitted['converged'], flush=True)
            save()
        if 'predictions' in record:
            continue
        barrier_key = min((key for key in record['fits'] if key.startswith('barrier_')),
                          key=lambda key:record['fits'][key]['rmse_decade'])
        record['selected_barrier_profile_key'] = barrier_key; record['predictions'] = {}
        for key in ('power', 'pf_like', barrier_key):
            fitted = record['fits'][key]; candidate = fitted['candidate']; height = fitted['height_ev']
            conditioning = va <= .16+1e-10
            if label == 'all_descriptive':
                identified = fitted
            else:
                identified = fit(va[conditioning], obs[target][:, conditioning], candidate, lookup, height,
                                 fixed_shared=fitted['shared_parameter'])
            output = simulate(va, fitted['shared_parameter'], identified['local_parameters'], candidate, lookup, height)
            score = metrics(obs[target], output['current_a'], zeros[target], va, lookup)
            high = ~conditioning
            record['predictions'][candidate] = {'source_fit_key':key, 'local_identification':identified,
                 'predicted_current_A':output['current_a'].tolist(), 'predicted_internal_voltage_V':output['internal_voltage_v'].tolist(),
                 'max_kcl_relative_error':output['max_kcl_relative_error'], 'all_voltage_score':score,
                 'higher_voltage_score':metrics(obs[target][:, high], output['current_a'][:, high], zeros[target], va[high], lookup)}
            print(label, candidate, 'higher voltage rmse', record['predictions'][candidate]['higher_voltage_score']['rmse_decade'], flush=True)
        save()
    report['status'] = 'complete' if stage == 'all' else 'descriptive_complete_validation_pending'
    save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--stage', choices=('all', 'descriptive'), default='all')
    run(parser.parse_args().stage)
