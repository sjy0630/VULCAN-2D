"""Fixed-geometry height state compared under the archived width-state protocol."""
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'analysis'))
from compare_barrier_states import assemble, metrics, FOLDS
from vulcan2d.barrier_state import forward_circuit
from vulcan2d.barrier_height_state import build_height_tables

WIDTHS = (.25, .5, .75, 1., 1.5, 2., 3., 4., 6.)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def simulate(voltage, log10_prefactor_S, heights_ev, table, lookup):
    heights = np.asarray(heights_ev)
    va = np.broadcast_to(voltage, (len(heights), len(voltage)))
    return forward_circuit(va, lambda v:table.internal(v, log10_prefactor_S, heights[:, None]),
                           lookup.voltage, lookup.current)


def fit(voltage, observed, table, lookup, fixed_shared=None, max_nfev=240):
    n, m = observed.shape; y = np.log10(observed); weight = 1/np.sqrt(n*m)
    offset = 1 if fixed_shared is None else 0
    lo = np.r_[([-7.] if offset else []), np.full(n, table.height[0])]
    hi = np.r_[([-1.] if offset else []), np.full(n, table.height[-1])]
    cache = {}; calls = 0
    def evaluate(p):
        nonlocal calls
        if 'p' not in cache or not np.array_equal(p, cache['p']):
            cache['p'] = p.copy(); calls += 1
            cache['out'] = simulate(voltage, p[0] if offset else fixed_shared, p[offset:], table, lookup)
        return cache['out']
    def residual(p):
        return (np.log10(evaluate(p)['current_a'])-y).ravel()*weight
    def jacobian(p):
        out = evaluate(p); row = np.arange(n*m); col = np.repeat(np.arange(n)+offset, m)
        if offset:
            values = np.r_[out['dlog10_current_dshared'].ravel(), out['dlog10_current_dlocal'].ravel()]*weight
            return csr_matrix((values, (np.r_[row, row], np.r_[np.zeros(n*m, int), col])), shape=(n*m, n+1))
        return csr_matrix((out['dlog10_current_dlocal'].ravel()*weight, (row, col)), shape=(n*m, n))
    starts = [fixed_shared] if fixed_shared is not None else [-4., -2.]
    attempts = []
    for shared in starts:
        logc = table.interpolate(np.full(len(table.height), .13), table.height)[0]
        expected = np.median(y-np.log10(voltage)[None, :], axis=1)-shared
        heights = np.interp(expected, logc[::-1], table.height[::-1])
        initial = np.clip(np.r_[([shared] if offset else []), heights], lo+1e-7, hi-1e-7)
        beginning = time.perf_counter()
        out = least_squares(residual, initial, jac=jacobian, bounds=(lo, hi), x_scale=np.full(len(initial), 1.),
                            max_nfev=max_nfev, ftol=1e-8, xtol=1e-8, gtol=1e-8,
                            tr_options={'atol':1e-10, 'btol':1e-10, 'maxiter':3000})
        attempts.append({'parameters':out.x.tolist(), 'success':bool(out.success), 'nfev':out.nfev,
                         'rmse_decade':float(np.sqrt(2*out.cost)), 'seconds':time.perf_counter()-beginning,
                         'message':out.message, 'optimality':float(out.optimality)})
    best = min(attempts, key=lambda a:a['rmse_decade']); p = np.array(best['parameters'])
    shared = float(p[0]) if offset else float(fixed_shared); local = p[offset:]
    out = simulate(voltage, shared, local, table, lookup)
    return {'fixed_width_nm':table.width_nm, 'log10_prefactor_S':shared, 'local_height_ev':local.tolist(),
            'rmse_decade':best['rmse_decade'], 'converged':best['success'], 'multistart':attempts,
            'max_kcl_relative_error':out['max_kcl_relative_error'], 'n_forward_calls':calls,
            'n_local_height_at_lower_bound':int(sum(local-table.height[0] < 1e-4)),
            'n_local_height_at_upper_bound':int(sum(table.height[-1]-local < 1e-4)),
            'shared_near_boundary':bool(min(shared+7, -1-shared) < 1e-4)}


def run():
    previous_path = ROOT / 'analysis/barrier_state_comparison.json'
    previous = json.loads(previous_path.read_text())
    for path, sha in previous['input_sha256'].items():
        if digest(ROOT / path) != sha:
            raise RuntimeError('Archived width comparison source changed: '+path)
    va, obs, zeros, branches, lookup = assemble(); ids = np.array([r['cycle_id'] for r in branches])
    np.testing.assert_array_equal(previous['observed_current_A'], obs)
    print('Building finite-temperature height-state tables', flush=True)
    tables = build_height_tables(WIDTHS)
    paths = [Path(__file__), ROOT / 'vulcan2d/barrier_height_state.py', previous_path,
             ROOT / 'analysis/BARRIER_HEIGHT_AND_POLARITY_DESIGN_2026-09-21.md']+[ROOT / p for p in previous['input_sha256']]
    inputs = {str(p.relative_to(ROOT)):digest(p) for p in dict.fromkeys(paths)}
    output = ROOT / 'analysis/barrier_height_comparison.json'
    result = {'scope':'Frozen low-bias barrier-height constitutive comparison; no microscopic attribution or full-history claim',
              'status':'running', 'fixed_width_profile_nm':list(WIDTHS), 'gate_V':1.3,
              'mass_ratio':.5, 'temperature_k':300., 'height_bounds_ev':[.02, 4.], 'log10_prefactor_bounds_S':[-7., -1.],
              'voltage_grid_step_V':.0025, 'height_grid_step_ev':.01, 'quadrature_order':24,
              'voltage_window_V':va.tolist(), 'branches':branches, 'observed_current_A':obs.tolist(),
              'splits':{}, 'input_sha256':inputs,
              'limitations':['The state is an effective height, not a measured contact charge or ion coordinate.',
                             'The low-voltage target segment identifies the state; held-out higher voltages are a constitutive extrapolation, not temporal prediction.',
                             'Temperature, geometry and supply priors are conditional assumptions; all observations, including noise/capacity flags, are retained.']}
    if output.exists():
        saved = json.loads(output.read_text())
        if saved['input_sha256'] != inputs:
            raise RuntimeError('Existing run uses changed source; preserve it before recomputing')
        result = saved
    def save():
        output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    splits = [('all_descriptive', np.ones(len(ids), bool), np.ones(len(ids), bool))]
    splits += [(f'fold_{lo}_{hi}', (ids < lo) | (ids > hi), (ids >= lo) & (ids <= hi)) for lo, hi in FOLDS]
    for label, training, target in splits:
        row = result['splits'].setdefault(label, {'training_branch_indices':np.flatnonzero(training).tolist(),
                                                'target_branch_indices':np.flatnonzero(target).tolist(), 'fits':{}})
        for width in WIDTHS:
            key = f'width_{width:g}'
            if key in row['fits']:
                continue
            fitted = fit(va, obs[training], tables[width], lookup)
            row['fits'][key] = fitted
            print(label, key, 'RMSE', fitted['rmse_decade'], 'supply', fitted['log10_prefactor_S'],
                  'bounds', fitted['n_local_height_at_lower_bound'], fitted['n_local_height_at_upper_bound'],
                  'converged', fitted['converged'], flush=True)
            save()
        if 'prediction' in row:
            continue
        key = min(row['fits'], key=lambda k:row['fits'][k]['rmse_decade']); row['selected_profile_key'] = key
        fitted = row['fits'][key]; table = tables[fitted['fixed_width_nm']]
        conditioning = va <= .16+1e-10
        identified = fitted if label == 'all_descriptive' else fit(va[conditioning], obs[target][:, conditioning], table, lookup,
                                                                 fixed_shared=fitted['log10_prefactor_S'])
        predicted = simulate(va, fitted['log10_prefactor_S'], identified['local_height_ev'], table, lookup)
        high = ~conditioning
        row['prediction'] = {'local_identification':identified, 'predicted_current_A':predicted['current_a'].tolist(),
                             'predicted_internal_voltage_V':predicted['internal_voltage_v'].tolist(),
                             'max_kcl_relative_error':predicted['max_kcl_relative_error'],
                             'all_voltage_score':metrics(obs[target], predicted['current_a'], zeros[target], va, lookup),
                             'higher_voltage_score':metrics(obs[target][:, high], predicted['current_a'][:, high], zeros[target], va[high], lookup)}
        print(label, 'selected', key, 'higher-voltage RMSE', row['prediction']['higher_voltage_score']['rmse_decade'], flush=True)
        save()
    result['status'] = 'complete'; save()


if __name__ == '__main__':
    run()
