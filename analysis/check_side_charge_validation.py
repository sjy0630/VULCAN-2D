"""Numerics, SET information profiles, symmetric charge bound and port forecasts."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'analysis'))
from compare_parallel_paths import assemble, VOLTAGE
from compare_side_charge_readout import profile_supply
from diagnose_cross_polarity_physics import read_pairs, GateLookup
from vulcan2d.transistor import load_reverse_lookup
from vulcan2d.side_charge import predict, shape_and_gradients, VT
from vulcan2d.side_charge_circuit import forward_initial


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def energy_profile(shared, fitted, y, weights, mode):
    """Fixed shared physics; diagnostic tolerance, not statistical confidence."""
    epsilon = np.unique(np.r_[np.linspace(-.6, .6, 1201), fitted['states'][0][1]])
    shape = shape_and_gradients(VOLTAGE, shared, epsilon, mode=mode, order=48)[0]
    positive = VOLTAGE > 0
    w = np.broadcast_to(weights[positive], shape[:, positive].shape)
    amplitude, _ = profile_supply(y[None, positive], shape[:, positive], w)
    error = shape[:, positive]+amplitude[:, None]-y[positive]
    rmse = np.sqrt(np.sum(w*error**2, axis=1)/w.sum(axis=1))
    retained = rmse <= rmse.min()+.005
    lower = amplitude[retained, None]+shape[retained][:, ~positive]
    optimal = int(np.argmin(rmse))
    negative_error = amplitude[:, None]+shape[:, ~positive]-y[~positive]
    best_negative = np.sqrt(np.mean(negative_error**2, axis=1))
    return {'definition':'Fixed shared physics; supply reprofiled from SET; all epsilon values within +0.005 decade SET RMSE of grid minimum. Not a confidence interval.',
            'epsilon_grid_ev':epsilon.tolist(), 'SET_rmse_profile_decade':rmse.tolist(),
            'profiled_log10_supply_S':amplitude.tolist(), 'grid_best_epsilon_ev':float(epsilon[optimal]),
            'grid_best_SET_rmse_decade':float(rmse[optimal]), 'n_retained_states':int(sum(retained)),
            'retained_epsilon_min_max_ev':[float(epsilon[retained].min()), float(epsilon[retained].max())],
            'negative_prediction_log10_A_min':lower.min(axis=0).tolist(),
            'negative_prediction_log10_A_max':lower.max(axis=0).tolist(),
            'negative_at_0p1_prediction_spread_decade':float(np.ptp(lower[:, 0])),
            'best_negative_rmse_inside_SET_tolerance_decade':float(best_negative[retained].min()),
            'worst_negative_rmse_inside_SET_tolerance_decade':float(best_negative[retained].max())}


def run():
    source = ROOT/'analysis/side_charge_readout.json'
    main = json.loads(source.read_text())
    if main['status'] != 'complete':
        raise RuntimeError('Comparison incomplete')
    for p, sha in main['input_sha256'].items():
        if digest(ROOT/p) != sha:
            raise RuntimeError('Executed input changed: '+p)
    ids, y, weights, selection = assemble()
    np.testing.assert_array_equal(main['observed_log10_current_A'], y)
    np.testing.assert_array_equal(main['weights'], weights)
    numerical, ports, profiles = [], {}, {}
    pairs = {p['cycle_id']:p for p in read_pairs('1.1')}
    lookup = GateLookup(load_reverse_lookup(), 1.1)
    for mode, candidate in main['candidates'].items():
        forecasts, info = [], []
        for label, split in candidate['splits'].items():
            fit = split['fit']
            original = predict(VOLTAGE, fit['shared_parameters'], fit['states'], mode=mode)
            refined = predict(VOLTAGE, fit['shared_parameters'], fit['states'], mode=mode, order=48)
            numerical.append({'mode':mode, 'split':label, 'kind':'training',
                              'maximum_24_48_difference_decade':float(np.max(np.abs(original-refined)))})
            for protocol, target in split['targets'].items():
                fitted = target['identification']
                predicted = predict(VOLTAGE, fit['shared_parameters'], fitted['states'], mode=mode, order=48)
                numerical.append({'mode':mode, 'split':label, 'kind':protocol,
                                  'maximum_24_48_difference_decade':float(np.max(np.abs(predicted-target['predicted_log10_A'])))})
            if not label.startswith('fold_'):
                continue
            identified = split['targets']['SET_only']['identification']
            for cid, state in zip(split['target_ids'], identified['states']):
                raw = pairs[cid]['reset']; va = np.asarray(raw['voltage']); obs = np.abs(raw['current'])
                apex = int(np.argmax(np.abs(va))); indices = np.flatnonzero((np.arange(len(va)) <= apex) & (np.abs(va) >= .1-1e-10) & (np.abs(va) <= .3+1e-10))
                zero = float(np.max(obs[np.abs(va) <= 1e-12]))
                primary = forward_initial(va[indices], fit['shared_parameters'], state, lookup, mode=mode, step=.001, order=24)
                refined = forward_initial(va[indices], fit['shared_parameters'], state, lookup, mode=mode, step=.0005, order=48)
                roots_same = all(len(a['roots']) == len(b['roots']) for a, b in zip(primary['root_diagnostics'], refined['root_diagnostics']))
                row = {'cycle_id':cid, 'training_ids':split['training_ids'], 'split':label, 'state':state,
                       'raw_indices':indices.tolist(), 'applied_voltage_V':va[indices].tolist(),
                       'observed_current_A':obs[indices].tolist(), 'zero_reference_A':zero,
                       'primary':primary, 'refined':refined, 'root_count_same_after_refinement':roots_same}
                if not primary['ambiguous_count'] and not refined['ambiguous_count']:
                    current = np.array(refined['predicted_current_A'])
                    error = np.log10(current/obs[indices]); strong = obs[indices] > 3*zero
                    row.update(rmse_decade=float(np.sqrt(np.mean(error**2))),
                               maximum_refinement_decade=float(np.max(np.abs(np.log10(current/primary['predicted_current_A'])))),
                               errors_decade=error.tolist(), high_signal_errors_decade=error[strong].tolist(),
                               high_signal_point_count=int(sum(strong)))
                forecasts.append(row)
                profile = energy_profile(fit['shared_parameters'], {'states':[state]}, y[cid-1], weights[cid-1], mode)
                profile.update(cycle_id=cid, split=label, shared_parameters=fit['shared_parameters'], selected_state=state)
                info.append(profile)
            print(mode, label, 'port forecasts and energy profiles complete', flush=True)
        groups = {}
        for group, predicate in [('all', lambda c:True), ('first3', lambda c:c <= 3), ('later', lambda c:c > 3)]:
            rows = [r for r in forecasts if predicate(r['cycle_id'])]
            completed = [r for r in rows if 'rmse_decade' in r]
            errors = [e for r in completed for e in r['errors_decade']]
            high = [e for r in completed for e in r['high_signal_errors_decade']]
            groups[group] = {'n_target_pairs':len(rows), 'n_unambiguous_scored_pairs':len(completed),
                             'n_scored_points':len(errors), 'n_high_signal_points':len(high),
                             'equal_pair_rmse_decade':float(np.sqrt(np.mean([r['rmse_decade']**2 for r in completed]))) if completed else None,
                             'high_signal_pooled_rmse_decade':float(np.sqrt(np.mean(np.square(high)))) if high else None}
        ports[mode] = {'groups':groups, 'cycles':forecasts}
        profiles[mode] = info
    observed_path = ROOT/'analysis/wkb_frozen_polarity_bound.json'
    observed = json.loads(observed_path.read_text())
    for p, sha in observed['input_sha256'].items():
        if digest(ROOT/p) != sha:
            raise RuntimeError('Observed ratio provenance changed: '+p)
    bound = []
    for row in observed['first3_all_processing_at_0p1V']:
        v = row['internal_voltage_V']; favorable = row['favorable_zero_reference_ratio_upper']
        raw = row['observed_ratio']
        bound.append({'cycle_id':row['cycle_id'], 'processing':row['processing'], 'internal_voltage_V':v,
                      'raw_ratio':raw, 'favorable_ratio_upper':favorable,
                      'necessary_abs_lever_arm_raw':float(VT*abs(np.log(raw))/(2*v)),
                      'necessary_abs_lever_arm_favorable':float(VT*abs(np.log(favorable))/(2*v)),
                      'interpretation':'Necessary, not sufficient, for a single equilibrated reservoir and two symmetric frozen configuration currents. Not inferred trap position.'})
    files = [Path(__file__), source, observed_path, ROOT/'vulcan2d/side_charge_circuit.py',
             ROOT/'tests/test_side_charge_circuit.py']+[ROOT/p for p in main['input_sha256']]+[ROOT/p for p in observed['input_sha256']]
    result = {'scope':'Charge-readout numerical refinement, target-RESET-free native port forecasts, and conditional SET-information profiles',
              'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in dict.fromkeys(files)},
              'quadrature_refinements':numerical, 'native_RESET_port_forecasts':ports,
              'SET_energy_information_profiles':profiles, 'symmetric_single_reservoir_lever_bound':bound,
              'limitations':['SET energy tolerance is a diagnostic choice, not a confidence interval.',
                             'Port scores condition on observed target SET; no target RESET currents enter the forward solver.',
                             'Multiple load roots are unresolved without further dynamics and would be reported, not selected using measurements.',
                             'No structural-state update law or full-cycle prediction is established.']}
    (ROOT/'analysis/side_charge_validation.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('Max quadrature difference', max(r['maximum_24_48_difference_decade'] for r in numerical), flush=True)
    for mode, result in ports.items():
        print(mode, result['groups'], flush=True)


if __name__ == '__main__':
    run()
