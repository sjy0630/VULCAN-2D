"""Frozen source-trace kinetics transferred to all other gate-1.3 RESETs.

This controlled homogeneous extension changes only the effective count N.
No target current enters transport, rates, initial state or load-line solves.
Monte Carlo frequencies are model probabilities, not experimental counts.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys
import numpy as np
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'analysis'))
from diagnose_cross_polarity_physics import read_pairs, GateLookup
from finite_path_observables import batch_events, log_readouts
from vulcan2d.transistor import load_reverse_lookup
from vulcan2d.finite_paths import transitions, marginal, sample_counts, state_currents
from vulcan2d.stochastic_bottleneck import clock_grid, transition_log_probabilities
from vulcan2d.reset_history import log_shapes

COUNTS = (1, 2, 4, 8, 16, 32)
CANDIDATES = ('constant', 'single_hop', 'polarizable_saddle')
QUANTILES = (.05, .1, .5, .9, .95)
N_PATHS = 20000
SEED = 512


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(values):
    value = np.asarray(values)
    return {'n': len(value), 'mean': float(np.mean(value)),
            'quantiles': dict(zip(map(str, QUANTILES), np.quantile(value, QUANTILES).tolist()))}


def event_summary(vapp, current, zero):
    detected = batch_events(vapp, current, zero)
    result = {}
    for kind in ('any_rise', 'drop_then_rise'):
        flags = detected[kind]; fraction = float(flags.mean())
        result[kind] = {'n': len(flags), 'n_detected': int(flags.sum()), 'fraction': fraction,
                        'model_monte_carlo_standard_error': float(np.sqrt(fraction*(1-fraction)/len(flags))),
                        'first_half_fraction': float(flags[:len(flags)//2].mean()),
                        'second_half_fraction': float(flags[len(flags)//2:].mean())}
    return result


def prepare():
    source_path = ROOT / 'analysis/stochastic_reentry_probe.json'
    source = json.loads(source_path.read_text())
    bad = [p for p, sha in source['input_sha256'].items() if digest(ROOT / p) != sha]
    if bad:
        raise RuntimeError('Frozen source inputs changed: '+str(bad))
    pairs = read_pairs('1.3')
    ids = np.array([p['cycle_id'] for p in pairs])
    va = np.abs(pairs[0]['reset']['voltage'])
    if not all(np.array_equal(va, np.abs(p['reset']['voltage'])) for p in pairs):
        raise RuntimeError('Different protocols must be evaluated separately')
    current = np.array([np.abs(p['reset']['current']) for p in pairs])
    if np.any(~np.isfinite(current)) or np.any(current <= 0):
        raise RuntimeError('Nonpositive/nonfinite observations require explicit handling')
    zeros = current[:, va <= 1e-12].max(axis=1)
    observed_events = batch_events(va, current, zeros)
    old = json.loads((ROOT / 'analysis/negative_reentry_audit.json').read_text())
    saved = {r['cycle_id']: r for r in old['traces'] if r['gate_V'] == 1.3}
    if set(ids) != set(saved):
        raise RuntimeError('Saved detector coverage differs')
    for j, cid in enumerate(ids):
        for kind in ('rise', 'drop'):
            expected = [r['raw_index_after'] for r in saved[cid]['events'] if r['kind'] == kind]
            if np.flatnonzero(observed_events[kind][j]).tolist() != expected:
                raise RuntimeError('Vectorized detector disagrees with archived raw-data audit')
        if observed_events['drop_then_rise'][j] != saved[cid]['has_persistent_drop_then_rise']:
            raise RuntimeError('Drop/rise classification changed')
    source_index = int(np.flatnonzero(ids == source['cycle_id'])[0])
    np.testing.assert_array_equal(va, source['applied_abs_voltage_V'])
    np.testing.assert_array_equal(current[source_index], source['observed_current_A'])
    target = ids != source['cycle_id']
    return source, va, ids, current, zeros, observed_events, source_index, target


def run():
    source, va, ids, observed, zeros, observed_events, source_index, target = prepare()
    lookup = GateLookup(load_reverse_lookup(), 1.3)
    transport = np.array(source['transport_fit_outward_only']['parameters'])
    apex = int(np.argmax(va)); scored = va >= .08-1e-10
    readouts = log_readouts(va, observed)
    cap = np.interp(va, lookup.voltage, lookup.current)
    capacity = (va[None, :] > 1e-12) & (observed > cap[None, :])
    _, envelopes = state_currents(va, transport, 1, lookup.voltage, lookup.current)
    support_low = (observed < envelopes[None, :, 0]) & scored
    support_high = (observed > envelopes[None, :, 1]) & scored
    # Favorable zero-reference perturbations are a sensitivity check, not a noise model.
    support_low_offset = (observed+zeros[:, None] < envelopes[None, :, 0]) & scored
    support_high_offset = (observed-zeros[:, None] > envelopes[None, :, 1]) & scored
    observed_rows = []
    for j, cid in enumerate(ids):
        observed_rows.append({'cycle_id': int(cid), 'role': 'target' if target[j] else 'source',
                             'zero_reference_A': float(zeros[j]),
                             'any_rise': bool(observed_events['any_rise'][j]),
                             'drop_then_rise': bool(observed_events['drop_then_rise'][j]),
                             'rise_raw_indices': np.flatnonzero(observed_events['rise'][j]).tolist(),
                             'drop_raw_indices': np.flatnonzero(observed_events['drop'][j]).tolist(),
                             'capacity_incompatible_nonzero_indices': np.flatnonzero(capacity[j]).tolist(),
                             'capacity_incompatible_scored_indices': np.flatnonzero(capacity[j] & scored).tolist(),
                             'below_frozen_floor_scored_indices': np.flatnonzero(support_low[j]).tolist(),
                             'above_frozen_ceiling_scored_indices': np.flatnonzero(support_high[j]).tolist(),
                             'below_frozen_floor_after_favorable_offset_indices': np.flatnonzero(support_low_offset[j]).tolist(),
                             'above_frozen_ceiling_after_favorable_offset_indices': np.flatnonzero(support_high_offset[j]).tolist(),
                             **{k:float(v[j]) for k, v in readouts.items()}})
    target_summary = {'n_traces': int(target.sum()),
                      'any_rise_trace_ids': ids[target & observed_events['any_rise']].tolist(),
                      'drop_then_rise_trace_ids': ids[target & observed_events['drop_then_rise']].tolist(),
                      'readouts': {k: summary(v[target]) for k, v in readouts.items()},
                      'n_capacity_incompatible_nonzero_points': int(capacity[target].sum()),
                      'n_capacity_incompatible_scored_points': int((capacity[target] & scored).sum()),
                      'n_below_frozen_floor_scored_points': int(support_low[target].sum()),
                      'n_above_frozen_ceiling_scored_points': int(support_high[target].sum()),
                      'n_traces_below_frozen_floor_after_favorable_offset': int(support_low_offset[target].any(axis=1).sum()),
                      'n_traces_above_frozen_ceiling_after_favorable_offset': int(support_high_offset[target].any(axis=1).sum())}
    zero_refs = {'zero': 0., 'source': float(zeros[source_index]), 'target_max': float(zeros[target].max())}
    rows, flat_rows = [], []
    for candidate in CANDIDATES:
        kinetic = source['candidates'][candidate]['fit']['parameters']
        for count in COUNTS:
            p8, d8 = transitions(va, transport, kinetic, candidate, count, lookup.voltage, lookup.current, substeps=8)
            p16, d16 = transitions(va, transport, kinetic, candidate, count, lookup.voltage, lookup.current, substeps=16)
            m8, m16 = marginal(p8), marginal(p16)
            vh, table = state_currents(va, transport, count, lookup.voltage, lookup.current)
            mean8, mean16 = (m8*table).sum(axis=1), (m16*table).sum(axis=1)
            lh, lc = log_shapes(vh, [transport[1], transport[3]])
            direct = 10**transport[0]*np.exp(lh)+10**transport[2]*np.exp(lc)*np.arange(count+1)/count
            transistor = np.interp(va[:, None]-vh, lookup.voltage, lookup.current)
            nonzero = va > 1e-12
            numerics = {'coarse': d8, 'refined': d16,
                        'maximum_transition_refinement_absolute_difference': float(np.max(np.abs(p8-p16))),
                        'maximum_marginal_refinement_absolute_difference': float(np.max(np.abs(m8-m16))),
                        'maximum_mean_current_refinement_decade': float(np.max(np.abs(np.log10(mean8[scored]/mean16[scored])))),
                        'maximum_marginal_row_sum_error': float(np.max(np.abs(m16.sum(axis=1)-1.))),
                        'minimum_transition_probability': float(np.min(p16)),
                        'maximum_kcl_relative_error': float(np.max(np.abs(table[nonzero]-transistor[nonzero])/table[nonzero])),
                        'maximum_transport_relative_error': float(np.max(np.abs(table[nonzero]-direct[nonzero])/table[nonzero]))}
            if count == 1:
                old_grid = clock_grid(va, transport, lookup.voltage, lookup.current, substeps=16)
                old_p = np.exp(transition_log_probabilities(old_grid, kinetic, candidate))
                numerics['maximum_previous_two_state_absolute_difference'] = float(np.max(np.abs(p16-old_p)))
            paths = sample_counts(p16, N_PATHS, SEED)
            model_current = table[np.arange(len(va))[None, :], paths]
            model_readouts = log_readouts(va, model_current)
            events = {key:event_summary(va, model_current, value) for key, value in zero_refs.items()}
            sampled_marginal = np.array([np.bincount(paths[:, j], minlength=count+1)/N_PATHS for j in range(len(va))])
            numerics['maximum_sampled_marginal_absolute_error'] = float(np.max(np.abs(sampled_marginal-m16)))
            secondary = sample_counts(p16, N_PATHS, SEED+1)
            secondary_current = table[np.arange(len(va))[None, :], secondary]
            secondary_events = event_summary(va, secondary_current, zero_refs['source'])
            secondary_readouts = log_readouts(va, secondary_current)
            replication = {'n_paths': N_PATHS, 'seed': SEED+1, 'source_zero_reference_events': secondary_events,
                           'readouts': {k:summary(v) for k, v in secondary_readouts.items()},
                           'two_sample_ecdf_max_difference': {k:float(ks_2samp(v, secondary_readouts[k]).statistic)
                                                             for k,v in model_readouts.items()}}
            # Exact pointwise current quantiles from the count marginal, not sample quantiles.
            cdf = m16.cumsum(axis=1); cdf[:, -1] = 1.
            qcurrent = np.array([table[np.arange(len(va)), np.argmax(cdf >= q, axis=1)] for q in QUANTILES])
            qlow, qhigh = qcurrent[0], qcurrent[-1]
            outside = ((observed < qlow) | (observed > qhigh))[:, scored]
            # These are marginal bands, not simultaneous bands for an entire experimental path.
            marginal_comparison = {'current_quantiles_A': dict(zip(map(str, QUANTILES), qcurrent.tolist())),
                                   'exact_mean_current_A': mean16.tolist(),
                                   'target_fraction_outside_pointwise_90_percent_band': float(outside[target].mean()),
                                   'target_outside_band_fraction_by_trace': dict(zip(map(str, ids[target]), outside[target].mean(axis=1).tolist()))}
            metrics = {}
            for key, value in model_readouts.items():
                low, median, high = np.quantile(value, [.05, .5, .95]); empirical = readouts[key][target]
                metrics[key] = {**summary(value),
                                'observed_n_below_model_q05': int((empirical < low).sum()),
                                'observed_n_above_model_q95': int((empirical > high).sum()),
                                'model_minus_observed_median_decade': float(median-np.median(empirical)),
                                'model_observed_ecdf_max_difference': float(ks_2samp(value, empirical).statistic)}
            rows.append({'candidate': candidate, 'effective_count': count, 'n_model_paths': N_PATHS, 'seed': SEED,
                         'frozen_kinetic_parameters': kinetic, 'zero_reference_event_sensitivity': events,
                         'readouts': metrics, 'independent_model_sampling_replication': replication,
                         'pointwise_current_comparison': marginal_comparison, 'numerical_checks': numerics})
            flat = {'candidate': candidate, 'effective_count': count,
                    'rise_probability_source_zero': events['source']['any_rise']['fraction'],
                    'paired_probability_source_zero': events['source']['drop_then_rise']['fraction']}
            for key, value in metrics.items():
                for q in ('0.05', '0.5', '0.95'):
                    flat[key+'_q'+q] = value['quantiles'][q]
                flat[key+'_observed_below_q05'] = value['observed_n_below_model_q05']
                flat[key+'_observed_above_q95'] = value['observed_n_above_model_q95']
            flat_rows.append(flat)
            print(candidate, count, 'rise/pair', flat['rise_probability_source_zero'], flat['paired_probability_source_zero'],
                  'median log return/out', metrics['log10_return_outward_ratio_0p5_0p8']['quantiles']['0.5'],
                  'median log low return', metrics['log10_return_current_A_0p1_0p3']['quantiles']['0.5'],
                  'max marginal refinement', numerics['maximum_marginal_refinement_absolute_difference'], flush=True)
    input_paths = [Path(__file__), ROOT / 'analysis/finite_path_observables.py', ROOT / 'vulcan2d/finite_paths.py',
                   ROOT / 'vulcan2d/reset_history.py', ROOT / 'analysis/stochastic_reentry_probe.json',
                   ROOT / 'analysis/FINITE_PATH_TRANSFER_DESIGN_2026-09-21.md', ROOT / '1.3V栅压.xlsx']
    input_paths += [ROOT / p for p in source['input_sha256']]
    result = {'scope': 'Conditional homogeneous transfer test; source trace 23 outward parameters frozen, other 100 gate-1.3 traces never fitted',
              'source_gate_V': 1.3, 'source_cycle_id': 23, 'target_cycle_ids': ids[target].tolist(),
              'effective_counts': list(COUNTS), 'candidates': list(CANDIDATES),
              'protocol': {'all_101_native_grids_identical': True, 'n_points': len(va), 'apex_index': apex,
                           'peak_applied_abs_voltage_V': float(va.max()), 'applied_abs_voltage_V': va.tolist(),
                           'clock': 'applied voltage path; no invented dwell or physical time'},
              'frozen_transport_parameters': transport.tolist(), 'initial_count_equals_N': True,
              'detector_equivalent_to_saved_audit_all_101_traces': True,
              'readout_definition': 'Mean log10 current or matched native equal-voltage ratio; uses all 0.01-V native points, not earlier 0.02-V order-audit grid',
              'zero_reference_sensitivity_A': zero_refs, 'observed_traces': observed_rows, 'target_summary': target_summary,
              'frozen_current_envelopes_A': {'low': envelopes[:, 0].tolist(), 'high': envelopes[:, 1].tolist()},
              'observed_target_pointwise_current_quantiles_A': dict(zip(map(str, QUANTILES), np.quantile(observed[target], QUANTILES, axis=0).tolist())),
              'results': rows,
              'limitations': ['Model parameters were informed by a selected anomaly; this is not external blind validation.',
                              'Count N changes granularity at fixed total current, not a measured channel/atom count.',
                              'No target-specific initial conductance, rate heterogeneity or measurement-noise process is added.',
                              'Within-group device/history identity and independence are unconfirmed; observed frequencies are descriptive, no binomial rejection p-value is assigned.',
                              'The zero reference tests event-detector sensitivity; it is not a complete noise/ranging model.',
                              'Frozen-envelope failures cannot alone diagnose whether transport, initial state, kinetics or measurement response changed.',
                              'Pointwise 90-percent current bands are not simultaneous path prediction regions.',
                              'All target curves and transistor capacity mismatches are retained.'],
              'input_sha256': {str(p.relative_to(ROOT)):digest(p) for p in dict.fromkeys(input_paths)}}
    (ROOT / 'analysis/finite_path_transfer.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    with (ROOT / 'analysis/finite_path_transfer_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0])); writer.writeheader(); writer.writerows(flat_rows)
    print('Observed targets:', json.dumps(target_summary, indent=2), flush=True)


if __name__ == '__main__':
    run()
