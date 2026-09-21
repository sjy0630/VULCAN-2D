"""Post-hoc sensitivity: compare targets with similar initial outward current.

Subsets use only the 0.1--0.3 V outward segment and never adjust predictions.
This follows the full-cohort result and is not a newly independent validation.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'analysis'))
from diagnose_cross_polarity_physics import read_pairs
from finite_path_observables import log_readouts


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    report = json.loads((ROOT / 'analysis/finite_path_transfer.json').read_text())
    pairs = read_pairs('1.3'); ids = np.array([p['cycle_id'] for p in pairs])
    va = np.abs(pairs[0]['reset']['voltage']); apex = int(np.argmax(va))
    current = np.array([np.abs(p['reset']['current']) for p in pairs])
    zeros = np.max(current[:, va <= 1e-12], axis=1)
    initial = (np.arange(len(va)) <= apex) & (va >= .1-1e-10) & (va <= .3+1e-10)
    initial_log = np.log10(current[:, initial]).mean(axis=1)
    source = int(np.flatnonzero(ids == 23)[0]); target = ids != 23
    readouts = log_readouts(va, current)
    re_low = (np.arange(len(va)) > apex) & (va >= .1-1e-10) & (va <= .3+1e-10)
    return_upper = np.log10(current[:, re_low]+zeros[:, None]).mean(axis=1)
    out = np.flatnonzero((np.arange(len(va)) <= apex) & (va >= .5-1e-10) & (va <= .8+1e-10))
    back = np.array([next(k for k in range(apex+1, len(va)) if abs(va[k]-va[j]) <= 1e-10) for j in out])
    ratio_upper, ratio_lower = np.full(len(ids), np.nan), np.full(len(ids), np.nan)
    valid_upper = np.all(current[:, out]-zeros[:, None] > 0, axis=1)
    ratio_upper[valid_upper] = np.log10((current[valid_upper][:, back]+zeros[valid_upper, None]) /
                                      (current[valid_upper][:, out]-zeros[valid_upper, None])).mean(axis=1)
    valid_lower = np.all(current[:, back]-zeros[:, None] > 0, axis=1)
    ratio_lower[valid_lower] = np.log10((current[valid_lower][:, back]-zeros[valid_lower, None]) /
                                      (current[valid_lower][:, out]+zeros[valid_lower, None])).mean(axis=1)
    cases = []
    for factor in (None, 2., 3., 10.):
        mask = target.copy()
        if factor is not None:
            mask &= np.abs(initial_log-initial_log[source]) <= np.log10(factor)
        n = int(mask.sum())
        if not n:
            raise RuntimeError('Empty matched subset must be reported explicitly')
        robust_upper = mask & valid_upper; robust_lower = mask & valid_lower
        data = {'maximum_initial_current_factor_from_source': factor, 'n_targets': n,
                'target_cycle_ids': ids[mask].tolist(),
                'initial_log_current_A_q05_q50_q95': np.quantile(initial_log[mask], [.05, .5, .95]).tolist(),
                'any_rise_n': sum(r['any_rise'] for r in report['observed_traces'] if r['cycle_id'] in ids[mask]),
                'drop_then_rise_n': sum(r['drop_then_rise'] for r in report['observed_traces'] if r['cycle_id'] in ids[mask]),
                'observed_readouts': {k:{'median': float(np.median(value[mask])),
                                          'q05_q50_q95': np.quantile(value[mask], [.05, .5, .95]).tolist()}
                                      for k, value in readouts.items()},
                'zero_reference_sensitivity': {
                    'return_current_upper_median_log10_A': float(np.median(return_upper[mask])),
                    'ratio_upper_valid_n': int(robust_upper.sum()),
                    'ratio_upper_invalid_cycle_ids': ids[mask & ~valid_upper].tolist(),
                    'ratio_upper_median_log10_valid_only': float(np.median(ratio_upper[robust_upper])) if robust_upper.any() else None,
                    'ratio_lower_valid_n': int(robust_lower.sum()),
                    'ratio_lower_invalid_cycle_ids': ids[mask & ~valid_lower].tolist(),
                    'ratio_lower_median_log10_valid_only': float(np.median(ratio_lower[robust_lower])) if robust_lower.any() else None},
                'frozen_model_comparisons': []}
        for model in report['results']:
            metrics = {}
            for key, predicted in model['readouts'].items():
                observed = readouts[key][mask]; quantiles = predicted['quantiles']
                metrics[key] = {'model_minus_observed_median_decade': float(quantiles['0.5']-np.median(observed)),
                                'observed_n_below_model_q05': int((observed < quantiles['0.05']).sum()),
                                'observed_n_above_model_q95': int((observed > quantiles['0.95']).sum())}
            data['frozen_model_comparisons'].append({'candidate': model['candidate'], 'effective_count': model['effective_count'],
                                                    'metrics': metrics})
        cases.append(data)
        print('Factor', factor, 'n', n, 'paired', data['drop_then_rise_n'],
              'median log return/out, low return:', [r['median'] for r in data['observed_readouts'].values()],
              'upper-current offset:', data['zero_reference_sensitivity']['return_current_upper_median_log10_A'], flush=True)
    inputs = [Path(__file__), ROOT / 'analysis/finite_path_transfer.json', ROOT / 'analysis/finite_path_observables.py',
              ROOT / 'analysis/diagnose_cross_polarity_physics.py', ROOT / '1.3V栅压.xlsx', ROOT / '1.3V栅压RESET.xlsx']
    result = {'scope': 'Post-hoc initial-current matching and zero-reference sensitivity; no parameter refit or independent validation claim',
              'source_initial_outward_geometric_current_A': float(10**initial_log[source]),
              'initial_window_applied_abs_voltage_V': [.1, .3], 'cases': cases,
              'limitations': ['Similar initial current does not establish the same microscopic state or same device.',
                              'Factors 2, 3 and 10 are post-hoc sensitivity windows, not pre-registered selection.',
                              'Offset extremes are favorable readout perturbations, not a calibrated measurement-noise model.',
                              'Targets with nonpositive perturbed denominator/numerator remain recorded as unbounded, not clipped.'],
              'input_sha256': {str(p.relative_to(ROOT)):digest(p) for p in inputs}}
    (ROOT / 'analysis/finite_path_initial_match.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    run()
