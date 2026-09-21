"""Native paired measurements against a conditional two-state MEAN bound."""
from pathlib import Path
import hashlib
import json
import numpy as np
from diagnose_cross_polarity_physics import read_pairs
ROOT = Path(__file__).resolve().parents[1]


def run():
    pairs = read_pairs('1.1'); result = {}
    for phase, direction in [('set', 'return'), ('reset', 'outward')]:
        curves = []
        for c in (3, 4):
            raw = pairs[c-1][phase]; voltage = np.abs(raw['voltage']); current = np.abs(raw['current'])
            apex = np.flatnonzero(np.isclose(voltage, voltage.max(), atol=1e-10, rtol=0))
            index = np.arange(len(voltage)); branch = index > apex[-1] if direction == 'return' else index < apex[0]
            mask = branch & (voltage >= .1-1e-10) & (voltage <= .3+1e-10)
            order = np.argsort(voltage[mask]); zero = current[voltage == 0]
            curves.append((voltage[mask][order], current[mask][order], float(np.max(zero))))
        (v3, i3, z3), (v4, i4, z4) = curves
        np.testing.assert_allclose(v3, v4, rtol=0, atol=1e-12)
        lower4 = np.maximum(i4-z4, 0); upper3 = i3+z3
        valid = lower4 > 0
        result[phase+'_'+direction] = {
            'voltage_magnitude_V': v3.tolist(), 'cycle3_current_A': i3.tolist(), 'cycle4_current_A': i4.tolist(),
            'mean_model_ratio_upper_bound': 1.5,
            'geometric_mean_observed_ratio': float(np.exp(np.mean(np.log(i4/i3)))),
            'all_points_positive_after_favorable_reference_shift': bool(valid.all()),
            'geometric_mean_ratio_after_favorable_reference_shift': float(np.exp(np.mean(np.log(lower4/upper3)))) if valid.all() else None,
            'observed_pointwise_ratios': (i4/i3).tolist(),
            'favorable_reference_pointwise_ratios': (lower4/upper3).tolist(),
            'observed_violation_count': int(np.count_nonzero(i4 > 1.5*i3)),
            'favorable_reference_violation_count': int(np.count_nonzero(lower4 > 1.5*upper3)),
            'zero_reference_scales_A': [z3, z4]}
    names = ['analysis/check_repeated_two_state_mean_bound.py', 'analysis/REPEATED_TWO_STATE_MEAN_BOUND_2026-09-21.md',
             'tests/test_repeated_two_state_mean.py', 'analysis/diagnose_cross_polarity_physics.py', '1T1M写入.xlsx', '1T1M擦除.xlsx']
    output = {'scope': 'Periodic independent two-state ensemble-mean current; not a sample-path bound',
              'novelty_verified': False, 'single_random_path_excluded': False,
              'input_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names},
              'data': result}
    (ROOT/'analysis/repeated_two_state_mean_bound.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': run()
