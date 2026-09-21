"""Direct, same-polarity terminal I-V hysteresis contrast on all 53 pairs.

No transport law, fitted state, transistor inverse, or amplitude normalization
is used. Contrasts show an observed scan-history distinction, not its cause.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'analysis'))
from diagnose_cross_polarity_physics import read_pairs
from compare_transport_candidates import write_csv

WINDOWS = {'low': (.1, .3), 'middle': (.5, .8), 'upper': (.8, 1.2)}


def branch_value(raw, v, returning, offset=0.):
    voltage = np.abs(np.asarray(raw['voltage'])); current = np.abs(np.asarray(raw['current']))+offset
    first = int(np.argmax(voltage)); last = len(voltage)-1-int(np.argmax(voltage[::-1]))
    indices = np.arange(len(voltage)-1, last-1, -1) if returning else np.arange(first+1)
    found = []
    for a, b in zip(indices[:-1], indices[1:]):
        if voltage[b] <= voltage[a] or not voltage[a]-1e-10 <= v <= voltage[b]+1e-10:
            continue
        if current[a] <= 0 or current[b] <= 0 or not np.isfinite(current[a]+current[b]):continue
        w = np.clip((v-voltage[a])/(voltage[b]-voltage[a]), 0., 1.)
        found.append(float((1-w)*np.log10(current[a])+w*np.log10(current[b])))
    if not found or max(found)-min(found) > 1e-8:return None
    return float(np.mean(found))


def run():
    points, cycles = [], []
    pairs = read_pairs('1.1')
    for pair in pairs:
        cid = pair['cycle_id']
        for phase in ('set', 'reset'):
            raw = pair[phase]; zero = float(np.max(np.abs(raw['current'][np.abs(raw['voltage']) <= 1e-12])))
            for window, (lo, hi) in WINDOWS.items():
                grid = np.round(np.arange(lo, hi+1e-8, .02), 10)
                rows = []
                for v in grid:
                    outward = branch_value(raw, v, False); returned = branch_value(raw, v, True)
                    if outward is None or returned is None:raise RuntimeError('Missing native-adjacent bracket')
                    # SET contrast is return/outward; RESET is outward/return.
                    numerator_return = phase == 'set'
                    numerator = branch_value(raw, v, numerator_return)
                    denominator = branch_value(raw, v, not numerator_return)
                    minimum_numerator = branch_value(raw, v, numerator_return, -zero)
                    maximum_denominator = branch_value(raw, v, not numerator_return, zero)
                    conservative = (minimum_numerator-maximum_denominator
                                    if minimum_numerator is not None and maximum_denominator is not None else None)
                    row = {'cycle_id': cid, 'phase': phase, 'window': window, 'applied_abs_voltage_V': float(v),
                           'log10_outward_A': outward, 'log10_return_A': returned,
                           'log10_history_contrast': numerator-denominator,
                           'zero_reference_lower_log10_contrast': conservative,
                           'zero_reference_A': zero}
                    rows.append(row); points.append(row)
                lower = [r['zero_reference_lower_log10_contrast'] for r in rows]
                cycles.append({'cycle_id': cid, 'phase': phase, 'window': window, 'n_points': len(rows),
                    'mean_log10_history_contrast': float(np.mean([r['log10_history_contrast'] for r in rows])),
                    'rms_log10_outward_return_difference': float(np.sqrt(np.mean([(r['log10_outward_A']-r['log10_return_A'])**2 for r in rows]))),
                    'zero_reference_lower_mean_log10_contrast': float(np.mean(lower)) if all(x is not None for x in lower) else None})
    groups = {}
    for tag, include in [('first3', lambda c: c <= 3), ('cycles4to53', lambda c: c >= 4)]:
        groups[tag] = {}
        for phase in ('set', 'reset'):
            for window in WINDOWS:
                selected = [r for r in cycles if r['phase'] == phase and r['window'] == window and include(r['cycle_id'])]
                x = np.array([r['mean_log10_history_contrast'] for r in selected])
                lower = [r['zero_reference_lower_mean_log10_contrast'] for r in selected]
                groups[tag][phase+'_'+window] = {'n_cycles': len(x), 'geometric_contrast_q10_median_q90': (10**np.quantile(x, [.1, .5, .9])).tolist(),
                    'minimum_geometric_contrast': float(10**x.min()), 'maximum_geometric_contrast': float(10**x.max()),
                    'zero_reference_lower_minimum_contrast': float(10**min(lower)) if all(t is not None for t in lower) else None}
    adjacent = []
    for before, after in zip(pairs[:-1], pairs[1:]):
        row = {'from_cycle': before['cycle_id'], 'to_cycle': after['cycle_id'], 'applied_window_V': [.1, .3]}
        for phase, returning in [('set', True), ('reset', False)]:
            delta = [branch_value(after[phase], v, returning)-branch_value(before[phase], v, returning)
                     for v in np.round(np.arange(.1, .300001, .02), 10)]
            row[phase+'_rmse_between_adjacent_curves_decade'] = float(np.sqrt(np.mean(np.array(delta)**2)))
            row[phase+'_geometric_after_before_ratio'] = float(10**np.mean(delta))
        adjacent.append(row)
    files = [Path(__file__), ROOT/'analysis/diagnose_cross_polarity_physics.py', ROOT/'1T1M写入.xlsx', ROOT/'1T1M擦除.xlsx']
    result = {'scope': 'Direct terminal scan-history contrasts; not intrinsic de-embedded transport or autonomous prediction',
              'definition': 'SET: geometric mean I_return/I_outward; RESET: geometric mean I_outward/I_return at matched applied |V|',
              'windows_V': WINDOWS, 'cycles': cycles, 'groups': groups, 'adjacent_cycle_changes': adjacent,
              'limitations': ['Windows follow the preceding low-bias/network analysis and exploratory inspection of the first six raw sweeps; this is a subsequent descriptive audit.',
                              'Uses measured terminal voltages; unequal currents can produce unequal internal voltages through the transistor.',
                              'A small reverse hysteresis does not prove the positive state relaxed: polarity-selective conduction, inter-scan changes and measurement history remain alternatives.',
                              'No wait time, retention lifetime, switching rate, defect species or minimum state dimension is inferred.',
                              'One early-to-later transition in one sequence is not repeated independent evidence of a universal cycle-count threshold.',
                              'Zero-reference perturbation is illustrative sensitivity, not calibrated instrumental uncertainty.'],
              'input_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (ROOT/'analysis/scan_memory_contrast.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    write_csv(ROOT/'analysis/scan_memory_contrast.csv', cycles)
    write_csv(ROOT/'analysis/scan_memory_contrast_points.csv', points)
    print(json.dumps(groups, indent=2))


if __name__ == '__main__':run()
