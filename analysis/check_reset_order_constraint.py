"""Test the terminal-current order preserved by one-way RESET state models.

If I_h(v,s) is monotone in v and in a conducting-state coordinate s, and the
series transistor is a fixed passive monotone load line, then s_return <=
s_outward implies I_return(Vapp) <= I_outward(Vapp) at matched applied bias.
At the outward internal-voltage root, decreasing s makes the residual I_h-I_T
nonpositive; its next root moves to larger internal voltage and smaller I_T.
This comparison does not need a numerical transistor inverse or a transport
formula. It does assume the recorded currents represent quasistatic conduction.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'analysis'))
from diagnose_cross_polarity_physics import read_pairs, DATASETS
from check_scan_memory_contrast import branch_value
from compare_transport_candidates import write_csv

GRID = np.round(np.arange(.5, .800001, .02), 10)


def run():
    rows, points, examples, groups = [], [], [], {}
    for gate in ('0.9', '1.1', '1.3'):
        pairs = read_pairs(gate)
        for pair in pairs:
            raw = pair['reset']; v, i = np.asarray(raw['voltage']), np.asarray(raw['current'])
            zero = float(np.max(np.abs(i[np.abs(v) <= 1e-12])))
            contrast, lower, signs = [], [], []
            for target in GRID:
                out, ret = branch_value(raw, target, False), branch_value(raw, target, True)
                out_upper = branch_value(raw, target, False, zero)
                ret_lower = branch_value(raw, target, True, -zero)
                if out is None or ret is None:raise RuntimeError('Missing observed comparison')
                minimum = ret_lower-out_upper if ret_lower is not None and out_upper is not None else None
                contrast.append(ret-out); lower.append(minimum)
                points.append({'gate_V': float(gate), 'cycle_id': pair['cycle_id'], 'applied_abs_voltage_V': float(target),
                               'outward_log10_current_A': out, 'return_log10_current_A': ret,
                               'log10_return_outward_ratio': ret-out, 'zero_reference_lower_log10_ratio': minimum})
            mask = (np.abs(v) >= .5-1e-10) & (np.abs(v) <= .8+1e-10)
            geometric = float(10**np.mean(contrast))
            conservative = float(10**np.mean(lower)) if all(x is not None for x in lower) else None
            row = {'gate_V': float(gate), 'cycle_id': pair['cycle_id'], 'reset_peak_abs_voltage_V': float(np.max(np.abs(v))),
                   'geometric_return_outward_ratio': geometric, 'zero_reference_lower_geometric_ratio': conservative,
                   'fraction_grid_return_above_outward': float(np.mean(np.array(contrast) > 0)),
                   'zero_reference_A': zero, 'zero_reference_lower_minimum_point_ratio': float(10**min(lower)) if conservative is not None else None,
                   'negative_voltage_fraction_in_window': float(np.mean(v[mask] < 0)),
                   'positive_stored_current_fraction_in_window': float(np.mean(i[mask] > 0)),
                   'strong_order_reversal': bool(conservative is not None and conservative > 3 and np.mean(np.array(contrast) > 0) >= .9)}
            rows.append(row)
            if row['strong_order_reversal']:
                examples.append({'gate_V': float(gate), 'cycle_id': pair['cycle_id'], 'voltage_v': v.tolist(), 'current_a': i.tolist(),
                                 'note': 'Selected by saved >3x conservative geometric contrast and >=90% point direction rule; not a distinct microscopic classification.'})
        selected = [r for r in rows if r['gate_V'] == float(gate)]
        groups[gate] = {'n_traces': len(selected), 'reset_peak_abs_voltages_V': sorted({r['reset_peak_abs_voltage_V'] for r in selected}),
                        'nominal_return_above_outward_ids': [r['cycle_id'] for r in selected if r['geometric_return_outward_ratio'] > 1],
                        'strong_conservative_reversal_ids': [r['cycle_id'] for r in selected if r['strong_order_reversal']],
                        'all_selected_recorded_voltages_negative': all(r['negative_voltage_fraction_in_window'] == 1 for r in selected),
                        'all_stored_currents_positive_in_window': all(r['positive_stored_current_fraction_in_window'] == 1 for r in selected)}
    files = [Path(__file__), ROOT/'analysis/diagnose_cross_polarity_physics.py', ROOT/'analysis/check_scan_memory_contrast.py']
    files += [ROOT/DATASETS[g][1] for g in ('0.9', '1.1', '1.3')]
    result = {'claim': 'Conditional order constraint of a one-way decreasing conducting state plus fixed passive monotone circuit',
              'comparison_window_applied_V': [.5, .8], 'groups': groups, 'cycles': rows, 'raw_strong_examples': examples,
              'selection_rule': 'Exploratory follow-up to prior scan audit: conservative zero-reference-perturbed geometric return/outward >3, at least 90% matched grid points increase.',
              'limitations': ['Same-polarity within-trace comparisons do not require cross-gate pairing or device identity; interpretation of sequence position does.',
                              'RESET workbooks store negative voltage and positive current columns throughout the comparison window. Treat current as a magnitude/convention; their algebraic product is not established signed physical power.',
                              '0.9/1.3-V datasets have a 1.5-V RESET peak versus 1.7 V at gate 1.1; cannot isolate a gate-only effect.',
                              'Cross-gate device identities, collection order and dwell/gap times are currently unconfirmed.',
                              'Presumes quasistatic conduction, a fixed gate/load line within a trace, and a state coordinate that increases conductance monotonically.',
                              'Capacitive/background/measurement-history contributions beyond the tested zero-reference scale have not been quantified.',
                              'Rejects a universally one-way loss rule under these assumptions; does not establish ion species or prove a required number of state variables.',
                              'The factor-three margin is a descriptive robustness rule, not a hypothesis-test significance threshold.'],
              'input_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (ROOT/'analysis/reset_order_constraint.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    write_csv(ROOT/'analysis/reset_order_constraint.csv', rows); write_csv(ROOT/'analysis/reset_order_constraint_points.csv', points)
    print(json.dumps(groups, indent=2))
    print('Robust examples', [(r['gate_V'], r['cycle_id'], r['geometric_return_outward_ratio'], r['zero_reference_lower_geometric_ratio']) for r in rows if r['strong_order_reversal']])


if __name__ == '__main__':run()
