"""Training-only inverse-state diagnostics; these are NOT forward predictions.

Material/circuit parameters are frozen. Bare SET compliance points produce
state intervals, never point estimates. Exact monotonicity is a mathematical
property of the candidate; observed violations remain conditional on measurement
noise, voltage/current conventions, circuit provenance and the quasi-static
approximation. No kernel or fitted parameters are modified by this script.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vulcan2d.shared_gap import GapLaw, BareGapCircuit, LookupGapCircuit, simulate_gap
from vulcan2d.transistor import measured_transistor_mapping

OUT = Path(__file__).resolve().parent
BASE = ROOT / 'analysis/shared_hbn_2026-11-03'
DATA = BASE / 'data/native_observations.npz'
FIT = BASE / 'gap_joint_fit.json'


def transistor_voltage_interval(circuit, magnitude_a, phase):
    """Keep the full voltage interval for an exact flat-current segment."""
    voltage, current = circuit._branches[phase]
    if magnitude_a > current[-1]:
        return None
    left = int(np.searchsorted(current, magnitude_a, side='left'))
    right = int(np.searchsorted(current, magnitude_a, side='right'))
    if left < len(current) and current[left] == magnitude_a:
        return float(voltage[left]), float(voltage[right-1])
    if left == 0:
        return 0., 0.
    weight = (magnitude_a-current[left-1])/(current[left]-current[left-1])
    v = float(voltage[left-1]+weight*(voltage[left]-voltage[left-1]))
    return v, v


def inverse_point(v, measured_i, top, phase, law, transistor):
    current, voltage = abs(float(measured_i)), abs(float(v))
    floor = 1e-11 if phase == 'set' else 1e-12
    row = {'voltage_V': float(v), 'abs_current_A': current}
    if voltage <= 1e-12 or current < floor:
        return {**row, 'status': 'excluded_zero_or_below_floor'}
    censored = top == '1r' and phase == 'set' and current >= .995e-3
    if top == '1t1r':
        bracket = transistor_voltage_interval(transistor, current, phase)
        if bracket is None:
            return {**row, 'status': 'above_measured_transistor_current_domain'}
        vt_min, vt_max = bracket
        vh_low, vh_high = voltage-vt_max, voltage-vt_min
    else:
        vt_min = vt_max = 0.
        vh_low, vh_high = (0., voltage) if censored else (voltage, voltage)
    row.update(hbn_voltage_interval_V=[vh_low, vh_high],
               transistor_voltage_interval_V=[vt_min, vt_max],
               allowed_ron_upper_ohm=vh_high/current)
    if vh_high < 0.:
        return {**row, 'status': 'no_passive_hbn_voltage_after_transistor'}
    ron_drop = law.ron_ohm*current
    if vh_high < ron_drop-1e-12:
        return {**row, 'status': 'below_fixed_ron_voltage_drop'}
    at_max = law.voltage(current, law.u_max)
    if vh_low > at_max+1e-12:
        return {**row, 'status': 'above_fixed_u_max_voltage_drop'}
    def invert(vh):
        if vh <= ron_drop:
            return 0.
        if vh >= at_max:
            return law.u_max
        return float(brentq(lambda u: law.voltage(current, u)-vh,
                            0., law.u_max, xtol=1e-12))
    low, high = invert(vh_low), invert(vh_high)
    row.update(u_interval=[low, high],
               status='compliance_upper_bound' if censored else
                      ('exact_inverse' if high-low < 1e-9 else 'flat_transistor_interval'))
    return row


def monotone_diagnostics(rows, phase):
    """Greedy interval feasibility is necessary and sufficient for monotonic u."""
    value = float('inf') if phase == 'set' else -float('inf')
    violations = []
    for index, row in enumerate(rows):
        if 'u_interval' not in row:
            continue
        low, high = row['u_interval']
        if phase == 'set':
            value = min(value, high)
            error = max(0., low-value)
        else:
            value = max(value, low)
            error = max(0., value-high)
        if error > 1e-10:
            violations.append({'index': index, 'voltage_V': row['voltage_V'],
                               'required_wrong_direction_u_change': error})
    return {'n_violating_points_exact': len(violations),
            'n_exceeding_u_change': {str(t): sum(x['required_wrong_direction_u_change'] > t
                                             for x in violations) for t in (.05, .1, .25)},
            'max_required_wrong_direction_u_change': max(
                (x['required_wrong_direction_u_change'] for x in violations), default=0.),
            'largest_violations': sorted(violations, key=lambda x:
                x['required_wrong_direction_u_change'], reverse=True)[:5]}


def loop_order(v, ii, phase):
    """Parameter-independent order: SET return>=outward; RESET return<=outward."""
    magnitude, current = abs(v), abs(ii)
    first = int(np.argmax(magnitude))
    last = len(v)-1-int(np.argmax(magnitude[::-1]))
    av, ai = magnitude[:first+1], current[:first+1]
    bv, bi = magnitude[last:][::-1], current[last:][::-1]
    ret = np.interp(av, bv, bi)
    floor = 1e-11 if phase == 'set' else 1e-12
    use = (av > 1e-12) & (av < magnitude[first]-1e-9) & (ai >= floor) & (ret >= floor)
    wrong = np.log10(ai[use]/ret[use]) if phase == 'set' else np.log10(ret[use]/ai[use])
    vv = av[use]
    order = np.argsort(wrong)[::-1]
    return {'eligible_matched_voltage_points': int(sum(use)),
            'wrong_order_count_by_decade': {str(t): int(sum(wrong > t)) for t in (0., .1, .3)},
            'max_wrong_order_decade': float(max(0., np.max(wrong))) if len(wrong) else 0.,
            'largest_violations': [{'abs_voltage_V': float(vv[j]),
                                   'wrong_order_decade': float(wrong[j])}
                                  for j in order[:5] if wrong[j] > 0.]}


def summarize(v, ii, top, phase, law, transistor, include_points=False):
    rows = [inverse_point(a, b, top, phase, law, transistor) for a, b in zip(v, ii)]
    output = {'point_status_counts': dict(Counter(row['status'] for row in rows)),
              'monotonic_u': monotone_diagnostics(rows, phase),
              'monotonic_u_excluding_low_abs_voltage': {
                  str(cut): monotone_diagnostics([row for row in rows
                      if abs(row['voltage_V']) >= cut], phase) for cut in (.1, .5, 1.)},
              'parameter_independent_loop_order': loop_order(v, ii, phase)}
    impossible = [{**row, 'index': i} for i, row in enumerate(rows)
                  if row['status'] == 'below_fixed_ron_voltage_drop']
    output['strongest_fixed_ron_conflicts'] = sorted(impossible,
                  key=lambda x: x['allowed_ron_upper_ohm'])[:4]
    if include_points:
        output['points'] = rows
    return output, rows


def set_resistance_contrast(v, current, points, earlier_v=2., later_v=4.):
    """A transport-parameter-independent two-point necessary condition.

    R_h=Ron+V0*u*asinh(I exp(u)/I0)/I is nonincreasing in I at fixed u,
    and increasing in u at fixed I. SET with u2<=u1 and I2>=I1 therefore
    requires R_h2<=R_h1. No fitted transport parameter enters this test.
    """
    apex = int(np.argmax(abs(v)))
    indices = [int(np.argmin(abs(v[:apex+1]-target))) for target in (earlier_v, later_v)]
    a, b = indices
    if max(abs(v[a]-earlier_v), abs(v[b]-later_v)) > 1e-6 or b <= a:
        return {'status': 'comparison_voltages_not_on_outward_grid'}
    selected = [points[j] for j in indices]
    if any('hbn_voltage_interval_V' not in p or p['status'] == 'compliance_upper_bound'
           for p in selected):
        return {'status': 'voltage_not_identified_or_censored'}
    resistances = [[vh/p['abs_current_A'] for vh in p['hbn_voltage_interval_V']] for p in selected]
    if min(*resistances[0], *resistances[1]) <= 0.:
        return {'status': 'nonpassive_or_zero_hbn_voltage'}
    increasing = bool(abs(current[b]) >= abs(current[a]))
    ratio_low = resistances[1][0]/resistances[0][1]
    ratio_high = resistances[1][1]/resistances[0][0]
    return {'status': 'identified_two_point_test', 'indices': indices,
            'command_voltages_V': [float(v[a]), float(v[b])],
            'currents_A': [float(abs(current[a])), float(abs(current[b]))],
            'hbn_resistance_intervals_ohm': resistances,
            'later_to_earlier_resistance_ratio_interval': [ratio_low, ratio_high],
            'current_increased': increasing,
            'violates_all_positive_transport_parameters': bool(increasing and ratio_low > 1.+1e-10)}


def endpoint_index(v, ending, minimum_abs_v):
    apex = int(np.argmax(abs(v)))
    use = np.flatnonzero((abs(v) >= minimum_abs_v) & (abs(v) > 1e-12) &
                        ((np.arange(len(v)) > apex) if ending else (np.arange(len(v)) < apex)))
    if not len(use):
        return None
    return int(use[-1] if ending else use[0])


def bridge(source, target, top, source_phase, law, transistor, minimum_abs_v):
    sv, si, sr = source
    tv, ti, tr = target
    a, b = endpoint_index(sv, True, minimum_abs_v), endpoint_index(tv, False, minimum_abs_v)
    if a is None or b is None:
        return {'status': 'no_nonzero_boundary_point'}
    left, right = sr[a], tr[b]
    result = {'source_voltage_V': float(sv[a]), 'target_voltage_V': float(tv[b]),
              'source_inverse_status': left['status'], 'target_inverse_status': right['status']}
    if 'u_interval' not in left or left['u_interval'][1]-left['u_interval'][0] > 1e-9:
        return {**result, 'status': 'source_state_not_identified_at_fixed_transport'}
    initial = float(np.mean(left['u_interval']))
    target_phase = 'reset' if source_phase == 'set' else 'set'
    def circuit(phase):
        return transistor if top == '1t1r' else BareGapCircuit(1e-3 if phase == 'set' else None)
    first = simulate_gap([0.], initial, law, circuit(source_phase),
                         initial_command_voltage_v=float(sv[a]), max_step_v=.001)
    second = simulate_gap([float(tv[b])], first['final_u'], law, circuit(target_phase), max_step_v=.001)
    prediction = abs(float(second['current_a'][-1]))
    result.update(status='conditional_oracle_endpoint_bridge',
                  source_inferred_u=initial, predicted_target_u=second['final_u'],
                  target_inferred_u_interval=right.get('u_interval'),
                  state_change_in_connector=second['final_u']-initial,
                  predicted_target_current_A=prediction, observed_target_current_A=abs(float(ti[b])),
                  current_error_decade=float(np.log10(prediction/abs(float(ti[b])))))
    if 'u_interval' in right:
        low, high = right['u_interval']
        result['distance_to_target_u_interval'] = max(low-second['final_u'], second['final_u']-high, 0.)
    return result


def main():
    fit = json.loads(FIT.read_text())
    law = GapLaw(**fit['selected_law'])
    transistor = LookupGapCircuit(measured_transistor_mapping(), 1.1)
    with np.load(DATA) as raw:
        ids = {'1r': sorted({int(k.split('_')[2]) for k in raw.files if k.startswith('1r_')})[:14],
               '1t1r': list(range(4, 36))}
        records, median, traces = {}, {}, {}
        for top, labels in ids.items():
            records[top], median[top] = {}, {}
            for phase in ('set', 'reset'):
                voltages = [raw[f'{top}_cycle_{c}_{phase}_voltage'] for c in labels]
                currents = abs(np.array([raw[f'{top}_cycle_{c}_{phase}_current'] for c in labels]))
                if not all(np.array_equal(v, voltages[0]) for v in voltages):
                    raise ValueError('Median diagnostic requires identical command grids')
                for cid, v, current in zip(labels, voltages, currents):
                    row, inverted = summarize(v, current, top, phase, law, transistor)
                    if phase == 'set':
                        row['set_2V_to_4V_resistance_test'] = set_resistance_contrast(v, current, inverted)
                    records[top].setdefault(str(cid), {})[phase] = row
                    traces[top, cid, phase] = (v, current, inverted)
                current = 10.**np.median(np.log10(np.maximum(currents, 1e-15)), axis=0)
                row, inverted = summarize(voltages[0], current, top, phase, law, transistor, True)
                if phase == 'set':
                    row['set_2V_to_4V_resistance_test'] = set_resistance_contrast(voltages[0], current, inverted)
                median[top][phase] = row
                traces[top, 'median', phase] = (voltages[0], current, inverted)
        bridges = {'set_to_reset': {}, 'reset_to_next_set': {}}
        for top, labels in ids.items():
            bridges['set_to_reset'][top] = {}
            for cid in [*labels, 'median']:
                bridges['set_to_reset'][top][str(cid)] = {
                    'nearest_nonzero': bridge(traces[top, cid, 'set'], traces[top, cid, 'reset'],
                        top, 'set', law, transistor, 0.),
                    'at_least_50mV_control': bridge(traces[top, cid, 'set'], traces[top, cid, 'reset'],
                        top, 'set', law, transistor, .0499)}
        for cid in ids['1t1r'][:-1]:
            bridges['reset_to_next_set'][str(cid)] = {
                'next_cycle': cid+1,
                'nearest_nonzero': bridge(traces['1t1r', cid, 'reset'], traces['1t1r', cid+1, 'set'],
                    '1t1r', 'reset', law, transistor, 0.),
                'at_least_50mV_control': bridge(traces['1t1r', cid, 'reset'], traces['1t1r', cid+1, 'set'],
                    '1t1r', 'reset', law, transistor, .0499)}
    aggregate = {}
    for top, by_id in records.items():
        aggregate[top] = {}
        for phase in ('set', 'reset'):
            counts = Counter()
            for row in by_id.values():
                counts.update(row[phase]['point_status_counts'])
            aggregate[top][phase] = {
                'n_records': len(by_id), 'point_status_counts': dict(counts),
                'records_with_wrong_u_change_over_0p1': sum(row[phase]['monotonic_u'][
                    'max_required_wrong_direction_u_change'] > .1 for row in by_id.values()),
                'records_with_wrong_loop_order_over_0p3_decade': sum(row[phase][
                    'parameter_independent_loop_order']['max_wrong_order_decade'] > .3
                    for row in by_id.values())}
            if phase == 'set':
                contrasts = [row[phase]['set_2V_to_4V_resistance_test'] for row in by_id.values()
                             if row[phase]['set_2V_to_4V_resistance_test']['status'] == 'identified_two_point_test']
                aggregate[top][phase]['set_2V_to_4V_resistance_test'] = {
                    'n_identified': len(contrasts),
                    'n_violating_all_positive_transport_parameters': sum(
                        row['violates_all_positive_transport_parameters'] for row in contrasts),
                    'ratio_lower_bound_min_median_max': np.quantile([
                        row['later_to_earlier_resistance_ratio_interval'][0] for row in contrasts],
                        [0., .5, 1.]).tolist() if contrasts else None}
    output = {'status': 'training_only_inverse_diagnostic_not_forward_prediction_or_unique_state_measurement',
              'fixed_material': asdict(law), 'training_ids': ids, 'aggregate': aggregate,
              'training_medians': median, 'records': records, 'endpoint_bridges': bridges,
              'interpretation': [
                  'Bare SET points at >=0.995mA are treated as censored: only 0<=u<=u_upper from Vh<=Vcommand.',
                  'Inverse states use observed currents and are diagnostics, not autonomous predictions.',
                  'The passive additive transport law requires Vh/I>=Ron; violating points cannot be fitted at this Ron, but a different shared Ron may resolve them.',
                  'SET requires nonincreasing u and RESET nondecreasing u, conditional on the voltage-path model and fixed transport/circuit parameters.',
                  'At matched command voltage, SET return current cannot be lower than outward; RESET return current cannot be higher. This order is parameter-independent for this monotone passive transport/circuit family.',
                  'Stronger SET condition: R_h=Ron+V0*u*asinh(I*exp(u)/I0)/I is nonincreasing in I at fixed u and increasing in u at fixed I. Hence I2>=I1 and u2<=u1 force R_h2<=R_h1, for any positive Ron,V0,I0. The 2V-to-4V contrast tests this using only measured I and the fixed transistor mapping.',
                  'A violation of the stronger resistance condition cannot be removed by re-estimating GapLaw transport parameters or a RESET-only initiation term; it remains conditional on transistor provenance, voltage readback and quasi-static interpretation. It does not exclude all common-material models.',
                  'Numerical thresholds 0.1u and 0.3decade are diagnostic sensitivities, not measured uncertainty estimates or statistical significance tests.',
                  'Monotonicity uses interval feasibility, so censored currents and flat transistor segments are not falsely inverted to exact states.',
                  'Endpoint bridges start from oracle-inferred state and only propagate the short known path through zero. They test compatibility, not predictive accuracy.',
                  'Bare same-label SET/RESET pairing is conditional; no bare record-to-record chronology is assumed.',
                  'Both transistor branches remain conditional on their measured mapping and device provenance; no transistor parameters were adjusted.',
                  'Experimental finite response, offsets, dwell times and readback conventions can invalidate the quasi-static diagnostic assumptions.',
                  'Only the training data identified above were loaded into diagnostics; no holdout curves or errors select a result.'],
              'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (Path(__file__), FIT, DATA, ROOT/'vulcan2d/shared_gap.py',
                            ROOT/'vulcan2d/transistor.py',
                            ROOT/'vulcan2d/data/fig2b_transistor_lookup.json',
                            ROOT/'vulcan2d/data/bookoutput2_reverse_transistor_lookup.json')}}
    path = OUT/'state_diagnostics.json'
    path.write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'file': str(path), 'aggregate': aggregate, 'median_summaries': {
        top: {phase: {key: val for key, val in row.items() if key != 'points'}
              for phase, row in phases.items()} for top, phases in median.items()}}, indent=2))


if __name__ == '__main__':
    main()
