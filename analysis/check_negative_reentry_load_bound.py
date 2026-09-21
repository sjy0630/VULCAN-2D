"""Finite-step load-line bound on current without an increase in conductance.

For a fixed passive monotone transistor and a memristor whose current is
nondecreasing in its internal voltage, a fixed or decreasing conductive state
requires v_h,after >= v_h,before whenever I_after > I_before. Therefore
I_after <= I_T(I_T^{-1}(I_before) + delta_Vapp).

Only the BEFORE current is inverted. This avoids interpreting the poorly
conditioned internal voltage after some large recoveries as a threshold.
The inequality does not identify a microscopic species or distinguish a
structural conductance change from a dynamic contact/readout change.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analysis'))
from diagnose_cross_polarity_physics import GateLookup
from vulcan2d.transistor import load_reverse_lookup


def no_enhancement_bound(lookup, current_before, applied_before, applied_after, current_scale=1.):
    """Return the exact finite-step upper bound for a given monotone load."""
    values = np.array([current_before, applied_before, applied_after, current_scale])
    if not np.all(np.isfinite(values)) or current_before < 0 or applied_before < 0 or current_scale <= 0:
        raise ValueError('Finite nonnegative observations and positive current scale required')
    if applied_after < applied_before:
        raise ValueError('This bound is for increasing applied-voltage magnitude')
    vt = lookup.inverse(current_before / current_scale)
    if vt > applied_before:
        raise ValueError('Before point exceeds the passive series-load capacity')
    limit_vt = vt + applied_after - applied_before
    limit = current_scale * lookup.at_voltage(limit_vt)
    return {'transistor_voltage_before_V': float(vt),
            'transistor_voltage_upper_without_enhancement_V': float(limit_vt),
            'current_upper_without_enhancement_A': float(limit)}


def run():
    source = ROOT / 'analysis/negative_reentry_audit.json'
    raw = json.loads(source.read_text())
    rows = []
    for trace in raw['traces']:
        for event in trace['events']:
            if event['kind'] != 'rise':
                continue
            perturbations = []
            for branch in ('mean', 'up'):
                lookup = GateLookup(load_reverse_lookup(branch=branch), trace['gate_V'])
                for current_scale in (.5, 1., 2.):
                    for conservative in (False, True):
                        zero = event['zero_reference_A'] if conservative else 0.
                        before = event['current_before_A'] + zero
                        after = event['current_after_A'] - zero
                        record = {'transistor_branch': branch, 'transistor_current_scale': current_scale,
                                  'zero_reference_favorable_perturbation': conservative}
                        try:
                            bound = no_enhancement_bound(lookup, before, event['applied_abs_voltage_before_V'],
                                                         event['applied_abs_voltage_after_V'], current_scale)
                            record.update(bound)
                            record.update({'valid': True, 'observed_after_A': after,
                                           'observed_over_no_enhancement_bound': float(after / bound['current_upper_without_enhancement_A'])})
                        except ValueError as error:
                            record.update({'valid': False, 'reason': str(error)})
                        perturbations.append(record)
            nominal = next(p for p in perturbations if p['transistor_branch'] == 'mean'
                           and p['transistor_current_scale'] == 1. and not p['zero_reference_favorable_perturbation'])
            valid = [p for p in perturbations if p['valid']]
            rows.append({'gate_V': trace['gate_V'], 'cycle_id': trace['cycle_id'],
                         'raw_index_after': event['raw_index_after'],
                         'part_of_drop_then_rise_trace': trace['has_persistent_drop_then_rise'],
                         'event': event, 'nominal': nominal, 'sensitivity_checks': perturbations,
                         'n_valid_sensitivity_checks': len(valid),
                         'minimum_observed_over_bound_in_valid_checks': min(p['observed_over_no_enhancement_bound'] for p in valid)})
    paths = [Path(__file__), source, ROOT / 'analysis/diagnose_cross_polarity_physics.py',
             ROOT / 'vulcan2d/transistor.py', ROOT / 'vulcan2d/data/bookoutput2_reverse_transistor_lookup.json']
    result = {'scope': 'Conditional finite-step passive load-line exclusion; all persistent upward steps in the saved exploratory detector',
              'equation': 'I_after <= I_T(I_T^{-1}(I_before) + Vapp_after - Vapp_before) if conducting ability does not increase',
              'rows': rows,
              'limitations': ['Assumes quasistatic conduction, a fixed transistor within the step, and nonnegative differential memristor conductance at frozen state.',
                              'A time-dependent interface/contact or charge state can change conducting ability without atomic reconstruction.',
                              'The transistor is a separately measured lookup; transfer to the measured cell is an assumption.',
                              'The factors 0.5 and 2 are stress tests, not a calibrated transistor uncertainty interval.',
                              'Zero-reference perturbations are sensitivity checks, not confidence intervals; measurement range/control artifacts remain unquantified.',
                              'Only the before-point transistor voltage is inverted; no precise intrinsic recovery threshold is inferred.',
                              'The step detector and this follow-up were designed after inspecting the data; not an external blind validation.'],
              'input_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (ROOT / 'analysis/negative_reentry_load_bound.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for row in rows:
        print(row['gate_V'], row['cycle_id'], 'nominal ratio', row['nominal'].get('observed_over_no_enhancement_bound'),
              'minimum stress ratio', row['minimum_observed_over_bound_in_valid_checks'],
              'valid checks', row['n_valid_sensitivity_checks'])


if __name__ == '__main__':
    run()
