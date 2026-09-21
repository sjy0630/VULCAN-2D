"""Raw within-outward drop/recovery signatures; no species assignment."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'analysis'))
from diagnose_cross_polarity_physics import read_pairs, GateLookup, DATASETS
from vulcan2d.transistor import load_reverse_lookup


def events(raw, lookup):
    v, i = np.abs(np.asarray(raw['voltage'])), np.abs(np.asarray(raw['current']))
    zero = float(np.max(i[v <= 1e-12])); apex = int(np.argmax(v)); found = []
    for j in range(3, apex-1):
        if not .2-1e-10 <= v[j-1] < v[j] or v[j]-v[j-1] > .05000001:continue
        following = float(np.median(i[j:j+3]))
        preceding = float(np.median(i[j-3:j]))
        for kind, numerator, denominator in [('drop', min(i[j-1], preceding)-zero, max(i[j], following)+zero),
                                              ('rise', min(i[j], following)-zero, max(i[j-1], preceding)+zero)]:
            ratio = numerator/denominator if numerator > 0 and denominator > 0 else 0.
            if ratio <= 10:continue
            row = {'kind': kind, 'raw_index_after': j, 'applied_abs_voltage_before_V': float(v[j-1]),
                   'applied_abs_voltage_after_V': float(v[j]), 'current_before_A': float(i[j-1]),
                   'current_after_A': float(i[j]), 'conservative_persistent_step_factor': float(ratio),
                   'zero_reference_A': zero}
            for label, index in [('before', j-1), ('after', j)]:
                try:
                    vd = lookup.inverse(float(i[index])); vh = float(v[index]-vd)
                    fraction = float(i[index]/lookup.at_voltage(float(v[index])))
                    row[label+'_hbn_voltage_V'] = vh
                    row[label+'_transistor_voltage_fraction'] = float(vd/v[index])
                    row[label+'_capacity_fraction'] = fraction
                    row[label+'_well_conditioned_lookup'] = bool(vh > 0 and vd/v[index] <= .5 and fraction <= .7)
                except ValueError:
                    row[label+'_hbn_voltage_V'] = None; row[label+'_well_conditioned_lookup'] = False
            found.append(row)
    return found


def run():
    rows, groups = [], {}
    for gate in ('0.9', '1.1', '1.3'):
        lookup = GateLookup(load_reverse_lookup(), float(gate))
        for pair in read_pairs(gate):
            ev = events(pair['reset'], lookup)
            sequences = [[a, b] for a in ev for b in ev if a['kind'] == 'drop' and b['kind'] == 'rise'
                         and a['raw_index_after'] < b['raw_index_after']]
            rows.append({'gate_V': float(gate), 'cycle_id': pair['cycle_id'], 'events': ev,
                         'has_persistent_drop_then_rise': bool(sequences), 'ordered_drop_rise_pairs': sequences})
        own = [r for r in rows if r['gate_V'] == float(gate)]
        groups[gate] = {'n_traces': len(own), 'drop_then_rise_trace_ids': [r['cycle_id'] for r in own if r['has_persistent_drop_then_rise']],
                        'any_rise_trace_ids': [r['cycle_id'] for r in own if any(e['kind'] == 'rise' for e in r['events'])]}
    paths = [Path(__file__), ROOT/'analysis/diagnose_cross_polarity_physics.py', ROOT/'vulcan2d/transistor.py',
             ROOT/'vulcan2d/data/bookoutput2_reverse_transistor_lookup.json']+[ROOT/DATASETS[g][1] for g in ('0.9', '1.1', '1.3')]
    result = {'scope': 'Exploratory signature scan after viewing gate-1.3 trace23; not a pre-registered switching detector',
              'rule': 'Outward negative magnitude >=0.2 V; native step <=0.05 V; conservative zero-reference-adjusted step factor >10, consistent with the median of three preceding and three following points; drop precedes rise.',
              'groups': groups, 'traces': rows,
              'limitations': ['Step selection is descriptive and the current resolution/ranging history is not independently calibrated.',
                              'All currents are treated as stored magnitudes; absolute voltage remains negative in the source.',
                              'A pre/post jump changes the voltage division. Neither endpoint is a precise intrinsic switching threshold.',
                              'A conductance recovery may involve competing state processes or measurement/control history; it does not establish filament chemistry, ion identity or a thermal cause.',
                              'Gate groups differ in reset endpoint and unconfirmed device/history metadata; no isolated gate-causal claim.'],
              'input_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (ROOT/'analysis/negative_reentry_audit.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(groups, indent=2))
    print('Gate1.3 trace23:', next(r for r in rows if r['gate_V'] == 1.3 and r['cycle_id'] == 23))


if __name__ == '__main__':run()
