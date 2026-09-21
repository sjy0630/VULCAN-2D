"""An exact conditional bound for any positive parallel fixed-rate level spectrum.

For one level, |j(v)| is proportional to sinh(|v|/2VT) divided by
cosh([E+(1/2-eta)v]/VT)+cosh(v/2VT). The two denominator arguments
differ by (1-2eta)v/VT. Since |d log(cosh(x)+c)/dx| <= 1 for c>=1,
|log(|j(-v)|/j(v))| <= |1-2eta| |v|/VT <= |v|/VT.

For any nonnegative, voltage-independent parallel weights, the total ratio
is a convex average of individual ratios, weighted by positive currents.
The bound survives arbitrary level energies, widths, lever arms, path counts
and per-cycle mixtures, provided the SAME state applies to both branches.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'analysis'))
from diagnose_cross_polarity_physics import GateLookup, read_pairs, deembed_branch, interpolate_branch
from audit_transistor_gate_interpolation import METHODS, gate_interpolate
from compare_transport_candidates import write_csv
from vulcan2d.transistor import load_default_lookup, load_reverse_lookup
from vulcan2d.transport_candidates import TransportAssumptions, Q, KB, LN10

GRID = np.round(np.arange(.1, .300001, .01), 10)


def ratio_bound_decade(voltage, temperature=300.):
    v = np.asarray(voltage, float)
    if np.any(~np.isfinite(v)) or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Finite voltage and positive temperature required')
    return np.abs(v)*Q/(KB*temperature*LN10)


def log_ratio_one_level(v, energy, eta, temperature=300.):
    """Natural log reverse/forward ratio; exact independent closed form."""
    if np.any(np.asarray(v) <= 0) or np.any((np.asarray(eta) < 0) | (np.asarray(eta) > 1)):
        raise ValueError('Positive voltage and passive lever arm required')
    vt = KB*temperature/Q
    def logcosh(x):return np.logaddexp(x, -x)-np.log(2.)
    common = logcosh(np.asarray(v)/(2*vt))
    def logden(sign):
        return np.logaddexp(logcosh((energy+sign*(.5-eta)*v)/vt), common)
    return logden(1)-logden(-1)


class InterpolatedLookup(GateLookup):
    def __init__(self, source, gate, method):
        self.voltage = np.unique(np.concatenate([c['drain_voltage_v'] for c in source.curves()]))
        gates = source.gate_voltages
        matrix = np.array([[source.current(v, g) for v in self.voltage] for g in gates])
        self.current = gate_interpolate(gates, matrix, gate, method)


def run():
    pairs = read_pairs('1.1'); rows = []; summaries = {}
    for policy in ('mean', 'up'):
        reverse = GateLookup(load_reverse_lookup(branch=policy), 1.1)
        for method in METHODS:
            forward = InterpolatedLookup(load_default_lookup(branch=policy), 1.1, method)
            for pair in pairs:
                branches = {}
                for phase, lookup in [('set', forward), ('reset', reverse)]:
                    branches[phase] = {sign: deembed_branch(pair[phase], phase, lookup, sign)
                                       for sign in (-1, 0, 1)}
                for v in GRID:
                    values, reasons = {}, {}
                    for phase in ('set', 'reset'):
                        for sign in (-1, 0, 1):
                            item, reason = interpolate_branch(branches[phase][sign], v)
                            values[phase, sign] = None if item is None else item['current_A']
                            if reason:reasons[phase+str(sign)] = reason
                    p, n = values['set', 0], values['reset', 0]
                    upper = (values['reset', 1]/values['set', -1]
                             if values['reset', 1] and values['set', -1] else None)
                    lower = (values['reset', -1]/values['set', 1]
                             if values['reset', -1] and values['set', 1] else None)
                    b = float(ratio_bound_decade(v)); ratio = n/p if n and p else None
                    d = np.log10(ratio) if ratio is not None else None
                    row = {'cycle_id': pair['cycle_id'], 'lookup_branch': policy, 'gate_interpolation': method,
                           'hbn_voltage_V': float(v), 'ratio': ratio, 'log10_ratio': d,
                           'bound_lower': 10**(-b), 'bound_upper': 10**b,
                           'violation_decade': max(abs(d)-b, 0.) if d is not None else None,
                           'zero_reference_ratio_upper': upper, 'zero_reference_ratio_lower': lower,
                           'upper_still_below_bound': bool(upper is not None and upper < 10**(-b)),
                           'zero_reference_lower_unavailable': lower is None,
                           'counterfactual_maximum_temperature_K': float(Q*v/(KB*abs(np.log(ratio)))) if ratio and ratio != 1 else None,
                           'set_zero_reference_fraction': branches['set'][0]['zero_bias_abs_current_max_A']/p if p else None,
                           'reset_zero_reference_fraction': branches['reset'][0]['zero_bias_abs_current_max_A']/n if n else None,
                           'missing_reasons': reasons}
                    rows.append(row)
            current = [r for r in rows if r['lookup_branch'] == policy and r['gate_interpolation'] == method]
            summaries[policy+'_'+method] = {'n_rows': len(current),
                'observed_violating_cycle_ids': sorted({r['cycle_id'] for r in current if r['violation_decade'] and r['violation_decade'] > 1e-10}),
                'upper_perturbed_violating_cycle_ids': sorted({r['cycle_id'] for r in current if r['upper_still_below_bound']}),
                'at_0p1_first3': [r for r in current if r['cycle_id'] <= 3 and abs(r['hbn_voltage_V']-.1)<1e-10]}
    primary = [r for r in rows if r['lookup_branch'] == 'mean' and r['gate_interpolation'] == 'linear_current']
    if any(r['violation_decade'] is None for r in primary):raise RuntimeError('Primary ratio coverage incomplete')
    groups = {}
    for tag, test in [('all53', lambda c: True), ('first3', lambda c: c <= 3), ('cycles4to53', lambda c: c >= 4)]:
        subset = [r for r in primary if test(r['cycle_id'])]
        e = np.array([r['violation_decade'] for r in subset])
        groups[tag] = {'n_pairs': len(subset)//len(GRID),
            # Weighted orthogonal projection onto a ratio interval. Per grid
            # point w+ = .25/21, w- = .5/21; w+w-/(w++w-)=1/(6*21).
            'three_window_metric_optimistic_bound_rmse_decade': float(np.sqrt(np.mean(e*e)/6)),
            'matched_low_equal_branch_optimistic_bound_rmse_decade': float(np.sqrt(np.mean(e*e))/2),
            'n_violating_grid_points': int(sum(e > 1e-10)),
            'temperature_sensitivity': {str(t): sorted({r['cycle_id'] for r in subset
                if abs(r['log10_ratio']) > ratio_bound_decade(r['hbn_voltage_V'], t)+1e-10}) for t in (250., 300., 350.)}}
    sources = [Path(__file__), ROOT/'analysis/diagnose_cross_polarity_physics.py',
               ROOT/'analysis/audit_transistor_gate_interpolation.py', ROOT/'vulcan2d/transistor.py',
               ROOT/'vulcan2d/transport_candidates.py', ROOT/'1T1M写入.xlsx', ROOT/'1T1M擦除.xlsx',
               ROOT/'vulcan2d/data/fig2b_transistor_lookup.json', ROOT/'vulcan2d/data/bookoutput2_reverse_transistor_lookup.json']
    report = {'claim': 'Conditional class exclusion: noninteracting fixed-rate full-bias localized levels in positive parallel mixtures, same frozen state for both polarities',
              'bound': 'abs(log10(abs(I(-V))/I(+V))) <= q*abs(V)/(kB*T*ln(10))',
              'temperature_K': 300., 'groups': groups, 'processing_sensitivity': summaries,
              'first3_primary_curve': [r for r in primary if r['cycle_id'] <= 3],
              'limitations': ['Zero-reference perturbations are an illustrative readout sensitivity, not calibrated error bars or a confidence region.',
                              'First three reverse currents are small relative to the zero-reference scale; the one-sided upper check remains available even when a lower perturbation is not.',
                              'Gate interpolation alternatives share unquantified device mismatch and measurement systematic errors.',
                              'The class assumes a common frozen state, voltage-independent positive path weights/contact rates, linear passive electrostatics, independent levels, and a specified temperature.',
                              'Violation does not rule out every localized-state mechanism: voltage-dependent contacts, interacting serial networks or changing states are outside this class.',
                              'Counterfactual temperature is a consistency bound, not thermometry; temperature is not fitted.',
                              'Interpolated points are correlated; no independent-sample significance claim.'],
              'input_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    (ROOT/'analysis/parallel_rectification_bound.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    write_csv(ROOT/'analysis/parallel_rectification_bound.csv', rows)
    print(json.dumps(groups, indent=2))
    print('0.1-V primary first3:', [(r['cycle_id'], r['ratio'], r['zero_reference_ratio_upper'], r['bound_lower'])
                                    for r in primary if r['cycle_id'] <= 3 and r['hbn_voltage_V'] == .1])


if __name__ == '__main__':run()
