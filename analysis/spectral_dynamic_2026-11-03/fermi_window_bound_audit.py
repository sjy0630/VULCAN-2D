"""Two-point analytic necessary bound; no fitting, trajectories, or spectral scan.

Reuses the declared strongest SET witness and the current measured transistor
mapping. Current uncertainty endpoints are deembedded together, not at fixed Vh.
"""
from pathlib import Path
import hashlib
import json
import math
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from vulcan2d.transistor import measured_transistor_mapping

KB_EV_PER_K = 8.617333262145e-5


def log_bound(high_v, low_v, temperature_k=300., division_max=1.):
    assert high_v > low_v > 0 and temperature_k > 0
    thermal = KB_EV_PER_K * temperature_k
    return (division_max * (high_v-low_v) / thermal
            + math.log(-math.expm1(-high_v/thermal))
            - math.log(-math.expm1(-low_v/thermal)))


def main():
    source = OUT / 'cross_gate_lowbias_robustness.json'
    payload = json.loads(source.read_text())
    groups = payload['scenarios']['records_ge4_high_current_floor']['groups']
    group = next(g for g in groups if g['gate_v'] == .9 and g['phase'] == 'set')
    witness = group['maximum_violation']
    assert witness['record_id'] == 12
    early, late = witness['witness_pair']
    assert [early['raw_index'], late['raw_index']] == [973, 983]

    mapping = measured_transistor_mapping()
    lookup = mapping._active_lookup('set')
    voltages = np.unique(np.concatenate([c['drain_voltage_v'] for c in lookup.curves()]))
    currents = np.array([mapping.current(v, .9, 'set') for v in voltages])

    def transistor_voltage(current):
        j = int(np.searchsorted(currents, current, side='left'))
        assert 0 < j < len(currents) and currents[j] > currents[j-1]
        return float(voltages[j-1] + (current-currents[j-1])
                     * (voltages[j]-voltages[j-1]) / (currents[j]-currents[j-1]))

    rows = []
    for fraction in [0., .01, .05]:
        # These ends simultaneously minimize observed ratio and maximize its
        # allowed bound because V_T(I) is increasing on both used segments.
        high_i = early['observed_abs_current_a'] / (1.+fraction)
        low_i = late['observed_abs_current_a'] * (1.+fraction)
        high_v = early['port_abs_voltage_v'] - transistor_voltage(high_i)
        low_v = late['port_abs_voltage_v'] - transistor_voltage(low_i)
        ratio = high_i / low_i
        bound = math.exp(log_bound(high_v, low_v))
        rows.append({
            'multiplicative_current_budget': fraction,
            'favorable_early_current_a': high_i,
            'favorable_late_current_a': low_i,
            'jointly_deembedded_early_voltage_v': high_v,
            'jointly_deembedded_late_voltage_v': low_v,
            'current_ratio_lower_bound': ratio,
            'all_eta_300K_fermi_window_ratio_upper_bound': bound,
            'ratio_over_allowed_bound': ratio/bound,
            'conditional_violation': ratio > bound,
        })
    assert abs(rows[0]['jointly_deembedded_early_voltage_v']
               - early['nominal_material_abs_voltage_v']) < 1e-12
    assert abs(rows[0]['jointly_deembedded_late_voltage_v']
               - late['nominal_material_abs_voltage_v']) < 1e-12
    report = {
        'status': 'analytic_two_point_necessary_bound_only',
        'gate_v': .9, 'phase': 'set', 'record_id': 12,
        'temperature_k': 300., 'allowed_voltage_division': [0., 1.],
        'observed_witness': witness['witness_pair'], 'sensitivity_rows': rows,
        'assumptions': [
            'Positive local voltages and quasistatic conductive current; fixed valid transistor mapping.',
            'Common equal reservoir temperature 300 K at both points.',
            'Nonnegative frozen transmission in the rigidly shifted energy reference.',
            'Constant linear chemical potential division a in [0,1].',
            'Later SET state has no smaller conductive current at fixed low voltage; Lorentz Gamma nondecrease suffices.',
            'Current bands are declared sensitivity intervals, not measured confidence limits.',
        ],
        'scope': [
            'No lognormal shape parameters, Ron, gamma inversion, sweep speed, or ODE used.',
            'No search for new witness pairs; only the preselected stored strongest 0.9 V SET pair.',
            'Cached observation fields, not newly extracted workbook data; no 1.1 V holdout currents.',
            'Does not exclude arbitrary bias-dependent spectra, reverse state drift, electrothermal or charge history, or all unified models.',
        ],
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'transistor_source_sha256': hashlib.sha256((ROOT/'vulcan2d/transistor.py').read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUT/'fermi_window_bound_audit.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': report['status'], 'sensitivity_rows': rows}, indent=2))


if __name__ == '__main__':
    main()
