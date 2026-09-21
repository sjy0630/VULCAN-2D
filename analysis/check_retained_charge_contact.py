"""Algebraic frozen-current compensation and conditional history sensitivity."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from scipy.special import log_expit
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'analysis'))
from compare_activated_configuration import assemble, lookups, scores
from vulcan2d.retained_charge_configuration import trajectory, surfaces, LO, HI


def ratio34(data, current, window):
    means = []
    for c in (3, 4):
        mask = (data['cycle'] == c) & (data['window'] == window) & (abs(data['voltage']) >= .1-1e-10)
        means.append(np.mean(np.log10(current[mask])))
    return float(10**(means[1]-means[0]))


def run():
    d = assemble(); lookup = lookups(); results = {}; hashes = {}
    for split in ('all_descriptive', 'chronological'):
        path = ROOT/'analysis'/f'retained_charge_configuration_{split}_single.json'
        source = json.loads(path.read_text()); assert source['status'] == 'complete'
        for name, sha in source['input_sha256'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == sha
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        shared = np.array(source['shared_parameters']); p = np.array(source['parameters'])
        v = np.linspace(-5., 5., 2001); reference = surfaces(v, shared, p)
        records = []
        for gl in sorted(set([-12., -6., -3., 0., 3., 6., 12., shared[4]])):
            altered_s = shared.copy(); altered_s[4] = gl
            altered_p = p.copy(); altered_p[0] += (log_expit(shared[4])-log_expit(gl))/np.log(10.)
            surf = surfaces(v, altered_s, altered_p)
            mask = reference['current'] > 0
            invariance = float(np.max(abs(np.log10(surf['current'][mask]/reference['current'][mask]))))
            assert invariance < 1e-12
            r = trajectory(d['voltage'], altered_s, altered_p, *lookup, **source['settings'])
            records.append({'ln_left_over_right': float(gl), 'log10_amplitude': float(altered_p[0]),
                            'amplitude_within_primary_bounds': bool(LO[0] <= altered_p[0] <= HI[0]),
                            'maximum_frozen_current_log_change': invariance,
                            'maximum_conditional_n_probability_change': float(np.max(abs(
                                np.exp(surf['n_log_probability'])-np.exp(reference['n_log_probability'])))),
                            'maximum_log_rate_change': float(np.max(abs(surf['log_rates']-reference['log_rates']))),
                            'full_scores': scores(d, r['current_A']),
                            'target_scores': scores(d, r['current_A'], source['target_ids']),
                            'maximum_kcl_relative_error': r['maximum_kcl_relative_error'],
                            'SET_return_4_over_3': ratio34(d, r['current_A'], 2),
                            'RESET_outward_4_over_3': ratio34(d, r['current_A'], 4)})
            print(split, gl, records[-1]['full_scores']['rmse_decade'], records[-1]['RESET_outward_4_over_3'], flush=True)
        results[split] = records
    for p in [Path(__file__), ROOT/'tests/test_retained_charge_contact.py',
              ROOT/'analysis/RETAINED_CHARGE_SENSITIVITY_DESIGN_2026-09-21.md']:
        hashes[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    out = {'scope': 'Fixed-parameter algebraic sensitivity, no refitting or model-family exclusion',
           'observed_SET_return_4_over_3': ratio34(d, d['observed'], 2),
           'observed_RESET_outward_4_over_3': ratio34(d, d['observed'], 4),
           'input_sha256': hashes, 'candidates': results}
    (ROOT/'analysis/retained_charge_configuration_contact_sensitivity.json').write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__': run()
