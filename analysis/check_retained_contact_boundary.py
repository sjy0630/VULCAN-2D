"""Post-fit saturation check after both selected contact ratios hit +12.

This is a fixed-parameter extension to +18/+24, preserving conditional current.
It is not a profile refit and does not establish absolute contact attempts.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.special import log_expit
from compare_activated_configuration import assemble, lookups, scores
from check_retained_charge_contact import ratio34
from vulcan2d.retained_charge_configuration import trajectory
ROOT = Path(__file__).resolve().parents[1]


def run():
    d = assemble(); lookup = lookups(); rows = {}; hashes = {}; mask = d['weights'] > 0
    for split in ('all_descriptive', 'chronological'):
        path = ROOT/'analysis'/f'retained_charge_configuration_{split}_single_contact.json'
        source = json.loads(path.read_text()); assert source['status'] == 'complete'
        shared = np.array(source['shared_parameters']); p = np.array(source['parameters'])
        assert shared[4] > 11.99
        z = np.load(path.with_suffix('.npz')); records = []
        for gl in (18., 24.):
            s = shared.copy(); s[4] = gl
            adjusted = p.copy(); adjusted[0] += (log_expit(shared[4])-log_expit(gl))/np.log(10.)
            r = trajectory(d['voltage'], s, adjusted, *lookup, **source['settings'])
            delta = np.log10(r['current_A'][mask]/z['current'][mask]); w = d['weights'][mask]
            records.append({'ln_contact_ratio': gl, 'log10_amplitude': float(adjusted[0]),
                            'maximum_log_current_change': float(np.max(abs(delta))),
                            'weighted_log_current_change': float(np.sqrt(np.dot(w, delta**2)/w.sum())),
                            'full_scores': scores(d, r['current_A']),
                            'target_scores': scores(d, r['current_A'], source['target_ids']),
                            'RESET_outward_4_over_3': ratio34(d, r['current_A'], 4),
                            'maximum_kcl_relative_error': r['maximum_kcl_relative_error']})
        rows[split] = records; z.close()
        for file in (path, path.with_suffix('.npz')): hashes[str(file.relative_to(ROOT))] = hashlib.sha256(file.read_bytes()).hexdigest()
    for file in (Path(__file__), ROOT/'vulcan2d/retained_charge_configuration.py', ROOT/'analysis/check_retained_charge_contact.py'):
        hashes[str(file.relative_to(ROOT))] = hashlib.sha256(file.read_bytes()).hexdigest()
    out = {'scope': 'Post-fit, fixed-parameter contact-bound saturation; no refitting', 'input_sha256': hashes, 'candidates': rows}
    (ROOT/'analysis/retained_charge_configuration_contact_boundary.json').write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
    for split, records in rows.items():
        print(split, [(r['ln_contact_ratio'], r['maximum_log_current_change']) for r in records])


if __name__ == '__main__': run()
