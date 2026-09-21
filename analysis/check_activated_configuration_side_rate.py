"""Conditional sensitivity of selected fits to unmeasured fast side rate.

No refitting and no physical seconds are inferred. The conditionally stationary
electronic approximation is retained; this is not a global profile likelihood.
"""
from pathlib import Path
import json
import hashlib
import numpy as np
from compare_activated_configuration import ROOT, assemble, lookups, scores
from vulcan2d import activated_configuration as ac


def run():
    d = assemble(); lookup = lookups(); base_surfaces = ac.surfaces; rows = {}; hashes = {}
    for split in ('all_descriptive', 'chronological'):
        for saddle in ('single', 'polarizable'):
            path = ROOT/'analysis'/f'activated_configuration_{split}_{saddle}_reconciled.json'
            if not path.exists(): continue
            r = json.loads(path.read_text()); assert r['status'] == 'complete'
            hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            baseline = np.load(path.with_suffix('.npz'))['current']; result = []
            for gt in (-12., -6., 0., 6., 12.):
                def modified(*args, **kwargs): return base_surfaces(*args, **kwargs, log_side_rate=gt)
                ac.surfaces = modified
                try:
                    predicted = ac.trajectory(d['voltage'], r['shared_parameters'], r['parameters'], *lookup, **r['settings'])
                finally: ac.surfaces = base_surfaces
                m = d['weights'] > 0; diff = np.log10(predicted['current_A'][m]/baseline[m]); w = d['weights'][m]
                early = []
                for c in (3, 4):
                    sample = (d['cycle'] == c) & (d['window'] == 4) & (abs(d['voltage']) >= .10-1e-10)
                    early.append(np.mean(np.log10(predicted['current_A'][sample])))
                result.append({'ln_side_over_right_rate': gt, 'refitted': False,
                               'maximum_log_current_change': float(abs(diff).max()),
                               'weighted_log_current_change': float(np.sqrt(np.dot(w, diff*diff)/w.sum())),
                               'full_scores': scores(d, predicted['current_A']),
                               'target_scores': scores(d, predicted['current_A'], r['target_ids']),
                               'predicted_R4_over_R3_low_bias': float(10**(early[1]-early[0])),
                               'maximum_kcl_relative_error': predicted['maximum_kcl_relative_error']})
            rows[path.stem] = result
            print(path.stem, [(x['ln_side_over_right_rate'], x['maximum_log_current_change']) for x in result], flush=True)
    for p in [Path(__file__), ROOT/'vulcan2d/activated_configuration.py', ROOT/'vulcan2d/_activated_configuration_core.cpp']:
        hashes[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    output = ROOT/'analysis/activated_configuration_side_rate_sensitivity.json'
    output.write_text(json.dumps({'scope': __doc__, 'input_sha256': hashes, 'candidates': rows}, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__': run()
