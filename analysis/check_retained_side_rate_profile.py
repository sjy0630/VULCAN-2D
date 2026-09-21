"""No-refit full-history profile of the side-charge exchange scale."""
from pathlib import Path
import json
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'analysis'))
from fit_retained_transport import digest
from compare_activated_configuration import assemble, lookups, scores
from check_retained_charge_contact import ratio34
from vulcan2d.retained_charge_configuration import trajectory


def run():
    d = assemble(); lookup = lookups(); rows = {}; arrays = {}; inputs = [Path(__file__),
        ROOT/'analysis/RETAINED_SIDE_RATE_PROFILE_DESIGN_2026-09-21.md']
    def feature(y, cycle, window):
        m = (d['cycle'] == cycle) & (d['window'] == window) & (abs(d['voltage']) >= .1-1e-10)
        return float(10**np.mean(np.log10(y[m])))
    observed = {str(c): {name:feature(d['observed'],c,wi) for name,wi in [('SET_return',2),('RESET_outward',4)]} for c in (3,4)}
    for split in ('all_descriptive','chronological'):
        path = ROOT/'analysis'/f'retained_charge_configuration_{split}_single_contact.json'
        source = json.loads(path.read_text()); inputs.append(path); rows[split] = []
        for name, expected in source['input_sha256'].items(): assert digest(ROOT/name) == expected, name
        for i, ellm in enumerate(sorted(set([-12.,-6.,-3.,0.,3.,6.,9.,12.,source['parameters'][8]]))):
            p = np.array(source['parameters']); p[8] = ellm
            r = trajectory(d['voltage'],source['shared_parameters'],p,*lookup,**source['settings'])
            entry = {'ellm':ellm,'full_scores':scores(d,r['current_A']),
                     'training_scores':scores(d,r['current_A'],source['training_ids']),
                     'SET_return_4_over_3':ratio34(d,r['current_A'],2),
                     'RESET_outward_4_over_3':ratio34(d,r['current_A'],4),
                     'maximum_kcl_relative_error':r['maximum_kcl_relative_error'],
                     'cycle_features':{str(c):{name:feature(r['current_A'],c,wi) for name,wi in [('SET_return',2),('RESET_outward',4)]} for c in (3,4)}}
            assert r['maximum_kcl_relative_error'] < 1e-8
            rows[split].append(entry); arrays[f'{split}_{i}_current_A'] = r['current_A']
            print(split,ellm,entry['training_scores']['rmse_decade'],entry['cycle_features']['3'],flush=True)
    destination = ROOT/'analysis/retained_side_rate_profile.npz'; np.savez_compressed(destination,**arrays)
    out = {'scope':'Exploratory fixed-parameter side-exchange profile, no rates inferred or time calibrated',
           'observed_cycle_features':observed,'candidates':rows,'arrays_sha256':digest(destination),
           'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in inputs}}
    (ROOT/'analysis/retained_side_rate_profile.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':run()
