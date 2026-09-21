"""Check declared instantaneous-readout starts before fitting."""
from pathlib import Path
import hashlib
import json
import numpy as np
from fit_stochastic_readout_support import initializations, ROOT
from compare_activated_configuration import assemble, lookups
from vulcan2d.finite_readout_support import readout


def run():
    d = assemble(); voltage = np.unique(d['voltage'][d['weights'] > 0]); lookup = lookups(); rows = []
    for split in ('all_descriptive', 'chronological'):
        for states in ('two', 'four'):
            for index, seed in enumerate(initializations(split)):
                p = np.array(seed)
                if states == 'two': p[5] = 0.
                main = readout(voltage, p, np.eye(4), *lookup)
                fine = readout(voltage, p, np.eye(4), *lookup, panels=24, table_step=.0025)
                error = float(np.max(abs(np.log10(fine['current_A']/main['current_A']))))
                rows.append({'split': split, 'states': states, 'start': index, 'maximum_log_change': error,
                             'maximum_kcl_relative_error': max(main['maximum_kcl_relative_error'], fine['maximum_kcl_relative_error'])})
                assert error < .005 and rows[-1]['maximum_kcl_relative_error'] < 1e-8
    paths = [Path(__file__), ROOT/'analysis/fit_stochastic_readout_support.py', ROOT/'vulcan2d/finite_readout_support.py',
             ROOT/'vulcan2d/_finite_readout_core.cpp', ROOT/'vulcan2d/_activated_configuration_core.cpp']
    output = {'starts': rows, 'input_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (ROOT/'analysis/stochastic_readout_prefit.json').write_text(json.dumps(output, indent=2)+'\n')
    print('Twenty starts passed; maximum change', max(r['maximum_log_change'] for r in rows))


if __name__ == '__main__': run()
