"""Check the unweighted 2--3 V local slope, retaining native-grid missing flags."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'analysis'))
from compare_parallel_paths import assemble, VOLTAGE
from vulcan2d.side_charge import predict


def run():
    source = ROOT/'analysis/side_charge_readout.json'; main = json.loads(source.read_text())
    digest = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for p, sha in main['input_sha256'].items():
        if digest(ROOT/p) != sha:
            raise RuntimeError('Input changed: '+p)
    ids, y, weights, _ = assemble(); rows = []
    for mode, item in main['candidates'].items():
        f = item['splits']['all_descriptive']['fit']
        predicted = predict(VOLTAGE, f['shared_parameters'], f['states'], mode=mode, order=48)
        for cid, obs, pr, w in zip(ids, y, predicted, weights):
            use = (VOLTAGE >= 2) & (VOLTAGE <= 3) & (w > 0)
            if np.sum(use) < 5:
                raise RuntimeError('Insufficient high-voltage coverage')
            x = np.log10(VOLTAGE[use]); x -= x.mean()
            rows.append({'mode':mode, 'cycle_id':int(cid), 'n_points':int(sum(use)),
                         'observed_slope':float(obs[use]@x/(x@x)),
                         'predicted_slope':float(pr[use]@x/(x@x))})
    summary = {}
    for mode in main['candidates']:
        r = [x for x in rows if x['mode'] == mode]
        summary[mode] = {key+'_q10_q50_q90':np.quantile([x[key] for x in r], [.1, .5, .9]).tolist()
                         for key in ('observed_slope', 'predicted_slope')}
        summary[mode].update(n_observed_below_one=sum(x['observed_slope'] < 1 for x in r),
                             n_predicted_below_one=sum(x['predicted_slope'] < 1 for x in r))
    files = [Path(__file__), source]+[ROOT/p for p in main['input_sha256']]
    result = {'scope':'Conditional common-grid shape check at internal 2--3 V; not independent observations or a universal model-class theorem',
              'rows':rows, 'summary':summary,
              'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in dict.fromkeys(files)}}
    (ROOT/'analysis/side_charge_high_voltage.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    run()
