"""Post-hoc grouped sensitivity at the passive single-reservoir limit a=1.

The limiting case was chosen after inspecting primary results and the ratio
bound. Shared geometry is still trained with each target pair excluded.
This does not replace primary scores or constitute an external blind test.
"""
from pathlib import Path
import json
import sys
import hashlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'analysis'))
from compare_parallel_paths import assemble, VOLTAGE, FOLDS
from compare_side_charge_readout import STARTS, fit, scores
from profile_side_charge_lever import fit_profile
from vulcan2d.side_charge import predict


def run():
    source = ROOT/'analysis/side_charge_readout.json'
    main = json.loads(source.read_text())
    digest = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for p, sha in main['input_sha256'].items():
        if digest(ROOT/p) != sha:
            raise RuntimeError('Executed input changed: '+p)
    ids, y, weights, _ = assemble(); positive = VOLTAGE > 0
    splits = [(f'fold_{lo}_{hi}', (ids < lo) | (ids > hi), (ids >= lo) & (ids <= hi)) for lo, hi in FOLDS]
    splits += [('chronological', ids <= 35, ids > 35)]
    files = [Path(__file__), ROOT/'analysis/profile_side_charge_lever.py', source]+[ROOT/p for p in main['input_sha256']]
    result = {'scope':'Post-hoc a=1 grouped sensitivity; data-informed limiting candidate, not a replacement of primary estimates',
              'splits':{}, 'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in dict.fromkeys(files)}}
    output = ROOT/'analysis/side_charge_maximal_gate.json'
    pooled = np.empty_like(y)
    for label, train, target in splits:
        fitted = fit_profile(y[train], weights[train], 1., STARTS[0])
        identified = fit(VOLTAGE[positive], y[target][:, positive], weights[target][:, positive], 'equilibrium',
                         fixed_shared=fitted['shared_parameters'])
        prediction = predict(VOLTAGE, fitted['shared_parameters'], identified['states'])
        refined = predict(VOLTAGE, fitted['shared_parameters'], identified['states'], order=48)
        result['splits'][label] = {'training_ids':ids[train].tolist(), 'target_ids':ids[target].tolist(),
                                  'fit':fitted, 'identification':identified, 'predicted_log10_A':prediction.tolist(),
                                  'scores':scores(y[target], prediction, weights[target], ids[target]),
                                  'maximum_24_48_difference_decade':float(np.max(np.abs(refined-prediction)))}
        if label.startswith('fold_'):
            pooled[target] = prediction
        output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(label, 'training',fitted['rmse_decade'],'forecast',result['splits'][label]['scores'], flush=True)
    result['five_fold_scores'] = scores(y, pooled, weights, ids)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('Five-fold',result['five_fold_scores'], flush=True)


if __name__ == '__main__':
    run()
