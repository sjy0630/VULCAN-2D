"""Post-hoc energy-prior sensitivity; all-data description only.

The primary center hits +0.5 eV, so its residual cannot by itself reject
broader energies. Explicitly configure the unchanged fitting implementation
for one wider-prior run; primary results and grouped forecasts remain intact.
"""
from pathlib import Path
from functools import partial
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'analysis'))
import compare_charge_spectrum as comparison
from vulcan2d.charge_spectrum import shape_and_gradient, predict


def run():
    path = ROOT/'analysis/charge_spectrum_comparison.json'
    main = json.loads(path.read_text())
    assert main['status'] == 'complete'
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for p, sha in main['input_sha256'].items():
        assert digest(ROOT/p) == sha
    p = np.array(main['splits']['all_descriptive']['fit']['shared_parameters'])
    extended = p.copy(); extended[[0, 5]] *= 1.5
    # Deliberate, recorded run configuration of the frozen fitting module.
    comparison.LO = comparison.LO.copy(); comparison.HI = comparison.HI.copy()
    comparison.LO[0] = -1.; comparison.HI[0] = 1.; comparison.HI[5] = 1.
    comparison.STARTS = (p, extended, [.8, .4, .5, .3, -5., .7], [-.3, .4, .5, .3, -2., .6])
    comparison.shape_and_gradient = partial(shape_and_gradient, panels=16)
    ids, y, w, _ = comparison.assemble()
    fit = comparison.fit(comparison.VOLTAGE, y, w)
    predicted = predict(comparison.VOLTAGE, fit['shared_parameters'], fit['states'], panels=16)
    files = [Path(__file__), path]+[ROOT/p for p in main['input_sha256']]
    result = {'scope': 'Post-hoc wider energy-prior sensitivity, all data consumed; not a new grouped forecast',
              'bounds': [comparison.LO.tolist(), comparison.HI.tolist()],
              'quadrature': {'panels': 16, 'order': 12}, 'fit': fit,
              'scores': comparison.scores(y, predicted, w, ids), 'predicted_log10_A': predicted.tolist(),
              'input_sha256': {str(p.relative_to(ROOT)): digest(p) for p in dict.fromkeys(files)}}
    (ROOT/'analysis/charge_spectrum_expanded_prior.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('Wider-prior selected fit', fit['shared_parameters'], result['scores'], flush=True)


if __name__ == '__main__':
    run()
