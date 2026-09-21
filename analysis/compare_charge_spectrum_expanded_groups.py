"""Post-hoc wider-prior grouped sensitivity after the all-data boundary check.

No all-data parameters initialize a held-out fold. Reuse only the matching
training subset's narrower-spectrum solution and fixed declared starts.
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
    sources = [ROOT/'analysis'/p for p in ['charge_spectrum_comparison.json', 'charge_spectrum_expanded_prior.json']]
    narrow, expanded = [json.loads(p.read_text()) for p in sources]
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for report in (narrow, expanded):
        for p, sha in report['input_sha256'].items(): assert digest(ROOT/p) == sha
    zero = json.loads((ROOT/'analysis/coupled_charge_comparison.json').read_text())
    comparison.LO = np.array(expanded['bounds'][0]); comparison.HI = np.array(expanded['bounds'][1])
    comparison.shape_and_gradient = partial(shape_and_gradient, panels=16)
    ids, y, w, _ = comparison.assemble()
    files = [Path(__file__)]+sources+[ROOT/p for r in (narrow, expanded) for p in r['input_sha256']]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in dict.fromkeys(files)}
    path = ROOT/'analysis/charge_spectrum_expanded_groups.json'
    result = {'scope': 'Post-hoc wider-prior whole-pair sensitivity; not an untouched external validation',
              'status': 'running', 'bounds': expanded['bounds'], 'quadrature': expanded['quadrature'],
              'input_sha256': hashes, 'splits': {}}
    if path.exists():
        result = json.loads(path.read_text()); assert result['input_sha256'] == hashes
    def save(): path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for label, old in narrow['splits'].items():
        if label == 'all_descriptive': continue
        train, target = np.array(old['training_ids'])-1, np.array(old['target_ids'])-1
        assert not set(train) & set(target)
        row = result['splits'].setdefault(label, {'training_ids': old['training_ids'], 'target_ids': old['target_ids'], 'targets': {}})
        if 'fit' not in row:
            comparison.STARTS = (old['fit']['shared_parameters'], [.6, .45, .55, .2, -5., .5],
                                 [.8, .4, .5, .3, -5., .7], [-.3, .4, .5, .3, -2., .6])
            row['initial_shared_vectors'] = [list(p) for p in comparison.STARTS]
            row['initialization_scope'] = 'First start from this exact training subset; others fixed in this script; plus same-training zero-width solution'
            print('Expanded grouped fit', label, flush=True)
            row['fit'] = comparison.fit(comparison.VOLTAGE, y[train], w[train], zero_width_fit=zero['splits'][label]['fit']); save()
        fit = row['fit']; p = predict(comparison.VOLTAGE, fit['shared_parameters'], fit['states'], panels=16)
        row['training_scores'] = comparison.scores(y[train], p, w[train], ids[train])
        for protocol, mask in [('SET_only', comparison.VOLTAGE > 0), ('joint_descriptive', np.ones(len(comparison.VOLTAGE), bool))]:
            if protocol in row['targets']: continue
            local = comparison.fit(comparison.VOLTAGE[mask], y[target][:, mask], w[target][:, mask], fixed_shared=fit['shared_parameters'])
            p = predict(comparison.VOLTAGE, fit['shared_parameters'], local['states'], panels=16)
            row['targets'][protocol] = {'identification': local, 'predicted_log10_A': p.tolist(),
                                       'scores': comparison.scores(y[target], p, w[target], ids[target])}; save()
            print(label, protocol, row['targets'][protocol]['scores'], flush=True)
        save()
    for protocol in ('SET_only', 'joint_descriptive'):
        pool = np.empty_like(y)
        for label, row in result['splits'].items():
            if label.startswith('fold_'): pool[np.array(row['target_ids'])-1] = row['targets'][protocol]['predicted_log10_A']
        result[protocol+'_five_fold_scores'] = comparison.scores(y, pool, w, ids)
    result['status'] = 'complete'; save(); print(path, flush=True)


if __name__ == '__main__': run()
