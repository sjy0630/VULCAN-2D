"""Refine every finite-domain configuration, including joint N=4 refits."""
from pathlib import Path
import json
import hashlib
import numpy as np
from compare_activated_configuration import assemble, lookups, scores
from check_stochastic_readout_support import ROOT, oracle, coverage
from vulcan2d.finite_readout_support import readout, fractions, conditional_currents
from vulcan2d.side_charge_circuit import load_roots


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    d = assemble(); lookup = lookups(); voltage, inverse = np.unique(d['voltage'], return_inverse=True)
    mask = d['weights'] > 0; valid = voltage != 0; rows = {}; hashes = {}; archive = {}
    for split in ('all_descriptive', 'chronological'):
        for joint in (False, True):
            name = f'stochastic_domain_support_{split}_N4' if joint else f'stochastic_readout_support_{split}_four_reconciled'
            path = ROOT/'analysis'/f'{name}.json'; source = json.loads(path.read_text()); assert source['status'] == 'complete'
            for file, sha in source['input_sha256'].items(): assert digest(ROOT/file) == sha, file
            assert digest(path.with_suffix('.npz')) == source['readout_sha256']
            for file in (path, path.with_suffix('.npz')): hashes[str(file.relative_to(ROOT))] = digest(file)
            for n in ((4,) if joint else (1, 2, 4, 8, 16)):
                pop = fractions(n); p = source['parameters']; settings = source['settings']
                main = readout(voltage, p, pop, *lookup, **settings); best, labels = oracle(d, main['current_A'], inverse)
                refinements = []; old_error = abs(np.log10(best[mask]/d['observed'][mask])); w = d['weights'][mask]
                for panels, step in ((24, .0025), (36, .00125)):
                    refined = readout(voltage, p, pop, *lookup, panels=panels, order=8, table_step=step)
                    fine, _ = oracle(d, refined['current_A'], inverse)
                    maximum = float(np.max(abs(np.log10(refined['current_A'][:, valid]/main['current_A'][:, valid]))))
                    error = abs(np.log10(fine[mask]/d['observed'][mask]))-old_error
                    weighted = float(np.sqrt(np.dot(w, error**2)/w.sum()))
                    refinements.append({'panels': panels, 'order': 8, 'table_step': step,
                                        'maximum_configuration_current_log_change': maximum,
                                        'weighted_absolute_residual_change': weighted,
                                        'maximum_absolute_residual_change': float(np.max(abs(error))),
                                        'passes': maximum < .005 and weighted < .001,
                                        'maximum_kcl_relative_error': refined['maximum_kcl_relative_error'],
                                        'training_support_scores': scores(d, fine, source['training_ids']),
                                        'target_support_scores': scores(d, fine, source['target_ids'])})
                root_errors = []
                if joint or n in (4, 16):
                    grid = np.linspace(0., 5., 1001)
                    tables = {sign: conditional_currents(sign*grid, p, panels=settings['panels'], order=8) for sign in (-1, 1)}
                    index = np.flatnonzero(valid)[np.linspace(0, np.count_nonzero(valid)-1, 8).astype(int)]
                    config_index = np.unique(np.linspace(0, len(pop)-1, 4).astype(int))
                    for c in config_index:
                        for j in index:
                            sign = 1 if voltage[j] > 0 else -1
                            roots = load_roots(abs(voltage[j]), grid, pop[c]@tables[sign], *lookup[0 if sign > 0 else 1])
                            assert len(roots) == 1 and roots[0]['stable_load_line']
                            error = abs(roots[0]['internal_voltage_V']-abs(main['internal_voltage_V'][c, j]))
                            assert error < 1e-8; root_errors.append(error)
                key = f'{split}_{"joint" if joint else "fixed"}_N{n}'
                rows[key] = {'source': str(path.relative_to(ROOT)), 'domains': n, 'configuration_count': len(pop),
                             'refitted_for_domain_count': joint, 'parameters': p,
                             'training_support_scores': scores(d, best, source['training_ids']),
                             'target_support_scores': scores(d, best, source['target_ids']), 'full_support_scores': scores(d, best),
                             'training_coverage': coverage(d, best, source['training_ids']), 'target_coverage': coverage(d, best, source['target_ids']),
                             'refinements': refinements, 'all_refinements_pass': all(r['passes'] for r in refinements),
                             'independent_mixed_root_checks': len(root_errors),
                             'maximum_mixed_root_voltage_error_V': max(root_errors, default=0.)}
                print(key, rows[key]['full_support_scores']['rmse_decade'], [r['maximum_configuration_current_log_change'] for r in refinements], flush=True)
                if joint:
                    stored = np.load(path.with_suffix('.npz'))
                    np.testing.assert_array_equal(labels, stored['oracle_index']); np.testing.assert_allclose(best, stored['current'], rtol=0, atol=0)
                    actual = source['starts'][source['selected_start']]
                    assert actual['converged'] and abs(actual['rmse_decade']-rows[key]['training_support_scores']['rmse_decade']) < 1e-10
                    archive[split+'_current'] = best; archive[split+'_labels'] = labels
                    archive[split+'_configuration_current'] = main['current_A']; archive[split+'_fractions'] = pop
                    stored.close()
    for file in (Path(__file__), ROOT/'analysis/check_stochastic_readout_support.py', ROOT/'tests/test_finite_readout_support.py'):
        hashes[str(file.relative_to(ROOT))] = digest(file)
    path = ROOT/'analysis/stochastic_domain_refinement.json'
    np.savez_compressed(path.with_suffix('.npz'), **archive, voltage=d['voltage'], observed=d['observed'], cycle=d['cycle'],
                        window=d['window'], weights=d['weights'], unique_voltage=voltage, voltage_inverse=inverse)
    result = {'scope': 'Fixed-current support only; oracle labels are not a kinetic path', 'input_sha256': hashes,
              'candidates': rows, 'arrays_sha256': digest(path.with_suffix('.npz'))}
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__': run()
