"""Forward circuit verification of saved held-out RESET histories, no refits."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'analysis'))
from compare_reset_histories import assemble, score, summarize
from compare_transport_candidates import write_csv
from vulcan2d.reset_history import terminal_trajectory


def run():
    source = ROOT/'analysis/reset_history_comparison.json'; fitreport = json.loads(source.read_text())
    data, unused, lookup = assemble(); scores, diagnostics = [], []
    for name, candidate in fitreport['candidates'].items():
        for fold in candidate['folds']:
            ids = fold['target_predictions']['RESET_outward_conditioned']['cycle_ids']; index = np.array(ids)-1
            protocols = list(fold['target_predictions'])
            local = np.concatenate([fold['target_predictions'][p]['local_parameters'] for p in protocols])
            va = np.tile(data['vapp'][index], (len(protocols), 1))
            shared = fold['fit']['shared_parameters']
            a = terminal_trajectory(va, name, shared, local, lookup.voltage, lookup.current, substeps=2)
            record = None
            for substeps in (4, 8, 16, 32):
                b = terminal_trajectory(va, name, shared, local, lookup.voltage, lookup.current, substeps=substeps)
                mask = np.tile(data['weights'][index] > 0, (len(protocols), 1))
                delta = np.abs(np.log10(a['current_a'][mask]/b['current_a'][mask]))
                record = {'candidate': name, 'split': fold['split'], 'substeps': substeps,
                          'maximum_refinement_difference_decade': float(delta.max()),
                          'rms_refinement_difference_decade': float(np.sqrt(np.mean(delta**2))),
                          'refinement_passed': bool(delta.max() < .002),
                          **{k: v for k, v in b.items() if not isinstance(v, np.ndarray)}}
                if record['refinement_passed']:break
                a = b
            diagnostics.append(record)
            with np.errstate(divide='ignore'):prediction = np.log10(b['current_a'])
            for pi, protocol in enumerate(protocols):
                for row, k in enumerate(index):
                    scores.append(score(k+1, data['y'][k], prediction[pi*len(index)+row], data['weights'][k],
                                        data['vapp'][k], data['vh'][k], name, protocol, fold['split'], 'predicted_terminal'))
            print(name, fold['split'], 'substeps', record['substeps'], 'maximum difference', round(record['maximum_refinement_difference_decade'], 7), flush=True)
            save(scores, diagnostics, source)


def save(scores, diagnostics, source):
    groups = {}
    for name in sorted({r['candidate'] for r in scores}):
        groups[name] = {}
        for tag, keep in [('all53', lambda i:True), ('first3', lambda i:i <= 3), ('cycles4to53', lambda i:i >= 4)]:
            groups[name][tag] = {}
            for protocol in sorted({r['protocol'] for r in scores}):
                for split in ('five_fold', 'chronological'):
                    rows = [r for r in scores if r['candidate'] == name and r['protocol'] == protocol and keep(r['cycle_id'])
                            and (r['split'].startswith('fold') if split == 'five_fold' else r['split'] == split)]
                    if rows:groups[name][tag][split+'__'+protocol] = summarize(rows)
    paths = [Path(__file__), source, ROOT/'vulcan2d/reset_history.py', ROOT/'analysis/compare_reset_histories.py',
             ROOT/'vulcan2d/transistor.py', ROOT/'vulcan2d/data/bookoutput2_reverse_transistor_lookup.json', ROOT/'1T1M写入.xlsx', ROOT/'1T1M擦除.xlsx']
    result = {'scope': 'Forward predictions at supplied terminal voltage, saved training/conditioning parameters, no target-current input to circuit trajectory',
              'n_score_rows': len(scores), 'refinement_acceptance_max_decade': .002, 'diagnostics': diagnostics, 'groups': groups,
              'limitations': ['Parameter identification used training de-embedded coordinates; target RESET outward is allowed only in the explicitly named outward-conditioned protocol.',
                              'SET handoff maps use observed target SET and preceding SET integrals but no target RESET values.',
                              'Numerical refinement is not physical validation; H/C are still effective transport shapes.',
                              'Unconverged/refinement-failed cases remain visible and cannot be claimed as validated predictions.'],
              'input_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (ROOT/'analysis/reset_history_terminal.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    write_csv(ROOT/'analysis/reset_history_terminal_scores.csv', scores)


if __name__ == '__main__':run()
