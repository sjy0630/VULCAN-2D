"""Low-bias-constrained paired-path/state experiment; production unchanged."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'analysis'))
from compare_occupation_transport import extract_curves
from compare_transport_candidates import extract_points, write_csv
from diagnose_cross_polarity_physics import interpolate_branch
from vulcan2d.parallel_paths import NAMES, fit_shared, predict, identify_state

LOW = np.round(np.arange(.1, .300001, .01), 10)
HIGH = np.round(np.arange(.35, 3.000001, .05), 10)
VOLTAGE = np.r_[LOW, HIGH, -LOW]
WEIGHTS = np.r_[np.full(len(LOW), .25/len(LOW)), np.full(len(HIGH), .25/len(HIGH)),
                np.full(len(LOW), .5/len(LOW))]
FOLDS = ((1, 11), (12, 22), (23, 33), (34, 43), (44, 53))


def assemble():
    _, _, pos = extract_curves('linear_current', 3.1)
    native = extract_points(); ids, values, weights, selection = [], [], [], []
    for cid in range(1, 54):
        branches = {}
        for side, source in [('positive', [r for r in pos if r['cycle_id'] == cid][::-1]),
                             ('negative', [r for r in native if r['cycle_id'] == cid and r['branch'] == 'reset_initial'])]:
            points = []
            for r in source:
                positive = side == 'positive'
                valid = r['reason'] in ('', 'outside_window') if positive else r['valid']
                points.append({'raw_index': r['raw_index'], 'status': 'valid' if valid else 'excluded',
                               'hbn_voltage_V': r['hbn_V'] if positive else r['hbn_voltage_V'],
                               'perturbed_current_A': r['current_A'],
                               # Only current/adjacency is used downstream; the
                               # existing interpolator also transports these keys.
                               'transistor_voltage_V': 0., 'transistor_voltage_fraction': 0.,
                               'transistor_capacity_fraction': 0., 'inverse_condition': 0.})
            branches[side] = {'points': points}
        observed = []; missing = []
        for v in VOLTAGE:
            item, reason = interpolate_branch(branches['positive' if v > 0 else 'negative'], abs(v))
            observed.append(None if item is None else np.log10(item['current_A']))
            if reason:missing.append({'voltage_V': float(v), 'reason': reason})
        valid = np.array([x is not None for x in observed])
        if np.any(~valid & (np.abs(VOLTAGE) <= .3)):
            raise RuntimeError('Missing required low-bias data; no cycle was dropped: '+str(missing))
        cw = np.zeros(len(VOLTAGE))
        for mass, mask in [(.25, (VOLTAGE > 0) & (VOLTAGE <= .3)),
                           (.25, VOLTAGE > .3), (.5, VOLTAGE < 0)]:
            selected = valid & mask
            if sum(selected) < 5:raise RuntimeError('Insufficient branch-window coverage')
            cw[selected] = mass/sum(selected)
        selection.append({'cycle_id': cid, 'included': True, 'missing': missing})
        ids.append(cid); values.append([0. if x is None else x for x in observed]); weights.append(cw)
    return np.array(ids), np.array(values), np.array(weights), selection


def score(cid, y, prediction, name, protocol, state, split, weights):
    e = prediction-y
    masks = {'positive_low': (VOLTAGE > 0) & (VOLTAGE <= .3),
             'positive_high': VOLTAGE > .3, 'negative': VOLTAGE < 0}
    return {'cycle_id': int(cid), 'candidate': name, 'protocol': protocol, 'split': split,
            'state': list(state), 'rmse_decade': float(np.sqrt(np.sum(weights*e*e))),
            **{key+'_rmse_decade': float(np.sqrt(np.mean(e[mask & (weights > 0)]**2))) for key, mask in masks.items()},
            'negative_bias_decade': float(np.mean(e[VOLTAGE < 0]))}


def summarize(rows):
    return {key: float(np.sqrt(np.mean([r[key]**2 for r in rows])))
            for key in ('rmse_decade', 'positive_low_rmse_decade', 'positive_high_rmse_decade', 'negative_rmse_decade')}


def evaluate(fit, ids, y, weights, selection, label, rows):
    pos = VOLTAGE > 0
    diagnostics = []
    for cid, obs, w in zip(ids[selection], y[selection], weights[selection]):
        for mode, mask in [('SET_only_forecast', pos), ('joint_target_descriptive', np.ones(len(pos), bool))]:
            mask = mask & (w > 0)
            found = identify_state(VOLTAGE[mask], obs[mask], w[mask], fit, int(cid))
            predicted = predict(VOLTAGE, fit['name'], fit['parameters'], found['state'])[0]
            rows.append(score(cid, obs, predicted, fit['name'], label+'_'+mode, found['state'], label, w))
            diagnostics.append({'cycle_id': int(cid), 'protocol': mode, **found})
    return diagnostics


def run(quick=False, only=None, max_nfev=600):
    ids, y, weights, selection = assemble(); rows, candidates = [], {}
    names = NAMES if only is None else [only]
    for name in names:
        fit = fit_shared(VOLTAGE, y, weights, ids.tolist(), name, max_nfev=max_nfev)
        pred = predict(VOLTAGE, name, fit['parameters'], fit['states'])
        for cid, obs, pr, state, w in zip(ids, y, pred, fit['states'], weights):
            rows.append(score(cid, obs, pr, name, 'all53_joint_descriptive', state, 'all53', w))
        convergence = predict(VOLTAGE, name, fit['parameters'], fit['states'], nodes=128)-pred
        item = {'all_data_fit': fit, 'quadrature_64_vs_128_max_decade': float(np.max(np.abs(convergence))), 'folds': []}
        candidates[name] = item
        print(name, 'all53', round(fit['rmse_decade'], 5), 'converged', fit['converged'],
              'attempts', [(f['nfev'], round(f['rmse_decade'], 5), f['success']) for f in fit['multistart']], flush=True)
        if not quick:
            splits = [('chronological', ids <= 35, ids > 35)]
            splits += [(f'fold{lo}_{hi}', (ids < lo) | (ids > hi), (ids >= lo) & (ids <= hi)) for lo, hi in FOLDS]
            for label, train, test in splits:
                f = fit_shared(VOLTAGE, y[train], weights[train], ids[train].tolist(), name, max_nfev=max_nfev)
                diagnostics = evaluate(f, ids, y, weights, test, label, rows)
                item['folds'].append({'split': label, 'fit': f, 'targets': diagnostics})
                print(name, label, round(f['rmse_decade'], 5), 'converged', f['converged'], flush=True)
        item['groups'] = {}
        for group, test in [('all53', lambda c: True), ('first3', lambda c: c <= 3), ('cycles4to53', lambda c: c >= 4)]:
            out = {}
            for protocol in ('all53_joint_descriptive', 'five_fold_SET_only_forecast', 'five_fold_joint_target_descriptive',
                             'chronological_SET_only_forecast', 'chronological_joint_target_descriptive'):
                subset = [r for r in rows if r['candidate'] == name and test(r['cycle_id'])
                          and r['protocol'] == protocol]
                if protocol.startswith('five_fold_'):
                    suffix = protocol[len('five_fold_'):]
                    subset = [r for r in rows if r['candidate'] == name and test(r['cycle_id'])
                              and r['split'].startswith('fold') and r['protocol'].endswith(suffix)]
                if subset:out[protocol] = {'n_pairs': len(subset), **summarize(subset)}
            item['groups'][group] = out
        # Save a restartable checkpoint after each completed candidate.
        save(candidates, rows, selection, weights, quick, only)


def save(candidates, rows, selection, weights, quick, only):
    files = [Path(__file__), ROOT/'analysis/PARALLEL_PATH_DESIGN_2026-09-20.md', ROOT/'vulcan2d/parallel_paths.py',
             ROOT/'analysis/compare_occupation_transport.py', ROOT/'analysis/compare_transport_candidates.py',
             ROOT/'analysis/diagnose_cross_polarity_physics.py', ROOT/'analysis/audit_transistor_gate_interpolation.py',
             ROOT/'vulcan2d/transport_candidates.py', ROOT/'vulcan2d/transistor.py',
             ROOT/'1T1M写入.xlsx', ROOT/'1T1M擦除.xlsx', ROOT/'vulcan2d/data/fig2b_transistor_lookup.json',
             ROOT/'vulcan2d/data/bookoutput2_reverse_transistor_lookup.json']
    report = {'scope': 'Effective parallel-path hypothesis; internal branch coordinates, not a full autonomous cycle',
              'quick_without_holdouts': quick, 'voltage_V': VOLTAGE.tolist(), 'weights_by_cycle': weights.tolist(),
              'selection': selection, 'candidates': candidates,
              'input_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              'limitations': ['All data previously used in research; no external blind test.',
                              'Uniform energy bands and frozen states are assumptions, not spectroscopy.',
                              'Internal voltage and eligibility depend on observed currents.',
                              'The two candidate state dimensions are not a proof of the unique number of microscopic variables.',
                              'Chronological targets use their SET return to identify state; not autonomous history prediction.',
                              'Optimization convergence and quadrature are checked; no guarantee of global optimum.']}
    suffix = ('_'+only if only else '') + ('_quick' if quick else '')
    out = ROOT/'analysis'/f'parallel_path_comparison{suffix}.json'
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    write_csv(ROOT/'analysis'/f'parallel_path_scores{suffix}.csv', rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--quick', action='store_true')
    parser.add_argument('--only', choices=NAMES); parser.add_argument('--max-nfev', type=int, default=600)
    args = parser.parse_args(); run(args.quick, args.only, args.max_nfev)
