"""Full RESET history discrimination and target-excluded handoff information."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'analysis'))
from diagnose_cross_polarity_physics import read_pairs, GateLookup
from check_scan_memory_contrast import branch_value
from compare_transport_candidates import write_csv
from vulcan2d.transistor import load_reverse_lookup
from vulcan2d.cycle_link import set_forward_exposure
from vulcan2d.reset_history import NAMES, fit_history, identify_from_outward, predict_internal, terminal_trajectory, domains

FOLDS = ((1, 11), (12, 22), (23, 33), (34, 43), (44, 53))
HANDOFFS = ('constant', 'set_only', 'set_cycle', 'set_dose')


def assemble():
    lookup = GateLookup(load_reverse_lookup(), 1.1); arrays = {k: [] for k in ('vapp', 'vh', 'y', 'weights', 'features', 'memory_contrast')}
    selections = []; cumulative = 0.
    for pair in read_pairs('1.1'):
        r = pair['reset']; va = np.abs(r['voltage']); i = np.abs(r['current']); apex = int(np.argmax(va))
        vd = np.array([lookup.inverse(float(x)) for x in i]); vh = va-vd; zero = va <= 1e-12
        if np.any((vh <= 0) & ~zero):raise RuntimeError('Invalid nonzero internal history')
        vh[zero] = 0.
        cap = np.interp(va, lookup.voltage, lookup.current)
        valid = (va >= .08-1e-10) & (i > 0) & (vd <= .5*va) & (i <= .7*cap)
        w = np.zeros(len(va)); index = np.arange(len(va)); masks = [
            (index <= apex) & (vh <= .3), (index <= apex) & (vh > .3), index > apex]
        for mass, mask in zip((.25, .25, .5), masks):
            chosen = mask & valid
            if sum(chosen) < 3:raise RuntimeError('Insufficient branch window')
            w[chosen] = mass/sum(chosen)
        positive = pair['set']; vset = np.abs(positive['voltage']); iset = np.abs(positive['current'])
        start = len(vset)-1-int(np.argmax(vset[::-1])); select = (np.arange(len(vset)) >= start) & (vset >= .1-1e-10) & (vset <= .3+1e-10)
        send = float(np.mean(np.log10(iset[select])))
        exposure = set_forward_exposure(positive['voltage'], positive['current']); cumulative += exposure
        contrast = np.mean([branch_value(r, v, False)-branch_value(r, v, True) for v in np.arange(.5, .80001, .02)])
        for key, value in [('vapp', va), ('vh', vh), ('y', np.log10(np.maximum(i, 1e-300))), ('weights', w),
                           ('features', [send, np.log10(pair['cycle_id']), np.log10(cumulative)]), ('memory_contrast', contrast)]:
            arrays[key].append(value)
        selections.append({'cycle_id': pair['cycle_id'], 'n_scored': int(sum(w > 0)), 'n_zero_bias': int(sum(zero)),
            'n_valid_low_outward': int(sum((w > 0) & masks[0])), 'n_valid_high_outward': int(sum((w > 0) & masks[1])),
            'n_valid_return': int(sum((w > 0) & masks[2])), 'set_exposure_A_V': exposure, 'cumulative_set_exposure_A_V': cumulative,
            'reset_same_polarity_log10_contrast': float(contrast), 'clock_calibration_eligible': bool(contrast > np.log10(5.))})
    return {k: np.array(v) for k, v in arrays.items()}, selections, lookup


def score(cid, y, pred, weights, vapp, vh, candidate, protocol, split, coordinate):
    valid = weights > 0
    if np.any(~np.isfinite(pred[valid])):raise RuntimeError('Nonfinite scored prediction')
    e = np.where(valid, pred-y, 0.); apex = int(np.argmax(vapp)); index = np.arange(len(y))
    masks = {'low_outward': valid & (index <= apex) & (vh <= .3),
             'high_outward': valid & (index <= apex) & (vh > .3), 'return': valid & (index > apex)}
    return {'cycle_id': int(cid), 'candidate': candidate, 'protocol': protocol, 'split': split, 'coordinate': coordinate,
            'rmse_decade': float(np.sqrt(np.sum(weights*e*e))),
            **{name+'_rmse_decade': float(np.sqrt(np.mean(e[mask]**2))) for name, mask in masks.items()},
            'bias_decade': float(np.sum(weights*e))}


def handoff(fit, training_features, clock_eligible, target_features, mode):
    cols = {'constant': [], 'set_only': [0], 'set_cycle': [0, 1], 'set_dose': [0, 2]}[mode]
    y = np.array(fit['local_parameters']); pred = []; records = []
    for output in range(3):
        mask = np.asarray(clock_eligible) if output == 2 else np.ones(len(y), bool)
        if sum(mask) < 3:raise ValueError('Insufficient identifiable training clocks')
        x = training_features[mask][:, cols]; target = target_features[:, cols]
        mean = x.mean(axis=0); scale = x.std(axis=0); scale = np.where(scale > 1e-10, scale, 1.)
        x = (x-mean)/scale; target = (target-mean)/scale
        center = y[mask, output].mean(); response = y[mask, output]-center
        coef = np.linalg.solve(x.T@x+np.eye(len(cols)), x.T@response) if cols else np.array([])
        pred.append(center+target@coef)
        records.append({'output': output, 'feature_columns': cols, 'training_center': mean.tolist(), 'training_scale': scale.tolist(),
                        'intercept': float(center), 'coefficients': coef.tolist(),
                        'training_cycle_ids': np.array(fit['training_cycle_ids'])[mask].tolist()})
    return np.array(pred).T, records


def summarize(rows):
    return {'n_pairs': len(rows), **{key: float(np.sqrt(np.mean([r[key]**2 for r in rows])))
            for key in ('rmse_decade', 'low_outward_rmse_decade', 'high_outward_rmse_decade', 'return_rmse_decade')}}


def run(quick=False, only=None, terminal=False):
    data, selection, lookup = assemble(); ids = np.arange(1, 54); scores, candidates = [], {}
    va, vh, y, w = [data[key] for key in ('vapp', 'vh', 'y', 'weights')]
    for name in NAMES if only is None else (only,):
        full = fit_history(va, vh, y, w, ids.tolist(), name)
        prediction = predict_internal(va, vh, name, full['shared_parameters'], full['local_parameters'])
        for k, cid in enumerate(ids):scores.append(score(cid, y[k], prediction[k], w[k], va[k], vh[k], name, 'joint_all_data_descriptive', 'all53', 'measured_internal'))
        item = {'all_data_fit': full, 'folds': []}; candidates[name] = item
        print(name, 'all53', round(full['rmse_decade'], 5), full['converged'],
              [(f['nfev'], round(f['rmse_decade'], 5), f['success']) for f in full['multistart']], flush=True)
        if not quick:
            splits = [('chronological', ids <= 35, ids > 35)]
            splits += [(f'fold{lo}_{hi}', (ids < lo) | (ids > hi), (ids >= lo) & (ids <= hi)) for lo, hi in FOLDS]
            for label, train, test in splits:
                fit = fit_history(va[train], vh[train], y[train], w[train], ids[train].tolist(), name)
                target_index = np.flatnonzero(test); local = []; target_diagnostics = []
                for k in target_index:
                    got = identify_from_outward(va[k], vh[k], y[k], w[k], fit, int(ids[k]))
                    local.append(got['local_parameters']); target_diagnostics.append({'cycle_id': int(ids[k]), **got})
                modes = {'RESET_outward_conditioned': np.array(local)}; maps = {}
                for mode in HANDOFFS:
                    q, record = handoff(fit, data['features'][train], data['memory_contrast'][train] > np.log10(5.), data['features'][test], mode)
                    modes['SET_handoff_'+mode] = q; maps[mode] = record
                fold = {'split': label, 'fit': fit, 'target_outward_fits': target_diagnostics, 'handoff_maps': maps,
                        'target_predictions': {}, 'terminal_diagnostics': {}}
                for protocol, q in modes.items():
                    pred = predict_internal(va[test], vh[test], name, fit['shared_parameters'], q)
                    _, _, qlo, qhi = domains(name)
                    fold['target_predictions'][protocol] = {'cycle_ids': ids[test].tolist(), 'local_parameters': q.tolist(),
                                                          'n_outside_optimization_bounds': int(sum(np.any((q < qlo) | (q > qhi), axis=1)))}
                    for row, k in enumerate(target_index):scores.append(score(ids[k], y[k], pred[row], w[k], va[k], vh[k], name, protocol, label, 'measured_internal'))
                    if terminal:
                        result = terminal_trajectory(va[test], name, fit['shared_parameters'], q, lookup.voltage, lookup.current)
                        with np.errstate(divide='ignore'):pp = np.log10(result['current_a'])
                        for row, k in enumerate(target_index):scores.append(score(ids[k], y[k], pp[row], w[k], va[k], vh[k], name, protocol, label, 'predicted_terminal'))
                        fold['terminal_diagnostics'][protocol] = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
                item['folds'].append(fold)
                print(name, label, round(fit['rmse_decade'], 5), fit['converged'], flush=True)
        item['groups'] = {}
        for group, choose in [('all53', lambda i: True), ('first3', lambda i: i <= 3), ('cycles4to53', lambda i: i >= 4)]:
            summary = {}
            selected = [r for r in scores if r['candidate'] == name and choose(r['cycle_id'])]
            for coord in sorted({r['coordinate'] for r in selected}):
                for protocol in sorted({r['protocol'] for r in selected}):
                    for split in ('all53', 'five_fold', 'chronological'):
                        rr = [r for r in selected if r['coordinate'] == coord and r['protocol'] == protocol
                              and (r['split'].startswith('fold') if split == 'five_fold' else r['split'] == split)]
                        if rr:summary[coord+'__'+split+'__'+protocol] = summarize(rr)
            item['groups'][group] = summary
        save(candidates, scores, selection, quick, only, terminal)


def save(candidates, scores, selection, quick, only, terminal):
    sources = [Path(__file__), ROOT/'vulcan2d/reset_history.py', ROOT/'analysis/RESET_HISTORY_DESIGN_2026-09-21.md',
               ROOT/'analysis/diagnose_cross_polarity_physics.py', ROOT/'analysis/check_scan_memory_contrast.py',
               ROOT/'vulcan2d/cycle_link.py', ROOT/'vulcan2d/transport_candidates.py', ROOT/'vulcan2d/transistor.py',
               ROOT/'1T1M写入.xlsx', ROOT/'1T1M擦除.xlsx', ROOT/'vulcan2d/data/bookoutput2_reverse_transistor_lookup.json']
    report = {'scope': 'Full RESET survival/state experiment; effective nuisance transport and explicit conditioning scopes',
              'quick_without_holdouts': quick, 'terminal_simulations_run': terminal, 'selection': selection, 'candidates': candidates,
              'feature_definitions': ['Mean log10 measured SET return current at applied 0.1-0.3 V', 'log10 cycle index', 'log10 cumulative preceding and current SET integral |I| dV in A V'],
              'limitations': ['This stage is not a unified microscopic or autonomous SET/RESET model.',
                              'Internal-coordinate fits use measured current to de-embed voltage; terminal simulations accept no target current.',
                              'RESET-outward-conditioned tests use the target initial/rising RESET, so do not establish SET-to-RESET handoff.',
                              'SET handoff features exclude every target RESET observation; the information map is not a microscopic law.',
                              'Rate/energy interpretation requires unrecorded scan speed and dwell times; no absolute kinetics inferred.',
                              'The HRS PF-like and connected sinh shapes are shared effective envelopes, not confirmed microscopic transport.',
                              'Weak-contrast training cycles do not identify the survival clock; they remain in amplitude fitting and in all target scores.',
                              'All data have informed model development; no external blind validation.'],
              'input_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    suffix = ('_'+only if only else '')+('_quick' if quick else '')
    out = ROOT/'analysis'/f'reset_history_comparison{suffix}.json'
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    write_csv(ROOT/'analysis'/f'reset_history_scores{suffix}.csv', scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--quick', action='store_true'); parser.add_argument('--only', choices=NAMES)
    parser.add_argument('--terminal', action='store_true'); args = parser.parse_args(); run(args.quick, args.only, args.terminal)
