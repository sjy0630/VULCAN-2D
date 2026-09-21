"""Recheck archived height fits and conditional WKB evidence without refitting.

Replays the forward circuit and all saved mesh refinements. WKB inequalities
are checked for every stored case; the 20 largest quadrature-error cases are
independently re-integrated. This is a numerical/provenance audit, not new
experimental evidence or proof of a microscopic mechanism.
"""
from pathlib import Path
import hashlib
import json
import platform
import re
import sys

import numpy as np
import scipy
from scipy.constants import Boltzmann, elementary_charge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analysis'))
from compare_barrier_states import assemble, metrics, FOLDS
from compare_barrier_height_states import WIDTHS, simulate
from vulcan2d.barrier_height_state import build_height_tables
from vulcan2d.wkb_polarity import log_current_integral


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected, tolerance=1e-10):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify_old_manifests():
    results = []
    for name in ('barrier_state', 'finite_path', 'bidirectional'):
        path = ROOT / f'analysis/{name}_artifact_manifest.json'
        manifest = json.loads(path.read_text())
        count = 0
        for key, base in (('artifacts', ROOT), ('external_workspace_logs', ROOT.parent)):
            for p, record in manifest.get(key, {}).items():
                require(digest(base / p) == record['sha256'], f'Old artifact changed: {p}')
                require((base / p).stat().st_size == record['bytes'], f'Old artifact size changed: {p}')
                count += 1
        results.append({'manifest':path.name, 'sha256':digest(path), 'verified_artifacts':count})
    return results


def run():
    paths = [ROOT / 'analysis' / name for name in (
        'barrier_height_comparison.json', 'barrier_height_validation.json',
        'wkb_frozen_polarity_bound.json')]
    main, validation, wkb = reports = [json.loads(p.read_text()) for p in paths]
    provenance = []
    for path, report in zip(paths, reports):
        for name, sha in report['input_sha256'].items():
            require(digest(ROOT / name) == sha, f'{path.name}: input changed: {name}')
        provenance.append({'report':path.name, 'verified_input_count':len(report['input_sha256'])})
    old = verify_old_manifests()
    require(main['status'] == 'complete', 'Main comparison incomplete')
    require(main['fixed_width_profile_nm'] == list(WIDTHS), 'Width profile changed')
    va, observed, zeros, branches, lookup = assemble()
    np.testing.assert_array_equal(main['voltage_window_V'], va)
    np.testing.assert_array_equal(main['observed_current_A'], observed)
    require(main['branches'] == branches and len(branches) == 202, 'Branch coverage differs')
    require(observed.size == 4646, 'Observed point coverage differs')
    capacity = np.interp(va, lookup.voltage, lookup.current)
    require(int(np.sum(observed > capacity)) == 49, 'Capacity flag count changed')
    require(int(np.sum(observed <= 3*zeros[:, None])) == 857, 'Signal flag count changed')
    labels = {'all_descriptive'} | {f'fold_{lo}_{hi}' for lo, hi in FOLDS}
    require(set(main['splits']) == labels, 'Missing or extra splits')
    profiles = {f'width_{g:g}' for g in WIDTHS}
    tables = build_height_tables(WIDTHS)
    quad = build_height_tables(WIDTHS, order=48)
    finer = build_height_tables(WIDTHS, voltage_step=.00125, height_step_ev=.005, order=48)
    refinements = {(r['split'], r['profile']):r for r in validation['training_profile_refinements']}
    require(len(validation['training_profile_refinements']) == 54 and
            set(refinements) == {(s, p) for s in labels for p in profiles}, 'Training refinements incomplete')
    target_refinements = {r['split']:r for r in validation['target_prediction_refinements']}
    require(len(validation['target_prediction_refinements']) == 5 and
            set(target_refinements) == labels-{'all_descriptive'}, 'Target refinements incomplete')
    ids = np.array([b['cycle_id'] for b in branches])
    tested, starts, selected_widths = [], [], {}
    fit_count = 0
    errors, reliable_errors = [], []
    phase_errors = {'outward':[], 'return':[]}
    max_rescore, max_kcl, max_quad, max_grid, max_target = 0., 0., 0., 0., 0.

    def forward(fit, table, voltage=va):
        nonlocal max_kcl
        out = simulate(voltage, fit['log10_prefactor_S'], fit['local_height_ev'], table, lookup)
        cap = np.interp(voltage, lookup.voltage, lookup.current)
        require(np.all(out['current_a'] > 0) and np.all(out['current_a'] <= cap*(1+1e-9)),
                'Invalid or above-capacity forward prediction')
        internal = out['internal_voltage_v']
        require(np.all(internal >= 0) and np.all(internal <= voltage), 'Nonpassive internal voltage')
        transistor = np.interp(voltage-internal, lookup.voltage, lookup.current)
        kcl = float(np.max(np.abs(transistor-out['current_a'])/out['current_a']))
        max_kcl = max(max_kcl, kcl, out['max_kcl_relative_error'])
        return out

    for label, split in main['splits'].items():
        training, target = [np.array(split[key]) for key in ('training_branch_indices', 'target_branch_indices')]
        require(set(split['fits']) == profiles, 'Incomplete geometry profile')
        if label == 'all_descriptive':
            np.testing.assert_array_equal(training, np.arange(202))
            np.testing.assert_array_equal(target, np.arange(202))
        else:
            lo, hi = map(int, label.split('_')[1:])
            np.testing.assert_array_equal(training, np.flatnonzero((ids < lo) | (ids > hi)))
            np.testing.assert_array_equal(target, np.flatnonzero((ids >= lo) & (ids <= hi)))
            require(not set(ids[training]) & set(ids[target]), 'Same trace in training and test')
            tested.extend(target.tolist())
        chosen = min(split['fits'], key=lambda k:split['fits'][k]['rmse_decade'])
        require(chosen == split['selected_profile_key'], 'Geometry not selected on training score')
        selected = split['fits'][chosen]
        selected_widths[label] = selected['fixed_width_nm']
        for key, fitted in split['fits'].items():
            g = fitted['fixed_width_nm']
            require(key == f'width_{g:g}', 'Geometry metadata mismatch')
            require(fitted['converged'] and len(fitted['local_height_ev']) == len(training), 'Training fit incomplete')
            require(len(fitted['multistart']) == 2 and all(a['success'] for a in fitted['multistart']),
                    'Missing or unconverged training starts')
            best = min(fitted['multistart'], key=lambda a:a['rmse_decade'])
            close(best['parameters'], [fitted['log10_prefactor_S']]+fitted['local_height_ev'])
            starts.extend(fitted['multistart']); fit_count += 1
            out = forward(fitted, tables[g])
            score = metrics(observed[training], out['current_a'], zeros[training], va, lookup)
            max_rescore = max(max_rescore, abs(score['rmse_decade']-fitted['rmse_decade']))
            close(score['rmse_decade'], fitted['rmse_decade'])
            q, f = forward(fitted, quad[g]), forward(fitted, finer[g])
            dq = float(np.max(np.abs(np.log10(q['current_a']/out['current_a']))))
            df = float(np.max(np.abs(np.log10(f['current_a']/out['current_a']))))
            saved = refinements[label, key]
            close(dq, saved['quadrature_maximum_difference_decade'])
            close(df, saved['combined_refinement_maximum_difference_decade'])
            max_quad, max_grid = max(max_quad, dq), max(max_grid, df)
        saved = split['prediction']; local = saved['local_identification']
        require(local['converged'] and len(local['local_height_ev']) == len(target), 'Target identification incomplete')
        require(local['log10_prefactor_S'] == selected['log10_prefactor_S'] and
                local['fixed_width_nm'] == selected['fixed_width_nm'], 'Shared parameter changed on target')
        out = forward(local, tables[selected['fixed_width_nm']])
        np.testing.assert_allclose(saved['predicted_current_A'], out['current_a'], rtol=2e-12, atol=0)
        close(saved['predicted_internal_voltage_V'], out['internal_voltage_v'], 1e-12)
        for name, mask in (('all_voltage_score', np.ones(len(va), bool)), ('higher_voltage_score', va > .16+1e-10)):
            score = metrics(observed[target][:, mask], out['current_a'][:, mask], zeros[target], va[mask], lookup)
            require(set(score) == set(saved[name]), 'Score fields changed')
            for key, value in score.items():
                close(value, saved[name][key])
        if label != 'all_descriptive':
            low = va <= .16+1e-10
            conditioning_rmse = float(np.sqrt(np.mean(np.log10(out['current_a'][:, low]/observed[target][:, low])**2)))
            close(conditioning_rmse, local['rmse_decade'])
            require(len(local['multistart']) == 1 and local['multistart'][0]['success'], 'Target fit failed')
            close(local['multistart'][0]['parameters'], local['local_height_ev'])
            fine = forward(local, finer[selected['fixed_width_nm']])
            difference = float(np.max(np.abs(np.log10(fine['current_a']/out['current_a']))))
            close(difference, target_refinements[label]['combined_refinement_maximum_difference_decade'])
            max_target = max(max_target, difference)
            error = np.log10(out['current_a'][:, ~low]/observed[target][:, ~low])
            errors.extend(error.ravel())
            reliable = (observed[target][:, ~low] > 3*zeros[target, None]) & (observed[target][:, ~low] <= capacity[~low])
            reliable_errors.extend(error[reliable])
            for j, index in enumerate(target):
                phase_errors[branches[index]['branch']].extend(error[j])
    require(sorted(tested) == list(range(202)) and fit_count == 54 and len(starts) == 108,
            'Grouped coverage or shared fit count failed')
    pooled = {'n_scored_points':len(errors), 'rmse_decade':float(np.sqrt(np.mean(np.square(errors)))),
              'n_high_signal_capacity_compatible_points':len(reliable_errors),
              'high_signal_capacity_compatible_rmse_decade':float(np.sqrt(np.mean(np.square(reliable_errors)))),
              **{p+'_rmse_decade':float(np.sqrt(np.mean(np.square(e)))) for p, e in phase_errors.items()}}
    require(pooled['n_scored_points'] == 2828 and pooled['n_high_signal_capacity_compatible_points'] == 2355,
            'Held-out score coverage changed')
    for key, value in pooled.items():
        close(value, validation['grouped_higher_voltage_scores'][key])
    require(max_quad < 1e-5 and max(max_grid, max_target) < 1e-3 and max_kcl < 1e-8,
            'Numerical refinement or circuit tolerance failed')
    # Preserve the declared sensitivity results, without using them to alter folds.
    require(len(validation['posthoc_parameter_checks']) == 7, 'Incomplete sensitivity profiles')
    for row in validation['posthoc_parameter_checks']:
        fit = row['fit']; g = fit['fixed_width_nm']
        table = (build_height_tables([g], min_height_ev=row['height_lower_ev'])[g]
                 if row['kind'] == 'height_lower_bound' else tables[g])
        require(fit['converged'], 'Post-hoc fit did not converge')
        out = forward(fit, table)
        close(np.sqrt(np.mean(np.log10(out['current_a']/observed)**2)), fit['rmse_decade'])

    single, mixtures = wkb['single_path_checks'], wkb['parallel_mixture_checks']
    require(len(single) == 1200 and len(mixtures) == 200, 'Incomplete numerical WKB cases')
    for row in single+mixtures:
        bound = row['voltage_V']/(Boltzmann*row['temperature_K']/elementary_charge)
        close(bound, row['absolute_log_ratio_bound'])
        require(abs(row['natural_log_ratio']) <= bound+1e-8, 'Frozen WKB bound violated in numerical cases')
    for row in single:
        close(row['natural_log_ratio'], row['log_negative_integral']-row['log_positive_integral'])
        require(row['spectral_order_max_residual'] <= 1e-10 and row['thermal_window_max_bound_residual'] <= 1e-10,
                'Spectral or thermal inequality failed')
    max_wkb_quad = max(r['quadrature_max_log_difference'] for r in single)
    require(max_wkb_quad <= 2e-6, 'WKB quadrature tolerance failed')
    close(max_wkb_quad, wkb['maximum_quadrature_refinement_natural_log'])
    close(min(r['absolute_log_ratio_bound']-abs(r['natural_log_ratio']) for r in single+mixtures),
          wkb['minimum_numerical_bound_margin'])
    close(max(abs(r['natural_log_ratio'])/r['absolute_log_ratio_bound'] for r in single),
          wkb['maximum_fraction_of_thermal_bound'])
    for row in sorted(single, key=lambda r:r['quadrature_max_log_difference'], reverse=True)[:20]:
        refined, original = [], []
        for sign, key in ((1, 'log_positive_integral'), (-1, 'log_negative_integral')):
            args = (row['voltage_V'], row['left_height_ev'], row['right_height_ev'], row['width_nm'])
            opts = {'sign':sign, 'temperature_k':row['temperature_K'], 'mass_ratio':row['mass_ratio']}
            refined.append(log_current_integral(*args, **opts, order=96))
            original.append(log_current_integral(*args, **opts, order=48))
            close(refined[-1], row[key])
        close(np.max(np.abs(np.array(refined)-original)), row['quadrature_max_log_difference'])

    archived = json.loads((ROOT / 'analysis/parallel_rectification_bound.json').read_text())
    require(wkb['all53_measured_coverage_inherited'] == archived['groups'] and
            archived['groups']['all53']['n_pairs'] == 53, 'Observed all53 provenance changed')
    inherited = {(method, r['cycle_id']):r for method, group in archived['processing_sensitivity'].items()
                 for r in group['at_0p1_first3']}
    methods = wkb['first3_all_processing_at_0p1V']
    require(len(archived['processing_sensitivity']) == 8 and len(methods) == 24 and
            {(r['processing'], r['cycle_id']) for r in methods} == set(inherited), 'Missing processing cases')
    for row in methods:
        source = inherited[row['processing'], row['cycle_id']]
        v, ratio = source['hbn_voltage_V'], source['zero_reference_ratio_upper']
        close(row['observed_ratio'], source['ratio'])
        close(row['internal_voltage_V'], v)
        close(row['favorable_zero_reference_ratio_upper'], ratio)
        vt = Boltzmann*300/elementary_charge
        close(row['frozen_passive_wkb_ratio_lower'], np.exp(-v/vt))
        require(0 < ratio < np.exp(-v/vt) and row['favorable_upper_still_below_bound'], 'First3 bound claim failed')
        close(row['conditional_extra_landscape_difference_lower_ev'], max(vt*abs(np.log(ratio))-v, 0))
        for t, value in row['temperature_sensitivity_lower_ev'].items():
            close(value, max(Boltzmann*float(t)/elementary_charge*abs(np.log(ratio))-v, 0))
    require(wkb['first3_primary_at_0p1V'] == [r for r in methods if r['processing'] == 'mean_linear_current'],
            'Primary processing mapping changed')
    require([r['cycle_id'] for r in wkb['robust_first3_summary']] == [1, 2, 3], 'Missing robust summary')
    for summary in wkb['robust_first3_summary']:
        close(summary['minimum_extra_landscape_difference_lower_ev_over_8_processing_methods'],
              min(r['conditional_extra_landscape_difference_lower_ev'] for r in methods if r['cycle_id'] == summary['cycle_id']))
        require(summary['all_8_favorable_ratio_upper_bounds_violate'], 'Robust summary inconsistent')

    log = ROOT.parent / 'verification/2026-09-20/barrier-height-tests-final.log'
    tests = re.findall(r'Ran (\d+) tests in ([\d.]+)s\s+OK', log.read_text())
    require(tests and int(tests[-1][0]) == 178, 'Passing full-suite log missing or wrong count')
    audit = {'scope':'Constitutive height-state prediction and conditional frozen-passive WKB bound; no full-history or microscopic validation claim',
             'provenance':provenance, 'old_manifests_verified_without_rewriting':old,
             'n_curves':101, 'n_branches':202, 'n_observed_points':observed.size,
             'capacity_incompatible_points_retained':49, 'not_above_3zero_points_retained':857,
             'each_trace_tested_once_and_never_split_across_train_test':True,
             'geometry_selected_by_training_score_only':True, 'selected_width_nm_by_split':selected_widths,
             'shared_supply_and_geometry_frozen_on_targets':True,
             'n_shared_fits':fit_count, 'n_converged_shared_fit_starts':len(starts), 'n_target_local_fits':5,
             'grouped_higher_voltage_scores_recomputed':pooled,
             'maximum_training_rescore_difference_decade':max_rescore, 'maximum_kcl_relative_error':max_kcl,
             'n_training_refinements_recomputed':len(refinements), 'n_target_refinements_recomputed':len(target_refinements),
             'maximum_quadrature_refinement_decade':max_quad, 'maximum_combined_refinement_decade':max_grid,
             'maximum_target_refinement_decade':max_target, 'n_posthoc_scores_recomputed':7,
             'wkb_stored_single_cases_checked':len(single), 'wkb_stored_positive_mixtures_checked':len(mixtures),
             'wkb_largest_quadrature_error_cases_reintegrated':20, 'maximum_wkb_quadrature_error_natural_log':max_wkb_quad,
             'first3_processing_cases_recomputed':len(methods), 'robust_first3_summary':wkb['robust_first3_summary'],
             'test_count':int(tests[-1][0]), 'test_seconds':float(tests[-1][1]), 'test_log_sha256':digest(log),
             'runtime':{'python':platform.python_version(), 'numpy':np.__version__, 'scipy':scipy.__version__, 'platform':platform.platform()},
             'verifier_source_sha256':digest(Path(__file__))}
    audit_path = ROOT / 'analysis/barrier_height_polarity_validation_audit.json'
    audit_path.write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
    files = paths+[audit_path, Path(__file__)]
    files += [ROOT / p for report in reports for p in report['input_sha256']]
    files += [ROOT / 'analysis' / p for p in ('BARRIER_HEIGHT_AND_POLARITY_RESULTS_2026-09-21.md',
              'plot_barrier_height_polarity.py', 'figures/35_barrier_height_polarity.png',
              'figures/35_barrier_height_polarity.pdf')]
    files += list((ROOT / 'tests').glob('test_*.py'))+list((ROOT / 'vulcan2d').glob('*.py'))
    files += list((ROOT / 'vulcan2d').glob('*.cpp'))
    logs = [ROOT.parent / 'verification/2026-09-20' / name for name in (
        'barrier-height-unit.log', 'wkb-polarity-unit.log', 'barrier-height-comparison.log',
        'barrier-height-validation.log', 'wkb-polarity-bound.log', 'barrier-height-tests-final.log')]
    manifest = {'scope':audit['scope'], 'date':'2026-09-21', 'production_model_changed':False,
                'artifacts':{str(p.relative_to(ROOT)):{'sha256':digest(p), 'bytes':p.stat().st_size} for p in dict.fromkeys(files)},
                'external_workspace_logs':{str(p.relative_to(ROOT.parent)):{'sha256':digest(p), 'bytes':p.stat().st_size} for p in logs}}
    (ROOT / 'analysis/barrier_height_polarity_artifact_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    run()
