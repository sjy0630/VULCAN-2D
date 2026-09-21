"""Numerical refinement and declared post-hoc barrier-parameter sensitivity."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'analysis'))
import compare_barrier_states as comparison
from vulcan2d.barrier_state import BarrierTable, supply_conductance


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    report = json.loads((ROOT / 'analysis/barrier_state_comparison.json').read_text())
    if report['status'] != 'complete':
        raise RuntimeError('Main grouped validation must finish first')
    for path, h in report['input_sha256'].items():
        if digest(ROOT / path) != h:
            raise RuntimeError('Main input changed: '+path)
    va, obs, zeros, branches, lookup = comparison.assemble()
    fine = {}; quadrature = {}; checks = []
    for height in comparison.HEIGHTS:
        fine[height] = BarrierTable(height, voltage_step=.00125, width_step_nm=.0125, order=48)
        quadrature[height] = BarrierTable(height, order=48)
    for label, split in report['splits'].items():
        for key, fit in split['fits'].items():
            if fit['candidate'] != 'barrier':
                continue
            height = fit['height_ev']; local = fit['local_parameters']; shared = fit['shared_parameter']
            original = comparison.simulate(va, shared, local, 'barrier', lookup, height)
            q = comparison.simulate(va, shared, local, 'barrier', lookup, table=quadrature[height])
            finer = comparison.simulate(va, shared, local, 'barrier', lookup, table=fine[height])
            checks.append({'split':label, 'profile':key,
                           'quadrature_maximum_difference_decade':float(np.max(np.abs(np.log10(q['current_a']/original['current_a'])))),
                           'combined_grid_quadrature_maximum_difference_decade':float(np.max(np.abs(np.log10(finer['current_a']/original['current_a'])))),
                           'refined_max_kcl_relative_error':finer['max_kcl_relative_error']})
    predictions = []; grouped_scores = {}
    for candidate in ('power', 'pf_like', 'barrier'):
        all_errors, high_signal_errors, phase_errors = [], [], {'outward':[], 'return':[]}
        for label, split in report['splits'].items():
            if label == 'all_descriptive':
                continue
            saved = split['predictions'][candidate]; indices = np.array(split['target_branch_indices'])
            predicted = np.array(saved['predicted_current_A']); target = obs[indices]
            score = va >= .17-1e-10; error = np.log10(predicted[:, score]/target[:, score])
            all_errors.extend(error.ravel())
            cap = np.interp(va[score], lookup.voltage, lookup.current)
            robust = (target[:, score] > 3*zeros[indices, None]) & (target[:, score] <= cap)
            high_signal_errors.extend(error[robust])
            for j, index in enumerate(indices):
                phase_errors[branches[index]['branch']].extend(error[j])
            if candidate == 'barrier':
                fit = split['fits'][saved['source_fit_key']]; local = saved['local_identification']['local_parameters']
                refined = comparison.simulate(va, fit['shared_parameter'], local, candidate, lookup, table=fine[fit['height_ev']])
                predictions.append({'split':label, 'maximum_difference_decade':float(np.max(np.abs(np.log10(refined['current_a']/predicted)))),
                                    'refined_max_kcl_relative_error':refined['max_kcl_relative_error']})
        grouped_scores[candidate] = {'n_scored_points':len(all_errors),
                                     'rmse_decade':float(np.sqrt(np.mean(np.square(all_errors)))),
                                     'n_high_signal_capacity_compatible_points':len(high_signal_errors),
                                     'high_signal_capacity_compatible_rmse_decade':float(np.sqrt(np.mean(np.square(high_signal_errors)))),
                                     **{phase+'_rmse_decade':float(np.sqrt(np.mean(np.square(errors)))) for phase, errors in phase_errors.items()}}
    sensitivity = []; original_bounds = comparison.bounds
    # These post-hoc descriptive fits probe a prior boundary; they do not replace
    # the fixed-grid, training-only model choices in the preceding five-fold test.
    for upper in (2., 6.):
        def relaxed_bounds(candidate, top=upper):
            return ((-7., top), (0., 6.)) if candidate == 'barrier' else original_bounds(candidate)
        comparison.bounds = relaxed_bounds
        for height in (.5, 1., 2., 4.):
            fit = comparison.fit(va, obs, 'barrier', lookup, height)
            sensitivity.append({'kind':'supply_prefactor_upper_bound', 'log10_prefactor_upper_S':upper,
                                'height_ev':height, 'mass_ratio':.5, 'fit':fit})
            print('Supply-bound sensitivity', upper, height, fit['rmse_decade'], fit['shared_parameter'], flush=True)
    comparison.bounds = original_bounds
    for mass in (.2, 1.):
        comparison.TABLES[.5] = BarrierTable(.5, mass_ratio=mass)
        fit = comparison.fit(va, obs, 'barrier', lookup, .5)
        sensitivity.append({'kind':'conditional_effective_mass', 'log10_prefactor_upper_S':-1.,
                            'height_ev':.5, 'mass_ratio':mass, 'fit':fit})
        print('Mass sensitivity', mass, fit['rmse_decade'], fit['shared_parameter'], flush=True)
    comparison.TABLES.pop(.5, None)
    default_fit = report['splits']['all_descriptive']['fits']['barrier_0.5']
    widths = np.array(default_fit['local_parameters']).reshape(-1, 2)
    width_summary = {'conditional_mass_ratio':.5, 'conditional_barrier_height_ev':.5,
                     'outward_width_nm_q10_q50_q90':np.quantile(widths[:, 0], [.1, .5, .9]).tolist(),
                     'return_width_nm_q10_q50_q90':np.quantile(widths[:, 1], [.1, .5, .9]).tolist(),
                     'return_minus_outward_nm_q10_q50_q90':np.quantile(widths[:, 1]-widths[:, 0], [.1, .5, .9]).tolist(),
                     'n_return_width_larger':int(sum(widths[:, 1] > widths[:, 0])),
                     'interpretation':'Two observed branches are fitted separately; these differences are not predictions or measured geometry.'}
    inputs = [Path(__file__), ROOT / 'analysis/barrier_state_comparison.json']+[ROOT / p for p in report['input_sha256']]
    result = {'scope':'Numerical audit and post-hoc prior/geometry sensitivity; main grouped model selection is unchanged',
              'all_training_profile_refinements':checks, 'target_prediction_refinements':predictions,
              'grouped_higher_voltage_scores':grouped_scores, 'posthoc_descriptive_parameter_sensitivities':sensitivity,
              'conditional_width_summary':width_summary,
              'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in dict.fromkeys(inputs)}}
    (ROOT / 'analysis/barrier_state_sensitivity.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('Grouped scores', json.dumps(grouped_scores, indent=2), flush=True)
    print('Maximum combined refinement', max(r['combined_grid_quadrature_maximum_difference_decade'] for r in checks), flush=True)


if __name__ == '__main__':
    run()
