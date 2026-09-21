"""Height-state refinement, predictive scores, and post-hoc local prior checks."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'analysis'))
from compare_barrier_states import assemble
from compare_barrier_height_states import WIDTHS, fit, simulate
from vulcan2d.barrier_height_state import build_height_tables


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    source = ROOT / 'analysis/barrier_height_comparison.json'
    report = json.loads(source.read_text())
    if report['status'] != 'complete':
        raise RuntimeError('Main comparison not complete')
    for p, h in report['input_sha256'].items():
        if digest(ROOT / p) != h:
            raise RuntimeError('Input changed: '+p)
    va, obs, zeros, branches, lookup = assemble()
    original = build_height_tables(WIDTHS)
    print('Building quadrature-refined tables', flush=True)
    quadrature = build_height_tables(WIDTHS, order=48)
    print('Building grid-and-quadrature-refined tables', flush=True)
    finer = build_height_tables(WIDTHS, voltage_step=.00125, height_step_ev=.005, order=48)
    checks, predicted_checks = [], []
    for label, split in report['splits'].items():
        for key, fitted in split['fits'].items():
            g = fitted['fixed_width_nm']; shared = fitted['log10_prefactor_S']; local = fitted['local_height_ev']
            out = simulate(va, shared, local, original[g], lookup)
            quad = simulate(va, shared, local, quadrature[g], lookup)
            fine = simulate(va, shared, local, finer[g], lookup)
            checks.append({'split':label, 'profile':key,
                           'quadrature_maximum_difference_decade':float(np.max(np.abs(np.log10(quad['current_a']/out['current_a'])))),
                           'combined_refinement_maximum_difference_decade':float(np.max(np.abs(np.log10(fine['current_a']/out['current_a'])))),
                           'refined_max_kcl_relative_error':fine['max_kcl_relative_error']})
        if label != 'all_descriptive':
            selected = split['fits'][split['selected_profile_key']]; local = split['prediction']['local_identification']
            out = simulate(va, selected['log10_prefactor_S'], local['local_height_ev'], finer[selected['fixed_width_nm']], lookup)
            old = np.array(split['prediction']['predicted_current_A'])
            predicted_checks.append({'split':label, 'combined_refinement_maximum_difference_decade':float(np.max(np.abs(np.log10(out['current_a']/old)))),
                                     'refined_max_kcl_relative_error':out['max_kcl_relative_error']})
    errors, robust_errors, phase_errors = [], [], {'outward':[], 'return':[]}
    for label, split in report['splits'].items():
        if label == 'all_descriptive':
            continue
        indices = np.array(split['target_branch_indices']); score = va >= .17-1e-10
        predicted = np.array(split['prediction']['predicted_current_A'])[:, score]; measured = obs[indices][:, score]
        error = np.log10(predicted/measured); errors.extend(error.ravel())
        capacity = np.interp(va[score], lookup.voltage, lookup.current)
        reliable = (measured > 3*zeros[indices, None]) & (measured <= capacity)
        robust_errors.extend(error[reliable])
        for j, index in enumerate(indices):
            phase_errors[branches[index]['branch']].extend(error[j])
    scores = {'n_scored_points':len(errors), 'rmse_decade':float(np.sqrt(np.mean(np.square(errors)))),
              'n_high_signal_capacity_compatible_points':len(robust_errors),
              'high_signal_capacity_compatible_rmse_decade':float(np.sqrt(np.mean(np.square(robust_errors)))),
              **{p+'_rmse_decade':float(np.sqrt(np.mean(np.square(e)))) for p, e in phase_errors.items()}}
    full = report['splits']['all_descriptive']; selected = full['fits'][full['selected_profile_key']]
    g = selected['fixed_width_nm']; prior_checks = []
    for lower in (.005, .05):
        table = build_height_tables([g], min_height_ev=lower)[g]
        fitted = fit(va, obs, table, lookup)
        prior_checks.append({'kind':'height_lower_bound', 'height_lower_ev':lower, 'fit':fitted})
        print('Height-bound sensitivity', lower, fitted['rmse_decade'], fitted['log10_prefactor_S'], flush=True)
    for shift in (-1., -.5, 0., .5, 1.):
        shared = selected['log10_prefactor_S']+shift
        fitted = fit(va, obs, original[g], lookup, fixed_shared=shared)
        prior_checks.append({'kind':'fixed_supply_profile', 'log10_prefactor_S':shared, 'fit':fitted})
        print('Supply profile', shared, fitted['rmse_decade'], flush=True)
    local = np.array(selected['local_height_ev']).reshape(-1, 2)
    descriptive_states = {'fixed_width_nm':g, 'log10_prefactor_S':selected['log10_prefactor_S'],
                          'outward_height_ev_q10_q50_q90':np.quantile(local[:, 0], [.1, .5, .9]).tolist(),
                          'return_height_ev_q10_q50_q90':np.quantile(local[:, 1], [.1, .5, .9]).tolist(),
                          'return_minus_outward_ev_q10_q50_q90':np.quantile(local[:, 1]-local[:, 0], [.1, .5, .9]).tolist(),
                          'interpretation':'Both branches independently fitted; these are conditional state coordinates, not independent barrier measurements or dynamical predictions.'}
    files = [Path(__file__), source]+[ROOT / p for p in report['input_sha256']]
    result = {'scope':'Numerical validation and post-hoc height/supply sensitivity; primary five-fold choices are unchanged',
              'training_profile_refinements':checks, 'target_prediction_refinements':predicted_checks,
              'grouped_higher_voltage_scores':scores, 'posthoc_parameter_checks':prior_checks,
              'conditional_descriptive_height_states':descriptive_states,
              'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in dict.fromkeys(files)}}
    (ROOT / 'analysis/barrier_height_validation.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('Pooled scores',json.dumps(scores, indent=2),flush=True)
    print('Max combined refinement',max(r['combined_refinement_maximum_difference_decade'] for r in checks),flush=True)


if __name__ == '__main__':
    run()
