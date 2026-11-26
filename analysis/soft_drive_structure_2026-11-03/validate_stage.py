"""Independent arithmetic review of saved results; no new trajectories or fits."""
from pathlib import Path
import hashlib
import json
import math
import time
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
started = time.monotonic()
probe = json.loads((HERE / 'sinh_field_probe.json').read_text())
inverse = json.loads((HERE / 'DRIVE_STRUCTURE_INVERSION.json').read_text())
checks = {}

def record(name, condition):
    checks[name] = bool(condition)
    assert condition, name

def close(name, actual, reported, atol=1e-12):
    record(name, np.allclose(actual, reported, rtol=1e-10, atol=atol))

record('two_completed_precision_runs_no_failures',
       probe['n_completed_runs'] == 2 and probe['n_failed_runs'] == 0)
record('one_fixed_effective_voltage_scale', probe['plan']['Vf_v'] == .5)
with np.load(HERE / 'sinh_field_probe.npz') as z:
    arr = {k: z[k].copy() for k in z.files}
v = arr['command_voltage_v']
obs = abs(arr['observed_current_a'])
mask = (v > 1e-12) & np.isfinite(obs) & (obs >= 1e-11)
record('original500_mask_preserved', np.array_equal(mask, arr['original_scoring_mask']) and mask.sum() == 500)
out = np.arange(len(v)) <= np.argmax(v)
with np.load(HERE.parent / 'reset_shape_2026-11-03/soft_field_mobility_probe.npz') as z:
    record('identical_observations_to_baseline', np.array_equal(obs, abs(z['observed_current_a'])) and
           np.array_equal(v, z['command_voltage_v']) and np.array_equal(mask, z['original_scoring_mask']))
    baseline = z['refined_current_a'].copy()

scores = {}
for run in probe['runs']:
    name = run['precision']['name']
    current, state, local = (arr[name + s] for s in ('_current_a', '_state_u', '_local_voltage_v'))
    groups = {'all500': mask, 'outward250': mask & out, 'return250': mask & ~out,
              'outward_low10': mask & out & (v <= .2 + 1e-12),
              'both_branches_low20': mask & (v <= .2 + 1e-12),
              'remaining480': mask & (v > .2 + 1e-12),
              'outward0p5_to1p5_50': mask & out & (v > .5 + 1e-12) & (v <= 1.5 + 1e-12)}
    scores[name] = {}
    for group, use in groups.items():
        e = np.log10(current[use] / obs[use])
        value = np.linalg.norm(e) / math.sqrt(use.sum())
        close(name + '_' + group, value, run['metrics'][group]['RMSE_decade'])
        record(name + '_' + group + '_count', use.sum() == run['metrics'][group]['n'])
        scores[name][group] = float(value)
    record(name + '_passive_finite_current', np.all(np.isfinite(current)) and np.all(current >= 0) and
           np.all(local >= 0) and np.all(local <= v + 1e-12))
    record(name + '_finite_monotone_state', np.all(np.isfinite(state)) and np.all(state > 0) and
           np.max(np.diff(state)) <= 5e-7)
    record(name + '_continuous_apex_handoff', run['segments'][1]['final_u'] == run['segments'][2]['initial_u'])
    for feature, result in run['reference_features'].items():
        idx = result['native_indices']
        value = current[idx[0]] if len(idx) == 1 else math.log(current[idx[1]] / current[idx[0]]) / math.log(v[idx[1]] / v[idx[0]])
        close(name + '_' + feature, value, result['single_cycle_prediction'])
        lo, hi = result['original32_aggregate_band']
        record(name + '_' + feature + '_decision', (lo <= value <= hi) == result['single_prediction_inside_reference_band'])
    for result in run['actual_budget_endpoints']:
        val = current[result['native_index']]
        close(name + '_' + result['name'], val, result['predicted_current_a'])
        passed = val >= result['bound_current_a'] if result['direction'] == 'at_least' else val <= result['bound_current_a']
        record(name + '_' + result['name'] + '_decision', passed == result['condition_satisfied'])
    record(name + '_explicit_failed_2V_arrival', not run['actual_budget_endpoints'][0]['condition_satisfied'])

screen = probe['necessary_screen']
early = arr['early_u']
new_lower = np.sum(np.diff(early) / arr['early_cell_upper']) / 1.8
close('necessary_lower_recomputed', new_lower, screen['kappa_lower_per_v2'])
close('unique_prespecified_rate', 2 * new_lower, screen['derived_unique_kappa_per_v2'])
small = probe['smallfield_bound']
kappa = screen['derived_unique_kappa_per_v2']
vf = .5
bound = 2 * kappa * small['Dmax'] * vf ** 2 * math.sinh(small['vmin_v'] / (2 * vf)) ** 2
close('stable_smallfield_integral_bound', bound, small['local_integral_upper_per_monotone_segment'], atol=0)
record('smallfield_bound_below_declared_threshold', bound < 1e-12)
comp = probe['precision_comparison']
close('precision_current', np.max(abs(np.log10(arr['nominal_current_a'][mask] / arr['refined_current_a'][mask]))), comp['max_native_log_current_change'])
close('precision_state', np.max(abs(arr['nominal_state_u'] - arr['refined_state_u'])), comp['max_native_state_change'])
record('KCL_independent_root_check_passed', probe['KCL_comparison']['passed'])

record('bounded_inverse_count', inverse['inverse_problem_count'] == 29)
pair, bands = inverse['pair_tradeoffs']
record('exact_pair_conflict_distinct_from_reference_bands', pair['order_conflict'] and not bands['order_conflict'])
close('minimax_pair_balance', sum(pair['signed_log10_current_errors']), 0., atol=2e-12)
close('pair_multiplicative_error', 10 ** pair['minimum_possible_max_absolute_log10_current_error'],
      pair['corresponding_symmetric_multiplicative_factor'])
record('inverse_kept_original_mask', inverse['original_scoring_count'] == 500 and not inverse['original_scoring_mask_modified'])

source_validation = {}
for label, mapping in [('probe', probe['source_sha256']), ('frozen_baseline', probe['frozen_baseline_hashes'])]:
    source_validation[label] = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected for path, expected in mapping.items()}
    record(label + '_source_hashes_unchanged', all(source_validation[label].values()))

bare_path = HERE / 'soft_state_budget.json'
if bare_path.exists():
    bare = json.loads(bare_path.read_text())
    record('bare_budget_completed', bare['status'] == 'completed_conditional_soft_state_budget')
    record('bare_budget_counts', bare['actual_inverse_problems'] == 16 and bare['actual_direct_spectrum_endpoints'] == 4)
    record('bare_direct_bounds_conservative', all(x['conservative_bound_direction_at_direct64'] for x in bare['direct_spectrum_checks']))
    for scenario in bare['scenarios']:
        uout = scenario['outward_inverse']['u_bound']
        for j, row in enumerate(scenario['return_bounds']):
            close(f'bare_delta_{scenario["relative_current_half_band"]}_{j}',
                  row['inverse']['u_bound'] - uout, row['necessary_signed_delta_u_lower'])

baseline_rmse = float(np.linalg.norm(np.log10(baseline[mask] / obs[mask])) / math.sqrt(500))
result = {'status': 'passed_saved_result_arithmetic_and_source_review', 'checks': checks,
          'source_validation': source_validation, 'scores': scores,
          'same_cycle_baseline_RMSE_decade': baseline_rmse,
          'fractional_RMSE_reduction': 1 - scores['refined']['all500'] / baseline_rmse,
          'new_ODE_or_parameter_fit_count': 0,
          'elapsed_s': time.monotonic() - started,
          'review_scope': 'Root source/mathematics review and independent saved-array arithmetic; delegated final review was interrupted by usage limits. No claim of complete unified-model or microscopic validation.'}
(HERE / 'STAGE_VALIDATION.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: result[k] for k in ('status', 'elapsed_s', 'fractional_RMSE_reduction')}, indent=2))
print('checks', len(checks))
