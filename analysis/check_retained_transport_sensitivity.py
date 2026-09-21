"""Local derivative and tangent-space audit of inherited transport priors."""
from pathlib import Path
import json
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'analysis'))
from fit_retained_transport import digest
from compare_activated_configuration import assemble, lookups
from vulcan2d.retained_charge_configuration import trajectory
from vulcan2d.retained_transport_parameters import pack, unpack, LO, HI, SCALE, NAMES, INHERITED, PREVIOUS_ACTIVE


def run():
    d = assemble(); lookup = lookups(); rows = {}; arrays = {}; inputs = [Path(__file__)]
    for split in ('all_descriptive', 'chronological'):
        path = ROOT/'analysis'/f'retained_charge_configuration_{split}_single_contact.json'
        source = json.loads(path.read_text()); inputs.append(path)
        for name, sha in source['input_sha256'].items(): assert digest(ROOT/name) == sha, name
        x = pack(source['shared_parameters'], source['parameters']); end = source['training_ids'][-1]*676
        voltage = d['voltage'][:end]; mask = d['weights'][:end] > 0
        weight = np.sqrt(d['weights'][:end][mask]/len(source['training_ids']))
        settings = source['settings']; cycle = d['cycle'][:end]; window = d['window'][:end]
        def calculate(xx):
            s, p = unpack(xx); r = trajectory(voltage, s, p, *lookup, **settings)
            assert r['maximum_kcl_relative_error'] < 1e-8
            logs = np.zeros(len(voltage)); nonzero = abs(voltage) > 0; logs[nonzero] = np.log10(r['current_A'][nonzero])
            return logs
        base = calculate(x); residual = weight*(base[mask]-np.log10(d['observed'][:end][mask])); jacobians = []; details = []
        def logratio(logs, wi):
            return float(np.mean(logs[(cycle == 4) & (window == wi) & (abs(voltage) >= .1-1e-10)])-
                         np.mean(logs[(cycle == 3) & (window == wi) & (abs(voltage) >= .1-1e-10)]))
        ratio_derivatives = []
        for epsilon in (1e-3, 5e-4):
            jac = np.empty((mask.sum(), 14)); ratio_gradient = np.empty((2, 14)); report = []
            for j, name in enumerate(NAMES):
                step = SCALE[j]*epsilon; left = x.copy(); right = x.copy()
                if x[j]-step < LO[j]:
                    right[j] += step; a, b, denominator, direction = base, calculate(right), epsilon, 'forward'
                elif x[j]+step > HI[j]:
                    left[j] -= step; a, b, denominator, direction = calculate(left), base, epsilon, 'backward'
                else:
                    left[j] -= step; right[j] += step
                    a, b, denominator, direction = calculate(left), calculate(right), 2*epsilon, 'central'
                derivative = (b-a)/denominator; jac[:, j] = weight*derivative[mask]
                ratio_gradient[:, j] = [logratio(derivative, wi) for wi in (2, 4)]
                report.append({'parameter': name, 'absolute_step': step, 'direction': direction})
            jacobians.append(jac); ratio_derivatives.append(ratio_gradient); details.append(report)
            print(split, 'epsilon', epsilon, 'done', flush=True)
        coarse, fine = jacobians; norms = np.linalg.norm(fine, axis=0)
        error = np.linalg.norm(fine-coarse, axis=0)
        comparisons = [{'parameter': name, 'fine_weighted_derivative_norm': float(norms[j]),
                        'halfstep_absolute_norm_change': float(error[j]),
                        'halfstep_relative_norm_change': float(error[j]/max(norms[j], 1e-8))}
                       for j, name in enumerate(NAMES)]
        u, sigma, vt = np.linalg.svd(fine[:, PREVIOUS_ACTIVE], full_matrices=False)
        keep = sigma > sigma[0]*1e-7; basis = u[:, keep]
        extra = fine[:, INHERITED]-basis@(basis.T@fine[:, INHERITED])
        old_residual = residual-basis@(basis.T@residual)
        u2, ss, vt2 = np.linalg.svd(fine, full_matrices=False); kept2 = ss > ss[0]*1e-7
        joint_residual = residual-u2[:, kept2]@(u2[:, kept2].T@residual)
        rows[split] = {'parameter_source': str(path.relative_to(ROOT)), 'coordinates': x.tolist(),
            'settings': settings, 'training_ids': source['training_ids'], 'derivative_steps': details,
            'halfstep_comparisons': comparisons, 'old_scaled_singular_values': sigma.tolist(),
            'joint_scaled_singular_values': ss.tolist(), 'relative_singular_cutoff': 1e-7,
            'old_local_numerical_rank': int(keep.sum()), 'joint_local_numerical_rank': int(kept2.sum()),
            'inherited_column_norm_after_old_projection': dict(zip([NAMES[j] for j in INHERITED], np.linalg.norm(extra, axis=0).tolist())),
            'baseline_training_rmse': float(np.linalg.norm(residual)),
            'old_unconstrained_linear_tangent_residual': float(np.linalg.norm(old_residual)),
            'joint_unconstrained_linear_tangent_residual': float(np.linalg.norm(joint_residual)),
            'interpretation': 'Local scaled derivative diagnostic only; unbounded tangent residual is not attainable-fit proof or statistical parameter uncertainty',
            'log_SET_4_over_3_scaled_gradient': ratio_derivatives[-1][0].tolist(),
            'log_RESET_4_over_3_scaled_gradient': ratio_derivatives[-1][1].tolist(),
            'log_selectivity_scaled_gradient': (ratio_derivatives[-1][1]-ratio_derivatives[-1][0]).tolist()}
        arrays[f'{split}_weighted_jacobian'] = fine; arrays[f'{split}_weighted_residual'] = residual
        arrays[f'{split}_scaled_right_singular_vectors'] = vt2
    for path in [ROOT/'vulcan2d/retained_transport_parameters.py', ROOT/'analysis/RETAINED_TRANSPORT_AUDIT_DESIGN_2026-09-21.md',
                 ROOT/'analysis/fit_retained_transport.py']:
        inputs.append(path)
    destination = ROOT/'analysis/retained_transport_sensitivity.npz'; np.savez_compressed(destination, **arrays)
    out = {'scope': 'Training-only local sensitivity; no microscopic identifiability or global exclusion claim',
           'parameter_names': list(NAMES), 'parameter_scales': SCALE.tolist(), 'candidates': rows,
           'arrays_sha256': digest(destination), 'input_sha256': {str(p.relative_to(ROOT)): digest(p) for p in inputs}}
    (ROOT/'analysis/retained_transport_sensitivity.json').write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
    print('COMPLETE', {k:(r['baseline_training_rmse'],r['old_unconstrained_linear_tangent_residual'],r['joint_unconstrained_linear_tangent_residual']) for k,r in rows.items()}, flush=True)


if __name__ == '__main__': run()
