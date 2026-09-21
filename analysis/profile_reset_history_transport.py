"""Profile nuisance transport to test if inferred RESET state is identifiable."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'analysis'))
from compare_reset_histories import assemble
from vulcan2d.reset_history import predict_internal, domains


def profile(data, fit, alpha):
    name = fit['name']; base = np.array(fit['shared_parameters']); base[1] = alpha
    free = [k for k in range(len(base)) if k != 1]
    glo, ghi, qlo, qhi = domains(name); n, m = data['y'].shape; ng = len(free)
    lo, hi = np.r_[glo[free], np.tile(qlo, n)], np.r_[ghi[free], np.tile(qhi, n)]
    weights = data['weights']; sw = np.sqrt(weights/n)
    sparsity = lil_matrix((n*m, ng+3*n), dtype=int); sparsity[:, :ng] = 1
    for k in range(n):sparsity[k*m:(k+1)*m, ng+3*k:ng+3*k+3] = 1
    sparsity = sparsity.tocsr()
    def unpack(x):
        p = base.copy(); p[free] = x[:ng]
        return p, x[ng:].reshape(n, 3)
    def residual(x):
        p, q = unpack(x); y = predict_internal(data['vapp'], data['vh'], name, p, q)
        return (np.where(weights > 0, y-data['y'], 0.)*sw).ravel()
    initial = np.r_[base[free], np.array(fit['local_parameters']).ravel()]
    trials = []
    for shift in (0., -.3, .3):
        guess = initial.copy(); guess[ng:].reshape(n, 3)[:, 2] += shift
        trials.append(least_squares(residual, np.clip(guess, lo+1e-8, hi-1e-8), bounds=(lo, hi), jac_sparsity=sparsity,
                                    x_scale=hi-lo, max_nfev=500, ftol=2e-7, xtol=2e-7, gtol=2e-7,
                                    tr_options={'atol': 1e-6, 'btol': 1e-6, 'maxiter': 300}))
    good = [f for f in trials if f.success]; f = min(good or trials, key=lambda f:f.cost); p, q = unpack(f.x)
    unused, state = predict_internal(data['vapp'], data['vh'], name, p, q, True)
    half = []
    for k in range(n):
        apex = int(np.argmax(data['vapp'][k])); logp = state[k, :apex+1]
        crossings = np.flatnonzero(logp <= -np.log(2.))
        if not len(crossings):half.append(None); continue
        j = int(crossings[0])
        if j == 0:half.append(float(data['vh'][k, j])); continue
        fraction = (-np.log(2.)-logp[j-1])/(logp[j]-logp[j-1])
        half.append(float(data['vh'][k, j-1]+fraction*(data['vh'][k, j]-data['vh'][k, j-1])))
    eligible = data['memory_contrast'] > np.log10(5.)
    values = [v for v, use in zip(half, eligible) if use and v is not None]
    result = {'candidate': name, 'fixed_connected_alpha_per_V': alpha, 'converged': bool(f.success),
              'rmse_decade': float(np.sqrt(2*f.cost)), 'shared_parameters': p.tolist(), 'local_parameters': q.tolist(),
              'inferred_half_survival_internal_voltage_V': half,
              'eligible_half_survival_q10_median_q90_V': np.quantile(values, [.1, .5, .9]).tolist(),
              'multistart': [{'success': bool(x.success), 'rmse_decade': float(np.sqrt(2*x.cost)), 'nfev': int(x.nfev)} for x in trials]}
    if name == 'thermal_rate':
        zero_hazard_per_V = np.log(2.)*np.exp(-.5*p[2])/10**q[:, 2]
        # A conditional zero-bias extension of the ONE-WAY loss law. The
        # original paper's retention is at positive read bias and on other
        # programmed states, so this is NOT an experimental contradiction.
        minimum_reset_duration = 3.4*zero_hazard_per_V*7*3600/(-np.log(.95))
        result['conditional_one_way_zero_bias_retention_check'] = {
            'hypothesis': 'Same fitted one-way erasure rate extrapolates to zero bias, 95% survival for 7 h, constant sweep speed on a 3.4-V total RESET path',
            'minimum_total_reset_duration_s_q10_median_q90': np.quantile(minimum_reset_duration[eligible], [.1, .5, .9]).tolist(),
            'limitation': 'Counterfactual model consistency check, not measured sweep duration or a direct test of the paper\'s +0.1-V retention protocol.'}
    return result


def run():
    data, unused, unused = assemble(); source = ROOT/'analysis/reset_history_comparison.json'; r = json.loads(source.read_text())
    rows = []
    for name, candidate in r['candidates'].items():
        for alpha in (2., 4., 8.044375371895217, 12., 20.):
            row = profile(data, candidate['all_data_fit'], alpha); rows.append(row)
            print(name, alpha, round(row['rmse_decade'], 5), row['eligible_half_survival_q10_median_q90_V'], row['converged'], flush=True)
    paths = [Path(__file__), source, ROOT/'analysis/compare_reset_histories.py', ROOT/'vulcan2d/reset_history.py']
    result = {'scope': 'Descriptive parameter profiles after full-history fitting, not confidence intervals or new validation', 'profiles': rows,
              'interpretation': 'Voltage-shape and survival can compensate. Small fit-cost differences with different half-survival voltages undermine a unique microscopic threshold interpretation.',
              'limitations': ['H/C remain effective envelopes; the fixed alpha grid is an exploratory identifiability probe, not measured transport uncertainty.',
                              'Weak-memory cycles remain in all fits; half-survival summaries use the prespecified >5-fold scan-contrast training information criterion.',
                              'No absolute hopping barrier, defect location, rate prefactor or lifetime is measured by these voltage-domain fits.'],
              'input_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (ROOT/'analysis/reset_history_transport_profiles.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':run()
