"""Joint feature/cut search under a cycle-vector exchangeability null."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from scipy.stats import beta
from diagnose_cross_polarity_physics import read_pairs
ROOT = Path(__file__).resolve().parents[1]


def cusum(values):
    x = np.asarray(values, float)
    if x.ndim != 2 or len(x) < 3 or not np.isfinite(x).all(): raise ValueError('At least three finite feature vectors required')
    n = len(x); centered = x-x.mean(axis=0); sd = x.std(axis=0, ddof=1)
    k = np.arange(1, n)
    scale = np.sqrt(k*(n-k)/n)[:, None]*np.where(sd > 0, sd, 1.)[None, :]
    return abs(np.cumsum(centered, axis=0)[:-1])/scale


def interval(exceedances, repetitions):
    b, n = exceedances, repetitions
    return [float(beta.ppf(.025, b, n-b+1)) if b else 0.,
            float(beta.ppf(.975, b+1, n-b)) if b < n else 1.]


def permutation_check(values, repetitions, seed):
    x = np.asarray(values, float); n = len(x); centered = x-x.mean(axis=0)
    sd = x.std(axis=0, ddof=1); k = np.arange(1, n)
    scale = np.sqrt(k*(n-k)/n)[:, None]*np.where(sd > 0, sd, 1.)[None, :]
    observed = cusum(x); feature_max = observed.max(axis=0); joint_max = float(observed.max())
    rng = np.random.default_rng(seed); count = 0; per_feature = np.zeros(x.shape[1], int)
    for start in range(0, repetitions, 2000):
        size = min(2000, repetitions-start)
        permutation = rng.permuted(np.broadcast_to(np.arange(n), (size, n)), axis=1)
        score = abs(np.cumsum(centered[permutation], axis=1)[:, :-1])/scale[None, :, :]
        maxima = score.max(axis=1)
        per_feature += np.count_nonzero(maxima >= feature_max[None, :]-1e-12, axis=0)
        count += int(np.count_nonzero(maxima.max(axis=1) >= joint_max-1e-12))
    return {'repetitions': repetitions, 'seed': seed, 'joint_max_statistic': joint_max,
            'joint_exceedances': count, 'joint_monte_carlo_p': (count+1)/(repetitions+1),
            'joint_monte_carlo_binomial_interval': interval(count, repetitions),
            'per_feature_best_cut_after_cycle': (np.argmax(observed, axis=0)+1).tolist(),
            'per_feature_max_statistic': feature_max.tolist(),
            'per_feature_exceedances': per_feature.tolist(),
            'per_feature_search_corrected_p': ((per_feature+1)/(repetitions+1)).tolist()}


def features(reference_shift=False):
    result = []
    for pair in read_pairs('1.1'):
        values = []
        for phase in ('set', 'reset'):
            raw = pair[phase]; v = abs(np.asarray(raw['voltage'])); y = abs(np.asarray(raw['current']))
            index = np.arange(len(v)); apex = np.flatnonzero(np.isclose(v, v.max(), atol=1e-10, rtol=0))
            branch = index > apex[-1] if phase == 'set' else index < apex[0]
            chosen = branch & (v >= .1-1e-10) & (v <= .3+1e-10)
            if reference_shift: y = y+np.max(y[v == 0])
            assert np.all(y[chosen] > 0)
            values.append(float(np.mean(np.log10(y[chosen]))))
        result.append([*values, values[1]-values[0]])
    return np.asarray(result)


def run():
    rows = {}
    for name, shift in [('raw', False), ('positive_zero_reference_shift', True)]:
        x = features(shift); results = [permutation_check(x, 200000, 92153), permutation_check(x, 100000, 92154)]
        ranks = (np.argsort(x[:, 2])[:3]+1).tolist(); n = len(x)
        rows[name] = {'feature_labels': ['SET_return_log10A', 'RESET_outward_log10A', 'log10_RESET_over_SET'],
                      'features': x.tolist(), 'permutation_checks': results,
                      'three_lowest_ratio_cycle_ids_in_rank_order': ranks,
                      'three_lowest_form_a_consecutive_block': max(ranks)-min(ranks) == 2,
                      'exact_rank_block_probability_anywhere_exploratory': (n-2)/math.comb(n, 3),
                      'exact_rank_probability_first_three_exploratory': 1/math.comb(n, 3),
                      'first3_mean_log_ratio': float(x[:3, 2].mean()), 'later50_mean_log_ratio': float(x[3:, 2].mean())}
        print(name, ranks, [(r['joint_monte_carlo_p'], r['joint_exceedances']) for r in results], flush=True)
    paths = [Path(__file__), ROOT/'analysis/CYCLE_ORDER_TEST_DESIGN_2026-09-21.md', ROOT/'analysis/diagnose_cross_polarity_physics.py',
             ROOT/'tests/test_cycle_order_evidence.py', ROOT/'1T1M写入.xlsx', ROOT/'1T1M擦除.xlsx']
    output = {'scope': 'Exploratory joint feature/cut permutation test of cycle-vector exchangeability; no microscopic identification',
              'input_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              'analyses': rows}
    (ROOT/'analysis/cycle_order_evidence.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__': run()
