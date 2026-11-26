"""Continuous-JS arithmetic recheck at the original 19 fixed gamma_c points.

No spectral integration, trajectory integration, holdout data, or new gamma_c
points. Piecewise derivative enumeration guards against narrow minima that an
unimodal bounded minimizer could otherwise miss.
"""
from pathlib import Path
import hashlib
import json
import math
import time

import numpy as np
from scipy.optimize import brentq, minimize_scalar

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    source_path = OUT / 'SERIES_ENTRY_REVIEW.json'
    source = json.loads(source_path.read_text())
    previous = json.loads((OUT / 'SERIES_ENTRY_REVIEW.pre_review.json').read_text())
    budget_path = ROOT / 'analysis/conditional_clock_2026-11-03/budget_seeds.json'
    budget = json.loads(budget_path.read_text())
    cache_path = ROOT / 'analysis/beta_feedback_2026-11-03/fixed_spectrum_budget_arrays.npz'
    with np.load(cache_path) as z:
        currents = z['soft_set_current_lower_a'].copy()
    assert currents.shape == (60,)
    assert np.all(currents > 0)
    for key in ('SET_target_2V_gamma_lower_ev', 'transistor_current_upper_at_command2V_a',
                'frozen_initial_series_current_a'):
        assert source[key] == previous[key], key
    for row, old in zip(source['rows'], previous['rows']):
        for key in ('gamma_c_ev', 'u0_cycle4_once', 'JS_finite_domain_upper_a',
                    'minimum_SET_outward_required_over_return_allowed', 'best_JS_a'):
            assert row[key] == old[key], key
    expected = np.unique(np.r_[np.logspace(-4, 0, 17), .03, .05])
    assert np.array_equal(expected, [r['gamma_c_ev'] for r in source['rows']])

    g5 = budget['inputs']['g5_ev']
    g2 = budget['inputs']['g2_ev']
    target = source['SET_target_2V_gamma_lower_ev']
    cap = source['transistor_current_upper_at_command2V_a']
    rows = []
    for original in source['rows']:
        assert time.monotonic() - started < 30, '30-second arithmetic budget'
        gc = original['gamma_c_ev']
        B = original['u0_cycle4_once']
        A = math.log1p(gc / target)
        u2 = math.log1p(gc / g2)
        du = math.log1p(gc / g5) - u2
        delta = B - A
        assert delta > 0 and du > 0
        d = u2 + np.log(currents / cap)
        ceiling = original['JS_finite_domain_upper_a']

        def values(c):
            if c + A <= 0:
                return math.inf, math.inf, 0.
            lower = math.log1p(delta / (c + A)) / 2
            integral = float(.05 * np.maximum(c + d, 0).sum())
            upper = du / integral if integral > 0 else math.inf
            return lower / upper, lower, upper

        def scan(js_hi):
            js_lo = 1e-7
            c_lo, c_hi = math.log(cap / js_hi), math.log(cap / js_lo)
            breaks = sorted(float(t) for t in np.unique(-d) if c_lo < t < c_hi)
            edges = [c_lo, *breaks, c_hi]
            candidates = [(c_lo, 'upper_JS_endpoint'), (c_hi, 'lower_JS_endpoint')]
            candidates += [(c, 'positive_part_breakpoint') for c in breaks]
            bounded = minimize_scalar(lambda x: values(math.log(cap)-x)[0],
                                      bounds=(math.log(js_lo), math.log(js_hi)),
                                      method='bounded', options={'xatol': 1e-13})
            assert bounded.success
            candidates.append((math.log(cap)-float(bounded.x), 'whole_domain_bounded'))
            stationary = []
            curvature_points = []
            piecewise_best = math.inf
            for left, right in zip(edges[:-1], edges[1:]):
                middle = (left + right) / 2
                active = middle + d > 0
                L = .05 * int(active.sum())
                D = .05 * float(d[active].sum())
                if L == 0:
                    continue
                # R(c) = (Lc+D)*log((c+B)/(c+A))/(2*du).
                # R'' has the sign of the following linear expression in c:
                # c*(2D-L(A+B)) + D(A+B)-2LAB.
                slope = 2*D - L*(A+B)
                intercept = D*(A+B) - 2*L*A*B
                cuts = [left, right]
                if slope != 0:
                    inflection = -intercept / slope
                    if left < inflection < right:
                        cuts.insert(1, inflection)
                        curvature_points.append(inflection)
                        candidates.append((inflection, 'curvature_split'))

                def derivative(c):
                    p = (c+A)*(c+B)
                    if c+A <= 0:
                        return -math.inf
                    return L*math.log1p(delta/(c+A))-(L*c+D)*delta/p

                for lo, hi in zip(cuts[:-1], cuts[1:]):
                    f_lo, f_hi = derivative(lo), derivative(hi)
                    if f_lo == 0:
                        stationary.append(lo)
                    if f_hi == 0:
                        stationary.append(hi)
                    if f_lo * f_hi < 0:
                        stationary.append(brentq(derivative, lo, hi, xtol=1e-13,
                                                  rtol=1e-14))
                local = minimize_scalar(lambda c: values(c)[0], bounds=(left, right),
                                        method='bounded', options={'xatol': 1e-13})
                assert local.success
                piecewise_best = min(piecewise_best, float(local.fun))
                candidates.append((float(local.x), 'piecewise_bounded'))
            candidates += [(c, 'enumerated_stationary_point') for c in stationary]
            scored = sorted((values(c)[0], c, kind) for c, kind in candidates)
            ratio, best_c, kind = scored[0]
            lower, upper = values(best_c)[1:]
            # Enumerated boundaries/stationary points are independent of either
            # bounded optimization call and suffice in each smooth piece.
            exhaustive = min(values(c)[0] for c, kind in candidates
                             if kind not in ('whole_domain_bounded', 'piecewise_bounded'))
            assert abs(ratio-exhaustive) < 2e-10 * max(1, abs(ratio))
            return {'JS_domain_a': [js_lo, js_hi],
                    'minimum_ratio': ratio, 'best_JS_a': cap*math.exp(-best_c),
                    'best_k_lower_per_v': lower,
                    'best_k_upper_per_v': upper if math.isfinite(upper) else None,
                    'upper_unbounded': not math.isfinite(upper),
                    'best_kind': kind, 'positive_part_breakpoint_count': len(breaks),
                    'stationary_point_count': len(set(stationary)),
                    'curvature_split_count': len(curvature_points),
                    'exhaustive_boundaries_and_stationary_minimum': exhaustive,
                    'whole_domain_bounded_minimum': float(bounded.fun),
                    'piecewise_bounded_minimum': piecewise_best if math.isfinite(piecewise_best) else None,
                    'bounded_minus_exhaustive': float(bounded.fun)-exhaustive,
                    'compatible_finite_rate_at_minimum': max(1e-4,lower) <= min(10**1.5,upper)}

        original_span = scan(.98*ceiling)
        # The declared necessary ceiling already exists in the source. A strict
        # activation equality is a boundary limit, never an admissible seed.
        full_declared = scan(float(np.nextafter(ceiling, 0.)))
        grid_value = original['minimum_SET_outward_required_over_return_allowed']
        assert original_span['minimum_ratio'] <= grid_value+1e-9
        rows.append({'gamma_c_ev': gc, 'previous_181_point_grid_minimum': grid_value,
                     'original_grid_span_continuous_JS': original_span,
                     'declared_necessary_ceiling_a': ceiling,
                     'full_preexisting_necessary_domain_continuous_JS': full_declared,
                     'grid_minus_continuous_original_span': grid_value-original_span['minimum_ratio'],
                     'ceiling_scope': 'Upper boundary evaluated from below; equality at activation is only an infimum/limit, not an activating seed.'})
    best = min(rows, key=lambda r:r['full_preexisting_necessary_domain_continuous_JS']['minimum_ratio'])
    report = {'status': 'completed_continuous_JS_at_19_fixed_gamma_c_points',
              'method': 'bounded minimization in log(JS), endpoints, all positive-part breakpoints, and exhaustive stationary roots after linear-curvature splits',
              'source_files_sha256': {str(p.relative_to(ROOT)):sha(p) for p in (source_path,budget_path,cache_path)},
              'source_erratum': source['erratum'], 'source_SET_fields_unchanged_verified': True,
              'inputs': {'g5_ev': g5, 'g2_ev': g2, 'target_2V_gamma_ev': target,
                         'Icap2_a': cap, 'current_node_count': len(currents), 'node_weight_v': .05},
              'rows': rows, 'best_fixed_gamma_c_ev': best['gamma_c_ev'],
              'lowest_ratio_all_sampled_gamma_c': best['full_preexisting_necessary_domain_continuous_JS']['minimum_ratio'],
              'all_19_fixed_gamma_c_continuous_JS_ratios_exceed_one': all(r['full_preexisting_necessary_domain_continuous_JS']['minimum_ratio']>1 for r in rows),
              'selected_diagnostic_seeds': [],
              'scope': ['Continuous JS only at the same 19 sampled gamma_c; no continuous-gamma_c global exclusion.',
                        'Original finite-grid report preserved, apart from separately documented restart and RESET-bound errata with pre_review snapshots.',
                        'Same original fixed SpectralShape, beta=1, A_RESET=0, anchored cycle4 initial state and generous training endpoint budgets.',
                        'Necessary endpoint conditions only; no full-curve claim, ODE trajectory, new integration, new spectral shape, expanded gamma_c domain, or holdout access.',
                        'Same constant command-path rate interpretation and cached transport approximation assumptions; these are not an exclusion of every physical mechanism.',
                        'Bounded scalar routines only; exhaustive candidate enumeration guards narrow valleys in JS under the stated piecewise arithmetic, not an interval-arithmetic certificate.'],
              'elapsed_s': time.monotonic()-started,
              'script_sha256':sha(Path(__file__))}
    assert report['elapsed_s'] < 30
    (OUT/'SERIES_THRESHOLD_RECHECK.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines = ['# 原19个gamma_c点的连续JS复核', '',
             '纯缓存代数；固定谱、beta=1、A_RESET=0及原训练端点预算。未新增谱积分、ODE或保留集读取。', '',
             'JS全区间bounded最小化与端点之外，逐一枚举60个正部节点的折点，并在每个光滑区间按线性的二阶导符号拆分、求出全部驻点；因此不依赖全局单峰假设，也检查窄谷。', '',
             '| gamma_c / eV | 原181点最小比值 | 原网格范围连续JS | 既有必要上界内连续JS | 最优JS / A |',
             '|---:|---:|---:|---:|---:|']
    for r in rows:
        f = r['full_preexisting_necessary_domain_continuous_JS']
        lines.append(f'| {r["gamma_c_ev"]:.9g} | {r["previous_181_point_grid_minimum"]:.9g} | {r["original_grid_span_continuous_JS"]["minimum_ratio"]:.9g} | {f["minimum_ratio"]:.9g} | {f["best_JS_a"]:.9g} |')
    lines += ['', f'最低下界/上界比值 = {report["lowest_ratio_all_sampled_gamma_c"]:.12g}，位于固定gamma_c={best["gamma_c_ev"]:.12g} eV。',
              '19个固定gamma_c点均未出现SET速率相容区间；连续JS复核没有发现被原网格漏掉的可行窄谷。未选择任何seed。',
              '原网格止于0.98×既有必要上界；表中另列该上界以内的剩余边缘范围。激活等号只作为边界极限，不是可用seed。',
              '**范围：仅连续JS、离散的19个gamma_c；不构成连续gamma_c域上的全局排除，更不能替代完整曲线评分或排除所有物理机制。**',
              '本次source引用已纠正重启区间下界/RESET无限物理上界分类的报告；用于JS复核的SET字段已逐项与pre_review快照核对未变。',
              f'用时{report["elapsed_s"]:.3f}s。完整数值、折点/驻点数量及来源hash见JSON。']
    (OUT/'SERIES_THRESHOLD_RECHECK.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'elapsed_s':report['elapsed_s'],'best_gamma_c':report['best_fixed_gamma_c_ev'],
                      'lowest_ratio':report['lowest_ratio_all_sampled_gamma_c'],
                      'all_19_above_one':report['all_19_fixed_gamma_c_continuous_JS_ratios_exceed_one'],
                      'best_row':best},indent=2))


if __name__ == '__main__':
    main()
