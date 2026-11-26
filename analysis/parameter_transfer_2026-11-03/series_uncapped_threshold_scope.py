"""Remove only the bare-Icc-derived JS upper cap in a series-only audit.

Reuse series_threshold_recheck.py's complete breakpoint/stationary enumeration.
No new gamma_c, spectrum, circuit evaluation, trajectory, or fit.  All inputs
are saved series budgets and current arrays; old outputs remain untouched.
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


def scan_threshold_interval(B, A, du, d, cap, js_low, js_high):
    """Closed arithmetic interval; a physical upper endpoint is a limit only."""
    delta = B - A
    assert delta > 0 and du > 0 and 0 < js_low < js_high
    c_low, c_high = math.log(cap / js_high), math.log(cap / js_low)
    assert c_low + A > 0

    def values(c):
        lower = math.log1p(delta / (c + A)) / 2.0
        integral = float(0.05 * np.maximum(c + d, 0).sum())
        upper = du / integral if integral > 0 else math.inf
        return lower / upper, lower, upper

    def enumerate_minimum(left_end, right_end):
        breaks = sorted(float(t) for t in np.unique(-d) if left_end < t < right_end)
        edges = [left_end, *breaks, right_end]
        candidates = [(left_end, 'upper_JS_endpoint'), (right_end, 'lower_JS_endpoint')]
        candidates += [(c, 'positive_part_breakpoint') for c in breaks]
        stationary, curvature_points = [], []
        for left, right in zip(edges[:-1], edges[1:]):
            active = (left + right) / 2.0 + d > 0
            L = 0.05 * int(active.sum())
            D = 0.05 * float(d[active].sum())
            if L == 0:
                continue
            # As in series_threshold_recheck.py: R'' has the sign of this
            # linear polynomial, so R' is monotone on every resulting cut.
            slope = 2.0 * D - L * (A + B)
            intercept = D * (A + B) - 2.0 * L * A * B
            cuts = [left, right]
            if slope != 0:
                inflection = -intercept / slope
                if left < inflection < right:
                    cuts.insert(1, inflection)
                    curvature_points.append(inflection)
                    candidates.append((inflection, 'curvature_split'))

            def derivative(c):
                return (L * math.log1p(delta / (c + A))
                        - (L * c + D) * delta / ((c + A) * (c + B)))

            for lo, hi in zip(cuts[:-1], cuts[1:]):
                flo, fhi = derivative(lo), derivative(hi)
                if flo == 0:
                    stationary.append(lo)
                if fhi == 0:
                    stationary.append(hi)
                if flo * fhi < 0:
                    stationary.append(brentq(derivative, lo, hi, xtol=1e-13, rtol=1e-14))
        candidates += [(c, 'enumerated_stationary_point') for c in stationary]
        best_ratio, best_c, kind = min((values(c)[0], c, kind) for c, kind in candidates)
        # Independent numerical cross-check, never the completeness argument.
        bounded = minimize_scalar(lambda c: values(c)[0], bounds=(left_end, right_end),
                                  method='bounded', options={'xatol': 1e-13})
        assert bounded.success
        assert best_ratio <= float(bounded.fun) + 2e-10 * max(1.0, abs(best_ratio))
        _, lower, upper = values(best_c)
        return {
            'ratio_infimum_over_interval_closure': best_ratio,
            # Clamp a floating-point round-trip to the declared closure.
            'best_JS_a': min(js_high, max(js_low, cap * math.exp(-best_c))),
            'best_log_cap_over_JS': best_c,
            'best_kind': kind,
            'best_k_lower_per_v': lower,
            'best_k_upper_per_v': upper if math.isfinite(upper) else None,
            'upper_unbounded': not math.isfinite(upper),
            'breakpoint_count': len(breaks),
            'stationary_point_count': len(set(stationary)),
            'curvature_split_count': len(curvature_points),
            'whole_domain_bounded_minimum': float(bounded.fun),
            'bounded_minus_enumerated': float(bounded.fun) - best_ratio,
            'closure_budget_overlap_without_k_box': best_ratio <= 1.0,
            'budget_overlap_with_k_box_at_this_minimum': max(1e-4, lower) <= min(10**1.5, upper),
        }

    unrestricted = enumerate_minimum(c_low, c_high)
    # The two extra box inequalities are monotone in c. Restrict the c-domain
    # first, then minimize the same ratio there; this tests existence over the
    # whole interval, rather than checking only the unrestricted minimizer.
    k_min, k_max = 1e-4, 10**1.5
    boxed_c_low = max(c_low, delta / math.expm1(2.0 * k_max) - A)
    boxed_c_high = c_high
    allowed_integral = du / k_min

    def integral(c):
        return float(0.05 * np.maximum(c + d, 0).sum())

    boxed = None
    if boxed_c_low <= boxed_c_high and integral(boxed_c_low) <= allowed_integral:
        if integral(boxed_c_high) > allowed_integral:
            boxed_c_high = brentq(lambda c: integral(c) - allowed_integral,
                                 boxed_c_low, boxed_c_high, xtol=1e-13, rtol=1e-14)
        boxed = enumerate_minimum(boxed_c_low, boxed_c_high)
    return {
        'JS_interval_closure_a': [js_low, js_high],
        'upper_equality_is_not_an_activating_seed': True,
        'without_algorithm_k_box': unrestricted,
        'with_original_algorithm_k_box': boxed,
        'original_k_box_per_v': [k_min, k_max],
        'any_budget_overlap_without_k_box': unrestricted['closure_budget_overlap_without_k_box'],
        'any_budget_overlap_with_original_k_box': bool(boxed and boxed['closure_budget_overlap_without_k_box']),
    }


def main():
    started = time.monotonic()
    source_path = OUT / 'SERIES_ENTRY_REVIEW.json'
    original_check_path = OUT / 'SERIES_THRESHOLD_RECHECK.json'
    algorithm_path = OUT / 'series_threshold_recheck.py'
    budget_path = ROOT / 'analysis/conditional_clock_2026-11-03/budget_seeds.json'
    cache_path = ROOT / 'analysis/beta_feedback_2026-11-03/fixed_spectrum_budget_arrays.npz'
    source = json.loads(source_path.read_text())
    original_check = json.loads(original_check_path.read_text())
    budget = json.loads(budget_path.read_text())
    with np.load(cache_path) as z:
        currents = z['soft_set_current_lower_a'].copy()
    assert currents.shape == (60,) and np.all(currents > 0)
    expected = np.unique(np.r_[np.logspace(-4, 0, 17), 0.03, 0.05])
    assert np.array_equal(expected, [r['gamma_c_ev'] for r in source['rows']])
    g5, g2 = budget['inputs']['g5_ev'], budget['inputs']['g2_ev']
    target = source['SET_target_2V_gamma_lower_ev']
    cap = source['transistor_current_upper_at_command2V_a']
    initial_i2 = source['frozen_initial_series_current_a']['command2V']
    rows = []
    for original, check in zip(source['rows'], original_check['rows']):
        assert time.monotonic() - started < 30.0
        gc = original['gamma_c_ev']
        assert gc == check['gamma_c_ev']
        B, A = original['u0_cycle4_once'], math.log1p(gc / target)
        u2 = math.log1p(gc / g2)
        du = math.log1p(gc / g5) - u2
        d = u2 + np.log(currents / cap)
        activation = initial_i2 * math.exp(B)
        arrival = cap * math.exp(A)
        assert math.isclose(activation, original['JS_activation_by_2V_upper_a'], rel_tol=1e-14)
        physical_ceiling = min(activation, arrival)
        old_ceiling = original['JS_finite_domain_upper_a']
        full = scan_threshold_interval(B, A, du, d, cap, 1e-7, physical_ceiling)
        omitted = (scan_threshold_interval(B, A, du, d, cap, old_ceiling, physical_ceiling)
                   if physical_ceiling > old_ceiling else None)
        above_1ma = (scan_threshold_interval(B, A, du, d, cap, 1e-3, physical_ceiling)
                     if physical_ceiling > 1e-3 else None)
        rows.append({
            'gamma_c_ev': gc,
            'old_algorithm_JS_ceiling_a': old_ceiling,
            'actual_anchor_activation_JS_ceiling_a': activation,
            'target_arrival_JS_ceiling_a': arrival,
            'uncapped_series_JS_ceiling_a': physical_ceiling,
            'old_capped_continuous_minimum_ratio': check['full_preexisting_necessary_domain_continuous_JS']['minimum_ratio'],
            'full_upper_relaxed_domain': full,
            'previously_omitted_upper_interval': omitted,
            'strictly_above_1mA_interval_closure': above_1ma,
        })
    best = min(rows, key=lambda r: r['full_upper_relaxed_domain']['without_algorithm_k_box']['ratio_infimum_over_interval_closure'])
    report = {
        'status': 'completed_series_only_uncapped_JS_upper_scope_audit_no_ODE_no_fit',
        'fixed_beta': 1.0,
        'fixed_A_RESET': 0.0,
        'n_original_gamma_c_values': len(rows),
        'n_gamma_c_with_previous_upper_truncation': sum(r['previously_omitted_upper_interval'] is not None for r in rows),
        'n_budget_compatible_without_k_box': sum(r['full_upper_relaxed_domain']['any_budget_overlap_without_k_box'] for r in rows),
        'n_budget_compatible_with_original_k_box': sum(r['full_upper_relaxed_domain']['any_budget_overlap_with_original_k_box'] for r in rows),
        'best_gamma_c_ev': best['gamma_c_ev'],
        'minimum_ratio': best['full_upper_relaxed_domain']['without_algorithm_k_box']['ratio_infimum_over_interval_closure'],
        'inputs': {'g5_ev': g5, 'g2_ev': g2, 'target_2V_gamma_ev': target,
                   'Icap2_a': cap, 'frozen_anchor_I2_a': initial_i2,
                   'JS_lower_bound_unchanged_a': 1e-7,
                   'current_node_count': len(currents), 'node_weight_v': 0.05},
        'rows': rows,
        'method': 'Same positive-part breakpoints and all stationary roots after linear-curvature splits as series_threshold_recheck.py; independent bounded numerical cross-check. Algorithm-k-box existence is checked after intersecting its monotone constraints in c=ln(Icap2/JS).',
        'scope': [
            'Only removes the bare-Icc-derived upper cap on series-only JS. It does not enlarge or validate the joint shared-parameter model.',
            'Same 19 discrete gamma_c points, beta1, fixed numerical spectrum/T mapping, 0.2V observation anchor and endpoint budgets; no continuous-gamma_c exclusion.',
            'JS lower boundary remains the original 1e-7 A; no claim about omitted smaller thresholds is newly made here.',
            'Physical upper equality is included only as a favorable closure/infimum. A zero initial drive at the activation boundary is not an admissible activating seed.',
            'The 2V outward budget generously allows 2V path after the 0.2V observation anchor; return endpoints retain old 5V Q10/1.05 and 2V Q90 conventions.',
            'Positive finite k without algorithm bounds and the original [1e-4,31.6227766] per-command-V box are distinguished. Even any necessary-budget overlap would not validate a full trajectory.',
            'JS above 1mA, if allowed by series budgets, could only establish independent-series mathematical room; bare compliance/read-state constraints must be intersected separately for shared parameters.',
            'No new spectral quadrature, ODE, circuit solve, fit, data gate, timing assumption, or holdout access. Frozen cache arithmetic is not an interval-arithmetic certificate.',
        ],
        'source_files_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [source_path, original_check_path, algorithm_path, budget_path, cache_path]},
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'elapsed_s': time.monotonic() - started,
    }
    assert report['elapsed_s'] < 30.0
    (OUT / 'SERIES_UNCAPPED_THRESHOLD_SCOPE.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    lines = ['# series 单组 JS 上界补查：移除裸1 mA算法截断', '',
             '**原19个 gamma_c 点均仍不相容，且不限制 k 的算法范围时也不相容。**',
             '只读取已有缓存，beta=1、原谱/晶体管/0.2 V观测锚和端点预算不变。旧报告不覆盖。', '',
             '独立series合法必要上界取 `min[I0,2 exp(u_anchor), Icap,2 exp(u_target2)]`；不再与裸1 mA取小值。JS下界仍保留原1e-7 A。', '',
             '| gamma_c/eV | 原JS上界/mA | 补查JS上界/mA | 原连续JS最小比 | 补查最小比 | 去掉k盒有交集 | 原k盒有交集 |',
             '|---:|---:|---:|---:|---:|:---:|:---:|']
    for r in rows:
        f = r['full_upper_relaxed_domain']
        lines.append(f'| {r["gamma_c_ev"]:.9g} | {r["old_algorithm_JS_ceiling_a"]*1e3:.9g} | {r["uncapped_series_JS_ceiling_a"]*1e3:.9g} | {r["old_capped_continuous_minimum_ratio"]:.9g} | {f["without_algorithm_k_box"]["ratio_infimum_over_interval_closure"]:.9g} | {"是" if f["any_budget_overlap_without_k_box"] else "否"} | {"是" if f["any_budget_overlap_with_original_k_box"] else "否"} |')
    f = best['full_upper_relaxed_domain']['without_algorithm_k_box']
    lines += ['', f'原5个 gamma_c≥0.1 eV 的域曾被1 mA截断。补查后的全表最低下/上界比为 {report["minimum_ratio"]:.12g}，位于 gamma_c={best["gamma_c_ev"]:.9g} eV、JS趋近 {f["best_JS_a"]*1e3:.9g} mA。',
              f'此处要求 k≥{f["best_k_lower_per_v"]:.9g}，同时 k≤{f["best_k_upper_per_v"]:.9g}（均为每命令路径V），所以冲突先于算法k范围限制。',
              '物理上界等号只作为最有利的边界极限参与计算；初锚启动驱动等于0并非可用起点。',
              '', '算法沿用原复核：枚举全部正部折点，在每个光滑段按二阶导线性符号拆分，再求全部驻点；不依赖整域单峰假设。另对原k盒约束先作单调域交集，再独立最小化，避免只在某个最优点检查k。',
              '', '**结论范围：** 此次确认原1 mA上界确实不属于独立series的物理限制，但移除它仍未解除这19个固定gamma_c点的SET必要预算冲突。不能把它表述为连续gamma_c全域排除或完整曲线拟合失败证明。',
              '这也不是扩展联合模型可行域：即使series在JS>1 mA出现空间，迁移到裸器件时仍须重新满足裸限流/低阻读态条件。',
              '0.2 V锚不自动等于真实0 V初态；2 V预算保留有利的2 V路径，返扫端点仍为5 V Q10/1.05与2 V原Q90。冻结数值缓存仍有既定精度范围，本次未认证新的谱误差。',
              f'用时 {report["elapsed_s"]:.3f} s。完整被截区间、JS>1 mA区间、折点/驻点及来源hash见同名JSON。']
    (OUT / 'SERIES_UNCAPPED_THRESHOLD_SCOPE.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: report[k] for k in ['elapsed_s', 'n_original_gamma_c_values', 'n_gamma_c_with_previous_upper_truncation', 'n_budget_compatible_without_k_box', 'n_budget_compatible_with_original_k_box', 'best_gamma_c_ev', 'minimum_ratio']}, indent=2))


if __name__ == '__main__':
    main()
