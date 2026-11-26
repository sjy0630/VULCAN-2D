#!/usr/bin/env python3
"""Algebraic lambda bounds from saved nodes only; no lambda evaluations or model calls."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    diagnostic_path = ROOT / "empirical_integral_diagnostic.json"
    arrays_path = ROOT / "empirical_integral_arrays.npz"
    before = {p.name: sha256(p) for p in (diagnostic_path, arrays_path)}
    diagnostic = json.loads(diagnostic_path.read_text())
    lambda0 = diagnostic["fixed_material"]["lambda_per_v"]
    variants = diagnostic["window_variants"]
    configurations = []
    saved_nodes_used = 0

    with np.load(arrays_path, allow_pickle=False) as arrays:
        for interpolant in ("linear", "pchip"):
            for gauss_order in (8, 16):
                def read_window(window_id: str) -> dict:
                    nonlocal saved_nodes_used
                    matches = [w for w in variants if
                               w["window_id"] == window_id and
                               w["interpolant"] == interpolant and
                               w["gauss_order"] == gauss_order]
                    assert len(matches) == 1
                    window = matches[0]
                    prefix = window["array_prefix"]
                    v = np.abs(arrays[prefix + "local_voltage_v"])
                    weights = arrays[prefix + "weights_path_v"]
                    # F is already saved at lambda0. No new exponential is evaluated.
                    mass = weights * arrays[prefix + "F"]
                    assert np.all(v > 0) and np.all(weights > 0) and np.all(mass > 0)
                    assert np.all(arrays[prefix + "completed_mask"])
                    saved_a = float(window["A_v2"])
                    mass_sum = float(np.sum(mass))
                    assert math.isclose(mass_sum, saved_a, rel_tol=1e-12, abs_tol=0)
                    saved_nodes_used += int(v.size)
                    return {
                        "window_id": window_id,
                        "saved_node_count": int(v.size),
                        "A_at_lambda0_v2": saved_a,
                        "B_a_v2": float(window["B_a_v2"]),
                        "y_delta_u": float(window["delta_u"]),
                        "local_abs_voltage_support_v": [float(v.min()), float(v.max())],
                        "A_weighted_voltage_mean_at_lambda0_v": float(np.dot(mass, v) / mass_sum),
                        "saved_A_vs_weighted_F_relative_difference": (mass_sum - saved_a) / saved_a,
                    }

                pairs = {}
                for name, low_id, high_id in (
                    ("reset_outward", "reset4_outward_0p2_to_0p8", "reset4_outward_0p8_to_1p6"),
                    ("set_return", "set4_return_0p8_to_0p2", "set4_return_1p6_to_0p8"),
                ):
                    low, high = read_window(low_id), read_window(high_id)
                    r0 = high["A_at_lambda0_v2"] / low["A_at_lambda0_v2"]
                    s = high["B_a_v2"] / low["B_a_v2"]
                    t = high["y_delta_u"] / low["y_delta_u"]
                    vl_min, vl_max = low["local_abs_voltage_support_v"]
                    vh_min, vh_max = high["local_abs_voltage_support_v"]
                    mu_l = low["A_weighted_voltage_mean_at_lambda0_v"]
                    mu_h = high["A_weighted_voltage_mean_at_lambda0_v"]
                    support_gap = vh_min - vl_max
                    d_upper_support = vh_max - vl_min
                    d_upper_jensen = vh_max - mu_l
                    d_lower_jensen = mu_h - vl_max
                    assert support_gap > 0 and d_lower_jensen > 0 and s > r0
                    lower = lambda0 + math.log(s / r0) / d_upper_jensen
                    sufficient = lambda0 + math.log(s / r0) / d_lower_jensen
                    pair = {
                        "low_window": low,
                        "high_window": high,
                        "R0_A_high_over_low": r0,
                        "S_B_high_over_low": s,
                        "T_y_high_over_low": t,
                        "T_minus_S_not_an_experimental_error_margin": t - s,
                        "strict_support_gap_v": support_gap,
                        "R_strictly_increasing_for_all_real_lambda_on_saved_nodes": True,
                        "D_upper_support_v": d_upper_support,
                        "D_upper_jensen_v": d_upper_jensen,
                        "D_lower_jensen_v": d_lower_jensen,
                        "strict_necessary_lambda_lower_bound_support_per_v": lambda0 + math.log(s / r0) / d_upper_support,
                        "strict_necessary_lambda_lower_bound_jensen_per_v": lower,
                        "lambda_above_this_suffices_for_R_greater_than_S_per_v": sufficient,
                    }
                    if name == "reset_outward":
                        assert high["y_delta_u"] > low["y_delta_u"] > 0 and t > s
                        pair.update({
                            "positive_coefficient_exact_condition": "S < R(lambda) < T",
                            "strict_necessary_lambda_upper_bound_jensen_per_v": lambda0 + math.log(t / r0) / d_lower_jensen,
                            "T_crossing_lambda_lower_bound_jensen_per_v": lambda0 + math.log(t / r0) / d_upper_jensen,
                            "original_lambda_interval_0_to_2_excluded_conditionally": lower > 2,
                        })
                    else:
                        assert high["y_delta_u"] > 0 > low["y_delta_u"]
                        pair["positive_coefficient_exact_condition"] = "R(lambda) > S"
                        pair["J_sign_interval"] = "B_high/A_high(lambda) < J < B_low/A_low(lambda)"
                    pairs[name] = pair

                reset = pairs["reset_outward"]
                ret = pairs["set_return"]
                configurations.append({
                    "interpolant": interpolant,
                    "gauss_order": gauss_order,
                    "pairs": pairs,
                    "reset_necessary_lower_bound_exceeds_set_return_sign_sufficient_bound":
                        reset["strict_necessary_lambda_lower_bound_jensen_per_v"] >
                        ret["lambda_above_this_suffices_for_R_greater_than_S_per_v"],
                })

    after = {p.name: sha256(p) for p in (diagnostic_path, arrays_path)}
    assert before == after
    assert len(configurations) == 4 and saved_nodes_used == 6720
    lower_bounds = [c["pairs"]["reset_outward"]["strict_necessary_lambda_lower_bound_jensen_per_v"] for c in configurations]
    upper_bounds = [c["pairs"]["reset_outward"]["strict_necessary_lambda_upper_bound_jensen_per_v"] for c in configurations]
    result = {
        "status": "completed_saved_node_algebra_only",
        "lambda0_per_v": lambda0,
        "inputs_sha256": before,
        "inputs_unchanged": before == after,
        "reproduction_script_sha256": sha256(Path(__file__).resolve()),
        "work_counts": {
            "saved_configurations": 4, "selected_window_variants": 16,
            "saved_nodes_used": saved_nodes_used,
            "new_lambda_point_evaluations": 0, "lambda_scans": 0,
            "roots": 0, "ODEs": 0, "state_inversions": 0,
            "static_circuit_solves": 0, "optimizations": 0,
            "new_quadrature_nodes": 0,
        },
        "scope": "Finite positive quadrature sums on the saved conditionally reconstructed trajectories; only lambda is varied symbolically, while readout, u, local voltage, B, y and path clock stay fixed.",
        "configurations": configurations,
        "cross_configuration_arithmetic_ranges_not_uncertainty_intervals": {
            "reset_necessary_lower_lambda_bound_per_v": [min(lower_bounds), max(lower_bounds)],
            "reset_necessary_upper_lambda_bound_per_v": [min(upper_bounds), max(upper_bounds)],
        },
        "limits": [
            "These are conservative necessary bounds, not numerical values of the lambda crossing roots or a sufficient interval for the complete model.",
            "The four-configuration range is interpolation/quadrature sensitivity, not an experimental-noise confidence interval or a true-between-sample-trajectory error bound.",
            "The low-voltage SET-return delta_u is small and its sign may depend on unknown experimental/readout uncertainty.",
            "The RESET condition S<R<T assumes y_high>y_low>0 and T>S. Both reconstructed RESET delta_u values, especially the small low-window value, have unknown measurement/readout uncertainty; T-S is not an error margin.",
            "Necessary lambda bounds are not recommended fit parameters, inferred physical values, or an uncertainty certificate.",
            "Original lambda in [0,2] is a previously used fitting box, not an independently established physical bound.",
            "Positive coefficients for either pair do not establish common kappa/r, shared b, all-window closure, forward agreement, or the full 1R/1T1R model.",
            "Unknown clock/scan-rate history and changes to the frozen readout can change these conditions; no unconditional mechanism rejection is made.",
            "A is symbolically reweighted along the fixed reconstructed path; no assertion is made that a forward solution stays on that path after lambda changes.",
        ],
    }
    (ROOT / "LAMBDA_NECESSARY_BOUND.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    rows = []
    for c in configurations:
        r, p = c["pairs"]["reset_outward"], c["pairs"]["set_return"]
        rows.append(f"| {c['interpolant']} / {c['gauss_order']} | {r['R0_A_high_over_low']:.9f} | {r['S_B_high_over_low']:.9f} | {r['T_y_high_over_low']:.9f} | {r['strict_necessary_lambda_lower_bound_jensen_per_v']:.9f} | {r['strict_necessary_lambda_upper_bound_jensen_per_v']:.9f} | {p['lambda_above_this_suffices_for_R_greater_than_S_per_v']:.9f} |")
    report = r"""# Common-lambda necessary bounds from saved empirical integral nodes

## Result and scope

On the fixed saved reconstructed trajectories, RESET4's two outward windows require **lambda > 2.37360 to 2.37410 V^-1**, depending on the interpolation/quadrature configuration, for both rate coefficients to be positive. Thus allowing lambda to vary within the former **[0, 2] fitting box** cannot repair this particular exact two-window conflict. That box is not an independently measured physical bound.

The upper necessary bounds are about 5.12 V^-1. These lower/upper bounds delimit a conservative enclosure; they do **not** identify the actual feasible interval, prove a common solution across windows, or provide a forward fit. No new lambda point, root, inversion, circuit solve or ODE was evaluated.

Moreover, the RESET necessary lower bound is already above a sufficient bound for removing the SET4 two-return-window **sign** conflict. Consequently, the fixed-lambda conflict alone does not justify claiming an extra state is required. Stronger field dependence remains a candidate to test, subject to physical justification and all-window/shared-coefficient validation.

## Algebra

For one saved window let c_j = w_j v_j exp(-u_j) > 0 and

    A_w(lambda) = sum_j c_j exp(lambda v_j),
    B_w and y_w = Delta u_w remain fixed.

The script does not evaluate this sum at a new lambda. It uses the saved masses w_j F_j(lambda0), supports and weighted voltage means at lambda0 = 0.4470398752554136 V^-1.

Use L/H for the lower/higher local-voltage windows and define

    R(lambda) = A_H(lambda) / A_L(lambda),
    S = B_H / B_L,  T = y_H / y_L.

For RESET, y_H > y_L > 0 and T > S. Solving y = alpha A - beta B gives

    alpha = (y_L/A_L) (T-S)/(R-S),
    beta  = (y_L/B_L) (T-R)/(R-S).

Therefore, under the stated y signs and T>S premise, alpha > 0 and beta > 0 **if and only if S < R(lambda) < T**. At R=S no finite solution exists because T != S; at R=T beta=0 and strict positivity fails. The current R0 is about 2.109, below S about 13.084 and T about 14.051.

The two RESET reconstructed changes are y_L=0.047940943168394146 and y_H=0.6736301308747978. T-S is approximately 0.96745 (linear) or 0.96594 (PCHIP); exact differences are saved per configuration. Both changes, especially the small y_L, may be sensitive to unknown measurement/readout errors. If these errors change T>S, the positive-coefficient criterion must be rederived. The observed T-S difference is not an experimental error margin.

For SET return, y_H > 0 > y_L. A positive kappa/r pair matching both windows exists **if and only if R(lambda) > S**. Its sign condition is equivalently

    B_H/A_H(lambda) < J=kappa/r < B_L/A_L(lambda).

This condition does not require the fitted pair to equal the positive coefficients identified from SET outward.

### Monotonicity and conservative bounds

For all four saved configurations, min(v_H) > max(v_L). For the finite positive sums,

    d log A_w / d lambda = mu_w(lambda),
    min(v_w) <= mu_w(lambda) <= max(v_w).

Thus d log R / d lambda > 0 for all real lambda. Since R(lambda0) < S, all lambda <= lambda0 are excluded. For delta=lambda-lambda0 >= 0, positivity and Jensen's inequality give

    exp(delta mu_w,0) <= A_w(lambda)/A_w(lambda0)
                         <= exp(delta v_w,max).

Combining the numerator and denominator inequalities yields

    R0 exp(D_lower delta) <= R(lambda) <= R0 exp(D_upper delta),
    D_lower = mu_H,0 - v_L,max > 0,
    D_upper = v_H,max - mu_L,0 > 0.

Consequently R>S requires

    lambda > lambda0 + log(S/R0)/D_upper,

and R<T requires

    lambda < lambda0 + log(T/R0)/D_lower.

The coarser support-only lower bound uses D_upper = v_H,max-v_L,min and gives RESET lambda > about 1.879 V^-1. Using the saved mean tightens it to > about 2.374 V^-1 without evaluating any new lambda.

For SET return, lambda > lambda0+log(S/R0)/D_lower is sufficient for R>S. This bound is about 2.017 V^-1, below the RESET necessary lower bound in every configuration. This proves only the return-window sign condition is automatically compatible at any lambda that could satisfy RESET's positive-coefficient condition.

## Saved-node arithmetic

All lambda bounds below are in V^-1. The last column is a sufficient threshold for SET-return sign compatibility, not for its magnitude/shared-coefficient closure.

| Configuration | RESET R0 | RESET S | RESET T | RESET necessary lower | RESET necessary upper | SET-return sign sufficient above |
|---|---:|---:|---:|---:|---:|---:|
"""
    report += "\n".join(rows)
    report += r"""

For linear/Gauss16, RESET local-voltage supports are [0.190699654, 0.707856660] and [0.708074715, 1.465242185] V, with mu_L,0=0.517819389 V and mu_H,0=1.113184840 V. Every window's support, mean, A/B/y and bounds for all four configurations are recorded in the JSON.

## Interpretation and next finite diagnostic

First determine whether expanding the physically defensible field coefficient beyond the earlier optimization box is justified. If so, a separately authorized finite diagnostic can reuse these saved nodes, bracket the two monotonic RESET crossings R=S and R=T, then check common positive kappa/r and the b relation against the remaining training windows. Only surviving candidates should receive continuous forward validation. No such search or new-model call is part of this artifact.

This distinguishes changing field dependence from adding states. It does not identify a microscopic process, and matching two reconstructed windows is insufficient evidence for the full unified model.

## Limitations and audit

- Frozen a=0 material readout, reconstructed u and local voltage, B/y, interpolation, and common command-path clock are assumed throughout. Varying lambda here does not re-solve the forward trajectory.
- The four-configuration spread measures only the selected interpolation/quadrature sensitivity. It is not experimental uncertainty, a noise bound, or a certificate for the unknown continuous trajectory.
- SET's low-return Delta u is small; unknown measurement/readout uncertainty can alter its sign.
- Both RESET Delta u values have unknown measurement/readout uncertainty, particularly the small low-window value. The RESET interval criterion assumes the observed T>S ordering remains valid.
- The necessary lambda bounds are not recommended parameters, experimentally identified values, or an error certificate.
- Shared b, common positive coefficients across all windows, training residual tolerance, forward accuracy and the complete 1R/1T1R objective remain unchecked by these bounds.
- Read-only arithmetic used 4 saved configurations, 16 window variants and 6720 saved node incidences. All new lambda point evaluations, scans, roots, ODEs, inversions, circuit solves, optimizations and quadrature-node generations: **0**. Both source input hashes were unchanged after the arithmetic.

Reproduce only this saved-node arithmetic and rewrite these two reports:

    .venv/bin/python -B analysis/empirical_integral_2026-11-04/lambda_necessary_bound.py

Input SHA-256 values:
"""
    for name, digest in before.items():
        report += f"\n- `{name}`: `{digest}`"
    report += f"\n- Reproduction script: `{result['reproduction_script_sha256']}`\n"
    (ROOT / "LAMBDA_NECESSARY_BOUND.md").write_text(report)
    print(json.dumps({"status": result["status"], "inputs_unchanged": result["inputs_unchanged"],
                      "work_counts": result["work_counts"],
                      "ranges": result["cross_configuration_arithmetic_ranges_not_uncertainty_intervals"]}, indent=2))


if __name__ == "__main__":
    main()
