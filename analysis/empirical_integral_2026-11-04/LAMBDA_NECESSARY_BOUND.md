# Common-lambda necessary bounds from saved empirical integral nodes

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
| linear / 8 | 2.108772542 | 13.083796128 | 14.051249023 | 2.373673522 | 5.122289383 | 2.017208146 |
| linear / 16 | 2.108772754 | 13.083797848 | 14.051249023 | 2.373601620 | 5.126223893 | 2.017506117 |
| pchip / 8 | 2.108491163 | 13.085312921 | 14.051249023 | 2.374097581 | 5.125616091 | 2.016542813 |
| pchip / 16 | 2.108491230 | 13.085309683 | 14.051249023 | 2.373871293 | 5.128754417 | 2.017431337 |

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

- `empirical_integral_diagnostic.json`: `b491cca8e7b0524c57f9093cb814bc2cbb96ab5d0251ee2d4189947d8516e9ba`
- `empirical_integral_arrays.npz`: `a4e00c5292c1c802105481cacf3db80f417ad85a83351e2487f2421403dc7da9`
- Reproduction script: `029fe9f7a07dc3e050a2ab3e8e9e128bc9ef6a568c62b8f18e60837ff15ae1a5`
