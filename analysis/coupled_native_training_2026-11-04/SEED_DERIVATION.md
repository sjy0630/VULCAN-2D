# Accepted one-seed initialization for actual full training

This saves the accepted initialization only. No forward experiment or optimization was run while preparing it. The preparation **did perform two analytic conditional state inverse mappings**; the absence of iterative roots does not make the inverse count zero.

## Frozen scalar material

Source: `analysis/constrained_history_convergence_2026-11-04/requests/102_final_refined.json`.

| Parameter | Value |
|---|---:|
| lambda | 0.4470398752554136 V^-1 |
| kappa_positive | 57.67798793605963 V^-2 |
| J_positive | 0.0037505935471625162 A |
| b | 0.769297378470313 |
| soft Ron | 58.463033350725695 ohm |
| Gamma_c | 0.03127031375421752 eV |

## One new parameter seed

Use reference low-resistance R*=93.4800875524324 ohm, Icc=1 mA, the reported bare SET transition median 2.675 V, and native maximum command 5 V. The exact R* number is also the old table's reference Ron and matches the rounded empirical feature median; it is **not** the current soft-branch Ron.

Define v*=Icc R*=0.0934800875524324 V and L=(5-2.675)+(5-v*)=7.231519912447568 V. With c=Icc/J_positive and E=exp(lambda v*), the positive stationary q² root is

\[
q_* = \frac{2c}{c+\sqrt{c^2+4Ec}}=0.39373690837844827.
\]

Choose equilibrium occupancy x_eq=0.9 and approach fraction 0.9. **Both 90% choices are arbitrary, explicit initialization conventions, not measurements or fitted outcomes.** Approximating q and actual voltage as q*,v* along the nominated compliance path gives

\[
\rho=\frac{e^{2b}q_*}{9(1-q_*)},\quad
A=\frac{1/R_*}{0.9\times0.9},\quad
\kappa_a=\frac{\ln10}{v_*L[e^bq_*+\rho e^{-b}(1-q_*)]}.
\]

| New parameter | Accepted seed |
|---|---:|
| A | 0.013206747378601957 S |
| kappa_a | 3.60746202417315 V^-2 |
| rho | 0.3361287809489076 |

Assigning the low-resistance conductance principally to a is another seed approximation. The soft contribution can make this initialization too conductive. The actual coupled/compliance trajectory must decide; this construction is not a forward solution.

## Three entrance states, each used once

| Continuous segment | u0 | q0 | a0 / S |
|---|---:|---:|---:|
| bare 1-3 | 10.054724499155975 | 4.298219923945171e-5 | 0 |
| bare 5-15 | 8.291769201636164 | 0.00025057076016083026 | 0 |
| series 4-35 | 9.464551672652638 | 7.755279151592979e-5 | 0 |

The two bare mappings use native index 1 (zero-based), V=0.0500000007450581 V. Currents are respectively 7.07061839633205e-11 A and 4.124164898478e-10 A. No zero-current subtraction or floor replacement was applied.

At that voltage, interpolate the original positive log-current table between voltage columns 307 and 308; add `ln(Ron_table/Ron_used)`. Invert this monotone piecewise-linear log-current versus log-Gamma vector analytically. The bracketing Gamma rows are 181/182 for bare1 and 188/189 for bare5. Convert by u=ln(1+Gamma_c/Gamma) and q=exp(-u). The JSON records the exact cell coordinates, all four raw log-current corner values, source keys and resulting Gamma, so these two mappings can be reproduced without a new model call or root search.

Each mapping is conditional on **a=0**. A current point cannot uniquely determine both q and a when the parallel path is active. Using the first 0.05 V state as the zero-voltage entrance is an additional seed approximation; evolution over that first interval has not been inverted. The series entry simply reuses the saved scalar-fit entrance as the common baseline. The old bare u=13.325 from a different model was not transplanted.

The user-confirmed chronology controls this stage: bare1-3 and bare5-15 are separate continuous segments, because missing cycle4 was actually measured. No entry is added to later observed cycles and no missing history is fabricated.

## Next comparison and counts

The root-owned next plan compares a-disabled and a-enabled using these same three entrance pairs, at two precisions each: 4 full evaluations x 92 phases = **368 planned phase calls**, budget **900 s**, **no optimization yet**. Preserve all 27023 original residuals and 20 aggregate features. This document does not claim that any of those forward calls has completed.

Preparation counts: one new parameter seed; **2 analytic conditional inverse mappings**; 0 iterative roots, 0 ODE calls, 0 parameter scans, 0 optimizations, 0 new spectral quadratures. Persistence only extracts the already-used table cells and hashes; it adds no inverse mapping.

Input SHA-256:

- `transport_g010_v0025.npz`: `00d6d9ea0b1dfe855181e7016afc6dd5e23d169f83f19cfcedfc9d3b35f9e4e2`
- `native_observations.npz`: `0aedbe51f41b4f4355e1de4d26048c6790a9d454b8b7d56f51bd76a4671465a8`
- `102_final_refined.json`: `d591b241d98279792b64f1e6a3093db5603f1b3db6be5185a110ac5e312a2b03`
- `feature_constraints.md`: `4249da103a0b77c317800e88a51c291e9f0364d0b7fcf1226ed595c591758956`

Absolute input paths and all arithmetic constants are in `SEED_DERIVATION.json`. Neither empirical feature ranges nor these seed assumptions are noise confidence intervals, microscopic evidence or calibrated physical-time rates.
