# Scientific scope of the four-parameter full-training calibration

Proceed with the actual full-training fit of **A, kappa_a, rho and one shared b**. Keep the scalar lambda/kappa_positive/J_positive, frozen readout, three declared segment entrance pairs, native histories, original masks/weights and all 20 feature definitions fixed. No q¹ substitution or new state law is introduced in this stage. This note adds no ODE, inverse mapping, spectral/current evaluation or parameter scan.

## Why b belongs in this calibration

For a negative-polarity RESET, nonnegative total-current capture gives

\[
q'\ge-\kappa_+e^{2b}v e^{\lambda v}q^2,
\qquad
q_{r,j}\ge\left[q_{0,j}^{-1}+\kappa_+e^{2b}D_j\right]^{-1}.
\]

Here q_0,j is the **actual continuously propagated RESET entrance** of record j. For bare RESET, actual voltage equals command because no 1 mA cap is applied. At the return 0.2 V point on 0→2.5→0,

\[
D_j=2B(V_{\max,j})-B(v_{r,j}),\qquad
B(V)=\frac{e^{\lambda V}(\lambda V-1)+1}{\lambda^2}.
\]

Use the native saved endpoint values in the per-record arithmetic; B(V)=V²/2 when lambda=0. The frozen baseline b=0.769297378470313 yields an inverse-q denominator increment of approximately 3649.82022 to the return read point.

Changing only A/kappa_a/rho does not change this explicit recovery coefficient. It changes the actual entrance and capture history; therefore leaving b fixed could unnecessarily restrict the recovery budget while fitting the new path. Including b tests that restriction with the existing shared law. **This does not prove that every fit with fixed b is impossible or that freeing b is sufficient.**

For varying b the correct bound is

\[
q_{r,j}(b)\ge\left[q_{0,j}(b)^{-1}+\kappa_+e^{2b}D_j\right]^{-1}.
\]

Even with the three segment entrances held fixed, the within-history RESET entrances q_0,j(b) generally change. Do not reuse baseline RESET entrances as universal constants. At a hypothetically fixed entrance, increasing b lowers this bound through exp(2b), but that conditional monotonic statement is not a proof of the full trajectory's behavior.

The tradeoff must remain visible: increasing b strengthens negative q recovery and negative a loss, but also strengthens positive a formation and weakens positive a loss. Positive q rates retain their positive-normalized coefficients, yet their total-current input changes through a. A better RESET tail can therefore worsen early SET and series behavior. Full 92-phase evaluation is required.

## What the already saved refined baselines show

Read-only sources: `analysis/coupled_native_training_2026-11-04/baseline_run_v2/disabled_refined.npz` and `enabled_refined.npz`. All 14 bare training RESET records were inspected at native return index 193, approximately 0.2 V. Only saved states/currents and the closed-form bound above were used.

| Saved quantity | a disabled | a enabled |
|---|---:|---:|
| RESET entrance q range | 0.031389828903–0.031389829502 | 0.383716722726–0.383799969947 |
| Calculated q lower bound range at return read | 0.000271615303758–0.000271615303803 | 0.000273790602423–0.000273790644796 |
| Median saved actual q at return read | 0.00150074031927 | 0.0426017374926 |
| Median saved soft current | 11.3826816262 nA | 352.459453475 nA |
| Median saved total current | 11.3826816262 nA | 109.848996654 microampere |
| Median per-record soft/observed current ratio | 150.7826 | 4668.9047 |

The observed median at this read point is 76.7438439 pA. Per-record saved soft-current excesses are:

| Bare record | Observed / pA | Disabled soft / observed | Enabled soft / observed |
|---|---:|---:|---:|
| 1 | 66.9372 | 170.05 | 5265.51 |
| 2 | 54.9247 | 207.24 | 6417.14 |
| 3 | 47.6621 | 238.82 | 7394.97 |
| 5 | 202.618 | 56.18 | 1739.52 |
| 6 | 1087.190 | 10.47 | 324.19 |
| 7 | 34.4991 | 329.94 | 10216.47 |
| 8 | 18.8615 | 603.49 | 18686.67 |
| 9 | 507.449 | 22.43 | 694.57 |
| 10 | 16.3870 | 694.62 | 21508.53 |
| 11 | 168.393 | 67.60 | 2093.07 |
| 12 | 405.961 | 28.04 | 868.21 |
| 13 | 27.1226 | 419.68 | 12995.04 |
| 14 | 194.053 | 58.66 | 1816.30 |
| 15 | 86.5505 | 131.51 | 4072.30 |

Since a>=0, cancelling the parallel current at a **fixed saved q** cannot remove these saved soft-current excesses. The coupled state trajectory also needs to change. However, the saved soft current was evaluated at the saved q, **not at the newly calculated q lower bound**. No new readout at q_min was performed. These numbers do not establish a current floor at q_min or exclude all parameter choices, including all b values. The state bound is deliberately a necessary budget, not a feasibility certificate.

## Stability result: relevant warning, not a fit veto

The original-spectrum q→0 limit with active a has Jacobian `[[alpha,beta],[gamma,-delta]]`, with alpha>=0 and beta,gamma,delta>0, hence negative determinant. The ideal insulating origin is a saddle. That limit lies outside the finite Gamma table; it does not rule out a finite low-state equilibrium or a sufficiently low current over the finite measured path. The corresponding series conclusion additionally requires a finite positive transistor origin slope. See the saved `LOW_STATE_STABILITY.md` for the precise hypotheses.

Do not use this limit to cancel the present q² calibration or switch exponents during the run. Conversely, a lower RMSE would not prove a stable microscopic insulating state, time-domain retention or an identified defect mechanism.

## Interpretation of the actual fit

- Every candidate shares A/kappa_a/rho/b across bare and series data and propagates all 92 phases within the three declared segments. No extra phase/cycle anchors, missing-cycle bridges or entry refits are introduced.
- Report the unchanged 27023-residual objective, branch errors and all 20 feature results; improving one tail or aggregate cost cannot conceal premature bare SET, lost series behavior or failed gates.
- Enforce the predeclared numerical/parameter bounds. Budget or integration failure is an unresolved trial, not physical rejection; preserve the attempted point and completed work. A bounded optimizer's failure or boundary hit is not a family-wide impossibility result.
- A successful calibration would establish an improved shared effective trajectory model within this fixed data/protocol assumption set. It would not yet establish microscopic uniqueness, physical-time parameters, holdout generalization or that the unchanged entrance assumptions are correct.

Only the already-specified four-dimensional calibration and its numerical validation are supported by this note. The purpose is to make the actual fit interpretable, not to insert another diagnostic loop before running it.
