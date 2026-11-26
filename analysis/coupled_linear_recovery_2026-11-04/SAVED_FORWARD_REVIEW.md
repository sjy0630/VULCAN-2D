# Independent saved-result review: q1 finite forward controls

Reviewed 2026-11-04. **Execution, selection and numerical checks pass; the scientific accuracy contract does not.** This review read saved training observations and predictions only. It performed no material-model evaluation, ODE integration, root solve, parameter search or holdout decoding. Root confirmed the authoritative numerical session 28266 exited 0; the independent saved-result audit session 78710 also exited 0.

## Integrity and execution

- All 77 frozen source/input SHA-256 values agree before execution, after execution and at review. The freeze file remained unchanged and the final declaration hash matches.
- The four predeclared controls were all attempted with the declared parameters. There were six actual evaluations and **370 completed phases**: four complete 92-phase evaluations and one completed phase in each failed control. Coarse numerical time was 23.3393 s; total numerical time was 97.4820 s.
- Every saved completed phase uses `fixed_recovery_order=1` and `material_variant=q1_a_linear_recovery_control`. Bare 1R and 1T1R therefore use the same q1 material law and parameter values. Their external circuits differ as declared.
- Bare records 1–3, bare records 5–15 and series records 4–35 are separate segments. All q/a handoffs inside each segment agree exactly; the measured-but-missing bare record 4 is not bridged. There are no per-cycle state resets.

| Request | Control | Completed phases | Original cost | Raw RMSE (decade) | Gates passed |
|---|---|---:|---:|---:|---:|
| 1 | order only | 1 | unavailable | unavailable | unavailable |
| 2 | capacity/rate ×1 | 92 | 3.2994840879623006 | 1.0700596000377642 | 3/20 |
| 3 | capacity/rate ×3 | 92 | 9.166099486956696 | 1.4379321551077027 | 5/20 |
| 4 | capacity/rate ×10 | 1 | unavailable | unavailable | unavailable |
| 5 | selected nominal | 92 | 3.2994834074629225 | 1.0700594500549294 | 3/20 |
| 6 | selected refined | 92 | 3.2994838376459232 | 1.0700595410004838 | 3/20 |

Request 2 is the correct minimum-cost eligible coarse candidate. Request 3 passes more gates but was not selected because the frozen selection rule is the original weighted cost, not gate count. Requests 1 and 4 both exit during bare record 1 RESET with `State Gamma outside frozen spectral table`. They have no full objective, no aggregate residual vectors and no feature gates. Their partial trajectories, last solver state and last RHS context are retained. These are **unresolved finite-table-domain failures**, not evidence physically rejecting either recovery family.

## Independently recomputed checks

For both final precisions the audit rebuilt the original 27,023-point raw and weighted residual vectors **bitwise** from observations and saved native predictions, preserving all original masks and six subgroup counts. Cost is `0.5 × sum(weighted residual²)`; it is distinct from all-point raw log-current RMSE. All four complete costs and raw RMSE values were independently recomputed.

The audit independently recomputed the 514 record-level features and all 20 primary feature gates for both final precisions. Features use the ordinary median across eligible records, fixed observed eligibility and the original Q10–Q90 bands; no model-based floor exclusions or replacement thresholds were introduced.

| Nominal–refined comparison | Maximum difference |
|---|---:|
| log10 current | 1.7328339732003428e-6 decade |
| internal state u | 2.7817749694847294e-6 |
| normalized state a/A | 4.872299006517228e-8 |

The a/A comparison uses this candidate's **A = 0.016870626999145072 S**, not the earlier seed capacity. There are 184 saved alternate-root checks per final precision (368 total); both methods share the frozen material table, so this is numerical consistency evidence, not independent validation of the material law.

## Accuracy and interpretation

The selected refined model passes only: bare SET transition midpoint, series SET return slope at 2–4 V, and series RESET peak voltage. Representative remaining discrepancies are:

| Feature | Model median | Observed median |
|---|---:|---:|
| Bare SET outward I(2 V) | 2.3351e-5 A | 5.4518e-8 A |
| Bare RESET return I(0.2 V) | 4.5882e-9 A | 7.6744e-11 A |
| Bare RESET peak current | 8.1140e-4 A | 5.0456e-3 A |
| Series SET return I(0.2 V) | 1.8355e-5 A | 7.5301e-7 A |
| Series RESET return I(0.2 V) | 9.8844e-8 A | 7.8996e-10 A |
| Series RESET plateau slope (1–1.4 V) | −5.5915 | 0.00147 |

Raw block RMSE values are 1.50949 (bare SET), 1.26997 (bare RESET), 0.916947 (series SET), and 1.10411 (series RESET) decade. Overall raw RMSE 1.07006 corresponds to a factor of about 11.75 when exponentiated; it is not a percentage error, mean multiplicative error or pointwise bound.

Compared with the previous q² calibrated refined candidate, weighted cost decreases 35.8662% and raw RMSE decreases 9.04673%; gate count increases from 2 to 3. **This is not an order-only comparison:** A and the positive-state rate also change. The pure order-only control did not complete, so these results cannot attribute the improvement specifically to linear recovery. Compared with the disabled-a refined baseline, cost is 48.9674% lower but raw RMSE is 5.56061% higher (1.07006 versus 1.01369), illustrating the distinct weighting of the two metrics.

No convergence, parameter uniqueness, microscopic mechanism or predictive holdout claim follows from these four controls. Missing scan timing still prevents interpreting the rates as rates per physical second. A common mathematical kernel is achieved here; an accurate unified explanation remains unmet.

## Evidence hashes

- `SAVED_FORWARD_REVIEW.json`: `1f90446ebdf7c14f8d8c088813f554951b54174bbb6aee19de116b0a099f8b0b`
- `SOURCE_FREEZE.json`: `70114e1a40a8d2124612df6c0c259b5e1ce8f7a0df4611f84c4491890d16c953`
- `FORWARD_DECLARATION.json`: `a0e4f9bf6ff99b2bc06f4064c86e3ee9a299bd5136c45a5eb27e1cef2c91f956`
- `run_rate_controls.py`: `5a6cd99a9680cc691ef7d97ed8e5ceba2eaed63a49cc0dc0e308ad402ca8d4d5`
- `q1_material.py`: `d5ff9715939315415fda4e772b4673aa7bf30465fc6c19bb7f4d01647d86aa1c`
- `rate_controls_run/cases/006_selected_refined.json`: `a1d23e9fdabe4fe498b2aaa73dcd6b8ccf344ac0fc95adb00c912c8b80d4ed22`
- `rate_controls_run/cases/006_selected_refined.npz`: `185fe1bcc73b9f4edb81788c70261916541f8b6f30e9c2ad16da018aea07b8c2`

The JSON review records all result/request hashes, exact features, selected parameters, failure diagnostics and recomputed numerical maxima.
