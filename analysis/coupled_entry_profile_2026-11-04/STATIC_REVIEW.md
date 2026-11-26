# Independent static review of entry profiling

Reviewed only by source reading, AST parsing and file hashing. No module was
imported, no model or optimizer was run, no data were exported, and no
author-owned file was changed.

Reviewed versions:

- `profile_entries.py`: `4f33b871c8e0c74f18ab647627ee9da659ff5c55e6ca03d3264ee55518b804af`
- `PROFILE_PLAN.md`: `82a48c855622597fae85ca0d9a4dad62f5dac27e43b16f9545517b44aa843ed2`

Outcome: **the numerical/scientific scope is coherent; add the small
identifiability reporting item below before freezing**. No new model family,
state, material parameter, fit start or trajectory is needed for that item.

## Checked and consistent

1. **Entry domain.** `_entry_bounds` uses valid conservative positive-prefix
   inequalities q′ ≥ −Rmax q and q′ ≤ Cmax(1−q). With passive local voltage
   bounded by the monotone command and the declared current cap, integration
   gives the implemented lower log-q bound and upper complement bound.
   Rmax·L is more conservative than κB(Vmax), but remains valid. The original
   Γ endpoints are converted correctly to u = log(1+Γc/Γ), intersected with
   the inherited u domain, and moved inward by the declared guard. The upper
   bound leaves a quantified maximum-capture margin; it is not merely an
   unexplained offset from q = 1. Empty domains trigger explicit fallback.

2. **Guess projection versus state clipping.** `np.minimum/maximum` projects
   only the three proposed optimizer starting parameters. Proposed/projected
   values and the optimizer's first interior adjustment are saved. The a
   parameter upper bound is moved inward by one representable step only when
   its own round trip exceeds A. Neither operation clips integrated states.
   Trials undergo the frozen physical/table-domain validation and failures
   terminate the start rather than yielding fabricated residuals.

3. **Prefix data and three regressions.** `_make_prefix` retains the original
   zero-to-outward native waveform through the final sample ≤0.25 V, uses the
   unchanged original scoring mask and requires at least two eligible points.
   Unscored waveform points remain integrated. All three old-entry regression
   calls precede fitting, compare to saved final q¹ native samples and use
   native checks plus two alternate roots. Failure blocks fitting and full
   evaluation. The three separate two-coordinate fits are equivalent to
   minimizing a separable six-coordinate prefix objective with fixed material
   parameters; they do not introduce cross-segment state coupling.

4. **Actual-call and time budgets.** The 900 cap increments at every actual
   `simulate_phase` invocation, including baseline regressions, explicit
   Jacobian calls and ending-point checks. Invalid states rejected before
   simulation are recorded without falsely counting an ODE. Jacobian base
   reuse requires exact full-coordinate equality. The enclosing fit section
   counts optimizer algebra and all finite differences; explicit artifact I/O
   pauses prevent double charging. The 300 s fit and 600 s total limits are
   checked during primitives and after sections. A primitive may finish just
   beyond a checkpoint; the actual time and failure remain visible, as the
   plan states. No implicit retry or tolerance relaxation was found.

5. **Normally returned endings.** Every normal optimizer return, including
   max_nfev exhaustion explicitly not called convergence, receives independent
   fit-precision and refined-prefix evaluations, native checks and alternate
   roots. Only a qualified ending point enters selection. A failed start's
   good intermediate iterates are not selected. Failure or budget exhaustion
   preserves already qualified entries and records old-entry fallback for
   unresolved segments.

6. **Fixed model, continuous histories and holdouts.** All seven selected
   material parameters are copied from the verified terminal parent and are
   never fitted or mutated. Only three initial pairs change. Full evaluations
   start at the original segment command zero, not at a prefix final state,
   and reuse the frozen original training adapter. There are at most two
   complete histories with 184 total phases, unchanged residuals and 20 gates,
   184 alternate roots per history and a final precision comparison. No
   per-cycle/per-phase reanchor or missing-cycle bridge was found. Only the
   original training loader and saved training control are accessed; prefix
   points are correctly labelled calibration data, with holdouts untouched.

## Minimal required reporting addition

The code retains all three starts' physical endpoints, costs, optimizer
termination, active-bound masks, finite-difference requests and failures.
That is sufficient raw evidence to audit multiple solutions. However, lines
around the Jacobian SVD and `valid_ends` selection currently save only singular
values/ratio and the selected endpoint. They do not explicitly report q/a
sensitivity correlation or near-equal-cost but different-state alternatives,
as required by `ENTRY_INFERENCE_SCOPE.md`.

Before freezing, add saved-arithmetic-only diagnostics:

- Save the small final two-column optimizer Jacobian with its coordinate
  labels, or its column norms/normalized column inner product together with
  SVD right vectors. A normalized column inner product is a sensitivity
  alignment diagnostic, not a measured statistical correlation or confidence
  interval. Retain undefined cases for a zero column rather than inventing a
  correlation.
- For each segment, list every pair of qualified ending points and their
  absolute cost difference, physical q/a values, Δln(q) and Δ(a/gref).
  Predeclare a simple descriptive near-equal-cost criterion and state its
  units/normalization and non-statistical status. It must not change the
  minimum-cost selection rule, acceptance gates or any model call.
- If that criterion is met while endpoints materially differ, mark the
  alternative states explicitly and report unresolved entry identifiability.
  If fewer than two endings qualify, state that the multi-start degeneracy
  check is unavailable, not that the inverse is unique.

All necessary values already exist in memory. These additions require zero
new prefix evaluations and no broader parameter search. The existing terminal
`initial_states_uniquely_identified=false` is appropriate and should remain.

## Small defensive check worth adding

The entry-envelope proof relies on 0≤Vh≤Vcommand and on the declared positive
current cap. The inherited frozen passive circuits implement those
conditions. `_verify_prefix` currently checks passivity via I·Vh≥0 and bare
compliance, but it does not explicitly record the local-voltage ordering or
the series current-cap check. Recording both inequalities from the already
saved native arrays would directly audit the assumptions used in the entry
bounds at no additional model cost. This is a validation improvement, not
an identified defect in the frozen circuit implementation.

## Interpretation limits

The selected minimum is a conditional boundary-state fit. Neither low prefix
error nor small numerical differences establish true microscopic entry
states. Full-history results overlap the training prefix and are not wholly
heldout prediction. The old a0=0 or old-q0 conditional capacity/early-SET
proofs must not silently be reused for newly fitted entries that violate
their assumptions. No conclusion about eventual accuracy can be drawn from
this static review.
