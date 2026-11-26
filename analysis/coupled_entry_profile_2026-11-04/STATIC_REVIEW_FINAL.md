# Final independent static review: entry profiling

**Passed for the predeclared bounded run. No blocking change remains.**
The author confirmed editing had stopped before this review. The initial
review remains in `STATIC_REVIEW.md`; this note records resolution of its
small reporting request and defensive native checks.

Final files independently hashed:

| File | SHA-256 |
|---|---|
| `profile_entries.py` | `94ca82432966ccd9dcc7366e31ec8ad48af315d975256c5ae0efe91d6ba9ddb9` |
| `PROFILE_PLAN.md` | `7d02698c9adab0473bc0c2dca56e76e6d2d9ed4cde56a444bda3e80c30b56bf9` |

This was source inspection and AST parsing only. No script import, model
evaluation, data export, optimizer, table readout, inverse or root was run;
no author-owned file was modified.

## Resolved review points

- The existing returned optimizer Jacobian is now retained in full, with
  coordinate labels, norms, normalized column directions, column alignment,
  singular values, right singular vectors and a declared numerical-rank
  threshold. The implementation correctly retains null alignment when a
  sensitivity column is zero. It records that vector signs are arbitrary,
  scaling matters and sensitivity alignment is not statistical correlation.
  These calculations add no derivative or model calls.
- Every pair of qualified endings within a segment now retains physical
  states, costs, signed and absolute cost differences, Δln(q), Δa and
  Δ(a/gref). The near-cost rule is explicitly a difference of mean squared
  log residuals, `abs(2 Δcost/n_eligible) <= 1e-4 decade²`; it is not a
  difference of RMSEs. The declared distinct-state rule is
  `abs(Δln(q)) >= ln(2)` or `abs(Δa/gref) >= 0.1`. Code and plan agree.
  All such thresholds are descriptive only and do not change endpoint
  selection, the prefix objective or the original acceptance gates.
- Near-cost/distinct-state alternatives are flagged and retained. Fewer
  than two qualified endings produces an explicit unavailable reason;
  absence of a flagged alternative does not establish uniqueness.
- Native checks now explicitly record the local voltage between zero and
  command and nonnegative current below the topology's declared cap, using
  the existing numerical tolerances. These check the passive-circuit
  assumptions used in the entry envelopes and add no model calls.
- The required source freeze now includes `ENTRY_INFERENCE_SCOPE.md` and
  the initial `STATIC_REVIEW.md`, so the governing scope and review record
  cannot silently change during the run. The root can include this final
  review in the manifest as an additional guarded input.

## Core scope remains unchanged

The reviewed additions do not alter the q0 envelope bounds, physical a0
interval, three fixed starting fractions, explicit Jacobian steps, 900-call
limit, 300/600 s budgets, three baseline regressions, normal-ending precision
qualification, minimum-prefix-cost selection, frozen seven material
parameters, original three segments, or the two complete-history checks.
Initial-parameter projection remains distinct from prohibited integrated
state clipping. No per-cycle/per-phase anchor or holdout access was added.

The smallest-prefix-cost states remain effective estimates under a frozen
model. Their physical uniqueness, future curve accuracy and mechanism
interpretation are not established by static approval. The root should
freeze the final reviewed files before the unique bounded execution and
retain all failures and identifiability diagnostics in its result.
