# At most four declared q1 controls

Status: general driver prepared; final candidates are NOT yet specified here.
The root agent must finish the capacity audit and finalize an explicit
FORWARD_DECLARATION.json before independent review and one root-agent launch.
Do not start a forward run while an author is editing any frozen input.

## Scope and hypotheses

All declared cases use Q1Material, fixed_recovery_order = 1, the unchanged total
current readout and both coupled states q and a. They use the original actual
bare/series circuits and one shared material parameter set per complete run.
This tests a narrowly declared structural and rate alternative. It is not
full q1 optimization, a proof of the microscopic mechanism, or a rejection
of the q² family.

Fix all three segment entry states at terminal q² calibration request 26.
The first case must keep every numerical material parameter equal to that
selected q² case, including κ+ = 57.67798793605963 V⁻². This is the pure
recovery-order perturbation. Up to three further cases may change only
explicitly declared parameters, as decided after the capacity audit.

The earlier analytic seed supplies a q1 κ reference and three rate-only
proposals. They are not automatically executable candidates. The capacity
audit may show that the selected A cannot meet a low-resistance constraint
for any κ under the declared assumptions; the root agent must resolve this
before finalizing the list. This document makes no capacity conclusion and
does not choose an alternative A or final rate schedule.

Every eventual case that changes κ is a rate-rescaled comparison: κ changes
capture and recovery together. Any A or other changes must also be explicitly
recorded, and such cases cannot be called order-only perturbations. The driver
does not generate a candidate, change a declared value, project parameters,
retry or adapt the list to outcomes.

## Required final declaration schema

The root-authored FORWARD_DECLARATION.json must contain:

- status = `final_predeclared_q1_controls_no_forward_evaluation`;
- parent_selected_request_id = 26, parent_result_sha256, fixed_recovery_order = 1;
- initial_states equal to the original bare1/bare5/series4 entries;
- allowed_changed_parameters: explicit unique names from the seven material parameters;
- candidates: an ordered list of one to four fully specified cases;
- input_sha256: absolute-path digests covering SEED_DERIVATION.json and all
  derivation/capacity evidence used to finalize candidates.

Each candidate contains a unique ASCII identifier label, a nonempty role,
parameters (all seven numbers), changed_parameters (exact changes versus
selected q²), initial_states, fixed_recovery_order = 1, forward_evaluated =
false, and only_recovery_order_changed_from_selected_q2 (true exactly when
the numeric parameters are unchanged). The first candidate must be that
unchanged-parameter control. Changed fields must be a subset of the global
allowed_changed_parameters. No default values or missing fields are inferred.

The driver separately checks the analytic seed's terminal status, two equal
positive κ reference fields, original entry states, unchanged non-κ seed
parameters, and input_sha256_before/after with inputs_unchanged = true. It
does not use the seed's earlier candidate array as the execution list.
Missing final declaration or any mismatch prevents starting a forward run.

## Original task preserved

Each actual candidate requests the complete 46 original record pairs,
92 phases and 27023 weighted residuals through training_problem_v2:

- bare1: records 1–3, one fixed entry;
- bare5: records 5–15, one fixed entry after missing measured cycle 4;
- series4: records 4–35, one fixed entry.

Every available native sample is propagated; original observed eligibility,
four-phase weights and bare SET three-subgroup weights are unchanged. Every
complete candidate evaluates all 20 original feature gates. Gate failures
do not cause early rejection or alter cost. Undefined features retain their
original failed status. No periodic trajectory reuse, per-cycle reinitializing,
new holdout access, inferred timing, missing-cycle bridge, state clipping,
table extension or artificial residual penalty is allowed.

The existing finite-domain solver can fail before all phases finish. Preserve
the completed and failed partial phase trajectories, error and count; mark
the run ineligible. Do not fabricate a full residual vector or call domain
failure a physical rejection. Continue to the next declaration if numerical
budget remains. This does not promise to continue a physically undefined
trajectory across a failed phase.

## Precision, ranking and checks

Use Radau at coarse settings rtol = 1e−6, atol_u = 1e−8, atol_x = 1e−10,
max_step = 0.05 command-path volts. After the full candidate finishes, call
the frozen candidate-aware calibration verify_native with its actual A.
Coarse checks include all native states, domains, conservation, current
components, compliance and exact two-state joins. All gates are recorded
regardless of the cost or gate count.

Among complete coarse candidates passing the numerical checks, select the
lowest unchanged original cost, breaking exact ties by declaration order.
No gate count participates in ranking. If all fail, select nothing.
If budget leaves candidates unrun, report that coverage is incomplete and
that a selected result is only the minimum among completed valid cases.

Independently rerun the same selected candidate twice, from the same original
segment entries; never reuse coarse trajectories:

- nominal: rtol = 1e−7, atol_u = 1e−9, atol_x = 1e−11, max_step = 0.02;
- refined: rtol = 1e−9, atol_u = 1e−11, atol_x = 1e−13, max_step = 0.01.

Both calls re-evaluate all original residuals and all 20 gates. Each calls
verify_native with alternate_roots=True: two reference roots per phase,
184 per complete run. Then call frozen compare_final with the selected A.
Required comparison thresholds remain max log-current difference 3e−4
decade, max u difference 1e−3, max a/A difference 1e−5. Numerical verification
is reported only if both final runs, all roots, comparison and source guards
pass. Passing these checks does not establish physical accuracy.

## Hard execution budget

At most four coarse evaluations and two final evaluations: six complete
requests, at most 552 completed phases. No retry or extra candidate.

Coarse numerical work is limited to 600 s and total numerical work to 900 s.
Reserve the remaining 300 s for final work; never extend or reset budgets.
Initial data/table loading and artifact I/O are excluded. Forward propagation,
feature calculation, native checks, alternate roots and precision comparison
are counted. Inline inherited source checks are conservatively included.
Checks inside the solver interrupt at its next budget checkpoint; a running
primitive may cross the wall-time boundary before that checkpoint. Record
actual elapsed time and any retained incomplete run rather than hiding it.

After the coarse budget is exhausted, unstarted declarations get explicit
not_run_budget records. Final requests are subject to the total limit. A
budget-interrupted candidate is ineligible even if some complete data exist.
No verified result is asserted if either final precision or comparison fails.

## Source guard and artifacts

The root agent creates a new SOURCE_FREEZE.json only after all authors stop
editing and independent review finishes. The driver does not create or
repair this manifest. It must cover every parent q² guarded source, terminal
RESULT, selected request-26 and final nominal/refined JSON/NPZ/declarations,
the parent freeze, q1 implementation, algebra script/result, structural note,
analytic seed script/plan/result and every seed input, final declaration and
all of its capacity/derivation inputs, this driver and plan,
and the original data/readout/feature contracts. The driver validates hashes
before loading the model and again at terminal completion. It also checks
the new manifest's own hash before versus after.

The output directory must not already exist (exist_ok=False). Each request
gets a declaration written before evaluation, followed by a result JSON.
All available native trajectories, states, residuals, features and checks are
saved through the existing artifact writer. RESULT.json records scope,
declarations, all outcomes, selection, budgets and guards. A failed/incomplete
run remains a recorded result, never an invented high-cost point.

Exit 0 requires all declared coarse candidates actually attempted and a
verified final candidate. All other outcomes return 2; scientific conclusions must use
the detailed terminal report rather than the process code alone.

Prepared invocation (not executed during preparation):

    .venv/bin/python analysis/coupled_linear_recovery_2026-11-04/run_rate_controls.py
