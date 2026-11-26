# Fixed-material q/a entry profiling for three continuous segments

Status: prepared implementation and predeclared protocol only. No prefix,
optimizer or full-history model has been run during preparation. The root
agent will arrange independent review, freeze all inputs, and uniquely launch.

## Scientific question and scope

The existing q1 control used segment entries derived in an earlier soft-only
model with a0 = 0. The saved EARLY_SET_BOUND shows a conditional early-SET
conflict for the old entries; it does not by itself reject the q1 material
family or prove that alternative entries solve the complete task.

Keep all seven selected q1 material parameters fixed at the terminal
rate-controls result, including q-linear recovery, A, both rate scales, J,
lambda, rho and shared b. Keep the frozen readout, near-zero treatment,
external circuits, solver law, waveform clock and original training split.
The only estimated quantities are six entry coordinates:

- bare1: q0,a0 at the start of record 1, propagated through records 1–3;
- bare5: q0,a0 at the start of record 5, propagated through records 5–15;
- series4: q0,a0 at the start of record 4, propagated through records 4–35.

These are three segment preparation states. They are never reset at another
cycle or phase, inferred from a later output, or inserted at the prefix end.
Missing measured cycle 4 remains an unknown-history boundary. No missing
waveform or physical-time duration is invented.

The complete saved q1 refined request 6 is retained as the frozen old-entry
control. Its original score, gates, precision, full native JSON/NPZ paths and
hashes remain available; no extra full control evaluation is added.

## Original prefix data

For each segment use only its first SET outward native samples with
0 < command voltage <= 0.25 V that satisfy the original training scoring
mask. No extra current threshold, smoothing, interpolated sample or point
selection by model output is introduced. At least two eligible samples are
required. All native samples from the original leading zero through the last
sample <= 0.25 V are propagated, including points not scored. The original
initial command voltage is zero. Do not append a return sweep or a second
zero, and do not continue from a previously evaluated prefix trajectory.

Each objective evaluation calls the same material and actual BareCircuit
(1 mA compliance) or SeriesCircuit with the same native prefix. Residuals are
log10(|I_model|) − log10(|I_observed|), one per original eligible prefix point;
the prefix objective is half their sum of squares. No background subtraction
or finite failure penalty is used. Prefix points are calibration data and
cannot be presented as heldout prediction or extrapolation.

## Three fixed starts and normalized variables

Let gref = |I_obs|/|Vcommand| from the first eligible native prefix sample,
in siemens. Use variables z = (ln(q0), a0/gref). This is numerical scaling;
the observed circuit conductance is not a direct material readout inversion.

For f = 0, 0.5, 0.99, propose q0 = old_q0(1−f), a0 = f*gref. This allocation
is not a measured soft/hard current decomposition. Project only these initial
parameter guesses into the declared search interval if necessary, retaining
the original proposal, projected coordinates and projection flags. The first
actual optimizer evaluation is also recorded, including SciPy's automatic
strict-interior displacement from a bound. No integrated state is clipped.

The physical a0 interval remains [0,A], expressed as [0,A/gref]. If the
floating-point upper endpoint round-trip exceeds A, move that parameter
bound inward by one representable step and record it; do not clip states.

## q0 interval from the unchanged finite table

Write q = exp(−u), with original finite limits 0 < u <= 80 and Gamma table
endpoints. Convert the table endpoints to u with
u = log(1 + Gamma_c/Gamma). Intersect with the inherited (0,80] domain and
place a fixed 1e−7 inward guard in u at each endpoint. The guard is a numerical
domain margin, not a physical parameter or an enlargement of the table.

For the monotone positive prefix, let L be its actual command-path length,
vmax its largest native command, and Imax = 1 mA for bare SET or the maximum
of the frozen positive transistor-current table for series SET. Passivity
gives 0 <= Vh <= Vcommand and total current <= Imax. With fixed lambda >= 0,
the q1 equation obeys

    dq/ds >= −Rmax q,
    Rmax = kappa+ vmax exp(lambda vmax),
    dq/ds <= Cmax (1−q),
    Cmax = kappa+ vmax Imax/J+.

Therefore define B_loss = Rmax L and B_capture = Cmax L, and require

    ln(q0) >= −u_guarded_max + B_loss,
    q0 <= 1 − (1−exp(−u_guarded_min)) exp(B_capture).

These conservative initial-parameter bounds leave room for both maximum
loss and maximum capture across the prefix. No state is extended outside
the table or replaced by the envelope. The actual solver still uses the
original coupled dynamics and can fail on a trial. The bounds are evaluated
with ordinary floating-point arithmetic, not an interval certificate.

If this declared interval is empty or invalid, do not expand it or add a
parameter. Retain that segment's old entry and record the reason. Restricting
the numerical search interval does not establish global infeasibility of
other initial conditions.

## Precision and numerical regression before fitting

For each segment define its fit precision once using that segment's gref
and the fixed A:

    rtol = 1e−8
    atol_u = 1e−10
    atol_x = min(1e−14, 1e−6*gref/A), with x = a/A
    max_step = 0.01 command-path V

All objective base and Jacobian perturbation evaluations for that segment
use this same precision. The dynamic atol_x rule prevents the normalized
a0/gref perturbation from lying below the integration error scale. It is a
numerical accuracy rule, not a fitted physical parameter.

The independent prefix-refined precision divides rtol, atol_u and atol_x
by 100 and uses max_step = 0.005. All calls use Radau. No tolerance is relaxed
or step coarsened to save time.

Before any fitting, run exactly the three old-entry prefixes at prefix-refined
precision. Compare each to its original samples in saved q1 refined request
6: commands and masks must match exactly, max scored log-current difference
<= 3e−4 decade, max u difference <= 1e−3 and max a/A difference <= 1e−5.
Check all native prefix domains, passivity, KCL, KVL, current components and
bare compliance, plus two alternate circuit roots (first eligible and final
native sample). Explicitly save 0 <= Vh <= Vcommand and nonnegative current
below the declared bare/transistor cap, using the already declared native
KVL tolerance 1e−9 V and current tolerance 1e−18 + 1e−7|I| A. These checks
audit the circuit assumptions used in the entry bounds from existing arrays;
they introduce no model calls. If any of the three regressions fails, stop: no entry fit
or full-history evaluation is launched. Save all regression outcomes.

## Finite optimizer and explicit difference calls

For each of the three starts in each segment call scipy least_squares with
method='trf', loss='linear', x_scale=(1,1), max_nfev=30 and
ftol=xtol=gtol=1e−6. Use an explicit two-column one-sided finite-difference
Jacobian. The requested step in coordinate j is

    h_j = 1e−4 max(1, |z_j|).

Prefer the positive direction when feasible, otherwise the negative one.
If neither full step fits, use half the larger available distance to a bound.
Reject a step that rounds to zero. Log every requested and actual step.
All perturbed points are evaluated with the same coupled prefix solver.

The base residual can be reused only when its complete z vector is exactly
equal to the most recent objective evaluation. This saves duplicate algebra
for a finite-difference base; it never reuses a different entry's trajectory
or substitutes a periodic waveform. Record every objective call, both extra
Jacobian calls, any separately needed base call, and each exact-base reuse.
The actual ODE-call count, not SciPy nfev, enforces the global bound.

Any domain, solver, numerical-check or budget failure raises through
least_squares and terminates that start. Save its parameters, error and any
partial arrays. Do not return a fabricated finite residual, silently switch
method, restart, or select a good intermediate iterate from a failed start.
Other declared starts may continue if budget remains.

For each normally returned ending point (including max_nfev exhaustion,
which is not claimed as convergence), independently rerun its prefix once
at fit precision and once at prefix-refined precision. Both include native
checks and two alternate roots. Require the same logI/u/a-A comparison
thresholds as the baseline regression. Only qualified ending points may be
selected. Select the minimum prefix-refined original log-residual objective
within each segment; exact ties use earlier start ID. If a segment has no valid ending point, preserve its old entry
and explicitly report fallback rather than fit success.

## Saved-arithmetic entry-identifiability diagnostics

The fitted entries add six estimated coordinates even though the material
parameters and the number of dynamical states are unchanged. Save the existing
returned two-column optimizer Jacobian, its coordinate labels (ln(q0),a0/gref),
column norms, normalized sensitivity directions, normalized column inner
product, singular values and SVD right vectors. The right-vector signs are
arbitrary. Save numerical rank using the declared floating-point threshold
max(J.shape)*machine_epsilon*largest_singular_value; rank and condition depend
on the chosen coordinate scaling. A zero sensitivity column gives an undefined
alignment (null plus its reason), not an invented correlation. The normalized
inner product describes local sensitivity alignment, not measured statistical
correlation or a confidence interval. No new Jacobian/model calls are added.

For every pair of numerically qualified ending points in the same segment,
save their physical q0,a0, original prefix costs, absolute cost difference,
signed delta ln(q0), signed delta a0 and signed delta(a0/gref). Define a
descriptive near-equal-cost flag before execution by

    |2*cost_left/n_eligible − 2*cost_right/n_eligible| <= 1e−4 decade².

This compares mean squared log residuals because cost is half the unnormalized
sum of squared prefix residuals. It is equivalent to a cost-difference bound
of n_eligible*5e−5 decade² for that segment, and does not mix different prefixes.
For describing distinct states, predeclare abs(delta ln(q0)) >= ln(2) OR
abs(delta(a0/gref)) >= 0.1. These numerical description thresholds are not
statistical confidence bands, uncertainty intervals, acceptance gates or
selection criteria. The original minimum-cost selection remains unchanged.

Flag any near-equal-cost pair with distinct entries as an alternative state
pair and retain unresolved entry identifiability. With fewer than two qualified
endings, mark the multistart degeneracy check unavailable; absence of a flagged
pair never establishes a unique inverse. Preserve all ending points and all
pair comparisons, not only the selected one. Everything in this subsection
uses values already returned by the declared optimizer and endpoint checks.

## Full-history consequence of the selected entries

Write the three selected initial states and their calibration provenance.
Then restart each original segment from command zero with the selected q0,a0.
Do not propagate from prefix-final states or reuse prefix outputs.

Run two independent complete histories through training_problem_v2 using
the same seven material parameters and original 46 pairs / 92 phases /
27023 weighted residuals. Calculate all 20 original feature gates. The
full-history original objective does not select or adjust the entry fit.

Predeclared full-history precisions account for possibly small nonzero a0:

- nominal: rtol 1e−7, atol_u 1e−9, atol_x 1e−13, max_step 0.02;
- refined: rtol 1e−9, atol_u 1e−11, atol_x 1e−15, max_step 0.01.

Use frozen calibration verify_native with the current fixed A, every exact
two-state handoff, and 184 alternate roots per complete history. Then use
compare_final at the current A with unchanged thresholds 3e−4 decade in
logI, 1e−3 in u and 1e−5 in a/A. Failures preserve partial histories and
never create a full score. Compare the final refined original score and
four weighted phase errors to the frozen old-entry control, noting the
recorded full-history tolerance differences. Accuracy remains separate
from numerical validity and feature-gate counts.

## Hard budgets, stopping and provenance

At most 900 actual prefix evaluations total, including all three baseline
regressions, objective calls, additional differences and ending-point checks.
At most nine optimizer starts, 30 nfev each. The explicit actual-call limit
has precedence over those nominal limits. No retry or additional start.

All prefix/optimizer/endpoint work shares 300 s numerical time; total
numerical time is at most 600 s, retaining 300 s for final work. Count the
optimizer's algebra, finite differences, prefix solves, native checks, roots
and comparisons. Initial loading and artifact I/O are excluded using explicit
I/O pauses. Inline inherited source checks remain conservatively counted.
Do not extend budgets or relax tolerances. Checkpoints can interrupt only
after a running primitive returns; record actual time and retained partial
results rather than hiding a small checkpoint overrun.

If fitting time/calls run out after baseline regression, keep any already
selected qualified segment entries; remaining segments use old entries.
Both final histories remain subject to total time, at most two full requests
and 184 completed phases. If either final check is incomplete, do not claim
numerical final verification.

The root freezes every parent guarded source, both parent final artifacts,
their declarations, saved independent review, EARLY_SET_BOUND, new script
and plan, ENTRY_INFERENCE_SCOPE and the independent STATIC_REVIEW in a new
SOURCE_FREEZE.json after authors stop editing. The script
verifies the full input map before loading and again at terminal completion;
it also guards the new manifest's own before/after digest. The output
directory must not already exist. Every actual prefix call saves its input
state, precision, residual/partial arrays and numerical outcome; every start
has a declaration and terminal record. Original full control files are kept
immutable and referenced, not overwritten.

This is bounded entry profiling conditioned on a frozen material model. It
is neither full-parameter optimization nor microscopic mechanism discovery.
The resulting q/a pair may remain nonidentifiable from the short prefix.
Earlier capacity or early-SET bounds conditional on a0=0, a0<=A*fcap or the
old q0 cannot automatically be applied to a newly selected entry that changes
those assumptions; no such reuse is part of this procedure.

Prepared invocation (not executed during preparation):

    .venv/bin/python analysis/coupled_entry_profile_2026-11-04/profile_entries.py
