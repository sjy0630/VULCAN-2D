# Scope of the three-segment entry-state inference

Independent mathematical/scope review only. No model run, new data export,
table evaluation, inverse, fit or author-script modification was performed
to write this note.

## What is estimated, and what remains fixed

Fix the selected q¹ law and all seven material parameters, readout parameters,
external circuits, original masks, weights and voltage-path clock. Estimate
only (q0,a0) at the three existing entries `bare1`, `bare5` and `series4`:
three unknown state pairs, at most six scalar coordinates. Their reference
instant is the original segment start before its first SET sweep, not the
first eligible measurement point. Initial-state fitting adds statistical
degrees of freedom even though no new dynamical state or material parameter
is introduced; report the count explicitly.

Fit only the original eligible points on each first SET outward prefix with
command voltage ≤ 0.25 V. Integrate the same frozen dynamics from the segment
start through the complete prefix waveform, including unscored intervals;
do not replace this by a static fit assuming constant states. Record the
exact native indices, masks, point counts, objective normalization and input
hashes before fitting. Voltage/current floors remain scoring rules, not
invented observations or state constraints.

After selection, propagate each inferred pair continuously over its entire
original segment and evaluate the full 46 pairs / 92 phases / 27023 original
residuals / 20 feature gates. There are no further per-cycle, per-polarity or
per-phase anchors. Do not bridge missing measured bare cycle 4. Prefix data
are training data; the complete-training evaluation overlaps those prefixes
and must not be called wholly independent prediction. Retain separate
descriptions of prefix fit and the rest of the same training segments without
replacing the original objective. Original holdout data remain untouched.

## Identifiability and required reporting

A single low-voltage current generally cannot separate soft-state q from
parallel conductance a: both enter the same current. Multiple prefix points
can still be nearly redundant, especially in a locally linear regime.
Dynamics and the series circuit may add information, but their presence is
not proof that the inverse is unique. The minimum local prefix error is an
effective boundary-state estimate under the frozen model, not identification
of a physically true microscopic initial state or prior device history.

Retain all three predeclared starts, labelled f = 0, 0.5 and 0.99, with the
explicit meaning of f, all six actual initial coordinates, fitted endpoints,
costs, termination reasons, domain failures, evaluation counts and active
bounds. Never hide an inferior start or a failed candidate. Compare endpoints
in physical q/a coordinates as well as solver coordinates. If near-equal
costs yield appreciably different q/a pairs, report that degeneracy rather
than presenting only the winning endpoint. Define the numerical meaning of
“near-equal” before judging results; it is not an experimental confidence band.

Report local q/a sensitivity or correlation when supported by the fitted
Jacobian: scaling, singular values/rank and condition measures should be
stated so artificial coordinate scaling is not mistaken for identification.
A correlation from only three optimizer endpoints is not a statistical
uncertainty estimate. If a Jacobian or noise model is inadequate, report
start-to-start spread and unresolved identifiability instead of inventing
confidence intervals. Lowest prefix cost alone must not establish a mechanism,
parameter uniqueness or a scientifically superior full-curve explanation.

## Lower table boundary during a positive prefix

For q¹, u = −ln(q), v = |Vh|, and nonnegative capture,

    du/ds = κ v exp(λv) − κ v (|I|/J)(1−q)/q
           ≤ κ v exp(λv).

This uses 0 < q ≤ 1 and positive-polarity material voltage. During the
monotone outward command prefix, require the passive circuit to return
0 ≤ v ≤ Vcommand and λ ≥ 0. Then v exp(λv) is increasing and ds = dVcommand,
including intervals where the material voltage is reduced by a transistor
or compliance. Consequently

    u(s) − u0 ≤ κ B(Vprefix),
    q(s) ≥ q0 exp[−κ B(Vprefix)],
    B(V) = ∫₀ⱽ x exp(λx) dx.

B has units V², κ has units V⁻², and the exponential is dimensionless.
Use the actual largest included native command voltage, at most 0.25 V,
rather than a rounded reading or a sum of eligible-point voltages. With
qtable_min = Γmin/(Γmin+Γc), an entrance lower bound

    q0 ≥ qtable_min exp[κ B(Vprefix)]

keeps the ideal prefix above the original Γ lower boundary. Add an explicit
numerical safety margin or use an interior inequality; do not claim an
interval-arithmetic certificate from floating-point evaluation. If the
derived domain is empty, report failure instead of clipping the bound.
The passive-voltage and sign conditions must be checked in actual solved
points. The formula does not certify the later complete trajectory or any
negative-polarity phase, which still requires the original domain checks.

## Upper table boundary and a domain

At an upper q boundary Q inside the table and v > 0, the positive q¹ law
points inward precisely when

    |I|/J ≤ exp(λv) Q/(1−Q).

A verified uniform circuit/current upper bound satisfying this condition
over the prefix establishes an invariant upper boundary. At v = 0 the rate
is zero. Alternatively, derive a quantitative upper-growth budget that
keeps all prefix states below the table limit. Merely choosing q0 a little
below the upper node is not a proof. If no analytic guarantee is available,
retain finite-domain checks throughout integration and explicitly fail any
escaping proposal; never repair it by clipping, extrapolation or reanchoring.

Use 0 ≤ a0 ≤ A. The original a law points inward at 0 and A for 0 ≤ q ≤ 1.
No learned entrance may bypass these physical or table constraints. Earlier
capacity proofs that assumed a0 = 0 or a0 ≤ A fcap cannot automatically be
reused if the new entrance estimator allows a0 > A fcap; likewise, the old
early-SET bound depending on a particular q0 must be recalculated before
application. State which prior conditional bounds still apply.

## Acceptance limits

An entry estimate is usable only after finite-domain and physical-state
checks, unchanged-material/circuit provenance, continuous state propagation,
original KCL/KVL checks and the planned numerical precision verification.
A successful prefix fit followed by poor full curves localizes a limitation
of the fixed dynamics/parameters or entry identifiability; it does not justify
adding anchors. Conversely, improved full-training error is evidence that
previous boundary assumptions mattered, not proof that the fitted entrances
are uniquely physical or that the microscopic mechanism is established.
