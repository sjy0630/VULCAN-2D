# Prepared static low-resistance capacity check for q¹

Status: prepared only. Do not execute until the root agent has reviewed and
authorized this separate one-run check. The earlier κ seed derivation remains
unexecuted. This check contains no ODE, fit, inverse or root solve.

## Joint invariant upper rectangle

Let Icc = 0.001 A be the original bare SET compliance, J the fixed selected
current scale, and λ ≥ 0, b ≥ 0, ρ > 0, κ > 0, κa ≥ 0. Define

    c = qcap = Icc/(Icc + J),
    fcap = c/[c + ρ exp(−2b)(1−c)],
    abar = A fcap.

For a positive bare segment, actual material |I| ≤ Icc. Therefore

    dq/ds ≤ κv[(Icc/J)(1−q) − q].

At q = c this derivative is nonpositive. At the a upper boundary, the
positive-polarity equilibrium fraction is

    f+(q) = q/[q + ρ exp(−2b)(1−q)],

which increases with q. The negative-polarity fraction replaces exp(−2b)
by exp(2b) and is no larger when b ≥ 0. Thus at a = A fcap and q ≤ c,
da/ds ≤ 0 for both polarities. At v = 0 all rates vanish. This statement
also covers κa = 0; no division by a rate coefficient is needed.

For negative polarity, at q = c and v > 0,

    dq/ds ≤ 0  iff  Isoft_abs(v,c) + a v ≤ Icc exp(4b + λv).

Let Vreset_max = 2.5 V be the maximum original bare RESET magnitude. Let
Mneg be the largest magnitude current over **all Γ rows and all stored
negative-table voltage columns from the first column through the right
bracketing column for Vreset_max**, scaled by original Ron/current Ron.
Include the right column even if its voltage exceeds Vreset_max; if the
endpoint is exactly a node, that node is sufficient. Log-bilinear
interpolation is a convex combination of node log currents, so this node
envelope covers every v in the actual RESET range. The inherited linear
near-zero voltage extension is bounded by the same Mneg. Columns farther
above that right bracket are excluded. Consequently

    Mneg + A fcap Vreset_max ≤ Icc exp(4b)

is a sufficient, deliberately conservative negative-boundary check. It is
not necessary. It uses no new negative readout evaluations. Require both
bare segment entries q0 ≤ c and a0 = 0, with no unrecorded transition
bridged. All upper faces of the rectangle q ≤ c, a ≤ A fcap then point
inward, and the continuous bare protocol remains in that rectangle while
the trajectory is valid in the original finite table domain. An exit from
the lower table boundary is an invalid trajectory, not an exemption from
the upper-capacity constraint. The argument concerns bare 1R only.

## One positive readout and the original median resistance gate

Use the original `1r.set.return.R_0p05V` native return index and gate. Require
the eligible records to have the same actual command read voltage, nominally
+0.05 V, preserving its stored floating-point value. Validate the fixed
bare protocol and actual 1 mA compliance from the original contract.
Also compute the original observed resistance median directly from the
original eligible per-record values, with no additional selection or
rounding. Save the eligible values, their median, the gate upper limit,
fcap, Fcap, and the negative node envelope, so later A declarations can use
these explicit quantities rather than an assumed approximate resistance.

Require the positive table to be nondecreasing in log Γ in both voltage
columns bracketing that native read voltage, without accepting or repairing
negative increments. Evaluate the frozen positive soft readout once at
q = c, yielding Fcap. This is one new forward table evaluation, not saved
q² current. For any q ≤ c, the unconstrained read current obeys

    Iread ≤ Fcap + vread A fcap = Icap(A).

When Icap(A) < Icc the existing bare-circuit rule uses the command voltage
directly, so this bound does not require an additional voltage-monotonicity
assumption. In particular, require Irequired = vread/Rgate_upper < Icc.
If Icap(selected A) < Irequired and the joint rectangle check passes, every
eligible bare read resistance exceeds Rgate_upper. Hence the original median
resistance gate fails for **every positive κ**, including the three planned
κ seeds. This is a necessary capacity obstruction, not a training score.

## Necessary A threshold with its own invariant check

The candidate lower threshold is

    Areq = [1/Rgate_upper − Fcap/vread]/fcap.

If the expression is nonpositive, the readout gives no positive A threshold.
Do not clip or claim a useful bound. If it is positive, recheck

    Mneg + Areq fcap Vreset_max ≤ Icc exp(4b).

If this passes, the same rectangle exists for every 0 ≤ A < Areq, because
the left-hand side increases with A. Every such A has Icap(A) < Irequired
and fails the original median low-resistance gate. Thus A ≥ Areq is a
necessary condition over A for this fixed J, λ, b, ρ, readout, compliance,
initial entries and protocol. It is independent of κ > 0 and κa ≥ 0.
Reaching Areq is not sufficient: the upper rectangle corner need never be
reached, and the other 19 gates and full curves are unconstrained by this
calculation. If the Areq negative-boundary check fails, retain only a
conditional formula, not a family-wide necessary bound. Changes to J, b,
ρ, λ, the readout, entries or protocol require a fresh argument.

## Work, provenance and outputs

The separate script `capacity_check.py` requires `--run --review-confirmed`,
the verified terminal q² request 35 selected from request 26, the unchanged
original feature contract and exact frozen table hash. It reads the saved
commands and original segment entries, not q² trajectories as q¹ states.
It hashes all inputs before and after and refuses an existing output or
exclusive run marker. Outputs are `CAPACITY_CHECK.json`, `CAPACITY_CHECK.md`
and `CAPACITY_CHECK_RUN_STARTED.json` in this directory only.

Maximum work: 14 saved protocol checks, one stored negative-node maximum
over the actual RESET voltage range with its right bracket,
one positive-table monotonicity audit, **at most one positive frozen readout**,
and scalar invariant/capacity arithmetic. Zero negative forward readouts,
new trajectories, inverses, roots, fits or quadratures. Arithmetic budget
30 s excluding I/O and hashing. All floating-point results are conditional
numerical evaluations of the analytic inequalities, not interval-arithmetic
certificates. Failures remain visible and no automatic retry occurs.

The pending κ seed and finite-forward plans must be reconsidered after this
capacity check; no deliberately capacity-excluded seed is presented as an
attempt at satisfying the shared low-resistance gate.
