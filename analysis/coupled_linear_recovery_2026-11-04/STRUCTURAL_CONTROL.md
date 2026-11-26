# First-order recovery: prepared structural control

Status: implementation and algebra check prepared only. No new data loading,
root solving, trajectory integration, calibration, candidate selection or
accuracy claim is part of this delivery. The q² family is not rejected by
the previous bounded calibration result.

## Precisely one structural change

Use the same two states, q = Γ/(Γ + Γc) and a in siemens. Let s be cumulative
command-voltage distance, v = |Vh| the actual solved material voltage,
σ = sign(Vh), and χ = (1 − σ)b. Retain

    I(Vh,q,a) = Isoft(Vh,q) + a Vh,
    C = κ+ v (|I|/J+) exp(−χ),
    R = κ+ v exp(χ + λv),
    dq/ds = C(1 − q) − R qⁿ,
    da/ds = κa v [exp(σb) q(A − a) − ρ exp(−σb)(1 − q)a].

The frozen parent has n = 2. Q1Material has n = 1, explicitly recorded as
`fixed_recovery_order=1`. The exponent is not fitted and is not a new state.
For Vh = 0 all four component rates stay exactly zero.

The original CoupledMaterial is unchanged. Q1Material inherits its readout,
near-zero linear extension, finite spectral domain, state validation,
current law and rate-difference method. Its rate_components first calls the
parent to keep all solved-point checks and three unchanged rate components,
then replaces only the recovery expression. Recovery is computed directly
as Rq; no division by q is used. The parent q² recovery is calculated for
validation but is neither used in the resulting q1 derivative nor saved as
the q1 rate. Both implementations retain overflow/domain failures.

## Algebra and boundary behavior

At a fixed v and total current (hence conditional fixed C and R), q1 gives

    q* = C/(C + R),             if C + R > 0,
    d(dq/ds)/dq = −(C + R).

For q2 the positive root, when C > 0, is

    q* = 2C/[C + sqrt(C² + 4RC)].

When C = 0 and R > 0 the root is zero in both cases; when C = R = 0 every
q is stationary. These are conditional balances, not solved equilibria of
the coupled circuit: I and Vh also depend on q and a in the full model.

For q ∈ [0,1], q1 recovery is at least q2 recovery at identical coefficients.
For 0 < q < 1 the ratio is 1/q. At q = 0 recovery is exactly zero for both;
at q = 1 it equals R. Capture points inward at q = 0 and recovery points
inward at q = 1. At a = 0 formation is nonnegative; at a = A the net a rate
is nonpositive. The algebraic rates are finite at q = 0 for finite rate
coefficients. This does not admit q = 0 into the finite spectral table.

For the inherited numerical coordinate u = −ln(q), inside the existing
allowed table domain only,

    q1: du/ds = R − C(1 − q)/q,
    q2: du/ds = Rq − C(1 − q)/q.

The existing integrator already computes −(dq/ds)/q using positive q, so
no integrator change or domain enlargement is needed. There is no state
clipping, boundary extrapolation, zero-current floor or missing-cycle bridge.

## What the change cannot establish

Unknown q1 kinetic parameters must not be treated as equivalent to fitted q2
parameters. At fixed q, matching recovery alone would require κ1 = κ2 q;
one constant cannot match an entire varying-q trajectory. Changing κ also
changes capture, so even a match of recovery at one q does not generally
match total kinetics. The unchanged J, λ, b, κa and ρ also participate in
coupled trajectories. A same-number comparison is an explicit structural
perturbation only; failure would not reject all q1 parameterizations.

This preparation sets no candidate, center, rate range or optimization plan.
A finite, predeclared rate comparison and numerical checks must precede
any quantitative comparison or structural conclusion. Neither recovery
order is established as a microscopic mechanism. Missing measured timing
remains unknown; all rates here refer to the command-voltage path, not seconds.

## Reuse and provenance

`q1_material.py` imports the exact frozen `training_problem_v2` and reuses
its exported kernel module. The subclass passes the existing isinstance
checks without monkey-patching either module. The existing adapter preserves
all three segment entries, measured native trajectories, circuits, scoring
masks, weights and unknown-history boundaries. No evaluation is launched
by importing this module or constructing an explicit material object.

`load_q1_material(...)` explicitly loads the inherited frozen readout;
`as_q1_material(parent)` reuses an existing q2 material without copying or
altering its immutable readout. Both record this new source file's SHA-256
plus the exact parent kernel and adapter hashes in material.source_sha256.
Candidate parameter provenance remains the caller's explicit ID and hash.

Frozen kernel SHA-256:
`c6873e8cec07965a3637bb695c942fab2fd75d699f0413ae1a2b0332097c2d2e`

Frozen training adapter SHA-256:
`de4539d573e3b0c13fbfbf8a9d89b28d29a2baa8d3458ef3145bb0e9f32480f7`

## Prepared algebra self-check

`algebra_selfcheck.py` checks rate identities, boundary behavior, polarity,
conditional fixed-point formulas, provenance and inherited-method identity
using a synthetic readout descriptor and synthetic solved-point records.
It forbids actual data/table loading, roots, circuits and integration while
running. Its fixture coefficients are arbitrary dimensioned algebra examples,
not q1 candidate parameters. It writes no files and prints a compact report.

Prepared invocation (not executed as part of this delivery):

    .venv/bin/python analysis/coupled_linear_recovery_2026-11-04/algebra_selfcheck.py
