# Predeclared finite q¹ recovery seed derivation

Status: preparation only. `derive_seed.py` must not run until the root agent
has reviewed this plan and script and explicitly authorizes its one run.
Preparing these files performs no table inversion, ODE, fit or root solve.

## Purpose and fixed inputs

The q² and q¹ laws have different state dependence. Reusing the fitted q²
value κ = 57.67798793605963 V⁻² as an equivalent q¹ kinetic scale would be
unjustified. Derive a transparent order-of-magnitude q¹ seed instead.

Use only final verified q² refined request 35, matching the formally selected
coarse request 26. Its 14 actual bare RESET entry states are from records
1, 2, 3, 5, ..., 15. Keep all other selected parameters fixed, including b,
λ, J, A, κa and ρ, as well as Ron, Γc, external circuits and original three
segment entries. No q¹ trajectory is generated or inferred at this step.

Use the original `1r.reset.return.I_0p2V` median gate upper limit, not a rounded
report number or a new per-record acceptance band. Use its original native
return index for each record and require a common actual read voltage. The
native voltage is nominally −0.2 V; preserve the stored float rather than
replacing it by exactly −0.2 V. Validate that bare RESET has no compliance,
local voltage equals command, and the out/return branches are monotone.

## One analytic inverse of the frozen readout

Require the frozen negative log-current table to be nondecreasing in
log Γ in both columns bracketing the original read voltage, with no negative
increment tolerance or repair. Interpolate those columns in log |V| and
apply the original Ron/current Ron amplitude scaling. Locate the cell
containing the logarithm of the gate upper limit, require a positive local
slope, and perform exactly one analytic linear interpolation in log Γ.
No iterative inverse, scalar root, spectral quadrature or extrapolation.

The result Γtarget gives qtarget = Γtarget/(Γtarget + Γc). It obeys

    |Isoft(−vread, qtarget)| = Igate_upper, with a = 0.

For a fixed record, nonnegative a makes q ≤ qtarget a necessary soft-channel
condition for total current ≤ Igate_upper. It is not sufficient when a > 0.
The actual original feature is a median over records: applying this same
current target to each entrance is a reference construction for a common
seed, not a new requirement that every record pass the median gate.

## Deleted-capture q¹ reference

Let s be cumulative command-voltage distance and v the actual bare voltage
magnitude. For negative polarity, with capture removed only for this
reference calculation,

    dq/ds = −κ1 v exp(2b + λv) q,
    qreturn = qentry exp[−κ1 exp(2b) D],
    D = ∫ v exp(λv) ds = 2 B(Vmax) − B(vread),
    B(V) = [exp(λV)(λV − 1) + 1]/λ²,
    B(V) = V²/2 when λ = 0.

For each saved q² entrance, compute

    κ1,record = ln(qentry/qtarget) / [exp(2b) D].

Require all ratios, doses and rates to be positive and finite; retain an
unresolved failure rather than clipping or replacing a nonpositive rate.
Take the median of the 14 individual rates as κ1,reference. q, b and λv
are dimensionless; D has units V² and κ1 has units V⁻². These are path-clock
coefficients, not measured inverse seconds or identified physical rates.

## Exactly three fixed candidates, no optimization

Predeclare multipliers [1, 3, 10]. Their only parameter replacement is

    kappa_positive_per_v2 = multiplier × κ1,reference.

Everything else is the final selected q² parameter set. Retain the original
fixed entries bare1, bare5 and series4, including a = 0 at those entries.
Record all resulting parameter tuples and entry states without evaluating
them. The subsequent authorized comparison must use the original complete
46-pair / 92-phase / 27023-residual / 20-gate training contract, propagate
both states continuously within its three segments, and avoid bridging the
missing measured cycle 4. It must retain failures and use numerical
validation before interpreting a successful trajectory. No optimization,
new states, extra free parameters, per-cycle anchors or rate-range expansion
is authorized by this seed derivation. The existing q² result is the comparator.

The same κ multiplies capture and recovery in both polarities. Scaling κ
therefore also changes positive capture and positive recovery, negative
capture, circuit feedback, a-state evolution through q, and all actual q¹
RESET entries. The q² entrances used above are reference data, not q¹
boundary conditions. Nonnegative capture makes the deleted-capture decay a
conditional lower envelope for a fixed entrance and bare waveform, but the
median seed is neither a family-wide necessary rate bound nor a reachability
guarantee. Passing this construction is not a fit result, evidence for a
microscopic first-order mechanism, or a rejection of the q² family.

## Work budget, provenance and output

One terminal-case read, 14 saved entrance/waveform checks, one negative-table
monotonicity audit, at most one analytic log-table inverse, 14 scalar dose/rate
calculations, and three parameter declarations. Zero new ODEs, fits, forward
trajectories, iterative inverses, roots or spectral quadratures. Numerical
arithmetic budget is 30 s excluding I/O and hashing. The arithmetic uses
ordinary floating point and is not an interval-error certificate.

The script requires `--run --review-confirmed`, checks the terminal status
and exact candidate, and refuses an existing result or exclusive run marker.
It hashes all immutable inputs before and after. It writes only
`SEED_DERIVATION.json`, `SEED_DERIVATION.md` and `SEED_DERIVATION_RUN_STARTED.json`
in this directory. Every failed guard is retained; no automatic retry.

The output's `q1_kappa_reference` and `kappa1_reference_per_v2` are the same
numeric V⁻² value. `seed_parameters` holds the seven parameters with κ set
to this reference; `initial_states` holds the original three entries.
`candidates` contains the three complete parameter declarations, each with
its fixed `multiplier`, `parameters`, `initial_states` and
`fixed_recovery_order=1`. These are inputs for the separately reviewed
forward-control driver; this script never runs that driver.

Prepared invocation, **not executed during preparation**:

    python3 analysis/coupled_linear_recovery_2026-11-04/derive_seed.py --run --review-confirmed
