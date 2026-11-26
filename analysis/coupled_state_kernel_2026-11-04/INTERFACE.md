# Executable common q/a kernel — prepared, not numerically validated

`coupled_forward.py` implements the existing two-state skeleton from
`analysis/soft_rate_compatibility_2026-11-03/SOFT_FEEDBACK_NEXT_DECISION.md`,
with the already studied **fixed q² recovery** structural choice. No old
source, fitted result, default application model, or holdout data is changed.
Import does not load data or run an experiment. Preparation used source
inspection and builtin compilation only, not model execution.

## Exact shared material law

Let `q=exp(-u)=Gamma/(Gamma+Gamma_c)`, `v=abs(Vh)`,
`sigma=sign(Vh)`, `chi=(1-sigma)*b`, and
`I=Isoft(Vh,q)+a*Vh`. At nonzero actual material voltage:

```text
dq/ds = kappa_positive*v*[(abs(I)/J_positive)*exp(-chi)*(1-q)
                        -exp(chi+lambda*v)*q**2]
da/ds = kappa_a*v*[exp(sigma*b)*q*(A-a)
                  -rho*exp(-sigma*b)*(1-q)*a]
```

Both derivatives are zero at Vh=0. Both use one actual total-current circuit
point. Material functions do not select equations by topology, cycle, file
phase, or outward/return direction. The same b is used in both transitions.
The old symmetric q notation maps to `kappa_q=kappa_positive*exp(b)` and
`J0=J_positive*exp(2*b)`; kappa_a retains its independently named symmetric
normalization. q² is an effective structural hypothesis, not a measured
two-particle mechanism. q/a do not identify a microscopic species or filament.

The clock is accumulated command-voltage distance, in volts. Unknown sweep
timing and waits are not assigned physical seconds. Duplicate commands add
no inferred dwell. Fixed parameter units: lambda V^-1; J_positive A;
kappa_positive/kappa_a V^-2; A and a S; q, b and rho dimensionless.

## Construction and entry points

```python
parameters = Parameters(
    lambda_per_v, J_positive_a, kappa_positive_per_v2, b,
    a_max_s, kappa_a_per_v2, rho, enable_a,
)
material = load_material(
    parameters,
    candidate_source_id=source_id,
    candidate_source_sha256=source_sha256,
    nearzero_policy='linear_endpoint',
    nearzero_anchor_v=None,  # None selects the original table minimum.
)
precision = Precision(name, rtol, atol_u, atol_x, max_step)
bare = BareCircuit(compliance_a=None)
series = SeriesCircuit(compliance_a=None)
point = circuit.solve(command_voltage, material, (q, a), reference=False)
current = material.current(actual_Vh, (q, a))
dq, da = material.rates((q, a), point)
result = simulate_phase(
    material, circuit, data, (q0, a0), precision, method, check_budget,
    collect_rhs=True,
)
history = simulate_history(
    material, [{'data': data1, 'circuit': circuit1},
               {'data': data2, 'circuit': circuit2}],
    (q0, a0), precision, method, check_budget, collect_rhs=True,
)
```

Method is explicitly `DOP853` or `Radau`; there is no automatic method switch
or retry. check_budget is a callback or None. Numerical integration uses
`(u=-ln(q), x=a/A)` with solver tolerances `(atol_u, atol_x)`. The physical
input and saved states remain `(q,a)`. For disabled a, the solver integrates
only scalar u with atol_u; x is absent from its numerical Jacobian and saved
as exactly zero, with no division by A. The requested atol_x is retained in
metadata and explicitly marked inactive. Thus Radau cannot perturb an
inactive x into a nonzero a, and no nonzero-a trial is silently accepted.

Parameters, material and readout containers are frozen dataclasses. Tables
have immutable bytes backing, and source metadata use immutable mappings.
`load_material` obtains the frozen spectral tables and measured transistor
branches through `minimal_static_loader`, checks their source hashes, and
copies them into that immutable readout. No observation array is decoded;
the inherited loader does verify the original data-file hash. Candidate ID
and SHA256 have syntax checks; the experiment manifest binds them to the
actual candidate source.

Require finite lambda>=0, J_positive>0, kappa_positive>=0, A>=0,
kappa_a>=0, rho>0 and finite b. Enabled a additionally requires A>0 and
kappa_a>0. A stricter b/parameter search domain belongs to the predeclared
experiment plan. `enable_a=False` requires a0=0 and keeps a exactly zero in
readout and dynamics; merely choosing a0=0 with enable_a=True does not
disable nucleation.

## Circuit and near-zero behavior

BareCircuit enforces ideal instrument compliance by solving for the reduced
actual Vh. It does not cap the output current while retaining the command
field. SeriesCircuit closes the transistor load line against **total** soft
plus parallel current. A series compliance cap also reduces actual supplied
voltage. Measured transistor plateaux retain their voltage intervals; source
regulation begins after the cap-current plateau's largest allowed drop.

The port convention is:

```text
supply_voltage = local_voltage + transistor_voltage
command_voltage = supply_voltage + compliance_voltage
KCL_residual = total_material_current - transistor_current  # series
KVL_residual = command - local - transistor - compliance
```

Bare KCL is zero by its direct source connection. At zero command, every
port current/drop is exactly zero. The normal series root uses transistor
voltage and Brent's method; `reference=True` instead uses actual material
voltage and 64 bisections, with a separately expanded four-corner spectral
interpolation. This cross-check is not a replacement for an independently
implemented root validator or a direct spectral-reference check.

For `0<abs(Vh)<nearzero_anchor_v`, the soft current uses the signed table
endpoint's positive linear slope, connecting continuously to zero. The
anchor must be inside the frozen voltage table; default is its minimum.
The two states are integrated throughout this region. There is no near-zero
state freeze and no reuse of the archived soft-only omission bound.
The extension is explicitly a numerical approximation, not a certified
zero-field spectrum. Comparing the original direct small-field spectrum or
a second in-table anchor, such as 2*vmin, is meaningful; halving the anchor
on the same extrapolated line is not an independent check. See
`RATE_AND_ZERO_FIELD_REVIEW.md` for the independent small-field reference.

## Domains, history and failure semantics

The physical rectangle is `0<=q<=1, 0<=a<=A`; the finite inherited spectral
table imposes stricter Gamma bounds and `0<u<=80`. Native, accepted and RHS
trial states are checked. No state is clipped or projected. A trial-domain
failure, including a stiff solver trial leaving the rectangle, is retained
as a numerical/domain failure; it is not declared a physical rejection.
Continuous-equation positivity alone is not a numerical validation result.

`data` requires `command_voltage_v` and `initial_command_voltage_v`.
Optional observations and their original boolean mask must be supplied
together as `observed_current_a` and `scoring_mask`. No masking threshold,
current floor, weighting change, or residual normalization is inferred.
Optional `record_id` and `phase` are audit labels only.

`simulate_history` represents one declared continuous segment. Each item
accepts only `data` and `circuit`; it has no per-phase initial-state input.
Both terminal states become the next phase's initial states. All command
joins and available numeric SET/RESET record labels are checked before the
first phase. Observed labels permit SET n -> RESET n -> SET n+1, preventing
silent bridges across missing cycles 4 or 23. Separate missing-history
segments require separate calls and explicitly declared unknown entry
states; these are not predictions through the missing measurements.

The material object is passed once to the entire history. External circuits
can change for declared compliance protocols. No state is inferred from a
phase's measured target. Setting a positive current-derived state before
every cycle is not supported by the history interface.

## Results and independent access

`simulate_phase` returns `{'summary', 'arrays', 'residual'}`. Arrays include:

- Original command, path, observations and mask where supplied.
- `state_u`, `state_q`, `state_a_s` at every native command point.
- `current_a`, `soft_current_a`, `hard_current_a`, `supply_voltage_v`,
  `local_voltage_v`, `transistor_voltage_v`, `compliance_voltage_v`,
  `KCL_residual_a`, `KVL_residual_v`, `compliance_limited`,
  `nearzero_extension_used`.
- `unique_path_voltage_s`, `unique_internal_state_ux`,
  `accepted_steps_ux` with columns `(path,u,x)`, and `rhs_trace` whose columns
  are explicitly provided by the module's `RHS_COLUMNS` and summary.

Summary includes immutable-source provenance, all parameters, circuit and
precision, initial/final q/a/u, segments, original residual count/RMSE when
observations exist, the last RHS trial, and execution status. Rate traces
retain separate nonnegative capture/recovery/formation/loss terms.
`PhaseFailure.summary/arrays` retain failures and partial traces.

`simulate_history` returns `{'summary', 'phases', 'residual'}`; summary joins
record both states entering/leaving every phase. Residuals concatenate the
original masks. `HistoryFailure.result` preserves completed earlier phases
and the failed phase. No failed phase is omitted or retried.

For independent validation, `material.readout` exposes `lg`, `lv`,
`li_positive`, `li_negative` (all natural logs), signed `tx_*`/`ti_*`
tables (nonnegative voltage/current magnitudes in V/A), `ron_ohm`,
`gamma_c_ev`, `log_scale`, `vmin_v`, `vmax_v`, and `shape_metadata`.
The spectral-shape metadata carries its original Ron; apply `log_scale` or
replace Ron consistently when comparing the exact spectrum. Local circuit
points also retain private material/state identity fields so rate evaluation
cannot accidentally use another material/state's solved point.

## Interpretation gates still outstanding

Disabled-a agreement with the frozen scalar solver must account for the
deliberate new full near-zero integration. Enabled-a needs independent total
roots, both-state convergence, nucleation from a0=0, genuine history handoffs,
and passive branch checks before any model-quality claim. The old scalar
inverse states and their empirical integral fits do not transfer after a is
enabled. Near-zero checks and synthetic histories do not establish the full
46-pair/27023-residual/20-feature experimental contract or microscopic proof.

The old single-peak exclusion still applies whenever the actual RESET q is
nonincreasing. Under ideal bare voltage drive, componentwise ordered states
also remain ordered for this cooperative skeleton; merely adding a variable
does not generate the observed crossing. The actual continuous history must
address those constraints. None of these experimental or numerical checks
was executed while preparing this module.
