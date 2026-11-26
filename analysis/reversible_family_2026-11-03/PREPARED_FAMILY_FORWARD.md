# Reversible-family forward kernel — preparation only

Status: prepared; only syntax compilation has been performed. The module has not been imported, loaded, validated numerically, integrated, or fitted. The parent will predeclare the finite three-parameter family calibration separately. Releasing the earlier two-return-point equilibrium constraint does not by itself add a fourth physical parameter, validate the family, or license an unrestricted search.

## Prepared source

- `forward_kernel.py`
- SHA256: `4382a56449b0b736a963356b1257873f49e0cf3125f74d6f2ec6f9aa865f0f17`
- No earlier stage, frozen runner, source dataset, transport table, or production default was edited.
- The module disables bytecode writing before importing reusable modules. No output writer, automatic seed/rate selection, timer, optimizer, or automatic retry is included.

## Explicit interface

`Parameters(lambda_per_v, J_positive_a, kappa_per_v2)` has no defaults. It requires finite lambda>=0, J>0, kappa>=0, matching the already selected family. The root's tighter predeclared fit bounds remain its responsibility; no parameter clipping occurs. `Precision(name, rtol, atol, max_step)` is also explicit.

`ForwardKernel.load(cache)` reads only the same cycle-4 arrays from `native_observations.npz`, the frozen spectral table, transistor lookups, original anchor and reference records. The four runtime-cache hashes and native503/eligible500/outward250/return250 identities are checked. This is a callable loader, not an import-time action. Its fixed Ron, gamma_c and .2V anchor are read from the existing saved sources; no lambda/J/kappa seed is read or inferred.

`validate_circuit(check_budget=...)` is an explicit future numerical action, not executed during preparation. It compares fast Brent and strict-domain42-bisection circuits against the original42 reference at the same36 fixed command/state pairs. The returned report preserves failures and mode-specific maxima. The caller chooses a mode; there is no silent fallback. Criteria retain the previous 5e-7 decade current, 1e-9 V local voltage and 1e-13 A KCL tolerances.

`evaluate(parameters, precision, circuit_mode=..., check_budget=..., collect_rhs=False)` returns a dictionary containing:

- `summary`: parameters/precision, segment records,7 scoring groups,3 actual endpoint checks,6 return references, threshold crossings, anchor/apex continuity, KCL and near-zero accounting;
- `arrays`: all503 native voltages/observations, original scoring mask, state u/q, current/local voltage/KCL, original-order log residuals, and accepted coordinates/states for every completed segment;
- `residual`: exactly the original500 eligible signed log10 residuals in native order, with no floor or exclusions.

`KernelFailure.summary` and `.arrays` carry failures/partial completed segments. Domain/resolution/solver/budget failures are explicitly not physical rejection. Root failures include command, state, segment and error; optional full RHS traces use NaN for deliberately unevaluated subtable current rather than inventing a zero. The parent must save every trial, impose the hard total/evaluation budgets, catch failures and decide stopping. A `check_budget()` callback is invoked before RHS/circuit work; the kernel does not override the parent's timeout handler.

`audit_native_legacy(result, check_budget=...)` is an optional explicit postfit action returning its report and reference arrays. It was not run during preparation.

## Conditioning and physical trajectory

The same signed ODE is used everywhere:

`u_prime = kappa * local_v * (exp(lambda * local_v) - current * expm1(u) / J)`.

The fixed outward .2 V anchor starts an outward segment .2→5 V; its actual terminal state starts the return segment s=5→10 with command A=10−s. A separate reverse integration of the same equation .2→0 supplies the native low-voltage outward points. Thus all original500 points are scored, the apex is continuous, duplicate endpoints add no dwell, and the anchor is imposed once without a backward-then-forward roundtrip discrepancy. The backward solution defines the conditional initial state; this is not a free fit or a reset between physical branches.

There are three segments per evaluation: anchor→apex, apex→zero, anchor→zero backward. Only the last two cross the subtable microsegment. There is no assertion that u or q is monotone. q remains the spectral-width coordinate exp(−u), not a measured trap occupation. A successful family calibration would remain conditional single-cycle SET, not full bare1R/1T1R history or32-cycle aggregate validation.

## Domain and circuit handling

All actual states must satisfy0<u<=80 and Gamma=gamma_c/expm1(u) inside the original finite table. No state/current/voltage clipping or domain expansion is allowed. Both root modes bracket transistor drop on [0,min(Tmax,A−vmin)]; an inward one-ULP adjustment is permitted only to keep the floating endpoint inside the actual material domain. A root requiring a subtable local voltage is unresolved, not silently evaluated.

The original cached42 solver contains a lower-voltage safeguard. It is used only as an audit reference after the kernel has checked that all its original bisection trial points stay inside the material/transistor domains. The ODE uses only the strict-domain fast or strict42 implementation.

At A=0, I=u_prime=0 exactly after state validation. For0<A<vmin, the Lorentz-area bound is

`B = 0.5*kappa*vmin**2*(exp(lambda*vmin) + pi*gamma_c/(Ron*J))`.

The kernel rejects an evaluation unless B<1e-12. It records omitted physical microsegments separately from RHS retries. This bounds a local omitted integral, not global trajectory error or exact inactivity.

## Verification actually performed

Syntax-only `compile(source, path, 'exec')` passed. No import, current/root evaluation, ODE, optimization, parameter scan or physical numerical acceptance check was performed. Independent read-only review by `/root/shared_kernel_review` found the ODE sign, anchor connection, scoring and root domains consistent. Its request to restrict lambda>=0 to the already selected family was implemented and syntax-checked. TimeoutError is wrapped as KernelFailure with failure_scope=caller_budget_exhausted; the parent must stop when this identifies exhaustion of its overall budget. This source review is not a numerical validation.
