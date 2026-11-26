# Reversible forward preparation — 2026-11-03

Status: **prepared and retained unexecuted by root decision**. The fixed initial-state barrier result excludes this seed as an error-reduction candidate for every finite κ≥0: even the 0.3% material-envelope optimistic original500-point RMSE lower bound is 0.490896 decade, above the prior conditional 0.182231-decade result. Independent barrier review has passed (see section 11 of `REVERSIBLE_BUDGET_REVIEW.md`); these values were not recomputed by this implementation task. Root therefore instructed that no κ be selected and no forward execution plan be created.

## Frozen prepared source

- File: `reversible_forward.py`
- SHA256: `801fb609e4d968d16f7752024ce8a05a83e18acbec842a101c66486940f098d8`
- Only this new analysis directory was edited. Earlier analysis stages were not edited or rerun.
- No `reversible_forward.plan.json`, forward result JSON/NPZ, or actual forward run was created during preparation.

## Implemented scope

One conditional cycle-4, soft-only SET probe using the exact `reversible_budget.json` seed. Both signs of `u′ = κv[exp(λv) − I expm1(u)/J]` are retained. The prepared implementation conditions on the saved outward 0.2 V anchor, integrates the same ODE backward to zero, then integrates outward 0→5 V and return on cumulative command path 5→10 V. The apex state is continuous; duplicate native endpoints add no dwell.

The two prescribed DOP853 precisions, original 500-point mask, all seven scoring groups, three actual endpoint checks, all six single-cycle reference comparisons, threshold crossings, native/RHS KCL checks, u/q/current arrays, failed-attempt evidence, and two-precision state/current comparison are implemented. The 120-second limit and one-seed/two-attempt/six-segment cap are encoded. No RESET, bare-history, holdout, or aggregate validation is included.

The scalar circuit root bracket is restricted to legal material/transistor domains: transistor drop ∈ [0,min(Tmax,A−vmin)]. A subtraction-rounding endpoint is adjusted inward by one floating-point unit only if needed; no state/current/solved-voltage clipping is applied. A root requiring local v<vmin is numerical-unknown. The prepared script compares 36 fixed roots with the original solver and preserves comparison failures before fallback.

For 0<A<vmin, the approved Lorentz-area bound gives B = ½κvmin²[exp(λvmin)+πΓc/(Ron J)]. The script permits a bounded RHS omission only for B<10⁻¹² and valid u/Γ, counts physical microsegments, and labels this a local integral bound rather than a trajectory-error certificate. Exact A=0 has I=u′=0. Bytecode writes are disabled before importing frozen analysis modules.

## Validation actually performed

1. Syntax-only `compile(source, path, 'exec')` completed successfully. This did **not** import or execute the script.
2. `/root/shared_kernel_review` performed a read-only code review. The reviewer confirmed the reversible-law sign, backward conditioning, two-precision/failure accounting, and original scoring mask. Their bracket-domain concern was repaired and the updated bracket was re-reviewed without another blocking concern.

No transport current, anchor inversion, rate grid, numerical root comparison, ODE, fitting, or numerical acceptance test was executed by this implementation task. Read-only review and syntax validity do not establish physical correctness, accuracy, or runtime success.

## Deferred run interface

If root later decides to run the prepared experiment, supply `--fixed-rate-plan <path>` with an internal experiment record containing `budget_sha256`, explicit positive `kappa_per_v2`, `selection_rule`, and `decision_reference`. There is no default rate, automatic rate derivation, or automatic rerun. This is internal experiment accounting, not a new user-permission requirement. The actual forward plan is written only at a future explicit execution, and existing output artifacts cannot be overwritten.

The reversible candidate remains a two-point-calibrated single-cycle hypothesis. The saved 4 V equilibrium is locally repelling and the 2 V equilibrium attracting. A future completed forward would not establish the complete common 1R/1T1R model or the original 32-cycle aggregate gates.
