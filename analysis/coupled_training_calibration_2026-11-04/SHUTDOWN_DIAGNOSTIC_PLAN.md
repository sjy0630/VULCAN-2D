# Predeclared finite shutdown diagnostic after calibration terminal state

Prepared only. Execute once only after the root agent explicitly confirms that calibration has terminated. The script additionally requires `RESULT.json` status `bounded_search_and_final_checks_finished`, unchanged sources, and a verified final candidate at both precisions. It must find the formally accepted coarse request 26 and the matching final refined request 35. No provisional or earlier best candidate is substituted.

## Scope and inputs

- Use the selected final refined artifact and its saved actual RESET entrances, return states and currents for all 14 bare training records: 1,2,3,5,...,15.
- Use the original `1r.reset.return.I_0p2V` record eligibility/native indices and empirical median gate. Compare against that existing gate upper bound; do not introduce new per-record pass bands.
- Use the frozen negative spectral table and its original Ron amplitude scaling. No module importing a model, no ODE, fit, inverse, root or new spectral quadrature is permitted. No original calibration file is written.
- Hash RESULT, selected JSON/NPZ, the original feature contract, frozen table, this script and this plan before/after. Refuse an existing diagnostic result or exclusive run marker.

## Exactly bounded work

For each record, validate the saved bare RESET waveform, absence of compliance, local voltage equal to command, actual `(q0,a0)` entrance matching its saved phase summary, and original return sample. Compute:

1. `Dq=2 B(Vmax)-B(vread)` and `Dv=Vmax^2-vread^2/2`, using actual native magnitudes, with B the existing analytic integral of v exp(lambda v).
2. `q_min=1/(1/q0+kappa_positive exp(2b)Dq)`.
3. `log(a_min)=log(a0)-kappa_a rho exp(b)Dv`, or a_min=0 if a0=0.
4. The existing saved soft/parallel/total currents, actual-state distances from these envelopes, and instantaneous capture/recovery/formation/loss terms by algebra at that saved point. These rate terms are point diagnostics, not an integral of trial RHS records.

Before any new soft-current evaluation, verify the negative log-current table is nondecreasing in log-Gamma in **both voltage columns bracketing that native read voltage**, across the full Gamma grid. Require positive ordered table axes, finite entries, in-domain read voltage, and in-domain Gamma(q_min) and saved Gamma. No negative increment is silently accepted or repaired. Also require the saved actual q/a to satisfy the computed lower bounds; violations are retained as numerical/unresolved diagnostics.

Then evaluate the same frozen log-bilinear negative soft readout **once at q_min per eligible record, at most 14 evaluations total**. This is new forward readout arithmetic and must be counted as such. It is not the saved current at the actual q. Monotonicity plus nonnegative parallel conduction yields the model-conditional mathematical bound

`I_total(read) >= I_soft(vread,q_min) + vread*a_min`.

The result is a necessary bound for this fixed candidate from its actual incoming states, under the fixed table/path/circuit model. Floating-point evaluation is recorded; it is not an interval-arithmetic certificate, experimental noise interval or proof for other parameter values/entrances. Table/domain failure skips that record's new readout and retains an unresolved result; no extrapolation or retry.

Maximum work: 14 record diagnostics, 14 new soft-current evaluations, 14 saved-point rate-component calculations, 0 ODEs, 0 fits, 0 inversions, 0 scalar/circuit roots, 0 quadratures. Numerical work budget 30 s, I/O and hashing excluded. No automatic retry. Inputs remain immutable.

## Outputs and interpretation

Write `SHUTDOWN_DIAGNOSTIC.json` and `.md` in this directory only, plus the exclusive `SHUTDOWN_DIAGNOSTIC_RUN_STARTED.json` marker. Preserve every record and failure. Aggregate lower-bound medians are defined only if all original eligible records have completed bounds; otherwise mark unresolved rather than omitting records.

- If the soft-only lower-bound median exceeds the existing upper gate, the selected q recovery budget is insufficient from its actual entries even with capture removed.
- If the parallel-only lower-bound median exceeds it, even maximal parallel-path loss with formation removed is insufficient from those entries.
- If only the combined bound exceeds it, report that combined conditional obstruction.
- If the bound permits the gate but actual current fails, the bound does not exclude the candidate's capacity; inspect actual sources and state gaps. An a gap reflects formation **and** reduced loss through `(1-q)`, not formation alone. A pointwise source ratio cannot establish its full-phase integrated contribution.

Per-record exact experimental ratios are diagnostics only. No family-wide rejection or new q¹ family follows. The original calibration result is not modified or re-ranked by this diagnostic.
