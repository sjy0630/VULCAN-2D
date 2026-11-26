# Soft drive structure and retained RESET memory

This stage follows the frozen `reset_shape_2026-11-03` stage. It is a
training-data diagnosis, not a replacement of the production material model.
The original goal of an accurate common 1R/1T1R model remains unachieved.

## Questions fixed before new calculations

1. Does the saved beta=6 local-field SET trajectory stop changing near the
   apex, and where does its outward error accumulate?
2. Under the same spectrum and transistor map, what state and drive are
   required by the measured training cycle? Can a nonnegative mobility
   multiplier alone give the required direction of evolution?
3. If the Ohmic branch can only decay after the preceding SET read, how
   large must a changing soft RESET contribution be? A second state that
   only changes a decay rate is not automatically sufficient.

Separate scripts declare their finite arithmetic/inverse budgets before
execution. No new physical ODE or optimization is authorized by this file.
A physical candidate, if justified, needs its own explicit finite plan.
No holdout records, desktop defaults or remote repository are changed.

## Fixed interpretation

- Bare records are one device in numeric order, same-label SET then RESET.
  Missing cycles 4 and 23 were measured but not supplied. Available segments
  are 1–3, 5–22, 24; the frozen training segments are 1–3 and 5–15.
- A future full simulation must carry every material state through the
  known history. No per-cycle fitted state resets, rate/threshold parameters,
  or fitted random realizations.
- The SET probe is conditioned on the same cycle's 0.2 V outward read.
  Its 500-point scores are conditional training diagnostics, not future
  predictions or a score for all 27,023 original joint residuals.
- The fixed empirical feature bands are aggregate training Q10–Q90 bands,
  not instrument error bars. Selected-point inverse conflicts do not by
  themselves establish unacceptable whole-curve error.
- State inversions depend on the assumed spectrum and external transistor
  map. Prior independent gate interpolation and mapping sensitivities remain
  in `CONSTRAINED_SHARED_RESULTS_2026-11-03.md`; do not treat them as measured
  calibration uncertainty or silently recalibrate the transistor.
- Unknown scan clocks and waits remain unknown. All current path rates are
  per command-voltage path, not identified rates per second.
- Additional memory, stochastic transitions, protocol differences, and
  alternative readout structure remain competing explanations. No unique
  microscopic state, species, or filament is established by a successful
  effective circuit fit.

## Stop and continuation rules

Preserve every failure, distinguish numerical resolution from physical
incompatibility, and do not expand a closed scan. Quantify finite residual
tradeoffs before rejecting approximate models. Any improvement to one phase
must be reported beside its scope and the untested remaining phases.
