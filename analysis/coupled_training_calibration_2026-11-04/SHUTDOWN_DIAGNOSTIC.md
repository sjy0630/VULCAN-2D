# Selected-candidate RESET shutdown diagnostic

Status: `completed_conditional_shutdown_diagnostic`. Selected coarse request 26; final refined request 35.
No new ODE, fit, inverse or root was performed. New frozen soft-readout evaluations are counted below.

The lower bounds are conditional on the selected parameters, actual RESET entrances, frozen table and path clock.
They are not parameter-family exclusions or experimental uncertainty intervals.

| Record | q entry | q lower | q saved | a lower / S | a saved / S | Soft lower / A | Total lower / A | Saved total / A | Observed / A |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.211865376 | 0.000123038395 | 0.000540625678 | 4.22469007e-37 | 6.62856616e-08 | 9.30544375e-10 | 9.30544375e-10 | 1.73494543e-08 | 6.6937178e-11 |
| 2 | 0.211865375 | 0.000123038395 | 0.00054062567 | 4.22469007e-37 | 6.62856608e-08 | 9.30544375e-10 | 9.30544375e-10 | 1.73494541e-08 | 5.49247314e-11 |
| 3 | 0.211865376 | 0.000123038395 | 0.00054062568 | 4.22469007e-37 | 6.62856619e-08 | 9.30544375e-10 | 9.30544375e-10 | 1.73494544e-08 | 4.76620514e-11 |
| 5 | 0.211865376 | 0.000123038395 | 0.000540625681 | 4.22469007e-37 | 6.6285662e-08 | 9.30544375e-10 | 9.30544375e-10 | 1.73494544e-08 | 2.02617895e-10 |
| 6 | 0.211865376 | 0.000123038395 | 0.000540625684 | 4.22469007e-37 | 6.62856625e-08 | 9.30544375e-10 | 9.30544375e-10 | 1.73494545e-08 | 1.08719045e-09 |
| 7 | 0.211865376 | 0.000123038395 | 0.000540625683 | 4.22469007e-37 | 6.62856623e-08 | 9.30544375e-10 | 9.30544375e-10 | 1.73494545e-08 | 3.44991362e-11 |
| 8 | 0.211865376 | 0.000123038395 | 0.000540625683 | 4.22469007e-37 | 6.62856623e-08 | 9.30544375e-10 | 9.30544375e-10 | 1.73494545e-08 | 1.88615443e-11 |
| 9 | 0.211865376 | 0.000123038395 | 0.000540625683 | 4.22469007e-37 | 6.62856623e-08 | 9.30544375e-10 | 9.30544375e-10 | 1.73494545e-08 | 5.07449194e-10 |
| 10 | 0.211865376 | 0.000123038395 | 0.000540625683 | 4.22469007e-37 | 6.62856623e-08 | 9.30544375e-10 | 9.30544375e-10 | 1.73494545e-08 | 1.63869578e-11 |
| 11 | 0.211865376 | 0.000123038395 | 0.000540625683 | 4.22469007e-37 | 6.62856623e-08 | 9.30544375e-10 | 9.30544375e-10 | 1.73494545e-08 | 1.68393452e-10 |
| 12 | 0.211865376 | 0.000123038395 | 0.000540625683 | 4.22469007e-37 | 6.62856623e-08 | 9.30544375e-10 | 9.30544375e-10 | 1.73494545e-08 | 4.0596107e-10 |
| 13 | 0.211865376 | 0.000123038395 | 0.000540625683 | 4.22469007e-37 | 6.62856623e-08 | 9.30544375e-10 | 9.30544375e-10 | 1.73494545e-08 | 2.71226062e-11 |
| 14 | 0.211865376 | 0.000123038395 | 0.000540625683 | 4.22469007e-37 | 6.62856623e-08 | 9.30544375e-10 | 9.30544375e-10 | 1.73494545e-08 | 1.94053162e-10 |
| 15 | 0.211865376 | 0.000123038395 | 0.000540625683 | 4.22469007e-37 | 6.62856623e-08 | 9.30544375e-10 | 9.30544375e-10 | 1.73494545e-08 | 8.65505098e-11 |

## Original median gate

- soft_lower_median_a: 9.305443748745389e-10
- hard_lower_median_a: 8.449380262735998e-38
- total_lower_median_a: 9.305443748745389e-10
- saved_total_median_a: 1.7349454476732183e-08
- saved_soft_median_a: 4.092321821129519e-09
- saved_hard_median_a: 1.3257132655602664e-08
- observed_median_a: 7.67438439186474e-11
- original_upper_gate_a: 4.770027567380094e-10
- soft_recovery_bound_exceeds_upper: True
- parallel_max_loss_bound_exceeds_upper: False
- combined_bound_exceeds_upper: True

The selected q recovery envelope already exceeds the original median upper gate from its actual RESET entrances, even with capture removed. These are fixed-candidate/actual-entry statements, not exclusions of the parameter family or a new model-family recommendation.

The saved actual soft current is distinct from the newly evaluated soft current at q_min.
An a-state gap above its envelope includes continued formation and the reduction of loss by (1-q).
Saved-point flux ratios cannot establish integrated contributions over the phase.
Per-record observed-current ratios are diagnostics only; the original median gate is unchanged.

## Work and provenance

```json
{
  "records_planned": 14,
  "records_attempted": 14,
  "records_completed": 14,
  "new_soft_current_evaluations_attempted": 14,
  "new_soft_current_evaluations_completed": 14,
  "saved_point_rate_arithmetic": 14,
  "ODEs": 0,
  "fits": 0,
  "inverse_mappings": 0,
  "scalar_roots": 0,
  "circuit_roots": 0,
  "new_spectral_quadratures": 0
}
```

- `/Users/sujiangyu/Desktop/SIM2RRAM/VULCAN-2D/analysis/coupled_training_calibration_2026-11-04/calibration_run/RESULT.json`: `b0c5a646aff8c2d139e4a975af600dda90b6861a3e595ab2d55f1ee81c686939`
- `/Users/sujiangyu/Desktop/SIM2RRAM/VULCAN-2D/analysis/coupled_training_calibration_2026-11-04/calibration_run/cases/035_best_refined.json`: `59005008be69e23a0579425e11060886734ba86ed485bdb7ed92ab84a1cdb524`
- `/Users/sujiangyu/Desktop/SIM2RRAM/VULCAN-2D/analysis/coupled_training_calibration_2026-11-04/calibration_run/cases/035_best_refined.npz`: `e122b92213d646a30ccdc2ceee5b4599a8f2f6b2cd7fe8d5dd899e44336f418c`
- `/Users/sujiangyu/Desktop/SIM2RRAM/VULCAN-2D/analysis/constrained_shared_2026-11-03/feature_constraints.json`: `7497d1d5a5462901a4d3121afae595f9213ee3eea6214f86c9e0c2af6fbc004f`
- `/private/tmp/vulcan-runtime-cache/transport_g010_v0025.npz`: `00d6d9ea0b1dfe855181e7016afc6dd5e23d169f83f19cfcedfc9d3b35f9e4e2`
- `/Users/sujiangyu/Desktop/SIM2RRAM/VULCAN-2D/analysis/coupled_training_calibration_2026-11-04/shutdown_diagnostic.py`: `61998584affe1eb5e81245fea0e36f3e4d0a25bf7bc24a523e7987188450d928`
- `/Users/sujiangyu/Desktop/SIM2RRAM/VULCAN-2D/analysis/coupled_training_calibration_2026-11-04/SHUTDOWN_DIAGNOSTIC_PLAN.md`: `f5206fdb021908d4b280459e28d7fac74ee67b75832e381581e439731f01148b`

Inputs unchanged: `True`. Floating-point arithmetic; no interval-error certificate.
