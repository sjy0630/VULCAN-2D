# Conditional early-SET lower bound for declared q¹ candidates

Status: completed using original saved inputs and closed-form arithmetic only.
No running q¹ trajectory, ODE, new transport-table readout, inverse, root, fit or new score was used.

## Derivation and scope

For an initially unregulated monotone positive sweep, local voltage v equals command voltage and ds=dv. Nonnegative capture in the q¹ law gives q′ >= -kappa*v*exp(lambda*v)*q. Thus on 0 <= v <= V*, q >= c = q0*exp[-kappa*B(V*)].

Since 0<=a<=A and q>=c, formation is at least exp(b)*c*(A-a), and the magnitude of loss is at most rho*exp(-b)*a. Therefore

    da/dv >= kappa_a*v*[exp(b)*c*A - (exp(b)*c + rho*exp(-b))*a],
    a(V*) >= A*exp(b)*c/h * [1-exp(-kappa_a*h*V*^2/2)],
    h = exp(b)*c + rho*exp(-b),
    I_total(V*) >= V* a_lower(V*).

B(V)=integral_0^V u exp(lambda*u) du has units V², kappa and kappa_a have units V^-2, a and A have units S, and the final current has units A. The comparison uses the original a0=0 entries, not later q² RESET states.

If compliance activated at any earlier time, v=command fails and this current bound is **not** an actual-current lower bound. The alternative is an early first crossing of compliance, ahead of the selected record's observed transition bracket.

## Original two-record observations and conditional bounds

The original approximately 2V median gate upper limit is 1.8768527638712884e-07 A. This limit is a comparison scale, not a new per-record gate.

| Candidate | Record | Conditional current lower / A | Observed / A | Lower / observation | Lower / original median upper | Observed transition bracket / V |
|---|---:|---:|---:|---:|---:|---|
| q1_capacity_rate_x1 | 1 | 1.03641842e-05 | 3.50065221e-08 | 296.064 | 55.2211 | [2.65000009536743, 2.70000004768372] |
| q1_capacity_rate_x3 | 1 | 6.75912687e-06 | 3.50065221e-08 | 193.082 | 36.0131 | [2.65000009536743, 2.70000004768372] |
| q1_capacity_rate_x10 | 1 | 1.51375476e-06 | 3.50065221e-08 | 43.2421 | 8.06539 | [2.65000009536743, 2.70000004768372] |
| q1_capacity_rate_x1 | 5 | 6.03447201e-05 | 2.02095066e-07 | 298.596 | 321.521 | [2.65000009536743, 2.70000004768372] |
| q1_capacity_rate_x3 | 5 | 3.93714776e-05 | 2.02095066e-07 | 194.817 | 209.774 | [2.65000009536743, 2.70000004768372] |
| q1_capacity_rate_x10 | 5 | 8.82304995e-06 | 2.02095066e-07 | 43.6579 | 47.0098 | [2.65000009536743, 2.70000004768372] |

For each candidate/entry pair, either 1mA compliance has already occurred before the approximately 2V read (earlier than that record's observed 0.995mA transition bracket), or the no-compliance hypothesis yields the listed total-current lower bound. These are two-record conditional shape diagnostics, not a full-median gate failure or family exclusion. No running q1 artifact was read.

Only records 1 and 5 are checked. No conclusion that the original 14-record median gate fails follows from these two records alone. The complete finite-run result remains separate.

## Provenance

```json
{
  "saved_protocol_records": 2,
  "closed_form_candidate_record_bounds": 6,
  "ODEs": 0,
  "new_table_readouts": 0,
  "inverses": 0,
  "roots": 0,
  "fits": 0,
  "new_score_evaluations": 0
}
```

- `/Users/sujiangyu/Desktop/SIM2RRAM/VULCAN-2D/analysis/coupled_linear_recovery_2026-11-04/FORWARD_DECLARATION.json`: `a0e4f9bf6ff99b2bc06f4064c86e3ee9a299bd5136c45a5eb27e1cef2c91f956`
- `/Users/sujiangyu/Desktop/SIM2RRAM/VULCAN-2D/analysis/constrained_shared_2026-11-03/feature_constraints.json`: `7497d1d5a5462901a4d3121afae595f9213ee3eea6214f86c9e0c2af6fbc004f`
- `/Users/sujiangyu/Desktop/SIM2RRAM/VULCAN-2D/analysis/coupled_training_calibration_2026-11-04/calibration_run/cases/035_best_refined.json`: `59005008be69e23a0579425e11060886734ba86ed485bdb7ed92ab84a1cdb524`
- `/Users/sujiangyu/Desktop/SIM2RRAM/VULCAN-2D/analysis/coupled_training_calibration_2026-11-04/calibration_run/cases/035_best_refined.npz`: `e122b92213d646a30ccdc2ceee5b4599a8f2f6b2cd7fe8d5dd899e44336f418c`

All inputs unchanged. Floating-point evaluation of analytic inequalities; no interval-arithmetic or experimental-noise certificate.
