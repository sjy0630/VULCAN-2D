# Fixed q¹ seed derivation

Status: `completed_reference_seed_declaration_no_forward_evaluation`.

This is a deleted-capture reference from saved q² RESET entrances, not a q¹ fit or trajectory.
The shared median-gate upper limit defines a common seed target only; no per-record gate is added.

- Native return voltage: -0.20000000298023199 V
- Original median-gate upper limit: 4.7700275673800943e-10 A
- q target at a = 0: 6.3078085707599324e-05

Reference κ₁: **0.057653196675049923 V⁻²**.

| Fixed multiplier | Shared κ / V⁻² |
|---:|---:|
| 1 | 0.057653196675049923 |
| 3 | 0.17295959002514977 |
| 10 | 0.57653196675049923 |

All other parameters and all three original segment entries are unchanged.

κ scales capture and recovery in both polarities. Actual q¹ entrances, capture feedback and a histories will change.
This reference is not a family-wide rate bound, a reachability guarantee or a physical-time calibration.
The subsequent three full forward comparisons require separate execution authorization.

```json
{
  "saved_record_checks": 14,
  "monotonicity_audits": 1,
  "analytic_inverse_attempts": 1,
  "analytic_inverse_completed": 1,
  "scalar_rate_calculations": 14,
  "candidate_declarations": 3,
  "ODEs": 0,
  "fits": 0,
  "forward_trajectories": 0,
  "iterative_inverses": 0,
  "scalar_roots": 0,
  "circuit_roots": 0,
  "spectral_quadratures": 0
}
```

Inputs unchanged: `True`. No interval-arithmetic certificate.
