# Common-lambda forward interface — prepared, not executed

This module is an explicit derivation of
`analysis/joint_history_calibration_2026-11-04/joint_forward.py`, whose frozen
SHA256 is `c7e25af2ccafffcf05ebfdcb9d28a6bca63c799b4dde5e92d733b334cfca53db`.
`FROZEN_PARENT_DIFF.patch` records all changes. The parent and its
`_joint_fixed_material` guard are untouched; no monkeypatch is used.

## Entry points

```python
readout = load_lambda_readout()
candidate = with_parameters(
    readout,
    lambda_per_v=declared_lambda,
    lambda_domain_per_v=(declared_lower, declared_upper),
    kappa_per_v2=declared_kappa,
    J_positive_a=declared_J,
    b=declared_b,
    candidate_source_id=declared_source_id,
    candidate_source_sha256=declared_source_sha256,
)
result = simulate_phase(
    candidate, data, u0, declared_b, precision, method, check_budget,
    collect_rhs=False,
)
```

Every argument in `with_parameters` is required and keyword-only. The old-λ
control must use this same candidate entry, with its own explicit values and
provenance. No λ range or candidate is selected by this module. The inclusive
domain must contain two finite numbers satisfying
`0 <= lower <= lambda_per_v <= upper`; negative λ is not supported because
the unchanged near-zero bound relies on nonnegative λ. The frozen parameter
class also requires finite κ ≥ 0 and J > 0. Shared b must be finite.
Any stricter scientific candidate constraints belong to the separate
experiment plan and runner, including the proposed b ∈ [0, ln(10)] range
and any strict-positivity screening; this interface does not enforce those
additional selections.

The source ID must be a nonempty string and the SHA256 must contain 64
hexadecimal characters (stored lowercase). This validates provenance syntax;
the caller's experiment manifest must bind the ID and hash to the actual
candidate source. It is not an automatic file lookup or verification.

## Snapshots and checks

- `_lambda_readout_snapshot` is an immutable mapping of Ron, Γc, base Ron/Γc,
  and base log scale. The original tables/views retain the parent's read-only
  flags, and an immutable identity tuple detects replacement of those arrays.
- `_lambda_dynamics_snapshot` is a separate immutable mapping of common λ,
  κ, J, b, the declared λ domain, and candidate provenance. At every phase
  entry, all four actual common parameters, including the supplied b, must
  equal this snapshot. A readout without a candidate snapshot cannot run.
- `_lambda_reference_parameters` is an immutable copy of the original
  parameters, used only for the parent's unchanged κ-based first-step rule.
  It is not the candidate dynamics snapshot.

The loader checks the frozen parent source hash. Per-phase results retain the
parent source hash, fixed readout, and shared dynamics/provenance. The array
flags and snapshots guard normal use; they are not a security boundary against
deliberately replacing private attributes or changing NumPy memory externally.

## Phase inputs and unchanged numerical contract

Required `data` fields are `command_voltage_v`,
`initial_command_voltage_v`, `nearzero_segment_budget`, and `circuit_mode`.
The near-zero budget must be explicitly supplied with
`0 < budget <= 1e-10`. Circuit mode is `fast` or `strict_bisection42`.
`observed_current_a` and the original boolean `scoring_mask` must either both
be supplied or both be absent. Optional `record_id` and `phase` are metadata.
The incoming u, b, signed-kernel `Precision`, explicit method (`DOP853` or
`Radau`), and budget callback (or `None`) remain caller-owned.

The signed q² kinetics, direct-u ODE, circuit, near-zero handling, state/domain
rules, waveform clock, precision, first-step formula, failure preservation,
and complete integration body remain unchanged. There is no candidate
selection, automatic retry, state reset, inversion, fitting, or experiment
driver here. Continuous state handoffs and any missing-cycle treatment must
be specified by a separate experiment plan; this interface does not infer them.

## Existing validation and preparation checks

Reuse the unmodified
`analysis/joint_history_calibration_2026-11-04/joint_validation.py`
(SHA256 `c254dccc6e2d51fd1bc0bd18ef686f01b2e7b263dce3b45f97ca0d5b383a3e1d`):

```python
phase_integrity(candidate, result, data, u0, b,
                independent=False, check_budget=check_budget)
compare_phases(nominal_result, refined_result)
```

Final independent validation requires `collect_rhs=True` during the forward
phase and `independent=True` in `phase_integrity`. These calls are interface
documentation, not actions performed during preparation.

Preparation checks are source-only: builtin `compile`, parent/hash checks,
and exact comparison of `simulate_phase` after normalizing its entry guard,
metadata fields, and renamed numerical-reference attribute. The module was
not imported, its loader was not called, and no numerical validation, forward
model, ODE, inversion, or fit was executed. Successful compilation is neither
numerical validation nor evidence that a common-λ model explains the data.
