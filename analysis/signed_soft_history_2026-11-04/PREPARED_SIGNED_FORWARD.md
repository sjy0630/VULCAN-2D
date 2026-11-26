# Signed soft m=2 kernel preparation

Prepared implementation: `signed_forward_kernel.py`.

This preparation performed only source inspection, text editing, SHA256 computation and Python built-in `compile(source, filename, 'exec')`. **The module has not been imported, loaded or executed by the implementation agent. No circuit roots, ODEs, fits, parameter scans or held-out observation reads were performed.** The parent owns every numerical run and the finite experiment ledger.

## Prepared behavior

- Keep the validated positive m=2 parameters, Ron, Γc, material current and positive circuit readout frozen. The positive circuit delegates to the preceding stage directly.
- The negative circuit uses material table `transport.li[1]` and measured transistor branch `transport.knots['reset']`. It enforces the actual negative transistor voltage domain, whose maximum magnitude is 2 V; it does not mirror the positive branch or extend a lookup.
- Solve a passive circuit with signed command, local voltage, transistor voltage and current. Restore signs only after solving the magnitude KCL root. Fast Brent and strict 42-step bisection are explicit choices.
- Select dynamics from the **actual local-voltage sign**, including return sweeps. For v=|VhBN|, i=|I|, a=0 and m=2:

  Positive: u′=κv[exp(λv−u)−(i/J+)expm1(u)].

  Negative: u′=κv[exp(2b+λv−u)−(i/J+)exp(−2b)expm1(u)].

- Supply `b`, initial state, previous command voltage and numerical precision explicitly. There is no conditioned 0.2 V anchor or hidden state reset.
- Use cumulative actual command-voltage distance as the clock. Duplicate endpoint voltages add no artificial dwell. Turnarounds and actual zero crossings split segments; no seconds or inter-cycle wait are inferred.
- Maintain the preceding finite state/table interface (`0<u<=80` plus original Γ domain). No clipping, projection, lower state floor, domain expansion, trial retry or hidden fallback is added.
- Permit signed near-zero omission only for command magnitudes strictly below the original voltage table minimum and only when the relevant analytic bound is below 1e−12. Exact zero has I=u′=0. A command at/above the table minimum whose local circuit root lies below it remains a resolution failure.
- Count each physical monotone near-zero microsegment at most once, independent of solver retries. The reported omitted integral is a local bound, not a full-trajectory error certificate.
- Separate waveform dynamics from observations and scoring masks. Observations are positive amplitudes in this dataset; residuals use log10(|Ipred|/|Iobs|) while predictions retain physical current signs.
- Preserve failures as `PhaseFailure(message, summary, arrays)` with partial arrays, root failures, segment ledger and last RHS. Numerical failures are not physical rejection. A caller TimeoutError maps to `caller_budget_exhausted`.

## Public interface

```python
Precision(name, rtol, atol, max_step)
PhaseFailure(message, summary, arrays)

kernel = SignedSoftKernel.load()
phase = kernel.load_training_phase(record_id, phase)
circuit = kernel.make_circuit(mode='fast')  # or 'strict_bisection42'
result = kernel.simulate_phase(
    command_voltage, circuit, initial_u, b, precision,
    initial_command_voltage_v=previous_command,
    observed_current_a=observed, scoring_mask=mask,
    check_budget=check_budget, collect_rhs=False,
)
```

`result` contains `summary`, `arrays` and the eligible `residual`. `load_training_phase` allows only SET4 positive regression, RESET4 and SET5. The caller passes each simulated terminal state to the next phase. In two-precision checks, each SET5 inherits its own precision's RESET4 terminal state.

Other methods: `validate_circuit(points, check_budget=...)`, `audit_native_legacy(result, check_budget=...)`, `signed_rate(state_u, solved_point, b)`, `smallfield_bound(sign, b)`, `load_set4_reference_arrays()`. Saved positive reference metadata are in `set4_reference_summary`.

## Provenance

The implementation refuses changed hashes for the preceding frozen kernel, final positive result, fit status and signed-extension readiness document. The delegated loader verifies all four frozen runtime-cache inputs. No frozen-stage files were edited.

| Input | SHA256 |
|---|---|
| preceding `forward_kernel.py` | `052971a4ee8e927769118a3bccac323c7d5ad9a81e08d0e5caf1fe7a38181302` |
| preceding `final_refined.json` | `5fa0b38f637d8a47b6681283f6e9b5c898f0662fbcfe4abd72004c2f0154ea8c` |
| preceding `final_refined.npz` | `9fd7f5274f84d1fb420c895736a38bee4fe2dc72b8b02f3b93ed444ff6df7f1a` |
| preceding `finite_family_fit.json` | `24aca3ee6bf65915814d1dceef8a425f50b5898b8298f3d657e1585628615c9e` |
| preceding `SIGNED_EXTENSION_READINESS.md` | `331580f2e20b6632fc10f814785f24bdbb262ffa80947912770d5abed7ed870d` |
| `native_observations.npz` | `0aedbe51f41b4f4355e1de4d26048c6790a9d454b8b7d56f51bd76a4671465a8` |
| `transport_g010_v0025.npz` | `00d6d9ea0b1dfe855181e7016afc6dd5e23d169f83f19cfcedfc9d3b35f9e4e2` |
| `fig2b_transistor_lookup.json` | `1c9ecb7073f74e3db9f1c4a041d17630c652b1a12e4f33ff7388aaf5f0b16f21` |
| `bookoutput2_reverse_transistor_lookup.json` | `28462c3dcb7e0184199635de72a4c5fb1832a9f6281f08900ede28198dfe1359` |

Final syntax-checked kernel hash after the reviewer-requested report-only validation strengthening: `30b3df1eeb9aa22e2f8d3e8827b2950b97a38e14a4700d829edcbae6b4c57d6e`. The strengthening adds finite signed outputs, material/transistor passivity and KVL≤1e−12 V to the static validation gate using already-solved points; it adds no roots. Initial reviewed hash: `0779f9acbeade6266eb29e038cff727439a8e34e85032288b468eeaaa61efc31`. A subsequent source change must receive a new reviewed hash before numerical execution. Static preparation and independent review do not establish physical validity or numerical convergence; the parent's declared experiment performs those checks.
