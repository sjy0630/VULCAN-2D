# Original training adapter

`training_problem.py` uses the unmodified coupled kernel at SHA256
`1288ae379fbefff679bebd3ad110a10cee344f970d8feb6a8dff7838b58a26d7`.
It implements the full original training histories and scorer; it does not
select initial states, fit parameters, reuse a periodic pair, or run on import.
Preparation included pure compilation and independent static review only.

## API

```python
import training_problem as tp
problem = tp.load_training_problem(check_budget=check_budget)
material = tp.kernel.load_material(parameters, candidate_source_id=source_id,
                                   candidate_source_sha256=source_sha256)
result = problem.evaluate(
    material,
    {'bare1': (q1,a1), 'bare5': (q5,a5), 'series4': (qt,at)},
    precision, method, check_budget, collect_rhs=False,
)
weighted_residual, score = problem.score_predictions(predictions)
scorer_check = tp.compare_legacy_scorer(problem, predictions=None)
```

`tp.kernel` is the same imported frozen module used for material/precision
class identity. Construct parameters/precision through this module, or the
same `coupled_forward` import. All evaluate arguments are explicit except
optional RHS trace collection. The adapter provides no default material,
initial state, method, precision, optimization, or execution loop.

## Data and history

The loader reads the frozen cache `native_observations.npz` and decodes only
184 explicitly named training voltage/current arrays. It verifies the data
file SHA256, original training-array digest, user-confirmed chronology,
six mask hashes/counts, and native zero endpoints. File hashing is not
decoding held-out observation arrays. Loaded arrays have immutable backing.

| Entry key | Topology | Consecutive records | Phases |
|---|---|---|---:|
| bare1 | 1r | 1–3 | 6 |
| bare5 | 1r | 5–15 | 22 |
| series4 | 1t1r | 4–35 | 64 |

There are exactly 46 pairs and 92 independently propagated phases. One
material object is passed to all three histories. Only those three entry
states are accepted; both q and a are inherited at every internal handoff.
bare5 is an unknown state after missing measured cycle 4, not a prediction
through it. No waveform for missing 4 or 23 is invented. Timing and waits
remain unspecified; the frozen kernel's command-voltage path clock applies.

Bare SET uses ideal 1 mA compliance with actual local-voltage solving; bare
RESET has no compliance. Series phases use the frozen measured 1.1 V gate
lookup with no additional instrument cap. Circuits change; material does not.

`problem.data[(topology, record_id, phase)]` exposes command, original signed
observation, original boolean score mask, initial command 0, record/phase,
topology, segment ID, outward mask and observed-limit mask. The corresponding
`waveforms`, `raw_training`, `log_observed`, `scoring_masks`, `groups`, and
`train_ids` are available for independent postprocessing. Group raw currents
are absolute values, matching the archived scorer exactly.

## Exact objective

Eligibility is determined only from observations: finite absolute current
at least 1e-11 A for SET or 1e-12 A for RESET, and |V|>1e-12 V. No measured
or predicted current is floored. Bare SET uses the original observed 0.995 mA
partition and native outward/return split.

| Original residual block, in vector order | Count | Division of raw log10 residual |
|---|---:|---|
| 1r SET, nonlimited outward | 730 | sqrt(730×3) |
| 1r SET, nonlimited return | 27 | sqrt(27×3) |
| 1r SET, limited | 2027 | sqrt(2027×3) |
| 1r RESET | 2799 | sqrt(2799) |
| 1t1r SET | 16000 | sqrt(16000) |
| 1t1r RESET | 5440 | sqrt(5440) |

The 27023-entry vector follows each block's original record/native-index
order. `verified_cost=0.5*dot(residual,residual)`, exactly as archived, with
no additional division by four. Each phase group has equal weight; bare SET
has three equal subgroup weights. The descriptive balanced four-phase RMSE
is separately `sqrt(dot(residual,residual)/4)`. Plain all-point RMSE is also
reported. Observed record scatter is never subtracted from the objective.

`score_predictions` accepts exactly the 92 original keys, with values as
coupled phase results, ordinary `current_a` traces, or native current arrays.
When command arrays are supplied, their identity with that record is checked.
No model call is made. Scores include phase/subgroup metrics and per-record
SET, RESET, combined, outward and return RMSE.

`compare_legacy_scorer` explicitly extracts only the frozen
`OrderedJointProblem.score_predictions` function using AST and invokes it
on the same groups/arrays. It does not import the old package initializer or
override production scoring. With predictions=None it creates one fixed,
record/node-varying arithmetic fixture from already loaded training arrays.
It checks residual bit equality and exact group details. It is not a fit or
simulation. This function was prepared but not executed by the author.

## Return and failure contract

Successful `evaluate` returns:

```text
summary      status='complete_original_training', metadata and exact counts
predictions  {(topology, record_id, phase): kernel_phase_result}
segments     [{segment_id, topology, record_ids, initial_state_q/a_s,
               final_state_q/a_s, status, result: kernel_history_result}]
residual     original weighted vector, length 27023
raw_residual same block order, without group normalization
score        JSON-compatible cost, four phase blocks and per-record metrics
```

Kernel phase results retain native states/currents, port quantities, optional
RHS traces, and solver details. Segment results retain all q/a joins.
`checkpoint_metadata()` gives provenance, exact decoded keys, mask hashes,
block offsets, normalizations and the three-entry history contract.

An evaluation failure raises `TrainingFailure`; its `.result` preserves
completed segments/phases and the failed history/phase where available.
Summary status is `incomplete_training_failure_retained`. No complete-score
claim is made: score remains None and full-objective residual vectors empty.
The caller must save this result instead of replacing it with a penalty or
discarding failed phases. Invalid input/state setup can fail before any
integration. The adapter never retries or changes parameters/initial states.

## Original feature gates

The separately prepared `feature_audit.py` supplies
`evaluate_features(problem, result['predictions'])` for the fixed original
20 aggregate gates. Run it as postprocessing on complete predictions; feature
failure does not change masks or hide the original objective. There is no
duplicate feature implementation in this adapter. Successful training-score
plumbing alone would not establish numerical convergence, an adequate fit,
held-out prediction, or microscopic mechanism evidence.
