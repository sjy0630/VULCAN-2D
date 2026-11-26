# Runner source timing note — 2026-11-04

The root agent reports that the one authorized run (session 10395) had already started when the implementation agent's start notification reached this child agent. After that start, but before receiving the notification, this child applied a source-review suggestion to `bounded_joint_calibration.py`: the top-level exception handler now explicitly sets `numerical_validation_passed=False` when an exception occurs. This edit was then reported to root. All three numerical source files are now frozen; this note does not assert that the runner source was completely frozen throughout startup.

## Exact versions

- Reviewed/startup runner, reconstructed here as `EXECUTED_RUNNER_SNAPSHOT.py`: SHA256 `4ffd87a98aca8550ccb04b62da1590159cc5746e7a731f8d8f2aefd14ef2b98a`.
- Current `bounded_joint_calibration.py`: SHA256 `5b99ed81c3f90f01f750774669967b03132a48c7f187c59d0239c1f488ec91cd`.

The snapshot was produced by reversing exactly the one exception-reporting edit in the current source. Its SHA256 exactly matches the reviewed startup version. No other source text was changed in reconstruction. The current runner, `joint_forward.py`, and `joint_validation.py` were not edited during snapshot preparation. The snapshot was neither imported nor executed.

## Scope and provenance limitation

The edit changes only a report flag in the top-level exceptional exit path, including a possible exception during the final provenance hash check. It changes no material equation, data, parameter, objective, optimization request, state handoff, numerical budget, solver setting, or normal-path validation. This does not justify claiming unchanged runner source during the run.

Input loading and source hashing may have occurred before or after this edit. Root must retain the hash actually recorded by the active run and reconcile it with the reviewed executable snapshot and the current file; neither an unchanged-hash result nor an input hash alone establishes that the source was unchanged across startup. Any discrepancy must remain explicit in the final report/manifest. Do not rerun automatically or overwrite the original numerical artifacts to remove this timing issue.

Only this reconstructed snapshot and this timing note were written in this recovery step. No model calculation, import of the runner, or automatic retry was performed.
