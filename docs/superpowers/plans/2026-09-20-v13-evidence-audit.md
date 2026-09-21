# v0.13 Evidence Audit Implementation Plan

**Goal:** Enforce the calibrated SET/RESET handoff domain and score all measured cycles without altering the v0.12 numerical laws.

**Architecture:** A small M13 wrapper audits M12 outputs and exposes separate state coordinates. A separate validation module reuses M12 calibration/metrics and fits target-excluding bundles for all raw pairs; old reports remain intact.

**Tech Stack:** Python, NumPy, SciPy, pandas, unittest; matplotlib for research figures.

## 1. Handoff API (primary agent)

- [x] Add public-API tests in `tests/test_model_v13.py`: partial SET must raise, central outputs match explicit M12, no composite physical state.
- [x] Run `.venv/bin/python -m unittest discover -s tests -p test_model_v13.py -v` against old API, retain failing log.
- [x] Add `vulcan2d/model_v13.py` with an explicit full-cycle signature, a reference-current handoff audit, and separate state semantics. Keep M12 equations unchanged. Update package exports/version.
- [x] Check finite/ordered envelopes and near-boundary tolerance using synthetic states; test both RESET modes and two voltage grids.

## 2. All-cycle validation (validation agent)

- [x] Add `vulcan2d/validate_v13.py` and `tests/test_validate_v13.py` only.
- [x] Separate raw scoring records from accepted training records. Split by ID, not list position.
- [x] Refit each target's 52 SET and 50/51 RESET records; simulate with target voltages only. Score 51/53/2 groups and fixed chronological split.
- [x] Record training IDs, invalid predictions, model handoff diagnostics, retrospective first35-reference RESET identification, JSON/CSV outputs and provenance.
- [x] Run short tests for exclusion, grouping, invalid scoring and independence from target currents.

## 3. Evidence and outputs

- [x] Evidence agent writes `analysis/V12_PHYSICS_EVIDENCE_2026-09-20.md`, distinguishing current equations from historical hypotheses.
- [x] Run all Python tests and the full M13 audit. Save logs outside historical outputs.
- [x] Read resulting JSON/CSV, compare legacy metrics with v0.12, generate scientific plots from saved results and inspect them.
- [x] Write `vulcan2d/README_v0.13.md` and update project handoff with actual findings, remaining limitations and next priority.
- [x] Review diff and validate affected build checks before reporting. No remote publish in this task.
