# VULCAN-2D

最新研究代码与复现范围见 [2026-09-21 同步说明](docs/RESEARCH_SYNC_2026-09-21.md)。
当前工作转向测量背景与独立可重复性核实；微观机制尚未闭合。

The research API uses the v0.12 quasi-static SET/RESET laws with the v0.13
handoff-domain audit. After calibration it receives the two voltage waveforms
and gate voltage, without target currents. Its SET and RESET states use
different transport envelopes; a partial SET endpoint is rejected because
that cross-polarity mapping has not been calibrated. See
[`vulcan2d/README_v0.13.md`](vulcan2d/README_v0.13.md) for the current API,
all-cycle validation and scientific limits. The archived
[`v0.12 report`](vulcan2d/README_v0.12.md) remains the numerical baseline.

**V**ariability-aware **U**nified simulator for **L**ayered-material **C**onduction **AN**alysis.

A reduced-order device simulator for **h-BN 1T1M**, based on Zhu et al.,
*Nature* 618, 57–62 (2023). The current research path uses measured transistor
load lines and polarity-specific h-BN transport/state laws. The interactive
3D UI still uses the earlier distributed-patch model and has not been migrated
to v0.13. The architecture and desktop instructions below describe that legacy
interface. Its patch visualization is not a microscopic validation of the
current research equations.

> Fudan University FDUROP 曦源项目. Advisor: 朱凯晨 (Zhu Kaichen). The model is calibrated
> against the group's hybrid 2D/CMOS h-BN 1T1M measurements (`1T1M写入/擦除.xlsx`).

## Architecture

Two processes during development:

```
┌─────────────────────────┐        HTTP /simulate        ┌──────────────────────────┐
│  Python engine           │ ───────────────────────────▶ │  TypeScript + Three.js UI │
│  vulcan2d/  (validated    │   model + measured I-V loops │  (Vite / Electron)        │
│  distributed-path model)  │ ◀───────────────────────────  │  3D device + I-V plot     │
└─────────────────────────┘                               └──────────────────────────┘
```

The calibrated model stays in Python (`vulcan2d/`, where the model research happens); the
UI fetches simulated + measured loops and renders them. (A later step can port the model
into TypeScript for a single double-click desktop binary.)

## Run

Two terminals:

```bash
# 1) physics engine  (needs a Python with numpy/scipy/pandas/openpyxl)
npm run engine            # → http://127.0.0.1:8000   (override interpreter with VULCAN_PY=…)

# 2) UI
npm run dev               # → http://localhost:5173
```

The UI shows "engine ✓" when connected. `npm run electron` packages the desktop window
(point it at a running engine).

## Desktop app (send to others)

Builds a double-click app with the Python engine bundled as a sidecar — recipients
need no Python, Node, or terminal.

```bash
npm run dist:mac     # → release/VULCAN-2D-<ver>-arm64.dmg   (Apple Silicon)
npm run dist:win     # → Windows installer (must be run ON Windows)
```

Pipeline: `scripts/build_engine.sh` (PyInstaller → a numpy-only engine binary, no raw
data inside) → `vite build` → electron-builder (bundles the engine as `extraResources`;
`electron/main.cjs` spawns it on launch and kills it on quit) → `scripts/afterPack.cjs`
ad-hoc code-signs the app (required for Apple Silicon).

Notes for recipients:
- The app is **ad-hoc signed but NOT Apple-notarized** (no paid Developer ID). Any copy
  received via download / WeChat / AirDrop gets a quarantine flag, so the first open shows
  *"Apple cannot verify…"*. To open it (once):
  - **Terminal:** `xattr -dr com.apple.quarantine /Applications/VULCAN-2D.app`, then double-click; **or**
  - **GUI (macOS 15/26):** try to open → click *Done* → **System Settings → Privacy & Security**
    → **Open Anyway** → confirm. (On macOS 15+, the old *right-click → Open* no longer bypasses this.)
  - To remove the prompt entirely, the app must be **Apple-notarized** (paid Developer ID) —
    wire signing+notarization into the build if the group has an account.
- Builds are per-platform: the macOS arm64 `.dmg` won't run on Intel Macs or Windows —
  build those on the respective OS.

## What the simulator does

- **I–V · model vs measured** — the model's SET / RESET median loops (amber) overlaid on
  the measured cell (grey), with a 10–90% variability band. The validation readout below
  shows V_set, V_reset, R_HRS, R_LRS, I_cc and the memory window, model / data.
- **3D device · soft-breakdown patches** — K coarse-grained local regions that light up
  progressively as φ̄ rises during SET. They are not a literal count or geometry of
  filaments; each can represent a defect bridge, metal-assisted path, or CAFM hotspot.
- **Live controls** — compliance I_cc, sub-populations K, variability σ, MC cycles,
  re-sample. Each re-queries the engine.

## The model (`vulcan2d/`)

See [vulcan2d/README_v0.3.md](vulcan2d/README_v0.3.md) for the equations and validation,
and [analysis/MODEL_REVIEW_2026-07-14.md](analysis/MODEL_REVIEW_2026-07-14.md) for the
reliability review. The paper-ready derivation and evidence grading are in
[analysis/LITERATURE_THEORY_REVIEW_2026-07-14.md](analysis/LITERATURE_THEORY_REVIEW_2026-07-14.md),
with citations in [references/vulcan2d_theory.bib](references/vulcan2d_theory.bib).
The advisor's proposed 1R/1T1R--CAFM--XTEM--atomistic paper logic is mapped in
[analysis/PAPER_EVIDENCE_CHAIN_2026-07-14.md](analysis/PAPER_EVIDENCE_CHAIN_2026-07-14.md).
The focused Zhu Kaichen/Mario Lanza h-BN literature audit, two-regime mechanism,
and testable predictions are in
[analysis/ZHU_LANZA_HBN_PHYSICS_REVIEW_2026-07-15.md](analysis/ZHU_LANZA_HBN_PHYSICS_REVIEW_2026-07-15.md).
The newly added sputtered h-BN CAFM current maps and 4,866-spot projected-area
distribution are analyzed in
[analysis/AFM_DATA_REVIEW_2026-07-14.md](analysis/AFM_DATA_REVIEW_2026-07-14.md),
with a reproducible parser and figure in `analysis/analyze_afm_spots.py`.
The v0.5 protocol correction, optional CAFM geometric prior, and 1.1 V gate
holdout are documented in [vulcan2d/README_v0.5.md](vulcan2d/README_v0.5.md)
and [analysis/V5_DATA_CONSTRAINED_PHYSICS_2026-07-26.md](analysis/V5_DATA_CONSTRAINED_PHYSICS_2026-07-26.md).
The measured reverse-transistor de-embedding and inverse RESET-state
identification are documented in [vulcan2d/README_v0.7.md](vulcan2d/README_v0.7.md),
with the machine-readable audit in
[analysis/v7_reset_identification_summary.json](analysis/v7_reset_identification_summary.json).
The read-initialized RESET prediction, endpoint-leakage removal and held-out
validation are documented in [vulcan2d/README_v0.8.md](vulcan2d/README_v0.8.md),
with the machine-readable audit in
[analysis/v8_read_initialized_reset_summary.json](analysis/v8_read_initialized_reset_summary.json).
The measured-SET-conditioned full-cycle hand-off and its leave-one-cycle-out
audit are documented in [vulcan2d/README_v0.9.md](vulcan2d/README_v0.9.md),
with machine-readable results in
[analysis/full_cycle_validation_summary.json](analysis/full_cycle_validation_summary.json).
The protocol-conditioned HRS endpoint, model-selection audit and improved
return branch are documented in
[vulcan2d/README_v0.10.md](vulcan2d/README_v0.10.md), with results in
[analysis/full_cycle_hrs_attractor_summary.json](analysis/full_cycle_hrs_attractor_summary.json).
The effective field-assisted HRS transport refinement and its target-excluding
and chronological audits are documented in
[vulcan2d/README_v0.11.md](vulcan2d/README_v0.11.md), with results in
[analysis/full_cycle_pf_refinement_summary.json](analysis/full_cycle_pf_refinement_summary.json).
The autonomous voltage-only SET-to-RESET loop, corrected SET de-embedding,
separate SET transport envelopes and strict target-excluding validation are
documented in [vulcan2d/README_v0.12.md](vulcan2d/README_v0.12.md), with results
in [analysis/full_cycle_autonomous_summary.json](analysis/full_cycle_autonomous_summary.json).
Reproduce the validation figure:

```bash
$VULCAN_PY -m vulcan2d.calibrate
$VULCAN_PY -m vulcan2d.validate
```

## Project layout

- `vulcan2d/` — the validated Python model + `serve.py` (engine HTTP server).
- `analysis/` — data exploration, feature extraction, figures.
- `src/` — the TypeScript UI: `engine.ts` (model client), `viz/ivplot.ts` (I–V overlay),
  `viz/device3d.ts` (distributed-path 3D), `main.ts`.
- See [CLAUDE.md](CLAUDE.md) for the full architecture.
