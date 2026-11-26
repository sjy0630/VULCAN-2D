# Bounded continuous shared-parameter calibration — 2026-11-04

Status at declaration: implementation/pre-execution review; no joint numerical run yet.

## Question and frozen scope

Can the same signed soft-only q² law improve continuous SET4 → RESET4 → SET5
when its common kinetic parameters and one whole-history initial state are
jointly calibrated? The previous frozen-parameter bounds do not exclude this
common-parameter family. This stage is a finite numerical probe, not proof of
a global optimum, full unified 1R/1T1R accuracy, or microscopic mechanism.

Keep the signed material spectrum, transistor tables, λ=0.4470398752554136/V,
Ron=58.463033350725695 Ω, Γc=0.03127031375421752 eV, q² recovery, a=0 and
command-voltage-distance clock fixed. Use the same equations for both signs:
q′=κq |v|[(|i|/J0) exp(σb)(1−q) − exp(−σb+λ|v|)q²],
κq=κ exp(b), J0=J+ exp(2b), q=exp(−u), Γ=Γc/expm1(u).
Unknown scan/dwell/wait times remain unknown. There is no zero-field waiting
evolution. No state clipping, phase/cycle re-anchor, topology-conditioned
material law, solver fallback after failure, original holdout access, default
model change or push.

## Data and continuity

The existing original-training waveform cache contains SET4/RESET4/SET5/RESET5/SET6.
Calibration uses only the first three phases and their original masks:
500/170/500=1170 residuals. Equal phase weight means RESET points receive
approximately 500/170 times each SET point's weight. Minimize the mean of the
three phase MSEs, and separately report each RMSE and the point-weighted
1170-point RMSE. This is not the full 27023-residual/20-gate objective.

Only one initial u0 is allowed. Every subsequent phase takes the actual terminal
state and command from the preceding phase, including the historical control.
Old inferred u0 is an initial seed, not a measurement. SET5 is now calibration
data. Final RESET5 → SET6 (170/500=670 residuals) is a fixed-parameter training
continuation, not an original holdout or pristine prediction. Its errors cannot
affect ranking, optimization, or parameter changes. Do not publish a combined
1840-point objective. Retain every low-current scored point.

The broader task remains 46 pairs/92 phases/27023 residuals/20 aggregate gates.
Bare cycles are chronological and same label means SET then RESET. Missing
4/23 were measured, so known segments 1–3, 5–22, 24 cannot be joined across the
gaps. Bare training is 1–3 and 5–15; series training is 4–35. This limited
series-only run does not solve the bare mA branch/reversal constraints.

## Parameters and fixed seeds

Optimize x=(log10 κ, log10 r, b, u0), r=κ/J+, with every coordinate normalized
to [0,1]. Computational domains:

- κ: [77.70013824898767/10, 100×14648.824351938409] /V².
- r: [(77.70013824898767/0.0041089272077381635)/10,
  100×426403.0838962568] /(A V²).
- b: [0, ln 10]; u0: [2,16]. These are search ranges, not established physical bounds.

Control: κ=77.70013824898767, J+=0.0041089272077381635 A,
b=0.5811330029022057, u0=9.860489172652638.
Necessary-region seeds use K=14648.824351938409, R=426403.0838962568,
B=ln 10, U=control u0: (K,R,B,U), (4K,R,B,U), (K,4R,B,U), (K,R,B/2,U).
K and R are endpoint-derived necessary-region guidance; they are not the
search-domain lower boundaries or hard constraints replacing finite residuals.

## Execution, finite comparison and budgets

Require explicit --run-authorized and refuse overwriting an existing final result.
First run the actual three-phase control with DOP853 and Radau, each at
rtol=1e−8, atol=1e−9, max_step=.01 V. Require max native current difference
≤1e−4 decade and max u difference ≤1e−3, native KCL/KVL/passivity/domain/joins.
Compare historical phase RMSEs (.1338949019268602, .705457880221811,
.5243272706818787) within 1e−4; actual new continuous control is authoritative.
The old mean phase MSE .2635059174365875 and point RMSE .4443640469714171
are comparison references, not forcibly reproduced state inlets.

All search forward solves use Radau, rtol=2e−6, atol=2e−7, max_step=.04 V.
Baseline plus four seeds cost five objective requests. Run two fixed
Nelder–Mead starts, at most 40 objective requests each: best completed new
seed first, old-control start second. If no new seed completes, skip its
optimizer and retain the old start; this is not family rejection. Initial
normalized simplex moves +.04 per axis if within domain, otherwise −.04.
No extra restarts. Count/log cached, invalid, failed and outside-domain
requests toward at most 85. Failed or invalid results have no finite score;
do not disguise them as a finite penalty. Cache only exact duplicates.

Numerical compute budgets (serialization/hash/I/O excluded): setup 60 s,
new optimizer 120 s, old optimizer 120 s, fit total ≤300 s, each fit phase
≤12 s. A timeout stops its declared block with failures/partials retained;
no retry. The final independent validation has a separate 100 s budget,
each final phase ≤20 s, and total numerical compute ≤400 s.

## Final validation and near-zero policy

Select only by complete first-three-phase scores. Run the selected candidate
through all five phases at two fixed Radau precisions, with all RHS calls
saved: nominal (1e−7,1e−8,.02), refined (1e−8,1e−9,.01). Check original
masks/counts, u/q/Γ domains, signed passive ports, KCL≤1e−13 A,
KVL≤1e−12 V, exact state handoffs, independent RHS algebra and near-zero
physical-microsegment accounting. Independently re-solve every native
readout using 48 bisections in local hBN voltage and the original finite
signed spectrum/transistor tables; current difference≤5e−7 decade and
local voltage difference≤1e−9 V. Two-precision limits are 1e−4 decade/1e−3 u.
Search checks without RHS must explicitly identify their native-only scope.

Pre-execution design correction: explicitly supply and save
nearzero_segment_budget=1e−10 (finite positive, ≤1e−10). The existing vmin=1e−9 V
and local freezing rule are unchanged. The former 1e−12 gate would reject a
declared seed before integration; the new budget covers the entire declared
parameter domain (worst recovery contribution about 7.3244e−11) and is below
refined atol=1e−9. Keep all other validation gates. These are local omission
bounds, not full-trajectory error certificates. Frozen prior sources unchanged.

## Evidence and reporting

Persist every objective request, parameters, actual state handoffs, original
arrays/masks, completed and failed phases, accepted partial trajectories,
solver errors, timing scopes, final arrays and independent root arrays.
Hash exact bytes at write time in an artifact catalog; deep-copy saved
summaries so later auditing cannot mutate prior snapshots. Do not repeatedly
load historical trajectories. Report numerical failures as unresolved
numerical/domain cases unless a separate physical exclusion is established.
After execution obtain an independent saved-result/source review, plot only
saved results, visually inspect figures, write the result report and final
manifest after writers stop. Keep the full scientific goal active until the
shared bare/series history and original contract are actually met.
