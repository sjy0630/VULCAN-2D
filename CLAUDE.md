# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current scope and status (2026-09-21; supersedes historical notes below)

USER STEERING OVERRIDES ALL UNEXECUTED MODEL PLANS BELOW: the user explicitly
said to stop autonomous mechanism ideation and find the most useful next route.
Read `analysis/RESEARCH_DIRECTION_RESET_2026-09-21.md` first. No further automatic
model expansion, parameter scans or implementation of the configuration-dependent
capture design. The current route is a concrete advisor/experimental discussion
brief, measurement/protocol provenance and independent reproducibility, then a
minimal discriminating comparison if feasible. Do not send external messages.
`analysis/MECHANISM_DISCUSSION_BRIEF_2026-09-21.md` is ready for the user.
The full research goal remains active and unachieved; this changes the approach,
not the completion standard. Additional experiments are not a prerequisite for
all useful work; existing records and targeted literature can supply constraints.

Latest numerical stage: `analysis/RETAINED_TRANSPORT_RESULTS_2026-09-21.md`.
Five inherited transport priors were released with B parameterization: eight
starts ended, selected all53 RMSE=0.364969, chronological training=0.382259,
last18=0.279460. A worse-training fast-start basin gives last18=0.263371 but
must not be reselected using held-out outcomes. Key jump still missing. Four
results pass refinement (max 0.000862 decade); 480 independent load roots and
13 related tests pass. All eight optimizer flags report success; a nonselected
xtol result has optimality 0.052, not proof of a stationary/global optimum.
No-refit ellm=6 at original chronological parameters gives refined cycle3 SET
1.03226 uA / RESET 1.66357 nA, but predicts no cycle4 recovery. All-data source
instead suppresses both directions. Eighteen rate profiles include six coarse
refinement failures; ellm6 separately passes two successive path refinements.
Extreme ellm9/12 checks remain incomplete: at ellm12, 2,272,640 substeps yield
probability-sum drift 2.133e-11 versus original 2e-11 gate (KCL still 2.20e-14).
Do not claim every profile is numerically certified. Logs and failures preserved.
No runs remain active; future model designs are parked under the user's steering.

The user's current goal is a physically constrained quasi-static I–V model
and evidence-based explanation of Zhu's Nature h-BN 1T1M device. The current
research API is `model_v13.py`, a domain-audited wrapper around the unchanged
v0.12 equations. Read `vulcan2d/README_v0.13.md` and
`analysis/V12_PHYSICS_EVIDENCE_2026-09-20.md` before changing physical claims.
The desktop server still imports the earlier model. Its patch visualization,
and the original proposal's QPC/thermal/pulse architecture below, do not
describe the current research execution path or add requirements to this task.
SET and RESET use separate envelope coordinates; partial-SET handoff has not
been calibrated. Do not infer unique ion species from an I–V fit.
The user explicitly prioritizes explaining the electrical curves with physical
evidence as the proposed research contribution. Read
`analysis/PHYSICS_DISCRIMINATION_2026-09-20.md` for the original-paper evidence,
competing hypotheses, and parameter checks. The original Nature paper already
discusses soft degradation, interface ionic exchange, and residual nanogaps;
these concepts alone are not this project's novelty. Model-validation scores
and domain guards do not establish a physical mechanism.
Latest completed evidence stage: `analysis/CONTACT_MEMORY_RESULTS_2026-09-21.md`.
The zero-bias contact-exchange counterfactual is now implemented and checked.
New sibling `retained_contact_memory.py` carries the entire P(q,m;E) without
reset, using the unchanged archived native propagator. Two matched-training
parameter sets x fixed +/-g and both swap directions give eight complete
53-pair trajectories. Fixed positive reproduces archived arrays exactly;
same-contact segmented and unsplit trajectories are bitwise identical.
The swap-after-pair3 hypothesis FAILS selective change at these parameters:
all-data -g to +g gives S4/S3=46.61 and R4/R3=42.39; chronological gives
349.50 and 215.23. Measured values are 1.00955 and 259.82. Their contrast
C=(R4/R3)/(S4/S3) is 257.36 measured, versus 0.909/0.616 for the swaps.
Do not promote the large RESET jump while ignoring the failed SET magnitude.
No parameters were refitted; this does NOT exclude the entire contact family.
Boundary is prescribed, not a predicted event time or stochastic likelihood.
15 tests, 16 full-trajectory refinements, 768 independent frozen load roots
pass; maximum current refinement difference 0.000358 decade (rounded upward).
Compensated A is never clipped; no invented seconds or atomic identity.
Next prioritize sensitivity/identifiability of the five remaining inherited
transport priors within the continuous fixed-contact benchmark before adding
another state. All53 windows, first3 absolute currents, selectivity, later drift,
same-split training and actual last18 prediction remain joint constraints.
Original goal remains active/unachieved; M13 and ten prior stage manifests
unchanged. The following stage descriptions are historical.
Previous completed evidence stage: `analysis/STOCHASTIC_READOUT_RESULTS_2026-09-21.md`.
A finite-domain frozen current-support module now tests instantaneous readout
feasibility. It DOES NOT provide a stochastic path, likelihood or prediction.
Using each observed current to choose the closest state is an oracle lower
bound at fixed parameters, not a validated trajectory. After refitting six
readout parameters, a single four-state domain gives 0.28643 decade; four
identical domains (35 configurations) give 0.17516, first3 0.21915. Restricting
parameters to first35 gives last18 SUPPORT error 0.18074, not prediction error.
The last genuine continuous chronological prediction is still the retained
charge/contact model's 0.27337 from the preceding stage. Do not conflate tasks.
Readouts sum domain currents at shared internal voltage before transistor
loading. Delta/U interchange permutes the four conditional spectra; their
physical labels cannot be inferred from current alone. More domain counts
expand the support by construction and do not identify the true channel count.
34 starts all converge; pure and finite-domain readouts refine with maximum
configuration-current change 0.000297 decade. 672 independently enumerated
pure/mixed load roots and 35 related tests pass. Initial overly tight ohmic
rounding tolerance failure is preserved; physical refinement limits unchanged.
Nine previous stage manifests remain unchanged.
New experimental ORDER evidence: whole-cycle vector permutation test jointly
searches SET-return, RESET-outward, log ratio and all 52 possible cuts. Exact
p=1/11713=8.5375e-5 under cycle-vector exchangeability. An extremal-subset
certificate checks all 312 directed subset-sum bounds; only the lowest three
RESET feature values in the first or last three positions reach the threshold.
200k/100k independent Monte Carlo checks agree within sampling precision.
The same exact p holds after declared positive zero-reference shifts. This
is exploratory, not externally blind/pre-registered, and does not identify
microscopic species or exclude acquisition-condition changes. It rejects an
exchangeable baseline, not correlated stochastic dynamics.
Next physical candidate: `analysis/CONTACT_ASYMMETRY_MEMORY_DESIGN_2026-09-21.md`.
Retained contact configuration c may exchange effective GL/GR while preserving
B=A*expit(gl), changing conditional n occupancy and hence q-m dynamics without
an instantaneous fixed-state current change. This is an UNIMPLEMENTED
hypothesis. First test a clearly labeled switch-at-zero counterfactual after
pair3 in both contact-order directions, continuously carrying the full P(q,m)
state. A hand-specified boundary is not a predicted event or a path probability.
If useful, infer a stochastic contact transition with an energy-constrained
rate instead of hard-coding cycle4. Preserve ordinary loops and later history;
no per-RESET initialization, no measured target-current drive, no invented
seconds or atomic species. Full original goal active and unachieved.
Previous completed model stage: `analysis/RETAINED_CHARGE_RESULTS_2026-09-21.md`.
Only fast conductor n is eliminated; joint side charge m and configuration q
are propagated continuously through all 53 pairs / 35,828 native points.
34,662 scored points, eight equal-mass windows and 787 low-signal flags remain.
20 original starts plus four same-split nested starts, two widened-energy
continuations and six contact-ratio starts give 32 starts; one nonselected
original start capped, all selected solutions converged. Primary fixed-contact
all-data RMSE is 0.379892 and chronological last18 prediction 0.299548.
Widening energy bounds gives 0.376893 all-data. Refitting contact asymmetry
(with the other five electronic priors still fixed) improves all-data to
0.370029 and chronological last18 to 0.273370. The flexible training-only
phase-template reference is 0.241405. Full physical explanation is unresolved.
Native S4/S3 return is 1.00955, R4/R3 outward 259.82. Selected refitted-contact
predictions give ~0.992 for both in all-data and ~1.051 in chronological fits;
first-three-cycle errors remain 1.06177 / 0.96133 decade respectively. Some
solutions retain charge across cycles, but the large transition is missing.
Both selected contact ratios reach ln(GL/GR)=+12. Fixed-readout compensation
extended to +18/+24 changes currents by at most 0.007184 decade without a
refit; it does not identify the contact ratio or reject the entire family.
Exact conditional-readout identity: A*expit(gl) controls frozen current, so
compensating A while changing gl preserves every fixed-state I-V but changes
n occupation and slow rates. The old equilibrium-model gl is a conditional
prior, not an independent measured input for this new time-scale ordering.
39 related tests pass. Eight selected results refine to 192 nodes and
0.00125/0.0025 V table/path steps: max change 0.0003554 decade, weighted
<0.000085; 960 independent frozen load-root checks pass. Eight old manifests
are intact; new manifest covers 71 artifacts, 28 logs and two archived sources.
The original contact-test effect-size assertion failure and exact initial
source are preserved; final tests check the analytic occupation identity.
Additional conditional result: `analysis/REPEATED_TWO_STATE_MEAN_BOUND_2026-09-21.md`.
Independent two-state channels under repeated fixed rate protocols with
nonnegative fixed phase readouts obey I_(k+1)<=k/(k-1)*I_k in the ENSEMBLE MEAN.
Thus I4/I3<=1.5; all 11 native RESET points violate it even under favorable
zero-reference shifts (geometric ratio 223.63). This is NOT a sample-path
bound: a single random channel can jump in cycle4. Collective load feedback,
additional hidden states and nonstationary protocols are outside its scope.
Do not infer a necessary extra state count or exclude stochastic mechanisms.
Next: test full stochastic path probabilities and readout support, constrained
by ordinary loops as well as initial suppressed RESETs and the observed jump.
Do not fit only one event, copy the old failed fixed-envelope transfer, or
force a probability mean to jump at a specified cycle. Inherited transport
priors, finite-rate terminal terms, late drift and cross-gate validity remain
unresolved. The retained q-m implementation is now complete; the former
unimplemented plan is historical. Its rapid-m limit matches the old small
side/right-rate limit, not the fixed ratio1 candidate. No per-cycle resets,
no target-current inversion, no calibrated seconds or microscopic species.
Full goal active/unachieved; M13 unchanged and work unpushed. Charge/configuration
coupling is prior art (Ducry2022); new bounds' novelty remains unverified.
Earlier continuous-configuration stage: `analysis/ACTIVATED_CONFIGURATION_RESULTS_2026-09-21.md`.
Previous completed stage: `analysis/CHARGE_SPECTRUM_RESULTS_2026-09-21.md`.
Added a bounded uniform distribution of independent conducting/side-charge
channels, averaging currents with exact feedback and analytic gradients.
All 53 pairs / 5072 valid internal points / 16 missing positions retained.
Zero width matches the prior reduced kernel. Initial 64-point quadrature
failed the declared tolerance before any fit; increased to 96 points, retained
the failure log, and verified all fitted states through 192/384 points.
Original energy prior gives joint RMSE 0.13509; five-fold SET-conditioned RESET
1.02563 (first3 3.14754), chronological last18 0.73368. The center reaches the
0.5 eV bound, so wider E_d +/-1 eV and width up to 1 eV were tested separately.
Wider all-data joint RMSE is 0.13318, high-voltage slope ~0.712 vs observed
0.594 (narrow prior ~0.068). Wider full grouped sensitivity is 1.02366 overall,
3.14755 first3 and 0.82577 chronological. Thus shape improves but predictive
closure still fails. Wider folds use only corresponding training-state starts;
no all-data warm start. Primary 35 starts and 12 targets converge. Wider
all-data 4 starts and all 6 selected grouped fits/12 targets converge; 1/30
nonselected grouped starts reaches 400 evaluations. Keep this failure record.
Added `charge_moments.py`: exact two-mean piecewise constant-voltage propagator,
tested against independent four-state matrix exponentials (100 checks).
It is an occupancy propagator with dimensionless Gamma_R*t, not a calibrated
retention or terminal transient model; displacement/current circuit not added.
Proved a conditional slow-mode bound for fixed occupation/bias-independent
attempt rates: lambda_slow(V)/lambda_slow(0) <= 1/(1-sqrt(f(E_min))).
The selected spectrum solutions have E_min ~0.086 eV and an all-data bound
~1.23, so their interaction alone provides little zero-bias/biased slow-mode
separation. This is not a general bound on every switching observable, nor a
rejection of charge memory with structural barriers or different contact rates.
203 tests passed (45.003 s); 42 adaptive integrals, full grouped scores,
refinements, provenance and six prior manifests verified. Read the accompanying
literature note: Michalek/Bulka arXiv:0907.2879 already studies dynamical Coulomb
blockade/metastable polarization with different, energy-dependent total rates.
Neither that concept nor our two analytic statements have verified novelty.
Next prioritize an explicit activated retained configuration and its energy
mapping/readout, driven by full SET/RESET history without target RESET resets.
Do not keep expanding static spectra as a substitute for history. All initial,
ordinary, recovery and drift constraints still apply. Goal active/unachieved;
M13 unchanged, no new native-terminal validation, work not committed/pushed.
Previous completed stage: `analysis/COUPLED_CHARGE_NETWORK_RESULTS_2026-09-21.md`.
Added a conserved four-state conductor/side-charge network with reciprocal
Coulomb shifts. All 53 pairs / 5072 valid internal-grid points / 16 missing
high-voltage positions retained. Seven shared fits, 28 starts and 12 target
identifications converged. Joint descriptive error is 0.14982 decade, but
five-fold SET-conditioned RESET error is 1.18099 (first3 2.93613), worse than
the previous equilibrium-WKB charge model at 0.81066 on the same test.
Chronological last18 error is 0.94266. These are internal-coordinate tests;
no new native-terminal circuit validation was run for this four-state model.
Exact binary first-moment closure eliminates side-exchange rate from steady
mean occupancy and current when total attempt rates are occupation-independent.
The new five-parameter `coupled_charge_reduced.py` matches all 19 saved state
groups within 3.56e-15 decade. Joint correlation and response times can still
depend on that rate; do not interpret fitted rate bounds as measured kinetics
or claim this holds for arbitrary occupation-dependent contact couplings.
Current-only fits saturate too early (2--3 V slope ~0.00136 versus 0.594).
Post-hoc slope penalties can recover 0.594, at larger original current RMSE
0.19145 and unresolved first3 low-bias errors. This does not prove a universal
model-class impossibility. Penalties reuse the same data and are descriptive.
Full-SET ridge controls, with nested training-only normalization/penalties,
do not robustly outperform simple references across five-fold, leave-one-pair
and chronological checks. All 295 outer records were recomputed; 25 target
perturbation checks passed. This is failure of those features/linear maps,
not a proof that every nonlinear structural-history model lacks information.
194 tests passed. Independent 160-digit master-equation checks (210 cases),
240-digit side-flux refinement, scores, provenance and five prior manifests
verified. Side-current subtraction residual is <=7.16e-15 relative to gross
exchange; failed initial numerical tolerance and witness logs are retained.
Next test bounded energy distributions or bias-dependent contact supply with
conserved feedback, while prioritizing an explicit full-SET/retained-history
state law. Do not keep adding stationary parameters just to reduce average
error. No microscopic species or literature novelty is established; the full
goal is active and unachieved, production M13 unchanged, work unpushed.
Previous completed stage: `analysis/SIDE_CHARGE_READOUT_RESULTS_2026-09-21.md`.
Implemented one-reservoir equilibrium side-charge readout with explicitly
averaged two-configuration WKB currents, and an equally parameterized frozen
mixture control. Both retain all 53 gate-1.1 pairs and the archived 5072-point
weighted internal grid. Charge improves descriptive error 0.32367 -> 0.15548,
but five-fold SET-conditioned RESET error is 0.81066 decade (frozen 0.74570).
Native applied-voltage forward circuits confirm 0.79109 versus 0.72885;
all 1166 predicted points have unique load roots after refinement, with
KCL <= 3.2e-14. Initial three remain ~2.56 decade at the port. This is not a
successful physical explanation or a general rejection of charge mechanisms.
At fixed shared physics, SET fits within +0.005 decade can give substantial
negative-current spread (median 0.418, maximum 1.375 decade), but choosing
the best target RESET within that tolerance still leaves first3 at 2.490.
The tolerance is not a confidence interval. The fitted charge model also
misses positive 2--3 V sublinearity: predicted slope ~1.031 versus measured
median 0.594, with all 53 observed slopes below one. Missing points are masked.
Post-hoc a=1 profiles and full grouped sensitivity do not repair prediction
(RESET 0.88813 decade). These do not replace primary folds. For symmetric
bare configurations only, abs(ln r) <= 2*abs(a)*v/V_T; arbitrary asymmetric
WKB configurations instead have the wider (1+2*abs(a))*v/V_T envelope.
Do not turn the noisy raw R2 passive-limit violation into a hard species claim.
14 selected primary fits converged, but 7/56 nonselected starts hit the
iteration cap; keep that failure accounting. 186 tests passed. Four prior
manifests, current provenance, grouping and scores verified. Next combine
positive supply/contact saturation with charge/structural readout in a single
conserved network, using full SET and prior history for the retained state.
Do not reinitialize each RESET from its observed current or merely add free
energy parameters to the same local window. Full-cycle/history, initial
anomalies, later drift and novelty remain unresolved; goal active, M13 intact.
Previous completed stage: `analysis/BARRIER_HEIGHT_AND_POLARITY_RESULTS_2026-09-21.md`.
The finite-temperature fixed-geometry barrier-height state was compared on
the identical 101 traces / 202 branches and five grouped voltage folds.
Height-state higher-voltage RMSE is 0.16665 decade versus width-state 0.20459
and PF-like 0.16591. This is constitutive voltage extrapolation after target
low-voltage state identification, not autonomous or chronological prediction.
All folds select 4 nm under the declared coarse profile and supply prior;
it is not measured geometry. Eleven descriptive local heights reach the lower
boundary. An independently derived frozen passive WKB polarity bound applies
to arbitrary static barrier asymmetry and common energy-independent positive
parallel supply: abs(ln(I_minus/I_plus)) <= v/V_T. Its assumptions exclude
changing occupation/structure, resonances and arbitrary energy-dependent
contacts. First three gate-1.1 SET-return/RESET-initial pairs violate it in
all eight processing sensitivities, including favorable zero-reference shifts.
This does not identify a charge species, prove slow relaxation, or establish
novelty. The conditional extra landscape bounds (52.36/72.80/54.32 meV)
are not measured trap energies. The verifier rechecks all 54 fits/refinements,
5 target refinements, scores, KCL, provenance and prior manifests; 178 tests
passed. Audit: `analysis/barrier_height_polarity_validation_audit.json`.
Next prioritize an explicit charge/structure readout and SET-to-RESET bridge
on all 53 confirmed gate-1.1 pairs. Average configuration currents, not the
current at an average barrier. A one-reservoir equilibrium side trap has no
stationary net side current; a two-reservoir trap requires its own conserved
transport current. Refer addition energies to the actual named reservoir:
the +/-v change can be 2*a*v, so do not assert a universal 2*v/V_T bound
by mixing reservoir and mean-potential references. A new candidate still
needs held-out initialization, ordinary cycles, initial anomalies and drift.
M13 remains unchanged; the full research goal is active and unachieved.
Previous completed stage: `analysis/BARRIER_STATE_RESULTS_2026-09-21.md`.
Added a finite-temperature Landauer-WKB barrier width state with forward
transistor coupling, including over-barrier thermal supply. All 101 gate-1.3
traces / 202 low-bias branches were retained. In five cycle-grouped folds,
target 0.08-0.16 V identifies one local state and 0.17-0.30 V is predicted:
power / PF-like / width-state RMSE = 0.1697 / 0.1659 / 0.2046 decade.
This is constitutive voltage extrapolation, not chronological prediction.
The nominal 0.5 eV, 2.22 mS solution is not uniquely identified: relaxing
the supply prior allows 1 eV, 3.58e5 S with comparable descriptive error;
the latter has no independently verified mode/area support. Do not call
conditional widths measured nanogaps. 171 tests, integration/grid refinement,
complete grouped coverage and provenance checks passed. Next compare a
fixed-width barrier-height/contact-potential state under the same folds,
then prioritize cross-polarity/full-history closure. M13 remains unchanged.
Prior stage: `analysis/FINITE_PATH_TRANSFER_RESULTS_2026-09-21.md`.
Frozen trace-23 outward transport/rates were transferred to all other 100
gate-1.3 traces, with 3 rate laws and N=1/2/4/8/16/32 equally weighted
contributions. N suppresses large jumps but does not fix ordinary return
states. Common-saddle N=32 gives a paired-event probability of 1.45%
(observed 2/100), yet median return/outward ratio 0.166 versus 0.0240 and
low-bias return 50.0 nA versus 1.60 nA. Post-hoc initial-current matching
does not close the gap. Even with favorable zero-reference shifts, 40
targets have points below the source's frozen floor and 23 above its
ceiling; rate/count changes alone cannot repair that support failure.
All 100 targets and transistor-capacity mismatches were retained. Numerical
refinement, independent matrix exponentials and 166 tests passed. Existing
executed kernels and M13 were preserved. Next constrain state/history,
transport variability and rates jointly with ordinary non-events, retained
return states and anomalies; do not optimize only trace 23 or event counts.
The complete research goal remains active; no microscopic species, required
state-variable count or general rejection of stochastic physics is established.
Earlier occupation discrimination is recorded in
`analysis/OCCUPATION_DISCRIMINATION_RESULTS_2026-09-20.md` and the active
checklist `analysis/RESEARCH_GOAL.md`. Stationary occupation kernels can improve
positive-return fits, but fixed-rate full-bias single-level models predict the
wrong cross-polarity ratio (roughly 8–31 versus observed median 0.53). Their
voltage and cross-gate extrapolation also remain inadequate. The original
partial-bias kernel has an unresolved local voltage partition; do not present
it as a self-consistent two-terminal model. Production M13 is unchanged.
The subsequent contact/localized-state network comparison is complete:
read `analysis/CONTACT_NETWORK_RESULTS_2026-09-20.md`. Finite-U rate networks
can describe the average branch shapes, but all-pair held-out errors remain
large. An exact common-grid bound rules out improving an amplitude-only,
shared-shape model below the observed floor (0.31545 decade for all 53 pairs).
Target-RESET-conditioned energy adjustments fail held positive low-voltage
points. The next task is low-bias-constrained shape-changing state models,
energy distributions or multiple paths, then full-scan/history identification.
The bound does not prove that every possible scalar state is insufficient.
Do not infer relaxation or ion identity from the single-level model failure.
The low-bias parallel-spectrum/state stage is now complete; see
`analysis/PARALLEL_PATH_RESULTS_2026-09-21.md`. Five candidates and all held-pair
and chronological checks show that improved descriptive state fits do not
improve RESET prediction. An exact conditional thermal rectification bound
rules out arbitrary positive parallel mixtures of independent fixed-rate
levels for the first three paired low-bias reads (including the tested upper
zero-reference perturbations). It does not rule out all interacting networks,
bias-dependent contacts or changing states; its literature novelty is unaudited.
Direct complete-scan contrasts also show little early reverse hysteresis,
despite large positive SET hysteresis. S3/S4 positive returns have a 1% mean
change while R3/R4 initial currents differ by about 260x. Do not call this a
measurement of relaxation or a universal fourth-cycle threshold. Next work
must test complete history/state handoff and polarity readout bottlenecks,
not keep adding fixed-rate independent level spectra. M13 remains unchanged.
The next full-RESET-history stage is complete; see
`analysis/RESET_HISTORY_RESULTS_2026-09-21.md`. Maximum-field, thermal-rate
and self-consistent power-loss histories have target-excluded return and
terminal-circuit checks. No unique winner; state thresholds trade off strongly
against nuisance transport curvature, and simple cumulative-SET maps did not
improve initialization. A cross-gate raw audit found robust reverse-loop
current ordering violations and five persistent outward drop-then-recovery
traces. Any universally one-way loss model with monotone state conductance
and a fixed passive load line cannot describe these. Gate 1.3 trace23 first
drops near applied -0.39 V, then recovers near -1.09 V; do not describe it as
starting HRS throughout. These observations do not identify ions or heating.
Next research must permit competing build/remove transitions with explicit
energy/retention and scan-history constraints, or test a competing contact
readability process; do not keep only refitting one-way decay laws. Gate-group
device identity/order and scan/gap timing were asked asynchronously and remain
unconfirmed. Test totals and source checks are in the latest numerical audit.
A finite-step passive-load bound now tests whether rises can be static nonlinearity: gate 1.3 trace23 has a 1.392 µA no-enhancement upper bound versus 61.158 µA measured, at least 27.61x above it under the specified sensitivity checks. All seven detected rises were tested; five are robust violations, while both gate 0.9 trace6 rises are not. This does not distinguish structural, contact, charge or measurement changes. See `analysis/BIDIRECTIONAL_STATE_DESIGN_2026-09-21.md` for the unimplemented next-stage design. Latest full suite: 146 tests. Original field terminal output retains 69 iteration-cap flags; the separately archived 240-iteration verification converges and changes currents by at most 3.93e-8 decade.
The subsequent bidirectional stage is complete as a research comparison; read
`analysis/BIDIRECTIONAL_STATE_RESULTS_2026-09-21.md`. Two energy-related
continuous-state candidates were fitted separately to 11 complete negative
traces (22 fits, 88 converged starts); both miss the sharp trace23 path despite
describing ordinary 1.1 V examples. This does not rule out microscopic rates:
a single stochastic trajectory is different from a population-mean trajectory.
An outward-only two-configuration Markov probe on trace23 gives held-return
all-high probabilities 0.152/0.297/0.139 for constant/field/common-saddle rates,
but the held high-current return envelope still has 0.328-decade error. Only
two observed jumps inform kinetics; no unique mechanism or channel count is
identified. Its same-state zero-bias retention counterfactual requires very
long scans and is not a direct comparison to Nature's +0.1 V retention states.
Current suite: 158 tests; source/numerical audit is
`analysis/bidirectional_validation_audit.json`. The small C++ load-line helper
is compiled on demand into `.venv/.cache/bidirectional`, with independent
Python root and ODE checks. Original M13 is unchanged. Next use ordinary
non-event intervals and anomalous events jointly, finite-path variability and
contact/charge readout competition, then actual SET/RESET history handoff.
Ranjan2018 fulltext already reports RTN/threshold behavior in a different
low-power h-BN/Cu CAFM system. Pan2017 and Pazos2023 publisher abstracts also
contain related behavior; Pazos fulltext remains unavailable (403/bot-check).
Do not claim stochastic jumps, ion exchange or an electrical comparison
inequality is itself a novel discovery without further literature audit.
The repository now has Git history and tracked electrical workbooks. The
local `.venv` supports the numerical audit; the historical d2l environment's
NumPy is too old for `np.trapezoid`. Keep the frontend build green when changed.

## Status: integrated simulator running (Python engine + TS UI)

VULCAN-2D is a Fudan University undergraduate research project (FDUROP 曦源项目). A
**preliminary complete simulator runs**: the validated Python model (`vulcan2d/`) serves
I–V loops over HTTP to the TypeScript/Three.js UI, which shows a **model-vs-measured I–V
overlay** + validation readout and a **non-filamentary 3D device** (areal soft-breakdown
patches lighting up with φ̄). See "App: stack & how to run".

Division of labor (important): **the student team does the physics-model research (in
Python, `vulcan2d/`); Claude writes and maintains all the UI/integration code.** Optimize
technical choices for performance and visual quality, not for student-maintainability.

PHYSICS NOTE — the device is **non-filamentary** (h-BN soft-breakdown, K areal patches in
series with the 1T transistor; faithful to Zhu/Lanza *Nature* 2023). Do **not** depict a
single conductive filament — that was an early wrong picture, now corrected.

The chosen integration is "Python engine first" (UI calls a local server); a later step can
port the model into TypeScript for a single double-click desktop binary. What remains:
expose the new `p_perc` percolation knob / endurance hooks in the UI if wanted; package the
engine as a sidecar (PyInstaller) for a one-click desktop build; optionally port to TS.

Current contents of the workspace:

- `1-2.【立项】曦源项目申请书（2026年版）v3(1).docx` — the project proposal / 开题报告. This is
  the authoritative spec for what VULCAN-2D should be. Re-read it (extract text from the
  zipped `word/document.xml`) when scope or physics details are unclear.
- `1T1M写入.xlsx`, `1T1M擦除.xlsx` — experimental I–V data for model calibration/validation
  (see "Experimental data" below).
- `比较2.opju` — an OriginLab project file (binary) used for plotting the measured data.
- `微信图片_*.jpg` — a reference image (h-BN memristor 1R vs. 1T1M I–V comparison).

This workspace **is a git repository** (`main` branch, no commits yet). A `.gitignore`
**excludes the unpublished experimental data (`*.xlsx`, `*.opju`) and the proposal
(`*.docx`, which carries personal info)** from version control — these stay local. If the
repo is kept *private* and you want the data versioned, override those lines (consider
git-lfs for the large `.xlsx`/`.opju`). Code, docs, and `CLAUDE.md` are tracked normally.

## Who's who (advisor)

The **actual advisor is 朱凯晨 (Zhu Kaichen)** — Young Researcher at Fudan's School of
Future Information / Institute of Optoelectronics (`zhukaichen@fudan.edu.cn`). His group's
2D-memristor / memristive-CMOS integration work is the experimental basis of this project.

Note on the proposal text: 王水源 (Wang Shuiyuan) appears as advisor in an *earlier* draft
(v4), but that was only a nominal/placeholder listing to file the application. The current
docx (v3) and the real supervision are **朱凯晨's**. Treat 朱凯晨 as the advisor in any
context, attribution, or correspondence.

## App: stack & how to run

Two processes (dev): a **Python physics engine** + a **TypeScript/Three.js UI**.

- **Engine** — `vulcan2d/serve.py` wraps the validated model in a stdlib HTTP server
  (no extra Python deps). `GET /simulate?icc_ua=&K=&sigma=&ncycles=&seed=&vset=` runs an
  N-cycle MC and returns model SET/RESET median loops, the **measured** loops (from the
  xlsx, for the overlay), the φ̄(V) trajectory, and features vs data targets.
- **UI** — TypeScript + Three.js (WebGL2) + Vite, Electron for desktop. Chosen because the
  dev machine is **Apple M4 / Metal**: WebGL runs native-class through Metal, while
  OpenGL-based stacks (Julia GLMakie / Python VTK) are the weakest 3D path on this hardware.

```
npm install
npm run engine     # Python model server → http://127.0.0.1:8000
                   #   (override the interpreter: VULCAN_PY=/path/to/python npm run engine;
                   #    needs numpy/scipy/pandas/openpyxl — the miniforge `d2l` env has them)
npm run dev        # Vite UI → http://localhost:5173   (shows "engine ✓" when connected)
npm run build      # tsc --noEmit + vite build → dist/  (must stay green)
npm run electron   # desktop window (point at a running engine)
```

Layout:
- `vulcan2d/` — **the validated Python model** (the team's research lands here): `model.py`
  (non-filamentary soft-breakdown + load-line divider + kinetics), `features.py`,
  `calibrate.py`, `validate.py`, `serve.py` (engine), `vulcan2d_calibrated.npz`. Verify with
  `$VULCAN_PY -m vulcan2d.validate`. The UI must NOT re-implement physics — it calls this.
- `analysis/` — data exploration, feature CSVs, figures (incl. `figures/09_*` validation).
- `src/engine.ts` — typed client for the engine (`simulate(params)`).
- `src/viz/ivplot.ts` — Canvas-2D model-vs-measured I–V overlay (clean, legible — NOT a
  cloud of curves); `src/viz/device3d.ts` — the non-filamentary 3D (areal patches,
  Three.js). Glow uses unlit basic materials + additive sprites, **no HDR bloom**
  (UnrealBloomPass produced a magenta artifact under software-GL; verify visuals on the
  real Metal GPU, e.g. headed Chrome `channel:'chrome'`, not headless SwiftShader).
- `src/main.ts` — fetch-from-engine wiring + render loop. `electron/main.cjs` — desktop shell.

The old TS phenomenological physics (`src/physics/`, QPC/Poole-Frenkel filament) was
**deleted** — superseded by the Python model. Do not reintroduce a second physics.

## What VULCAN-2D is

VULCAN = **V**ariability-aware **U**nified simulator for **L**ayered-material **C**onduction
**AN**alysis. It is a device-level **physical simulator for 2D-material memristors** (RRAM
built from layered materials such as h-BN, graphene, TMDs), filling a gap left by existing
physical simulators (e.g. SIM²RRAM) that target HfO₂-class oxide RRAM.

Goal: reproduce, quantitatively, a device's **I–V characteristics** and **set/reset
switching behavior**, including **device-to-device and cycle-to-cycle variability**, then
calibrate/validate against the advisor group's real 2D-memristor measurements to form an
"experiment ↔ simulation" closed loop.

**SIM²RRAM is the architectural reference model** — the project's plan is to mirror its
solver structure (coupled kinetic / thermal / transport equations) and then swap in
physics specific to 2D layered materials. When in doubt about overall structure, consult
the SIM²RRAM paper (Villena et al., *J. Comput. Electron.* 2017, ref [1] in the proposal).

## Planned simulator architecture (the physics modules to build)

The proposal specifies four coupled physical models. Treat these as the module breakdown:

1. **Conductive filament formation / rupture** — filament geometry + defect (vacancy)
   dynamics governing the set/reset transitions.
2. **Charge transport** — regime-dependent conduction:
   - Low-resistance state (LRS): **Quantum Point Contact (QPC)** model (quantized
     conductance through the filament).
   - High-resistance state (HRS): **Poole–Frenkel** (and similar trap-assisted) emission.
3. **Joule heating / temperature field** — local temperature drives the thermally
   activated reset; electro-thermal coupling.
4. **Device variability** — statistical modeling of parameter spread so simulated I–V and
   set/reset distributions match measured variability. The advisor group's
   two-dimensional variability-coefficient method (ref [4]) is the intended analysis lens.

These are **multiphysics-coupled** (electrical ↔ thermal ↔ defect). Numerical stability
and convergence of the coupled solve is called out in the proposal as a primary
difficulty — expect this to need care, not a naive fixed-point loop.

**Implementation language:** the proposal names **Python or MATLAB**. No choice is locked
in; if starting the codebase, confirm with the user which to use before scaffolding.

## Experimental data (calibration / validation targets)

These are real measurements from advisor **朱凯晨's group's work on hybrid 2D/CMOS
memristive microchips** (cf. K. Zhu et al., "Hybrid 2D/CMOS microchips for memristive
applications," *Nature* 618, 57–62, 2023). The 1T1M structure = an h-BN memristor in series
with a CMOS transistor that provides compliance/current-limiting (the red, narrowed I–V in
the reference image; the bare 1R memristor is the wide, variable blue I–V). Notably the
corresponding author and co-authors of that Nature paper (Lanza, Villena, Roldán) are also
the authors of SIM²RRAM — the simulator this project takes as its architectural reference.

`1T1M写入.xlsx` (write) and `1T1M擦除.xlsx` (erase) hold measured I–V curves from a 1T1M
(one-transistor–one-memristor) cell:

- Each file has **53 worksheets** (one per cycle/measurement run).
- Each sheet has columns headed `voltage` and `current` (write sheets carry a third
  column). Write sheets run ~500 rows; erase sheets ~170 rows.

These are the ground truth the simulator's I–V and set/reset output must reproduce.
Reading them programmatically: they are standard `.xlsx`, so `pandas.read_excel(path,
sheet_name=None)` (Python) loads all 53 sheets as a dict. `.opju` is OriginLab's binary
format and is not readable without Origin — use the `.xlsx` files as the data source for
any code.

## Working notes

- The project is bilingual: the proposal and filenames are Chinese; code and docs can be
  English. Preserve the Chinese filenames as-is (they are referenced elsewhere).
- The 12-month plan (from the proposal): months 1–2 literature/framework; 3–5 core physics
  modules (filament, QPC transport, thermal); 6–8 add variability + initial calibration;
  9–10 validation/optimization + variability analysis; 11–12 report + docs. Use this to
  judge what stage a request belongs to.
- Deliverable is "runnable simulator + source + usage docs" reproducing measured I–V,
  HRS/LRS, and set/reset behavior. Quantitative agreement with the `.xlsx` data is the bar.
