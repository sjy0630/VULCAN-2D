# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

For model changes and experiments, follow the [model iteration protocol](docs/model-exploration/PROTOCOL.md) and [project rules](AGENTS.md). Archive the result before reporting it, including failed comparisons.

## Latest common-lambda feasibility (2026-11-04; no admissible forward candidate)

Read `analysis/COMMON_LAMBDA_FEASIBILITY_RESULTS_2026-11-04.md` and
`analysis/common_lambda_feasibility_2026-11-04/` first. Previous turn was progress;
this turn also executed a new diagnostic, not just planning. Unique session98817
exit0, full4 configurations, frozen5 sources unchanged, numerical.176828875s.
Runner SHA041f7e5cbcd73694102a98e3b1ea79609208eced5d21b341099d561fb1f4a00e.

Actual RESET positive-rate lambda intervals: linear16(3.273462176,3.376316130),
PCHIP16(3.274232225,3.376925074)/V. Four shared-product H roots near3.28409–
3.28488/V all require b≈-.175737 to-.173926, outside declared[0,ln10].
All4 eligible_for_continuous_forward_check=false. Even retaining negative b
diagnostically, linear16 predicts RESET high/low return Delta_u -4.233433/
-.569561 vs reconstructed+2.087996/+2.228142. SET5 low-return -.285321 vs+.381373.
Other8 windows max normalized integral closure≈.99316–.99318, NOT current
percentage/RMSE. b-domain is a model assumption, not measured uncertainty;
return-direction failure is an additional reason not to accept these roots.

All work:8 R boundary+4 H parameter roots,383+172 bisection iterations,
399 R functions,296 H points including124 grid points,592 2x2 identifications,
2014 saved-window sums/846160 node incidences,4 candidates*12 closure pairs.
No new circuit/material roots, state inversions, ODE or nonlinear forward fit.
Shared b projected as(bF+bG)/2, negative coefficients recomputed from same tuple;
max relative projection difference6.47e-14. Sources frozen BEFORE sole run.
Saved-result review9320 checks pass,9 inputs/144 NPZ members, no new trial points.

Independent finite-sum H monotonicity theorem/check: P=SET outward,N=RESET;
H' upper bound Cplus-Cminus<-28.34V over entire RESET positive-coefficient
interval, with H endpoint limits+infinity/-infinity. Therefore exactly one
product root per reconstructed configuration;180 saved arithmetic checks pass.
This is conditional finite positive sums, not an experimental trajectory/noise
certificate. T>S and small RESET Delta_u=.047941 remain uncertainty-sensitive.
Do not claim all unified physical models are excluded. No accuracy update;
prior3-phase candidate RMSE.405892 remains the last actual forward score.

Prepared lambda_forward.py SHAde28520a24f36cda66e5923548af73183ef74f33ed98ae0682512d5b231a980c:
explicit immutable readout and lambda/kappa/J/b snapshots; parent joint_forward
unchanged. Only compile/body-diff/static review done, not imported/run. Because
no eligible lambda candidate, NO forward experiment executed with this interface.

NEXT is implementation, not another new suggestion/constant-family scan:
`COUPLED_IMPLEMENTATION_REQUIREMENTS.md` records the already-proposed reversible
q/a positive-rate skeleton in soft_rate_compatibility/SOFT_FEEDBACK_NEXT_DECISION.
Read-only code audit found NO actual joint q/a history solver for both topologies.
Current shared_spectral/bare segmented/signed forwards are scalar; field_rupture
is fixedGamma + conditional analytic a loss, not joint history. Implement the
existing skeleton with explicit retained q² structure choice if used; preserve
J_positive/J0 polarity normalization. Material same forbare/series; only external
circuit differs; total current/local voltage feedback shared; no per-cycle seeds.
Opening a invalidates old I->u inversions, old soft-only roots and smallfield_bound
(additional a*Vh*expm1(u) capture plus a evolution). Must rederive near-zero control,
validate both states and disabled-a regression before physical fitting.
If q still declines throughout RESET, old unimodality rejection persists.
Bare voltage-drive q/a skeleton is cooperative: componentwise-ordered entry
states still preserve current order; true history must create incomparable states
to reproduce circles11/14. Two state names alone are not a solution.
Original full46pairs/92phases/27023residuals/20gates active; bare4/23 not bridged,
clocks unknown, no default change/holdout/push. Goal remains active.

## Latest empirical integral diagnostic and common-lambda bounds (2026-11-04)

Read `analysis/EMPIRICAL_INTEGRAL_RESULTS_2026-11-04.md` and
`analysis/empirical_integral_2026-11-04/` first. Actual new evidence, no new
forward fit: unique session26234 exit0;426/426 conditional native inversions,
48/48 window/interpolant/order variants,20160 quadrature nodes,21108 static
circuit solves,20448 material-bisection iterations (22152 material evaluations),
8 linear two-coefficient identifications. Numerical time1.401817872s; I/O/hash
separate.16 sources/inputs frozen BEFORE launch, author DONE, unchanged after.
Runner SHA6ca987d6db74fc8af96ace80288a69a0db910dbfcbf79ecd200fdc95017e7a16.

Frozen a=0/readout/lambda=.4470398752554136/path clock.12 windows only:
each SET4/RESET4/SET5 outward .2->.8->1.6 and return1.6->.8->.2.
432 incidences/426 unique native points; windows cover6 of60 low-voltage
points, not all low20. Full1170 scoring contract unchanged.
Linear/PCHIP over merged71-point branches; Gauss8/16 per native interval.
All complete, no failures, retries, point removal, state anchoring or holdout.

Actual RESET two outward windows force negative coefficients for all4 variants:
alpha≈-.202,beta≈-3997. Linear/G16 rho=B/A .0006245623->.0038750723A;
eta=Delta_u/A2.2943769->15.2879732/V². Both increase, contrary to
eta=alpha-beta*rho with beta>0. This already conflicts before imposing shared b.
Column-normalized conditions positive≈20.35,negative≈5.395,rank2.
Positive coefficients identified from SET4 outward kappa≈448.068/r≈23686.176
predict SET4 high-return Delta_u25.2199 vs reconstructed.234628.
SET4 return sign constraints require J>3.772619mA AND J<2.042536mA;
low-return Delta_u=-.028548 is small and lacks experimental noise bounds.
Gauss8/16 max relative A/B difference8.87e-7/4.75e-7; interpolation difference
.7953%/.02284%. Inverse amplification chi≈2.1723–2.7536, not noise bounds.

Independent saved-result review12156 checks passes,19 actual input files;
all saved circuit outputs/20160 nodes/A/B/8 systems/closures checked, no new
root/inverse/ODE. PCHIP polynomial and material derivative not independently
rebuilt; actual review scope is in SAVED_RESULT_REVIEW.md. Root reviewed theory
and separately verified lambda-bound arithmetic from original saved nodes.

IMPORTANT next direction: don't treat frozen-lambda failure as all-family
exclusion or immediately add a state. New LAMBDA_NECESSARY_BOUND.md/.json:
R(lambda)=A_RESET_high/A_RESET_low strictly increasing on saved finite nodes.
With y_high>y_low>0 and T=y_high/y_low≈14.051>S=B_high/B_low≈13.084,
positive alpha/beta iff S<R<T. Jensen/support bounds imply lambda>2.37360–
2.37410/V and lambda<about5.12/V; conservative enclosure, NOT actual feasible
interval or selected parameter. Earlier lambda[0,2] search box cannot resolve
this exact reconstructed conflict, but was never an experimental bound.
SET4-return sign condition is sufficient above≈2.0176/V, so any possible
RESET-compatible lambda already removes that SET sign conflict; amplitudes,
common positive coefficients and b remain unchecked. Four configurations,
16 saved windows/6720 node incidences; zero new lambda evaluations/roots/ODE.
T>S depends on a small RESET Delta_u=.047941 and unknown observation/readout
errors. Bounds concern finite quadrature reconstructions, not true trajectory
or noise certificates. Next: bounded common-lambda feasibility THEN continuous
forward validation, one initial state; no blind extension of fixed-lambda96.

Bare circle10 ordered soft budget ALREADY executed in
soft_drive_structure_2026-11-03/soft_state_budget.json/.md:16 inversions,
20% sensitivity B_out>=.465058mA and Delta_u>=12.5628 to return.2V.
Don't rerun it as new work. Not yet compared to current kappa exp(2b) recovery
upper bound; can reuse saved bounds/path for a future finite arithmetic check.
Bare reversal/pure-loss requirements remain. Any added a changes readout and
feedback and invalidates automatic reuse of a=0 inversions/SET precision.
Full46pairs/92phases/27023residuals/20gates unfinished; missing bare4/23 not
bridged, clocks unknown. No default change/push. Goal remains active.

## Latest constrained history check (2026-11-04; all-three improvement, not converged)

Read `analysis/CONSTRAINED_HISTORY_CONVERGENCE_RESULTS_2026-11-04.md` and
`analysis/constrained_history_convergence_2026-11-04/` first. Actual new progress:
unique session47591 exit0, fixed old/request57 nominal+refined checks, then
constrained coordinate search and final two-precision3-phase chain. Same q²,
a=0, frozen readout/lambda/Ron/Gamma_c/T tables/path clock, common4 parameters
and ONE history initial state. Source freeze5 hashes written BEFORE launch;
source hash comparison passes, no edits during execution.

Guardrail adopted BEFORE execution: ALL THREE phase RMSEs <=same-precision
old control (not just SET4/SET5). Modeling goal only, not noise/confidence band.
Verified request57 refined RMSE .1286857445/.6927145550/.4908701818;
all1170 .4239938975. Current final .1325768352/.6938689323/.4519293826;
all1170 .4058920620 vs old .4443640076 (-8.65775%). Mean phase MSE
.2344236264 vs old .2635058877 (-11.03666%), seed .2457890037 (-4.62404%).
This candidate preserves old SET4 and improves all3 relative OLD, but SET4/
RESET4 slightly worsen relative request57. Previous unconstrained .391205
candidate had SET4 .235947; don't claim current .405892 beats that total score.

Selected request93 round12_axis0_minus: kappa57.6779879361/V²,
r15378.3627073/(A V²), J=.00375059354716A, b=.769297378470313,
initialu9.46455167265. True state chain ->3.49797843452 ->6.45197446951
->3.49797840604. Low20 RMSE .31123029/.88841823/.87058264, all retained.
Major failures persist: RESET4 peak .788037uA vs8.6317uA (~10.95x low),
return-.2V12.0477nA vs1.2504nA (9.635x high), SET5 outward+.2V25.2583nA
vs.725nA (34.839x high). RESET outward RMSE worsens .68838->.73710 while
return improves. Phase RMSE gains do NOT mean all branches/features improve.

102requests=4preflight+96search+2final;86 actual completed chains,16cache hits.
12complete rounds with step .02/.01/.005; request budget exhausted,
grid_stopped=false. Last full neighborhood centered on91, then93 accepted;
93 itself not fully polled. NOT grid convergence/stationarity/global optimum.
No restart/extra quota. Numerical time6.737229+84.574403+3.647435=94.959067s.
Final max precision logI diff1.83751e-6dec,u6.00157e-6. Both precisions pass
all3 goals, actual joins/RHS/independent48 native roots. No new extension
evaluation; do not transplant prior candidate's RESET5/SET6 performance.
Targeted independent review1786 checks pass:102request/12round scalars,
18verification phases+27search phases,98 actual file reads/45phase NPZs.
Not a full rehash of626 artifacts. Root visually reviewed1x3 I-V PNG.

Integral diagnostic derivation: fixed a=0/readout/lambda/path clock gives
positive Δu=kappa A-rB, negative Δu=alpha A-beta B, alpha/kappa=r/beta=e^(2b).
Discrete I-V does not give true between-point A/B; unknown real scan times
can change effective polarity coefficients. Never treat small interpolated
coefficient mismatch as an unconditional mechanism exclusion.
Actual saved-model-only selfcheck session32356 exit0,.009792s,13source hashes:
refined native12windows max|E|8.77686e-4, normalized2.06646e-4;
thinned3.62017e-3/8.39555e-4. Known true model identifies product ratio1.010755
native /1.044802 thinned, kappa error-.9439%/-3.8196%, positive scaledcond159.34.
Same-sampling solver E difference1.96695e-6. This diagnoses sampling/inverse
conditioning bias, NOT experimental physics. No new roots/inversions/ODE.
See integral_selfcheck/SELF_CHECK_REPORT.md and INTEGRAL_IDENTIFIABILITY_NOTE.md.

NEXT: do not blindly append another96 local requests or call this a completed
unified model. Mechanism work must address RESET peak/low-tail/next onset AND
bare mA branch variation/cross-cycle reversal. Integral compatibility may
diagnose the frozen soft family only with explicit interpolation sensitivity;
it does not identify a microscopic species or remain valid automatically when
parallel a opens. Any (u,a) proposal must use the same equations/feedback for
1R and1T1R and true history; re-evaluate readout/SET as a changes. Full original
46pairs/92phases/27023residuals/20gates remain; bare4/23 missing measured cycles
cannot be bridged; real timing unknown. No holdout/default change/push; active.

## Latest continuous joint calibration (2026-11-04; limited improvement, goal active)

Read `analysis/JOINT_HISTORY_CALIBRATION_RESULTS_2026-11-04.md` and
`analysis/joint_history_calibration_2026-11-04/` first. This turn made actual
numerical progress: one bounded shared-parameter SET4→RESET4→SET5 fit, followed
by fixed-parameter RESET5→SET6 training continuation. Session10395 exit0.
Same signed q² law, a=0, frozen lambda/Ron/Gamma_c/spectra/T tables/path clock;
common kappa/J/b and ONE whole-history initial u. Actual state/command handoffs,
no per-phase anchors, clipping, hidden retry, holdout, default change or push.

Old actual continuous Radau control RMSE SET4/RESET4/SET5:
.1338949308/.7054578723/.5243271889. Selected final_refined:
.2359465361/.6471679505/.4000564418. All1170 RMSE .4443640076→.3912050666
(-11.9629%); equal-phase MSE .2635058877→.2115140936 (-19.7308%).
Do not mix those two weighting conventions or claim all phases improve.
Low20 retained; new low20 RMSE .2778827477/.8096661016/.8159456765.
RESET4 peak .983131uA vs8.6317uA, return-.2V9.893576nA vs1.2504nA;
SET5 outward+.2V20.809831nA vs.725nA (28.70x high). Mechanism still inadequate.

Selected evaluated request86 old_optimizer_40: kappa27.4455429988/V²,
kappa/J8147.68328042/(A V²), J=.00336850882075A, b1.00945160737,
whole-history initial u8.52694901994 (fitted, not measured).
Final refined terminals SET4 3.19956595191, RESET4 6.64560320922,
SET5 3.19956610632, RESET5 6.64560321220, SET6 3.19956595668.
Near-repeating predicted cycles: max logI RESET4/5 difference7.44e-8dec,
SET5/6 difference7.23e-8dec; observed RMS differences .22552/.15692dec.
This property is of this candidate, not a proof for every parameter set.
Extension RMSE RESET5 .5046987580, SET6 .3632531874; excluded from selection.
These are training-internal continuation checks, not independent holdout.
No old extension baseline was run; no combined1840 objective.

85 objective requests + DOP853 control +2 final chains=88 records; one exact
cache hit, no numerical failure/timeouts. Both predeclared Nelder-Mead starts
stop at40 evaluations WITHOUT convergence. Best new-start MSE .46904935;
do not exclude high-rate region/all-q² family from this finite comparison.
Request57 old_optimizer_11 search-only RMSE .128686/.692714/.490870 improves
all3 slightly but wasn't selected by declared MSE and lacks final validation.
Thus selected SET degradation is not a proven unavoidable physical tradeoff.

Controls pass both methods. Final two Radau precisions use same candidate,
each own continuous5-phase chain, complete RHS+48-bisection independent native
root audits. All3710 native points included; max precision logI difference
9.30e-7dec. Fit numerical time73.49075s, final5.82325s, I/O separate.
New-stage nearzero per-physical-segment budget explicitly1e-10, vmin/rule
unchanged; local bound only, not trajectory error certificate. Frozen prior
sources unchanged. Root visually inspected the saved2x3 comparison figure.
Independent saved review finished:15054 checks,16 sources,622 artifacts,
265 phases/530 segments,48903 final RHS rows pass;136 additional feature
checks pass. Strict Gamma table endpoints rely on original phase_integrity;
independent saved arithmetic checked only the broader finite Gamma domain.

Source timing caveat: after process launch, before its startup notification
reached author, runner main-except reporting flag changed (only that line).
EXECUTED_RUNNER_SNAPSHOT.py exactly4ffd87...; input/source hash current5b99ed....
Normal model/optimizer/budget path identical; exception not entered. Do not
call process-launch-to-end source wholly frozen; source_files_unchanged checks
only input-hash-to-end interval. See RUNNER_SOURCE_TIMING_NOTE.md. No rerun.

NEXT: predeclare a bounded convergence check of the common-parameter model
with an explicit SET degradation limit, separating finite-search incompleteness
from structural failure. Do not blindly repeat40-evaluation searches or claim
family impossibility. Preserve one state history and per-phase scores. Full
bare/series model still needs mA branches/cross-cycle reversal and the original
46pairs/92phases/27023residuals/20gates contract. Bare4/23 measured but missing,
do not bridge missing cycles; timing remains unknown. Goal stays active.

## Latest training-history bounds (2026-11-04; common-parameter family still open)

Read `analysis/ZERO_CROSSING_POPULATION_RESULTS_2026-11-04.md` and
`analysis/zero_crossing_population_2026-11-04/` first. Previous goal turn and
this turn both made actual progress; full accurate shared1R/1T1R remains
unachieved. No new I-V fit/ODE/default change/push or original holdout access.

All31 RESET k return-.2 -> SET k+1 outward+.2 bridges, k4..34, are audited
training points with .4V paths. Reuse2 frozen inversions for4->5, execute60
new readpoint inversions/120 circuit rechecks (session23036 exit0,.030004s).
Fixed-m2 recovery budget violated24/31, early9/10,middle5/10,late10/11;
ratios1.213448--188.530222, strongest7->8. Seven others require negative
DeltaZ, so the growth bound alone cannot exclude them. Same history,
not31 independent trials; never replace original27023 residual/20gate contract.

Pure saved-endpoint structure screen: fixed other parameters, q^m requires
m<=1.443121658; q^2+epsilon*q requires epsilon>=.01791860167, both7->8.
Necessary boundaries only, not selected/fitted parameters or species.
Independent15-source/3405-check saved-result review passes. Eleven adjacent
float pass endpoints differ at~1e-14 in high precision; actual reported
conservative limits remain valid. No strict interval certificate/noise claim.

Separate complete-table capture bound executed once(session22467 exit0,
.006789s), no new roots/inversions/ODE. For a=0, i/Gamma<=Csign over full
Gamma domain, local v<=.2, plus declared near-zero freezing. Cpositive=
.00130274518657,Cnegative=.00122495417206 A/eV. With b>=0 and ANY nonnegative
recovery, Delta_u>=-.029893850984 under fixed kappa/J/readout/path clock.
All7 other bridges require decrease.077662--.674076, exceeding2.598--22.549x.
Original fixed m2 violations union31/31; changing recovery does not automatically
preserve the old24 growth violations. The7 capture violations do remain for
recovery-only changes under these other fixed assumptions. Four-source/87-check
capture review passes. Both figures root visually reviewed, all points retained.

Do NOT claim all shared-parameter q2 models excluded. Same frozen readout,
lambda,b domain,clock: exact endpoint matching necessarily requires
kappa_new>=188.530222*kappa_old=14648.82435194/V^2 and
(kappa/J)_new>=22.54898476*(kappa/J)_old=426403.083896/(A V^2).
See JOINT_PARAMETER_REGION.json. This does not force J alone to decrease:
at the necessary kappa lower boundary J<=.03435440527A. These are exact-point
necessary constraints/next-probe guidance, not a replacement fitting objective,
parameter selection or sufficiency. Readout/lambda/domain changes update bounds.

NEXT substantive step: finitely test continuous SET4->RESET4->SET5 under
the SAME q2 equations with jointly varied common parameters, using the new
necessary region to choose a finite comparison and keeping the old control.
Predeclare one whole-history initial state or consistent preparation history;
old SET4 initial u was inferred with old parameters, not measured. No phase/cycle
re-anchor. Test numerical stiffness fairly; numerical-domain failure is not a
physical family rejection. Report all original500/170/500 scores separately.
Do not immediately add arbitrary recovery or parallel path merely to evade
these checks. Full32-series and segmented bare history still must follow.

Bare1R additionally needs mA-level branch variation and cross-cycle reversal;
soft-only progress does not satisfy it. Any retained a must share the same
material/feedback definition and arise from history, never topology switches
or per-cycle fitted states. Opening a invalidates automatic SET accuracy and
old soft-only inversions. Bare4/23 measured but missing; timing remains unknown.

## Latest zero-crossing necessary condition (2026-11-04; goal active)

Read `analysis/ZERO_CROSSING_COMPATIBILITY_RESULTS_2026-11-04.md` and
`analysis/zero_crossing_compatibility_2026-11-04/` first. Same frozen
soft-only readout/electrical map/positive parameters; no new ODE/fit/b point.
Exactly two conditional readpoint inversions: RESET4 return-.2V1.2504nA
gives u=8.70744814347,Z=6047.789688; SET5 outward+.2V.725nA gives
u=10.00167059803,Z=22063.293919. These are pointwise conditional states,
not zero-voltage initial states, physical trap populations or model re-anchors.

Z'=kappa*v[exp(2b+lambda*v)-(i/J)*exp(-2b)*Z*(Z-1)] negative;
positive removes b. Passivity and nonnegative capture bound -.2->0->+.2
DeltaZ by166.630873, even for variable b(s) in[0,ln10]. Wider simple
analytic bound plus1e-6 margin171.633866 is exceeded by padded required
lower16015.503950, ratio93.312027. This is a conditional state-budget
conflict, NOT current-error factor/full-RMSE bound/original20 aggregate gate.
No experimental noise band or rigorous interval-arithmetic certificate.

Two inverse algorithms per point, four circuit rechecks, analytic/Gauss16/32
checks pass; numerical compute.004055s. First input-only attempt interrupted
inside historical loader before numerics (session49404 exit130), preserved.
Minimal loader now instantiates same frozen static classes using only actual
tables/parameters and imports transistor.py directly, avoiding package-wide
old-model imports/unused trace hydration. Executed numerical session11706 exit0.
Independent saved-result review passes:13 actual sources and125 checks.
Initial audit's KCL reconstruction discrepancy1.72e-21A is covered by actual
slope*voltage-ulp propagation; preserve audit correction, no physical rerun.

Stop tuning this frozen b family, including bounded field/state-dependent b.
Next check the same necessary condition on the other known continuous
training-cycle transitions to establish whether this gap is systematic;
then distinguish low-q recovery structure, zero-field/interphase real-time
evolution, and fixed single-state spectral readout. Unknown timing remains
unknown; the test excludes extra unrecorded waiting evolution by assumption.
Does not prove a specific microscopic process or rescue by adding path a.
No holdout/default change/push; full shared1R/1T1R goal remains unachieved.

## Latest signed continuous-history result (2026-11-04; accuracy insufficient)

Read `analysis/SIGNED_SOFT_HISTORY_RESULTS_2026-11-04.md` and
`analysis/signed_soft_history_2026-11-04/` first. The signed extension proposed
below is now implemented and executed once. Same fixed positive m=2 law,
true negative spectrum and measured reverse transistor; a=0. New constant
b preserves positive kinetics through kappaq=kappaplus*exp(b),
J0=Jplus*exp(2b). Freeze lambda/Jplus/kappaplus/readout/SET4 terminal.
No phase/cycle re-anchor, state clipping, invented clock seconds or holdout.

12 static points and24 RHS checks pass. Fixed SET4 regression reproduces
.133894906 with max logI difference4.45e-7dec. Two endpoint +8 bounded
RESET4 calls select b=.5811330029022057 (interior, optimizer success;
no global-optimum claim). Total15 complete phases/30segments,0 failures,
5.497727s numerical compute, separate I/O. SET5 never selects b but belongs
to original training, not an independent holdout or untouched experiment.

Final original170 RESET RMSE=.7054578802dec (out.68838,ret.72213);
original500 SET5=.5243272707dec (out.73921,ret.05832). Low20 retained.
RESET native peak1.183713uA vs8.6317uA; return.2V19.03213nA vs1.2504nA.
SET5 1uA onset.82--.84V vs1.18--1.20V;10uA1.28--1.30vs1.56--1.58V.
Actual u chain3.655082141628295 ->5.999383986199006 ->3.655082161884186,
joins exactly0. Do not equate old SET4 inverse initial state with required
SET5 initial state or infer32-cycle stability from these close final states.

Independent review:22 actual sources,26 reread artifact hashes,all15 trial
JSON and6 main arrays pass. Two precision max logI differences RESET1.26e-7,
SET5 2.36e-6dec; full native legacy/KCL/RHS/joins pass. Numerical success
does not establish physical accuracy or full unified1R/1T1R explanation.

Conditional comparison proof: fixed-command/state negative du'/db>=0,
so admissible unique scalar trajectories have u nondecreasing with b and
native currents nonincreasing. b>=b* leaves peak>=7.29205x too low;
b<=b* leaves fixed.2V tail>=15.22084x too high. Hence worst feature factor
at least approximately7.29205 in this frozen family. Not170-point RMSE bound,
MSE-unimodality/global-optimality proof, original20 aggregate gate or all-m2
rejection. Numerical ratios are not rigorous interval arithmetic bounds.
Stop further tuning of this b family. Next derive RESET state/field-history
requirements from existing training data under fixed readout before adding
new rates/channels. Adding a changes I/q and invalidates automatic SET-fit
preservation. Full46pairs/27023residuals/20gates remain outstanding.

User confirms bare numbering is chronological, same label SET then RESET,
and missing4/23 were actually measured. Known segments1--3,5--22,24 cannot
be directly bridged across missing cycles; no per-cycle fitted reset states.
Unknown scan/dwell/wait times stay unknown. No default-model change or push.
Stage frozen after root validation/manifest; goal active and unachieved.
Use runner write-time catalog plus independent final-array/source audit;
do not rehydrate all unused historical traces for the next-stage check.

## Latest fixed-m2 SET improvement (2026-11-04; full unified goal still active)

Read `analysis/COOPERATIVE_RELAXATION_RESULTS_2026-11-04.md` and
`analysis/cooperative_relaxation_2026-11-04/` first. This stage made actual
accuracy progress on the conditional cycle4 soft-only SET component, not a
complete common 1R/1T1R model. No RESET/bare history/original20 aggregate
evaluation, original holdout access, desktop default change or push.

Fixed new discrete hypothesis m=2: q'=kappa*v[(I/J)*(1-q)-exp(lambda*v)*q^2],
q=exp(-u), Gamma=gc/expm1(u). Same spectrum/Ron/gc/transistor/.2V anchor and
three continuous fitted parameters as m1. q is not measured occupancy; q^2
does not establish bimolecular recombination, microscopic reversibility or
unique species. At fixed coefficients q^2<=q, so not universally faster loss.
Current rates use command path, not seconds. The local stability motivation
does not assume actual dynamic return points are equilibria.

Same26 saved inversions/24intervals/82NNLS initializer,3 seeds,one bounded
80-evaluation fit and2precision checks. All85 actual trajectories/255segments
completed,0 domain-outside requests or numerical failures,32.183s computation.
Optimizer hit evaluation cap; NOT converged/global optimum. Relative kappa/J
bounds use each model's own seed, so m1/m2 absolute search domains differ.
Do not attribute all improvement uniquely to structure or declare m1 no-go.

Final lambda=.4470398752554136/V,J=.0041089272077381635A,
kappa=77.70013824898767/V^2; not at bounds. Original500 RMSE=.133894906044272
decade vs old sinh .182230639877782 (-26.5%) and finite m1 .345118456388450.
Outward=.166806051 improves;return=.089618264 remains worse than sinh .063518.
Low20=.405804217;remaining480=.108688984 is diagnostic ONLY, not primary score.
Absolute-factor median1.14921 is separate from RMSE; .134dec is not13.4% error.
1uA crossing matches1.32--1.34V;10uA still early1.56--1.58vs1.62--1.64.

High-V downturn remains:5V prediction39.28695uA vs observedout50.94/ret50.97;
ret4=46.91909vs44.4uA,ret2=31.31559vs29.778. Only1/3 conditional endpoint
inequalities and1/6 single-cycle references pass; NOT original aggregate gates.
Two-precision max logI change1.14655e-6dec,u3.50531e-6; native/RHS KCL
~6e-19/~6.8e-19A,legacy503 logI difference1.97e-13dec,continuous apex/anchor.
All500-point comparison PNG/PDF root visually reviewed,high-V deficit retained.

Matched m1 control executed ONCE with qref=.05490380671259502,
kappa1=kappa2*qref,J1=J2*qref,lambda1=lambda2, no refit. Both precisions
failed in first outward segment at~4.707V,u~51.92 as Gamma<1e-24 table floor.
0.432s actual compute;20 source hashes unchanged. No complete500 trajectory,
RMSE,full precision comparison or native legacy audit exists for this control.
This is numerical-domain unresolved, NOT physical rejection or verified causal
advantage. Preserve failures, do not expand domain/retry silently. Capture
coefficient match is not same flux once trajectories differ.

Separate last computation reused old initial_barrier saved circuit corners for
this unique mapped m1 lambda/J, with original qbar/readout/500 mask; no new
root/inverse/ODE/fit or parameter choice. Four cases only,two grids x0/.003
material envelopes,10s cap,actual.000441s. New continuous-negative prefix is
intersected with old certified domain. Refined nominal .2--1.42V,62points,
necessary original500 RMSE floor.324509556; .003 envelope .2--1.41V,61points,
floor.312600351. Coarse floors.313246/.301399 agree in direction. All other
errors optimistically0,denominator500. This is a conditional bound for fixed
mapped lambda/J and initial state (any finite kappa>=0),NOT actual control RMSE,
all-m1-family rejection,experimental noise/CI or microscopic proof. It supplies
model-internal evidence of a startup barrier under the matched linear law;
the failed full control remains numerical-domain unresolved.

Independent actual audit now COMPLETE:15 source hashes,all85 saved trajectories,
255segments,original500/group scores,state/RHS/joins,final precision/legacy pass.
Matched-control failure audit and separate30-source/four-case barrier review
also pass. First audit comparison incorrectly demanded early trial JSON equal
to final summary with later-added legacy field; original audit/script failure
retained,comparator corrected and full read-only audit passed,no physical rerun.
All writers/calculations stopped before completed_stage_manifest; stage frozen.

Next substantive work: read SIGNED_EXTENSION_READINESS.md. Freeze positive
parameters; test actual SET4terminal->RESET4->SET5 with no phase/cycle re-anchor.
One explicit effective polarity b can preserve positive law via
kappaq=kappaplus*exp(b),J0=Jplus*exp(2b). Negative law would have capture
exp(-2b) and loss exp(2b+lambda*v)*q^2. This is a design, NOT implemented or run.
Calibrate b on RESET4 only in a predeclared finite trial; SET5 is unused in that
trial's fit but belongs to original training, not a pristine holdout. First
test a=0; activating a changes totalI/q and invalidates automatic preservation
of this SET fit and its old soft-only .2V inverse anchor. Old scalar spectral
integrator's stop floor/clipping must not be reused for reversible q/a.
Keep missing4/23 history segments and original27023 residual/20feature contract.
Do not continue polishing isolated SET or call current goal complete.

For future stage verification, hash actual inputs/source dependencies and
newly changed artifacts. Do not repeatedly hydrate all unused historical
optimizer traces unless a discrepancy or specific audit requires it. This
stage's explicit prior-manifest audit already preserved179/179 artifacts
(mutable CLAUDE excluded); keep that record instead of repeating it per run.

## Latest reversible-family calibration (2026-11-04; stage began 11-03)

Read `analysis/REVERSIBLE_FAMILY_RESULTS_2026-11-04.md` and
`analysis/reversible_family_2026-11-03/`. Previous goal turn made progress;
this turn completed an actual finite three-parameter calibration and numerical
verification. Goal remains active/unachieved. No full 1R/1T1R history, RESET,
bare model, original20 aggregate contract, holdout, default change or push.

The old .490895645 all-kappa bound covered one lambda/J seed calibrated by
treating return2/4 measurements as equilibria. It did not exclude the family.
New stage keeps same q/u law, spectrum, Ron/gc, transistor and .2V anchor,
but calibrates lambda,J,kappa from dynamics without either equilibrium rule.
No additional state or physical term; q is still not measured trap occupancy.

Initialization uses all26 saved inversions,24 branch-local intervals, fixed
lambda0:.05:2 and Gauss8/16;82 finite NNLS profiles, no new inverse or KCL.
Interpolated u,v,logI are only a surrogate, generally not intermediate KCL.
Three fixed-bin seeds (.65,.95,1.35)/V are numerically valid but actual500
curve RMSE3.88027,4.72819,5.58258dec. Do not call integral loss an IV score.

One bounded Nelder-Mead from best seed:80 objective requests,72 in-domain
full trajectories and8 out-of-domain inf requests; no clipping/fake residual.
Search lambda[0,2], each kappa/J individually within seed/100..seed*100.
Max evaluations reached after43 iterations; NOT converged or global optimum.
77 actual complete ODE trajectories=3seeds+72fit+2precision;231 segments,
0 numerical failures,26.252s actual computation. Backward .2->0 plus
.2->5->return0 preserves same conditional solution and continuous apex.

Final lambda=.10421299497906249/V, kappa=5.254465516635359/V^2,
J=.00014169711494164486A. Refined all500 RMSE=.3451184564dec;
outward=.36969672,return=.31865002. Better than old fixed-seed necessary
floor .490896, but WORSE than existing sinh .182230640. No project-best
replacement or accuracy-improvement claim. This is conditional same-cycle
calibration with3 fitted kinetic parameters, not independent prediction.

Main errors: outward(.5,1.5]50-point RMSE.64506;1uA crossing1.40--1.42V
vs observed1.32--1.34,10uA1.56--1.58vs1.62--1.64 (too steep).
Return2/4 currents52.6389/77.7496uA vs observed29.778/44.4uA.
Only2/3 old endpoint inequalities and1/6 single-cycle reference comparisons
pass. Passing a lower endpoint bound does not mean amplitude is accurate;
these are NOT original32-cycle aggregate results.

Two precision max current difference2.44118e-6dec/state5.71140e-6;
native/RHS KCL about2.4e-17A, full503 legacy current difference1.06e-12dec,
continuous apex/anchor errors0. Numerical mismatch is negligible compared
with model residuals. Comparison PNG/PDF retain500 points and explicitly
label higher new training error; root visually inspected PNG.

Whole-stage saved-array/source audit is complete: all77 stored trajectories,
85 request records and15 run-source hashes checked independently. All22 prior
frozen research artifacts match. Use completed stage manifest and
RESULT_REVIEW.md for final provenance before further work.
Keep this bounded fit and all85 request logs/77 traces frozen. Do not call
local unconverged fit a no-go theorem, and do not silently extend its budget.
Next physical question: can activation and return self-limitation coexist
under this stopping surface/rate structure and fixed readout? A new extension
requires evidence-constrained state/readout coupling; no arbitrary capture
function or renaming q as microscopic occupancy. Original unified goal and
known histories/missing cycles remain intact.

## Latest reversible-capture screen (2026-11-03; no improved full-model fit)

Read `analysis/REVERSIBLE_CAPTURE_RESULTS_2026-11-03.md` and
`analysis/reversible_capture_2026-11-03/` first. The goal remains active and
unachieved. This stage made actual mathematical/numerical screening progress;
the full common 1R/1T1R model, continuous known histories and original 20
aggregate feature gates remain unverified. No original holdout/default/push.

One fixed two-return-point-conditioned seed was screened:
q=exp(-u)=Gamma/(Gamma+gc), q'=kappa*v[(I/Jplus)*(1-q)-exp(lambda*v)*q].
lambda=.34972785733630246/V, Jplus=.00028291476913783176 A; same fixed
spectrum, Ron and transistor map, positive SET, soft-only a=0. q is not a
measured trap occupation. Path kinetics do not identify seconds or species.

Reversible first-passage budget: 0.461107 s, 175972 KCL nodes, no new inverse,
ODE or fit. Refined necessary kappa lower 7.6325119 nominal / 7.4293790 with
.3% material envelope; refinement tightens but does not establish convergence.
No whole-q-cell barrier in that broad screen does NOT prove reachability.
Saved ret2 equilibrium attracts, ret4 repels; not two stable working points.
Stability-label-only correction preserves execution provenance and NPZ hash.

Sharper fixed-initial-q barrier: 0.036146 s, 2462 KCL nodes. For every finite
kappa>=0, q cannot cross its .2 V conditional upper bound while command stays
in certified outward prefix .2--1.675 V nominal / .2--1.665 V with .3%
material envelope. On original 500-point mask, 74 constrained native points
alone imply RMSE >= .491627993 / .490895645 decade, setting all other errors
optimistically zero. At 1.4 V, envelope current ceiling .02192733 uA vs
observed 1.5682 uA, at least 71.518x underprediction. Prior same-conditioned
sinh actual RMSE .182230640 is lower; stop this fixed seed's kappa tuning.
Do not promote this to a rejection of all reversible models, free lambda/J,
other initial states, added paths or original32-cycle aggregate contract.

Independent review verified the proof, saved corner arrays, contiguous prefix,
original data/mask and score denominator. Prepared reversible_forward.py was
syntax checked and reviewed, including legal root bracket repair, but was NOT
run and no kappa was selected. All previous21 frozen research artifacts match
the previous manifest (mutable CLAUDE excluded).

One further parameter-free saved-array check is complete and independently
reviewed: same-temperature terminal forward event flux Iplus=I/[1-exp(-v/VT)]
with frozen 300 K (VT=.025851999786435 V). Multiply saved capture upper by
C(vmin), recompute guard, keep lambda/J/qbar/readout fixed. Maximum correction
1.000436961565; all four prefix endpoints and original500 RMSE floors remain
unchanged. No new KCL/inverse/quadrature/ODE/fit. This simple replacement also
cannot rescue the seed; it is not a rejection of local trap-capture physics.

Read CAPTURE_FLUX_MODEL_AUDIT.md before a further kinetic extension: terminal
transport does not identify local capture, Gamma is an energy-width/readout
coordinate rather than a measured slow rate, and q cannot just be renamed
physical occupancy. Preserve all rejection scopes; do not restart a kappa,
field-factor, temperature or nearby lambda/J scan without a new evidence-based
finite plan. Read the completed stage manifest for final provenance/state.

## Latest rate-structure screening (2026-11-03; no new full-curve fit)

Read `analysis/RATE_STRUCTURE_RESULTS_2026-11-03.md` and
`analysis/soft_rate_compatibility_2026-11-03/` first. This goal turn made
actual progress; goal remains active/unachieved. Earlier conditionalcycle4
SET .18223064dec remains, no newODE/fit/holdout/default/push. Fullcommon
1R/1T1R46pairhistory and20aggregatefeatures still unverified.

Completed6anchorinverse early/latebudget(.051s): fixedbeta6JS/backbone,
out2lower26.105524uA vsret4upper44.4812uA permits tinyu loss. All3tested
mobility rateintervals disjoint; refined .3%materialenvelope lower/upper
ratios58.03constant,99.62linear,1301.30sinh. General finite nonnegative
nondecreasing voltage-onlyM: K=kappa*M(2V)>=.48229498 but<=.00854037V^-1,
ratio56.47. Strong/weakmaterial corners re-closeKCL andrelaxanchors.
No more fieldfactor/rate scans atthisfixedstoppinglaw/conditions.

Completed4moreinverse+4terminalroots(.006s): same nonnegative r(A,u) with
botharguments nondecreasing implies out4->5->ret4 loss>=ret4->2loss
(each2Vcommandpath), hence2L<=H+U2. Conservative .3%case violatesby
.06590315u. Spectral chi<=2/(1-exp(-u)),u>=2.88024 gives sufficient
beta>=2.118916 forfullpathmonotonicrate, includesbeta4/6ANYJS,kappa
andnondecreasingM. Holdingout2+ret4forcesret2<=26.970uA vs28.515lower;
holdingout2+ret2forcesret4>=45.510uA vs44.4812upper. Theseareconditional
finitecurrenttradeoffs, notglobalRMSEfloors orallmaterial-family no-go.
Crucial: out2ispriorconditionalarrivalanchor,NOToneoforiginal20primary
features;ret2/4bands are32recordmedianreference,notpercyclenoisebounds.
DoNOTupgrade thissinglecurveconstraint tooriginalaggregate infeasibility.
Equalcommands-paths are NOTknown equalphysicaltimes;no clocksinvented.

Actual newhardcomponent screen(4.713s): q=exp(-u), a'=v[kn exp(sigma*b)
q(Acap-a)-kd exp(-sigma*b)(1-q)a],u nondecreasingduringRESET implies
anondec->noninc(singlepeak). FixedsmallsoftGamma_cap=.0018213724515eV
isHYPOTHESISnotbaremeasurement. All2799nativepoints and14training
recordskept;allowBpointwisefreein[0,1.003Fcap]. Nominal14/14and20%raw
currentband7/14fail(1,6,9,12,13,14,15),allwiththree>=10uAwitnesses.
Record15at|V|.075/.725/.95V requires a>=9.85621mS,<=.152900mS,
>=.301852mS under20%;weakside/valley1.97418. Stopthiscomponent's
ratefittingunderthemonotoneu+smallsoftpremises;notalltwo-state rejection.
216integerintervalcontrols+6longcontrols andindependent28caseauditpass.

Independent reviews coverrates,threepointproof,intervalalgorithm andsource
hashes. Newstage ROOT_VALIDATION.json records10newanchorinversions,
4tradeoffroots,0newODE/fit. Earlierresearchartifacts preserved;one stale
agentrevisedoldTHEORYbutrootrestored exactfrozenhash42a23c...a427,
newrevisionkeptonlyinnewstage. Oldbareattemptstoppedduringimport,
noadditionalinversions/resultwrites. No livecalculations atstagefreeze.

Nextsinglecandidate is REVERSIBLE q=e^-u replacingu (notaddingthirdstate),
Gamma=gc*q/(1-q), q'=kq*v[(i/Jplus)*(1-q)-exp(lambda*v)*q], a remains
independentretainedpath;totaliandactualsignedV consistentlyused.
Two-pointstoredret2/4algebraexamplelambda~.349728V^-1,Jplus~.282915mA
isconditional2pointcalibration,notpredictionorvalidatedtrajectory.
Important localstability: savedderivativeexample is repellingat4V(~+.238)
andattractingat2V(~-.105);not2stableequilibria. Firstpredeclareone-seed
reversiblefirstpassage/stabilitybudget(twofixedgrids,.3%envelope,60s,
noODE/optimization),then decidewhetheractual2precisionforwardwarranted.
DoNOTreuseoldmonotone-statebudgetunchanged. Fullsignedq/a skeleton and
assumptionlimitsin SOFT_FEEDBACK_NEXT_DECISION.md. No newcandidate
implementation,ratebudgetorforwardhasyetbeenrunforthisreversiblelaw.

## Latest SET drive structure and one fixed sinh probe (2026-11-03)

Read `analysis/SOFT_DRIVE_STRUCTURE_RESULTS_2026-11-03.md` and
`analysis/soft_drive_structure_2026-11-03/` first. This goal turn made actual
progress, goal stays active/unachieved. Full unified model, original20
aggregate gates and46pair history are NOT completed. No holdout/default/push.

Saved linear-field beta6 SET diagnosis:87.19%SSE is outward(.5,1.5]V;
1uA crossing.76--.78V vs measured1.32--1.34;10uA1.18--1.20 vs1.62--1.64.
Return D=lnI/JS+6u crosses nonpositive4.96->4.94V;247/250 nonzero native
return states then freeze. Positive mobility changes lag, not D=0 surface.

29fixed diagnostic inversions: exact out2/ret4 required u3.000437/3.047070,
but minimax necessary log error only.00776739dec(~1.8%factor), not cause to
reject useful approximate model. Original out2lower26.105524uA/ret4upper
44.4812uA permit u in[3.044494830,3.057700121]. No pairwise band conflict.
Next sharper necessary late budget: out2->5->ret4 can lose at most.013205291u.
Compare its integrated positive-drive lower bound against early .2->2 rate
lower. Formula written in THEORY_DRIVE_STRUCTURE.md, no new grid run yet.
All inverse conclusions conditional on frozen spectrum and transistor map.

One data-informed exploratory fixed effective scale Vf=.5V added in
M=Vf*sinh(absVh/Vf), lowfield M~absVh. Samebeta6/JS/gc/Ron/.2Vanchor.
Unique kappa=2*64cellnecessarylower=.08061866593V^-2, no scan/fit or later
rate tuning. Actual same500cycle4 RMSE .18223064 vs prior .47073536(-61.3%);
outward .24976273,return .06351842,lowout10 .474615,remaining480 .172535.
Tradeoff important:out2 only15.4756uA, fails26.1055lower. Other2oldbudget
endpoints pass. Return4~48.796uA stillhigh,2reference slopes stillfail;
NOT a passing unified model. Crossings now1.46--1.48/1.86--1.88V, too late.
Two precision runs completed1.153s;maxcurrent difference4.004e-7dec,
maxstate9.622e-7. Nearzero omitted integral <=1.8044e-17permonotone
microsegment is local only, not a fulltrajectory certificate. Old sources
unchanged. No recoveries/additional seeds needed for this candidate.

Bare record10 pure-loss-a hypothesis now has actual soft-state budget:
B_out>=Iout-vGset and B_return<=Ireturn. Nominal/20%rawcurrent sensitivity
requires B_out>=5.15852/.465058mA,Gamma_out>=.969054/.091518eV.
At20% Gamma_out is>=50.25x prior actualseriesmaximum;tail.2V requires
delta_u>=12.5628(nominal13.0073). A tiny softstate modulation of a loss
rate cannot supply required background amplitude under this decomposition.
This is NOT measured softfraction/state, not proof of2continuous states or
species, not a clock constraint.16inversions within originaltable,4direct
32/64endpoints within inherited.3%table envelope. Initial malformed
reference path failure at0inversions archived underpre_reference_path_fix;
same16problem plan completed after correction, no physical expansion.

Delegated agents hit usage limit after probe completed; root verified no
matching calculation processes live and took over remaining docs/bare run.
Final independent saved-array/source arithmetic review is ROOT review,
not a completed separate-agent review.107checks passed;scope is thisstage,
not fullmodel test suite. New figure visually inspected; baseline and failure
archives retained. Next preserve all known histories and test coupled memory
or explicit competing mechanisms before promoting a shared candidate.

## Latest RESET structure and conditional SET improvement (2026-11-03)

Read `analysis/RESET_STRUCTURE_RESULTS_2026-11-03.md` and
`analysis/reset_shape_2026-11-03/` first. This goal turn made concrete progress;
goal remains active and unachieved. No complete unified model promoted, no
original holdout read, desktop change or push. Prior frozen stages preserved.

New spectrum-independent RESET constraints (training14 only): static within-
cycle B(v), Ohmic hard path, and no growth since preceding SET read imply
0 <= Iout-Ireturn <= v*Gset. Record10 at.975V gives ratio1.563502; even20%
symmetric raw-current sensitivity leaves.465054mA excess. This comparison
allowance is not measured instrument noise. Shared-background relaxation
also fails at three all-record high-signal nodes under20%. Do not use enormous
ratios with nA denominators as the main evidence; negative branch-difference
violations are all low signal. Branch-product test has0/1400 robust usable
pairs and is NOT experimentally rejected. Common normalized shapes span215.5x
at.925V; SET+RESET joint sensitivity needed87.25% merely for pointwise contact.

Independent same-polarity RESET order witness avoids the SET-boundary issue:
record11/14 currents at.05V are.155152/.790152mA, but at.725V are1.421159/
.125306mA. Remains reversed at20%; contact delta67.1742% is not measured noise.
Common protocol, unique scalar continuous dynamics and same-direction
monotone readout preserve ordering even for lambda(v,a), so that extension
alone is insufficient. Protocol variation, stochasticity, additional memory,
nonmonotone readout or non-order-preserving jumps can change the premise;
no unique microscopic mechanism or mandatory2continuous-states claim.

Soft RESET Gamma_eff at six points is only an upper bound when hard current
is nonnegative. Old static Gamma /1.685 suffices for tail upper gate alone;
do not assume dynamics is required just because old background was too high.
Without background fraction lower bound a small static Gamma remains possible.
At assumed soft fraction>=50%, monotone Gamma is feasible13/14 but common
dlnGamma/ds=-kGamma*|V| intervals are empty; q>=90% even stricter. These are
conditional on fixed spectrum/fractions, not measured soft-state trajectories.

Completed soft SET beta1/2/4/6 x5 JS conditional budget (20fixed pairs), no
extra5% low-bias gate: respectively0/0/3/4 pairs not excluded. Then ACTUAL
two fixed singlecycle4 SET forwards: beta4/f.95 and beta6/f.8, same gc/Ron.
Backward .2->0 supplies conditional initial state, then full0->5->0; all500
original eligible points retained. beta6 meets three original endpoint
budgets, yet full/outward RMSE .698574/.985926dec. beta4 full .774736.
Two-prescribed-precision current differences <=5.45e-6dec; initial interface
boundary bug and two nearzero table-domain stops are preserved, not physical
no-go results. Two recoveries use proven exact inactive drive, not current
extrapolation.6solver attempts,4completed traces,2physical seeds;3.505s total.

One further PREDECLARED linear-local-field probe: beta6/f.8 same JS/gc/Ron,
du/ds=-kappa*|Vh|*positive(logIT/JS+beta*u). kappa numeric .8900744371 now
V^-2, no fit/new exponent. Conditional necessary lower .234575881; upper
unbounded is not a pass. Actual full500 RMSE .470735, outward .662697,
return .063375, outwardlow10 .621857, remaining480 .471845. Compared to
same-cycle beta6 constant mobility, full RMSE decreases32.6%; not an improved
full shared-model score. Three actual endpoint budgets pass; Ireturn4V and
two reference slopes still miss. No32cycle aggregate gates computed.

Field probe nominal was recovered once using a BOUNDED nearzero neglect:
I<=A/Ron, Vh<=A,u<=80 imply local omitted state integral <=1.9922e-16
per monotone0<->vmin passage. This is not exact inactivity or a rigorous
whole-trajectory error bound. No domain-extrapolated current or state clipping.
Refined trace bitwise preserved; native current/state differences2.96e-7dec/
6.93e-7;3solver attempts,1physical seed,1.242s. Archived failures remain.
All complete scores include anchor/lowbias, so are conditional calibration
diagnostics, not holdout predictions. Basic drive is +beta*u for both polarities
in u=log1p(gc/Gamma); external derivative sign differs. RESET not changed/tested.

Next focus: SET still rises too early over roughly.2--1.7V; constrain activation
and acceleration before increasing a global rate. Bare RESET cannot simply
reuse pure loss/static background or a scalar feedback fix. Any joint extension
must generate its extra memory/path from actual history and preserve all known
segments; do not fit per-cycle rates, thresholds or random realizations.

## Latest mechanism selection (2026-11-03; conditional trial not promoted)

Read `analysis/MECHANISM_SELECTION_RESULTS_2026-11-03.md` and
`analysis/mechanism_selection_2026-11-03/` first. Goal remains active and
unachieved; this stage made concrete progress, not a blocked turn. No full
unified model improvement or real 46-pair joint fit. No holdout, desktop
default change or push. Three checked primary sources support considering
`I=Fsoft(V,u)+aV`, but do not establish two paths or a metal species. Nature
VG1.1 stable 1T1M is interpreted as nonfilamentary; preserve SI limitations.

Prior-SET low-read log conductance correlates .99549 with following RESET
low-bias slope and .69806 with peak current. Peak-only internal LOO can reduce
RMSE while worsening median factor error; same-cycle measured Vpeak is target
information, not prediction. Keep all 14 records including >10mA peaks.

Completed bounded conditional RESET trial: common frozen background Gamma,
Ron58.4630ohm, `a'= -kappa*(abs(Vh)-Vr)_+*a`; prior-SET-derived a0 versus
fold-training SET-geomean constant. Three common fitted parameters, 3 fixed
starts x (full+14 folds) x2 models=90, all finite solver success, no expansion.
Original2799 mask/4 feature gates preserved. Conditional/control training
RMSE .86952/.88813dec; internal LOO .89663/.91856dec, median factors
2.6695/2.7216; both only1/4 gates. This is fixed-preselected-backbone LOO:
Ron uses full-training low reads and spectrum was previously selected.
Only 3 new parameters are fold-refit; NOT end-to-end CV or future prediction.
Measured SET input means it does NOT predict a formation or carry full history.

All90 fits Vr near0; full-fit peaks all.375V vs observed.45--.975V. Conditional
aggregate peak2.551mA and tail803.9pA fail original gates. At return.2V,
hard path only5.70e-22A; fixed background causes tail failure, not inadequate
path loss. Increasing kappa cannot fix it. No proof against all exponential
laws. Independent implementation/metric and scientific-scope review passed.
Next necessary field-only-loss check: with frozen common B(v), symmetric
triangle and same path rates, Qout*Qreturn constant in v, where Q=(I-B)/v;
normalized trajectories must share shape across cycles. Check positive
resolvable excess before logs; do not subtract invalid backgrounds silently.

Soft-only series cycle4 .1/.2V inversion fails an EXTRA5% local band, but
this is not the original primary contract. Minimal relative half-band35.76%;
optimistic two-point common-Gamma RMSE.16377dec contributes only1.676e-6
to original totalcost (series SET denominator16000). Raw zero .22379nA is
not subtracted or identified as an offset. Beta1/2/4/6 budget actual count0,
JS optimization0, ODE0: do not claim four betas rejected or prohibit forward
with finite low-bias residuals. Preserve possible approximate compromise.

Runtime cloud placeholders stalled reads. Four temporary copied inputs at
`/private/tmp/vulcan-runtime-cache/` have matched frozen hashes; verification
is in SOFT_FEEDBACK_LOW_BIAS_BUDGET.json. Do not confuse input waiting with
solver failure or duplicate a live run. Trial/audit writers are terminal.

## Latest Ron and scalar-cycle constraints (2026-11-03; no improved fit)

Read `analysis/RON_AND_CYCLE_ORDER_RESULTS_2026-11-03.md` and
`analysis/ron_scope_2026-11-03/` first. Prior goal turn made concrete progress;
goal remains active and unachieved.112bare training low-read points imply a
conditional Ron ceiling61.5400ohm (64.6170/67.6940with5/10%comparison allowance),
not a lower bound or a new fittedRon. Prespecified commonRon scenarios are
93.4801,58.4630,29.2315ohm. Re-invert anchors/re-solve KCL at changedRon;
never directly rescale series terminal currents. SmallGamma simultaneous
Ron/Gamma/gc scaling nearly preservesu,current,and original dynamics.
Closed19originalgc x3Ron screen gives0/57SET rate overlaps, including no k
algorithm box; minimum ratios1.806969/1.802936/1.799571. JS1mA cap removed for
series, JS>=1e-7retained. No newODE/fit; do not restart or expand this screen.

Confirmed order supports a separate conditional model-class constraint:
unique deterministic scalar flows under identical COMPLETE periodic protocol
are order-preserving. Monotone common readout then makes all same-phase-node
currents monotone across a segment. Bare5--15low reads reverse strongly;
both directions fail5/10/20%linear relative bands, two-node common direction
needs delta>=.653703. This is NOT an instrument-noise estimate or proof that
all models need a second state; stochastic driving, protocol variation or
nonmonotone readout can invalidate assumptions. Missing4still splits history.
Original-weight isotonic relaxation floor=.259414jointcost; weaker raw-phase
floors are.360555/.566689/.094723/.170684dec (bareS/R,seriesS/R). These are
necessary floors for that conditional class, not fitted prediction errors;
current dynamic failures remain much larger. Do not subtract old scatter.

New segmented_joint_problem.py has one shared material object, fixed external
Ron, q8=5log material coordinates+u_bare1+u_bare5+u_series4, full46pairs only.
Synthetic92phase/mask/history checks and10point amplitude identity passed;
no real46pair run or new calibration. Preserve15->16state in future holdout;
24after missing23 remains conditional. Next distinguish mean-dynamics failure
from cross-cycle variation with constrained physics; no per-cycle material
parameters or fitted random seeds, no claim that lowerRon solved the model.

## Latest bare-record identity correction (2026-11-03; user confirmed)

Read `analysis/BARE_HISTORY_RESULTS_2026-11-03.md` and
`analysis/bare_history_2026-11-03/` first. The segmented bare adapter now uses
one common law, two explicit segment-entry states (u1/u5), and unchanged5583
training residuals.9targeted wiring checks pass; it is not a newly fitted joint
model. Four existing candidates were repeated3times each without state resets:
SET thresholds change from3.025--3.175V to1.525--1.925V (target2.590--2.775),
features3/7->2/7->2/7; RESET failures persist. First cycles exactly reproduce
previous traces. Not returning to the originalu0 is NOT itself a failure.
The fixedRon93.4801ohm ceiling independently excludes some higher-conductance
bare readings (SET50mV7/14, max51.9%). Its fullRESET RMSE floor is only0.022dec,
so it does not explain the much larger dynamic errors. Ron was a training
median, not an independently established absolute conductance ceiling; no
newRon/shape was fitted. Never promote these diagnostics to model improvement.

The user explicitly confirmed that the 22 numeric labels in the two bare 1R
workbooks are repeated cycles of ONE physical device, not first-forming sweeps
of 22 independently prepared devices. This supersedes earlier identity-unknown
statements below. The user subsequently confirmed numeric chronological order and same-label
SET->RESET, and that cycles4/23 were actually measured but not supplied.
Known supplied segments are1--3,5--22,24; training segments remain1--3,5--15.
Missing-cycle waveforms/states and waits are unknown. Do not re-ask unavailable
scan times. A single explicit gap-entry latent state at5 is permitted for
conditional segment fitting, not an arbitrary state reset at every cycle.
Future heldout16 must inherit15;24 follows missing23 and is not an unconditional
holdout prediction if its initial state is freely set.
See `analysis/bare_history_2026-11-03/identity_confirmation.json` and the current
bare-history report before designing any further fit.

The one-prepared-bare-pair objectives and their frozen scores remain historical
conditional diagnostics, not validated repeated-device predictions. Preserve
their artifacts. Future actual-history models must carry material state between
known consecutive operations; do not introduce per-cycle initial states or
silently bridge unobserved operations. A no-wait repetition of the same command
is an explicitly conditional diagnostic, not a reconstruction of label order.
Keep original training/holdout IDs and feature gates; bare1--3 remain training.
Same-cell cycling does not prove a particular defect species, first-record
forming status, waiting protocol, or identical parameters to the series cell.

## Latest within-group diagnosis (2026-11-03; no improved model yet)

Read `analysis/PARAMETER_TRANSFER_RESULTS_2026-11-03.md` and
`analysis/parameter_transfer_2026-11-03/` first. Previous goal turns made
concrete progress; the overall accurate unified-model goal remains unachieved.
The selected-group objective preserves5583bare+21440series residuals, weights,
and state continuity;10synthetic checks pass, not a physical validation.
No separate optimization or actual bidirectional parameter transfer was run.
Four budget-derived bareA0/beta1 candidates were actually propagated as complete
SET/RESET pairs. All only3/7bare primary features pass; SET too late, RESET
peak too small/early and return leakage far too large. Same-mask branch errors
are worse than block GapLaw; do not promote them or mistake entry budgets for
successful trajectories. The first candidate RESET is blocked by a scalar
no-crossing state; raisingkR cannot fix it. Cached barrier check rejects2of5old
entries, and the other3were forwarded once each (no optimization).
Series-only early SET versus late SET-return budgets fail at19existinggc.
ContinuousJS stationary/breakpoint recheck leaves minratio2.385 in original
box (2.597 within old98%grid). Removing the inappropriate independent-series
JS<1mA cutoff still leaves minratio1.80787 (gc1eV,JS upperlimit12.027mA), even
without k algorithm bounds. This is fixed19gc/JS>=1e-7/conditional anchors,
not a proof over continuousgc or arbitrary initial/transport/clock choices.
The0.2V state anchor is not automatically the0V prepared state; main bound
allows extra path, while actual seed use needs inactive initial/tail checks.
Series restart lower and RESET infinite-upper labels were corrected with
pre_review snapshots retained; selectedSET results unchanged.
The beta>1 feedback-sign review is a derivation only. Its rough soft-stop
windows must also satisfy the ACTUAL2Vtarget state barrier, not just initial
drive activation. Preserve SET/RESET opposite feedback signs; do not choose
separate betas or boundary states. No newbeta core/ODE/fit has been implemented.

## Current conditional protocol and kinetic audit (2026-11-03)

Read `analysis/CONDITIONAL_PROTOCOL_RESULTS_2026-11-03.md` before the earlier
spectral stage. The user cannot obtain scan/dwell settings now; preserve this
unknown and do not re-ask or fabricate seconds. The full model goal is still
unachieved. Current diagnostics are not improved full-trajectory accuracy.
Systematic Fermi-window pair checking found NO violations in 1.1V training
records4--35 (1440 original retained pairs). Stronger general witnesses mostly
belong to some0.9V records; do not use them to claim the primary1.1V target
necessarily needs non-rigid transport or relaxation. Neither zero-field
relaxation alone nor ordinary passive WKB automatically resolves the previous
conditional constraints. Their reviews are in spectral_dynamic_2026-11-03/.
The external-clock adapter in conditional_clock_2026-11-03/ passes12 checks
and independent review, but no scenario has been dynamically fitted/evaluated.
Final budget_seeds has EMPTY selected_seeds: do not use revoked provisional
seeds from the loose7.95V path budget. Native outward compliance brackets are
about0.05V and must not be replaced with return-path closure.
The same-record beta1 diagnostic gives incompatible slow/jump rate bounds at
three fixed gamma_c values under declared5% endpoint bands. Its original
analytic all-threshold statement covers ONLY0<JS<Icc; JS>Icc is not forbidden
by the core law. A separately reviewed low-read stopping-bound extension covers
JS<=Icc*exp(uL) and still has minimum ratios3.564/5.701/7.805. Its common read
budget187.68ohm is a trainingQ90, not each record's actual read upper bound;
bare1/11 exceed it. Twelve conditional rate exclusions become10 directly
compatible with their native read states, with6/15 separate direction failures.
No cross-topology alpha is used there, but a constant within-record command-
path rate is assumed; adaptive dwell times would invalidate that comparison.
Do not generalize finite points/sensitivity bands to all material laws or
quietly promote5% diagnostic bands into the original feature acceptance gates.
The finite beta-feedback screen has completed57.713s,24predeclared cases
(beta.25/.5/.75/1, original3gamma_c, alpha20/50). SAME beta for both signs and
circuits, SAME alpha both signs, actual initial activation and low-read bound
preserved, including allowed JS>Icc. NO seeds: minimumSET required/allowed
ratio7.9491. No ODE/fit/holdout, no domain expansion; do not restart this screen.
RESET ratio0 denotes a loose infinite upper bound, not perfect prediction;
the final beta table separates unbounded cases from finite-interval statistics.
The parameter-transfer scope review is a proposal, not completed separate
calibrations: first establish within-group feasibility before attributing a
shared-parameter failure to different cells. A separate zero-extra-parameter
carrier-supply counterfactual (both signs multiply abs(I)/1mA, beta1,A0) has now
completed6predeclared cases in59.677s. New128-state-piece jump bounds preserve
the low-current onset; allSET budgets fail, best required/allowed ratio79.0723.
No ODE/fit/production implementation, no wider domain; do not restart this
screen. Infinite upper intervals remain mathematically allowed, not failures.
Source-identity check re-opened the public Zenodo preview and local workbook
metadata only: no hard-bare forming/cycling mapping was established, no holdout
numeric access. Do not equate public2c form (series1T1M) with local bare1R.
Missing file->figure/cell/stage/order mapping is distinct from unknown scan
settings; preserve both without inferring acquisition stage from thresholds.

## Latest continuous spectral trial (2026-11-03; evaluated, not promoted)

Read `analysis/SPECTRAL_DYNAMIC_RESULTS_2026-11-03.md` and
`analysis/spectral_dynamic_2026-11-03/` first. The active goal remains unachieved.
The shared spectral material and signed circuit/continuous-state backend now
exist and pass targeted validation; local static errors do not describe their
full dynamic accuracy. A bounded conditional common-path-rate initiation trial
has terminated on its declared wall budget (exit0): 20 new calls plus 10 charged
from the interrupted numerical-difference attempt, one start actually run,
second not started. Best recorded training cost1.70625585 versus GapLaw0.71314272;
no convergence claim. Do not restart it. Frozen full32 reevaluation is complete:
state/current/cost reproduce exactly; four raw phase RMSEs are1.261803,1.107448,
0.596821,0.967398dec, all worse than block GapLaw, and only7/20 primary gates pass
versus11/20. This bounded candidate is not promoted, and no holdout is evaluated.
The fitted block/accurate GapLaw training baselines are rescored on identical
masks; comparison only to the poor unfitted diagnostic is unacceptable.
The 47-point existing-spectrum screen is finished without expanding its domain:
26 initially evaluable,19 recovered by a separate bounded numerical recheck,
2 still numerical unknown (not physical failures). No sampled point passed all
selected necessary conditions. See analysis/spectral_shape_scope_2026-11-03/;
do not turn this finite screen into a global family impossibility claim.
Existing0.9/1.3V native-port caches now give additional timing-independent
within-branch state-direction constraints (not pristine holdouts). Preserve
startup-excluded/high-signal sensitivity in cross_gate_lowbias_robustness.md.
More strongly, FERMI_WINDOW_BOUND_REVIEW.md proves a bound for any nonnegative
frozen Landauer spectrum with fixed linear eta in[0,1], fixed300K and later
same-voltage transport not smaller during SET return. Gate0.9 record12, with
joint5% current/voltage deembedding allowance, requires ratio158.111 versus
the widest bound31.172. Changing frozen DOS or only rate parameters cannot fix
this conditional witness. Preserve quasistatic mapping and temperature limits;
it does not prove relaxation or reject arbitrary bias-dependent/thermal models.
Do not resume blind fixed-spectrum scans. Next distinguish return-state
relaxation from non-rigid bias-dependent transport while retaining other gates.

User says scan settings are currently unavailable. Preserve this unknown.
Same mathematical material model is the user requirement; common numeric
cell parameters, fixed selected spectrum and equal path rates are additional
test assumptions. Nominal stack equality does not prove identical defects.
The conditional scalar-state SET crossing bound needs roughly r_series/r_bare
>=20 under fixed mapping and declared endpoint budgets; it is not measured
timing or a no-go theorem for all unified physics. Common transistor-scale
sensitivity can invalidate its voltage-order premise without proving a fit;
see circuit_order_sensitivity.md. RESET initiation cannot repair this SET bound.
Do not fabricate times or claim nuisance scenarios are calibrated physics.
Restore the recorded user confirmations from ../PROJECT_HANDOFF.md:178--183:
reverse column4 is ID, independent/series T have the same type and dimensions,
and terminal/body-well connections match. Body potential and identical physical
cell/current-scale calibration were not established. Later generic "wiring
unconfirmed" statements and old converter metadata are stale; see
analysis/spectral_dynamic_2026-11-03/CIRCUIT_SOURCE_CONSTRAINT_AUDIT.md. Preserve frozen
numerical artifacts instead of silently rewriting their provenance snapshots.
Keep cycles1--3 excluded, one series initial state, and original failure records.
No desktop defaults changed and no new push was requested for this stage.

## Previous constrained screening (2026-11-03)

Read `analysis/CONSTRAINED_SHARED_RESULTS_2026-11-03.md` and
`analysis/constrained_shared_2026-11-03/` before continuing the active goal.
The goal is still unachieved; a static readout improvement is not a better full
shared switching model. Original GapLaw remains the complete-trajectory reference.
No desktop defaults changed or push requested for this stage.

Default-transistor Rh-increase evidence survives24 common interpolation/branch
choices; a hypothetical common0.789 current scale can remove the two-point
constraint but is NOT independently established. Independent1T interpolation
CV changes VG1.1 current by only about1.4--4.2%; no direct positive1.1V curve exists.

Actual-current cubic has a conditional reachability failure at the selected
frozen-envelope parameters: required state drop>=50.5056 versus<=2.96963 from
1mA compliance/frozen soft state. This is not an all-parameter no-go theorem.
Gaussian spectral readout passes independent numerics but saturates too early;
fixed-shape gamma inversion conflicts with monotone SET on32/32 return records.
Same-four-parameter lognormal readout reduces local autonomous SET-return
mean/raw RMSE to0.027945/0.051320 decade, but only12/16 displayed static features
pass, and no state kinetics or full bare1R/1T1R prediction exists. Do not compare
these partial-window mean scores with the full baseline scores. Its initial
Hermite failure is preserved; fixed-energy quadrature passes selected-point
independent validation, not arbitrary-gamma/Ohmic-limit validation. Lognormal
energy tails are assumptions, not measured DOS.
Lognormal residual mean-state inversions do not alone reject a useful model:
the strongest examined mean witness needs only0.23% expansion of training
Q10--Q90 current intervals (Gaussian witness18.8%);5->4V needs0.73%. These are
point-pair diagnostics, not whole-trajectory proof or relaxed main feature gates.
Next follow SPECTRAL_DYNAMIC_TRIAL_PLAN.md for a bounded single-state shared-G
trial; do not keep expanding static spectra in pursuit of zero error.

The new ordered objective scores original records with continuous history;
checked periodic shortcut must fall back to full32 when not validated. Final
parameters always require full32 comparison. Four workbook schemas/150sheets
have no acquisition-time columns; SET columnC contains two A/B resistance
formulas. Shared physical dz/dt does not imply shared voltage-path dz/ds when
scan rates differ. Unknown protocol rates cannot be invented to rescue fits.
Read DATA_OR_DERIVATION.md for prioritized provenance and measurement needs.

## Latest refinement status (2026-11-03)

Read `analysis/SHARED_GAP_REFINEMENT_RESULTS_2026-11-03.md` and the corresponding
`shared_gap_refinement_2026-11-03/` artifacts before continuing. The user requested
both an accurate unified explanation and lower error. Neither is complete.
Same GapLaw material parameters/state dynamics remain shared across bare1R and
1T1R; accurate adaptive recalibration and RESET-block polishing are comparison
baselines, not validated microscopic replacements. Optimizer stopping flags do
not imply stationarity: accurate fit ends on xtol; block fit reaches its 70-nfev
limit. Read the finite-difference audit before interpreting Jacobian singular values.

Training-only analysis proves a conditional original-family constraint: with
monotone decreasing SET u and the original transport law, Rh cannot increase
as I increases, yet all32 training series SET curves show Rh rising from command
2V to4V after fixed-transistor deembedding. Mapping applicability and quasi-static
voltage interpretation are assumptions. Do not claim all shared models excluded.
`shared_access_gap.py` adds one shared cubic voltage term with unchanged drive;
the tested fitted trajectory still gives Rh4/Rh2 approximately0.497. Its microscopic
origin is unverified. Nonzero-B fits, a bounded common-initial-state profile,
and B0 no-refit ablation are preserved; do not silently promote extra complexity.

`shared_gap_ivp.py` checks current refinement but current compliance can hide
state error. Independent final audits compare current AND whole state/endpoints.
Some tighter Radau/ordinary reference configurations fail at the absorbing bound;
the B0 fast layer needed exact inverse-coordinate integration ds/du=1/G, with two
independent integration methods. Keep failures and delimit verified parameters.
The fitting two-pair training surrogate is not the exact ordered32-cycle loss;
`objective_scope_audit.json` quantifies this. All final scores retain raw scatter,
use continuous cycle4→53 without a split reset and label holdouts internal only.
Bare50mV read resistance is not a measurement of true zero-bias Ohmic resistance;
the unfitted low-bias predictions offer a discriminating follow-up, not validation.
No desktop defaults changed and no new push was requested for this refinement.

## Current scope (2026-11-03; user corrected the device pair)

The user subsequently required actual joint modelling. Four original workbooks
have now been recovered from Git HEAD without overwriting their cloud placeholders;
use `analysis/shared_hbn_2026-11-03/data/native_observations.npz` and its inventory.
The 96-ohm bare reference is supported by these observations. Current experiment
is `vulcan2d/shared_gap.py`, one implicit material law and one continuous state
ODE, shared across bare and transistor circuits. It is a jointly fitted candidate,
not a validated microscopic explanation or a replacement for desktop defaults.
Read `analysis/SHARED_GAP_JOINT_PLAN_2026-11-03.md` and
`analysis/SHARED_GAP_JOINT_RESULTS_2026-11-03.md` before
continuing. Baseline `shared_hbn.py` is retained as the previous limited reference.
Use one cycle4 initial state through cycle53, including the cycle35/36 split;
never reset state for scoring. Bare records have only conditional same-label pairing.
No fit score alone validates the effective-gap coordinate, physical time, or species.
Preserve failed attempts, source hashes and independent integration checks.

The user explicitly corrected the target: unify **1R and 1T1R**, NOT standalone
1T and 1T1R. The previous transistor consistency repair remains valid local work
but does not solve this request. The actual h-BN constitutive and state laws
currently differ: legacy bare1R hard-switch/clipped-current model versus the
v12/v13 soft-envelope protocol model. Current task is a shared material kernel
I_h=F(V_h,z;theta), dz/ds=G(V_h,z;theta) with only external circuit/protocol
adapters differing. No topology or Vg input may choose a different material law.
Instrument compliance must reduce actual device voltage before state evolution,
not merely clip reported current. Read
`analysis/SHARED_HBN_STATUS_2026-11-03.md` for the scoped candidate and limitations.
The new candidate is not jointly calibrated, does not replace v13 defaults, and
must not be advertised as having validated soft/hard mechanism selection.
Nature evidence now confirms the SAME NOMINAL Au–Ti–h-BN–W stack and geometry
for the paper's bare1R / Fig.2c 1T1M comparison: ~6 nm h-BN, ~260 nm W via,
area at most ~0.053 um², 3 nm Ti + 40 nm Au. These are different cells, not
paired measurements of one physical device. See
`analysis/NATURE_1R_1T1R_PROVENANCE_2026-11-03.md`. Exact local workbook/source
figure mapping and forming versus repeat-switching history still need checking;
SI Fig.6 reports 1 kohm LRS, not the legacy 96 ohm reference. The reported
~70 us / 300 pF versus transistor / ~50 pF circuit difference motivates a
specific transient check, but the present path model does not simulate it.
Do not invent paired bare1R
SET/RESET records or physical dwell times. Continue excluding startup cycles1--3
from primary normal-operation evaluation, preserving raw/history records.

## Previous scope and status (2026-10-29; transistor repair only)

The user explicitly authorized targeted repair of the inconsistent 1T / 1T1R
interpretation. This is not authorization to resume unconstrained mechanism
searches. Read `analysis/UNIFIED_TRANSISTOR_REPAIR_2026-10-29.md` first.

Advisor correction supplied by the user on 2026-09-22: cycles 1--3 may have
failed SET or self-recovered; the advisor said to ignore them for the intended
simulation. Preserve their raw records, exclude them from primary normal-cycle
fitting/scoring, and do not use the cycle3-to4 jump as the required innovation.
Those two explanations are alternatives, not an identified microscopic cause.
This supersedes first-three-cycle requirements in ALL historical notes below.
Excluding scores does not authorize arbitrary resets in continuous-state models.

Research v0.13.1 now shares one measured fixed-physical-source transistor law
between 1T and both 1T1R phases. Independent branch overrides must agree on gate
reference, current scale and access resistance. Use `measured_transistor_mapping`
and `simulate_calibrated_cycle(..., transistor_mapping=mapping)`. A common KCL
solver serves SET/RESET. Legacy desktop defaults remain the historical model;
do not identify its curves/patch picture with the research API.

The default research curves were already consistent with their respective 1T
lookups. This repair prevents inconsistent overrides and fixes legacy measured
mapping polarity handling; it does not improve fitted accuracy or establish a
microscopic MOS/h-BN mechanism. The earlier handoff records user confirmation of
ID, matching type/dimensions and terminal/body-well wiring; body potential,
same-physical-cell identity and numerical scale remain distinct unresolved items.
SET/RESET h-BN envelope-state unification
also remains unresolved. The new cycles4--53 audit uses unchanged historical
calibration and archived native observations: descriptive scoring, not a refit
or a newly independent holdout. Historical results must retain their old scope.

## Historical scope and status (2026-09-21; superseded where noted above)

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
