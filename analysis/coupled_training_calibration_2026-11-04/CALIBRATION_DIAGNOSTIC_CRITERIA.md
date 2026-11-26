# Minimum post-calibration evidence for the q²+a shutdown problem

This is a read-only interpretation plan for the already running four-parameter calibration. It introduces no ODE, fit, inverse/current evaluation, model family or frozen-file change. Do not declare the family unsuccessful before the actual result and its numerical validation are available.

## 1. The minimum result bundle

Use only the selected candidate's saved complete histories and final numerical checks:

1. The unchanged 27023-residual objective, branch errors, all 20 feature outcomes, optimizer exit/budget status, parameter-bound hits, and nominal/refined disagreement. An incomplete history or unconverged numerical result is not a physical score. A bounded/local optimization outcome is not a global parameter-family result.
2. For each of the 14 bare RESET records, retain its **actual candidate-specific** entrance `(q0,a0)` and the saved return-0.2-V `(q,a,Vh,Isoft,Ihard,Itotal,Iobserved)`. Use the existing native mask/index; do not create a new interpolated observation.
3. At that read point, compute the two conservative shutdown bounds below and compare them with the saved actual states. Inspect native capture/recovery/formation/loss terms only where needed to explain a large gap from the bounds. These rate components can be obtained by algebra from saved q/a/Vh/Itotal and the selected parameters, without a new circuit or ODE.

The a-disabled and initial enabled runs are comparisons, not substitutes for the selected candidate's RESET entries. The three *segment* entrances are fixed in this calibration; the later RESET entrances nevertheless depend on A/kappa_a/rho/b through preceding phases.

## 2. Candidate-specific q recovery budget

For negative actual voltage and nonnegative capture,

\[
q'\ge-\kappa_+e^{2b}v e^{\lambda v}q^2,
\qquad
q_r\ge q_{\min}=\left[q_0^{-1}+\kappa_+e^{2b}D_q\right]^{-1}.
\]

For bare RESET, `v=|Vcommand|`; on 0→Vmax→vr,

\[
D_q=2B(V_{\max})-B(v_r),\qquad
B(V)=\frac{e^{\lambda V}(\lambda V-1)+1}{\lambda^2}
\]

(B(V)=V²/2 at lambda=0). Use actual native endpoints rather than rounding the read voltage. Passive series voltage division permits the same command-based expression as a conservative upper bound on the integral, but a different waveform must use its own path.

**Required inputs:** selected b, fixed lambda/kappa_positive, and this candidate's actual q0 for each RESET. The earlier constant 3649.82022 applies only to the former b=0.769297378470313 and the approximately 0.2 V bare return path; it must not be reused after b changes.

Report q_r/q_min. If it is near one, the trajectory approaches the maximal recovery permitted when capture is removed. If it is large, the saved trajectory is far from this recovery envelope; capture/history can be the dominant obstruction instead of a shortage of recovery budget alone. This comparison is descriptive and requires numerical convergence; no arbitrary ratio is a physical pass threshold.

The bound is on **state**. The existing Isoft is evaluated at saved q_r, not at q_min. Without an additional declared readout calculation or independently established readout bound, do not label that Isoft as a current floor at q_min, and do not infer infeasibility from this state bound alone.

## 3. A directly testable lower bound from the parallel path

The same negative-polarity a law gives, without assuming any q history,

\[
a'=\kappa_av[e^{-b}q(A-a)-\rho e^b(1-q)a]
\ge-\kappa_a\rho e^b v a.
\]

Therefore

\[
\boxed{a_r\ge a_{\min}=a_0\exp[-\kappa_a\rho e^bD_v]},\qquad
\boxed{|I_r|\ge v_r a_{\min}},
\quad D_v=\int v\,ds.
\]

For bare 0→Vmax→vr, D_v=Vmax²-vr²/2; with 2.5 V and 0.2 V it is 6.23 V². For a passive load, replacing v by command magnitude gives a conservative bound. This is a total-current lower bound because the soft and parallel branch magnitudes are nonnegative and cannot cancel.

**Required inputs:** this candidate's actual a0, kappa_a/rho/b, actual read voltage and native path. Store the logarithmic form `ln(a_min)=ln(a0)-kappa_a*rho*exp(b)*D_v` to avoid underflow. If a0=0, this particular bound is zero and gives no obstruction; a positive saved a_r then comes from formation during the phase.

For an explicitly declared per-record target I_target>0, a necessary condition for reaching it is

\[
\kappa_a\rho e^b D_v\ge\ln(v_r a_0/I_{\rm target})
\]

when the logarithm is positive. If this fails, that fixed candidate starting from its actual entrance could not reach the target even if formation were removed entirely. This is a rigorous **candidate/entrance-conditional budget conflict**, not a proof against every possible A/kappa_a/rho/b, since those parameters also change a0.

Using a single observed current as I_target tests exact reproduction of that datum, not a certified measurement-error interval. The original low-tail feature gate is a *median across records*: if `median(v_r*a_min)` exceeds its original upper gate, the candidate necessarily fails that aggregate gate. A failed per-record exact target alone does not establish failure of the median gate.

Report the saved a_r relative to a_min in log form. A lower bound already above the target indicates inadequate available loss at the selected rates/entrance. A very small lower bound but a large actual a_r indicates continuing formation or history feedback; it does not support calling the loss coefficient alone too small.

## 4. Current decomposition and the shortest useful flux check

At a failed native read point, saved `Isoft`, `Ihard=a*Vh` and their sum directly identify which branch contributes the error. If Isoft alone exceeds the observation, subtracting only the parallel current at the **same state** cannot fix that point. Changing A/kappa_a/rho/b may also change q, so this is not a counterfactual prediction of a separately simulated a-disabled model.

If the state lies well above its conservative recovery envelope, inspect the saved-point signs and magnitudes of:

- `capture_q - recovery_q`, using the same saved total current;
- `formation_a - loss_a`, using the same saved q/a/local field.

Pointwise ratios diagnose whether the trajectory is still being replenished or is already losing state. They do not measure integrated causal contributions. In particular, **do not integrate a sorted raw RHS trace**: it can contain rejected steps, multiple trial states and repeated coordinates. Native-point trapezoidal integration would be an approximation requiring a resolution check, not a strict bound. The closed-form bounds above avoid that extra diagnostic loop.

## 5. What conclusions are and are not supported

| Evidence after the fit | Supported conclusion |
|---|---|
| Improved converged full histories and feature outcomes | The former seed was inadequate; the shared effective model has improved within this fixed calibration scope. |
| Optimizer evaluation cap, boundary hit or numerical failure | This finite search is incomplete or constrained; no parameter-family rejection follows. |
| Candidate-specific hard-path lower bound exceeds a declared target | Even removing formation cannot shut that candidate down fast enough from its actual entry; the conditional rate/history budget is insufficient. |
| Actual states far above permissive recovery bounds and native sources remain substantial | Coupled replenishment/formation is a plausible obstruction requiring interpretation of the fitted trajectory; the conservative loss budget itself has not excluded the target. |
| Persistent current errors but all conservative bounds are permissive | The current fit remains inadequate; these diagnostics do not distinguish local optimization failure from an insufficient state-law structure. |

A family-wide necessary-condition exclusion would require bounds on **all allowed parameter values and all corresponding reachable RESET entrances**. Holding a baseline entrance fixed while maximizing b or kappa_a/rho is not such a proof. No such reachability enclosure is established by this run.

The existing original-spectrum Jacobian shows an unstable ideal q=a=0 limit for active q²+a, but this point lies outside the finite table and does not rule out finite low-state equilibria or finite-path HRS. Keep it as a structural warning, not a veto on the present fit. No q¹ or other family is introduced by this document.
