# Coupled q/a kernel: rate normalization and zero-field review

Prepared by read-only source inspection and algebra. No observation inversions, ODEs, parameter searches, circuit roots or new spectral evaluations were performed for this review. Old stages are unchanged.

## 1. Implement these equations with one actual circuit point

Let `s = integral |dV_command|` (path volts, not measured seconds),
`q = exp(-u) = Gamma/(Gamma + Gamma_c)`, `v = abs(V_hBN)`,
`sigma = sign(V_hBN)`, `chi = (1-sigma)b`, and

\[
I=F_{\rm soft}(V_h,\Gamma(q))+aV_h,\qquad i=|I|.
\]

The current driving the q law **must be this total current**, solved in the same KCL/KVL calculation. Neither the soft current alone nor a command-voltage estimate is interchangeable with it.

The existing positive-normalized q² law is

\[
\boxed{\frac{dq}{ds}=\kappa_+v\left[
 \frac{i}{J_+}e^{-\chi}(1-q)-e^{\chi+\lambda v}q^2\right].}
\]

Its exactly equivalent u law is

\[
\boxed{\frac{du}{ds}=\kappa_+v\left[
 e^{\chi+\lambda v-u}-\frac{i}{J_+}e^{-\chi}\operatorname{expm1}(u)\right].}
\]

Thus positive voltage has no b shift; negative voltage has recovery factor `exp(2b)` and capture factor `exp(-2b)`. This agrees with `signed_forward_kernel.py::signed_rate` and the current `joint_forward.py` parameterization. The older symmetric q notation is equivalent only after `kappa_q = kappa_positive exp(b)` and `J_0 = J_positive exp(2b)`; do not mix these conventions.

Preserve the declared second-state candidate, with its **separately named symmetric coefficient**:

\[
\boxed{\frac{da}{ds}=\kappa_a v\left[
 e^{\sigma b}q(A-a)-\rho e^{-\sigma b}(1-q)a\right],\quad A=a_{\max}.}
\]

Sharing b between these laws is an explicit structural hypothesis. It does not require identifying the independently normalized kappa_a with kappa_positive. For internal `x=a/A`, A>0,

\[
\frac{dx}{ds}=\kappa_a v\left[e^{\sigma b}q(1-x)-\rho e^{-\sigma b}(1-q)x\right].
\]

At exactly V_h=0, a passive readout gives I=0 and both rates equal zero. No sign convention is needed there. At nonzero fields, sigma comes from the solved local voltage, including return sweeps; it is not a phase/file label.

Recommended shared parameter names: `lambda_per_v`, `J_positive_a`, `kappa_positive_per_v2`, `b`, `a_max_s`, `kappa_a_per_v2`, `rho`, `enable_a`. Use one material object and one rate function for bare 1R and 1T1R; only the external circuit changes. Bare compliance must solve the reduced actual material voltage before evaluating rates.

## 2. State domain and what “cooperative” does not establish

Require finite lambda>=0, J_positive>0, kappa_positive>=0, kappa_a>=0, rho>=0 and A>=0; keep the declared b domain rather than silently enlarging it.

- At q=0 the q derivative is nonnegative; at q=1 it is nonpositive. At a=0 the a derivative is nonnegative; at a=A it is nonpositive. The ideal physical rectangle `[0,1] x [0,A]` is invariant for continuous passive currents.
- This does **not** prove invariance of the narrower numerical Gamma table. The frozen table spans approximately Gamma=`1e-24 ... 1e6` eV, so admissible q lies between `Gamma_min/(Gamma_min+Gamma_c)` and `Gamma_max/(Gamma_max+Gamma_c)`. Preserve explicit finite-state/table checks, retain trial-domain failures, and do not clip or re-anchor states.
- If A=0, handle the disabled-a limit explicitly; do not form x=a/A. With active a, a=0 is generally not invariant: its derivative is positive when kappa_a, v, q and A are positive. An initially zero a is not a disabled path.
- At fixed local voltage, assuming soft current increases with q, the (q,a) off-diagonal derivatives are nonnegative: `partial_q a_rate = kappa_a v[exp(sigma b)(A-a)+rho exp(-sigma b)a] >= 0` and `partial_a q_rate = kappa_positive v^2 exp(-chi)(1-q)/J_positive >= 0`.
- That fixed-voltage result does not automatically survive a series load: increasing a changes both current and local voltage. For example, with total material current M(v,q,a), transistor current T(V_command-v), and D=M_v+T'>0, `v_a=-v/D`. Consequently

\[
\partial_a q'=\frac{\kappa_+v}{D}\left[
 \frac{e^{-\chi}(1-q)}{J_+}(vT'-i)
 + e^{\chi+\lambda v}q^2(1+\lambda v)\right],
\]

which need not be positive in a saturated transistor regime. Do not apply a global cooperative-system/ordering theorem to the coupled loaded circuit without checking its hypotheses. The q² exponent also remains an effective hypothesis, not evidence of a microscopic two-particle mechanism.

## 3. Near zero: integrate a declared extension instead of freezing

The frozen material table starts at delta=about `1e-9 V`; it does not provide nonzero values below that. A practical extension for each sign is

\[
c_\sigma(q)=\frac{|F_{\rm table}(\sigma\delta,\Gamma(q))|}{\delta},\qquad
I_{\rm soft}(V_h,q)=c_\sigma(q)V_h\quad(0<|V_h|<\delta).
\]

Use the correctly rescaled frozen Ron before forming c_sigma. This extension is positive, connects exactly to the table at each signed endpoint, and connects continuously to I=0 at zero. Adding the parallel path gives

\[
I=(c_\sigma+a)V_h,\qquad I V_h=(c_\sigma+a)V_h^2\ge0.
\]

It is strictly monotone in local voltage on each subtable interval for c_sigma+a>0. Against a continuous nondecreasing transistor branch, the local circuit residual has positive derivative `c_sigma+a+T'`, so a sign-bracketed in-domain root is unique. This does not authorize extrapolating the transistor lookup or skipping the root-bracket check.

Solve the same total-current circuit throughout the microsegment and evaluate both rates. For finite admissible states, i=O(v), q'=O(v), a'=O(v), so both rates approach zero continuously; polarity-dependent one-sided slopes are allowed. The two finite-delta slopes c_+ and c_- need not be identical. This is a **numerical endpoint extension**, not a proof of the original spectrum's exact zero-field tangent or an error bound.

Do not reuse the archived soft-only near-zero omission bound: adding a changes the total-current bound and the capture term, especially in the u coordinate where it contains `i*expm1(u)`. The new kernel should not omit this integration or claim its omission is covered by the old a=0 certificate.

## 4. Existing independent small-field reference and useful sensitivity

The original spectral law has the common two-sided zero-field conductance

\[
G_0(\Gamma)=\frac{\Gamma}{2\pi k_BT R_{\rm on}}
\left\langle\Re\,\psi_1\!\left(\frac12+\frac{\Gamma+iE}{2\pi k_BT}\right)\right\rangle_E,
\qquad 0<G_0\le 1/R_{\rm on}.
\]

Equivalently it is the positive Lorentzian transmission averaged against `-f'(epsilon)` and the energy distribution. The zero-field derivative is independent of eta, although higher voltage terms need not be odd in voltage. The current source code does not expose a complex-trigamma interface, so do not assume ordinary real `polygamma` supports this formula directly.

There is already a usable stable reference in `vulcan2d/shared_spectral.py::SpectralShape.current_at_log_gamma`: for `|V|<1e-5 V`, it obtains g0 and higher coefficients from direct spectrum values at `+/-1e-4` and `+/-5e-5 V` using Richardson combinations, then evaluates a fourth-order small-voltage polynomial. The broad-Gamma branch has its separate moment/remainder checks. The default spectral domain is `|V|<=5 V`, finite ln(Gamma)>=-700 or +infinity; the frozen table interface remains stricter. Construct the reference with Ron=58.463033350725695 ohm, or apply the exact reference-Ron scaling consistently.

For a separately declared finite static validation, reuse a few predetermined in-table Gamma states and both signs at delta/2, delta and 2delta, comparing the table extension to this spectral interface. Report this as a numerical reference check, not an experimental uncertainty certificate. No such evaluations were performed here.

For an extension sensitivity check, anchoring a second line at **2delta** uses a different in-table voltage and is informative. Evaluating the existing delta-anchored line at delta/2 and re-anchoring that same line is algebraically identical and provides no independent check. Neither a new state inversion nor a free parameter scan is needed for these checks.

## 5. Required disabled-a and coupling checks before interpretation

1. `enable_a=False` must require a0=0 and keep a exactly zero in readout and dynamics. At identical actual circuit points, verify `u'=-q'/q` and agreement with the frozen signed a=0 rate for both signs and arbitrary declared b.
2. Compare the new a-disabled circuit to the frozen circuit at matched states and voltages where the old table/root is valid. Preserve KCL, KVL, current sign, Ron scaling and actual-local-voltage polarity. The new small-field extension is the only intended low-voltage readout difference.
3. Before an experimental trial, compare one predeclared a-disabled continuous trajectory with its frozen counterpart using the same initial state, parameters and waveform. The old kernel omitted subtable evolution; therefore report that deliberate difference separately from numerical tolerance rather than demanding bitwise identity or silently changing the baseline.
4. A coupled smoke test must show the same a entering the readout, KCL and both rates; record soft/parallel/total currents separately, all phase terminal (q,a), and exact state handoff. State-domain or integration failures remain numerical unresolved outcomes. No state is inferred anew at a phase boundary.

Source basis: `analysis/joint_history_calibration_2026-11-04/joint_forward.py`, `analysis/signed_soft_history_2026-11-04/signed_forward_kernel.py`, `analysis/cooperative_relaxation_2026-11-04/SIGNED_EXTENSION_READINESS.md`, `analysis/zero_crossing_compatibility_2026-11-04/minimal_static_loader.py`, `vulcan2d/shared_spectral.py`, and the read-only metadata of `/private/tmp/vulcan-runtime-cache/transport_g010_v0025.npz`.
