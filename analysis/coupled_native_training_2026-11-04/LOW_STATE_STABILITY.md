# Low-state stability and q² RESET recovery bound

Algebraic review only; no kernel changes, ODEs, state inversions, parameter scans or new model family runs. This does not delay or replace the fixed v2 full-training comparison.

## 1. Linearization at the ideal insulating limit

Use q,a coordinates, fixed local magnitude v>0, polarity sigma, chi=(1-sigma)b, and the implemented total-current drive. Extend the original positive spectrum mathematically to Gamma→0, with Gamma=Gamma_c q/(1-q). Write

\[
|I_{\rm soft}|=C_\sigma(v)q+o(q),\qquad C_\sigma(v)\ge0,
\quad |I|=C_\sigma(v)q+va+o(q,a).
\]

The nonnegative coefficient follows from the narrow-Lorentzian limit of the original positive spectral integral. This is **a limit of the original spectrum**, not an evaluation at q=0 in the finite Gamma table. The implemented table excludes that endpoint.

Define

\[
\alpha=\frac{\kappa_+v e^{-\chi}C_\sigma(v)}{J_+},\quad
\beta=\frac{\kappa_+v^2e^{-\chi}}{J_+},\quad
\gamma=\kappa_av e^{\sigma b}A,\quad
\delta=\kappa_av\rho e^{-\sigma b}.
\]

For q² recovery, the exact limiting Jacobian is

\[
M_2=\begin{pmatrix}\alpha&\beta\\\gamma&-\delta\end{pmatrix},
\qquad \det M_2=-\alpha\delta-\beta\gamma<0
\]

when v,kappa_positive,kappa_a,A,rho,J_positive are strictly positive. Hence the origin is a saddle, with

\[
\mu_+=\tfrac12\left[\alpha-\delta+
\sqrt{(\alpha+\delta)^2+4\beta\gamma}\right]>0.
\]

The q² recovery contributes no linear restoring term. Varying lambda or any finite positive rho cannot make this ideal origin attracting while the active cross-coupling remains. Even with a disabled, the scalar q² origin is unstable when C_sigma(v)>0; enabling a increases the growing mode.

This is not a proof that every finite low-conductance equilibrium or finite-duration HRS trajectory is impossible. The exact q=a=0 solution remains absorbing; actual initial states are positive and finite. At small v, C_sigma(v)=O(v), and with rho>0 the growing eigenvalue is O(v²); at exactly zero voltage both implemented rates vanish. Growth is per command-path volt, not per measured second.

For a series circuit, the same first-order matrix applies **conditionally** when the relevant signed transistor branch has a finite, strictly positive origin slope T'(0), and positive instrument compliance is inactive near zero current. Then V_h=V_command+O(q,a), and its state perturbation multiplies rate terms already of first order, affecting only higher orders. No first-slope value was computed here. Zero-slope, threshold/dead-zone, singular or zero-current-plateau loads require a separate analysis; this result is not asserted for every load.

## 2. Contrast with the original q¹ skeleton

Keeping all other factors the same, q¹ recovery subtracts

\[
E=\kappa_+v e^{\chi+\lambda v}>0
\]

from the top-left entry:

\[
M_1=\begin{pmatrix}\alpha-E&\beta\\\gamma&-\delta\end{pmatrix},\quad
\operatorname{tr}M_1=\alpha-E-\delta,\quad
\det M_1=\delta(E-\alpha)-\beta\gamma.
\]

Local asymptotic stability requires **trace<0 and determinant>0**. For delta>0 these reduce to

\[
E>\alpha+\frac{\beta\gamma}{\delta}
\quad\Longleftrightarrow\quad
J_+e^{2\chi+\lambda v}>C_\sigma(v)+\frac{Av e^{2\sigma b}}{\rho}.
\]

Thus q¹ permits an attracting insulating limit under an explicit condition; it is not automatically stable. Equality is nonhyperbolic and needs higher-order analysis. This comparison is structural reasoning, not permission to change the frozen q² kernel or evidence that q¹ fits the full data better.

## 3. An independent finite-path q² RESET bound

On a negative-polarity path, total-current capture is nonnegative for 0<q<=1, so

\[
q'\ge-\kappa_+e^{2b}v e^{\lambda v}q^2,
\qquad
\boxed{q_{\rm end}\ge
\frac{1}{q_{\rm start}^{-1}+\kappa_+e^{2b}\int v e^{\lambda v}\,ds}}.
\]

Equivalently, u_end<=ln(exp(u_start)+kappa_positive exp(2b) integral). The starting state must be the actual incoming RESET state of the continuous trajectory, not a new fitted anchor. This bound uses the full nonnegative capture term and remains valid with a active.

For passive voltage division, v<=|V_command|. With lambda>=0 the function v exp(lambda v) is increasing, so replacing local voltage by command magnitude yields a conservative bound. Bare RESET has no 1 mA compliance here, hence its actual material voltage equals the command. For the full native 0→2.5→0 path,

\[
\int v e^{\lambda v}ds=2B(2.5),\qquad
B(V)=\begin{cases}[e^{\lambda V}(\lambda V-1)+1]/\lambda^2,&\lambda>0,\\V^2/2,&\lambda=0.\end{cases}
\]

At the return 0.2 V read point use `2B(2.5)-B(0.2)` instead. With the frozen scalar parameters, these are:

| Endpoint | Integral / V² | Increment in denominator 1/q |
|---|---:|---:|
| End at zero | 13.60618661484256 | 3655.5248275684094 |
| Return 0.2 V | 13.584953568608423 | 3649.8202220186718 |

No incoming state was assumed and no current was evaluated. In particular, the zero-voltage endpoint has zero current regardless of q; this bound alone cannot give a nonzero tail-current floor. A later saved-trajectory check can use the actual q_start and the 0.2 V bound, then the monotone soft readout and a>=0 to test a current lower bound within its validated domain.

The finite-path bound and the origin Jacobian are distinct statements. They identify possible reasons for slow recovery and premature activation, but neither substitutes for the completed native-data comparison or establishes a microscopic mechanism.
