# Shared 1R / 1T1R material model

This research candidate uses **one material law, one evolving state and one
material parameter object** for bare and transistor-connected h-BN devices.
It is jointly calibrated against recovered observations; remaining curve
errors prevent claiming a validated physical explanation. Desktop defaults
and the older v13 calibration remain historical baselines.

The subsequent refinement uses `shared_gap_ivp.py` for adaptive integration.
It also tests the nested `shared_access_gap.py` transport candidate described
below. Fitting outputs, rejected trials, training-only diagnostics and
independent checks are retained in
`analysis/shared_gap_refinement_2026-11-03/`; they do not overwrite the original
joint benchmark. See `analysis/SHARED_GAP_REFINEMENT_PLAN_2026-11-03.md` for the
comparison rules. Internal test records have historical development exposure
and must not be described as a new blind validation set.
The completed comparisons and their scientific limits are in
`analysis/SHARED_GAP_REFINEMENT_RESULTS_2026-11-03.md`. No tested candidate is
designated an accurate, independently validated physical explanation.

## Equations and circuit separation

The effective bottleneck coordinate `u` is dimensionless, bounded by 0 and 30.
It is not a measured atomic gap in nanometres. Signed material voltage obeys

\[
V_h=R_{on} I+V_0u\operatorname{asinh}(I e^u/I_0).
\]

At `u=0` this is exactly Ohmic; for positive `u` it is nonlinear and passive.
The same equation applies for either polarity and topology. The reduced drive is

\[
e=V_0\operatorname{asinh}(|I|e^u/I_0).
\]

Positive current reduces `u`, while negative current increases it:

\[
\frac{du}{ds}=\begin{cases}
-k_s\sinh([e-e_s]_+/e_0),& I>0,\\
+k_r\sinh([e-e_r-e_i e^{-u}]_+/e_0),& I<0.
\end{cases}
\]

Bounds prohibit outward motion. The additional RESET initiation drive `e_i`
is a phenomenological, state-dependent threshold; its units are V, not energy.
The exponential form and its unit decay scale are hypotheses. Setting `e_i=0`
recovers the nested baseline. No topology, gate or observed current selects a
different material law.

`s` is cumulative commanded-voltage variation in V. It is not physical time.
Repeated voltage entries have no additional exposure; this model cannot predict
dwells, relaxation, pulse timing, capacitive overshoot or a 70-microsecond response.
Both return sweeps continue the same dynamics. SET and RESET share state without
empirical amplitude conversion or manual reinitialization.

### Evidence-driven transport comparison

Under the fixed measured transistor mapping, all 32 training SET curves have
increasing deembedded `Vh/I` from command 2 V to 4 V while current increases.
The original transport law has decreasing resistance with current at fixed
`u`, and increasing resistance with `u`. Combined with SET-only decreasing
`u`, it cannot exactly reproduce that trend for any positive transport
parameters. This conclusion is conditional on the transistor mapping and
quasi-static voltage interpretation; it does not identify the missing physics.
The proof, raw-cycle results and limitations are in
`analysis/shared_gap_refinement_2026-11-03/STATE_DIAGNOSTICS.md`.

`AccessGapLaw` tests one common additional voltage term:

\[
V_h=R_{on}I+uV_0\operatorname{asinh}(Ie^u/I_0)
    +uB(Ie^u/I_{ref})^3,\qquad I_{ref}=1\,\mathrm{mA},\ B\ge0.
\]

The cubic term permits resistance to rise with current at fixed state. It
vanishes at `u=0` and at `B=0`; the latter recovers the original law exactly.
There is still one state and one material parameter object across both
circuits. The migration drive and state equation are inherited unchanged:
the additional series voltage is not silently counted as gap field.
The cubic form, `exp(3u)` dependence and disappearance at `u=0` are effective
functional assumptions, not independently verified heating/contact/atomic
mechanisms. Passivity and KCL consistency alone do not establish a complete
thermodynamic model. A fitted coefficient is not a microscopic measurement.

### Accurate path integration

`simulate_gap_ivp` has the same current/state fields as `simulate_gap` and
uses adaptive integration on monotone command-voltage segments. It enforces
the existing current-cap stopping invariant, without changing the material
ODE. Fivefold tighter state tolerances check returned currents; the tighter
trajectory is returned. This is an a posteriori consistency estimate, not a
rigorous global error guarantee. End-state and next-cycle checks with an
independently implemented integrator remain necessary, especially during
current compliance when current alone can hide a state error.

Calibration uses training log-current sufficient statistics with original
per-record eligibility. The two continuous 1T1R pairs used during optimization
are a declared repeated-curve surrogate, not the exact 32-cycle chronological
likelihood. Final scores must include record-to-record scatter and propagate
cycle4 through cycle53 continuously. Scatter is not automatically measurement
noise or an irreducible lower bound for a history-dependent model.

Bare SET uses ideal instrument compliance of 1 mA, solved by reducing actual
device voltage. Bare RESET is not capped at 1 mA. 1T1R uses the common measured
signed transistor port model at VG=1.1 V. Gate voltage enters only that circuit.
The negative transistor lookup is restricted to |VDS|<=2 V; extrapolation fails
explicitly. The provenance of positive and negative transistor measurements
still does not establish that they came from the same physical transistor.

## Use the same fitted law for both connections

Run from the repository root after the joint fit exists:

```python
import json
from pathlib import Path
import numpy as np
from vulcan2d.shared_gap import GapLaw, BareGapCircuit, LookupGapCircuit, simulate_gap
from vulcan2d.transistor import measured_transistor_mapping

fit = json.loads(Path('analysis/shared_hbn_2026-11-03/gap_joint_fit.json').read_text())
law = GapLaw(**fit['selected_law'])
q = fit['fits'][fit['selected_start']]['q']
t = LookupGapCircuit(measured_transistor_mapping(), 1.1)

def full_pair(v_set, v_reset, initial_u, set_circuit, reset_circuit):
    set_result = simulate_gap(v_set, initial_u, law, set_circuit, max_step_v=.001)
    reset_result = simulate_gap(v_reset, set_result['final_u'], law, reset_circuit,
                                max_step_v=.001, initial_command_voltage_v=v_set[-1])
    return set_result, reset_result

# v_set/v_reset are supplied command waveforms, never measured currents.
# The two initial values reflect different unknown preparation histories.
# r1 = full_pair(v_set_1r, v_reset_1r, q[7], BareGapCircuit(1e-3), BareGapCircuit())
# tr = full_pair(v_set_1t1r, v_reset_1t1r, q[8], t, t)
# For another cycle, pass the previous RESET final_u; do not reset to q[8].
```

The actual benchmark starts 1T1R at cycle4 and propagates through cycle53,
including the training/holdout boundary. Bare records use a common initial
value for each conditionally paired record because their chronological
continuity has not been established. Two different initial values are allowed;
there are no per-cycle fitted initial values or per-topology material tables.

## Reproduce and interpret

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest tests.test_shared_gap -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python analysis/evaluate_shared_gap_joint.py --step .001
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python analysis/check_shared_gap_independent.py
```

Training plans and source inventory are under `analysis/SHARED_GAP_JOINT_PLAN_2026-11-03.md`
and `analysis/shared_hbn_2026-11-03/data_inventory.json`. Full scores, parameter
search limits, numerical checks, and plots are stored beside the fit. A common
framework is implemented; fitting quality and microscopic evidence are separate
questions. Do not report median-curve scores as all-record accuracy, suppress
the bare nonlimited SET score, or describe the optimizer's evaluation-limit stop
as convergence to an optimum.
