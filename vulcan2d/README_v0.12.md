# VULCAN-2D autonomous quasi-static SET-RESET loop

This update closes the main modelling gap left by v0.11: after calibration,
the simulator now receives the **SET voltage waveform, RESET voltage waveform
and gate voltage**, and predicts both current branches without receiving the
target SET or RESET current.

The present validation domain is deliberately narrow: the measured 1.1-V
gate, 0-to-5-to-0 V DC SET and 0-to--1.7-to-0 V DC RESET protocols.  This is
not yet an arbitrary waveform, time-domain or new-device predictor.

## Corrections to the September SET handoff

The earlier exploratory handoff was useful but contained five problems:

1. It reused the v0.6 affine cross-gate endpoint correction with the public
   Fig. 2b transistor lookup.  That correction had been fitted to the 0.9/1.3-V
   SET endpoints, so it was not an arbitrary coding mistake; however, at an
   applied 1.1-V gate it evaluates the lookup near 0.896 V and makes this
   inverse reconstruction fold back.  The literal alternative interpolates
   the measured 1.0- and 1.5-V curves at 1.1 V.  In a controlled comparison,
   the literal mapping identifies all 53 SET trajectories and gives a
   full-group median-loop error of 0.078/0.085 decade, whereas the affine
   mapping leaves fewer than three trajectories with enough state range for
   the corrected two-envelope model.  SET therefore uses literal interpolation
   at 1.1 V.  This is a data-selected circuit-interface choice, not a claim
   that transistor threshold offsets can never occur.
2. It forced the pre-SET and connected states to share one `sinh` curvature.
   The de-embedded stable branches instead support separate effective shapes.
3. It rescaled every inferred state trajectory to end at zero and one.  That
   procedure guarantees apparent complete switching and was removed.
4. It labelled `integral |V_hBN| dV_app` in a millivolt-like unit.  Its unit is
   V2; it is a voltage-domain exposure, not time, energy or charge.
5. Its 0.06 state RMS measured how well a state inferred from the target
   current could be reconstructed.  It was not a current-prediction test.  The
   new audit excludes each target pair from calibration and reports current
   error directly.

## Circuit and SET equations

At every voltage point the series circuit obeys

\[
V_{\rm app}=V_{\rm hBN}+V_{DS},\qquad
I_{\rm hBN}(V_{\rm hBN},x)=I_T(V_{DS},V_G).
\]

`I_T` is interpolated from the public positive-bias transistor output curves
used for SET.  The two stable h-BN envelopes are

\[
I_{\rm pre}=A_{\rm pre}V_{\rm hBN}
\exp(\beta_{\rm pre}\sqrt{V_{\rm hBN}}),
\]

\[
I_{\rm connected}=A_{\rm connected}V_{\rm hBN}^{n},
\]

and the observable connected fraction mixes them:

\[
I_{\rm hBN}=(1-x)I_{\rm pre}+xI_{\rm connected}.
\]

The fitted full-group values are

- `beta_pre = 6.538 V^-1/2`;
- `n = 1.326`;
- pre-envelope shape RMSE `0.211 decade`;
- connected-envelope shape RMSE `0.113 decade`.

The power-law connected envelope is called **effective super-ohmic**.  A good
power-law fit is not sufficient to identify space-charge-limited conduction or
any other unique microscopic process.

The rising-branch state uses the step-size-invariant exposure

\[
\Xi=\int |V_{\rm hBN}|\,dV_{\rm app},\qquad [\Xi]={\rm V^2},
\]

\[
x(\Xi)=\frac{1}{1+\exp[-(\Xi-\Xi_{50})/w_{\Xi}]}.
\]

`V_hBN`, exposure and state are solved self-consistently with the transistor
load line.  The state is irreversible during the rising branch and is frozen
on the return branch.  Full-group values are `Xi50 = 1.135 V2` and
`w_Xi = 0.0479 V2`.  This is a quasi-static voltage-coordinate law; without
point dwell times it does not predict switching time or pulse-width effects.

## SET-to-RESET handoff

The final connected SET envelope and initial RESET LRS law use different bias
directions and read protocols.  They are first matched at 0.1 V, then linked by
a robust factor fitted only from paired training cycles:

\[
G_{\rm LRS,RESET}=\eta\,
\frac{A_{\rm connected}(0.1)^{n}}
{\sinh(\alpha_{\rm LRS,RESET}\,0.1)}.
\]

The full-group value is `eta = 0.258`, with `0.225 decade` robust scatter.  It
absorbs cross-protocol relaxation, polarity/read differences and unresolved
history.  It is **not** a measured ion-retention or migration efficiency.

RESET then uses the measured negative-bias transistor load line, effective
field-assisted HRS curvature and a protocol-conditioned endpoint amplitude.
The PF-envelope observable state identified at 1.1 V gives
`E50 = 0.793 MV/cm` and width `0.162 MV/cm`, with state-fit RMS `0.037`.
Unlike the older RESET identifier, this path does not force every cycle's
state endpoints to exactly zero and one.

Two RESET-transition modes are retained because the data reveal a real
distribution shift:

- `same_protocol` uses the target-excluding 1.1-V state law and best describes
  randomly held-out cycles from the full dataset;
- `cross_gate_prior` uses the earlier 0.9/1.3-V transition prior and is more
  robust when only cycles 1--35 are available and cycles 36--53 arrive later.

This is not two hidden transport models: the transistor, LRS/HRS equations,
SET handoff and endpoint law are unchanged.  Only the protocol-level RESET
state threshold and width differ.

## Target-excluding validation

There are 53 SET cycles and 53 paired RESET cycles.  RESET cycles 2 and 3 fail
the pre-existing separable-envelope quality check, leaving 51 pairs.  For each
target pair, the following are all refitted without that pair:

- SET envelope shapes and amplitudes;
- SET exposure-transition distribution;
- SET-to-RESET link;
- RESET HRS curvature;
- 1.1-V RESET state threshold and width;
- RESET endpoint distribution.

Only the target voltage grids enter the simulator.  Its SET and RESET currents
are used afterwards for scoring.

| Leave-one-cycle-out median-loop RMSE | Rising/forward | Return |
|---|---:|---:|
| SET | 0.078 decade | 0.085 decade |
| RESET, same-protocol state law | 0.192 decade | 0.124 decade |
| RESET, cross-gate transition prior | 0.230 decade | 0.123 decade |

Across individual points above the stated current floors, pooled RMSE is
`0.166 decade` for SET and `0.293 decade` for RESET with the same-protocol
state law.  Median per-cycle RMSE is `0.160 decade` for SET and
`0.256 decade` for RESET.  These larger values are expected because a central
deterministic trajectory cannot reproduce all cycle-to-cycle fluctuations.

The chronological check fits cycles 1--35 and predicts the later 18 accepted
pairs:

| Chronological median-loop RMSE | Rising/forward | Return |
|---|---:|---:|
| SET | 0.121 decade | 0.097 decade |
| RESET, same-protocol state law | 0.273 decade | 0.107 decade |
| RESET, cross-gate transition prior | 0.230 decade | 0.107 decade |

The SET and post-RESET return accuracy survive the time-ordered split.  The
RESET forward branch does not remain stationary: its identified `E50` median
is about `0.753 MV/cm` in the first 33 accepted RESET cycles and
`0.870 MV/cm` in the later 18.  There is no significant trend within the first
group that reliably forecasts the jump.  The cross-gate prior is therefore
the safer sequential-prediction mode, while the 1.1-V law is the better
same-population reconstruction mode.  A future model should condition this
shift on a measured history/batch variable rather than hide it in another
free curve-correction parameter.

## What can and cannot be claimed

The model can now, within the calibrated protocol:

- predict a typical complete SET/RESET I-V loop from voltage and gate input;
- separate transistor voltage from h-BN voltage through a measured load line;
- expose a continuous, irreversible connected-fraction state for circuit and
  visualization use;
- generate parameter-distribution variants using the measured scatter.

It still cannot:

- predict an arbitrary gate, peak voltage, scan rate, pulse width or
  temperature outside the calibration domain;
- reproduce the identity of every stochastic cycle with one central input;
- assign the state uniquely to boron vacancies, metal ions, a filament radius
  or a particular transport mechanism;
- infer physical time constants from the DC sweep index;
- transfer to a new device geometry without recalibration.

## Reproduce

```bash
python3 -m vulcan2d.validate_v12
python3 -m unittest discover -s tests
```

After calibration, the central model can be called without loading Excel
workbooks:

```python
from vulcan2d import simulate_calibrated_cycle

result = simulate_calibrated_cycle(set_voltage, reset_voltage, gate_voltage=1.1)
set_current = result["set"]["Is"]
reset_current = result["reset"]["Is"]

# More conservative when predicting a later, potentially shifted batch:
robust = simulate_calibrated_cycle(
    set_voltage, reset_voltage, gate_voltage=1.1,
    reset_transition_mode="cross_gate_prior",
)
```

The current desktop UI server still calls the legacy interactive ensemble.  It
has not been silently redirected because its old controls (`K`, `Icc`, generic
`Vset`) do not yet map one-to-one onto the audited v0.12 parameters.  Updating
that interface is a separate integration task; the Python research model and
its voltage-only API are complete for the stated calibration domain.

Outputs:

- `analysis/full_cycle_autonomous_summary.json`
- `analysis/figures/20_vulcan_autonomous_loop.png`

## Literature boundary

The device geometry, transistor current limiting, progressive high-resistance
switching regime and public electrical data originate from:

- Zhu et al., *Nature* **618**, 57--62 (2023),
  https://doi.org/10.1038/s41586-023-05973-1.

Field-assisted Frenkel--Poole leakage has been reported in h-BN dielectrics:

- Pierret et al., *Materials Research Express* **9**, 065901 (2022),
  https://doi.org/10.1088/2053-1591/ac4fe1.

The caution against converting a visually good field-assisted linearization
into a unique microscopic assignment is supported by:

- Voelkel et al., *Advanced Functional Materials* **34**, 2300428 (2024),
  https://doi.org/10.1002/adfm.202300428.

Accordingly, the equations above are evidence-constrained compact laws, not a
claim that one atomistic mechanism has been proven.
