# VULCAN-2D v0.7 — inverse-identified RESET state

## Purpose

v0.6 removed explicit gate-dependent h-BN RESET compensation, but it still had
two unresolved assumptions: the RESET transistor branch was a magnitude copy of
the positive Figure 2b curve, and the RESET state trajectory was generated from
preassigned thresholds rather than inferred from the measured loop.

v0.7 closes the first gap with the independently measured negative-`V_DS`
transistor curves and changes RESET identification from terminal-curve tuning to
a de-embedding and state-reconstruction problem.

## Circuit reconstruction

For every measured point `(V_app, I, V_G)`, the measured negative transistor
lookup is inverted:

```text
V_DS = I_T^-1(|I|, V_G)
|V_hBN| = |V_app| - |V_DS|
E_hBN = 10 |V_hBN| / d_hBN    [MV/cm, d_hBN in nm]
```

The JSON lookup preserves all 56 gate sweeps from `V_G=0` to `5.5 V`. The
fourth source-workbook column is provisionally interpreted as `I_D`; terminal
and body/well metadata still require experimenter confirmation.

## Observable RESET state

Stable pre-RESET and post-RESET portions of each cycle identify two transport
envelopes:

```text
I_LRS(V_hBN) = G_LRS sinh(alpha_LRS |V_hBN|)
I_HRS(V_hBN) = G_HRS sinh(alpha_HRS |V_hBN|)
```

The state reconstructed from the measured current is

```text
s = (I - I_HRS) / (I_LRS - I_HRS),  0 <= s <= 1
```

and is monotonically projected during the forward RESET ramp. `s` is the
remaining fraction of resettable conductance between the two electrical
envelopes. It does not identify a vacancy concentration, filament radius, or
chemical species.

Only the 0.9-V and 1.3-V ensembles identify the shared coefficients. The 1.1-V
ensemble is held out:

| Quantity | Endpoint-trained value | 1.1-V holdout median |
| --- | ---: | ---: |
| `alpha_LRS` | 8.044 V^-1 | not refitted |
| `alpha_HRS` | 6.097 V^-1 | not refitted |
| `E50` | 0.837 MV/cm | 0.779 MV/cm |
| 90%-to-10% logistic scale | 0.147 MV/cm | 0.151 MV/cm |

The common quasi-static law is

```text
s(E) = 1 / (1 + exp((E - E50) / width)).
```

The held-out median state RMSE is about `0.041` on the normalized 0–1 state
scale. This is the strongest current evidence that the three gate conditions
share an approximately common internal-field RESET trajectory once the
transistor is de-embedded.

## Terminal-loop result

When each cycle is conditioned on its low-disturbance LRS/HRS envelope, the
1.1-V median terminal-loop RMSE changes from the v0.6 unconditioned
`0.943/1.067 decade` to approximately `0.23/0.36 decade` for forward/return
branches. The scope is intentionally different: v0.7 tests the RESET transition
law after observing the boundary states; it does not yet predict absolute
`G_LRS` and `G_HRS` from geometry or SET history alone.

## Physical and statistical scope

- Gate voltage does not enter the h-BN RESET survival equation.
- Gate voltage enters through the measured transistor load line and through the
  initial state produced by the preceding SET operation.
- The logistic survival curve is the continuum representation of distributed
  local rupture thresholds. It is compatible with localized CAFM switching but
  is not an atomistic image.
- DC point timing is not known accurately. The identified law is a voltage-ramp
  law and must not be reported as a switching rate in seconds.
- The next model-identification task is to predict the LRS/HRS envelope
  amplitudes from the preceding SET read state, rather than conditioning on the
  RESET loop endpoints.

## Reproduce

```bash
python -m vulcan2d.precompute_reverse_transistor /path/to/Bookoutput2.xls
python -m unittest tests.test_model_v7 -v
python -m vulcan2d.validate_v7
```

Outputs:

- `analysis/v7_reset_identification_summary.json`
- `analysis/figures/15_vulcan_v7_reset_identification.png`
