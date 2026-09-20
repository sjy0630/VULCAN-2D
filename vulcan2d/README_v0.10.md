# VULCAN-2D v0.10 — protocol-conditioned RESET endpoint

## Why the return branch needed a different boundary condition

The measured-SET-conditioned cycle predicts the RESET starting conductance,
but its return-branch error did not improve. The cause is the former endpoint
assumption:

```text
G_HRS = rho(V_G) G_LRS .
```

At `V_G=1.1 V` under the paired DC RESET protocol, the 51 accepted cycles do
not show the required positive LRS-to-HRS scaling:

| Quantity | Observed value |
| --- | ---: |
| Standard deviation of `log10(G_LRS)` | 0.329 decade |
| Standard deviation of `log10(G_HRS)` | 0.177 decade |
| Pearson correlation of `log10(G_LRS)` and `log10(G_HRS)` | -0.137 (`p=0.337`) |
| Spearman correlation | -0.442 (`p=0.0012`) |

The final HRS is narrower than the starting LRS and is not positively
correlated with it. A fixed HRS/LRS ratio therefore transfers unsupported LRS
variation into the RESET return branch.

## Implemented endpoint law

For one fixed gate voltage and one fixed DC protocol, the endpoint is now
represented as a log-domain attractor distribution:

```text
log10(G_HRS) = mu_HRS(V_G, protocol) + epsilon .
```

For the `0 -> -1.7 -> 0 V`, `V_G=1.1 V` data:

```text
mu_HRS = -8.7261
G_HRS,center = 1.879e-9 A
robust sigma = 0.202 decade
```

The center is the mean of `log10(G_HRS)`. The residual scatter is retained so
an ensemble can reproduce endpoint variability. If a sampled HRS would exceed
the SET-predicted LRS, it is capped below LRS and the output records that the
guard was activated.

The complete data flow is now:

```text
measured S_i forward trace
    -> SET exposure proxy
    -> predicted initial RESET G_LRS
    -> transistor/h-BN voltage division
    -> internal-field RESET survival state
    -> protocol-conditioned final G_HRS distribution
    -> predicted R_i loop
```

The target `R_i` current is not a prediction input. It is used only to apply
the existing quality rule, construct the validation label and calculate the
error. Every target endpoint is predicted by an attractor fitted to the other
50 accepted cycles.

## Model selection

Three endpoint choices were compared under the same leave-one-cycle-out
protocol:

| Endpoint model | Final-HRS RMSE | RESET return RMSE |
| --- | ---: | ---: |
| Fixed HRS/LRS ratio | 0.304 decade | 0.393 decade |
| Linear SET-exposure-to-HRS law | 0.180 decade | 0.346 decade |
| Protocol HRS attractor | 0.180 decade | 0.343 decade |

The SET-to-HRS slope changes final-HRS RMSE by less than `0.001 decade` and
worsens the terminal return curve slightly. It was rejected. The attractor is
the simpler supported model.

## Full-cycle result

| Prediction path | RESET forward RMSE | RESET return RMSE |
| --- | ---: | ---: |
| No SET baseline | 0.244 decade | 0.391 decade |
| SET plus fixed HRS/LRS ratio | 0.220 decade | 0.393 decade |
| SET plus protocol HRS attractor | **0.202 decade** | **0.343 decade** |

Relative to the preceding full-cycle path, the new endpoint reduces the
forward error by about `8%` and the return error by about `13%`. It improves
both branches without reading the target RESET current during prediction.

### Chronological stress test

A stricter test uses only the first 35 accepted pairs to predict the later 16.
The later cycles have a higher mean `log10(G_HRS)` (`-8.612`) than the first
35 (`-8.778`), revealing a slow cycle-order drift. The stationary attractor's
individual HRS RMSE is consequently `0.202 decade`, compared with
`0.148 decade` for the old ratio in this particular chronological split.
Nevertheless, its complete return-branch RMSE remains lower:
`0.308 decade` versus `0.333 decade`.

This distinction matters. The attractor improves the representative I-V
loop, but it does not yet explain slow conditioning or degradation across
cycle number. A future history-dependent endpoint law must be validated on
more than one sequential run before it replaces the stationary distribution.

## Physical interpretation and boundary

The result is consistent with a RESET process that approaches a
protocol-dependent residual-conduction distribution after sufficient field
exposure. It does not determine whether that residual conduction comes from a
partly ruptured metal-assisted path, a reconstructed defect bridge, boron
vacancies or another microscopic configuration.

The endpoint center is calibrated only for `V_G=1.1 V` and the recorded DC
waveform. The 0.9-V and 1.3-V workbooks have gate, batch and protocol changes
that cannot presently be separated. The attractor must therefore not be
extrapolated across gate voltage, device, area, temperature, sweep rate or
pulse width.

The 1T1M circuit context and transistor current control follow K. Zhu et al.,
“Hybrid 2D–CMOS microchips for memristive applications,” *Nature* 618, 57–62
(2023), DOI: 10.1038/s41586-023-05973-1. The HRS attractor is a
quality-controlled empirical result from the paired local data, not a law
reported in that paper.

Cycle-to-cycle distributions are standard objects in compact RRAM models;
for example, Aguirre et al. model randomly distributed programmed states and
validate variability against several RRAM technologies. That work supports
representing endpoint scatter, but it does not establish the numerical center
or mechanism used here. See F. L. Aguirre et al., *Microelectronic
Engineering* 266, 111886 (2022), DOI: 10.1016/j.mee.2022.111886. Direct h-BN
experiments also attribute cycle variation to stochastic atomic rearrangement
during conductive-path formation and disruption; see “Stochastic resonance
in 2D materials based memristors,” *npj 2D Materials and Applications* 8
(2024), DOI: 10.1038/s41699-024-00444-1. These references motivate the model
form only; the present device's microscopic species remains unresolved.

## Reproduce

```bash
python -m unittest tests.test_cycle_link -v
python -m vulcan2d.validate_full_cycle
```

Outputs:

- `analysis/full_cycle_hrs_attractor_summary.json`
- `analysis/figures/18_vulcan_full_cycle_hrs_attractor.png`
