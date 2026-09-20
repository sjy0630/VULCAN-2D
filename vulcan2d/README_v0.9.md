# VULCAN-2D v0.9 — measured SET to predicted RESET

## What is now connected

The 53 sheet pairs have a confirmed temporal meaning: `S1` is followed by
`R1`, `S2` by `R2`, and so on. The model now passes information from each SET
operation into the prediction of the following RESET operation. Cycles 2 and 3
are excluded by the pre-existing RESET quality rule because their LRS and HRS
envelopes are not separable; 51 pairs remain.

For a target pair, the code receives:

- the measured voltage and current of `S_i`;
- the known negative-voltage waveform of `R_i`;
- the gate voltage.

It does **not** receive the current of `R_i`. That current is used only to
quality-control the validation set, define the held-out label and calculate
error; it is not a feature of that target's prediction.

## State hand-off

Only the rising branch of SET, from zero voltage to its positive apex, is
used. The return branch is excluded. The SET feature is

```text
A_SET = integral |I_SET(V)| dV .
```

`A_SET` has units of A V and is called a **voltage-domain exposure proxy**.
The measurement did not record point dwell time or sweep rate, so it must not
be reported as charge or energy. If every curve used the same linear ramp
rate, however, it is proportional to transferred charge.

The following RESET starting conductance is described by a centered log law:

```text
log10(G_RESET,0) = log10(G_ref)
                 + b [log10(A_SET) - log10(A_ref)] .
```

For all 51 accepted pairs, `A_ref = 1.331e-4 A V`,
`G_ref = 1.374e-7 A`, and `b = 31.11`. The large numerical slope reflects the
very narrow observed exposure range (roughly `1.28e-4` to `1.37e-4 A V`), not
a universal material exponent. This law should not be extrapolated beyond the
recorded protocol.

The predicted `G_RESET,0` replaces the low-voltage RESET read previously used
to initialize the model. The rest of the model is unchanged:

```text
V_app = V_hBN + V_DS,          I_hBN = I_transistor
G_HRS = rho(V_G) G_RESET,0
s(E) = 1 / [1 + exp((E - E50) / width)]
I_hBN = s G_RESET,0 sinh(alpha_LRS V_hBN)
      + (1-s) G_HRS sinh(alpha_HRS V_hBN) .
```

The reverse transistor curve determines the series voltage division, and the
h-BN internal field determines the irreversible RESET progress on the outward
ramp. The state remains ruptured on the return ramp.

## Leakage-controlled validation

Each target cycle is predicted with a transfer law fitted on the other 50
accepted pairs. This is leave-one-cycle-out validation: the target RESET
starting-state label is never used to fit its own prediction.

Against the common endpoint-identified starting-state target:

| Initial-state method | Median absolute error | RMSE |
| --- | ---: | ---: |
| No-SET leave-one-out median | 0.143 decade | 0.333 decade |
| SET-conditioned leave-one-out | 0.130 decade | 0.320 decade |
| Initial RESET read | 0.047 decade | 0.111 decade |

The SET exposure and next RESET initial conductance have Pearson
`r = 0.523` in log space and Spearman `rho = 0.503` (`p < 2e-4`). In a
chronological check trained on the first 35 accepted pairs and tested on the
later 16, the SET-conditioned RMSE is `0.304 decade`, versus `0.362 decade`
for the no-SET median baseline.

For the median terminal RESET loop:

| Method | Forward RMSE | Return RMSE |
| --- | ---: | ---: |
| No-SET median baseline | 0.244 decade | 0.391 decade |
| SET-conditioned | 0.220 decade | 0.393 decade |
| RESET-read initialized reference | 0.234 decade | 0.397 decade |

The new link gives a modest improvement on the blind starting state and the
forward RESET branch. It does not improve the return branch, whose error is
now dominated by the empirical final-HRS ratio rather than initial-state
knowledge.

## Scientific claim boundary

- This is the first data-validated **measured-SET-conditioned full electrical
  cycle** in the repository. It is not yet a fully autonomous SET simulator.
- The observed correlation says that the SET history contains predictive
  information about the next RESET initial state. It does not prove a unique
  microscopic cause.
- The result is protocol-specific. The narrow exposure range and single
  paired gate condition do not support extrapolation to new sweep rates,
  pulse widths, temperatures, areas or devices.
- The state variable remains an electrically observable distributed-path
  fraction. It cannot be renamed a metal-ion population, boron-vacancy
  density or filament radius without direct structural evidence.

The circuit-level use of a CMOS transistor to control current through the
h-BN memristor follows K. Zhu et al., “Hybrid 2D–CMOS microchips for
memristive applications,” *Nature* 618, 57–62 (2023),
DOI: 10.1038/s41586-023-05973-1. The SET-to-RESET transfer coefficient and its
validation are new empirical results from the paired local workbooks, not a
parameter quoted from that article.

## Reproduce

```bash
python -m unittest tests.test_cycle_link -v
python -m vulcan2d.validate_full_cycle
```

Outputs:

- `analysis/full_cycle_validation_summary.json`
- `analysis/figures/17_vulcan_full_cycle_link.png`
