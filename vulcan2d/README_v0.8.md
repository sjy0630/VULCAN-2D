# VULCAN-2D v0.8 — read-initialized RESET prediction

## What changed

The preceding RESET model reproduced the transition only after observing both
the LRS and HRS envelopes of each measured cycle.  That is useful for
identifying the internal state law, but the final HRS endpoint is already part
of the answer to be predicted.

The new path removes that endpoint leakage.  It uses only the low-voltage
segment at the beginning of RESET to initialize the cycle and predicts the
post-RESET HRS before the return branch is seen.  It does not require another
experiment because the existing RESET workbooks already contain these initial
low-voltage points.

## Initial-state read

The measured negative-`V_DS` transistor lookup is inverted point by point:

```text
|V_hBN| = |V_app| - |V_DS(I, V_G)|
```

Only the initial forward-ramp points with `0.08 <= |V_hBN| <= 0.12 V` are used:

```text
G_LRS,read = median[I / sinh(alpha_LRS |V_hBN|)]
```

Low-voltage points on the post-RESET return branch are explicitly excluded.
Changing or deleting the later HRS data therefore cannot alter the estimated
initial state.

## Predicted RESET endpoint

For every accepted training cycle, the previously identified envelopes give
`rho = G_HRS/G_LRS`.  The two training conditions contribute equal-weight
group medians:

```text
median log10(rho) at V_G=0.9 V = -1.518
median log10(rho) at V_G=1.3 V = -2.080
```

Within the measured range, the endpoint law is the straight interpolation
between those anchors:

```text
log10 rho(V_G) = -1.799 - 1.406 (V_G - 1.1 V)
G_HRS,pred = rho(V_G) G_LRS,read
```

Outside `0.9--1.3 V`, the nearest boundary value is used instead of
extrapolating.  This gate-conditioned endpoint is phenomenological: it can
represent how the transistor load line and the recorded protocol affect the
RESET depth.  It is not evidence that gate voltage directly changes h-BN
chemistry, and it may contain batch effects.

The internal h-BN transition law is unchanged and remains gate-neutral:

```text
s(E_hBN) = 1 / [1 + exp((E_hBN - E50) / width)]
I = s I_LRS + (1-s) I_HRS
```

## Model selection

An endpoint law based on the reconstructed maximum h-BN field was also tested
without using the 1.1-V holdout.  It increased cross-group HRS RMSE in both
directions:

| Training → test group | Constant ratio | Maximum-field ratio |
| --- | ---: | ---: |
| 0.9 V → 1.3 V | 2.122 decade | 2.697 decade |
| 1.3 V → 0.9 V | 1.329 decade | 1.583 decade |

The maximum-field endpoint law was therefore rejected.  The implemented
two-anchor interpolation is simpler and does not add an unstable fitted
coefficient.

## Held-out result at 1.1 V

All transport, state and endpoint parameters are fitted on the 0.9-V and
1.3-V groups.  The 51 accepted 1.1-V cycles remain the interpolation holdout.

| Quantity | Held-out result |
| --- | ---: |
| Initial `G_LRS` median absolute error | 0.047 decade |
| Initial `G_LRS` RMSE | 0.111 decade |
| Predicted `G_HRS` median absolute error | 0.309 decade |
| Predicted `G_HRS` RMSE | 0.452 decade |
| Terminal RESET forward-branch RMSE | 0.234 decade |
| Terminal RESET return-branch RMSE | 0.397 decade |

For comparison, the older unconditioned terminal errors were `0.943/1.067
decade`.  The endpoint-conditioned model reached `0.230/0.361 decade` but used
the measured HRS endpoint.  The read-initialized model stays close to that
accuracy without reading the endpoint it is meant to predict.

## Claim boundary

- This is a quasi-static voltage-ramp model.  Unknown DC dwell times do not
  support a calibrated switching rate in seconds.
- The initial state still comes from an electrical read; it is not predicted
  from geometry alone.
- The observable state is compatible with distributed local rupture, but it
  is not a measured metal-ion concentration, boron-vacancy population or
  filament radius.
- The HRS endpoint interpolation is validated only from `V_G=0.9` to `1.3 V`.
- A causal microscopic endpoint law would still require stronger structural
  evidence or deliberately varied protocols.  It is not required for the
  present numerical upgrade.

## References

1. K. Zhu et al., “Hybrid 2D–CMOS microchips for memristive applications,”
   *Nature* 618, 57–62 (2023). DOI: 10.1038/s41586-023-05973-1.
2. F. L. Aguirre et al., “Comprehensive physics-based RRAM compact model
   including the effect of variability and multi-level random telegraph
   noise,” *Microelectronic Engineering* 266, 111886 (2022).
   DOI: 10.1016/j.mee.2022.111886.

## Reproduce

```bash
python -m unittest tests.test_model_v8 -v
python -m vulcan2d.validate_v8
```

Outputs:

- `analysis/v8_read_initialized_reset_summary.json`
- `analysis/figures/16_vulcan_read_initialized_reset.png`
