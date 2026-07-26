# VULCAN-2D v0.6 — measured transistor load line

## Why this version exists

v0.5 exposed a real failure: after removing every direct 1.1-V RESET anchor,
the median-loop RMSE changed from about `0.22/0.43` to `0.98/1.06 decade`.
The reason was structural as well as numerical. Gate voltage entered an
effective transistor curve and six separate h-BN RESET transport/state
families, so the model could hide gate-specific compensation inside the memory
element.

v0.6 removes that double counting. It does **not** claim that RESET is solved.

## Architecture

1. The transistor branch uses the public Nature Figure 2b
   `I_D(V_DS,V_G)` curves from DOI `10.5281/zenodo.7607096` (CC BY 4.0).
2. The series circuit solves `V_app = V_hBN + V_DS` and
   `I_hBN(V_hBN,state) = I_D(V_DS,V_G)` by bisection.
3. An affine gate mapping is fitted only to the 0.9/1.3-V SET current endpoints.
   The 1.1-V gate is held out. This mapping belongs to the electrical interface,
   not the h-BN material model.
4. All six formerly gate-indexed h-BN RESET parameter families are pooled to
   one endpoint-derived value. The SET path exponent is also set to zero.
5. The v0.5 implementation remains available as `model_v4.py` for comparison.

## Result and interpretation

The structural correction succeeds: the count of explicit gate-dependent h-BN
RESET families falls from six to zero. The SET endpoint currents are reproduced
by construction, while the held-out 1.1-V SET loop remains reasonably close.

RESET does not materially improve. The v0.6 1.1-V RESET RMSE is about
`0.94/1.07 decade`, comparable with the v0.5 endpoint holdout
`0.98/1.06 decade`. This falsifies the stronger claim that simply inserting
Figure 2b is sufficient.

The public Figure 2b data contain only non-negative `V_DS`, whereas RESET uses
the opposite applied-voltage polarity in the 1T1R circuit. v0.6 therefore labels
its RESET use as a **magnitude proxy**. A verified source/drain/body convention,
reverse-orientation transistor curves, and device/contact mapping remain needed
before the RESET load line can be called measured rather than hypothesized.

## Reproduce

```bash
python -m unittest discover -s tests -v
python -m vulcan2d.validate_v6
```

Outputs:

- `analysis/v6_validation_summary.json`
- `analysis/figures/14_vulcan_v6_transistor_loadline.png`
