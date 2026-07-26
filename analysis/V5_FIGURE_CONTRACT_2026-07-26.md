# VULCAN-2D v0.5 figure contract

## Analytical questions and takeaways

1. **CAFM distribution:** what lateral size heterogeneity is actually present?
   Takeaway: the count is dominated by small apparent conductive spots, while
   the area contribution has a heavier tail; this is a geometric prior only.
2. **Multi-regime validation:** does one reduced-order model reproduce the
   measured 1T1R soft regime and standalone 1R hard regime after the protocol
   correction? Takeaway: the 1 mA correction closes the 1R SET return branch,
   while the 1R RESET full-loop mismatch remains visible.
3. **Constraint audit:** which relationships transfer without a direct 1.1 V
   anchor? Takeaway: SET interpolates, RESET fails by about one decade, and the
   CAFM area prior cannot repair the missing transistor/load-line explanation.

## Chart map

| output | family / variant | data grain | palette and non-color encoding |
|---|---|---|---|
| `11_afm_conductive_spot_distribution.png` | distribution: stem + cumulative lines | 59 bins, 4,866 spots | teal/orange; marker and line distinction |
| `12_vulcan_v4_multiregime_validation.png` | small-multiple I-V lines | 100/53/101 1T1R cycles and 22/22 1R cycles | gate colors plus solid data / dashed model |
| `13_vulcan_v5_constraints_audit.png` | 2x2 line and grouped-bar audit | measured medians, direct model, endpoint holdout, CAFM sensitivity | neutral direct, orange holdout, blue CAFM; line styles and direct bar labels |

All current plots use logarithmic current axes where orders of magnitude are
the analytical comparison. Absolute comparison bars start at zero. Images are
exported as static PNG files under `analysis/figures/` and visually inspected at
their final resolution before GitHub publication.
