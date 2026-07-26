# VULCAN-2D v0.5 data-constraint audit

v0.5 keeps the v0.4 soft/hard state equations but strengthens their evidence
boundary and validation protocol:

- standalone 1R SET now uses the confirmed 1 mA compliance;
- `spatial_prior.py` converts a two-column CAFM spot-area distribution into an
  optional K-patch geometric prior;
- measured area is not coupled to switching threshold unless joint data later
  support that relation;
- `validate_v5.py` removes all direct 1.1 V gate anchors and evaluates a
  0.9/1.3 -> 1.1 V interpolation holdout.

Run the numeric audit with the local CAFM file:

```bash
python -m vulcan2d.validate_v5 --cafm /path/to/spot_distribution.txt
```

The outputs are `analysis/v5_validation_summary.json` and
`analysis/figures/13_vulcan_v5_constraints_audit.png`. Plotting is optional:
when Matplotlib is unavailable the complete numeric report is still written.

The central result is asymmetric. SET survives the gate holdout, but RESET
does not: its forward/return RMSE rises to about 0.98/1.06 decade after the
1.1 V anchors are removed. The next model change should therefore replace the
many gate-specific RESET factors with a measured transistor load line plus one
shared h-BN transport/state law. The current standalone-transistor data do not
yet establish the RESET terminal orientation or exact 1T-to-1T1R mapping.

See `analysis/V5_DATA_CONSTRAINED_PHYSICS_2026-07-26.md` for the quantitative
results and article-safe claim hierarchy.
