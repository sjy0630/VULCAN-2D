# VULCAN-2D: field-assisted HRS transport refinement

This update corrects the largest remaining error in the measured-SET-
conditioned full cycle: the post-RESET return branch.  It does not add a free
terminal-current correction and it does not read the target RESET current when
making a prediction.

## What changed

The earlier model used the same generic field-activated shape for both stable
RESET envelopes:

\[
I_{\mathrm{LRS}}=G_{\mathrm{LRS}}\sinh(\alpha_{\mathrm{LRS}}V_{\mathrm{hBN}}),
\qquad
I_{\mathrm{HRS}}=G_{\mathrm{HRS}}\sinh(\alpha_{\mathrm{HRS}}V_{\mathrm{hBN}}).
\]

The LRS expression is retained.  The HRS expression is replaced by a
fixed-thickness, effective Poole--Frenkel-like curvature:

\[
I_{\mathrm{HRS}}
=A_{\mathrm{HRS}}V_{\mathrm{hBN}}
\exp\!\left(\beta_{\mathrm{PF}}\sqrt{V_{\mathrm{hBN}}}\right).
\]

For a fixed h-BN thickness, this is the voltage-domain form of the usual
field-assisted expression \(J\propto E\exp(\gamma\sqrt E)\).  One shared
\(\beta_{\mathrm{PF}}\) describes the curvature of a gate/protocol group, while
each cycle retains its own endpoint amplitude \(A_{\mathrm{HRS}}\).  Profiling
out the amplitude prevents cycle-to-cycle HRS variability from being mistaken
for a different transport curvature.

The fitted coefficient is an **effective compact-model parameter**.  The
available data contain one temperature and do not support extracting a trap
energy or dielectric constant.  The fit therefore does not prove that
Poole--Frenkel emission is the unique microscopic mechanism.

## Leakage-free validation

There are 53 measured `S_i -> R_i` pairs.  Two RESET cycles fail the previously
defined state-separation quality check, leaving 51 accepted pairs.  For every
target pair, the validation excludes its complete RESET current from:

1. the SET-to-initial-RESET transfer fit;
2. the shared HRS curvature fit;
3. the HRS endpoint-amplitude distribution.

The target inputs are its measured SET trace and RESET voltage protocol.  Its
RESET current is used only for quality control and scoring.  The irreversible
transition-field law remains the independent 0.9/1.3-V identification, so this
update isolates HRS transport instead of refitting the complete target loop.

## Results

### Leave one complete cycle out

| Median-loop RMSE | Original `sinh` HRS | Effective PF HRS | Change |
|---|---:|---:|---:|
| RESET forward | 0.202 decade | 0.176 decade | −13.0% |
| RESET return | 0.343 decade | 0.124 decade | −63.9% |

The HRS within-cycle shape MAD decreases from 0.138 to 0.063 decade.  The
leave-one-out \(\beta_{\mathrm{PF}}\) range is narrow, 6.13--6.30
\(\mathrm{V}^{-1/2}\), with median 6.25 \(\mathrm{V}^{-1/2}\).  This stability
is important: the improvement does not come from assigning an unconstrained
curvature to every cycle.

The effective-PF return median lies inside the measured 10--90% current band at
every scored voltage point.  Its residual RMSE is 0.056 decade at 1.2--1.7 V,
compared with 0.470 decade for the previous HRS expression.  The remaining
largest local error is the 0.1--0.4 V return interval (0.258 decade), where
terminal currents are smallest and de-embedding is most sensitive.

### Chronological check

Using only the first 35 accepted pairs to fit the SET transfer, HRS curvature
and endpoint distribution, then predicting the later 16 pairs:

| Later-16 median-loop RMSE | Original `sinh` HRS | Effective PF HRS |
|---|---:|---:|
| RESET forward | 0.229 decade | 0.238 decade |
| RESET return | 0.308 decade | 0.130 decade |

The return improvement survives time-ordered validation.  The small forward
increase shows that the next optimization target is the 0.8--1.2 V transition
region, not another HRS endpoint correction.

## What remains unchanged

- `S_i` is paired with the immediately following `R_i`.
- The measured forward SET trace supplies a voltage-domain exposure proxy.
- The independent negative-\(V_{DS}\) transistor lookup supplies the RESET
  load line.
- The RESET state remains a monotone observable fraction between stable LRS
  and HRS envelopes.
- The protocol-conditioned HRS endpoint remains stochastic rather than a fixed
  HRS/LRS ratio.

## Remaining limits

- The model is measured-SET-conditioned, not a fully autonomous SET simulator.
- The DC data do not provide a physical switching-time constant.
- The transport fit does not identify a unique metal-ion, boron-vacancy or
  other microscopic species.
- The PF coefficient is validated only for the 1.1-V gate and the paired
  0 to −1.7 to 0 V DC RESET protocol.
- Cross-gate validation of the new HRS curvature still requires applying the
  same target-excluding procedure to the 0.9-V and 1.3-V groups.

## Reproduce

```bash
python -m vulcan2d.validate_v11
```

Outputs:

- `analysis/full_cycle_pf_refinement_summary.json`
- `analysis/figures/19_vulcan_pf_hrs_refinement.png`

## Literature boundary

Field-assisted Frenkel--Poole leakage has been measured in h-BN dielectrics:

- Pierret et al., *Materials Research Express* **9**, 065901 (2022),
  https://doi.org/10.1088/2053-1591/ac4fe1.

The project-specific physical motivation is also consistent with the HRS
Poole--Frenkel discussion in Kaichen Zhu's dissertation.  Conversely, other
h-BN stacks can favor hopping or trap-assisted tunnelling, and apparently good
Poole--Frenkel linearizations can yield unphysical parameters:

- Völkel et al., *Advanced Functional Materials* **34**, 2300428 (2024),
  https://doi.org/10.1002/adfm.202300428.

Accordingly, this update claims a predictive **PF-like curvature**, not unique
microscopic proof.
