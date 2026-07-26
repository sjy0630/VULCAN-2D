# VULCAN-2D v0.3

VULCAN-2D v0.3 is a variability-aware reduced-order model for the measured h-BN
1T1M cell. It is calibrated to 53 SET and 53 RESET sweeps. It is suitable for
reproducing the present quasi-static experiment, but it is not yet a predictive
microscopic model for temperature, pulse width, area, or device-to-device scaling.

## Model equations

The h-BN area is represented by `K` parallel sub-populations with effective
contribution weights `w_k`
and soft-breakdown state `phi_k` in `[phi_min, 1]`. With
`Theta(phi)=phi**p_perc`, the ensemble transport prefactor for polarity `s` is

```text
G_ens^(s) = I_s^(s) sum_k w_k [exp(xi_H,k)(1-Theta(phi_k))
                                + G_on,eff Theta(phi_k)]
I_hBN = G_ens^(s) sinh(alpha_s V_h)
```

This interpolation keeps the HRS residual `xi_H` separate from the LRS
soft-breakdown fluctuation in `G_on,eff`. The `sinh` law follows from detailed
balance between forward and backward hopping rates,
`Gamma_+-Gamma_- proportional to sinh(Delta mu/2kBT)`. With
`Delta mu=q a_eff V_h/d`, `alpha=q a_eff/(2kBTd)`. At 300 K and `d=6 nm`, the
fitted SET/RESET coefficients correspond to effective potential-drop lengths of
2.09/2.38 nm. These are coarse-grained lengths, not literal nearest-neighbour
atomic hopping distances. The current data still do not uniquely distinguish
hopping, TAT, Poole-Frenkel, Schottky, or another defect-assisted microscopic
mechanism.

Here `w_k` is not necessarily a pure geometric area fraction. A more physical
interpretation is `w_k,eff proportional to A_k g_k`, where `A_k` is the CAFM-
visible projected area and `g_k` collects effective thickness, defect depth,
barrier, and electrode-access effects. This follows the h-BN weak-region form
`I=J(t_eff)A` used by Chen et al. (Nature Electronics 2020).

The transistor and h-BN are solved self-consistently as a series load line:

```text
I_tr = I_sat^(s) [1 + lambda_s |V_tr|] tanh(|V_tr|/V_k,s)
V_app = V_h + V_tr
I_hBN(V_h) = I_tr(V_tr)
```

An experimental M1 path can replace the empirical transistor law with a
measured `I_D(V_DS,V_G)` lookup generated from the advisor-provided
`Fig.2b.xlsx`:

```bash
python -m vulcan2d.precompute_transistor /path/to/Fig.2b.xlsx
```

The generated `vulcan2d/transistor_lookup.json` is deliberately ignored by git
because it contains unpublished measurement data. Load it with
`load_default_lookup()` and pass it explicitly to `simulate_cycles` together
with `gate_voltage_set` and/or `gate_voltage_reset`. The empirical law remains
the default M0 path until transistor terminal orientation, the exact 53-cycle
gate bias, and reverse-bias output characteristics are confirmed. The current
workbook covers `0.5 <= V_G <= 3 V` and `0 <= V_DS <= 5 V`; extrapolation is
rejected.

Each simulated cycle now retains `Vhs` and `Vhr`, the self-consistent h-BN
internal-voltage traces for SET and RESET. This makes the next local-field
kinetics audit possible without changing the calibrated default, which still
drives state evolution with applied voltage.

`phi_k` is the occupancy of high-transmission configurations in an areal
sub-population, not the literal radius or atom count of a filament. A patch may
coarse-grain an intrinsic interlayer bridge, a metal-assisted confined path, or
a CAFM hotspot. Device-level progressive switching does not by itself exclude
nanoscale localized paths. Patch states follow the SET/RESET limits of a bounded
two-state master equation:

```text
SET:   dphi_k/dt = g_k (1-phi_k)
RESET: dphi_k/dt = r_k (phi_min-phi_k)
g_k,r_k proportional to sinh(beta_s (|V_app|-V_th,k))_+
```

The kinetic `sinh` also has a direct h-BN modeling precedent: the supplementary
model of Chen et al. uses `Delta G approximately G^gamma sinh(beta V)` for
exponential ionic switching kinetics. This supports the effective rate form but
does not identify Ti, Ag, or a particular vacancy as the unique moving species.

The default `Theta(phi)=phi` is a first-order parallel-area mixture. It is not a
validated Bruggeman or critical-percolation law; `K=10` is a quadrature/coarse-
graining choice rather than evidence for ten physical channels. Extrapolating
the new sputtered-h-BN CAFM apparent density to 0.053 um2 gives about 10.3 spots,
so `K=10` passes an order-of-magnitude cross-check only; the CAFM and 1T1M films
are different samples and processes.

The use of `V_app` as the state-driving stress is a reduced-order assumption. A
fully self-consistent local-field model would require independent transistor and
internal-voltage constraints. Thermal feedback is optional and defaults to
`Rth=0`, because one room-temperature quasi-static data set cannot identify it.

The patch structure is frozen for one calibrated device; changing the cycle seed
resamples C2C noise without silently creating a different device. Cycle-to-cycle
threshold variability is a Weibull draw with weak AR(1) memory.
Endurance damage persists through RESET and changes SET threshold, residual HRS
state, and LRS conductance:

```text
D_(n+1) = D_n + kappa (|Delta phi_SET| + |Delta phi_RESET|)
V_th,SET = V_th,SET,0 + c_E D
phi_min = phi_floor + c_f D
G_on,eff = G_on exp(xi_on - c_g D)
```

## Calibration audit

Median across 12 independent 53-cycle C2C sequences, with one frozen device:

| feature | model | data |
|---|---:|---:|
| V_SET | 1.300 V, CV 25.4% | 1.298 V, CV 25.0% |
| V_RESET | -1.075 V, CV 24.5% | -1.075 V, CV 24.1% |
| R_HRS | 2.038e8 ohm, CV 44.6% | 2.045e8 ohm, CV 45.5% |
| R_LRS | 2.934e5 ohm, CV 28.6% | 2.939e5 ohm, CV 28.4% |
| I_cc | 5.145e-5 A, CV 1.0% | 5.146e-5 A, CV 1.0% |
| median memory window | 649 | 668 |

For the representative sequence nearest the ensemble median, full-loop
log-current RMSE is 0.146/0.106 decade for SET forward/return and 0.278/0.181
decade for RESET forward/return. The validation figure is
[10_vulcan_v3_validation.png](../analysis/figures/10_vulcan_v3_validation.png).

## Interpretation discipline

- Weibull thresholds, log-normal-like resistance, progressive patch switching,
  and independent SET/RESET draws are model assumptions or calibrated inputs.
- The low I_cc variability relative to resistance and threshold variability is
  a structural consequence of the series transistor, although its 1% floor is
  calibrated with `icc_noise`.
- The model should not be used to claim a uniquely identified TAT mechanism,
  filament geometry, Joule-heating strength, or pulse-time prediction.
- The present high-resistance, progressive 1T1M data are interpreted as a
  transistor-controlled soft-path regime. Nanoscale localized bridges remain
  possible, but a complete CNF/QPC should not be claimed without Ohmic low
  resistance, abrupt rupture, or conductance-quantization evidence.
- `Ea_set` and `Ea_reset` are unidentifiable placeholder values at the present
  single temperature, not experimentally extracted activation energies.

The paper-ready derivation, evidence grading, reading notes, and proposed
experiments are in
[LITERATURE_THEORY_REVIEW_2026-07-14.md](../analysis/LITERATURE_THEORY_REVIEW_2026-07-14.md).
The matching citation database is
[vulcan2d_theory.bib](../references/vulcan2d_theory.bib).
The advisor's proposed 1R/1T1R, CAFM, XTEM, and atomistic evidence chain is in
[PAPER_EVIDENCE_CHAIN_2026-07-14.md](../analysis/PAPER_EVIDENCE_CHAIN_2026-07-14.md).
The focused Zhu/Lanza literature audit and the resulting two-regime physical
interpretation are in
[ZHU_LANZA_HBN_PHYSICS_REVIEW_2026-07-15.md](../analysis/ZHU_LANZA_HBN_PHYSICS_REVIEW_2026-07-15.md).

Run the audit and tests with:

```bash
python3 -m vulcan2d.calibrate
python3 -m vulcan2d.validate
python3 -m unittest discover -s tests -v
```
