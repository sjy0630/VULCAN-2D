# VULCAN-2D v0.4: XTEM-constrained soft/hard SET-RESET model

## Status

v0.4 is the research model built beside the already calibrated v0.3 desktop
engine. It now uses all supplied electrical ensembles:

- 1T1R SET and RESET at `V_G=0.9, 1.1, 1.3 V`;
- standalone 1R SET with the confirmed 10 mA SMU compliance;
- standalone 1R RESET from the hard low-resistance state;
- the Fig. 1h cross-sectional TEM geometry from Zhu et al., Nature 2023.

It adds four capabilities that v0.3 did not have:

1. explicit gate-voltage dependence on both SET and RESET;
2. continuous SET-to-RESET state transfer instead of reinitializing RESET;
3. an XTEM-thickness-normalized internal h-BN field;
4. standalone hard-path formation and distributed hard-link rupture.

Run the full audit with:

```bash
python -m vulcan2d.validate_v4
```

The desktop application's default endpoint still calls v0.3. Promotion of v0.4
to that endpoint should happen only after the research validation and protocol
assumptions below are accepted.

## Evidence and what it constrains

### Electrical ensembles

| condition | cycles | direct median constraints |
|---|---:|---|
| 1T1R SET, 0.9/1.1/1.3 V | 100/53/101 | apex current 36.912/51.46/93.304 uA |
| 1T1R RESET, 0.9/1.1/1.3 V | 100/53/101 | per-cycle peak current 0.425/3.774/21.524 uA |
| standalone 1R SET | 22 | hard crossing near 2.70 V; 10 mA SET clamp |
| standalone 1R RESET | 22 | initial resistance 95.5 ohm; peak 4.96 mA near -0.51 V |

The measurement protocol has now been confirmed: standalone 1R SET uses a
10 mA instrument compliance, whereas standalone 1R RESET has no user-set
compliance. v0.4 therefore clips the SET branch at 10 mA but does not clip the
RESET branch. The finite current range of the analyser is not treated as an
intrinsic device limit.

### Fig. 1h XTEM constraint

The Fig. 1h inset is a 20 nm x 16 nm cross-sectional TEM image of the
Au-Ti-h-BN-W memristor. The paper reports approximately 18 h-BN layers and a
thickness of approximately 6 nm; the tungsten via is approximately 260 nm in
diameter and the device area is at most 0.053 um2. The image confirms the
as-fabricated layered stack and interfaces.

It does **not** show the same device in virgin, LRS and HRS states, and it does
not provide a Ti/Au/B elemental map of a switched path. Therefore it cannot by
itself identify a metal filament or distinguish metal-ion motion from vacancy
reconfiguration.

`K=10` in the model remains the number of lateral coarse-grained weak-region
quadrature points. It is not changed to 18: the TEM layer count is a vertical
geometry constraint, whereas `K` represents lateral statistical heterogeneity.

## Model equations

### 1T1R circuit and XTEM-normalized field

```text
V_app = V_hBN + V_DS
I_hBN(V_hBN, phi) = I_tr(V_DS, V_G, polarity)
E_hBN [MV/cm] = 10 |V_hBN| / d_hBN[nm]
```

The kinetic drive is expressed as an equivalent voltage at the 6 nm reference
thickness:

```text
V_drive = [0.8 |V_hBN| + 0.2 |V_app|] (6 nm / d_hBN)
```

At the measured default `d_hBN=6 nm`, this preserves the calibrated voltage
scale. Changing the thickness now changes the switching drive through the
average field, rather than treating 3 nm and 12 nm devices as identical.

The 20% terminal-stress term remains a reduced-order local-field correction.
The XTEM image does not independently determine this coefficient.

### Distributed soft state and state continuity

`phi_k` is the connectivity/occupation of a local high-transmission
configuration. Candidate realizations include intrinsic vacancy/inter-layer
bridges and confined metal-assisted paths, but `phi_k` is not a measured defect
concentration.

SET drives `phi_k` upward and RESET drives it toward the residual floor. The
final SET vector is now passed directly into the following RESET sweep:

```text
phi_initial_RESET = phi_final_SET
```

SET and RESET thresholds remain separate random variables because the supplied
electrical data do not identify one shared atomistic barrier.

For RESET, the connected and off-state currents use separate effective field
coefficients:

```text
I_hBN = G_off sinh(alpha_off(V_G) V_hBN)
      + G_on  sinh(alpha_on(V_G)  V_hBN)
```

This split is required by the measured low-voltage initial state, peak-collapse
shape and HRS return branch. The gate-dependent factors are terminal/circuit
parameters; they must not be presented as independently measured intrinsic
h-BN conductivities.

The low-current state written at 0.9 V is assigned a lower effective RESET
stability threshold. This is supported by its much smaller initial RESET
current, but it is not a second measured activation energy.

### Grid-independent voltage-domain exposure

```text
exposure = |Delta V| / 0.02 V
```

This removes false dependence on the 0.01, 0.02 and 0.025 V spreadsheet grids.
It does not supply real time, pulse width or dwell time.

### Standalone 1R hard SET and hard-link RESET

The standalone SET branch remains:

```text
I_pre = I0_1R sinh(alpha_1R V)
h_SET = max[h_SET, sigmoid((|V|-V_hard)/delta_V)]
I = min((1-h_SET) I_pre + h_SET |V|/R_hard, Icomp_SET)
```

The RESET data independently constrain `R_hard` to approximately 96 ohm. A
small ensemble of coarse hard links is then ruptured at distributed negative
thresholds:

```text
G_hard(V) = sum_j w_j h_j(V) / R_hard
I_RESET = I_HRS(V) + |V| G_hard(V)
h_j(V) decreases irreversibly when |V| crosses Vrupture_j
```

Each `h_j` is a complete coarse path through the layered stack, not one h-BN
layer and not automatically one atomic or metallic filament. Multiple links are
the minimum structure needed to reproduce the staged conductance collapse.

The measured terminal peak power is about 2.58 mW. This makes Joule-heating
assistance plausible, but without time- or temperature-dependent measurements
v0.4 reports power diagnostically and does not fit a thermal resistance or an
activation energy from it.

## Validation snapshot

The generated figure is
`analysis/figures/12_vulcan_v4_multiregime_validation.png`; exact distributions
and RMSE values are in `analysis/v4_validation_summary.json`.

### 1T1R RESET median-curve comparison

| V_G | initial LRS at -0.2 V, data/model | return HRS at -0.2 V, data/model | peak V, data/model | forward/return RMSE |
|---:|---:|---:|---:|---:|
| 0.9 V | 0.00300/0.00445 uA | 0.0108/0.0105 nA | -1.49/-1.50 V | 0.495/0.364 decade |
| 1.1 V | 0.361/0.388 uA | 1.163/0.936 nA | -0.98/-0.98 V | 0.219/0.427 decade |
| 1.3 V | 0.960/0.971 uA | 1.842/1.852 nA | -1.12/-1.15 V | 0.194/0.292 decade |

Per-cycle peak medians and the peak of the ensemble-median curve are both kept
in the JSON report. They differ when individual cycles peak at different
voltages; they must not be mixed in one comparison.

### Standalone 1R RESET comparison

| quantity | data | v0.4 |
|---|---:|---:|
| initial resistance at -0.2 V | 95.5 ohm | 109 ohm |
| peak current | 4.96 mA | 4.43 mA |
| median-curve peak voltage | -0.50 V | -0.475 V |
| return HRS current at -0.2 V | 78.1 pA | 49.4 pA |
| terminal peak power | 2.58 mW | 2.12 mW |

The hard RESET feature anchors agree, but the full forward/return loop RMSE is
0.901/0.599 decade. The remaining error is visible: six smooth coarse links do
not reproduce every measured plateau and re-rise. Adding more hidden states
merely to reduce RMSE is not justified until the RESET protocol and structural
state evidence are available.

## Literature mapping and claim boundary

The working-regime separation follows Zhu et al., Nature 2023: a CMOS
transistor suppresses overshoot and produces progressive high-resistance 1T1M
switching, whereas sufficiently high-current standalone devices can enter a
hard filamentary regime. CAFM studies support spatially localized weak regions;
DFT/NEGF supports field-reconfigurable vacancy/inter-layer bridges; electrode
dependence supports possible metal assistance.

The evidence hierarchy is:

1. electrical data identify soft versus hard device-level regimes and constrain
   their terminal state transitions;
2. Fig. 1h XTEM fixes the approximate layer count, thickness and stack geometry;
3. CAFM supports lateral non-uniformity and localized weak regions;
4. vacancy and metal-ion literature supplies competing microscopic candidates;
5. state-resolved XTEM plus EDS/EELS is still required to identify the switched
   path chemistry in the target Au-Ti-h-BN-W device.

v0.4 therefore does not claim:

- that the Fig. 1h inset images a conductive filament;
- that `phi_k` is a boron-vacancy concentration;
- that each hard link is a metal filament or one atomic layer;
- a QPC/Landauer law without quantized-conductance evidence;
- real pulse-time, dwell-time, temperature or activation-energy prediction;
- unique separation of metal-ion and vacancy populations from port I-V alone.
