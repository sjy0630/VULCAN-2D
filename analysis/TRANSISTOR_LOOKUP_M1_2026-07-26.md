# Measured-transistor M1 audit — 2026-07-26

## Scope and data boundary

This audit uses the advisor-provided `Fig.2b.xlsx` standalone-transistor output
characteristics: six gate voltages (`0.5, 1.0, 1.5, 2.0, 2.5, 3.0 V`), each
with 202 measured points in a `0 -> 5 -> 0 V` `V_DS` sweep. The source SHA-256
is `4009913b0d35ae02f1030d25e2fadba5a9fdd2dcf830f050d3e59c41eb51a518`.

The workbook and derived `vulcan2d/transistor_lookup.json` contain unpublished
measurements and remain outside git. The repository contains only the generic
converter, interpolation code, tests, and this result summary.

## Implemented M1 seam

- `vulcan2d/transistor.py` validates and interpolates measured
  `I_D(V_DS,V_G)` curves.
- Ascending and descending sweeps are averaged on the ascending voltage grid;
  the origin is set to zero and a cumulative maximum ensures the load-line law
  is monotone. The raw JSON values remain unchanged.
- Gate interpolation is linear; extrapolation outside `0.5–3.0 V` or
  `0–5.0 V` is rejected.
- `model.i_tr`, `divider`, `run_sweep`, and `simulate_cycles` accept an optional
  measured lookup and explicit SET/RESET gate voltages.
- The calibrated empirical `tanh` transistor remains the default M0 path.
- Each cycle now exposes `Vhs` and `Vhr`, the self-consistent h-BN internal
  voltage traces.
- `analysis/audit_transistor_lookup.py` reproduces the M0/M1 comparison.

## Numerical result

One fixed 53-cycle sequence (`seed=110`) was evaluated. Only the SET-side
transistor was replaced because `Fig.2b` does not establish the RESET terminal
orientation or reverse-bias characteristic.

| model | assumed SET `V_G` | mean `V_SET` | mean `R_HRS` | mean `R_LRS` | mean `I_cc` | mean `V_h` at extracted `V_SET` | median window |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M0 empirical | — | 1.258 V | 197.7 MΩ | 289.5 kΩ | 51.47 µA | 0.842 V | 639 |
| M1 lookup | 0.8 V | 1.151 V | 197.6 MΩ | 215.9 kΩ | 51.85 µA | 1.110 V | 946 |
| M1 lookup | 1.1 V | 1.241 V | 197.6 MΩ | 206.7 kΩ | 91.04 µA | 1.191 V | 1013 |

The measured 53-cycle targets used by v0.3 are approximately
`R_LRS=293.9 kΩ`, `I_cc=51.46 µA`, and median window `668`.

## What the new data resolved

1. The empirical transistor equation is no longer the only available load-line
   representation. A measured and testable interface now exists.
2. The measured 1T curves directly constrain how current changes with `V_G` and
   `V_DS`; they can falsify the old assumption that changing compliance is just
   a scalar `I_sat` rescaling.
3. Internal h-BN voltage can now be exported and compared rather than remaining
   hidden inside the divider.

## What it falsified or exposed

At the literature/PPT operating point `V_G=1.1 V`, direct substitution predicts
about `91 µA`, not the observed `51 µA`, and lowers `R_LRS` to about `207 kΩ`.
Choosing an effective `V_G≈0.8 V` recovers `I_cc` but still misses `R_LRS`, the
memory window, the extracted switching voltage, and the internal voltage.

Therefore a single gate offset or current scale is not enough to make the
standalone 1T curve and the calibrated 1T1M loop mutually consistent. Likely
missing mappings include transistor W/L or device identity, source/body
potential, terminal orientation, series/contact resistance, the exact gate
voltage of the 53 cycles, and whether `DrainV` corresponds to the transistor
drop used in the 1T1M wiring.

The comparison also shows that changing the transistor law shifts mean
`V_h@V_SET` from `0.842 V` to `1.191 V`, while the state kinetics still use
applied voltage. The measured lookup therefore makes local-field kinetics
possible, but the existing threshold and rate parameters cannot simply be
reused: they were calibrated on `V_app` and an implicit per-voltage-step time.

## Current bottlenecks, in order

1. **1T-to-1T1M electrical mapping:** exact transistor identity/W/L, terminal
   convention, body/source connection, 53-cycle `V_G`, and any series/contact
   resistance.
2. **Time axis:** point timestamps or source/measure delay, sweep rate, dwell,
   and SET/RESET waiting time. “Seconds per DC cycle” does not identify the
   `ms–us` pulse kinetics.
3. **RESET transistor law:** reverse-bias or actual RESET-orientation
   `I_D–V_DS–V_G` curves and RESET gate voltage.
4. **Multi-gate raw 1T1M loops:** the PPT gives state-count summaries and Origin
   graph objects, not the point-level curves needed for held-out validation.
5. **Geometry and D2D evidence:** area, h-BN thickness/layers, device mapping,
   and multiple devices are needed before `K`, `w_k`, or the mixing exponent can
   be tied to physical structure.

## Next model gate

Do not recalibrate M1 to the current 53 loops yet. First resolve bottleneck 1
and import the 1500 raw files. Then:

1. hold the transistor lookup fixed;
2. solve and store `V_h(t)`;
3. refit only the state threshold/rate layer against a subset of gate/time
   conditions;
4. test on a held-out `V_G`, sweep rate, or device.

Until that gate is passed, M0 remains the reproducible quasi-static baseline
and M1 is an experimental, partially constrained branch.

