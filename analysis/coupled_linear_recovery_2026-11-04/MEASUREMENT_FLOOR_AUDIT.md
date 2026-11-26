# Measurement-floor evidence audit

2026-11-04. **There is insufficient evidence to attribute the q1 rate ×3 bare-RESET tail mismatch to a measurement floor.** This audit changes no model, masks, thresholds, scores or feature gates. It uses local cached paper/SI and only the 184 named original training voltage/current arrays; no new model evaluation or holdout decoding was performed.

## What the paper establishes

Zhu et al., *Nature* 618, 57–62 (2023), DOI [10.1038/s41586-023-05973-1](https://www.nature.com/articles/s41586-023-05973-1), Methods / Characterization states that DC I–V measurements use **Keithley 4200 in ramped voltage sweep mode**. Pulse measurements use Keysight B1500A with two WGFMUs; the Agilent E3631A supplies gate bias for those pulse measurements. The probe stations are Cascade M150.

Locator: `../verification/2026-09-20/physics-sources/zhu2023-nature.txt:925`, relative to the repository root. The cached original HTML and SI PDF hashes match their source manifest. The reviewed text contains no explicit acquisition-current range, current noise-floor value, current resolution, particular SMU card/preamp configuration, or integration/dwell settings for these local DC records. There is no located statement classifying bare RESET return currents of 10⁻¹¹–10⁻¹⁰ A as measurement noise.

The reported bare-device compliance of 1 mA or higher is a current limit, not the low-current measurement range or resolution. The SI's “limited resolution” at text line 523 refers to identifying data points in published endurance plots, not current-meter resolution. The main text's temporal-resolution statement at line 1426 concerns pulse switching. The cited 2600B SMU datasheet does not establish that the DC curves used that model, and a generic instrument capability would not establish the floor of this acquisition. The local-workbook-to-public-figure mapping remains incomplete, as already recorded in `NATURE_1R_1T1R_PROVENANCE_2026-11-03.md`.

## Training-data clues and their limits

All stored current values in the reviewed 1R and 1T1R training arrays are positive, including negative-voltage RESET. The existing inventory explicitly identifies bare-R currents as **magnitudes**. Therefore the arrays do not support a signed, zero-centered noise test. Assigning `sign(V) × abs(I)` reconstructs a passive convention; it does not recover the instrument's measured sign.

The table uses each record's first and last zero-voltage entry once. Duplicate terminal zero entries in the series records are retained in the JSON but are not counted as independent measurements.

| Training block | Records | Initial V=0 current median (A) | Final V=0 current median (A) |
|---|---:|---:|---:|
| Bare SET | 14 | 1.480e-13 | 4.702e-5 |
| Bare RESET | 14 | 2.796e-5 | 4.896e-13 |
| Series SET | 32 | 2.230e-10 | 5.482e-10 |
| Series RESET | 32 | 3.387e-10 | 2.217e-10 |

For bare RESET, the 14 final zero-voltage values span **2.661e-13–1.747e-12 A**. At return |V|=0.2 V, the measured magnitudes span **1.639e-11–1.087e-9 A**, with median **7.674e-11 A**. Each record's 0.2-V return value is **31.7–2211.6 times** its own final zero-voltage magnitude; the median ratio is **170.3**. These values do not support equating the entire 0.2-V tail with the zero-voltage baseline. They also do not prove every tail point is free of measurement artifacts.

Zero-bias endpoints are history dependent, and the large LRS endpoint currents make pooling all endpoints as a single noise distribution inappropriate. They are not repeated blank measurements, and unknown range switching, offsets or transient contributions cannot be separated here. The roughly 2.2e-10 A series endpoint level is an offset/floor **clue**, not a calibrated floor and not transferable to the bare acquisition. No acquisition timing is inferred.

## Consequence for the q1 rate ×3 control

The saved ×3 candidate predicts a bare RESET return I(0.2 V) median of **8.834e-16 A**, versus measured **7.674e-11 A**: the ratio of these medians is **86,874**, or **4.939 decades**. Its original feature gate fails. A model prediction below a possible instrument floor is not itself evidence that the observed tail is floor dominated.

Keep that mismatch and the acquisition uncertainty separate. There is currently no measured uncertainty model supporting censoring, dropping these points, increasing the scoring floor, or claiming the ×3 candidate explains the tail. Useful missing evidence would be the actual current-range/SMU settings, genuine signed exports and comparable blank or repeated-baseline measurements. The absence of those details in the saved paper/SI does not establish that no laboratory record exists.

`MEASUREMENT_FLOOR_AUDIT.json` contains all 92 per-record zero endpoints, low-voltage stored-sign counts and return values, training-only decoded keys, source hashes and exact numerical summaries. The original masks and all model outputs remain unchanged.
