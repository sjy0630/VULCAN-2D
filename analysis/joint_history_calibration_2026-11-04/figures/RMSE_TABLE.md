# Saved-result RMSE comparison

All values are log10-current RMSE in decades, on the original eligible masks. Each phase uses its full native waveform and actual incoming state.

| Phase | Role | Scored points | Old Radau control | Joint refined | Joint low-bias RMSE (20 points) |
|---|---|---:|---:|---:|---:|
| SET4 | Calibration | 500 | 0.133894931 | 0.235946536 | 0.277882748 |
| RESET4 | Calibration | 170 | 0.705457872 | 0.647167951 | 0.809666102 |
| SET5 | Calibration | 500 | 0.524327189 | 0.400056442 | 0.815945677 |
| RESET5 | Training continuation | 170 | Not simulated | 0.504698758 | 0.604568193 |
| SET6 | Training continuation | 500 | Not simulated | 0.363253187 | 0.789163441 |

Calibration only: equal-phase mean squared error (the selection objective): 0.263505888 → 0.211514094 decade².
Calibration only: point-weighted 1170-point RMSE: 0.444364008 → 0.391205067 decade.

SET4 accuracy worsens while RESET4 and SET5 improve. This is a limited compromise from bounded calibration, not completion of the unified 1R/1T1R model or identification of a microscopic mechanism.
RESET5 and SET6 did not enter candidate selection. No old continuation baseline or combined five-phase objective is invented.
