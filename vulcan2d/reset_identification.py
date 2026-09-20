"""Inverse identification of the observable 1T1M RESET state.

The measured terminal pair ``(V_app, I)`` is first de-embedded with the
independent negative-``V_DS`` transistor lookup.  Two stable transport
envelopes are then identified for every cycle:

``I_LRS(V_h) = G_LRS sinh(alpha_LRS V_h)``
``I_HRS(V_h) = G_HRS sinh(alpha_HRS V_h)``

The electrically observable state is the remaining fraction of resettable
conductance between those envelopes.  It is a device-level state and must not
be interpreted as a measured vacancy concentration or filament radius.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .transistor import TransistorMapping


@dataclass(frozen=True)
class ReconstructedReset:
    gate_voltage_v: float
    voltage_app_v: np.ndarray
    current_a: np.ndarray
    voltage_ds_v: np.ndarray
    voltage_hbn_v: np.ndarray
    valid: np.ndarray
    apex_index: int


@dataclass(frozen=True)
class IdentifiedResetState:
    gate_voltage_v: float
    field_mvc: np.ndarray
    state_raw: np.ndarray
    state_monotone: np.ndarray
    g_lrs_a: float
    g_hrs_a: float
    e90_mvc: float
    e50_mvc: float
    e10_mvc: float
    reconstruction_valid_fraction: float


@dataclass(frozen=True)
class IdentifiedPFResetState:
    """RESET state between a sinh LRS and effective-PF HRS envelope."""

    gate_voltage_v: float
    field_mvc: np.ndarray
    state_raw: np.ndarray
    state_monotone: np.ndarray
    g_lrs_a: float
    a_hrs: float
    e50_mvc: float
    width_mvc: float
    state_fit_rms: float
    reconstruction_valid_fraction: float


@dataclass(frozen=True)
class PFResetTransitionLaw:
    """Protocol-level RESET survival law identified with PF HRS transport."""

    e50_mvc: float
    width_mvc: float
    e50_sigma_mvc: float
    width_sigma_mvc: float
    residual_rms_state: float
    gate_voltage_v: float
    protocol_label: str
    n_training_cycles: int
    rejected_cycles: tuple[str, ...] = ()
    reset_peak_abs_v: float = 1.7
    reset_peak_tolerance_v: float = 0.03

    def validate_gate(self, gate_voltage):
        gate = float(gate_voltage)
        if not np.isfinite(gate) or not np.isclose(
                gate, self.gate_voltage_v, rtol=0.0, atol=1e-9):
            raise ValueError("RESET transition gate voltage does not match fit")

    def validate_protocol(self, reset_voltage):
        voltage = np.asarray(reset_voltage, dtype=float)
        finite = voltage[np.isfinite(voltage)]
        if finite.ndim != 1 or len(finite) < 3:
            raise ValueError("RESET protocol must contain finite voltage points")
        peak = float(np.max(np.abs(finite)))
        if (not np.isclose(peak, self.reset_peak_abs_v, rtol=0.0,
                           atol=self.reset_peak_tolerance_v)
                or np.min(finite) >= 0.0):
            raise ValueError("RESET peak does not match transition fit")


def reconstruct_reset(voltage, current, gate_voltage, mapping,
                      hbn_thickness_nm=6.0):
    """De-embed the transistor voltage from one measured RESET loop.

    Magnitudes are used because the stored circuit solver is magnitude-domain.
    Points that exceed the measured transistor domain or imply ``V_DS>V_app``
    are retained as invalid rather than silently clipped.
    """
    voltage = np.asarray(voltage, dtype=float)
    current = np.asarray(current, dtype=float)
    if voltage.ndim != 1 or current.ndim != 1 or len(voltage) != len(current):
        raise ValueError("voltage and current must be same-length vectors")
    vapp = np.abs(voltage)
    imag = np.abs(current)
    vds = np.full_like(vapp, np.nan)
    valid = np.isfinite(vapp) & np.isfinite(imag)
    for idx in np.flatnonzero(valid):
        try:
            vds[idx] = mapping.voltage_for_current(
                imag[idx], gate_voltage, side="reset")
        except ValueError:
            valid[idx] = False
    vh = vapp - vds
    valid &= np.isfinite(vh) & (vh >= -2e-6)
    vh[valid] = np.maximum(vh[valid], 0.0)
    vh[~valid] = np.nan
    apex = int(np.nanargmax(vapp))
    return ReconstructedReset(
        gate_voltage_v=float(gate_voltage),
        voltage_app_v=vapp,
        current_a=imag,
        voltage_ds_v=vds,
        voltage_hbn_v=vh,
        valid=valid,
        apex_index=apex,
    )


def _log_sinh(alpha, voltage):
    value = np.sinh(np.clip(float(alpha) * np.asarray(voltage), 0.0, 39.0))
    return np.log10(np.clip(value, 1e-300, None))


def _branch_mask(cycle, branch, low_v=0.08, high_v=None):
    index = np.arange(len(cycle.voltage_hbn_v))
    if branch == "lrs":
        side = index <= cycle.apex_index
        high_v = 0.35 if high_v is None else high_v
    elif branch == "hrs":
        side = index >= cycle.apex_index
        # The return branch is already RESET and can constrain the complete
        # HRS non-linearity over a wider field interval than the initial LRS.
        high_v = 1.20 if high_v is None else high_v
    else:
        raise ValueError("branch must be 'lrs' or 'hrs'")
    return (cycle.valid & side &
            (cycle.voltage_hbn_v >= low_v) &
            (cycle.voltage_hbn_v <= high_v) &
            (cycle.current_a > 1e-13))


def fit_shared_alpha(cycles, branch, bounds=(1.0, 15.0)):
    """Identify one field coefficient while allowing per-cycle amplitudes.

    The objective uses only within-cycle residuals, so cycle-to-cycle LRS/HRS
    magnitude variation does not masquerade as a different field coefficient.
    """
    from scipy.optimize import minimize_scalar

    def objective(alpha):
        residuals = []
        for cycle in cycles:
            mask = _branch_mask(cycle, branch)
            if np.count_nonzero(mask) < 5:
                continue
            log_g = (np.log10(cycle.current_a[mask]) -
                     _log_sinh(alpha, cycle.voltage_hbn_v[mask]))
            residuals.extend(log_g - np.median(log_g))
        if not residuals:
            return np.inf
        return float(np.median(np.abs(residuals)))

    fit = minimize_scalar(objective, bounds=bounds, method="bounded")
    if not fit.success or not np.isfinite(fit.fun):
        raise RuntimeError("Could not identify RESET transport coefficient")
    return float(fit.x), float(fit.fun)


def _isotonic_decreasing(values, weights=None):
    """Small pool-adjacent-violators implementation for a decreasing state."""
    y = np.asarray(values, dtype=float)
    if weights is None:
        weights = np.ones_like(y)
    weights = np.asarray(weights, dtype=float)
    # Increasing PAVA on -y.
    blocks = []
    for idx, (value, weight) in enumerate(zip(-y, weights)):
        blocks.append([idx, idx + 1, float(value), float(weight)])
        while len(blocks) >= 2 and blocks[-2][2] > blocks[-1][2]:
            left, right = blocks[-2], blocks[-1]
            total = left[3] + right[3]
            mean = (left[2] * left[3] + right[2] * right[3]) / total
            blocks[-2:] = [[left[0], right[1], mean, total]]
    out = np.empty_like(y)
    for start, stop, mean, _ in blocks:
        out[start:stop] = -mean
    return out


def _crossing(field, state, level):
    good = np.isfinite(field) & np.isfinite(state)
    field = np.asarray(field)[good]
    state = np.asarray(state)[good]
    if len(field) < 2 or np.nanmin(state) > level or np.nanmax(state) < level:
        return np.nan
    # State is decreasing; reverse arrays so interpolation sees an increasing x.
    return float(np.interp(level, state[::-1], field[::-1]))


def identify_reset_state(cycle, alpha_lrs, alpha_hrs,
                         hbn_thickness_nm=6.0):
    """Return the observable, monotone RESET state for one reconstructed loop."""
    lrs = _branch_mask(cycle, "lrs")
    hrs = _branch_mask(cycle, "hrs")
    if np.count_nonzero(lrs) < 5 or np.count_nonzero(hrs) < 5:
        raise ValueError("Insufficient stable LRS/HRS points")
    g_lrs = 10.0 ** float(np.median(
        np.log10(cycle.current_a[lrs]) -
        _log_sinh(alpha_lrs, cycle.voltage_hbn_v[lrs])
    ))
    g_hrs = 10.0 ** float(np.median(
        np.log10(cycle.current_a[hrs]) -
        _log_sinh(alpha_hrs, cycle.voltage_hbn_v[hrs])
    ))
    if not np.isfinite(g_lrs) or not np.isfinite(g_hrs) or g_lrs <= g_hrs:
        raise ValueError("RESET cycle does not have separable LRS and HRS envelopes")

    stop = cycle.apex_index + 1
    vh = cycle.voltage_hbn_v[:stop]
    current = cycle.current_a[:stop]
    i_lrs = g_lrs * np.sinh(np.clip(alpha_lrs * vh, 0.0, 39.0))
    i_hrs = g_hrs * np.sinh(np.clip(alpha_hrs * vh, 0.0, 39.0))
    denominator = i_lrs - i_hrs
    valid = (cycle.valid[:stop] & np.isfinite(denominator) &
             (denominator > 1e-15) & (vh >= 0.04))
    raw = np.full(stop, np.nan)
    raw[valid] = np.clip(
        (current[valid] - i_hrs[valid]) / denominator[valid], 0.0, 1.0)

    monotone = np.full(stop, np.nan)
    idx = np.flatnonzero(valid)
    if len(idx) < 8:
        raise ValueError("Too few points to identify a RESET trajectory")
    fitted = _isotonic_decreasing(raw[idx])
    # Normalize small envelope-fit offsets without changing the transition shape.
    early = fitted[:max(3, len(fitted) // 10)]
    late = fitted[-max(3, len(fitted) // 10):]
    top = float(np.median(early))
    bottom = float(np.median(late))
    if top - bottom < 0.05:
        raise ValueError("Observable RESET state has insufficient dynamic range")
    fitted = np.clip((fitted - bottom) / (top - bottom), 0.0, 1.0)
    monotone[idx] = fitted
    field = 10.0 * vh / float(hbn_thickness_nm)
    return IdentifiedResetState(
        gate_voltage_v=cycle.gate_voltage_v,
        field_mvc=field,
        state_raw=raw,
        state_monotone=monotone,
        g_lrs_a=float(g_lrs),
        g_hrs_a=float(g_hrs),
        e90_mvc=_crossing(field, monotone, 0.9),
        e50_mvc=_crossing(field, monotone, 0.5),
        e10_mvc=_crossing(field, monotone, 0.1),
        reconstruction_valid_fraction=float(np.mean(cycle.valid)),
    )


def logistic_state(field_mvc, e50_mvc, width_mvc):
    """Irreversible-ramp survival curve for the resettable conductance."""
    z = np.clip((np.asarray(field_mvc) - float(e50_mvc)) /
                max(float(width_mvc), 1e-9), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(z))


def _fit_pf_transition(field, state):
    """Fit one decreasing logistic trajectory in observable-state space."""
    from scipy.optimize import minimize

    field = np.asarray(field, dtype=float)
    state = np.asarray(state, dtype=float)
    valid = np.isfinite(field) & np.isfinite(state)
    field = field[valid]
    state = state[valid]
    if (len(field) < 8 or np.max(state) - np.min(state) < 0.5
            or np.max(state) < 0.7 or np.min(state) > 0.3):
        raise ValueError("Observable RESET state has insufficient dynamic range")
    e50_initial = float(np.interp(0.5, state[::-1], field[::-1]))
    span = max(float(np.ptp(field)), 0.1)

    def objective(values):
        predicted = logistic_state(field, values[0], values[1])
        return float(np.mean((predicted - state) ** 2))

    fit = minimize(
        objective, [e50_initial, max(span / 12.0, 0.02)],
        method="Nelder-Mead",
        bounds=[(0.0, max(float(np.max(field)) * 2.0, 0.5)),
                (max(span / 2000.0, 0.002), span)],
        options={"maxiter": 1000, "xatol": 1e-9, "fatol": 1e-12},
    )
    if not fit.success or not np.isfinite(fit.fun):
        raise RuntimeError("Could not fit RESET transition")
    return float(fit.x[0]), float(fit.x[1]), float(np.sqrt(fit.fun))


def identify_pf_reset_state(cycle, alpha_lrs, pf_law,
                            hbn_thickness_nm=6.0):
    """Identify RESET state without forcing per-cycle zero/one endpoints."""
    # Local import avoids a module cycle: hrs_transport uses _branch_mask.
    from .hrs_transport import EffectivePooleFrenkelLaw, effective_pf_basis

    if not isinstance(pf_law, EffectivePooleFrenkelLaw):
        raise TypeError("pf_law must be an EffectivePooleFrenkelLaw")
    lrs = _branch_mask(cycle, "lrs")
    hrs = _branch_mask(cycle, "hrs")
    if np.count_nonzero(lrs) < 5 or np.count_nonzero(hrs) < 5:
        raise ValueError("Insufficient stable LRS/HRS points")
    g_lrs = 10.0 ** float(np.median(
        np.log10(cycle.current_a[lrs])
        - _log_sinh(alpha_lrs, cycle.voltage_hbn_v[lrs])
    ))
    hrs_basis = effective_pf_basis(
        cycle.voltage_hbn_v[hrs], pf_law.beta_per_sqrt_v)
    a_hrs = 10.0 ** float(np.median(
        np.log10(cycle.current_a[hrs]) - np.log10(hrs_basis)
    ))
    if (not np.isfinite(g_lrs) or not np.isfinite(a_hrs)
            or g_lrs <= 0.0 or a_hrs <= 0.0):
        raise ValueError("Invalid RESET envelope amplitude")

    stop = cycle.apex_index + 1
    voltage = cycle.voltage_hbn_v[:stop]
    current = cycle.current_a[:stop]
    valid = (cycle.valid[:stop] & np.isfinite(voltage)
             & (voltage >= 0.04))
    index = np.flatnonzero(valid)
    voltage_valid = voltage[index]
    i_lrs = g_lrs * np.sinh(np.clip(
        float(alpha_lrs) * voltage_valid, 0.0, 39.0))
    i_hrs = a_hrs * effective_pf_basis(
        voltage_valid, pf_law.beta_per_sqrt_v)
    denominator = i_lrs - i_hrs
    separable = np.isfinite(denominator) & (denominator > 1e-15)
    index = index[separable]
    i_lrs = i_lrs[separable]
    i_hrs = i_hrs[separable]
    denominator = denominator[separable]
    if len(index) < 8:
        raise ValueError("Too few separable points for RESET state")

    raw_values = np.clip(
        (current[index] - i_hrs) / denominator, 0.0, 1.0)
    monotone_values = np.clip(
        _isotonic_decreasing(raw_values), 0.0, 1.0)
    field_values = 10.0 * voltage[index] / float(hbn_thickness_nm)
    e50, width, rms = _fit_pf_transition(field_values, monotone_values)
    raw = np.full(stop, np.nan)
    monotone = np.full(stop, np.nan)
    field = 10.0 * voltage / float(hbn_thickness_nm)
    raw[index] = raw_values
    monotone[index] = monotone_values
    return IdentifiedPFResetState(
        gate_voltage_v=cycle.gate_voltage_v,
        field_mvc=field,
        state_raw=raw,
        state_monotone=monotone,
        g_lrs_a=float(g_lrs),
        a_hrs=float(a_hrs),
        e50_mvc=e50,
        width_mvc=width,
        state_fit_rms=rms,
        reconstruction_valid_fraction=float(np.mean(cycle.valid)),
    )


def fit_pf_reset_transition_law(
        named_cycles, alpha_lrs, pf_law, gate_voltage, protocol_label,
        reset_peak_abs_v=1.7):
    """Fit a target-excludable 1.1-V RESET transition distribution."""
    states = []
    rejected = []
    for item in named_cycles:
        if len(item) == 2:
            name, cycle = item
        else:
            name, cycle = str(len(states) + len(rejected) + 1), item
        try:
            states.append(identify_pf_reset_state(
                cycle, alpha_lrs, pf_law))
        except (ValueError, RuntimeError) as error:
            rejected.append(f"{name}: {error}")
    if len(states) < 3:
        raise ValueError("At least three identifiable RESET cycles are required")
    e50 = np.asarray([state.e50_mvc for state in states])
    width = np.asarray([state.width_mvc for state in states])
    residual = np.asarray([state.state_fit_rms for state in states])
    e50_center = float(np.median(e50))
    width_center = float(np.median(width))
    return PFResetTransitionLaw(
        e50_mvc=e50_center,
        width_mvc=width_center,
        e50_sigma_mvc=float(
            1.4826 * np.median(np.abs(e50 - e50_center))),
        width_sigma_mvc=float(
            1.4826 * np.median(np.abs(width - width_center))),
        residual_rms_state=float(np.median(residual)),
        gate_voltage_v=float(gate_voltage),
        protocol_label=str(protocol_label),
        n_training_cycles=len(states),
        rejected_cycles=tuple(rejected),
        reset_peak_abs_v=float(reset_peak_abs_v),
    )


def state_width_from_levels(e90_mvc, e10_mvc):
    """Convert 90-to-10 transition width to logistic scale."""
    if not np.isfinite(e90_mvc) or not np.isfinite(e10_mvc):
        return np.nan
    return float((e10_mvc - e90_mvc) / (2.0 * np.log(9.0)))


def load_reset_workbook(path):
    """Load one RESET workbook without modifying the recorded observations."""
    import pandas as pd

    frames = pd.read_excel(Path(path), sheet_name=None)
    curves = []
    for sheet, frame in frames.items():
        voltage = frame.iloc[:, 0].to_numpy(float)
        current = frame.iloc[:, 1].to_numpy(float)
        good = np.isfinite(voltage) & np.isfinite(current)
        curves.append((sheet, voltage[good], current[good]))
    return curves
