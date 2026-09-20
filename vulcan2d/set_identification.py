"""Identify a quasi-static SET law from the 1T1M voltage sweeps.

The positive SET branch is de-embedded with the literal measured
``I_D(V_DS, V_G)`` lookup from Nature Fig. 2b.  The requested 1.1-V gate lies
inside the measured 1.0--1.5-V interval and is directly interpolated.  No
extra affine gate correction is used: the older cross-gate correction shifts
1.1 V to about 0.896 V and makes the reconstructed h-BN voltage fold back
artificially.

Two polarity-specific compact envelopes are identified:

``I_pre = A_pre V_h exp(beta_pre sqrt(V_h))``
``I_connected = A_connected V_h ** n_connected``

The observable state ``x=(I-I_pre)/(I_connected-I_pre)`` is monotonized
without re-scaling its endpoints.  The envelopes already define the zero and
one states, so per-cycle endpoint normalization would force apparent complete
switching even when the data do not support it.

The SET driver is ``integral |V_hBN| dV_app`` with unit ``V^2``.  It is a
step-invariant voltage-domain exposure, not time, charge, or energy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .transistor import TransistorMapping, load_default_lookup


@dataclass(frozen=True)
class ReconstructedSet:
    gate_voltage_v: float
    voltage_app_v: np.ndarray
    current_a: np.ndarray
    voltage_ds_v: np.ndarray
    voltage_hbn_v: np.ndarray
    valid: np.ndarray
    apex_index: int


@dataclass(frozen=True)
class SetTransportLaw:
    """Shared SET envelope shapes and protocol-level amplitude centers."""

    beta_pre_per_sqrt_v: float
    connected_exponent: float
    log10_a_pre_ref: float
    log10_a_connected_ref: float
    pre_amplitude_sigma_decade: float
    connected_amplitude_sigma_decade: float
    pre_shape_rmse_decade: float
    connected_shape_rmse_decade: float
    gate_voltage_v: float
    protocol_label: str
    n_training_cycles: int
    set_peak_abs_v: float = 5.0
    set_peak_tolerance_v: float = 0.05

    def validate_gate(self, gate_voltage):
        gate = float(gate_voltage)
        if not np.isfinite(gate) or not np.isclose(
                gate, self.gate_voltage_v, rtol=0.0, atol=1e-9):
            raise ValueError("SET transport gate voltage does not match fit")

    def validate_protocol(self, voltage):
        values = np.asarray(voltage, dtype=float)
        finite = values[np.isfinite(values)]
        if finite.ndim != 1 or len(finite) < 3:
            raise ValueError("SET protocol must contain finite voltage points")
        peak = float(np.max(np.abs(finite)))
        if (not np.isclose(peak, self.set_peak_abs_v, rtol=0.0,
                           atol=self.set_peak_tolerance_v)
                or np.max(finite) <= 0.0):
            raise ValueError("SET peak does not match transport fit")

    def pre_amplitude(self, residual_offset_decade=0.0):
        return float(10.0 ** (
            self.log10_a_pre_ref + float(residual_offset_decade)))

    def connected_amplitude(self, residual_offset_decade=0.0):
        return float(10.0 ** (
            self.log10_a_connected_ref + float(residual_offset_decade)))


@dataclass(frozen=True)
class IdentifiedSetState:
    gate_voltage_v: float
    exposure_v2: np.ndarray
    state_raw: np.ndarray
    state_monotone: np.ndarray
    a_pre: float
    a_connected: float
    e10_v2: float
    e50_v2: float
    e90_v2: float
    width_v2: float
    state_fit_rms: float
    reconstruction_valid_fraction: float


@dataclass(frozen=True)
class SetTransitionLaw:
    """Protocol-specific logistic SET state versus voltage exposure."""

    e50_v2: float
    width_v2: float
    e50_sigma_v2: float
    width_sigma_v2: float
    residual_rms_state: float
    gate_voltage_v: float
    protocol_label: str
    n_training_cycles: int
    rejected_cycles: tuple[str, ...] = ()
    set_peak_abs_v: float = 5.0
    set_peak_tolerance_v: float = 0.05

    def validate_gate(self, gate_voltage):
        gate = float(gate_voltage)
        if not np.isfinite(gate) or not np.isclose(
                gate, self.gate_voltage_v, rtol=0.0, atol=1e-9):
            raise ValueError("SET transition gate voltage does not match fit")

    def validate_protocol(self, voltage):
        values = np.asarray(voltage, dtype=float)
        finite = values[np.isfinite(values)]
        if finite.ndim != 1 or len(finite) < 3:
            raise ValueError("SET protocol must contain finite voltage points")
        peak = float(np.max(np.abs(finite)))
        if (not np.isclose(peak, self.set_peak_abs_v, rtol=0.0,
                           atol=self.set_peak_tolerance_v)
                or np.max(finite) <= 0.0):
            raise ValueError("SET peak does not match transition fit")

    def state(self, exposure_v2):
        return set_transition_state(
            exposure_v2, self.e50_v2, self.width_v2)

    def state_with_e50(self, exposure_v2, e50_v2):
        """Evaluate the shared-width law at a selected ensemble threshold."""
        return set_transition_state(exposure_v2, e50_v2, self.width_v2)


def direct_set_transistor_mapping(positive_lookup=None):
    """Return the literal positive-``V_DS`` Nature Fig. 2b mapping."""
    lookup = load_default_lookup() if positive_lookup is None else positive_lookup
    return TransistorMapping(
        lookup=lookup,
        gate_gain=1.0,
        gate_offset_v=0.0,
        current_scale=1.0,
        series_resistance_ohm=0.0,
    )


def reconstruct_set(voltage, current, gate_voltage, mapping=None):
    """De-embed the transistor voltage from one measured SET loop."""
    mapping = direct_set_transistor_mapping() if mapping is None else mapping
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
                imag[idx], gate_voltage, side="set")
        except ValueError:
            valid[idx] = False
    vh = vapp - vds
    valid &= np.isfinite(vh) & (vh >= -2e-6)
    vh[valid] = np.maximum(vh[valid], 0.0)
    vh[~valid] = np.nan
    return ReconstructedSet(
        gate_voltage_v=float(gate_voltage),
        voltage_app_v=vapp,
        current_a=imag,
        voltage_ds_v=vds,
        voltage_hbn_v=vh,
        valid=valid,
        apex_index=int(np.nanargmax(vapp)),
    )


def effective_pre_basis(voltage_hbn_v, beta_per_sqrt_v):
    voltage = np.asarray(voltage_hbn_v, dtype=float)
    if np.any(~np.isfinite(voltage)) or np.any(voltage < 0.0):
        raise ValueError("voltage_hbn_v must be finite and non-negative")
    beta = float(beta_per_sqrt_v)
    if not np.isfinite(beta) or beta < 0.0:
        raise ValueError("beta_per_sqrt_v must be finite and non-negative")
    value = voltage * np.exp(np.clip(beta * np.sqrt(voltage), 0.0, 39.0))
    return float(value) if value.ndim == 0 else value


def connected_basis(voltage_hbn_v, exponent):
    voltage = np.asarray(voltage_hbn_v, dtype=float)
    if np.any(~np.isfinite(voltage)) or np.any(voltage < 0.0):
        raise ValueError("voltage_hbn_v must be finite and non-negative")
    power = float(exponent)
    if not np.isfinite(power) or power <= 0.0:
        raise ValueError("exponent must be positive and finite")
    value = voltage ** power
    return float(value) if value.ndim == 0 else value


def _set_branch_mask(cycle, branch):
    index = np.arange(len(cycle.voltage_hbn_v))
    if branch == "pre":
        side = index <= cycle.apex_index
        window = ((cycle.voltage_hbn_v >= 0.08)
                  & (cycle.voltage_hbn_v <= 1.20)
                  & (cycle.voltage_app_v <= 1.25))
    elif branch == "connected":
        side = index >= cycle.apex_index
        window = ((cycle.voltage_hbn_v >= 0.08)
                  & (cycle.voltage_hbn_v <= 4.20))
    else:
        raise ValueError("branch must be 'pre' or 'connected'")
    return (cycle.valid & side & window & (cycle.current_a > 1e-13))


def _profile_log_amplitude(cycle, branch, shape_parameter):
    mask = _set_branch_mask(cycle, branch)
    if np.count_nonzero(mask) < 5:
        raise ValueError(f"Insufficient stable {branch} SET points")
    voltage = cycle.voltage_hbn_v[mask]
    basis = (effective_pre_basis(voltage, shape_parameter)
             if branch == "pre"
             else connected_basis(voltage, shape_parameter))
    valid = np.isfinite(basis) & (basis > 0.0)
    if np.count_nonzero(valid) < 5:
        raise ValueError(f"Invalid {branch} SET basis")
    return float(np.mean(
        np.log10(cycle.current_a[mask][valid]) - np.log10(basis[valid])))


def _shape_rmse(cycles, branch, shape_parameter):
    residuals = []
    used = 0
    for cycle in cycles:
        try:
            log_amplitude = _profile_log_amplitude(
                cycle, branch, shape_parameter)
        except ValueError:
            continue
        mask = _set_branch_mask(cycle, branch)
        voltage = cycle.voltage_hbn_v[mask]
        basis = (effective_pre_basis(voltage, shape_parameter)
                 if branch == "pre"
                 else connected_basis(voltage, shape_parameter))
        residuals.extend(
            np.log10(cycle.current_a[mask])
            - np.log10(basis) - log_amplitude)
        used += 1
    if not residuals:
        return np.inf, 0
    values = np.asarray(residuals, dtype=float)
    return float(np.sqrt(np.mean(values ** 2))), used


def _robust_log_center(values):
    logs = np.log10(np.asarray(values, dtype=float))
    center = float(np.median(logs))
    return center, float(1.4826 * np.median(np.abs(logs - center)))


def fit_set_transport_law(cycles, gate_voltage, protocol_label,
                          set_peak_abs_v=5.0):
    """Fit shared SET envelope shapes while profiling cycle amplitudes."""
    from scipy.optimize import minimize_scalar

    cycles = list(cycles)
    if len(cycles) < 3:
        raise ValueError("At least three reconstructed SET cycles are required")
    pre_fit = minimize_scalar(
        lambda beta: _shape_rmse(cycles, "pre", beta)[0],
        bounds=(0.0, 15.0), method="bounded", options={"xatol": 1e-7})
    connected_fit = minimize_scalar(
        lambda exponent: _shape_rmse(cycles, "connected", exponent)[0],
        bounds=(0.2, 3.0), method="bounded", options={"xatol": 1e-7})
    pre_rmse, pre_used = _shape_rmse(cycles, "pre", pre_fit.x)
    connected_rmse, connected_used = _shape_rmse(
        cycles, "connected", connected_fit.x)
    if (not pre_fit.success or not connected_fit.success
            or pre_used < 3 or connected_used < 3):
        raise RuntimeError("Could not identify SET transport")
    pre_amplitudes = [10.0 ** _profile_log_amplitude(
        cycle, "pre", pre_fit.x) for cycle in cycles]
    connected_amplitudes = [10.0 ** _profile_log_amplitude(
        cycle, "connected", connected_fit.x) for cycle in cycles]
    pre_center, pre_sigma = _robust_log_center(pre_amplitudes)
    connected_center, connected_sigma = _robust_log_center(
        connected_amplitudes)
    return SetTransportLaw(
        beta_pre_per_sqrt_v=float(pre_fit.x),
        connected_exponent=float(connected_fit.x),
        log10_a_pre_ref=pre_center,
        log10_a_connected_ref=connected_center,
        pre_amplitude_sigma_decade=pre_sigma,
        connected_amplitude_sigma_decade=connected_sigma,
        pre_shape_rmse_decade=float(pre_rmse),
        connected_shape_rmse_decade=float(connected_rmse),
        gate_voltage_v=float(gate_voltage),
        protocol_label=str(protocol_label),
        n_training_cycles=int(min(pre_used, connected_used)),
        set_peak_abs_v=float(set_peak_abs_v),
    )


def estimate_set_amplitudes(cycle, law):
    """Estimate one cycle's pre/connected amplitudes for identification."""
    if not isinstance(law, SetTransportLaw):
        raise TypeError("law must be a SetTransportLaw")
    return (
        float(10.0 ** _profile_log_amplitude(
            cycle, "pre", law.beta_pre_per_sqrt_v)),
        float(10.0 ** _profile_log_amplitude(
            cycle, "connected", law.connected_exponent)),
    )


def set_voltage_exposure_v2(cycle):
    """Return cumulative ``integral |V_hBN| dV_app`` in ``V^2``."""
    index = np.arange(len(cycle.voltage_hbn_v))
    forward = index <= cycle.apex_index
    vapp = cycle.voltage_app_v
    vh = np.where(np.isfinite(cycle.voltage_hbn_v),
                  cycle.voltage_hbn_v, 0.0)
    good = forward & np.asarray(cycle.valid, dtype=bool)
    if np.count_nonzero(good) < 3:
        raise ValueError("SET cycle has too few valid rising-branch points")
    exposure = np.zeros_like(vapp)
    dv = np.maximum(np.diff(vapp), 0.0)
    exposure[1:] = np.cumsum(0.5 * (vh[1:] + vh[:-1]) * dv)
    exposure[~good] = np.nan
    return exposure


def _isotonic_increasing(values, weights=None):
    """Pool-adjacent-violators for a monotone increasing state."""
    y = np.asarray(values, dtype=float)
    weights = (np.ones_like(y) if weights is None
               else np.asarray(weights, dtype=float))
    blocks = []
    for idx, (value, weight) in enumerate(zip(y, weights)):
        blocks.append([idx, idx + 1, float(value), float(weight)])
        while len(blocks) >= 2 and blocks[-2][2] > blocks[-1][2]:
            left, right = blocks[-2], blocks[-1]
            total = left[3] + right[3]
            mean = (left[2] * left[3] + right[2] * right[3]) / total
            blocks[-2:] = [[left[0], right[1], mean, total]]
    out = np.empty_like(y)
    for start, stop, mean, _ in blocks:
        out[start:stop] = mean
    return out


def set_transition_state(exposure_v2, e50_v2, width_v2):
    """Rising logistic SET transition versus voltage exposure in ``V^2``."""
    width = float(width_v2)
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("width_v2 must be positive and finite")
    z = np.clip((np.asarray(exposure_v2, dtype=float) - float(e50_v2))
                / width, -60.0, 60.0)
    value = 1.0 / (1.0 + np.exp(-z))
    return float(value) if value.ndim == 0 else value


def _fit_one_transition(exposure, state):
    from scipy.optimize import minimize

    exposure = np.asarray(exposure, dtype=float)
    state = np.asarray(state, dtype=float)
    valid = np.isfinite(exposure) & np.isfinite(state)
    exposure = exposure[valid]
    state = state[valid]
    if len(exposure) < 8 or np.max(state) < 0.70:
        raise ValueError("Observable SET state has insufficient dynamic range")
    span = max(float(np.ptp(exposure)), 1e-3)

    def objective(values):
        prediction = set_transition_state(exposure, values[0], values[1])
        return float(np.mean((prediction - state) ** 2))

    initial = [float(np.interp(0.5, state, exposure)),
               max(span / 30.0, 0.005)]
    fit = minimize(
        objective, initial, method="Nelder-Mead",
        bounds=[(0.0, max(float(np.max(exposure)) * 2.0, 0.1)),
                (max(span / 2000.0, 0.001), span)],
        options={"maxiter": 1000, "xatol": 1e-9, "fatol": 1e-12})
    if not fit.success or not np.isfinite(fit.fun):
        raise RuntimeError("Could not fit SET transition")
    return float(fit.x[0]), float(fit.x[1]), float(np.sqrt(fit.fun))


def identify_set_state(cycle, law):
    """Identify one absolute observable SET state without endpoint rescaling."""
    if not isinstance(law, SetTransportLaw):
        raise TypeError("law must be a SetTransportLaw")
    a_pre, a_connected = estimate_set_amplitudes(cycle, law)
    stop = cycle.apex_index + 1
    voltage = cycle.voltage_hbn_v[:stop]
    current = cycle.current_a[:stop]
    i_pre = a_pre * effective_pre_basis(
        voltage, law.beta_pre_per_sqrt_v)
    i_connected = a_connected * connected_basis(
        voltage, law.connected_exponent)
    denominator = i_connected - i_pre
    valid = (cycle.valid[:stop] & np.isfinite(denominator)
             & (denominator > 1e-12) & (voltage >= 0.04))
    raw = np.full(stop, np.nan)
    raw[valid] = np.clip(
        (current[valid] - i_pre[valid]) / denominator[valid], 0.0, 1.0)
    order = np.flatnonzero(valid)
    if len(order) < 8:
        raise ValueError("Too few points to identify a SET trajectory")
    monotone = np.full(stop, np.nan)
    monotone_values = np.clip(_isotonic_increasing(raw[order]), 0.0, 1.0)
    monotone[order] = monotone_values
    exposure = set_voltage_exposure_v2(cycle)[:stop]
    e50, width, state_rms = _fit_one_transition(
        exposure[order], monotone_values)
    return IdentifiedSetState(
        gate_voltage_v=cycle.gate_voltage_v,
        exposure_v2=set_voltage_exposure_v2(cycle),
        state_raw=raw,
        state_monotone=monotone,
        a_pre=float(a_pre),
        a_connected=float(a_connected),
        e10_v2=float(e50 - np.log(9.0) * width),
        e50_v2=float(e50),
        e90_v2=float(e50 + np.log(9.0) * width),
        width_v2=float(width),
        state_fit_rms=float(state_rms),
        reconstruction_valid_fraction=float(np.mean(cycle.valid)),
    )


def fit_set_transition_law(named_cycles, transport_law, gate_voltage,
                           protocol_label, set_peak_abs_v=5.0):
    """Fit the protocol-level transition distribution from training cycles."""
    states = []
    rejected = []
    for item in named_cycles:
        if len(item) == 2:
            name, cycle = item
        else:
            name, cycle = str(len(states) + len(rejected) + 1), item
        try:
            states.append(identify_set_state(cycle, transport_law))
        except (ValueError, RuntimeError) as error:
            rejected.append(f"{name}: {error}")
    if len(states) < 3:
        raise ValueError("At least three identifiable SET cycles are required")
    e50 = np.asarray([state.e50_v2 for state in states])
    width = np.asarray([state.width_v2 for state in states])
    residual = np.asarray([state.state_fit_rms for state in states])
    e50_center = float(np.median(e50))
    width_center = float(np.median(width))
    return SetTransitionLaw(
        e50_v2=e50_center,
        width_v2=width_center,
        e50_sigma_v2=float(1.4826 * np.median(np.abs(e50 - e50_center))),
        width_sigma_v2=float(
            1.4826 * np.median(np.abs(width - width_center))),
        residual_rms_state=float(np.median(residual)),
        gate_voltage_v=float(gate_voltage),
        protocol_label=str(protocol_label),
        n_training_cycles=len(states),
        rejected_cycles=tuple(rejected),
        set_peak_abs_v=float(set_peak_abs_v),
    )


def claim_scope():
    return {
        "transistor_mapping": (
            "literal Nature Fig. 2b positive-V_DS lookup at the requested gate"),
        "pre_transport": (
            "effective field-assisted V_hBN*exp(beta*sqrt(V_hBN)) curvature"),
        "connected_transport": "effective super-ohmic A*V_hBN^n curvature",
        "state": (
            "observable fraction between fitted pre/connected envelopes; "
            "not a vacancy concentration or filament radius"),
        "drive": (
            "integral |V_hBN| dV_app in V^2; a protocol-specific, "
            "step-invariant voltage exposure"),
        "time_limitation": (
            "no physical time, charge or energy is inferred without dwell time"),
    }
