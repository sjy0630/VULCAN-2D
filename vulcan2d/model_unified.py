"""Unified local-connectivity model for quasi-static h-BN 1T1M loops.

The model uses one persistent state vector ``z_k`` for both polarities.  Each
entry is a coarse-grained local conductive-connectivity fraction, not a
measured vacancy count, metal-ion concentration, or filament radius.  The same
state that mixes the local transport envelopes is also the state advanced by
the SET/RESET electrical stress laws.

The existing calibrated positive and negative transport envelopes are kept.
For K=1 the model is the mean-field limit of the distributed construction and
reproduces the established voltage-only current calculation.  For K>1 a
quantile-preserving coarse graining allocates the same macroscopic state among
local regions.  Consequently the local currents sum exactly to the validated
mean-field current; spatial detail is exposed without creating a second,
inconsistent terminal model.

The SET-to-RESET factor remains a polarity/protocol transport-amplitude
conversion.  It is not used to modify ``z_k`` and must not be interpreted as
ion retention or state-transfer efficiency.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import model_v7 as v7
from . import model_v8 as v8
from . import model_v11 as v11
from . import model_v12 as v12
from .reset_identification import PFResetTransitionLaw, logistic_state
from .set_identification import (
    SetTransitionLaw,
    SetTransportLaw,
    connected_basis,
    direct_set_transistor_mapping,
    effective_pre_basis,
)


@dataclass(frozen=True)
class SpatialConfiguration:
    """Numerical coarse graining of local conductive regions.

    ``weights`` may be informed by a CAFM projected-area distribution, but the
    ordering is deliberately independent of spot size because the available
    CAFM image does not identify electrical thresholds.  Reversing the SET
    order for RESET implements a nested hysteresis construction; the terminal
    current is invariant to this unobserved permutation at fixed mean state.
    """

    weights: np.ndarray
    set_order: np.ndarray
    reset_order: np.ndarray
    source: str = "numerical coarse graining"
    order_identified: bool = False

    def __post_init__(self):
        weights = np.asarray(self.weights, dtype=float).copy()
        set_order = np.asarray(self.set_order, dtype=int).copy()
        reset_order = np.asarray(self.reset_order, dtype=int).copy()
        if (weights.ndim != 1 or len(weights) == 0
                or np.any(~np.isfinite(weights)) or np.any(weights <= 0.0)):
            raise ValueError("weights must be a positive finite vector")
        weights /= np.sum(weights)
        expected = np.arange(len(weights))
        for name, order in (("set_order", set_order),
                            ("reset_order", reset_order)):
            if (order.shape != expected.shape
                    or not np.array_equal(np.sort(order), expected)):
                raise ValueError(f"{name} must be a permutation of patch indices")
        for array in (weights, set_order, reset_order):
            array.setflags(write=False)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "set_order", set_order)
        object.__setattr__(self, "reset_order", reset_order)

    @property
    def k(self):
        return int(len(self.weights))

    @classmethod
    def mean_field(cls):
        return cls(np.ones(1), np.zeros(1, dtype=int),
                   np.zeros(1, dtype=int), source="K=1 mean-field limit")

    @classmethod
    def from_weights(cls, weights, seed=2026, source="geometric prior"):
        weights = np.asarray(weights, dtype=float)
        rng = np.random.default_rng(int(seed))
        order = rng.permutation(len(weights))
        return cls(weights, order, order[::-1], source=source,
                   order_identified=False)


def _state_vector(state, k):
    state = np.asarray(state, dtype=float)
    if state.ndim == 0:
        state = np.full(int(k), float(state))
    if (state.shape != (int(k),) or np.any(~np.isfinite(state))
            or np.any(state < 0.0) or np.any(state > 1.0)):
        raise ValueError("local state must be finite, in [0,1], and match K")
    return state.copy()


def weighted_state(state, spatial):
    state = _state_vector(state, spatial.k)
    return float(np.dot(spatial.weights, state))


def _increase_to_mean(state, target, spatial):
    """Increase local states in SET order to an exact weighted mean."""
    state = _state_vector(state, spatial.k)
    current = float(np.dot(spatial.weights, state))
    target = float(target)
    if not np.isfinite(target) or target < current - 2e-13 or target > 1.0 + 2e-13:
        raise ValueError("SET target mean must lie between current mean and one")
    remaining = max(min(target, 1.0) - current, 0.0)
    for index in spatial.set_order:
        capacity = spatial.weights[index] * (1.0 - state[index])
        addition = min(remaining, capacity)
        state[index] += addition / spatial.weights[index]
        remaining -= addition
        if remaining <= 2e-15:
            break
    if remaining > 2e-12:
        raise RuntimeError("Could not allocate the requested SET state")
    return np.clip(state, 0.0, 1.0)


def _decrease_to_mean(state, target, spatial):
    """Decrease local states in RESET order to an exact weighted mean."""
    state = _state_vector(state, spatial.k)
    current = float(np.dot(spatial.weights, state))
    target = float(target)
    if not np.isfinite(target) or target < -2e-13 or target > current + 2e-13:
        raise ValueError("RESET target mean must lie between zero and current mean")
    remaining = max(current - max(target, 0.0), 0.0)
    for index in spatial.reset_order:
        capacity = spatial.weights[index] * state[index]
        removal = min(remaining, capacity)
        state[index] -= removal / spatial.weights[index]
        remaining -= removal
        if remaining <= 2e-15:
            break
    if remaining > 2e-12:
        raise RuntimeError("Could not allocate the requested RESET state")
    return np.clip(state, 0.0, 1.0)


def set_hbn_current(voltage_hbn_v, local_state, a_pre, a_connected,
                    transport_law, spatial):
    """Sum local positive-polarity currents using the persistent state."""
    if not isinstance(transport_law, SetTransportLaw):
        raise TypeError("transport_law must be a SetTransportLaw")
    state = _state_vector(local_state, spatial.k)
    voltage = abs(float(voltage_hbn_v))
    a_pre, a_connected = float(a_pre), float(a_connected)
    if (not np.isfinite(a_pre) or not np.isfinite(a_connected)
            or a_pre <= 0.0 or a_connected <= 0.0):
        raise ValueError("SET envelope amplitudes must be positive and finite")
    pre = a_pre * effective_pre_basis(
        voltage, transport_law.beta_pre_per_sqrt_v)
    connected = a_connected * connected_basis(
        voltage, transport_law.connected_exponent)
    local = (1.0 - state) * pre + state * connected
    return float(np.dot(spatial.weights, local))


def divider_set(applied_voltage_v, local_state, a_pre, a_connected,
                gate_voltage, transport_law, spatial, mapping=None, nbis=44):
    mapping = direct_set_transistor_mapping() if mapping is None else mapping
    applied = abs(float(applied_voltage_v))
    if applied <= 1e-15:
        return 0.0, 0.0
    lo, hi = 0.0, applied
    for _ in range(int(nbis)):
        voltage_hbn = 0.5 * (lo + hi)
        ih = set_hbn_current(
            voltage_hbn, local_state, a_pre, a_connected,
            transport_law, spatial)
        it = mapping.current(
            applied - voltage_hbn, gate_voltage, side="set")
        if ih < it:
            lo = voltage_hbn
        else:
            hi = voltage_hbn
    voltage_hbn = 0.5 * (lo + hi)
    return (set_hbn_current(
        voltage_hbn, local_state, a_pre, a_connected,
        transport_law, spatial), voltage_hbn)


def run_unified_set(
        voltage_wave, gate_voltage, transport_law, transition_law,
        spatial=None, mapping=None, a_pre=None, a_connected=None,
        initial_state=0.0, transition_e50_offset_v2=0.0,
        max_fixed_point_iterations=80):
    """Run SET with one local state vector and voltage-domain exposure."""
    if not isinstance(transport_law, SetTransportLaw):
        raise TypeError("transport_law must be a SetTransportLaw")
    if not isinstance(transition_law, SetTransitionLaw):
        raise TypeError("transition_law must be a SetTransitionLaw")
    spatial = SpatialConfiguration.mean_field() if spatial is None else spatial
    if not isinstance(spatial, SpatialConfiguration):
        raise TypeError("spatial must be a SpatialConfiguration")
    voltage = np.asarray(voltage_wave, dtype=float)
    if (voltage.ndim != 1 or len(voltage) < 3
            or np.any(~np.isfinite(voltage))):
        raise ValueError("voltage_wave must be a finite complete SET loop")
    transport_law.validate_gate(gate_voltage)
    transition_law.validate_gate(gate_voltage)
    transport_law.validate_protocol(voltage)
    transition_law.validate_protocol(voltage)
    mapping = direct_set_transistor_mapping() if mapping is None else mapping
    a_pre = (transport_law.pre_amplitude() if a_pre is None
             else float(a_pre))
    a_connected = (transport_law.connected_amplitude()
                   if a_connected is None else float(a_connected))
    state = _state_vector(initial_state, spatial.k)
    e50 = transition_law.e50_v2 + float(transition_e50_offset_v2)
    if not np.isfinite(e50):
        raise ValueError("transition_e50_offset_v2 must be finite")

    magnitude = np.abs(voltage)
    apex = int(np.argmax(magnitude))
    if apex < 1 or np.any(np.diff(magnitude[:apex + 1]) < -1e-9):
        raise ValueError("SET rising branch must be non-decreasing")
    current = np.empty_like(voltage)
    voltage_hbn = np.empty_like(voltage)
    state_mean = np.empty_like(voltage)
    state_local = np.empty((len(voltage), spatial.k), dtype=float)
    exposure_history = np.empty_like(voltage)
    exposure = 0.0
    previous_magnitude = 0.0
    previous_voltage_hbn = 0.0

    for index, applied in enumerate(magnitude):
        if index <= apex:
            delta_voltage = max(float(applied - previous_magnitude), 0.0)
            previous_state = state.copy()
            candidate = state.copy()
            candidate_exposure = exposure
            for _ in range(int(max_fixed_point_iterations)):
                _, candidate_vh = divider_set(
                    applied, candidate, a_pre, a_connected, gate_voltage,
                    transport_law, spatial, mapping)
                candidate_exposure = (
                    exposure
                    + 0.5 * (previous_voltage_hbn + candidate_vh)
                    * delta_voltage
                )
                target = max(
                    weighted_state(previous_state, spatial),
                    float(transition_law.state_with_e50(
                        candidate_exposure, e50)),
                )
                updated = _increase_to_mean(previous_state, target, spatial)
                if np.max(np.abs(updated - candidate)) <= 1e-10:
                    candidate = updated
                    break
                candidate = 0.5 * candidate + 0.5 * updated
            state = candidate
            idevice, vh = divider_set(
                applied, state, a_pre, a_connected, gate_voltage,
                transport_law, spatial, mapping)
            exposure = (
                exposure
                + 0.5 * (previous_voltage_hbn + vh) * delta_voltage
            )
            target = max(
                weighted_state(previous_state, spatial),
                float(transition_law.state_with_e50(exposure, e50)),
            )
            state = _increase_to_mean(previous_state, target, spatial)
            idevice, vh = divider_set(
                applied, state, a_pre, a_connected, gate_voltage,
                transport_law, spatial, mapping)
            previous_magnitude = float(applied)
            previous_voltage_hbn = float(vh)
        else:
            idevice, vh = divider_set(
                applied, state, a_pre, a_connected, gate_voltage,
                transport_law, spatial, mapping)
        current[index] = np.sign(voltage[index]) * idevice
        voltage_hbn[index] = np.sign(voltage[index]) * vh
        state_mean[index] = weighted_state(state, spatial)
        state_local[index] = state
        exposure_history[index] = exposure

    return {
        "Vs": voltage.copy(), "Is": current, "Vhs": voltage_hbn,
        "zs": state_mean, "z_local": state_local,
        # Compatibility alias: x is now explicitly the mean of z_k.
        "xs": state_mean,
        "set_exposure_v2": exposure_history,
        "a_pre": float(a_pre), "a_connected": float(a_connected),
        "transition_e50_v2": float(e50),
        "transition_width_v2": float(transition_law.width_v2),
        "Vg": float(gate_voltage),
        "target_set_current_used_for_prediction": False,
    }


def reset_hbn_current(voltage_hbn_v, local_state, g_lrs_a, a_hrs,
                      reset_params, spatial):
    """Sum local negative-polarity currents using the same state vector."""
    state = _state_vector(local_state, spatial.k)
    voltage = abs(float(voltage_hbn_v))
    i_lrs = float(g_lrs_a) * np.sinh(np.clip(
        reset_params.reset_alpha_lrs_identified * voltage, 0.0, 39.0))
    i_hrs = float(a_hrs) * v7.reset_hrs_basis(voltage, reset_params)
    local = state * i_lrs + (1.0 - state) * i_hrs
    return float(np.dot(spatial.weights, local))


def divider_reset(applied_voltage_v, local_state, g_lrs_a, a_hrs,
                  gate_voltage, reset_params, spatial, mapping, nbis=36):
    applied = abs(float(applied_voltage_v))
    if applied <= 1e-15:
        return 0.0, 0.0
    lo, hi = 0.0, applied
    for _ in range(int(nbis)):
        voltage_hbn = 0.5 * (lo + hi)
        ih = reset_hbn_current(
            voltage_hbn, local_state, g_lrs_a, a_hrs,
            reset_params, spatial)
        it = mapping.current(
            applied - voltage_hbn, gate_voltage, side="reset")
        if ih < it:
            lo = voltage_hbn
        else:
            hi = voltage_hbn
    voltage_hbn = 0.5 * (lo + hi)
    return (reset_hbn_current(
        voltage_hbn, local_state, g_lrs_a, a_hrs,
        reset_params, spatial), voltage_hbn)


def run_unified_reset(
        voltage_wave, gate_voltage, initial_state, g_lrs_a, a_hrs,
        reset_params, spatial, mapping, e50_mvc, width_mvc):
    """Run RESET by decreasing the exact local state left by SET."""
    voltage = np.asarray(voltage_wave, dtype=float)
    if (voltage.ndim != 1 or len(voltage) < 3
            or np.any(~np.isfinite(voltage))):
        raise ValueError("voltage_wave must be a finite complete RESET loop")
    if g_lrs_a <= 0.0 or a_hrs <= 0.0:
        raise ValueError("RESET envelope amplitudes must be positive")
    state = _state_vector(initial_state, spatial.k)
    reference = 0.1
    if reset_hbn_current(reference, np.ones(spatial.k), g_lrs_a, a_hrs,
                         reset_params, spatial) <= reset_hbn_current(
                             reference, np.zeros(spatial.k), g_lrs_a, a_hrs,
                             reset_params, spatial):
        raise ValueError("RESET envelopes require LRS current above HRS")
    e50, width = float(e50_mvc), float(width_mvc)
    if not np.isfinite(e50) or not np.isfinite(width) or width <= 0.0:
        raise ValueError("RESET transition parameters must be finite and positive")
    apex = int(np.argmax(np.abs(voltage)))
    current = np.empty_like(voltage)
    voltage_hbn = np.empty_like(voltage)
    field_history = np.empty_like(voltage)
    state_mean = np.empty_like(voltage)
    state_local = np.empty((len(voltage), spatial.k), dtype=float)

    for index, applied in enumerate(voltage):
        magnitude = abs(float(applied))
        _, vh = divider_reset(
            magnitude, state, g_lrs_a, a_hrs, gate_voltage,
            reset_params, spatial, mapping)
        field = 10.0 * vh / reset_params.hbn_thickness_nm
        # Preserve the SET endpoint exactly at the zero-bias handoff.  At the
        # first finite RESET point the same calibrated survival law takes over.
        if index <= apex and magnitude > 1e-15:
            survival = float(logistic_state(field, e50, width))
            target = min(weighted_state(state, spatial), survival)
            state = _decrease_to_mean(state, target, spatial)
            idevice, vh = divider_reset(
                magnitude, state, g_lrs_a, a_hrs, gate_voltage,
                reset_params, spatial, mapping)
            field = 10.0 * vh / reset_params.hbn_thickness_nm
        else:
            idevice, vh = divider_reset(
                magnitude, state, g_lrs_a, a_hrs, gate_voltage,
                reset_params, spatial, mapping)
        current[index] = np.sign(applied) * idevice
        voltage_hbn[index] = np.sign(applied) * vh
        field_history[index] = field
        state_mean[index] = weighted_state(state, spatial)
        state_local[index] = state

    return {
        "Vs": voltage.copy(), "Is": current, "Vhs": voltage_hbn,
        "zs": state_mean, "z_local": state_local,
        # Compatibility alias: p is now explicitly the same mean z state.
        "pbs": state_mean,
        "Ehs_MV_cm": field_history,
        "g_lrs_a": float(g_lrs_a), "g_hrs_a": float(a_hrs),
        "e50_mvc": e50, "width_mvc": width,
        "Vg": float(gate_voltage),
    }


def run_unified_full_cycle(
        set_voltage, reset_voltage, gate_voltage, set_transport_law,
        set_transition_law, set_reset_link_law, reset_endpoint_law,
        reset_transport_law, reset_base_params=None, set_mapping=None,
        reset_mapping=None, reset_transition_law=None,
        reset_e50_mvc=None, reset_width_mvc=None,
        set_transition_e50_offset_v2=0.0,
        link_residual_offset_decade=0.0,
        endpoint_residual_offset_decade=0.0, spatial=None):
    """Predict a full loop with one continuous local-connectivity state."""
    spatial = SpatialConfiguration.mean_field() if spatial is None else spatial
    if not isinstance(spatial, SpatialConfiguration):
        raise TypeError("spatial must be a SpatialConfiguration")
    reset_transport_law.validate_gate(gate_voltage)
    reset_transport_law.validate_protocol(reset_voltage)
    reset_endpoint_law.validate_gate(gate_voltage)
    reset_endpoint_law.validate_protocol(reset_voltage)
    if reset_transition_law is not None:
        if not isinstance(reset_transition_law, PFResetTransitionLaw):
            raise TypeError("reset_transition_law must be a PFResetTransitionLaw")
        reset_transition_law.validate_gate(gate_voltage)
        reset_transition_law.validate_protocol(reset_voltage)
        if reset_e50_mvc is None:
            reset_e50_mvc = reset_transition_law.e50_mvc
        if reset_width_mvc is None:
            reset_width_mvc = reset_transition_law.width_mvc
    base = v8.Params() if reset_base_params is None else reset_base_params
    reset_params = v11.params_with_transport_law(base, reset_transport_law)
    reset_mapping = (v11.transistor_mapping(reset_params)
                     if reset_mapping is None else reset_mapping)

    set_result = run_unified_set(
        set_voltage, gate_voltage, set_transport_law, set_transition_law,
        spatial=spatial, mapping=set_mapping,
        transition_e50_offset_v2=set_transition_e50_offset_v2)
    g_lrs = set_reset_link_law.predict_reset_lrs_amplitude(
        set_result["a_connected"], set_transport_law.connected_exponent,
        reset_params.reset_alpha_lrs_identified,
        residual_offset_decade=link_residual_offset_decade)
    a_hrs_uncapped = reset_endpoint_law.predict(
        endpoint_residual_offset_decade)
    a_hrs = v12._cap_reset_endpoint(a_hrs_uncapped, g_lrs, reset_params)
    if reset_e50_mvc is None:
        reset_e50_mvc = reset_params.reset_e50_mvc
    if reset_width_mvc is None:
        reset_width_mvc = reset_params.reset_width_mvc
    set_endpoint = set_result["z_local"][-1].copy()
    reset_result = run_unified_reset(
        reset_voltage, gate_voltage, set_endpoint, g_lrs, a_hrs,
        reset_params, spatial, reset_mapping,
        reset_e50_mvc, reset_width_mvc)

    set_end_mean = weighted_state(set_endpoint, spatial)
    reset_start_mean = float(reset_result["zs"][0])
    continuity_error = abs(set_end_mean - reset_start_mean)
    combined_local = np.vstack(
        [set_result["z_local"], reset_result["z_local"]])
    combined_mean = np.concatenate(
        [set_result["zs"], reset_result["zs"]])
    return {
        "set": set_result, "reset": reset_result,
        "Vs": np.concatenate([set_result["Vs"], reset_result["Vs"]]),
        "Is": np.concatenate([set_result["Is"], reset_result["Is"]]),
        "zs": combined_mean, "z_local": combined_local,
        "connected_fraction": combined_mean,
        "g_reset_initial_predicted_a": float(g_lrs),
        "a_reset_hrs_predicted": float(a_hrs),
        "a_reset_hrs_uncapped": float(a_hrs_uncapped),
        "reset_endpoint_capped": bool(a_hrs < a_hrs_uncapped),
        "polarity_transport_factor": float(
            set_reset_link_law.retention_factor),
        "reset_transition_source": (
            "protocol_fitted_PF_envelope_state"
            if reset_transition_law is not None
            else "base_parameter_or_explicit_override"),
        "target_set_current_used_for_prediction": False,
        "target_reset_current_used_for_prediction": False,
        "handoff": {
            "method": "continuous_local_connectivity_state",
            "unified_physical_state": True,
            "set_end_state": set_end_mean,
            "reset_first_recorded_state": reset_start_mean,
            "state_continuity_absolute_error": continuity_error,
            "reference_current_relative_error": 0.0,
            "partial_set_supported_mathematically": True,
            "partial_set_experimentally_calibrated": False,
            "polarity_factor_interpretation": (
                "transport/protocol amplitude asymmetry, not state retention"),
        },
        "spatial": {
            "K": spatial.k,
            "weights": spatial.weights.tolist(),
            "source": spatial.source,
            "order_identified": spatial.order_identified,
            "terminal_identifiability": (
                "only the weighted mean is identifiable from the present I-V data"),
        },
        "state_coordinates": {
            "z_k": "coarse local conductive-connectivity fraction",
            "set.xs": "compatibility alias for sum_k w_k z_k",
            "reset.pbs": "compatibility alias for the same sum_k w_k z_k",
        },
        "model_family": "unified local-connectivity quasi-static model",
    }


def simulate_calibrated_cycle(
        set_voltage, reset_voltage, gate_voltage=1.1, calibration=None,
        reset_transition_mode="same_protocol", spatial=None, **kwargs):
    calibration = v12.default_calibration() if calibration is None else calibration
    if not isinstance(calibration, v12.AutonomousCalibration):
        raise TypeError("calibration must be an AutonomousCalibration")
    if reset_transition_mode == "same_protocol":
        transition = calibration.reset_transition
    elif reset_transition_mode == "cross_gate_prior":
        transition = None
    else:
        raise ValueError(
            "reset_transition_mode must be 'same_protocol' or 'cross_gate_prior'")
    return run_unified_full_cycle(
        set_voltage, reset_voltage, gate_voltage,
        calibration.set_transport, calibration.set_transition,
        calibration.set_reset_link, calibration.reset_endpoint,
        calibration.reset_transport, reset_transition_law=transition,
        spatial=spatial, **kwargs)


def claim_scope():
    return {
        "state": (
            "one persistent local-connectivity vector z_k generates both "
            "terminal current and quasi-static SET/RESET evolution"),
        "mean_field_limit": (
            "K=1 is the established envelope model; K>1 is an exact "
            "weighted coarse graining at fixed mean state"),
        "physical_interpretation": (
            "effective local interlayer connectivity consistent with defect- "
            "or metal-assisted paths; no unique atomic species is identified"),
        "spatial_limit": (
            "CAFM may constrain area weights, but present I-V data do not "
            "identify local ordering or threshold chemistry"),
        "time_limit": (
            "voltage-domain quasi-static exposure; no calibrated dwell or "
            "pulse-time prediction"),
        "validation_domain": (
            "1.1-V gate, recorded +5-V SET and -1.7-V RESET DC protocols"),
    }
