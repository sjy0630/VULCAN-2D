"""Autonomous quasi-static SET-to-RESET simulation for VULCAN-2D.

This module closes the voltage-domain loop left open in v0.11.  Its public
simulation API accepts the SET/RESET voltage waveforms and gate voltage, but
does not accept either target current trace.

SET uses a measured positive-bias transistor load line, two independently
identified h-BN transport envelopes and an irreversible observable state
driven by ``integral |V_hBN| dV_app``.  RESET reuses the measured reverse-bias
transistor branch and effective field-assisted HRS law from v0.11.  A fitted
SET-to-RESET link accounts for the different bias/read protocols; it is an
empirical state-transfer factor, not a microscopic migration efficiency.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import model_v8 as v8
from . import model_v11 as v11
from .cycle_link import ResetEndpointAttractorLaw
from .hrs_transport import EffectivePooleFrenkelLaw
from .reset_identification import PFResetTransitionLaw
from .set_identification import (
    SetTransitionLaw,
    SetTransportLaw,
    connected_basis,
    direct_set_transistor_mapping,
    effective_pre_basis,
)


@dataclass(frozen=True)
class SetResetLinkLaw:
    """Protocol-level conversion from final SET to initial RESET scale.

    ``retention_factor`` is fitted after matching the two compact transport
    laws at ``reference_voltage_v``.  It absorbs the change of polarity,
    inter-sweep waiting/read conditions and any unresolved state relaxation.
    It must not be interpreted as a directly measured material efficiency.
    """

    retention_factor: float
    residual_sigma_decade: float
    reference_voltage_v: float
    n_training_cycles: int
    set_protocol_label: str
    reset_protocol_label: str

    def predict_reset_lrs_amplitude(
            self, a_connected, connected_exponent, reset_alpha_lrs,
            residual_offset_decade=0.0):
        a_connected = float(a_connected)
        if not np.isfinite(a_connected) or a_connected <= 0.0:
            raise ValueError("a_connected must be positive and finite")
        reference = float(self.reference_voltage_v)
        set_current = a_connected * connected_basis(
            reference, connected_exponent)
        reset_basis = np.sinh(float(reset_alpha_lrs) * reference)
        if not np.isfinite(reset_basis) or reset_basis <= 0.0:
            raise ValueError("RESET LRS basis must be positive and finite")
        return float(
            self.retention_factor
            * 10.0 ** float(residual_offset_decade)
            * set_current / reset_basis
        )


@dataclass(frozen=True)
class AutonomousCalibration:
    """Versioned central calibration for the validated 1.1-V DC protocol."""

    set_transport: SetTransportLaw
    set_transition: SetTransitionLaw
    set_reset_link: SetResetLinkLaw
    reset_endpoint: ResetEndpointAttractorLaw
    reset_transport: EffectivePooleFrenkelLaw
    reset_transition: PFResetTransitionLaw


def default_calibration():
    """Return the v0.12 full-group calibration without reading workbooks."""
    set_protocol = "0 to +5 to 0 V DC SET at Vg=1.1 V"
    reset_protocol = "0 to -1.7 to 0 V DC RESET at Vg=1.1 V"
    return AutonomousCalibration(
        set_transport=SetTransportLaw(
            beta_pre_per_sqrt_v=6.5376046042521665,
            connected_exponent=1.326001856827809,
            log10_a_pre_ref=-9.585926896788138,
            log10_a_connected_ref=-4.960982904913941,
            pre_amplitude_sigma_decade=0.20532850592002452,
            connected_amplitude_sigma_decade=0.01696395752318905,
            pre_shape_rmse_decade=0.21106327030432845,
            connected_shape_rmse_decade=0.1127273147353709,
            gate_voltage_v=1.1,
            protocol_label=set_protocol,
            n_training_cycles=53,
        ),
        set_transition=SetTransitionLaw(
            e50_v2=1.1350926441681073,
            width_v2=0.047923217612974855,
            e50_sigma_v2=0.15290593098017197,
            width_sigma_v2=0.04180404720051404,
            residual_rms_state=0.05739906659612289,
            gate_voltage_v=1.1,
            protocol_label=set_protocol,
            n_training_cycles=53,
        ),
        set_reset_link=SetResetLinkLaw(
            retention_factor=0.2575148876811263,
            residual_sigma_decade=0.22505495491078945,
            reference_voltage_v=0.1,
            n_training_cycles=51,
            set_protocol_label=set_protocol,
            reset_protocol_label=reset_protocol,
        ),
        reset_endpoint=ResetEndpointAttractorLaw(
            gate_voltage_v=1.1,
            protocol_label=reset_protocol,
            log10_g_hrs_ref=-9.200845203689065,
            residual_sigma_decade=0.1873992974557137,
            n_training=51,
        ),
        reset_transport=EffectivePooleFrenkelLaw(
            beta_per_sqrt_v=6.28586925832311,
            residual_mad_decade=0.06297597576540603,
            gate_voltage_v=1.1,
            protocol_label=reset_protocol,
            n_training_cycles=51,
        ),
        reset_transition=PFResetTransitionLaw(
            e50_mvc=0.7926771526923162,
            width_mvc=0.1621629633540858,
            e50_sigma_mvc=0.11872365101688878,
            width_sigma_mvc=0.034187646646737034,
            residual_rms_state=0.037157534735097614,
            gate_voltage_v=1.1,
            protocol_label=reset_protocol,
            n_training_cycles=51,
        ),
    )


def fit_set_reset_link_law(
        connected_amplitudes, reset_lrs_amplitudes, connected_exponent,
        reset_alpha_lrs, set_protocol_label, reset_protocol_label,
        reference_voltage_v=0.1):
    """Fit a robust cross-protocol SET-to-RESET state-transfer factor."""
    connected_amplitudes = np.asarray(connected_amplitudes, dtype=float)
    reset_lrs_amplitudes = np.asarray(reset_lrs_amplitudes, dtype=float)
    if (connected_amplitudes.ndim != 1
            or connected_amplitudes.shape != reset_lrs_amplitudes.shape):
        raise ValueError("SET and RESET amplitudes must be same-length vectors")
    reference = float(reference_voltage_v)
    if not np.isfinite(reference) or reference <= 0.0:
        raise ValueError("reference_voltage_v must be positive and finite")
    set_current = connected_amplitudes * connected_basis(
        reference, connected_exponent)
    direct_reset_scale = set_current / np.sinh(
        float(reset_alpha_lrs) * reference)
    valid = (
        np.isfinite(direct_reset_scale)
        & np.isfinite(reset_lrs_amplitudes)
        & (direct_reset_scale > 0.0)
        & (reset_lrs_amplitudes > 0.0)
    )
    if np.count_nonzero(valid) < 3:
        raise ValueError("At least three valid paired cycles are required")
    log_ratio = np.log10(
        reset_lrs_amplitudes[valid] / direct_reset_scale[valid])
    center = float(np.median(log_ratio))
    sigma = float(1.4826 * np.median(np.abs(log_ratio - center)))
    return SetResetLinkLaw(
        retention_factor=float(10.0 ** center),
        residual_sigma_decade=sigma,
        reference_voltage_v=reference,
        n_training_cycles=int(np.count_nonzero(valid)),
        set_protocol_label=str(set_protocol_label),
        reset_protocol_label=str(reset_protocol_label),
    )


def set_hbn_current(voltage_hbn_v, state, a_pre, a_connected,
                    transport_law):
    """Return the two-envelope SET current at one observable state."""
    if not isinstance(transport_law, SetTransportLaw):
        raise TypeError("transport_law must be a SetTransportLaw")
    voltage = abs(float(voltage_hbn_v))
    state = float(np.clip(state, 0.0, 1.0))
    a_pre = float(a_pre)
    a_connected = float(a_connected)
    if (not np.isfinite(a_pre) or not np.isfinite(a_connected)
            or a_pre <= 0.0 or a_connected <= 0.0):
        raise ValueError("SET envelope amplitudes must be positive and finite")
    pre = a_pre * effective_pre_basis(
        voltage, transport_law.beta_pre_per_sqrt_v)
    connected = a_connected * connected_basis(
        voltage, transport_law.connected_exponent)
    return float((1.0 - state) * pre + state * connected)


def divider_set(applied_voltage_v, state, a_pre, a_connected, gate_voltage,
                transport_law, mapping=None, nbis=44):
    """Solve one self-consistent SET load-line intersection."""
    mapping = (direct_set_transistor_mapping()
               if mapping is None else mapping)
    applied = abs(float(applied_voltage_v))
    if applied <= 1e-15:
        return 0.0, 0.0
    lo, hi = 0.0, applied
    for _ in range(int(nbis)):
        voltage_hbn = 0.5 * (lo + hi)
        hbn_current = set_hbn_current(
            voltage_hbn, state, a_pre, a_connected, transport_law)
        transistor_current = mapping.current(
            applied - voltage_hbn, gate_voltage, side="set")
        if hbn_current < transistor_current:
            lo = voltage_hbn
        else:
            hi = voltage_hbn
    voltage_hbn = 0.5 * (lo + hi)
    current = set_hbn_current(
        voltage_hbn, state, a_pre, a_connected, transport_law)
    return current, voltage_hbn


def run_autonomous_set(
        voltage_wave, gate_voltage, transport_law, transition_law,
        mapping=None, a_pre=None, a_connected=None, initial_state=0.0,
        transition_e50_offset_v2=0.0, max_fixed_point_iterations=80):
    """Predict a complete SET loop from voltage alone after calibration.

    The rising branch couples the h-BN voltage, cumulative voltage exposure
    and observable state by fixed-point iteration.  The state is frozen on
    the return branch, representing nonvolatile retention over this DC loop.
    """
    if not isinstance(transport_law, SetTransportLaw):
        raise TypeError("transport_law must be a SetTransportLaw")
    if not isinstance(transition_law, SetTransitionLaw):
        raise TypeError("transition_law must be a SetTransitionLaw")
    voltage = np.asarray(voltage_wave, dtype=float)
    if (voltage.ndim != 1 or len(voltage) < 3
            or np.any(~np.isfinite(voltage))):
        raise ValueError("voltage_wave must be a finite complete SET loop")
    transport_law.validate_gate(gate_voltage)
    transition_law.validate_gate(gate_voltage)
    transport_law.validate_protocol(voltage)
    transition_law.validate_protocol(voltage)
    mapping = (direct_set_transistor_mapping()
               if mapping is None else mapping)
    a_pre = (transport_law.pre_amplitude() if a_pre is None
             else float(a_pre))
    a_connected = (transport_law.connected_amplitude()
                   if a_connected is None else float(a_connected))
    state = float(np.clip(initial_state, 0.0, 1.0))
    if not np.isfinite(state):
        raise ValueError("initial_state must be finite")
    e50 = transition_law.e50_v2 + float(transition_e50_offset_v2)
    if not np.isfinite(e50):
        raise ValueError("transition_e50_offset_v2 must be finite")

    magnitude = np.abs(voltage)
    apex = int(np.argmax(magnitude))
    if apex < 1 or np.any(np.diff(magnitude[:apex + 1]) < -1e-9):
        raise ValueError("SET rising branch must be non-decreasing")
    current = np.empty_like(voltage)
    voltage_hbn = np.empty_like(voltage)
    state_history = np.empty_like(voltage)
    exposure_history = np.empty_like(voltage)
    exposure = 0.0
    previous_magnitude = 0.0
    previous_voltage_hbn = 0.0

    for index, applied in enumerate(magnitude):
        if index <= apex:
            delta_voltage = max(float(applied - previous_magnitude), 0.0)
            candidate = state
            candidate_exposure = exposure
            for _ in range(int(max_fixed_point_iterations)):
                _, candidate_vh = divider_set(
                    applied, candidate, a_pre, a_connected, gate_voltage,
                    transport_law, mapping)
                candidate_exposure = (
                    exposure
                    + 0.5 * (previous_voltage_hbn + candidate_vh)
                    * delta_voltage
                )
                driven = transition_law.state_with_e50(
                    candidate_exposure, e50)
                updated = max(state, float(driven))
                if abs(updated - candidate) <= 1e-10:
                    candidate = updated
                    break
                candidate = 0.5 * candidate + 0.5 * updated
            state = float(np.clip(candidate, state, 1.0))
            idevice, vh = divider_set(
                applied, state, a_pre, a_connected, gate_voltage,
                transport_law, mapping)
            exposure = (
                exposure
                + 0.5 * (previous_voltage_hbn + vh) * delta_voltage
            )
            # One last state/load-line correction makes the stored point
            # self-consistent at the converged exposure.
            state = max(state, float(transition_law.state_with_e50(
                exposure, e50)))
            idevice, vh = divider_set(
                applied, state, a_pre, a_connected, gate_voltage,
                transport_law, mapping)
            previous_magnitude = float(applied)
            previous_voltage_hbn = float(vh)
        else:
            idevice, vh = divider_set(
                applied, state, a_pre, a_connected, gate_voltage,
                transport_law, mapping)
        current[index] = np.sign(voltage[index]) * idevice
        voltage_hbn[index] = np.sign(voltage[index]) * vh
        state_history[index] = state
        exposure_history[index] = exposure

    return {
        "Vs": voltage.copy(),
        "Is": current,
        "Vhs": voltage_hbn,
        "xs": state_history,
        "set_exposure_v2": exposure_history,
        "a_pre": float(a_pre),
        "a_connected": float(a_connected),
        "transition_e50_v2": float(e50),
        "transition_width_v2": float(transition_law.width_v2),
        "Vg": float(gate_voltage),
        "target_set_current_used_for_prediction": False,
    }


def _cap_reset_endpoint(a_hrs, g_lrs, reset_params, reference_voltage=0.1):
    lrs_current = (
        float(g_lrs)
        * np.sinh(reset_params.reset_alpha_lrs_identified * reference_voltage)
    )
    hrs_basis = (
        reference_voltage
        * np.exp(reset_params.reset_pf_beta_per_sqrt_v
                 * np.sqrt(reference_voltage))
    )
    return float(min(float(a_hrs), 0.95 * lrs_current / hrs_basis))


def run_autonomous_full_cycle(
        set_voltage, reset_voltage, gate_voltage, set_transport_law,
        set_transition_law, set_reset_link_law, reset_endpoint_law,
        reset_transport_law, reset_base_params=None, set_mapping=None,
        reset_mapping=None, reset_transition_law=None,
        reset_e50_mvc=None, reset_width_mvc=None,
        set_transition_e50_offset_v2=0.0,
        link_residual_offset_decade=0.0,
        endpoint_residual_offset_decade=0.0):
    """Predict SET and RESET using voltage waveforms only.

    Calibration laws are supplied explicitly.  No target SET or RESET current
    appears in this API; measured currents are reserved for external scoring.
    """
    reset_transport_law.validate_gate(gate_voltage)
    reset_transport_law.validate_protocol(reset_voltage)
    reset_endpoint_law.validate_gate(gate_voltage)
    reset_endpoint_law.validate_protocol(reset_voltage)
    if reset_transition_law is not None:
        if not isinstance(reset_transition_law, PFResetTransitionLaw):
            raise TypeError(
                "reset_transition_law must be a PFResetTransitionLaw")
        reset_transition_law.validate_gate(gate_voltage)
        reset_transition_law.validate_protocol(reset_voltage)
        if reset_e50_mvc is None:
            reset_e50_mvc = reset_transition_law.e50_mvc
        if reset_width_mvc is None:
            reset_width_mvc = reset_transition_law.width_mvc
    base = v8.Params() if reset_base_params is None else reset_base_params
    reset_params = v11.params_with_transport_law(
        base, reset_transport_law)
    if reset_mapping is None:
        reset_mapping = v11.transistor_mapping(reset_params)

    set_result = run_autonomous_set(
        set_voltage, gate_voltage, set_transport_law, set_transition_law,
        mapping=set_mapping,
        transition_e50_offset_v2=set_transition_e50_offset_v2,
    )
    g_lrs = set_reset_link_law.predict_reset_lrs_amplitude(
        set_result["a_connected"],
        set_transport_law.connected_exponent,
        reset_params.reset_alpha_lrs_identified,
        residual_offset_decade=link_residual_offset_decade,
    )
    a_hrs_uncapped = reset_endpoint_law.predict(
        endpoint_residual_offset_decade)
    a_hrs = _cap_reset_endpoint(
        a_hrs_uncapped, g_lrs, reset_params)
    reset_result = v11.run_effective_pf_reset(
        reset_voltage, gate_voltage, g_lrs, a_hrs,
        p=reset_params, mapping=reset_mapping,
        e50_mvc=reset_e50_mvc, width_mvc=reset_width_mvc,
    )
    combined_voltage = np.concatenate(
        [set_result["Vs"], reset_result["Vs"]])
    combined_current = np.concatenate(
        [set_result["Is"], reset_result["Is"]])
    combined_state = np.concatenate(
        [set_result["xs"], reset_result["pbs"]])
    return {
        "set": set_result,
        "reset": reset_result,
        "Vs": combined_voltage,
        "Is": combined_current,
        "connected_fraction": combined_state,
        "g_reset_initial_predicted_a": float(g_lrs),
        "a_reset_hrs_predicted": float(a_hrs),
        "a_reset_hrs_uncapped": float(a_hrs_uncapped),
        "reset_endpoint_capped": bool(a_hrs < a_hrs_uncapped),
        "set_reset_retention_factor": (
            set_reset_link_law.retention_factor),
        "reset_transition_source": (
            "protocol_fitted_PF_envelope_state"
            if reset_transition_law is not None
            else "base_parameter_or_explicit_override"),
        "target_set_current_used_for_prediction": False,
        "target_reset_current_used_for_prediction": False,
    }


def simulate_calibrated_cycle(
        set_voltage, reset_voltage, gate_voltage=1.1, calibration=None,
        reset_transition_mode="same_protocol", **kwargs):
    """Run the central v0.12 calibration from voltage inputs only.

    ``same_protocol`` uses the target-excludable 1.1-V transition law and is
    the default for reproducing this dataset. ``cross_gate_prior`` retains the
    older 0.9/1.3-V transition law, which is less accurate in random holdout
    but more stable across the observed late-cycle distribution shift.
    """
    calibration = (default_calibration() if calibration is None
                   else calibration)
    if not isinstance(calibration, AutonomousCalibration):
        raise TypeError("calibration must be an AutonomousCalibration")
    if reset_transition_mode == "same_protocol":
        transition = calibration.reset_transition
    elif reset_transition_mode == "cross_gate_prior":
        transition = None
    else:
        raise ValueError(
            "reset_transition_mode must be 'same_protocol' or "
            "'cross_gate_prior'")
    return run_autonomous_full_cycle(
        set_voltage,
        reset_voltage,
        gate_voltage,
        calibration.set_transport,
        calibration.set_transition,
        calibration.set_reset_link,
        calibration.reset_endpoint,
        calibration.reset_transport,
        reset_transition_law=transition,
        **kwargs,
    )


def claim_scope():
    """Return the evidence boundary of the autonomous full-cycle model."""
    return {
        "input_after_calibration": (
            "SET voltage waveform, RESET voltage waveform and gate voltage"),
        "set_transport": (
            "effective pre-SET field-assisted and connected super-ohmic "
            "envelopes, coupled to the measured positive transistor load line"),
        "set_state": (
            "irreversible observable connected fraction driven by voltage-domain "
            "exposure; not a filament radius or vacancy concentration"),
        "set_reset_link": (
            "paired-data protocol transfer factor; not a microscopic retention "
            "or ion-migration efficiency"),
        "reset": (
            "measured reverse-transistor load line, internal-field state and "
            "effective field-assisted HRS transport"),
        "reset_transition_modes": (
            "same-protocol 1.1-V state law improves random holdout; the "
            "0.9/1.3-V prior is retained for late-cycle robustness"),
        "nonstationarity": (
            "RESET transition threshold shifts between early and late cycle "
            "groups and is not predictable from voltage input alone"),
        "time_limitation": (
            "quasi-static DC waveform model; changing physical dwell time or "
            "pulse width is outside the current validation domain"),
        "mechanism_limitation": (
            "electrical curves do not uniquely distinguish boron-vacancy and "
            "metal-assisted paths"),
    }
