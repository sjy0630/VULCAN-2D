"""Audited domain for the v0.12 quasi-static full-cycle laws.

The two polarity-specific state coordinates are deliberately kept separate.
The calibrated SET-to-RESET link maps a connected SET envelope to a RESET
LRS envelope; it has not been calibrated after partial SET. No new transport
law or microscopic state-transfer hypothesis is introduced here.
"""
from __future__ import annotations

import numpy as np

from . import model_v12 as legacy
from .set_identification import direct_set_transistor_mapping
from .transistor import TransistorMapping, measured_transistor_mapping


# Numerical tolerance for using the saturated-envelope approximation. This is
# neither a measured material threshold nor a bound on prediction error.
HANDOFF_SATURATION_TOLERANCE = 1e-3


class HandoffDomainError(ValueError):
    """The simulated SET endpoint is outside the calibrated handoff domain."""

    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def _shared_transistor(set_mapping, reset_mapping, reset_base_params,
                       transistor_mapping):
    """Resolve historical branch options without silently splitting one device."""
    if all(value is None for value in (
            set_mapping, reset_mapping, reset_base_params, transistor_mapping)):
        return measured_transistor_mapping()
    if transistor_mapping is not None:
        if set_mapping is not None or reset_mapping is not None:
            raise ValueError(
                "Use transistor_mapping alone for the same transistor in both phases")
        mapping = transistor_mapping
    else:
        # Old validation callers supply the independently calibrated RESET
        # branch. Only its actually used reverse settings are compared: its
        # unused historical positive affine gate map is not part of this law.
        positive = (direct_set_transistor_mapping()
                    if set_mapping is None else set_mapping)
        reverse = (legacy.v11.transistor_mapping(reset_base_params)
                   if reset_mapping is None else reset_mapping)
        if not isinstance(positive, TransistorMapping) or not isinstance(reverse, TransistorMapping):
            raise TypeError("Shared-device mappings must be TransistorMapping objects")
        forward_settings = (positive.gate_gain, positive.gate_offset_v,
                            positive.current_scale, positive.series_resistance_ohm)
        reverse_settings = (reverse.reset_gate_gain, reverse.reset_gate_offset_v,
                            reverse.current_scale, reverse.series_resistance_ohm)
        if forward_settings != reverse_settings:
            raise ValueError(
                "SET and RESET must use the same transistor parameters; "
                "pass one bipolar transistor_mapping for a shared-device change. "
                "Historical independent branch hypotheses remain in model_v12.")
        mapping = TransistorMapping(
            lookup=positive.lookup, reset_lookup=reverse.reset_lookup,
            reset_policy=reverse.reset_policy,
            gate_gain=positive.gate_gain, gate_offset_v=positive.gate_offset_v,
            reset_gate_gain=reverse.reset_gate_gain,
            reset_gate_offset_v=reverse.reset_gate_offset_v,
            current_scale=positive.current_scale,
            series_resistance_ohm=positive.series_resistance_ohm)
    if not isinstance(mapping, TransistorMapping):
        raise TypeError("transistor_mapping must be a TransistorMapping")
    values = (mapping.gate_gain, mapping.gate_offset_v, mapping.reset_gate_gain,
              mapping.reset_gate_offset_v, mapping.current_scale,
              mapping.series_resistance_ohm)
    if (not np.all(np.isfinite(values)) or mapping.current_scale <= 0.0
            or mapping.series_resistance_ohm < 0.0):
        raise ValueError("Shared transistor parameters must be finite and passive")
    if ((mapping.gate_gain, mapping.gate_offset_v)
            != (mapping.reset_gate_gain, mapping.reset_gate_offset_v)):
        raise ValueError("Both phases must keep the same transistor gate reference")
    if mapping.reset_policy != "measured_reverse" or mapping.reset_lookup is None:
        raise ValueError("The same transistor requires measured reverse data for RESET")
    return mapping


def audit_set_handoff(set_result, transport_law, link_law):
    """Check the final SET state against its connected reference envelope.

    The reference is an *internal h-BN* voltage, not the final applied voltage
    (usually zero). The current check alone can be uninformative when the two
    envelopes almost coincide, so closeness of x to one is required as well.
    """
    reference = float(link_law.reference_voltage_v)
    states = np.asarray(set_result["xs"], dtype=float)
    if states.ndim != 1 or len(states) == 0:
        raise ValueError("SET state history must be a nonempty vector")
    state = float(states[-1])
    if not np.isfinite(reference) or reference <= 0.0:
        raise ValueError("Handoff reference voltage must be positive and finite")
    if not np.isfinite(state) or not 0.0 <= state <= 1.0:
        raise ValueError("Final SET state must be finite and between zero and one")
    args = (set_result["a_pre"], set_result["a_connected"], transport_law)
    pre = legacy.set_hbn_current(reference, 0.0, *args)
    connected = legacy.set_hbn_current(reference, 1.0, *args)
    final = legacy.set_hbn_current(reference, state, *args)
    if (not np.all(np.isfinite([pre, connected, final]))
            or not connected > pre > 0.0):
        raise ValueError(
            "Handoff requires finite, positive, ordered SET envelopes "
            "at the internal reference voltage")
    mismatch = float(abs(final - connected) / connected)
    diagnostics = {
        "method": "saturated_SET_envelope_to_RESET_LRS_envelope",
        "set_end_state": state,
        "set_state_distance_from_connected": 1.0 - state,
        "reference_voltage_hbn_v": reference,
        "set_reference_current_a": final,
        "pre_reference_current_a": pre,
        "connected_reference_current_a": connected,
        "reference_current_relative_error": mismatch,
        "saturation_tolerance": HANDOFF_SATURATION_TOLERANCE,
        "tolerance_meaning": "numerical saturated-envelope approximation",
        "partial_set_mapping_calibrated": False,
        "unified_physical_state": False,
    }
    # Tiny roundoff allowance makes the inclusive tolerance deterministic.
    limit = HANDOFF_SATURATION_TOLERANCE + 8 * np.finfo(float).eps
    if 1.0 - state > limit or mismatch > limit:
        raise HandoffDomainError(
            "The SET-to-RESET mapping after partial SET is not calibrated: "
            f"x_end={state:.6g}, reference-current mismatch={mismatch:.6g}. "
            "Use run_autonomous_set for SET-only diagnostics; the full-cycle "
            "API requires a saturated SET endpoint.", diagnostics)
    return diagnostics


def run_autonomous_full_cycle(
        set_voltage, reset_voltage, gate_voltage, set_transport_law,
        set_transition_law, set_reset_link_law, reset_endpoint_law,
        reset_transport_law, reset_base_params=None, set_mapping=None,
        reset_mapping=None, reset_transition_law=None,
        reset_e50_mvc=None, reset_width_mvc=None,
        set_transition_e50_offset_v2=0.0,
        link_residual_offset_decade=0.0,
        endpoint_residual_offset_decade=0.0, transistor_mapping=None):
    """Return v0.12 currents only within the audited envelope-handoff domain.

    SET ``xs`` and RESET ``pbs`` remain distinct observable coordinates.
    No combined physical-state array is returned. One bipolar transistor
    mapping is used in both phases. Legacy per-phase options must agree on
    the transistor scale, access resistance and physical gate reference.
    Default transport/calibration is unchanged; an invalid handoff raises
    before any full-cycle result is returned to the caller.
    """
    for name, value in (
            ("reset_e50_mvc", reset_e50_mvc),
            ("reset_width_mvc", reset_width_mvc),
            ("link_residual_offset_decade", link_residual_offset_decade),
            ("endpoint_residual_offset_decade", endpoint_residual_offset_decade)):
        if value is not None and not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    factor = float(set_reset_link_law.retention_factor)
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError("SET-to-RESET conversion factor must be positive and finite")
    mapping = _shared_transistor(
        set_mapping, reset_mapping, reset_base_params, transistor_mapping)
    result = legacy.run_autonomous_full_cycle(
        set_voltage, reset_voltage, gate_voltage, set_transport_law,
        set_transition_law, set_reset_link_law, reset_endpoint_law,
        reset_transport_law, reset_base_params=reset_base_params,
        set_mapping=mapping, reset_mapping=mapping,
        reset_transition_law=reset_transition_law,
        reset_e50_mvc=reset_e50_mvc, reset_width_mvc=reset_width_mvc,
        set_transition_e50_offset_v2=set_transition_e50_offset_v2,
        link_residual_offset_decade=link_residual_offset_decade,
        endpoint_residual_offset_decade=endpoint_residual_offset_decade,
    )
    reset = result["reset"]
    # A NaN threshold can leave finite currents because the legacy irreversible
    # min update retains the old state. Check parameters as well as arrays.
    if (not np.isfinite(reset["e50_mvc"])
            or not np.isfinite(reset["width_mvc"])
            or reset["width_mvc"] <= 0.0):
        raise ValueError("RESET threshold must be finite and width positive and finite")
    for phase, keys in (("set", ("Vs", "Is", "Vhs", "xs")),
                        ("reset", ("Vs", "Is", "Vhs", "pbs", "Ehs_MV_cm"))):
        for key in keys:
            if not np.all(np.isfinite(result[phase][key])):
                raise ValueError(f"Nonfinite model output: {phase}.{key}")
    for key in ("g_reset_initial_predicted_a", "a_reset_hrs_predicted"):
        if not np.isfinite(result[key]) or result[key] <= 0.0:
            raise ValueError(f"{key} must be positive and finite")
    handoff = audit_set_handoff(
        result["set"], set_transport_law, set_reset_link_law)
    handoff["reset_lrs_envelope_amplitude_a"] = (
        result["g_reset_initial_predicted_a"])
    handoff["reset_first_recorded_state"] = float(result["reset"]["pbs"][0])
    result.pop("connected_fraction")
    result["handoff"] = handoff
    result["state_coordinates"] = {
        "set.xs": "fraction between positive pre-SET and connected envelopes",
        "reset.pbs": "fraction between negative HRS and LRS envelopes",
    }
    result["model_version"] = "0.13.1"
    result["transistor_consistency"] = {
        "shared_between_phases": True,
        "law": "signed_measured_fixed_physical_source",
        "gate_reference": "V_G-V_S; physical source held at zero",
        "gate_gain": mapping.gate_gain,
        "gate_offset_v": mapping.gate_offset_v,
        "current_scale": mapping.current_scale,
        "series_resistance_ohm": mapping.series_resistance_ohm,
        "same_physical_device_confirmed": False,
        "microscopic_mos_model": False,
    }
    if reset_e50_mvc is not None or reset_width_mvc is not None:
        result["reset_transition_source"] = "explicit_parameter_override"
    return result


def simulate_calibrated_cycle(
        set_voltage, reset_voltage, gate_voltage=1.1, calibration=None,
        reset_transition_mode="same_protocol", **kwargs):
    """Run the existing central calibration with an explicit handoff check."""
    calibration = (legacy.default_calibration() if calibration is None
                   else calibration)
    if not isinstance(calibration, legacy.AutonomousCalibration):
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
        set_voltage, reset_voltage, gate_voltage,
        calibration.set_transport, calibration.set_transition,
        calibration.set_reset_link, calibration.reset_endpoint,
        calibration.reset_transport, reset_transition_law=transition,
        **kwargs,
    )


def claim_scope():
    """Describe the current interpretation without adding mechanism claims."""
    scope = dict(legacy.claim_scope())
    scope.update({
        "set_reset_link": (
            "protocol conversion of a saturated SET envelope into a RESET "
            "LRS envelope; partial SET mapping is not calibrated"),
        "state_coordinates": (
            "SET x and RESET p use different polarity-specific envelopes; "
            "they are not one continuous microscopic fraction"),
        "history_limitation": (
            "SET paths ending at the same calibrated connected envelope "
            "produce the same central RESET law; finer history is unresolved"),
        "validation": (
            "target-excluding parameter fits with current-based training "
            "eligibility; model selection and chronological diagnostics "
            "are retrospective"),
    })
    return scope
