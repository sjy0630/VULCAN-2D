"""Effective field-assisted HRS transport for the full-cycle RESET model.

This revision keeps the measured reverse-transistor load line, the SET-to-LRS
handoff and the irreversible internal-field RESET state.  It changes only the
post-RESET HRS envelope from a generic ``sinh`` curve to the protocol-fitted,
effective Poole--Frenkel-like curvature implemented in :mod:`hrs_transport`.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np

from . import model_v7 as v7
from . import model_v8 as v8
from . import cycle_link as cl
from .hrs_transport import EffectivePooleFrenkelLaw


@dataclass
class Params(v8.Params):
    """RESET parameters with an effective Poole--Frenkel-like HRS branch."""

    reset_hrs_transport_model: str = "poole_frenkel_effective"
    # Full-group deployment value. Strict validation refits this coefficient
    # without the target cycle instead of reading this default.
    reset_pf_beta_per_sqrt_v: float = 6.28586925832311


def params_with_transport_law(base_params, law):
    """Copy calibrated legacy parameters and attach one fitted HRS law."""
    if not isinstance(law, EffectivePooleFrenkelLaw):
        raise TypeError("law must be an EffectivePooleFrenkelLaw")
    inherited = {
        field.name: getattr(base_params, field.name)
        for field in fields(v8.Params)
    }
    return replace(
        Params(),
        **inherited,
        reset_hrs_transport_model="poole_frenkel_effective",
        reset_pf_beta_per_sqrt_v=law.beta_per_sqrt_v,
    )


def transistor_mapping(p=None, positive_lookup=None, reverse_lookup=None):
    """Use the same measured positive/negative transistor lookups."""
    p = Params() if p is None else p
    return v8.transistor_mapping(p, positive_lookup, reverse_lookup)


def run_effective_pf_reset(voltage_wave, gate_voltage, g_lrs_a, a_hrs,
                           p=None, mapping=None, e50_mvc=None,
                           width_mvc=None):
    """Run RESET with the effective field-assisted HRS envelope."""
    p = Params() if p is None else p
    if p.reset_hrs_transport_model != "poole_frenkel_effective":
        raise ValueError("run_effective_pf_reset requires the PF HRS branch")
    result = v7.run_identified_reset(
        voltage_wave,
        gate_voltage,
        g_lrs_a,
        a_hrs,
        p=p,
        mapping=mapping,
        e50_mvc=e50_mvc,
        width_mvc=width_mvc,
    )
    result.update({
        "hrs_transport_model": p.reset_hrs_transport_model,
        "reset_pf_beta_per_sqrt_v": p.reset_pf_beta_per_sqrt_v,
        "a_hrs_effective": float(a_hrs),
    })
    return result


def run_refined_set_conditioned_cycle(
        set_voltage, set_current, reset_voltage, gate_voltage,
        transfer_law, endpoint_law, transport_law, base_params=None,
        mapping=None, e50_mvc=None, width_mvc=None):
    """Run measured SET -> predicted RESET with no target RESET current.

    All three statistical laws must already have been fitted on an eligible
    training population.  Cross-validation is responsible for excluding the
    target RESET labels from those fits.
    """
    transport_law.validate_gate(gate_voltage)
    transport_law.validate_protocol(reset_voltage)
    endpoint_law.validate_gate(gate_voltage)
    endpoint_law.validate_protocol(reset_voltage)
    base = v8.Params() if base_params is None else base_params
    p = params_with_transport_law(base, transport_law)
    set_v, set_i = cl._same_length_vectors(set_voltage, set_current)
    exposure = cl.set_forward_exposure(set_v, set_i)
    g_lrs = transfer_law.predict(exposure)
    a_hrs = endpoint_law.predict()
    result = run_effective_pf_reset(
        reset_voltage,
        gate_voltage,
        g_lrs,
        a_hrs,
        p=p,
        mapping=mapping,
        e50_mvc=e50_mvc,
        width_mvc=(p.reset_width_mvc if width_mvc is None else width_mvc),
    )
    result.update({
        "set_voltage_v": np.asarray(set_v, dtype=float),
        "set_current_a": np.asarray(set_i, dtype=float),
        "set_exposure_A_V": exposure,
        "g_reset_initial_predicted_a": g_lrs,
        "state_source": "predicted_from_preceding_set",
        "endpoint_source": "protocol_conditioned_pf_amplitude_attractor",
        "endpoint_protocol_label": endpoint_law.protocol_label,
        "endpoint_sigma_decade": endpoint_law.residual_sigma_decade,
        "set_reset_transfer_sigma_decade": (
            transfer_law.residual_sigma_decade),
        "target_reset_current_used_for_prediction": False,
    })
    return result


def claim_scope(p=None):
    """Return the inherited model boundary plus the HRS refinement."""
    p = Params() if p is None else p
    scope = dict(v8.claim_scope(p))
    scope.update({
        "absolute_reset_boundary": (
            "G_LRS is predicted from the preceding measured SET trace; "
            "the effective HRS amplitude is supplied by a fixed-gate, "
            "fixed-protocol endpoint distribution"
        ),
        "endpoint_leakage": (
            "strict validation excludes the target endpoint amplitude from "
            "its protocol distribution"
        ),
        "validated_gate_range": (
            "the transition law spans the 0.9--1.3 V interpolation range; "
            "the effective HRS curvature is currently validated at 1.1 V"
        ),
        "remaining_endpoint_limitation": (
            "the endpoint distribution is protocol-specific and can include "
            "cycle-history or batch effects"
        ),
        "hrs_transport": (
            "effective V_hBN*exp(beta_PF*sqrt(V_hBN)) curvature fitted "
            "within one gate and DC RESET protocol"
        ),
        "hrs_transport_limit": (
            "beta_PF is a compact voltage-domain coefficient; it is not a "
            "trap energy or dielectric constant without temperature data"
        ),
    })
    return scope
