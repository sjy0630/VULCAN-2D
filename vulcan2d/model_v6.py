"""VULCAN-2D v0.6: measured-transistor load line with a gate-neutral h-BN layer.

v0.5 allowed gate voltage to enter both an effective transistor curve and six
RESET transport/state interpolants.  v0.6 removes that double counting:

* the transistor branch is the public Nature Fig. 2b ``I_D(V_DS, V_G)`` lookup;
* one endpoint-calibrated affine gate mapping represents the unresolved 1T to
  1T1R electrical offset, outside the h-BN material model;
* the h-BN RESET transport and state parameters are shared across gate voltage;
* the series divider still solves ``V_app = V_hBN + V_DS`` self-consistently.

Fig. 2b has no negative-V_DS data.  RESET currently uses its positive-V_DS
magnitude as a declared proxy, not as a measured reverse-bias characteristic.
That limitation is deliberately surfaced in validation output.
"""

from dataclasses import dataclass
from math import sqrt

from . import model_v4 as v5
from .transistor import TransistorMapping, load_default_lookup


@dataclass
class Params(v5.Params):
    """v0.6 parameters with no gate-indexed h-BN RESET compensation."""

    # SET endpoints (0.9 and 1.3 V only) identify this affine electrical map.
    # The held-out 1.1 V condition is not used.  These are circuit mapping
    # coefficients, not MOSFET threshold voltage or h-BN material parameters.
    transistor_gate_gain: float = 1.0953794419765472
    transistor_gate_offset_v: float = -0.3090066462755203
    transistor_current_scale: float = 1.0
    transistor_series_resistance_ohm: float = 0.0

    # Do not let SET compliance alter the intrinsic h-BN path prefactor.
    gate_path_exponent: float = 0.0

    # One shared h-BN RESET law.  Geometric endpoint pooling uses only the
    # v0.5 0.9/1.3-V endpoint fits and removes every explicit 1.1-V anchor.
    reset_connected_scale_low: float = sqrt(0.0022 * 0.42)
    reset_connected_scale_ref: float = sqrt(0.0022 * 0.42)
    reset_connected_scale_high: float = sqrt(0.0022 * 0.42)
    reset_off_scale_low: float = sqrt(0.008 * 0.95)
    reset_off_scale_ref: float = sqrt(0.008 * 0.95)
    reset_off_scale_high: float = sqrt(0.008 * 0.95)
    alpha_reset_low_v4: float = sqrt(7.53 * 6.51)
    alpha_reset_ref_v4: float = sqrt(7.53 * 6.51)
    alpha_reset_high_v4: float = sqrt(7.53 * 6.51)
    alpha_reset_on_low_v4: float = sqrt(10.0 * 7.666)
    alpha_reset_on_ref_v4: float = sqrt(10.0 * 7.666)
    alpha_reset_on_high_v4: float = sqrt(10.0 * 7.666)
    reset_drive_scale_low: float = sqrt(0.62 * 0.90)
    reset_drive_scale_ref: float = sqrt(0.62 * 0.90)
    reset_drive_scale_high: float = sqrt(0.62 * 0.90)
    reset_threshold_factor_low: float = sqrt(0.44 * 1.00)
    reset_threshold_factor_ref: float = sqrt(0.44 * 1.00)
    reset_threshold_factor_high: float = sqrt(0.44 * 1.00)


def transistor_mapping(p=None, lookup=None):
    p = Params() if p is None else p
    lookup = load_default_lookup() if lookup is None else lookup
    return TransistorMapping(
        lookup=lookup,
        gate_gain=p.transistor_gate_gain,
        gate_offset_v=p.transistor_gate_offset_v,
        current_scale=p.transistor_current_scale,
        series_resistance_ohm=p.transistor_series_resistance_ohm,
        reset_policy="magnitude_proxy",
    )


def simulate_1t1r_set_ensemble(p=None, **kwargs):
    p = Params() if p is None else p
    mapping = kwargs.pop("transistor_mapping", None) or transistor_mapping(p)
    return v5.simulate_1t1r_set_ensemble(
        p, transistor_mapping=mapping, **kwargs)


def simulate_1t1r_cycle_ensemble(p=None, **kwargs):
    p = Params() if p is None else p
    mapping = kwargs.pop("transistor_mapping", None) or transistor_mapping(p)
    return v5.simulate_1t1r_cycle_ensemble(
        p, transistor_mapping=mapping, **kwargs)


def simulate_1t1r_reset_ensemble(p=None, **kwargs):
    p = Params() if p is None else p
    mapping = kwargs.pop("transistor_mapping", None) or transistor_mapping(p)
    return v5.simulate_1t1r_reset_ensemble(
        p, transistor_mapping=mapping, **kwargs)


def claim_scope(p=None):
    p = Params() if p is None else p
    mapping = transistor_mapping(p)
    return {
        "transistor": "public Nature Fig. 2b lookup, bilinear in V_DS and V_G",
        "gate_mapping": {
            "gain": mapping.gate_gain,
            "offset_V": mapping.gate_offset_v,
            "fit_conditions_V": [p.Vg_low, p.Vg_high],
            "held_out_V": p.Vg_ref,
            "meaning": "1T-to-1T1R electrical mapping; not an h-BN parameter",
        },
        "hbn_reset": "one gate-neutral transport/state law pooled from endpoints",
        "reset_limitation": (
            "Fig. 2b contains only non-negative V_DS; RESET uses a magnitude "
            "proxy until reverse-orientation transistor data are available"
        ),
    }
