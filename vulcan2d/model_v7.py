"""VULCAN-2D v0.7: inverse-identified quasi-static RESET state.

This module keeps the v0.6 SET implementation and replaces the RESET magnitude
proxy plus assumed soft-state kinetics with:

* a measured negative-``V_DS`` standalone-transistor lookup;
* a scalar observable state between cycle-specific LRS/HRS envelopes;
* a gate-neutral logistic survival law versus reconstructed h-BN field.

The observable state is the fraction of resettable conductance that remains.
It is the continuum limit of a distribution of local weak-region rupture
thresholds.  It is not a measured vacancy count or filament radius.  Because
the source loops lack reliable point dwell times, this is a voltage-ramp model,
not a calibrated physical-time switching law.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import model_v6 as v6
from .reset_identification import logistic_state
from .transistor import (
    TransistorMapping,
    load_default_lookup,
    load_reverse_lookup,
)


@dataclass
class Params(v6.Params):
    """RESET parameters identified without the 1.1-V holdout condition."""

    # Shared transport coefficients fitted to stable portions of the 0.9-V and
    # 1.3-V RESET ensembles after reverse-1T de-embedding.
    reset_alpha_lrs_identified: float = 8.044375371895217
    reset_alpha_hrs_identified: float = 6.097290909897696

    # Median endpoint-trained observable-state law.  The 1.1-V data are not
    # used in these values and serve as the interpolation holdout.
    reset_e50_mvc: float = 0.837043889173497
    reset_width_mvc: float = 0.1468665790608002
    reset_e50_sigma_mvc: float = 0.319080202201558


def transistor_mapping(p=None, positive_lookup=None, reverse_lookup=None):
    """Measured positive SET and measured negative RESET circuit branches."""
    p = Params() if p is None else p
    positive_lookup = (load_default_lookup() if positive_lookup is None
                       else positive_lookup)
    reverse_lookup = (load_reverse_lookup() if reverse_lookup is None
                      else reverse_lookup)
    return TransistorMapping(
        lookup=positive_lookup,
        gate_gain=p.transistor_gate_gain,
        gate_offset_v=p.transistor_gate_offset_v,
        current_scale=p.transistor_current_scale,
        series_resistance_ohm=p.transistor_series_resistance_ohm,
        reset_policy="measured_reverse",
        reset_lookup=reverse_lookup,
        reset_gate_gain=1.0,
        reset_gate_offset_v=0.0,
    )


def reset_hrs_basis(vh_abs, p=None):
    """Return the dimensionless HRS voltage dependence.

    The v0.7/v0.8 path keeps its original ``sinh(alpha*V_hBN)`` law.  A newer
    parameter object may request the effective Poole--Frenkel-like form

    ``V_hBN * exp(beta_PF * sqrt(V_hBN))``.

    The latter is deliberately a fixed-thickness voltage-domain law.  Its
    fitted ``beta_PF`` must not be interpreted as a dielectric constant or a
    trap energy without temperature-dependent measurements.
    """
    p = Params() if p is None else p
    vh = np.abs(np.asarray(vh_abs, dtype=float))
    transport = getattr(p, "reset_hrs_transport_model", "sinh")
    if transport == "sinh":
        value = np.sinh(np.clip(
            p.reset_alpha_hrs_identified * vh, 0.0, 39.0))
    elif transport == "poole_frenkel_effective":
        beta = float(getattr(p, "reset_pf_beta_per_sqrt_v"))
        value = vh * np.exp(np.clip(beta * np.sqrt(vh), 0.0, 39.0))
    else:
        raise ValueError(f"Unsupported RESET HRS transport: {transport}")
    return float(value) if value.ndim == 0 else value


def reset_hbn_current(vh_abs, state, g_lrs_a, g_hrs_a, p=None):
    """Two-envelope h-BN current at fixed observable RESET state."""
    p = Params() if p is None else p
    vh = abs(float(vh_abs))
    state = float(np.clip(state, 0.0, 1.0))
    i_lrs = float(g_lrs_a) * np.sinh(np.clip(
        p.reset_alpha_lrs_identified * vh, 0.0, 39.0))
    i_hrs = float(g_hrs_a) * reset_hrs_basis(vh, p)
    return float(state * i_lrs + (1.0 - state) * i_hrs)


def divider_reset(vapp_abs, state, g_lrs_a, g_hrs_a, gate_voltage,
                  p=None, mapping=None, nbis=36):
    """Solve one fixed-state RESET load-line intersection."""
    p = Params() if p is None else p
    mapping = transistor_mapping(p) if mapping is None else mapping
    va = abs(float(vapp_abs))
    if va <= 1e-15:
        return 0.0, 0.0
    lo, hi = 0.0, va
    for _ in range(nbis):
        vh = 0.5 * (lo + hi)
        ih = reset_hbn_current(vh, state, g_lrs_a, g_hrs_a, p)
        itr = mapping.current(va - vh, gate_voltage, side="reset")
        if ih < itr:
            lo = vh
        else:
            hi = vh
    vh = 0.5 * (lo + hi)
    return reset_hbn_current(vh, state, g_lrs_a, g_hrs_a, p), vh


def run_identified_reset(voltage_wave, gate_voltage, g_lrs_a, g_hrs_a,
                         p=None, mapping=None, e50_mvc=None,
                         width_mvc=None):
    """Simulate one irreversible quasi-static RESET loop.

    ``g_lrs_a`` and ``g_hrs_a`` are cycle/device boundary states obtained from
    non-disturbing low-voltage portions of the loop or from a separate state
    estimator.  Supplying them is state conditioning, not a claim that the
    present model predicts absolute LRS/HRS conductance from geometry alone.
    """
    p = Params() if p is None else p
    mapping = transistor_mapping(p) if mapping is None else mapping
    voltage = np.asarray(voltage_wave, dtype=float)
    if voltage.ndim != 1 or len(voltage) < 3:
        raise ValueError("voltage_wave must contain a complete RESET loop")
    if g_lrs_a <= 0.0 or g_hrs_a <= 0.0:
        raise ValueError("RESET envelope amplitudes must be positive")
    reference_v = 0.1
    i_lrs_reference = reset_hbn_current(
        reference_v, 1.0, g_lrs_a, g_hrs_a, p)
    i_hrs_reference = reset_hbn_current(
        reference_v, 0.0, g_lrs_a, g_hrs_a, p)
    if i_lrs_reference <= i_hrs_reference:
        raise ValueError(
            "RESET envelopes require LRS current above HRS at 0.1 V")
    e50 = p.reset_e50_mvc if e50_mvc is None else float(e50_mvc)
    width = p.reset_width_mvc if width_mvc is None else float(width_mvc)
    apex = int(np.argmax(np.abs(voltage)))
    current = np.empty_like(voltage)
    vh_hist = np.empty_like(voltage)
    state_hist = np.empty_like(voltage)
    field_hist = np.empty_like(voltage)
    state = 1.0

    for idx, applied in enumerate(voltage):
        magnitude = abs(float(applied))
        _, vh = divider_reset(
            magnitude, state, g_lrs_a, g_hrs_a, gate_voltage, p, mapping)
        field = 10.0 * vh / p.hbn_thickness_nm
        if idx <= apex:
            state = min(state, float(logistic_state(field, e50, width)))
            # Re-solve after the irreversible state update so the recorded
            # terminal point is self-consistent with the new conductance.
            idev, vh = divider_reset(
                magnitude, state, g_lrs_a, g_hrs_a, gate_voltage, p, mapping)
            field = 10.0 * vh / p.hbn_thickness_nm
        else:
            idev, vh = divider_reset(
                magnitude, state, g_lrs_a, g_hrs_a, gate_voltage, p, mapping)
            field = 10.0 * vh / p.hbn_thickness_nm
        current[idx] = np.sign(applied) * idev
        vh_hist[idx] = np.sign(applied) * vh
        state_hist[idx] = state
        field_hist[idx] = field

    return {
        "Vs": voltage.copy(),
        "Is": current,
        "Vhs": vh_hist,
        "pbs": state_hist,
        "Ehs_MV_cm": field_hist,
        "g_lrs_a": float(g_lrs_a),
        "g_hrs_a": float(g_hrs_a),
        "e50_mvc": float(e50),
        "width_mvc": float(width),
        "Vg": float(gate_voltage),
    }


def simulate_1t1r_set_ensemble(p=None, **kwargs):
    """Keep the measured-positive-1T SET implementation from v0.6."""
    p = Params() if p is None else p
    mapping = kwargs.pop("transistor_mapping", None) or transistor_mapping(p)
    return v6.simulate_1t1r_set_ensemble(
        p, transistor_mapping=mapping, **kwargs)


def claim_scope(p=None):
    p = Params() if p is None else p
    return {
        "transistor_set": "public Nature Fig. 2b positive-V_DS lookup",
        "transistor_reset": "Bookoutput2 measured negative-V_DS lookup",
        "reset_state": (
            "observable fraction of resettable conductance between identified "
            "cycle LRS/HRS envelopes"
        ),
        "reset_drive": "reconstructed average h-BN field only; no terminal-voltage blend",
        "gate_dependence": (
            "none in the h-BN RESET survival law; gate enters through the "
            "measured transistor and the conditioned initial state"
        ),
        "time_limitation": (
            "quasi-static voltage-ramp identification; no physical time constant"
        ),
        "microscopic_limitation": (
            "state is not uniquely assigned to metal ions or boron vacancies"
        ),
    }
