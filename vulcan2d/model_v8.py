"""Read-initialized quasi-static RESET prediction for VULCAN-2D.

The RESET transition law and reverse-transistor load line are inherited from
``model_v7``.  The difference is the boundary condition: the model estimates
the initial LRS envelope from the non-disturbing 0.08--0.12 V portion at the
start of the measured RESET ramp, then predicts the final HRS envelope from a
gate-conditioned ratio identified only from the 0.9-V and 1.3-V ensembles.

No point on the post-RESET return branch is used.  This closes the endpoint
leakage in v0.7, but it remains a read-initialized, quasi-static model rather
than a geometry-only or physical-time prediction.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from . import model_v7 as v7


@dataclass
class Params(v7.Params):
    """RESET endpoint law trained on 0.9-V and 1.3-V ensembles only."""

    # Group-median log10(G_HRS/G_LRS) at the interpolation midpoint.
    reset_log10_ratio_ref: float = -1.7986903428238656
    reset_ratio_gate_ref_v: float = 1.1
    # Linear interpolation through the two training-group medians.
    reset_log10_ratio_slope_per_v: float = -1.405653208352818
    reset_ratio_gate_min_v: float = 0.9
    reset_ratio_gate_max_v: float = 1.3


def estimate_initial_lrs(cycle, alpha_lrs=None, low_v=0.08, high_v=0.12,
                         min_points=2, p=None):
    """Estimate the LRS amplitude from the initial low-voltage RESET segment.

    Only valid points before the applied-voltage apex are eligible.  In
    particular, low-voltage points on the post-RESET return branch are never
    read, so the fitted value cannot contain the final HRS endpoint.
    """
    p = Params() if p is None else p
    alpha = (p.reset_alpha_lrs_identified if alpha_lrs is None
             else float(alpha_lrs))
    index = np.arange(len(cycle.voltage_hbn_v))
    mask = (
        np.asarray(cycle.valid, dtype=bool)
        & (index <= int(cycle.apex_index))
        & (np.asarray(cycle.voltage_hbn_v) >= float(low_v))
        & (np.asarray(cycle.voltage_hbn_v) <= float(high_v))
        & (np.asarray(cycle.current_a) > 1e-13)
    )
    if np.count_nonzero(mask) < int(min_points):
        raise ValueError("Insufficient initial low-voltage points for LRS read")
    vh = np.asarray(cycle.voltage_hbn_v, dtype=float)[mask]
    current = np.asarray(cycle.current_a, dtype=float)[mask]
    denominator = np.sinh(np.clip(alpha * vh, 0.0, 39.0))
    valid = np.isfinite(current) & np.isfinite(denominator) & (denominator > 0)
    if np.count_nonzero(valid) < int(min_points):
        raise ValueError("Initial LRS read contains invalid points")
    return float(10.0 ** np.median(
        np.log10(current[valid]) - np.log10(denominator[valid])
    ))


def predicted_hrs_ratio(gate_voltage, p=None):
    """Predict ``G_HRS/G_LRS`` inside the measured 0.9--1.3 V gate range.

    Values outside that range use the nearest training boundary because the
    present data support interpolation, not unconstrained extrapolation.
    """
    p = Params() if p is None else p
    gate = float(np.clip(
        gate_voltage, p.reset_ratio_gate_min_v, p.reset_ratio_gate_max_v))
    log_ratio = (
        p.reset_log10_ratio_ref
        + p.reset_log10_ratio_slope_per_v
        * (gate - p.reset_ratio_gate_ref_v)
    )
    return float(np.clip(10.0 ** log_ratio, 1e-6, 0.95))


def fit_endpoint_ratio_params(gate_voltage, g_lrs, g_hrs, p=None,
                              reference_gate_v=1.1):
    """Fit a robust gate interpolation to training-group endpoint ratios.

    Each gate condition contributes one median ``log10(G_HRS/G_LRS)`` value,
    so a condition with more accepted cycles cannot dominate the fit.  The
    fitted slope is constrained to be non-positive, matching the observed
    deeper RESET at the higher-gate training condition.
    """
    p = Params() if p is None else p
    gate = np.asarray(gate_voltage, dtype=float)
    lrs = np.asarray(g_lrs, dtype=float)
    hrs = np.asarray(g_hrs, dtype=float)
    if gate.shape != lrs.shape or gate.shape != hrs.shape or gate.ndim != 1:
        raise ValueError("gate_voltage, g_lrs and g_hrs must be same-length vectors")
    valid = (
        np.isfinite(gate) & np.isfinite(lrs) & np.isfinite(hrs)
        & (lrs > hrs) & (hrs > 0.0)
    )
    unique = np.unique(gate[valid])
    if len(unique) < 2:
        raise ValueError("At least two training gate conditions are required")
    group_log_ratio = np.asarray([
        np.median(np.log10(hrs[valid & (gate == value)] /
                           lrs[valid & (gate == value)]))
        for value in unique
    ])
    slope, intercept = np.polyfit(unique, group_log_ratio, 1)
    slope = min(float(slope), 0.0)
    reference = float(reference_gate_v)
    log_ratio_ref = float(intercept + slope * reference)
    return replace(
        p,
        reset_log10_ratio_ref=log_ratio_ref,
        reset_ratio_gate_ref_v=reference,
        reset_log10_ratio_slope_per_v=slope,
        reset_ratio_gate_min_v=float(unique[0]),
        reset_ratio_gate_max_v=float(unique[-1]),
    )


def run_read_initialized_reset(voltage_wave, gate_voltage, g_lrs_read_a,
                               p=None, mapping=None, e50_mvc=None,
                               width_mvc=None):
    """Predict a RESET loop using the initial read state and no HRS endpoint."""
    p = Params() if p is None else p
    g_lrs = float(g_lrs_read_a)
    if not np.isfinite(g_lrs) or g_lrs <= 0.0:
        raise ValueError("g_lrs_read_a must be a positive finite value")
    g_hrs = g_lrs * predicted_hrs_ratio(gate_voltage, p)
    result = v7.run_identified_reset(
        voltage_wave,
        gate_voltage,
        g_lrs,
        g_hrs,
        p=p,
        mapping=mapping,
        e50_mvc=e50_mvc,
        width_mvc=width_mvc,
    )
    result["endpoint_source"] = "predicted_from_initial_read_and_gate"
    result["hrs_lrs_ratio"] = float(g_hrs / g_lrs)
    return result


def transistor_mapping(p=None, positive_lookup=None, reverse_lookup=None):
    """Use the same measured SET/RESET transistor mapping as v0.7."""
    p = Params() if p is None else p
    return v7.transistor_mapping(p, positive_lookup, reverse_lookup)


def simulate_1t1r_set_ensemble(p=None, **kwargs):
    """Keep the measured-positive-1T SET implementation from v0.7."""
    p = Params() if p is None else p
    return v7.simulate_1t1r_set_ensemble(p, **kwargs)


def claim_scope(p=None):
    p = Params() if p is None else p
    scope = dict(v7.claim_scope(p))
    scope.update({
        "absolute_reset_boundary": (
            "G_LRS is estimated from the initial 0.08--0.12 V RESET read; "
            "G_HRS/G_LRS is interpolated from 0.9-V and 1.3-V training groups"
        ),
        "endpoint_leakage": "no post-RESET HRS point is used for prediction",
        "validated_gate_range": "0.9--1.3 V interpolation only",
        "remaining_endpoint_limitation": (
            "gate-conditioned HRS ratio is phenomenological and may include "
            "batch/protocol effects; it is not an atomistic rupture law"
        ),
    })
    return scope
