"""SET-conditioned hand-off to the VULCAN-2D RESET model.

The experimentally recorded ``S_i`` curve is paired with the immediately
following ``R_i`` curve.  A voltage-domain SET exposure proxy is calculated
from the forward SET ramp and mapped to the conductance at the start of RESET.
The target RESET current is deliberately absent from the prediction API.

Because the dwell time and sweep rate were not recorded, ``integral |I| dV``
is neither labelled charge nor energy.  It is a protocol-specific exposure
proxy; for a common linear ramp it is proportional to transferred charge.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np

from . import model_v7 as v7
from . import model_v8 as v8


@dataclass(frozen=True)
class SetResetTransferLaw:
    """Centered log-linear map from SET exposure to initial RESET state."""

    log10_exposure_ref: float
    log10_g_ref: float
    slope: float
    residual_sigma_decade: float
    n_training: int

    def predict(self, exposure_a_v, residual_offset_decade=0.0):
        """Predict the initial RESET conductance-amplitude parameter."""
        exposure = float(exposure_a_v)
        if not np.isfinite(exposure) or exposure <= 0.0:
            raise ValueError("exposure_a_v must be a positive finite value")
        log_g = (
            self.log10_g_ref
            + self.slope * (np.log10(exposure) - self.log10_exposure_ref)
            + float(residual_offset_decade)
        )
        return float(10.0 ** log_g)


@dataclass(frozen=True)
class ResetEndpointAttractorLaw:
    """Protocol-specific distribution of the final RESET HRS amplitude.

    Within one fixed gate and DC protocol, the final HRS is represented by a
    log-domain center and residual scatter.  It is intentionally independent
    of the predicted LRS scale unless future data demonstrate a reproducible
    coupling.
    """

    gate_voltage_v: float
    protocol_label: str
    log10_g_hrs_ref: float
    residual_sigma_decade: float
    n_training: int
    reset_peak_abs_v: float = 1.7
    reset_peak_tolerance_v: float = 0.03

    def validate_gate(self, gate_voltage):
        gate = float(gate_voltage)
        if not np.isfinite(gate) or not np.isclose(
                gate, self.gate_voltage_v, rtol=0.0, atol=1e-9):
            raise ValueError(
                "endpoint attractor gate voltage does not match prediction")

    def predict(self, residual_offset_decade=0.0):
        offset = float(residual_offset_decade)
        if not np.isfinite(offset):
            raise ValueError("residual_offset_decade must be finite")
        return float(10.0 ** (self.log10_g_hrs_ref + offset))

    def validate_protocol(self, reset_voltage):
        voltage = np.asarray(reset_voltage, dtype=float)
        finite = voltage[np.isfinite(voltage)]
        if finite.ndim != 1 or len(finite) < 3:
            raise ValueError("RESET protocol must contain finite voltage points")
        peak = float(np.max(np.abs(finite)))
        if (not np.isclose(peak, self.reset_peak_abs_v, rtol=0.0,
                           atol=self.reset_peak_tolerance_v)
                or np.min(finite) >= 0.0):
            raise ValueError(
                "RESET peak does not match the endpoint-attractor protocol")


def _same_length_vectors(voltage, current):
    voltage = np.asarray(voltage, dtype=float)
    current = np.asarray(current, dtype=float)
    if (voltage.ndim != 1 or current.ndim != 1
            or len(voltage) != len(current) or len(voltage) < 3):
        raise ValueError("voltage and current must be same-length vectors")
    finite = np.isfinite(voltage) & np.isfinite(current)
    if np.count_nonzero(finite) < 3:
        raise ValueError("voltage and current contain too few finite points")
    return voltage[finite], current[finite]


def set_forward_exposure(voltage, current):
    """Return ``integral |I| dV`` on the initial, rising SET branch only.

    The first maximum positive voltage is the branch boundary.  Return-branch
    samples are excluded so the feature describes SET drive rather than the
    subsequent low-voltage LRS readout.
    """
    voltage, current = _same_length_vectors(voltage, current)
    apex = int(np.argmax(voltage))
    if apex < 1:
        raise ValueError("SET waveform has no rising positive-voltage branch")
    forward_v = voltage[:apex + 1]
    forward_i = np.abs(current[:apex + 1])
    if np.any(np.diff(forward_v) < -1e-9):
        raise ValueError("SET forward branch must be non-decreasing")
    exposure = float(np.trapezoid(forward_i, forward_v))
    if not np.isfinite(exposure) or exposure <= 0.0:
        raise ValueError("SET exposure must be positive and finite")
    return exposure


def fit_set_reset_transfer(exposure_a_v, g_reset_initial_a):
    """Fit a centered ordinary-least-squares SET-to-RESET hand-off law."""
    exposure = np.asarray(exposure_a_v, dtype=float)
    conductance = np.asarray(g_reset_initial_a, dtype=float)
    if (exposure.ndim != 1 or conductance.ndim != 1
            or exposure.shape != conductance.shape):
        raise ValueError("exposure and conductance must be same-length vectors")
    valid = (
        np.isfinite(exposure) & np.isfinite(conductance)
        & (exposure > 0.0) & (conductance > 0.0)
    )
    if np.count_nonzero(valid) < 3:
        raise ValueError("At least three valid paired cycles are required")
    x = np.log10(exposure[valid])
    y = np.log10(conductance[valid])
    x_ref = float(np.median(x))
    centered = x - x_ref
    variance = float(np.sum(centered ** 2))
    if variance <= 1e-20:
        raise ValueError("SET exposures do not span a usable range")
    # The intercept is refit after centering, rather than carrying an opaque
    # coefficient tied to the small SI-valued exposure scale.
    slope = float(np.sum((centered - np.mean(centered))
                         * (y - np.mean(y))) /
                  np.sum((centered - np.mean(centered)) ** 2))
    log_g_ref = float(np.mean(y - slope * centered))
    residual = y - (log_g_ref + slope * centered)
    center = float(np.median(residual))
    sigma = float(1.4826 * np.median(np.abs(residual - center)))
    return SetResetTransferLaw(
        log10_exposure_ref=x_ref,
        log10_g_ref=log_g_ref,
        slope=slope,
        residual_sigma_decade=sigma,
        n_training=int(np.count_nonzero(valid)),
    )


def leave_one_out_transfer_predictions(exposure_a_v, g_reset_initial_a):
    """Predict each paired cycle from a law fitted to all other cycles."""
    exposure = np.asarray(exposure_a_v, dtype=float)
    conductance = np.asarray(g_reset_initial_a, dtype=float)
    if (exposure.ndim != 1 or conductance.ndim != 1
            or exposure.shape != conductance.shape or len(exposure) < 4):
        raise ValueError("At least four paired exposure/conductance values are required")
    predictions = np.empty_like(exposure)
    laws = []
    for target in range(len(exposure)):
        training = np.ones(len(exposure), dtype=bool)
        training[target] = False
        law = fit_set_reset_transfer(
            exposure[training], conductance[training])
        predictions[target] = law.predict(exposure[target])
        laws.append(law)
    return predictions, laws


def fit_reset_endpoint_attractor(g_hrs_a, gate_voltage, protocol_label,
                                 reset_peak_abs_v=1.7):
    """Fit the geometric-mean HRS endpoint and its robust log scatter."""
    conductance = np.asarray(g_hrs_a, dtype=float)
    valid = np.isfinite(conductance) & (conductance > 0.0)
    if conductance.ndim != 1 or np.count_nonzero(valid) < 3:
        raise ValueError("At least three positive finite HRS endpoints are required")
    log_g = np.log10(conductance[valid])
    center = float(np.mean(log_g))
    median = float(np.median(log_g))
    sigma = float(1.4826 * np.median(np.abs(log_g - median)))
    return ResetEndpointAttractorLaw(
        gate_voltage_v=float(gate_voltage),
        protocol_label=str(protocol_label),
        log10_g_hrs_ref=center,
        residual_sigma_decade=sigma,
        n_training=int(np.count_nonzero(valid)),
        reset_peak_abs_v=float(reset_peak_abs_v),
    )


def leave_one_out_endpoint_predictions(g_hrs_a, gate_voltage,
                                       protocol_label):
    """Predict every HRS endpoint from all other endpoints in its protocol."""
    conductance = np.asarray(g_hrs_a, dtype=float)
    if (conductance.ndim != 1 or len(conductance) < 4
            or np.any(~np.isfinite(conductance))
            or np.any(conductance <= 0.0)):
        raise ValueError("At least four positive finite HRS endpoints are required")
    predictions = np.empty_like(conductance)
    laws = []
    for target in range(len(conductance)):
        training = np.ones(len(conductance), dtype=bool)
        training[target] = False
        law = fit_reset_endpoint_attractor(
            conductance[training], gate_voltage, protocol_label)
        predictions[target] = law.predict()
        laws.append(law)
    return predictions, laws


def _sheet_index(name, prefix):
    match = re.fullmatch(rf"{re.escape(prefix)}\s*(\d+)", str(name), re.I)
    return None if match is None else int(match.group(1))


def pair_cycle_sheets(set_sheet_names, reset_sheet_names):
    """Pair ``S_i`` and ``R_i`` sheets by numeric cycle index."""
    def indexed(names, prefix):
        result = {}
        for name in names:
            index = _sheet_index(name, prefix)
            if index is None:
                continue
            if index in result:
                raise ValueError(f"Duplicate {prefix} cycle index {index}")
            result[index] = str(name)
        return result

    set_index = indexed(set_sheet_names, "S")
    reset_index = indexed(reset_sheet_names, "R")
    if not set_index or not reset_index:
        raise ValueError("No S_i/R_i cycle sheets were found")
    missing_set = sorted(set(reset_index) - set(set_index))
    missing_reset = sorted(set(set_index) - set(reset_index))
    if missing_set or missing_reset:
        raise ValueError(
            f"Unpaired cycles: missing SET {missing_set}, "
            f"missing RESET {missing_reset}"
        )
    return [(index, set_index[index], reset_index[index])
            for index in sorted(set_index)]


def run_set_conditioned_cycle(set_voltage, set_current, reset_voltage,
                              gate_voltage, transfer_law, p=None,
                              mapping=None, e50_mvc=None, width_mvc=None,
                              endpoint_law=None,
                              endpoint_residual_offset_decade=0.0):
    """Run measured SET -> predicted RESET without target RESET current.

    The returned structure contains the measured SET trace for plotting and
    the simulated RESET result.  With ``endpoint_law=None`` it retains the
    legacy gate-conditioned HRS/LRS ratio.  A supplied endpoint attractor
    replaces that ratio with its fixed-gate, fixed-protocol HRS distribution.
    """
    p = v8.Params() if p is None else p
    set_v, set_i = _same_length_vectors(set_voltage, set_current)
    exposure = set_forward_exposure(set_v, set_i)
    g_lrs = transfer_law.predict(exposure)
    if endpoint_law is None:
        result = v8.run_read_initialized_reset(
            reset_voltage,
            gate_voltage,
            g_lrs,
            p=p,
            mapping=mapping,
            e50_mvc=e50_mvc,
            width_mvc=width_mvc,
        )
        endpoint_source = "predicted_from_preceding_set_and_gate"
        endpoint_capped = False
        g_hrs_uncapped = result["g_hrs_a"]
        endpoint_protocol_label = None
        endpoint_sigma_decade = None
    else:
        endpoint_law.validate_gate(gate_voltage)
        endpoint_law.validate_protocol(reset_voltage)
        g_hrs_uncapped = endpoint_law.predict(
            endpoint_residual_offset_decade)
        g_hrs = min(g_hrs_uncapped, 0.95 * g_lrs)
        endpoint_capped = bool(g_hrs < g_hrs_uncapped)
        result = v7.run_identified_reset(
            reset_voltage,
            gate_voltage,
            g_lrs,
            g_hrs,
            p=p,
            mapping=mapping,
            e50_mvc=e50_mvc,
            width_mvc=width_mvc,
        )
        endpoint_source = "protocol_conditioned_hrs_attractor"
        endpoint_protocol_label = endpoint_law.protocol_label
        endpoint_sigma_decade = endpoint_law.residual_sigma_decade
    result.update({
        "set_voltage_v": set_v,
        "set_current_a": set_i,
        "set_exposure_A_V": exposure,
        "g_reset_initial_predicted_a": g_lrs,
        "state_source": "predicted_from_preceding_set",
        "endpoint_source": endpoint_source,
        "g_hrs_uncapped_a": g_hrs_uncapped,
        "endpoint_capped_below_lrs": endpoint_capped,
        "endpoint_protocol_label": endpoint_protocol_label,
        "endpoint_sigma_decade": endpoint_sigma_decade,
        "hrs_transport_model": getattr(
            p, "reset_hrs_transport_model", "sinh"),
        "reset_pf_beta_per_sqrt_v": getattr(
            p, "reset_pf_beta_per_sqrt_v", None),
        "set_reset_transfer_sigma_decade": (
            transfer_law.residual_sigma_decade),
    })
    return result


def claim_scope():
    """Return an explicit interpretation boundary for the cycle link."""
    return {
        "paired_order": "S_i is the measured operation immediately before R_i",
        "set_input": "measured forward SET I-V trace",
        "reset_target_leakage": "target RESET current is not a model input",
        "set_exposure": (
            "integral |I| dV is a voltage-domain proxy; without sweep time "
            "it is not an absolute charge or energy"
        ),
        "physical_interpretation": (
            "SET history contains statistical information about the following "
            "RESET starting conductance; correlation does not identify a "
            "unique atomistic migration species"
        ),
        "hrs_endpoint": (
            "within a fixed gate and DC protocol, final HRS is represented by "
            "a log-domain attractor distribution rather than a fixed HRS/LRS "
            "ratio"
        ),
        "hrs_endpoint_generalization": (
            "the attractor may be used only at its fitted gate and protocol; "
            "cross-gate changes can include batch effects"
        ),
        "remaining_limit": (
            "the SET trace is experimental input, so this is a measured-SET-"
            "conditioned full cycle, not a fully autonomous SET simulator"
        ),
    }
