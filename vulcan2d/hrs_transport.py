"""Protocol-level HRS transport identification for paired RESET cycles.

The measured reverse-transistor lookup first reconstructs the voltage across
h-BN.  The stable post-RESET branch is then fitted to the fixed-thickness,
effective Poole--Frenkel-like basis

``I_HRS = A_HRS * V_hBN * exp(beta_PF * sqrt(V_hBN))``.

Every cycle retains its own amplitude ``A_HRS`` while the curvature ``beta_PF``
is shared by one gate/protocol group.  Consequently, cycle-to-cycle endpoint
variation cannot masquerade as a different transport curvature.

This voltage-domain fit is a compact transport law, not a microscopic
identification.  Temperature-dependent measurements would be required to
extract a trap energy or dielectric constant and to distinguish Poole--Frenkel
emission from other field-assisted mechanisms.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .reset_identification import _branch_mask


@dataclass(frozen=True)
class EffectivePooleFrenkelLaw:
    """Shared HRS curvature for one fixed gate and RESET protocol."""

    beta_per_sqrt_v: float
    residual_mad_decade: float
    gate_voltage_v: float
    protocol_label: str
    n_training_cycles: int
    reset_peak_abs_v: float = 1.7
    reset_peak_tolerance_v: float = 0.03

    def validate_gate(self, gate_voltage):
        gate = float(gate_voltage)
        if not np.isfinite(gate) or not np.isclose(
                gate, self.gate_voltage_v, rtol=0.0, atol=1e-9):
            raise ValueError("HRS transport gate voltage does not match fit")

    def validate_protocol(self, reset_voltage):
        voltage = np.asarray(reset_voltage, dtype=float)
        finite = voltage[np.isfinite(voltage)]
        if finite.ndim != 1 or len(finite) < 3:
            raise ValueError("RESET protocol must contain finite voltage points")
        peak = float(np.max(np.abs(finite)))
        if (not np.isclose(peak, self.reset_peak_abs_v, rtol=0.0,
                           atol=self.reset_peak_tolerance_v)
                or np.min(finite) >= 0.0):
            raise ValueError("RESET peak does not match HRS transport fit")


def effective_pf_basis(voltage_hbn_v, beta_per_sqrt_v):
    """Return ``V*exp(beta*sqrt(V))`` for non-negative h-BN voltage."""
    voltage = np.asarray(voltage_hbn_v, dtype=float)
    if np.any(~np.isfinite(voltage)) or np.any(voltage < 0.0):
        raise ValueError("voltage_hbn_v must be finite and non-negative")
    beta = float(beta_per_sqrt_v)
    if not np.isfinite(beta) or beta < 0.0:
        raise ValueError("beta_per_sqrt_v must be finite and non-negative")
    value = voltage * np.exp(np.clip(beta * np.sqrt(voltage), 0.0, 39.0))
    return float(value) if value.ndim == 0 else value


def estimate_hrs_amplitude(cycle, law_or_beta, min_points=5):
    """Estimate one cycle's HRS amplitude from its stable return branch."""
    beta = (law_or_beta.beta_per_sqrt_v
            if isinstance(law_or_beta, EffectivePooleFrenkelLaw)
            else float(law_or_beta))
    mask = _branch_mask(cycle, "hrs")
    if np.count_nonzero(mask) < int(min_points):
        raise ValueError("Insufficient stable HRS points")
    basis = effective_pf_basis(cycle.voltage_hbn_v[mask], beta)
    valid = (
        np.isfinite(cycle.current_a[mask])
        & np.isfinite(basis)
        & (cycle.current_a[mask] > 0.0)
        & (basis > 0.0)
    )
    if np.count_nonzero(valid) < int(min_points):
        raise ValueError("Stable HRS branch contains too few valid points")
    return float(10.0 ** np.median(
        np.log10(cycle.current_a[mask][valid])
        - np.log10(basis[valid])
    ))


def _shape_residuals(cycles, beta):
    residuals = []
    used = 0
    for cycle in cycles:
        mask = _branch_mask(cycle, "hrs")
        if np.count_nonzero(mask) < 5:
            continue
        basis = effective_pf_basis(cycle.voltage_hbn_v[mask], beta)
        valid = (
            np.isfinite(cycle.current_a[mask])
            & np.isfinite(basis)
            & (cycle.current_a[mask] > 0.0)
            & (basis > 0.0)
        )
        if np.count_nonzero(valid) < 5:
            continue
        log_amplitude = (
            np.log10(cycle.current_a[mask][valid])
            - np.log10(basis[valid])
        )
        residuals.extend(log_amplitude - np.median(log_amplitude))
        used += 1
    return np.asarray(residuals, dtype=float), used


def fit_effective_pf_law(cycles, gate_voltage, protocol_label,
                         beta_bounds=(0.0, 20.0), reset_peak_abs_v=1.7):
    """Fit shared HRS curvature while profiling out cycle amplitudes."""
    from scipy.optimize import minimize_scalar

    cycles = list(cycles)
    if len(cycles) < 3:
        raise ValueError("At least three reconstructed RESET cycles are required")

    def objective(beta):
        residual, _ = _shape_residuals(cycles, beta)
        if len(residual) == 0:
            return np.inf
        return float(np.median(np.abs(residual)))

    fit = minimize_scalar(
        objective,
        bounds=tuple(map(float, beta_bounds)),
        method="bounded",
        options={"xatol": 1e-7},
    )
    residual, used = _shape_residuals(cycles, fit.x)
    if (not fit.success or not np.isfinite(fit.fun)
            or used < 3 or len(residual) == 0):
        raise RuntimeError("Could not identify effective HRS transport")
    return EffectivePooleFrenkelLaw(
        beta_per_sqrt_v=float(fit.x),
        residual_mad_decade=float(np.median(np.abs(residual))),
        gate_voltage_v=float(gate_voltage),
        protocol_label=str(protocol_label),
        n_training_cycles=int(used),
        reset_peak_abs_v=float(reset_peak_abs_v),
    )


def leave_one_out_pf_laws(cycles, gate_voltage, protocol_label):
    """Fit one target-excluding HRS transport law per RESET cycle."""
    cycles = list(cycles)
    if len(cycles) < 4:
        raise ValueError("At least four RESET cycles are required")
    laws = []
    for target in range(len(cycles)):
        training = [cycle for index, cycle in enumerate(cycles)
                    if index != target]
        laws.append(fit_effective_pf_law(
            training, gate_voltage, protocol_label))
    return laws


def claim_scope():
    """Return the scientific interpretation boundary of the HRS law."""
    return {
        "equation": "I_HRS=A_HRS*V_hBN*exp(beta_PF*sqrt(V_hBN))",
        "fit_level": (
            "beta_PF is shared within one gate/protocol group; A_HRS varies "
            "between cycles"
        ),
        "interpretation": (
            "effective fixed-thickness field-assisted HRS curvature"
        ),
        "not_identified": (
            "no trap energy, dielectric constant or unique microscopic "
            "transport mechanism is inferred without temperature data"
        ),
    }
