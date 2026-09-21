"""Normalized, passive transport candidates for mechanism discrimination.

These are research candidates, not a replacement of the autonomous M13 model.
Temperature/thickness conversions are conditional, not fitted measurements.
The contact-emission branch assumes the local barrier sees the de-embedded
voltage; it is not a microscopic solution of both asymmetric contacts.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize_scalar

Q = 1.602176634e-19
KB = 1.380649e-23
EPS0 = 8.8541878188e-12
LN10 = float(np.log(10.0))
NAMES = ("ohmic", "power", "pf_free", "pf_constrained",
         "schottky_free", "schottky_constrained", "hopping")


@dataclass(frozen=True)
class TransportAssumptions:
    temperature_k: float = 300.0
    thickness_nm: float = 6.0
    reference_voltage_v: float = 0.1
    epsilon_min: float = 2.0
    epsilon_max: float = 8.0

    def __post_init__(self):
        values = (self.temperature_k, self.thickness_nm, self.reference_voltage_v,
                  self.epsilon_min, self.epsilon_max)
        if not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError("All transport assumptions must be finite and positive")
        if self.epsilon_max < self.epsilon_min:
            raise ValueError("epsilon_max must be >= epsilon_min")

    @property
    def thermal_voltage_v(self):
        return KB * self.temperature_k / Q


def pf_beta(epsilon_r, assumptions=TransportAssumptions()):
    epsilon = float(epsilon_r)
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon_r must be positive")
    return float(np.sqrt(Q ** 3 / (np.pi * EPS0 * epsilon
                         * assumptions.thickness_nm * 1e-9))
                 / (KB * assumptions.temperature_k))


def parameter_bounds(name, assumptions=TransportAssumptions()):
    if name == "ohmic":
        return None
    if name == "power":
        return (0.2, 8.0)
    if name in ("pf_free", "schottky_free"):
        return (0.0, 40.0)
    if name in ("pf_constrained", "schottky_constrained"):
        scale = 1.0 if name.startswith("pf") else 0.5
        return tuple(scale * pf_beta(e, assumptions)
                     for e in (assumptions.epsilon_max, assumptions.epsilon_min))
    if name == "hopping":
        # q*a/(2*d*kBT), with conditional a <= d. Small alpha tends to Ohmic.
        return (1e-4, 1.0 / (2.0 * assumptions.thermal_voltage_v))
    raise ValueError(f"Unknown transport candidate: {name}")


def _log_sinh_positive(z):
    return z + np.log(-np.expm1(-2.0 * z)) - np.log(2.0)


def log10_shape(voltage_magnitude, name, parameter=None,
                assumptions=TransportAssumptions()):
    """Log of normalized positive-voltage shape, with value zero at v_ref."""
    v = np.asarray(voltage_magnitude, dtype=float)
    if np.any(~np.isfinite(v)) or np.any(v <= 0):
        raise ValueError("log10_shape requires strictly positive finite voltage")
    bounds = parameter_bounds(name, assumptions)
    if bounds is not None:
        if parameter is None or not np.isfinite(parameter):
            raise ValueError("Finite shape parameter required")
        if not bounds[0] - 1e-12 <= parameter <= bounds[1] + 1e-12:
            raise ValueError("Shape parameter outside declared candidate domain")
    vr = assumptions.reference_voltage_v
    if name == "ohmic":
        log_value = np.log(v / vr)
    elif name == "power":
        log_value = parameter * np.log(v / vr)
    elif name.startswith("pf"):
        log_value = np.log(v / vr) + parameter * (np.sqrt(v) - np.sqrt(vr))
    elif name.startswith("schottky"):
        vt = assumptions.thermal_voltage_v
        log_value = (parameter * (np.sqrt(v) - np.sqrt(vr))
                     + np.log(-np.expm1(-v / vt)) - np.log(-np.expm1(-vr / vt)))
    elif name == "hopping":
        log_value = (_log_sinh_positive(parameter * v)
                     - _log_sinh_positive(parameter * vr))
    return log_value / LN10


def current(voltage, log10_reference_current_a, name, parameter=None,
            assumptions=TransportAssumptions()):
    """Odd, passive branch extension; zero current at exactly zero voltage.

    Oddness is an extension of a single branch, not an assertion that the
    actual positive/negative device has equal parameters or state.
    """
    v = np.asarray(voltage, dtype=float)
    if np.any(~np.isfinite(v)) or not np.isfinite(log10_reference_current_a):
        raise ValueError("Finite voltage and log amplitude required")
    out = np.zeros_like(v)
    nonzero = v != 0
    logs = log10_reference_current_a + log10_shape(
        np.abs(v[nonzero]), name, parameter, assumptions)
    if np.any(logs > np.log10(np.finfo(float).max)):
        raise OverflowError("Candidate current overflows; no silent clipping")
    out[nonzero] = np.sign(v[nonzero]) * 10.0 ** logs
    return float(out) if out.ndim == 0 else out


@dataclass(frozen=True)
class Curve:
    cycle_id: int
    voltage_v: np.ndarray
    log10_current_a: np.ndarray

    def __post_init__(self):
        v, y = np.asarray(self.voltage_v), np.asarray(self.log10_current_a)
        if v.ndim != 1 or y.shape != v.shape or len(v) < 3:
            raise ValueError("A curve needs at least three paired 1D points")
        if np.any(~np.isfinite(v)) or np.any(~np.isfinite(y)) or np.any(v <= 0):
            raise ValueError("Invalid transport curve")


def profile_amplitude(curve, name, parameter, assumptions=TransportAssumptions()):
    return float(np.mean(curve.log10_current_a
                         - log10_shape(curve.voltage_v, name, parameter, assumptions)))


def fit_shared_shape(curves, name, assumptions=TransportAssumptions()):
    """Equal-cycle squared log error, with one profiled amplitude per curve."""
    curves = tuple(curves)
    if len(curves) < 2:
        raise ValueError("At least two training curves required")
    if len({c.cycle_id for c in curves}) != len(curves):
        raise ValueError("Duplicate cycle IDs in transport training")
    bounds = parameter_bounds(name, assumptions)

    def loss(parameter):
        values = []
        for c in curves:
            residual = c.log10_current_a - log10_shape(c.voltage_v, name, parameter, assumptions)
            values.append(float(np.mean((residual - np.mean(residual)) ** 2)))
        return float(np.mean(values))

    if bounds is None:
        parameter = None
    elif bounds[0] == bounds[1]:
        parameter = bounds[0]
    elif name == "hopping":
        opt = minimize_scalar(loss, bounds=bounds, method="bounded",
                              options={"xatol": 1e-8})
        if not opt.success or not np.isfinite(opt.fun):
            raise RuntimeError("Could not optimize hopping shape")
        # Include boundaries explicitly; a lower-bound optimum means Ohmic.
        parameter = min((float(opt.x), *bounds), key=loss)
    else:
        # All remaining families are affine in the shape parameter in log I.
        # Profile amplitudes analytically, preventing large currents/cycles
        # from taking larger weights than other cycles.
        xxs, xys = [], []
        p0, p1 = bounds
        for c in curves:
            b0 = log10_shape(c.voltage_v, name, p0, assumptions)
            x = (log10_shape(c.voltage_v, name, p1, assumptions) - b0) / (p1 - p0)
            y = c.log10_current_a - b0
            x, y = x - np.mean(x), y - np.mean(y)
            xxs.append(float(np.mean(x * x)))
            xys.append(float(np.mean(x * y)))
        if np.mean(xxs) <= 1e-20:
            raise ValueError("No voltage spread to identify shared shape")
        parameter = float(np.clip(p0 + np.mean(xys) / np.mean(xxs), *bounds))

    amplitudes = [profile_amplitude(c, name, parameter, assumptions) for c in curves]
    at_boundary = (bounds is not None and any(
        np.isclose(parameter, b, atol=1e-6, rtol=0) for b in bounds))
    return {"name": name, "parameter": parameter, "parameter_bounds": bounds,
            "parameter_at_boundary": at_boundary,
            "n_training_cycles": len(curves),
            "training_cycle_ids": [c.cycle_id for c in curves],
            "log10_reference_amplitude_center": float(np.median(amplitudes)),
            "log10_reference_amplitudes": dict(zip([c.cycle_id for c in curves], amplitudes)),
            "profiled_equal_cycle_rmse_decade": float(np.sqrt(loss(parameter))),
            "conditional_physics": conditional_physics(name, parameter, assumptions)}


def conditional_physics(name, parameter, assumptions=TransportAssumptions()):
    result = {"assumed_temperature_K": assumptions.temperature_k,
              "assumed_thickness_nm": assumptions.thickness_nm,
              "interpretation": "Conditional correspondence, not microscopic identification"}
    if name.startswith(("pf", "schottky")):
        scale = 1.0 if name.startswith("pf") else 0.5
        result["apparent_epsilon_r"] = (None if parameter <= 0
            else (scale * pf_beta(1.0, assumptions) / parameter) ** 2)
        result["coefficient_unit"] = "V^-1/2"
    elif name == "hopping":
        result["effective_hop_length_nm"] = (
            2 * assumptions.thickness_nm * assumptions.thermal_voltage_v * parameter)
        result["coefficient_unit"] = "V^-1"
    elif name == "power":
        result["interpretation"] = "Effective power exponent; does not identify SCLC"
    return result
