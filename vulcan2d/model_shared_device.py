"""Quasi-static h-BN model with a device state independent of circuit topology.

States are effective fractions of soft connected regions and strong paths.
The simulator uses voltage ramps. Its sustained-current SET rule tests the
load line before and after a candidate strong connection. It describes
quasi-static state access and has no switching-transient time scale. Archived
calibrations retain the earlier available-current rule.
"""
from dataclasses import dataclass

import numpy as np
from scipy.special import expit


@dataclass(frozen=True)
class Device:
    beta_set: float
    soft_exponent: float
    set_center_v: float
    set_width_v: float
    beta_reset: float
    reset_soft_alpha: float
    reset_center_v: float
    reset_width_v: float
    hard_set_v: float
    hard_access_a: float
    hard_conductance_s: float
    hard_reset_centers_v: tuple
    hard_reset_widths_v: tuple
    hard_reset_late_fraction: float
    set_skew: float = 1.0
    reset_off_power: float = 1.0
    reset_on_power: float = None
    reset_late_fraction: float = 0.0
    reset_center_gap_v: float = 0.0
    reset_hazard_per_v: float = None
    hard_access_mode: str = "available"
    soft_growth_current_a: float = None
    reset_power_assist_v: float = 0.0
    reset_tail_power_fraction: float = 1.0

    def __post_init__(self):
        if self.hard_access_mode not in ("available", "sustained"):
            raise ValueError("hard_access_mode must be available or sustained")
        if self.soft_growth_current_a is not None and (
                not np.isfinite(self.soft_growth_current_a) or self.soft_growth_current_a <= 0):
            raise ValueError("soft_growth_current_a must be positive")
        if not np.isfinite(self.reset_power_assist_v) or self.reset_power_assist_v < 0:
            raise ValueError("reset_power_assist_v must be finite and nonnegative")
        if not 0 <= self.reset_tail_power_fraction <= 1:
            raise ValueError("reset_tail_power_fraction must be between zero and one")


@dataclass(frozen=True)
class Preparation:
    off_set_a: float
    soft_set_a: float
    off_reset_a: float
    soft_reset_a: float
    soft_available: float = 1.0


@dataclass(frozen=True)
class State:
    soft: float = 0.0
    hard: float = 0.0


@dataclass(frozen=True)
class Circuit:
    """Series transistor magnitude curve and/or an ideal source current cap."""
    transistor: tuple = None
    compliance_a: float = None

    def transistor_current(self, voltage):
        grid, current = self.transistor
        return np.interp(voltage, grid, current)

    def supply_bound(self, voltage):
        if self.transistor is None:
            available = np.full_like(voltage, np.inf)
        else:
            available = self.transistor_current(voltage)
        return (available if self.compliance_a is None
                else np.minimum(available, self.compliance_a))

    def operating_point(self, applied, current_law, iterations=27):
        """Find the load-line intersection for a monotone current law."""
        if self.transistor is None and self.compliance_a is None:
            return current_law(applied), applied
        low = np.zeros_like(applied)
        high = applied.copy()
        free = self.transistor is None and self.compliance_a is not None
        if free:
            clamped = current_law(applied) > self.compliance_a
            high = np.where(clamped, applied, 0.0)
        for _ in range(iterations):
            v = 0.5 * (low + high)
            available = self.supply_bound(applied - v)
            below = current_law(v) < available
            low = np.where(below, v, low)
            high = np.where(below, high, v)
        v = 0.5 * (low + high)
        if free:
            v = np.where(clamped, v, applied)
        current = current_law(v)
        if self.compliance_a is not None:
            current = np.minimum(current, self.compliance_a)
        return current, v


def device_current(v, soft, hard, device, preparation, phase):
    if phase == "SET":
        off = preparation.off_set_a * v * np.exp(np.clip(device.beta_set * np.sqrt(v), 0, 40))
        on = preparation.soft_set_a * v ** device.soft_exponent
    else:
        off = (preparation.off_reset_a * v ** device.reset_off_power
               * np.exp(np.clip(device.beta_reset * np.sqrt(v), 0, 40)))
        if device.reset_on_power is None:
            on = preparation.soft_reset_a * np.sinh(np.clip(device.reset_soft_alpha * v, 0, 40))
        else:
            on = preparation.soft_reset_a * v ** device.reset_on_power
    connected = (1.0 - soft) * off + soft * on
    return (1.0 - hard) * connected + hard * device.hard_conductance_s * v


def _soft_formation(v, initial, device, preparation):
    baseline = expit(-device.set_center_v / device.set_width_v) ** device.set_skew
    activated = expit((v - device.set_center_v) / device.set_width_v) ** device.set_skew
    progress = (activated - baseline) / (1 - baseline)
    return np.maximum(initial, preparation.soft_available * progress)


def _hard_survival(v, device):
    weights = device.hard_reset_late_fraction
    centers = device.hard_reset_centers_v
    widths = device.hard_reset_widths_v
    weakening = expit((centers[0] - v) / widths[0]) / expit(centers[0] / widths[0])
    disconnect = expit((centers[1] - v) / widths[1]) / expit(centers[1] / widths[1])
    return (weights + (1 - weights) * weakening) * disconnect


def _soft_survival(stress, initial, device, tail_stress=None):
    center, width = device.reset_center_v, device.reset_width_v
    first = initial * expit((center - stress) / width) / expit(center / width)
    if device.reset_late_fraction == 0.0:
        return first
    tail_stress = stress if tail_stress is None else tail_stress
    late_center = center + device.reset_center_gap_v
    second = initial * expit((late_center - tail_stress) / width) / expit(late_center / width)
    return (1-device.reset_late_fraction)*first + device.reset_late_fraction*second


def _reset_feedback(voltage, initial, device, preparation, circuit, hard):
    """Integrate RESET hazard against ramp voltage using the changing load line.

    The trap is evaluated on the voltage grid as a causal Volterra equation.
    No physical time constant is implied without a known sweep rate.
    """
    rate = device.reset_hazard_per_v
    if not np.isfinite(rate) or rate <= 0 or device.reset_width_v <= 0:
        raise ValueError("RESET feedback needs a positive rate and width")
    dv = np.diff(np.r_[0., voltage])
    center, width = device.reset_center_v, device.reset_width_v
    soft = np.full_like(voltage, initial.soft)
    for _ in range(60):
        current, local = circuit.operating_point(voltage, lambda vh: device_current(
            vh, soft, hard, device, preparation, "RESET"))
        hazard = rate * expit((local - center) / width)
        integral = np.cumsum(.5 * (hazard + np.r_[rate*expit(-center/width), hazard[:-1]]) * dv)
        updated = initial.soft * np.exp(-integral)
        if np.max(np.abs(updated - soft)) <= 2e-9:
            return current, local, soft
        soft = updated
    raise RuntimeError("RESET feedback did not converge on the supplied voltage grid")


def reset_outward(v, initial, device, preparation, circuit):
    """Solve the outgoing RESET limb for one trace or a batch of prepared states."""
    # The prepared-state load line provides a stress reference. Power is a
    # quasi-static proxy here; there is no inferred temperature or time axis.
    reference_current, stress = circuit.operating_point(v, lambda vh: device_current(
        vh, initial.soft, initial.hard, device, preparation, "RESET"))
    soft_stress = stress
    tail_stress = None
    if device.reset_power_assist_v:
        assist = device.reset_power_assist_v * np.log1p(reference_current * stress / 5e-5)
        soft_stress = stress + assist
        tail_stress = stress + device.reset_tail_power_fraction * assist
    soft = np.minimum.accumulate(_soft_survival(soft_stress, initial.soft, device, tail_stress), axis=-1)
    hard = np.minimum.accumulate(initial.hard * _hard_survival(stress, device), axis=-1)
    if device.reset_hazard_per_v is not None and initial.soft > 0.0:
        current, local, soft = _reset_feedback(v, initial, device, preparation, circuit, hard)
    else:
        current, local = circuit.operating_point(v, lambda vh: device_current(
            vh, soft, hard, device, preparation, "RESET"))
    return current, local, soft, hard


def solve_soft_set_outward(v, initial, device, preparation, circuit):
    """Operating point before strong path formation, including batched fits."""
    def current(vh):
        soft = _soft_formation(vh, initial.soft, device, preparation)
        return device_current(vh, soft, initial.hard, device, preparation, "SET")

    measured, vh = circuit.operating_point(v, current)
    soft = _soft_formation(vh, initial.soft, device, preparation)
    if device.soft_growth_current_a is not None:
        # The field-only load line provides the candidate programming current.
        # Limited current reduces the newly retained fraction. Recompute the
        # load line for that actual retained state before returning its current.
        factor = measured / (measured + device.soft_growth_current_a)
        soft = np.maximum.accumulate(initial.soft + (soft - initial.soft) * factor, axis=-1)
        measured, vh = circuit.operating_point(v, lambda local: device_current(
            local, soft, initial.hard, device, preparation, "SET"))
    return measured, vh, soft


def strong_operating_point(applied, device, circuit):
    """Load-line current and device voltage with the strong state fully on."""
    return circuit.operating_point(applied, lambda vh: device.hard_conductance_s * vh)


def _strong_formation(applied, pre_transition_voltage, device, circuit, initial):
    eligible = pre_transition_voltage >= device.hard_set_v
    if device.hard_access_mode == "sustained":
        # Test the current the candidate strong path can actually draw while
        # leaving its own voltage drop in the load line.
        available = np.zeros_like(applied)
        if np.any(eligible):
            available[eligible] = strong_operating_point(applied[eligible], device, circuit)[0]
    else:
        available = circuit.supply_bound(applied)
    formed = eligible & (available >= device.hard_access_a)
    return np.maximum.accumulate(np.where(formed, 1.0, initial.hard))


def simulate_sweep(voltage, phase, device, preparation, circuit, initial=State()):
    """A triangular sweep, followed by a return at the stable apex state."""
    if phase not in ("SET", "RESET"):
        raise ValueError("phase must be SET or RESET")
    voltage = np.asarray(voltage, dtype=float)
    if (voltage.ndim != 1 or len(voltage) < 3 or not np.isfinite(voltage).all()
            or (np.any(voltage < 0) if phase == "SET" else np.any(voltage > 0))):
        raise ValueError("Expected a finite single-polarity waveform")
    v = np.abs(voltage)
    apex = int(np.argmax(v))
    if np.any(np.diff(v[:apex + 1]) < -1e-10) or np.any(np.diff(v[apex:]) > 1e-10):
        raise ValueError("Expected an outward sweep followed by its return")
    if circuit.transistor is not None and v[apex] > circuit.transistor[0][-1] + 1e-8:
        raise ValueError("Waveform exceeds the measured transistor voltage range")
    if phase == "SET":
        outward_current, outward_vh, soft = solve_soft_set_outward(
            v[:apex + 1], initial, device, preparation, circuit)
        hard = _strong_formation(v[:apex + 1], outward_vh, device, circuit, initial)
        just_formed = hard > initial.hard
        if np.any(just_formed):
            outward_current[just_formed], outward_vh[just_formed] = strong_operating_point(
                v[:apex + 1][just_formed], device, circuit)
            if device.hard_access_mode == "sustained":
                first = np.flatnonzero(just_formed)[0]
                retained = soft[first]
                candidate = _soft_formation(outward_vh[just_formed], retained, device, preparation)
                if device.soft_growth_current_a is not None:
                    current = outward_current[just_formed]
                    candidate = retained + (candidate - retained) * (
                        current / (current + device.soft_growth_current_a))
                soft[just_formed] = np.maximum.accumulate(candidate)
    else:
        outward_current, outward_vh, soft, hard = reset_outward(
            v[:apex + 1], initial, device, preparation, circuit)

    final = State(float(soft[-1]), float(hard[-1]))
    return_current, return_vh = circuit.operating_point(v[apex + 1:], lambda vh: device_current(
        vh, final.soft, final.hard, device, preparation, phase))
    return {
        "voltage_V": voltage.copy(),
        "current_A": np.sign(voltage) * np.r_[outward_current, return_current],
        "device_voltage_V": np.sign(voltage) * np.r_[outward_vh, return_vh],
        "soft_state": np.r_[soft, np.full(len(return_vh), final.soft)],
        "hard_state": np.r_[hard, np.full(len(return_vh), final.hard)],
        "final": final,
    }


def simulate_cycle(set_voltage, reset_voltage, device, preparation, set_circuit,
                   reset_circuit, initial=State()):
    programmed = simulate_sweep(set_voltage, "SET", device, preparation, set_circuit, initial)
    erased = simulate_sweep(reset_voltage, "RESET", device, preparation, reset_circuit, programmed["final"])
    return programmed, erased
