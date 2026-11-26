"""Uncalibrated common h-BN reference kernel for 1R and 1T1R experiments.

One additive transport law and one bounded connectivity update are used in
every circuit. This is a reduced-order candidate, not a jointly fitted model,
an atomic mechanism, or a replacement for the archived v0.13 calibration.
Rates use command-voltage path length as their clock (per volt, not per second).
Their transfer between protocols is an explicit approximation requiring data.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .circuit import solve_series_load_line


@dataclass(frozen=True)
class SharedHbnLaw:
    """Material parameters, with no circuit label or gate-dependent adjustment.

    I(V,z) = I0*sinh(alpha*V) + G_on*V*sum(w_k*z_k).
    The connectivity z_k lies in [0,1]. Under positive internal bias it grows
    toward one; under negative bias it decays toward zero. The frozen-bias
    update is an exact bounded exponential. Thresholds are effective internal
    voltages at the device thickness, not activation energies or atom counts.
    All parameters are mandatory: this module contains no experimental fit.
    """

    leakage_scale_a: float
    leakage_alpha_per_v: float
    on_conductance_s: float
    weights: tuple[float, ...]
    set_thresholds_v: tuple[float, ...]
    reset_thresholds_v: tuple[float, ...]
    set_rate_per_v: float
    reset_rate_per_v: float
    acceleration_per_v: float

    def __post_init__(self):
        for name in ("weights", "set_thresholds_v", "reset_thresholds_v"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be a nonempty finite vector")
            object.__setattr__(self, name, tuple(float(v) for v in values))
        if not len(self.weights) == len(self.set_thresholds_v) == len(self.reset_thresholds_v):
            raise ValueError("Weights and threshold vectors must have matching lengths")
        if (np.any(np.asarray(self.weights) < 0.)
                or not np.isclose(sum(self.weights), 1., rtol=0., atol=1e-12)):
            raise ValueError("Weights must be nonnegative and sum to one")
        if min(*self.set_thresholds_v, *self.reset_thresholds_v) < 0.:
            raise ValueError("Internal switching thresholds must be nonnegative")
        for name in ("leakage_scale_a", "leakage_alpha_per_v", "on_conductance_s",
                     "set_rate_per_v", "reset_rate_per_v", "acceleration_per_v"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        if self.on_conductance_s <= 0.:
            raise ValueError("on_conductance_s must be positive")

    def validate_state(self, state):
        values = np.asarray(state, dtype=float)
        if (values.shape != (len(self.weights),) or not np.all(np.isfinite(values))
                or np.any(values < 0.) or np.any(values > 1.)):
            raise ValueError("State must contain one finite connectivity in [0,1] per weight")
        return values.copy()

    def current(self, voltage_hbn_v, state):
        """Signed intrinsic current; the external circuit does not enter this law."""
        voltage = float(voltage_hbn_v)
        if not np.isfinite(voltage):
            raise ValueError("Internal h-BN voltage must be finite")
        z = self.validate_state(state)
        with np.errstate(over="raise", invalid="raise"):
            result = (self.leakage_scale_a * np.sinh(self.leakage_alpha_per_v * voltage)
                      + self.on_conductance_s * float(np.dot(self.weights, z)) * voltage)
        if not np.isfinite(result):
            raise ValueError("Material law overflow outside its numerical voltage domain")
        return float(result)

    def advance(self, state, voltage_hbn_v, command_path_increment_v):
        """Frozen-bias evolution per command-path volt, with no physical seconds.

        The common rate family is inherited from the bounded field-threshold
        reduced-order model. It is a hypothesis for both regimes, not evidence
        that hard and soft transport share a unique microscopic mechanism.
        """
        z = self.validate_state(state)
        voltage, distance = float(voltage_hbn_v), float(command_path_increment_v)
        if not np.all(np.isfinite([voltage, distance])) or distance < 0.:
            raise ValueError("Internal voltage and nonnegative command path must be finite")
        if distance == 0. or voltage == 0.:
            return z
        if voltage > 0.:
            thresholds, coefficient, equilibrium = self.set_thresholds_v, self.set_rate_per_v, 1.
        else:
            thresholds, coefficient, equilibrium = self.reset_thresholds_v, self.reset_rate_per_v, 0.
        overdrive = np.maximum(abs(voltage) - np.asarray(thresholds), 0.)
        with np.errstate(over="raise", invalid="raise"):
            rate = coefficient * np.sinh(self.acceleration_per_v * overdrive)
            switched_fraction = -np.expm1(-distance * rate)
        # This form preserves an exactly frozen state when rate is zero and
        # avoids cancellation for very small rate*distance.
        result = z + (equilibrium - z) * switched_fraction
        return np.clip(result, 0., 1.)


@dataclass(frozen=True)
class OperatingPoint:
    command_voltage_v: float
    supply_voltage_v: float
    hbn_voltage_v: float
    transistor_voltage_v: float
    current_a: float
    compliance_drop_v: float
    compliance_limited: bool


def _check_compliance(value):
    if value is not None and (not np.isfinite(float(value)) or float(value) <= 0.):
        raise ValueError("Compliance must be positive and finite, or None")


def _voltage_at_current(material, state, target_a, upper_v):
    """Invert the monotone intrinsic law in a declared magnitude bracket."""
    target, upper = float(target_a), float(upper_v)
    if target == 0.:
        return 0.
    if target < 0. or material.current(upper, state) < target:
        raise ValueError("Current is outside the intrinsic voltage bracket")
    lo, hi = 0., upper
    for _ in range(64):
        mid = .5 * (lo + hi)
        if material.current(mid, state) < target:
            lo = mid
        else:
            hi = mid
    return .5 * (lo + hi)


@dataclass(frozen=True)
class BareCircuit:
    """Ideal voltage source with optional instrument compliance for bare 1R.

    A current limit reduces actual supplied voltage. It is never implemented
    as clipping the reported current while keeping the full h-BN field.
    The command-minus-supply difference is source regulation, not device heat.
    """

    compliance_a: float | None = None

    def __post_init__(self):
        _check_compliance(self.compliance_a)

    def solve(self, command_voltage_v, material, state):
        command = float(command_voltage_v)
        current = material.current(command, state)
        voltage, limited = command, False
        if self.compliance_a is not None and abs(current) > self.compliance_a:
            voltage = np.copysign(_voltage_at_current(
                material, state, self.compliance_a, abs(command)), command)
            current, limited = material.current(voltage, state), True
        return OperatingPoint(command, voltage, voltage, 0., current,
                              command - voltage, limited)


@dataclass(frozen=True)
class ZeroDropTransistor:
    """Ideal-short circuit adapter used only to test the exact 1T1R→1R limit."""


@dataclass(frozen=True)
class TransistorCircuit:
    """The same material in series with a fixed-source transistor port law."""

    mapping: object
    gate_voltage: float
    compliance_a: float | None = None

    def __post_init__(self):
        if not np.isfinite(float(self.gate_voltage)):
            raise ValueError("Gate voltage must be finite")
        _check_compliance(self.compliance_a)

    def solve(self, command_voltage_v, material, state):
        command = float(command_voltage_v)
        if isinstance(self.mapping, ZeroDropTransistor):
            return BareCircuit(self.compliance_a).solve(command, material, state)
        current, vh = solve_series_load_line(
            command, self.gate_voltage, lambda v: material.current(v, state),
            self.mapping, nbis=56)
        vt, supply, limited = command - vh, command, False
        if self.compliance_a is not None and abs(current) > self.compliance_a:
            if not hasattr(self.mapping, "voltage_for_current"):
                raise TypeError("A current-limited transistor adapter requires voltage_for_current")
            side = "reset" if np.signbit(command) else "set"
            vh = np.copysign(_voltage_at_current(
                material, state, self.compliance_a, abs(command)), command)
            vt = np.copysign(self.mapping.voltage_for_current(
                self.compliance_a, self.gate_voltage, side=side), command)
            supply = vh + vt
            if not np.isfinite(supply) or abs(supply) > abs(command) + 1e-10:
                raise ValueError("Compliance solution is outside the commanded voltage bracket")
            current, limited = material.current(vh, state), True
        return OperatingPoint(command, supply, vh, vt, current,
                              command - supply, limited)


def _advance_interval(state, start, stop, law, circuit, max_step_v,
                      state_atol, state_rtol):
    """Bounded adaptive exponential Euler with accepted two-half-step updates.

    The first half-step is retained even if it quenches the later rate. A
    midpoint predictor used only to evaluate a *replacement* full update can
    instead erase real growth under strong negative load feedback. Step
    doubling controls local error, and a maximum trial state change prevents
    two grossly saturated updates from looking spuriously converged.
    """
    direction = 1. if stop > start else -1.
    position = float(start)
    step = min(max_step_v, abs(stop - start))
    accepted = rejected = attempts = 0
    while direction * (stop - position) > 0.:
        attempts += 1
        if attempts > 200000:
            raise RuntimeError("Adaptive state integration did not converge within its step budget")
        remaining = abs(stop - position)
        step = min(step, remaining, max_step_v)
        if position + direction * step == position:
            raise RuntimeError("Adaptive state step reached floating-point resolution")
        point = circuit.solve(position, law, state)
        full = law.advance(state, point.hbn_voltage_v, step)
        if np.max(np.abs(full - state)) > .05:
            rejected += 1
            step *= .5
            continue
        half = law.advance(state, point.hbn_voltage_v, .5 * step)
        midpoint = circuit.solve(position + direction * .5 * step, law, half)
        twice = law.advance(half, midpoint.hbn_voltage_v, .5 * step)
        scale = state_atol + state_rtol * np.maximum(np.abs(state), np.abs(twice))
        error = float(np.max(np.abs(twice - full) / scale))
        if np.array_equal(twice, state):
            # Detect a threshold crossed only near this step's right endpoint;
            # otherwise both zero left rates could hide the onset completely.
            endpoint = circuit.solve(position + direction * step, law, state)
            end_trial = law.advance(state, endpoint.hbn_voltage_v, step)
            error = max(error, float(np.max(np.abs(end_trial - state) / scale)))
        if error > 1.:
            rejected += 1
            step *= max(.1, min(.8, .9 / np.sqrt(error)))
            continue
        state = twice
        position = stop if step == remaining else position + direction * step
        accepted += 1
        step *= 2. if error == 0. else min(2., max(.5, .9 / np.sqrt(error)))
    return state, accepted, rejected


def run_waveform(command_voltage, initial_state, law, circuit, *, max_step_v=.02,
                 initial_command_voltage_v=0., state_atol=1e-9, state_rtol=1e-6):
    """Integrate continuous state with bounded adaptive command-path stepping.

    Only voltage sign selects SET/RESET kinetics. No scan segment reinitializes
    the state or substitutes a fitted endpoint. Repeated commands have zero
    path increment: this model has no calibrated dwell, relaxation or pulse
    time. To continue another segment, pass final_state and the previous last
    command explicitly. Outputs are resolved at the stored post-update state.
    """
    if not isinstance(law, SharedHbnLaw):
        raise TypeError("law must be a SharedHbnLaw")
    commands = np.asarray(command_voltage, dtype=float)
    step, previous = float(max_step_v), float(initial_command_voltage_v)
    if (commands.ndim != 1 or len(commands) == 0 or not np.all(np.isfinite(commands))
            or not np.isfinite(previous)):
        raise ValueError("Command waveform and initial command must be finite")
    if not np.isfinite(step) or step <= 0.:
        raise ValueError("max_step_v must be positive and finite")
    if (not np.all(np.isfinite([state_atol, state_rtol]))
            or state_atol <= 0. or state_rtol < 0.):
        raise ValueError("State error tolerances must be finite with atol>0 and rtol>=0")
    state = law.validate_state(initial_state)
    history, points = [], []
    n_substeps = n_rejected = 0
    for target in commands:
        edges = (0., float(target)) if previous * target < 0. else (float(target),)
        for edge in edges:
            distance = abs(edge - previous)
            if distance > 0.:
                state, accepted, rejected = _advance_interval(
                    state, previous, edge, law, circuit, step, state_atol, state_rtol)
                n_substeps += accepted
                n_rejected += rejected
            previous = edge
        points.append(circuit.solve(target, law, state))
        history.append(state.copy())
    fields = ("command_voltage_v", "supply_voltage_v", "hbn_voltage_v",
              "transistor_voltage_v", "current_a", "compliance_drop_v", "compliance_limited")
    result = {name: np.asarray([getattr(p, name) for p in points]) for name in fields}
    result.update({
        "state": np.asarray(history), "final_state": state.copy(),
        "final_command_voltage_v": previous, "n_substeps": n_substeps,
        "n_rejected_substeps": n_rejected,
        "state_atol": state_atol, "state_rtol": state_rtol,
        "integration": "adaptive bounded exponential Euler with step doubling",
        "jointly_calibrated": False,
        "model_status": "uncalibrated_shared_material_reference",
        "clock": "absolute command-voltage path in volts; not physical seconds",
        "microscopic_species_identified": False,
    })
    return result


def simulate_1r(command_voltage, initial_state, law, *, compliance_a=None,
                max_step_v=.02, initial_command_voltage_v=0.,
                state_atol=1e-9, state_rtol=1e-6):
    """Bare-device wrapper; uses exactly the same material and integrator."""
    return run_waveform(command_voltage, initial_state, law, BareCircuit(compliance_a),
                        max_step_v=max_step_v,
                        initial_command_voltage_v=initial_command_voltage_v,
                        state_atol=state_atol, state_rtol=state_rtol)


def simulate_1t1r(command_voltage, initial_state, law, mapping, gate_voltage, *,
                  compliance_a=None, max_step_v=.02, initial_command_voltage_v=0.,
                  state_atol=1e-9, state_rtol=1e-6):
    """Series-transistor wrapper; no alteration of material parameters or state."""
    return run_waveform(command_voltage, initial_state, law,
                        TransistorCircuit(mapping, gate_voltage, compliance_a),
                        max_step_v=max_step_v,
                        initial_command_voltage_v=initial_command_voltage_v,
                        state_atol=state_atol, state_rtol=state_rtol)
