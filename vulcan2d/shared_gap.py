"""Shared, uncalibrated effective-gap candidate for bare 1R and series 1T1R.

The *same* material law and parameters operate in either external circuit.
The scalar gap u is dimensionless, not an identified atomic distance. Rates
use cumulative absolute commanded-voltage path, not physical seconds. A
positive return sweep continues SET dynamics; no branch is frozen or reset.
This module does not replace the archived validated calibration.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy.optimize import brentq

_LOG2 = math.log(2.)


def _asinh_exp(logx):
    return logx + _LOG2 if logx > 35. else math.asinh(math.exp(logx))


@dataclass(frozen=True)
class GapLaw:
    """Vh = Ron*I + V0*u*asinh(I*exp(u)/I0), with bounded field dynamics.

    Field proxy e = V0*asinh(|I|*exp(u)/I0) has a continuous u→0 limit.
    du/ds = -kset*sinh((e-eth_set)+/escale) for positive current and the
    corresponding positive reset rate for negative current. The optional
    global reset threshold increment reset_initiation_drive_v*exp(-u) represents a
    candidate effective opening threshold near a closed gap; it identifies no
    atomic species or proven bond-breaking mechanism. No topology,
    gate voltage, measured target current, or cycle number enters this law.
    """
    ron_ohm: float
    v0_v: float
    i0_a: float
    u_max: float
    kset_per_v: float
    kreset_per_v: float
    eth_set_v: float
    eth_reset_v: float
    escale_v: float
    reset_initiation_drive_v: float = 0.

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = float(getattr(self, name))
            positive = name in ("ron_ohm", "v0_v", "i0_a", "u_max", "escale_v")
            if not math.isfinite(value) or value < 0. or (positive and value == 0.):
                raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
            object.__setattr__(self, name, value)

    def validate_state(self, u):
        state = float(u)
        if not math.isfinite(state) or not 0. <= state <= self.u_max:
            raise ValueError("Effective gap u must be finite and within [0,u_max]")
        return state

    def drive(self, current_a, u):
        state = self.validate_state(u)
        current = abs(float(current_a))
        if not math.isfinite(current):
            raise ValueError("Current must be finite")
        return 0. if current == 0. else self.v0_v * _asinh_exp(math.log(current) + state - math.log(self.i0_a))

    def voltage(self, current_a, u):
        state = self.validate_state(u)
        currents = np.asarray(current_a, dtype=float)
        if not np.all(np.isfinite(currents)):
            raise ValueError("Current must be finite")
        if currents.ndim == 0:
            current = float(currents)
            return self.ron_ohm * current if state == 0. else self.ron_ohm * current + math.copysign(state * self.drive(current, state), current)
        if state == 0.:
            return self.ron_ohm * currents
        magnitude = np.abs(currents)
        nonzero = magnitude > 0.
        term = np.zeros_like(magnitude)
        logx = np.log(magnitude[nonzero]) + state - math.log(self.i0_a)
        value = np.empty_like(logx)
        large = logx > 35.
        value[large] = logx[large] + _LOG2
        value[~large] = np.arcsinh(np.exp(logx[~large]))
        term[nonzero] = self.v0_v * state * value
        return self.ron_ohm * currents + np.copysign(term, currents)

    def _current_with_series(self, magnitude_v, u, extra_resistance_ohm=0.):
        """Invert in log-current coordinates to retain tiny HRS currents."""
        resistance = self.ron_ohm + extra_resistance_ohm
        if magnitude_v == 0.:
            return 0.
        if u == 0.:
            return magnitude_v / resistance
        logv, logr = math.log(magnitude_v), math.log(resistance)
        logi0 = math.log(self.i0_a)
        # asinh(x) <= x: zero-bias tangent gives a rigorous lower bound.
        lower = logv - float(np.logaddexp(logr, math.log(self.v0_v * u) + u - logi0))
        upper = logv - logr
        if upper - lower < 2e-14:
            return math.exp(upper)
        def residual(logi):
            return resistance * math.exp(logi) + self.v0_v * u * _asinh_exp(logi + u - logi0) - magnitude_v
        # Tiny rounding errors at analytical bounds must not invalidate them.
        root = brentq(residual, lower - 2e-12, upper + 2e-12, xtol=2e-12, rtol=9e-15)
        result = math.exp(root)
        if result == 0.:
            raise FloatingPointError("Intrinsic current underflow outside numerical domain")
        return result

    def current(self, voltage_hbn_v, u):
        state, voltage = self.validate_state(u), float(voltage_hbn_v)
        if not math.isfinite(voltage):
            raise ValueError("Voltage must be finite")
        return math.copysign(self._current_with_series(abs(voltage), state), voltage)

    def derivative(self, u, current_a):
        state, current = self.validate_state(u), float(current_a)
        if not math.isfinite(current):
            raise ValueError("Current must be finite")
        if current == 0. or (current > 0. and state == 0.) or (current < 0. and state == self.u_max):
            return 0.
        threshold = (self.eth_set_v if current > 0. else
                     self.eth_reset_v + self.reset_initiation_drive_v * math.exp(-state))
        rate = self.kset_per_v if current > 0. else self.kreset_per_v
        overdrive = max(self.drive(current, state) - threshold, 0.) / self.escale_v
        if rate == 0. or overdrive == 0.:
            return 0.
        try:
            magnitude = rate * math.sinh(overdrive)
        except OverflowError as exc:
            raise FloatingPointError("Gap dynamics overflow outside numerical domain") from exc
        if not math.isfinite(magnitude):
            raise FloatingPointError("Gap dynamics overflow outside numerical domain")
        return -magnitude if current > 0. else magnitude


@dataclass(frozen=True)
class GapPoint:
    command_voltage_v: float
    supply_voltage_v: float
    hbn_voltage_v: float
    transistor_voltage_v: float
    current_a: float
    compliance_drop_v: float
    compliance_limited: bool


def _check_compliance(value):
    if value is not None and (not math.isfinite(float(value)) or float(value) <= 0.):
        raise ValueError("Compliance must be positive and finite or None")


@dataclass(frozen=True)
class BareGapCircuit:
    """Bare device with ideal optional current regulation of actual voltage."""
    compliance_a: float | None = None

    def __post_init__(self):
        _check_compliance(self.compliance_a)

    def solve(self, command_v, law, u):
        command = float(command_v)
        current = law.current(command, u)
        limited = self.compliance_a is not None and abs(current) > self.compliance_a
        if limited:
            current = math.copysign(self.compliance_a, command)
            voltage = law.voltage(current, u)
        else:
            voltage = command
        return GapPoint(command, voltage, voltage, 0., current, command - voltage, limited)


class LookupGapCircuit:
    """Exact fixed-gate piecewise-linear transistor adapter, without extrapolation.

    Lookup knots include the union of measured voltage breakpoints at the
    requested gate. Flat current segments retain their voltage interval.
    Original physical source remains the reference in either bias direction.
    """
    def __init__(self, mapping, gate_voltage, compliance_a=None):
        _check_compliance(compliance_a)
        self.mapping, self.gate_voltage, self.compliance_a = mapping, float(gate_voltage), compliance_a
        if not math.isfinite(self.gate_voltage):
            raise ValueError("Gate voltage must be finite")
        if not math.isfinite(mapping.current_scale) or mapping.current_scale <= 0.:
            raise ValueError("Transistor current scale must be positive and finite")
        if not math.isfinite(mapping.series_resistance_ohm) or mapping.series_resistance_ohm < 0.:
            raise ValueError("Transistor series resistance must be nonnegative and finite")
        if mapping.reset_policy != "measured_reverse":
            raise ValueError("Shared signed circuit requires measured reverse transistor data")
        self._branches = {}
        for side in ("set", "reset"):
            lookup = mapping._active_lookup(side)
            gate = mapping.effective_gate_voltage(self.gate_voltage, side)
            curves = lookup.curves()
            knots = np.unique(np.concatenate([curve['drain_voltage_v'] for curve in curves]))
            knots = knots[(knots >= lookup.vds_min) & (knots <= lookup.vds_max)]
            currents = np.array([mapping.current_scale * lookup.current(v, gate) for v in knots])
            terminal = knots + currents * mapping.series_resistance_ohm
            if (len(knots) < 2 or knots[0] != 0. or currents[0] != 0.
                    or np.any(np.diff(terminal) <= 0.) or np.any(np.diff(currents) < 0.)):
                raise ValueError("Transistor lookup must be monotone with a zero-voltage/current origin")
            self._branches[side] = (terminal, currents)

    @staticmethod
    def _minimum_transistor_voltage(target, voltage, current):
        if target == 0.:
            return 0.
        index = int(np.searchsorted(current, target, side="left"))
        if index >= len(current):
            raise ValueError("Compliance exceeds transistor lookup range")
        if current[index] == target:
            return float(voltage[index])
        left = index - 1
        fraction = (target - current[left]) / (current[index] - current[left])
        return float(voltage[left] + fraction * (voltage[index] - voltage[left]))

    def solve(self, command_v, law, u):
        command, state = float(command_v), law.validate_state(u)
        if not math.isfinite(command):
            raise ValueError("Command voltage must be finite")
        if command == 0.:
            return GapPoint(command, command, command, command, 0., 0., False)
        side, sign = ("reset", -1.) if command < 0. else ("set", 1.)
        voltage, current = self._branches[side]
        target = abs(command)
        load = law.voltage(current, state) + voltage
        # Compliance can solve a high command within the measured domain even
        # if the uncapped command would require unmeasured transistor voltage.
        if self.compliance_a is not None and self.compliance_a <= current[-1]:
            cap_vt = self._minimum_transistor_voltage(self.compliance_a, voltage, current)
            cap_vh = law.voltage(self.compliance_a, state)
            # At a plateau exactly equal to the cap, the unregulated
            # circuit already satisfies compliance. Keep command voltage
            # until the entire equal-current voltage interval is passed.
            equal_cap = np.flatnonzero(current == self.compliance_a)
            cap_vt_end = float(voltage[equal_cap[-1]]) if len(equal_cap) else cap_vt
            if cap_vh + cap_vt_end < target:
                supply = sign * (cap_vh + cap_vt)
                return GapPoint(command, supply, sign * cap_vh, sign * cap_vt,
                                sign * self.compliance_a, command - supply, True)
        if target > load[-1] + 1e-12:
            raise ValueError("Series operating point requires extrapolation outside transistor domain")
        target = min(target, float(load[-1]))
        right = min(int(np.searchsorted(load, target, side="right")), len(load) - 1)
        left = max(right - 1, 0)
        if current[right] == current[left]:
            magnitude = float(current[left])
        else:
            resistance = float((voltage[right] - voltage[left]) / (current[right] - current[left]))
            offset = float(voltage[left] - resistance * current[left])
            magnitude = law._current_with_series(target - offset, state, resistance)
            tolerance = 1e-10 * max(float(current[right]), 1e-30)
            if not current[left] - tolerance <= magnitude <= current[right] + tolerance:
                raise RuntimeError("Piecewise transistor inversion escaped its bracket")
        vh = law.voltage(magnitude, state)
        vt = target - vh
        return GapPoint(command, command, sign * vh, sign * vt, sign * magnitude, 0., False)


def simulate_gap(command_voltage, initial_u, law, circuit, *, max_step_v=.01,
                 initial_command_voltage_v=0., method="rk4"):
    """Integrate one continuous command path with projected RK2 or RK4.

    Fixed maximum path increments are subdivided if any RK stage implies a
    state change above 0.05. Bounds are projected only after this guard; this
    controls gross excursions but is NOT an error tolerance or a stiffness
    certificate. Reported results require step refinement and independent
    integration checks. Repeated command values represent zero path/time.
    """
    command = np.asarray(command_voltage, dtype=float)
    if command.ndim != 1 or not np.all(np.isfinite(command)):
        raise ValueError("Command waveform must be a finite one-dimensional vector")
    state, position = law.validate_state(initial_u), float(initial_command_voltage_v)
    if not math.isfinite(position) or not math.isfinite(max_step_v) or max_step_v <= 0.:
        raise ValueError("Initial command and positive maximum step must be finite")
    if method not in ("rk2", "rk4"):
        raise ValueError("method must be 'rk2' or 'rk4'")
    rows, states, accepted, rejected = [], [], 0, 0
    max_change = min(.05, law.u_max * .05)
    def project(value):
        return min(law.u_max, max(0., value))
    def derivative(v, value):
        value = project(value)
        return law.derivative(value, circuit.solve(v, law, value).current_a)
    for endpoint in command:
        stops = [0., float(endpoint)] if position * endpoint < 0. else [float(endpoint)]
        for stop in stops:
            direction = 1. if stop > position else -1.
            step = min(max_step_v, abs(stop - position))
            while direction * (stop - position) > 0.:
                remaining = abs(stop - position)
                step = min(step, remaining, max_step_v)
                if step <= 1e-14 * max(1., abs(position)):
                    if remaining < 1e-13:
                        position = stop
                        break
                    raise RuntimeError("Gap integration reached minimum path step; stiff parameter set")
                if accepted + rejected > 2000000:
                    raise RuntimeError("Gap integration exceeded step budget; stiff parameter set")
                k1 = derivative(position, state)
                if abs(step * k1) > max_change:
                    step *= max(.05, .8 * max_change / abs(step * k1))
                    rejected += 1
                    continue
                if method == "rk4":
                    k2 = derivative(position + direction * step / 2., state + step * k1 / 2.)
                    k3 = derivative(position + direction * step / 2., state + step * k2 / 2.)
                    k4 = derivative(position + direction * step, state + step * k3)
                    rates = (k1, k2, k3, k4)
                    trial = state + step * (k1 + 2. * k2 + 2. * k3 + k4) / 6.
                else:
                    # Heun retains the initial increment if its predictor
                    # reaches a bound or extinguishes the later rate.
                    k2 = derivative(position + direction * step, state + step * k1)
                    rates = (k1, k2)
                    trial = state + step * (k1 + k2) / 2.
                if max(abs(step * rate) for rate in rates) > max_change:
                    rejected += 1
                    step *= .5
                    continue
                state = project(trial)
                position = stop if step == remaining else position + direction * step
                accepted += 1
                step = min(2. * step, max_step_v)
        rows.append(circuit.solve(float(endpoint), law, state))
        states.append(state)
    result = {name: np.array([getattr(row, name) for row in rows])
              for name in GapPoint.__dataclass_fields__}
    result.update(state_u=np.asarray(states), final_u=state,
                  metadata={"clock": "cumulative_absolute_command_voltage_path_v",
                            "method": method, "max_step_v": float(max_step_v),
                            "accepted_steps": accepted, "rejected_steps": rejected,
                            "max_stage_state_change": max_change,
                            "initial_u": float(initial_u),
                            "initial_command_voltage_v": float(initial_command_voltage_v),
                            "independent_accuracy_validation_required": True})
    return result
