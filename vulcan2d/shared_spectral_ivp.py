"""Signed circuit adapters and continuous path integration for one spectral law.

Research implementation only. The shape is common to both circuits and both
polarities. A tabulation is an explicitly testable numerical approximation,
not an additional material parameter. Commands are not physical seconds.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import math
import time
import warnings

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from .shared_gap import GapPoint, LookupGapCircuit
from .shared_gap_ivp import _clock


class SpectralTable:
    """Log-current interpolation, monotone in voltage and level broadening.

    Both signs are sampled independently. Interpolation never substitutes a
    positive material voltage for the negative branch. Outside the sampled
    domain callers use the actual law rather than extrapolate this table.
    """
    def __init__(self, shape, log_gamma, log_voltage, log_current, metadata=None):
        self.shape = shape
        self.log_gamma = np.asarray(log_gamma, dtype=float)
        self.log_voltage = np.asarray(log_voltage, dtype=float)
        self.log_current = np.asarray(log_current, dtype=float)
        self.metadata = metadata or {}
        expected = (2, len(self.log_gamma), len(self.log_voltage))
        if self.log_current.shape != expected or not np.all(np.isfinite(self.log_current)):
            raise ValueError('Incomplete or invalid spectral table')
        if np.any(np.diff(self.log_gamma) <= 0) or np.any(np.diff(self.log_voltage) <= 0):
            raise ValueError('Nonmonotone table coordinates')
        if np.min(np.diff(self.log_current, axis=1)) < -1e-9:
            raise ValueError('Table violates state ordering')
        if np.min(np.diff(self.log_current, axis=2)) < -1e-9:
            raise ValueError('Table violates voltage monotonicity')
        ceiling = self.log_voltage-math.log(self.shape.ron_ohm)
        if np.max(self.log_current-ceiling[None, None, :]) > 1e-9:
            raise ValueError('Table violates Ohmic transmission bound')

    @classmethod
    def build(cls, shape, *, gamma_decade_step=.1, voltage_decade_step=.025,
              gamma_decade_bounds=(-24., 6.), minimum_voltage=1e-9,
              maximum_voltage=5., order=32, progress=None):
        started = time.monotonic()
        g0, g1 = gamma_decade_bounds
        lg = np.linspace(g0, g1, int(np.ceil((g1-g0)/gamma_decade_step))+1)*np.log(10.)
        v0, v1 = np.log10(minimum_voltage), np.log10(maximum_voltage)
        lv = np.linspace(v0, v1, int(np.ceil((v1-v0)/voltage_decade_step))+1)*np.log(10.)
        lv[0], lv[-1] = math.log(minimum_voltage), math.log(maximum_voltage)
        values = np.empty((2, len(lg), len(lv)))
        voltage = np.minimum(maximum_voltage, np.exp(lv))
        for j, gamma in enumerate(lg):
            for branch, sign in enumerate((1., -1.)):
                current = abs(shape.current_at_log_gamma(sign*voltage, gamma, order=order))
                if np.any(~np.isfinite(current)) or np.any(current <= 0):
                    raise FloatingPointError('Invalid current while building table')
                values[branch, j] = np.log(current)
            if progress is not None and (j % 25 == 0 or j == len(lg)-1):
                progress({'row': j+1, 'total_rows': len(lg), 'elapsed_s': time.monotonic()-started})
        metadata = {'shape': asdict(shape), 'gamma_decade_step_max': gamma_decade_step,
                    'voltage_decade_step_max': voltage_decade_step, 'quadrature_order': order,
                    'elapsed_s': time.monotonic()-started,
                    'scope': 'Numerical table, must validate interpolation at used parameters'}
        return cls(shape, lg, lv, values, metadata)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as stream:
            np.savez_compressed(stream, log_gamma=self.log_gamma, log_voltage=self.log_voltage,
                                log_current=self.log_current, metadata=json.dumps(self.metadata))

    @classmethod
    def load(cls, path, shape):
        with np.load(path, allow_pickle=False) as source:
            metadata = json.loads(str(source['metadata']))
            if metadata['shape'] != asdict(shape):
                raise ValueError('Cached table belongs to another material shape')
            return cls(shape, source['log_gamma'], source['log_voltage'],
                       source['log_current'], metadata)

    def profile(self, log_gamma, sign):
        if not self.log_gamma[0] <= log_gamma <= self.log_gamma[-1]:
            return None
        index = int(np.searchsorted(self.log_gamma, log_gamma, side='right'))-1
        index = min(len(self.log_gamma)-2, max(index, 0))
        fraction = (log_gamma-self.log_gamma[index])/(self.log_gamma[index+1]-self.log_gamma[index])
        branch = 0 if sign > 0 else 1
        return ((1.-fraction)*self.log_current[branch, index]
                +fraction*self.log_current[branch, index+1])


class TabulatedSpectralLaw:
    def __init__(self, exact_law, table):
        if asdict(exact_law.shape) != asdict(table.shape):
            raise ValueError('Law and table shape differ')
        self.exact_law, self.table = exact_law, table
        self.shape = exact_law.shape
        self.state_limit = exact_law.state_limit
        self.jset_a = exact_law.jset_a
        self.ron_ohm = exact_law.shape.ron_ohm

    def validate_state(self, u):
        return self.exact_law.validate_state(u)

    def derivative(self, u, current_a):
        return self.exact_law.derivative(u, current_a)

    def _profile(self, u, sign):
        state = self.validate_state(u)
        if state == 0.:
            return None
        log_gamma = math.log(self.exact_law.gamma_c_ev)-math.log(math.expm1(state))
        return self.table.profile(log_gamma, sign)

    def current(self, voltage, u):
        v, state = float(voltage), self.validate_state(u)
        if v == 0. or state == 0.:
            return v/self.ron_ohm
        profile = self._profile(state, v)
        logv = math.log(abs(v))
        if profile is None or not self.table.log_voltage[0] <= logv <= self.table.log_voltage[-1]:
            return float(self.exact_law.current(v, state))
        return math.copysign(math.exp(float(np.interp(logv, self.table.log_voltage, profile))), v)

    def voltage(self, current, u):
        i, state = float(current), self.validate_state(u)
        if i == 0. or state == 0.:
            return i*self.ron_ohm
        profile = self._profile(state, i)
        logi = math.log(abs(i))
        if profile is None or not profile[0] <= logi <= profile[-1]:
            return float(self.exact_law.voltage(i, state))
        return math.copysign(math.exp(float(np.interp(logi, profile, self.table.log_voltage))), i)


class BareSpectralCircuit:
    def __init__(self, compliance_a=None):
        if compliance_a is not None and (not math.isfinite(compliance_a) or compliance_a <= 0):
            raise ValueError('Positive compliance required')
        self.compliance_a = compliance_a

    def positive_current_bound(self):
        return self.compliance_a

    def solve(self, command_v, law, u):
        v = float(command_v)
        if not math.isfinite(v):
            raise ValueError('Nonfinite command')
        current = law.current(v, u)
        limited = self.compliance_a is not None and abs(current) > self.compliance_a
        if limited:
            current = math.copysign(self.compliance_a, v)
            vh = law.voltage(current, u)
        else:
            vh = v
        return GapPoint(v, vh, vh, 0., current, v-vh, limited)


class LookupSpectralCircuit:
    """Measured signed transistor in series with a possibly rectifying material.

    LookupGapCircuit is used ONLY to parse the independently measured transistor
    knots, never for its antisymmetric-material load equation.
    """
    def __init__(self, mapping, gate_voltage, compliance_a=None):
        if compliance_a is not None:
            raise ValueError('This trial has no extra series instrument compliance')
        parsed = LookupGapCircuit(mapping, gate_voltage)
        self.mapping, self.gate_voltage = mapping, float(gate_voltage)
        self._branches = parsed._branches
        self.compliance_a = None
        self._flat_intervals = {}
        for name, (volts, currents) in self._branches.items():
            intervals = []
            for value in np.unique(currents[np.flatnonzero(np.diff(currents) == 0)]):
                equal = np.flatnonzero(currents == value)
                intervals.append((float(value), float(volts[equal[0]]), float(volts[equal[-1]])))
            self._flat_intervals[name] = intervals

    def positive_current_bound(self):
        return float(self._branches['set'][1][-1])

    @staticmethod
    def _transistor_voltage(target, voltage, current):
        return LookupGapCircuit._minimum_transistor_voltage(target, voltage, current)

    def solve(self, command_v, law, u):
        v, state = float(command_v), law.validate_state(u)
        if not math.isfinite(v):
            raise ValueError('Nonfinite command')
        if v == 0.:
            return GapPoint(0., 0., 0., 0., 0., 0., False)
        sign, magnitude = (1. if v > 0. else -1.), abs(v)
        side = 'set' if v > 0. else 'reset'
        vt, it = self._branches[side]
        # Bracket in log current for relative accuracy even for tiny HRS I.
        material_max = abs(law.current(v, state))
        # A measured transistor plateau is an interval-valued inverse. Solve
        # such loads explicitly; a scalar inverse would jump over valid KVL.
        for plateau_i, first, last in self._flat_intervals[side]:
            if plateau_i > material_max:
                continue
            vh = abs(law.voltage(sign*plateau_i, state))
            needed = magnitude-vh
            if first <= needed <= last:
                return GapPoint(v, v, sign*vh, sign*needed, sign*plateau_i, 0., False)
        upper = min(material_max, float(it[-1]))
        if upper <= 0.:
            raise FloatingPointError('Nonpositive circuit upper current')

        def terms(logi):
            # The bracketing variable never exceeds log(upper); exp/log
            # roundoff may nevertheless exceed the original knot by one ulp.
            i = min(upper, math.exp(logi))
            h = abs(law.voltage(sign*i, state))
            t = self._transistor_voltage(i, vt, it)
            return h+t-magnitude, h, t

        logupper = math.log(upper)
        end, eh, et = terms(logupper)
        if end < -1e-10:
            raise ValueError('Series circuit would exceed measured transistor voltage range')
        if abs(end) <= 1e-12:
            return GapPoint(v, v, sign*eh, sign*(magnitude-eh), sign*upper, 0., False)
        lower = logupper-2.
        value = terms(lower)[0]
        while value > 0. and lower > -700.:
            lower -= 2.
            value = terms(lower)[0]
        if value > 0.:
            raise FloatingPointError('Could not bracket spectral circuit root')
        root = brentq(lambda x: terms(x)[0], lower, logupper, xtol=2e-11, rtol=9e-15)
        _, h, t = terms(root)
        current = sign*math.exp(root)
        return GapPoint(v, v, sign*h, sign*t, current, 0., False)


def _positive_floor(initial_u, law, circuit):
    cap = circuit.positive_current_bound()
    if cap is None:
        return 0.
    # Derived for THIS beta=1 log-threshold law, not inherited from GapLaw.
    return min(initial_u, max(0., math.log(law.jset_a)-math.log(cap)))


def _integrate(command, initial_u, law, circuit, initial_command, method, rtol, atol, max_step):
    path, voltage, inverse, segments = _clock(command, initial_command)
    saved = np.full(len(path), initial_u)
    state, nfev, njev, nlu = initial_u, 0, 0, 0
    warning_text, lower_events = [], 0
    for start, stop, va, vb in segments:
        sign = np.sign(va if va != 0. else vb)
        lower = _positive_floor(state, law, circuit) if sign > 0 else 0.
        upper = law.state_limit
        indices = np.flatnonzero((path >= start) & (path <= stop))
        if sign > 0 and state == lower:
            saved[indices] = state
            continue
        direction = (vb-va)/(stop-start)

        def rhs(s, y):
            proposed = min(upper, max(lower, float(y[0])))
            v = np.clip(va+direction*(s-start), min(va, vb), max(va, vb))
            return [law.derivative(proposed, circuit.solve(v, law, proposed).current_a)]

        def boundary(s, y):
            return y[0]-(lower if sign > 0 else upper)

        boundary.terminal = True
        boundary.direction = -1. if sign > 0 else 1.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            sol = solve_ivp(rhs, (start, stop), [state], method=method, rtol=rtol, atol=atol,
                            max_step=max_step, events=boundary, dense_output=True)
        warning_text.extend(str(w.message) for w in caught)
        nfev += sol.nfev
        njev += sol.njev
        nlu += sol.nlu
        if not sol.success:
            raise RuntimeError(f'Spectral state integration failed: {sol.message}')
        hit = bool(len(sol.t_events[0]))
        if hit and sign < 0:
            raise RuntimeError('State left certified numerical domain; no RESET saturation inserted')
        endpoint = sol.t[-1]
        before = indices[path[indices] <= endpoint]
        saved[before] = sol.sol(path[before])[0]
        if hit:
            saved[indices[path[indices] > endpoint]] = lower
            state = lower
            lower_events += 1
        else:
            state = float(sol.y[0, -1])
        tolerance = 10*(atol+rtol*max(1., abs(state)))
        if np.any(saved[indices] < lower-tolerance) or np.any(saved[indices] > upper+tolerance):
            raise RuntimeError('Spectral dense trajectory violated the state interval')
        saved[indices] = np.clip(saved[indices], lower, upper)
        state = min(upper, max(lower, state))
    states = saved[inverse][1:]
    points = [circuit.solve(float(v), law, float(u)) for v, u in zip(command, states)]
    result = {name: np.array([getattr(p, name) for p in points],
                            dtype=bool if name == 'compliance_limited' else float)
              for name in GapPoint.__dataclass_fields__}
    result.update(state_u=states, final_u=state)
    return result, {'nfev': nfev, 'njev': njev, 'nlu': nlu,
                    'segments': len(segments), 'lower_bound_events': lower_events,
                    'warnings': sorted(set(warning_text))}


def simulate_spectral_ivp(command_voltage, initial_u, law, circuit, *,
                          initial_command_voltage_v=0., method='DOP853',
                          state_rtol=1e-6, state_atol=1e-8, current_rtol=1e-3,
                          state_refine_atol=2e-5, max_step_v=np.inf, max_refinements=2):
    """Continuous state solve, with both current AND state refinement checks."""
    started = time.monotonic()
    command = np.asarray(command_voltage, dtype=float)
    if command.ndim != 1 or not np.all(np.isfinite(command)):
        raise ValueError('Finite one-dimensional command required')
    initial = law.validate_state(initial_u)
    if not np.isfinite(initial_command_voltage_v):
        raise ValueError('Initial command voltage must be finite')
    if method not in ('DOP853', 'Radau', 'BDF'):
        raise ValueError('Unsupported state solver')
    if not all(np.isfinite(x) and x > 0 for x in (state_rtol, state_atol, current_rtol, state_refine_atol)):
        raise ValueError('Positive finite tolerances required')
    if max_refinements < 1 or np.isnan(max_step_v) or max_step_v <= 0:
        raise ValueError('Invalid refinement settings')
    coarse, info = _integrate(command, initial, law, circuit, initial_command_voltage_v,
                              method, state_rtol, state_atol, max_step_v)
    rounds = [dict(info, state_rtol=state_rtol, state_atol=state_atol)]
    for iteration in range(1, max_refinements+1):
        rtol, atol = state_rtol/5**iteration, state_atol/5**iteration
        fine, info = _integrate(command, initial, law, circuit, initial_command_voltage_v,
                                method, rtol, atol, max_step_v)
        rounds.append(dict(info, state_rtol=rtol, state_atol=atol))
        a, b = abs(coarse['current_a']), abs(fine['current_a'])
        use = (a != 0) | (b != 0)
        logerr = float(np.max(abs(np.log(a[use])-np.log(b[use])))) if np.any(use) else 0.
        stateerr = float(np.max(abs(coarse['state_u']-fine['state_u']))) if len(command) else 0.
        enderr = abs(coarse['final_u']-fine['final_u'])
        if logerr <= np.log1p(current_rtol) and max(stateerr, enderr) <= state_refine_atol:
            fine['metadata'] = {'clock': 'cumulative_absolute_command_voltage_path_v',
                'method': method, 'rounds': rounds, 'max_log_current_refinement_natural': logerr,
                'max_state_refinement_u': stateerr, 'endpoint_refinement_u': enderr,
                'elapsed_s': time.monotonic()-started,
                'scope': 'ODE refinement; material-table errors need separate certification'}
            return fine
        coarse = fine
    raise RuntimeError(f'Spectral trajectory refinement failed: logI={logerr:g}, state={stateerr:g}')
