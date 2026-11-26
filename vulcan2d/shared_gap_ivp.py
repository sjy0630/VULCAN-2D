"""Adaptive integration of the existing shared-gap law, with no new physics.

State tolerances control the IVP solver's local error. An additional complete
trajectory at tighter tolerances estimates current error at every returned
observation. That refinement comparison is a numerical estimate, not a
rigorous global error bound. Independent reference audits remain necessary.
"""
from __future__ import annotations
import math
import time
import warnings
import numpy as np
from scipy.integrate import solve_ivp

from .shared_gap import GapPoint, BareGapCircuit, LookupGapCircuit


def _clock(command, initial_command):
    values = np.r_[initial_command, command]
    path = np.r_[0., np.cumsum(np.abs(np.diff(values)))]
    unique, index, inverse = np.unique(path, return_index=True, return_inverse=True)
    voltage = values[index]
    # A polarity change is an explicit segment boundary even when the caller
    # omits the intervening zero-voltage observation.
    add = []
    for j in range(len(unique)-1):
        if voltage[j]*voltage[j+1] < 0.:
            add.append(unique[j]+abs(voltage[j]))
    knots = np.unique(np.r_[unique, add])
    knot_v = np.interp(knots, unique, voltage)
    if add:
        knot_v[np.searchsorted(knots, add)] = 0.
    if len(knots) < 2:
        return unique, voltage, inverse, []
    directions = np.sign(np.diff(knot_v))
    cuts = np.unique(np.r_[0, np.flatnonzero(np.diff(directions) != 0.)+1,
                           np.flatnonzero(knot_v == 0.), len(knots)-1]).astype(int)
    segments = [(float(knots[a]), float(knots[b]), float(knot_v[a]), float(knot_v[b]))
                for a, b in zip(cuts[:-1], cuts[1:]) if knots[b] > knots[a]]
    return unique, voltage, inverse, segments


def _positive_floor(initial_u, law, circuit):
    """Exact invariant from the existing positive-current threshold and cap.

    I <= Imax implies du/ds=0 at u<=log(Icrit/Imax). This prevents numerical
    overshoot through the compliance stopping state; it does not introduce
    a state reset or change the differential equation in its feasible region.
    """
    cap = None
    if isinstance(circuit, BareGapCircuit):
        cap = circuit.compliance_a
    elif isinstance(circuit, LookupGapCircuit):
        cap = float(circuit._branches['set'][1][-1])
        if circuit.compliance_a is not None:
            cap = min(cap, circuit.compliance_a)
    if cap is None or cap <= 0. or law.eth_set_v == 0.:
        return 0.
    ratio = law.eth_set_v/law.v0_v
    logsinh = ratio-math.log(2.) if ratio > 35. else math.log(math.sinh(ratio))
    floor = max(0., math.log(law.i0_a)+logsinh-math.log(cap))
    return min(float(initial_u), floor)


def _integrate(command, initial_u, law, circuit, initial_command, method, rtol, atol, max_step):
    path, voltage, inverse, segments = _clock(command, initial_command)
    saved = np.full(len(path), initial_u, dtype=float)
    state = float(initial_u)
    diagnostics = {'nfev':0, 'njev':0, 'nlu':0, 'segments':len(segments),
                   'absorbing_bound_events':0, 'warnings':[]}
    for start, stop, va, vb in segments:
        sign = np.sign(va if va != 0. else vb)
        lower = _positive_floor(state, law, circuit) if sign > 0. else 0.
        upper = law.u_max
        bound = lower if sign > 0. else upper
        indices = np.flatnonzero((path >= start) & (path <= stop))
        direction = (vb-va)/(stop-start)
        if state == bound:
            saved[indices] = state
            continue
        def rhs(s, y):
            u = min(upper, max(lower, float(y[0])))
            v = min(max(va, vb), max(min(va, vb), va+direction*(s-start)))
            return [law.derivative(u, circuit.solve(v, law, u).current_a)]
        def at_bound(s, y):
            del s
            return float(y[0])-bound
        at_bound.terminal = True
        at_bound.direction = -1. if sign > 0. else 1.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = solve_ivp(rhs, (start, stop), [state], method=method,
                               rtol=rtol, atol=atol, max_step=max_step,
                               events=at_bound, dense_output=True)
        diagnostics['warnings'].extend(str(item.message) for item in caught)
        for name in ('nfev','njev','nlu'):
            diagnostics[name] += int(getattr(result, name))
        if not result.success:
            raise RuntimeError(f'Adaptive gap integration failed: {result.message}')
        event = bool(len(result.t_events[0]))
        endpoint = float(result.t[-1])
        before = indices[path[indices] <= endpoint]
        if len(before):
            saved[before] = result.sol(path[before])[0]
        if event:
            saved[indices[path[indices] > endpoint]] = bound
            state = bound
            diagnostics['absorbing_bound_events'] += 1
        else:
            state = float(result.y[0,-1])
        # Roundoff-sized endpoint interpolation excursions are permitted only
        # within declared numerical tolerance; larger violations fail loudly.
        tolerance = 10.*(atol+rtol*max(abs(state),1.))
        values = saved[indices]
        if np.any(values < lower-tolerance) or np.any(values > upper+tolerance):
            raise RuntimeError('Adaptive dense output left invariant state bounds')
        saved[indices] = np.clip(values, lower, upper)
        if not lower-tolerance <= state <= upper+tolerance:
            raise RuntimeError('Adaptive endpoint left invariant state bounds')
        state = min(upper, max(lower, state))
    observations = saved[inverse][1:]
    points = [circuit.solve(float(v), law, float(u)) for v,u in zip(command, observations)]
    output = {name: np.array([getattr(point,name) for point in points],
                             dtype=bool if name == 'compliance_limited' else float)
              for name in GapPoint.__dataclass_fields__}
    output.update(state_u=observations, final_u=state)
    diagnostics['warnings'] = sorted(set(diagnostics['warnings']))
    return output, diagnostics


def simulate_gap_ivp(command_voltage, initial_u, law, circuit, *,
                     initial_command_voltage_v=0., method='Radau',
                     state_rtol=1e-7, state_atol=1e-9, max_step_v=np.inf,
                     current_rtol=1e-3, max_refinements=3):
    """Return the simulate_gap fields using adaptive, continuously carried state.

    Commands are piecewise linear in cumulative absolute voltage path. Turning
    points and polarity crossings are explicit integration boundaries, and
    repeated commands consume zero path. Bounds are absorbing only in their
    outward-flow polarity. Current relative tolerance is checked by tightening
    state tolerances fivefold and comparing all nonzero-current outputs; the
    tighter trajectory is returned. It is an a posteriori consistency estimate,
    not an absolute accuracy guarantee or a test against measured current.
    """
    started = time.monotonic()
    command = np.asarray(command_voltage, dtype=float)
    if command.ndim != 1 or not np.all(np.isfinite(command)):
        raise ValueError('Command waveform must be a finite one-dimensional vector')
    state = law.validate_state(initial_u)
    initial = float(initial_command_voltage_v)
    if not math.isfinite(initial):
        raise ValueError('Initial command must be finite')
    if method not in ('Radau','DOP853','BDF'):
        raise ValueError('method must be Radau, DOP853, or BDF')
    for name, value in (('state_rtol',state_rtol),('state_atol',state_atol),('current_rtol',current_rtol)):
        if not math.isfinite(value) or value <= 0.:
            raise ValueError(f'{name} must be positive and finite')
    if math.isnan(max_step_v) or max_step_v <= 0.:
        raise ValueError('max_step_v must be positive')
    if not isinstance(max_refinements,int) or max_refinements < 1:
        raise ValueError('max_refinements must be a positive integer')
    coarse, detail = _integrate(command,state,law,circuit,initial,method,
                                state_rtol,state_atol,max_step_v)
    rounds = [dict(detail,state_rtol=state_rtol,state_atol=state_atol)]
    log_error, state_error = 0., 0.
    for iteration in range(1,max_refinements+1):
        rtol, atol = state_rtol*5.**(-iteration), state_atol*5.**(-iteration)
        fine, detail = _integrate(command,state,law,circuit,initial,method,rtol,atol,max_step_v)
        rounds.append(dict(detail,state_rtol=rtol,state_atol=atol))
        ci, fi = np.abs(coarse['current_a']),np.abs(fine['current_a'])
        nonzero = (ci != 0.) | (fi != 0.)
        if np.any((ci[nonzero] == 0.) | (fi[nonzero] == 0.)):
            log_error = math.inf
        elif np.any(nonzero):
            log_error = float(np.max(np.abs(np.log(ci[nonzero])-np.log(fi[nonzero]))))
        else:
            log_error = 0.
        state_error = float(np.max(np.abs(coarse['state_u']-fine['state_u']))) if len(command) else 0.
        if log_error <= math.log1p(current_rtol):
            fine['metadata'] = {'clock':'cumulative_absolute_command_voltage_path_v',
                'method':method,'state_rtol':state_rtol,'state_atol':state_atol,
                'effective_state_rtol':rtol,'effective_state_atol':atol,
                'current_rtol':current_rtol,
                'current_tolerance_refinement_passed':True,
                'current_relative_error_estimate':math.expm1(log_error),
                'current_log10_error_estimate':log_error/math.log(10.),
                'max_state_difference_last_refinement':state_error,
                'max_step_v':max_step_v if math.isfinite(max_step_v) else None,
                'initial_u':state,'initial_command_voltage_v':initial,
                'refinement_rounds':iteration,'integrations':rounds,
                'elapsed_s':time.monotonic()-started,
                'error_estimate_scope':'tolerance refinement at returned observation points; not a rigorous global bound',
                'positive_compliance_invariant_enforced':True}
            return fine
        coarse = fine
    raise RuntimeError(f'Adaptive current refinement did not converge: estimated relative error {math.expm1(log_error):.6g}')
