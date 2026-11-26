"""Prepared signed m=2 soft phase kernel; no experiment runs on import.

Frozen positive parameters/readout, explicit b/precision/initial state, a=0.
The parent owns finite experiment plans, budgets, trial persistence and fits.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import sys

sys.dont_write_bytecode = True
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT/'analysis/cooperative_relaxation_2026-11-04'
CACHE = Path('/private/tmp/vulcan-runtime-cache')
FROZEN_INPUTS = {
    PREVIOUS/'forward_kernel.py': '052971a4ee8e927769118a3bccac323c7d5ad9a81e08d0e5caf1fe7a38181302',
    PREVIOUS/'final_refined.json': '5fa0b38f637d8a47b6681283f6e9b5c898f0662fbcfe4abd72004c2f0154ea8c',
    PREVIOUS/'final_refined.npz': '9fd7f5274f84d1fb420c895736a38bee4fe2dc72b8b02f3b93ed444ff6df7f1a',
    PREVIOUS/'finite_family_fit.json': '24aca3ee6bf65915814d1dceef8a425f50b5898b8298f3d657e1585628615c9e',
    PREVIOUS/'SIGNED_EXTENSION_READINESS.md': '331580f2e20b6632fc10f814785f24bdbb262ffa80947912770d5abed7ed870d',
}


class DomainFailure(ValueError):
    pass


class ResolutionFailure(ValueError):
    pass


class PhaseFailure(RuntimeError):
    def __init__(self, message, summary, arrays):
        super().__init__(message)
        self.summary, self.arrays = summary, arrays


@dataclass(frozen=True)
class Precision:
    name: str
    rtol: float
    atol: float
    max_step: float

    def __post_init__(self):
        if not self.name or any(not math.isfinite(float(value)) or value <= 0.
                                for value in (self.rtol, self.atol, self.max_step)):
            raise ValueError('Explicit named positive finite precision required')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_frozen_module(path):
    name = 'vulcan_frozen_m2_positive_for_signed_history'
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != path.resolve():
            raise RuntimeError('Frozen module name already used by a different source')
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _path_clock(command, initial_command):
    """Native cumulative voltage distance, with turning and actual zero crossings.

    This copies only the path geometry idea, not any old state clipping/floor.
    Repeated native voltages are retained through inverse indices with no dwell.
    """
    values = np.r_[initial_command, command]
    path = np.r_[0., np.cumsum(abs(np.diff(values)))]
    unique, first, inverse = np.unique(path, return_index=True, return_inverse=True)
    voltage = values[first]
    additions = [unique[index]+abs(voltage[index]) for index in range(len(unique)-1)
                 if voltage[index]*voltage[index+1] < 0.]
    knots = np.unique(np.r_[unique, additions])
    knot_v = np.interp(knots, unique, voltage)
    if additions:
        # The analytic zero of an interpolated sign-changing command edge.
        knot_v[np.searchsorted(knots, additions)] = 0.
    if len(knots) < 2:
        return unique, inverse, []
    direction = np.sign(np.diff(knot_v))
    cuts = np.unique(np.r_[0, np.flatnonzero(np.diff(direction) != 0.)+1,
                          np.flatnonzero(knot_v == 0.), len(knots)-1]).astype(int)
    segments = [(float(knots[a]), float(knots[b]), float(knot_v[a]), float(knot_v[b]))
                for a, b in zip(cuts[:-1], cuts[1:]) if knots[b] > knots[a]]
    return unique, inverse, segments


class SignedSeriesCircuit:
    """Actual sign selects both material and measured transistor branches."""
    def __init__(self, kernel, mode):
        if mode not in ('fast', 'strict_bisection42'):
            raise ValueError('Choose explicit fast or strict_bisection42 mode')
        self.kernel, self.mode = kernel, mode

    def solve(self, command_voltage, state_u):
        return self.kernel._solve_signed(float(command_voltage), float(state_u), self.mode)


class SignedSoftKernel:
    @classmethod
    def load(cls, cache=CACHE):
        """Callable input loader only; no current root, ODE, or fit evaluation."""
        for path, digest in FROZEN_INPUTS.items():
            if sha(path) != digest:
                raise ValueError(f'Frozen preceding-stage input changed: {path}')
        self = cls()
        self.cache = Path(cache)
        self.frozen_module = _load_frozen_module(PREVIOUS/'forward_kernel.py')
        self.base = self.frozen_module.ForwardKernel.load(self.cache)
        self.set4_reference_summary = json.loads((PREVIOUS/'final_refined.json').read_text())
        fit = json.loads((PREVIOUS/'finite_family_fit.json').read_text())
        if fit.get('numerical_validation_passed') is not True or self.set4_reference_summary['fixed_recovery_order'] != 2:
            raise ValueError('Frozen positive m2 source is not numerically validated')
        self.parameters = self.frozen_module.Parameters(**self.set4_reference_summary['parameters'])
        self.ron, self.gc, self.vmin = self.base.ron, self.base.gc, self.base.vmin
        self.lg, self.lv = self.base.lg, self.base.lv
        self.li_negative = self.base.transport.li[1]
        self.negative_tx, self.negative_ti = self.base.transport.knots['reset']
        self.li_negative.flags.writeable = False
        self.negative_tx.flags.writeable = False
        self.negative_ti.flags.writeable = False
        self.source_sha256 = dict(self.base.source_sha256)
        self.source_sha256.update({str(path): digest for path, digest in FROZEN_INPUTS.items()})
        self.source_sha256[str(Path(__file__))] = sha(Path(__file__))
        return self

    def load_training_phase(self, record_id, phase):
        if (record_id, phase) not in ((4, 'set'), (4, 'reset'), (5, 'set')):
            raise ValueError('This stage only permits SET4 regression, RESET4 and SET5 training arrays')
        prefix = f'1t1r_cycle_{record_id}_{phase}'
        with np.load(self.cache/'native_observations.npz') as archive:
            command = archive[prefix+'_voltage'].copy()
            observed = archive[prefix+'_current'].copy()
        threshold = 1e-11 if phase == 'set' else 1e-12
        mask = np.isfinite(abs(observed)) & (abs(observed) >= threshold) & (abs(command) > 1e-12)
        return dict(command_voltage_v=command, observed_current_a=observed, scoring_mask=mask,
                    record_id=record_id, phase=phase,
                    source_sha256=self.source_sha256[str(self.cache/'native_observations.npz')])

    def load_set4_reference_arrays(self):
        with np.load(PREVIOUS/'final_refined.npz') as archive:
            return {key: archive[key].copy() for key in archive.files}

    def make_circuit(self, *, mode):
        return SignedSeriesCircuit(self, mode)

    def gamma(self, state_u):
        try:
            return self.base.gamma(float(state_u))
        except self.frozen_module.DomainFailure as exc:
            raise DomainFailure(str(exc)) from exc

    def _material_abs(self, magnitude, log_gamma, sign):
        if sign == 1:
            try:
                return self.base.material(magnitude, log_gamma)
            except self.frozen_module.DomainFailure as exc:
                raise DomainFailure(str(exc)) from exc
            except self.frozen_module.ResolutionFailure as exc:
                raise ResolutionFailure(str(exc)) from exc
        if sign != -1:
            raise ValueError('Nonzero material branch sign must be+1 or-1')
        if not math.isfinite(magnitude) or magnitude < 0.:
            raise DomainFailure('Invalid material voltage magnitude')
        if magnitude == 0.:
            return 0.
        y = math.log(magnitude)
        if y > self.lv[-1] or not self.lg[0] <= log_gamma <= self.lg[-1]:
            raise DomainFailure('Negative material query outside original table domain')
        if y < self.lv[0]:
            raise ResolutionFailure('Negative material local voltage below table resolution')
        ix = min(max(int(np.searchsorted(self.lg, log_gamma)-1), 0), len(self.lg)-2)
        iy = min(max(int(np.searchsorted(self.lv, y)-1), 0), len(self.lv)-2)
        a = (log_gamma-self.lg[ix])/(self.lg[ix+1]-self.lg[ix])
        c = (y-self.lv[iy])/(self.lv[iy+1]-self.lv[iy])
        li = self.li_negative
        return math.exp((1-a)*((1-c)*li[ix, iy]+c*li[ix, iy+1])
                       +a*((1-c)*li[ix+1, iy]+c*li[ix+1, iy+1])+self.base.log_scale)

    @staticmethod
    def _point(command, current, local, transistor, kcl):
        return dict(command_voltage_v=command, current_a=current, local_voltage_v=local,
                    transistor_voltage_v=transistor, KCL_residual_a=kcl,
                    KVL_residual_v=command-local-transistor)

    def _negative_bracket(self, magnitude, state_u):
        log_gamma = math.log(self.gamma(state_u))
        if not math.isfinite(magnitude) or magnitude <= 0.:
            raise DomainFailure('Invalid negative command magnitude')
        if math.log(magnitude) > self.lv[-1]:
            raise DomainFailure('Negative command beyond original material table')
        if math.log(magnitude) < self.lv[0]:
            raise ResolutionFailure('Negative command below table resolution')
        tx, ti = self.negative_tx, self.negative_ti
        def residual(transistor):
            if not 0. <= transistor <= tx[-1]:
                raise DomainFailure('Reverse transistor trial outside actual lookup domain')
            return float(np.interp(transistor, tx, ti))-self._material_abs(magnitude-transistor, log_gamma, -1)
        material_top = magnitude-self.vmin
        material_limited = material_top <= tx[-1]
        top = min(material_top, float(tx[-1]))
        if top > 0. and magnitude-top < self.vmin:
            top = math.nextafter(top, -math.inf)
        low_residual = residual(0.)
        if low_residual > 0.:
            raise DomainFailure('Reverse root requires negative transistor-drop magnitude')
        if top <= 0.:
            if low_residual == 0.:
                return log_gamma, residual, 0.
            raise ResolutionFailure('Reverse root at command=vmin needs subtable local voltage')
        if magnitude-top < self.vmin:
            raise ResolutionFailure('No representable in-table reverse root bracket')
        if residual(top) < 0.:
            if material_limited:
                raise ResolutionFailure('Reverse root requires subtable local voltage')
            raise DomainFailure('Reverse root requires transistor voltage beyond actual lookup')
        return log_gamma, residual, top

    def _solve_signed(self, command, state_u, mode):
        self.gamma(state_u)
        if not math.isfinite(command):
            raise DomainFailure('Nonfinite signed command voltage')
        if command >= 0.:
            try:
                current, local, kcl = self.base.circuit(command, state_u, mode=mode)
            except self.frozen_module.DomainFailure as exc:
                raise DomainFailure(str(exc)) from exc
            except self.frozen_module.ResolutionFailure as exc:
                raise ResolutionFailure(str(exc)) from exc
            return self._point(command, current, local, command-local, kcl)
        magnitude = -command
        log_gamma, residual, top = self._negative_bracket(magnitude, state_u)
        if mode == 'fast':
            transistor = 0. if top == 0. else brentq(residual, 0., top, xtol=1e-15, rtol=1e-13, maxiter=100)
        elif mode == 'strict_bisection42':
            low, high = 0., top
            for _ in range(42):
                mid = .5*(low+high)
                if residual(mid) >= 0.:
                    high = mid
                else:
                    low = mid
            transistor = .5*(low+high)
        else:
            raise ValueError('Unknown explicit circuit mode')
        local = magnitude-transistor
        current = self._material_abs(local, log_gamma, -1)
        kcl = current-float(np.interp(transistor, self.negative_tx, self.negative_ti))
        return self._point(command, -current, -local, -transistor, -kcl)

    def signed_rate(self, state_u, point, b):
        """RHS algebra at an already solved actual signed point; no new root."""
        self.gamma(state_u)
        if not math.isfinite(b):
            raise ValueError('Explicit finite effective polarity b required')
        vh, current = point['local_voltage_v'], point['current_a']
        if not math.isfinite(vh) or not math.isfinite(current) or vh*current < 0.:
            raise DomainFailure('Nonfinite or nonpassive actual circuit point')
        if vh == 0.:
            if current != 0.:
                raise DomainFailure('Nonzero passive soft current at zero local voltage')
            return 0.
        p = self.parameters
        v, i = abs(vh), abs(current)
        if vh > 0.:
            value = p.kappa_per_v2*v*(math.exp(p.lambda_per_v*v-state_u)
                                      -i*math.expm1(state_u)/p.J_positive_a)
        else:
            value = p.kappa_per_v2*v*(math.exp(2*b+p.lambda_per_v*v-state_u)
                                      -(i/p.J_positive_a)*math.exp(-2*b)*math.expm1(state_u))
        if not math.isfinite(value):
            raise DomainFailure('Nonfinite signed m2 derivative')
        return value

    def smallfield_bound(self, sign, b):
        if sign not in (-1, 1) or not math.isfinite(b):
            raise ValueError('Explicit nonzero polarity and finite b required')
        p, delta = self.parameters, self.vmin
        recovery_shift, capture_shift = (0., 0.) if sign > 0 else (2*b, -2*b)
        return .5*p.kappa_per_v2*delta**2*(math.exp(recovery_shift+p.lambda_per_v*delta)
            +math.pi*self.gc/(self.ron*p.J_positive_a)*math.exp(capture_shift))

    def legacy_reference(self, command, state_u):
        if command >= 0.:
            try:
                current, local, kcl = self.base.legacy_reference(command, state_u)
            except self.frozen_module.DomainFailure as exc:
                raise DomainFailure(str(exc)) from exc
            except self.frozen_module.ResolutionFailure as exc:
                raise ResolutionFailure(str(exc)) from exc
            return self._point(command, current, local, command-local, kcl)
        magnitude, gamma = -float(command), self.gamma(state_u)
        self._negative_bracket(magnitude, state_u)
        low, high, log_gamma = 0., magnitude, math.log(gamma)
        for _ in range(42):
            mid = .5*(low+high)
            transistor = magnitude-mid
            if not 0. <= transistor <= self.negative_tx[-1]:
                raise DomainFailure('Legacy reverse trial exceeds actual transistor lookup')
            value = self._material_abs(mid, log_gamma, -1)-float(np.interp(transistor, self.negative_tx, self.negative_ti))
            if value > 0.:
                high = mid
            else:
                low = mid
        if low < self.vmin:
            raise ResolutionFailure('Legacy reverse would invoke its subtable safeguard')
        current, local = self.base.transport.series(magnitude, gamma, self.ron, phase='reset')
        current, local = float(current), float(local)
        self._material_abs(local, log_gamma, -1)
        transistor = magnitude-local
        if not 0. <= transistor <= self.negative_tx[-1]:
            raise DomainFailure('Legacy reverse solution beyond actual transistor lookup')
        kcl = current-float(np.interp(transistor, self.negative_tx, self.negative_ti))
        return self._point(command, -current, -local, -transistor, -kcl)

    def validate_circuit(self, points, *, check_budget=None):
        rows, failures = [], []
        points = list(points)
        def signed_passive(point):
            command, current = point['command_voltage_v'], point['current_a']
            local, transistor = point['local_voltage_v'], point['transistor_voltage_v']
            return bool(all(math.isfinite(value) for value in point.values())
                and command*current >= 0. and local*current >= 0. and transistor*current >= 0.
                and command*local >= 0. and command*transistor >= 0.)
        for command, state in points:
            if check_budget:
                check_budget()
            try:
                reference = self.legacy_reference(float(command), float(state))
            except Exception as exc:
                if isinstance(exc, TimeoutError):
                    raise
                failures.append(dict(command_v=command, state_u=state, mode='legacy_reference',
                                     error_type=type(exc).__name__, error=str(exc)))
                continue
            for mode in ('fast', 'strict_bisection42'):
                try:
                    value = self._solve_signed(float(command), float(state), mode)
                    error = (0. if command == 0. else abs(math.log10(abs(value['current_a']/reference['current_a']))))
                    rows.append(dict(command_v=command, state_u=state, mode=mode,
                        log_current_difference_decade=error,
                        local_voltage_difference_v=abs(value['local_voltage_v']-reference['local_voltage_v']),
                        KCL_residual_a=abs(value['KCL_residual_a']),
                        KVL_residual_v=abs(value['KVL_residual_v']),
                        signed_passive=signed_passive(value),
                        reference_signed_passive=signed_passive(reference), point=value))
                except Exception as exc:
                    if isinstance(exc, TimeoutError):
                        raise
                    failures.append(dict(command_v=command, state_u=state, mode=mode,
                                         error_type=type(exc).__name__, error=str(exc)))
        report = dict(n_reference_points=len(points), rows=rows, failures=failures, modes={})
        for mode in ('fast', 'strict_bisection42'):
            selected = [row for row in rows if row['mode'] == mode]
            maxima = {key: max((row[key] for row in selected), default=None) for key in
                      ('log_current_difference_decade', 'local_voltage_difference_v',
                       'KCL_residual_a', 'KVL_residual_v')}
            bad = any(row['mode'] in ('legacy_reference', mode) for row in failures)
            passed = bool(len(selected) == len(points) and len(points) > 0 and not bad
                and all(row['signed_passive'] and row['reference_signed_passive'] for row in selected)
                and maxima['log_current_difference_decade'] <= 5e-7
                and maxima['local_voltage_difference_v'] <= 1e-9 and maxima['KCL_residual_a'] <= 1e-13
                and maxima['KVL_residual_v'] <= 1e-12)
            report['modes'][mode] = dict(n_points=len(selected), passed=passed, **maxima)
        return report

    def simulate_phase(self, command_voltage, circuit, initial_u, b, precision, *,
                       initial_command_voltage_v, observed_current_a=None, scoring_mask=None,
                       check_budget=None, collect_rhs=False):
        """Integrate one actual waveform from its actual incoming state, with no anchor."""
        if not isinstance(circuit, SignedSeriesCircuit) or circuit.kernel is not self:
            raise TypeError('Use a sign-correct circuit from this frozen material kernel')
        if not isinstance(precision, Precision) or not math.isfinite(b):
            raise ValueError('Explicit precision and finite b are required')
        command = np.asarray(command_voltage, dtype=float)
        if command.ndim != 1 or not len(command) or not np.all(np.isfinite(command)) or not math.isfinite(initial_command_voltage_v):
            raise ValueError('Actual waveform and previous command must be finite')
        if (observed_current_a is None) != (scoring_mask is None):
            raise ValueError('Supply observations and their original mask together, or neither')
        observed, mask = None, None
        if observed_current_a is not None:
            observed, mask = np.asarray(observed_current_a, dtype=float), np.asarray(scoring_mask, dtype=bool)
            if observed.shape != command.shape or mask.shape != command.shape:
                raise ValueError('Observation/mask shapes differ from native waveform')
            if np.any(mask & (~np.isfinite(observed) | (abs(observed) <= 0.) | (abs(command) <= 1e-12))):
                raise ValueError('Invalid supplied original scoring mask')
        summary = dict(status='running', parameters=asdict(self.parameters), b=float(b),
            precision=asdict(precision), circuit_mode=circuit.mode, fixed_recovery_order=2, parallel_path_a_s=0.,
            initial_u=float(initial_u), initial_command_voltage_v=float(initial_command_voltage_v),
            n_native=len(command), segments=[], root_failures=[], last_rhs={},
            no_conditioned_anchor=True, no_hidden_state_reset=True,
            clock='Cumulative actual command-voltage distance; duplicate endpoints add no dwell; no seconds inferred')
        arrays = dict(command_voltage_v=command.copy())
        residual = np.empty(0)
        rhs_rows, omissions = [], []
        rhs_calls, rhs_kcl = 0, 0.
        current_segment = None
        unique, inverse, segments = _path_clock(command, initial_command_voltage_v)
        states_unique = np.full(len(unique), np.nan)
        arrays['path_voltage_s'] = unique[inverse][1:]
        if observed is not None:
            arrays.update(observed_current_a=observed.copy(), original_scoring_mask=mask.copy())

        def solve(command_value, state):
            if check_budget:
                check_budget()
            try:
                return circuit.solve(command_value, state)
            except Exception as exc:
                summary['root_failures'].append(dict(segment=current_segment, command_v=float(command_value),
                    state_u=float(state), error_type=type(exc).__name__, error=str(exc)))
                raise

        try:
            self.gamma(float(initial_u))
            states_unique[0] = float(initial_u)
            present_signs = {1 if value > 0 else -1 for value in np.r_[initial_command_voltage_v, command] if value != 0.}
            bounds = {sign: self.smallfield_bound(sign, b) for sign in present_signs}
            summary['smallfield_bounds_by_sign'] = {str(sign): value for sign, value in bounds.items()}
            if any(not math.isfinite(value) or value >= 1e-12 for value in bounds.values()):
                raise ResolutionFailure('Required signed near-zero omission bound is not below1e-12')
            state = float(initial_u)
            for index, (start, stop, va, vb) in enumerate(segments):
                if check_budget:
                    check_budget()
                current_segment = index
                sign = 1 if (va if va != 0. else vb) > 0. else -1
                row = dict(index=index, path_start_v=start, path_stop_v=stop,
                    command_start_v=va, command_stop_v=vb, sign=sign, initial_u=state,
                    status='running', full_zero_microsegment=(min(abs(va), abs(vb)) == 0. and max(abs(va), abs(vb)) >= self.vmin))
                summary['segments'].append(row)
                def actual_command(coordinate):
                    fraction = (coordinate-start)/(stop-start)
                    if not 0. <= fraction <= 1.:
                        raise DomainFailure('Solver coordinate outside the actual command segment')
                    return (1.-fraction)*va+fraction*vb  # Convex interpolation, not clipping.
                def rhs(coordinate, y):
                    nonlocal rhs_calls, rhs_kcl
                    if check_budget:
                        check_budget()
                    proposed, applied = float(y[0]), float(actual_command(coordinate))
                    rhs_calls += 1
                    summary['last_rhs'] = dict(segment=index, coordinate=coordinate, command_v=applied, state_u=proposed)
                    self.gamma(proposed)
                    if applied == 0.:
                        point = self._point(0., 0., 0., 0., 0.)
                        rate = 0.
                    elif abs(applied) < self.vmin:
                        actual_sign = 1 if applied > 0 else -1
                        if actual_sign != sign:
                            raise DomainFailure('Unexpected sign crossing inside a sign-constant segment')
                        omissions.append(dict(segment=index, sign=sign, command_v=applied, state_u=proposed))
                        if collect_rhs:
                            rhs_rows.append([applied, proposed, np.nan, np.nan, np.nan, 0., np.nan])
                        return [0.]
                    else:
                        point = solve(applied, proposed)
                        rhs_kcl = max(rhs_kcl, abs(point['KCL_residual_a']))
                        rate = self.signed_rate(proposed, point, b)
                    if collect_rhs:
                        rhs_rows.append([applied, proposed, point['current_a'], point['local_voltage_v'],
                                         point['transistor_voltage_v'], rate, point['KCL_residual_a']])
                    return [rate]
                solution = solve_ivp(rhs, (start, stop), [state], method='DOP853',
                    rtol=precision.rtol, atol=precision.atol, max_step=precision.max_step, dense_output=True)
                row.update(status='completed' if solution.success else 'solver_failed', success=bool(solution.success),
                    message=str(solution.message), nfev=int(solution.nfev), accepted_steps=len(solution.t)-1,
                    final_u=float(solution.y[0, -1]))
                arrays[f'segment_{index}_coordinate'] = solution.t
                arrays[f'segment_{index}_state_u'] = solution.y[0]
                arrays[f'segment_{index}_state_q'] = np.exp(-solution.y[0])
                if not solution.success:
                    raise RuntimeError(solution.message)
                selected = (unique >= start) & (unique <= stop)
                states_unique[selected] = solution.sol(unique[selected])[0]
                state = float(solution.y[0, -1])
                self.gamma(state)
            native_u = states_unique[inverse][1:]
            if not np.all(np.isfinite(native_u)):
                raise DomainFailure('Missing native state after actual waveform integration')
            current_segment = 'native_scoring'
            points = [solve(float(applied), float(value)) for applied, value in zip(command, native_u)]
            for key in ('current_a', 'local_voltage_v', 'transistor_voltage_v', 'KCL_residual_a', 'KVL_residual_v'):
                arrays[key] = np.array([point[key] for point in points])
            arrays.update(state_u=native_u, state_q=np.exp(-native_u))
            if np.any(arrays['current_a']*command < 0.):
                raise DomainFailure('Nonpassive signed native output')
            if observed is not None:
                if np.any(~np.isfinite(arrays['current_a'][mask])) or np.any(abs(arrays['current_a'][mask]) <= 0.):
                    raise DomainFailure('Invalid model current on original eligible mask')
                residual = np.log10(abs(arrays['current_a'][mask])/abs(observed[mask]))
                errors = np.full(len(command), np.nan)
                errors[mask] = residual
                arrays['log10_current_residual'] = errors
                groups = dict(all_eligible=mask, low_abs_le_0p2=mask & (abs(command) <= .2+1e-12))
                if len(present_signs) == 1:
                    outward = np.arange(len(command)) <= np.argmax(abs(command))
                    groups.update(outward=mask & outward, return_=mask & ~outward)
                summary['metrics'] = {('return' if name == 'return_' else name):
                    dict(n=int(use.sum()), RMSE_decade=float(np.sqrt(np.mean(errors[use]**2))),
                         mean_signed_log_error_decade=float(np.mean(errors[use])),
                         median_absolute_factor=float(10**np.median(abs(errors[use]))))
                    for name, use in groups.items() if np.any(use)}
            joins = [abs(a['final_u']-z['initial_u']) for a, z in zip(summary['segments'][:-1], summary['segments'][1:])]
            summary.update(status='completed', final_u=state, terminal_u=state,
                final_command_voltage_v=float(command[-1]), max_segment_state_join_error=max(joins, default=0.),
                max_native_KCL_residual_a=float(max(abs(arrays['KCL_residual_a']), default=0.)),
                max_native_KVL_residual_v=float(max(abs(arrays['KVL_residual_v']), default=0.)),
                n_scored=int(mask.sum()) if mask is not None else 0)
        except Exception as exc:
            category = ('numerical_domain_unresolved' if isinstance(exc, DomainFailure) else
                'numerical_resolution_unresolved' if isinstance(exc, ResolutionFailure) else
                'caller_budget_exhausted' if isinstance(exc, TimeoutError) else 'solver_or_audit_failure')
            summary.update(status='failed_preserved_for_caller', error_type=type(exc).__name__, error=str(exc),
                           failure_scope=category, physical_rejection=False)
            if summary['segments'] and summary['segments'][-1]['status'] == 'running':
                summary['segments'][-1]['status'] = 'exception_before_completion'
            raise PhaseFailure(str(exc), summary, arrays) from exc
        finally:
            counted = {row['index']: row['sign'] for row in summary['segments']
                       if row['status'] == 'completed' and row['full_zero_microsegment']}
            counted.update({row['segment']: row['sign'] for row in omissions})
            summary.update(n_rhs_calls=rhs_calls, max_rhs_KCL_residual_a=rhs_kcl,
                smallfield_omission=dict(n_RHS_omissions=len(omissions), physical_microsegment_indices=sorted(counted),
                    local_bound_sum=sum(summary.get('smallfield_bounds_by_sign', {}).get(str(sign), 0.) for sign in counted.values()),
                    exact_inactive_claim=False, evaluations=omissions,
                    scope='At most once per physical monotone microsegment; partial omitted segments counted conservatively. '
                          'Local omitted integral only, not a full-trajectory error certificate.'))
            arrays['partial_unique_path_v'] = unique
            arrays['partial_unique_state_u'] = states_unique
            if collect_rhs:
                arrays['rhs_trace'] = np.asarray(rhs_rows).reshape((-1, 7))
                summary['rhs_trace_columns'] = ['command_v', 'state_u', 'current_a', 'local_voltage_v',
                    'transistor_voltage_v', 'signed_u_derivative_per_v', 'KCL_residual_a']
        return dict(summary=summary, arrays=arrays, residual=residual)

    def audit_native_legacy(self, result, *, check_budget=None):
        arrays = result['arrays']
        commands, states = arrays['command_voltage_v'], arrays['state_u']
        rows, failures = [], []
        for index, (command, state) in enumerate(zip(commands, states)):
            if check_budget:
                check_budget()
            try:
                point = self.legacy_reference(float(command), float(state))
                rows.append([index, point['current_a'], point['local_voltage_v'],
                             point['transistor_voltage_v'], point['KCL_residual_a']])
            except Exception as exc:
                if isinstance(exc, TimeoutError):
                    raise
                failures.append(dict(native_index=index, command_v=float(command),
                                     error_type=type(exc).__name__, error=str(exc)))
        report = dict(n_native_points=len(commands), completed_points=len(rows), failures=failures)
        if len(rows) == len(commands) and not failures:
            values = np.asarray(rows)[:, 1:]
            use = arrays.get('original_scoring_mask', commands != 0.)
            report.update(max_log_current_difference_decade=float(max(abs(np.log10(
                abs(arrays['current_a'][use]/values[use, 0]))), default=0.)),
                max_local_voltage_difference_v=float(max(abs(arrays['local_voltage_v']-values[:, 1]), default=0.)),
                max_reference_KCL_residual_a=float(max(abs(values[:, 3]), default=0.)))
        return report, np.asarray(rows).reshape((-1, 5))
