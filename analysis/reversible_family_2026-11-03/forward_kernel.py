"""Reusable conditional cycle-4 forward kernel; import runs no experiment.

lambda/J/kappa and solver precision are explicit inputs. No seed selection,
optimizer, timer, output writer, or automatic retry lives in this module.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import math
import sys
from typing import Callable

sys.dont_write_bytecode = True
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path('/private/tmp/vulcan-runtime-cache')


class DomainFailure(ValueError):
    pass


class ResolutionFailure(ValueError):
    pass


class KernelFailure(RuntimeError):
    """Caller must preserve summary and partial arrays; never a physical rejection."""
    def __init__(self, message, summary, arrays):
        super().__init__(message)
        self.summary = summary
        self.arrays = arrays


@dataclass(frozen=True)
class Parameters:
    lambda_per_v: float
    J_positive_a: float
    kappa_per_v2: float

    def __post_init__(self):
        values = (self.lambda_per_v, self.J_positive_a, self.kappa_per_v2)
        if any(not math.isfinite(float(x)) for x in values):
            raise ValueError('All three physical parameters must be finite')
        if self.lambda_per_v < 0. or self.J_positive_a <= 0. or self.kappa_per_v2 < 0.:
            raise ValueError('Require lambda>=0, J>0 and kappa>=0; no parameter clipping')


@dataclass(frozen=True)
class Precision:
    name: str
    rtol: float
    atol: float
    max_step: float

    def __post_init__(self):
        if not self.name or any(not math.isfinite(float(x)) or x <= 0.
                                for x in (self.rtol, self.atol, self.max_step)):
            raise ValueError('An explicit named positive finite solver precision is required')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ForwardKernel:
    """Load once, evaluate explicit parameter triples after root predeclaration."""

    @classmethod
    def load(cls, cache=CACHE):
        """Read fixed data/transport only; does not solve a root or integrate an ODE."""
        cache = Path(cache)
        sys.path[:0] = [str(ROOT), str(ROOT/'analysis/mechanism_selection_2026-11-03')]
        from SOFT_FEEDBACK_LOW_BIAS_BUDGET import cached_transport
        self = cls()
        self.transport, lookup_paths = cached_transport(cache)
        t = self.transport
        expected = {
            'native_observations.npz': '0aedbe51f41b4f4355e1de4d26048c6790a9d454b8b7d56f51bd76a4671465a8',
            'transport_g010_v0025.npz': '00d6d9ea0b1dfe855181e7016afc6dd5e23d169f83f19cfcedfc9d3b35f9e4e2',
            'fig2b_transistor_lookup.json': '1c9ecb7073f74e3db9f1c4a041d17630c652b1a12e4f33ff7388aaf5f0b16f21',
            'bookoutput2_reverse_transistor_lookup.json': '28462c3dcb7e0184199635de72a4c5fb1832a9f6281f08900ede28198dfe1359',
        }
        for name, digest in expected.items():
            if sha(cache/name) != digest:
                raise ValueError(f'Frozen runtime-cache input changed: {name}')
        previous = ROOT/'analysis/reset_shape_2026-11-03'
        base_path = previous/'soft_field_mobility_probe.json'
        base = json.loads(base_path.read_text())
        budget_path = previous/'soft_beta_relaxed_budget.json'
        budget = json.loads(budget_path.read_text())
        self.ron = float(base['plan']['Ron_ohm'])
        self.gc = float(base['plan']['gamma_c_ev'])
        self.anchor_u = float(budget['anchors_u']['initial_0p2'])
        self.legacy_check_states = (self.anchor_u, budget['anchors_u']['outward_2V_lower'],
            budget['anchors_u']['return_5V_lower'], budget['anchors_u']['return_2V_upper'], 15., 25.)
        with np.load(cache/'native_observations.npz') as archive:
            self.voltage = archive['1t1r_cycle_4_set_voltage'].copy()
            self.observed_signed = archive['1t1r_cycle_4_set_current'].copy()
        with np.load(base_path.with_suffix('.npz')) as archive:
            np.testing.assert_array_equal(self.voltage, archive['command_voltage_v'])
            np.testing.assert_array_equal(self.observed_signed, archive['observed_current_a'])
            self.path = archive['path_voltage_s'].copy()
            self.mask = archive['original_scoring_mask'].copy()
        self.observed_abs = abs(self.observed_signed)
        self.outward = np.arange(len(self.voltage)) <= np.argmax(self.voltage)
        np.testing.assert_array_equal(self.mask, np.isfinite(self.observed_abs)
            & (self.observed_abs >= 1e-11) & (self.voltage > 1e-12))
        if (len(self.voltage), int(self.mask.sum()), int((self.mask & self.outward).sum()),
                int((self.mask & ~self.outward).sum())) != (503, 500, 250, 250):
            raise ValueError('Native cycle-4 sample identity/count changed')
        np.testing.assert_allclose(self.path[self.outward], self.voltage[self.outward], rtol=0., atol=1e-12)
        np.testing.assert_allclose(self.path[~self.outward], 10.-self.voltage[~self.outward], rtol=0., atol=1e-12)
        anchors_path = cache/'RON_SCALING_REVIEW.json'
        self.anchors = next(row for row in json.loads(anchors_path.read_text())['rows']
                            if row['Ron_ohm'] == self.ron)['anchor_reinversions']
        self.references = next(row for row in base['runs']
            if row['precision']['name'] == 'refined' and 'metrics' in row)['reference_features']
        self.lg, self.lv, self.li = t.lg, t.lv, t.li[0]
        self.tx, self.ti = t.knots['set']
        self.vmin = math.exp(self.lv[0])
        self.log_scale = math.log(t.r0/self.ron)
        sources = [cache/name for name in expected] + [base_path, base_path.with_suffix('.npz'),
            budget_path, anchors_path, Path(__file__),
            ROOT/'analysis/mechanism_selection_2026-11-03/SOFT_FEEDBACK_LOW_BIAS_BUDGET.py',
            ROOT/'analysis/ron_scope_2026-11-03/ron_scaling_review.py',
            ROOT/'vulcan2d/transistor.py']
        self.source_sha256 = {str(path): sha(path) for path in sources}
        for array in (self.voltage, self.observed_signed, self.observed_abs, self.path,
                      self.mask, self.outward, self.lg, self.lv, self.li, self.tx, self.ti):
            array.flags.writeable = False
        return self

    def gamma(self, u):
        if not math.isfinite(u) or not 0. < u <= 80.:
            raise DomainFailure(f'State outside finite table interface 0<u<=80: {u!r}')
        result = self.gc/math.expm1(u)
        if not self.lg[0] <= math.log(result) <= self.lg[-1]:
            raise DomainFailure(f'Gamma outside existing table: {result!r}')
        return result

    def material(self, voltage, log_gamma):
        if not math.isfinite(voltage):
            raise DomainFailure('Nonfinite material voltage')
        if voltage == 0.:
            return 0.
        if voltage < 0. or math.log(voltage) > self.lv[-1]:
            raise DomainFailure('Material voltage outside existing positive table')
        y = math.log(voltage)
        if y < self.lv[0]:
            raise ResolutionFailure('Positive material voltage below table resolution')
        if not self.lg[0] <= log_gamma <= self.lg[-1]:
            raise DomainFailure('log Gamma outside existing table')
        ix = min(max(int(np.searchsorted(self.lg, log_gamma)-1), 0), len(self.lg)-2)
        iy = min(max(int(np.searchsorted(self.lv, y)-1), 0), len(self.lv)-2)
        a = (log_gamma-self.lg[ix])/(self.lg[ix+1]-self.lg[ix])
        b = (y-self.lv[iy])/(self.lv[iy+1]-self.lv[iy])
        return math.exp((1-a)*((1-b)*self.li[ix, iy]+b*self.li[ix, iy+1])
                       +a*((1-b)*self.li[ix+1, iy]+b*self.li[ix+1, iy+1])+self.log_scale)

    def _bracket(self, command, u):
        log_gamma = math.log(self.gamma(float(u)))
        command = float(command)
        if not math.isfinite(command) or command < 0.:
            raise DomainFailure('Invalid command voltage')
        if command == 0.:
            return None
        if math.log(command) > self.lv[-1]:
            raise DomainFailure('Command beyond original table domain')
        if math.log(command) < self.lv[0]:
            raise ResolutionFailure('Positive command below table resolution')
        def residual(vt):
            return float(np.interp(vt, self.tx, self.ti))-self.material(command-vt, log_gamma)
        material_top = command-self.vmin
        material_limited = material_top <= self.tx[-1]
        top = min(material_top, float(self.tx[-1]))
        if top > 0. and command-top < self.vmin:
            top = math.nextafter(top, -math.inf)
        low_value = residual(0.)
        if low_value > 0.:
            raise DomainFailure('KCL root requires a negative transistor drop')
        if top <= 0.:
            if low_value == 0.:
                return log_gamma, residual, 0.
            raise ResolutionFailure('KCL root at command=vmin requires subtable local voltage')
        if command-top < self.vmin:
            raise ResolutionFailure('No representable in-table root bracket')
        if residual(top) < 0.:
            if material_limited:
                raise ResolutionFailure('KCL root requires subtable local voltage')
            raise DomainFailure('KCL root requires transistor drop beyond its table')
        return log_gamma, residual, top

    def circuit(self, command, u, *, mode):
        """Both modes evaluate only actual in-table currents; no silent fallback."""
        bracket = self._bracket(command, u)
        if bracket is None:
            return 0., 0., 0.
        log_gamma, residual, top = bracket
        if mode == 'fast':
            vt = 0. if top == 0. else brentq(residual, 0., top, xtol=1e-15, rtol=1e-13, maxiter=100)
        elif mode == 'strict_bisection42':
            low, high = 0., top
            for _ in range(42):
                middle = .5*(low+high)
                if residual(middle) >= 0.:
                    high = middle
                else:
                    low = middle
            vt = .5*(low+high)
        else:
            raise ValueError('Explicit circuit mode must be fast or strict_bisection42')
        vh = float(command)-vt
        current = self.material(vh, log_gamma)
        return current, vh, current-float(np.interp(vt, self.tx, self.ti))

    def legacy_reference(self, command, u):
        """Original42 reference for fixed/native audits, never used by the ODE.

        The original implementation contains a subtable safeguard. Verify its
        entire original bisection path remains in-domain before calling it;
        otherwise report unresolved and do not invoke that implementation.
        """
        g = self.gamma(float(u))
        if command == 0.:
            return 0., 0., 0.
        self._bracket(command, u)
        low, high = 0., float(command)
        log_gamma = math.log(g)
        for _ in range(42):
            middle = .5*(low+high)
            if not 0. <= command-middle <= self.tx[-1]:
                raise DomainFailure('Original42 reference trial outside the transistor table')
            value = self.material(middle, log_gamma)-float(np.interp(command-middle, self.tx, self.ti))
            if value > 0.:
                high = middle
            else:
                low = middle
        if low < self.vmin:
            raise ResolutionFailure('Original42 reference would evaluate its subtable safeguard')
        current, vh = self.transport.series(float(command), g, self.ron)
        current, vh = float(current), float(vh)
        self.material(vh, log_gamma)
        if not 0. <= command-vh <= self.tx[-1]:
            raise DomainFailure('Original42 reference outside transistor table')
        return current, vh, current-float(np.interp(command-vh, self.tx, self.ti))

    def validate_circuit(self, *, check_budget: Callable[[], None] | None = None):
        """Explicit optional36-point audit; caller saves this before choosing mode."""
        rows, failures = [], []
        for command in (.02, .1, .2, 1., 2., 5.):
            for state in self.legacy_check_states:
                if check_budget:
                    check_budget()
                try:
                    reference = self.legacy_reference(command, state)
                    for mode in ('fast', 'strict_bisection42'):
                        value = self.circuit(command, state, mode=mode)
                        rows.append(dict(command_v=command, state_u=state, mode=mode,
                            log_current_difference_decade=abs(math.log10(value[0]/reference[0])),
                            local_voltage_difference_v=abs(value[1]-reference[1]), KCL_residual_a=abs(value[2])))
                except Exception as exc:
                    if isinstance(exc, TimeoutError):
                        raise
                    failures.append(dict(command_v=command, state_u=state, error_type=type(exc).__name__, error=str(exc)))
        summary = dict(n_reference_points=36, failures=failures, rows=rows, modes={})
        for mode in ('fast', 'strict_bisection42'):
            selected = [row for row in rows if row['mode'] == mode]
            maxima = {key: max((row[key] for row in selected), default=None) for key in
                      ('log_current_difference_decade', 'local_voltage_difference_v', 'KCL_residual_a')}
            passed = bool(len(selected) == 36 and not failures and maxima['log_current_difference_decade'] <= 5e-7
                and maxima['local_voltage_difference_v'] <= 1e-9 and maxima['KCL_residual_a'] <= 1e-13)
            summary['modes'][mode] = dict(n_points=len(selected), passed=passed, **maxima)
        return summary

    def evaluate(self, parameters: Parameters, precision: Precision, *, circuit_mode: str,
                 check_budget: Callable[[], None] | None = None, collect_rhs=False):
        """One explicit triple/precision; caller budgets, saves failures and selects trials.

        Returns {summary, arrays, residual}; residual has all original500 values
        in native order. A KernelFailure carries the partial summary/arrays.
        """
        if not isinstance(parameters, Parameters) or not isinstance(precision, Precision):
            raise TypeError('Use explicit Parameters and Precision objects')
        if circuit_mode not in ('fast', 'strict_bisection42'):
            raise ValueError('Choose an explicitly validated circuit mode')
        lam, js, kappa = parameters.lambda_per_v, parameters.J_positive_a, parameters.kappa_per_v2
        summary = dict(status='running', parameters=asdict(parameters), precision=asdict(precision),
            circuit_mode=circuit_mode, segments=[], root_failures=[], last_rhs={},
            conditional_anchor_u=self.anchor_u, parameter_count=3,
            initialization='Anchor .2V ->5V->return0 plus independent same-ODE backward .2V->0; '
                           'one continuous conditional solution, no initial-state fit or reset.',
            scope='Single cycle4 conditional SET; no holdout,RESET,bare history or aggregate validation')
        arrays, rhs_rows, omissions = {}, [], []
        segment = None
        rhs_calls, rhs_kcl = 0, 0.

        def solve_circuit(command, state):
            if check_budget:
                check_budget()
            try:
                return self.circuit(command, state, mode=circuit_mode)
            except Exception as exc:
                summary['root_failures'].append(dict(segment=segment, command_v=float(command),
                    state_u=float(state), error_type=type(exc).__name__, error=str(exc)))
                raise

        def rate(command, state):
            nonlocal rhs_calls, rhs_kcl
            if check_budget:
                check_budget()
            rhs_calls += 1
            summary['last_rhs'] = dict(segment=segment, command_v=float(command), state_u=float(state))
            self.gamma(state)
            if command == 0.:
                if collect_rhs:
                    rhs_rows.append([command, state, 0., 0., 0., 0.])
                return 0.
            if 0. < command < self.vmin:
                omissions.append(dict(segment=segment, command_v=float(command), state_u=float(state)))
                if collect_rhs:
                    rhs_rows.append([command, state, np.nan, np.nan, 0., np.nan])
                return 0.
            current, vh, kcl = solve_circuit(command, state)
            rhs_kcl = max(rhs_kcl, abs(kcl))
            derivative = kappa*vh*(math.exp(lam*vh)-current*math.expm1(state)/js)
            if not math.isfinite(derivative):
                raise DomainFailure('Nonfinite reversible derivative')
            if collect_rhs:
                rhs_rows.append([command, state, current, vh, derivative, kcl])
            return derivative

        def integrate(name, span, initial_u, command):
            nonlocal segment
            segment = name
            row = dict(name=name, span=list(span), initial_u=float(initial_u), status='running')
            summary['segments'].append(row)
            solution = solve_ivp(lambda coordinate, y: [rate(command(coordinate), float(y[0]))],
                span, [initial_u], method='DOP853', rtol=precision.rtol, atol=precision.atol,
                max_step=precision.max_step, dense_output=True)
            row.update(status='completed' if solution.success else 'solver_failed', success=bool(solution.success),
                message=str(solution.message), nfev=int(solution.nfev), accepted_steps=len(solution.t)-1,
                final_u=float(solution.y[0, -1]))
            arrays[name+'_coordinate'] = solution.t
            arrays[name+'_state_u'] = solution.y[0]
            arrays[name+'_state_q'] = np.exp(-solution.y[0])
            if not solution.success:
                raise RuntimeError(solution.message)
            return solution

        try:
            bound = .5*kappa*self.vmin**2*(math.exp(lam*self.vmin)+math.pi*self.gc/(self.ron*js))
            summary['smallfield_bound'] = dict(value_per_microsegment=bound, vmin_v=self.vmin,
                scope='Local Lorentz-area omitted integral, not exact inactivity or a global error bound')
            if not math.isfinite(bound) or bound >= 1e-12:
                raise ResolutionFailure('Subtable microsegment bound is not below1e-12')
            outward = integrate('outward_anchor_to_apex', (.2, 5.), self.anchor_u, lambda x: x)
            returned = integrate('return_apex_to_zero', (5., 10.), float(outward.y[0, -1]), lambda x: 10.-x)
            backward = integrate('backward_anchor_to_zero', (.2, 0.), self.anchor_u, lambda x: x)
            v, out, path = self.voltage, self.outward, self.path
            low = out & (v <= .2)
            high = out & ~low
            states = np.empty_like(v)
            states[low] = backward.sol(path[low])[0]
            states[high] = outward.sol(path[high])[0]
            states[~out] = returned.sol(path[~out])[0]
            segment = 'native_scoring'
            values = np.asarray([solve_circuit(command, state) for command, state in zip(v, states)])
            current, local_v, kcl = values.T
            if not (np.all(np.isfinite(current)) and np.all(current >= 0.) and np.all(current[self.mask] > 0.)):
                raise DomainFailure('Invalid native current')
            residual = np.log10(current[self.mask]/self.observed_abs[self.mask])
            errors = np.full(len(v), np.nan)
            errors[self.mask] = residual
            groups = dict(all500=self.mask, outward250=self.mask & out, return250=self.mask & ~out,
                outward_low10=self.mask & out & (v <= .2+1e-12),
                both_branches_low20=self.mask & (v <= .2+1e-12), remaining480=self.mask & (v > .2+1e-12),
                outward0p5_to1p5_50=self.mask & out & (v > .5+1e-12) & (v <= 1.5+1e-12))
            metrics = {name: dict(n=int(use.sum()), RMSE_decade=float(np.sqrt(np.mean(errors[use]**2))),
                mean_signed_log_error_decade=float(np.mean(errors[use])),
                median_absolute_factor=float(10**np.median(abs(errors[use])))) for name, use in groups.items()}
            arrays.update(command_voltage_v=v.copy(), path_voltage_s=path.copy(),
                observed_current_a=self.observed_signed.copy(), original_scoring_mask=self.mask.copy(),
                state_u=states, state_q=np.exp(-states), current_a=current, local_voltage_v=local_v,
                KCL_residual_a=kcl, log10_current_residual=errors)
            summary.update(status='completed', metrics=metrics, n_scored=500,
                max_native_KCL_residual_a=float(max(abs(kcl))),
                derived_initial_u=float(backward.y[0, -1]), terminal_u=float(returned.y[0, -1]),
                apex_state_continuity_error=float(returned.y[0, 0]-outward.y[0, -1]),
                anchor_outward_join_error=float(backward.y[0, 0]-outward.y[0, 0]))
            self._diagnostic_features(summary, current, states)
        except Exception as exc:
            category = ('numerical_domain_unresolved' if isinstance(exc, DomainFailure) else
                        'numerical_resolution_unresolved' if isinstance(exc, ResolutionFailure) else
                        'caller_budget_exhausted' if isinstance(exc, TimeoutError) else 'solver_or_audit_failure')
            summary.update(status='failed_preserved_for_caller', error_type=type(exc).__name__,
                error=str(exc), failure_scope=category, physical_rejection=False)
            if summary['segments'] and summary['segments'][-1]['status'] == 'running':
                summary['segments'][-1]['status'] = 'exception_before_completion'
            raise KernelFailure(str(exc), summary, arrays) from exc
        finally:
            # Outward .2->5 has no near-zero crossing. Return and backward each have one.
            complete_micro = {row['name'] for row in summary['segments'] if row['status'] == 'completed'
                              and row['name'] in ('return_apex_to_zero', 'backward_anchor_to_zero')}
            physical_micro = sorted(complete_micro | {row['segment'] for row in omissions})
            summary.update(n_rhs_calls=rhs_calls, max_rhs_KCL_residual_a=rhs_kcl,
                smallfield_omission=dict(n_RHS_omissions=len(omissions), physical_microsegments=physical_micro,
                    local_bound_sum=len(physical_micro)*summary.get('smallfield_bound', {}).get('value_per_microsegment', 0.),
                    exact_inactive_claim=False, evaluations=omissions))
            if collect_rhs:
                arrays['rhs_trace'] = np.asarray(rhs_rows).reshape((-1, 6))
                summary['rhs_trace_columns'] = ['command_v', 'state_u', 'current_a', 'local_voltage_v',
                                                'signed_u_derivative_per_v', 'KCL_residual_a']
        return dict(summary=summary, arrays=arrays, residual=residual)

    def _diagnostic_features(self, summary, current, state):
        v, out = self.voltage, self.outward
        references = {}
        for name, reference in self.references.items():
            indices = reference['native_indices']
            value = (float(current[indices[0]]) if len(indices) == 1 else
                math.log(current[indices[1]]/current[indices[0]])/math.log(v[indices[1]]/v[indices[0]]))
            low, high = reference['original32_aggregate_band']
            references[name] = dict(native_indices=indices, single_cycle_prediction=value,
                original32_aggregate_band=[low, high], single_prediction_inside_reference_band=low <= value <= high)
        endpoints = []
        for name, command, branch, greater in (
            ('outward_2V_lower', 2., out, True), ('return_5V_lower', 5., ~out, True),
            ('return_2V_upper', 2., ~out, False)):
            index = int(np.argmin(np.where(branch, abs(v-command), np.inf)))
            limit = self.anchors[name]['current_a']
            endpoints.append(dict(name=name, native_index=index, predicted_current_a=float(current[index]),
                bound_current_a=limit, actual_u=float(state[index]),
                condition_satisfied=bool(current[index] >= limit if greater else current[index] <= limit)))
        crossings = []
        for threshold in (1e-6, 1e-5):
            entry = dict(threshold_a=threshold)
            for name, trace in (('observed', self.observed_abs), ('predicted', current)):
                hits = np.flatnonzero(out & self.mask & (trace >= threshold))
                entry[name] = dict(reached=False) if not len(hits) else dict(reached=True,
                    last_below_v=float(v[hits[0]-1]), first_at_or_above_v=float(v[hits[0]]))
            crossings.append(entry)
        summary.update(reference_features=references, actual_budget_endpoints=endpoints,
            outward_current_crossings=crossings,
            all_three_actual_endpoints_satisfied=all(row['condition_satisfied'] for row in endpoints),
            reference_scope='Single-cycle references, not original32-cycle aggregate gates')

    def audit_native_legacy(self, result, *, check_budget=None):
        """Optional explicit postfit audit; root decides budget and persists failures."""
        rows, failures = [], []
        arrays = result['arrays']
        for index, (command, state) in enumerate(zip(self.voltage, arrays['state_u'])):
            if check_budget:
                check_budget()
            try:
                value = self.legacy_reference(float(command), float(state))
                rows.append([index, *value])
            except Exception as exc:
                if isinstance(exc, TimeoutError):
                    raise
                failures.append(dict(native_index=index, error_type=type(exc).__name__, error=str(exc)))
        report = dict(n_native_points=len(self.voltage), completed_points=len(rows), failures=failures)
        if len(rows) == len(self.voltage) and not failures:
            values = np.asarray(rows)[:, 1:]
            report.update(max_log_current_difference_decade=float(max(abs(np.log10(
                arrays['current_a'][self.mask]/values[self.mask, 0])))),
                max_local_voltage_difference_v=float(max(abs(arrays['local_voltage_v']-values[:, 1]))),
                max_reference_KCL_residual_a=float(max(abs(values[:, 2]))))
        return report, np.asarray(rows).reshape((-1, 4))
