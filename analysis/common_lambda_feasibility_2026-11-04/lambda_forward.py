"""Prepared common-lambda phase forward, explicitly derived from joint_forward.

Frozen parent SHA256: c7e25af2ccafffcf05ebfdcb9d28a6bca63c799b4dde5e92d733b334cfca53db.
No experiment runs on import. The parent ODE/circuit integration is preserved.
Every candidate, including the old-lambda control, must declare shared
lambda/kappa/J/b, a finite nonnegative lambda domain, and source ID/SHA.
The caller owns history handoffs, method choice, numerical budgets and plans.
"""
from __future__ import annotations

from copy import copy
from dataclasses import asdict
from pathlib import Path
from types import MappingProxyType
import math
import sys

sys.dont_write_bytecode = True
import numpy as np
from scipy.integrate import DOP853, Radau

ROOT = Path(__file__).resolve().parents[2]
STATIC_STAGE = ROOT / 'analysis/zero_crossing_compatibility_2026-11-04'
SIGNED_STAGE = ROOT / 'analysis/signed_soft_history_2026-11-04'
for directory in (STATIC_STAGE, SIGNED_STAGE):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
from minimal_static_loader import load_minimal
from signed_forward_kernel import (Precision, PhaseFailure, DomainFailure,
                                   ResolutionFailure, _path_clock, sha)


FROZEN_PARENT = ROOT / 'analysis/joint_history_calibration_2026-11-04/joint_forward.py'
FROZEN_PARENT_SHA256 = 'c7e25af2ccafffcf05ebfdcb9d28a6bca63c799b4dde5e92d733b334cfca53db'


def _readout_arrays(kernel):
    return (kernel.lg, kernel.lv, kernel.base.li, kernel.li_negative,
            kernel.base.tx, kernel.base.ti, kernel.negative_tx, kernel.negative_ti)


def _readout_values(kernel):
    return {
        'Ron_ohm': kernel.ron, 'gamma_c_ev': kernel.gc,
        'base_Ron_ohm': kernel.base.ron, 'base_gamma_c_ev': kernel.base.gc,
        'base_log_scale': kernel.base.log_scale,
    }


def load_lambda_readout():
    """Load immutable readout only; with_parameters() is required before use.

    No roots, ODEs or observation arrays are evaluated here. The frozen
    parent hash is checked explicitly; no old guard or snapshot is modified.
    """
    if sha(FROZEN_PARENT) != FROZEN_PARENT_SHA256:
        raise ValueError('Frozen parent source hash changed')
    kernel = load_minimal()
    # Preserve the parent owner/view read-only policy without changing values.
    for array in _readout_arrays(kernel):
        array.flags.writeable = False
    kernel._lambda_readout_snapshot = MappingProxyType(_readout_values(kernel))
    kernel._lambda_readout_array_ids = tuple(id(array) for array in _readout_arrays(kernel))
    # This immutable historical reference controls only the unchanged first-step
    # rule. It is not the candidate's material-rate parameter snapshot.
    kernel._lambda_reference_parameters = MappingProxyType(asdict(kernel.parameters))
    kernel.source_sha256 = dict(kernel.source_sha256)
    kernel.source_sha256[str(FROZEN_PARENT)] = FROZEN_PARENT_SHA256
    kernel.source_sha256[str(Path(__file__).resolve())] = sha(Path(__file__))
    return kernel


def _check_readout(kernel):
    if not isinstance(getattr(kernel, '_lambda_readout_snapshot', None), MappingProxyType):
        raise TypeError('Use load_lambda_readout() before configuring a candidate')
    if _readout_values(kernel) != kernel._lambda_readout_snapshot:
        raise ValueError('Frozen Ron/Gamma_c/readout scaling changed')
    arrays = _readout_arrays(kernel)
    if tuple(id(array) for array in arrays) != kernel._lambda_readout_array_ids:
        raise ValueError('Frozen readout array objects changed')
    if any(array.flags.writeable for array in arrays):
        raise ValueError('Frozen readout arrays must remain read-only')
    if not isinstance(kernel._lambda_reference_parameters, MappingProxyType):
        raise TypeError('Immutable numerical reference snapshot is required')


def with_parameters(readout, *, lambda_per_v, lambda_domain_per_v,
                    kappa_per_v2, J_positive_a, b,
                    candidate_source_id, candidate_source_sha256):
    """Declare one shared candidate, including an explicit old-lambda control.

    lambda_domain_per_v is a finite inclusive (lower, upper) pair with
    0 <= lower <= upper. Provenance is caller supplied, not a file selector:
    source ID must be nonempty and SHA256 must be exactly 64 hexadecimal
    characters. The experiment manifest must bind it to the actual source.
    All snapshots contain only immutable scalar/tuple values.
    """
    _check_readout(readout)
    domain = tuple(float(value) for value in lambda_domain_per_v)
    if (len(domain) != 2 or not all(math.isfinite(value) for value in domain)
            or not 0. <= domain[0] <= domain[1]):
        raise ValueError('Declare a finite inclusive nonnegative lambda domain')
    lam, shared_b = float(lambda_per_v), float(b)
    if not math.isfinite(lam) or not domain[0] <= lam <= domain[1]:
        raise ValueError('Explicit lambda is outside its declared domain')
    if not math.isfinite(shared_b):
        raise ValueError('Explicit finite shared b is required')
    if not isinstance(candidate_source_id, str) or not candidate_source_id.strip():
        raise ValueError('Explicit nonempty candidate source ID is required')
    if (not isinstance(candidate_source_sha256, str) or len(candidate_source_sha256) != 64
            or any(character not in '0123456789abcdefABCDEF' for character in candidate_source_sha256)):
        raise ValueError('Explicit hexadecimal candidate SHA256 is required')
    candidate = copy(readout)
    candidate.parameters = readout.frozen_module.Parameters(
        lambda_per_v=lam, J_positive_a=float(J_positive_a), kappa_per_v2=float(kappa_per_v2))
    candidate._lambda_dynamics_snapshot = MappingProxyType({
        **asdict(candidate.parameters), 'b': shared_b,
        'lambda_domain_per_v': domain, 'candidate_source_id': candidate_source_id,
        'candidate_source_sha256': candidate_source_sha256.lower(),
    })
    _check_candidate(candidate, shared_b)
    return candidate


def _check_candidate(kernel, b):
    _check_readout(kernel)
    snapshot = getattr(kernel, '_lambda_dynamics_snapshot', None)
    if not isinstance(snapshot, MappingProxyType):
        raise TypeError('Every candidate, including old lambda, must pass with_parameters()')
    actual = {**asdict(kernel.parameters), 'b': float(b)}
    if any(actual[key] != snapshot[key] for key in actual):
        raise ValueError('Shared lambda/kappa/J/b changed between candidate declaration and phase')


def simulate_phase(kernel, data, u0, b, precision, method, check_budget,
                   collect_rhs=False):
    """Run one actual phase; preserve numerical failures as PhaseFailure.

    Required data keys: command_voltage_v, initial_command_voltage_v,
    nearzero_segment_budget (explicit positive bound, at most 1e-10),
    circuit_mode ('fast' or 'strict_bisection42'). Optional observations and
    their original scoring mask must be supplied together, as
    observed_current_a/scoring_mask. No state is derived from observations.
    Explicit methods: DOP853 or Radau. All physics uses the same signed kernel.
    """
    _check_candidate(kernel, b)
    if method not in ('DOP853', 'Radau'):
        raise ValueError('Choose explicit DOP853 or Radau; no fallback method')
    if not isinstance(precision, Precision):
        raise TypeError('Use an explicit signed-kernel Precision')
    if not math.isfinite(float(b)) or not math.isfinite(float(u0)):
        raise ValueError('Explicit finite b and incoming state are required')
    if check_budget is not None and not callable(check_budget):
        raise TypeError('check_budget must be callable or None')
    command = np.asarray(data['command_voltage_v'], dtype=float)
    initial_command = float(data['initial_command_voltage_v'])
    nearzero_budget = float(data['nearzero_segment_budget'])
    if not math.isfinite(nearzero_budget) or not 0. < nearzero_budget <= 1e-10:
        raise ValueError('Explicit nearzero_segment_budget must be positive and at most 1e-10')
    if command.ndim != 1 or not len(command) or not np.all(np.isfinite(command)) or not math.isfinite(initial_command):
        raise ValueError('Actual native waveform and incoming command must be finite')
    circuit = kernel.make_circuit(mode=data['circuit_mode'])
    observed = data.get('observed_current_a')
    mask = data.get('scoring_mask')
    if (observed is None) != (mask is None):
        raise ValueError('Supply observations and the original score mask together')
    if observed is not None:
        observed = np.asarray(observed, dtype=float)
        original_mask = np.asarray(mask)
        if original_mask.dtype != np.bool_:
            raise ValueError('Original scoring mask must have boolean dtype')
        mask = original_mask.copy()
        if observed.shape != command.shape or mask.shape != command.shape:
            raise ValueError('Native waveform/observation/mask shapes differ')
        if np.any(mask & (~np.isfinite(observed) | (abs(observed) <= 0.) | (abs(command) <= 1e-12))):
            raise ValueError('Invalid supplied original scoring mask')
    summary = {
        'status': 'running', 'parameters': asdict(kernel.parameters), 'b': float(b),
        'precision': asdict(precision), 'method': method, 'state_representation': 'direct_u',
        'circuit_mode': circuit.mode, 'fixed_recovery_order': 2, 'parallel_path_a_s': 0.,
        'fixed_readout': dict(kernel._lambda_readout_snapshot),
        'shared_dynamics': dict(kernel._lambda_dynamics_snapshot),
        'frozen_parent_source_sha256': FROZEN_PARENT_SHA256,
        'initial_u': float(u0), 'initial_command_voltage_v': initial_command,
        'n_native': len(command), 'segments': [], 'root_failures': [], 'state_domain_failures': [],
        'last_rhs': {}, 'no_conditioned_anchor': True, 'no_hidden_state_reset': True,
        'no_state_clipping': True, 'no_failed_phase_retry': True,
        'nearzero_segment_budget': nearzero_budget, 'rhs_trace_requested': bool(collect_rhs),
        'clock': 'Cumulative actual command-voltage distance; duplicate endpoints add no dwell; no physical time inferred',
        'first_step_rule': 'min(segment span, max_step, 1e-4 V / max(1,kappa/kappa_frozen)); numerical initialization only',
        'record_id': data.get('record_id'), 'phase': data.get('phase'),
    }
    arrays = {'command_voltage_v': command.copy()}
    if observed is not None:
        arrays.update(observed_current_a=observed.copy(), original_scoring_mask=mask.copy())
    unique, inverse, segments = _path_clock(command, initial_command)
    arrays['path_voltage_s'] = unique[inverse][1:]
    unique_u = np.full(len(unique), np.nan)
    residual = np.empty(0)
    rhs_rows, omissions, accepted = [], [], {}
    rhs_calls, rhs_kcl, rhs_kvl = 0, 0., 0.
    current_segment, active_solver = None, None

    def check():
        if check_budget is not None:
            check_budget()

    def valid_state(value, location):
        try:
            return kernel.gamma(float(value))
        except Exception as exc:
            summary['state_domain_failures'].append({
                'segment': current_segment, 'location': location, 'state_u': float(value),
                'error_type': type(exc).__name__, 'error': str(exc),
                'interpretation': 'Numerical trial/domain unresolved; not a physical rejection'})
            raise

    def solve(applied, state):
        check()
        try:
            return circuit.solve(float(applied), float(state))
        except Exception as exc:
            summary['root_failures'].append({
                'segment': current_segment, 'command_v': float(applied), 'state_u': float(state),
                'error_type': type(exc).__name__, 'error': str(exc)})
            raise

    try:
        check(); valid_state(u0, 'incoming_state')
        unique_u[0] = float(u0)
        present_signs = {1 if v > 0 else -1 for v in np.r_[initial_command, command] if v != 0.}
        bounds = {sign: kernel.smallfield_bound(sign, b) for sign in present_signs}
        summary['smallfield_bounds_by_sign'] = {str(sign): value for sign, value in bounds.items()}
        if any(not math.isfinite(value) or value >= nearzero_budget for value in bounds.values()):
            raise ResolutionFailure('Signed near-zero omission bound is not below the explicit per-segment budget')
        state = float(u0)
        for index, (start, stop, va, vb) in enumerate(segments):
            check(); current_segment = index; active_solver = None
            sign = 1 if (va if va != 0. else vb) > 0. else -1
            reference_kappa = kernel._lambda_reference_parameters['kappa_per_v2']
            kappa_ratio = kernel.parameters.kappa_per_v2 / reference_kappa
            first_step = min(stop-start, precision.max_step, 1e-4 / max(1., kappa_ratio))
            row = {
                'index': index, 'path_start_v': start, 'path_stop_v': stop,
                'command_start_v': va, 'command_stop_v': vb, 'sign': sign,
                'initial_u': state, 'status': 'running', 'first_step_v': first_step,
                'full_zero_microsegment': min(abs(va), abs(vb)) == 0. and max(abs(va), abs(vb)) >= kernel.vmin,
            }
            summary['segments'].append(row)
            accepted[index] = {'coordinate': [start], 'state_u': [state]}
            unique_u[unique == start] = state

            def actual_command(coordinate):
                fraction = (coordinate-start)/(stop-start)
                if not 0. <= fraction <= 1.:
                    raise DomainFailure('Solver coordinate outside actual waveform segment')
                return (1.-fraction)*va + fraction*vb

            def rhs(coordinate, y):
                nonlocal rhs_calls, rhs_kcl, rhs_kvl
                check()
                proposed, applied = float(y[0]), float(actual_command(coordinate))
                rhs_calls += 1
                summary['last_rhs'] = {'segment': index, 'coordinate': float(coordinate),
                    'command_v': applied, 'state_u': proposed}
                valid_state(proposed, 'RHS_trial')
                if applied == 0.:
                    point = kernel._point(0., 0., 0., 0., 0.)
                    rate = 0.
                elif abs(applied) < kernel.vmin:
                    actual_sign = 1 if applied > 0. else -1
                    if actual_sign != sign:
                        raise DomainFailure('Sign changed inside sign-constant waveform segment')
                    omissions.append({'segment': index, 'sign': sign, 'command_v': applied, 'state_u': proposed})
                    if collect_rhs:
                        rhs_rows.append([applied, proposed, np.nan, np.nan, np.nan, 0., np.nan])
                    return np.array([0.])
                else:
                    point = solve(applied, proposed)
                    rhs_kcl = max(rhs_kcl, abs(point['KCL_residual_a']))
                    rhs_kvl = max(rhs_kvl, abs(point['KVL_residual_v']))
                    rate = kernel.signed_rate(proposed, point, b)
                if collect_rhs:
                    rhs_rows.append([applied, proposed, point['current_a'], point['local_voltage_v'],
                                     point['transistor_voltage_v'], rate, point['KCL_residual_a']])
                return np.array([rate])

            solver_type = DOP853 if method == 'DOP853' else Radau
            active_solver = solver_type(rhs, start, np.array([state]), stop,
                rtol=precision.rtol, atol=precision.atol, max_step=precision.max_step,
                first_step=first_step, vectorized=False)
            while active_solver.status == 'running':
                check()
                previous_t = float(active_solver.t)
                message = active_solver.step()
                row.update(nfev=int(active_solver.nfev), njev=int(active_solver.njev),
                           nlu=int(active_solver.nlu), solver_status=active_solver.status)
                if active_solver.status == 'failed':
                    row.update(status='solver_failed', message=str(message))
                    raise RuntimeError(str(message))
                accepted_t, accepted_u = float(active_solver.t), float(active_solver.y[0])
                accepted[index]['coordinate'].append(accepted_t)
                accepted[index]['state_u'].append(accepted_u)
                row.update(last_accepted_coordinate_v=accepted_t, last_accepted_u=accepted_u,
                           accepted_steps=len(accepted[index]['coordinate'])-1)
                valid_state(accepted_u, 'accepted_step')
                # DOP853 dense output may evaluate extra RHS points; it is still
                # part of the same accepted step and has no hidden retry policy.
                dense = active_solver.dense_output()
                use = (unique > previous_t) & (unique <= accepted_t)
                if np.any(use):
                    sampled = np.asarray(dense(unique[use]))[0]
                    for value in sampled:
                        valid_state(value, 'native_dense_output')
                    unique_u[use] = sampled
                check()
            if active_solver.status != 'finished':
                raise RuntimeError('Solver ended without reaching the segment boundary')
            state = float(active_solver.y[0])
            valid_state(state, 'segment_terminal')
            unique_u[unique == stop] = state
            row.update(status='completed', success=True, message='Reached actual segment boundary',
                       final_u=state, nfev=int(active_solver.nfev), njev=int(active_solver.njev),
                       nlu=int(active_solver.nlu), accepted_steps=len(accepted[index]['coordinate'])-1)
        native_u = unique_u[inverse][1:]
        arrays.update(state_u=native_u.copy(), state_q=np.exp(-native_u))
        if not np.all(np.isfinite(native_u)):
            raise DomainFailure('Missing native state after waveform integration')
        current_segment = 'native_readout'
        for key in ('current_a', 'local_voltage_v', 'transistor_voltage_v', 'KCL_residual_a', 'KVL_residual_v'):
            arrays[key] = np.full(len(command), np.nan)
        for j, (applied, value) in enumerate(zip(command, native_u)):
            point = solve(applied, value)
            for key in ('current_a', 'local_voltage_v', 'transistor_voltage_v', 'KCL_residual_a', 'KVL_residual_v'):
                arrays[key][j] = point[key]
            summary['last_completed_native_readout_index'] = j
        if any(not np.all(np.isfinite(arrays[key])) for key in (
                'current_a', 'local_voltage_v', 'transistor_voltage_v', 'KCL_residual_a', 'KVL_residual_v')):
            raise DomainFailure('Nonfinite native circuit outputs')
        if (np.any(arrays['current_a']*command < 0.)
                or np.any(arrays['current_a']*arrays['local_voltage_v'] < 0.)
                or np.any(arrays['current_a']*arrays['transistor_voltage_v'] < 0.)):
            raise DomainFailure('Nonpassive signed native circuit output')
        if observed is not None:
            if np.any(abs(arrays['current_a'][mask]) <= 0.):
                raise DomainFailure('Invalid current on original eligible scoring mask')
            residual = np.log10(abs(arrays['current_a'][mask])/abs(observed[mask]))
            errors = np.full(len(command), np.nan); errors[mask] = residual
            arrays['log10_current_residual'] = errors
            groups = {'all_eligible': mask, 'low_abs_le_0p2': mask & (abs(command) <= .2+1e-12)}
            if len(present_signs) == 1:
                outward = np.arange(len(command)) <= np.argmax(abs(command))
                groups.update(outward=mask & outward, return_=mask & ~outward)
            summary['metrics'] = {('return' if name == 'return_' else name): {
                'n': int(use.sum()), 'RMSE_decade': float(np.sqrt(np.mean(errors[use]**2))),
                'mean_signed_log_error_decade': float(np.mean(errors[use])),
                'median_absolute_factor': float(10**np.median(abs(errors[use])))}
                for name, use in groups.items() if np.any(use)}
        joins = [abs(a['final_u']-z['initial_u'])
                 for a, z in zip(summary['segments'][:-1], summary['segments'][1:])]
        summary.update(status='completed', final_u=state, terminal_u=state,
            final_command_voltage_v=float(command[-1]), max_segment_state_join_error=max(joins, default=0.),
            max_native_KCL_residual_a=float(max(abs(arrays['KCL_residual_a']), default=0.)),
            max_native_KVL_residual_v=float(max(abs(arrays['KVL_residual_v']), default=0.)),
            n_scored=int(mask.sum()) if mask is not None else 0)
        check()
    except Exception as exc:
        category = ('caller_budget_exhausted' if isinstance(exc, TimeoutError) else
            'numerical_domain_unresolved' if isinstance(exc, DomainFailure) else
            'numerical_resolution_unresolved' if isinstance(exc, ResolutionFailure) else 'solver_or_audit_failure')
        summary.update(status='failed_preserved_for_caller', error_type=type(exc).__name__, error=str(exc),
            failure_scope=category, physical_rejection=False,
            failure_interpretation='Integrator trial, table domain, resolution or budget failure; never physical exclusion')
        if summary['segments'] and summary['segments'][-1]['status'] == 'running':
            summary['segments'][-1]['status'] = 'exception_before_completion'
        raise PhaseFailure(str(exc), summary, arrays) from exc
    finally:
        for index, values in accepted.items():
            arrays[f'segment_{index}_coordinate'] = np.asarray(values['coordinate'])
            arrays[f'segment_{index}_state_u'] = np.asarray(values['state_u'])
            arrays[f'segment_{index}_state_q'] = np.exp(-np.asarray(values['state_u']))
        counted = {row['index']: row['sign'] for row in summary['segments']
                   if row['status'] == 'completed' and row['full_zero_microsegment']}
        counted.update({row['segment']: row['sign'] for row in omissions})
        summary.update(n_rhs_calls=rhs_calls, max_rhs_KCL_residual_a=rhs_kcl,
            max_rhs_KVL_residual_v=rhs_kvl,
            smallfield_omission={'n_RHS_omissions': len(omissions),
                'physical_microsegment_indices': sorted(counted),
                'local_bound_sum': sum(summary.get('smallfield_bounds_by_sign', {}).get(str(sign), 0.) for sign in counted.values()),
                'exact_inactive_claim': False, 'evaluations': omissions,
                'scope': 'At most once per physical monotone near-zero microsegment; partial omitted segments counted conservatively. Local integral bound only, not trajectory error.'})
        arrays['partial_unique_path_v'] = unique
        arrays['partial_unique_state_u'] = unique_u
        if 'state_u' not in arrays:
            arrays['partial_native_state_u'] = unique_u[inverse][1:]
        if collect_rhs:
            arrays['rhs_trace'] = np.asarray(rhs_rows, dtype=float).reshape((-1, 7))
            summary['rhs_trace_columns'] = ['command_v', 'state_u', 'current_a', 'local_voltage_v',
                'transistor_voltage_v', 'signed_u_derivative_per_v', 'KCL_residual_a']
    return {'summary': summary, 'arrays': arrays, 'residual': residual}
