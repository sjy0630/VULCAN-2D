"""Two fixed conditional readout inversions and a necessary path-budget bound.

No ODE, fitting, b scan, initial-state replacement or holdout access.
Execution is explicit; prior research sources and results remain read-only.
"""
from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import time

sys.dont_write_bytecode = True
import numpy as np

HERE = Path(__file__).resolve().parent
SIGNED = HERE.parent / 'signed_soft_history_2026-11-04'
sys.path.insert(0, str(SIGNED))
from minimal_static_loader import load_minimal


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inverse_piecewise(x, y, target):
    """Invert only an in-domain unique target of a nondecreasing table."""
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(y))
    assert np.all(np.diff(x) > 0) and np.all(np.diff(y) >= 0)
    assert y[0] <= target <= y[-1], 'Target outside frozen table'
    equal = np.flatnonzero(y == target)
    if len(equal):
        assert len(equal) == 1, 'Nonunique plateau target; no point chosen'
        j = int(equal[0])
        return float(x[j]), [j, j]
    j = int(np.searchsorted(y, target))
    assert 0 < j < len(x) and y[j - 1] < target < y[j]
    value = x[j - 1] + (x[j] - x[j - 1]) * (target - y[j - 1]) / (y[j] - y[j - 1])
    return float(value), [j - 1, j]


def invert_readpoint(kernel, label, command, observed, check_budget):
    check_budget()
    sign = 1 if command > 0 else -1
    tx, ti = (kernel.base.tx, kernel.base.ti) if sign > 0 else (kernel.negative_tx, kernel.negative_ti)
    drop, tj = inverse_piecewise(tx, ti, observed)
    local = abs(command) - drop
    assert kernel.vmin <= local <= abs(command) and 0 <= drop <= tx[-1]
    assert 0 <= float(np.interp(drop, tx, ti))
    lv, lg = kernel.lv, kernel.lg
    y = math.log(local)
    assert lv[0] <= y <= lv[-1]
    j = min(max(int(np.searchsorted(lv, y) - 1), 0), len(lv) - 2)
    fraction = (y - lv[j]) / (lv[j + 1] - lv[j])
    spectrum = kernel.base.li if sign > 0 else kernel.li_negative
    log_current_slice = (1 - fraction) * spectrum[:, j] + fraction * spectrum[:, j + 1] + kernel.base.log_scale
    target = math.log(observed)
    lg_direct, gj = inverse_piecewise(lg, log_current_slice, target)

    # Independent fixed-count bisection uses the kernel material evaluator,
    # not the vector slice used for the direct piecewise inverse.
    low, high = float(lg[0]), float(lg[-1])
    rlow = math.log(kernel._material_abs(local, low, sign)) - target
    rhigh = math.log(kernel._material_abs(local, high, sign)) - target
    assert rlow <= 0 <= rhigh
    for _ in range(48):
        check_budget()
        middle = (low + high) / 2
        residual = math.log(kernel._material_abs(local, middle, sign)) - target
        if residual >= 0:
            high = middle
        else:
            low = middle
    u = math.log1p(kernel.gc * math.exp(-lg_direct))
    u_bracket = [math.log1p(kernel.gc * math.exp(-high)), math.log1p(kernel.gc * math.exp(-low))]
    u_bisect = math.log1p(kernel.gc * math.exp(-(low + high) / 2))
    assert abs(u - u_bisect) <= 1e-8
    kernel.gamma(u)
    circuits = {}
    for mode in ['fast', 'strict_bisection42']:
        check_budget()
        point = kernel.make_circuit(mode=mode).solve(command, u)
        log_error = abs(math.log10(abs(point['current_a']) / observed))
        assert log_error <= 5e-7
        assert abs(point['KCL_residual_a']) <= 1e-13 and abs(point['KVL_residual_v']) <= 1e-12
        assert point['current_a'] * command > 0 and point['local_voltage_v'] * point['current_a'] > 0
        circuits[mode] = dict(point, observed_current_log_error_decade=log_error)
    # This is an explicitly padded floating-point bracket, not a rigorous
    # interval-arithmetic certificate or an experimental uncertainty interval.
    padded_u = [min(u, u_bracket[0]) - 1e-8, max(u, u_bracket[1]) + 1e-8]
    return {
        'label': label, 'command_voltage_v': command, 'observed_current_magnitude_a': observed,
        'transistor_drop_magnitude_v': drop, 'local_voltage_magnitude_v': local,
        'transistor_bracket_indices': tj, 'transistor_bracket_voltages': tx[tj].tolist(),
        'transistor_bracket_currents': ti[tj].tolist(),
        'transistor_monotonic_checked': True, 'material_slice_monotonic_checked': True,
        'unique_target': True, 'material_bracket_indices': gj,
        'material_bracket_log_gamma': lg[gj].tolist(),
        'material_bracket_log_current': log_current_slice[gj].tolist(),
        'direct_log_gamma': lg_direct, 'direct_gamma_ev': math.exp(lg_direct),
        'direct_u': u, 'direct_Z': math.exp(u), 'bisect_u': u_bisect,
        'bisect_log_gamma_bracket': [low, high], 'bisect_u_bracket': u_bracket,
        'bisect_log_current_residual_bracket': [math.log(kernel._material_abs(local, x, sign)) - target for x in [low, high]],
        'bisect_iterations': 48, 'numerical_padded_u_bracket': padded_u,
        'numerical_padded_Z_bracket': [math.exp(x) for x in padded_u],
        'circuit_rechecks': circuits,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-authorized', action='store_true')
    args = parser.parse_args()
    if not args.run_authorized:
        parser.error('Explicit finite-plan execution flag required')
    destination = HERE / 'compatibility_result.json'
    if destination.exists():
        raise FileExistsError('This bounded stage has already executed; no automatic rerun')
    load_started = time.monotonic()
    kernel = load_minimal()
    previous = json.loads((SIGNED / 'bounded_signed_trial.json').read_text())
    for p, digest in kernel.source_sha256.items():
        if Path(p).parent != HERE:
            assert previous['source_sha256'][p] == digest, p
    reset = kernel.load_training_phase(4, 'reset')
    onset = kernel.load_training_phase(5, 'set')
    assert math.isclose(reset['command_voltage_v'][161], -.2, abs_tol=1e-14)
    assert math.isclose(onset['command_voltage_v'][10], .2, abs_tol=1e-14)
    assert reset['observed_current_a'][161] == 1.2504e-9
    assert onset['observed_current_a'][10] == 7.25e-10
    bridge = np.r_[reset['command_voltage_v'][161:], onset['command_voltage_v'][:11]]
    assert np.all(np.diff(bridge) >= -1e-14)
    assert math.isclose(float(abs(np.diff(bridge)).sum()), .4, abs_tol=1e-14)
    sources = dict(kernel.source_sha256)
    for p in [Path(__file__), HERE / 'EXPERIMENT_PLAN.md', SIGNED / 'bounded_signed_trial.json']:
        sources[str(p)] = sha(p)
    result = {'status': 'started', 'source_sha256': sources, 'parameters': asdict(kernel.parameters),
              'Ron_ohm': kernel.ron, 'gamma_c_ev': kernel.gc,
              'input_IO_wall_s': time.monotonic() - load_started,
              'readpoints': [], 'command_bridge_v': bridge.tolist(), 'bridge_path_length_v': float(abs(np.diff(bridge)).sum()),
              'new_ODEs': 0, 'new_fits': 0, 'new_b_trials': 0, 'original_holdout_accessed': False,
              'state_reanchoring': False, 'bare_history_accessed': False}
    started = time.monotonic()
    def check_budget():
        if time.monotonic() - started > 10:
            raise TimeoutError('Fixed10s numerical budget exceeded')
    try:
        for label, record, index in [('RESET4_return_0p2', reset, 161), ('SET5_outward_0p2', onset, 10)]:
            result['readpoints'].append(invert_readpoint(kernel, label,
                float(record['command_voltage_v'][index]), float(abs(record['observed_current_a'][index])), check_budget))
        p = kernel.parameters
        delta, B = .2, math.log(10)
        x = p.lambda_per_v * delta
        integral = (math.exp(x) * (x - 1) + 1) / p.lambda_per_v**2 if p.lambda_per_v else .5 * delta**2
        quadratures = {}
        for order in [16, 32]:
            nodes, weights = np.polynomial.legendre.leggauss(order)
            a = .5 * delta * (nodes + 1)
            quadratures[str(order)] = float(.5 * delta * np.dot(weights, a * np.exp(p.lambda_per_v * a)))
            assert math.isclose(quadratures[str(order)], integral, rel_tol=1e-12, abs_tol=1e-15)
        tight_bound = p.kappa_per_v2 * (math.exp(2 * B) + 1) * integral
        simple_bound = .5 * p.kappa_per_v2 * delta**2 * math.exp(x) * (math.exp(2 * B) + 1)
        negative, positive = result['readpoints']
        needed = positive['direct_Z'] - negative['direct_Z']
        needed_lower = positive['numerical_padded_Z_bracket'][0] - negative['numerical_padded_Z_bracket'][1]
        needed_upper = positive['numerical_padded_Z_bracket'][1] - negative['numerical_padded_Z_bracket'][0]
        conservative_bound = simple_bound * (1 + 1e-6)
        result['necessary_budget'] = {
            'delta_v': delta, 'B': B, 'analytic_integral_v2': integral, 'gauss_integral_v2': quadratures,
            'tight_upper_deltaZ': tight_bound, 'simple_upper_deltaZ': simple_bound,
            'numerical_margin_fraction': 1e-6, 'decision_upper_deltaZ': conservative_bound,
            'required_deltaZ': needed, 'required_deltaZ_padded_bracket': [needed_lower, needed_upper],
            'lower_required_over_decision_upper': needed_lower / conservative_bound,
            'conditional_incompatibility': needed_lower > conservative_bound,
        }
        check_budget()
        result['status'] = 'completed_conditional_incompatibility' if needed_lower > conservative_bound else 'completed_necessary_condition_not_excluding'
        result['numerical_checks_passed'] = True
    except Exception as exc:
        result['status'] = 'failed_or_numerically_unresolved'
        result['error'] = repr(exc)
        result['numerical_checks_passed'] = False
        raise
    finally:
        result['numerical_wall_s'] = time.monotonic() - started
        result['completed_conditional_readpoint_inversions'] = len(result['readpoints'])
        with destination.open('x') as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
        print(json.dumps({k: result[k] for k in ['status', 'numerical_wall_s', 'completed_conditional_readpoint_inversions']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
