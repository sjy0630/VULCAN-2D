"""Small algebra checks only: synthetic inputs, no real data/roots/ODEs.

Fixture coefficients below are arbitrary algebra examples, never a candidate
rate schedule or fitted parameter transfer from q2 to q1.
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import FrozenInstanceError
import json
import math
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from unittest.mock import patch

import q1_material as q1


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(left, right, message):
    require(math.isclose(left, right, rel_tol=3e-14, abs_tol=1e-20), message)


def forbidden(*args, **kwargs):
    raise AssertionError('Data, table, circuit, root or ODE execution is forbidden in this algebra check')


def synthetic_point(material, state, voltage):
    # These positive branch values exercise rate algebra only. They are NOT
    # evaluated material currents or an actual solved electrical circuit.
    soft = 2e-5 * voltage
    hard = state[1] * voltage
    return dict(_material_id=id(material), _state=tuple(state),
                local_voltage_v=voltage, current_a=soft + hard,
                soft_current_a=soft, hard_current_a=hard)


def run_checks():
    k = q1.kernel
    sources = q1.source_hashes()
    n_rate_checks = 0
    n_equilibrium_checks = 0
    with ExitStack() as stack:
        for owner, name in (
            (k, 'load_material'), (k, 'load_minimal'), (k, '_root'),
            (k, 'simulate_phase'), (k, 'simulate_history'),
            (k, 'DOP853'), (k, 'Radau'),
            (k.BareCircuit, 'solve'), (k.SeriesCircuit, 'solve'),
            (q1.training, 'load_training_problem'),
            (q1.training.TrainingProblem, 'evaluate'), (k.np, 'load'),
        ):
            stack.enter_context(patch.object(owner, name, forbidden))
        for method in ('validate_state', '_table_abs', 'components', 'current', 'rates'):
            require(getattr(q1.Q1Material, method) is getattr(k.CoupledMaterial, method),
                    f'Unexpected override of {method}')
        for enabled in (False, True):
            # Synthetic values chosen for arithmetic coverage; no rate center.
            parameters = k.Parameters(lambda_per_v=.4, J_positive_a=3e-5,
                kappa_positive_per_v2=.7, b=.3, a_max_s=.002,
                kappa_a_per_v2=.6, rho=1.4, enable_a=enabled)
            parent = k.CoupledMaterial(
                SimpleNamespace(ron_ohm=1., gamma_c_ev=.1), parameters, .025,
                'synthetic_algebra_fixture_not_candidate', q1.sha(__file__),
                MappingProxyType({str(q1.KERNEL_PATH.resolve()): q1.KERNEL_SHA256}),
            )
            child = q1.as_q1_material(parent)
            require(isinstance(child, q1.training.kernel.CoupledMaterial), 'Adapter type identity')
            require(child.readout is parent.readout, 'Readout object changed')
            require(child.parameters is parent.parameters, 'Parameters changed')
            require(parent.metadata()['fixed_recovery_order'] == 2, 'Parent metadata mutated')
            require(child.fixed_recovery_order == child.metadata()['fixed_recovery_order'] == 1,
                    'Q1 order not explicitly recorded')
            require(child.metadata()['source_sha256'].items() >= sources.items(),
                    'New and frozen source provenance missing')
            try:
                child.fixed_recovery_order = 2
            except FrozenInstanceError:
                pass
            else:
                raise AssertionError('Recovery order is mutable')
            for q in (0., 1e-12, .01, .2, .8, 1.):
                for a in ((0., .0006, .002) if enabled else (0.,)):
                    state = q, a
                    for vh in (-2., -.02, 0., .02, 2.):
                        parent_terms = parent.rate_components(state, synthetic_point(parent, state, vh))
                        point = synthetic_point(child, state, vh)
                        terms = child.rate_components(state, point)
                        require(all(math.isfinite(x) and x >= 0 for x in terms.values()),
                                'Nonfinite or negative component')
                        for name in ('capture_q_per_v', 'formation_a_s_per_v', 'loss_a_s_per_v'):
                            require(terms[name] == parent_terms[name], f'Changed inherited {name}')
                        close(parent_terms['recovery_q_per_v'], q * terms['recovery_q_per_v'],
                              'q2 recovery != q * q1 recovery')
                        require(terms['recovery_q_per_v'] >= parent_terms['recovery_q_per_v'],
                                'q1 recovery smaller at fixed coefficients')
                        sign = 1 if vh > 0 else -1
                        C = parameters.kappa_positive_per_v2 * abs(vh) * (abs(point['current_a']) /
                            parameters.J_positive_a) * math.exp(-(1 - sign) * parameters.b)
                        R = parameters.kappa_positive_per_v2 * abs(vh) * math.exp(
                            (1 - sign) * parameters.b + parameters.lambda_per_v * abs(vh))
                        close(terms['capture_q_per_v'], C * (1 - q), 'Capture formula')
                        close(terms['recovery_q_per_v'], R * q, 'Direct q1 recovery formula')
                        dq, da = child.rates(state, point)
                        if q == 0:
                            require(terms['recovery_q_per_v'] == 0 and dq >= 0, 'q=0 boundary')
                        if q == 1:
                            require(dq <= 0, 'q=1 boundary')
                        if a == 0:
                            require(da >= 0, 'a=0 boundary')
                        if a == parameters.a_max_s:
                            require(da <= 0, 'a=A boundary')
                        if vh == 0:
                            require(all(x == 0 for x in terms.values()), 'Zero-voltage rates')
                        if 0 < q <= 1:
                            close(-dq / q, R - C * (1 - q) / q, 'Internal u derivative')
                        n_rate_checks += 1
            # Algebraic q=0 validity must not expand the actual table domain.
            for boundary in (0., 1.):
                try:
                    child.validate_state((boundary, 0.), table=True)
                except k.DomainFailure:
                    pass
                else:
                    raise AssertionError('Finite table endpoint restrictions expanded')
            bad_point = synthetic_point(parent, (.2, 0.), 1.)
            try:
                child.rate_components((.2, 0.), bad_point)
            except ValueError:
                pass
            else:
                raise AssertionError('Material identity validation bypassed')
        # Conditional constant-coefficient balances; no coupled equilibrium solve.
        for C in (0., .1, 1., 10.):
            for R in (0., .1, 1., 10.):
                if C + R == 0:
                    continue
                qstar1 = C / (C + R)
                qstar2 = 2 * C / (C + math.sqrt(C * C + 4 * R * C)) if C > 0 else 0.
                close(C * (1 - qstar1), R * qstar1, 'Q1 conditional balance')
                close(C * (1 - qstar2), R * qstar2 * qstar2, 'Q2 conditional balance')
                require(qstar1 <= qstar2 + 1e-15, 'Conditional equilibrium ordering')
                n_equilibrium_checks += 1
    require(q1.source_hashes() == sources, 'Sources changed during algebra check')
    return dict(status='algebra_checks_passed_only',
        fixed_recovery_order=1, parent_fixed_recovery_order=2,
        n_rate_cases=n_rate_checks, n_conditional_balance_cases=n_equilibrium_checks,
        data_loaded=False, tables_loaded=False, roots_solved=False,
        trajectories_integrated=False, candidates_selected=False,
        fixture_parameters_are_candidate=False, source_sha256=sources,
        selfcheck_sha256=q1.sha(Path(__file__)))


if __name__ == '__main__':
    print(json.dumps(run_checks(), indent=2, sort_keys=True))
