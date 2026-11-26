"""Short independent ODE/adapter checks; no full32 run or optimization."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from protocol_problem import (
    ROOT, ProtocolClock, ProtocolClockSimulator, _ClockedDerivativeView,
    make_problem, SpectralJointProblem, PARAMETER_NAMES,
    BareSpectralCircuit, simulate_spectral_ivp,
)

HERE = Path(__file__).resolve().parent
OPTIONS = dict(method="DOP853", state_rtol=1e-9, state_atol=1e-11,
               current_rtol=1e-5, state_refine_atol=1e-8,
               max_step_v=.02, max_refinements=4)


@dataclass(frozen=True)
class AnalyticLinearLaw:
    """Independent synthetic du/ds= +/-k*u; I=G*V is state independent."""
    rate: float = .7
    conductance: float = 1e-4
    state_limit: float = 50.
    jset_a: float = 1e-12

    def validate_state(self, u):
        value = float(u)
        if not np.isfinite(value) or not 0 <= value <= self.state_limit:
            raise ValueError("Synthetic state outside declared interval")
        return value

    def current(self, v, u):
        self.validate_state(u)
        return self.conductance * v

    def voltage(self, current, u):
        self.validate_state(u)
        return current / self.conductance

    def derivative(self, u, current_a):
        self.validate_state(u)
        return -self.rate * u * np.sign(current_a)


def equal_trace(a, b):
    assert set(a) == set(b)
    for key, value in a.items():
        if isinstance(value, np.ndarray):
            assert value.dtype == b[key].dtype and value.tobytes() == b[key].tobytes(), key
        elif key != "metadata":
            assert value == b[key], key
    # Wall-clock elapsed is inherently different, not a numerical result.
    ma, mb = dict(a["metadata"]), dict(b["metadata"])
    ma.pop("elapsed_s"); mb.pop("elapsed_s")
    assert ma == mb


def main():
    started = time.monotonic(); checks = {}; numerical = []
    for bad in (0, -1, np.inf, np.nan, True, 1e-320):
        try:
            ProtocolClock(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid alpha accepted: {bad}")
    checks["invalid_or_reciprocal_overflow_alpha_rejected"] = True

    problems = {alpha: make_problem(alpha, OPTIONS) for alpha in (1., 4.)}
    p1, p4 = problems[1.], problems[4.]
    reference = SpectralJointProblem(p1.law_factory, circuits=p1.circuits,
                                     integrator_options=OPTIONS)
    assert p1.simulate_fn is simulate_spectral_ivp
    for p in (p1, p4):
        assert p.train_ids == reference.train_ids
        assert p.train_ids["1t1r"] == list(range(4, 36))
        assert p.training_arrays_sha256 == reference.training_arrays_sha256
        assert p.n_residuals == reference.n_residuals
        assert p.scatter_constant_cost == reference.scatter_constant_cost
        assert p.initial_indices == {"1r": 5, "1t1r": 6}
        assert p.parameter_names == PARAMETER_NAMES and p.n_parameters == 7
        assert not p.include_reset_initiation
        assert p._pair.__func__ is SpectralJointProblem._pair
        assert p.evaluate.__func__ is SpectralJointProblem.evaluate
        for key in p.groups:
            np.testing.assert_array_equal(p.waveforms[key], reference.waveforms[key])
            for group, original in zip(p.groups[key], reference.groups[key]):
                np.testing.assert_array_equal(group["mask"], original["mask"])
                assert group["normalization"] == original["normalization"]
    checks["exact_training_masks_order_scatter_initial_indices_and_inherited_propagation"] = True
    checks["alpha1_injects_original_simulator_without_wrapper"] = True

    q = np.array([-1.5, -8., -9., -2., -2., 6., 6.])
    law = p1.law_factory(q, p1.ron)
    other = p4.law_factory(q, p4.ron)
    assert law.exact_law == other.exact_law
    assert law.exact_law.reset_initiation_log == 0.
    np.testing.assert_array_equal(law.table.log_current, other.table.log_current)
    checks["same_A0_material_parameters_shape_and_transport_table"] = True
    for topology in ("1r", "1t1r"):
        for phase_index, sign in enumerate((1., -1.)):
            command = sign * np.array([0., .02, .04, .06])
            circuit = p1.circuits[topology][phase_index]
            a = simulate_spectral_ivp(command, 6., law, circuit, **OPTIONS)
            b = p1.simulate_fn(command, 6., law, circuit, **OPTIONS)
            equal_trace(a, b)
            view = _ClockedDerivativeView(law, .25)
            assert view.base_law is law
            assert view.state_limit == law.state_limit and view.jset_a == law.jset_a
            assert view.current(sign*.06, 6.) == law.current(sign*.06, 6.)
            assert view.voltage(view.current(sign*.06, 6.), 6.) == law.voltage(law.current(sign*.06, 6.), 6.)
            assert circuit.solve(sign*.06, law, 6.) == circuit.solve(sign*.06, view, 6.)
    checks["alpha1_short_spectral_traces_bit_identical_both_circuits_and_polarities"] = True
    checks["clock_view_keeps_material_readout_inverse_thresholds_bounds_and_KCL_exact"] = True

    bare, series = BareSpectralCircuit(), BareSpectralCircuit()
    circuits = {"1r": (bare, bare), "1t1r": (series, series)}
    toy = AnalyticLinearLaw(); path = np.array([0., .02, .07, .15])
    analytic_options = dict(OPTIONS, state_rtol=1e-12, state_atol=1e-14,
                            state_refine_atol=1e-11)
    for alpha in (.5, 1., 4.):
        simulator = ProtocolClockSimulator(alpha, circuits)
        for topology, circuit in (("1r", bare), ("1t1r", series)):
            multiplier = 1. if topology == "1r" else 1. / alpha
            for sign in (1., -1.):
                trace = simulator(sign*path, 1., toy, circuit, **analytic_options)
                exact = np.exp(-sign*toy.rate*multiplier*path)
                error = float(np.max(abs(trace["state_u"]-exact)))
                assert error < 1e-9, (alpha, topology, sign, error, trace["state_u"], exact)
                np.testing.assert_array_equal(trace["current_a"], toy.conductance*sign*path)
                numerical.append(dict(alpha=alpha, topology=topology, polarity=int(sign),
                                      maximum_analytic_state_error=error,
                                      final_state=trace["final_u"], analytic_final_state=float(exact[-1])))
    checks["independent_linear_ODE_analytic_solution_verifies_inverse_alpha_for_both_polarities"] = True
    checks["bare_clock_stays_one_and_no_current_rescaling"] = True

    # Independent phase handoff: the second initial state comes from the first.
    simulator = ProtocolClockSimulator(4., circuits)
    first = simulator(path, 1., toy, series, **analytic_options)
    second = simulator(-path, first["final_u"], toy, series,
                       initial_command_voltage_v=path[-1], **analytic_options)
    # From +.15 to0 closes by another exp(-k*.15/alpha), then RESET opens.
    expected = np.exp(-toy.rate*.15/4.)
    assert abs(second["final_u"]-expected) < 1e-9
    checks["phase_handoff_uses_previous_final_state_and_actual_initial_command"] = True

    limited = BareSpectralCircuit(.001)
    capped_circuits = {"1r": (bare, bare), "1t1r": (limited, limited)}
    capped = ProtocolClockSimulator(4., capped_circuits)(
        path, 1., replace(toy, conductance=1.), limited, **OPTIONS)
    assert np.max(abs(capped["current_a"])) <= .001
    assert np.all(capped["compliance_limited"][1:])
    np.testing.assert_allclose(capped["hbn_voltage_v"], capped["current_a"], rtol=0, atol=0)
    checks["existing_compliance_and_voltage_accounting_unchanged"] = True
    for alpha in (1., 4.):
        simulator = ProtocolClockSimulator(alpha, circuits)
        try:
            simulator(-path, 1., replace(toy, state_limit=1.01), series, **OPTIONS)
        except RuntimeError as exc:
            assert "State left certified numerical domain" in str(exc)
        else:
            raise AssertionError("Clock adapter suppressed the existing upper-domain failure")
        zero = simulator(path, 0., toy, series, **OPTIONS)
        assert np.all(zero["state_u"] == 0.)
    checks["existing_zero_boundary_and_upper_domain_failure_preserved"] = True

    metadata = p4.checkpoint_metadata()
    assert metadata["conditional_protocol_clock"]["alpha_r_series_over_r_bare"] == 4.
    assert metadata["conditional_protocol_clock"]["series_path_derivative_multiplier"] == .25
    assert metadata["series_initial_state_used_once_at_record"] == 4
    assert metadata["evaluation_split"] == "training_only_no_heldout_arrays_loaded"
    assert str(HERE/"protocol_problem.py") in metadata["source_sha256"]
    assert metadata["reset_initiation_log_fixed"] == 0.
    checks["checkpoint_records_alpha_code_hashes_and_no_physical_time_claim"] = True
    report = dict(status="short_adapter_and_analytic_ODE_validation_passed_no_full32_no_optimization",
                  checks=checks, all_passed=all(checks.values()), analytic_runs=numerical,
                  elapsed_s=time.monotonic()-started, checkpoint_example=metadata,
                  validation_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  limitations=["No actual32-cycle candidate or parameter optimization was run.",
                               "Analytic synthetic ODE verifies protocol-clock conversion, not hBN physics.",
                               "Alpha remains an assumed relative scan-rate scenario, not a measured speed."])
    (HERE/"protocol_validation.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: report[key] for key in ("status", "all_passed", "checks", "elapsed_s")}, indent=2))


if __name__ == "__main__":
    main()
