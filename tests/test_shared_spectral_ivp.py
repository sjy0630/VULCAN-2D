"""Independent circuit and state-path contracts for the spectral IVP adapter.

Analytic asymmetric toy material isolates circuit/integration errors from the
spectral quadrature. Its positive branch admits exact path solutions; signed
series loads are independently rooted in transistor voltage. These checks do
not use observed 1R/1T1R currents or establish a microscopic mechanism.
"""
from dataclasses import dataclass, replace
import math
import unittest

import numpy as np
from scipy.optimize import brentq

from vulcan2d.shared_spectral_ivp import (
    BareSpectralCircuit, LookupSpectralCircuit, SpectralTable,
    TabulatedSpectralLaw, simulate_spectral_ivp)
from vulcan2d.transistor import TransistorLookup, measured_transistor_mapping


@dataclass(frozen=True)
class AsymmetricToyShape:
    ron_ohm: float = 100.
    negative_state_exponent: float = 1.7

    def current_at_log_gamma(self, voltage, log_gamma, order=32):
        del order
        v, lg = np.broadcast_arrays(np.asarray(voltage), np.asarray(log_gamma))
        state = np.logaddexp(0., -lg)
        a = np.where(v >= 0., 1., self.negative_state_exponent)
        return v/self.ron_ohm*np.exp(-a*state)


@dataclass(frozen=True)
class AsymmetricToyLaw:
    shape: AsymmetricToyShape = AsymmetricToyShape()
    gamma_c_ev: float = 1.
    jset_a: float = 1e-3
    jreset_a: float = 2e-5
    kset_per_v: float = .3
    kreset_per_v: float = .08
    state_limit: float = 20.

    def validate_state(self, u):
        u = float(u)
        if not math.isfinite(u) or not 0. <= u <= self.state_limit:
            raise ValueError("State outside toy numerical domain")
        return u

    def current(self, voltage, u):
        state = self.validate_state(u)
        a = 1. if voltage >= 0. else self.shape.negative_state_exponent
        return float(voltage)/self.shape.ron_ohm*math.exp(-a*state)

    def voltage(self, current, u):
        state = self.validate_state(u)
        a = 1. if current >= 0. else self.shape.negative_state_exponent
        return float(current)*self.shape.ron_ohm*math.exp(a*state)

    def derivative(self, u, current):
        state = self.validate_state(u)
        if current == 0. or (current > 0. and state == 0.):
            return 0.
        threshold = self.jset_a if current > 0. else self.jreset_a
        rate = self.kset_per_v if current > 0. else self.kreset_per_v
        amplitude = rate*max(math.log(abs(current))-math.log(threshold)+state, 0.)
        return -amplitude if current > 0. else amplitude


class SignedSpectralCircuitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mapping = measured_transistor_mapping()
        cls.circuit = LookupSpectralCircuit(cls.mapping, 1.1)

    def test_nonodd_material_real_transistor_kcl_matches_voltage_root(self):
        law = AsymmetricToyLaw()
        self.assertNotAlmostEqual(law.current(-1., 1.), -law.current(1., 1.), delta=1e-4)
        for state in (0., 1., 8.):
            for command in (.05, .4, 1.7, 5., -.05, -.4, -1.7):
                with self.subTest(state=state, command=command):
                    sign, magnitude = math.copysign(1., command), abs(command)
                    # Independent coordinate: root the actual signed transistor
                    # voltage, not the implementation's inverse/log-current root.
                    vt_mag = brentq(lambda vt: sign*law.current(sign*(magnitude-vt), state)
                                    -sign*self.mapping.drain_current(sign*vt, 1.1),
                                    0., magnitude, xtol=1e-14, rtol=1e-14)
                    expected = self.mapping.drain_current(sign*vt_mag, 1.1)
                    point = self.circuit.solve(command, law, state)
                    self.assertAlmostEqual(point.current_a, expected, delta=max(2e-14, abs(expected)*2e-9))
                    self.assertAlmostEqual(point.hbn_voltage_v+point.transistor_voltage_v, command, delta=2e-9)
                    self.assertAlmostEqual(law.current(point.hbn_voltage_v, state), point.current_a,
                                           delta=max(2e-14, abs(expected)*2e-9))
                    self.assertAlmostEqual(self.mapping.drain_current(point.transistor_voltage_v, 1.1), point.current_a,
                                           delta=max(2e-14, abs(expected)*2e-9))
                    self.assertGreaterEqual(point.current_a*point.hbn_voltage_v, 0.)
                    self.assertGreaterEqual(point.current_a*point.transistor_voltage_v, 0.)

    def test_bare_compliance_reduces_actual_voltage_with_correct_sign(self):
        law, circuit, state = AsymmetricToyLaw(), BareSpectralCircuit(.001), 1.
        for command in (2., -2.):
            with self.subTest(command=command):
                point = circuit.solve(command, law, state)
                expected_i = math.copysign(.001, command)
                expected_v = law.voltage(expected_i, state)
                self.assertTrue(point.compliance_limited)
                self.assertEqual(point.current_a, expected_i)
                self.assertAlmostEqual(point.hbn_voltage_v, expected_v, places=14)
                self.assertLess(abs(point.hbn_voltage_v), abs(command))
                self.assertEqual(point.supply_voltage_v, point.hbn_voltage_v)
                self.assertAlmostEqual(point.hbn_voltage_v+point.compliance_drop_v, command, places=14)
                self.assertAlmostEqual(law.current(point.hbn_voltage_v, state), expected_i, places=14)

    def test_declared_synthetic_transistor_plateau_has_valid_interval_inverse(self):
        # The actual VG=1.1-V processed data need not have a flat interval.
        # This fixture explicitly exercises the allowed interval-valued case.
        def lookup(scale):
            return TransistorLookup.from_payload({"schema_version": 1, "curves": [
                {"gate_voltage_v": gate, "drain_voltage_v": [0., 1., 2., 3.],
                 "drain_current_a": (scale*np.array([0., 1e-4, 1e-4, 3e-4])).tolist()}
                for gate in (1., 1.5)]})
        fixture_mapping = measured_transistor_mapping(lookup(1.), lookup(2.))
        circuit = LookupSpectralCircuit(fixture_mapping, 1.1)
        law = AsymmetricToyLaw()
        checked = 0
        for side, intervals in circuit._flat_intervals.items():
            sign = 1. if side == "set" else -1.
            for current, first, last in intervals:
                if current <= 0. or first == last:
                    continue
                desired_vt = .5*(first+last)
                vh = law.voltage(sign*current, 1.)
                point = circuit.solve(vh+sign*desired_vt, law, 1.)
                self.assertAlmostEqual(point.current_a, sign*current, delta=1e-13)
                self.assertAlmostEqual(point.transistor_voltage_v, sign*desired_vt, delta=1e-9)
                checked += 1
                break
        self.assertEqual(checked, 2)

    def test_real_transistor_domain_exit_is_rejected_not_extrapolated(self):
        for command in (10., -10.):
            with self.subTest(command=command):
                with self.assertRaisesRegex(ValueError, "exceed measured transistor voltage range"):
                    self.circuit.solve(command, AsymmetricToyLaw(), 0.)


class SpectralPathIntegrationTests(unittest.TestCase):
    def test_positive_outward_and_return_match_analytic_path_integral(self):
        law, initial = AsymmetricToyLaw(), 2.
        outward = np.array([0., .05, .1, .2, .4, .7, 1.])
        returning = np.array([1., .7, .4, .2, .1, .05, 0.])
        voltage = np.r_[outward, returning]
        threshold_v = law.shape.ron_ohm*law.jset_a
        def primitive(values):
            values = np.asarray(values)
            result = np.zeros_like(values)
            use = values > threshold_v
            x = values[use]
            result[use] = x*np.log(x/threshold_v)-x+threshold_v
            return result
        distance = np.r_[primitive(outward), 2*primitive([1.])[0]-primitive(returning)]
        expected = initial-law.kset_per_v*distance
        trace = simulate_spectral_ivp(voltage, initial, law, BareSpectralCircuit(),
                                      method="Radau", state_rtol=2e-8, state_atol=1e-10,
                                      state_refine_atol=1e-8, current_rtol=1e-7, max_refinements=3)
        np.testing.assert_allclose(trace["state_u"], expected, atol=8e-8, rtol=0.)
        self.assertEqual(trace["state_u"][6], trace["state_u"][7])
        self.assertLess(trace["final_u"], trace["state_u"][6])

    def test_own_compliance_floor_matches_exponential_and_never_projects_up(self):
        law = replace(AsymmetricToyLaw(), jset_a=.002, kset_per_v=4.)
        circuit, initial = BareSpectralCircuit(.001), 3.
        voltage = np.array([5., 4., 5., 4.])
        path = np.array([0., 1., 2., 3.])
        stopping = math.log(law.jset_a/.001)
        expected = stopping+(initial-stopping)*np.exp(-law.kset_per_v*path)
        trace = simulate_spectral_ivp(voltage, initial, law, circuit,
                                      initial_command_voltage_v=5., method="Radau",
                                      state_rtol=1e-8, state_atol=1e-10,
                                      state_refine_atol=1e-8, max_refinements=3)
        np.testing.assert_allclose(trace["state_u"], expected, atol=1e-8, rtol=0.)
        np.testing.assert_allclose(trace["current_a"], .001, atol=1e-15, rtol=0.)
        self.assertGreaterEqual(float(trace["state_u"].min()), stopping)
        self.assertLessEqual(trace["metadata"]["max_state_refinement_u"], 1e-8)
        lower_initial = .3
        frozen = simulate_spectral_ivp(voltage, lower_initial, law, circuit, initial_command_voltage_v=5.)
        np.testing.assert_array_equal(frozen["state_u"], np.full(len(voltage), lower_initial))

    def test_zero_state_absorbs_set_but_releases_on_reset(self):
        law = replace(AsymmetricToyLaw(), jset_a=1e-7, jreset_a=1e-6, kreset_per_v=.1)
        positive = simulate_spectral_ivp([0., .1, .5, .1, 0.], 0., law, BareSpectralCircuit())
        np.testing.assert_array_equal(positive["state_u"], np.zeros(5))
        negative = simulate_spectral_ivp([0., -.1, -.5, -.1, 0.], positive["final_u"], law, BareSpectralCircuit())
        self.assertGreater(negative["final_u"], 0.)
        self.assertTrue(np.all(np.diff(negative["state_u"]) >= -1e-10))

    def test_two_cycles_and_nonzero_phase_handoff_equal_unsplit_path(self):
        law = replace(AsymmetricToyLaw(), jset_a=1e-4, kset_per_v=.15)
        circuit, initial = BareSpectralCircuit(), 1.5
        pieces = [[0., .4, 1.2, .6], [.2, 0., -.2, -.9, -.4, 0.],
                  [0., .4, 1.2, .4, 0.], [0., -.2, -.9, -.2, 0.]]
        command = np.concatenate(pieces)
        controls = dict(state_rtol=1e-8, state_atol=1e-10, current_rtol=2e-6,
                        state_refine_atol=1e-7, max_refinements=3)
        whole = simulate_spectral_ivp(command, initial, law, circuit, method="DOP853", **controls)
        state, previous, split_states = initial, 0., []
        for piece in pieces:
            trace = simulate_spectral_ivp(piece, state, law, circuit, method="Radau",
                                          initial_command_voltage_v=previous, **controls)
            split_states.extend(trace["state_u"])
            state, previous = trace["final_u"], piece[-1]
        np.testing.assert_allclose(whole["state_u"], split_states, atol=8e-7, rtol=0.)
        self.assertAlmostEqual(whole["final_u"], state, delta=8e-7)
        self.assertNotAlmostEqual(float(split_states[len(pieces[0])+len(pieces[1])]), initial, delta=1e-3)

    def test_reset_state_limit_is_domain_failure_not_artificial_saturation(self):
        law = replace(AsymmetricToyLaw(), jreset_a=1e-9, kreset_per_v=2., state_limit=.1)
        with self.assertRaisesRegex(RuntimeError, "no RESET saturation"):
            simulate_spectral_ivp([0., -.2, -1.], .09, law, BareSpectralCircuit())

    def test_empty_and_repeated_commands_consume_no_path(self):
        law = AsymmetricToyLaw()
        empty = simulate_spectral_ivp([], 1., law, BareSpectralCircuit())
        self.assertEqual(empty["final_u"], 1.)
        self.assertEqual(empty["state_u"].shape, (0,))
        same = simulate_spectral_ivp([.5, .5, .5], 1., law, BareSpectralCircuit(), initial_command_voltage_v=.5)
        np.testing.assert_array_equal(same["state_u"], np.ones(3))

    def test_nonfinite_initial_command_is_rejected_not_zero_path(self):
        for initial_command in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(initial_command=initial_command):
                with self.assertRaisesRegex(ValueError, "Initial command voltage must be finite"):
                    simulate_spectral_ivp([0., .5, 1.], 1., AsymmetricToyLaw(), BareSpectralCircuit(),
                                          initial_command_voltage_v=initial_command)


class SpectralTableApproximationTests(unittest.TestCase):
    def test_table_retains_rectification_and_converges_to_same_law(self):
        law, shape = AsymmetricToyLaw(), AsymmetricToyShape()
        coarse = SpectralTable.build(shape, gamma_decade_step=1., voltage_decade_step=.5,
                                     gamma_decade_bounds=(-3., 3.), minimum_voltage=.001)
        fine = SpectralTable.build(shape, gamma_decade_step=.1, voltage_decade_step=.1,
                                   gamma_decade_bounds=(-3., 3.), minimum_voltage=.001)
        approximate = [TabulatedSpectralLaw(law, table) for table in (coarse, fine)]
        errors = [[], []]
        for u in (.12, .3, .8, 1.7, 4.3):
            for v in (.012, .18, 1.2, 4.2, -.012, -.18, -1.2, -4.2):
                expected = law.current(v, u)
                for j, candidate in enumerate(approximate):
                    current = candidate.current(v, u)
                    errors[j].append(abs(math.log10(abs(current/expected))))
                    self.assertAlmostEqual(candidate.voltage(current, u), v, delta=1e-11)
                    self.assertEqual(candidate.derivative(u, current), law.derivative(u, current))
        self.assertGreater(max(errors[0]), 1e-4, "Coarse table should expose actual interpolation error")
        self.assertLess(max(errors[1]), max(errors[0])/8.)
        candidate = approximate[1]
        self.assertNotAlmostEqual(candidate.current(-1., 1.), -candidate.current(1., 1.), delta=1e-4)
        self.assertEqual(candidate.current(1e-5, 1.), law.current(1e-5, 1.))
        self.assertEqual(candidate.current(-1., 16.), law.current(-1., 16.))
        self.assertEqual(candidate.current(-1., 0.), -.01)

    def test_true_spectral_frozen_state_signed_series_small_check(self):
        from vulcan2d.shared_spectral import SpectralLaw, SpectralShape
        shape = SpectralShape.from_json()
        law = SpectralLaw(shape, gamma_c_ev=.003, jset_a=1e-4, jreset_a=1e-4,
                          kset_per_v=0., kreset_per_v=0.)
        mapping = measured_transistor_mapping()
        circuit = LookupSpectralCircuit(mapping, 1.1)
        for command in (.6, -.6):
            point = circuit.solve(command, law, 1.)
            self.assertAlmostEqual(point.hbn_voltage_v+point.transistor_voltage_v, command, delta=2e-9)
            self.assertAlmostEqual(law.current(point.hbn_voltage_v, 1.), point.current_a, delta=2e-13)
            self.assertAlmostEqual(mapping.drain_current(point.transistor_voltage_v, 1.1), point.current_a, delta=2e-13)


if __name__ == "__main__":
    unittest.main()
