"""Contract and independent numerical checks for the uncalibrated gap candidate."""
import numpy as np
import unittest
import inspect
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from vulcan2d.shared_gap import GapLaw, BareGapCircuit, LookupGapCircuit, simulate_gap
from vulcan2d.transistor import TransistorLookup, TransistorMapping, measured_transistor_mapping


def law(**kwargs):
    values = dict(ron_ohm=100., v0_v=.2, i0_a=2e-5, u_max=12.,
                  kset_per_v=.5, kreset_per_v=.4,
                  eth_set_v=.15, eth_reset_v=.2, escale_v=.3)
    values.update(kwargs)
    return GapLaw(**values)


def mapping(plateau=False, resistance=0.):
    currents = [0., 1e-4, 1e-4, 3e-4] if plateau else [0., 1e-4, 2e-4, 5e-4]
    payload = {"schema_version": 1, "curves": [
        {"gate_voltage_v": gate, "drain_voltage_v": [0., 1., 2., 5.],
         "drain_current_a": currents} for gate in (1., 2.)]}
    lookup = TransistorLookup.from_payload(payload)
    return TransistorMapping(lookup, reset_lookup=lookup,
                             reset_policy="measured_reverse", series_resistance_ohm=resistance)


class SharedGapTests(unittest.TestCase):
    def test_zero_initiation_threshold_preserves_existing_trajectory(self):
        command = [0., .5, 1., .5, 0., -.5, -1., -.5, 0.]
        expected_state = [3.0, 2.99930668695399, 2.9104334094534754, 2.813086038442491,
                          2.811276068645261, 2.811276068645261, 2.8566337969852955,
                          2.8988972112114246, 2.8988972112114246]
        expected_current = [0.0, 9.299104531712163e-07, 2.935404800011365e-06,
                            1.2124979792854304e-06, 0.0, -1.2156715468310045e-06,
                            -3.2059990438158523e-06, -1.0720251697365282e-06, 0.0]
        for model in (law(), law(reset_initiation_drive_v=0.)):
            result = simulate_gap(command, 3., model, BareGapCircuit(1e-4), max_step_v=.01)
            np.testing.assert_array_equal(result['state_u'], expected_state)
            np.testing.assert_array_equal(result['current_a'], expected_current)

    def test_reset_initiation_parameter_must_be_nonnegative_and_finite(self):
        for invalid in (-1., np.nan, np.inf):
            with self.assertRaises(ValueError):
                law(reset_initiation_drive_v=invalid)

    def test_global_initiation_threshold_suppresses_closed_gap_reset_only(self):
        base, barrier = law(), law(reset_initiation_drive_v=.25)
        current = -5e-5
        self.assertGreater(base.derivative(0., current), 0.)
        self.assertEqual(barrier.derivative(0., current), 0.)
        self.assertGreater(barrier.derivative(1., current), 0.)
        for u in (0., 1., 8.):
            self.assertEqual(base.voltage(current, u), barrier.voltage(current, u))
            self.assertEqual(base.derivative(u, -current), barrier.derivative(u, -current))
        self.assertGreater(base.derivative(1., current), barrier.derivative(1., current))
        # The added barrier becomes negligible for an already open gap.
        ratio = barrier.derivative(8., current) / base.derivative(8., current)
        self.assertGreater(ratio, .999)

    def test_material_interface_has_no_topology_or_gate(self):
        for target in (GapLaw, GapLaw.voltage, GapLaw.current, GapLaw.drive, GapLaw.derivative):
            forbidden = {"topology", "gate_voltage", "device", "mode", "hard"}
            self.assertFalse(forbidden & set(inspect.signature(target).parameters))

    def test_measured_transistor_branches_match_independent_load_root(self):
        model = law()
        transistor = measured_transistor_mapping()
        circuit = LookupGapCircuit(transistor, 1.1)
        for u in (.1, 2., 10.):
            for command in (-1.8, -.1, .1, 4.):
                point = circuit.solve(command, model, u)
                sign = np.sign(command)
                voltage = brentq(lambda vh: model.current(vh, u)
                                 - abs(transistor.drain_current(sign * (abs(command) - vh), 1.1)),
                                 0., abs(command), xtol=1e-13)
                np.testing.assert_allclose(point.hbn_voltage_v, sign * voltage, atol=2e-11, rtol=1e-10)
                np.testing.assert_allclose(point.current_a, model.current(sign * voltage, u), atol=1e-16, rtol=1e-8)

    def test_ohmic_limit_and_tiny_current_inverse(self):
        model = law()
        for voltage in (-2., 0., 2.):
            assert model.current(voltage, 0.) == voltage / model.ron_ohm
            assert model.voltage(voltage / model.ron_ohm, 0.) == voltage
        for u in (0., .5, 12.):
            for current in (1e-30, 1e-14, 1e-6, .01):
                np.testing.assert_allclose(model.current(model.voltage(current, u), u), current, rtol=2e-10, atol=1e-40)
                assert model.voltage(-current, u) == -model.voltage(current, u)
    
    
    def test_critical_compliance_and_regulation_voltage(self):
        model = law()
        threshold_current = model.i0_a * np.sinh(model.eth_set_v / model.v0_v)
        cap = threshold_current / np.exp(2.)
        assert abs(model.derivative(2., cap)) < 1e-14
        assert model.derivative(2.1, cap) < 0.
        point = BareGapCircuit(cap).solve(4., model, 2.)
        assert point.compliance_limited
        assert point.current_a == cap
        np.testing.assert_allclose(point.hbn_voltage_v, model.voltage(cap, 2.))
        assert point.supply_voltage_v < point.command_voltage_v
        assert point.compliance_drop_v + point.supply_voltage_v == 4.
    
    
    def test_shared_law_signed_load_and_flat_lookup_segment(self):
        model = law(kset_per_v=0., kreset_per_v=0.)
        transistor = mapping(plateau=True, resistance=25.)
        circuit = LookupGapCircuit(transistor, 1.5)
        for sign in (-1., 1.):
            for vt in (.2, 1.5, 3.):
                current = sign * transistor.current(vt, 1.5, side="set")
                command = model.voltage(current, 1.) + sign * vt
                point = circuit.solve(command, model, 1.)
                np.testing.assert_allclose(point.current_a, current, rtol=2e-9, atol=1e-15)
                np.testing.assert_allclose(point.hbn_voltage_v + point.transistor_voltage_v, command)
                np.testing.assert_allclose(transistor.drain_current(point.transistor_voltage_v, 1.5), point.current_a, rtol=2e-9, atol=1e-15)
        with self.assertRaisesRegex(ValueError, "range|domain|extrapolat"):
            circuit.solve(100., model, 1.)
    
    
    def test_series_compliance_uses_actual_voltage(self):
        model = law()
        circuit = LookupGapCircuit(mapping(plateau=True), 1.5, compliance_a=1e-4)
        point = circuit.solve(4., model, .2)
        assert point.compliance_limited
        assert point.current_a == 1e-4
        np.testing.assert_allclose(point.transistor_voltage_v, 1.)
        np.testing.assert_allclose(point.supply_voltage_v, point.hbn_voltage_v + point.transistor_voltage_v)
        assert point.supply_voltage_v + point.compliance_drop_v == 4.
    
    
    def test_transistor_plateau_at_compliance_keeps_command_voltage(self):
        model = law()
        circuit = LookupGapCircuit(mapping(plateau=True), 1.5, compliance_a=1e-4)
        command = model.voltage(1e-4, .2) + 1.5
        point = circuit.solve(command, model, .2)
        self.assertFalse(point.compliance_limited)
        self.assertEqual(point.supply_voltage_v, command)
        np.testing.assert_allclose(point.current_a, 1e-4)

    def test_continuous_set_return_and_reset_handoff(self):
        model = law()
        circuit = BareGapCircuit(1e-4)
        command = np.array([0., .5, 1., .5, 0., -.5, -1., -.5, 0.])
        full = simulate_gap(command, 3., model, circuit, max_step_v=.01)
        first = simulate_gap(command[:5], 3., model, circuit, max_step_v=.01)
        second = simulate_gap(command[5:], first['final_u'], model, circuit,
                              initial_command_voltage_v=command[4], max_step_v=.01)
        np.testing.assert_allclose(full['state_u'][5:], second['state_u'], atol=1e-12, rtol=0.)
        assert full['state_u'][3] < full['state_u'][2]  # positive return remains active
        assert full['state_u'][-1] > full['state_u'][4]
        assert np.all(full['state_u'] >= 0.)
    
    
    def test_positive_initial_rate_not_erased_by_feedback_and_independent_ivp(self):
        model = law(kset_per_v=10.)
        transistor = mapping()
        circuit = LookupGapCircuit(transistor, 1.5)
        initial_u, start, stop = 2., 1., 1.12
    
        def independent_current(command, u):
            # Independent voltage-space load root, no prepared knot inversion.
            vh = brentq(lambda voltage: model.current(voltage, u)
                        - transistor.drain_current(command - voltage, 1.5),
                        0., command, xtol=1e-13)
            return model.current(vh, u)
    
        assert model.derivative(initial_u, independent_current(start, initial_u)) < 0.
        reference = solve_ivp(lambda s, y: [model.derivative(float(y[0]), independent_current(start + s, float(y[0])))],
                              [0., stop - start], [initial_u], method="Radau", rtol=1e-10, atol=1e-12)
        assert reference.success
        result = simulate_gap([stop], initial_u, model, circuit,
                              initial_command_voltage_v=start, max_step_v=.002)
        assert result['final_u'] < initial_u
        np.testing.assert_allclose(result['final_u'], reference.y[0, -1], rtol=0., atol=2e-7)
    
    
    def test_rk2_preserves_initial_rate_near_projected_boundary(self):
        model = law()
        circuit = BareGapCircuit()
        self.assertLess(model.derivative(.01, circuit.solve(1., model, .01).current_a), 0.)
        result = simulate_gap([1., 1.1], .01, model, circuit,
                              initial_command_voltage_v=1., max_step_v=.01, method="rk2")
        self.assertEqual(result['final_u'], 0.)

    def test_refinement_both_polarities_and_bounds(self):
        model = law(kset_per_v=2., kreset_per_v=1.)
        circuit = BareGapCircuit(2e-4)
        command = [0., .5, 1., .5, 0., -.5, -1., -.5, 0.]
        coarse = simulate_gap(command, 1., model, circuit, max_step_v=.01)
        fine = simulate_gap(command, 1., model, circuit, max_step_v=.0025)
        np.testing.assert_allclose(coarse['state_u'], fine['state_u'], atol=2e-4, rtol=2e-4)
        assert np.all(np.isfinite(fine['current_a']))
        assert model.derivative(0., .001) == 0.
        assert model.derivative(model.u_max, -.001) == 0.

if __name__ == "__main__":
    unittest.main()
