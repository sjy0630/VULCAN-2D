"""Accuracy and protocol contracts for adaptive shared-gap integration."""
import unittest
from dataclasses import replace
import numpy as np

from vulcan2d.shared_gap import GapLaw, BareGapCircuit, LookupGapCircuit
from vulcan2d.shared_gap_ivp import simulate_gap_ivp
from vulcan2d.transistor import measured_transistor_mapping


def selected_law():
    return GapLaw(ron_ohm=93.4800875524324, v0_v=.039954620313492985,
                  i0_a=2.71150055588204e-7, u_max=30., kset_per_v=843.3159003419391,
                  kreset_per_v=.3310806330955371, eth_set_v=.35771554271889355,
                  eth_reset_v=.27371071502507793, escale_v=.020925256503739546,
                  reset_initiation_drive_v=.14439326619795345)


class AdaptiveGapTests(unittest.TestCase):
    def test_zero_path_repeated_points_and_ohmic_limit(self):
        law = replace(selected_law(), kset_per_v=0., kreset_per_v=0.)
        result = simulate_gap_ivp([0., 0., .1, .1, 0., -.1, -.1], 0., law, BareGapCircuit())
        np.testing.assert_array_equal(result['state_u'], np.zeros(7))
        np.testing.assert_allclose(result['current_a'], np.array([0., 0., .1, .1, 0., -.1, -.1])/law.ron_ohm)
        empty = simulate_gap_ivp([], 1., law, BareGapCircuit())
        self.assertEqual(empty['final_u'], 1.)
        self.assertEqual(empty['current_a'].shape, (0,))

    def test_coarse_step_stop_state_error_is_removed(self):
        law = selected_law()
        voltage = np.r_[np.linspace(0., 5., 101), np.linspace(4.95, 0., 100)]
        result = simulate_gap_ivp(voltage, 7.5368589569794615, law, BareGapCircuit(.001))
        stop = np.log(law.i0_a*np.sinh(law.eth_set_v/law.v0_v)/.001)
        self.assertGreaterEqual(np.min(result['state_u']), stop-1e-12)
        self.assertAlmostEqual(result['final_u'], stop, delta=1e-8)
        self.assertTrue(result['metadata']['current_tolerance_refinement_passed'])

    def test_continuous_pair_split_preserves_state(self):
        law, circuit = selected_law(), LookupGapCircuit(measured_transistor_mapping(), 1.1)
        first_v = [0., 1., 3., 5., 3., 1., 0.]
        second_v = [-.25, -.7, -1.7, -.7, 0.]
        whole = simulate_gap_ivp(first_v+second_v, 4.643852229872415, law, circuit)
        first = simulate_gap_ivp(first_v, 4.643852229872415, law, circuit)
        second = simulate_gap_ivp(second_v, first['final_u'], law, circuit)
        np.testing.assert_allclose(whole['state_u'][len(first_v):], second['state_u'], atol=2e-6, rtol=0.)
        self.assertLess(first['final_u'], 4.643852229872415)
        self.assertGreater(second['final_u'], first['final_u'])

    def test_selected_parameter_reference_endpoints(self):
        law = selected_law()
        circuit = LookupGapCircuit(measured_transistor_mapping(), 1.1)
        set_v = np.r_[np.linspace(0., 5., 201), np.linspace(4.975, 0., 200)]
        reset_v = np.r_[np.linspace(0., -1.7, 69), np.linspace(-1.675, 0., 68)]
        set_trace = simulate_gap_ivp(set_v, 4.643852229872415, law, circuit)
        reset_trace = simulate_gap_ivp(reset_v, set_trace['final_u'], law, circuit)
        self.assertAlmostEqual(set_trace['final_u'], 2.4542437559319166, delta=3e-6)
        self.assertAlmostEqual(reset_trace['final_u'], 4.82092965885897, delta=3e-6)

    def test_invalid_controls_rejected(self):
        for kw in ({'state_atol':0.}, {'state_rtol':-1.}, {'current_rtol':0.}, {'method':'RK4'}):
            with self.assertRaises(ValueError):
                simulate_gap_ivp([0., 1.], 1., selected_law(), BareGapCircuit(), **kw)


if __name__ == '__main__':
    unittest.main()
