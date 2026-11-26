"""The same transistor must be used alone and in both 1T1R polarities."""
import unittest
from dataclasses import replace

import numpy as np

from vulcan2d import model_v13 as M13
from vulcan2d.set_identification import direct_set_transistor_mapping
from vulcan2d.transistor import load_default_lookup, load_reverse_lookup


def waves():
    vs = np.linspace(0., 5., 11)
    vr = np.linspace(0., -1.7, 11)
    return np.r_[vs, vs[-2::-1]], np.r_[vr, vr[-2::-1]]


class TransistorConsistencyTests(unittest.TestCase):
    def test_legacy_measured_override_respects_physical_polarity(self):
        from vulcan2d import model
        from vulcan2d.transistor import measured_transistor_mapping
        mapping = measured_transistor_mapping()
        for side, voltage in (("set", 1.), ("reset", -1.)):
            actual = model.i_tr(
                abs(voltage), side, model.Params(), 1.,
                transistor_lookup=mapping, gate_voltage=1.1)
            self.assertEqual(actual, abs(mapping.drain_current(voltage, 1.1)))

    def test_circuit_rejects_nonzero_zero_bias_hbn_current(self):
        from vulcan2d.circuit import solve_series_load_line
        from vulcan2d.transistor import measured_transistor_mapping
        with self.assertRaisesRegex(ValueError, "origin"):
            solve_series_load_line(1., 1.1, lambda v: 1e-3 + v * 1e-3,
                                   measured_transistor_mapping())

    def test_one_sided_transistor_scaling_is_rejected(self):
        # Previously SET changed by ~21 uA while RESET stayed bitwise identical.
        mapping = replace(direct_set_transistor_mapping(), current_scale=0.5)
        with self.assertRaisesRegex(ValueError, "same transistor"):
            M13.simulate_calibrated_cycle(*waves(), set_mapping=mapping)

    def test_one_mapping_drives_both_phases_and_standalone(self):
        from vulcan2d.transistor import measured_transistor_mapping
        mapping = replace(measured_transistor_mapping(), current_scale=0.5)
        baseline = M13.simulate_calibrated_cycle(*waves())
        result = M13.simulate_calibrated_cycle(
            *waves(), transistor_mapping=mapping)
        for phase in ("set", "reset"):
            self.assertGreater(np.max(np.abs(
                baseline[phase]["Is"] - result[phase]["Is"])), 1e-9)
            branch = result[phase]
            for applied, vh, current in zip(
                    branch["Vs"], branch["Vhs"], branch["Is"]):
                expected = mapping.drain_current(applied - vh, 1.1)
                self.assertAlmostEqual(current, expected, delta=2e-12)

    def test_signed_law_retains_physical_gate_reference_and_data(self):
        from vulcan2d.transistor import measured_transistor_mapping
        mapping = measured_transistor_mapping()
        for sign, lookup in ((1, load_default_lookup()),
                             (-1, load_reverse_lookup())):
            for gate in (0.9, 1.1, 1.3):
                for vd in (0., .01, .2, 1., 1.7):
                    expected = sign * lookup.current(vd, gate)
                    actual = mapping.drain_current(sign * vd, gate)
                    self.assertEqual(actual, expected)
                    self.assertGreaterEqual(actual * sign * vd, 0.)

    def test_signed_reverse_requires_reverse_measurements(self):
        with self.assertRaisesRegex(ValueError, "measured reverse"):
            direct_set_transistor_mapping().drain_current(-.2, 1.1)

    def test_series_solver_ohmic_limits_both_polarities(self):
        from vulcan2d.circuit import solve_series_load_line
        from vulcan2d.transistor import measured_transistor_mapping
        mapping = measured_transistor_mapping()
        for applied in (-1., 1.):
            for resistance in (1., 1e4, 1e14):
                current, vh = solve_series_load_line(
                    applied, 1.1, lambda v: v / resistance, mapping, nbis=60)
                standalone = mapping.drain_current(applied - vh, 1.1)
                self.assertAlmostEqual(current, standalone, delta=1e-16)
                self.assertAlmostEqual(current, vh / resistance, delta=1e-16)
                self.assertLessEqual(abs(current), abs(mapping.drain_current(applied, 1.1)))
                if resistance == 1.:
                    self.assertAlmostEqual(current / mapping.drain_current(applied, 1.1), 1., delta=1e-3)
                if resistance == 1e14:
                    self.assertLessEqual(abs(current), 1.01e-14)

    def test_invalid_terminal_inputs_are_rejected(self):
        from vulcan2d.transistor import measured_transistor_mapping
        mapping = measured_transistor_mapping()
        for vd, gate in ((np.nan, 1.1), (1., np.nan), (np.inf, 1.1)):
            with self.assertRaises(ValueError):
                mapping.drain_current(vd, gate)

    def test_reset_zero_uses_reverse_gate_domain(self):
        from vulcan2d.model_v7 import divider_reset
        self.assertEqual(divider_reset(0., .5, 1e-8, 1e-10, .2), (0., 0.))

    def test_explicit_historical_reset_proxy_remains_available(self):
        from vulcan2d.model_v6 import transistor_mapping
        from vulcan2d.model_v7 import divider_reset
        proxy = transistor_mapping()
        current, vh = divider_reset(1., .5, 1e-8, 1e-10, 1.1, mapping=proxy)
        self.assertAlmostEqual(current, proxy.current(1. - vh, 1.1, "reset"),
                               delta=2e-12)


if __name__ == "__main__":
    unittest.main()
