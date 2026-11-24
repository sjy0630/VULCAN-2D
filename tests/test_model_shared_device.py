"""Circuit and state invariants for the shared h-BN sweep model."""
import unittest
from dataclasses import replace

import numpy as np

from vulcan2d import model_shared_device as model
from vulcan2d.validate_shared_device import PREPARATION, REFERENCE


class SharedDeviceTest(unittest.TestCase):
    def setUp(self):
        outward = np.linspace(0, 5, 251)
        reset = np.linspace(0, -1.5, 76)
        self.set_voltage = np.r_[outward, outward[-2::-1]]
        self.reset_voltage = np.r_[reset, reset[-2::-1]]

    def test_no_load_obeys_the_same_device_current(self):
        output = model.simulate_sweep(
            self.set_voltage, "SET", REFERENCE, PREPARATION, model.Circuit())
        np.testing.assert_equal(output["device_voltage_V"], self.set_voltage)
        expected = model.device_current(
            self.set_voltage, output["soft_state"], output["hard_state"],
            REFERENCE, PREPARATION, "SET")
        np.testing.assert_allclose(output["current_A"], expected, rtol=1e-11)

    def test_series_load_enforces_kcl_and_limits_hard_formation(self):
        grid = np.linspace(0, 5, 501)
        transistor = (grid, 24e-6 * grid)
        circuit = model.Circuit(transistor=transistor)
        out = model.simulate_sweep(
            self.set_voltage, "SET", REFERENCE, PREPARATION, circuit)
        vds = self.set_voltage - out["device_voltage_V"]
        np.testing.assert_allclose(out["current_A"], circuit.transistor_current(vds),
                                   rtol=2e-6, atol=4e-12)
        self.assertGreater(out["final"].soft, .99)
        self.assertEqual(out["final"].hard, 0)

    def test_current_compliance_and_exact_cycle_handoff(self):
        limit = 1e-3
        written, erased = model.simulate_cycle(
            self.set_voltage, self.reset_voltage, REFERENCE, PREPARATION,
            model.Circuit(compliance_a=limit), model.Circuit())
        self.assertTrue(np.any(np.isclose(written["current_A"], limit, rtol=0, atol=5e-10)))
        self.assertTrue(np.all(written["current_A"] <= limit))
        self.assertEqual(written["final"].hard, 1)
        self.assertEqual(erased["hard_state"][0], written["final"].hard)
        self.assertEqual(erased["soft_state"][0], written["final"].soft)
        self.assertTrue(np.all(np.diff(erased["hard_state"]) <= 0))
        self.assertTrue(np.all(np.diff(erased["soft_state"]) <= 0))

    def test_strong_access_uses_loaded_current_and_preserves_state_handoff(self):
        device = replace(REFERENCE, hard_set_v=2., hard_access_a=.6e-3,
                         hard_conductance_s=.2e-3, hard_access_mode="sustained")
        grid = np.linspace(0., 5., 501)
        weak = model.Circuit(transistor=(grid, .2e-3 * grid))
        strong = model.Circuit(transistor=(grid, 2e-3 * grid))
        bare = model.Circuit(compliance_a=1e-3)
        self.assertGreater(weak.supply_bound(grid)[-1], device.hard_access_a)
        self.assertEqual(model.simulate_sweep(self.set_voltage, "SET", device,
                                             PREPARATION, weak)["final"].hard, 0.)
        for circuit in (strong, bare):
            written, erased = model.simulate_cycle(self.set_voltage, self.reset_voltage,
                                                   device, PREPARATION, circuit, model.Circuit())
            self.assertEqual(written["final"].hard, 1.)
            self.assertEqual(erased["hard_state"][0], 1.)
            self.assertEqual(erased["soft_state"][0], written["final"].soft)
            hit = written["hard_state"] == 1.
            np.testing.assert_allclose(written["current_A"][hit],
                                       device.hard_conductance_s * written["device_voltage_V"][hit],
                                       atol=5e-11, rtol=1e-7)
            if circuit.transistor is not None:
                np.testing.assert_allclose(written["current_A"][hit], circuit.transistor_current(
                    written["voltage_V"][hit] - written["device_voltage_V"][hit]),
                    atol=2e-10, rtol=1e-7)

    def test_current_limited_soft_growth_keeps_load_line_and_retained_state(self):
        grid = np.linspace(0., 5., 501)
        circuit = model.Circuit(transistor=(grid, 24e-6 * grid))
        device = replace(REFERENCE, soft_growth_current_a=200e-6, hard_access_mode="sustained")
        ordinary = model.simulate_sweep(self.set_voltage, "SET", REFERENCE, PREPARATION, circuit)
        limited = model.simulate_sweep(self.set_voltage, "SET", device, PREPARATION, circuit)
        self.assertLess(limited["final"].soft, ordinary["final"].soft)
        self.assertGreater(limited["final"].soft, 0.)
        self.assertTrue(np.all(np.diff(limited["soft_state"]) >= 0))
        expected = circuit.transistor_current(self.set_voltage - limited["device_voltage_V"])
        np.testing.assert_allclose(limited["current_A"], expected, rtol=1e-5, atol=4e-12)

    def test_power_assist_preserves_start_and_accelerates_soft_reset(self):
        initial = model.State(.8, 0.)
        circuit = model.Circuit()
        base = model.simulate_sweep(self.reset_voltage, "RESET", REFERENCE,
                                    PREPARATION, circuit, initial)
        device = replace(REFERENCE, reset_power_assist_v=.5)
        assisted = model.simulate_sweep(self.reset_voltage, "RESET", device,
                                        PREPARATION, circuit, initial)
        self.assertAlmostEqual(assisted["soft_state"][0], initial.soft)
        self.assertLess(assisted["final"].soft, base["final"].soft)
        self.assertTrue(np.all(np.diff(assisted["soft_state"]) <= 0.))
        apex = int(np.argmax(np.abs(self.reset_voltage)))
        np.testing.assert_equal(assisted["soft_state"][apex:], assisted["final"].soft)
        np.testing.assert_allclose(assisted["device_voltage_V"], self.reset_voltage)


if __name__ == "__main__":
    unittest.main()
