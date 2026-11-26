"""Scientific invariants of the uncalibrated common 1R / 1T1R kernel."""
import inspect
import unittest
from dataclasses import replace

import numpy as np

from vulcan2d.shared_hbn import (
    SharedHbnLaw, BareCircuit, TransistorCircuit, ZeroDropTransistor,
    run_waveform, simulate_1r, simulate_1t1r,
)


def law():
    # Synthetic parameters only: these are not fits to either measured device.
    return SharedHbnLaw(
        leakage_scale_a=1e-8, leakage_alpha_per_v=1.2,
        on_conductance_s=1e-2, weights=(0.4, 0.6),
        set_thresholds_v=(0.5, 0.7), reset_thresholds_v=(0.3, 0.5),
        set_rate_per_v=2.0, reset_rate_per_v=1.5,
        acceleration_per_v=2.0,
    )


class LinearTransistor:
    """Synthetic passive load with an independently checkable Ohmic law."""
    def __init__(self, resistance=1000.):
        self.resistance = resistance

    def current(self, voltage, gate_voltage, side="set"):
        del gate_voltage, side
        return abs(voltage) / self.resistance

    def drain_current(self, voltage, gate_voltage):
        del gate_voltage
        return voltage / self.resistance

    def voltage_for_current(self, current, gate_voltage, side="set"):
        del gate_voltage, side
        return abs(current) * self.resistance


class SharedHbnTests(unittest.TestCase):
    def test_material_law_has_no_topology_or_gate_arguments(self):
        for method in (SharedHbnLaw, SharedHbnLaw.current, SharedHbnLaw.advance):
            names = inspect.signature(method).parameters
            self.assertFalse(set(names) & {"topology", "gate_voltage", "device", "mode", "hard"})
        material = law()
        state = np.array([.2, .7])
        for voltage in (-1., -.2, 0., .2, 1.):
            expected = (1e-8 * np.sinh(1.2 * voltage)
                        + .01 * np.dot([.4, .6], state) * voltage)
            self.assertAlmostEqual(material.current(voltage, state), expected)

    def test_zero_drop_transistor_reproduces_bare_trace_bitwise(self):
        voltage = np.r_[np.linspace(0., 2., 17), np.linspace(1.875, 0., 16),
                        np.linspace(-.125, -1.5, 12), np.linspace(-1.375, 0., 12)]
        initial = np.array([.01, .02])
        bare = simulate_1r(voltage, initial, law(), max_step_v=.03)
        series = simulate_1t1r(
            voltage, initial, law(), ZeroDropTransistor(), 1.1, max_step_v=.03)
        for key in ("current_a", "hbn_voltage_v", "state", "final_state"):
            np.testing.assert_array_equal(bare[key], series[key])
        self.assertFalse(bare["jointly_calibrated"])

    def test_compliance_is_voltage_feedback_not_readout_clipping(self):
        material = law()
        state = np.array([1., 1.])
        limited = BareCircuit(compliance_a=1e-3).solve(5., material, state)
        self.assertAlmostEqual(limited.current_a, 1e-3, delta=1e-14)
        self.assertAlmostEqual(limited.hbn_voltage_v, .1, delta=2e-6)
        self.assertAlmostEqual(limited.current_a,
                               material.current(limited.hbn_voltage_v, state), delta=1e-14)
        self.assertAlmostEqual(limited.command_voltage_v,
                               limited.supply_voltage_v + limited.compliance_drop_v)
        self.assertEqual(limited.transistor_voltage_v, 0.)
        voltage = np.linspace(0., 2., 21)
        initial = np.array([.1, .1])
        free = simulate_1r(voltage, initial, material, max_step_v=.02)
        capped = simulate_1r(voltage, initial, material, compliance_a=1e-5, max_step_v=.02)
        self.assertLess(np.max(capped["final_state"]), np.min(free["final_state"]))
        np.testing.assert_array_equal(initial, [.1, .1])

    def test_transistor_and_compliance_obey_same_material_and_kcl(self):
        material = law()
        transistor = LinearTransistor()
        for voltage in (-2., 2.):
            for compliance in (None, 2e-5):
                point = TransistorCircuit(transistor, 1.1, compliance).solve(
                    voltage, material, np.array([.3, .6]))
                self.assertAlmostEqual(point.current_a,
                                       material.current(point.hbn_voltage_v, [.3, .6]), delta=1e-13)
                self.assertAlmostEqual(point.current_a,
                                       transistor.drain_current(point.transistor_voltage_v, 1.1), delta=1e-13)
                self.assertAlmostEqual(point.supply_voltage_v,
                                       point.hbn_voltage_v + point.transistor_voltage_v)
                self.assertAlmostEqual(point.command_voltage_v,
                                       point.supply_voltage_v + point.compliance_drop_v)

    def test_measured_transistor_coupling_uses_the_same_frozen_material(self):
        from vulcan2d.transistor import measured_transistor_mapping
        material = replace(law(), set_rate_per_v=0., reset_rate_per_v=0.)
        mapping = measured_transistor_mapping()
        voltage = [0., .2, 1., 0., -.2, -1., 0.]
        state = [.3, .6]
        result = simulate_1t1r(voltage, state, material, mapping, 1.1, max_step_v=.2)
        np.testing.assert_array_equal(result["state"], np.tile(state, (len(voltage), 1)))
        for i, vh, vt in zip(result["current_a"], result["hbn_voltage_v"],
                             result["transistor_voltage_v"]):
            self.assertAlmostEqual(i, material.current(vh, state), delta=1e-14)
            self.assertAlmostEqual(i, mapping.drain_current(vt, 1.1), delta=1e-13)

    def test_loaded_positive_rate_cannot_disappear_in_midpoint_predictor(self):
        from vulcan2d.transistor import measured_transistor_mapping
        material = SharedHbnLaw(
            leakage_scale_a=9.99e-11, leakage_alpha_per_v=3.315,
            on_conductance_s=1. / 96., weights=(1.,),
            set_thresholds_v=(2.675,), reset_thresholds_v=(.55,),
            set_rate_per_v=2., reset_rate_per_v=2., acceleration_per_v=4.)
        mapping = measured_transistor_mapping()
        result = simulate_1t1r([3., 3.02], [0.], material, mapping, 1.1,
                               initial_command_voltage_v=3., max_step_v=.02)
        self.assertGreater(result["hbn_voltage_v"][0], 2.675)
        self.assertGreater(result["final_state"][0], 0.)

    def test_loaded_trajectory_matches_independent_path_ode(self):
        from scipy.integrate import solve_ivp
        from scipy.optimize import brentq
        from vulcan2d.transistor import measured_transistor_mapping
        material = SharedHbnLaw(
            leakage_scale_a=9.99e-11, leakage_alpha_per_v=3.315,
            on_conductance_s=1. / 96., weights=(1.,),
            set_thresholds_v=(2.675,), reset_thresholds_v=(.55,),
            set_rate_per_v=2., reset_rate_per_v=2., acceleration_per_v=4.)
        mapping = measured_transistor_mapping()
        commands = np.linspace(3., 3.05, 6)

        def reference_rhs(command, state):
            # Independent scalar load root and differential rate equation;
            # no production advance or circuit solver is used here.
            x = float(np.clip(state[0], 0., 1.))
            vh = brentq(lambda v: 9.99e-11 * np.sinh(3.315 * v)
                        + x * v / 96. - mapping.drain_current(command - v, 1.1),
                        0., command, xtol=1e-13)
            rate = 2. * np.sinh(4. * max(vh - 2.675, 0.))
            return [(1. - x) * rate]

        reference = solve_ivp(reference_rhs, (commands[0], commands[-1]), [0.],
                              method="Radau", rtol=1e-9, atol=1e-12,
                              t_eval=commands, max_step=.001)
        self.assertTrue(reference.success)
        result = simulate_1t1r(commands, [0.], material, mapping, 1.1,
                               initial_command_voltage_v=3., max_step_v=.02)
        np.testing.assert_allclose(result["state"][:, 0], reference.y[0],
                                   rtol=2e-4, atol=2e-8)

    def test_set_reset_carries_state_and_segmented_run_is_identical(self):
        voltage = np.r_[np.linspace(0., 2., 17), np.linspace(1.875, 0., 16),
                        np.linspace(-.125, -1.5, 12), np.linspace(-1.375, 0., 12)]
        split = 32
        initial = [.01, .02]
        full = simulate_1r(voltage, initial, law(), max_step_v=.025)
        first = simulate_1r(voltage[:split + 1], initial, law(), max_step_v=.025)
        second = simulate_1r(
            voltage[split:], first["final_state"], law(), max_step_v=.025,
            initial_command_voltage_v=voltage[split])
        np.testing.assert_array_equal(second["state"][0], first["final_state"])
        np.testing.assert_array_equal(np.r_[first["current_a"], second["current_a"][1:]],
                                      full["current_a"])
        np.testing.assert_array_equal(second["final_state"], full["final_state"])
        self.assertTrue(np.all(second["final_state"] < first["final_state"]))

    def test_current_and_state_converge_under_path_refinement(self):
        voltage = np.r_[np.linspace(0., 2., 11), np.linspace(1.8, 0., 10)]
        material = law()
        # With adaptation, reducing a cap above the accepted step size alone
        # need not reduce error. Tighten the actual error controls as well;
        # retain the original 2e-4 absolute state accuracy requirement.
        settings = ((.1, 1e-6, 1e-3), (.05, 1e-8, 1e-5),
                    (.025, 1e-9, 1e-6), (.0125, 1e-10, 1e-7))
        values = [simulate_1t1r(
            voltage, [.01, .02], material, LinearTransistor(), 1.1,
            max_step_v=step, state_atol=atol, state_rtol=rtol)
            for step, atol, rtol in settings]
        errors = [np.max(abs(row["state"] - values[-1]["state"])) for row in values[:-1]]
        self.assertGreater(errors[0], errors[1])
        self.assertGreater(errors[1], errors[2])
        self.assertLess(errors[2], 2e-4)
        self.assertTrue(np.all((values[-1]["state"] >= 0.) & (values[-1]["state"] <= 1.)))

    def test_invalid_inputs_are_rejected(self):
        for initial in ([1.1, .2], [.2], [np.nan, .3]):
            with self.assertRaises(ValueError):
                simulate_1r([0., 1.], initial, law())
        for compliance in (0., -1., np.nan):
            with self.assertRaises(ValueError):
                BareCircuit(compliance)
        for step in (0., -1., np.nan):
            with self.assertRaises(ValueError):
                simulate_1r([0., 1.], [.1, .2], law(), max_step_v=step)


if __name__ == "__main__":
    unittest.main()
