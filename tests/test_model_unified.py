"""Scientific invariants for the unified local-connectivity model."""
import inspect
import unittest

import numpy as np

from vulcan2d import model_v12 as legacy
from vulcan2d import model_unified as unified


def _waves(points=21):
    set_forward = np.linspace(0.0, 5.0, points)
    reset_forward = np.linspace(0.0, -1.7, points)
    return (
        np.r_[set_forward, set_forward[-2::-1]],
        np.r_[reset_forward, reset_forward[-2::-1]],
    )


class UnifiedModelTests(unittest.TestCase):
    def test_mean_field_preserves_the_calibrated_terminal_solution(self):
        set_voltage, reset_voltage = _waves()
        for mode in ("same_protocol", "cross_gate_prior"):
            with self.subTest(mode=mode):
                reference = legacy.simulate_calibrated_cycle(
                    set_voltage, reset_voltage,
                    reset_transition_mode=mode)
                result = unified.simulate_calibrated_cycle(
                    set_voltage, reset_voltage,
                    reset_transition_mode=mode)
                np.testing.assert_allclose(
                    result["set"]["Is"], reference["set"]["Is"],
                    rtol=2e-8, atol=1e-15)
                np.testing.assert_allclose(
                    result["set"]["Vhs"], reference["set"]["Vhs"],
                    rtol=2e-8, atol=1e-12)
                np.testing.assert_allclose(
                    result["reset"]["Is"], reference["reset"]["Is"],
                    rtol=2e-8, atol=3e-13)
                np.testing.assert_allclose(
                    result["reset"]["Vhs"], reference["reset"]["Vhs"],
                    rtol=2e-8, atol=6e-9)

    def test_set_endpoint_is_the_exact_reset_initial_state(self):
        set_voltage, reset_voltage = _waves()
        result = unified.simulate_calibrated_cycle(
            set_voltage, reset_voltage,
            spatial=unified.SpatialConfiguration.from_weights(
                [0.05, 0.10, 0.15, 0.20, 0.50]))
        np.testing.assert_array_equal(
            result["set"]["z_local"][-1],
            result["reset"]["z_local"][0])
        self.assertEqual(
            result["handoff"]["state_continuity_absolute_error"], 0.0)
        self.assertTrue(result["handoff"]["unified_physical_state"])

    def test_partial_set_is_propagated_without_silent_saturation(self):
        set_voltage, reset_voltage = _waves()
        result = unified.simulate_calibrated_cycle(
            set_voltage, reset_voltage,
            set_transition_e50_offset_v2=100.0)
        self.assertLess(result["set"]["zs"][-1], 0.01)
        self.assertAlmostEqual(
            result["set"]["zs"][-1], result["reset"]["zs"][0])
        self.assertTrue(
            result["handoff"]["partial_set_supported_mathematically"])
        self.assertFalse(
            result["handoff"]["partial_set_experimentally_calibrated"])

    def test_spatial_coarse_graining_is_exact_at_the_terminal(self):
        set_voltage, reset_voltage = _waves()
        spatial = unified.SpatialConfiguration.from_weights(
            [0.04, 0.06, 0.08, 0.10, 0.12, 0.60], seed=17)
        distributed = unified.simulate_calibrated_cycle(
            set_voltage, reset_voltage, spatial=spatial)
        mean_field = unified.simulate_calibrated_cycle(
            set_voltage, reset_voltage)
        np.testing.assert_allclose(
            distributed["Is"], mean_field["Is"], rtol=2e-8, atol=1e-15)
        np.testing.assert_allclose(
            distributed["zs"],
            distributed["z_local"] @ spatial.weights,
            rtol=0.0, atol=2e-15)

    def test_local_states_follow_the_irreversible_branch_directions(self):
        set_voltage, reset_voltage = _waves(41)
        result = unified.simulate_calibrated_cycle(
            set_voltage, reset_voltage,
            spatial=unified.SpatialConfiguration.from_weights(
                np.linspace(1.0, 3.0, 10)))
        set_apex = int(np.argmax(np.abs(set_voltage)))
        reset_apex = int(np.argmax(np.abs(reset_voltage)))
        self.assertTrue(np.all(np.diff(
            result["set"]["zs"][:set_apex + 1]) >= -2e-12))
        self.assertTrue(np.all(np.diff(
            result["reset"]["zs"][:reset_apex + 1]) <= 2e-12))

    def test_public_prediction_api_has_no_target_current_argument(self):
        parameters = inspect.signature(
            unified.run_unified_full_cycle).parameters
        self.assertNotIn("set_current", parameters)
        self.assertNotIn("reset_current", parameters)

    def test_invalid_spatial_configuration_is_rejected(self):
        for weights in ([1.0, 0.0], [1.0, -1.0], [1.0, np.nan], []):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                unified.SpatialConfiguration(
                    weights, np.arange(len(weights)), np.arange(len(weights)))
        with self.assertRaisesRegex(ValueError, "permutation"):
            unified.SpatialConfiguration(
                [0.5, 0.5], [0, 0], [1, 0])


if __name__ == "__main__":
    unittest.main()
