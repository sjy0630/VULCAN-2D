"""Scientific domain checks for the audited full-cycle public API."""
import unittest
from dataclasses import replace

import numpy as np

from vulcan2d import simulate_calibrated_cycle
from vulcan2d import model_v12 as M12
from vulcan2d import model_v13 as M13


def _waves(points=21):
    set_forward = np.linspace(0.0, 5.0, points)
    reset_forward = np.linspace(0.0, -1.7, points)
    return (np.r_[set_forward, set_forward[-2::-1]],
            np.r_[reset_forward, reset_forward[-2::-1]])


class ModelV13Tests(unittest.TestCase):
    def test_partial_set_does_not_silently_start_a_saturated_reset(self):
        vs, vr = _waves()
        # Artificial domain stress, not an experimental partial-SET claim.
        with self.assertRaisesRegex(ValueError, "partial SET"):
            simulate_calibrated_cycle(
                vs, vr, set_transition_e50_offset_v2=100.0)

    def test_calibrated_currents_match_legacy_in_both_transition_modes(self):
        vs, vr = _waves()
        for mode in ("same_protocol", "cross_gate_prior"):
            with self.subTest(mode=mode):
                legacy = M12.simulate_calibrated_cycle(
                    vs, vr, reset_transition_mode=mode)
                audited = simulate_calibrated_cycle(
                    vs, vr, reset_transition_mode=mode)
                np.testing.assert_array_equal(audited["Is"], legacy["Is"])
                np.testing.assert_array_equal(
                    audited["set"]["xs"], legacy["set"]["xs"])
                np.testing.assert_array_equal(
                    audited["reset"]["pbs"], legacy["reset"]["pbs"])

    def test_distinct_state_coordinates_are_not_a_single_physical_fraction(self):
        vs, vr = _waves()
        result = simulate_calibrated_cycle(vs, vr)
        self.assertNotIn("connected_fraction", result)
        self.assertIn("handoff", result)
        self.assertFalse(result["handoff"]["unified_physical_state"])

    def test_saturation_domain_is_stable_under_grid_refinement(self):
        for points in (11, 81):
            with self.subTest(points=points):
                vs, vr = _waves(points)
                result = simulate_calibrated_cycle(vs, vr)
                self.assertLessEqual(
                    result["handoff"]["reference_current_relative_error"],
                    M13.HANDOFF_SATURATION_TOLERANCE)

    def test_saturation_tolerance_boundary_is_inclusive(self):
        calibration = M12.default_calibration()
        law = calibration.set_transport
        result = {"a_pre": law.pre_amplitude(),
                  "a_connected": law.connected_amplitude(),
                  "xs": np.array([0.999])}
        M13.audit_set_handoff(result, law, calibration.set_reset_link)
        result["xs"][-1] = 0.9989
        with self.assertRaises(M13.HandoffDomainError):
            M13.audit_set_handoff(result, law, calibration.set_reset_link)

    def test_nearly_identical_envelopes_cannot_hide_an_incomplete_state(self):
        calibration = M12.default_calibration()
        law, link = calibration.set_transport, calibration.set_reset_link
        vref = link.reference_voltage_v
        connected = law.connected_amplitude() * vref ** law.connected_exponent
        pre_basis = vref * np.exp(law.beta_pre_per_sqrt_v * np.sqrt(vref))
        result = {"a_pre": 0.999999 * connected / pre_basis,
                  "a_connected": law.connected_amplitude(),
                  "xs": np.array([0.5])}
        with self.assertRaises(M13.HandoffDomainError):
            M13.audit_set_handoff(result, law, link)

    def test_invalid_reference_state_and_envelope_order_are_rejected(self):
        calibration = M12.default_calibration()
        law, link = calibration.set_transport, calibration.set_reset_link
        base = {"a_pre": law.pre_amplitude(),
                "a_connected": law.connected_amplitude(),
                "xs": np.array([1.0])}
        for reference in (0.0, -0.1, np.nan):
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                M13.audit_set_handoff(
                    base, law, replace(link, reference_voltage_v=reference))
        for state in (np.nan, -0.1, 1.1):
            with self.subTest(state=state), self.assertRaises(ValueError):
                M13.audit_set_handoff(dict(base, xs=np.array([state])), law, link)
        with self.assertRaisesRegex(ValueError, "ordered"):
            M13.audit_set_handoff(dict(base, a_pre=1.0), law, link)

    def test_nonfinite_reset_parameters_cannot_pass_the_handoff_audit(self):
        vs, vr = _waves(7)
        for parameter in ("reset_e50_mvc", "reset_width_mvc",
                          "link_residual_offset_decade",
                          "endpoint_residual_offset_decade"):
            with self.subTest(parameter=parameter), self.assertRaises(ValueError):
                simulate_calibrated_cycle(vs, vr, **{parameter: np.nan})

    def test_nonfinite_reset_voltage_is_rejected(self):
        vs, vr = _waves(7)
        vr[2] = np.nan
        with self.assertRaises(ValueError):
            simulate_calibrated_cycle(vs, vr)

    def test_nonfinite_link_calibration_is_rejected(self):
        vs, vr = _waves(7)
        calibration = M12.default_calibration()
        bad_link = replace(calibration.set_reset_link, retention_factor=np.nan)
        with self.assertRaises(ValueError):
            simulate_calibrated_cycle(
                vs, vr, calibration=replace(calibration, set_reset_link=bad_link))

    def test_explicit_reset_threshold_is_not_labeled_as_a_fitted_value(self):
        vs, vr = _waves(7)
        result = simulate_calibrated_cycle(vs, vr, reset_e50_mvc=0.85)
        self.assertEqual(result["reset_transition_source"],
                         "explicit_parameter_override")


if __name__ == "__main__":
    unittest.main()
