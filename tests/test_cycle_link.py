import unittest

import numpy as np

from vulcan2d import cycle_link as CL
from vulcan2d import model_v8 as M8


class CycleLinkTests(unittest.TestCase):
    def test_set_exposure_uses_only_forward_branch(self):
        voltage = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
        current = np.array([0.0, 1.0, 3.0, 5.0, 9.0])
        changed_return = current.copy()
        changed_return[3:] *= 1e6

        exposure = CL.set_forward_exposure(voltage, current)
        changed = CL.set_forward_exposure(voltage, changed_return)

        self.assertAlmostEqual(exposure, 2.5)
        self.assertAlmostEqual(changed, exposure)

    def test_transfer_law_recovers_a_centered_log_relation(self):
        exposure = 10.0 ** np.array([-3.0, -2.5, -2.0, -1.5, -1.0])
        conductance = 10.0 ** (-6.0 + 0.4 * (np.log10(exposure) + 2.0))

        law = CL.fit_set_reset_transfer(exposure, conductance)

        self.assertAlmostEqual(law.log10_exposure_ref, -2.0)
        self.assertAlmostEqual(law.log10_g_ref, -6.0)
        self.assertAlmostEqual(law.slope, 0.4)
        self.assertAlmostEqual(law.predict(1e-2), 1e-6)

    def test_endpoint_attractor_fits_the_log_domain_center(self):
        law = CL.fit_reset_endpoint_attractor(
            [1e-9, 1e-8, 1e-7], gate_voltage=1.1,
            protocol_label="paired DC RESET")

        self.assertAlmostEqual(law.log10_g_hrs_ref, -8.0)
        self.assertAlmostEqual(law.predict(), 1e-8)
        self.assertAlmostEqual(law.predict(0.3), 10 ** -7.7)
        self.assertEqual(law.n_training, 3)

    def test_endpoint_attractor_leave_one_out_excludes_target(self):
        predicted, laws = CL.leave_one_out_endpoint_predictions(
            10.0 ** np.array([-9.0, -8.0, -7.0, -6.0]),
            gate_voltage=1.1,
            protocol_label="paired DC RESET",
        )

        np.testing.assert_allclose(
            np.log10(predicted),
            np.array([-7.0, -7.333333333333333, -7.666666666666667, -8.0]),
            rtol=0.0,
            atol=1e-12,
        )
        self.assertTrue(all(law.n_training == 3 for law in laws))

    def test_sheet_pairing_is_numeric_not_lexicographic(self):
        pairs = CL.pair_cycle_sheets(
            ["S10", "S2", "S1"], ["R2", "R1", "R10"])

        self.assertEqual(
            pairs,
            [(1, "S1", "R1"), (2, "S2", "R2"), (10, "S10", "R10")],
        )

    def test_leave_one_out_predictions_exclude_each_target_label(self):
        exposure = 10.0 ** np.array([-3.0, -2.5, -2.0, -1.5, -1.0])
        conductance = 10.0 ** (-6.0 + 0.4 * (np.log10(exposure) + 2.0))

        predictions, laws = CL.leave_one_out_transfer_predictions(
            exposure, conductance)

        np.testing.assert_allclose(predictions, conductance, rtol=1e-12)
        self.assertEqual(len(laws), len(exposure))
        self.assertTrue(all(law.n_training == len(exposure) - 1
                            for law in laws))

    def test_set_conditioned_reset_does_not_require_reset_current(self):
        law = CL.SetResetTransferLaw(
            log10_exposure_ref=0.0,
            log10_g_ref=-7.0,
            slope=0.5,
            residual_sigma_decade=0.1,
            n_training=8,
        )
        set_voltage = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
        set_current = np.array([0.0, 0.5, 1.5, 8.0, 7.0])
        reset_voltage = np.concatenate([
            np.arange(0.0, -1.5 - 1e-12, -0.02),
            np.arange(-1.48, 0.0 + 1e-12, 0.02),
        ])

        result = CL.run_set_conditioned_cycle(
            set_voltage,
            set_current,
            reset_voltage,
            gate_voltage=1.1,
            transfer_law=law,
            p=M8.Params(reset_e50_sigma_mvc=0.0),
        )

        expected_exposure = 1.25
        expected_g = law.predict(expected_exposure)
        self.assertAlmostEqual(result["set_exposure_A_V"], expected_exposure)
        self.assertAlmostEqual(result["g_lrs_a"], expected_g)
        self.assertEqual(result["state_source"],
                         "predicted_from_preceding_set")
        self.assertEqual(result["endpoint_source"],
                         "predicted_from_preceding_set_and_gate")
        self.assertEqual(len(result["Is"]), len(reset_voltage))

    def test_protocol_endpoint_is_independent_of_predicted_lrs_scale(self):
        endpoint_law = CL.ResetEndpointAttractorLaw(
            gate_voltage_v=1.1,
            protocol_label="paired DC RESET",
            log10_g_hrs_ref=-9.0,
            residual_sigma_decade=0.2,
            n_training=50,
            reset_peak_abs_v=1.5,
        )
        set_voltage = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
        set_current = np.array([0.0, 0.5, 1.5, 8.0, 7.0])
        reset_voltage = np.concatenate([
            np.arange(0.0, -1.5 - 1e-12, -0.02),
            np.arange(-1.48, 0.0 + 1e-12, 0.02),
        ])
        weak_lrs = CL.SetResetTransferLaw(
            0.0, -7.0, 0.0, 0.1, 8)
        strong_lrs = CL.SetResetTransferLaw(
            0.0, -6.0, 0.0, 0.1, 8)

        weak = CL.run_set_conditioned_cycle(
            set_voltage, set_current, reset_voltage, 1.1, weak_lrs,
            endpoint_law=endpoint_law,
            p=M8.Params(reset_e50_sigma_mvc=0.0),
        )
        strong = CL.run_set_conditioned_cycle(
            set_voltage, set_current, reset_voltage, 1.1, strong_lrs,
            endpoint_law=endpoint_law,
            p=M8.Params(reset_e50_sigma_mvc=0.0),
        )

        self.assertNotEqual(weak["g_lrs_a"], strong["g_lrs_a"])
        self.assertAlmostEqual(weak["g_hrs_a"], 1e-9)
        self.assertAlmostEqual(strong["g_hrs_a"], 1e-9)
        self.assertEqual(weak["endpoint_source"],
                         "protocol_conditioned_hrs_attractor")
        self.assertEqual(weak["endpoint_protocol_label"],
                         "paired DC RESET")
        self.assertAlmostEqual(weak["endpoint_sigma_decade"], 0.2)

    def test_protocol_endpoint_rejects_a_different_gate(self):
        endpoint_law = CL.ResetEndpointAttractorLaw(
            gate_voltage_v=1.1,
            protocol_label="paired DC RESET",
            log10_g_hrs_ref=-9.0,
            residual_sigma_decade=0.2,
            n_training=50,
        )

        with self.assertRaisesRegex(ValueError, "gate voltage"):
            endpoint_law.validate_gate(0.9)

    def test_protocol_endpoint_rejects_a_different_reset_peak(self):
        endpoint_law = CL.ResetEndpointAttractorLaw(
            gate_voltage_v=1.1,
            protocol_label="paired DC RESET",
            log10_g_hrs_ref=-9.0,
            residual_sigma_decade=0.2,
            n_training=50,
        )

        with self.assertRaisesRegex(ValueError, "RESET peak"):
            endpoint_law.validate_protocol([0.0, -1.3, 0.0])


if __name__ == "__main__":
    unittest.main()
