import unittest

import numpy as np

from vulcan2d import model_v8 as M8
from vulcan2d.reset_identification import ReconstructedReset


class ModelV8Tests(unittest.TestCase):
    def test_initial_lrs_estimator_ignores_return_branch(self):
        alpha = 8.0
        vh = np.array([0.0, 0.05, 0.09, 0.10, 0.11, 0.20,
                       0.30, 0.11, 0.10, 0.09, 0.0])
        g_true = 2.5e-8
        current = g_true * np.sinh(alpha * vh)
        cycle = ReconstructedReset(
            gate_voltage_v=1.1,
            voltage_app_v=vh,
            current_a=current,
            voltage_ds_v=np.zeros_like(vh),
            voltage_hbn_v=vh,
            valid=np.ones_like(vh, dtype=bool),
            apex_index=6,
        )
        changed = current.copy()
        changed[7:] *= 1e-5
        changed_cycle = ReconstructedReset(
            gate_voltage_v=cycle.gate_voltage_v,
            voltage_app_v=cycle.voltage_app_v,
            current_a=changed,
            voltage_ds_v=cycle.voltage_ds_v,
            voltage_hbn_v=cycle.voltage_hbn_v,
            valid=cycle.valid,
            apex_index=cycle.apex_index,
        )

        estimate = M8.estimate_initial_lrs(cycle, alpha)
        changed_estimate = M8.estimate_initial_lrs(changed_cycle, alpha)

        self.assertAlmostEqual(estimate, g_true, delta=g_true * 1e-12)
        self.assertAlmostEqual(changed_estimate, estimate, delta=estimate * 1e-12)

    def test_endpoint_ratio_is_a_monotone_gate_interpolation(self):
        p = M8.Params()
        r09 = M8.predicted_hrs_ratio(0.9, p)
        r11 = M8.predicted_hrs_ratio(1.1, p)
        r13 = M8.predicted_hrs_ratio(1.3, p)

        self.assertGreater(r09, r11)
        self.assertGreater(r11, r13)
        self.assertAlmostEqual(np.log10(r11),
                               0.5 * (np.log10(r09) + np.log10(r13)))

    def test_endpoint_ratio_fit_uses_group_medians(self):
        fitted = M8.fit_endpoint_ratio_params(
            gate_voltage=[0.9, 0.9, 0.9, 1.3, 1.3, 1.3],
            g_lrs=[1, 1, 1, 1, 1, 1],
            g_hrs=[0.02, 0.03, 0.90, 0.004, 0.008, 0.50],
            p=M8.Params(),
        )

        self.assertAlmostEqual(M8.predicted_hrs_ratio(0.9, fitted), 0.03)
        self.assertAlmostEqual(M8.predicted_hrs_ratio(1.3, fitted), 0.008)

    def test_read_initialized_reset_does_not_require_an_hrs_endpoint(self):
        p = M8.Params(reset_e50_sigma_mvc=0.0)
        wave = np.concatenate([
            np.arange(0.0, -1.5 - 1e-12, -0.02),
            np.arange(-1.48, 0.0 + 1e-12, 0.02),
        ])

        result = M8.run_read_initialized_reset(
            wave, gate_voltage=1.1, g_lrs_read_a=1e-7, p=p)

        self.assertEqual(len(result["Is"]), len(wave))
        self.assertAlmostEqual(result["g_lrs_a"], 1e-7)
        self.assertAlmostEqual(
            result["g_hrs_a"], 1e-7 * M8.predicted_hrs_ratio(1.1, p))
        self.assertLess(result["g_hrs_a"], result["g_lrs_a"])
        self.assertEqual(result["endpoint_source"],
                         "predicted_from_initial_read_and_gate")


if __name__ == "__main__":
    unittest.main()
