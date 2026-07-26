import os
import unittest

import numpy as np

from vulcan2d import model as M
from vulcan2d import serve
from vulcan2d import calibrate
from vulcan2d.transistor import TransistorLookup


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class ModelTests(unittest.TestCase):
    @staticmethod
    def _synthetic_lookup():
        payload = {
            "schema_version": 1,
            "curves": [
                {
                    "gate_voltage_v": 1.0,
                    "drain_voltage_v": [0.0, 1.0, 2.0, 1.0, 0.0],
                    "drain_current_a": [1e-8, 1e-5, 2e-5, 1.2e-5, 2e-8],
                },
                {
                    "gate_voltage_v": 2.0,
                    "drain_voltage_v": [0.0, 1.0, 2.0, 1.0, 0.0],
                    "drain_current_a": [2e-8, 3e-5, 5e-5, 3.2e-5, 3e-8],
                },
            ],
        }
        return TransistorLookup.from_payload(payload)

    def test_measured_transistor_lookup_interpolates_and_preserves_origin(self):
        lookup = self._synthetic_lookup()
        self.assertEqual(lookup.current(0.0, 1.5), 0.0)
        self.assertAlmostEqual(lookup.current(1.0, 1.0), 1.1e-5)
        self.assertAlmostEqual(lookup.current(1.0, 1.5), 2.1e-5)
        with self.assertRaises(ValueError):
            lookup.current(1.0, 0.9)

    def test_measured_transistor_lookup_accepts_duplicated_apex(self):
        payload = {
            "schema_version": 1,
            "curves": [
                {"gate_voltage_v": 1.0,
                 "drain_voltage_v": [0.0, 1.0, 1.0, 0.0],
                 "drain_current_a": [0.0, 1e-5, 1.1e-5, 0.0]},
                {"gate_voltage_v": 2.0,
                 "drain_voltage_v": [0.0, 1.0, 1.0, 0.0],
                 "drain_current_a": [0.0, 2e-5, 2.1e-5, 0.0]},
            ],
        }
        lookup = TransistorLookup.from_payload(payload)
        self.assertAlmostEqual(lookup.current(1.0, 1.0), 1.05e-5)

    def test_divider_accepts_measured_transistor_lookup(self):
        p = M.Params()
        lookup = self._synthetic_lookup()
        current, vh = M.divider(
            1.5, 1e-7, p, "set", 1.0,
            transistor_lookup=lookup, gate_voltage=1.5)
        transistor_current = lookup.current(1.5 - abs(vh), 1.5)
        self.assertAlmostEqual(current, transistor_current, delta=2e-10)

    def test_triangle_has_one_apex_and_monotone_branches(self):
        for peak in (5.0, -1.7):
            wave = M.triangle(peak)
            apex = int(np.argmax(np.abs(wave)))
            direction = np.sign(peak)
            self.assertTrue(np.all(direction * np.diff(wave[:apex + 1]) > 0))
            self.assertTrue(np.all(direction * np.diff(wave[apex:]) < 0))
            self.assertEqual(np.count_nonzero(wave == wave[apex]), 1)

    def test_hrs_and_lrs_random_terms_are_decoupled(self):
        p = M.Params()
        w = np.array([0.4, 0.6])
        low_residual = np.array([-1.0, -1.0])
        high_residual = np.array([1.0, 1.0])

        hrs_low = M.G_ensemble(np.zeros(2), low_residual, w, p, p.Gon)
        hrs_high = M.G_ensemble(np.zeros(2), high_residual, w, p, p.Gon)
        lrs_low = M.G_ensemble(np.ones(2), low_residual, w, p, p.Gon)
        lrs_high = M.G_ensemble(np.ones(2), high_residual, w, p, p.Gon)

        self.assertGreater(hrs_high, hrs_low)
        self.assertAlmostEqual(lrs_high, lrs_low)

    def test_calibrated_profile_matches_multi_seed_targets(self):
        path = os.path.join(ROOT, "vulcan2d", "vulcan2d_calibrated.npz")
        p = M.Params.from_npz(path)
        result = calibrate.audit(p)

        targets = {
            "Vset": (1.298, 25.0, 0.05, 3.0),
            "Vreset": (-1.075, 24.1, 0.05, 3.0),
            "R_HRS": (2.045e8, 45.5, 0.10, 5.0),
            "R_LRS": (2.939e5, 28.4, 0.10, 5.0),
            "Icc": (5.146e-5, 1.0, 0.03, 1.0),
        }
        for column, (mean, target_cv, mean_tol, cv_tol) in targets.items():
            actual_mean = np.median(result[column + "_mean"])
            actual_cv = np.median(result[column + "_cv"])
            self.assertLess(abs(actual_mean / mean - 1.0), mean_tol, column)
            self.assertLess(abs(actual_cv - target_cv), cv_tol, column)

        self.assertGreater(np.median(result["Vset_slope"]), 0.0)
        self.assertLess(np.median(result["lnR_HRS_slope"]), 0.0)
        self.assertGreater(np.median(result["lnR_LRS_slope"]), 0.0)

    def test_ui_default_variability_preserves_calibration(self):
        p = M.Params()
        expected = (p.sigma_lnG, p.sigma_Gon, p.sigma_theta, p.m_set, p.m_reset)
        serve._apply_overrides(p, {"sigma": [str(serve.SIGMA_UI_DEFAULT)]})
        actual = (p.sigma_lnG, p.sigma_Gon, p.sigma_theta, p.m_set, p.m_reset)
        np.testing.assert_allclose(actual, expected, rtol=1e-9)

    def test_cycle_resampling_keeps_the_same_device_structure(self):
        p = M.Params()
        _, frozen_a = M.simulate_cycles(p, n_cycles=1, seed=1)
        _, frozen_b = M.simulate_cycles(p, n_cycles=1, seed=2)
        np.testing.assert_allclose(frozen_a["w"], frozen_b["w"])
        np.testing.assert_allclose(frozen_a["dtheta"], frozen_b["dtheta"])

    def test_simulation_exposes_internal_voltage_traces(self):
        cycles, _ = M.simulate_cycles(M.Params(), n_cycles=1, seed=3)
        cycle = cycles[0]
        self.assertEqual(len(cycle["Vhs"]), len(cycle["Vs"]))
        self.assertEqual(len(cycle["Vhr"]), len(cycle["Vr"]))
        self.assertTrue(np.all(np.abs(cycle["Vhs"]) <= np.abs(cycle["Vs"]) + 1e-12))
        self.assertTrue(np.all(np.abs(cycle["Vhr"]) <= np.abs(cycle["Vr"]) + 1e-12))


if __name__ == "__main__":
    unittest.main()
