import unittest

import numpy as np

from vulcan2d import model_v4 as M


class ModelV4Tests(unittest.TestCase):
    def test_gate_current_anchors_are_monotone_and_exact(self):
        p = M.Params()
        values = [M.gate_current_ceiling(vg, p) for vg in (0.9, 1.1, 1.3)]
        self.assertTrue(np.all(np.diff(values) > 0))
        np.testing.assert_allclose(values, [p.Icc_low, p.Icc_ref, p.Icc_high],
                                   rtol=1e-12)

    def test_reset_current_anchors_are_monotone_and_exact(self):
        p = M.Params()
        values = [M.reset_peak_anchor(vg, p) for vg in (0.9, 1.1, 1.3)]
        self.assertTrue(np.all(np.diff(values) > 0))
        np.testing.assert_allclose(
            values, [p.Ireset_peak_low, p.Ireset_peak_ref, p.Ireset_peak_high],
            rtol=1e-12,
        )

    def test_xtem_geometry_sets_field_but_not_patch_count(self):
        p = M.Params()
        self.assertEqual(p.hbn_layers, 18)
        self.assertEqual(p.K, 10)
        self.assertAlmostEqual(p.hbn_thickness_nm / p.hbn_layers, 1.0 / 3.0)
        self.assertAlmostEqual(M.hbn_field_MV_cm(1.0, p), 10.0 / 6.0)
        thick = M.Params(hbn_thickness_nm=12.0)
        self.assertAlmostEqual(M._state_drive(1.0, 1.0, thick), 0.5)

    def test_soft_update_is_exposure_composable(self):
        p = M.Params()
        phi = np.array([0.02, 0.2, 0.7])
        thresholds = np.array([1.0, 1.2, 1.4])
        one = M.step_soft(phi, 1.8, 1.0, p.T0, thresholds, "set", p)
        half = M.step_soft(phi, 1.8, 0.5, p.T0, thresholds, "set", p)
        two_halves = M.step_soft(half, 1.8, 0.5, p.T0, thresholds, "set", p)
        np.testing.assert_allclose(one, two_halves, rtol=1e-12, atol=1e-12)

    def test_standalone_1r_enters_hard_compliance_regime(self):
        p = M.Params()
        self.assertAlmostEqual(p.Icomp_1r, 10e-3)
        cycles = M.simulate_1r_set_ensemble(p, n_cycles=12, seed=8)
        crossings = []
        for cycle in cycles:
            current = np.abs(cycle["Is"])
            hit = np.where(current >= 0.5e-3)[0]
            self.assertTrue(len(hit))
            crossings.append(cycle["Vs"][hit[0]])
            self.assertLessEqual(np.max(current), p.Icomp_1r * (1 + 1e-12))
            self.assertGreater(cycle["hard"][-1], 0.99)
        self.assertGreater(np.median(crossings), 2.4)
        self.assertLess(np.median(crossings), 3.0)

    def test_multigate_reset_is_sequential_and_gate_ordered(self):
        p = M.Params()
        peaks = []
        for gate, step, vpeak in ((0.9, 0.01, 1.5),
                                  (1.1, 0.02, 1.7),
                                  (1.3, 0.01, 1.5)):
            cycles, _ = M.simulate_1t1r_reset_ensemble(
                p, gate, n_cycles=8, seed=3, set_step=0.02,
                reset_step=step, reset_vpeak=vpeak,
            )
            peak = np.median([np.max(np.abs(c["Is"])) for c in cycles])
            peaks.append(peak)
            state_drops = []
            for cycle in cycles:
                initial = np.mean(cycle["phi_initial"])
                final = np.mean(cycle["phi_final"])
                self.assertLessEqual(final, initial + 1e-12)
                state_drops.append(initial - final)
                self.assertEqual(len(cycle["Ehs_MV_cm"]), len(cycle["Vs"]))
            self.assertGreater(np.median(state_drops), 0.0)
        self.assertTrue(np.all(np.diff(peaks) > 0))

    def test_standalone_1r_reset_ruptures_hard_links(self):
        p = M.Params()
        self.assertIsNone(p.Icomp_1r_reset)
        cycles = M.simulate_1r_reset_ensemble(p, n_cycles=10, seed=8)
        peak_voltages = []
        for cycle in cycles:
            hard = cycle["hard"]
            self.assertTrue(np.all(np.diff(hard) <= 1e-12))
            self.assertGreater(hard[0], 0.99)
            self.assertLess(hard[-1], 1e-5)
            self.assertEqual(cycle["links"].shape[1], p.hard_reset_links)
            apex = int(np.argmax(np.abs(cycle["Vs"])))
            peak = int(np.argmax(np.abs(cycle["Is"][:apex + 1])))
            peak_voltages.append(abs(cycle["Vs"][peak]))
        self.assertGreater(np.median(peak_voltages), 0.35)
        self.assertLess(np.median(peak_voltages), 0.75)

    def test_regime_summary_does_not_claim_a_unique_species(self):
        text = M.regime_summary()["interpretation"]["hard"]
        self.assertIn("chemical species unresolved", text)
        xtem = M.regime_summary()["interpretation"]["xtem"]
        self.assertIn("no unique migrating species", xtem)


if __name__ == "__main__":
    unittest.main()
