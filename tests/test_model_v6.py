import unittest

import numpy as np

from vulcan2d import model_v4 as M5
from vulcan2d import model_v6 as M6
from vulcan2d.transistor import load_default_lookup


class ModelV6Tests(unittest.TestCase):
    def test_public_lookup_has_expected_provenance_and_domain(self):
        lookup = load_default_lookup()
        np.testing.assert_allclose(lookup.gate_voltages,
                                   [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        self.assertAlmostEqual(lookup.vds_min, 0.0)
        self.assertAlmostEqual(lookup.vds_max, 5.0)

    def test_gate_mapping_uses_endpoints_and_holds_out_1p1V(self):
        p = M6.Params()
        mapping = M6.transistor_mapping(p)
        self.assertAlmostEqual(mapping.effective_gate_voltage(0.9),
                               0.6768348515, places=8)
        self.assertAlmostEqual(mapping.effective_gate_voltage(1.3),
                               1.1149866283, places=8)
        expected_midpoint = 0.5 * (
            mapping.effective_gate_voltage(0.9) +
            mapping.effective_gate_voltage(1.3))
        self.assertAlmostEqual(mapping.effective_gate_voltage(1.1),
                               expected_midpoint, places=12)

    def test_hbn_reset_parameters_are_gate_neutral(self):
        spread = []
        p = M6.Params()
        for family in (
            (p.reset_connected_scale_low, p.reset_connected_scale_ref,
             p.reset_connected_scale_high),
            (p.reset_off_scale_low, p.reset_off_scale_ref,
             p.reset_off_scale_high),
            (p.alpha_reset_low_v4, p.alpha_reset_ref_v4,
             p.alpha_reset_high_v4),
            (p.alpha_reset_on_low_v4, p.alpha_reset_on_ref_v4,
             p.alpha_reset_on_high_v4),
            (p.reset_drive_scale_low, p.reset_drive_scale_ref,
             p.reset_drive_scale_high),
            (p.reset_threshold_factor_low, p.reset_threshold_factor_ref,
             p.reset_threshold_factor_high),
        ):
            spread.append(max(family) - min(family))
        np.testing.assert_allclose(spread, 0.0)
        self.assertEqual(p.gate_path_exponent, 0.0)

    def test_v5_default_path_remains_backward_compatible(self):
        p = M5.Params()
        expected = M5.reset_peak_anchor(1.1, p)
        actual = M5.i_tr(1.0, 1.1, "reset", p)
        self.assertGreater(actual, 0.0)
        self.assertGreater(expected, 0.0)

    def test_v6_load_line_uses_lookup_current(self):
        p = M6.Params()
        mapping = M6.transistor_mapping(p)
        state = M5._G_ensemble(
            np.full(p.K, 0.2), np.zeros(p.K), np.full(p.K, 1 / p.K),
            p, p.Gon, "reset", 1.1)
        current, vh = M5.divider_1t1r(
            -1.0, state, 1.1, p, "reset", transistor_mapping=mapping)
        transistor_current = mapping.current(1.0 - abs(vh), 1.1, side="reset")
        self.assertAlmostEqual(current, transistor_current, delta=2e-10)

    def test_reset_proxy_limitation_is_explicit(self):
        scope = M6.claim_scope()
        self.assertIn("non-negative V_DS", scope["reset_limitation"])
        self.assertIn("proxy", scope["reset_limitation"])


if __name__ == "__main__":
    unittest.main()
