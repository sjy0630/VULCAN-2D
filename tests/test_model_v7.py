import unittest

import numpy as np

from vulcan2d import model_v7 as M7
from vulcan2d.reset_identification import (
    identify_reset_state,
    reconstruct_reset,
)
from vulcan2d.transistor import load_reverse_lookup


class ModelV7Tests(unittest.TestCase):
    def test_reverse_lookup_domain_and_inverse_are_measured(self):
        lookup = load_reverse_lookup()
        self.assertAlmostEqual(lookup.vg_min, 0.0)
        self.assertAlmostEqual(lookup.vg_max, 5.5)
        self.assertAlmostEqual(lookup.vds_max, 2.0)
        for gate in (0.9, 1.1, 1.3):
            current = lookup.current(0.2, gate)
            self.assertAlmostEqual(
                lookup.voltage_for_current(current, gate), 0.2, places=10)

    def test_reset_mapping_does_not_apply_positive_gate_offset(self):
        mapping = M7.transistor_mapping()
        self.assertEqual(mapping.reset_policy, "measured_reverse")
        self.assertAlmostEqual(mapping.effective_gate_voltage(1.1, "reset"), 1.1)
        self.assertNotAlmostEqual(mapping.effective_gate_voltage(1.1, "set"), 1.1)

    def test_reset_divider_obeys_current_continuity(self):
        p = M7.Params()
        mapping = M7.transistor_mapping(p)
        current, vh = M7.divider_reset(
            1.0, 0.4, 1e-7, 1e-10, 1.1, p=p, mapping=mapping)
        transistor = mapping.current(1.0 - vh, 1.1, side="reset")
        self.assertAlmostEqual(current, transistor, delta=2e-10)

    def test_synthetic_state_is_recovered_from_terminal_loop(self):
        p = M7.Params(reset_e50_sigma_mvc=0.0)
        wave = np.concatenate([
            np.arange(0.0, -1.5 - 1e-12, -0.02),
            np.arange(-1.48, 0.0 + 1e-12, 0.02),
        ])
        simulated = M7.run_identified_reset(
            wave, 1.1, 1e-7, 1e-10, p=p,
            e50_mvc=p.reset_e50_mvc,
            width_mvc=p.reset_width_mvc,
        )
        reconstructed = reconstruct_reset(
            simulated["Vs"], simulated["Is"], 1.1, M7.transistor_mapping(p))
        state = identify_reset_state(
            reconstructed,
            p.reset_alpha_lrs_identified,
            p.reset_alpha_hrs_identified,
        )
        self.assertLess(abs(state.e50_mvc - p.reset_e50_mvc), 0.18)
        self.assertGreater(state.e10_mvc, state.e50_mvc)
        self.assertGreater(state.e50_mvc, state.e90_mvc)

    def test_reset_state_is_irreversible_on_return(self):
        p = M7.Params(reset_e50_sigma_mvc=0.0)
        wave = np.concatenate([
            np.arange(0.0, -1.5 - 1e-12, -0.02),
            np.arange(-1.48, 0.0 + 1e-12, 0.02),
        ])
        result = M7.run_identified_reset(wave, 1.1, 1e-7, 1e-10, p=p)
        apex = int(np.argmax(np.abs(wave)))
        self.assertTrue(np.all(np.diff(result["pbs"][:apex + 1]) <= 1e-12))
        np.testing.assert_allclose(result["pbs"][apex:], result["pbs"][apex])


if __name__ == "__main__":
    unittest.main()
