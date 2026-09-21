import sys
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from vulcan2d.barrier_state import supply_conductance, forward_circuit
from vulcan2d.barrier_height_state import build_height_tables


class HeightStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = build_height_tables([0., 2.], max_voltage=.3, min_height_ev=.2, max_height_ev=1.,
                                         voltage_step=.05, height_step_ev=.1, cache=False)

    def test_nodes_match_original_supply_and_ballistic_limit(self):
        table = self.tables[2.]
        for k, v in enumerate(table.voltage):
            for j, h in enumerate(table.height):
                self.assertAlmostEqual(table.log_conductance[k, j], np.log10(float(supply_conductance(v, [2.], h)[0])), places=12)
        np.testing.assert_allclose(self.tables[0.].log_conductance, 0., atol=1e-10)

    def test_implicit_height_derivative_and_zero_current(self):
        table = self.tables[2.]; va = np.array([0., .1137, .1937, .2817]); heights = np.array([.43, .43, .63, .73])
        tv = np.linspace(0., 1., 201); ti = 1e-4*tv
        def predict(height):
            return forward_circuit(va, lambda v:table.internal(v, -3., height), tv, ti)
        base = predict(heights); delta = 1e-6
        plus, minus = predict(heights+delta)['current_a'], predict(heights-delta)['current_a']
        self.assertEqual(base['current_a'][0], 0.)
        np.testing.assert_allclose(base['dlog10_current_dlocal'][1:], np.log10(plus[1:]/minus[1:])/(2*delta), rtol=2e-4, atol=1e-6)
        self.assertLess(base['max_kcl_relative_error'], 1e-8)

    def test_outside_height_domain_is_not_clipped(self):
        with self.assertRaises(ValueError):
            self.tables[2.].internal([.1], -3., [.01])


if __name__ == '__main__':
    unittest.main()
