import unittest
from types import SimpleNamespace
import numpy as np
from vulcan2d.side_charge_circuit import load_roots, forward_initial


class SideChargeCircuitTests(unittest.TestCase):
    def test_all_three_roots_reported_for_nonmonotone_device(self):
        roots = load_roots(1., [0, .2, .4, .6, .8, 1], [0, 1., .1, .1, .8, 1.], [0, 1], [0, 1])
        self.assertEqual(len(roots), 3)
        self.assertEqual([r['stable_load_line'] for r in roots], [True, False, True])

    def test_ohmic_load_intersection(self):
        roots = load_roots(.3, [0, 1], [0, 2e-5], [0, 1], [0, 1e-5])
        self.assertEqual(len(roots), 1)
        self.assertAlmostEqual(roots[0]['internal_voltage_V'], .1, places=12)

    def test_direct_charge_kernel_and_circuit_conservation(self):
        lookup = SimpleNamespace(voltage=np.array([0., 1.]), current=np.array([0., .001]))
        result = forward_initial([-.1, -.2, -.3], [2., .15, .4, .8], [-3., -.08], lookup)
        self.assertEqual(result['ambiguous_count'], 0)
        self.assertLess(result['max_kcl_relative_error'], 1e-10)
        self.assertTrue(all(x > 0 for x in result['predicted_current_A']))


if __name__ == '__main__':
    unittest.main()
