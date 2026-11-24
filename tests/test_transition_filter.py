import unittest

import numpy as np

from vulcan2d import validate_transition_regions as audit


class TransitionFilterTest(unittest.TestCase):
    def test_filter_does_not_cross_limits_or_branches(self):
        voltage = np.r_[np.arange(9), np.arange(7, -1, -1)] * .1
        current = np.r_[0., 2e-8, 8e-6, 2e-8, 1e-3, 1e-3, 1e-3,
                        7e-7, 8e-7, 9e-7, 8e-7, 7e-7, 6e-7, 5e-7,
                        4e-7, 3e-7, 0.]
        old = current.copy()
        masks, _ = audit.prepare_curve(voltage, current, "SET", smu_limit=1e-3)
        filtered, flags, _ = audit.filter_unclamped(voltage, current, "SET", masks)
        np.testing.assert_array_equal(current, old)
        self.assertAlmostEqual(filtered[2], 2e-8)
        self.assertFalse(flags["use_unclamped"][4:7].any())
        np.testing.assert_array_equal(filtered[3:10], old[3:10])

    def test_near_cap_marks_only_persistent_runs_within_a_branch(self):
        voltage = np.r_[np.arange(6), np.arange(4, -1, -1)] * .1
        current = 1e-7 * np.r_[0., 1., 2., 3., 4., 5., 4., 3., 2., 1., 0.]
        masks, _ = audit.prepare_curve(voltage, current, "SET")
        masks["current_to_transistor_cap"][:] = .4
        masks["current_to_transistor_cap"][[1, 2, 3, 5, 6, 7]] = .96
        filtered, flags, _ = audit.filter_unclamped(voltage, current, "SET", masks)
        self.assertEqual(np.flatnonzero(flags["transistor_near_cap"]).tolist(), [1, 2, 3])
        np.testing.assert_array_equal(filtered, current)


if __name__ == "__main__":
    unittest.main()
