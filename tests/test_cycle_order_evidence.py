import itertools
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'analysis'))
from check_cycle_order_evidence import cusum, permutation_check


class CycleOrderTests(unittest.TestCase):
    def test_statistic_against_direct_group_means_and_scale_invariance(self):
        x = np.array([[1., 8., 4.], [2., 7., 4.], [5., 1., 4.], [4., 3., 4.], [8., 4., 4.]])
        actual = cusum(x); expected = np.zeros_like(actual)
        for k in range(1, len(x)):
            for j in range(2):
                expected[k-1, j] = abs(x[:k, j].mean()-x[k:, j].mean())*np.sqrt(k*(len(x)-k)/len(x))/x[:, j].std(ddof=1)
        np.testing.assert_allclose(actual, expected, atol=1e-14)
        np.testing.assert_allclose(cusum(x*np.array([3., -2., 4.])+np.array([9., 5., 2.])), actual, atol=1e-14)

    def test_joint_permutation_against_small_exact_enumeration(self):
        x = np.array([[0., 1.], [1., 3.], [2., 0.], [5., 4.], [6., 8.], [7., 6.]])
        observed = cusum(x).max(); count = 0; total = 0
        for order in itertools.permutations(range(len(x))):
            count += int(cusum(x[list(order)]).max() >= observed-1e-12); total += 1
        exact = count/total
        sampled = permutation_check(x, 20000, 709)
        self.assertAlmostEqual(sampled['joint_monte_carlo_p'], exact, delta=.015)
        self.assertEqual(len(sampled['per_feature_best_cut_after_cycle']), 2)


if __name__ == '__main__': unittest.main()
