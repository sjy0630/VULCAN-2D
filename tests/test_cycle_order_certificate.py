import itertools
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'analysis'))
from check_cycle_order_evidence import cusum
from certify_cycle_order_permutation import certify


class CycleOrderCertificateTests(unittest.TestCase):
    def test_unique_subset_certificate_against_full_permutation_enumeration(self):
        x = np.array([0., .01, .02, 10., 10.01, 10.02, 10.03])[:, None]
        result = certify(x); self.assertTrue(result['certified'])
        threshold = cusum(x).max(); count = 0
        for order in itertools.permutations(range(len(x))):
            count += int(cusum(x[list(order)]).max() >= threshold-1e-12)
        self.assertEqual(result['exact_probability'], count/5040)
        self.assertAlmostEqual(result['exact_probability'], 2/35)

    def test_nonunique_qualifying_subsets_are_not_falsely_certified(self):
        result = certify(np.arange(7., dtype=float)[:, None], threshold=.1)
        self.assertFalse(result['certified']); self.assertIsNone(result['exact_probability'])


if __name__ == '__main__': unittest.main()
