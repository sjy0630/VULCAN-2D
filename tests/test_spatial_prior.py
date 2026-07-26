import unittest

import numpy as np

from vulcan2d.spatial_prior import SpotAreaDistribution
from vulcan2d.validate_v5 import endpoint_interpolated_params
from vulcan2d import model_v4 as M


class SpatialPriorTests(unittest.TestCase):
    def test_area_distribution_summary_and_coarse_weights(self):
        distribution = SpotAreaDistribution(
            area_m2=np.asarray([1e-18, 2e-18, 8e-18]),
            counts=np.asarray([4, 4, 2]),
        )
        summary = distribution.summary(k=2)
        weights = np.asarray(summary["coarse_prior"]["area_weights"])
        self.assertEqual(summary["spot_count"], 10)
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertGreater(weights[1], weights[0])
        self.assertIn("not a filament diameter", summary["claim_scope"])

    def test_1p1_gate_holdout_removes_direct_anchors(self):
        p = M.Params()
        holdout = endpoint_interpolated_params(p)
        self.assertAlmostEqual(holdout.Icc_ref,
                               np.sqrt(p.Icc_low * p.Icc_high))
        self.assertAlmostEqual(holdout.Ireset_peak_ref,
                               np.sqrt(p.Ireset_peak_low * p.Ireset_peak_high))
        self.assertNotAlmostEqual(holdout.Icc_ref, p.Icc_ref)


if __name__ == "__main__":
    unittest.main()
