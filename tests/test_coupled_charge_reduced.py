import unittest
import numpy as np
from vulcan2d.coupled_charge import stationary
from vulcan2d.coupled_charge_reduced import stationary_current


class ReducedCoupledChargeTests(unittest.TestCase):
    def test_exact_reduction_for_random_extreme_parameters(self):
        rng=np.random.default_rng(711)
        for _ in range(100):
            shared=np.r_[rng.uniform(-.5,.5),rng.uniform(.01,.99,2),rng.uniform(0,1),rng.uniform(-8,8)]
            v=np.r_[-.01,-.1,-.3,.01,.1,1.,3.];et=rng.uniform(-.6,.6)
            reduced=stationary_current(v,shared,et)
            full=stationary(v,np.r_[shared,rng.uniform(-8,8)],et)
            np.testing.assert_allclose(reduced['log10_abs_current_scale'],full['log10_abs_current_scale'],rtol=0,atol=1e-11)
            p=full['probabilities']
            np.testing.assert_allclose(reduced['mean_conductor_occupation'],p[1]+p[3],rtol=1e-11,atol=1e-13)
            np.testing.assert_allclose(reduced['mean_side_occupation'],p[2]+p[3],rtol=1e-11,atol=1e-13)

    def test_no_coupling_and_zero_bias(self):
        result=stationary_current(np.array([-.1,0,.1]),[.08,.2,.7,0.,2.],-.1)
        self.assertEqual(result['current_scale'][1],0.)
        self.assertLess(np.max(np.abs(result['side_normalization_error'])),1e-13)
        self.assertLess(np.max(np.abs(result['conductor_normalization_error'])),1e-13)

    def test_same_current_different_correlations(self):
        shared=[-.1,.3,.35,.02,0.]
        slow=stationary(.2,np.r_[shared,-7.],-.13)
        fast=stationary(.2,np.r_[shared,7.],-.13)
        self.assertAlmostEqual(slow['current_scale'],fast['current_scale'],places=13)
        self.assertGreater(np.max(np.abs(slow['probabilities']-fast['probabilities'])),1e-5)


if __name__=='__main__':unittest.main()
