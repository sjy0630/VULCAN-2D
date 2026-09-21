import unittest
import numpy as np
from vulcan2d.barrier_state import supply_conductance
from vulcan2d.side_charge import configuration_supply, shape_and_gradients, predict, stationary_charge, symmetric_log_ratio_bound


class SideChargeTests(unittest.TestCase):
    def test_broadcast_supply_matches_archived_scalar(self):
        v = np.array([0., .02, .1, .3, 1., 3.])
        heights = np.array([.005, .2, 1.5])
        c = configuration_supply(v, heights[:, None], 2., order=48)[0]
        expected = [[supply_conductance(x, 2., h, order=48) for x in v] for h in heights]
        np.testing.assert_allclose(c, expected, rtol=2e-12, atol=0)

    def test_action_and_occupation_gradients(self):
        v = np.array([-.3, -.1, .1, .7, 2.5])
        shared = np.array([1.7, .18, .34, .63]); states = np.array([-.13, .06])
        value, grad, local, _ = shape_and_gradients(v, shared, states, order=64)
        for j in range(4):
            step = np.eye(4)[j]*1e-6
            finite = (shape_and_gradients(v, shared+step, states, order=64)[0]-
                      shape_and_gradients(v, shared-step, states, order=64)[0])/2e-6
            np.testing.assert_allclose(grad[:, :, j], finite, rtol=3e-3, atol=3e-4)
        finite = (shape_and_gradients(v, shared, states+1e-6, order=64)[0]-
                  shape_and_gradients(v, shared, states-1e-6, order=64)[0])/2e-6
        np.testing.assert_allclose(local, finite, rtol=1e-6, atol=1e-8)

    def test_current_mixture_not_mean_barrier(self):
        shared = [4., .1, .5, 0.]
        predicted = 10**predict([.1], shared, [-3., 0.])[0, 0]
        c = configuration_supply(.1, np.array([.1, .6]), 4.)[0]
        self.assertAlmostEqual(predicted/(1e-3*.1*c.mean()), 1, places=12)
        mean_barrier = 1e-3*.1*configuration_supply(.1, .35, 4.)[0]
        self.assertGreater(predicted/mean_barrier, 10)

    def test_detailed_balance_and_zero_stationary_side_current(self):
        p, entering, leaving = stationary_charge(np.array([-.3, 0, .3]), .08, .7, 23.)
        np.testing.assert_allclose(entering, leaving, rtol=3e-16, atol=0)
        self.assertTrue(np.all(np.diff(p) > 0))

    def test_frozen_symmetry_and_equilibrium_ratio_bound(self):
        rng = np.random.default_rng(934)
        v = np.array([.02, .1, .3, 1.])
        for _ in range(30):
            shared = [rng.uniform(.25, 6), rng.uniform(.005, 1), rng.uniform(0, 2), rng.uniform(-1, 1)]
            states = np.column_stack((rng.uniform(-7, -1, 8), rng.uniform(-.6, .6, 8)))
            value = predict(np.r_[v, -v], shared, states)
            ratio = (value[:, len(v):]-value[:, :len(v)])*np.log(10)
            self.assertTrue(np.all(np.abs(ratio) <= symmetric_log_ratio_bound(v, shared[3])+1e-10))
            frozen = predict(np.r_[v, -v], shared, states, mode='frozen')
            np.testing.assert_allclose(frozen[:, :len(v)], frozen[:, len(v):], atol=1e-12)


if __name__ == '__main__':
    unittest.main()
