import unittest
import numpy as np
from scipy.special import softmax
from vulcan2d.coupled_charge import EDGES, VT, charging, log_rates, stationary, predict


class CoupledChargeTests(unittest.TestCase):
    def test_equilibrium_boltzmann_distribution(self):
        shared = [.06, .2, .27, .3, 2., -1.]; et = -.04; u, _ = charging(shared)
        result = stationary(0., shared, et)
        expected = softmax(-np.array([0, shared[0], et, shared[0]+et+u])/VT)
        np.testing.assert_allclose(result['probabilities'], expected, rtol=1e-12, atol=1e-15)
        self.assertEqual(result['current_scale'], 0.)

    def test_trees_against_independent_linear_solve(self):
        rng = np.random.default_rng(1507)
        for _ in range(40):
            shared = [rng.uniform(-.12,.12),rng.uniform(.1,.9),rng.uniform(.1,.9),rng.uniform(0,.2),rng.uniform(-2,2),rng.uniform(-2,2)]
            v = rng.choice([-1.,1.])*rng.uniform(.03,.2); et = rng.uniform(-.12,.12)
            rates = np.exp(log_rates(v, shared, et)[0]); q = np.zeros((4,4))
            for rate, (a,b) in zip(rates, EDGES):q[b,a] += rate; q[a,a] -= rate
            matrix = q.copy(); matrix[-1] = 1
            p = np.linalg.solve(matrix, [0,0,0,1]); result = stationary(v, shared, et)
            np.testing.assert_allclose(result['probabilities'], p, rtol=2e-8, atol=1e-10)
            np.testing.assert_allclose(result['current_scale'], result['gross_left_current'], rtol=2e-8, atol=1e-13)
            self.assertLess(abs(result['gross_side_current']), 1e-12)
            self.assertLess(abs(result['gross_left_current']+result['gross_right_current']), 1e-12)
            self.assertGreater(result['current_scale']*v, 0)
            self.assertLess(abs(result['forward_reverse_cycle_log_ratio_residual']), 1e-12)

    def test_uncoupled_limit_is_independent_of_side_trap(self):
        v = np.array([-.3,-.1,.1,.5,2.]); shared = [.08,.13,.8,0.,.5,-5.]
        a = predict(v, shared, [-5.,-.4]); shared[-1] = 6.
        b = predict(v, shared, [-5.,.5])
        np.testing.assert_allclose(a,b,rtol=0,atol=1e-12)

    def test_stable_tiny_current_and_cycle_affinity(self):
        v = np.array([-.02,.02]); shared = [.4,.02,.05,1.,-8.,8.]
        result = stationary(v, shared, .5)
        self.assertTrue(np.all(np.isfinite(result['log10_abs_current_scale'])))
        self.assertLess(np.max(np.abs(result['forward_reverse_cycle_log_ratio_residual'])),1e-11)

    def test_shared_and_structural_energy_gradients(self):
        v = np.array([-.3,-.1,.1,.8,2.]); shared = np.array([.08,.12,.32,.25,1.,-.6]); et = -.08
        analytic = stationary(v, shared, et)['derivatives']
        for j in range(7):
            step = 1e-6
            if j < 6:
                delta = np.eye(6)[j]*step
                difference = stationary(v, shared+delta, et)['log10_abs_current_scale']-stationary(v, shared-delta, et)['log10_abs_current_scale']
            else:
                difference = stationary(v, shared, et+step)['log10_abs_current_scale']-stationary(v, shared, et-step)['log10_abs_current_scale']
            np.testing.assert_allclose(analytic[...,j], difference/(2*step), rtol=2e-5, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
