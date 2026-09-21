import unittest
import numpy as np
from scipy.linalg import expm
from vulcan2d.charge_moments import coefficients, propagate_means
from vulcan2d.coupled_charge import EDGES, log_rates


class ChargeMomentTests(unittest.TestCase):
    def test_exact_multistep_propagation_against_four_state_generator(self):
        shared = [.03, .2, .3, .1, -.3]; et = -.07; lograte = .7
        voltages = [.1, .5, 1., .2, 0., -.1, -.6, -.1, 0.]
        durations = np.linspace(.01, 2., len(voltages))
        probability = np.array([.1, .2, .3, .4])
        predicted = propagate_means(voltages, durations, shared, et, [.6, .7], lograte)
        for row, voltage, dt in zip(predicted, voltages, durations):
            q = np.zeros((4, 4))
            for (a, b), rate in zip(EDGES, np.exp(log_rates(voltage, [*shared, lograte], et)[0])):
                q[b, a] += rate; q[a, a] -= rate
            probability = expm(q*dt)@probability
            np.testing.assert_allclose(row, [probability[1]+probability[3], probability[2]+probability[3]], rtol=0, atol=2e-14)

    def test_decay_rates_and_uncoupled_limit(self):
        for u in (0., .2, 1.):
            p = [-.1, .2, .3, u, -.5]
            c = coefficients(.1, p, -.2, 2.)
            g, t = c['conductor_attempt_rate'], c['side_attempt_rate']
            matrix = np.array([[-g, g*c['B']], [t*c['D'], -t]])
            np.testing.assert_allclose(sorted(-np.linalg.eigvals(matrix)),
                                       [c['slow_decay_rate'], c['fast_decay_rate']], atol=1e-13)
            self.assertGreater(c['slow_decay_rate'], 0.)
            self.assertLessEqual(c['slow_rate_relative_to_uncoupled'], 1.+1e-14)
            if u == 0.: self.assertAlmostEqual(c['slow_rate_relative_to_uncoupled'], 1.)

    def test_positive_zero_bias_level_has_no_large_collective_slowdown(self):
        c = coefficients(0., [.128, .157, .334, 1., -5.6], np.linspace(-.6, .6, 25), 0.)
        self.assertTrue(np.all(c['closure_determinant'] > .99))

    def test_strong_blockade_stable_small_decay_rate(self):
        c = coefficients(0., [-.5, .2, .21, 1., 0.], -.5, 0.)
        self.assertGreater(c['slow_decay_rate'], 0.)
        self.assertLess(c['slow_rate_relative_to_uncoupled'], 1e-7)


if __name__ == '__main__':
    unittest.main()
