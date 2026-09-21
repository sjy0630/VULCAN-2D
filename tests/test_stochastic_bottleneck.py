import sys
from pathlib import Path
import unittest
import numpy as np
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from vulcan2d.stochastic_bottleneck import (step_log_probabilities, transition_log_probabilities,
                                          negative_log_likelihood, population, sample_paths)


class StochasticBottleneckTests(unittest.TestCase):
    def test_transition_matches_matrix_exponential(self):
        for a, b, dx in [(2., 5., .2), (1e-9, 1e5, .02), (3., 7., 0.)]:
            result = np.exp(step_log_probabilities(np.log(a), np.log(b), dx))
            np.testing.assert_allclose(result, expm(np.array([[-a, a], [b, -b]])*dx), rtol=1e-10, atol=1e-13)
            np.testing.assert_allclose(result.sum(axis=1), 1., atol=1e-14)

    def test_substeps_match_constant_rate_semigroup(self):
        p = [2., .04]
        one = {'off_voltage': np.zeros((3, 1)), 'on_voltage': np.zeros((3, 1)), 'path_increment': np.array([.1, 0., .3])}
        four = {'off_voltage': np.zeros((3, 4)), 'on_voltage': np.zeros((3, 4)), 'path_increment': np.array([.025, 0., .075])}
        np.testing.assert_allclose(np.exp(transition_log_probabilities(one, p, 'constant')),
                                   np.exp(transition_log_probabilities(four, p, 'constant')), atol=1e-14)

    def test_probability_and_likelihood_for_known_sequence(self):
        matrix = np.array([[[.8, .2], [.3, .7]], [[.8, .2], [.3, .7]]])
        self.assertAlmostEqual(negative_log_likelihood(np.log(matrix), [1, 0, 1]), -np.log(.3*.2))
        self.assertAlmostEqual(negative_log_likelihood(np.log(matrix), [1, 0, 1], [True, False]), -np.log(.3))
        np.testing.assert_allclose(population(np.log(matrix)), [1., .7, .55])

    def test_sample_frequency_matches_population(self):
        matrix = np.tile(np.array([[.8, .2], [.3, .7]])[None], (5, 1, 1))
        samples = sample_paths(np.log(matrix), n_paths=100000, seed=7)
        np.testing.assert_allclose(samples.mean(axis=0), population(np.log(matrix)), atol=.005)

    def test_very_fast_rates_remain_log_stochastic(self):
        result = step_log_probabilities(500., 498., .1)
        self.assertTrue(np.all(np.isfinite(result)))
        np.testing.assert_allclose(np.exp(result).sum(axis=-1), 1., atol=1e-13)

    def test_return_current_poisoning_cannot_change_transport_calibration(self):
        from types import SimpleNamespace
        sys.path.insert(0, str(ROOT / 'analysis'))
        from check_stochastic_reentry import calibrate_outward_transport
        from vulcan2d.stochastic_bottleneck import frozen_load
        up = np.linspace(0, 1.5, 61); va = np.r_[up, up[::-1]]
        labels = np.ones(len(up), dtype=int); labels[15:43] = 0
        full_labels = np.r_[labels, np.ones(len(up), dtype=int)]
        tv = np.linspace(0, 2, 401); ti = 1e-4*tv
        _, observed = frozen_load(va, [-9., 6., -5., 3.], full_labels, tv, ti)
        lookup = SimpleNamespace(voltage=tv, current=ti)
        before = calibrate_outward_transport(va, observed, labels, lookup)
        corrupted = observed.copy(); corrupted[len(up):] = np.nan
        after = calibrate_outward_transport(va, corrupted, labels, lookup)
        np.testing.assert_array_equal(before['parameters'], after['parameters'])
        self.assertEqual(before['rmse_decade'], after['rmse_decade'])


if __name__ == '__main__':
    unittest.main()
