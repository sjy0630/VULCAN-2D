import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from scipy.linalg import expm
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'analysis'))
from vulcan2d.finite_paths import generator, stochastic_exponential, transitions, marginal, sample_counts
from vulcan2d.stochastic_bottleneck import clock_grid, transition_log_probabilities, step_log_probabilities
from finite_path_observables import batch_events, log_readouts
from check_negative_reentry import events


class FinitePathTests(unittest.TestCase):
    def test_generator_boundary_and_conservation(self):
        q = generator(np.full((2, 5), .3), 4, [2., .03], 'constant')
        np.testing.assert_allclose(q.sum(axis=-1), 0., atol=1e-13)
        self.assertEqual(q[0, 0, 4], 0.); self.assertEqual(q[0, 4, 0], 0.)
        self.assertTrue(np.all(q[:, np.arange(5), np.arange(5)] <= 0))

    def test_positive_exponential_matches_independent_scipy(self):
        for peak in (-8., 1., 12.):
            q = generator(np.full((3, 9), .3), 8, [peak, .03], 'constant')
            dt = np.array([0., .01, .7]); actual, _ = stochastic_exponential(q, dt)
            for j in range(3): np.testing.assert_allclose(actual[j], expm(q[j]*dt[j]), atol=3e-10)
            np.testing.assert_allclose(actual.sum(axis=-1), 1., atol=1e-13)
            self.assertTrue(np.all(actual >= 0))

    def test_independent_channels_have_binomial_count_transition(self):
        count = 7; q = generator(np.zeros((1, count+1)), count, [1., 0.], 'constant')
        transition, _ = stochastic_exponential(q, [.2])
        two = np.exp(step_log_probabilities(1., 1., .2))
        for initial in range(count+1):
            expected = np.convolve(binom.pmf(np.arange(initial+1), initial, two[1, 1]),
                                   binom.pmf(np.arange(count-initial+1), count-initial, two[0, 1]))
            np.testing.assert_allclose(transition[0, initial], expected, atol=1e-13)

    def test_state_dependent_rates_match_exponential_and_stationary_law(self):
        voltage = np.array([[1.2, 1., .8, .5, .2], [.9, .85, .6, .2, .05]])
        q = generator(voltage, 4, [7., .1, .3, .04], 'polarizable_saddle')
        actual, _ = stochastic_exponential(q, [.01, .3])
        for j, duration in enumerate((.01, .3)):
            np.testing.assert_allclose(actual[j], expm(q[j]*duration), atol=1e-12)
            # A finite birth/death chain has a known stationary adjacent-flux ratio.
            log_weights = np.r_[0., np.cumsum(np.log(np.diag(q[j], 1))-np.log(np.diag(q[j], -1)))]
            weights = np.exp(log_weights-log_weights.max()); weights /= weights.sum()
            np.testing.assert_allclose(weights @ actual[j], weights, atol=1e-12)

    def test_one_path_matches_previous_two_state_solver(self):
        va = np.r_[np.linspace(0, 1.5, 51), np.linspace(1.5, 0, 51)]
        tv = np.linspace(0, 2, 201); ti = 1e-4*tv; transport = [-9., 6., -5., 3.]; kinetic = [7., .2, .5]
        got, _ = transitions(va, transport, kinetic, 'single_hop', 1, tv, ti, substeps=4)
        grid = clock_grid(va, transport, tv, ti, substeps=4)
        expected = np.exp(transition_log_probabilities(grid, kinetic, 'single_hop'))
        np.testing.assert_allclose(got, expected, atol=3e-14)

    def test_sample_count_frequency_matches_marginal(self):
        q = generator(np.full((5, 5), .4), 4, [1., 0.], 'constant'); p, _ = stochastic_exponential(q, .1)
        dist = marginal(p); paths = sample_counts(p, n_paths=50000, seed=6)
        observed = np.bincount(paths[:, -1], minlength=5)/len(paths)
        np.testing.assert_allclose(observed, dist[-1], atol=.006)

    def test_batch_detector_matches_original_and_rejects_isolated_spikes(self):
        up = np.linspace(0, 1.5, 151); v = np.r_[up, up[::-1]]
        good = np.where(v < .4, 1e-5, np.where(v < 1.1, 1e-8, 1e-4)); good[v == 0] = 1e-10
        spike = np.full(len(v), 1e-8); spike[v == 0] = 1e-10; spike[60] = 1e-4
        lookup = SimpleNamespace(inverse=lambda i:i/1e-4, at_voltage=lambda voltage:voltage*1e-4)
        all_currents = np.array([good, spike]); got = batch_events(v, all_currents, [1e-10, 1e-10])
        for k, current in enumerate(all_currents):
            old = events({'voltage': -v, 'current': current}, lookup)
            for kind in ('rise', 'drop'):
                self.assertEqual(np.flatnonzero(got[kind][k]).tolist(), [r['raw_index_after'] for r in old if r['kind'] == kind])

    def test_matched_bias_readouts_and_nonpositive_guard(self):
        up = np.linspace(0, 1.5, 151); v = np.r_[up, up[::-1]]
        current = 1e-5*v; current[151:] *= .01
        got = log_readouts(v, current)
        np.testing.assert_allclose(got['log10_return_outward_ratio_0p5_0p8'], -2., atol=1e-12)
        read = np.linspace(.1, .3, 21)*1e-7
        np.testing.assert_allclose(got['log10_return_current_A_0p1_0p3'], np.mean(np.log10(read)), atol=1e-12)
        current[-21] = 0.
        with self.assertRaises(ValueError):
            log_readouts(v, current)


if __name__ == '__main__': unittest.main()
