"""Independent continuous-time two-state protocols obey the mean bound."""
import unittest
import numpy as np
from scipy.linalg import expm


class RepeatedTwoStateMeanTests(unittest.TestCase):
    def test_arbitrary_periodic_rates_and_positive_channel_mixtures(self):
        rng = np.random.default_rng(38295)
        for _ in range(120):
            currents = []
            for channel in range(7):
                cycle = np.eye(2); integrated_rate = 0.
                for segment in range(5):
                    k01, k10 = 10**rng.uniform(-4, 2, 2); duration = 10**rng.uniform(-2, 1)
                    q = np.array([[-k01, k10], [k01, -k10]])
                    cycle = expm(q*duration)@cycle; integrated_rate += (k01+k10)*duration
                np.testing.assert_allclose(cycle.sum(axis=0), 1., atol=2e-12, rtol=0)
                self.assertAlmostEqual(np.linalg.det(cycle), np.exp(-integrated_rate), delta=3e-12)
                probability = rng.dirichlet([1., 1.]); readout = 10**rng.uniform(-12, 0, 2); trace = []
                for c in range(10):
                    trace.append(readout@probability); probability = cycle@probability
                currents.append(trace)
            currents = np.asarray(currents)
            for trace in [*currents, currents.sum(axis=0)]:
                for k in range(2, 10):
                    self.assertLessEqual(trace[k], (k/(k-1))*trace[k-1]*(1+3e-12))

    def test_bound_is_sharp_in_slow_initially_dark_limit(self):
        for epsilon in (1e-2, 1e-4, 1e-6):
            current = -np.expm1(-epsilon*np.arange(4))
            ratio = current[3]/current[2]
            self.assertLess(ratio, 1.5)
            self.assertLess(1.5-ratio, epsilon)


if __name__ == '__main__': unittest.main()
