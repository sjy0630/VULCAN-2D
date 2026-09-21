import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'analysis'))
from check_negative_reentry_load_bound import no_enhancement_bound


class ReentryLoadBoundTests(unittest.TestCase):
    def test_linear_transistor_bound_matches_exact_series_limit(self):
        conductance = 1e-4
        lookup = SimpleNamespace(inverse=lambda current: current / conductance,
                                 at_voltage=lambda voltage: conductance * voltage)
        for scale in (.5, 1., 2.):
            bound = no_enhancement_bound(lookup, 1e-6, .5, .51, scale)
            self.assertAlmostEqual(bound['current_upper_without_enhancement_A'], 1e-6 + scale*conductance*.01, places=16)

    def test_fixed_and_weakened_nonlinear_device_obey_bound(self):
        transistor = lambda v: 1e-4 * np.expm1(v)
        lookup = SimpleNamespace(inverse=lambda current: np.log1p(current/1e-4), at_voltage=transistor)
        def current(applied, amplitude):
            internal = brentq(lambda v: amplitude*np.sinh(5*v)-transistor(applied-v), 0., applied)
            return transistor(applied-internal)
        before = current(.5, 1e-6)
        bound = no_enhancement_bound(lookup, before, .5, .51)['current_upper_without_enhancement_A']
        for amplitude in (1e-6, 5e-7, 1e-7):
            self.assertLessEqual(current(.51, amplitude), bound)
        self.assertGreater(current(.51, 1e-3), bound)

    def test_invalid_direction_and_passive_capacity_are_rejected(self):
        lookup = SimpleNamespace(inverse=lambda current: current / 1e-4, at_voltage=lambda voltage: 1e-4*voltage)
        for before, v0, v1, scale in [(1e-6, .5, .4, 1.), (1e-3, .5, .6, 1.), (1e-6, .5, .6, 0.)]:
            with self.assertRaises(ValueError):
                no_enhancement_bound(lookup, before, v0, v1, scale)


if __name__ == '__main__':
    unittest.main()
