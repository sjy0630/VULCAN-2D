import sys
from pathlib import Path
import unittest
import numpy as np
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from vulcan2d.wkb_polarity import linear_root_average, log_fermi_window, log_transmission, log_current_integral, thermal_log_ratio_bound
from vulcan2d.barrier_state import supply_conductance


class WKBPolarityTests(unittest.TestCase):
    def test_arbitrary_linear_profile_action_matches_quadrature(self):
        for a, b in ((-.3, .7), (.7, -.3), (.2, .2), (-.2, -.1), (3., .01)):
            expected = quad(lambda x:np.sqrt(max((1-x)*a+x*b, 0.)), 0., 1., epsabs=1e-11)[0]
            self.assertAlmostEqual(float(linear_root_average(a, b)), expected, places=9)

    def test_symmetric_profile_matches_archived_supply_and_polarities(self):
        for phi, g, v in ((.1, 6., .1), (.5, 2., .2), (3., 5., .3)):
            positive = log_current_integral(v, phi, phi, g)
            negative = log_current_integral(v, phi, phi, g, sign=-1)
            self.assertAlmostEqual(positive, negative, places=11)
            old = np.log(float(supply_conductance(v, [g], phi, order=48)[0])*v)
            self.assertAlmostEqual(positive, old, places=6)

    def test_spectral_order_and_thermal_shift_inequalities(self):
        e = np.linspace(-3., 5., 801); v = .13
        plus = log_transmission(e, v, .12, 1.2, 5.)
        shifted = log_transmission(e+v, v, .12, 1.2, 5., sign=-1)
        self.assertTrue(np.all(plus <= shifted+1e-12))
        difference = log_fermi_window(e-v, v)-log_fermi_window(e, v)
        self.assertLessEqual(float(np.max(np.abs(difference))), float(thermal_log_ratio_bound(v))+1e-12)

    def test_asymmetric_current_respects_bound_and_independent_integral(self):
        from scipy.constants import Boltzmann, elementary_charge
        for left, right, g, v in ((.3, .01, 6., .1), (.1, 2., 2., .3), (1.2, .2, 4., .2)):
            lp = log_current_integral(v, left, right, g, order=64)
            lm = log_current_integral(v, left, right, g, sign=-1, order=64)
            self.assertLessEqual(abs(lm-lp), float(thermal_log_ratio_bound(v))+1e-10)
            # Normalize by the quadrature estimate so absolute-error control is
            # meaningful even for extremely small physical currents.
            def scaled(energy):
                return np.exp(log_transmission(energy, v, left, right, g)+log_fermi_window(energy, v)-lp)
            vt = Boltzmann*300/elementary_charge
            nodes = [-v/2-60*vt, -v/2, v/2, left+v/2, right-v/2, max(left, right)+v/2+60*vt]
            nodes = sorted(set(nodes))
            total = sum(quad(scaled, a, b, epsabs=1e-10, epsrel=1e-9, limit=300)[0] for a, b in zip(nodes[:-1], nodes[1:]))
            self.assertAlmostEqual(total, 1., places=6)


if __name__ == '__main__':
    unittest.main()
