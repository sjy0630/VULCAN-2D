import unittest
import numpy as np
from vulcan2d.retained_transport_parameters import pack, unpack, LO, HI
from vulcan2d.retained_charge_configuration import surfaces
from vulcan2d.coupled_charge import charging


class RetainedTransportParameterTests(unittest.TestCase):
    def test_round_trip_and_compensated_current(self):
        s = np.array([.6, .45, .55, .15, 12., .5])
        p = np.array([-4., .6, .1, -.1, -3., .2, .02, 0., -4.])
        x = pack(s, p); ss, pp = unpack(x)
        np.testing.assert_allclose(ss, s, atol=0, rtol=0)
        np.testing.assert_allclose(pp, p, atol=1e-15, rtol=0)
        original = surfaces([-.3, -.1, .1, .3], s, p)
        for gl in [-12., -6., 0., 6., 12.]:
            changed = x.copy(); changed[6] = gl
            ss, pp = unpack(changed); np.testing.assert_allclose(pack(ss, pp), changed, atol=1e-15, rtol=0)
            np.testing.assert_allclose(surfaces([-.3, -.1, .1, .3], ss, pp)['current'], original['current'], rtol=4e-14, atol=0)

    def test_bounded_geometry_and_arbitrary_parameter_round_trip(self):
        rng = np.random.default_rng(921141)
        for x in rng.uniform(LO, HI, size=(40, 14)):
            s, p = unpack(x); np.testing.assert_allclose(pack(s, p), x, rtol=0, atol=2e-15)
            u = charging(np.r_[s[:5], 0.])[0]
            cap = 1.439964/(3.1*max(6*abs(s[1]-s[2]), .25))
            self.assertGreaterEqual(u, 0); self.assertLessEqual(u, cap)
            self.assertEqual(p[7], 0.)

    def test_rejects_nonfinite_or_wrong_branch(self):
        for x in [np.zeros(13), np.full(14, np.nan)]:
            with self.assertRaises(ValueError): unpack(x)
        s = np.zeros(6); p = np.zeros(9); p[7] = .1
        with self.assertRaises(ValueError): pack(s, p)


if __name__ == '__main__': unittest.main()
