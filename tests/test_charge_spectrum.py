import unittest
import numpy as np
from vulcan2d.charge_spectrum import shape_and_gradient, quadrature, predict
from vulcan2d.coupled_charge_reduced import predict as reduced_predict


class ChargeSpectrumTests(unittest.TestCase):
    def test_zero_width_exact_control(self):
        voltage = np.r_[-3., -.3, -.1, .1, .3, 2., 3.]
        states = [[-4., -.2], [-5., .1]]
        for p in ([.13, .16, .34, 1., -5.6], [-.3, .8, .2, .0, 4.]):
            np.testing.assert_allclose(predict(voltage, [*p, 0.], states),
                                       reduced_predict(voltage, p, states), atol=3e-13, rtol=0)

    def test_analytic_gradients(self):
        v = np.array([-.3, -.1, .1, .3, 2., 3.]); energies = np.array([-.2, .15])
        p = np.array([.13, .16, .34, .6, -2., .35])
        exact = shape_and_gradient(v, p, energies)['derivatives']
        for j in range(7):
            step = 2e-6
            pp, pm, ep, em = p.copy(), p.copy(), energies.copy(), energies.copy()
            if j < 6: pp[j] += step; pm[j] -= step
            else: ep += step; em -= step
            plus = shape_and_gradient(v, pp, ep)['log10_abs_current_scale']
            minus = shape_and_gradient(v, pm, em)['log10_abs_current_scale']
            np.testing.assert_allclose((plus-minus)/(2*step), exact[..., j], atol=2e-6, rtol=2e-6)

    def test_quadrature_refinement(self):
        v = np.r_[np.linspace(-.3, -.1, 7), np.linspace(.1, 3., 35)]
        for p in ([.1, .16, .34, 1., -5., .6], [-.3, .8, .15, .8, 2., .6]):
            a = shape_and_gradient(v, p, [-.5, .0, .5], panels=8)['log10_abs_current_scale']
            b = shape_and_gradient(v, p, [-.5, .0, .5], panels=16)['log10_abs_current_scale']
            np.testing.assert_allclose(a, b, atol=2e-7, rtol=0)

    def test_zero_bias_dissipation_and_normalized_weights(self):
        v = np.array([-3., -.1, 0., .1, 3.])
        r = shape_and_gradient(v, [.1, .2, .7, .6, -2., .3], [.0, .1])
        self.assertTrue(np.all(r['current_scale']*v >= 0))
        np.testing.assert_array_equal(r['current_scale'][:, 2], [0., 0.])
        x, w = quadrature()
        self.assertAlmostEqual(w.sum(), 1.)
        self.assertTrue(np.all(w > 0) and np.all(np.abs(x) < 1))

    def test_no_coupling_makes_side_state_irrelevant(self):
        r = shape_and_gradient([-.3, .1, 3.], [.1, .2, .7, 0., -2., .3], [-.5, .5])
        np.testing.assert_allclose(r['log10_abs_current_scale'][0], r['log10_abs_current_scale'][1], atol=1e-13)
        np.testing.assert_allclose(r['derivatives'][..., 6], 0., atol=1e-12)


if __name__ == '__main__':
    unittest.main()
