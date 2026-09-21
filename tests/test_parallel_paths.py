import sys
from pathlib import Path
import unittest
import numpy as np
from scipy.integrate import quad
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'analysis'))
from check_parallel_rectification_bound import log_ratio_one_level, ratio_bound_decade
from check_scan_memory_contrast import branch_value
from vulcan2d.parallel_paths import band_log10_shape, predict, current, fit_shared, identify_state, _evaluate, component_shapes
from vulcan2d.contact_network import log10_shape
from vulcan2d.transport_candidates import TransportAssumptions


class ParallelPathTests(unittest.TestCase):
    def test_zero_width_single_level_limit(self):
        v = np.r_[-np.geomspace(.001, 3, 30), np.geomspace(.001, 3, 30)]
        np.testing.assert_allclose(band_log10_shape(v, [.2, 0., .3]),
                                   log10_shape(v, 'fixed_level', [.2, .3]), atol=2e-13)

    def test_independent_integral_and_quadrature_convergence(self):
        vt = TransportAssumptions().thermal_voltage_v
        for p in ([.1, .4, .13], [.7, 1., .9], [.1, .01, .5]):
            e, w, eta = p
            def direct(v):
                return quad(lambda energy: expit((eta*v-energy)/vt)-expit((-(1-eta)*v-energy)/vt),
                            e-w/2, e+w/2, epsabs=1e-14, epsrel=1e-11)[0]/w
            v = np.array([-.7, -.2, .1, .6, 3.])
            expected = np.log10(np.abs([direct(x)/direct(.1) for x in v]))
            np.testing.assert_allclose(band_log10_shape(v, p), expected, atol=3e-7)
            np.testing.assert_allclose(band_log10_shape(v, p), band_log10_shape(v, p, nodes=128), atol=3e-7)

    def test_zero_equilibrium_passivity_and_linear_response(self):
        p = [.1, .2, .15, .3, .5, .8]
        for name, parameters, state in [('two_band_free', p, [-8., 2.]),
                                        ('two_band_conserved', p+[-8., -7.], [2.])]:
            v = np.r_[np.linspace(-3, -.001, 31), 0., np.linspace(.001, 3, 31)]
            out = current(v, name, parameters, state)
            self.assertTrue(np.all(out*v >= 0)); self.assertEqual(current(0., name, parameters, state), 0.)
            small = current(np.array([-1e-9, 1e-9]), name, parameters, state)
            self.assertAlmostEqual(-small[0]/small[1], 1., places=6)

    def test_state_jacobian_matches_independent_differences(self):
        v = np.r_[np.linspace(.1, 3, 20), -np.linspace(.1, .3, 11)]
        p = [.1, .2, .15, .3, .5, .8]
        for name, pars, q in [('two_band_free', p, np.array([[-8., 2.], [-9., -3.]])),
                              ('two_band_conserved', p+[-8., -7.], np.array([[2.], [-3.]]))]:
            shapes = component_shapes(v, name, pars); value, jac = _evaluate(shapes, name, pars, q)
            for j in range(q.shape[1]):
                other = q.copy(); other[:, j] += 1e-6
                actual = (_evaluate(shapes, name, pars, other)[0]-value)/1e-6
                np.testing.assert_allclose(actual, jac[:, :, j], atol=1e-7)

    def test_conserved_candidate_is_nested_in_free_paths(self):
        p = [.1, .2, .15, .3, .5, .8]; a1, a2, t = -8., -7., 2.
        z = expit(t); total = (1-z)*10**a1+z*10**a2
        free_t = t+(a2-a1)*np.log(10.)
        v = np.r_[np.linspace(.1, 3, 20), -np.linspace(.1, .3, 11)]
        np.testing.assert_allclose(predict(v, 'two_band_conserved', p+[a1, a2], [[t]]),
                                   predict(v, 'two_band_free', p, [[np.log10(total), free_t]]), atol=1e-13)

    def test_SET_state_recovery_and_target_exclusion(self):
        p = [.1, .2, .15, .3, .5, .8]; q = [-8., 1.]
        v = np.linspace(.1, 3, 61); y = predict(v, 'two_band_free', p, [q])[0]
        fit = {'name': 'two_band_free', 'parameters': p, 'training_cycle_ids': [1, 2, 3]}
        got = identify_state(v, y, np.ones(len(v)), fit, 4)
        np.testing.assert_allclose(got['state'], q, atol=1e-6)
        # Negative target data is absent from this interface and cannot enter
        # state fitting. Shared-fit leakage is an explicit error.
        with self.assertRaises(ValueError):identify_state(v, y, np.ones(len(v)), fit, 2)

    def test_training_fit_recovers_synthetic_band_predictions(self):
        v = np.r_[np.linspace(.1, 3, 24), -np.linspace(.1, .3, 12)]
        y = np.array([-8., -7.7, -8.1])[:, None]+band_log10_shape(v, [.14, .18, .25])
        w = np.full(len(v), 1/len(v))
        fit = fit_shared(v, y, w, [1, 2, 3], 'band_amplitude', nstarts=2)
        self.assertTrue(fit['converged'])
        np.testing.assert_allclose(predict(v, 'band_amplitude', fit['parameters'], fit['states']), y, atol=2e-5)

    def test_analytic_rectification_bound_and_saturation(self):
        rng = np.random.default_rng(301)
        energy = rng.uniform(-3, 3, 1000); eta = rng.uniform(0, 1, 1000); v = rng.uniform(.001, 3, 1000)
        ratio = log_ratio_one_level(v, energy, eta)
        self.assertTrue(np.all(np.abs(ratio)/np.log(10) <= ratio_bound_decade(v)+1e-12))
        for eta in (0., 1.):
            r = log_ratio_one_level(.1, 10., eta)/np.log(10)
            self.assertAlmostEqual(abs(r), float(ratio_bound_decade(.1)), places=11)
        p = [.12, .8]; volts = np.geomspace(.001, 3, 100)
        actual = log10_shape(-volts, 'fixed_level', p)-log10_shape(volts, 'fixed_level', p)
        np.testing.assert_allclose(actual, log_ratio_one_level(volts, p[0], p[1])/np.log(10), atol=2e-13)

    def test_positive_mixtures_obey_class_bound(self):
        rng = np.random.default_rng(407); v = np.geomspace(.001, 3, 101)
        positive = np.zeros(len(v)); negative = positive.copy()
        for k in range(30):
            e, w, eta = rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(.01, .99)
            a = 10**rng.uniform(-10, -4)
            positive += a*10**band_log10_shape(v, [e, w, eta])
            negative += a*10**band_log10_shape(-v, [e, w, eta])
        self.assertTrue(np.all(np.abs(np.log10(negative/positive)) <= ratio_bound_decade(v)+1e-12))

    def test_masked_high_points_do_not_change_fit(self):
        v = np.r_[np.linspace(.1, 3, 24), -np.linspace(.1, .3, 12)]
        y = np.array([-8., -7.7, -8.1])[:, None]+band_log10_shape(v, [.14, .18, .25])
        w = np.ones_like(y); w[:, 15] = 0.; w /= w.sum(axis=1, keepdims=True)
        altered = y.copy(); altered[:, 15] = 150.
        f = fit_shared(v, y, w, [1, 2, 3], 'band_amplitude', nstarts=1)
        g = fit_shared(v, altered, w, [1, 2, 3], 'band_amplitude', nstarts=1)
        np.testing.assert_allclose(f['parameters'], g['parameters'], atol=1e-12)

    def test_scan_contrast_respects_order_and_duplicate_apex(self):
        raw = {'voltage': np.array([0., .2, .4, .4, .2, 0.]),
               'current': np.array([1e-10, 1e-9, 1e-8, 1e-8, 1e-7, 1e-10])}
        self.assertAlmostEqual(branch_value(raw, .2, False), -9.)
        self.assertAlmostEqual(branch_value(raw, .2, True), -7.)
        self.assertAlmostEqual(branch_value(raw, .3, False), -8.5)
        self.assertAlmostEqual(branch_value(raw, .3, True), -7.5)
        self.assertIsNone(branch_value(raw, .5, True))

    def test_scan_offset_does_not_log_nonpositive_current(self):
        raw = {'voltage': np.array([0., .2, .4, .2, 0.]),
               'current': np.array([1e-10, 1e-9, 1e-8, 1e-7, 1e-10])}
        self.assertIsNone(branch_value(raw, .1, False, offset=-2e-9))


if __name__ == '__main__':unittest.main()
