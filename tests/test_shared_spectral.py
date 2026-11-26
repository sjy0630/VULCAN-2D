"""Independent full-state numerical and signed-law checks for spectral research.

References use adaptive integration in log energy; the narrow-width and
zero-bias checks also use positive Fermi integrals without digamma subtraction.
No fitted experimental accuracy is asserted by these tests.
"""
from dataclasses import replace
import inspect
import math
import unittest

import numpy as np
from scipy.integrate import quad, quad_vec
from scipy.special import digamma, expit, roots_legendre

from vulcan2d.shared_spectral import SpectralShape, SpectralLaw, SpectralDomainError


def model(**kwargs):
    values = dict(shape=SpectralShape(), gamma_c_ev=.01, jset_a=1e-4,
                  jreset_a=1e-5, kset_per_v=.7, kreset_per_v=.3, state_limit=80.)
    values.update(kwargs)
    return SpectralLaw(**values)


def independent_current(shape, voltage, gamma):
    """Infinite-energy-support reference in Z coordinates, to probability 1e-23."""
    v = np.asarray(voltage)
    t = shape.thermal_ev
    def integrand(z):
        epsilon = shape.median_energy_ev*np.exp(shape.sigma_log_energy*z)
        a = .5+(gamma+1j*(epsilon+shape.eta*v))/(2*np.pi*t)
        b = .5+(gamma+1j*(epsilon-(1-shape.eta)*v))/(2*np.pi*t)
        return (digamma(a)-digamma(b)).imag*np.exp(-z*z/2)/np.sqrt(2*np.pi)
    # Integrate dimensionless spectral weight so tiny gamma does not evade an
    # absolute-current stopping tolerance.
    result, error = quad_vec(integrand, -10., 10., epsabs=2e-12, epsrel=2e-11,
                             norm='max', limit=500)
    return gamma/shape.ron_ohm*result


def narrow_reference(shape, voltage, gamma):
    """gamma->0 positive Fermi-window integral, independent of digamma."""
    v = float(voltage)
    t = shape.thermal_ev
    def integrand(z):
        epsilon = shape.median_energy_ev*np.exp(shape.sigma_log_energy*z)
        a, b = epsilon-(1-shape.eta)*v, epsilon+shape.eta*v
        low, high = min(a, b), max(a, b)
        window = math.copysign(1., v)*expit(-low/t)*expit(high/t)*(-np.expm1(-abs(v)/t))
        return window*np.exp(-z*z/2)/np.sqrt(2*np.pi)
    result = quad(integrand, -10., 10., epsabs=1e-25, epsrel=2e-11, limit=500)[0]
    return np.pi*gamma/shape.ron_ohm*result


def independent_wide_zero_bias_conductance(shape, gamma):
    """Positive Fermi-derivative integral; no complex special functions.

    Used only for gamma>=1 eV, so the electron-energy integrand is smooth.
    """
    x, w = roots_legendre(256)
    energy = 40*shape.thermal_ev*x
    thermal_weight = 40*w*expit(energy/shape.thermal_ev)*expit(-energy/shape.thermal_ev)
    def integrand(z):
        epsilon = shape.median_energy_ev*np.exp(shape.sigma_log_energy*z)
        transmission = gamma**2/((energy-epsilon)**2+gamma**2)
        return float(transmission@thermal_weight)*np.exp(-z*z/2)/np.sqrt(2*np.pi)
    return quad(integrand, -10., 10., epsabs=1e-12, epsrel=2e-11, limit=200)[0]/shape.ron_ohm


class SharedSpectralTests(unittest.TestCase):
    def test_fixed_shape_matches_selected_readout_and_no_topology_input(self):
        self.assertEqual(SpectralShape(), SpectralShape.from_json())
        for function in (SpectralShape.current_at_log_gamma, SpectralLaw,
                         SpectralLaw.current, SpectralLaw.voltage, SpectralLaw.derivative):
            self.assertFalse({'topology', 'gate', 'gate_voltage', 'cycle', 'phase', 'device'}
                             & set(inspect.signature(function).parameters))
        self.assertNotIn('beta', inspect.signature(SpectralLaw).parameters)
        self.assertEqual(SpectralLaw.beta, 1.)

    def test_full_tail_probability_and_quadrature_refinement(self):
        shape = SpectralShape()
        meta = shape.quadrature_metadata()
        self.assertGreater(meta['energy_upper_ev'], 1e5)
        self.assertLessEqual(meta['omitted_probability'], 1.01e-24)
        self.assertLess(abs(meta['probability_mass_error']), 2e-10)
        v = np.array([-5., -2.5, -.2, -.06, .06, .2, 1., 2., 5.])
        for gamma in (1e-24, 1e-8, .0025943212886790456, .1, 1., 10., 1e3, 1e6):
            a = shape.current_at_gamma(v, gamma, order=32)
            b = shape.current_at_gamma(v, gamma, order=64)
            np.testing.assert_allclose(a, b, rtol=2e-7, atol=0.)
            reference = independent_current(shape, v, gamma)
            np.testing.assert_allclose(b, reference, rtol=2e-7, atol=0.)

    def test_narrow_and_tiny_voltage_limits_without_digamma_reference(self):
        shape = SpectralShape()
        gamma = 1e-24
        for v in (-5., -.2, -1e-9, 1e-9, .2, 5.):
            expected = narrow_reference(shape, v, gamma)
            actual = shape.current_at_gamma(v, gamma)
            np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=0.)
        for gamma in (1., 100.):
            conductance = independent_wide_zero_bias_conductance(shape, gamma)
            for v in (-1e-11, 1e-11):
                np.testing.assert_allclose(shape.current_at_gamma(v, gamma)/v,
                                           conductance, rtol=2e-8, atol=0.)

    def test_voltage_monotonicity_state_order_and_passivity_across_full_gamma(self):
        shape = SpectralShape()
        v = np.r_[-np.geomspace(5., 1e-9, 120), 0., np.geomspace(1e-9, 5., 120)]
        gammas = np.geomspace(1e-24, 1e6, 35)
        result = shape.current_at_log_gamma(v[None, :], np.log(gammas)[:, None])
        self.assertTrue(np.all(np.diff(result, axis=1) > 0.))
        self.assertTrue(np.all(result*v >= 0.))
        self.assertTrue(np.all(abs(result) <= abs(v)[None, :]/shape.ron_ohm*(1+1e-10)))
        nonzero = v != 0.
        self.assertTrue(np.all(np.diff(abs(result[:, nonzero]), axis=0) >= 0.))

    def test_signed_asymmetry_and_inverse_do_not_assume_oddness(self):
        law = model()
        self.assertNotAlmostEqual(abs(law.current(-1., 2.))/law.current(1., 2.), 1., places=3)
        voltage = np.array([-5., -1., -.2, -1e-9, 0., 1e-9, .2, 1., 5.])
        for u in (0., 1e-7, .005, 2., 20., 60.):
            current = law.current(voltage, u)
            np.testing.assert_allclose(law.voltage(current, u), voltage, rtol=3e-8, atol=1e-15)
        current = 1e-6
        self.assertGreater(abs(law.voltage(-current, 2.)+law.voltage(current, 2.)), .01)

    def test_exact_and_continuous_ohmic_limit(self):
        law = model()
        v = np.array([-5., -.2, 0., .2, 5.])
        np.testing.assert_array_equal(law.current(v, 0.), v/law.ron_ohm)
        np.testing.assert_array_equal(law.shape.current_at_log_gamma(v, np.inf), v/law.ron_ohm)
        previous = np.inf
        for u in (1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10):
            error = np.max(abs(law.current(v, u)-v/law.ron_ohm))
            self.assertLessEqual(error, previous)
            previous = error
        np.testing.assert_allclose(law.current(v, 1e-10), v/law.ron_ohm, rtol=1e-12, atol=0.)

    def test_numerical_switches_are_continuous_and_independently_checked(self):
        shape = SpectralShape()
        # Both sides of the tiny-voltage path.
        for gamma in (1e-24, .002, 1., 1e3):
            for sign in (-1., 1.):
                v = sign*np.array([1e-5*(1-1e-7), 1e-5*(1+1e-7)])
                current = shape.current_at_gamma(v, gamma)
                np.testing.assert_allclose(current[0]/v[0], current[1]/v[1], rtol=2e-8, atol=0.)
        # Both sides of the analytically bounded broad-width switch.
        for v in (-5., -.2, .2, 5.):
            _, fourth = shape._moments(v)
            boundary = (fourth/1e-11)**.25
            for gamma in (boundary*(1-1e-6), boundary*(1+1e-6)):
                expected = independent_current(shape, np.array([v]), gamma)[0]
                actual = shape.current_at_gamma(v, gamma)
                np.testing.assert_allclose(actual, expected, rtol=3e-10, atol=0.)

    def test_derivative_log_overdrive_and_nonabsorbing_reset_boundary(self):
        law = model(state_limit=20.)
        self.assertEqual(law.derivative(0., 1e-2), 0.)
        self.assertGreater(law.derivative(0., -1e-3), 0.)
        self.assertGreater(law.derivative(law.state_limit, -1e-3), 0.)
        self.assertEqual(law.derivative(10., 0.), 0.)
        for u in (0.1, 2., 12.):
            for factor in (.5, 1., 2.):
                i = law.jset_a*math.exp(-u)*factor
                self.assertAlmostEqual(law.derivative(u, i), -law.kset_per_v*max(math.log(factor), 0.), places=13)
                i = -law.jreset_a*math.exp(-u)*factor
                self.assertAlmostEqual(law.derivative(u, i), law.kreset_per_v*max(math.log(factor), 0.), places=13)
        with self.assertRaises(SpectralDomainError):
            law.derivative(law.state_limit+1e-8, -1e-3)

    def test_own_set_stopping_bound_is_necessary_not_state_projection(self):
        law = model(jset_a=2e-3)
        bound = law.set_stopping_bound(1e-3)
        self.assertAlmostEqual(bound, math.log(2.))
        self.assertEqual(law.derivative(bound-.1, 1e-3), 0.)
        self.assertLess(law.derivative(bound+.1, 1e-3), 0.)
        self.assertEqual(law.validate_state(.1), .1)
        self.assertEqual(law.set_stopping_bound(0.), math.inf)

    def test_domain_failures_do_not_clip_current_or_state(self):
        law = model()
        for current in (law.saturation_current(2.), 2*law.saturation_current(2.)):
            with self.assertRaises(SpectralDomainError):
                law.voltage(current, 2.)
            with self.assertRaises(SpectralDomainError):
                law.voltage(-current, 2.)
        with self.assertRaises(SpectralDomainError):
            law.voltage(1.01*law.current(5., 2.), 2.)
        for state in (-1., math.nan, math.inf, 81.):
            with self.assertRaises(SpectralDomainError):
                law.current(.1, state)
        with self.assertRaises(SpectralDomainError):
            law.current(5.01, 2.)
        with self.assertRaises(ValueError):
            model(jset_a=0.)
        with self.assertRaises(ValueError):
            replace(SpectralShape(), eta=1.)

    def test_reset_initiation_zero_preserves_original_operation_exactly(self):
        base = model()
        explicit_zero = model(reset_initiation_log=0.)
        for state in (0., .01, 1., 8., 80.):
            for current in (-1., -1e-3, -1e-9, 0., 1e-9, 1e-3, 1.):
                if current == 0. or (current > 0. and state == 0.):
                    expected = 0.
                else:
                    threshold = base.jset_a if current > 0. else base.jreset_a
                    rate = base.kset_per_v if current > 0. else base.kreset_per_v
                    original = rate*max(math.log(abs(current))-math.log(threshold)+base.beta*state, 0.)
                    expected = -original if current > 0. else original
                self.assertEqual(base.derivative(state, current), expected)
                self.assertEqual(explicit_zero.derivative(state, current), expected)

    def test_reset_initiation_threshold_and_fixed_supply_feedback_term(self):
        law = model(reset_initiation_log=5.)
        base = model()
        for state in (0., .3, 2., 12.):
            # The exponential is used only for moderate, representable test
            # thresholds; production never evaluates exp(A).
            threshold = law.jreset_a*math.exp(-state+5.*math.exp(-state))
            self.assertEqual(law.derivative(state, -.5*threshold), 0.)
            self.assertAlmostEqual(law.derivative(state, -threshold), 0., places=13)
            self.assertAlmostEqual(law.derivative(state, -2*threshold),
                                   law.kreset_per_v*math.log(2.), places=13)
            self.assertEqual(law.derivative(state, 1e-3), base.derivative(state, 1e-3))
            self.assertLessEqual(law.derivative(state, -1e-3), base.derivative(state, -1e-3))
        self.assertEqual(law.derivative(0., -100*law.jreset_a), 0.)
        self.assertGreater(law.derivative(0., -200*law.jreset_a), 0.)
        self.assertGreater(law.derivative(law.state_limit, -1e-3), 0.)
        # Isolate the new state-feedback contribution at fixed current. The
        # full fixed-supply derivative additionally includes -chi_external.
        state, step, current = 1., 1e-5, -1.
        difference = ((law.derivative(state+step, current)-law.derivative(state-step, current))-
                      (base.derivative(state+step, current)-base.derivative(state-step, current)))
        np.testing.assert_allclose(difference/(2*step),
                                   law.kreset_per_v*5.*math.exp(-state), rtol=2e-9)

    def test_reset_initiation_does_not_change_transport_and_is_log_safe(self):
        base, modified = model(), model(reset_initiation_log=5.)
        voltage = np.array([-5., -.2, 0., .2, 5.])
        for state in (0., .1, 3., 12.):
            np.testing.assert_array_equal(base.current(voltage, state), modified.current(voltage, state))
            currents = base.current(voltage, state)
            np.testing.assert_array_equal(base.voltage(currents, state), modified.voltage(currents, state))
            self.assertEqual(base.saturation_current(state), modified.saturation_current(state))
            self.assertEqual(base.set_stopping_bound(1e-3), modified.set_stopping_bound(1e-3))
        extreme = model(reset_initiation_log=1e308)
        self.assertEqual(extreme.derivative(0., -1e300), 0.)
        self.assertEqual(extreme.derivative(80., -1e300), 0.)
        self.assertEqual(extreme.derivative(2., 1e-3), base.derivative(2., 1e-3))
        for invalid in (-1., math.nan, math.inf):
            with self.assertRaises(ValueError):
                model(reset_initiation_log=invalid)


if __name__ == '__main__':
    unittest.main()
