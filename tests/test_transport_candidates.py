import unittest
import numpy as np

from vulcan2d.transport_candidates import (
    NAMES, Curve, TransportAssumptions, conditional_physics, current,
    fit_shared_shape, log10_shape, parameter_bounds, pf_beta,
)


class TestTransportPhysics(unittest.TestCase):
    def test_passivity_origin_reference_and_monotonicity(self):
        positive = np.geomspace(1e-7, 2.0, 100)
        for name in NAMES:
            bounds = parameter_bounds(name)
            parameter = None if bounds is None else sum(bounds) / 2
            with self.subTest(candidate=name):
                plus = current(positive, -8, name, parameter)
                minus = current(-positive, -8, name, parameter)
                self.assertEqual(current(0.0, -8, name, parameter), 0.0)
                self.assertAlmostEqual(current(.1, -8, name, parameter), 1e-8, places=18)
                np.testing.assert_allclose(minus, -plus)
                self.assertTrue(np.all(np.diff(plus) > 0))

    def test_classical_field_lowering_factors(self):
        beta = pf_beta(3.1)
        self.assertAlmostEqual(beta, 21.5256006905, places=7)
        self.assertAlmostEqual(conditional_physics("pf_free", beta)["apparent_epsilon_r"], 3.1)
        self.assertAlmostEqual(conditional_physics("schottky_free", beta/2)["apparent_epsilon_r"], 3.1)

    def test_hopping_low_field_and_conditional_length(self):
        v = np.array([.01, .1, .2])
        np.testing.assert_allclose(current(v, -8, "hopping", 1e-4), 1e-7*v, rtol=1e-8)
        upper = parameter_bounds("hopping")[1]
        self.assertAlmostEqual(conditional_physics("hopping", upper)["effective_hop_length_nm"], 6.0)

    def test_contact_prefactor_is_not_pf_prefactor(self):
        v = np.array([.5, 1.0])
        beta = 8.0
        change = np.diff(log10_shape(v, "schottky_free", beta))[0]
        self.assertAlmostEqual(change, beta*(1-np.sqrt(.5))/np.log(10), places=7)
        pf_change = np.diff(log10_shape(v, "pf_free", beta))[0]
        self.assertAlmostEqual(pf_change-change, np.log10(2), places=7)

    def test_recovers_shape_despite_six_decades_of_amplitude_variation(self):
        v = np.linspace(.2, 1.2, 30)
        for name, parameter in (("power", 1.4), ("pf_free", 6.0),
                                ("schottky_free", 9.0), ("hopping", 4.0)):
            curves = [Curve(i, v, a+log10_shape(v, name, parameter))
                      for i, a in enumerate((-12., -9., -6.))]
            fit = fit_shared_shape(curves, name)
            with self.subTest(candidate=name):
                self.assertAlmostEqual(fit["parameter"], parameter, places=5)
                self.assertLess(fit["profiled_equal_cycle_rmse_decade"], 1e-7)

    def test_physical_constraint_cannot_silently_be_violated(self):
        v = np.linspace(.5, 1.2, 30)
        curves = [Curve(i, v-1e-3*i, -9+i+log10_shape(v-1e-3*i, "pf_free", 6.0))
                  for i in range(3)]
        fit = fit_shared_shape(curves, "pf_constrained")
        self.assertTrue(fit["parameter_at_boundary"])
        self.assertAlmostEqual(fit["conditional_physics"]["apparent_epsilon_r"], 8.0)
        self.assertGreater(fit["profiled_equal_cycle_rmse_decade"], 0.1)

    def test_fixed_permittivity_is_a_zero_shape_freedom_constraint(self):
        assumptions = TransportAssumptions(epsilon_min=3.1, epsilon_max=3.1)
        v = np.linspace(.5, 1.2, 20)
        beta = pf_beta(3.1, assumptions)
        curves = [Curve(i, v, -10+i+log10_shape(v,"pf_constrained",beta,assumptions)) for i in range(3)]
        fit = fit_shared_shape(curves,"pf_constrained",assumptions)
        self.assertEqual(fit["parameter"], beta)
        self.assertLess(fit["profiled_equal_cycle_rmse_decade"],1e-10)

    def test_invalid_inputs_and_overflow_are_rejected(self):
        for v in (0, -1, np.nan):
            with self.assertRaises(ValueError):
                log10_shape(v, "pf_free", 2)
        with self.assertRaises(ValueError):
            TransportAssumptions(temperature_k=0)
        with self.assertRaises(ValueError):
            current(1, -8, "power", -1)
        with self.assertRaises(OverflowError):
            current(1e6, 10, "pf_free", 40)


if __name__ == "__main__":
    unittest.main()
