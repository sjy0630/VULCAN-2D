import sys
from pathlib import Path
import unittest
import numpy as np
from scipy.integrate import quad
from scipy.special import expit
from scipy.constants import Boltzmann, elementary_charge

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from vulcan2d.barrier_state import mean_barrier_root, transmission, supply_conductance, BarrierTable, forward_circuit, effective_internal


class BarrierTests(unittest.TestCase):
    def test_spatial_action_matches_independent_quadrature(self):
        for energy in (-.5, .05, .35, 1.):
            for voltage in (0., 1e-8, .2, .7):
                expected = quad(lambda x: np.sqrt(max(.4-energy-voltage*x, 0.)), 0., 1., epsabs=1e-11)[0]
                self.assertAlmostEqual(float(mean_barrier_root(energy, voltage, .4)), expected, places=8)

    def test_ballistic_supply_and_mass_width_degeneracy(self):
        for v in (0., .01, .3):
            self.assertAlmostEqual(float(supply_conductance(v, [0.], .4)[0]), 1., places=11)
        np.testing.assert_allclose(transmission([-.2, .05, .2], .3, 2., .5, .25),
                                   transmission([-.2, .05, .2], .3, 1., .5, 1.), rtol=1e-13)

    def test_energy_integral_includes_thermal_over_barrier_supply(self):
        vt = Boltzmann*300/elementary_charge
        for v, g, phi in ((.2, 2., .5), (.1, 6., .1), (.3, 4., 1.)):
            def integrand(energy):
                supply = expit(-energy/vt)*expit((energy+v)/vt)*(-np.expm1(-v/vt))/v
                return float(transmission(energy, v, g, phi))*supply
            independent = quad(integrand, -v-2., 2., points=[-v, 0., phi-v, phi], epsabs=1e-20, epsrel=2e-9, limit=500)[0]
            got = float(supply_conductance(v, [g], phi, order=48)[0])
            self.assertAlmostEqual(np.log10(got/independent), 0., places=5)
        thermionic = float(supply_conductance(.1, [6.], .1, order=48)[0])
        self.assertGreater(thermionic, 1e-3)

    def test_table_circuit_derivatives_and_zero_bias(self):
        table = BarrierTable(.6, voltage_step=.005, width_step_nm=.05)
        voltage = np.array([0., .0937, .1843, .2817]); width = np.array([1.173, 1.173, 2.123, 3.173])
        tv = np.linspace(0., 1., 101); ti = 1e-4*tv
        def predict(amplitude, g):
            return forward_circuit(voltage, lambda v:table.internal(v, amplitude, g), tv, ti)
        actual = predict(-3., width)
        self.assertEqual(actual['current_a'][0], 0.)
        self.assertLess(actual['max_kcl_relative_error'], 1e-8)
        for key, delta in (('shared', 1e-5), ('local', 1e-5)):
            plus = predict(-3.+(delta if key == 'shared' else 0), width+(delta if key == 'local' else 0))['current_a'][1:]
            minus = predict(-3.-(delta if key == 'shared' else 0), width-(delta if key == 'local' else 0))['current_a'][1:]
            finite = np.log10(plus/minus)/(2*delta)
            np.testing.assert_allclose(actual['dlog10_current_d'+key][1:], finite, rtol=2e-4, atol=1e-6)

    def test_effective_ohmic_control_matches_series_resistors(self):
        va = np.array([0., .1, .2, .3]); tv = np.linspace(0., 1., 101); ti = 1e-4*tv
        result = forward_circuit(va, lambda v:effective_internal(v, 1., -6., 'power'), tv, ti)
        g = 1e-6/.2; expected = va/(1/g+1/1e-4)
        np.testing.assert_allclose(result['current_a'], expected, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
