import unittest
import numpy as np
from scipy.integrate import quad
from scipy.special import expit, log_expit
from vulcan2d import finite_readout_support as fr
from vulcan2d import retained_charge_configuration as rc
from vulcan2d.coupled_charge import VT, charging
from vulcan2d.side_charge_circuit import load_roots


class FiniteReadoutTests(unittest.TestCase):
    def test_shared_series_load_matches_ohmic_solution(self):
        grid = np.linspace(-1., 1., 401); conductances = np.array([.0001, .0003, .0007, .002])
        populations = np.array([[.1, .2, .3, .4], [1., 0., 0., 0.]])
        applied = np.array([0., .01, .1, .7, 0., -.1, -.8]); gt = .0004
        r = fr.table_load(applied, grid, abs(grid[:, None])*conductances, populations, ([0., 1.], [0., gt]), ([0., 1.], [0., gt]))
        g = populations@conductances
        expected = (g*gt/(g+gt))[:, None]*abs(applied[None, :])
        # Uniform signed-grid coordinate rounding is amplified at tiny voltage.
        # This remains much tighter than the independently required 1e-8 KCL.
        np.testing.assert_allclose(r['current_A'], expected, rtol=1e-11, atol=1e-22)
        # Averages of separately loaded branches give a different circuit.
        wrong = populations@(conductances*gt/(conductances+gt))
        self.assertGreater(abs(wrong[0]-g[0]*gt/(g[0]+gt)), 1e-5)

    def test_exact_mapping_from_previous_conditional_readout(self):
        shared = [.45, .31, .54, .14, -3., .35]
        p = [-3., .13, -.1, .03, -3., .02, .08, .002, -.5]
        mapped = [p[0]+log_expit(shared[4])/np.log(10.), shared[0], shared[5], shared[1], p[1], charging(np.r_[shared[:5], 0.])[0]]
        v = np.linspace(-2., 2., 181)
        old = rc.surfaces(v, shared, p, panels=12, order=8)
        np.testing.assert_allclose(fr.conditional_currents(v, mapped), old['current']@old['weights'], rtol=3e-14, atol=0.)

    def test_zero_width_and_adaptive_spectrum_reference(self):
        p = np.array([-4., .18, .12, .4, .13, .2]); v = np.array([-.8, -.2, .05, .3, 1.])
        actual = fr.conditional_currents(v, p)
        for state, shift in enumerate((p[4], p[4]+p[5], 0., p[5])):
            for j, voltage in enumerate(v):
                def kernel(e):
                    return abs(expit(-(e-p[3]*voltage)/VT)-expit(-(e+(1-p[3])*voltage)/VT))
                reference = 10**p[0]*quad(kernel, p[1]+shift-p[2], p[1]+shift+p[2], epsabs=1e-18, epsrel=1e-12)[0]/(2*p[2])
                self.assertAlmostEqual(actual[state, j]/reference, 1., delta=2e-10)
        p[2] = 0.; values = fr.conditional_currents(v, p)
        energy = p[1]+np.array([p[4], p[4]+p[5], 0., p[5]])[:, None]
        exact = 10**p[0]*abs(expit(-(energy-p[3]*v)/VT)-expit(-(energy+(1-p[3])*v)/VT))
        np.testing.assert_allclose(values, exact, rtol=1e-11, atol=1e-20)

    def test_frozen_roots_against_independent_piecewise_solver(self):
        p = [-4., .13, .08, .4, .12, .2]; grid = np.linspace(-1., 1., 1001)
        currents = fr.conditional_currents(grid, p).T; pop = fr.fractions(2)
        va = np.array([.1, .5, .8, -.1, -.5, -.8]); lookup = ([0., .2, 1.], [0., 2e-5, 3e-5])
        result = fr.table_load(va, grid, currents, pop, lookup, lookup)
        for i in (0, 3, 8):
            for j, applied in enumerate(va):
                sign = np.sign(applied); v = np.linspace(0., 1., 501)
                values = fr.conditional_currents(sign*v, p).T@pop[i]
                roots = load_roots(abs(applied), v, values, *lookup)
                self.assertEqual(len(roots), 1)
                self.assertAlmostEqual(abs(result['internal_voltage_V'][i, j]), roots[0]['internal_voltage_V'], delta=1e-11)

    def test_nested_integer_configurations_and_oracle_floor(self):
        rng = np.random.default_rng(82)
        coarse, fine = fr.fractions(2), fr.fractions(4)
        self.assertEqual(len(coarse), 10); self.assertEqual(len(fine), 35)
        for row in coarse: self.assertTrue(np.any(np.all(fine == row, axis=1)))
        readouts = np.exp(rng.normal(size=(4, 80))); chosen = rng.integers(0, 4, 80)
        y = readouts[chosen, np.arange(80)]*np.exp(rng.normal(scale=.2, size=80))
        _, best = fr.nearest_configuration(y, readouts)
        realized = np.log10(readouts[chosen, np.arange(80)]/y)
        self.assertTrue(np.all(best**2 <= realized**2+1e-16))


if __name__ == '__main__': unittest.main()
