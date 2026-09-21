import sys
from pathlib import Path
import unittest
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vulcan2d.bidirectional_state import (THERMAL_VOLTAGE, log_rates, exact_fixed_voltage_step,
                                         trajectory, conditional_barriers)
from vulcan2d.reset_history import log_shapes


class BidirectionalStateTests(unittest.TestCase):
    def parameters(self, polar=False):
        base = [-9., 6., -5.5, 5., 6., .2, .3, .5]
        return base+[.1] if polar else base

    def test_energy_ratio_and_positive_conditional_barriers(self):
        v = np.linspace(0, 1.7, 101)
        for name, p in [('single_hop', self.parameters()), ('polarizable_saddle', self.parameters(True))]:
            build, remove = log_rates(v, p, name)
            np.testing.assert_allclose(build-remove, -(p[5]-p[6]*v)/THERMAL_VOLTAGE, atol=2e-14)
            eb, er = conditional_barriers(v, p, name)
            self.assertGreater(eb.min(), 0.); self.assertGreater(er.min(), 0.)
            np.testing.assert_allclose(eb-er, p[5]-p[6]*v, atol=2e-15)
            self.assertAlmostEqual(max(build.max(), remove.max()), p[4], places=12)

    def test_single_hop_rates_have_opposing_field_slopes(self):
        p = self.parameters(); v = np.linspace(0, 1.7, 51)
        for charge in (-.8, .8):
            p[6] = charge; build, remove = log_rates(v, p, 'single_hop')
            self.assertTrue(np.all(np.diff(build)*np.diff(remove) < 0))

    def test_exact_step_preserves_probability_and_matches_independent_ode(self):
        for state in (0., .3, 1.):
            got = exact_fixed_voltage_step(state, .3, np.log(2.), np.log(5.))
            independent = solve_ivp(lambda x, p:2*(1-p)-5*p, (0., .3), [state], rtol=1e-11, atol=1e-13)
            self.assertAlmostEqual(float(got), independent.y[0, -1], places=10)
            self.assertGreaterEqual(got, 0.); self.assertLessEqual(got, 1.)
            self.assertEqual(exact_fixed_voltage_step(state, 0., 5., 7.), state)
        self.assertAlmostEqual(float(exact_fixed_voltage_step(.9, 1., 800., 800.)), .5)

    def test_compiled_load_line_matches_independent_scalar_root(self):
        va = np.array([0., .1, .1, .1]); tv = np.linspace(0, 2, 1001); ti = tv*1e-4
        p = self.parameters(); p[4] = -20.
        got = trajectory(va, p, 'single_hop', tv, ti)
        for j, applied in enumerate(va[1:], 1):
            def residual(v):
                h, c = log_shapes(np.array([v]), [p[1], p[3]])
                current = 10**p[0]*np.exp(h[0])+10**p[2]*np.exp(c[0]+got['log_state'][j])
                return current-1e-4*(applied-v)
            root = brentq(residual, 0., applied, xtol=1e-14)
            self.assertAlmostEqual(got['internal_voltage_v'][j], root, places=12)
        self.assertLess(got['max_kcl_relative_error'], 1e-8)
        np.testing.assert_array_equal(got['log_state'][1:], got['log_state'][1])

    def test_clock_zero_path_and_explicit_state_handoff(self):
        tv = np.linspace(0, 2, 101); ti = 1e-4*tv
        zero = trajectory([0., 0., 0.], self.parameters(), 'single_hop', tv, ti, initial_log_state=-3.)
        np.testing.assert_array_equal(zero['log_state'], -3.)
        np.testing.assert_array_equal(zero['current_a'], 0.)
        va = np.r_[np.linspace(0, 1.5, 61), np.linspace(1.5, 0, 61)[1:]]
        full = trajectory(va, self.parameters(), 'single_hop', tv, ti, rtol=1e-8, atol=1e-10)
        first = trajectory(va[:61], self.parameters(), 'single_hop', tv, ti, rtol=1e-8, atol=1e-10)
        second = trajectory(va[60:], self.parameters(), 'single_hop', tv, ti,
                            initial_log_state=first['log_state'][-1], rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(full['log_state'][60:], second['log_state'], atol=1e-6)

    def test_independent_integrators_and_tolerances_agree(self):
        tv = np.linspace(0, 2, 501); ti = 1e-4*np.expm1(tv)
        va = np.r_[np.linspace(0, 1.5, 61), np.linspace(1.5, 0, 61)[1:]]
        p = self.parameters(True)
        a = trajectory(va, p, 'polarizable_saddle', tv, ti, rtol=1e-8, atol=1e-10)
        b = trajectory(va, p, 'polarizable_saddle', tv, ti, rtol=1e-8, atol=1e-10, method='Radau')
        valid = va > .08
        self.assertLess(np.max(np.abs(np.log10(a['current_a'][valid]/b['current_a'][valid]))), 1e-5)
        self.assertTrue(np.all(a['log_state'] <= 1e-9))


if __name__ == '__main__':
    unittest.main()
