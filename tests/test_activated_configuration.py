import unittest
from unittest.mock import patch
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import logsumexp
from vulcan2d import activated_configuration as ac
from vulcan2d.coupled_charge import EDGES, log_rates, stationary, charging, VT, LN10
from vulcan2d.side_charge_circuit import load_roots


def full_generator(voltage, shared, p, ed, fast_factor=1.):
    """Independent eight-state generator, q-major then 00,10,01,11."""
    generator = np.zeros((8, 8))
    for config in (0, 1):
        s = np.r_[ed+(1-config)*p[1], shared[1:5], 0.]
        for (a, b), rate in zip(EDGES, np.exp(log_rates(voltage, s, p[2+config])[0])):
            a += 4*config; b += 4*config
            generator[b, a] += fast_factor*rate
            generator[a, a] -= fast_factor*rate
    for j, (n, m) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
        delta = p[5]-p[6]*voltage-n*p[1]+m*(p[3]-p[2])
        base = p[4]+p[7]*voltage**2/VT
        for a, b, rate in ((j, j+4, np.exp(base-delta/(2*VT))),
                           (j+4, j, np.exp(base+delta/(2*VT)))):
            generator[b, a] += rate; generator[a, a] -= rate
    return generator


class ActivatedConfigurationTests(unittest.TestCase):
    def test_fast_joint_against_independent_tree_solution(self):
        rng = np.random.default_rng(46321)
        for _ in range(40):
            s = np.array([rng.uniform(-.7, .7), *rng.uniform(0, 1, 3), rng.uniform(-7, 5), 0.])
            et = rng.uniform(-.7, .7); gt = rng.uniform(-6, 6)
            v = np.array([-3., -.1, 0., .1, 3.])
            fast = ac.fast_joint(v, s[0], et, s, gt)
            exact = stationary(v, np.r_[s[:5], gt], et)
            np.testing.assert_allclose(np.exp(fast['log_probability']), exact['probabilities'], rtol=2e-12, atol=2e-14)
            np.testing.assert_allclose(logsumexp(fast['log_probability'], axis=0), 0., atol=2e-14)
            mask = v != 0
            np.testing.assert_allclose(fast['log_abs_current_scale'][mask]/LN10,
                                       exact['log10_abs_current_scale'][mask], atol=2e-12)

    def test_zero_bias_partition_ratio_and_microscopic_balance(self):
        shared = np.array([.02, .3, .41, .13, -1., .11])
        p = np.array([-3., .08, -.02, -.06, -3., .07, .12, .002])
        table = ac.surfaces([0.], shared, p)
        u = charging(np.r_[shared[:5], 0.])[0]
        for j, z in enumerate(table['nodes']):
            ed = shared[0]+shared[5]*z
            energies = np.array([n*(ed+(1-q)*p[1])+m*p[2+q]+u*n*m+q*p[5]
                                 for q in (0, 1) for n, m in ((0, 0), (1, 0), (0, 1), (1, 1))])
            expected = logsumexp(-energies[4:]/VT)-logsumexp(-energies[:4]/VT)
            self.assertAlmostEqual(table['log_rates'][0, 0, j]-table['log_rates'][1, 0, j], expected, places=12)
            prob = np.exp(-energies/VT-logsumexp(-energies/VT))
            generator = full_generator(0., shared, p, ed)
            flux = generator*prob[None, :]
            np.testing.assert_allclose(flux, flux.T, rtol=2e-13, atol=1e-17)
            np.testing.assert_allclose(generator@prob, 0., atol=1e-16)

    def test_adiabatic_limit_against_full_eight_state_dynamics(self):
        shared = np.array([.015, .3, .5, .025, -.3, 0.])
        p = np.array([-3., .025, -.02, -.04, -2., .02, .04, .002])
        for v in (-.15, 0., .2):
            table = ac.surfaces([v], shared, p, panels=1, order=1)
            pi = np.exp(table['log_probability'][:, :, 0, 0])
            k01, k10 = np.exp(table['log_rates'][:, 0, 0]); rate = k01+k10
            initial = np.r_[.8*pi[0], .2*pi[1]]
            dt = .7; q1 = k01/rate+(.2-k01/rate)*np.exp(-rate*dt)
            expected = np.r_[(1-q1)*pi[0], q1*pi[1]]
            errors = []
            for speed in (1e2, 1e3, 1e4):
                actual = expm(dt*full_generator(v, shared, p, shared[0], speed))@initial
                errors.append(np.max(np.abs(actual-expected)))
            self.assertLess(errors[-1], 3e-6)
            self.assertLess(errors[-1], errors[0]/80)

    def test_native_circuit_history_zero_path_and_refinement(self):
        shared = [.12, .3, .4, .1, -2., .06]
        p = [-3., .1, .01, -.07, -4., .03, .1, .003]
        lookup = (np.array([0., .03, .2, 1.]), np.array([0., 1e-5, 3e-5, 5e-5]))
        one = np.r_[np.linspace(0, 1., 41), np.linspace(1., 0., 41), np.linspace(0, -1., 41), np.linspace(-1., 0., 41)]
        va = np.tile(one, 2)
        r = ac.trajectory(va, shared, p, lookup, lookup, diagnostics=True, max_step=.01, table_step=.005)
        np.testing.assert_allclose(r['states'].sum(axis=1), 1., atol=2e-14)
        self.assertTrue(np.all((r['states'] >= 0) & (r['states'] <= 1)))
        self.assertLess(r['maximum_kcl_relative_error'], 1e-8)
        repeated = np.flatnonzero(np.diff(va) == 0)+1
        np.testing.assert_array_equal(r['states'][repeated], r['states'][repeated-1])
        self.assertGreater(r['states'][len(one), 1].max(), .01)
        np.testing.assert_array_equal(r['states'][len(one)], r['states'][len(one)-1])
        self.assertTrue(np.all(r['root_count'] >= 1))
        finer = ac.trajectory(va, shared, p, lookup, lookup, max_step=.005, table_step=.0025, panels=8, order=12)
        mask = np.abs(va) >= .08
        error = np.max(np.abs(np.log10(r['current_A'][mask]/finer['current_A'][mask])))
        self.assertLess(error, .005)

    def test_native_attracting_root_matches_independent_enumeration(self):
        device_v = np.array([0, .2, .4, .6, .8, 1.])
        device_i = np.array([0, 1., .1, .1, .8, 1.])
        lookup = (np.array([0., 1.]), np.array([0., 1.]))
        def synthetic(v, *args, **kwargs):
            j = np.interp(np.abs(v), device_v, device_i)[:, None]
            return {'current': np.array([j, j]), 'log_rates': np.full((2, len(v), 1), -1000.), 'weights': np.ones(1)}
        with patch.object(ac, 'surfaces', synthetic):
            r = ac.trajectory([0., 1.], np.zeros(6), np.zeros(8), lookup, lookup,
                              table_step=.2, max_step=1., diagnostics=True)
        roots = load_roots(1., device_v, device_i, *lookup)
        self.assertEqual(r['root_count'][-1], len(roots))
        self.assertAlmostEqual(r['internal_voltage_V'][-1], roots[0]['internal_voltage_V'], places=12)
        self.assertTrue(roots[0]['stable_load_line'])

    def test_continuous_circuit_against_independent_adaptive_ode(self):
        shared = [.12, .3, .4, .1, -2., 0.]
        p = [-3., .1, .03, -.02, -3., .02, .08, .002]
        va = np.r_[np.linspace(0, .6, 9), np.linspace(.6, 0., 9),
                   np.linspace(0, -.5, 9), np.linspace(-.5, 0., 9)]
        conductance = .001; lookup = (np.array([0., 1.]), np.array([0., conductance]))
        def rate(applied, q):
            sign = 1 if applied >= 0 else -1
            def table(v): return ac.surfaces([sign*v], shared, p, panels=1, order=1)
            def mismatch(vd):
                t = table(abs(applied)-vd)['current'][:, 0, 0]
                return (1-q)*t[0]+q*t[1]-conductance*vd
            vd = brentq(mismatch, 0., abs(applied), xtol=1e-15) if applied else 0.
            k01, k10 = np.exp(table(abs(applied)-vd)['log_rates'][:, 0, 0])
            return k01*(1-q)-k10*q
        expected = [0.]
        for first, last in zip(va[:-1], va[1:]):
            distance = abs(last-first)
            if distance == 0: expected.append(expected[-1]); continue
            solution = solve_ivp(lambda s, q: [rate(first+(last-first)*s/distance, q[0])],
                                 (0., distance), [expected[-1]], rtol=2e-9, atol=1e-11)
            self.assertTrue(solution.success); expected.append(solution.y[0, -1])
        native = ac.trajectory(va, shared, p, lookup, lookup, panels=1, order=1, table_step=.001, max_step=.005)
        np.testing.assert_allclose(native['states'][:, 1, 0], expected, rtol=0, atol=2e-5)

    def test_nonfinite_or_empty_lookup_rejected_before_native_call(self):
        good = ([0., 1.], [0., .001]); shared = [.12, .3, .4, .1, -2., 0.]
        p = [-3., .1, .03, -.02, -3., .02, .08, .002]
        for bad in (([], []), ([0., np.nan], [0., .001]), ([0., 1.], [0., np.inf])):
            with self.assertRaises(ValueError): ac.trajectory([0., .1], shared, p, bad, good)


if __name__ == '__main__':
    unittest.main()
