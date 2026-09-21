import unittest
from unittest.mock import patch
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import expit, logsumexp
from vulcan2d import retained_charge_configuration as rc
from vulcan2d import activated_configuration as ac
from vulcan2d.coupled_charge import EDGES, log_rates, charging, VT


def full_eight(voltage, shared, p, ed, fast=1.):
    """Independent microscopic generator, q-major, then n+2*m."""
    g = np.zeros((8, 8))
    for q in (0, 1):
        baseline = np.r_[ed+(1-q)*p[1], shared[1:5], 0.]
        elementary = np.exp(log_rates(voltage, baseline, p[2+q])[0])
        for i, ((a, b), rate) in enumerate(zip(EDGES, elementary)):
            rate *= fast if i < 8 else np.exp(p[8])
            a += 4*q; b += 4*q; g[b, a] += rate; g[a, a] -= rate
    for i, (n, m) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
        delta = p[5]-p[6]*voltage-n*p[1]+m*(p[3]-p[2])
        base = p[4]+p[7]*voltage**2/VT
        for a, b, rate in ((i, i+4, np.exp(base-delta/(2*VT))),
                           (i+4, i, np.exp(base+delta/(2*VT)))):
            g[b, a] += rate; g[a, a] -= rate
    return g


def from_logs(logs):
    q = np.zeros((4, 4))
    for (a, b), k in zip(rc.EDGES, np.exp(logs)):
        q[b, a] += k; q[a, a] -= k
    return q


class RetainedChargeConfigurationTests(unittest.TestCase):
    def test_zero_bias_full_partition_balance(self):
        shared = [.03, .3, .41, .1, -2., .12]
        p = [-3., .05, -.04, -.02, -3., .02, .08, .003, -.3]
        table = rc.surfaces([0.], shared, p)
        u = charging(np.r_[shared[:5], 0.])[0]
        for h, z in enumerate(table['nodes']):
            ed = shared[0]+shared[5]*z
            energy = np.array([q*p[5]+n*(ed+(1-q)*p[1])+m*p[2+q]+u*n*m
                               for q in (0, 1) for m in (0, 1) for n in (0, 1)])
            logz = logsumexp(-energy.reshape(4, 2)/VT, axis=1)
            pi = np.exp(logz-logsumexp(logz))
            q = from_logs(table['log_rates'][:, 0, h]); flux = q*pi[None, :]
            np.testing.assert_allclose(flux, flux.T, rtol=4e-13, atol=1e-17)
            np.testing.assert_allclose(q@pi, 0., atol=2e-16)

    def test_conductor_fast_limit_against_full_eight_states(self):
        shared = [.02, .3, .5, .025, -.3, 0.]
        p = [-3., .025, -.02, -.04, -2., .02, .04, .002, -.5]
        slow_initial = np.array([.1, .2, .3, .4]); dt = .4
        for v in (-.15, 0., .2):
            table = rc.surfaces([v], shared, p, panels=1, order=1)
            conditional = np.exp(table['n_log_probability'][:, :, 0, 0]).T
            initial = (slow_initial[:, None]*conditional).ravel()
            slow = expm(dt*from_logs(table['log_rates'][:, 0, 0]))@slow_initial
            expected = (slow[:, None]*conditional).ravel(); errors = []
            for fast in (1e2, 1e3, 1e4):
                actual = expm(dt*full_eight(v, shared, p, shared[0], fast))@initial
                errors.append(np.max(abs(actual-expected)))
            self.assertLess(errors[-1], 1e-5)
            self.assertLess(errors[-1], errors[0]/80)

    def test_positive_native_exponential_against_scipy(self):
        rng = np.random.default_rng(9917)
        for _ in range(60):
            logs = rng.uniform(-6, 6, 8); dt = 10**rng.uniform(-3, 1)
            actual = rc.propagator(logs, dt); exact = expm(from_logs(logs)*dt)
            self.assertGreaterEqual(actual.min(), 0.)
            np.testing.assert_allclose(actual.sum(axis=0), 1., atol=3e-16)
            np.testing.assert_allclose(actual, exact, rtol=1e-10, atol=2e-12)

    def test_stiff_exponential_retains_slow_and_tiny_transfers(self):
        for side_logs in ([0., -2., 1., -3.], [-100., -101., -102., -99.]):
            logs = np.r_[side_logs, 120., 115., 119., 118.]; dt = .4
            # q equilibrates almost instantly; m still evolves on its own scale.
            q1 = np.array([expit(logs[4]-logs[5]), expit(logs[6]-logs[7])]); q0 = 1-q1
            k01 = q0[0]*np.exp(logs[0])+q1[0]*np.exp(logs[2])
            k10 = q0[1]*np.exp(logs[1])+q1[1]*np.exp(logs[3]); total = k01+k10
            moved = -np.expm1(-total*dt); mprop = np.array([[1-k01/total*moved, k10/total*moved],
                                                         [k01/total*moved, 1-k10/total*moved]])
            expected = np.array([[q0[0]*mprop[0, 0], q0[0]*mprop[0, 1], q0[0]*mprop[0, 0], q0[0]*mprop[0, 1]],
                                 [q0[1]*mprop[1, 0], q0[1]*mprop[1, 1], q0[1]*mprop[1, 0], q0[1]*mprop[1, 1]],
                                 [q1[0]*mprop[0, 0], q1[0]*mprop[0, 1], q1[0]*mprop[0, 0], q1[0]*mprop[0, 1]],
                                 [q1[1]*mprop[1, 0], q1[1]*mprop[1, 1], q1[1]*mprop[1, 0], q1[1]*mprop[1, 1]]])
            np.testing.assert_allclose(rc.propagator(logs, dt), expected, rtol=2e-11, atol=0.)

    def test_fast_side_limit_matches_correct_small_side_over_right_limit(self):
        shared = [.03, .3, .41, .1, -2., .12]
        p = [-3., .05, -.04, -.02, -3., .02, .08, .003, 8.]
        voltage = np.array([-.5, -.15, 0., .15, .5])
        table = rc.surfaces(voltage, shared, p); rates = np.exp(table['log_rates'])
        effective = []
        side = []
        for q in (0, 1):
            m1 = rates[2*q]/(rates[2*q]+rates[2*q+1]); side.append(m1)
        effective = np.array([(1-side[0])*rates[4]+side[0]*rates[6],
                              (1-side[1])*rates[5]+side[1]*rates[7]])
        old = ac.surfaces(voltage, shared, p[:8], panels=8, order=8, log_side_rate=-35.)
        np.testing.assert_allclose(np.log(effective), old['log_rates'], rtol=0, atol=1e-11)
        currents = np.array([(1-side[q])*table['current'][2*q]+side[q]*table['current'][2*q+1] for q in (0, 1)])
        np.testing.assert_allclose(currents, old['current'], rtol=3e-13, atol=1e-25)

    def test_native_history_zero_duration_and_refinement(self):
        shared = [.12, .3, .4, .1, -2., .06]
        p = [-3., .1, .03, -.02, -3., .02, .08, .002, -.5]
        lookup = ([0., .1, 1.], [0., 3e-5, 1e-4])
        one = np.r_[np.linspace(0, .6, 31), np.linspace(.6, 0., 31), np.linspace(0, -.5, 26), np.linspace(-.5, 0., 26)]
        va = np.tile(one, 3)
        first = rc.trajectory(va, shared, p, lookup, lookup, panels=4, order=8, table_step=.005, max_step=.01)
        fine = rc.trajectory(va, shared, p, lookup, lookup, panels=8, order=8, table_step=.0025, max_step=.005)
        np.testing.assert_allclose(first['states'].sum(axis=1), 1., atol=2e-13)
        self.assertTrue(np.all(first['states'] >= 0))
        repeated = np.flatnonzero(np.diff(va) == 0)+1
        np.testing.assert_array_equal(first['states'][repeated], first['states'][repeated-1])
        self.assertGreater(np.max(abs(first['states'][len(one)]-first['states'][0])), .01)
        m = abs(va) >= .08
        self.assertLess(np.max(abs(np.log10(first['current_A'][m]/fine['current_A'][m]))), .002)
        self.assertLess(first['maximum_kcl_relative_error'], 1e-8)

    def test_continuous_circuit_against_independent_adaptive_ode(self):
        shared = [.12, .3, .4, .1, -2., 0.]
        p = [-3., .1, .03, -.02, -3., .02, .08, .002, -.5]
        va = np.r_[np.linspace(0, .6, 9), np.linspace(.6, 0., 9),
                   np.linspace(0, -.5, 9), np.linspace(-.5, 0., 9)]
        conductance = .001; lookup = ([0., 1.], [0., conductance])
        def rhs(applied, state):
            sign = 1 if applied >= 0 else -1
            def table(v): return rc.surfaces([sign*v], shared, p, panels=1, order=1)
            def mismatch(vd): return state@table(abs(applied)-vd)['current'][:, 0, 0]-conductance*vd
            vd = brentq(mismatch, 0., abs(applied), xtol=1e-15) if applied else 0.
            return from_logs(table(abs(applied)-vd)['log_rates'][:, 0, 0])@state
        reference = [rc.initial_distribution(shared, p, panels=1, order=1)[:, 0]]
        for first, last in zip(va[:-1], va[1:]):
            ds = abs(last-first)
            if ds == 0: reference.append(reference[-1]); continue
            solved = solve_ivp(lambda s, state: rhs(first+(last-first)*s/ds, state),
                               (0., ds), reference[-1], rtol=2e-9, atol=1e-11)
            self.assertTrue(solved.success); reference.append(solved.y[:, -1])
        native = rc.trajectory(va, shared, p, lookup, lookup, panels=1, order=1, table_step=.001, max_step=.005)
        np.testing.assert_allclose(native['states'][:, :, 0], reference, atol=2e-5, rtol=0)

    def test_exact_contact_current_identity_away_from_electronic_equilibrium(self):
        shared = [.02, .3, .5, .025, -.3, 0.]
        p = [-3., .025, -.02, -.04, -2., .02, .04, .002, -.5]
        probability = np.array([.05, .1, .2, .15, .03, .12, .21, .14]); fast = 17.
        for v in (-.2, 0., .2):
            g = full_eight(v, shared, p, shared[0], fast); derivative = g@probability
            dn = derivative.reshape(4, 2)[:, 1].sum()
            dm = derivative[[2, 3, 6, 7]].sum(); left = 0.; right = 0.
            for q in (0, 1):
                s = np.r_[shared[0]+(1-q)*p[1], shared[1:5], 0.]
                k = np.exp(log_rates(v, s, p[2+q])[0]); z = probability[q*4:q*4+4]
                left += fast*(k[0]*z[0]-k[1]*z[1]+k[4]*z[2]-k[5]*z[3])
                right += fast*(k[2]*z[0]-k[3]*z[1]+k[6]*z[2]-k[7]*z[3])
                right += np.exp(p[8])*(k[8]*z[0]-k[9]*z[2]+k[10]*z[1]-k[11]*z[3])
            table = rc.surfaces([v], shared, p, panels=1, order=1)
            marginal = probability.reshape(4, 2).sum(axis=1)
            stationary_current = fast*np.sign(v)*(marginal@table['current'][:, 0, 0])/10**p[0]
            wl = expit(shared[4])
            self.assertAlmostEqual(left, stationary_current+wl*dn, places=12)
            self.assertAlmostEqual(right, -stationary_current+(1-wl)*dn+dm, places=12)

    def test_frozen_conditional_readout_is_monotone(self):
        shared = [.12, .3, .4, .1, -2., .35]
        p = [-3., .1, .03, -.02, -3., .02, .08, .002, -.5]
        v = np.linspace(-3., 3., 801)
        current = rc.surfaces(v, shared, p)['current']*np.sign(v)[None, :, None]
        self.assertGreaterEqual(np.diff(current, axis=1).min(), -1e-18)


if __name__ == '__main__': unittest.main()
