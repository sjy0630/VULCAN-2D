import unittest
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from vulcan2d import retained_charge_configuration as rc
from vulcan2d.retained_contact_memory import compensated_contact, continue_from_zero, switched_trajectory


class RetainedContactMemoryTests(unittest.TestCase):
    def setUp(self):
        self.s = np.array([.12, .3, .4, .1, -2., .06])
        self.p = np.array([-3., .1, .03, -.02, -3., .02, .08, .002, -.5])
        self.lookup = ([0., .1, 1.], [0., 3e-5, 1e-4])
        self.one = np.r_[np.linspace(0, .6, 31), np.linspace(.6, 0., 31),
                         np.linspace(0, -.5, 26), np.linspace(-.5, 0., 26)]
        self.settings = dict(panels=2, order=8, table_step=.005, max_step=.01)

    def test_unchanged_contact_split_matches_archived_trajectory_exactly(self):
        va = np.tile(self.one, 3)
        full = rc.trajectory(va, self.s, self.p, self.lookup, self.lookup, **self.settings)
        segmented = switched_trajectory(va, self.s, self.p, self.lookup, self.lookup,
                                        len(self.one), self.s[4], self.s[4], **self.settings)
        for field in ('current_A', 'states', 'internal_voltage_V', 'transistor_voltage_V'):
            np.testing.assert_array_equal(full[field], segmented[field])
        self.assertEqual(segmented['boundary_state_change'], 0.)

    def test_arbitrary_correlated_probability_is_carried_without_reset(self):
        rng = np.random.default_rng(92211)
        initial = rng.dirichlet([.7, 1.1, 1.3, .5], size=16).T
        saved = initial.copy(); va = np.r_[0., 0., self.one, 0., 0.]
        r = continue_from_zero(va, self.s, self.p, self.lookup, self.lookup, initial, **self.settings)
        np.testing.assert_array_equal(initial, saved)
        np.testing.assert_array_equal(r['states'][0], initial)
        repeat = np.flatnonzero(np.diff(va) == 0)+1
        np.testing.assert_array_equal(r['states'][repeat], r['states'][repeat-1])
        np.testing.assert_allclose(r['states'].sum(axis=1), 1., atol=2e-13, rtol=0)
        self.assertGreaterEqual(r['states'].min(), 0.)
        self.assertGreater(np.max(abs(r['states'][-1]-initial)), .01)

    def test_zero_duration_and_invalid_state_or_boundary(self):
        initial = np.full((4, 16), .25)
        r = continue_from_zero([0., 0., 0.], self.s, self.p, self.lookup, self.lookup, initial, **self.settings)
        np.testing.assert_array_equal(r['states'], np.broadcast_to(initial, (3, 4, 16)))
        np.testing.assert_array_equal(r['current_A'], 0.)
        for state in [initial[:, :-1], initial*.99, -initial, np.full_like(initial, np.nan)]:
            with self.assertRaises(ValueError):
                continue_from_zero(self.one, self.s, self.p, self.lookup, self.lookup, state, **self.settings)
        for at in [-1, 1, 17, len(self.one)]:
            with self.assertRaises(ValueError):
                switched_trajectory(self.one, self.s, self.p, self.lookup, self.lookup, at, -2., 2., **self.settings)
        with self.assertRaises(ValueError):
            continue_from_zero([.1, .2], self.s, self.p, self.lookup, self.lookup, initial, **self.settings)

    def test_contact_swap_preserves_readout_zero_bias_rates_and_state(self):
        s, p = compensated_contact(self.s, self.p, 12.)
        opposite_s, opposite_p = compensated_contact(s, p, -12.)
        self.assertGreater(opposite_p[0], rc.HI[0])
        v = np.array([-.5, 0., .5]); a = rc.surfaces(v, s, p); b = rc.surfaces(v, opposite_s, opposite_p)
        np.testing.assert_allclose(a['current'], b['current'], rtol=3e-14, atol=0)
        np.testing.assert_allclose(a['n_log_probability'][:, :, 1], b['n_log_probability'][:, :, 1], atol=1e-14, rtol=0)
        np.testing.assert_allclose(a['log_rates'][:, 1], b['log_rates'][:, 1], atol=2e-14, rtol=0)
        self.assertGreater(np.max(abs(a['log_rates']-b['log_rates'])), .1)
        r = switched_trajectory(np.tile(self.one, 2), s, p, self.lookup, self.lookup,
                                len(self.one), 12., -12., **self.settings)
        self.assertEqual(r['boundary_state_change'], 0.)

    def test_arbitrary_state_dynamics_against_independent_adaptive_ode(self):
        s = self.s.copy(); s[5] = 0.
        initial = np.array([[.11], [.31], [.23], [.35]])
        va = np.r_[np.linspace(0, .5, 9), np.linspace(.5, 0, 9),
                   np.linspace(0, -.4, 9), np.linspace(-.4, 0, 9)]
        conductance = .001; lookup = ([0., 1.], [0., conductance])
        def rhs(v, state):
            sign = 1 if v >= 0 else -1
            def surf(vd): return rc.surfaces([sign*(abs(v)-vd)], s, self.p, panels=1, order=1)
            def mismatch(vd): return state@surf(vd)['current'][:, 0, 0]-conductance*vd
            vd = brentq(mismatch, 0., abs(v), xtol=1e-15) if v else 0.
            generator = np.zeros((4, 4))
            for (a, b), rate in zip(rc.EDGES, np.exp(surf(vd)['log_rates'][:, 0, 0])):
                generator[b, a] += rate; generator[a, a] -= rate
            return generator@state
        expected = [initial[:, 0]]
        for start, end in zip(va[:-1], va[1:]):
            duration = abs(end-start)
            if duration == 0: expected.append(expected[-1]); continue
            r = solve_ivp(lambda t, state: rhs(start+(end-start)*t/duration, state),
                          (0., duration), expected[-1], rtol=2e-9, atol=1e-11)
            self.assertTrue(r.success); expected.append(r.y[:, -1])
        native = continue_from_zero(va, s, self.p, lookup, lookup, initial,
                                    panels=1, order=1, table_step=.001, max_step=.005)
        np.testing.assert_allclose(native['states'][:, :, 0], expected, atol=2e-5, rtol=0)


if __name__ == '__main__': unittest.main()
