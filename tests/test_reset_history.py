import sys
from pathlib import Path
import unittest
import numpy as np
from scipy.integrate import solve_ivp, quad

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'analysis'))
from vulcan2d.reset_history import thermal_clock, power_step, log_shapes, predict_internal, terminal_trajectory, identify_from_outward
from check_negative_reentry import events


class ResetHistoryTests(unittest.TestCase):
    def waves(self):
        v = np.r_[np.linspace(0, 1.7, 50), np.linspace(1.7, 0, 50)]
        return v[None, :], v[None, :]*.95

    def test_exact_piecewise_linear_thermal_integral(self):
        va = np.array([[0., .3, .3, .8, .5, 0.]])
        vh = np.array([[0., .25, .25, .7, .45, 0.]])
        actual = thermal_clock(va, vh, 9.2)[0]; expected = [0.]
        for j in range(1, len(va[0])):
            integral = abs(va[0, j]-va[0, j-1])*quad(lambda t: np.exp(9.2*(vh[0, j-1]+t*(vh[0, j]-vh[0, j-1])-.5)), 0, 1)[0]
            expected.append(expected[-1]+integral)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-13)

    def test_power_feedback_exact_step_matches_independent_ode(self):
        p = np.array([6., 5.]); q = np.array([[-9., -6., -.5]])
        v = np.array([.7]); duration = np.array([.08]); initial = .8
        h, c = log_shapes(v, p); H = float(10**q[0, 0]*np.exp(h[0])); C = float(10**q[0, 1]*np.exp(c[0]))
        k = np.log(2.)/10**q[0, 2]
        solution = solve_ivp(lambda t, x: -k*.7*(H+C*x)*x/1e-6, (0, duration[0]), [initial], rtol=1e-11, atol=1e-13)
        got = np.exp(power_step(np.log(np.array([initial])), v, duration, p, q))[0]
        self.assertAlmostEqual(got, solution.y[0, -1], places=10)

    def test_power_zero_drive_and_zero_path_preserve_state(self):
        p = np.array([6., 5.]); q = np.array([[-9., -6., -.5], [-9., -6., -.5]])
        initial = np.log([.2, .8])
        np.testing.assert_array_equal(power_step(initial, np.array([0., .8]), np.array([.2, 0.]), p, q), initial)

    def test_all_states_irreversible_and_effective_current_passive(self):
        va, vh = self.waves()
        for name, shared, scale in [('field_weibull', [6., 5., 4.], np.log10(.6)),
                                    ('thermal_rate', [6., 5., 9.], -1.), ('power_damage', [6., 5.], -.5)]:
            y, state = predict_internal(va, vh, name, shared, [[-9., -6., scale]], True)
            self.assertTrue(np.all(np.diff(state) <= 1e-12)); self.assertTrue(np.all(state <= 0))
            self.assertTrue(np.all(np.isfinite(y[vh > 0]))); self.assertTrue(np.all(np.isneginf(y[vh == 0])))
            if name == 'field_weibull':np.testing.assert_allclose(state[0, 49:], state[0, 49])
            else:self.assertLess(state[0, -1], state[0, 49])

    def test_terminal_solver_kcl_and_internal_time_step_refinement(self):
        va, vh = self.waves(); tv = np.linspace(0, 2, 1001); ti = tv*1e-4
        for name, shared, scale in [('field_weibull', [6., 5., 4.], np.log10(.6)),
                                    ('thermal_rate', [6., 5., 9.], -1.), ('power_damage', [6., 5.], -.5)]:
            a = terminal_trajectory(va, name, shared, [[-9., -6., scale]], tv, ti, substeps=4)
            b = terminal_trajectory(va, name, shared, [[-9., -6., scale]], tv, ti, substeps=8)
            mask = va > .08
            self.assertLess(np.max(np.abs(np.log10(a['current_a'][mask]/b['current_a'][mask]))), .005)
            self.assertLess(a['max_kcl_relative_error'], 1e-4)
            self.assertEqual(a['unconverged_athermal_steps'], 0)

    def test_return_values_cannot_enter_outward_identification(self):
        va, vh = self.waves(); shared = [6., 5., 4.]; q = [-9., -6., np.log10(.6)]
        y = predict_internal(va, vh, 'field_weibull', shared, [q])[0]
        y[~np.isfinite(y)] = -12.; w = np.where(va[0] > .08, 1., 0.); w /= w.sum()
        fit = {'name': 'field_weibull', 'shared_parameters': shared, 'training_cycle_ids': [1, 2]}
        a = identify_from_outward(va[0], vh[0], y, w, fit, 3)
        changed = y.copy(); changed[50:] += 5.; changed_vh = vh[0].copy(); changed_vh[50:] *= .2
        b = identify_from_outward(va[0], changed_vh, changed, w, fit, 3)
        np.testing.assert_allclose(a['local_parameters'], b['local_parameters'], atol=0.)
        with self.assertRaises(ValueError):identify_from_outward(va[0], vh[0], y, w, fit, 1)

    def test_tiny_survival_times_large_envelope_needs_relative_convergence(self):
        va, unused = self.waves(); tv = np.linspace(0, 2, 1001); ti = tv*1e-4
        shared = [6.87, 20., 1.35]; local = [[-8.825, -5.655, -1.008]]
        a = terminal_trajectory(va, 'field_weibull', shared, local, tv, ti, substeps=1)
        b = terminal_trajectory(va, 'field_weibull', shared, local, tv, ti, substeps=2)
        mask = va > .08
        self.assertLess(np.max(np.abs(np.log10(a['current_a'][mask]/b['current_a'][mask]))), 1e-6)
        self.assertEqual(a['unconverged_athermal_steps'], 0)

    def test_one_way_states_preserve_terminal_current_order(self):
        va, unused = self.waves(); tv = np.linspace(0, 2, 1001); ti = 1e-4*(1-np.exp(-tv))
        for name, shared, scale in [('field_weibull', [6., 5., 4.], np.log10(.6)),
                                    ('thermal_rate', [6., 5., 9.], -1.), ('power_damage', [6., 5.], -.5)]:
            result = terminal_trajectory(va, name, shared, [[-9., -6., scale]], tv, ti, substeps=2)
            outward = result['current_a'][0, :50]; returned = result['current_a'][0, 50:][::-1]
            self.assertTrue(np.all(returned <= outward*(1+1e-8)+1e-18))

    def test_persistent_drop_and_recovery_detection(self):
        from types import SimpleNamespace
        up = np.linspace(0, 1.5, 151); v = np.r_[up, up[::-1]]
        i = np.where(v < .4, 1e-5, np.where(v < 1.1, 1e-8, 1e-4)); i[v == 0] = 1e-10
        raw = {'voltage': -v, 'current': i}
        lookup = SimpleNamespace(inverse=lambda current:current/1e-4, at_voltage=lambda voltage:voltage*1e-4)
        found = events(raw, lookup)
        self.assertEqual([r['kind'] for r in found], ['drop', 'rise'])
        self.assertEqual([r['raw_index_after'] for r in found], [40, 110])
        self.assertTrue(found[1]['before_well_conditioned_lookup'])
        self.assertFalse(found[1]['after_well_conditioned_lookup'])

    def test_isolated_spike_is_not_a_persistent_event(self):
        from types import SimpleNamespace
        up = np.linspace(0, 1.5, 151); v = np.r_[up, up[::-1]]
        i = np.full(len(v), 1e-8); i[v == 0] = 1e-10; i[60] = 1e-4
        lookup = SimpleNamespace(inverse=lambda current:current/1e-4, at_voltage=lambda voltage:voltage*1e-4)
        self.assertEqual(events({'voltage': -v, 'current': i}, lookup), [])


if __name__ == '__main__':unittest.main()
