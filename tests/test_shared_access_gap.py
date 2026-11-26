"""Independent transport/circuit checks for the shared access-gap extension."""
from dataclasses import asdict
import inspect
import math
import unittest

import numpy as np
from scipy.optimize import brentq

from vulcan2d.shared_access_gap import AccessGapLaw, ACCESS_REFERENCE_CURRENT_A
from vulcan2d.shared_gap import GapLaw, BareGapCircuit, LookupGapCircuit, simulate_gap
from vulcan2d.transistor import TransistorLookup, TransistorMapping


def law(**kwargs):
    values = dict(ron_ohm=100., v0_v=.2, i0_a=2e-5, u_max=12.,
                  kset_per_v=.5, kreset_per_v=.4,
                  eth_set_v=.15, eth_reset_v=.2, escale_v=.3,
                  access_voltage_v=.5)
    values.update(kwargs)
    return AccessGapLaw(**values)


def base_law(model):
    values = asdict(model)
    values.pop("access_voltage_v")
    return GapLaw(**values)


def mapping(plateau=False):
    currents = [0., 1e-4, 1e-4, 4e-4] if plateau else [0., 1e-4, 2e-4, 5e-4]
    payload = {"schema_version": 1, "curves": [
        {"gate_voltage_v": gate, "drain_voltage_v": [0., 1., 2., 5.],
         "drain_current_a": currents} for gate in (1., 2.)]}
    lookup = TransistorLookup.from_payload(payload)
    return TransistorMapping(lookup, reset_lookup=lookup,
                             reset_policy="measured_reverse", series_resistance_ohm=37.)


def explicit_voltage(model, current, u, extra_resistance=0.):
    """Independent direct equation; never calls the production inverse."""
    return ((model.ron_ohm+extra_resistance)*current
            + u*model.v0_v*math.asinh(current*math.exp(u)/model.i0_a)
            + u*model.access_voltage_v*(current*math.exp(u)/1e-3)**3)


class SharedAccessGapTests(unittest.TestCase):
    def test_only_one_nonnegative_finite_extra_material_parameter(self):
        self.assertEqual(ACCESS_REFERENCE_CURRENT_A, 1e-3)
        self.assertEqual(set(AccessGapLaw.__dataclass_fields__)-set(GapLaw.__dataclass_fields__),
                         {"access_voltage_v"})
        for invalid in (-.1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                law(access_voltage_v=invalid)
        for target in (AccessGapLaw, AccessGapLaw.voltage, AccessGapLaw.current,
                       AccessGapLaw.drive, AccessGapLaw.derivative):
            self.assertFalse({"topology", "device", "gate_voltage", "cycle"}
                             & set(inspect.signature(target).parameters))

    def test_zero_extension_is_bit_identical_to_base_in_both_circuits(self):
        model = law(access_voltage_v=0.)
        base = base_law(model)
        currents = np.array([-1e-3, -1e-12, 0., 1e-12, 1e-3])
        for u in (0., .1, 3., 12.):
            np.testing.assert_array_equal(model.voltage(currents, u), base.voltage(currents, u))
            for v in (-5., -1e-18, 0., 1e-18, 5.):
                self.assertEqual(model.current(v, u), base.current(v, u))
                self.assertEqual(model._current_with_series(abs(v), u, 123.),
                                 base._current_with_series(abs(v), u, 123.))
        command = [0., .5, 1., .5, 0., -.5, -1., -.5, 0.]
        for circuit in (BareGapCircuit(1e-4), LookupGapCircuit(mapping(), 1.5)):
            reference = simulate_gap(command, 3., base, circuit, max_step_v=.02)
            result = simulate_gap(command, 3., model, circuit, max_step_v=.02)
            for field in ("state_u", "current_a", "hbn_voltage_v"):
                np.testing.assert_array_equal(result[field], reference[field])

    def test_scalar_vector_odd_passive_monotone_and_direct_equation(self):
        model = law()
        currents = np.r_[-np.geomspace(1e-2, 1e-16, 50), 0., np.geomspace(1e-16, 1e-2, 50)]
        for u in (0., .01, 2., 12.):
            voltage = model.voltage(currents, u)
            self.assertTrue(np.all(np.diff(voltage) > 0.))
            self.assertTrue(np.all(voltage*currents >= 0.))
            np.testing.assert_array_equal(model.voltage(-currents, u), -voltage)
            np.testing.assert_allclose(voltage, [model.voltage(i, u) for i in currents],
                                       rtol=2e-14, atol=0.)
            np.testing.assert_allclose(voltage, [explicit_voltage(model, i, u) for i in currents],
                                       rtol=2e-14, atol=0.)

    def test_closed_state_and_small_current_limits(self):
        model = law(access_voltage_v=100.)
        for v in (-2., 0., 2.):
            self.assertEqual(model.current(v, 0.), v/model.ron_ohm)
            self.assertEqual(model.voltage(v/model.ron_ohm, 0.), v)
        for u in (.1, 3., 12.):
            tangent = model.ron_ohm + u*model.v0_v*math.exp(u)/model.i0_a
            for current in (1e-30, 1e-18):
                np.testing.assert_allclose(model.voltage(current, u)/current, tangent, rtol=2e-13)
                np.testing.assert_allclose(model.current(model.voltage(current, u), u), current,
                                           rtol=2e-10, atol=0.)

    def test_added_series_drop_does_not_enter_migration_drive(self):
        model = law(access_voltage_v=3.)
        base = base_law(model)
        self.assertIs(AccessGapLaw.drive, GapLaw.drive)
        self.assertIs(AccessGapLaw.derivative, GapLaw.derivative)
        for u in (0., .1, 2., 8.):
            for current in (-1e-4, -1e-8, 0., 1e-8, 1e-4):
                self.assertEqual(model.drive(current, u), base.drive(current, u))
                self.assertEqual(model.derivative(u, current), base.derivative(u, current))
                if u and current:
                    self.assertGreater(abs(model.voltage(current, u)), abs(base.voltage(current, u)))

    def test_secant_resistance_can_increase_with_current(self):
        model = law(access_voltage_v=.5)
        base = base_law(model)
        u, low, high = 2., 1e-4, 5e-4
        self.assertLess(base.voltage(high, u)/high, base.voltage(low, u)/low)
        self.assertGreater(model.voltage(high, u)/high, model.voltage(low, u)/low)

    def test_passive_power_split_keeps_access_drop_out_of_gap_drive(self):
        model = law(access_voltage_v=.3)
        for u in (0., .3, 3., 8.):
            for current in (-2e-4, -1e-10, 0., 1e-10, 2e-4):
                ohmic_power = model.ron_ohm*current**2
                gap_power = abs(current)*u*model.drive(current, u)
                access_power = u*model.access_voltage_v*math.exp(3.*u)*current**4/1e-9
                for component in (ohmic_power, gap_power, access_power):
                    self.assertGreaterEqual(component, 0.)
                np.testing.assert_allclose(current*model.voltage(current, u),
                                           ohmic_power+gap_power+access_power,
                                           rtol=2e-14, atol=0.)

    def test_safe_bracket_when_old_tangent_is_not_lower_bound(self):
        model = law(access_voltage_v=1e6)
        u, target = 2., 1.
        old_lower = target/(model.ron_ohm + u*model.v0_v*math.exp(u)/model.i0_a)
        self.assertGreater(model.voltage(old_lower, u), target)
        current = model.current(target, u)
        self.assertLess(current, old_lower)
        np.testing.assert_allclose(model.voltage(current, u), target, rtol=2e-11)

    def test_inverse_against_independent_current_space_brent(self):
        for coefficient in (1e-8, .2, 200.):
            model = law(access_voltage_v=coefficient)
            for u in (.01, 2., 8.):
                for target in (1e-12, .01, 5.):
                    for extra in (0., 12000.):
                        scale = target/(model.ron_ohm+extra)
                        root = brentq(lambda x: explicit_voltage(model, scale*x, u, extra)-target,
                                      0., 1., xtol=1e-250, rtol=1e-14)
                        predicted = model._current_with_series(target, u, extra)
                        np.testing.assert_allclose(predicted, root*scale, rtol=3e-11, atol=0.)
                        np.testing.assert_allclose(explicit_voltage(model, predicted, u, extra),
                                                   target, rtol=3e-11, atol=0.)
                        self.assertEqual(model.current(-target, u), -model.current(target, u))

    def test_same_material_kcl_for_bare_and_series_signed_circuits(self):
        model = law(access_voltage_v=.2)
        transistor = mapping(plateau=True)
        series = LookupGapCircuit(transistor, 1.5)
        for u in (0., .3, 2., 6.):
            for command in (-3., -.1, .1, 4.):
                bare = BareGapCircuit(1e-4).solve(command, model, u)
                np.testing.assert_allclose(model.voltage(bare.current_a, u), bare.hbn_voltage_v,
                                           rtol=3e-11, atol=2e-13)
                np.testing.assert_allclose(bare.supply_voltage_v+bare.compliance_drop_v, command)
                point = series.solve(command, model, u)
                np.testing.assert_allclose(point.hbn_voltage_v+point.transistor_voltage_v, command,
                                           atol=2e-12, rtol=0.)
                np.testing.assert_allclose(transistor.drain_current(point.transistor_voltage_v, 1.5),
                                           point.current_a, rtol=2e-9, atol=1e-15)
                np.testing.assert_allclose(explicit_voltage(model, point.current_a, u),
                                           point.hbn_voltage_v, rtol=3e-11, atol=2e-13)


if __name__ == "__main__":
    unittest.main()
