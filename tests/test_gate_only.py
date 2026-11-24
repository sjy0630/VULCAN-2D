import unittest

import numpy as np

from vulcan2d import gate_only
from vulcan2d.model_shared_device import Circuit


class GateOnlyTest(unittest.TestCase):
    def setUp(self):
        grid = np.linspace(0., 4., 401)
        self.circuit = Circuit(transistor=(grid, 3e-4*grid))
        self.wave = np.array([0., .25, .5, 1., .5, .25, 0.])

    def test_apparent_conductance_inverts_the_transistor_load(self):
        expected = 1e-4
        current, _ = self.circuit.operating_point(self.wave, lambda v: expected*v)
        log_g, usable = gate_only.apparent_log_conductance(self.wave,current,self.circuit.transistor)
        self.assertEqual(int(usable.sum()),5)
        np.testing.assert_allclose(log_g[usable],np.log10(expected),atol=1e-7)

    def test_zero_edge_holds_correction_and_target_uses_its_load_line(self):
        correction=np.full_like(self.wave,.3)
        profile={.9:{"local_delta": correction},1.1:{"local_delta": correction}}
        current, _ = self.circuit.operating_point(self.wave, lambda v: 1e-4*v)
        prediction=gate_only.current_for_method(self.wave,self.circuit,gate_only.log_abs(current),
                                                 profile,1.4,"local_delta",0.)
        expected, _ = self.circuit.operating_point(self.wave,lambda v: 10**.3*1e-4*v)
        np.testing.assert_allclose(prediction,expected,atol=4e-12)
        np.testing.assert_equal(gate_only.interpolate((.9,1.1), np.array([[1.],[2.]]),1.4),[2.])

    def test_erased_state_is_used_at_the_apex_and_on_the_return(self):
        wave = -self.wave
        baseline, _ = self.circuit.operating_point(self.wave, lambda v: 2e-3*v)
        transport = {"beta_sqrt_v": 2., "log_conductances": {.9: [-7., -5.], 1.1: [-6., -4.]}}
        result = gate_only.current_with_reset_return(wave, self.circuit, -baseline, 1.3, transport)
        expected, _ = self.circuit.operating_point(
            self.wave, lambda v: v*(1e-6*np.exp(2.*np.sqrt(v)) + 1e-4))
        np.testing.assert_array_equal(result[:3], -baseline[:3])
        np.testing.assert_allclose(result[3:], -expected[3:], atol=1e-12)
        self.assertTrue(np.all(np.abs(result) <= self.circuit.supply_bound(self.wave)))


if __name__ == "__main__":
    unittest.main()
