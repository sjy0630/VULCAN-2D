"""The inherited contact prior need not transfer across time-scale orderings."""
import unittest
import numpy as np
from scipy.special import log_expit, expit
from vulcan2d import retained_charge_configuration as rc
from vulcan2d.coupled_charge import charging, VT


class RetainedContactTests(unittest.TestCase):
    def test_frozen_current_invariant_but_occupation_and_slow_generator_change(self):
        shared = np.array([.12, .3, .4, .1, -5.2, .2])
        p = np.array([-3., .1, .03, -.02, -3., .02, .08, .002, -.5])
        voltage = np.r_[np.linspace(-3., -.005, 181), 0., np.linspace(.005, 3., 181)]
        reference = rc.surfaces(voltage, shared, p)
        for gl in (-12., -6., -3., 0., 3., 6., 12.):
            changed = shared.copy(); changed[4] = gl
            adjusted = p.copy(); adjusted[0] += (log_expit(shared[4])-log_expit(gl))/np.log(10.)
            actual = rc.surfaces(voltage, changed, adjusted)
            np.testing.assert_allclose(actual['current'], reference['current'], rtol=3e-14, atol=0.)
            # This invariance does not extend to the dynamics.
            self.assertGreater(np.max(abs(actual['n_log_probability']-reference['n_log_probability'])), .1)
            self.assertFalse(np.allclose(actual['log_rates'], reference['log_rates'], atol=1e-10, rtol=0))
            q = np.array([0, 0, 1, 1])[:, None, None]
            m = np.array([0, 1, 0, 1])[:, None, None]
            energy = (shared[0]+shared[5]*reference['nodes'][None, None, :]+(1-q)*p[1]
                      +(1-shared[1])*voltage[None, :, None]+charging(np.r_[shared[:5], 0.])[0]*m)
            fleft = expit((voltage[None, :, None]-energy)/VT); fright = expit(-energy/VT)
            expected_change = (expit(gl)-expit(shared[4]))*(fleft-fright)
            np.testing.assert_allclose(np.exp(actual['n_log_probability'][1])-np.exp(reference['n_log_probability'][1]),
                                       expected_change, atol=4e-15, rtol=1e-11)
            np.testing.assert_allclose(rc.initial_distribution(changed, adjusted),
                                       rc.initial_distribution(shared, p), rtol=2e-14, atol=0.)


if __name__ == '__main__': unittest.main()
