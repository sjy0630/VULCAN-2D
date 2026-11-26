"""Independent integral/limit tests for the proposed readout, not model evidence."""
import unittest
import numpy as np
from scipy.integrate import quad
from scipy.special import expit
from spectral_readout import spectral_current,SpectralReadout,KB_EV_PER_K,gaussian_rule

class SpectralTests(unittest.TestCase):
    def test_single_level_against_energy_quadrature(self):
        thermal=KB_EV_PER_K*300
        for v,gamma,epsilon,eta in [(2.,.003,.4,.2),(-.3,.05,.1,.7),(.03,.4,-.2,.5),(1.,2.,.5,.9)]:
            center=epsilon+eta*v
            f=lambda e:gamma**2/((e-center)**2+gamma**2)*(expit((v-e)/thermal)-expit(-e/thermal))
            # Split explicitly at the resonance; this integration does not
            # use the digamma representation of the production formula.
            lo,hi=min(0,v,center)-2.,max(0,v,center)+2.
            intervals=sorted(set([lo,hi,0.,v,center]))
            reference=sum(quad(f,a,b,epsabs=1e-13,epsrel=1e-10,limit=300)[0] for a,b in zip(intervals[:-1],intervals[1:]))/93.48
            self.assertAlmostEqual(spectral_current(v,gamma,epsilon,0,eta,93.48)/reference,1.,places=8)

    def test_gaussian_average_normalization_and_passivity(self):
        self.assertAlmostEqual(gaussian_rule(96)[1].sum(),1.,places=14)
        v=np.linspace(-5,5,1001)
        for eta in (0.,.2,.5,1.):
            current=spectral_current(v,.005,.4,.25,eta,93.48)
            self.assertTrue(np.all(current*v>=0))
            self.assertTrue(np.all(np.diff(current)>=-1e-16))
            self.assertTrue(np.all(np.diff(current)/np.diff(v)<=1/93.48+1e-12))

    def test_ohmic_limit_and_state_order(self):
        law=SpectralReadout(93.48,1.,.3,.2,.2)
        v=np.array([-4.,-.1,0.,.1,4.])
        np.testing.assert_array_equal(law.current(v,0.),v/93.48)
        np.testing.assert_allclose(law.current(v,1e-8),v/93.48,rtol=1e-10)
        self.assertTrue(np.all(abs(law.current(v,2.))>=abs(law.current(v,4.))))

    def test_zero_bias_and_nested_width(self):
        self.assertEqual(spectral_current(0.,.01,.2,.4,.2,100.),0.)
        self.assertEqual(spectral_current(.5,.01,.2,0.,.2,100.,order=16),spectral_current(.5,.01,.2,0.,.2,100.,order=256))

if __name__=='__main__':unittest.main()
