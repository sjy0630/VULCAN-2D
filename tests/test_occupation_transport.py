import sys
import unittest
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'analysis'))
from vulcan2d.transport_candidates import Curve, TransportAssumptions, Q
from vulcan2d.occupation_transport import (current,log10_shape,fit_shared_shape,
    stationary_level,conditional_physics,NAMES,series_current)
from compare_occupation_transport import predict_target


class OccupationTransportTests(unittest.TestCase):
    def test_explicit_stationary_rates_match_shape_and_conserve_charge(self):
        a=TransportAssumptions();energy=.08;coupling=.2;gl=1e10;gr=2e10
        v=np.linspace(.1,3.,40);w=2*a.thermal_voltage_v/coupling;v0=2*energy/coupling
        p,i=stationary_level(v,energy,coupling,gl,gr,3.,a)
        from scipy.special import expit
        fl=expit((coupling*v/2-energy)/a.thermal_voltage_v)
        fr=expit((-coupling*v/2-energy)/a.thermal_voltage_v)
        np.testing.assert_allclose(gl*(fl-p),gr*(p-fr),rtol=1e-12)
        np.testing.assert_allclose(10**log10_shape(v,'occupation',[v0,w]),i/i[0],rtol=1e-12)
        self.assertTrue(np.all((p>=0)&(p<=1)))

    def test_equilibrium_and_large_bias_rate_limit(self):
        p,i=stationary_level(np.array([0.,100.]),.08,.2,1e10,2e10,3.)
        self.assertEqual(i[0],0.)
        self.assertAlmostEqual(i[1]/(Q*3*1e10*2e10/3e10),1.,places=12)
        self.assertAlmostEqual(p[1],1/3,places=12)

    def test_zero_odd_passive_and_monotone(self):
        for name,p in [('occupation',[1.,.25]),('occupation_background',[1.,.25,-2.]),
                       ('rollover',[2.,.5,1.]),('power',[1.7]),('schottky_free',[5.])]:
            v=np.r_[0.,np.geomspace(1e-9,8,250)]
            i=current(v,-8,name,p)
            self.assertEqual(i[0],0.)
            self.assertTrue(np.all(i[1:]>0))
            self.assertTrue(np.all(np.diff(i)>=0))
            np.testing.assert_array_equal(current(-v,-8,name,p),-i)
            self.assertAlmostEqual(float(log10_shape(.1,name,p)),0.)

    def test_asymptotes_and_crossover(self):
        p=[1.,.25,-2.]
        low=current(np.array([1e-7,2e-7]),-8,'occupation_background',p)
        self.assertAlmostEqual(low[1]/low[0],2.,places=10)
        high=current(np.array([1e6,2e6]),-8,'occupation_background',p)
        self.assertAlmostEqual(high[1]/high[0],2.,places=4)
        sat=current(np.array([30.,60.]),-8,'occupation',p[:2])
        self.assertAlmostEqual(sat[1]/sat[0],1.,places=12)
        for v,greater in ((.6,True),(2.,False)):
            span=np.array([v*.999,v*1.001])
            n=np.diff(log10_shape(span,'occupation_background',p))/np.diff(np.log10(span))
            self.assertEqual(bool(n[0]>1),greater)

    def test_full_window_synthetic_shape_recovery(self):
        p=np.array([1.1,.28,-2.2]);v=np.geomspace(.15,3,90)
        curves=[Curve(k,v,-8+k*.1+log10_shape(v,'occupation_background',p)) for k in range(1,5)]
        fit=fit_shared_shape(curves,'occupation_background')
        np.testing.assert_allclose(fit['parameters'],p,atol=1e-5)
        self.assertLess(fit['profiled_equal_cycle_rmse_decade'],1e-7)
        self.assertEqual(fit['training_cycle_ids'],[1,2,3,4])

    def test_very_low_voltage_cannot_distinguish_level_parameters(self):
        # With independent amplitude, both collapse to I~Gv; this is not a
        # statistical confidence interval but a concrete limiting degeneracy.
        v=np.geomspace(1e-7,1e-5,20)
        s1=log10_shape(v,'occupation',[.5,.2]);s2=log10_shape(v,'occupation',[2.,.8])
        delta=s1-s2
        self.assertLess(np.ptp(delta),1e-8)

    def test_conditional_parameter_mapping(self):
        a=TransportAssumptions();p=[1.,.25,-2.]
        d=conditional_physics('occupation_background',p,a)
        self.assertAlmostEqual(d['conditional_voltage_coupling'],2*a.thermal_voltage_v/.25)
        self.assertAlmostEqual(d['conditional_level_energy_eV'],a.thermal_voltage_v/.25)
        self.assertAlmostEqual(d['background_to_saturation_at_reference'],.01)

    def test_invalid_domains_are_rejected(self):
        for p in ([1.,0.],[-1.,.2],[1.,np.nan],[1.,.04]):
            with self.assertRaises(ValueError):log10_shape(.5,'occupation',p)
        with self.assertRaises(ValueError):log10_shape(0.,'occupation',[1.,.2])
        with self.assertRaises(ValueError):stationary_level(1.,.1,1.1,1e10,1e10)

    def test_series_solver_matches_two_resistors_and_nonlinear_balance(self):
        grid=np.linspace(0,5,100);it=2e-6*grid;va=np.linspace(0,4,80)
        i,vh=series_current(va,grid,it,np.log10(1e-6*.1),'power',[1.])
        np.testing.assert_allclose(i,va/(1e6+.5e6),rtol=1e-13)
        np.testing.assert_allclose(vh,va*2/3,rtol=1e-13)
        i,vh=series_current(va,grid,it,-8,'occupation_background',[1.,.25,-2.])
        np.testing.assert_allclose(i,np.interp(va-vh,grid,it),rtol=1e-12,atol=1e-20)
        self.assertTrue(np.all((vh>=0)&(vh<=va)))
        with self.assertRaises(ValueError):series_current([6.],grid,it,-8,'power',[1.])

    def test_full_bias_kernel_preserves_reservoir_bias_and_reverse_asymmetry(self):
        from scipy.special import expit
        p=[.08,.15];vt=TransportAssumptions().thermal_voltage_v
        v=np.linspace(-2,2,81)
        a=(p[0]-p[1]*v)/vt;b=(p[0]+(1-p[1])*v)/vt
        np.testing.assert_allclose(b-a,v/vt,atol=1e-13)
        f=expit(-a)-expit(-b)
        ref=expit((p[1]*.1-p[0])/vt)-expit((-(1-p[1])*.1-p[0])/vt)
        observed=current(v,np.log10(ref),'full_bias_level',p)
        np.testing.assert_allclose(observed,f,rtol=1e-12,atol=1e-14)
        self.assertTrue(np.all(np.diff(observed)>=0))
        self.assertGreater(abs(float(current(-.2,np.log10(ref),'full_bias_level',p))),
                           float(current(.2,np.log10(ref),'full_bias_level',p)))

    def test_high_target_values_do_not_change_prediction_parameters(self):
        v=np.linspace(.15,3,50);p=[1.,.25,-2.]
        cs=[Curve(k,v,-8+k*.1+log10_shape(v,'occupation_background',p)) for k in range(1,5)]
        y=cs[0].log10_current_a.copy();y[v>1.2]+=2
        changed=Curve(1,v,y)
        for protocol in ('loco_autonomous_branch','loco_full_shape_low_anchor','loco_voltage_extrapolation'):
            before=predict_target(cs[1:],cs[0],'occupation_background',protocol)
            after=predict_target(cs[1:],changed,'occupation_background',protocol)
            self.assertEqual(before['parameters'],after['parameters'])
            self.assertEqual(before['log10_reference_amplitude'],after['log10_reference_amplitude'])
            self.assertNotEqual(before['rmse_decade'],after['rmse_decade'])
        with self.assertRaises(ValueError):predict_target(cs,cs[0],'power','loco_autonomous_branch')


if __name__=='__main__':unittest.main()
