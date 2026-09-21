import sys
from pathlib import Path
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'analysis'))
from vulcan2d.contact_network import (network_log_rates,stationary_network,current,log10_shape,
                                     Pair,fit_shared_shape,profile_amplitude,charging_energy,ChargingAssumptions)
from vulcan2d.occupation_transport import current as single_current
from vulcan2d.transport_candidates import TransportAssumptions
from compare_contact_networks import predict_targets
from check_contact_state_information import adjust_energies


class ContactNetworkTests(unittest.TestCase):
    def test_detailed_balance_full_voltage_and_zero_equilibrium(self):
        p=[.1,-.08,.2,.5,1.,-2.];v=np.linspace(-3,3,51)
        a,b,c,d,e,f=network_log_rates(v,p)
        np.testing.assert_allclose(a+b+c-d-e-f,v/TransportAssumptions().thermal_voltage_v,atol=1e-12)
        prob,edges=stationary_network(0.,p)
        np.testing.assert_allclose(edges,0.,atol=1e-15)
        self.assertAlmostEqual(sum(prob),1.)
        self.assertEqual(current(0.,-8,'two_site_chain',p),0.)

    def test_stationary_probabilities_solve_master_equation(self):
        p=[.1,-.08,.2,.5,1.,-2.]
        for v in [-1.,-.2,.1,.8,2.]:
            logs=network_log_rates(v,p);a,b,c,d,e,f=np.exp(logs)
            generator=np.array([[-a-f,d,c],[a,-d-b,e],[f,b,-c-e]])
            matrix=generator.copy();matrix[-1]=1.
            expected=np.linalg.solve(matrix,np.array([0.,0.,1.]))
            prob,edges=stationary_network(v,p)
            np.testing.assert_allclose(prob,expected,rtol=1e-10,atol=1e-12)
            np.testing.assert_allclose(edges,edges[0],rtol=1e-8,atol=1e-14)
            self.assertTrue(np.all(prob>=0));self.assertAlmostEqual(sum(prob),1.)

    def test_tree_current_matches_normalized_kernel(self):
        p=[.05,-.03,.25,.4,-1.,1.];v=np.r_[np.linspace(-2,-.02,60),np.linspace(.02,2,60)]
        _,edges=stationary_network(v,p);_,ref=stationary_network(.1,p)
        got=current(v,np.log10(ref[0]),'two_site_chain',p)
        np.testing.assert_allclose(got,edges[0],rtol=1e-9,atol=1e-14)
        self.assertTrue(np.all(got*v>0))

    def test_contact_rate_reduces_to_fixed_level(self):
        v=np.r_[np.linspace(-2,-.01,40),np.linspace(.01,2,40)]
        base=single_current(v,-8,'full_bias_level',[.1,.3])
        for name,p in [('fixed_level',[.1,.3]),('contact_rate',[.1,.3,-2.,0.]),
                       ('two_contact_rates',[.1,.3,2.,0.,0.])]:
            np.testing.assert_allclose(current(v,-8,name,p),base,rtol=1e-12)

    def test_low_voltage_conductance_is_same_on_both_sides(self):
        for name,p in [('fixed_level',[.1,.3]),('contact_rate',[.1,.3,1.,2.]),
                       ('two_contact_rates',[.1,.3,1.,2.,-1.]),('two_site_chain',[.1,-.05,.2,.6,-1.,2.])]:
            i=current(np.array([-1e-9,1e-9]),-8,name,p)
            self.assertAlmostEqual(-i[0]/i[1],1.,places=6)

    def test_separated_rate_limits_match_full_network(self):
        p=[-.01,.01,.3,.5];v=np.r_[np.linspace(-.3,-.01,30),np.linspace(.01,.3,30)]
        for name,rates in [('equilibrium_left',[-12.,6.]),('equilibrium_right',[6.,-12.])]:
            approx=current(v,-8,name,p);exact=current(v,-8,'two_site_chain',p+rates)
            np.testing.assert_allclose(approx,exact,rtol=.002)
            near=current(np.array([-1e-9,1e-9]),-8,name,p)
            self.assertAlmostEqual(-near[0]/near[1],1.,places=6)

    def test_finite_charging_zero_and_large_limits(self):
        p=[.08,-.1,.3,.5];v=np.r_[np.linspace(-1,-.01,70),np.linspace(.01,1,70)]
        x2=p[2]+(1-p[2])*p[3]
        for side,singlep in [('left',[p[0],p[2]]),('right',[p[1],x2])]:
            # Negative-level energy maps to positive energy with the opposite
            # electrostatic lever arm, retaining the same full-bias law.
            if singlep[0]<0:singlep=[-singlep[0],1-singlep[1]]
            np.testing.assert_allclose(current(v,-8,'finite_'+side,p+[0.]),
                                       current(v,-8,'fixed_level',singlep),rtol=1e-12)
            np.testing.assert_allclose(current(v,-8,'finite_'+side,p+[3.]),
                                       current(v,-8,'equilibrium_'+side,p),rtol=1e-12)

    def test_charging_geometry_and_equilibrium_conductance(self):
        p=[.08,-.1,.3,.5,.8];a=ChargingAssumptions(dielectric_r=2.)
        u=charging_energy('coulomb_left',p,a)
        self.assertAlmostEqual(u,.8*1.439964/(2*6*.7*.5))
        for side in ['left','right']:
            i=current(np.array([-1e-9,1e-9]),-8,'coulomb_'+side,p,a)
            self.assertAlmostEqual(-i[0]/i[1],1.,places=6)

    def test_slow_contact_duality_is_not_spatial_identification(self):
        p=[.15,-.1,.25,.6];x1=p[2];x2=x1+(1-x1)*p[3]
        rx1=x2-x1;rx2=1-x1
        other=[p[1]-p[0],-p[0],rx1,(rx2-rx1)/(1-rx1)]
        v=np.r_[np.linspace(-3,-.01,60),np.linspace(.01,3,60)]
        np.testing.assert_allclose(current(v,-8,'equilibrium_left',p),
                                   current(v,-8,'equilibrium_right',other),rtol=1e-12)

    def test_finite_U_limit_matches_full_four_state_master_equation(self):
        from scipy.special import expit
        p=[.05,-.04,.25,.55,.15];vt=TransportAssumptions().thermal_voltage_v
        volts=np.array([-.2,-.1,.1,.2]);x2=p[2]+(1-p[2])*p[3]
        for side in ['left','right']:
            gl,gr=(1e-8,1.) if side=='left' else (1.,1e-8)
            actual=[]
            for v in volts:
                e1=p[0]+(1-p[2])*v;e2=p[1]+(1-x2)*v
                fl0=expit((v-e1)/vt);flu=expit((v-e1-p[4])/vt)
                fr0=expit(-e2/vt);fru=expit(-(e2+p[4])/vt)
                rates={(0,1):gl*fl0,(1,0):gl*(1-fl0),(2,3):gl*flu,(3,2):gl*(1-flu),
                       (0,2):gr*fr0,(2,0):gr*(1-fr0),(1,3):gr*fru,(3,1):gr*(1-fru),
                       (1,2):np.exp(-max(e2-e1,0)/vt),(2,1):np.exp(-max(e1-e2,0)/vt)}
                matrix=np.zeros((4,4))
                for (a,b),rate in rates.items():matrix[b,a]+=rate;matrix[a,a]-=rate
                matrix[-1]=1.;prob=np.linalg.solve(matrix,np.array([0.,0.,0.,1.]))
                if side=='left':j=gl*(fl0*prob[0]-(1-fl0)*prob[1]+flu*prob[2]-(1-flu)*prob[3])
                else:j=gr*((1-fr0)*prob[2]-fr0*prob[0]+(1-fru)*prob[3]-fru*prob[1])
                actual.append(j)
            predicted=current(volts,np.log10(actual[2]),'finite_'+side,p)
            np.testing.assert_allclose(predicted,actual,rtol=2e-5,atol=1e-18)

    def synthetic(self):
        vp=np.linspace(.15,3,40);vn=-np.linspace(.1,.3,12);p=[.1,.65]
        return [Pair(k,vp,-8+.1*k+log10_shape(vp,'fixed_level',p),
                     vn,-8+.1*k+log10_shape(vn,'fixed_level',p)) for k in range(1,5)]

    def test_joint_synthetic_recovery_and_branch_weights(self):
        pairs=self.synthetic();fit=fit_shared_shape(pairs,'fixed_level')
        np.testing.assert_allclose(fit['parameters'],[.1,.65],atol=1e-6)
        repeated=[Pair(p.cycle_id,np.repeat(p.positive_voltage_v,3),np.repeat(p.positive_log10_current_a,3),
                       p.negative_voltage_v,p.negative_log10_current_a) for p in pairs]
        other=fit_shared_shape(repeated,'fixed_level')
        np.testing.assert_allclose(other['parameters'],fit['parameters'],atol=1e-6)
        self.assertLess(fit['profiled_pair_branch_equal_rmse_decade'],1e-7)

    def test_target_reverse_values_do_not_leak_into_predictions(self):
        pairs=self.synthetic();p=pairs[0]
        changed=Pair(p.cycle_id,p.positive_voltage_v,p.positive_log10_current_a,
                     p.negative_voltage_v,p.negative_log10_current_a+2)
        f,a=predict_targets(pairs[1:],[p],'fixed_level');g,b=predict_targets(pairs[1:],[changed],'fixed_level')
        self.assertEqual(f['parameters'],g['parameters'])
        for left,right in zip(a,b):
            self.assertEqual(left['log10_reference_amplitude'],right['log10_reference_amplitude'])
            self.assertNotEqual(left['negative_rmse_decade'],right['negative_rmse_decade'])
        with self.assertRaises(ValueError):predict_targets(pairs,[p],'fixed_level')

    def test_SET_energy_initialization_does_not_read_target_RESET(self):
        p=[-.15,-.05,.22,.13,.8];vp=np.linspace(.15,3,50);vn=-np.linspace(.1,.3,12)
        pair=Pair(1,vp,-7+log10_shape(vp,'coulomb_right',p),vn,-7+log10_shape(vn,'coulomb_right',p))
        changed=Pair(1,vp,pair.positive_log10_current_a,vn,pair.negative_log10_current_a+3)
        source={'name':'coulomb_right','parameters':[-.16,-.05,.22,.13,.8]}
        before,amp1=adjust_energies(pair,source,[0],True)
        after,amp2=adjust_energies(changed,source,[0],True)
        self.assertEqual(before['parameters'],after['parameters']);self.assertEqual(amp1,amp2)


if __name__=='__main__':unittest.main()
