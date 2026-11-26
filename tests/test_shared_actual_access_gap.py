"""Scientific contracts of the actual-current access-voltage candidate."""
import math
import unittest
from dataclasses import asdict, replace
import numpy as np
from scipy.optimize import brentq

from vulcan2d.shared_gap import GapLaw, BareGapCircuit, LookupGapCircuit, simulate_gap
from vulcan2d.shared_actual_access_gap import ActualAccessGapLaw
from vulcan2d.transistor import measured_transistor_mapping


def base():
    return GapLaw(ron_ohm=100.,v0_v=.04,i0_a=3e-7,u_max=15.,
                  kset_per_v=.5,kreset_per_v=.3,eth_set_v=.2,eth_reset_v=.2,
                  escale_v=.15,reset_initiation_drive_v=.1)


def model(b=1e4):
    return ActualAccessGapLaw(**asdict(base()),access_voltage_v=b)


def independent_voltage(i,u,law):
    return law.ron_ohm*i+law.v0_v*u*math.asinh(i*math.exp(u)/law.i0_a)+u*law.access_voltage_v*(i/1e-3)**3


def independent_current(v,u,law):
    if v==0.:return 0.
    sign=np.sign(v);magnitude=abs(v)
    root=brentq(lambda z:independent_voltage(math.exp(z),u,law)-magnitude,
                -740.,math.log(magnitude/law.ron_ohm)+1e-10,xtol=1e-12)
    return float(sign*math.exp(root))


class ActualAccessTests(unittest.TestCase):
    def test_b_zero_keeps_old_transport_and_complete_trajectory_bitwise(self):
        old,new=base(),model(0.)
        for u in (0.,.2,3.,10.):
            for i in (-1e-3,-1e-20,0.,1e-20,1e-3):
                self.assertEqual(old.voltage(i,u),new.voltage(i,u))
                self.assertEqual(old.current(i,u),new.current(i,u))
        waveform=[0.,.3,1.,2.,1.,0.,-.3,-1.,-.3,0.]
        circuits=[BareGapCircuit(.001),LookupGapCircuit(measured_transistor_mapping(),1.1)]
        for circuit in circuits:
            a=simulate_gap(waveform,3.,old,circuit,max_step_v=.02)
            b=simulate_gap(waveform,3.,new,circuit,max_step_v=.02)
            for key in ('current_a','state_u','hbn_voltage_v','transistor_voltage_v','supply_voltage_v'):
                np.testing.assert_array_equal(a[key],b[key])

    def test_ohmic_limit_parameter_validation_and_unchanged_dynamics(self):
        for value in (-1.,math.inf,math.nan):
            with self.assertRaises(ValueError):model(value)
        old,new=base(),model()
        for voltage in (-5.,0.,5.):
            self.assertEqual(new.current(voltage,0.),voltage/new.ron_ohm)
        for u in (0.,.2,3.,10.):
            for i in (-1e-4,0.,1e-4):
                self.assertEqual(old.drive(i,u),new.drive(i,u))
                self.assertEqual(old.derivative(u,i),new.derivative(u,i))

    def test_actual_current_coefficient_has_no_exponential_state_factor(self):
        old,new=base(),model()
        i=1e-4
        for u in (.7,1.4):
            extra=new.voltage(i,u)-old.voltage(i,u)
            self.assertAlmostEqual(extra,u*new.access_voltage_v*(i/1e-3)**3,delta=1e-12)
        np.testing.assert_allclose(new.voltage(np.array([-i,0.,i]),1.4),
                                   [independent_voltage(-i,1.4,new),0.,independent_voltage(i,1.4,new)],rtol=1e-14)

    def test_inverse_against_independent_direct_voltage_root_and_tiny_current(self):
        for b in (1e3,1e4,1e5):
            law=model(b)
            for u in (.01,.5,3.,10.):
                for voltage in (-3.,-.01,1e-8,.1,5.):
                    np.testing.assert_allclose(law.current(voltage,u),independent_current(voltage,u,law),rtol=2e-10,atol=1e-30)
                for i in (1e-30,1e-14,1e-6,1e-3):
                    np.testing.assert_allclose(law.current(law.voltage(i,u),u),i,rtol=2e-10,atol=1e-40)

    def test_passivity_monotonicity_and_signed_transistor_load(self):
        law=model()
        current=np.r_[0.,np.geomspace(1e-15,.01,60)]
        for u in (0.,.1,2.,10.):
            voltage=law.voltage(current,u)
            self.assertTrue(np.all(np.diff(voltage)>0.))
            np.testing.assert_array_equal(law.voltage(-current,u),-voltage)
        transistor=measured_transistor_mapping()
        circuit=LookupGapCircuit(transistor,1.1)
        for u in (.05,2.,5.):
            for command in (-1.5,-.1,.1,4.):
                point=circuit.solve(command,law,u)
                magnitude=abs(command);sign=np.sign(command)
                vh=brentq(lambda v:independent_current(v,u,law)-abs(transistor.drain_current(sign*(magnitude-v),1.1)),0.,magnitude,xtol=1e-12)
                self.assertAlmostEqual(point.hbn_voltage_v,sign*vh,delta=3e-10)
                self.assertAlmostEqual(point.hbn_voltage_v+point.transistor_voltage_v,command,delta=1e-12)
                np.testing.assert_allclose(point.current_a,transistor.drain_current(point.transistor_voltage_v,1.1),rtol=1e-8,atol=1e-15)

    def test_locked_normalized_current_can_raise_resistance_under_closure(self):
        law=model(3000.)
        j=1e-3
        u1,u2=3.,2.8
        i1,i2=j*np.exp(-u1),j*np.exp(-u2)
        h=law.v0_v*np.arcsinh(j/law.i0_a)
        for u in np.linspace(u2,u1,11):
            i=j*np.exp(-u)
            rgap=u*h/i
            raccess=u*law.access_voltage_v*i**2/(1e-3)**3
            self.assertGreater(u,.5)
            self.assertGreater(raccess/rgap,(1.+u)/(2.*u-1.))
        self.assertGreater(i2,i1)
        self.assertGreater(law.voltage(i2,u2)/i2,law.voltage(i1,u1)/i1)
        # Under the same locked J, the original pure gap has the opposite trend.
        self.assertLess(base().voltage(i2,u2)/i2,base().voltage(i1,u1)/i1)


if __name__=='__main__':unittest.main()
