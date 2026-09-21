"""Tests the data separation that makes the physical comparison meaningful."""
import unittest
import numpy as np

from analysis.compare_transport_candidates import evaluate_loco
from vulcan2d.transport_candidates import Curve, log10_shape


class TestTransportHoldout(unittest.TestCase):
    def test_target_high_current_cannot_change_its_prediction_parameters(self):
        voltage = np.linspace(.5,1.2,36)
        curves = [Curve(i,voltage,-9+i*.1+log10_shape(voltage,"schottky_free",9.0))
                  for i in range(1,5)]
        original = evaluate_loco(curves,"schottky_free",.92)
        changed = list(curves)
        y = changed[0].log10_current_a.copy()
        y[voltage>.92] += 2.0
        changed[0] = Curve(1,voltage,y)
        altered = evaluate_loco(changed,"schottky_free",.92)
        before = {r["protocol"]:r for r in original if r["cycle_id"]==1}
        after = {r["protocol"]:r for r in altered if r["cycle_id"]==1}
        for protocol,row in before.items():
            self.assertEqual(row["parameter"],after[protocol]["parameter"])
            self.assertEqual(row["log10_reference_amplitude"],after[protocol]["log10_reference_amplitude"])
            self.assertGreater(after[protocol]["high_rmse_decade"],row["high_rmse_decade"]+1)

    def test_target_low_read_changes_only_conditioned_amplitude(self):
        voltage=np.linspace(.5,1.2,36)
        curves=[Curve(i,voltage,-9+log10_shape(voltage,"pf_free",6.0)) for i in range(1,5)]
        before=[r for r in evaluate_loco(curves,"pf_free",.92) if r["cycle_id"]==1]
        y=curves[0].log10_current_a.copy();y[voltage<=.92]+=1
        curves[0]=Curve(1,voltage,y)
        after=[r for r in evaluate_loco(curves,"pf_free",.92) if r["cycle_id"]==1]
        for a,b in zip(before,after):
            self.assertEqual(a["parameter"],b["parameter"])
            delta=b["log10_reference_amplitude"]-a["log10_reference_amplitude"]
            self.assertAlmostEqual(delta,0 if a["protocol"]=="loco_autonomous" else 1)


if __name__=="__main__":unittest.main()
