import unittest
from types import SimpleNamespace

import numpy as np

from vulcan2d import cycle_quality as quality


def sweep(label, outward_g, return_g):
    voltage = np.r_[np.linspace(0., 1., 101), np.linspace(.99, 0., 100)]
    outward = np.arange(len(voltage)) <= 100
    masks = dict(branch_outward=outward)
    masks.update({key: np.zeros_like(outward) for key in (
        "transistor_dominated", "lookup_mismatch", "smu_compliance")})
    return SimpleNamespace(label=label, voltage=voltage,
                           current=voltage*np.where(outward, outward_g, return_g),
                           masks=masks)


class CycleQualityTests(unittest.TestCase):
    def test_single_weak_read_band_does_not_exclude(self):
        key = ("1T1M", 1.3, "SET")
        typical = sweep("Sheet1", 1e-8, 1e-6)
        weak = sweep("Sheet2", 1e-8, 1e-8)
        weak.current[np.abs(weak.voltage) > .53] *= np.where(
            weak.masks["branch_outward"][weak.voltage > .53], 1, 40)
        measured = quality.screen_groups({key: [typical, weak]})[key]
        self.assertFalse(measured[0].excluded)
        self.assertFalse(measured[1].excluded)

    def test_exclusions_preserve_original_test_boundary(self):
        key = ("1T1M", 1.3, "SET")
        rows = [sweep("Sheet1", 1e-8, 1e-6), sweep("Sheet2", 1e-8, 1e-8),
                sweep("Sheet3", 1e-8, 1e-6)]
        flags = quality.screen_groups({key: rows})[key]
        self.assertTrue(flags[1].excluded)
        self.assertEqual([row.label for row in quality.split_with_screen(rows, flags, stop=2)],
                         ["Sheet1"])
        self.assertEqual([row.label for row in quality.split_with_screen(rows, flags, start=2)],
                         ["Sheet3"])

    def test_known_pair_propagates_reset_failure_to_set(self):
        ks, kr = ("1T1M", 1.1, "SET"), ("1T1M", 1.1, "RESET")
        rows = {ks: [sweep("S4", 1e-8, 1e-6)], kr: [sweep("R4", 1e-8, 1e-8)]}
        flags = quality.screen_groups(rows)
        self.assertEqual(flags[ks][0].status, "exclude_corresponding_pair")
        self.assertTrue(flags[kr][0].excluded)


if __name__ == "__main__":
    unittest.main()
