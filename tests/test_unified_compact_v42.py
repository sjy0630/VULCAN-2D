from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from analysis.et_two_channel_model import StandaloneNMOS
from analysis.unified_compact_v42 import NMOS_GATE_OFFSET_V, UnifiedCompactV42

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_JSON = (
    REPO_ROOT
    / "docs"
    / "model-exploration"
    / "iterations"
    / "2026-11-27-01-回程坍塌与全自动整圈交接"
    / "iteration_2026_11_27_01_results.json"
)


class TestUnifiedCompactV42(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        payload = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
        cls.model = UnifiedCompactV42.from_dict(payload["parameters"])
        cls.nmos = StandaloneNMOS.fit_fig2b(REPO_ROOT / "vulcan2d/data/fig2b_transistor_lookup.json")

    def test_invert_read_round_trip(self):
        fc = self.nmos.monotone_table(1.1, scale=1.0, offset_v=NMOS_GATE_OFFSET_V, vmax=6.0)
        g_true = np.array([2.0e-11, 5.0e-9, 8.0e-7, 2.5e-5])
        v_read = 0.25
        # Forward solve at +0.25 V
        vd_tab, id_tab = fc
        levels, iu = np.unique(id_tab, return_index=True)
        lo, hi = np.zeros_like(g_true), np.full_like(g_true, v_read)
        for _ in range(28):
            vh = 0.5 * (lo + hi)
            it = self.model.current_mag(vh, g_true, lam=1.0)
            v1t = np.interp(np.minimum(it, levels[-1]), levels, vd_tab[iu])
            below = vh + self.model.rc * it + v1t < v_read
            lo = np.where(below, vh, lo)
            hi = np.where(below, hi, vh)
        i_meas = self.model.current_mag(0.5 * (lo + hi), g_true, lam=1.0)
        g_rec = self.model.invert_read_to_g0(v_read, i_meas, fc, lam=1.0)
        np.testing.assert_allclose(np.log10(g_rec), np.log10(g_true), atol=5e-3)

    def test_soft_bridge_collapses_while_welded_bridge_persists(self):
        wave = np.r_[np.linspace(0.0, 5.0, 101), np.linspace(4.95, 0.05, 99)]
        idx_025 = 100 + int(np.argmin(np.abs(wave[100:] - 0.25)))
        c_09, _, _, g_end_09, _ = self.model.simulate_set(wave, np.array([1.0e-11]), self.nmos, 0.9, 1.5, step=0.0125)
        c_14, _, _, g_end_14, _ = self.model.simulate_set(wave, np.array([1.0e-11]), self.nmos, 1.4, 1.5, step=0.0125)
        self.assertLess(float(c_09[0, idx_025]), 0.30e-6)
        self.assertGreater(float(c_14[0, idx_025]), 6.0e-6)
        self.assertGreater(float(g_end_14[0] / g_end_09[0]), 25.0)

    def test_set_and_reset_step_convergence(self):
        wave_s = np.r_[np.linspace(0.0, 5.0, 101), np.linspace(4.95, 0.10, 98)]
        c_coarse = self.model.simulate_set(wave_s, np.array([1.2e-11]), self.nmos, 0.9, 1.5, step=0.0125)[0]
        c_fine = self.model.simulate_set(wave_s, np.array([1.2e-11]), self.nmos, 0.9, 1.5, step=0.00625)[0]
        mask = wave_s >= 0.25
        max_diff_set = float(np.max(np.abs(np.log10(c_coarse[:, mask] / c_fine[:, mask]))))
        self.assertLess(max_diff_set, 2.0e-3)

        wave_r = -np.r_[np.linspace(0.02, 1.50, 75), np.linspace(1.48, 0.02, 74)]
        r_coarse = self.model.simulate_reset(wave_r, np.array([5.0e-5]), None, g_floor=1.5e-11, step=0.012)[0]
        r_fine = self.model.simulate_reset(wave_r, np.array([5.0e-5]), None, g_floor=1.5e-11, step=0.006)[0]
        max_diff_rst = float(np.max(np.abs(np.log10(r_coarse / r_fine))))
        self.assertLess(max_diff_rst, 5.0e-3)


if __name__ == "__main__":
    unittest.main()
