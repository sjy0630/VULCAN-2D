from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from analysis.et_two_channel_model import StandaloneNMOS
from analysis.unified_compact_v43 import NMOS_GATE_OFFSET_V, UnifiedCompactV43

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_JSON = (
    REPO_ROOT
    / "docs"
    / "model-exploration"
    / "iterations"
    / "2026-11-27-02-十二参数精简整圈闭环模型"
    / "iteration_2026_11_27_02_results.json"
)


class TestUnifiedCompactV43(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        payload = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
        assert payload["parameter_count"] == 12
        cls.model = UnifiedCompactV43.from_dict(payload["parameters"])
        cls.nmos = StandaloneNMOS.fit_fig2b(REPO_ROOT / "vulcan2d/data/fig2b_transistor_lookup.json")

    def test_strict_monotonicity_in_reset_and_low_bias(self):
        g_grid = np.logspace(-15, -2, 200)
        # Negative RESET polarity is strictly monotonic across all voltages (0..2.7 V)
        for v_test in (0.1, 0.8, 1.5, 2.7):
            cur_neg = self.model.current_mag(np.full_like(g_grid, v_test), g_grid, lam=1.0, polarity=-1)
            self.assertTrue(np.all(np.diff(cur_neg) > 0.0), f"Non-monotonic RESET current at V={v_test}")
        # Positive readback & Bare 1M pre-switch envelope are strictly monotonic
        for v_test, lam_v in ((0.25, 1.0), (0.50, 1.0), (2.7, (2.5 / 1.5) ** self.model.p_vrst)):
            cur_pos = self.model.current_mag(np.full_like(g_grid, v_test), g_grid, lam=lam_v, polarity=1)
            self.assertTrue(np.all(np.diff(cur_pos) > 0.0), f"Non-monotonic SET current at V={v_test}, lam={lam_v}")

    def test_markov_necking_separates_1p1V_and_1p4V(self):
        wave = np.r_[np.linspace(0.0, 5.0, 101), np.linspace(4.95, 0.05, 99)]
        idx_025 = 100 + int(np.argmin(np.abs(wave[100:] - 0.25)))
        c_11, _, _, g0_11, _ = self.model.simulate_set(wave, np.array([1.5e-11]), self.nmos, 1.1, 1.5, step=0.0125)
        c_14, _, _, g0_14, _ = self.model.simulate_set(wave, np.array([3.0e-11]), self.nmos, 1.4, 1.5, step=0.0125)
        self.assertLess(float(c_11[0, idx_025]), 0.50e-6)
        self.assertGreater(float(c_14[0, idx_025]), 7.0e-6)
        self.assertGreater(float(g0_14[0] / g0_11[0]), 100.0)

    def test_invert_and_step_convergence(self):
        fc = self.nmos.monotone_table(1.1, scale=1.0, offset_v=NMOS_GATE_OFFSET_V, vmax=6.0)
        g_true = np.array([5.0e-11, 2.0e-8, 3.0e-6, 4.0e-5])
        vd_tab, id_tab = fc
        levels, iu = np.unique(id_tab, return_index=True)
        lo, hi = np.zeros_like(g_true), np.full_like(g_true, 0.25)
        for _ in range(28):
            vh = 0.5 * (lo + hi)
            it = self.model.current_mag(vh, g_true, lam=1.0, polarity=1)
            v1t = np.interp(np.minimum(it, levels[-1]), levels, vd_tab[iu])
            below = vh + self.model.rc * it + v1t < 0.25
            lo = np.where(below, vh, lo)
            hi = np.where(below, hi, vh)
        i_obs = self.model.current_mag(0.5 * (lo + hi), g_true, lam=1.0, polarity=1)
        g_rec = self.model.invert_read_to_g0(0.25, i_obs, fc, lam=1.0)
        np.testing.assert_allclose(np.log10(g_rec), np.log10(g_true), atol=5e-3)


if __name__ == "__main__":
    unittest.main()
