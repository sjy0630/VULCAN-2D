"""Invariants for the training-only RESET terminal refinement."""
import inspect
import json
from pathlib import Path
import unittest

import numpy as np

from vulcan2d import reset_terminal_refinement as refinement


ROOT = Path(__file__).resolve().parents[1]


class ResetTerminalRefinementTests(unittest.TestCase):
    def test_common_grid_is_required(self):
        rows = [
            {"reset_voltage": np.array([0.0, -0.1, -0.2])},
            {"reset_voltage": np.array([0.0, -0.1, -0.2])},
        ]
        np.testing.assert_array_equal(
            refinement._same_reset_grid(rows), rows[0]["reset_voltage"])
        rows[1]["reset_voltage"] = np.array([0.0, -0.11, -0.2])
        with self.assertRaisesRegex(ValueError, "common voltage grid"):
            refinement._same_reset_grid(rows)

    def test_prediction_api_cannot_receive_target_current(self):
        parameters = inspect.signature(
            refinement.simulate_with_terminal_refinement).parameters
        self.assertNotIn("set_current", parameters)
        self.assertNotIn("reset_current", parameters)

    def test_saved_strict_validation_excludes_every_target(self):
        path = ROOT / "analysis" / "reset_platform_terminal_refinement.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        records = report["cycle_predictions"]
        loo = [record for record in records
               if record["split"] == "leave_one_cycle_out"]
        chronological = [record for record in records
                         if record["split"] ==
                         "chronological_first35_to_later"]
        self.assertEqual(len(loo), 50)
        self.assertEqual(len(chronological), 18)
        for record in loo:
            target = record["cycle_id"]
            training = record["terminal_refinement_training_cycle_ids"]
            self.assertEqual(len(training), 49)
            self.assertNotIn(target, training)
            self.assertFalse(
                record["terminal_refinement_target_current_used"])
            self.assertTrue(record["target_excluded"])
            self.assertEqual(record["status"], "ok")
        for record in chronological:
            training = record["terminal_refinement_training_cycle_ids"]
            self.assertEqual(training, list(range(4, 36)))
            self.assertNotIn(record["cycle_id"], training)
            self.assertFalse(
                record["terminal_refinement_target_current_used"])
            self.assertTrue(record["target_excluded"])
            self.assertEqual(record["status"], "ok")

        baseline = report["baseline_postconditioning_region_scores"][
            "outward_plateau_0p70_to_1p40V"]["rmse_decade"]
        refined = report["refined_region_scores"][
            "outward_plateau_0p70_to_1p40V"]["rmse_decade"]
        self.assertLess(refined, baseline)


if __name__ == "__main__":
    unittest.main()
