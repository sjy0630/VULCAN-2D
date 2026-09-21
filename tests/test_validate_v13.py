import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from vulcan2d import validate_v13 as audit


def _raw_pair(index):
    return {
        "cycle": index, "set_sheet": f"S{index}", "reset_sheet": f"R{index}",
        "set_voltage": np.array([0.0, 2.5, 5.0, 2.5, 0.0]),
        "reset_voltage": np.array([0.0, -0.85, -1.7, -0.85, 0.0]),
        "set_current": np.array([0.0, 1e-6, 2e-6, 1e-6, 0.0]),
        "reset_current": np.array([0.0, -1e-6, -2e-6, -1e-6, 0.0]),
    }


class ValidationAuditV13Tests(unittest.TestCase):
    def setUp(self):
        self.raw = [_raw_pair(index) for index in range(1, 54)]
        self.named_set = [(f"S{index}", object()) for index in range(1, 54)]
        self.accepted = [row for row in self.raw if row["cycle"] not in (2, 3)]
        self.reset_by_id = {row["cycle"]: object() for row in self.accepted}
        self.eligible = set(self.reset_by_id)

    def test_target_exclusion_for_accepted_and_rejected_pairs(self):
        for target, expected_resets in ((1, 50), (2, 51), (3, 51), (53, 50)):
            selected = audit._select_training(
                self.named_set, self.accepted, self.reset_by_id, target_cycle=target)
            sets, records, resets, provenance = selected
            self.assertEqual(len(sets), 52)
            self.assertEqual(len(records), expected_resets)
            self.assertEqual(len(resets), expected_resets)
            self.assertNotIn(target, provenance["training_SET_cycle_ids"])
            self.assertNotIn(target, provenance["training_RESET_cycle_ids"])
            for row, reset in zip(records, resets):
                self.assertIs(reset, self.reset_by_id[row["cycle"]])

    def test_chronological_membership_preserves_first35_split(self):
        *_, provenance = audit._select_training(
            self.named_set, self.accepted, self.reset_by_id, last_training_cycle=35)
        self.assertEqual(provenance["training_SET_cycle_ids"], list(range(1, 36)))
        self.assertEqual(provenance["training_paired_RESET_cycles"], 33)
        self.assertTrue(all(index <= 35 for index in provenance["training_RESET_cycle_ids"]))

    def test_all53_groups_keep_rejected_pairs_scoreable(self):
        groups = audit._group_ids(self.raw, self.eligible)
        self.assertEqual(groups["all_pairs"], list(range(1, 54)))
        self.assertEqual(len(groups["legacy_accepted"]), 51)
        self.assertEqual(groups["legacy_rejected"], [2, 3])
        self.assertEqual(set(groups["all_pairs"]),
                         set(groups["legacy_accepted"]) | set(groups["legacy_rejected"]))

    def test_invalid_prediction_is_counted_and_score_is_not_masked(self):
        metrics = audit._point_metrics(
            [0.1, 0.2, 0.3, 0.4], [1e-6] * 4,
            [1e-6, np.nan, 0.0, 1e-6], 1e-11)
        self.assertEqual(metrics["status"], "invalid_prediction")
        self.assertEqual(metrics["n_eligible_points"], 4)
        self.assertEqual(metrics["n_scored_points"], 2)
        self.assertEqual(metrics["n_invalid_predictions_at_eligible_points"], 2)
        self.assertIsNone(metrics["rmse_decade"])

    def test_zero_bias_offsets_are_reported_outside_log_score(self):
        voltage = [0.0, 5e-13, 0.2, 0.0]
        observed = [2e-8, -4e-8, 1e-6, 6e-8]
        for predicted in ([0.0, 0.0, 1e-6, 0.0],
                          [1e-3, 2e-3, 1e-6, 3e-3]):
            metrics = audit._point_metrics(voltage, observed, predicted, 1e-11)
            self.assertEqual(metrics["status"], "ok")
            self.assertEqual(metrics["n_zero_bias_points"], 3)
            self.assertEqual(metrics["n_eligible_points"], 1)
            self.assertEqual(metrics["n_scored_points"], 1)
            self.assertEqual(metrics["rmse_decade"], 0.0)
            self.assertEqual(metrics["zero_bias_observed_abs_current_median_A"], 4e-8)
            self.assertEqual(metrics["zero_bias_observed_abs_current_max_A"], 6e-8)

    def test_zero_prediction_at_nonzero_bias_remains_invalid(self):
        metrics = audit._point_metrics(
            [0.0, 0.2], [2e-8, 1e-6], [0.0, 0.0], 1e-11)
        self.assertEqual(metrics["status"], "invalid_prediction")
        self.assertEqual(metrics["n_invalid_predictions_at_eligible_points"], 1)
        self.assertIsNone(metrics["rmse_decade"])

    def test_nonfinite_prediction_at_zero_bias_remains_invalid(self):
        metrics = audit._point_metrics(
            [0.0, 0.2], [2e-8, 1e-6], [np.nan, 1e-6], 1e-11)
        self.assertEqual(metrics["status"], "invalid_prediction")
        self.assertEqual(metrics["n_nonfinite_predictions_all_points"], 1)
        self.assertIsNone(metrics["rmse_decade"])

    def test_chronological_calibration_failure_retains_all_test_records(self):
        with patch.object(audit.V12, "_fit_bundle", side_effect=ValueError("synthetic")):
            records, predictions, reference, provenance = audit._evaluate(
                self.raw, self.named_set, self.accepted, self.reset_by_id,
                None, None, {2: "rejected", 3: "rejected"}, last_training_cycle=35)
        self.assertEqual(len(records), 36)
        self.assertEqual({row["cycle_id"] for row in records}, set(range(36, 54)))
        self.assertTrue(all(row["status"] == "calibration_failed" for row in records))
        self.assertTrue(all(row["target_excluded"] for row in records))
        self.assertEqual(predictions, {})
        self.assertIsNone(reference)
        diagnostics = audit._retrospective_diagnostics(
            self.raw, reference, None, None,
            reference_failure_reason=provenance["reference_calibration_reason"])
        self.assertEqual(len(diagnostics), 53)
        self.assertTrue(all(row["status"] == "reference_unavailable" for row in diagnostics))
        self.assertTrue(all("synthetic" in row["reason"] for row in diagnostics))

    def test_incomplete_group_does_not_publish_successful_subset_score(self):
        records = [{"split": "leave_one_cycle_out", "cycle_id": 2,
                    "reset_transition_mode": mode, "status": "prediction_failed",
                    "reason": "synthetic failure"} for mode in audit.MODES]
        result = audit._summarize_group(
            [2], "leave_one_cycle_out", records, {}, {2: self.raw[1]})
        for mode in audit.MODES:
            self.assertEqual(result["modes"][mode]["status"], "incomplete")
            self.assertIsNone(result["modes"][mode]["scores"])
            self.assertEqual(result["modes"][mode]["failed_pairs"][0]["cycle_id"], 2)

    def test_target_currents_are_never_read_by_prediction_adapter(self):
        class VoltageOnlyRow(dict):
            def __getitem__(self, key):
                if key not in ("set_voltage", "reset_voltage"):
                    raise AssertionError(f"Prediction read target label: {key}")
                return super().__getitem__(key)

        row = VoltageOnlyRow(self.raw[1])
        bundle = {key: object() for key in (
            "set_transport", "set_transition", "link", "endpoint",
            "reset_transport", "reset_transition")}
        marker = object()
        with patch.object(audit, "_simulate", return_value=marker) as simulate:
            self.assertIs(audit._run(row, bundle, None, None, "same_protocol"), marker)
        args, kwargs = simulate.call_args
        self.assertIs(args[0], row["set_voltage"])
        self.assertIs(args[1], row["reset_voltage"])
        self.assertNotIn("set_current", kwargs)
        self.assertNotIn("reset_current", kwargs)
        self.assertIs(kwargs["reset_transition_law"], bundle["reset_transition"])

    def test_missing_target_current_does_not_change_simulation_voltage_grid(self):
        set_frame = pd.DataFrame({"voltage": [0.0, 5.0, 0.0],
                                  "current": [0.0, np.nan, 0.0]})
        reset_frame = pd.DataFrame({"voltage": [0.0, -1.7, 0.0],
                                    "current": [0.0, 1e-6, 0.0]})
        with patch.object(audit.pd, "read_excel", side_effect=[
                {"S2": set_frame}, {"R2": reset_frame}]):
            raw = audit._load_raw_pairs()
        self.assertEqual(len(raw), 1)
        np.testing.assert_array_equal(raw[0]["set_voltage"], [0.0, 5.0, 0.0])
        self.assertTrue(np.isnan(raw[0]["set_current"][1]))

    def test_rejected_pair_scores_without_identified_envelopes(self):
        row = self.raw[1]
        _, _, _, provenance = audit._select_training(
            self.named_set, self.accepted, self.reset_by_id, target_cycle=2)
        prediction = {
            phase: {"Vs": row[f"{phase}_voltage"], "Is": row[f"{phase}_current"]}
            for phase in ("set", "reset")
        }
        with patch.object(audit, "identify_pf_reset_state",
                          side_effect=AssertionError("Scoring must not identify state")):
            record = audit._prediction_record(
                row, prediction, "leave_one_cycle_out", "same_protocol",
                provenance, self.eligible, {2: "legacy rejection"})
        self.assertEqual(record["status"], "ok")
        self.assertFalse(record["legacy_training_eligible"])
        self.assertTrue(record["target_excluded"])
        self.assertEqual(record["metrics"]["RESET"]["all"]["rmse_decade"], 0.0)

    def test_nonidentifiable_diagnostic_retains_cycle_and_reason(self):
        with patch.object(audit, "reconstruct_reset", side_effect=ValueError("synthetic")):
            records = audit._retrospective_diagnostics([self.raw[1]], {}, None, None)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["cycle_id"], 2)
        self.assertEqual(records[0]["status"], "not_identifiable")
        self.assertIn("synthetic", records[0]["reason"])
        self.assertIsNone(records[0]["identified_e50_mvc"])
        self.assertFalse(records[0]["used_for_prediction"])


if __name__ == "__main__":
    unittest.main()
