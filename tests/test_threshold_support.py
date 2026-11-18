"""Small-sample guard for empirical alarm thresholds."""

import unittest

import numpy as np

from src.revision_report import with_class_intervals
from src.utils import empirical_alarm_threshold


class ClassIntervalTests(unittest.TestCase):
    def test_small_support_interval_is_wide(self):
        row = with_class_intervals({"support": 3, "predicted": 4, "recall": 1 / 3, "precision": 0.25})
        self.assertLess(row["recall_ci95_low"], 0.1)
        self.assertGreater(row["recall_ci95_high"], 0.7)
        self.assertLess(row["precision_ci95_low"], 0.05)

    def test_empty_support_has_no_interval(self):
        row = with_class_intervals({"support": 0, "predicted": 0, "recall": None, "precision": None})
        self.assertNotIn("recall_ci95_low", row)
        self.assertNotIn("precision_ci95_low", row)


class ThresholdSupportTests(unittest.TestCase):
    def test_small_benign_holdout_falls_back_to_floor(self):
        probabilities = np.array([[0.6, 0.4, 0], [0.7, 0.3, 0], [0.8, 0.2, 0], [0.2, 0.8, 0]])
        labels = np.array([0, 0, 0, 1])
        threshold, details = empirical_alarm_threshold(
            probabilities, labels, target_far=0.01, floor=0.5, min_benign_rows=1000
        )
        self.assertEqual(threshold, 0.5)
        self.assertTrue(details["insufficient_benign_support"])
        self.assertEqual(details["calibration_rule"], "insufficient_benign_support_floor_fallback")

    def test_sufficient_benign_holdout_keeps_empirical_quantile(self):
        q = np.linspace(0.0, 0.8, 2000)
        probabilities = np.column_stack((1 - q, q, np.zeros(2000)))
        labels = np.zeros(2000, dtype=int)
        threshold, details = empirical_alarm_threshold(
            probabilities, labels, target_far=0.01, floor=0.5, min_benign_rows=1000
        )
        self.assertGreater(threshold, 0.7)
        self.assertFalse(details["insufficient_benign_support"])

    def test_insufficient_support_never_opens_donor_gate(self):
        from argparse import Namespace

        from src.revision_adaptation import AdapterClient
        from src.revision_supplement import REPAIR_RESCUE, REPAIR_UNION, missing_class_repair

        local = np.array([[0.6, 0.4], [0.2, 0.8], [0.9, 0.1]])
        donor = np.array([[0.05, 0.95], [0.01, 0.99], [0.1, 0.9]])
        labels = np.array([0, 1, 0])
        client = AdapterClient(local, labels, local, np.zeros(2))
        args = Namespace(
            target_far=0.01,
            threshold_floor=0.5,
            threshold_min_benign_rows=1000,
            threshold_insufficient_policy="disable",
            repair_union_fallback="base",
        )
        threshold, status = empirical_alarm_threshold(local, labels, 0.01, 0.5, 1000, "disable")
        self.assertEqual(threshold, 1.0)
        self.assertTrue(status["alert_gate_disabled_for_support"])
        base = np.zeros(3, dtype=int)
        predictions, _, receipts = missing_class_repair(
            client, np.arange(3), donor, donor, base, np.array([False, True]), args
        )
        np.testing.assert_array_equal(predictions[REPAIR_UNION], base)
        np.testing.assert_array_equal(predictions[REPAIR_RESCUE], base)
        self.assertEqual(receipts[1]["calibration_far"], 0.0)

    def test_base_union_cannot_spend_full_budget_twice(self):
        from argparse import Namespace

        from src.revision_adaptation import AdapterClient
        from src.revision_supplement import REPAIR_UNION, missing_class_repair

        scores = np.linspace(0.6, 0.99, 12)
        local = np.column_stack((1 - scores, scores))
        donor = local[::-1].copy()
        labels = np.zeros(12, dtype=int)
        client = AdapterClient(local, labels, local, np.zeros(2))
        args = Namespace(
            target_far=0.25,
            threshold_floor=0.5,
            threshold_min_benign_rows=1,
            threshold_insufficient_policy="disable",
            repair_union_fallback="base",
        )
        threshold, _ = empirical_alarm_threshold(local, labels, 0.25, 0.5, 1, "disable")
        base = (scores > threshold).astype(int)
        predictions, _, receipts = missing_class_repair(
            client, np.arange(12), donor, donor, base, np.array([False, True]), args
        )
        self.assertTrue(np.all(predictions[REPAIR_UNION] >= base))
        self.assertLessEqual(receipts[1]["calibration_far"], 0.25)

    def test_default_preserves_legacy_rule(self):
        probabilities = np.array([[0.9, 0.1, 0], [0.2, 0.8, 0]])
        labels = np.array([0, 1])
        threshold, details = empirical_alarm_threshold(probabilities, labels, target_far=0.5, floor=0.5)
        self.assertEqual(threshold, 0.5)
        self.assertFalse(details["insufficient_benign_support"])


if __name__ == "__main__":
    unittest.main()
