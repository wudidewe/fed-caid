"""Exercise donor-only selection and receiver-only safety checks."""

import unittest

import numpy as np

from src.revision_donor_check_guard import EXACT_SEARCH, fit_donor_check_guard, predict_donor_check_guard


def _cal(base, labels, groups, margins):
    """Encode scalar margins as nonnegative scores for target class 2."""
    size = len(base)
    scores = np.column_stack((np.zeros(size), np.ones(size), np.exp(margins), np.ones(size)))
    return {
        "base_predictions": np.array(base, dtype=np.int64),
        "scores": scores,
        "labels": np.array(labels, dtype=np.int64),
        "group_ids": np.array(groups, dtype=np.int64),
    }


class DonorCheckGuardTests(unittest.TestCase):
    def setUp(self):
        self.donor = _cal([1, 1, 1, 1, 0], [2, 2, 2, 3, 2], [1, 1, 2, 3, 4], [0.5, 0.5, 2, -2, 4])
        self.receiver = _cal([1, 1, 3], [1, 1, 3], [10, 11, 12], [-1, -2, -1])

    def _fit(self, receiver):
        return fit_donor_check_guard(
            [self.donor],
            receiver,
            2,
            thresholds=(0.0, 1.0),
            min_donor_target_groups=2,
            min_donor_negative_groups=1,
            min_receiver_groups=2,
            min_correct_class_groups=2,
        )

    def test_selects_using_donor_groups_and_reports_logical_bytes(self):
        receipt = self._fit(self.receiver)
        self.assertEqual(receipt["status"], "enabled")
        self.assertEqual(receipt["selected_threshold"], 0)
        self.assertEqual(receipt["donor_summaries"][0]["missed_alerted_target_groups"], 2)
        self.assertEqual(receipt["candidates"][0]["corrected_target_groups"], 2)
        self.assertEqual(
            receipt["receiver_check"]["source_routes"]["3"]["status"], "insufficient_class_check_support"
        )
        self.assertEqual(receipt["communication"]["logical_payload_bytes"], 56)

    def test_prediction_rewrites_only_already_alerted_supported_classes(self):
        receipt = self._fit(self.receiver)
        evaluation = _cal([0, 1, 3, 1], [2, 2, 2, 1], [20, 21, 22, 23], [3, 3, 3, -1])
        predictions = predict_donor_check_guard(
            evaluation["base_predictions"], evaluation["scores"], 2, receipt
        )
        np.testing.assert_array_equal(predictions, [0, 2, 3, 1])
        np.testing.assert_array_equal(predictions > 0, evaluation["base_predictions"] > 0)

    def test_receiver_check_can_disable_original_cutoff_but_cannot_tighten_it(self):
        changed = _cal([1, 1, 3], [1, 1, 3], [10, 11, 12], [0.5, -2, -1])
        receipt = self._fit(changed)
        self.assertEqual(receipt["selected_threshold"], 0)
        self.assertEqual(receipt["status"], "receiver_check_budget_exceeded")
        self.assertIsNone(receipt["threshold"])
        self.assertEqual(receipt["receiver_check"]["source_routes"]["1"]["spoiled_groups"], 1)
        np.testing.assert_array_equal(
            predict_donor_check_guard(changed["base_predictions"], changed["scores"], 2, receipt),
            changed["base_predictions"],
        )

    def test_rejects_receiver_target_contamination(self):
        contaminated = {**self.receiver, "labels": np.array([2, 1, 3])}
        with self.assertRaisesRegex(ValueError, "negative-only"):
            self._fit(contaminated)

    def test_rejects_overlapping_donor_and_receiver_groups(self):
        duplicate = {**self.receiver, "group_ids": np.array([3, 11, 12])}
        with self.assertRaisesRegex(ValueError, "disjoint feature groups"):
            self._fit(duplicate)

    def test_rejects_overlapping_donor_groups(self):
        with self.assertRaisesRegex(ValueError, "disjoint feature groups"):
            fit_donor_check_guard([self.donor, self.donor], self.receiver, 2)

    def test_absent_support_leaves_no_cutoff_and_skips_receiver_check(self):
        receipt = fit_donor_check_guard([self.donor], self.receiver, 2)
        self.assertEqual(receipt["status"], "insufficient_donor_support")
        self.assertIsNone(receipt["selected_threshold"])
        self.assertIsNone(receipt["receiver_check"])
        self.assertEqual(receipt["communication"]["logical_payload_bytes"], 128)


class ExactDonorFrontierTests(unittest.TestCase):
    """Check group breakpoints, strict ties, and isolated CAL roles."""

    def setUp(self):
        self.donor = _cal([1] * 5, [2, 2, 2, 1, 1], [1, 1, 2, 3, 3], [1.0, 1.6, 1.8, 0.9, 1.2])
        self.receiver = _cal([1, 1], [1, 1], [10, 11], [-1, 0])

    def _fit(self, thresholds=None, receiver=None):
        return fit_donor_check_guard(
            [self.donor],
            self.receiver if receiver is None else receiver,
            2,
            thresholds=thresholds,
            min_donor_target_groups=2,
            min_donor_negative_groups=1,
            min_receiver_groups=2,
            min_correct_class_groups=2,
        )

    def test_exact_recovers_at_a_cutoff_between_fixed_grid_points(self):
        self.assertEqual(self._fit((1.0, 2.0))["status"], "no_feasible_donor_threshold")
        exact = self._fit()
        self.assertEqual(exact["status"], "enabled")
        self.assertAlmostEqual(exact["selected_threshold"], 1.2)
        self.assertEqual(exact["settings"]["threshold_search"], EXACT_SEARCH)
        self.assertIsNone(exact["communication"]["logical_payload_bytes"])
        self.assertEqual(exact["frontier"]["unique_group_margin_breakpoints"], 3)
        self.assertEqual(exact["frontier"]["evaluated_cutoff_regimes"], 4)
        self.assertEqual(exact["frontier"]["feasible_cutoff_regimes"], 2)
        self.assertEqual(exact["frontier"]["min_wrong_rewrite_rate_with_recovery"], 0)
        self.assertEqual(exact["candidates"][0]["corrected_target_groups"], 2)
        self.assertEqual(exact["candidates"][0]["wrong_rewrite_groups"], 0)

    def test_exact_prediction_keeps_equal_cutoff_and_benign_unchanged(self):
        evaluation = _cal([1, 1, 0], [1, 2, 2], [20, 21, 22], [1.2, 1.3, 2.0])
        predictions = predict_donor_check_guard(
            evaluation["base_predictions"], evaluation["scores"], 2, self._fit()
        )
        np.testing.assert_array_equal(predictions, [1, 2, 0])

    def test_unsafe_receiver_cannot_try_second_feasible_candidate(self):
        unsafe = _cal([1, 1], [1, 1], [10, 11], [1.4, 0])
        receipt = self._fit(receiver=unsafe)
        self.assertEqual(receipt["frontier"]["feasible_cutoff_regimes"], 2)
        self.assertAlmostEqual(receipt["selected_threshold"], 1.2)
        self.assertEqual(receipt["receiver_check"]["checked_threshold"], receipt["selected_threshold"])
        self.assertEqual(receipt["status"], "receiver_check_budget_exceeded")
        self.assertIsNone(receipt["threshold"])


if __name__ == "__main__":
    unittest.main()
