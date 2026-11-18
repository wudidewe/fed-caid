from unittest import TestCase

import numpy as np

from src.revision_fixed_subtype import eligible_missing_classes, fixed_subtype_predictions, subtype_transitions


class FixedSubtypeTests(TestCase):
    def test_only_preferred_eligible_attacks_can_replace_existing_alerts(self):
        base = np.array([0, 1, 2, 3, 3])
        p = np.array([
            [0.9, 0.01, 0.08, 0.01],
            [0.9, 0.01, 0.08, 0.01],
            [0.9, 0.08, 0.01, 0.01],
            [0.9, 0.01, 0.03, 0.06],
            [0.9, 0.02, 0.04, 0.04],
        ])
        pred = fixed_subtype_predictions(base, p, (2,))
        np.testing.assert_array_equal(pred, [0, 2, 2, 3, 2])
        np.testing.assert_array_equal(pred > 0, base > 0)
        np.testing.assert_array_equal(fixed_subtype_predictions(base, p, ()), base)
        np.testing.assert_array_equal(eligible_missing_classes(np.array([0, 1, 1]), 4), (2, 3))
        self.assertEqual(eligible_missing_classes(np.array([0, 1]), 4, target=1), ())

    def test_report_counts_coverage_and_directed_errors(self):
        names = ["BENIGN", "CLASS1", "CLASS2"]
        y = np.array([2, 2, 1, 0, 2])
        base = np.array([0, 1, 1, 1, 2])
        p = np.tile([0.9, 0.01, 0.09], (len(y), 1))
        candidate = fixed_subtype_predictions(base, p, (2,))
        stats, rows = subtype_transitions(y, base, candidate, names, (2,), subtype_scores=p)
        self.assertEqual(stats["base_alert_rows"], 4)
        self.assertEqual(stats["changed_rows"], 3)
        self.assertEqual(stats["corrected_attack_subtypes"], 1)
        self.assertEqual(stats["spoiled_attack_subtypes"], 1)
        self.assertEqual(stats["relabeled_benign_alerts"], 1)
        self.assertEqual(stats["eligible_recall_ceiling"], 2 / 3)
        self.assertEqual(rows[2]["base_false_positive"], 0)
        self.assertEqual(rows[2]["candidate_false_positive"], 2)
        self.assertEqual(rows[2]["candidate_benign_false_positive"], 1)
        self.assertEqual(rows[2]["candidate_other_attack_false_positive"], 1)
        self.assertEqual(rows[2]["head_attack_top1_support"], 3)
        self.assertEqual(rows[2]["head_attack_top1_alerted"], 2)
        self.assertEqual(rows[2]["head_attack_top1_blocked"], 1)
        self.assertEqual(rows[2]["base_recall_given_alert"], 0.5)
        self.assertEqual(rows[2]["candidate_recall_given_alert"], 1)
        self.assertEqual(sum(r["count"] for r in rows if r["row_type"] == "prediction_change"), 3)

    def test_empty_support_and_validation(self):
        empty = np.array([], dtype=np.int64)
        self.assertEqual(len(fixed_subtype_predictions(empty, np.empty((0, 2)), (1,))), 0)
        stats, rows = subtype_transitions(empty, empty, empty, ["BENIGN", "ATTACK"], (1,))
        self.assertIsNone(stats["eligible_recall_ceiling"])
        self.assertIsNone(rows[1]["recall_ceiling"])
        self.assertIsNone(rows[1]["candidate_recall_given_alert"])
        with self.assertRaisesRegex(ValueError, "alarm mask"):
            subtype_transitions([1], [0], [1], ["BENIGN", "ATTACK"], (1,))
        with self.assertRaisesRegex(ValueError, "ineligible subtype"):
            subtype_transitions([1], [1], [2], ["BENIGN", "ATTACK", "OTHER"], (1,))
        with self.assertRaisesRegex(ValueError, "aligned"):
            fixed_subtype_predictions([0, 1], [[0.9, 0.1]], (1,))
        with self.assertRaisesRegex(ValueError, "non-benign"):
            fixed_subtype_predictions([0], [[0.9, 0.1]], (0,))
