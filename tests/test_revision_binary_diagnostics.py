"""Tests for the CAL/VAL binary diagnostic report."""

import json
import unittest

import numpy as np

from src.revision_binary_diagnostics import class_diagnostic_rows, summarize_binary_diagnostics


def case(
    *,
    cal_benign,
    cal_attack,
    val_benign,
    val_attack,
    threshold=0.5,
    fit_benign=None,
    select_benign=None,
    min_benign_rows=10,
):
    cal_benign = np.asarray(cal_benign, dtype=float)
    cal_attack = np.asarray(cal_attack, dtype=float)
    val_benign = np.asarray(val_benign, dtype=float)
    val_attack = np.asarray(val_attack, dtype=float)
    fit = cal_benign if fit_benign is None else np.asarray(fit_benign, dtype=float)
    select = cal_benign if select_benign is None else np.asarray(select_benign, dtype=float)
    return summarize_binary_diagnostics(
        np.concatenate((cal_benign, cal_attack)),
        np.concatenate((np.zeros(len(cal_benign), dtype=int), np.ones(len(cal_attack), dtype=int))),
        threshold,
        threshold_fit_scores=np.concatenate((fit, cal_attack)),
        threshold_fit_labels=np.concatenate(
            (np.zeros(len(fit), dtype=int), np.ones(len(cal_attack), dtype=int))
        ),
        operating_select_scores=np.concatenate((select, cal_attack)),
        operating_select_labels=np.concatenate(
            (np.zeros(len(select), dtype=int), np.ones(len(cal_attack), dtype=int))
        ),
        val_scores=np.concatenate((val_benign, val_attack)),
        val_labels=np.concatenate(
            (np.zeros(len(val_benign), dtype=int), np.ones(len(val_attack), dtype=int))
        ),
        reference_threshold=0.5,
        target=1,
        min_benign_rows=min_benign_rows,
    )


class BinaryDiagnosticTests(unittest.TestCase):
    def test_separable_case_is_consistent(self):
        result = case(
            cal_benign=[0.1] * 100, cal_attack=[0.9] * 20, val_benign=[0.1] * 100, val_attack=[0.9] * 20
        )
        self.assertEqual(result["primary_diagnosis"], "consistent")
        self.assertEqual(result["cal_far"], 0.0)
        self.assertEqual(result["val_far"], 0.0)
        self.assertEqual(result["val_fnr"], 0.0)
        self.assertEqual(result["target_support"], 20)
        self.assertEqual(result["target_recall"], 1.0)
        self.assertGreater(result["auc_val"], 0.99)
        self.assertEqual(result["val_tpr_at_far_1pct"], 1.0)
        self.assertIsNone(result["val_tpr_at_far_0_5pct"])
        self.assertEqual(result["cal_standardized_partial_auc_at_far_1pct"], 1.0)

    def test_benign_shift_is_reported(self):
        result = case(
            cal_benign=[0.1] * 100, cal_attack=[0.9] * 20, val_benign=[0.8] * 100, val_attack=[0.95] * 20
        )
        self.assertTrue(result["benign_score_shift"])
        self.assertEqual(result["primary_diagnosis"], "benign_score_shift")
        self.assertGreater(result["val_far"], 0.9)

    def test_attack_benign_overlap_is_reported(self):
        result = case(
            cal_benign=[0.1] * 50 + [0.2] * 50,
            cal_attack=[0.1] * 10 + [0.2] * 10,
            val_benign=[0.1] * 50 + [0.2] * 50,
            val_attack=[0.1] * 10 + [0.2] * 10,
        )
        self.assertTrue(result["attack_benign_overlap"])
        self.assertEqual(result["primary_diagnosis"], "attack_benign_overlap")
        self.assertEqual(result["val_fnr"], 1.0)

    def test_support_insufficiency_takes_priority(self):
        result = case(
            cal_benign=[0.1] * 100,
            cal_attack=[0.9] * 20,
            val_benign=[0.1] * 100,
            val_attack=[0.9] * 20,
            fit_benign=[0.1] * 5,
            min_benign_rows=100,
        )
        self.assertTrue(result["support_insufficient"])
        self.assertEqual(result["primary_diagnosis"], "threshold_support_insufficient")

    def test_target_class_loss_uses_exact_predictions(self):
        result = summarize_binary_diagnostics(
            np.array([0.1] * 100 + [0.9] * 20),
            np.array([0] * 100 + [1] * 20),
            0.5,
            threshold_fit_scores=np.array([0.1] * 100 + [0.9] * 20),
            threshold_fit_labels=np.array([0] * 100 + [1] * 20),
            operating_select_scores=np.array([0.1] * 100 + [0.9] * 20),
            operating_select_labels=np.array([0] * 100 + [1] * 20),
            val_scores=np.array([0.1] * 100 + [0.9] * 20),
            val_labels=np.array([0] * 100 + [1] * 20),
            val_predictions=np.array([0] * 100 + [1] * 2 + [2] * 18),
            reference_threshold=0.5,
            target=1,
            min_benign_rows=10,
        )
        self.assertTrue(result["target_class_loss"])
        self.assertEqual(result["primary_diagnosis"], "target_class_loss")
        self.assertAlmostEqual(result["target_f1"], 2 * 1.0 * 0.1 / 1.1)
        self.assertEqual(result["target_binary_fnr"], 0.0)
        self.assertEqual(result["target_subtype_miss_rate"], 0.9)

    def test_attack_class_losses_include_support_and_missed(self):
        result = case(
            cal_benign=[0.1] * 100,
            cal_attack=[0.2, 0.9, 0.95],
            val_benign=[0.1] * 100,
            val_attack=[0.2, 0.9, 0.95],
        )
        losses = json.loads(result["attack_class_losses"])
        self.assertEqual(losses["1"]["support"], 3)
        self.assertEqual(losses["1"]["missed"], 1)

    def test_operating_limit_excess_is_reported_even_with_small_shift(self):
        result = case(
            cal_benign=[0.1] * 98 + [0.55] * 2,
            cal_attack=[0.9] * 20,
            val_benign=[0.1] * 98 + [0.55] * 2,
            val_attack=[0.9] * 20,
        )
        self.assertTrue(result["val_far_exceeds_1pct"])
        self.assertEqual(result["primary_diagnosis"], "val_far_limit_exceeded")
        self.assertEqual(result["val_true_positives"], 20)
        self.assertEqual(result["val_true_negatives"], 98)

    def test_row_thresholds_preserve_distinct_binary_and_subtype_misses(self):
        data = np.array([0.3, 0.8, 0.8, 0.9, 0.7])
        labels = np.array([0, 0, 1, 1, 2])
        cutoff = np.array([0.7, 0.6, 0.85, 0.88, 0.65])
        mask = np.array([False, True, False, True, True])

        def run(**changes):
            parameters = {
                "threshold_fit_scores": data[:2],
                "threshold_fit_labels": labels[:2],
                "operating_select_scores": data[2:],
                "operating_select_labels": labels[2:],
                "val_scores": data,
                "val_labels": labels,
                "reference_threshold": 0.5,
                "target": 1,
                "val_predictions": np.array([0, 2, 0, 2, 2]),
                "cal_thresholds": cutoff,
                "threshold_fit_thresholds": cutoff[:2],
                "operating_select_thresholds": cutoff[2:],
                "val_thresholds": cutoff,
                "cal_alerts": mask,
                "val_alerts": mask,
            }
            return summarize_binary_diagnostics(data, labels, 0.5, **{**parameters, **changes})

        result = run()
        self.assertNotIn("target_class", result)
        self.assertEqual(result["target_class_index"], 1)
        self.assertEqual(result["target_binary_fnr"], 0.5)
        self.assertEqual(result["target_subtype_miss_rate"], 1.0)
        self.assertEqual(result["target_precision"], 0.0)
        self.assertEqual(result["target_f1"], 0.0)
        self.assertEqual(result["val_false_positives"], 1)
        self.assertEqual(result["val_false_negatives"], 1)
        self.assertEqual(result["operating_select_fnr"], 1 / 3)
        first, second, unobserved = class_diagnostic_rows(result, ["BENIGN", "One", "Two", "Three"])
        self.assertEqual(first["class"], "One")
        self.assertEqual(first["binary_missed"], 1)
        self.assertEqual(first["subtype_missed_among_alerted"], 1)
        self.assertEqual(second["subtype_false_positives"], 2)
        self.assertEqual(unobserved["val_support"], 0)
        self.assertEqual(unobserved["subtype_true_positives"], 0)
        self.assertEqual(unobserved["subtype_missed_among_alerted"], 0)
        self.assertIsNone(unobserved["binary_fnr"])
        with self.assertRaises(ValueError):
            run(val_alerts=~mask)
        with self.assertRaises(ValueError):
            run(val_thresholds=cutoff[1:])


if __name__ == "__main__":
    unittest.main()
