"""Regress the locked CAL point and paired low-FAR acceptance."""

import unittest

import numpy as np

from src.revision_nb15_low_far_evaluation import ARMS, assess_results, evaluate_seed


def _probabilities(scores):
    """Create complete three-class scores with an unsupported rare subtype."""
    scores = np.asarray(scores)
    return np.column_stack((1 - scores, scores, np.zeros(len(scores))))


def _experiment(seed=42, *, short_cal=False, reverse_val=False):
    """Give the clients different CAL cutoffs with ties and pooled VAL counts."""
    benign_count = 20 if short_cal else 100
    labels = (np.r_[np.zeros(benign_count, dtype=int), 1], np.r_[np.zeros(100, dtype=int), 1, 1])
    val_scores = [0.3] * 99 + [0.8, 0.9, 0.9]
    if reverse_val:
        val_scores.reverse()
    probabilities = [
        (_probabilities([0.1] * (benign_count - 2) + [top, top, 0.7]), _probabilities(val_scores))
        for top in (0.8, 0.4)
    ]
    return evaluate_seed(
        seed, ["BENIGN", "Exploits", "Worms"], 0, [labels, labels], dict.fromkeys(ARMS, probabilities)
    )


class LowFarEvaluationTests(unittest.TestCase):
    def test_cal_thresholds_exclude_ties_and_overall_pools_client_counts(self):
        tables = _experiment()
        rows = tables["client_working_points"][:2]
        self.assertEqual([row["threshold"] for row in rows], [0.8, 0.4])
        self.assertEqual([row["val_false_positives"] for row in rows], [0, 1])
        receiver, overall = tables["working_points"][:2]
        self.assertEqual(receiver["val_far"], 0)
        self.assertEqual(overall["val_far"], 1 / 200)
        self.assertEqual(overall["val_true_positives"], 4)
        rare = [row for row in tables["per_class"] if row["class_name"] == "Worms"]
        self.assertEqual([row["support"] for row in rare], [0] * 6)
        self.assertTrue(all(row["recall"] is None for row in rare))

    def test_val_scores_do_not_influence_cal_thresholds(self):
        original = _experiment()
        changed = _experiment(reverse_val=True)
        left, right = original["client_working_points"], changed["client_working_points"]
        self.assertEqual([row["threshold"] for row in left], [row["threshold"] for row in right])
        self.assertNotEqual(left[0]["val_fnr"], right[0]["val_fnr"])

    def test_insufficient_benign_cal_disables_gate_and_smoke_does_not_promote(self):
        tables = _experiment(short_cal=True)
        first = tables["client_working_points"][0]
        self.assertEqual(first["threshold"], 1.0)
        self.assertTrue(first["alert_gate_disabled_for_support"])
        status = assess_results(tables["working_points"], [42])
        self.assertFalse(status["selection_on_val"])
        self.assertIsNone(status["candidates"]["frozen_head"]["development_guard_pass"])

    def test_paired_guards_reject_absolute_far_regression_even_if_other_metrics_match(self):
        seeds = range(42, 47)
        rows = [row for seed in seeds for row in _experiment(seed)["working_points"]]
        for row in rows:
            if row["arm"] == "frozen_head" and row["seed"] == 46:
                row["val_far"] = 0.011
        status = assess_results(rows, seeds)
        marks = status["candidates"]
        self.assertFalse(marks["frozen_head"]["development_guard_pass"])
        self.assertTrue(marks["full_finetune"]["development_guard_pass"])
        self.assertEqual(marks["frozen_head"]["scopes"]["receiver"]["seeds_above_far_cap"], 1)
        self.assertFalse(status["automatic_promotion"])


if __name__ == "__main__":
    unittest.main()
