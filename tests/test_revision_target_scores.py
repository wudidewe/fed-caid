"""Check paired target score diagnostics against direct classification rules."""

import unittest

import numpy as np

from src.revision_target_scores import FULL, LOCAL, TargetScoreObserver


class TargetScoreObserverTests(unittest.TestCase):
    def test_shifted_ranks_true_false_transitions_and_gate(self):
        y = np.array([2, 2, 0, 1, 0, 3])
        local = np.array([
            [0.1, 0.4, 0.4, 0.1],  # stable tie favors attack 1
            [0.1, 0.2, 0.5, 0.2],
            [0.7, 0.1, 0.15, 0.05],  # attack top-1 blocked by gate
            [0.1, 0.5, 0.3, 0.1],
            [0.2, 0.2, 0.5, 0.1],
            [0.05, 0.20, 0.55, 0.20],
        ])
        full = np.array([
            [0.1, 0.2, 0.5, 0.2],
            [0.6, 0.1, 0.2, 0.1],
            [0.6, 0.1, 0.2, 0.1],
            [0.1, 0.3, 0.5, 0.1],
            [0.2, 0.5, 0.2, 0.1],
            [0.05, 0.20, 0.55, 0.20],
        ])
        observer = TargetScoreObserver(2, 1, y, max_block_rows=2)
        observer(LOCAL, 0, None)  # ignore all nonreceivers
        observer("base", 1, None)
        observer(LOCAL, 1, local)
        observer(FULL, 1, full)
        report = {r["stratum"]: r for r in observer.summarize([1, 2, 0, 1, 2, 2], [2, 0, 0, 2, 1, 2])}
        positives = report["true_target"]
        self.assertEqual(
            (positives["rows"], positives["local_attack_top1"], positives["full_attack_top1"]), (2, 1, 2)
        )
        self.assertEqual(
            (positives["new_gated_true_positive"], positives["lost_gated_true_positive"]), (1, 1)
        )
        self.assertEqual(positives["new_gated_target_from_other_alert"], 1)
        self.assertEqual(positives["full_attack_top1_blocked_by_gate"], 1)
        self.assertAlmostEqual(positives["delta_attack_rank_mean"], -0.5)
        self.assertAlmostEqual(
            positives["delta_target_logp_mean"], (np.log(0.5 / 0.4) + np.log(0.2 / 0.5)) / 2
        )
        self.assertAlmostEqual(positives["delta_attack_log_margin_mean"], np.log(2) / 2)
        negatives = report["non_target"]
        self.assertEqual(
            (negatives["new_gated_false_positive"], negatives["removed_gated_false_positive"]), (1, 1)
        )
        self.assertEqual(report["benign"]["removed_gated_false_positive"], 1)
        self.assertEqual(report["other_attack"]["new_gated_false_positive"], 1)

    def test_missing_competing_class_and_empty_strata_are_null(self):
        observer = TargetScoreObserver(1, 0, np.array([0, 0]))
        observer(LOCAL, 0, np.array([[1.0, 0.0], [0.6, 0.4]]))
        observer(FULL, 0, np.array([[1.0, 0.0], [0.2, 0.8]]))
        report = {r["stratum"]: r for r in observer.summarize()}
        self.assertIsNone(report["true_target"]["delta_target_logp_mean"])
        self.assertEqual(report["true_target"]["rows"], 0)
        self.assertEqual(report["non_target"]["margin_rows"], 0)
        self.assertIsNone(report["non_target"]["delta_attack_log_margin_mean"])
        self.assertAlmostEqual(report["non_target"]["delta_target_logp_mean"], np.log(2) / 2)
        self.assertFalse(report["non_target"]["gated_decisions_available"])

    def test_checks_alignment_and_ignores_unobserved_modes(self):
        observer = TargetScoreObserver(2, 0, np.array([2]))
        p = np.array([[0.05, 0.15, 0.8]])
        with self.assertRaisesRegex(ValueError, "Both conditional"):
            observer.summarize()
        with self.assertRaisesRegex(ValueError, "class space"):
            observer(LOCAL, 0, p[:0])
        with self.assertRaisesRegex(ValueError, "normalized"):
            observer(LOCAL, 0, p / 2)
        observer(LOCAL, 0, p)
        with self.assertRaisesRegex(ValueError, "already observed"):
            observer(LOCAL, 0, p)
        observer(FULL, 0, p)
        with self.assertRaisesRegex(ValueError, "both gated"):
            observer.summarize([2])
        with self.assertRaisesRegex(ValueError, "differ from"):
            observer.summarize([1], [2])


if __name__ == "__main__":
    unittest.main()
