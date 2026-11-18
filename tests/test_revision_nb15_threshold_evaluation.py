"""Regression checks for matched NB15 threshold fitting and locked evaluation."""

from collections import defaultdict
import json
import unittest

import numpy as np

from src.revision_nb15_threshold_evaluation import (
    CONSERVATIVE,
    MATCHED,
    REFERENCE,
    assess_results,
    evaluate_locked_seed,
    freeze_thresholds,
)


NAMES = ("BENIGN", "Exploits", "Analysis")


def _probabilities(scores):
    """Keep one common multiclass subtype rule across all tests."""
    return np.column_stack((1 - scores, 0.75 * scores, 0.25 * scores))


def _populations(check_high=0):
    """Make an adversarial CAL-check tail and independently fixed VAL predictions."""
    fit = np.linspace(0.001, 0.9, 1_200)
    check = np.full(1_200, 0.15)
    check[:check_high] = 0.97
    cal_y = np.array([0] * (len(fit) + len(check)) + [1, 2])
    cal_p = _probabilities(np.concatenate((fit, check, [0.98, 0.99])))
    subsets = {
        "threshold_fit": np.arange(len(fit)),
        "operating_select": np.arange(len(fit), len(cal_y)),
    }
    val_y = np.array([0] * 400 + [1] * 10 + [2] * 10)
    val_p = _probabilities(np.where(val_y, 0.99, 0.10))
    return cal_y, cal_p, subsets, val_y, val_p


class ThresholdEvaluationTests(unittest.TestCase):
    def test_lock_is_serializable_disjoint_and_independent_of_val(self):
        cal_y, cal_p, subsets, val_y, val_p = _populations(check_high=30)
        locked = freeze_thresholds([cal_y], [cal_p], [subsets], len(NAMES))
        self.assertEqual(locked, json.loads(json.dumps(locked, allow_nan=False)))
        original, matched, conservative = (locked[arm][0] for arm in (REFERENCE, MATCHED, CONSERVATIVE))
        self.assertFalse(original["cal_check_independent"])
        self.assertIsNone(original["cal_check_far_pass"])
        self.assertLessEqual(matched["threshold"], conservative["threshold"])
        self.assertEqual(conservative["cal_fit_benign_rows"], 1_200)
        self.assertEqual(conservative["cal_check_benign_rows"], 1_200)
        self.assertFalse(conservative["cal_check_far_pass"])
        changed = _probabilities(np.full(len(val_y), 0.02))
        before = evaluate_locked_seed(42, NAMES, 0, [val_y], [val_p], locked)
        after = evaluate_locked_seed(42, NAMES, 0, [val_y], [changed], locked)
        self.assertEqual(
            [row["threshold"] for row in before["client_working_points"]],
            [row["threshold"] for row in after["client_working_points"]],
        )
        self.assertNotEqual(before["working_points"][0]["val_fnr"], after["working_points"][0]["val_fnr"])

    def test_overlap_is_rejected_before_threshold_fit(self):
        cal_y, cal_p, subsets, _, _ = _populations()
        subsets["operating_select"][0] = subsets["threshold_fit"][0]
        with self.assertRaisesRegex(ValueError, "without overlap"):
            freeze_thresholds([cal_y], [cal_p], [subsets], len(NAMES))

    def test_cal_check_failure_does_not_replace_locked_val_predictions(self):
        cal_y, cal_p, subsets, val_y, val_p = _populations(check_high=30)
        locked = freeze_thresholds([cal_y], [cal_p], [subsets], len(NAMES))
        seeds, tables = tuple(range(42, 47)), defaultdict(list)
        for seed in seeds:
            outcome = evaluate_locked_seed(seed, NAMES, 0, [val_y], [val_p], locked)
            for name, rows in outcome.items():
                tables[name].extend(rows)
        analysis = assess_results(tables, seeds)
        candidate = analysis["candidates"][CONSERVATIVE]
        self.assertTrue(candidate["scopes"]["receiver"]["far_pass"])
        self.assertFalse(candidate["cal_check"]["pass"])
        self.assertFalse(analysis["development_guard_pass"])
        self.assertFalse(analysis["automatic_promotion"])
        key = f"{CONSERVATIVE}_minus_{REFERENCE}"
        self.assertEqual(analysis["target_class_intervals"][key]["receiver"]["Analysis"]["recall"]["mean"], 0)

    def test_incomplete_seeds_have_no_scientific_decision(self):
        cal_y, cal_p, subsets, val_y, val_p = _populations()
        locked = freeze_thresholds([cal_y], [cal_p], [subsets], len(NAMES))
        tables = evaluate_locked_seed(42, NAMES, 0, [val_y], [val_p], locked)
        analysis = assess_results(tables, (42,))
        self.assertIsNone(analysis["development_guard_pass"])
        self.assertTrue(analysis["candidates"][CONSERVATIVE]["cal_check"]["pass"])
