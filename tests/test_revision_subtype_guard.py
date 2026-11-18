import json
from unittest import TestCase

import numpy as np

from src.revision_subtype_guard import (
    apply_cal_check,
    audit_subtype_guard,
    fit_subtype_guard,
    guarded_subtype_predictions,
    target_attack_margin,
    validate_subtype_guard,
)


class SubtypeGuardTests(TestCase):
    def test_group_budget_uses_group_max_and_ignores_target_and_nonalerts(self):
        base = [1, 1, 1, 1, 1, 1, 0]
        labels = [0, 1, 1, 1, 0, 2, 1]
        scores = [
            [0.9, 0.01, 0.10],
            [0.3, 0.02, 0.68],
            [0.1, 0.01, 0.09],
            [0.1, 0.12, 0.19],
            [0.1, 0.17, 0.10],
            [0.1, 0.01, 0.89],
            [0.1, 0.01, 0.89],
        ]
        groups = [16, 16, 22, 23, 24, 25, 26]
        guard = fit_subtype_guard(
            base, scores, labels, (2,), max_rewrite_rate=0.25, min_negative_groups=4, group_ids=groups
        )
        self.assertEqual(guard["fit_negative_groups"], 4)
        self.assertEqual(guard["fit_rewritten_groups"], 1)
        self.assertAlmostEqual(guard["threshold"], np.log(9))
        np.testing.assert_array_equal(
            guarded_subtype_predictions(base, scores, (2,), guard), [2, 2, 1, 1, 1, 2, 0]
        )
        audit = audit_subtype_guard(base, scores, labels, (2,), guard, groups)
        self.assertEqual(audit["negative_rows"], 5)
        self.assertEqual(audit["rewritten_negative_rows"], 2)
        self.assertEqual(audit["negative_group_rewrite_rate"], 0.25)
        self.assertTrue(audit["meets_empirical_budget"])

    def test_negative_threshold_can_rescue_alert_without_top_rank(self):
        calibration = [[0.3, 0.5, 0.20], [0.3, 0.45, 0.25], [0.3, 0.40, 0.30], [0.3, 0.35, 0.35]]
        guard = fit_subtype_guard(
            [1] * 4, calibration, [1] * 4, [2], max_rewrite_rate=0.25, min_negative_groups=4
        )
        self.assertLess(guard["threshold"], 0)
        # First target margin is < 0 but higher than the CAL threshold; the
        # second row sticks on equality and the third never had an alert.
        values = [[0.3, 0.37, 0.33], calibration[2], [0.2, 0.1, 0.9]]
        prediction = guarded_subtype_predictions([1, 1, 0], values, [2], guard)
        np.testing.assert_array_equal(prediction, [2, 1, 0])
        np.testing.assert_array_equal(prediction > 0, [True, True, False])

    def test_multiple_eligible_attacks_share_one_budget_and_deterministic_ties(self):
        calibration = [[0, 0.20, 0.6, 0.2], [0, 0.20, 0.2, 0.6], [0, 0.6, 0.3, 0.1], [0, 0.6, 0.1, 0.3]]
        guard = fit_subtype_guard(
            [1] * 4, calibration, [1] * 4, [3, 2], max_rewrite_rate=0.25, min_negative_groups=4
        )
        # Both equally high calibration groups abstain rather than break a tie
        # in their favor and overspend the configured budget.
        self.assertEqual(guard["fit_rewritten_groups"], 0)
        self.assertEqual(guard["eligible_classes"], [2, 3])
        proposal = [[0.1, 0.2, 0.5, 0.2], [0.1, 0.1, 0.1, 0.7], [0.1, 0.05, 0.85, 0.05]]
        np.testing.assert_array_equal(
            guarded_subtype_predictions([1, 1, 2], proposal, [2, 3], guard), [1, 3, 2]
        )

    def test_unsupported_cases_abstain_and_receipt_is_json_safe(self):
        cases = [
            ([1], [[0.1, 0.1, 0.8]], [1], [2], "insufficient_negative_groups"),
            ([1], [[0.1, 0.1, 0.8]], [1], [], "no_eligible_attacks"),
            ([2], [[0.1, 0.1, 0.8]], [1], [2], "no_rewrite_candidates"),
        ]
        for base, scores, labels, eligible, status in cases:
            with self.subTest(status=status):
                guard = fit_subtype_guard(
                    base, scores, labels, eligible, min_negative_groups=1 if base == [2] else 2
                )
                self.assertEqual(guard["status"], status)
                self.assertIsNone(guard["threshold"])
                guard = json.loads(json.dumps(guard, allow_nan=False))
                np.testing.assert_array_equal(
                    guarded_subtype_predictions(base, scores, eligible, guard), base
                )

    def test_validation_and_single_attack_margin(self):
        values = [[1, 0], [0.8, 0.2]]
        np.testing.assert_allclose(target_attack_margin(values, 1), np.log([1e-12, 0.2]))
        guard = fit_subtype_guard([1, 0], values, [0, 0], [1], min_negative_groups=1)
        with self.assertRaisesRegex(ValueError, "eligible classes"):
            validate_subtype_guard(guard, eligible=[3])
        with self.assertRaisesRegex(ValueError, "status or threshold"):
            validate_subtype_guard({**guard, "threshold": 0})
        with self.assertRaisesRegex(ValueError, "group IDs"):
            fit_subtype_guard([1, 0], values, [0, 0], [1], group_ids=[0])
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            target_attack_margin([[1, -0.3]], 1)

    def test_rounding_does_not_move_the_empirical_budget_past_boundary(self):
        scores = [[0, 0.5, value] for value in np.linspace(0.1, 0.6, 6)]
        rate = float(np.nextafter(5 / 6, -np.inf))  # 6 * rate rounds to 5.0
        guard = fit_subtype_guard([1] * 6, scores, [1] * 6, [2], max_rewrite_rate=rate, min_negative_groups=1)
        self.assertEqual(guard["fit_rewritten_groups"], 4)

    def test_check_budget_can_disable_guard_without_refitting(self):
        guards = [1] * 4
        scores = [[0, 0.5, score] for score in (0.1, 0.2, 0.3, 0.4)]
        fitted = fit_subtype_guard(guards, scores, guards, [2], max_rewrite_rate=0.25, min_negative_groups=4)
        shifted_scores = [[0, 0.5, score] for score in (0.4, 0.5, 0.6, 0.7)]
        effective, audit = apply_cal_check(guards, shifted_scores, guards, [2], fitted, min_check_groups=4)
        self.assertEqual(fitted["status"], "calibrated")
        self.assertEqual(effective["status"], "check_rewrite_budget_exceeded")
        self.assertIsNone(effective["threshold"])
        self.assertEqual(effective["fit_rewritten_groups"], fitted["fit_rewritten_groups"])
        self.assertEqual(audit["original_fit_threshold"], fitted["threshold"])
        self.assertEqual(audit["rewritten_negative_groups"], 4)
        self.assertEqual(audit["check_limit"], 0.25)
        validate_subtype_guard(json.loads(json.dumps(effective)), eligible=[2])
        np.testing.assert_array_equal(
            guarded_subtype_predictions(guards, shifted_scores, [2], effective), guards
        )

    def test_check_requires_group_support_and_respects_tolerance(self):
        base = [1] * 4
        scores = [[0, 0.5, score] for score in (0.1, 0.2, 0.3, 0.4)]
        fitted = fit_subtype_guard(base, scores, base, [2], max_rewrite_rate=0.25, min_negative_groups=4)
        shifted_scores = [[0, 0.5, score] for score in (0.2, 0.3, 0.4, 0.5)]
        effective, audit = apply_cal_check(
            base, shifted_scores, base, [2], fitted, min_check_groups=4, check_tolerance=0.25
        )
        self.assertEqual(effective, fitted)
        self.assertEqual(audit["check_status"], "passed")
        self.assertEqual(audit["negative_group_rewrite_rate"], audit["check_limit"])
        self.assertFalse(audit["meets_empirical_budget"])
        effective, audit = apply_cal_check(
            base, shifted_scores, base, [2], fitted, min_check_groups=4, group_ids=[0, 0, 1, 2]
        )
        self.assertEqual(effective["status"], "insufficient_check_negative_groups")
        self.assertEqual(audit["negative_groups"], 3)
        self.assertEqual(audit["check_support"], 4)
        with self.assertRaisesRegex(ValueError, "rewrite rate"):
            apply_cal_check(
                base, shifted_scores, base, [2], fitted, min_check_groups=4, check_tolerance=float("nan")
            )
