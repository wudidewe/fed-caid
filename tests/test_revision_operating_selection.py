"""Pure-CAL operating selection and group-isolated threshold split tests."""

import unittest

import numpy as np

from src.revision_operating_selection import SCHEMA, select_operating_point, split_operating_subsets


def client(*, fit_benign, fit_attack, select_benign, select_attack, receiver=False):
    return {
        "threshold_fit_scores": np.concatenate((fit_benign, fit_attack)),
        "threshold_fit_labels": np.concatenate(
            (np.zeros(len(fit_benign), dtype=int), np.ones(len(fit_attack), dtype=int))
        ),
        "operating_select_scores": np.concatenate((select_benign, select_attack)),
        "operating_select_labels": np.concatenate(
            (np.zeros(len(select_benign), dtype=int), np.ones(len(select_attack), dtype=int))
        ),
        "receiver": receiver,
    }


def scenario():
    fit_benign = np.concatenate((np.full(1000, 0.1), np.full(10, 0.8), np.full(10, 0.9)))
    fit_attack = np.full(60, 0.99)
    select_benign = np.concatenate((np.full(1000, 0.1), np.full(20, 0.85)))
    select_attack = np.full(60, 0.99)
    return [
        client(
            fit_benign=fit_benign,
            fit_attack=fit_attack,
            select_benign=select_benign,
            select_attack=select_attack,
            receiver=True,
        ),
        client(
            fit_benign=np.full(1000, 0.1),
            fit_attack=np.full(60, 0.99),
            select_benign=np.full(1000, 0.1),
            select_attack=np.full(60, 0.99),
        ),
        client(
            fit_benign=np.full(1000, 0.1),
            fit_attack=np.full(60, 0.99),
            select_benign=np.full(1000, 0.1),
            select_attack=np.full(60, 0.99),
        ),
    ]


class OperatingSelectionTests(unittest.TestCase):
    def limits(self, **overrides):
        values = dict(
            candidates=(0.01, 0.005),
            max_far=0.01,
            reference_far=0.01,
            max_fnr_increase=0.01,
            min_benign_rows=100,
            min_select_benign_rows=100,
            min_select_attack_rows=20,
            min_receiver_attack_rows=20,
            min_attack_coverage=0.5,
            floor=0.5,
        )
        values.update(overrides)
        return values

    def test_selects_loosest_candidate_that_satisfies_receiver_far(self):
        result = select_operating_point(scenario(), **self.limits())
        self.assertEqual(result["status"], "selected")
        self.assertEqual(result["selected_cal_far"], 0.005)
        self.assertEqual(result["schema"], SCHEMA)
        self.assertEqual(result["candidates"][0]["candidate_cal_far"], 0.01)
        self.assertFalse(result["candidates"][0]["feasible"])
        self.assertEqual(result["candidates"][0]["reason"], "receiver_far")
        self.assertTrue(result["candidates"][1]["feasible"])
        self.assertEqual(len(result["thresholds"]), 3)
        self.assertEqual(len(result["reference_thresholds"]), 3)
        self.assertEqual(result["thresholds"][0], 0.9)

    def test_no_feasible_point_reports_reason_and_candidates(self):
        bad = scenario()
        bad[0]["operating_select_scores"] = np.concatenate((np.full(1020, 0.95), np.full(60, 0.99)))
        result = select_operating_point(bad, **self.limits())
        self.assertEqual(result["status"], "no_feasible_operating_point")
        self.assertEqual(result["reason"], "no_candidate_satisfied_constraints")
        self.assertIn("overall_far", result["failed_constraints"])
        self.assertIsNone(result["selected_cal_far"])
        self.assertEqual(result["thresholds"], [])

    def test_insufficient_reference_support_is_explicit(self):
        small = scenario()
        result = select_operating_point(small, **self.limits(min_benign_rows=5000))
        self.assertEqual(result["status"], "no_feasible_operating_point")
        self.assertEqual(result["reason"], "reference_1pct_support_insufficient")
        self.assertFalse(result["support"]["reference_valid"])

    def test_pooled_benign_fallback_covers_a_small_client(self):
        data = scenario()
        data[2]["threshold_fit_scores"] = np.concatenate((np.full(45, 0.1), np.full(60, 0.99)))
        data[2]["threshold_fit_labels"] = np.concatenate((np.zeros(45, dtype=int), np.ones(60, dtype=int)))
        data[2]["operating_select_scores"] = np.concatenate((np.full(45, 0.1), np.full(60, 0.99)))
        data[2]["operating_select_labels"] = np.concatenate((np.zeros(45, dtype=int), np.ones(60, dtype=int)))
        result = select_operating_point(data, **self.limits(min_select_benign_rows=30, pooled_fallback=True))
        self.assertEqual(result["status"], "selected")
        self.assertEqual(result["support"]["threshold_sources"][2], "pooled_cal_threshold_fit")
        self.assertGreaterEqual(result["support"]["pooled_benign_rows"], 100)

    def test_receiver_fnr_uses_only_cal_attack_labels(self):
        data = scenario()
        data[0]["threshold_fit_labels"][-60:] = 0
        data[0]["operating_select_labels"][-60:] = 0
        result = select_operating_point(data, **self.limits())
        self.assertEqual(result["status"], "no_feasible_operating_point")
        self.assertEqual(result["support"]["receiver_operating_select_attack_rows"], 0)
        self.assertEqual(result["candidates"][0]["reason"], "receiver_attack_support")

    def test_frozen_thresholds_check_fnr_against_parent_not_refit(self):
        data = scenario()
        data[0]["operating_select_scores"][-60:] = 0.84
        refitted = select_operating_point(data, **self.limits())
        frozen = select_operating_point(data, reference_thresholds=(0.85, 0.5, 0.5), **self.limits())
        self.assertEqual(refitted["status"], "no_feasible_operating_point")
        self.assertEqual(refitted["candidates"][1]["receiver"]["fnr_delta"], 1.0)
        self.assertEqual(frozen["status"], "selected")
        self.assertEqual(frozen["selected_cal_far"], 0.005)
        self.assertEqual(frozen["reference_thresholds"], [0.85, 0.5, 0.5])
        self.assertEqual(frozen["candidates"][1]["receiver"]["fnr_delta"], 0.0)
        self.assertEqual(frozen["support"]["reference_source"], "external_frozen_reference")
        self.assertIsNone(frozen["support"]["reference_support"])

    def test_frozen_reference_preserves_candidate_support_requirements(self):
        result = select_operating_point(
            scenario(), reference_thresholds=(0.85, 0.5, 1), **self.limits(min_benign_rows=5000)
        )
        self.assertEqual(result["status"], "no_feasible_operating_point")
        self.assertEqual(result["support"]["reference_disabled_clients"], [2])
        self.assertIn("threshold_support", result["failed_constraints"])
        self.assertEqual(result["support"]["threshold_fit_benign_rows"], [1020, 1000, 1000])
        self.assertEqual(result["support"]["operating_select_benign_rows"], [1020, 1000, 1000])

    def test_full_candidate_frontier_keeps_first_feasible_rate(self):
        result = select_operating_point(scenario(), **self.limits(candidates=(0.0025, 0.01, 0.005)))
        self.assertEqual(result["selected_cal_far"], 0.005)
        self.assertEqual([r["candidate_cal_far"] for r in result["candidates"]], [0.01, 0.005, 0.0025])
        self.assertTrue(result["candidates"][-1]["feasible"])

    def test_frozen_reference_thresholds_require_one_finite_value_per_client(self):
        for references in (
            [0.5],
            [0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, float("nan")],
            [0.5, 0.5, 1.01],
            [0.5, 0.5, -0.01],
            [".5", ".5", ".5"],
            [[0.5], [0.5], [0.5]],
        ):
            with self.subTest(references=references), self.assertRaisesRegex(ValueError, "Frozen reference"):
                select_operating_point(scenario(), reference_thresholds=references, **self.limits())


class OperatingSplitTests(unittest.TestCase):
    def test_holdout_groups_split_without_overlap(self):
        rows, clients = 60, 3
        x = np.arange(rows, dtype=np.float32).reshape(rows, 1)
        y = np.tile(np.array([0, 1] * 10, dtype=np.int64), clients)
        roster = [{"cal": np.arange(i * 20, (i + 1) * 20)} for i in range(clients)]
        splits = [(np.arange(0, 10), np.arange(10, 20)) for _ in range(clients)]
        operating, audit = split_operating_subsets(x, y, roster, splits, 0.5, 42, min_select_benign=1)
        for (parent_fit, parent_hold), row in zip(splits, operating, strict=True):
            self.assertTrue(set(row["threshold_fit"]).isdisjoint(row["operating_select"]))
            np.testing.assert_array_equal(
                np.sort(np.concatenate((row["threshold_fit"], row["operating_select"]))),
                np.arange(len(parent_hold)),
            )
        self.assertEqual(audit["global_exact_feature_overlap"], 0)
        self.assertIn("assignment_sha256", audit)

    def test_split_rejects_nonpositive_fraction(self):
        with self.assertRaises(ValueError):
            split_operating_subsets(
                np.zeros((4, 1)),
                np.zeros(4, dtype=int),
                [{"cal": np.arange(4)}],
                [(np.array([0, 1]), np.array([2, 3]))],
                0,
                1,
            )


if __name__ == "__main__":
    unittest.main()
