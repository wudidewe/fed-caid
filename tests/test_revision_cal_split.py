"""Global CAL separation and infeasibility checks, without detector training."""

import unittest

import numpy as np

from src.revision_cal_split import cal_group_splits
from src.revision_data import _feature_keys


def fixture(groups_by_client, labels=None):
    groups = np.concatenate(groups_by_client)
    x = np.column_stack((groups, groups + 100)).astype(np.float32)
    y = np.zeros(len(x), dtype=np.int64) if labels is None else np.asarray(labels, dtype=np.int64)
    offsets = np.cumsum([0, *map(len, groups_by_client)])
    roster = [{"cal": np.arange(offsets[i], offsets[i + 1])} for i in range(len(groups_by_client))]
    return x, y, roster


class GlobalCalSplitTests(unittest.TestCase):
    def assert_global_disjoint(self, x, roster, splits):
        all_fit, all_hold = [], []
        for client, (fit, hold) in zip(roster, splits, strict=True):
            self.assertGreater(len(fit), 0)
            self.assertGreater(len(hold), 0)
            np.testing.assert_array_equal(np.sort(np.concatenate((fit, hold))), np.arange(len(client["cal"])))
            all_fit.extend(client["cal"][fit])
            all_hold.extend(client["cal"][hold])
        self.assertEqual(len(np.intersect1d(_feature_keys(x[all_fit]), _feature_keys(x[all_hold]))), 0)

    def test_shared_feature_groups_stay_on_one_side_across_clients(self):
        x, y, roster = fixture([[0, 0, 1, 2, 3], [0, 1, 1, 2, 4], [1, 2, 2, 3, 4]])
        splits, audit = cal_group_splits(x, y, roster, 0.5, 42)
        self.assert_global_disjoint(x, roster, splits)
        self.assertEqual(audit["global_exact_feature_overlap"], 0)
        self.assertEqual(audit["global_fit_rows"] + audit["global_holdout_rows"], len(y))
        self.assertEqual(audit["cal_feature_groups"], 5)
        self.assertEqual(audit, cal_group_splits(x, y, roster, 0.5, 42)[1])

    def test_default_assignment_is_label_blind_and_row_order_independent(self):
        x, y, roster = fixture([[0, 1], [1, 2], [2, 3]])
        first, audit = cal_group_splits(x, y, roster, 0.5, 4)
        relabeled = np.array([0, 1, 2, 1, 2, 0])
        second, other = cal_group_splits(x, relabeled, roster, 0.5, 4)
        self.assertEqual(audit["assignment_sha256"], other["assignment_sha256"])
        for a, b in zip(first, second, strict=True):
            np.testing.assert_array_equal(a[0], b[0])
            np.testing.assert_array_equal(a[1], b[1])
        reversed_roster = [{"cal": client["cal"][::-1]} for client in roster]
        _, reversed_audit = cal_group_splits(x, y, reversed_roster, 0.5, 4)
        self.assertEqual(audit["assignment_sha256"], reversed_audit["assignment_sha256"])

    def test_canonical_nonfinite_and_signed_zero_features_share_groups(self):
        x = np.array([[0, np.nan], [-0.0, np.inf], [1, 1], [2, 2], [3, 3], [4, 4]], dtype=float)
        roster = [{"cal": np.array([0, 2, 4])}, {"cal": np.array([1, 3, 5])}]
        splits, audit = cal_group_splits(x, np.array([0, 1, 0, 1, 0, 1]), roster, 0.5, 0)
        self.assert_global_disjoint(x, roster, splits)
        self.assertEqual(audit["cal_feature_groups"], 5)

    def test_feasibility_repair_keeps_every_client_nonempty(self):
        x, y, roster = fixture([[0, 1], [1, 2]])
        repairs = 0
        for seed in range(5):
            splits, audit = cal_group_splits(x, y, roster, 0.5, seed)
            repairs += audit["group_assignment_repaired"]
            self.assert_global_disjoint(x, roster, splits)
        self.assertGreater(repairs, 0)

    def test_jointly_infeasible_triangle_fails_without_row_fallback(self):
        # Each pair must straddle fit/holdout; an odd cycle cannot be two-colored.
        x, y, roster = fixture([[0, 1], [1, 2], [2, 0]])
        with self.assertRaisesRegex(ValueError, "infeasible global group constraints"):
            cal_group_splits(x, y, roster, 0.5, 0)

    def test_one_distinct_group_or_duplicate_ownership_is_invalid(self):
        x, y, roster = fixture([[0, 0], [1, 2]])
        with self.assertRaisesRegex(ValueError, "distinct feature groups"):
            cal_group_splits(x, y, roster, 0.5, 0)
        roster[0]["cal"] = np.array([0, 2])
        with self.assertRaisesRegex(ValueError, "single client owner"):
            cal_group_splits(x, y, roster, 0.5, 0)

    def test_explicit_benign_floor_respected_or_rejected(self):
        x, y, roster = fixture([[0, 1], [2, 3]], labels=[0, 1, 0, 1])
        splits, audit = cal_group_splits(x, y, roster, 0.5, 7, min_holdout_benign=1)
        self.assert_global_disjoint(x, roster, splits)
        self.assertTrue(all(client["holdout_benign_rows"] >= 1 for client in audit["clients"]))
        with self.assertRaisesRegex(ValueError, "insufficient benign"):
            cal_group_splits(x, y, roster, 0.5, 7, min_holdout_benign=2)


if __name__ == "__main__":
    unittest.main()
