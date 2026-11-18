"""Controlled benign CAL reservation without moving training or attack rows."""

import unittest

import numpy as np

from src.revision_benign_cal import reserve_benign_cal
from src.revision_cal_split import cal_group_splits


class BenignCalReservationTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array(
            [[i, 2 * i] for i in [0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 11]], dtype=np.float32
        )
        self.y = np.array([1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0], dtype=np.int64)
        self.roster = [
            {"cal": np.arange(0, 5), "train": np.array([10]), "val": np.array([0])},
            {"cal": np.arange(5, 10), "train": np.array([20]), "val": np.array([1])},
            {"cal": np.arange(10, 14), "train": np.array([30]), "val": np.array([2])},
        ]

    def test_reserves_floor_without_changing_noncal_or_attack_ownership(self):
        actual, receipt = reserve_benign_cal(self.x, self.y, self.roster, 3, 23)
        repeated, repeat_receipt = reserve_benign_cal(self.x, self.y, self.roster, 3, 23)
        self.assertEqual(receipt, repeat_receipt)
        np.testing.assert_array_equal(
            np.sort(np.concatenate([v["cal"] for v in actual])), np.arange(len(self.y))
        )
        self.assertGreater(receipt["moved_benign_rows"], 0)
        self.assertEqual(receipt["moved_attack_rows"], 0)
        for source, result, repeat in zip(self.roster, actual, repeated, strict=True):
            self.assertIs(source["train"], result["train"])
            self.assertIs(source["val"], result["val"])
            np.testing.assert_array_equal(result["cal"], repeat["cal"])
            np.testing.assert_array_equal(
                source["cal"][self.y[source["cal"]] > 0], result["cal"][self.y[result["cal"]] > 0]
            )
            self.assertGreaterEqual(len(np.unique(self.x[result["cal"], 0])), 2)
        _, split_receipt = cal_group_splits(self.x, self.y, actual, 0.5, 23, min_holdout_benign=3)
        self.assertEqual(split_receipt["global_exact_feature_overlap"], 0)
        self.assertTrue(all(c["holdout_benign_rows"] >= 3 for c in split_receipt["clients"]))

    def test_respects_group_atomicity_and_leaves_mixed_groups(self):
        x = np.repeat(self.x, 2, axis=0)
        y = np.repeat(self.y, 2)
        roster = [{"cal": np.arange(2 * a, 2 * b)} for a, b in ((0, 5), (5, 10), (10, 14))]
        actual, _ = reserve_benign_cal(x, y, roster, 6, 20)
        before = {index: i for i, row in enumerate(roster) for index in row["cal"]}
        after = {index: i for i, row in enumerate(actual) for index in row["cal"]}
        for key in np.unique(x[:, 0]):
            group = np.flatnonzero(x[:, 0] == key)
            if np.any(y[group] > 0):
                self.assertTrue(all(before[index] == after[index] for index in group))
            if any(before[index] != after[index] for index in group):
                self.assertEqual(len({after[index] for index in group}), 1)

    def test_validates_explicit_groups_and_inventory(self):
        with self.assertRaisesRegex(ValueError, "split an exact feature group"):
            reserve_benign_cal(self.x, self.y, self.roster, 3, 5, group_ids=np.arange(len(self.y)))
        _, audit = reserve_benign_cal(self.x, self.y, self.roster, 3, 5, group_ids=self.x[:, 0].astype(int))
        self.assertIn("explicit CAL groups", audit["grouping"])
        duplicate = [*self.roster, self.roster[0]]
        with self.assertRaisesRegex(ValueError, "one owner"):
            reserve_benign_cal(self.x, self.y, duplicate, 3, 5)
        with self.assertRaisesRegex(ValueError, "insufficient total benign rows"):
            reserve_benign_cal(self.x, self.y, self.roster, 4, 5)

    def test_owners_are_unchanged_when_every_client_has_enough_benign_cal(self):
        result, audit = reserve_benign_cal(self.x, self.y, self.roster, 2, 5)
        self.assertEqual(audit["moved_benign_rows"], 0)
        for source, actual in zip(self.roster, result, strict=True):
            np.testing.assert_array_equal(source["cal"], actual["cal"])


if __name__ == "__main__":
    unittest.main()
