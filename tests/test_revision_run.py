"""Regression tests for paired partitioning and evaluation edge cases."""

import unittest

import numpy as np

from src.revision_report import score
from src.revision_run import _synthetic, allocate_clients


class RevisionRunTests(unittest.TestCase):
    def test_partition_is_deterministic_and_complete_under_skew(self):
        data = _synthetic("CIC-IoV-2024", "grouped")
        first, repairs = allocate_clients(data.parts, len(data.names), 10, 0.03, 43)
        second, repeated_repairs = allocate_clients(data.parts, len(data.names), 10, 0.03, 43)
        self.assertEqual(repairs, repeated_repairs)
        for role, (_, y) in data.parts.items():
            seen = np.concatenate([client[role] for client in first])
            np.testing.assert_array_equal(np.sort(seen), np.arange(len(y)))
            for client, replay in zip(first, second, strict=True):
                np.testing.assert_array_equal(client[role], replay[role])
                self.assertGreaterEqual(len(client[role]), 2 if role in ("train", "cal") else 1)

    def test_score_marks_missing_classes_and_missing_binary_denominator(self):
        names = ["BENIGN", "ATTACK", "MISSING"]
        actual, predicted = np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])
        metrics, per_class, matrix = score(actual, predicted, np.array([0, 1, 0, 1]), names)
        self.assertEqual(metrics["far"], 0.5)
        self.assertEqual(metrics["fnr"], 0.5)
        self.assertEqual(metrics["macro_f1"], 0.5)
        self.assertEqual(metrics["balanced_accuracy"], 0.5)
        self.assertEqual(per_class[-1]["support"], 0)
        self.assertIsNone(per_class[-1]["recall"])
        self.assertEqual(matrix, [[1, 1, 0], [1, 1, 0], [0, 0, 0]])
        single, _, _ = score(actual[:2], predicted[:2], np.array([0, 1]), names)
        self.assertIsNone(single["fnr"])
        self.assertIsNone(single["macro_f1"])

    def test_test_assignment_does_not_access_test_labels(self):
        data = _synthetic("CIC-IoV-2024", "grouped")
        original, _ = allocate_clients(data.parts, len(data.names), 3, 0.8, 43)
        x, y = data.parts["test"]
        altered = {**data.parts, "test": (x, y[::-1])}
        replay, _ = allocate_clients(altered, len(data.names), 3, 0.8, 43)
        for first, second in zip(original, replay, strict=True):
            np.testing.assert_array_equal(first["test"], second["test"])


if __name__ == "__main__":
    unittest.main()
