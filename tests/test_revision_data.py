"""Regression checks for held-out identities and duplicate feature groups."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.revision_data import ROLES, _audit_exact_inputs, _feature_keys, _grouped_resplit, load_revision_data


class RevisionDataTest(unittest.TestCase):
    names = ["BENIGN", "ATTACK"]

    def test_feature_novelty_ignores_label_and_normalizes_missing_and_signed_zero(self):
        parts = {
            "train": (np.array([[-0.0, np.nan], [3.0, 0.0]]), np.array([0, 1])),
            "cal": (np.array([[4.0, 0.0]]), np.array([0])),
            "val": (np.array([[5.0, 0.0]]), np.array([1])),
            "test": (
                np.array([
                    [0.0, np.inf], [4.0, -0.0], [5.0, 0.0], [6.0, 0.0], [6.0, 0.0], [7.0, 0.0], [7.0, 0.0]
                ]),
                np.array([1, 1, 0, 1, 1, 0, 1]),
            ),
        }
        audit, selected = _audit_exact_inputs(parts, self.names, chunk_size=1)
        self.assertEqual(np.flatnonzero(selected).tolist(), [3])
        self.assertEqual(audit["cross_role_exact_overlap"]["train:test"]["test_rows"], 1)
        self.assertEqual(audit["cross_role_exact_overlap"]["cal:test"]["test_rows"], 1)
        self.assertEqual(audit["cross_role_exact_overlap"]["val:test"]["test_rows"], 1)
        self.assertEqual(audit["unseen_test_groups_with_conflicting_labels_excluded"], 1)
        self.assertEqual(audit["test_exact_input_unseen_deduplicated"], {"BENIGN": 0, "ATTACK": 1})

    def test_grouped_split_keeps_duplicate_and_conflicting_groups_indivisible(self):
        x = np.repeat(np.arange(8, dtype=np.float32), 3).reshape(-1, 1)
        y = np.repeat(np.arange(8) % 2, 3)
        y[0] = 1  # Both labels occur at the same feature key.
        indices = np.arange(len(y), dtype=np.uint64)
        parts = {r: (x[i::4], y[i::4]) for i, r in enumerate(ROLES)}
        ids = {r: indices[i::4] for i, r in enumerate(ROLES)}
        grouped, grouped_ids, summary = _grouped_resplit(parts, ids, self.names)
        _, repeated_ids, _ = _grouped_resplit(parts, ids, self.names)
        self.assertEqual(summary["groups_with_multiple_labels"], 1)
        for role in ROLES:
            np.testing.assert_array_equal(grouped_ids[role], repeated_ids[role])
            for other in ROLES:
                if role != other:
                    self.assertFalse(
                        np.intersect1d(_feature_keys(grouped[role][0]), _feature_keys(grouped[other][0])).size
                    )
        np.testing.assert_array_equal(np.sort(np.concatenate(list(grouped_ids.values()))), indices)

    def test_revision_loader_matches_frozen_replay_and_validates_reserved_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            raw = folder / "source.csv"
            raw.write_text(
                "f,Label\n10,BENIGN\n11,ATTACK\n12,BENIGN\n13,ATTACK\n14,BENIGN\n"
                "15,ATTACK\n16,BENIGN\n17,ATTACK\n18,BENIGN\n19,ATTACK\n",
                encoding="utf-8",
            )
            held = folder / "held.csv"
            held.write_text(
                "role,class,source_file_id,zero_based_data_row,row_in_role\n"
                "cal,BENIGN,0,0,1\ncal,ATTACK,0,1,0\n"
                "val,BENIGN,0,2,1\nval,ATTACK,0,3,0\n",
                encoding="utf-8",
            )
            excluded = folder / "excluded.csv"
            initial = (
                "source_file_id,zero_based_data_row,class\n"
                "0,0,BENIGN\n0,1,ATTACK\n0,2,BENIGN\n0,3,ATTACK\n0,4,BENIGN\n0,5,ATTACK\n"
            )
            excluded.write_text(initial, encoding="utf-8")
            spec = {
                "name": "fixture",
                "class_names": self.names,
                "feature_names": ["f"],
                "encoding": "utf-8",
                "label_column": "Label",
                "pipeline": None,
                "seed": 42,
                "train_rows": 4,
                "train_class_counts": [2, 2],
                "heldout_class_counts": {"cal": [1, 1], "val": [1, 1]},
                "source_class_counts": [5, 5],
                "heldout_identities_path": held,
                "excluded_identities_path": excluded,
                "recipe_sha256": "test fixture",
                "source_files": [
                    {
                        "filename": raw.name,
                        "path": raw,
                        "bytes": raw.stat().st_size,
                        "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                        "csv_data_rows": 10,
                    }
                ],
            }
            with patch("src.revision_data.check_raw_sources", return_value=spec):
                result = load_revision_data("fixture", folder, chunk_size=2)
                again = load_revision_data("fixture", folder, chunk_size=7)
                expected = np.arange(16, 20)[np.random.default_rng(42).permutation(4)]
                np.testing.assert_array_equal(result.parts["train"][0].ravel(), expected)
                np.testing.assert_array_equal(result.parts["cal"][0].ravel(), [11, 10])
                np.testing.assert_array_equal(result.parts["val"][0].ravel(), [13, 12])
                np.testing.assert_array_equal(result.parts["test"][0].ravel(), [14, 15])
                self.assertEqual(
                    result.audit["source_identity_sha256_by_role"],
                    again.audit["source_identity_sha256_by_role"],
                )
                self.assertTrue(result.test_unseen_mask.all())
                # Retain both TEST labels so only the raw identity-label check catches the error.
                excluded.write_text(
                    initial.replace("0,4,BENIGN", "0,4,ATTACK").replace("0,5,ATTACK", "0,5,BENIGN"),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "TEST identity manifest"):
                    load_revision_data("fixture", folder, chunk_size=2)


if __name__ == "__main__":
    unittest.main()
