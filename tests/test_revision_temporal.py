"""Check that chronological roles follow timestamps and preserve row identities."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.revision_data import ROLES, _audit_exact_inputs, _feature_keys
from src.revision_temporal import _purge_temporal_inputs, load_temporal_data


class RevisionTemporalTest(unittest.TestCase):
    def test_forward_purge_canonical_inputs_and_conflict_ownership(self):
        parts = {
            "train": (
                np.array([[-0.0, 1], [np.nan, 2], [10, 10], [10, 10]], dtype=np.float32),
                np.array([0, 0, 1, 1]),
            ),
            "cal": (
                np.array(
                    [[0.0, 1], [np.inf, 2], [20, 20], [20, 20], [21, 21], [21, 21], [22, 22]],
                    dtype=np.float32,
                ),
                np.array([1, 0, 0, 0, 0, 1, 1]),
            ),
            "val": (
                np.array([[21, 21], [20, 20], [30, 30], [30, 30]], dtype=np.float32),
                np.array([1, 1, 0, 0]),
            ),
            "test": (
                np.array(
                    [[21, 21], [30, 30], [40, 40], [40, 40], [41, 41], [41, 41], [-np.inf, 2], [0.0, 1]],
                    dtype=np.float32,
                ),
                np.array([1, 0, 1, 1, 0, 1, 0, 1]),
            ),
        }
        names = ["BENIGN", "ATTACK"]
        keep, report = _purge_temporal_inputs(parts, names, 2)
        again, repeated_report = _purge_temporal_inputs(parts, names, 20)
        expected = {
            "train": [True] * 4,
            "cal": [False, False, True, False, False, False, True],
            "val": [False, False, True, True],
            "test": [False, False, True, True, True, True, False, False],
        }
        for role in ROLES:
            np.testing.assert_array_equal(keep[role], expected[role])
            np.testing.assert_array_equal(keep[role], again[role])
        self.assertEqual(report, repeated_report)
        self.assertEqual(report["cal_conflicting_groups_excluded"], 1)
        self.assertEqual(report["cal_conflicting_rows_excluded"], 2)
        self.assertEqual(report["cal_duplicate_rows_excluded"], 1)
        clean = {role: (x[keep[role]], y[keep[role]]) for role, (x, y) in parts.items()}
        exact, unseen = _audit_exact_inputs(clean, names, 2)
        self.assertTrue(
            all(pair["unique_shared_inputs"] == 0 for pair in exact["cross_role_exact_overlap"].values())
        )
        np.testing.assert_array_equal(unseen, [True, False, False, False])
        # A unique CAL input can never be in both sides of any row-wise split.
        self.assertEqual(len(np.unique(_feature_keys(clean["cal"][0]))), len(clean["cal"][1]))
        # CAL labels do not change TRAIN or permit a discarded CAL group later.
        parts["test"][1][:] = 0
        changed, _ = _purge_temporal_inputs(parts, names, 2)
        for role in ROLES:
            np.testing.assert_array_equal(changed[role], keep[role])

    def test_purged_loader_preserves_retained_source_identities(self):
        entries = [
            ("18/02/2015 01:00:00 AM", 8, "ATTACK"),
            ("17/02/2015 01:00:00 AM", 2, "BENIGN"),
            ("22/01/2015 01:00:00 AM", 1, "BENIGN"),
            ("17/02/2015 02:00:00 AM", 2, "BENIGN"),
            ("17/02/2015 10:00:00 PM", 4, "ATTACK"),
            ("17/02/2015 03:00:00 AM", 4, "BENIGN"),
            ("17/02/2015 04:00:00 AM", 4, "ATTACK"),
            ("17/02/2015 11:00:00 PM", 5, "ATTACK"),
            ("22/01/2015 02:00:00 AM", 0, "BENIGN"),
            ("17/02/2015 05:00:00 AM", 3, "BENIGN"),
            ("17/02/2015 11:30:00 PM", 5, "ATTACK"),
            ("18/02/2015 02:00:00 AM", 4, "ATTACK"),
            ("18/02/2015 03:00:00 AM", 5, "ATTACK"),
            ("18/02/2015 04:00:00 AM", 8, "ATTACK"),
            ("18/02/2015 05:00:00 AM", 9, "BENIGN"),
            ("18/02/2015 06:00:00 AM", 9, "ATTACK"),
            ("17/02/2015 06:00:00 AM", 1, "BENIGN"),
        ]
        with tempfile.TemporaryDirectory() as folder:
            raw = Path(folder) / "CICFlowMeter_out.csv"
            raw.write_text(
                "Timestamp,f,Label\n"
                + "".join(f"{stamp},{feature},{label}\n" for stamp, feature, label in entries),
                encoding="utf-8",
            )
            spec = {
                "name": "CIC-UNSW-NB15",
                "class_names": ["BENIGN", "ATTACK"],
                "feature_names": ["f"],
                "encoding": "utf-8",
                "label_column": "Label",
                "pipeline": None,
                "source_class_counts": [
                    sum(label == name for _, _, label in entries) for name in ("BENIGN", "ATTACK")
                ],
                "recipe_sha256": "test fixture",
                "source_files": [
                    {
                        "filename": raw.name,
                        "path": raw,
                        "bytes": raw.stat().st_size,
                        "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                        "csv_data_rows": len(entries),
                    }
                ],
            }
            with patch("src.revision_temporal.check_raw_sources", return_value=spec):
                result = load_temporal_data(folder, chunk_size=2, purge_exact=True)
                repeated = load_temporal_data(folder, chunk_size=7, purge_exact=True)
                historical = load_temporal_data(folder, chunk_size=3)
        expected_rows = {"train": [2, 8], "cal": [1, 9], "val": [7, 10], "test": [0, 13, 14, 15]}
        self.assertEqual(result.audit["split_protocol"], "temporal_purged")
        self.assertEqual(historical.audit["split_protocol"], "temporal")
        self.assertFalse(historical.audit["temporal_exact_purge"]["enabled"])
        for role, rows in expected_rows.items():
            self.assertEqual(
                result.audit["source_identity_sha256_by_role"][role],
                hashlib.sha256(np.array(rows, dtype="<u8")).hexdigest(),
            )
            np.testing.assert_array_equal(result.parts[role][0], repeated.parts[role][0])
        self.assertEqual(result.audit, repeated.audit)
        purge = result.audit["temporal_exact_purge"]
        self.assertEqual(purge["role_rows_before"], {"train": 2, "cal": 6, "val": 3, "test": 6})
        self.assertEqual(purge["role_rows_after"], {"train": 2, "cal": 2, "val": 2, "test": 4})
        self.assertEqual(
            purge["source_identity_sha256_by_role_before"], historical.audit["source_identity_sha256_by_role"]
        )
        self.assertEqual(
            purge["source_identity_sha256_by_role_after"], result.audit["source_identity_sha256_by_role"]
        )
        np.testing.assert_array_equal(result.test_unseen_mask, [True, False, False, False])

    def test_unsorted_rows_respect_time_windows_and_audit_test_overlap(self):
        with tempfile.TemporaryDirectory() as folder:
            raw = Path(folder) / "CICFlowMeter_out.csv"
            raw.write_text(
                "Timestamp,f,Label\n"
                "18/02/2015 12:00:00 AM,10,BENIGN\n"
                "17/02/2015 10:00:00 PM,30,ATTACK\n"
                "22/01/2015 07:00:00 AM,10,BENIGN\n"
                "17/02/2015 09:59:59 PM,20,BENIGN\n"
                "18/02/2015 01:00:00 AM,40,ATTACK\n"
                "18/02/2015 02:00:00 AM,40,ATTACK\n",
                encoding="utf-8",
            )
            spec = {
                "name": "CIC-UNSW-NB15",
                "class_names": ["BENIGN", "ATTACK"],
                "feature_names": ["f"],
                "encoding": "utf-8",
                "label_column": "Label",
                "pipeline": None,
                "source_class_counts": [3, 3],
                "recipe_sha256": "test fixture",
                "source_files": [
                    {
                        "filename": raw.name,
                        "path": raw,
                        "bytes": raw.stat().st_size,
                        "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                        "csv_data_rows": 6,
                    }
                ],
            }
            with patch("src.revision_temporal.check_raw_sources", return_value=spec):
                result = load_temporal_data(folder, chunk_size=2)
                repeated = load_temporal_data(folder, chunk_size=5)

            self.assertEqual(result.features, ["f"])
            np.testing.assert_array_equal(result.parts["train"][0].ravel(), [10])
            np.testing.assert_array_equal(result.parts["cal"][0].ravel(), [20])
            np.testing.assert_array_equal(result.parts["val"][0].ravel(), [30])
            np.testing.assert_array_equal(result.parts["test"][0].ravel(), [10, 40, 40])
            np.testing.assert_array_equal(result.test_unseen_mask, [False, True, False])
            self.assertEqual(
                result.audit["source_identity_sha256_by_role"],
                repeated.audit["source_identity_sha256_by_role"],
            )
            self.assertEqual(
                result.audit["exact_feature_audit"]["test_rows_matching_development_class_support"],
                {"BENIGN": 1, "ATTACK": 0},
            )


if __name__ == "__main__":
    unittest.main()
