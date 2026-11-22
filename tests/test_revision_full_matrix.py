"""Verify that supplementary stages bind their declared cases and evidence scope."""

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.revision_full_matrix import (
    CAN_PHASES,
    KMINUS1_TRANSFER,
    NB15_TEMPORAL,
    _checked_metrics,
    _scale_cases,
    _supplement_cases,
    _validate_scale_budget,
    _validate_temporal_audit,
    run_kminus1_transfer,
    run_scale_matrix,
)
from src.revision_run import SEEDS


class FullMatrixContractTests(unittest.TestCase):
    """Keep case design and safety boundaries explicit without formal model training."""

    def test_native_workloads_match_the_declared_scales_and_protocols(self):
        cases = _scale_cases(SEEDS, smoke=False)
        self.assertEqual(len(cases), 90)
        self.assertEqual({case[2] for case in cases if "_alpha_" not in case[1]}, {3, 5, 10})
        self.assertEqual({case[3] for case in cases if "_alpha_" in case[1]}, {0.3, 0.8, 2.0})
        self.assertEqual(sum(len(_supplement_cases(phase, SEEDS)) for phase in CAN_PHASES), 20)
        self.assertEqual(len(_supplement_cases(NB15_TEMPORAL, SEEDS)), 5)
        self.assertEqual(len(_supplement_cases(KMINUS1_TRANSFER, SEEDS)), 10)
        self.assertIn("--transfer-full-train", KMINUS1_TRANSFER.native_flags)
        self.assertNotIn("--repair-missing-classes", KMINUS1_TRANSFER.native_flags)

    def test_scale_fixed_per_client_requires_each_client_budget(self):
        case = ("CIC-UNSW-NB15", "scale_local_c3", 3, 0.8, 42)
        audit = {
            "test_evaluated": False,
            "budget_policy": "local",
            "train_rows": 99000,
            "cal_rows": 16500,
            "train_rows_by_client": [33000, 33000, 33000],
            "original_cal_rows_by_client": [5500, 5500, 5500],
        }
        _validate_scale_budget(audit, case, smoke=False)
        audit["train_rows_by_client"] = [32999, 33001, 33000]
        with self.assertRaisesRegex(ValueError, "fixed per-client"):
            _validate_scale_budget(audit, case, smoke=False)

    def test_temporal_stage_validates_cross_role_exact_duplicates(self):
        pairs = ("train:cal", "train:val", "train:test", "cal:val", "cal:test", "val:test")
        overlaps = {pair: {"unique_shared_inputs": 0} for pair in pairs}
        audit = {
            "split_protocol": "temporal_purged",
            "external_captures": False,
            "exact_feature_audit": {"cross_role_exact_overlap": overlaps},
        }
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "audit.json"
            path.write_text(json.dumps(audit), encoding="utf8")
            _validate_temporal_audit(path)
            overlaps["train:test"]["unique_shared_inputs"] = 1
            path.write_text(json.dumps(audit), encoding="utf8")
            with self.assertRaisesRegex(ValueError, "exact-input role separation"):
                _validate_temporal_audit(path)

    def test_scope_validator_rejects_unplanned_and_duplicate_evaluations(self):
        row = {
            "dataset": "CIC-UNSW-NB15",
            "scenario": "scale_total_c3",
            "clients": 3,
            "alpha": 0.8,
            "seed": 42,
            "role": "val",
            "mode": "base_threshold",
        }
        cases = {("CIC-UNSW-NB15", "scale_total_c3", 3, 0.8, 42)}
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (root / "metrics.csv").open("w", encoding="utf8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=row)
                writer.writeheader()
                writer.writerows([row, row])
            with self.assertRaisesRegex(ValueError, "Duplicate supplementary"):
                _checked_metrics(root, "metrics.csv", cases, ("base_threshold",), ("val",))
            with self.assertRaisesRegex(ValueError, "Unexpected supplementary metric population"):
                _checked_metrics(root, "metrics.csv", cases, ("base_threshold",), ("test",))

    def test_supplements_cannot_open_an_existing_result_directory(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            existing = root / "existing"
            existing.mkdir()
            for runner in (run_scale_matrix, run_kminus1_transfer):
                with (
                    patch("src.revision_full_matrix.native.main") as native,
                    patch("src.revision_full_matrix.shared.main") as shared,
                    self.assertRaisesRegex(ValueError, "new output directory"),
                ):
                    runner(existing, root / "raw", device="cpu")
                native.assert_not_called()
                shared.assert_not_called()


if __name__ == "__main__":
    unittest.main()
