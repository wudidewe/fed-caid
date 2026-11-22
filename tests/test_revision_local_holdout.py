"""Check separation of current support and historical evaluation exposure."""

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.revision_local_holdout import CURRENT_SCOPE, HISTORICAL_SCOPE, NB15, holdout_rows


def _audit(protocol, test_digest="current-test"):
    """Create class support for a reserve that includes an unsupported class."""
    roles = {
        "val": {"BENIGN": 60, "Attack A": 7, "Attack B": 0},
        "test": {"BENIGN": 70, "Attack A": 12, "Attack B": 0},
    }
    return {
        "split_protocol": protocol,
        "external_captures": False,
        "source_files_sha256": {"capture.csv": "capture-contents"},
        "source_identity_sha256_by_role": {"val": f"{protocol}-val", "test": test_digest},
        "role_rows": {role: sum(support.values()) for role, support in roles.items()},
        "role_class_support": roles,
        "exact_feature_audit": {
            "within_role_repeats": {"val": {"distinct_inputs": 20}, "test": {"distinct_inputs": 14}},
            "test_exact_input_unseen_deduplicated": {"BENIGN": 10, "Attack A": 4, "Attack B": 0},
        },
        "grouped_resplit": {"test_rows_formerly_in_original_train": 17},
    }


def _metric(dataset, protocol, role, rows, *, mode="base", seed=42):
    """Simulate measurement identity without model scores."""
    scenario = "main" if protocol == "grouped" else "supplement_temporal"
    return {
        "dataset": dataset,
        "protocol": protocol,
        "scenario": scenario,
        "mode": mode,
        "role": role,
        "rows": rows,
        "seed": seed,
    }


def _temporal(test_digest="historical-test", seed=42):
    """Supply explicit historical, within-source plan, audit and result metadata."""
    return {
        "source": "verified/temporal/metrics_by_seed.csv",
        "run": {
            "matrix": [
                {"dataset": NB15, "protocol": "temporal", "scenarios": [["supplement_temporal", 3, 0.8]]}
            ],
            "synthetic_only": False,
            "data_only": False,
            "seeds": [42],
        },
        "audit": _audit("temporal_purged", test_digest),
        "metrics": [
            _metric(NB15, "temporal", "test", 82, seed=seed),
            _metric(NB15, "temporal", "test", 82, mode="full", seed=seed),
            _metric(NB15, "temporal", "test_unique_unseen", 14, seed=seed),
        ],
    }


class LocalHoldoutAuditTests(unittest.TestCase):
    """Treat missing provenance as unknown and keep unsupported classes visible."""

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tables = Path(temporary.name)

    def _write_current(self, dataset=NB15, role="val", rows=67):
        path = self.tables / "benchmark_matrix" / dataset / "grouped" / "data_audit.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_audit("grouped")), encoding="utf-8")
        metric = _metric(dataset, "grouped", role, rows)
        with (self.tables / "benchmark_matrix/metrics_by_seed.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=metric)
            writer.writeheader()
            writer.writerow(metric)

    def _rows(self, temporal=None):
        keys = ("dataset", "evidence_scope", "role")
        return {tuple(row[key] for key in keys): row for row in holdout_rows(self.tables, temporal)}

    def test_can_reserve_reports_zero_class_and_deduplicated_attack_support(self):
        dataset = "CIC-IoV-2024"
        self._write_current(dataset)
        result = self._rows()
        test = result[dataset, CURRENT_SCOPE, "TEST_EXACT_UNIQUE"]
        self.assertEqual(
            (test["support_unit"], test["support"], test["attack_support"]), ("exact_input_groups", 14, 4)
        )
        self.assertEqual(test["missing_attack_classes"], "Attack B")
        self.assertEqual(test["attack_classes_with_1_to_4_units"], "Attack A")
        self.assertEqual(test["scored_in_current_run"], "no")
        self.assertEqual(test["scored_in_verified_history"], "unknown")
        self.assertFalse(test["independent_confirmation_eligible"])
        self.assertEqual(result[dataset, CURRENT_SCOPE, "VAL"]["scored_in_current_run"], "yes")

    def test_temporal_test_proves_past_execution_only_at_its_own_role(self):
        self._write_current()
        result = self._rows(_temporal())
        previous = result[NB15, HISTORICAL_SCOPE, "TEST"]
        current = result[NB15, CURRENT_SCOPE, "TEST"]
        self.assertEqual(previous["scored_in_verified_history"], "yes")
        self.assertEqual(previous["seeds_scored_in_verified_history"], 1)
        self.assertEqual(current["exposure_status"], "same_source_overlap_unknown")
        self.assertEqual(current["scored_in_verified_history"], "unknown")
        self.assertEqual(current["grouped_test_rows_from_original_train_role"], 17)
        self.assertFalse(previous["independent_confirmation_eligible"])

    def test_matching_ordered_role_with_same_source_marks_previous_exposure(self):
        self._write_current()
        result = self._rows(_temporal(test_digest="current-test"))
        current = result[NB15, CURRENT_SCOPE, "TEST"]
        self.assertEqual(current["exposure_status"], "same_ordered_role_previously_scored")
        self.assertEqual(current["scored_in_current_run"], "no")
        self.assertEqual(current["scored_in_verified_history"], "yes")

    def test_missing_and_inconsistent_records(self):
        current = self._rows()[NB15, CURRENT_SCOPE, "TEST"]
        self.assertEqual(current["scored_in_current_run"], "unknown")
        self.assertFalse(current["independent_confirmation_eligible"])
        with self.assertRaisesRegex(ValueError, "recorded seeds"):
            self._rows(_temporal(seed=999))
        self._write_current(rows=5)
        with self.assertRaisesRegex(ValueError, "differ from the .* audit"):
            self._rows()


if __name__ == "__main__":
    unittest.main()
