"""Verify the no-training local evidence bundle and its derived IoV table."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.revision_local_review import IOV_TABLES, _iov_attribution, _iov_lines, run_local_review
from src.revision_report import write_csv
from src.revision_v12 import _hash, _read


def _operating_rows():
    for mode, counts in (
        ("base_threshold", (9, 0, 0)),
        ("local_only", (9, 7, 0)),
        ("full", (9, 7, 0)),
    ):
        for client, count in enumerate(counts):
            yield {
                "seed": 42, "client": client, "mode": mode, "role": "val",
                "false_alarm_rows": count, "benign_rows": 20,
                "false_alarm_groups": int(count > 0), "top_false_alarm_group_rows": count,
            }


def _overlap_rows():
    for reference, mode in (("base_threshold", "local_only"), ("local_only", "full")):
        for client in range(3):
            yield {
                "seed": 42, "client": client, "role": "val", "reference": reference, "candidate": mode,
                "benign_alert_rows_added": 7 if mode == "local_only" and client == 1 else 0,
                "benign_alert_rows_removed": 0,
            }


class LocalReviewTests(unittest.TestCase):
    def test_missing_inputs_are_recorded_and_runs_have_separate_receipts(self):
        """An empty checkout can explain missing evidence without starting training."""
        with TemporaryDirectory() as temporary:
            project = Path(temporary).resolve()
            with patch("src.revision_run.main", side_effect=AssertionError("Unexpected training")):
                first, second = run_local_review(project), run_local_review(project)
            outputs = [Path(record["output"]) for record in (first, second)]
            self.assertNotEqual(*outputs)
            for output in outputs:
                receipt = _read(output / "complete.json")
                self.assertEqual(receipt["trained_cases"], 0)
                self.assertFalse(receipt["new_test_scoring"])
                self.assertFalse(receipt["new_network_measurement"])
                self.assertEqual(receipt["artifacts"]["SUMMARY.md"], _hash(output / "SUMMARY.md"))
                self.assertIn("当前不可用", (output / "SUMMARY.md").read_text(encoding="utf-8"))

    def test_invalid_evidence_cannot_complete_a_report(self):
        """Keep a terminal failure receipt if source verification fails."""
        with (
            TemporaryDirectory() as temporary,
            patch("src.revision_local_review._source_records", side_effect=ValueError("Source artifact changed")),
        ):
            project = Path(temporary).resolve()
            with self.assertRaisesRegex(ValueError, "Source artifact changed"):
                run_local_review(project)
            receipt, = (project / "outputs").rglob("complete.json")
            self.assertEqual(_read(receipt)["status"], "failed")

    def test_iov_distinguishes_added_rows_from_client_group_occurrences(self):
        """Keep one output per seed and identify the correct comparison reference."""
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            tables = root / IOV_TABLES
            tables.mkdir(parents=True)
            write_csv(tables / "iov_operating_diagnostics.csv", list(_operating_rows()))
            write_csv(tables / "iov_alarm_overlap.csv", list(_overlap_rows()))
            base, local, full = _iov_attribution(root)
        self.assertEqual(base["row_far"], 9 / 60)
        self.assertEqual(local["false_alarm_client_group_occurrences"], 2)
        self.assertEqual(local["largest_client_group_share_of_seed_false_alarms"], 9 / 16)
        self.assertEqual(local["benign_alert_rows_added"], 7)
        self.assertEqual(full["comparison_reference"], "local_only")
        self.assertEqual(full["benign_alert_rows_added"], 0)
        self.assertIn("15.000%", "\n".join(_iov_lines([base], root)))
