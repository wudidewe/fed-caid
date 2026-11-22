"""Verify synthetic, fresh and continuation routes through the public entry."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import main as entry


class ReviewEntryTests(unittest.TestCase):
    def setUp(self):
        self.stdout = self.enterContext(redirect_stdout(StringIO()))
        self.stderr = self.enterContext(redirect_stderr(StringIO()))
        self.demo = self.enterContext(patch.object(entry, "_demo", return_value=0))
        self.full = self.enterContext(patch("src.revision_final_review.run_final_review", return_value=0))

    def test_no_datasets_require_explicit_synthetic_preview(self):
        imports_before = sys.path.copy()
        with TemporaryDirectory() as directory, patch.object(entry, "PROJECT", Path(directory).resolve()):
            self.assertEqual(entry.main([]), 2)
            self.demo.assert_not_called()
            self.assertIn('--demo', self.stderr.getvalue())
            self.assertEqual(entry.main(['--demo']), 0)
        self.assertEqual(sys.path, imports_before)
        self.demo.assert_called_once_with()
        self.full.assert_not_called()

    def test_partial_dataset_does_not_launch_demo_or_train(self):
        with TemporaryDirectory() as directory:
            project = Path(directory).resolve()
            marker = project / "dataset" / "CIC-UNSW-NB15" / "CICFlowMeter_out.csv"
            marker.parent.mkdir(parents=True)
            marker.touch()
            with patch.object(entry, "PROJECT", project):
                self.assertEqual(entry.main([]), 2)
        self.assertIn("dataset", self.stderr.getvalue())
        self.demo.assert_not_called()
        self.full.assert_not_called()

    def test_ready_datasets_select_fresh_full_review(self):
        with TemporaryDirectory() as directory:
            project = Path(directory).resolve()
            with (
                patch.object(entry, "PROJECT", project),
                patch.object(entry, "_dataset_files", return_value=(4, ())),
            ):
                self.assertEqual(entry.main([]), 0)
                self.assertEqual(entry.main(["--full-review"]), 0)
        self.assertEqual(self.full.call_count, 2)
        self.full.assert_called_with(project, hulk_manifest=None, lan_manifest=None)
        self.demo.assert_not_called()

    def test_ready_2018_capture_is_appended_to_the_full_review(self):
        from src.revision_hulk_2018 import DEFAULT_CAPTURE

        with TemporaryDirectory() as directory:
            project = Path(directory).resolve()
            capture = project / DEFAULT_CAPTURE
            capture.parent.mkdir(parents=True)
            capture.touch()
            with (
                patch.object(entry, "PROJECT", project),
                patch.object(entry, "_dataset_files", return_value=(4, ())),
            ):
                self.assertEqual(entry.main([]), 0)
        self.full.assert_called_once_with(
            project, hulk_manifest=None, lan_manifest=None, hulk_2018_source=capture
        )

    def test_2018_replay_uses_selected_run_without_retraining(self):
        csv = Path("capture.csv")
        run = Path("run_fixture")
        with (
            patch.object(entry, "_dataset_files", return_value=(8, ())) as check,
            patch("src.revision_final_review.run_hulk_2018_review", return_value=0) as replay,
        ):
            self.assertEqual(entry.main(["--hulk-2018", str(run), "--hulk-2018-source", str(csv)]), 0)
        check.assert_called_once_with(("CIC-IDS-2017",))
        replay.assert_called_once_with(entry.PROJECT, run, csv)
        self.full.assert_not_called()

    def test_explicit_full_review_requires_real_inputs(self):
        with TemporaryDirectory() as directory, patch.object(entry, "PROJECT", Path(directory).resolve()):
            self.assertEqual(entry.main(["--full-review"]), 2)
        self.demo.assert_not_called()
        self.full.assert_not_called()

    def test_iov_review_is_explicit(self):
        with (
            patch.object(entry, "_dataset_files", return_value=(4, ())),
            patch("src.revision_final_review.run_iov_review", return_value=0) as focused,
        ):
            self.assertEqual(entry.main(["--iov-review"]), 0)
        focused.assert_called_once_with(entry.PROJECT)
        self.full.assert_not_called()

    def test_resume_targets_the_explicit_run(self):
        with patch("src.revision_resume_review.resume_review", return_value=0) as resume:
            self.assertEqual(entry.main(["--resume", "run_fixture"]), 0)
        resume.assert_called_once_with(entry.PROJECT, "run_fixture")
        self.full.assert_not_called()
        self.demo.assert_not_called()

    def test_local_evidence_remains_available_without_training(self):
        with patch.object(entry, "_run_local_evidence", return_value=0) as evidence:
            self.assertEqual(entry.main(["--local-evidence"]), 0)
        evidence.assert_called_once_with()
        self.full.assert_not_called()
        self.demo.assert_not_called()

    def test_nb15_specialists_share_the_public_entry(self):
        with (
            patch.object(entry, "_dataset_files", return_value=(1, ())),
            patch("src.revision_nb15_low_far.run_nb15_low_far") as fit,
            patch("src.revision_nb15_threshold.run_nb15_threshold") as threshold,
        ):
            self.assertEqual(entry.main(["--nb15-low-far"]), 0)
            self.assertEqual(entry.main(["--nb15-threshold"]), 0)
        fit.assert_called_once_with(entry.PROJECT)
        threshold.assert_called_once_with(entry.PROJECT)
        self.full.assert_not_called()

    def test_can_launch_checkout_from_an_unrelated_working_directory(self):
        with TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-E", str(entry.PROJECT / "main.py"), "--help"],
                cwd=directory, check=True, capture_output=True, text=True, timeout=30,
            )
        self.assertIn("--full-review", result.stdout)
        self.assertIn("--resume", result.stdout)


class DownloadDemoTests(unittest.TestCase):
    def _synthetic_run(self, arguments):
        self.assertIn("--smoke", arguments)
        output = Path(arguments[arguments.index("--out") + 1])
        output.mkdir(parents=True, exist_ok=False)
        receipt = {"smoke": True, "test_evaluated": False, "evaluated": 1}
        (output / "complete.json").write_text(json.dumps(receipt), encoding="utf-8")
        return 0

    def test_demo_can_allocate_output_for_the_native_runner(self):
        with (
            TemporaryDirectory() as directory,
            patch.object(entry, "PROJECT", Path(directory).resolve()),
            patch("src.revision_next.main", side_effect=self._synthetic_run),
            redirect_stdout(StringIO()),
        ):
            self.assertEqual(entry._demo(), 0)
            outputs = (Path(directory).resolve() / "outputs" / "fedcaid_demo").rglob("complete.json")
            self.assertEqual(len(list(outputs)), 1)
