"""Exercise same-run continuation of native supplementary cases."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src import revision_full_matrix as matrix
from src import revision_resume_can as resume
from src import revision_run as native


def _interrupt_second_case(output, data_root):
    """Leave one atomic case result from a real synthetic native phase."""
    train = native._run_case
    trained = []

    def interrupt(*args, **kwargs):
        if trained:
            raise KeyboardInterrupt()
        trained.append(args[1])
        return train(*args, **kwargs)

    with patch.object(native, "_run_case", side_effect=interrupt):
        try:
            matrix.run_can_quantization(output, data_root, device="cpu", smoke=True)
        except KeyboardInterrupt:
            return next(output.rglob("result.json"))
    raise AssertionError("The synthetic supplement was not interrupted")


class NativeSupplementResumeTests(unittest.TestCase):
    """Preserve completed cases and stop before consuming unexpected evidence."""

    def test_can_resumes_three_unfinished_cases_and_keeps_the_first_one(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            output, data = root / "can_quantization", root / "dataset"
            first = _interrupt_second_case(output, data)
            original = first.read_bytes()
            with patch.object(native, "_run_case", wraps=native._run_case) as train:
                receipt = resume.resume_can_quantization(output, data, device="cpu", smoke=True)
                self.assertEqual(train.call_count, 3)
            self.assertEqual(first.read_bytes(), original)
            self.assertEqual(receipt["cases"], 4)
            self.assertEqual(receipt["status"], "completed")
            with patch.object(native, "main") as launch:
                self.assertEqual(resume.resume_can_quantization(output, data, device="cpu", smoke=True), receipt)
            launch.assert_not_called()

    def test_disagreeing_case_is_refused_before_training(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            output, data = root / "can_quantization", root / "dataset"
            first = _interrupt_second_case(output, data)
            corrupted = json.loads(first.read_text(encoding="utf-8"))
            corrupted["evaluations"].pop()
            first.write_text(json.dumps(corrupted), encoding="utf-8")
            with (
                patch.object(native, "main") as launch,
                self.assertRaisesRegex(ValueError, "incomplete or duplicate evaluations"),
            ):
                resume.resume_can_quantization(output, data, device="cpu", smoke=True)
            launch.assert_not_called()

    def test_single_phase_subsequent_invocations_preserve_the_published_receipt(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            for runner in (resume.resume_nb15_temporal, resume.resume_kminus1_transfer):
                output, data = root / runner.__name__, root / "dataset"
                result = runner(output, data, device="cpu", smoke=True)
                with patch.object(native, "main") as launch:
                    self.assertEqual(runner(output, data, device="cpu", smoke=True), result)
                launch.assert_not_called()

    def test_uncommitted_native_plan_can_restart_without_losing_evidence(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            output = root / "nb15_temporal"
            output.mkdir()
            (output / "run.partial").write_text('{"incomplete":', encoding="utf-8")
            result = resume.resume_nb15_temporal(output, root / "dataset", device="cpu", smoke=True)
            self.assertEqual(result["status"], "completed")
            self.assertFalse((output / "run.partial").exists())

    def test_native_phase_without_plan_keeps_unexpected_files(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            output = root / "nb15_temporal"
            output.mkdir()
            orphan = output / "result.json"
            orphan.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no committed run.json"):
                resume.resume_nb15_temporal(output, root / "dataset", device="cpu", smoke=True)
            self.assertEqual(orphan.read_text(encoding="utf-8"), "{}")


if __name__ == "__main__":
    unittest.main()
