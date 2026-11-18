"""Exercise in-place review continuation without training on real data."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src import revision_resume_review as resume
from src.revision_v12 import _hash, _read, _write


class ResumeReviewTests(unittest.TestCase):
    def setUp(self):
        temporary = self.enterContext(TemporaryDirectory())
        self.enterContext(redirect_stdout(StringIO()))
        self.project = Path(temporary).resolve()
        self.run_name = "run_fixture"
        self.output = self.project / "outputs" / resume.CURRENT_REVIEW / self.run_name
        self.output.mkdir(parents=True)
        self.plan_path = self.output / "final_plan.json"
        self.state_path = self.output / "final_state.json"
        self.marker_path = self.output / "final_complete.json"
        self.calls = []
        self.prefix = self._write_fixture()
        self.enterContext(patch.object(resume, "_operations", self._create_operations))

    def _write_fixture(self):
        """Create a stopped full-run plan with valid artifacts through scale."""
        preflight = {name: {"status": "pending", "reason": "fixture"} for name in ("lan", "external")}
        source = Path(resume.__file__).with_name("revision_final_review.py")
        plan = {
            "project": str(self.project),
            "run_directory": str(self.output),
            "schema_version": resume.SCHEMA_VERSION,
            "workflow": resume.FULL_WORKFLOW,
            "stages": list(resume.STAGE_LABELS),
            "input_preflight": preflight,
            "entry_sha256": {},
            "source_sha256": {source.name: _hash(source)},
            "requested_device": "auto",
            "synthetic_only": False,
            "previous_run_inputs_allowed": False,
        }
        names = plan["stages"][:-len(resume.SUPPLEMENTARY_STAGES)]
        state = {name: self._stage_record(name, preflight) for name in names}
        _write(self.plan_path, plan)
        _write(self.state_path, state)
        _write(
            self.marker_path,
            {"status": "running", "schema_version": resume.SCHEMA_VERSION, "plan_sha256": _hash(self.plan_path)},
        )
        return state

    def _stage_record(self, name, pending):
        """Write a small hash-bound artifact or preserve a pending input."""
        path = self.output / name
        if name in pending:
            return {"output": str(path), **pending[name]}
        path.mkdir()
        file = path / ("SUMMARY.md" if name == "report" else "run.json")
        _write(file, {"stage": name, "device": "cpu"})
        return {"status": "completed", "output": str(path), "artifacts": {file.name: _hash(file)}}

    def _create_operations(self, project, state, device):
        """Replace training and reporting with deterministic local callbacks."""
        self.assertEqual(project, self.project)
        self.assertEqual(device, "cpu")
        self.assertEqual({name: state[name] for name in self.prefix}, self.prefix)
        return {name: self._perform for name in resume.SUPPLEMENTARY_STAGES}

    def _perform(self, path):
        """Record a scheduled stage and write its fixture evidence."""
        self.calls.append(path.name)
        return self._stage_record(path.name, {})

    def test_only_remaining_stages_execute_and_completed_run_is_idempotent(self):
        original_plan = _hash(self.plan_path)
        self.assertEqual(resume.resume_review(self.project, self.run_name), 0)
        self.assertEqual(self.calls, list(resume.SUPPLEMENTARY_STAGES))
        state = _read(self.state_path)
        self.assertEqual({name: state[name] for name in self.prefix}, self.prefix)
        self.assertEqual(_hash(self.plan_path), original_plan)
        result = _read(self.marker_path)
        self.assertEqual(result["status"], "completed_with_pending")
        self.assertEqual(result["pending_steps"], ["lan", "external"])
        self.assertEqual(resume.resume_review(self.project, self.run_name), 0)
        self.assertEqual(self.calls, list(resume.SUPPLEMENTARY_STAGES))

    def test_changed_preserved_artifact_stops_before_execution(self):
        (self.output / "scale_matrix" / "run.json").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Current-run artifact changed"):
            resume.resume_review(self.project, self.run_name)
        self.assertEqual(self.calls, [])

    def test_missing_earlier_stage_stops_before_execution(self):
        state = _read(self.state_path)
        state.pop("nb15_pair")
        _write(self.state_path, state)
        with self.assertRaisesRegex(ValueError, "completed prefix"):
            resume.resume_review(self.project, self.run_name)
        self.assertEqual(self.calls, [])

    def test_report_publish_can_recover_without_replacing_report(self):
        record = self._stage_record("report", {})
        report = self.output / "report"
        evidence = report / "evidence_plan.json"
        sources = {
            str(self.output / stage / name): digest
            for stage, previous in self.prefix.items()
            for name, digest in previous.get("artifacts", {}).items()
        }
        _write(evidence, {"sources": sources, "run_plan_sha256": _hash(self.plan_path)})
        record["artifacts"][evidence.name] = _hash(evidence)
        completion = report / "complete.json"
        _write(completion, record)
        resumed = resume._recover_report(report, self.prefix)
        self.assertEqual(resumed["artifacts"][completion.name], _hash(completion))

    def test_interruption_keeps_saved_progress_for_next_click(self):
        perform = self._perform

        def interrupt(path):
            """Stop when the next stage starts for the first time."""
            if path.name == "nb15_temporal":
                raise KeyboardInterrupt()
            return perform(path)

        with patch.object(self, "_perform", side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                resume.resume_review(self.project, self.run_name)
        self.assertEqual(self.calls, ["can_quantization"])
        self.assertEqual(_read(self.marker_path)["status"], "aborted")
        self.assertIn("can_quantization", _read(self.state_path))
        self.assertEqual(resume.resume_review(self.project, self.run_name), 0)
        self.assertEqual(self.calls, list(resume.SUPPLEMENTARY_STAGES))

if __name__ == "__main__":
    unittest.main()
