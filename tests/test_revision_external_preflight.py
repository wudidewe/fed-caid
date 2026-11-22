"""Preflight must not score or read capture contents before the immutable seal."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.revision_external_preflight import preflight_hulk, run_verified_hulk
from src.revision_followup_hulk import RESULT_FILES
from src.revision_v12 import _hash


class ExternalPreflightTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / "v12_hulk_attribution"
        self.manifest = self.root / "hulk_manifest.json"
        self.output = self.root / "external_result"
        self.capture = self.root / "prepared.csv"
        self.raw = self.root / "raw.capture"
        self.version = 7
        self.source_files = {"fixture_source": "a" * 64}

    def _declare(self):
        self.capture.write_bytes(b"x,label,group_id\n1,BENIGN,new/session-1\n")
        self.raw.write_bytes(b"original independent capture fixture")
        self.document = {
            "version": 1,
            "dataset": "CIC-IDS-2017",
            "target_class": "DoS Hulk",
            "attestation": {"custodian": "Test fixture custodian", "unseen_in_development": True},
            "exact_overlap_policy": "reject",
            "captures": [
                {
                    "capture_id": "fixture-acquisition",
                    "prepared_csv": "prepared.csv",
                    "sha256": _hash(self.capture),
                    "raw_files": [{"path": "raw.capture", "sha256": _hash(self.raw)}],
                }
            ],
            "criteria": {
                "min_target_f1_effect_pp": 1.0,
                "max_far": 0.01,
                "max_far_increase": 0.001,
                "max_fnr_increase": 0.005,
                "min_target_groups": 5,
                "min_benign_groups": 5,
                "min_seeds": 5,
                "min_captures": 1,
            },
        }
        self._write_manifest()
        self.heads = []
        for seed in range(42, 47):
            path = self.source / f"seed_{seed}" / "head_frozen.npz"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"frozen fixture")
            path.with_suffix(".json").write_text(
                json.dumps({"feature_order": ["x"], "class_order": ["BENIGN", "DoS Hulk"]}), encoding="utf-8"
            )
            self.heads.append(path)

    def _write_manifest(self):
        self.manifest.write_text(json.dumps(self.document), encoding="utf-8")

    def _plan(self, heads, manifest):
        from src.revision_external import _capture_files

        document = json.loads(manifest.read_text())
        return {
            "version": self.version,
            "dataset": "CIC-IDS-2017",
            "target_class": "DoS Hulk",
            "manifest": str(manifest),
            "manifest_sha256": _hash(manifest),
            "seeds": list(range(42, 47)),
            "cases": [{"file": str(path)} for path in heads],
            "captures": _capture_files(manifest, document["captures"], {"f" * 64}),
        }, []

    def _patch_sources(self):
        patches = (
            patch("src.revision_external_preflight._source_artifacts", return_value=self.source_files),
            patch("src.revision_followup_hulk._source_artifacts", return_value=self.source_files),
            patch("src.revision_external_preflight._frozen_hulk_heads", return_value=self.heads),
            patch("src.revision_followup_hulk._frozen_hulk_heads", return_value=self.heads),
            patch("src.revision_external._plan", side_effect=self._plan),
        )
        for item in patches:
            self.addCleanup(item.stop)
            item.start()

    def _confirm(self, seal, result):
        self.assertTrue(seal.is_file(), "Cannot score before sealing")
        result.mkdir()
        for name in RESULT_FILES - {"complete.json"}:
            (result / name).write_text("fixture result", encoding="utf-8")
        (result / "complete.json").write_text(
            json.dumps({
                "seal": str(seal),
                "seal_sha256": _hash(seal),
                "seeds": list(range(42, 47)),
                "captures": 1,
                "detector_training": False,
                "threshold_selection_on_test": False,
                "status": "criteria_not_met",
            }),
            encoding="utf-8",
        )

    def test_missing_manifest_and_example_never_read_source_or_start_confirmation(self):
        with (
            patch(
                "src.revision_external_preflight._source_artifacts",
                side_effect=AssertionError("Unexpected source read"),
            ),
            patch("src.revision_external_preflight.run_hulk") as run,
        ):
            self.assertEqual(run_verified_hulk(self.output, self.source, self.manifest)["status"], "pending")
            example = self.root / "hulk_manifest.example.json"
            example.write_text("{}", encoding="utf-8")
            self.assertEqual(preflight_hulk(self.source, example)["status"], "pending")
            run.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_unfilled_attestation_and_missing_data_are_pending_without_output(self):
        self._declare()
        self._patch_sources()
        self.document["attestation"]["unseen_in_development"] = False
        self._write_manifest()
        self.assertIn("incomplete", preflight_hulk(self.source, self.manifest)["reason"])
        self.document["attestation"]["unseen_in_development"] = True
        self._write_manifest()
        self.capture.unlink()
        with patch("src.revision_external_preflight.run_hulk") as run:
            state = run_verified_hulk(self.output, self.source, self.manifest)
            self.assertEqual(state["status"], "pending")
            self.assertEqual(state["missing_files"], [str(self.capture)])
            run.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_ready_preflight_reads_no_capture_contents(self):
        self._declare()
        self._patch_sources()
        original = Path.open

        def guarded_open(path, *args, **kwargs):
            if path in (self.capture, self.raw):
                raise AssertionError("Preflight cannot read capture contents")
            return original(path, *args, **kwargs)

        with patch.object(Path, "open", guarded_open):
            state = preflight_hulk(self.source, self.manifest)
        self.assertEqual(state["status"], "ready")
        self.assertEqual(state["expected_prepared_columns"], ["x", "label", "group_id"])
        self.assertFalse(state["capture_contents_read"])
        self.assertFalse(state["independence_verified"])
        self.assertFalse(self.output.exists())

    def test_too_few_declared_captures_are_pending_before_source_read(self):
        self._declare()
        self.document["criteria"]["min_captures"] = 2
        self._write_manifest()
        with patch("src.revision_external_preflight._source_artifacts") as source:
            state = preflight_hulk(self.source, self.manifest)
        self.assertEqual(state["status"], "pending")
        self.assertIn("below declared minimum", state["reason"])
        self.assertFalse(state["capture_contents_read"])
        source.assert_not_called()

    def test_version_and_reused_development_source_are_rejected(self):
        self._declare()
        self._patch_sources()
        self.version = 5
        self.assertIn("version-7", preflight_hulk(self.source, self.manifest)["reason"])
        self.version = 7
        self.document["captures"][0]["raw_files"][0]["sha256"] = "f" * 64
        self._write_manifest()
        self.assertIn("development source", preflight_hulk(self.source, self.manifest)["reason"])

    def test_manifest_and_capture_directory_paths_report_configuration_failure(self):
        self._declare()
        self._patch_sources()
        self.assertEqual(preflight_hulk(self.source, self.root)["status"], "failed")
        self.document["captures"][0]["prepared_csv"] = str(self.source)
        self._write_manifest()
        self.assertIn("not a file", preflight_hulk(self.source, self.manifest)["reason"])

    def test_scoring_starts_only_after_seal_and_completed_run_is_reused(self):
        self._declare()
        self._patch_sources()
        original = Path.open

        def guarded_open(path, *args, **kwargs):
            if path in (self.capture, self.raw) and not (self.output / "seal.json").is_file():
                raise AssertionError("Capture read before seal")
            return original(path, *args, **kwargs)

        with (
            patch("src.revision_external.confirm_external", side_effect=self._confirm) as confirm,
            patch.object(Path, "open", guarded_open),
        ):
            first = run_verified_hulk(self.output, self.source, self.manifest)
            self.assertEqual(first["status"], "completed")
            self.assertEqual(first["assessment"], "criteria_not_met")
            self.assertEqual(run_verified_hulk(self.output, self.source, self.manifest), first)
            confirm.assert_called_once()

    def test_unsealed_path_correction_archives_workflow_but_sealed_change_fails(self):
        self._declare()
        self._patch_sources()
        self.output.mkdir()
        old = {
            "schema_version": 1,
            "source": str(self.source),
            "manifest": str(self.root / "old_manifest.json"),
            "source_artifacts": self.source_files,
            "frozen_version": 7,
        }
        workflow = self.output / "workflow.json"
        workflow.write_text(json.dumps(old), encoding="utf-8")
        original = workflow.read_bytes()
        with patch("src.revision_external.confirm_external", side_effect=self._confirm) as confirm:
            result = run_verified_hulk(self.output, self.source, self.manifest)
            self.assertEqual(result["status"], "completed")
            archives = list(self.output.glob("workflow_before_manifest_override_*.json"))
            self.assertEqual(len(archives), 1)
            self.assertEqual(archives[0].read_bytes(), original)
            self.assertIn(archives[0].name, result["artifacts"])
            renamed = self.root / "renamed.json"
            renamed.write_bytes(self.manifest.read_bytes())
            seal_hash = _hash(self.output / "seal.json")
            failed = run_verified_hulk(self.output, self.source, renamed)
            self.assertEqual(failed["status"], "failed")
            self.assertIn("cannot change", failed["reason"])
            self.assertEqual(_hash(self.output / "seal.json"), seal_hash)
            confirm.assert_called_once()

    def test_invalid_existing_workflow_is_failed_and_preserved(self):
        self._declare()
        self._patch_sources()
        self.output.mkdir()
        workflow = self.output / "workflow.json"
        workflow.write_text("[]", encoding="utf-8")
        with patch("src.revision_external_preflight.run_hulk") as run:
            result = run_verified_hulk(self.output, self.source, self.manifest)
            self.assertEqual(result["status"], "failed")
            self.assertIn("JSON object", result["reason"])
            self.assertEqual(workflow.read_text(), "[]")
            run.assert_not_called()

    def test_modified_manifest_path_archive_is_not_accepted_on_resume(self):
        self._declare()
        self._patch_sources()
        self.output.mkdir()
        old = {
            "schema_version": 1,
            "source": str(self.source),
            "manifest": str(self.root / "old_manifest.json"),
            "source_artifacts": self.source_files,
            "frozen_version": 7,
        }
        (self.output / "workflow.json").write_text(json.dumps(old), encoding="utf-8")
        with patch("src.revision_external.confirm_external", side_effect=self._confirm) as confirm:
            self.assertEqual(
                run_verified_hulk(self.output, self.source, self.manifest)["status"], "completed"
            )
            archive = next(self.output.glob("workflow_before_manifest_override_*.json"))
            archive.write_text("{}", encoding="utf-8")
            result = run_verified_hulk(self.output, self.source, self.manifest)
            self.assertEqual(result["status"], "failed")
            self.assertIn("archive changed", result["reason"])
            confirm.assert_called_once()


if __name__ == "__main__":
    unittest.main()
