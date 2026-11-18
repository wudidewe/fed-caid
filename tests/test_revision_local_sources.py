"""Tests for independently sourced official and historical evidence."""

import json
import unittest
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from src.revision_local_sources import (
    BENCHMARK_DATASETS,
    FULL_CORE,
    SCALE_CAN_REQUIRED,
    SCALE_CAN_ROOT,
    SEED_SET,
    copy_legacy_evidence,
    copy_official_reports,
    discover_official_runs,
)
from src.revision_v12 import _hash, _read, _write


def _stage(run, name, **metadata):
    """Bind a small completed stage and deliberately omit training."""
    directory = run / name
    directory.mkdir(parents=True)
    table = directory / "bound_fixture.csv"
    table.write_text("metric,value\nf1,0.5\n", encoding="utf-8")
    return {
        "status": "completed",
        "output": str(directory),
        "artifacts": {table.name: _hash(table)},
        **metadata,
    }


def _sealed_run(run, workflow, *, synthetic=False, legacy=False):
    """Create minimal finalized metadata with the supported workflow schema."""
    if workflow == "full_review":
        stages = {name: _stage(run, name) for name in FULL_CORE}
        stages["benchmark_matrix"].update(
            execution_origin="current_fresh_training",
            test_evaluated=False,
            datasets=sorted(BENCHMARK_DATASETS),
            cases_per_dataset=dict.fromkeys(BENCHMARK_DATASETS, len(SEED_SET)),
        )
        for name in ("lan", "external"):
            stages[name] = {"status": "pending", "output": str(run / name)}
    else:
        stages = {
            "iov_diagnosis": _stage(
                run,
                "iov_diagnosis",
                execution_origin="current_fresh_training",
                test_evaluated=False,
                threshold_candidate_evaluated=False,
                synthetic_only=synthetic,
                seeds=sorted(SEED_SET),
            )
        }
    stages["report"] = _stage(run, "report")
    _seal(run, workflow, stages, synthetic=synthetic, legacy=legacy)


def _seal(run, workflow, stages, *, synthetic, legacy):
    """Write the same completion envelope as the real invocation."""
    plan, state = run / "final_plan.json", run / "final_state.json"
    details = dict(run_directory=str(run), previous_run_inputs_allowed=False)
    plan_data = dict(details, stages=list(stages), schema_version=5 if legacy else 6)
    if not legacy:
        plan_data.update(workflow=workflow, synthetic_only=synthetic)
    _write(plan, plan_data)
    _write(state, stages)
    _write(
        run / "final_complete.json",
        dict(
            details,
            status="completed_with_pending" if workflow == "full_review" else "completed",
            plan_sha256=_hash(plan),
            state_sha256=_hash(state),
            steps={name: {k: v for k, v in row.items() if k != "artifacts"} for name, row in stages.items()},
        ),
    )


def _interrupted_run(run):
    """Represent the native aborted stage marker and its plan hash."""
    run.mkdir(parents=True)
    _write(run / "final_plan.json", {"workflow": "iov_diagnosis"})
    _write(run / "final_complete.json", {"status": "aborted", "plan_sha256": _hash(run / "final_plan.json")})


def _old_scale_can(outputs):
    """Create an older report with a canonical source plan and artifact inventory."""
    source = outputs / "native.csv"
    source.write_text("setting,seconds\n3,7\n", encoding="utf-8")
    plan = {"source_hashes": {str(source): _hash(source)}, "missing_sources": []}
    digest = sha256(json.dumps(plan, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    root = outputs / SCALE_CAN_ROOT
    snapshot = root / f"report_{digest[:16]}"
    snapshot.mkdir(parents=True)
    _write(snapshot / "evidence_plan.json", plan)
    for name in SCALE_CAN_REQUIRED:
        (snapshot / name).write_text("quantity,value\n5,8\n", encoding="utf-8")
    hashes = {path.name: _hash(path) for path in snapshot.iterdir()}
    marker = snapshot / "complete.json"
    _write(marker, {"artifacts": hashes, "source_digest": digest, "status": "completed"})
    _write(
        root / "latest.json",
        {
            "report_path": str(snapshot),
            "receipt_sha256": _hash(marker),
            "artifacts": {**hashes, "complete.json": _hash(marker)},
        },
    )
    return source, snapshot


class LocalSourceTests(unittest.TestCase):
    def test_picks_only_newest_finalized_real_workflows(self):
        with TemporaryDirectory() as temporary:
            outputs = Path(temporary).resolve() / "outputs"
            parent = outputs / "fedcaid_review"
            full = parent / "run_20261116_002634_all"
            recent = parent / "run_20261116_100647_iov"
            _sealed_run(full, "full_review", legacy=True)
            _sealed_run(recent, "iov_diagnosis")
            _sealed_run(parent / "run_20261115_100647_iov", "iov_diagnosis")
            _sealed_run(parent / "run_20261117_100647_synthetic", "iov_diagnosis", synthetic=True)
            _interrupted_run(parent / "run_20261118_100647_aborted")
            (parent / "qa_launcher_test_run_20261118_100647_invalid").mkdir()
            self.assertEqual(discover_official_runs(outputs), {"full_review": full, "iov_diagnosis": recent})
            before = _hash(full / "final_complete.json")
            destination = Path(temporary).resolve() / "refresh"
            manifest = copy_official_reports(outputs, destination)
            self.assertEqual(manifest["full_review"]["trained_cases"], 0)
            self.assertEqual(manifest["iov_diagnosis"]["source_run"], recent.name)
            self.assertEqual(_hash(full / "final_complete.json"), before)
            self.assertTrue((destination / "full_review/refresh_source.json").is_file())

    def test_rejects_changed_native_input_instead_of_falling_back(self):
        with TemporaryDirectory() as temporary:
            outputs = Path(temporary).resolve()
            parent = outputs / "fedcaid_review"
            _sealed_run(parent / "run_20261116_100647_old", "iov_diagnosis")
            latest = parent / "run_20261118_100647_changed"
            _sealed_run(latest, "iov_diagnosis")
            (latest / "iov_diagnosis/bound_fixture.csv").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Current-run artifact changed"):
                discover_official_runs(outputs)

    def test_copies_legacy_only_when_source_inventory_still_verifies(self):
        with TemporaryDirectory() as temporary:
            outputs = Path(temporary).resolve() / "outputs"
            outputs.mkdir()
            native, snapshot = _old_scale_can(outputs)
            history = outputs.parent / "history"
            status = copy_legacy_evidence(outputs, history)
            self.assertEqual(status["scale_can"]["source_receipt_sha256"], _hash(snapshot / "complete.json"))
            self.assertEqual(status["nb15_temporal"]["status"], "unavailable")
            manifest = _read(history / "scale_can/archive_provenance.json")
            self.assertEqual(manifest["verified_native_artifact_sha256"], {"native.csv": _hash(native)})
            native.write_text("changed", encoding="utf-8")
            retry = outputs.parent / "retry"
            with self.assertRaisesRegex(ValueError, "Current-run artifact changed"):
                copy_legacy_evidence(outputs, retry)
            self.assertFalse(retry.exists())

    def test_missing_archives_are_explicit(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            self.assertEqual(discover_official_runs(root / "outputs"), {})
            legacy = copy_legacy_evidence(root / "outputs", root / "history")
            self.assertEqual({row["status"] for row in legacy.values()}, {"unavailable"})
            self.assertFalse((root / "history").exists())
