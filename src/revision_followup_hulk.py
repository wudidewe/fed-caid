"""Seal, confirm, and verify the frozen V7 Hulk attribution run."""

from pathlib import Path

from . import revision_external
from . import revision_v12_launch_plan
from .revision_v12 import _hash, _output_lock, _read, _write
from .revision_v12_external_suite import _result_bindings, _verify_capture_hashes


SOURCE_RECEIPTS = ("v12_complete.json", "v12_plan.json", "v12_state.json")
CONFIRMATION_PATHS = ("seal.json", "result", "receipt.json", "workflow.json")
STAGE_RECEIPTS = {"run.json", "complete.json", "case_status.csv"}
NESTED_STAGE_RECEIPTS = {"CIC-IDS-2017/data_audit.json"}
HULK_SEEDS = [42, 43, 44, 45, 46]
FROZEN_VERSION = 7
WORKFLOW_ERRORS = (KeyError, OSError, RuntimeError, TypeError, ValueError)


def _validate_source_receipts(marker, plan, state, files):
    if not all(isinstance(value, dict) for value in (marker, plan, state)):
        raise ValueError("V12 attribution receipts must be JSON objects")
    identity = (marker.get("mode"), marker.get("scope"), marker.get("status"))
    if (
        identity != ("attribution", "development", "completed")
        or marker.get("test_evaluated") is not False
        or (plan.get("mode"), plan.get("scope")) != identity[:2]
        or set(state) != {"attribution"}
        or marker.get("plan_sha256") != files["v12_plan.json"]
        or marker.get("state_sha256") != files["v12_state.json"]
    ):
        raise ValueError("Expected an intact completed V12 attribution development wrapper")


def _archived_attribution_files(stage, files):
    if stage.resolve() != stage or not isinstance(files, dict) or not STAGE_RECEIPTS <= files.keys():
        raise ValueError("Attribution stage is redirected or its receipt is incomplete")
    for name, expected in files.items():
        path = stage / name
        allowed = Path(name).name == name or name in NESTED_STAGE_RECEIPTS
        if not allowed or path.resolve() != path or not path.is_file() or _hash(path) != expected:
            raise ValueError(f"Completed attribution source artifact changed: {name}")
    return {f"attribution/{name}": expected for name, expected in files.items()}


def _source_artifacts(source):
    """Verify the completed development wrapper without changing its receipts."""
    marker, plan, state = (_read(source / name) for name in SOURCE_RECEIPTS)
    files = {name: _hash(source / name) for name in SOURCE_RECEIPTS}
    _validate_source_receipts(marker, plan, state, files)
    files.update(_archived_attribution_files(source / "attribution", state["attribution"]))
    return files


def _attribution_plan(heads, manifest):
    plan, _ = revision_external._plan(heads, manifest)
    identity = (plan.get("version"), plan.get("dataset"), plan.get("target_class"))
    if identity != (FROZEN_VERSION, "CIC-IDS-2017", "DoS Hulk") or plan.get("seeds") != HULK_SEEDS:
        raise ValueError("Independent attribution requires the five version-7 Hulk bundles for seeds 42-46")
    return plan


def _validate_output(output, source, manifest):
    if output.is_relative_to(source) or source.is_relative_to(output) or manifest.is_relative_to(output):
        raise ValueError("Follow-up output must be disjoint from the frozen source and external manifest")


def _check_capture_paths(output, plan):
    for capture in plan["captures"]:
        paths = [capture["prepared_csv"], *(path for path, _ in capture["raw_files"])]
        if any(Path(path).resolve().is_relative_to(output) for path in paths):
            raise ValueError("Capture inputs must be outside the follow-up output directory")


def _workflow_paths(output):
    paths = {Path(name).stem: output / name for name in CONFIRMATION_PATHS}
    for path in (*paths.values(), output / ".v12_lock"):
        if path.resolve() != path:
            raise ValueError(f"Follow-up artifact is redirected: {path}")
    if (
        output.exists()
        and not paths["workflow"].exists()
        and any(p.name != ".v12_lock" for p in output.iterdir())
    ):
        raise ValueError("Follow-up needs a new output directory; existing outputs were preserved")
    output.mkdir(parents=True, exist_ok=True)
    return paths


def _write_workflow(workflow, expected):
    if workflow.exists():
        if _read(workflow) != expected:
            raise ValueError("Follow-up source or configuration changed; existing outputs were preserved")
    elif any(path.name != ".v12_lock" for path in workflow.parent.iterdir()):
        raise ValueError(
            "Follow-up artifacts have no original workflow receipt; existing outputs were preserved"
        )
    else:
        _write(workflow, expected)


def _ensure_seal(paths, heads, manifest, plan):
    seal = paths["seal"]
    if seal.exists():
        if _read(seal) != plan:
            raise ValueError(
                "Existing attribution seal differs from frozen models, manifest or analysis code"
            )
    elif paths["result"].exists() or paths["receipt"].exists():
        raise ValueError("Follow-up results have no original seal; existing outputs were preserved")
    elif revision_external.seal_confirmation(heads, manifest, seal) != plan:
        raise ValueError("Attribution inputs changed while creating the seal")


def _confirmed_files(paths, plan, bindings):
    result, seal, receipt = (paths[name] for name in ("result", "seal", "receipt"))
    if not result.exists() and not receipt.exists():
        revision_external.confirm_external(seal, result)
    files, marker = _result_bindings(result, seal, plan, bindings["seal_sha256"])
    if receipt.exists() and _read(receipt) != {"schema_version": 1, **bindings, "files": files}:
        raise ValueError("Completed attribution confirmation receipt or artifacts changed")
    return files, marker


def _check_unchanged(source, manifest, source_files, heads, plan, paths, bindings):
    _verify_capture_hashes(plan)
    if (
        _source_artifacts(source) != source_files
        or _attribution_plan(heads, manifest) != plan
        or _hash(paths["seal"]) != bindings["seal_sha256"]
        or _hash(paths["workflow"]) != bindings["workflow_sha256"]
    ):
        raise ValueError("Attribution inputs or receipts changed during confirmation")


def _confirmation_details(marker):
    """Expose the verified result's evidence scope to the single-entry report."""
    return {
        "capture_count": marker["captures"],
        "seed_count": len(marker["seeds"]),
        "criteria_met": marker["status"] == "criteria_met_on_declared_external_receiver",
        "detector_training": marker["detector_training"],
        "threshold_selection_on_test": marker["threshold_selection_on_test"],
        "seed_interval_scope": marker.get("interval_scope"),
        "capture_interval_scope": marker.get("capture_interval_scope"),
        "capture_evidence": marker.get("capture_evidence"),
    }


def _save_confirmation(paths, bindings, files, marker):
    receipt = paths["receipt"]
    if not receipt.exists():
        _write(receipt, {"schema_version": 1, **bindings, "files": files})
    artifacts = {
        "seal.json": bindings["seal_sha256"],
        "workflow.json": bindings["workflow_sha256"],
        "receipt.json": _hash(receipt),
        **{f"result/{name}": digest for name, digest in files.items()},
    }
    return {
        "status": "completed",
        "reason": "",
        "result_path": str(paths["result"]),
        "seal": str(paths["seal"]),
        "assessment": marker["status"],
        "scoring": True,
        "capture_contents_read": True,
        "confirmation": _confirmation_details(marker),
        "artifacts": artifacts,
    }


def _run_confirmation(output, source, manifest):
    _validate_output(output, source, manifest)
    source_files = _source_artifacts(source)
    heads = revision_v12_launch_plan._frozen_hulk_heads(source)
    plan = _attribution_plan(heads, manifest)
    _check_capture_paths(output, plan)
    paths = _workflow_paths(output)
    expected = {
        "schema_version": 1,
        "source": str(source),
        "manifest": str(manifest),
        "source_artifacts": source_files,
        "frozen_version": FROZEN_VERSION,
    }
    with _output_lock(output):
        _write_workflow(paths["workflow"], expected)
        _ensure_seal(paths, heads, manifest, plan)
        bindings = {
            "seal_sha256": _hash(paths["seal"]),
            "manifest_sha256": plan["manifest_sha256"],
            "workflow_sha256": _hash(paths["workflow"]),
        }
        files, marker = _confirmed_files(paths, plan, bindings)
        _check_unchanged(source, manifest, source_files, heads, plan, paths, bindings)
        return _save_confirmation(paths, bindings, files, marker)


def run_hulk(output: Path, source: Path, manifest: Path) -> dict:
    """Seal and confirm once, then verify the matching artifacts on subsequent runs."""
    output, source, manifest = (Path(path).resolve() for path in (output, source, manifest))
    identity = {
        "output": str(output),
        "source": str(source),
        "required_manifest": str(manifest),
        "frozen_version": FROZEN_VERSION,
    }
    if not manifest.is_file():
        return {
            **identity,
            "status": "pending",
            "reason": "Independent capture manifest is missing",
            "artifacts": {},
        }
    try:
        return {**identity, **_run_confirmation(output, source, manifest)}
    except WORKFLOW_ERRORS as error:
        return {**identity, "status": "failed", "reason": str(error), "artifacts": {}}
