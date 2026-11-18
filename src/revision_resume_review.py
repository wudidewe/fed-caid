"""Continue the supplementary suffix of an interrupted full review in place."""

from pathlib import Path

from .revision_final_review import (
    CURRENT_REVIEW,
    FULL_WORKFLOW,
    SCHEMA_VERSION,
    STAGE_LABELS,
    _run_stage,
    _verify_source,
    _write_abort,
    _write_completion,
)
from .revision_fresh_report import _artifact_path, _stage_root, _verified_files, run_fresh_report
from .revision_v12 import _bindings, _hash, _output_lock, _read, _write


SUPPLEMENTARY_STAGES = ("can_quantization", "nb15_temporal", "kminus1_transfer", "report")
TERMINAL_MARKERS = {"completed", "completed_with_pending"}


def _run_path(project, name):
    """Resolve an explicit run name under the review output container."""
    if Path(name).name != name or not name.startswith("run_"):
        raise ValueError("Expected an explicit review run name")
    output = project / "outputs" / CURRENT_REVIEW / name
    if output.resolve(strict=True) != output or not output.is_dir():
        raise ValueError(f"Invalid review run directory: {output}")
    return output


def _read_plan(project, output, plan_path, marker):
    """Check the original run identity and all previously bound source files."""
    plan = _read(plan_path)
    if (
        plan["schema_version"] != SCHEMA_VERSION
        or marker["schema_version"] != SCHEMA_VERSION
        or marker["plan_sha256"] != _hash(plan_path)
        or Path(plan["project"]) != project
        or Path(plan["run_directory"]) != output
        or plan["workflow"] != FULL_WORKFLOW
        or plan["synthetic_only"] is not False
        or plan["previous_run_inputs_allowed"] is not False
        or plan["stages"] != list(STAGE_LABELS)
        or tuple(plan["stages"][-len(SUPPLEMENTARY_STAGES):]) != SUPPLEMENTARY_STAGES
    ):
        raise ValueError("The interrupted run differs from its declared full-review plan")
    _verify_source(plan)
    return plan


def _verify_stage(output, stage, record, optional):
    """Verify completed evidence or the original optional pending declaration."""
    root = _stage_root(output, stage, record)
    if record["status"] == "completed" and record.get("artifacts"):
        for _ in _verified_files(root, record["artifacts"]):
            pass
        return
    preflight = optional.get(stage, {})
    if preflight.get("status") != "pending" or not all(record.get(k) == v for k, v in preflight.items()):
        raise ValueError(f"Cannot preserve an unverified stage: {stage}")


def _read_state(output, state_path, marker, plan):
    """Require a contiguous completed prefix and verify each preserved stage."""
    state = _read(state_path)
    stages = plan["stages"]
    minimum = len(stages) - len(SUPPLEMENTARY_STAGES)
    if len(state) < minimum or set(state) != set(stages[:len(state)]):
        raise ValueError("Saved stages do not cover the completed prefix through the scale matrix")
    if marker["status"] in TERMINAL_MARKERS and (
        len(state) != len(stages) or marker.get("state_sha256") != _hash(state_path)
    ):
        raise ValueError("Finalized run state differs from its completion receipt")
    ordered = {name: state[name] for name in stages if name in state}
    for name, record in ordered.items():
        _verify_stage(output, name, record, plan["input_preflight"])
    return ordered


def _continuation_receipt(output, plan_path, state_path, state):
    """Bind the original plan, initial progress and the added continuation code."""
    receipt = _artifact_path(output, "continuation_plan.json")
    project = Path(__file__).resolve().parent.parent
    names = ("main.py", "src/revision_resume_review.py", "src/revision_resume_can.py")
    binding = {"original_plan_sha256": _hash(plan_path), "source_sha256": _bindings(project, names)}
    if receipt.exists():
        recorded = _read(receipt)
        if any(recorded.get(key) != value for key, value in binding.items()):
            raise ValueError("Continuation code or the original plan changed")
        for _ in _verified_files(output, recorded["preserved_case_sha256"]):
            pass
        return
    _write(
        receipt,
        {
            **binding,
            "initial_state_sha256": _hash(state_path),
            "preserved_stages": list(state),
            "preserved_case_sha256": _case_bindings(output, state),
        },
    )


def _case_bindings(output, state):
    """Bind atomic case files saved before the missing stage completed."""
    paths = (
        path
        for stage in SUPPLEMENTARY_STAGES[:-1]
        if stage not in state
        for path in (output / stage).rglob("result.json")
    )
    names = sorted(path.relative_to(output).as_posix() for path in paths)
    return {name: _hash(_artifact_path(output, name)) for name in names}


def _recover_report(output, state):
    """Reuse a verified report published before its stage record was saved."""
    path = _artifact_path(output, "complete.json")
    report = _read(path)
    _verify_stage(output.parent, "report", report, {})
    sources = {
        str(_stage_root(output.parent, stage, record) / name): digest
        for stage, record in state.items()
        for name, digest in record.get("artifacts", {}).items()
    }
    evidence = _read(_artifact_path(output, "evidence_plan.json"))
    if evidence["sources"] != sources or evidence["run_plan_sha256"] != _hash(output.parent / "final_plan.json"):
        raise ValueError("The published report does not match the current continuation")
    return {**report, "artifacts": {**report["artifacts"], path.name: _hash(path)}}


def _resume_device(output, plan):
    """Use the actual device recorded by the completed benchmark."""
    config = _read(_artifact_path(output / "benchmark_matrix", "run.json"))
    device = config["device"]
    if device not in {"cpu", "cuda"} or plan["requested_device"] not in {"auto", device}:
        raise ValueError("The benchmark device differs from the original run plan")
    return device


def _operations(project, state, device):
    """Use the current run's original supplement settings and reporting pipeline."""
    from .revision_resume_can import resume_can_quantization, resume_kminus1_transfer, resume_nb15_temporal

    data = project / "dataset"
    return {
        "can_quantization": lambda path: resume_can_quantization(path, data, device),
        "nb15_temporal": lambda path: resume_nb15_temporal(path, data, device),
        "kminus1_transfer": lambda path: resume_kminus1_transfer(path, data, device),
        "report": lambda path: _recover_report(path, state) if path.exists() else run_fresh_report(path, state),
    }


def _run_remaining(output, state_path, plan, state, operations):
    """Execute only missing stages and atomically commit each completed record."""
    for name in SUPPLEMENTARY_STAGES:
        if name in state:
            continue
        _verify_source(plan)
        record = _run_stage(name, output / name, operations[name])
        if record["status"] != "completed":
            raise RuntimeError(f"{STAGE_LABELS[name]}: {record.get('reason', record['status'])}")
        _verify_stage(output, name, record, {})
        state[name] = record
        _write(state_path, state)
    _verify_source(plan)


def _continue_locked(project, output):
    """Bind a stopped run and finish its supplementary stages and report."""
    plan_path, state_path, marker_path = (
        _artifact_path(output, name)
        for name in ("final_plan.json", "final_state.json", "final_complete.json")
    )
    marker = _read(marker_path)
    plan = _read_plan(project, output, plan_path, marker)
    state = _read_state(output, state_path, marker, plan)
    if marker["status"] in TERMINAL_MARKERS:
        print(f"FedCAID | 本轮已完成 | {output / 'report' / 'SUMMARY.md'}", flush=True)
        return 0
    if marker["status"] not in {"running", "aborted"}:
        raise ValueError(f"Cannot continue a run with status {marker['status']}")
    _continuation_receipt(output, plan_path, state_path, state)
    return _execute_remaining(project, output, marker_path, plan_path, state_path, plan, state)


def _execute_remaining(project, output, marker_path, plan_path, state_path, plan, state):
    """Record an interruption as resumable and publish the normal final receipt."""
    _write(marker_path, {"schema_version": SCHEMA_VERSION, "status": "running", "plan_sha256": _hash(plan_path)})
    print(f"FedCAID | 补跑目录 | {output}", flush=True)
    print(f"FedCAID | 已校验并保留的阶段 | {'、'.join(STAGE_LABELS[name] for name in state)}", flush=True)
    try:
        operations = _operations(project, state, _resume_device(output, plan))
        _run_remaining(output, state_path, plan, state, operations)
    except (KeyboardInterrupt, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        _write_abort(marker_path, plan_path, error)
        raise
    return _write_completion(marker_path, plan_path, state_path, state)


def resume_review(project, name):
    """Resume the designated full review without loading another run's evidence."""
    project = Path(project).resolve(strict=True)
    output = _run_path(project, name)
    with _output_lock(output):
        return _continue_locked(project, output)
