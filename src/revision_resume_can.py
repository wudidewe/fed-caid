"""Continue interrupted native supplements inside their original run directory."""

from pathlib import Path

from . import revision_full_matrix as matrix
from . import revision_run as native
from .revision_v12 import _bindings, _hash, _read


def _resume_paths(output, data_root):
    """Keep partial outputs and their children inside the declared stage."""
    output, data_root = Path(output).absolute(), Path(data_root).resolve()
    if output.resolve() != output or output.is_relative_to(data_root) or data_root.is_relative_to(output):
        raise ValueError("Supplementary output must be an unredirected path separate from the raw dataset")
    if any(path.resolve() != path for path in output.rglob("*")):
        raise ValueError(f"Redirected supplementary artifact in {output}")
    if (output / matrix.STAGE_RECEIPT).exists():
        _verify_published_bindings(output)
    return output, data_root


def _verify_published_bindings(output):
    """Reject changes to published evidence before native steps can run."""
    bindings = _read(output / matrix.STAGE_RECEIPT)["artifacts"]
    for location, digest in bindings.items():
        path = output / location
        if path.resolve() != path or not path.is_relative_to(output) or not path.is_file() or _hash(path) != digest:
            raise ValueError(f"Existing supplementary evidence changed: {path}")


def _case_paths(output, phase, cases):
    """Map each declared case to the atomic native result location."""
    return {
        output / case[0] / phase.protocol / case[1] / f"seed_{case[-1]}" / "result.json": case
        for case in cases
    }


def _validate_case(path, phase, case, device, smoke):
    """Check the identity, settings and completeness of an atomic case result."""
    record = _read(path)
    if record["protocol"] != phase.protocol or matrix._case_key(record) != case:
        raise ValueError(f"Saved supplementary case identity differs at {path}")
    settings = record["settings"]
    if settings["device"] != device or record.get("prediction_cache") is not None:
        raise ValueError(f"Saved supplementary case used different runtime or cache inputs: {path}")
    matrix._validate_supplement_case(settings, path.parents[2], phase, smoke)
    required = {(role, mode) for role in matrix.TEST_ROLES for mode in phase.modes}
    evaluations = [(row["role"], row["mode"]) for row in record["evaluations"]]
    if (
        not required.issubset(evaluations)
        or len(evaluations) != len(set(evaluations))
        or any(role not in matrix.TEST_ROLES for role, _ in evaluations)
    ):
        raise ValueError(f"Saved supplementary case has incomplete or duplicate evaluations: {path}")


def _verify_partial_phase(output, data_root, device, seeds, smoke, phase, cases):
    """Reject wrong or incomplete saved case files before entering native resume."""
    saved_plan = _read(output / "run.json")
    expected_matrix = [
        {"dataset": name, "protocol": phase.protocol, "scenarios": [[phase.scenario, 3, phase.alpha]]}
        for name in phase.datasets
    ]
    coordinates = (saved_plan["matrix"], saved_plan["seeds"], saved_plan["synthetic_only"], saved_plan["data_root"])
    saved_device = saved_plan["device"]
    if (
        coordinates != (expected_matrix, list(seeds), smoke, None if smoke else str(data_root))
        or saved_plan["supplement"]["profile"] != phase.profile
        or saved_plan["source_sha256"] != native._source_hashes(supplement=True)
        or saved_device not in {"cpu", "cuda"}
        or device not in {"auto", saved_device}
    ):
        raise ValueError(f"Saved supplementary plan differs from this run: {output}")
    _verify_saved_cases(output, phase, cases, saved_device, smoke)
    return saved_device


def _verify_saved_cases(output, phase, cases, device, smoke):
    """Validate the declared file set and require a source audit for every result."""
    paths = _case_paths(output, phase, cases)
    for path in output.rglob("result.json"):
        if path not in paths:
            raise ValueError(f"Saved supplementary case is outside the declared workload: {path}")
        if not (path.parents[2] / "data_audit.json").is_file():
            raise ValueError(f"Saved supplementary case lacks its original dataset audit: {path}")
        _validate_case(path, phase, paths[path], device, smoke)


def _resume_native(output, data_root, device, seeds, smoke, phase):
    """Use the native runner's exact-plan and dataset-audit resume checks."""
    arguments = [
        "--profile",
        phase.profile,
        "--dataset",
        *phase.datasets,
        "--no-tcp",
        "--no-figures",
        "--resume",
        *phase.native_flags,
        *matrix._common_arguments(output, data_root, device, seeds, smoke),
    ]
    if native.main(arguments, supplement=True) != 0:
        raise RuntimeError(f"Native supplement could not resume: {phase.profile}")


def _clear_uncommitted_phase(output):
    """Release only an empty phase that stopped before publishing its plan."""
    partial = output / "run.partial"
    entries = list(output.iterdir())
    if entries:
        if entries != [partial] or not partial.is_file() or partial.resolve() != partial:
            raise ValueError(f"Native phase has evidence but no committed run.json: {output}")
        partial.unlink()
    output.rmdir()


def _resume_phase(output, data_root, device, seeds, smoke, phase):
    """Train missing cases and bind the same evidence as the fresh runner."""
    if output.is_dir() and not (output / "run.json").exists():
        _clear_uncommitted_phase(output)
    if not output.exists():
        return matrix._run_supplement(output, data_root, device, seeds, smoke, phase)
    cases = matrix._supplement_cases(phase, seeds)
    device = _verify_partial_phase(output, data_root, device, seeds, smoke, phase, cases)
    if not (output / "complete.json").exists():
        _resume_native(output, data_root, device, seeds, smoke, phase)
    matrix._validate_supplement(output, data_root, phase, seeds, cases, smoke)
    matrix._checked_metrics(
        output, "metrics_by_seed.csv", cases, phase.modes, matrix.TEST_ROLES, protocol=phase.protocol
    )
    return matrix._phase_artifacts(output, phase, cases, smoke)


def _publish_resume(output, names, details):
    """Publish a new stage receipt or preserve an already finished, verified one."""
    marker = output / matrix.STAGE_RECEIPT
    if not marker.exists():
        return matrix._publish(output, names, details)
    receipt = _read(marker)
    expected = {
        **details,
        "datasets": list(details["datasets"]),
        "seeds": list(details["seeds"]),
        "status": "completed",
        "output": str(output),
        "execution_origin": "current_fresh_training",
        "independent_confirmation": False,
        "artifacts": _bindings(output, names),
    }
    if receipt != expected:
        raise ValueError(f"Existing supplementary stage receipt differs from its artifacts: {marker}")
    return {**receipt, "artifacts": {**receipt["artifacts"], matrix.STAGE_RECEIPT: _hash(marker)}}


def resume_can_quantization(output, data_root, device="auto", *, smoke=False):
    """Resume the grouped and frozen quantization phases, reusing only this run's cases."""
    output, data_root = _resume_paths(output, data_root)
    seeds = native.SEEDS[:1] if smoke else native.SEEDS
    names = []
    for phase in matrix.CAN_PHASES:
        phase_output = output / phase.protocol
        names.extend(
            f"{phase.protocol}/{name}" for name in _resume_phase(phase_output, data_root, device, seeds, smoke, phase)
        )
    details = {
        "datasets": matrix.CAN_DATASETS,
        "protocols": [phase.protocol for phase in matrix.CAN_PHASES],
        "seeds": seeds,
        "cases": len(matrix.CAN_PHASES) * len(matrix.CAN_DATASETS) * len(seeds),
        "synthetic_only": smoke,
        "test_evaluated": True,
        "scientific_status": "same_source_kminus1_quantization_diagnostic",
        "secure_scales": [10**6, 10**9],
        "evaluation_scope": "Retrospective grouped and frozen CAN VAL and benchmark TEST, original K−1 adapters",
    }
    return _publish_resume(output, names, details)


def resume_nb15_temporal(output, data_root, device="auto", *, smoke=False):
    """Continue the exact-purged NB15 later-day holdout."""
    output, data_root = _resume_paths(output, data_root)
    seeds = native.SEEDS[:1] if smoke else native.SEEDS
    phase = matrix.NB15_TEMPORAL
    names = _resume_phase(output, data_root, device, seeds, smoke, phase)
    details = {
        "datasets": phase.datasets,
        "protocols": ["temporal_purged"],
        "seeds": seeds,
        "cases": len(seeds),
        "synthetic_only": smoke,
        "test_evaluated": True,
        "scientific_status": "same_source_exact_purged_temporal_test",
        "evaluation_scope": "Retrospective later-day TEST within the same pinned NB15 capture",
    }
    return _publish_resume(output, names, details)


def resume_kminus1_transfer(output, data_root, device="auto", *, smoke=False):
    """Continue the full-TRAIN original K−1 transfer score diagnostics."""
    output, data_root = _resume_paths(output, data_root)
    seeds = native.SEEDS[:1] if smoke else native.SEEDS
    phase = matrix.KMINUS1_TRANSFER
    names = _resume_phase(output, data_root, device, seeds, smoke, phase)
    scores = "transfer_score_deltas.csv"
    details = {
        "datasets": phase.datasets,
        "protocols": [phase.protocol],
        "seeds": seeds,
        "alpha": phase.alpha,
        "cases": len(phase.datasets) * len(seeds),
        "synthetic_only": smoke,
        "test_evaluated": True,
        "eligible_score_contrast_rows": len(list(matrix._rows(output, scores))) if scores in names else 0,
        "full_train": True,
        "scientific_status": "same_source_kminus1_transfer_diagnostic",
        "evaluation_scope": "Retrospective exact-unseen benchmark TEST; original K−1 missing-class transfer strata",
    }
    return _publish_resume(output, names, details)
