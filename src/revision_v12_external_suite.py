"""Verify sealed external capture inputs and collected evidence files."""

from pathlib import Path

from .revision_v12 import _bindings, _hash, _read


RESULT_FILES = frozenset({
    "complete.json",
    "metrics.csv",
    "per_class.csv",
    "paired_by_seed.csv",
    "paired_intervals.csv",
    "capture_support.csv",
    "capture_metric_summaries.csv",
    "capture_intervals.csv",
    "confusion_counts.json",
    "fixed_alarm_transitions.csv",
})
VALID_ASSESSMENTS = frozenset({
    "criteria_met_on_declared_external_receiver",
    "criteria_not_met",
    "insufficient_support_or_gate",
})


def _verify_capture_hashes(plan):
    """Check that every prepared and raw capture matches the sealed hash."""
    for capture in plan["captures"]:
        files = [(capture["prepared_csv"], capture["sha256"]), *capture["raw_files"]]
        for path, expected in files:
            if _hash(Path(path)) != expected:
                raise ValueError(f"Independent capture file differs from its original seal: {path}")


def _result_inventory(result):
    """Validate inventory completeness and containment before hashing files."""
    if not result.is_dir() or not (result / "complete.json").is_file():
        raise ValueError(
            f"External result is missing or incomplete at {result}; existing outputs were preserved"
        )
    inventory = {path.name for path in result.iterdir() if path.is_file()}
    if not RESULT_FILES <= inventory:
        raise ValueError(f"External result at {result} is incomplete; existing outputs were preserved")
    if any((result / name).resolve() != result / name for name in inventory):
        raise ValueError(f"External result artifact leaves its managed path: {result}")
    return sorted(inventory)


def _validate_result_marker(marker, seal, plan, seal_sha256):
    """Require prospective seal identity and a valid external assessment."""
    if (
        marker.get("seal_sha256") != seal_sha256
        or Path(marker.get("seal", "")).resolve() != seal
        or marker.get("seeds") != plan["seeds"]
        or marker.get("captures") != len(plan["captures"])
        or marker.get("detector_training") is not False
        or marker.get("threshold_selection_on_test") is not False
        or marker.get("status") not in VALID_ASSESSMENTS
    ):
        raise ValueError("External result does not match the prospective Hulk seal")


def _result_bindings(result, seal, plan, seal_sha256):
    """Return verified artifact hashes and the recorded assessment."""
    inventory = _result_inventory(result)
    marker = _read(result / "complete.json")
    _validate_result_marker(marker, seal, plan, seal_sha256)
    return _bindings(result, inventory), marker
