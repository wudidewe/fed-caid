"""Locate the five evaluated Hulk frozen heads from a development receipt."""

import csv
import json
from pathlib import Path


HULK_SEEDS = (42, 43, 44, 45, 46)
HULK_CASE = ("CIC-IDS-2017", "train_absence", "DoS Hulk", "reserved", "evaluated")
CASE_FIELDS = ("dataset", "scenario", "target_class", "cal_policy", "status")


def _json(path):
    """Read an object from a saved JSON receipt."""
    try:
        with path.open(encoding="utf-8-sig") as stream:
            value = json.load(stream)
    except (OSError, ValueError) as error:
        raise ValueError(f"Could not read the saved JSON receipt: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return value


def _bundle_root(root):
    """Resolve a wrapper to its native attribution run, when present."""
    root = Path(root)
    native = root / "attribution"
    if (root / "case_status.csv").is_file() or not (native / "case_status.csv").is_file():
        return root
    resolved = native.resolve()
    if resolved.parent != root:
        raise ValueError(f"The attribution stage leaves the selected run directory: {root}")
    return resolved


def _completed_hulk_run(root):
    """Require a completed real development evaluation."""
    marker = _json(root / "complete.json")
    if (
        marker.get("task") != "train-absence"
        or marker.get("mode") != "evaluation"
        or marker.get("smoke") is not False
        or marker.get("head_only") is True
        or marker.get("test_evaluated")
    ):
        raise ValueError(f"Choose a completed real Hulk development run: {root}")
    return marker


def _hulk_case_rows(inventory, marker):
    """Bind the full case inventory to the completion counts."""
    if not inventory.is_file():
        raise ValueError(f"Case status file was not found: {inventory}")
    with inventory.open(encoding="utf-8-sig", newline="") as stream:
        cases = list(csv.DictReader(stream))
    evaluated = sum(row.get("status") == "evaluated" for row in cases)
    if len(cases) != marker.get("planned_case_policies") or evaluated != marker.get("evaluated"):
        raise ValueError(f"The case inventory does not match its completion receipt: {inventory.parent}")
    return [row for row in cases if tuple(row.get(field) for field in CASE_FIELDS) == HULK_CASE]


def _hulk_seed_index(inventory, marker):
    """Require exactly one selected evaluated case for every expected seed."""
    indexed = {}
    for row in _hulk_case_rows(inventory, marker):
        try:
            seed = int(row["seed"])
        except (KeyError, ValueError) as error:
            raise ValueError(f"Invalid seed in Hulk case inventory: {inventory}") from error
        if seed in indexed:
            raise ValueError(f"Hulk seed {seed} appears twice in the case inventory.")
        indexed[seed] = row
    if set(indexed) != set(HULK_SEEDS):
        raise ValueError("The run must contain the five evaluated reserved Hulk cases for seeds 42–46.")
    return indexed


def _check_head_details(head, seed, row):
    """Check the frozen metadata against the evaluated sample and receiver."""
    info = _json(head.with_suffix(".json"))
    thresholds = info.get("thresholds")
    receiver = info.get("receiver")
    if (
        type(info.get("seed")) is not int
        or info["seed"] != seed
        or info.get("dataset") != HULK_CASE[0]
        or info.get("target_class") != HULK_CASE[2]
        or type(receiver) is not int
        or str(receiver) != row.get("receiver")
        or not isinstance(thresholds, dict)
        or not {"head_local", "head_federated"} <= thresholds.keys()
    ):
        raise ValueError(f"Frozen receipt does not match the evaluated Hulk case: {head}")


def _frozen_head_path(root, row, seed):
    """Resolve the expected NPZ bundle without permitting a path escape."""
    name = row.get("frozen_head")
    if not name:
        raise ValueError(f"No frozen head was recorded for Hulk seed {seed}.")
    head = (root / name).resolve()
    if not head.is_relative_to(root) or head.suffix.lower() != ".npz" or not head.is_file():
        raise ValueError(f"The recorded Hulk seed {seed} bundle is missing or leaves its run directory: {head}")
    _check_head_details(head, seed, row)
    return head


def _frozen_hulk_heads(root):
    """Return five distinct frozen NPZ paths in seed order."""
    root = _bundle_root(root)
    indexed = _hulk_seed_index(root / "case_status.csv", _completed_hulk_run(root))
    heads = [_frozen_head_path(root, indexed[seed], seed) for seed in HULK_SEEDS]
    if len(set(heads)) != len(HULK_SEEDS):
        raise ValueError("Hulk frozen head paths must be distinct.")
    return heads
