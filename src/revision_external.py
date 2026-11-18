"""Seal and score a frozen receiver pipeline on separately acquired, schema-matched traffic."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from hashlib import file_digest, sha256
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd

from . import fedcaid_resources
from .revision_data import _feature_keys
from .revision_report import score, summarize, write_csv
from .revision_shared_head import MODEL_SCHEMA, head_probabilities
from .revision_subtype_guard import GUARDED_ROUTING, guarded_subtype_predictions, validate_subtype_guard
from .utils import hierarchical_decisions

PRIMARY_MODES = ("head_local", "head_federated")
DONOR_MODE = "head_donor"
SINGLE_DONOR_MODE = "head_donor_single"
SUPPORTED_MODES = (*PRIMARY_MODES, DONOR_MODE)
ATTRIBUTION_VERSION = 7
DONOR_POOL_CONTRAST = "donor_pool_vs_single"
DONOR_FULL_CONTRAST = "federated_vs_single"
BASE_MODE = "base_threshold"
OPERATING_REFERENCE_MODE = "base_reference_1pct"
OPERATING_CONTRAST = "operating_fnr_vs_reference"
REFERENCE_SUBTYPE_MODE = "guarded_federated_reference_1pct"
REFERENCE_SUBTYPE_CONTRAST = "subtype_nonregression_vs_reference"
FIXED_ROUTING = "missing_attack_top1_v1"
FIXED_VERSIONS = (2, 3, 4, 5, 6, ATTRIBUTION_VERSION)
GUARDED_VERSIONS = (3, 4, 5, 6, ATTRIBUTION_VERSION)
OPERATING_VERSIONS = (4, 5, 6, ATTRIBUTION_VERSION)
SELECTION_VERSIONS = (5, 6, ATTRIBUTION_VERSION)
OPERATING_FIELDS = {"reference_threshold", "max_operating_fnr_increase"}
OPERATING_SELECTION_SCHEMA = "cal_operating_selection_v1"
OPERATING_REFERENCE_CAL_FAR = 0.01
CLASS_ALARM_ROUTING = "local_tabm_attack_top1_v1"
CLASS_SELECTION_FIELDS = {"class_thresholds", "class_routing"}
OPERATING_SELECTION_FIELDS = {
    "schema",
    "status",
    "reason",
    "candidate_grid",
    "constraints",
    "selected_cal_far",
    "reference_cal_far",
    "receiver_client",
    "receiver_threshold",
    "receiver_reference_threshold",
    "thresholds",
    "reference_thresholds",
    "cal_group_assignment_sha256",
    "group_overlap",
    "selection_sha256",
    "support",
    "selection_rule",
}
FIXED_CRITERIA = {"min_target_f1_over_base_pp": 1.0, "max_macro_f1_loss_pp": 1.0}
ABSOLUTE_TARGET_CRITERIA = {
    "target_f1": "min_absolute_target_f1",
    "target_precision": "min_absolute_target_precision",
}
CAPTURE_GROUP_CRITERIA = {
    "target_groups": "min_target_groups_per_capture",
    "benign_groups": "min_benign_groups_per_capture",
}
CAPTURE_F1_LOWER_CRITERION = "min_capture_target_f1_ci95_lower_pp"
CAPTURE_METRICS = ("target_f1", "target_precision", "target_recall", "macro_f1", "far", "fnr")
DEV_ROLES = ("train", "cal", "val")
FINGERPRINT = "ordered-canonical-float32-fnv64-v1"
SHARED_HEAD_SOURCE = "revision_shared_head.py"
# The V12 donor-training edit leaves the three scoring functions and their
# dependencies unchanged. Only the exact reviewed source pair is compatible.
_COMPATIBLE_SHARED_HEAD_SOURCE = {
    "a4de88679030aedeb0d565b56f1d8613bf33d44600b5a5a8ff54b570f0b518a8": (
        "9faba2ca0b1ef0f358346ce1d72de1cec19595c8f419092dee5416409f7b2ed0"
    ),
}
CRITERIA = {
    "min_target_f1_effect_pp": 1.0,
    "max_far": 0.01,
    "max_far_increase": 0.001,
    "max_fnr_increase": 0.005,
    "min_target_groups": 5,
    "min_benign_groups": 5,
    "min_seeds": 5,
    "min_captures": 1,
}


def _sha(path):
    with Path(path).open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def _digest(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", value):
        raise ValueError("Expected a 64-digit SHA256")
    return value.lower()


def _source_digests(items):
    if not isinstance(items, dict) or not items:
        raise ValueError("Provide development source names mapped to original SHA256 checksums")
    return {name: _digest(value) for name, value in items.items()}


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as dest:
        json.dump(value, dest, indent=2, allow_nan=False)
        dest.write("\n")


def _fingerprints(x):
    keys = _feature_keys(x).view("<u4").reshape(len(x), x.shape[1])
    result = np.full(len(x), 14695981039346656037, dtype=np.uint64)
    # Screening is conservative. Collisions can flag extra overlap only.
    for column in keys.T:
        result ^= column
        result *= np.uint64(1099511628211)
    return result


def _group_ids(ids):
    groups = tuple(ids)
    if any(not isinstance(x, str) or not x or x.strip() != x for x in groups):
        raise ValueError("Expected nonempty, whitespace-normalized group identifiers")
    return groups


def prepare_development_screen(
    path, development_parts, development_source_hashes, *, development_group_ids=()
):
    """Create a reusable, private overlap index of canonical TRAIN/CAL/VAL inputs."""
    path = Path(path).resolve()
    if path.suffix != ".npy":
        raise ValueError("Development screen must use a .npy path")
    receipt_path = path.with_suffix(".json")
    if path.exists() or receipt_path.exists():
        raise ValueError("Development screen requires a new .npy path")
    hashes = _source_digests(development_source_hashes)
    groups = sorted(set(_group_ids(development_group_ids)))
    matrices = [np.asarray(development_parts[role][0]) for role in DEV_ROLES]
    if any(x.ndim != 2 or x.shape[1] != matrices[0].shape[1] for x in matrices):
        raise ValueError("Canonical development feature widths differ between roles")
    width = matrices[0].shape[1]
    if width < 1 or not all(len(x) for x in matrices):
        raise ValueError("Development screen requires nonempty TRAIN, CAL and VAL")
    # Chunk the canonicalization to bound peak RAM on flow corpora.
    blocks = [_fingerprints(x[start : start + 65536]) for x in matrices for start in range(0, len(x), 65536)]
    keys = np.unique(np.concatenate(blocks)).astype("<u8", copy=False)
    with path.open("xb") as out:
        np.save(out, keys, allow_pickle=False)
    receipt = {
        "path": str(path),
        "sha256": _sha(path),
        "algorithm": FINGERPRINT,
        "rows_by_role": dict(zip(DEV_ROLES, map(len, matrices), strict=True)),
        "keys": len(keys),
        "feature_count": width,
        "source_files_sha256": hashes,
        "development_group_ids": groups,
    }
    _write_json(receipt_path, receipt)
    return receipt


def load_development_screen(path):
    """Recover a development index after verifying its recorded checksum."""
    path = Path(path).resolve()
    receipt = _read_json(path.with_suffix(".json"))
    if (
        path.suffix != ".npy"
        or receipt["algorithm"] != FINGERPRINT
        or Path(receipt["path"]).resolve() != path
        or _sha(path) != _digest(receipt["sha256"])
    ):
        raise ValueError("Development screen differs from its recorded provenance")
    return receipt


def _distinct_names(values):
    values = tuple(values)
    if any(not isinstance(x, str) or not x for x in values) or len(set(values)) != len(values):
        raise ValueError("Frozen feature and class names must be distinct nonempty strings")
    return values


def _validate_configuration(info):
    names, features = _distinct_names(info["class_order"]), _distinct_names(info["feature_order"])
    if (
        len(names) < 2
        or names[0] != "BENIGN"
        or info["target_class"] not in names[1:]
        or not features
        or {"label", "group_id"}.intersection(features)
    ):
        raise ValueError("Invalid frozen class, target or feature schema")
    if (
        not isinstance(info["dataset"], str)
        or not info["dataset"]
        or type(info["seed"]) is not int
        or info["seed"] < 0
        or type(info["receiver"]) is not int
        or info["receiver"] < 0
        or info["model_schema"] != MODEL_SCHEMA
        or type(info["version"]) is not int
        or info["version"] not in (1, *FIXED_VERSIONS)
    ):
        raise ValueError("Invalid frozen receiver or model schema")
    modes = set(info["thresholds"])
    attribution = info["version"] == ATTRIBUTION_VERSION
    if attribution and modes != {*SUPPORTED_MODES, SINGLE_DONOR_MODE}:
        raise ValueError(
            "Frozen donor attribution requires both primary modes, the donor pool and the single donor"
        )
    if not attribution and not set(PRIMARY_MODES) <= modes <= set(SUPPORTED_MODES):
        raise ValueError("Frozen artifact requires both primary modes and optionally head_donor")
    for value in info["thresholds"].values():
        if type(value) not in (float, int) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Frozen thresholds must be finite probabilities")
    settings = info["settings"]
    if not isinstance(settings.get("protocol"), dict) or not settings["protocol"]:
        raise ValueError("Frozen cases must declare their matched method protocol")
    if attribution:
        donor, clients = settings.get("single_donor_client"), settings["protocol"].get("clients")
        if (
            type(clients) is not int
            or clients < 3
            or info["receiver"] >= clients
            or type(donor) is not int
            or not 0 <= donor < clients
            or donor == info["receiver"]
        ):
            raise ValueError("Frozen donor attribution requires a valid non-receiver single donor client")
    gate = settings.get("gate_disabled", {})
    if (
        not isinstance(gate, dict)
        or set(gate) - modes
        or any(type(value) is not bool for value in gate.values())
    ):
        raise ValueError("Gate status must be a boolean for each declared mode")
    if info["version"] in FIXED_VERSIONS:
        _validate_fixed_gate(info, names, modes, gate)
    elif "fixed_gate" in info:
        raise ValueError("Legacy head artifacts cannot carry a fixed TabM gate")
    screen = info["development_screen"]
    if screen["algorithm"] != FINGERPRINT or screen["feature_count"] != len(features):
        raise ValueError("Development screen schema differs from frozen features")
    _source_digests(screen["source_files_sha256"])


def _validate_fixed_gate(info, names, modes, gate):
    fixed = info["fixed_gate"]
    guarded = info["version"] in GUARDED_VERSIONS
    operating = info["version"] in OPERATING_VERSIONS
    fields = {"checkpoint", "sha256", "threshold", "gate_disabled", "routing", "eligible_attack_indices"}
    if guarded:
        fields.add("subtype_guards")
    if operating:
        fields.update(OPERATING_FIELDS)
    if info["version"] in SELECTION_VERSIONS:
        fields.add("operating_selection")
    routing = GUARDED_ROUTING if guarded else FIXED_ROUTING
    if (
        not isinstance(fixed, dict)
        or set(fixed) != fields
        or not Path(fixed["checkpoint"]).is_absolute()
        or fixed["routing"] != routing
        or fixed["eligible_attack_indices"] != [names.index(info["target_class"])]
        or any(type(value) is not int for value in fixed["eligible_attack_indices"])
        or type(fixed["gate_disabled"]) is not bool
        or type(fixed["threshold"]) not in (int, float)
        or not 0 <= fixed["threshold"] <= 1
        or (info["version"] != 6 and fixed["gate_disabled"] != (fixed["threshold"] == 1))
        or set(gate) != modes
        or any(gate[mode] != fixed["gate_disabled"] for mode in modes)
        or any(info["thresholds"][mode] != fixed["threshold"] for mode in modes)
    ):
        raise ValueError(
            "Fixed branches must use the same TabM threshold, active status and target-only routing"
        )
    _digest(fixed["sha256"])
    if operating and any(
        type(fixed[key]) not in (int, float) or not 0 <= fixed[key] <= 1 for key in OPERATING_FIELDS
    ):
        raise ValueError(
            "Fixed operating reference threshold and FNR allowance must be finite rates in [0, 1]"
        )
    if info["version"] in SELECTION_VERSIONS:
        _validate_operating_selection(
            fixed["operating_selection"],
            fixed,
            info["receiver"],
            info["settings"]["protocol"],
            class_count=len(names) if info["version"] == 6 else None,
            require_reference=info["version"] == ATTRIBUTION_VERSION,
        )
    if guarded:
        guards = fixed["subtype_guards"]
        if not isinstance(guards, dict) or set(guards) != modes:
            raise ValueError("Frozen subtype guards must match the declared head modes")
        if info["settings"]["protocol"].get("routing", routing) != routing:
            raise ValueError("Frozen subtype guard routing differs from the matched method protocol")
        for guard in guards.values():
            validate_subtype_guard(guard, eligible=fixed["eligible_attack_indices"])
        if len({(g["max_rewrite_rate"], g["min_negative_groups"]) for g in guards.values()}) != 1:
            raise ValueError("Frozen head branches must use the same subtype guard fit settings")


def _selection_digest(selection):
    payload = {
        key: selection[key]
        for key in (
            "status",
            "selected_cal_far",
            "candidate_grid",
            "constraints",
            "thresholds",
            "reference_thresholds",
            "receiver_client",
            "receiver_threshold",
            "receiver_reference_threshold",
            "cal_group_assignment_sha256",
            "group_overlap",
            "support",
            "selection_rule",
        )
    }
    if "reference_subtype_guard" in selection:
        payload["reference_subtype_guard"] = selection["reference_subtype_guard"]
    for field in CLASS_SELECTION_FIELDS:
        if field in selection:
            payload[field] = selection[field]
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def frozen_selection_digest(selection):
    return _selection_digest(selection)


def _validate_operating_selection(
    selection, fixed, receiver_id, protocol, *, class_count=None, require_reference=False
):
    reference_fields = OPERATING_SELECTION_FIELDS | {"reference_subtype_guard"}
    valid_fields = (
        (reference_fields | CLASS_SELECTION_FIELDS,)
        if class_count is not None
        else (reference_fields,) if require_reference else (OPERATING_SELECTION_FIELDS, reference_fields)
    )
    if not isinstance(selection, dict) or set(selection) not in valid_fields:
        raise ValueError("Frozen operating selection has an unexpected field inventory")
    schema, rule = OPERATING_SELECTION_SCHEMA, "loosest_first_descending_nominal_far"
    if class_count is not None:
        from .revision_class_operating import SCHEMA, SELECTION_RULE

        schema, rule = SCHEMA, SELECTION_RULE
    if selection["schema"] != schema or selection["status"] != "selected":
        raise ValueError("Frozen operating selection must be a selected CAL-only candidate")
    if selection["selection_rule"] != rule:
        raise ValueError("Frozen operating selection rule differs from the declared rule")
    grid = selection["candidate_grid"]
    if (
        not isinstance(grid, list)
        or not grid
        or any(type(value) not in (int, float) or not 0 < value < 1 for value in grid)
        or grid != sorted(set(grid), reverse=True)
    ):
        raise ValueError("Frozen operating candidate grid must be distinct descending rates in (0, 1)")
    if type(selection["selected_cal_far"]) not in (int, float) or selection["selected_cal_far"] not in grid:
        raise ValueError("Frozen selected CAL FAR must be a member of the candidate grid")
    if (
        type(selection["reference_cal_far"]) not in (int, float)
        or selection["reference_cal_far"] != OPERATING_REFERENCE_CAL_FAR
    ):
        raise ValueError("Frozen reference CAL FAR must be 1%")
    constraints = selection["constraints"]
    required = {
        "max_far",
        "reference_far",
        "max_fnr_increase",
        "min_benign_rows",
        "min_select_benign_rows",
        "min_select_attack_rows",
        "min_receiver_attack_rows",
        "min_attack_coverage",
        "threshold_floor",
        "pooled_benign_fallback",
    }
    if class_count is not None:
        required = (required - {"pooled_benign_fallback"}) | {"min_route_benign"}
    allowed = (required, required | {"pooled_benign_fallback"}) if class_count is not None else (required,)
    if not isinstance(constraints, dict) or set(constraints) not in allowed:
        raise ValueError("Frozen operating constraints have an unexpected schema")
    for key in ("max_far", "reference_far", "max_fnr_increase", "min_attack_coverage", "threshold_floor"):
        if type(constraints[key]) not in (int, float) or not 0 <= constraints[key] <= 1:
            raise ValueError("Frozen operating constraints must be rates in [0, 1]")
    for key in (
        "min_benign_rows",
        "min_select_benign_rows",
        "min_select_attack_rows",
        "min_receiver_attack_rows",
    ):
        if type(constraints[key]) is not int or constraints[key] < 0:
            raise ValueError("Frozen operating support minima must be nonnegative integers")
    if class_count is not None and (
        type(constraints["min_route_benign"]) is not int or constraints["min_route_benign"] < 1
    ):
        raise ValueError("Frozen class alarm route support minimum must be a positive integer")
    if type(constraints.get("pooled_benign_fallback", False)) is not bool:
        raise ValueError("Frozen pooled benign fallback flag must be boolean")
    if class_count is not None and constraints.get("pooled_benign_fallback", False):
        raise ValueError("Frozen class alarms require client-local threshold fitting")
    if (
        constraints["reference_far"] != OPERATING_REFERENCE_CAL_FAR
        or protocol.get("operating_reference_cal_far", OPERATING_REFERENCE_CAL_FAR)
        != OPERATING_REFERENCE_CAL_FAR
    ):
        raise ValueError("Frozen reference CAL FAR differs between selection constraints and protocol")
    if constraints["max_fnr_increase"] != fixed["max_operating_fnr_increase"]:
        raise ValueError("Frozen FNR allowance differs from the selection constraints")
    if (
        protocol.get("operating_candidates", grid) != grid
        or protocol.get("operating_constraints", constraints) != constraints
    ):
        raise ValueError("Frozen operating selection differs from the declared grid or protocol constraints")
    thresholds, references = selection["thresholds"], selection["reference_thresholds"]
    receiver = selection["receiver_client"]
    if (
        not isinstance(thresholds, list)
        or not isinstance(references, list)
        or not thresholds
        or len(thresholds) != len(references)
        or type(receiver) is not int
        or not 0 <= receiver < len(thresholds)
        or any(type(value) not in (int, float) or not 0 <= value <= 1 for value in (*thresholds, *references))
        or (class_count is None and any(value >= 1 for value in references))
        or (class_count is not None and references[receiver] >= 1)
    ):
        raise ValueError("Frozen operating thresholds and receiver index are invalid")
    clients = protocol.get("clients", len(thresholds))
    if type(clients) is not int or clients != len(thresholds):
        raise ValueError("Frozen operating threshold vector length differs from the client roster")
    if receiver != receiver_id:
        raise ValueError("Frozen operating selection receiver differs from the model receiver")
    if fixed["threshold"] != thresholds[receiver] or fixed["reference_threshold"] != references[receiver]:
        raise ValueError("Frozen gate thresholds differ from the selected CAL operating point")
    if (
        selection["receiver_threshold"] != thresholds[receiver]
        or selection["receiver_reference_threshold"] != references[receiver]
    ):
        raise ValueError("Frozen receiver operating thresholds differ from the selected vector")
    if class_count is not None:
        if thresholds != references:
            raise ValueError("Frozen class-gate scalar placeholders must equal the reference thresholds")
        _validate_class_selection(selection, fixed, protocol, clients, receiver, class_count)
    _digest(selection["cal_group_assignment_sha256"])
    _digest(selection["selection_sha256"])
    if selection["group_overlap"] != 0:
        raise ValueError("Frozen threshold-fit and operating-select groups must be disjoint")
    support = selection["support"]
    if not isinstance(support, dict):
        raise ValueError("Frozen operating selection must record support")
    receiver_attack = support.get("receiver_operating_select_attack_rows")
    if type(receiver_attack) is not int or receiver_attack < constraints["min_receiver_attack_rows"]:
        raise ValueError("Frozen operating selection contradicts the receiver attack support constraint")
    for key, minimum in (
        ("threshold_fit_benign_rows", 0),
        ("threshold_fit_attack_rows", 0),
        ("operating_select_benign_rows", constraints["min_select_benign_rows"]),
        ("operating_select_attack_rows", 0),
    ):
        if key not in support:
            continue
        counts = support[key]
        if (
            not isinstance(counts, list)
            or len(counts) != clients
            or any(type(count) is not int or count < minimum for count in counts)
        ):
            raise ValueError("Frozen operating selection has invalid per-client support")
    if "operating_select_attack_rows" in support:
        attacks = support["operating_select_attack_rows"]
        if attacks[receiver] != receiver_attack or sum(attacks) < constraints["min_select_attack_rows"]:
            raise ValueError("Frozen operating selection contradicts the attack support constraints")
    if "reference_subtype_guard" in selection:
        guard = validate_subtype_guard(
            selection["reference_subtype_guard"], eligible=fixed["eligible_attack_indices"]
        )
        candidate_guard = validate_subtype_guard(
            fixed["subtype_guards"][PRIMARY_MODES[1]], eligible=fixed["eligible_attack_indices"]
        )
        if any(
            guard[field] != candidate_guard[field] for field in ("max_rewrite_rate", "min_negative_groups")
        ):
            raise ValueError("Frozen reference and candidate subtype guards must use the same fit settings")
    if selection["selection_sha256"] != _selection_digest(selection):
        raise ValueError("Frozen operating selection digest differs from its contents")


def _validate_class_selection(selection, fixed, protocol, clients, receiver, class_count):
    routing = selection["class_routing"]
    if routing != CLASS_ALARM_ROUTING or protocol.get("class_routing", routing) != routing:
        raise ValueError("Frozen class alarm routing differs from the declared routing")
    matrix = selection["class_thresholds"]
    if (
        not isinstance(matrix, list)
        or len(matrix) != clients
        or any(not isinstance(row, list) or len(row) != class_count - 1 for row in matrix)
        or any(type(value) not in (int, float) or not 0 <= value <= 1 for row in matrix for value in row)
    ):
        raise ValueError(
            "Frozen class alarm thresholds must match the client roster and ordered attack classes"
        )
    if fixed["gate_disabled"] != all(value == 1 for value in matrix[receiver]):
        raise ValueError("Frozen class alarm gate status differs from the receiver threshold vector")


def save_frozen_head(
    path,
    *,
    dataset,
    names,
    features,
    seed,
    receiver,
    target_class,
    vectors,
    thresholds,
    development_parts=None,
    development_source_hashes=None,
    development_screen=None,
    settings=None,
    fixed_gate=None,
):
    """Save the receiver head states and thresholds, optionally bound to its fixed TabM gate."""
    path = Path(path).resolve()
    receipt = path.with_suffix(".json")
    if path.suffix != ".npz" or path.exists() or receipt.exists():
        raise ValueError("Frozen case requires a new .npz path")
    if development_screen is None:
        development_screen = prepare_development_screen(
            path.with_name(path.stem + "_development.npy"), development_parts, development_source_hashes
        )
    version = 1
    if fixed_gate is not None:
        if fixed_gate.get("routing") == GUARDED_ROUTING:
            operating_selection = fixed_gate.get("operating_selection")
            if operating_selection is not None:
                if SINGLE_DONOR_MODE in vectors:
                    version = ATTRIBUTION_VERSION
                else:
                    version = 6 if CLASS_SELECTION_FIELDS.intersection(operating_selection) else 5
            else:
                version = 4 if OPERATING_FIELDS.intersection(fixed_gate) else 3
        else:
            version = 2
    info = {
        "version": version,
        "dataset": dataset,
        "target_class": target_class,
        "seed": seed,
        "receiver": receiver,
        "class_order": list(names),
        "feature_order": list(features),
        "model_schema": MODEL_SCHEMA,
        "thresholds": thresholds,
        "settings": settings or {},
        "development_screen": {**development_screen, "path": str(Path(development_screen["path"]).resolve())},
    }
    if fixed_gate is not None:
        required = {"checkpoint", "threshold", "gate_disabled", "routing"}
        if version in GUARDED_VERSIONS:
            required.add("subtype_guards")
        if version == 4:
            required.update(OPERATING_FIELDS)
        if version in SELECTION_VERSIONS:
            required.update(OPERATING_FIELDS)
            required.add("operating_selection")
        if set(fixed_gate) not in (required, required | {"eligible_attack_indices"}):
            raise ValueError("Fixed gate fields differ from the frozen routing schema")
        checkpoint = Path(fixed_gate["checkpoint"]).resolve(strict=True)
        info["fixed_gate"] = {
            **fixed_gate,
            "checkpoint": str(checkpoint),
            "sha256": _sha(checkpoint),
            "eligible_attack_indices": list(
                fixed_gate.get("eligible_attack_indices", [info["class_order"].index(target_class)])
            ),
        }
    _validate_configuration(info)
    if fixed_gate is not None:
        _load_fixed_detector(info)
    vectors = {k: np.asarray(value, dtype="<f4") for k, value in vectors.items()}
    shape = (len(info["class_order"]), len(info["feature_order"]) + 1)
    if set(vectors) != set(thresholds) or any(
        v.shape != shape or not np.isfinite(v).all() for v in vectors.values()
    ):
        raise ValueError("Head coefficient inventory/dimensions differ from the frozen schema")
    info["inference_sha256"] = _inference_hashes(info["version"])
    with path.open("xb") as out:
        np.savez(out, **vectors)
    info["sha256"] = _sha(path)
    _write_json(receipt, info)
    return _artifact_entry(path, receipt, info)


def _artifact_entry(path, receipt, info):
    entry = {
        "file": str(path),
        "sha256": info["sha256"],
        "receipt": str(receipt),
        "receipt_sha256": _sha(receipt),
    }
    if info["version"] == ATTRIBUTION_VERSION:
        entry.update(seed=info["seed"], single_donor_client=info["settings"]["single_donor_client"])
    if info["version"] in FIXED_VERSIONS:
        entry.update(
            tabm_checkpoint=info["fixed_gate"]["checkpoint"],
            tabm_checkpoint_sha256=info["fixed_gate"]["sha256"],
        )
    return entry


def _load_fixed_detector(info):
    from .revision_tabm_checkpoint import load_tabm_checkpoint

    fixed = info["fixed_gate"]
    detector = load_tabm_checkpoint(
        fixed["checkpoint"],
        expected_sha256=fixed["sha256"],
        names=info["class_order"],
        features=info["feature_order"],
    )
    if detector.client_id != f"client_{info['receiver']}":
        raise ValueError("Frozen TabM checkpoint belongs to a different receiver")
    protocol = info["settings"]["protocol"]
    if detector.inference_device != "cpu" or protocol.get("base_inference_device", "cpu") != "cpu":
        raise ValueError("Fixed TabM inference must use the recorded CPU path")
    if protocol.get("base_batch_size", detector.batch_size) != detector.batch_size:
        raise ValueError("Frozen TabM batch size differs from the fixed protocol")
    return detector


def _load_receipt(path):
    path = Path(path).resolve()
    if path.suffix != ".npz":
        raise ValueError("Expected a frozen .npz case")
    receipt = path.with_suffix(".json")
    info = _read_json(receipt)
    _validate_configuration(info)
    if not _compatible_inference_hashes(info["inference_sha256"], _inference_hashes(info["version"])):
        raise ValueError("Frozen CAL/inference implementation changed since export")
    if _sha(path) != _digest(info["sha256"]):
        raise ValueError("Frozen head coefficients differ from their receipt")
    if (
        info["version"] in FIXED_VERSIONS
        and _sha(info["fixed_gate"]["checkpoint"]) != info["fixed_gate"]["sha256"]
    ):
        raise ValueError("Frozen TabM checkpoint differs from its receipt")
    return info, _artifact_entry(path, receipt, info)


def _compatible_inference_hashes(frozen, current):
    """Accept current source and exact reviewed packaging or formatting predecessors."""
    if frozen == current:
        return True
    unpackaged = _previous_inference_hashes(current, "package_import_compatibility.json", "packaged_sha256")
    if frozen == unpackaged:
        return True
    original = _preformat_inference_hashes(unpackaged)
    previous = _COMPATIBLE_SHARED_HEAD_SOURCE.get(original.get(SHARED_HEAD_SOURCE))
    return frozen == original or (
        previous is not None and frozen == {**original, SHARED_HEAD_SOURCE: previous}
    )


def _preformat_inference_hashes(current):
    """Recognize only exact current hashes verified against the original Python AST."""
    return _previous_inference_hashes(current, "frozen_format_compatibility.json", "formatted_sha256")


def _previous_inference_hashes(current, resource_name, current_key):
    """Look up a verified predecessor only when the corresponding current bytes match."""
    resource = files(fedcaid_resources).joinpath(resource_name)
    if not resource.is_file():
        return current
    conversions = json.loads(resource.read_text(encoding="utf-8"))
    original = current.copy()
    for name, pair in conversions.items():
        if current.get(name) == pair[current_key]:
            original[name] = pair["original_sha256"]
    return original


def _config(manifest, version):
    if manifest.get("version") != 1:
        raise ValueError("External manifest requires version 1")
    attestation = manifest["attestation"]
    if (
        not isinstance(attestation.get("custodian"), str)
        or not attestation["custodian"].strip()
        or attestation.get("unseen_in_development") is not True
    ):
        raise ValueError("External capture custodian must attest that captures were unseen in development")
    captures = manifest["captures"]
    if not isinstance(captures, list) or not captures:
        raise ValueError("Declare at least one external capture")
    ids = _group_ids(capture["capture_id"] for capture in captures)
    if len(set(ids)) != len(ids) or "ALL_CAPTURES" in ids:
        raise ValueError("Capture IDs must be distinct and cannot use the pooled row label")
    declared = manifest["criteria"]
    optional = FIXED_CRITERIA if version in FIXED_VERSIONS else {}
    allowed = {
        *CRITERIA,
        *optional,
        *ABSOLUTE_TARGET_CRITERIA.values(),
        *CAPTURE_GROUP_CRITERIA.values(),
        CAPTURE_F1_LOWER_CRITERION,
    }
    if not isinstance(declared, dict) or not set(CRITERIA) <= declared.keys() <= allowed:
        raise ValueError(f"Invalid confirmation criteria (required: {', '.join(CRITERIA)})")
    criteria = {**optional, **declared}
    per_capture_minima = [key for key in CAPTURE_GROUP_CRITERIA.values() if key in criteria]
    for key in ("min_target_groups", "min_benign_groups", "min_seeds", "min_captures", *per_capture_minima):
        if type(criteria[key]) is not int or criteria[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if criteria["min_seeds"] < 2:
        raise ValueError("A paired interval needs at least two seeds")
    if CAPTURE_F1_LOWER_CRITERION in criteria and criteria["min_captures"] < 2:
        raise ValueError(f"{CAPTURE_F1_LOWER_CRITERION} requires min_captures >= 2")
    absolute_floors = [key for key in ABSOLUTE_TARGET_CRITERIA.values() if key in criteria]
    capture_floor = (CAPTURE_F1_LOWER_CRITERION,) if CAPTURE_F1_LOWER_CRITERION in criteria else ()
    for key in (
        "min_target_f1_effect_pp",
        "max_far",
        "max_far_increase",
        "max_fnr_increase",
        *optional,
        *absolute_floors,
        *capture_floor,
    ):
        bound = 100 if key.endswith("_pp") else 1
        if type(criteria[key]) not in (float, int) or not 0 <= criteria[key] <= bound:
            raise ValueError(f"Invalid confirmation criterion: {key}")
    policy = manifest.get("exact_overlap_policy", "reject")
    if policy not in ("reject", "exclude_overlapping_groups"):
        raise ValueError("Unknown external overlap policy")
    for capture in captures:
        _digest(capture["sha256"])
        if not isinstance(capture["raw_files"], list) or not capture["raw_files"]:
            raise ValueError("Each capture must list its original raw files and hashes")
        for source in capture["raw_files"]:
            _digest(source["sha256"])
    return criteria, policy


def _capture_files(manifest_path, captures, development_hashes):
    source_paths, source_hashes, prepared_paths, prepared_hashes = set(), set(), set(), set()
    resolved = []
    for capture in captures:
        prepared = (manifest_path.parent / capture["prepared_csv"]).resolve()
        prepared_hash = _digest(capture["sha256"])
        raw = [
            [str((manifest_path.parent / row["path"]).resolve()), _digest(row["sha256"])]
            for row in capture["raw_files"]
        ]
        paths, digests = {p for p, _ in raw}, {d for _, d in raw}
        if prepared_hash in development_hashes or digests & development_hashes:
            raise ValueError(f"External capture reuses a development source SHA256: {capture['capture_id']}")
        if (
            len(paths) != len(raw)
            or len(digests) != len(raw)
            or source_paths & paths
            or source_hashes & digests
            or str(prepared) in prepared_paths
            or prepared_hash in prepared_hashes
            or str(prepared) in source_paths
            or prepared_hash in source_hashes
            or paths & prepared_paths
            or digests & prepared_hashes
        ):
            raise ValueError("External capture files must be unique across captures")
        source_paths.update(paths)
        source_hashes.update(digests)
        prepared_paths.add(str(prepared))
        prepared_hashes.add(prepared_hash)
        resolved.append({
            "capture_id": capture["capture_id"],
            "prepared_csv": str(prepared),
            "sha256": prepared_hash,
            "raw_files": raw,
        })
    return resolved


def _analysis_hashes(version):
    names = ("revision_external.py", "revision_data.py", "revision_report.py")
    if version in SELECTION_VERSIONS:
        names = (*names, "revision_operating_selection.py")
    return {**_inference_hashes(version), **{name: _sha(Path(__file__).with_name(name)) for name in names}}


def _inference_hashes(version):
    names = (SHARED_HEAD_SOURCE, "utils.py")
    if version in FIXED_VERSIONS:
        names += (
            "revision_tabm_checkpoint.py",
            "revision_fixed_subtype.py",
            "vendor/tabm.py",
            "vendor/rtdl_num_embeddings.py",
        )
    if version in GUARDED_VERSIONS:
        names += ("revision_subtype_guard.py",)
    if version == 6:
        names += ("revision_class_operating.py",)
    source = Path(__file__).parent
    return {name: _sha(source / name) for name in names}


def _plan(frozen, manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = _read_json(manifest_path)
    cases = [_load_receipt(path) for path in frozen]
    if not cases:
        raise ValueError("Provide frozen head cases")
    first = cases[0][0]
    version = first["version"]
    criteria, overlap_policy = _config(manifest, version)
    invariants = (
        "version",
        "dataset",
        "target_class",
        "class_order",
        "feature_order",
        "model_schema",
        "development_screen",
    )
    guard_fit = _guard_fit_settings(first) if version in GUARDED_VERSIONS else None
    max_operating_fnr = (
        first["fixed_gate"]["max_operating_fnr_increase"] if version in OPERATING_VERSIONS else None
    )
    selection_protocol = _operating_selection_protocol(first) if version in SELECTION_VERSIONS else None
    seeds = set()
    for info, _ in cases:
        if (
            any(info[k] != first[k] for k in invariants)
            or info["settings"].get("protocol") != first["settings"].get("protocol")
            or set(info["thresholds"]) != set(first["thresholds"])
            or (version in GUARDED_VERSIONS and _guard_fit_settings(info) != guard_fit)
            or (
                version in OPERATING_VERSIONS
                and info["fixed_gate"]["max_operating_fnr_increase"] != max_operating_fnr
            )
            or (version in SELECTION_VERSIONS and _operating_selection_protocol(info) != selection_protocol)
            or info["seed"] in seeds
        ):
            raise ValueError(
                "Sealed cases must share dataset, target, method protocol, overlap screen and have unique"
                " seeds"
            )
        seeds.add(info["seed"])
    if (
        first["dataset"] != manifest["dataset"]
        or first["target_class"] != manifest["target_class"]
        or len(seeds) < criteria["min_seeds"]
        or len(manifest["captures"]) < criteria["min_captures"]
    ):
        raise ValueError("Frozen case identity or coverage does not satisfy the declared confirmation design")
    screen = first["development_screen"]
    if _sha(screen["path"]) != _digest(screen["sha256"]):
        raise ValueError("Development overlap screen differs from its receipt")
    captures = _capture_files(
        manifest_path, manifest["captures"], set(screen["source_files_sha256"].values())
    )
    return {
        "version": version,
        "dataset": first["dataset"],
        "target_class": first["target_class"],
        "manifest": str(manifest_path),
        "manifest_sha256": _sha(manifest_path),
        "analysis_sha256": _analysis_hashes(version),
        "seeds": sorted(seeds),
        "criteria": criteria,
        "exact_overlap_policy": overlap_policy,
        **({"max_operating_fnr_increase": max_operating_fnr} if version in OPERATING_VERSIONS else {}),
        **({"operating_selection_protocol": selection_protocol} if version in SELECTION_VERSIONS else {}),
        "cases": [case for _, case in cases],
        "captures": captures,
        "attestation_scope": (
            "Custodian attestation; local hashes neither prove acquisition independence nor timestamp custody"
        ),
        "evaluation_scope": (
            "Frozen receiver TabM alarm and subtype heads; one declared dataset and target; no TEST"
            " calibration or fitting"
            if version in FIXED_VERSIONS
            else "Frozen receiver head; one declared dataset and target; no TEST calibration or fitting"
        ),
    }, [info for info, _ in cases]


def _guard_fit_settings(info):
    guards = info["fixed_gate"]["subtype_guards"]
    return {mode: (g["max_rewrite_rate"], g["min_negative_groups"]) for mode, g in guards.items()}


def _operating_selection_protocol(info):
    selection = info["fixed_gate"]["operating_selection"]
    return {
        **{key: selection[key] for key in ("schema", "candidate_grid", "constraints", "selection_rule")},
        **({"reference_subtype_nonregression": True} if "reference_subtype_guard" in selection else {}),
        **({"class_routing": selection["class_routing"]} if info["version"] == 6 else {}),
    }


def seal_confirmation(frozen, manifest, out):
    """Pin exact model and manifest receipts without reading held-out captures/labels."""
    plan, _ = _plan(frozen, manifest)
    _write_json(out, plan)
    return plan


def _load_capture(capture, info, keys, used_groups, overlap_policy):
    path = Path(capture["prepared_csv"])
    for raw_path, wanted in capture["raw_files"]:
        if _sha(raw_path) != wanted:
            raise ValueError(f"External raw file differs from its sealed hash: {raw_path}")
    if _sha(path) != capture["sha256"]:
        raise ValueError(f"External prepared CSV differs from its sealed hash: {path}")
    columns = [*info["feature_order"], "label", "group_id"]
    table = pd.read_csv(
        path, encoding="utf-8-sig", dtype={"label": str, "group_id": str}, keep_default_na=False
    )
    if list(table.columns) != columns or table.empty or not isinstance(table.index, pd.RangeIndex):
        raise ValueError(f"Expected nonempty canonical features followed by label,group_id: {path}")
    if _sha(path) != capture["sha256"]:
        raise ValueError(f"External prepared CSV changed during loading: {path}")
    values = table[info["feature_order"]].replace({"": np.nan, "NaN": np.nan, "nan": np.nan})
    x = values.apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
    labels = {name: index for index, name in enumerate(info["class_order"])}
    values = table.label.map(labels)
    if values.isna().any():
        raise ValueError(f"External CSV contains a label outside the frozen class order: {path}")
    y = values.to_numpy(dtype=np.int64)
    group_ids = np.asarray(table.group_id, dtype=str)
    distinct = set(_group_ids(np.unique(group_ids).tolist()))
    if distinct & used_groups:
        raise ValueError(f"External group identifiers overlap development or another capture: {path}")
    used_groups.update(distinct)
    fingerprints = _fingerprints(x)
    positions = np.minimum(np.searchsorted(keys, fingerprints), len(keys) - 1)
    overlaps = keys[positions] == fingerprints
    overlapping_groups = np.unique(group_ids[overlaps])
    if overlaps.any() and overlap_policy == "reject":
        raise ValueError(f"External exact-input overlap screen flagged {int(overlaps.sum())} rows: {path}")
    keep = ~np.isin(group_ids, overlapping_groups)
    if not keep.any():
        raise ValueError(f"External overlap exclusions leave the capture empty: {path}")
    target = labels[info["target_class"]]
    row = {
        "capture_id": capture["capture_id"],
        "before_rows": len(y),
        "before_groups": len(distinct),
        "rows_flagged_by_development_screen": int(overlaps.sum()),
        "excluded_groups": len(overlapping_groups),
        "excluded_rows": int((~keep).sum()),
        "rows": int(keep.sum()),
        "groups": len(np.unique(group_ids[keep])),
        "target_groups": len(np.unique(group_ids[keep & (y == target)])),
        "benign_groups": len(np.unique(group_ids[keep & (y == 0)])),
    }
    return x[keep], y[keep], group_ids[keep], row


def _model_states(case, info):
    with np.load(case["file"], allow_pickle=False) as arrays:
        states = {mode: arrays[mode] for mode in arrays.files}
    shape = (len(info["class_order"]), len(info["feature_order"]) + 1)
    if set(states) != set(info["thresholds"]) or any(
        value.dtype != np.dtype("<f4") or value.shape != shape or not np.isfinite(value).all()
        for value in states.values()
    ):
        raise ValueError("Unexpected frozen model inventory or coefficients")
    if _sha(case["file"]) != case["sha256"]:
        raise ValueError("Frozen model changed before confirmation")
    return states


def _evaluation(info, capture_id, mode, y, decisions, probabilities, groups):
    metrics, classes, counts = score(y, decisions, probabilities, info["class_order"])
    target = info["class_order"].index(info["target_class"])
    ident = {
        "dataset": info["dataset"],
        "target_class": info["target_class"],
        "capture_id": capture_id,
        "seed": info["seed"],
        "receiver": info["receiver"],
        "mode": mode,
    }
    fixed = info.get("fixed_gate")
    threshold = fixed["threshold"] if fixed is not None else info["thresholds"][mode]
    if mode in (OPERATING_REFERENCE_MODE, REFERENCE_SUBTYPE_MODE):
        threshold = fixed["reference_threshold"]
    class_gate = info["version"] == 6 and mode not in (OPERATING_REFERENCE_MODE, REFERENCE_SUBTYPE_MODE)
    gate = (
        fixed["gate_disabled"]
        if class_gate
        else (threshold >= 1 or info["settings"].get("gate_disabled", {}).get(mode, False))
    )
    for index, row in enumerate(classes):
        row["declared_group_count"] = len(np.unique(groups[y == index]))
    results = {
        **ident,
        **metrics,
        "gate_disabled": bool(gate),
        "threshold": threshold,
        "target_f1": classes[target]["f1"],
        "target_precision": classes[target]["precision"],
        "target_recall": classes[target]["recall"],
        "target_support": classes[target]["support"],
        "target_groups": classes[target]["declared_group_count"],
        "benign_groups": classes[0]["declared_group_count"],
    }
    if info["version"] in GUARDED_VERSIONS:
        guard = (
            fixed["operating_selection"]["reference_subtype_guard"]
            if mode == REFERENCE_SUBTYPE_MODE
            else fixed["subtype_guards"].get(mode)
        )
        results["subtype_guard_status"] = guard["status"] if guard is not None else None
        results["subtype_guard_threshold"] = guard["threshold"] if guard is not None else None
    if class_gate:
        selection = fixed["operating_selection"]
        results["class_alarm_routing"] = selection["class_routing"]
        results["class_alarm_thresholds"] = selection["class_thresholds"][info["receiver"]]
    if info["version"] == ATTRIBUTION_VERSION:
        results["single_donor_client"] = info["settings"]["single_donor_client"]
    return results, [{**ident, **row} for row in classes], {**ident, "counts": counts}


def _paired(rows, *, fixed=False):
    metrics = (
        "target_f1",
        "target_precision",
        "target_recall",
        "macro_f1",
        "far",
        "fnr",
        "ece",
        "alert_rate",
    )
    populations = {}
    for row in rows:
        populations.setdefault((row["capture_id"], row["seed"]), {})[row["mode"]] = row
    by_capture, deltas = {}, []
    for (capture, seed), modes in populations.items():
        fed_mode = PRIMARY_MODES[1]
        comparisons = [("aggregation", fed_mode, PRIMARY_MODES[0])]
        if fixed:
            comparisons.append(("against_local_detector", fed_mode, BASE_MODE))
            if DONOR_MODE in modes:
                comparisons.append(("receiver_participation", fed_mode, DONOR_MODE))
            if SINGLE_DONOR_MODE in modes:
                comparisons.extend(
                    (
                        (DONOR_POOL_CONTRAST, DONOR_MODE, SINGLE_DONOR_MODE),
                        (DONOR_FULL_CONTRAST, fed_mode, SINGLE_DONOR_MODE),
                    )
                )
            if OPERATING_REFERENCE_MODE in modes:
                comparisons.append((OPERATING_CONTRAST, fed_mode, OPERATING_REFERENCE_MODE))
            if REFERENCE_SUBTYPE_MODE in modes:
                comparisons.append((REFERENCE_SUBTYPE_CONTRAST, fed_mode, REFERENCE_SUBTYPE_MODE))
        for name, candidate, reference in comparisons:
            branch = modes[candidate]
            changes = {}
            for metric in metrics:
                a, b = modes[reference][metric], branch[metric]
                changes[metric] = b - a if a is not None and b is not None else None
            label = {"contrast": name} if fixed else {}
            deltas.append({
                "capture_id": capture, "seed": seed, "receiver": branch["receiver"], **label, **changes
            })
            by_capture.setdefault((capture, name), []).append(changes)
    summaries = [
        {
            "capture_id": capture,
            **({"contrast": name} if fixed else {}),
            "metric": metric,
            **summarize([row[metric] for row in values], bounds=(-1, 1)),
        }
        for (capture, name), values in by_capture.items()
        for metric in metrics
    ]
    return deltas, summaries


def _seed_metric_summary(values):
    observed = [value for value in values if value is not None]
    return {
        **summarize(observed),
        "minimum": min(observed, default=None),
        "maximum": max(observed, default=None),
    }


def _capture_summaries(metrics, paired):
    """Summarize cutoff outcomes across seeds, then compare independent capture units."""
    by_mode = {}
    for row in metrics:
        by_mode.setdefault((row["capture_id"], row["mode"]), []).append(row)
    absolute = [
        {
            "capture_id": capture_id,
            "mode": mode,
            "metric": metric,
            "expected_seeds": len(rows),
            **_seed_metric_summary(row[metric] for row in rows),
        }
        for (capture_id, mode), rows in by_mode.items()
        for metric in CAPTURE_METRICS
    ]
    per_capture = {}
    for statistic, rows, branch in (
        ("absolute", absolute, "mode"),
        ("paired_difference", paired, "contrast"),
    ):
        for row in rows:
            if row["capture_id"] != "ALL_CAPTURES" and row["metric"] in CAPTURE_METRICS:
                key = (statistic, row.get(branch, "aggregation"), row["metric"])
                per_capture.setdefault(key, []).append(row["mean"])
    intervals = []
    for (statistic, comparison, metric), values in per_capture.items():
        interval = summarize(values, bounds=(0, 1) if statistic == "absolute" else (-1, 1))
        intervals.append({
            "statistic": statistic,
            "comparison": comparison,
            "metric": metric,
            "declared_captures": len(values),
            "captures_with_metric": interval["seeds_with_metric"],
            "mean": interval["mean"],
            "capture_sd": interval["seed_sd"],
            "ci95_low": interval["ci95_low"],
            "ci95_high": interval["ci95_high"],
        })
    return absolute, intervals


def _capture_group_checks(row, criteria):
    return {
        f"{count}_min_met": row[count] >= criteria[limit]
        for count, limit in CAPTURE_GROUP_CRITERIA.items()
        if limit in criteria
    }


def _capture_target_f1(plan, summaries, supported):
    """Estimate the primary paired effect using complete per-capture seed means."""
    values = [
        row["mean"]
        for row in summaries
        if row["capture_id"] in supported
        and row["metric"] == "target_f1"
        and row.get("contrast", "aggregation") == "aggregation"
        and row["seeds_with_metric"] == len(plan["seeds"])
    ]
    interval = summarize(values, bounds=(-1, 1))
    return {
        "contrast": "federated_minus_local_subtype",
        "metric": "target_f1",
        "captures_with_target_and_benign": len(supported),
        "captures_with_complete_seed_pairs": interval["seeds_with_metric"],
        "mean": interval["mean"],
        "ci95_low": interval["ci95_low"],
        "ci95_high": interval["ci95_high"],
        "required_lower_pp": plan["criteria"].get(CAPTURE_F1_LOWER_CRITERION),
    }


def _external_capture_evidence(plan, summaries, audit):
    """Record tested overlap controls and the capture-level reporting unit."""
    supported = {row["capture_id"] for row in audit if row["target_groups"] and row["benign_groups"]}
    return {
        "capture_units": len(audit),
        "seed_models": len(plan["seeds"]),
        "target_f1_pairing": _capture_target_f1(plan, summaries, supported),
        "row_screen": {
            "policy": plan["exact_overlap_policy"],
            "development_roles": list(DEV_ROLES),
            **{
                field: sum(row[field] for row in audit)
                for field in (
                    "before_rows",
                    "rows_flagged_by_development_screen",
                    "excluded_rows",
                    "excluded_groups",
                    "rows",
                )
            },
        },
        "group_check": "supplied group IDs must not overlap declared development IDs or another capture",
        "provenance_scope": "custodian declaration and file hash checks",
    }


def _assessment(plan, metrics, summaries, audit):
    criteria = plan["criteria"]
    pooled = [r for r in metrics if r["capture_id"] == "ALL_CAPTURES" and r["mode"] in PRIMARY_MODES]
    federated = [r for r in pooled if r["mode"] == PRIMARY_MODES[1]]
    report = [r for r in summaries if r["capture_id"] == "ALL_CAPTURES"]
    interval = {r["metric"]: r for r in report if r.get("contrast", "aggregation") == "aggregation"}
    absolute = {
        metric: _seed_metric_summary(r[metric] for r in federated)
        for metric in ("target_f1", "target_precision", "target_recall")
    }
    capture_evidence = _external_capture_evidence(plan, summaries, audit)
    capture_f1 = capture_evidence["target_f1_pairing"]
    checks = {
        "target_groups": pooled[0]["target_groups"] >= criteria["min_target_groups"],
        "benign_groups": pooled[0]["benign_groups"] >= criteria["min_benign_groups"],
        "captures_with_target_and_benign": (
            sum(bool(r["target_groups"] and r["benign_groups"]) for r in audit) >= criteria["min_captures"]
        ),
        "paired_seed_support": all(
            interval[k]["seeds_with_metric"] >= criteria["min_seeds"] for k in ("target_f1", "far", "fnr")
        ),
        "gates_active": all(not r["gate_disabled"] for r in pooled),
    }
    for count, limit in CAPTURE_GROUP_CRITERIA.items():
        if limit in criteria:
            checks[f"{count}_each_capture"] = all(r[count] >= criteria[limit] for r in audit)
    for metric, limit in ABSOLUTE_TARGET_CRITERIA.items():
        if limit in criteria:
            checks[f"absolute_{metric}_complete_seed_support"] = absolute[metric]["seeds_with_metric"] == len(
                plan["seeds"]
            )
    f1, far, fnr = (interval[k] for k in ("target_f1", "far", "fnr"))
    if plan["version"] in FIXED_VERSIONS:
        versus_base = {r["metric"]: r for r in report if r["contrast"] == "against_local_detector"}
        checks["paired_baseline_support"] = all(
            versus_base[key]["seeds_with_metric"] >= criteria["min_seeds"]
            for key in ("target_f1", "macro_f1")
        )
    if plan["version"] in OPERATING_VERSIONS:
        operating = next(r for r in report if r["contrast"] == OPERATING_CONTRAST and r["metric"] == "fnr")
        checks["operating_fnr_paired_seed_support"] = operating["seeds_with_metric"] >= criteria["min_seeds"]
    if plan["version"] in SELECTION_VERSIONS:
        checks["cal_operating_selection_complete"] = (
            plan.get("operating_selection_protocol", {}).get("candidate_grid") is not None
        )
    reference_subtype = {}
    if plan.get("operating_selection_protocol", {}).get("reference_subtype_nonregression"):
        reference_subtype = {r["metric"]: r for r in report if r["contrast"] == REFERENCE_SUBTYPE_CONTRAST}
        checks["reference_subtype_paired_seed_support"] = all(
            reference_subtype.get(metric, {}).get("seeds_with_metric", 0) >= criteria["min_seeds"]
            for metric in ("target_f1", "macro_f1")
        )
    if CAPTURE_F1_LOWER_CRITERION in criteria:
        checks["capture_target_f1_paired_support"] = (
            capture_f1["captures_with_complete_seed_pairs"] >= criteria["min_captures"]
        )
    supported = all(checks.values())
    checks.update({
        "target_f1_practical_effect": (
            f1["mean"] is not None and f1["mean"] * 100 >= criteria["min_target_f1_effect_pp"]
        ),
        "target_f1_interval_positive": f1["ci95_low"] is not None and f1["ci95_low"] > 0,
        "absolute_far_each_seed": all(
            r["far"] is not None and r["far"] <= criteria["max_far"] for r in federated
        ),
        "far_increase_limit": (
            far["ci95_high"] is not None and far["ci95_high"] <= criteria["max_far_increase"]
        ),
        "fnr_increase_limit": (
            fnr["ci95_high"] is not None and fnr["ci95_high"] <= criteria["max_fnr_increase"]
        ),
    })
    for metric, limit in ABSOLUTE_TARGET_CRITERIA.items():
        if limit in criteria:
            average = absolute[metric]["mean"]
            checks[f"absolute_{metric}_seed_mean_met"] = average is not None and average >= criteria[limit]
    if plan["version"] in FIXED_VERSIONS:
        target = versus_base["target_f1"]
        macro = versus_base["macro_f1"]
        checks.update({
            "target_f1_gain_over_base": (
                target["mean"] is not None
                and target["ci95_low"] is not None
                and target["mean"] * 100 >= criteria["min_target_f1_over_base_pp"]
                and target["ci95_low"] > 0
            ),
            "macro_f1_loss_vs_base_limit": (
                macro["ci95_low"] is not None and macro["ci95_low"] * 100 >= -criteria["max_macro_f1_loss_pp"]
            ),
            "binary_alarm_metrics_equal_to_base": all(
                versus_base[key]["mean"] == 0
                and versus_base[key]["ci95_low"] == 0
                and versus_base[key]["ci95_high"] == 0
                for key in ("far", "fnr", "alert_rate")
            ),
        })
    if plan["version"] in OPERATING_VERSIONS:
        limit = plan["max_operating_fnr_increase"]
        checks["operating_fnr_limit_vs_1pct_reference"] = (
            operating["ci95_high"] is not None and operating["ci95_high"] <= limit
        )
    if reference_subtype:
        for metric in ("target_f1", "macro_f1"):
            mean = reference_subtype[metric]["mean"]
            checks[f"{metric}_mean_nonregression_vs_1pct_reference"] = mean is not None and mean >= 0
    if CAPTURE_F1_LOWER_CRITERION in criteria:
        low = capture_f1["ci95_low"]
        checks["capture_target_f1_lower_limit"] = (
            low is not None and 100 * low > criteria[CAPTURE_F1_LOWER_CRITERION]
        )
    state = (
        "insufficient_support_or_gate"
        if not supported
        else "criteria_met_on_declared_external_receiver" if all(checks.values()) else "criteria_not_met"
    )
    return {
        "status": state,
        "criteria": criteria,
        "checks": checks,
        "capture_evidence": capture_evidence,
        "absolute_federated_target_metrics": absolute,
        **(
            {"operating_fnr_vs_1pct_reference": {"max_increase": limit, **operating}}
            if plan["version"] in OPERATING_VERSIONS
            else {}
        ),
        **(
            {
                "subtype_nonregression_vs_1pct_reference": {
                    metric: reference_subtype[metric] for metric in ("target_f1", "macro_f1")
                }
            }
            if reference_subtype
            else {}
        ),
        "interval_scope": (
            "Paired t intervals across seeds on the same captures; capture-level summaries reported"
            " separately"
        ),
        "capture_interval_scope": (
            "Equal-weight t intervals of seed means across captures with metrics; exploratory for these"
            " capture units"
        ),
        "claim_scope": plan["evaluation_scope"],
        "attestation_scope": plan["attestation_scope"],
        "overlap_scope": (
            "Canonical pretransform equality screen; no near-duplicate or session inference beyond supplied"
            " group IDs"
        ),
        "detector_training": False,
        "threshold_selection_on_test": False,
    }


def _predict_modes(x, y, info, model, detector, transitions, capture_id):
    if detector is None:
        for mode, coefficients in model.items():
            p = head_probabilities(coefficients, x)
            decision = hierarchical_decisions(p, info["thresholds"][mode])
            yield mode, decision["predicted"], decision["attack_score"]
        return

    from .revision_fixed_subtype import fixed_subtype_predictions, subtype_transitions

    fixed = info["fixed_gate"]
    eligible = fixed["eligible_attack_indices"]
    guards = fixed.get("subtype_guards")
    reference_guard = fixed.get("operating_selection", {}).get("reference_subtype_guard")
    tabm_p = detector(x)
    alarm_threshold = fixed["threshold"]
    if info["version"] == 6:
        from .revision_class_operating import class_alarm_thresholds

        values = fixed["operating_selection"]["class_thresholds"][info["receiver"]]
        alarm_threshold = class_alarm_thresholds(tabm_p, values)
    base_decision = hierarchical_decisions(tabm_p, alarm_threshold)
    base, attack = base_decision["predicted"], base_decision["attack_score"]
    yield BASE_MODE, base, attack
    if info["version"] in OPERATING_VERSIONS:
        reference = hierarchical_decisions(tabm_p, fixed["reference_threshold"])
        yield OPERATING_REFERENCE_MODE, reference["predicted"], attack
    batch_size = info["settings"]["protocol"].get("head_batch_size", 65_536)
    for mode, coefficients in model.items():
        probabilities = head_probabilities(coefficients, x, device="cpu", batch_size=batch_size)
        guess = (
            fixed_subtype_predictions(base, probabilities, eligible)
            if guards is None
            else guarded_subtype_predictions(base, probabilities, eligible, guards[mode])
        )
        if not np.array_equal(guess > 0, base > 0):
            raise RuntimeError("Frozen fixed-alarm route changed the binary alarm mask")
        _, details = subtype_transitions(
            y, base, guess, info["class_order"], eligible, subtype_scores=probabilities
        )
        ident = {
            "dataset": info["dataset"],
            "target_class": info["target_class"],
            "capture_id": capture_id,
            "seed": info["seed"],
            "receiver": info["receiver"],
            "mode": mode,
        }
        transitions.extend({**ident, **row} for row in details)
        yield mode, guess, attack
        if reference_guard is not None and mode == PRIMARY_MODES[1]:
            reference_base = reference["predicted"]
            reference_guess = guarded_subtype_predictions(
                reference_base, probabilities, eligible, reference_guard
            )
            if not np.array_equal(reference_guess > 0, reference_base > 0):
                raise RuntimeError("Frozen reference subtype route changed its binary alarm mask")
            _, details = subtype_transitions(
                y,
                reference_base,
                reference_guess,
                info["class_order"],
                eligible,
                subtype_scores=probabilities,
            )
            transitions.extend({**ident, "mode": REFERENCE_SUBTYPE_MODE, **row} for row in details)
            yield REFERENCE_SUBTYPE_MODE, reference_guess, attack


@dataclass
class _CaptureResults:
    """Keep per-capture metrics and aligned predictions for the pooled evaluation."""

    predictions: dict = field(default_factory=dict)
    labels: list = field(default_factory=list)
    groups: list = field(default_factory=list)
    metrics: list = field(default_factory=list)
    per_class: list = field(default_factory=list)
    confusion: list = field(default_factory=list)
    audit: list = field(default_factory=list)
    transitions: list = field(default_factory=list)

    def measure(self, info, capture_id, mode, labels, predictions, attack, groups):
        """Record metrics and their class denominators for one case and mode."""
        row, classes, counts = _evaluation(info, capture_id, mode, labels, predictions, attack, groups)
        self.metrics.append(row)
        self.per_class.extend(classes)
        self.confusion.append(counts)


def _read_screen(screen):
    """Load and validate the sealed development-overlap index."""
    keys = np.load(screen["path"], mmap_mode="r", allow_pickle=False)
    if keys.ndim != 1 or keys.dtype != np.dtype("<u8") or not len(keys) or len(keys) != screen["keys"]:
        raise ValueError("Invalid development overlap index")
    return keys


def _frozen_models(plan, infos, fixed):
    """Load each seed's frozen heads and optional receiver detector once."""
    models = [_model_states(case, info) for case, info in zip(plan["cases"], infos, strict=True)]
    detectors = [_load_fixed_detector(info) for info in infos] if fixed else [None] * len(infos)
    return list(zip(infos, models, detectors, strict=True))


def _measure_capture(results, capture_id, features, labels, groups, models):
    """Score one capture in the same order as the frozen seed inventory."""
    results.labels.append(labels)
    results.groups.append(groups)
    for info, model, detector in models:
        for mode, guess, attack in _predict_modes(
            features, labels, info, model, detector, results.transitions, capture_id
        ):
            guesses, scores = results.predictions.setdefault((info["seed"], mode), ([], []))
            guesses.append(guess)
            scores.append(attack)
            results.measure(info, capture_id, mode, labels, guess, attack, groups)


def _measure_captures(plan, infos, fixed):
    """Read each capture, audit its support and accumulate frozen predictions."""
    screen = infos[0]["development_screen"]
    keys = _read_screen(screen)
    models = _frozen_models(plan, infos, fixed)
    used_groups = set(screen["development_group_ids"])
    results = _CaptureResults()
    for capture in plan["captures"]:
        features, labels, groups, entry = _load_capture(
            capture, infos[0], keys, used_groups, plan["exact_overlap_policy"]
        )
        results.audit.append({**entry, **_capture_group_checks(entry, plan["criteria"])})
        _measure_capture(results, capture["capture_id"], features, labels, groups, models)
    return results


def _measure_pooled(plan, infos, fixed, results):
    """Add pooled summaries with the original frozen mode ordering."""
    labels, groups = np.concatenate(results.labels), np.concatenate(results.groups)
    for info in infos:
        modes = (BASE_MODE, *info["thresholds"]) if fixed else info["thresholds"]
        if info["version"] in OPERATING_VERSIONS:
            modes = (*modes, OPERATING_REFERENCE_MODE)
        if plan.get("operating_selection_protocol", {}).get("reference_subtype_nonregression"):
            modes = (*modes, REFERENCE_SUBTYPE_MODE)
        for mode in modes:
            guess, attack = map(np.concatenate, results.predictions[info["seed"], mode])
            results.measure(info, "ALL_CAPTURES", mode, labels, guess, attack, groups)


def _write_confirmation_tables(out, results, pairs, intervals, fixed):
    """Write class, paired, capture and confusion reports from the measured populations."""
    capture_metrics, capture_intervals = _capture_summaries(results.metrics, intervals)
    out.mkdir(parents=True, exist_ok=False)
    for stem, values in (
        ("metrics", results.metrics),
        ("per_class", results.per_class),
        ("paired_by_seed", pairs),
        ("paired_intervals", intervals),
        ("capture_support", results.audit),
        ("capture_metric_summaries", capture_metrics),
        ("capture_intervals", capture_intervals),
    ):
        write_csv(out / f"{stem}.csv", values)
    if fixed:
        write_csv(out / "fixed_alarm_transitions.csv", results.transitions)
    _write_json(out / "confusion_counts.json", results.confusion)


def confirm_external(seal, out):
    """Verify and score the sealed pipeline without fitting or threshold selection."""
    seal, out = Path(seal).resolve(), Path(out).resolve()
    plan = _read_json(seal)
    if out.exists():
        raise ValueError("Confirmation results need a new output directory")
    current, infos = _plan([case["file"] for case in plan["cases"]], plan["manifest"])
    if current != plan:
        raise ValueError(
            "Seal verification failed: frozen coefficients, data declaration or analysis code changed"
        )
    fixed = plan["version"] in FIXED_VERSIONS
    results = _measure_captures(plan, infos, fixed)
    _measure_pooled(plan, infos, fixed, results)
    pairs, intervals = _paired(results.metrics, fixed=fixed)
    assessment = _assessment(plan, results.metrics, intervals, results.audit)
    _write_confirmation_tables(out, results, pairs, intervals, fixed)
    receipt = {
        "seal": str(seal),
        "seal_sha256": _sha(seal),
        "captures": len(results.audit),
        "seeds": plan["seeds"],
    }
    _write_json(out / "complete.json", {**assessment, **receipt})
    return assessment
