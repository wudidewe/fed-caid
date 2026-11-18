"""Bind private CAL/VAL predictions to a reproducible case and row order."""

import hashlib
import json

import numpy as np

FILENAME = "private_cal_val_scores.npz"
CACHE_SCHEMA = 2
_METADATA_KEY = "score_cache_metadata"


def _hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _identity(identity, route, configuration, union):
    order = [
        {
            role: hashlib.sha256(np.asarray(binding[role], dtype="<i8").tobytes()).hexdigest()
            for role in ("cal", "val")
        }
        for binding in union
    ]
    encoded = json.dumps([identity, route, configuration, order], sort_keys=True, allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _validate(probabilities, union):
    shapes = set()
    for scores in probabilities.values():
        if len(scores) != len(union):
            raise ValueError("Cached score client roster differs")
        for array, binding in zip(scores, union, strict=True):
            if (
                array.ndim != 2
                or array.dtype.kind != "f"
                or array.shape[1] < 2
                or len(array) != len(binding["cal"]) + len(binding["val"])
                or not np.isfinite(array).all()
                or (array < 0).any()
                or (array > 1).any()
            ):
                raise ValueError("Cached probabilities must be finite, in [0, 1] and aligned to CAL then VAL")
            shapes.add(array.shape[1])
    if len(shapes) != 1:
        raise ValueError("Cached class columns differ across branches")
    return shapes.pop()


def _mode_names(modes):
    if (
        not isinstance(modes, list)
        or not modes
        or any(type(mode) is not str or not mode for mode in modes)
        or len(set(modes)) != len(modes)
    ):
        raise ValueError("Invalid private score cache modes")
    return modes


def _class_names(names, class_count):
    if names is None:
        return None
    if isinstance(names, (str, bytes)):
        raise ValueError("Expected an ordered sequence of private score cache class names")
    names = list(names)
    if (
        len(names) != class_count
        or any(type(name) is not str or not name for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("Invalid private score cache class names")
    return names


def save_score_cache(directory, identity, route, configuration, union, probabilities, *, class_names=None):
    classes = _validate(probabilities, union)
    path = directory / FILENAME
    partial = directory / "private_cal_val_scores_partial.npz"
    modes = _mode_names(list(probabilities))
    binding = _identity(identity, route, configuration, union)
    metadata = {
        "schema_version": CACHE_SCHEMA,
        "binding_sha256": binding,
        "modes": modes,
        "clients": len(union),
        "classes": classes,
        "class_names": _class_names(class_names, classes),
    }
    arrays = {
        f"m{i}_c{j}": scores for i, mode in enumerate(modes) for j, scores in enumerate(probabilities[mode])
    }
    arrays[_METADATA_KEY] = np.asarray(json.dumps(metadata, sort_keys=True))
    try:
        np.savez_compressed(partial, **arrays)
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    return {"file": FILENAME, "sha256": _hash(path), **metadata}


def load_score_cache(
    directory, receipt, identity, route, configuration, union, *, expected_modes=None, expected_classes=None
):
    if receipt.get("schema_version") != CACHE_SCHEMA:
        raise ValueError("Private score cache is an unsupported legacy schema without embedded metadata")
    if receipt.get("file") != FILENAME or receipt.get("binding_sha256") != _identity(
        identity, route, configuration, union
    ):
        raise ValueError("Private score cache belongs to a different configuration or route")
    path = directory / FILENAME
    if _hash(path) != receipt.get("sha256"):
        raise ValueError("Private score cache hash differs")
    with np.load(path, allow_pickle=False) as archive:
        if _METADATA_KEY not in archive.files:
            raise ValueError("Private score cache has no embedded metadata")
        embedded = archive[_METADATA_KEY]
        if embedded.shape != () or embedded.dtype.kind != "U":
            raise ValueError("Invalid private score cache metadata array")
        metadata = json.loads(embedded.item())
        fields = {"schema_version", "binding_sha256", "modes", "clients", "classes", "class_names"}
        if (
            not isinstance(metadata, dict)
            or set(metadata) != fields
            or type(metadata["schema_version"]) is not int
            or metadata != {field: receipt.get(field) for field in fields}
        ):
            raise ValueError("Private score cache embedded metadata differs from the receipt")
        modes = _mode_names(metadata["modes"])
        if expected_modes is not None and set(modes) != set(expected_modes):
            raise ValueError("Private score cache modes differ from the expected branches")
        if type(metadata["clients"]) is not int or metadata["clients"] != len(union):
            raise ValueError("Private score cache client count differs from the expected roster")
        classes = metadata["classes"]
        if type(classes) is not int or classes < 2:
            raise ValueError("Invalid private score cache class count")
        names = _class_names(metadata["class_names"], classes)
        if expected_classes is not None:
            correct = (
                expected_classes == classes
                if type(expected_classes) is int
                else names == _class_names(expected_classes, classes)
            )
            if not correct:
                raise ValueError("Private score cache classes differ from the expected columns")
        keys = {_METADATA_KEY, *(f"m{i}_c{j}" for i in range(len(modes)) for j in range(len(union)))}
        if set(archive.files) != keys or len(archive.files) != len(keys):
            raise ValueError("Private score cache array roster differs")
        result = {mode: [archive[f"m{i}_c{j}"] for j in range(len(union))] for i, mode in enumerate(modes)}
    if _validate(result, union) != classes:
        raise ValueError("Private score cache probability columns differ from the embedded class count")
    return result
