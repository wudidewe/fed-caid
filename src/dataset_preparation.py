"""Rebuild the frozen development roles from verified public source CSV files.

Recipes contain file hashes, feature maps and source row identities, not raw
traffic. No new split, sampling, feature selection or model tuning is applied.
"""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import tempfile

import numpy as np
import pandas as pd

from data_io import stream_source
from ids_core import save_json, sha


def recipe_path(dataset, dataset_root):
    """Return the public recipe path without touching the filesystem."""
    from dataset_access import DATASET_PROFILES
    if dataset not in DATASET_PROFILES:
        raise ValueError("Unknown dataset identifier: " + str(dataset))
    return Path(dataset_root) / "preparation" / dataset / "recipe.json"


def resolve_raw_directory(recipe, dataset_root):
    """Resolve a declared relative raw directory, including the IoV archive/."""
    relative = recipe.get("raw_directory", "raw/" + recipe["name"])
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or Path(relative).is_absolute() or PureWindowsPath(relative).drive
            or any(part in (".", "..") for part in relative.split("/"))
            or PurePosixPath(relative).is_absolute()):
        raise ValueError("Raw directory must be a safe relative path inside dataset/")
    root = Path(dataset_root).resolve()
    directory = root.joinpath(*PurePosixPath(relative).parts)
    if not directory.resolve().is_relative_to(root):
        raise ValueError("Raw directory must stay inside dataset/ after resolving links")
    return directory


def _recipe(dataset, dataset_root):
    from dataset_access import DATASET_PROFILES
    root = Path(dataset_root)
    path = recipe_path(dataset, root)
    directory = path.parent
    spec = json.loads(path.read_text(encoding="utf-8"))
    profile = DATASET_PROFILES[dataset]
    if (spec.get("schema_version") != 1 or spec.get("name") != dataset
            or spec.get("class_names") != profile["class_names"]
            or spec.get("feature_names") != profile["feature_names"]
            or spec.get("seed") != 42 or spec.get("fraction") != "full"):
        raise ValueError("Preparation recipe differs from the frozen dataset profile")
    for field in ("heldout_identities", "excluded_identities"):
        entry = spec[field]
        filename = entry["filename"]
        if (Path(filename).name != filename or filename in (".", "..")
                or "\\" in filename or "/" in filename or PureWindowsPath(filename).drive):
            raise ValueError("Recipe metadata must use a plain file name")
        metadata = directory / filename
        if not metadata.is_file() or metadata.is_symlink() or sha(metadata) != entry["sha256"]:
            raise ValueError("Frozen source row metadata hash differs: " + filename)
        spec[field + "_path"] = metadata
    sources = []
    for source in spec["source_files"]:
        filename = source["filename"]
        if (Path(filename).name != filename or filename in (".", "..")
                or "\\" in filename or "/" in filename or PureWindowsPath(filename).drive):
            raise ValueError("Raw source must use a plain file name")
        sources.append({**source, "path": resolve_raw_directory(spec, root) / filename})
    if not sources or len({s["filename"] for s in sources}) != len(sources):
        raise ValueError("Preparation recipe needs distinct ordered source files")
    spec["source_files"] = sources
    spec["recipe_sha256"] = sha(path)
    return spec


def check_raw_sources(dataset, dataset_root, allow_download=True):
    """Preflight availability without downloading, creating folders or training."""
    from dataset_access import _validate_download_url
    spec = _recipe(dataset, dataset_root)
    missing = []
    for source in spec["source_files"]:
        path = source["path"]
        if path.is_file() and not path.is_symlink():
            continue
        if path.exists():
            raise ValueError("Raw source must be a regular CSV file: " + str(path))
        if not allow_download or (not source.get("url") and not spec.get("kaggle", {}).get("handle")):
            missing.append(source["filename"])
        elif source.get("url"):
            _validate_download_url(source["url"])
    if missing:
        reason = "Downloads are disabled" if not allow_download else "No verified direct download URL is configured"
        raise FileNotFoundError(
            f"{dataset}: {reason}. Obtain the original CSV files from "
            f"{spec['official_source']} and place them in "
            f"{resolve_raw_directory(spec, dataset_root)}: " + ", ".join(missing)
        )


def _verify_raw(source):
    path = source["path"]
    if (not path.is_file() or path.is_symlink()
            or path.stat().st_size != source["bytes"]
            or sha(path) != source["sha256"]):
        raise ValueError("Raw CSV version differs from the published SHA256: " + source["filename"])


def _ensure_raw(spec, dataset_root):
    from dataset_access import _download_archive
    if (any(not source["path"].exists() for source in spec["source_files"])
            and spec.get("kaggle", {}).get("handle")):
        from kaggle_source import ensure_kaggle_sources
        ensure_kaggle_sources(spec["name"], dataset_root, spec)
    for source in spec["source_files"]:
        path = source["path"]
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".download_", dir=path.parent) as scratch:
                temporary = Path(scratch) / source["filename"]
                print("DOWNLOAD " + spec["name"] + ": " + source["filename"], flush=True)
                _download_archive(source["url"], temporary, source["bytes"])
                _verify_raw({**source, "path": temporary})
                if path.exists():
                    raise FileExistsError("Raw file appeared during download; refusing replacement: " + str(path))
                temporary.rename(path)
        _verify_raw(source)


def _identity_codes(frame, source_files):
    for column in ("source_file_id", "zero_based_data_row"):
        values = frame[column].to_numpy()
        if not np.issubdtype(values.dtype, np.integer) or (values < 0).any():
            raise ValueError("Source row identities must be nonnegative integers")
    fids = frame.source_file_id.to_numpy(np.int64)
    rows = frame.zero_based_data_row.to_numpy(np.int64)
    if (fids >= len(source_files)).any() or (rows >= (1 << 48)).any():
        raise ValueError("Source row identity is outside the declared source space")
    for fid, source in enumerate(source_files):
        if (rows[fids == fid] >= source["csv_data_rows"]).any():
            raise ValueError("Source row identity exceeds the declared CSV length")
    codes = (fids.astype(np.uint64) << np.uint64(48)) | rows.astype(np.uint64)
    if len(np.unique(codes)) != len(codes):
        raise ValueError("Repeated source row identities are forbidden")
    return codes


def prepare_dataset(dataset, dataset_root, allow_download=True, *, chunk_size=250000):
    """Prepare full TRAIN and the original ordered CAL/VAL under dataset_root.

    Existing caches and raw files are never replaced. Missing CSVs are fetched
    only when their recipe has a real public HTTPS URL and pinned file hash.
    """
    root = Path(dataset_root)
    spec = _recipe(dataset, root)
    output = root / dataset
    if output.exists():
        from dataset_access import read_local_dataset
        read_local_dataset(output / "development.npz", dataset)
        return output / "development.npz"
    check_raw_sources(dataset, root, allow_download)
    _ensure_raw(spec, root)
    names, features = spec["class_names"], spec["feature_names"]
    exclusion = pd.read_csv(spec["excluded_identities_path"])
    excluded = _identity_codes(exclusion, spec["source_files"])
    heldout = pd.read_csv(spec["heldout_identities_path"])
    if not set(heldout.role).issubset({"cal", "val"}) or set(heldout.role) != {"cal", "val"}:
        raise ValueError("Public heldout identities must contain CAL and VAL only")
    held_codes = _identity_codes(heldout, spec["source_files"])
    if not np.isin(held_codes, excluded).all():
        raise ValueError("CAL and VAL identities must be excluded from TRAIN")
    role_data, role_codes = {}, {}
    for role in ("cal", "val"):
        selected = heldout.loc[heldout.role == role].sort_values("row_in_role")
        if not np.array_equal(selected.row_in_role, np.arange(len(selected))):
            raise ValueError("Heldout role positions must be unique and consecutive")
        y = np.asarray([names.index(value) for value in selected["class"]], dtype=np.int64)
        if np.bincount(y, minlength=len(names)).tolist() != spec["heldout_class_counts"][role]:
            raise ValueError("Frozen heldout class supports differ")
        codes = _identity_codes(selected, spec["source_files"])
        order = np.argsort(codes)
        role_codes[role] = codes[order], order
        role_data[role] = np.empty((len(y), len(features)), np.float32), y, np.zeros(len(y), bool)
    print("PREPARE " + dataset + ": replaying frozen full TRAIN/CAL/VAL", flush=True)
    train_codes, train_labels = [], []
    counts = np.zeros(len(names), np.int64)
    for fid, rows, y, _ in stream_source(spec, False, chunk_size):
        codes = (np.uint64(fid) << np.uint64(48)) | rows.astype(np.uint64)
        counts += np.bincount(y, minlength=len(names))
        eligible = ~np.isin(codes, excluded)
        train_codes.append(codes[eligible])
        train_labels.append(y[eligible])
    if counts.tolist() != spec["source_class_counts"]:
        raise ValueError("Raw source class supports differ from the frozen recipe")
    codes, y = np.concatenate(train_codes), np.concatenate(train_labels)
    order = np.argsort(codes)
    codes, y = codes[order], y[order]
    if (len(codes) != spec["train_rows"] or (np.diff(codes) == 0).any()
            or np.bincount(y, minlength=len(names)).tolist() != spec["train_class_counts"]):
        raise ValueError("Full TRAIN source identities differ from the frozen recipe")
    del train_codes, train_labels
    x = np.empty((len(y), len(features)), np.float32)
    found = np.zeros(len(y), bool)
    for fid, rows, labels, values in stream_source(spec, True, chunk_size):
        raw_codes = (np.uint64(fid) << np.uint64(48)) | rows.astype(np.uint64)
        position = np.searchsorted(codes, raw_codes)
        matched = position < len(codes)
        matched[matched] &= codes[position[matched]] == raw_codes[matched]
        target = position[matched]
        if not np.array_equal(labels[matched], y[target]):
            raise ValueError("TRAIN label changed between source passes")
        x[target], found[target] = values[matched], True
        for role in ("cal", "val"):
            sorted_codes, positions = role_codes[role]
            location = np.searchsorted(sorted_codes, raw_codes)
            selected = location < len(sorted_codes)
            selected[selected] &= sorted_codes[location[selected]] == raw_codes[selected]
            destination = positions[location[selected]]
            hx, hy, hf = role_data[role]
            if not np.array_equal(labels[selected], hy[destination]):
                raise ValueError("Heldout source labels differ: " + role)
            hx[destination], hf[destination] = values[selected], True
    if not found.all() or any(not value[2].all() for value in role_data.values()):
        raise ValueError("Incomplete raw source replay")
    keep = np.isfinite(np.fmin.reduce(x, axis=0)) & (np.fmin.reduce(x, axis=0) < np.fmax.reduce(x, axis=0))
    if not keep.all():
        raise ValueError("Prepared feature map would change; check the exact raw release")
    for source in spec["source_files"]:
        _verify_raw(source)
    permutation = np.random.default_rng(spec["seed"]).permutation(len(y))
    with tempfile.TemporaryDirectory(prefix=".prepare_", dir=root) as scratch:
        staged = Path(scratch) / dataset
        staged.mkdir()
        data = staged / "development.npz"
        # Retain the original writer's role order and column-selection memory
        # layout as well as its values, preserving the frozen NPZ signature.
        parts = {"train": (x[permutation][:, keep], y[permutation]),
                 **{role: (role_data[role][0][:, keep], role_data[role][1]) for role in ("val", "cal")}}
        np.savez_compressed(data, class_names=names, feature_names=features,
                            **{"X_" + role: value[0] for role, value in parts.items()},
                            **{"y_" + role: value[1] for role, value in parts.items()})
        identities = staged / "selected_train_identities.npz"
        np.savez_compressed(identities,
                            source_file_id=(codes[permutation] >> np.uint64(48)).astype(np.int32),
                            source_row=(codes[permutation] & np.uint64((1 << 48) - 1)).astype(np.int64),
                            y=y[permutation])
        save_json(staged / "preparation.json", {
            "source_recipe_sha256": spec["recipe_sha256"], "data_sha256": sha(data),
            "identity_sha256": sha(identities), "train_rows": len(y),
            "train_class_counts": np.bincount(y, minlength=len(names)).tolist(),
            "feature_names": features, "dropped_train_constants": [], "fraction": "full",
            "selection_description": "all eligible original TRAIN source identities; known heldout and confirmation excluded",
            "all_eligible_train_included": True, "known_heldout_identities_excluded": len(exclusion),
            "verified_frozen_heldout_source_rows": len(heldout), "heldout_values_unchanged": True,
            "test_scored": False, "confirmation_scored": False, "duplicates_removed": 0,
            "seed": spec["seed"], "preparation_route": "verified_raw_source_replay",
        })
        from data_io import load_data
        load_data(data)
        if output.exists():
            raise FileExistsError("Prepared cache appeared during preparation; refusing replacement")
        staged.rename(output)
    print("PREPARED " + dataset + ": frozen roles verified", flush=True)
    return output / "development.npz"
