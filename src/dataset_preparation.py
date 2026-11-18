"""Rebuild frozen TRAIN/CAL/VAL from pinned CSVs and source row identities."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
import pandas as pd

from ._iov_layout import project_root
from .data_io import stream_source
from .utils import save_json, sha

_PREPARED_FILES = ("development.npz", "selected_train_identities.npz", "preparation.json")
PREPARATION_ROOT = Path(__file__).resolve().parent / "fedcaid_resources" / "preparation"
DEFAULT_CACHE = project_root(__file__) / "data_cache"
RAW_DIRECTORIES = {
    "Car-Hacking": "Car-Hacking Dataset",
    "CIC-IDS-2017": "CIC-IDS-2017 Dataset",
    "CIC-IoV-2024": "CICIoV2024/decimal",
    "CIC-UNSW-NB15": "CIC-UNSW-NB15",
}


def recipe_path(dataset, dataset_root=None):
    """Read the bundled role recipe, independently of the raw data directory."""
    from .dataset_access import DATASET_PROFILES

    if dataset not in DATASET_PROFILES:
        raise ValueError("Unknown dataset identifier: " + str(dataset))
    return PREPARATION_ROOT / dataset / "recipe.json"


def resolve_raw_directory(recipe, dataset_root):
    """Resolve the documented local raw directory inside ./dataset/."""
    relative = RAW_DIRECTORIES[recipe["name"]]
    if recipe.get("raw_directory", relative) != relative:
        raise ValueError("Raw directory differs from the fixed local dataset layout")
    if (
        not isinstance(relative, str)
        or not relative
        or "\\" in relative
        or Path(relative).is_absolute()
        or PureWindowsPath(relative).drive
        or any(part in (".", "..") for part in relative.split("/"))
        or PurePosixPath(relative).is_absolute()
    ):
        raise ValueError("Raw directory must be a safe relative path inside dataset/")
    root = Path(dataset_root).resolve()
    directory = root.joinpath(*PurePosixPath(relative).parts)
    if not directory.resolve().is_relative_to(root):
        raise ValueError("Raw directory must stay inside dataset/ after resolving links")
    return directory


def _is_link(path):
    if path.is_symlink():
        return True
    if not path.exists():
        return False
    attributes = getattr(path.lstat(), "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def checked_source_path(root, path):
    """Require a path inside root with no symbolic links or junctions."""
    root, path = Path(root).resolve(), Path(path).absolute()
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError("Raw source path escapes its dataset or staging directory") from error
    current = root
    for part in relative.parts:
        current = current / part
        if _is_link(current):
            raise ValueError("Raw source paths must not contain symbolic links or junctions")
    if not path.resolve().is_relative_to(root):
        raise ValueError("Raw source path escapes its dataset or staging directory")
    return path


def _source_aliases(source):
    aliases = source.get("aliases", [])
    if not isinstance(aliases, list) or not all(isinstance(name, str) for name in aliases):
        raise ValueError("Source aliases must be a list of CSV filenames")
    names = {source["filename"], *aliases}
    for name in names:
        if (
            not name
            or name != name.strip()
            or Path(name).name != name
            or "/" in name
            or "\\" in name
            or ":" in name
            or "\x00" in name
            or name in {".", ".."}
            or Path(name).suffix.lower() != ".csv"
        ):
            raise ValueError("Source aliases must be plain CSV filenames")
    result = {name.casefold() for name in names}
    if result & {"data.csv", "label.csv"}:
        raise ValueError("Data.csv and Label.csv are not frozen experiment sources")
    return result


def locate_raw_sources(recipe, dataset_root, *, verify=True):
    """Read CSVs only from this dataset's fixed local directory."""
    root = Path(dataset_root).resolve()
    default_directory = resolve_raw_directory(recipe, root)
    sources = recipe["source_files"]
    aliases = [_source_aliases(source) for source in sources]
    wanted = set().union(*aliases)
    candidates = {name: [] for name in wanted}
    checked_source_path(root, default_directory)
    if default_directory.is_dir():
        for path in sorted(default_directory.iterdir()):
            if path.name.casefold() in wanted:
                path = checked_source_path(root, path)
                candidates[path.name.casefold()].append(path)
    resolved = []
    for source, names in zip(sources, aliases):
        matches = sorted(
            {path for name in names for path in candidates[name]},
            key=lambda path: (path.relative_to(root).as_posix().casefold(), str(path)),
        )
        if verify:
            for path in matches:
                _verify_raw({**source, "path": path})
        path = matches[0] if matches else default_directory / source["filename"]
        resolved.append({**source, "path": checked_source_path(root, path)})
    return resolved


def raw_source_paths(dataset, dataset_root):
    """List expected local inputs without hashing the raw CSVs."""
    recipe = _recipe(dataset, dataset_root, verify_raw=False)
    return tuple(source["path"] for source in recipe["source_files"])


def _recipe(dataset, dataset_root, *, verify_raw=True):
    from .dataset_access import DATASET_PROFILES

    root = Path(dataset_root)
    path = checked_source_path(PREPARATION_ROOT, recipe_path(dataset))
    directory = path.parent
    spec = json.loads(path.read_text(encoding="utf-8"))
    profile = DATASET_PROFILES[dataset]
    if (
        spec.get("schema_version") != 1
        or spec.get("name") != dataset
        or spec.get("class_names") != profile["class_names"]
        or spec.get("feature_names") != profile["feature_names"]
        or spec.get("seed") != 42
        or spec.get("fraction") != "full"
    ):
        raise ValueError("Preparation recipe differs from the frozen dataset profile")
    for field in ("heldout_identities", "excluded_identities"):
        entry = spec[field]
        filename = entry["filename"]
        if (
            Path(filename).name != filename
            or filename in (".", "..")
            or "\\" in filename
            or "/" in filename
            or PureWindowsPath(filename).drive
        ):
            raise ValueError("Recipe metadata must use a plain file name")
        metadata = checked_source_path(PREPARATION_ROOT, directory / filename)
        if not metadata.is_file() or metadata.is_symlink() or sha(metadata) != entry["sha256"]:
            raise ValueError("Frozen source row metadata hash differs: " + filename)
        spec[field + "_path"] = metadata
    sources = []
    for source in spec["source_files"]:
        filename = source["filename"]
        if (
            Path(filename).name != filename
            or filename in (".", "..")
            or "\\" in filename
            or "/" in filename
            or PureWindowsPath(filename).drive
        ):
            raise ValueError("Raw source must use a plain file name")
        sources.append(dict(source))
    if not sources or len({s["filename"] for s in sources}) != len(sources):
        raise ValueError("Preparation recipe needs distinct ordered source files")
    spec["source_files"] = sources
    spec["source_files"] = locate_raw_sources(spec, root, verify=verify_raw)
    spec["recipe_sha256"] = sha(path)
    return spec


def check_raw_sources(dataset, dataset_root):
    """Report missing local CSV files and their official manual download address."""
    spec = _recipe(dataset, dataset_root)
    missing = [source["filename"] for source in spec["source_files"] if not source["path"].is_file()]
    if missing:
        raise FileNotFoundError(
            f"{dataset}: missing local CSV files: {', '.join(missing)}. "
            f"Download from {spec['official_source']} and extract under "
            f"{resolve_raw_directory(spec, dataset_root)}. Automatic downloads are disabled."
        )
    return spec


def _verify_raw(source):
    path = source["path"]
    if not path.is_file() or path.is_symlink():
        raise ValueError("Raw CSV version differs from the published SHA256: " + source["filename"])
    if path.stat().st_size != source["bytes"]:
        raise ValueError("Raw CSV byte count differs from the frozen experiment: " + source["filename"])
    if sha(path) != source["sha256"]:
        raise ValueError("Raw CSV SHA256 differs from the frozen experiment: " + source["filename"])


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


def _install_prepared(staged, output, root):
    output = checked_source_path(root, output)
    if not output.exists():
        staged.rename(output)
        return
    if not output.is_dir():
        raise ValueError("Prepared dataset location must be a regular directory")
    if any((output / name).exists() or (output / name).is_symlink() for name in _PREPARED_FILES):
        raise FileExistsError("Prepared files appeared during preparation; existing files preserved")
    installed = []
    try:
        # Exclusive links install complete files atomically without replacing raw data.
        for name in _PREPARED_FILES:
            source, target = staged / name, output / name
            os.link(source, target)
            installed.append((source, target))
    except BaseException:
        for source, target in reversed(installed):
            if target.exists() and not _is_link(target) and os.path.samefile(source, target):
                target.unlink()
        raise


def prepare_dataset(dataset, dataset_root, *, cache_root=None, chunk_size=250000):
    """Rebuild full TRAIN and ordered CAL/VAL without replacing existing data."""
    root = Path(dataset_root)
    spec = _recipe(dataset, root)
    cache_root = Path(cache_root) if cache_root is not None else DEFAULT_CACHE
    cache_root = cache_root.resolve()
    output = cache_root / dataset
    if output.exists():
        checked_source_path(cache_root, output)
        if not output.is_dir():
            raise ValueError("Prepared dataset location must be a regular directory")
        if any((output / name).exists() or (output / name).is_symlink() for name in _PREPARED_FILES):
            if not all((output / name).is_file() and not _is_link(output / name) for name in _PREPARED_FILES):
                raise ValueError("Prepared dataset is incomplete; existing files preserved")
            from .dataset_access import read_local_dataset

            audit = json.loads((output / "preparation.json").read_text(encoding="utf-8"))
            if audit.get("source_recipe_sha256") != spec["recipe_sha256"]:
                raise ValueError("Prepared cache belongs to a different frozen role recipe")
            read_local_dataset(output / "development.npz", dataset)
            return output / "development.npz"
    check_raw_sources(dataset, root)
    for source in spec["source_files"]:
        _verify_raw(source)
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
    if (
        len(codes) != spec["train_rows"]
        or (np.diff(codes) == 0).any()
        or np.bincount(y, minlength=len(names)).tolist() != spec["train_class_counts"]
    ):
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
    cache_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare_", dir=cache_root) as scratch:
        staged = Path(scratch) / dataset
        staged.mkdir()
        data = staged / "development.npz"
        # Preserve the original role order and array layout for identical NPZ hashes.
        parts = {
            "train": (x[permutation][:, keep], y[permutation]),
            **{role: (role_data[role][0][:, keep], role_data[role][1]) for role in ("val", "cal")},
        }
        np.savez_compressed(
            data,
            class_names=names,
            feature_names=features,
            **{"X_" + role: value[0] for role, value in parts.items()},
            **{"y_" + role: value[1] for role, value in parts.items()},
        )
        identities = staged / "selected_train_identities.npz"
        np.savez_compressed(
            identities,
            source_file_id=(codes[permutation] >> np.uint64(48)).astype(np.int32),
            source_row=(codes[permutation] & np.uint64((1 << 48) - 1)).astype(np.int64),
            y=y[permutation],
        )
        save_json(
            staged / "preparation.json",
            {
                "source_recipe_sha256": spec["recipe_sha256"],
                "data_sha256": sha(data),
                "identity_sha256": sha(identities),
                "train_rows": len(y),
                "train_class_counts": np.bincount(y, minlength=len(names)).tolist(),
                "feature_names": features,
                "dropped_train_constants": [],
                "fraction": "full",
                "selection_description": (
                    "all eligible original TRAIN source identities; known heldout and confirmation excluded"
                ),
                "all_eligible_train_included": True,
                "known_heldout_identities_excluded": len(exclusion),
                "verified_frozen_heldout_source_rows": len(heldout),
                "heldout_values_unchanged": True,
                "test_scored": False,
                "confirmation_scored": False,
                "duplicates_removed": 0,
                "seed": spec["seed"],
                "preparation_route": "verified_raw_source_replay",
            },
        )
        from .data_io import load_data

        load_data(data)
        _install_prepared(staged, output, cache_root)
    print("PREPARED " + dataset + ": frozen roles verified", flush=True)
    return output / "development.npz"
