"""Prepare and load frozen TRAIN/CAL/VAL data."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from ids_core import save_json, sha

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def label(value):
    value = str(value).strip()
    if value.lower() in ("benign", "normal"):
        return "BENIGN"
    value = value.replace("\u00ef\u00bf\u00bd", "-").replace("\ufffd", "-")
    value = unicodedata.normalize("NFKD", value)
    return re.sub(r"\s+", " ", re.sub(r"[^\x20-\x7e]", "-", value)).strip()


def apply_recipes(x, features, recipes):
    output = []
    for recipe in recipes:
        values = x[:, [features.index(n) for n in recipe["columns"]]].astype(float)
        valid = np.isfinite(values).all(1) & (values >= 0).all(1)
        if recipe["operation"] != "balance":
            raise ValueError(
                "Only the frozen network balance recipes are needed in this clean route"
            )
        with np.errstate(all="ignore"):
            result = (values[:, 0] - values[:, 1]) / (values.sum(1) + 1.0)
        output.append(np.where(valid & np.isfinite(result), result, np.nan).astype(np.float32))
    return np.column_stack(output) if output else np.empty((len(x), 0), np.float32)


def car_chunk(frame, source_name, with_features):
    frame = frame.fillna("")
    dlc = pd.to_numeric(frame[2], errors="raise").to_numpy(float)
    if not np.isfinite(dlc).all() or not ((dlc >= 0) & (dlc <= 8) & (dlc == np.floor(dlc))).all():
        raise ValueError("Invalid Car-Hacking DLC")
    dlc = dlc.astype(int)
    flags = np.empty(len(frame), object)
    for size in np.unique(dlc):
        mask = dlc == size
        flags[mask] = frame.loc[mask, 3 + int(size)].astype(str).str.strip()
        if (
            frame.loc[mask, list(range(4 + int(size), 12))]
            .astype(str)
            .apply(lambda c: c.str.strip())
            .to_numpy()
            != ""
        ).any():
            raise ValueError("Extra fields after Car-Hacking flag")
    if not np.isin(flags, ["R", "T"]).all():
        raise ValueError("Car-Hacking R/T flag missing or shifted")
    subtype = {
        "dos_dataset.csv": "DoS",
        "fuzzy_dataset.csv": "Fuzzy",
        "gear_dataset.csv": "Gear spoofing",
        "rpm_dataset.csv": "RPM spoofing",
    }[source_name.lower()]
    labels = np.where(flags == "T", subtype, "BENIGN")
    if not with_features:
        return labels, None
    ts = pd.to_numeric(frame[0], errors="raise").to_numpy(float)
    if not np.isfinite(ts).all() or (ts < 0).any():
        raise ValueError("Invalid CAN timestamp")
    x = np.zeros((len(frame), 10), np.float32)
    x[:, 1] = dlc
    for column, destination, maximum in [(1, 0, 0x1FFFFFFF)] + [
        (3 + j, 2 + j, 255) for j in range(8)
    ]:
        observed = np.ones(len(frame), bool) if column == 1 else dlc > column - 3
        text = frame.loc[observed, column].astype(str).str.strip()
        if not text.str.fullmatch(r"[0-9A-Fa-f]+").all():
            raise ValueError("CAN bytes and IDs must be hexadecimal")
        mapping = {s: int(s, 16) for s in text.unique()}
        values = text.map(mapping).to_numpy(np.int64)
        if (values > maximum).any() or (values.astype(np.float32).astype(np.int64) != values).any():
            raise ValueError("CAN value exceeds range or exact float32 representation")
        x[observed, destination] = values
    return labels, x


def stream_source(spec, with_features, chunk_size):
    """Retain physical CSV row identities, including positions of repeated headers."""
    names = spec["class_names"]
    for file_id, source in enumerate(spec["source_files"]):
        path = Path(source["path"])
        offset = 0
        if spec["name"] == "Car-Hacking":
            reader = pd.read_csv(
                path,
                header=None,
                names=list(range(12)),
                dtype=str,
                keep_default_na=False,
                chunksize=chunk_size,
            )
        else:
            header = pd.read_csv(
                path, nrows=0, encoding=spec["encoding"], encoding_errors="replace"
            ).columns
            mapping = {str(c).strip(): c for c in header}
            if len(mapping) != len(header):
                raise ValueError("Ambiguous source columns")
            pipeline = spec["pipeline"]
            originals = pipeline["original_feature_names"] if pipeline else spec["feature_names"]
            needed = originals + [spec["label_column"]] if with_features else [spec["label_column"]]
            reader = pd.read_csv(
                path,
                usecols=[mapping[c] for c in needed],
                encoding=spec["encoding"],
                encoding_errors="replace",
                low_memory=False,
                chunksize=chunk_size,
            )
        with reader:
            for frame in reader:
                rows = np.arange(offset, offset + len(frame), dtype=np.int64)
                offset += len(frame)
                if spec["name"] == "Car-Hacking":
                    labels, x = car_chunk(frame, path.name, with_features)
                else:
                    frame.columns = frame.columns.str.strip()
                    text = frame[spec["label_column"]].astype(str).str.strip()
                    valid = text.str.lower().ne(spec["label_column"].lower()).to_numpy()
                    frame, text, rows = frame.loc[valid], text[valid], rows[valid]
                    labels = text.map({s: label(s) for s in text.unique()}).to_numpy()
                    x = None
                    if with_features:
                        with np.errstate(over="ignore", invalid="ignore"):
                            raw = (
                                frame[originals]
                                .apply(pd.to_numeric, errors="coerce")
                                .to_numpy(np.float32)
                            )
                        raw[~np.isfinite(raw)] = np.nan
                        raw[raw == 0] = 0.0
                        if pipeline:
                            selected = pipeline["selected_original_features"]
                            x = np.column_stack(
                                (
                                    raw[:, [originals.index(c) for c in selected]],
                                    apply_recipes(raw, originals, pipeline["engineered_features"]),
                                )
                            )
                        else:
                            x = raw
                if not set(labels).issubset(names):
                    raise ValueError("Source labels differ from the frozen class space")
                y = np.asarray([names.index(s) for s in labels], np.int64)
                yield file_id, rows, y, x
        if offset != source["csv_data_rows"]:
            raise ValueError("Physical source row count changed")


def verify_spec(spec, content=True):
    for pin in spec["metadata_pins"] + [
        {"path": spec["data"], "sha256": spec["data_sha256"]},
        {"path": spec["excluded_identity_file"], "sha256": spec["excluded_identity_sha256"]},
    ]:
        if sha(pin["path"]) != pin["sha256"]:
            raise ValueError("Frozen input metadata changed: " + pin["path"])
    for source in spec["source_files"]:
        stat = Path(source["path"]).stat()
        if (
            stat.st_size != source["bytes"]
            or stat.st_mtime_ns != source["mtime_ns"]
            or (content and sha(source["path"]) != source["sha256"])
        ):
            raise ValueError("Raw source version changed: " + source["path"])


def training_spec(spec, fraction):
    counts = np.asarray(spec["available_train_counts"], dtype=np.int64)
    if counts.shape != (len(spec["class_names"]),) or (counts < 1).any():
        raise ValueError("Every declared class needs eligible TRAIN source identities")
    if fraction == "full":
        selected = counts
        description = (
            "all eligible original TRAIN source identities; known heldout and confirmation excluded"
        )
    elif fraction == "third":
        selected = np.maximum(1, counts // 3)
        if selected.tolist() != spec["third_train_counts"]:
            raise ValueError("Historical one-third quotas differ from source inventory")
        description = "floor(eligible source count / 3) per class, min1 if positive"
    else:
        raise ValueError("Unknown TRAIN source fraction")
    return {
        **spec,
        "selected_train_counts": selected.tolist(),
        "train_rows": int(selected.sum()),
        "train_fraction": fraction,
        "selection_description": description,
    }


def prepare_data(spec, out, chunk_size):
    out = Path(out)
    if out.exists():
        audit = read_json(out / "preparation.json")
        if audit["source_spec_sha256"] != spec_digest(spec) or audit["data_sha256"] != sha(
            out / "development.npz"
        ):
            raise ValueError("Prepared TRAIN cache differs")
        if audit["identity_sha256"] != sha(out / "selected_train_identities.npz"):
            raise ValueError("Prepared TRAIN identities changed")
        verify_spec(spec)
        return out / "development.npz"
    verify_spec(spec)
    names = spec["class_names"]
    exclusion = pd.read_csv(spec["excluded_identity_file"])
    excluded = {
        i: exclusion.loc[exclusion.source_file_id == i, "zero_based_data_row"].to_numpy(np.int64)
        for i in range(len(spec["source_files"]))
    }
    pools = [[] for _ in names]
    actual = np.zeros(len(names), np.int64)
    print(
        f"PREPARE {spec['name']}: {spec['train_fraction']} eligible TRAIN, target={spec['train_rows']:,}; heldout excluded",
        flush=True,
    )
    for fid, rows, y, _ in stream_source(spec, False, chunk_size):
        actual += np.bincount(y, minlength=len(names))
        eligible = ~np.isin(rows, excluded[fid])
        for c in np.unique(y[eligible]):
            codes = (np.uint64(fid) << np.uint64(48)) | rows[eligible & (y == c)].astype(np.uint64)
            pools[int(c)].append(codes)
    if actual.tolist() != spec["source_class_counts"]:
        raise ValueError("Source class supports changed")
    rng, chosen, chosen_labels = np.random.default_rng(42), [], []
    for c, chunks in enumerate(pools):
        pool = np.concatenate(chunks) if chunks else np.empty(0, np.uint64)
        if len(pool) != spec["available_train_counts"][c]:
            raise ValueError("Eligible class support differs from inventory")
        quota = spec["selected_train_counts"][c]
        if quota < 1:
            raise ValueError(
                "A declared class has no eligible TRAIN identities; no label may be invented"
            )
        chosen.append(pool if quota == len(pool) else rng.choice(pool, quota, replace=False))
        chosen_labels.append(np.full(quota, c, np.int64))
    del pools
    codes, y = np.concatenate(chosen), np.concatenate(chosen_labels)
    order = np.argsort(codes)
    codes, y = codes[order], y[order]
    if len(codes) != spec["train_rows"] or (np.diff(codes) == 0).any():
        raise ValueError("TRAIN source identity count or uniqueness differs")
    x = np.empty((len(y), len(spec["feature_names"])), np.float32)
    with np.load(spec["data"], allow_pickle=False) as old:
        if (
            old["class_names"].tolist() != names
            or old["feature_names"].tolist() != spec["feature_names"]
        ):
            raise ValueError("Frozen class/feature map differs")
        held = {r: (old["X_" + r], old["y_" + r]) for r in ("val", "cal")}
    identity_file = Path(spec["data"]).parent / "selected_source_rows.csv"
    if not identity_file.is_file():
        identity_file = next(
            Path(p["path"])
            for p in spec["metadata_pins"]
            if Path(p["path"]).name == "selected_source_rows.csv"
        )
    frozen_rows = pd.read_csv(identity_file)
    lookup = {}
    for role in ("val", "cal"):
        selected = frozen_rows.loc[frozen_rows.role == role].sort_values("row_in_role")
        if not np.array_equal(selected.row_in_role, np.arange(len(held[role][1]))):
            raise ValueError("Frozen heldout positions differ")
        for row in selected.to_dict("records"):
            key = (int(row["source_file_id"]) << 48) | int(row["zero_based_data_row"])
            lookup[key] = (role, int(row["row_in_role"]))
    replayed = 0
    found = np.zeros(len(y), bool)
    for fid, rows, labels, values in stream_source(spec, True, chunk_size):
        raw_codes = (np.uint64(fid) << np.uint64(48)) | rows.astype(np.uint64)
        positions = np.searchsorted(codes, raw_codes)
        matches = positions < len(codes)
        matches[matches] &= codes[positions[matches]] == raw_codes[matches]
        wanted = positions[matches]
        if not np.array_equal(labels[matches], y[wanted]):
            raise ValueError("Selected TRAIN label changed between passes")
        x[wanted] = values[matches]
        found[wanted] = True
        for i in np.flatnonzero(np.isin(raw_codes, np.fromiter(lookup, dtype=np.uint64))):
            role, position = lookup[int(raw_codes[i])]
            if labels[i] != held[role][1][position] or not np.array_equal(
                values[i], held[role][0][position], equal_nan=True
            ):
                raise ValueError("Frozen raw-to-feature replay differs: " + spec["name"])
            replayed += 1
    if not found.all() or replayed != sum(len(v[1]) for v in held.values()):
        raise ValueError("Incomplete feature extraction")
    keep = np.isfinite(np.fmin.reduce(x, axis=0)) & (
        np.fmin.reduce(x, axis=0) < np.fmax.reduce(x, axis=0)
    )
    if not keep.any():
        raise ValueError("No finite varying TRAIN feature")
    features = np.asarray(spec["feature_names"])[keep].tolist()
    permutation = rng.permutation(len(y))
    parts = {
        "train": (x[permutation][:, keep], y[permutation]),
        **{r: (a[:, keep], b) for r, (a, b) in held.items()},
    }
    out.mkdir(parents=True, exist_ok=False)
    data = out / "development.npz"
    np.savez_compressed(
        data,
        class_names=names,
        feature_names=features,
        **{f"X_{r}": a for r, (a, _) in parts.items()},
        **{f"y_{r}": b for r, (_, b) in parts.items()},
    )
    np.savez_compressed(
        out / "selected_train_identities.npz",
        source_file_id=(codes[permutation] >> np.uint64(48)).astype(np.int32),
        source_row=(codes[permutation] & np.uint64((1 << 48) - 1)).astype(np.int64),
        y=parts["train"][1],
    )
    verify_spec(spec)
    save_json(
        out / "preparation.json",
        {
            "source_spec_sha256": spec_digest(spec),
            "data_sha256": sha(data),
            "identity_sha256": sha(out / "selected_train_identities.npz"),
            "train_rows": len(y),
            "train_class_counts": np.bincount(y).tolist(),
            "test_scored": False,
            "confirmation_scored": False,
            "known_heldout_identities_excluded": len(exclusion),
            "duplicates_removed": 0,
            "feature_names": features,
            "dropped_train_constants": np.asarray(spec["feature_names"])[~keep].tolist(),
            "tiny_train_classes": [names[i] for i, n in enumerate(np.bincount(y)) if n < 10],
            "verified_frozen_heldout_source_rows": replayed,
            "fraction": spec["train_fraction"],
            "selection_description": spec["selection_description"],
            "all_eligible_train_included": spec["train_fraction"] == "full",
            "heldout_values_unchanged": True,
        },
    )
    print(
        f"PREPARED {spec['name']}: TRAIN={len(y):,}, VAL={len(held['val'][1]):,}, CAL={len(held['cal'][1]):,}, features={len(features)}",
        flush=True,
    )
    return data


def spec_digest(spec):
    import hashlib

    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def load_data(path):
    with np.load(path, allow_pickle=False) as data:
        if any(k.endswith("_test") for k in data.files):
            raise ValueError("Fit file must not contain TEST members")
        names, features = data["class_names"].tolist(), data["feature_names"].tolist()
        parts = {r: (data["X_" + r], data["y_" + r]) for r in ("train", "val", "cal")}
    if (
        len(names) < 2
        or names[0] != "BENIGN"
        or len(set(names)) != len(names)
        or len(set(features)) != len(features)
    ):
        raise ValueError("Unique class/feature names and BENIGN at class zero required")
    for r, (x, y) in parts.items():
        if (
            x.ndim != 2
            or x.shape != (len(y), len(features))
            or not np.issubdtype(y.dtype, np.integer)
            or not np.array_equal(np.unique(y), np.arange(len(names)))
        ):
            raise ValueError("All development roles must cover the frozen class space: " + r)
    return parts, names, features
