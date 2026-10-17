"""Parse raw CSV streams and load frozen development data."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


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
