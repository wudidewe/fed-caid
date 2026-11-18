"""Build a chronological evaluation from the pinned CIC-UNSW-NB15 flow CSV."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .data_io import stream_source
from .dataset_preparation import _verify_raw, check_raw_sources
from .revision_data import (
    ROLES,
    RevisionDataset,
    _audit_exact_inputs,
    _class_counts,
    _feature_keys,
    _matching_positions,
)

DATASET = "CIC-UNSW-NB15"
TIMESTAMP_FORMAT = "%d/%m/%Y %I:%M:%S %p"
ROLE_WINDOWS = {
    "train": ("2015-01-22T00:00:00", "2015-01-23T00:00:00"),
    "cal": ("2015-02-17T00:00:00", "2015-02-17T22:00:00"),
    "val": ("2015-02-17T22:00:00", "2015-02-18T00:00:00"),
    "test": ("2015-02-18T00:00:00", "2015-02-19T00:00:00"),
}


def _timestamp_roles(source, encoding, chunk_size):
    """Read one timestamp per physical CSV row, independent of file ordering."""
    options = {"encoding": encoding, "encoding_errors": "replace"}
    header = pd.read_csv(source["path"], nrows=0, **options).columns
    columns = [name for name in header if name.strip() == "Timestamp"]
    if len(columns) != 1:
        raise ValueError("Temporal CIC-UNSW-NB15 needs one Timestamp column")

    roles = np.full(source["csv_data_rows"], -1, dtype=np.int8)
    observed = {}
    windows = [
        (role, np.datetime64(start), np.datetime64(stop)) for role, (start, stop) in ROLE_WINDOWS.items()
    ]
    offset = 0
    with pd.read_csv(source["path"], usecols=columns, dtype=str, chunksize=chunk_size, **options) as reader:
        for frame in reader:
            stamps = pd.to_datetime(frame[columns[0]].str.strip(), format=TIMESTAMP_FORMAT)
            stamps = stamps.to_numpy(dtype="datetime64[s]")
            end = offset + len(stamps)
            if end > len(roles):
                raise ValueError("Timestamp count exceeds the pinned physical row count")
            block = roles[offset:end]
            for index, (role, start, stop) in enumerate(windows):
                mask = (stamps >= start) & (stamps < stop)
                if mask.any():
                    block[mask] = index
                    earliest, latest = stamps[mask].min(), stamps[mask].max()
                    previous = observed.get(role, (earliest, latest))
                    observed[role] = min(earliest, previous[0]), max(latest, previous[1])
            offset = end

    if offset != len(roles) or (roles < 0).any() or set(observed) != set(ROLES):
        raise ValueError("Pinned timestamps must all belong to the declared nonempty temporal roles")
    return roles, {role: [str(stamp) for stamp in observed[role]] for role in ROLES}


def _purge_temporal_inputs(parts, names, chunk_size):
    """Keep earlier-role ownership of exact inputs, without copying all TRAIN keys.

    Only evaluation keys share a sorted lookup table. TRAIN is compared in
    bounded chunks. Every raw earlier-role input blocks its later occurrences,
    including a CAL conflict group discarded before fitting or calibration.
    """
    evaluation_roles = ROLES[1:]
    lengths = [len(parts[role][1]) for role in evaluation_roles]
    feature_bytes = parts["train"][0].shape[1] * np.dtype("<f4").itemsize
    raw_keys = np.empty(sum(lengths), dtype=np.dtype((np.void, feature_bytes)))
    offset = 0
    for role in evaluation_roles:
        x = parts[role][0]
        for start in range(0, len(x), chunk_size):
            end = min(start + chunk_size, len(x))
            raw_keys[offset + start : offset + end] = _feature_keys(x[start:end])
        offset += len(x)
    keys, inverse = np.unique(raw_keys, return_inverse=True)
    del raw_keys
    groups = dict(zip(evaluation_roles, np.split(inverse, np.cumsum(lengths)[:-1])))
    counts = {role: np.bincount(groups[role], minlength=len(keys)) for role in evaluation_roles}
    counts["train"] = np.zeros(len(keys), dtype=np.int64)
    train = parts["train"][0]
    for start in range(0, len(train), chunk_size):
        _, matches = _matching_positions(keys, _feature_keys(train[start : start + chunk_size]))
        counts["train"] += np.bincount(matches, minlength=len(keys))

    overlaps = {}
    for index, role in enumerate(ROLES):
        for other in ROLES[index + 1 :]:
            shared = (counts[role] > 0) & (counts[other] > 0)
            overlaps[f"{role}:{other}"] = {
                "unique_shared_inputs": int(shared.sum()),
                f"{role}_rows": int(counts[role][shared].sum()),
                f"{other}_rows": int(counts[other][shared].sum()),
            }

    keep = {"train": np.ones(len(train), dtype=bool)}
    removed_earlier = {"train": 0}
    seen = counts["train"] > 0
    for role in evaluation_roles:
        keep[role] = ~seen[groups[role]]
        removed_earlier[role] = int((~keep[role]).sum())
        # Ownership uses the raw role, even if CAL conflict filtering drops it.
        seen |= counts[role] > 0

    cal_groups, cal_labels = groups["cal"], parts["cal"][1]
    low = np.full(len(keys), len(names), dtype=np.int64)
    high = np.full(len(keys), -1, dtype=np.int64)
    np.minimum.at(low, cal_groups, cal_labels)
    np.maximum.at(high, cal_groups, cal_labels)
    unique, first = np.unique(cal_groups, return_index=True)
    novel = keep["cal"][first]
    conflicting = low[unique] != high[unique]
    cal_keep = np.zeros(len(cal_labels), dtype=bool)
    cal_keep[first[novel & ~conflicting]] = True
    conflict_rows = int(counts["cal"][unique[novel & conflicting]].sum())
    duplicate_rows = int(keep["cal"].sum() - conflict_rows - cal_keep.sum())
    keep["cal"] = cal_keep
    report = {
        "enabled": True,
        "role_order": list(ROLES),
        "comparison": "Canonical ordered pre-normalization float32 feature bytes; labels excluded",
        "cross_role_rule": "Every raw earlier-role input blocks that input in every later role",
        "within_role_rule": "TRAIN, VAL and raw TEST retain their within-role repetitions",
        "cal_rule": (
            "Discard conflicting-label CAL groups and retain the first source row per remaining input"
        ),
        "cal_fit_hold_exact_input_disjoint_by_construction": True,
        "selection": (
            "Cross-role purge uses feature identity only; CAL conflicts use CAL labels only; no TEST label"
            " selects training or calibration rows"
        ),
        "removed_matching_earlier_role_rows": removed_earlier,
        "cal_conflicting_groups_excluded": int((novel & conflicting).sum()),
        "cal_conflicting_rows_excluded": conflict_rows,
        "cal_duplicate_rows_excluded": duplicate_rows,
        "raw_cross_role_exact_overlap": overlaps,
        "scope_limit": "Does not check near-duplicates, sessions or equality after imputation",
    }
    return keep, report


def load_temporal_data(dataset_root, *, chunk_size=250000, purge_exact=False):
    """Load ordered development days and the later TEST day without date features."""
    if chunk_size < 1:
        raise ValueError("Chunk size must be positive")
    spec = check_raw_sources(DATASET, Path(dataset_root))
    names, features = spec["class_names"], spec["feature_names"]
    (source,) = spec["source_files"]
    roles, observed = _timestamp_roles(source, spec["encoding"], chunk_size)
    counts = np.bincount(roles, minlength=len(ROLES))
    parts = {
        role: (np.empty((count, len(features)), dtype=np.float32), np.empty(count, dtype=np.int64))
        for role, count in zip(ROLES, counts, strict=True)
    }
    digests = {role: hashlib.sha256() for role in ROLES}
    positions = dict.fromkeys(ROLES, 0)
    offset = 0
    for file_id, rows, labels, values in stream_source(spec, True, chunk_size):
        if file_id != 0 or not np.array_equal(rows, np.arange(offset, offset + len(rows))):
            raise ValueError("Feature rows do not match the physical timestamp row identities")
        offset += len(rows)
        block_roles = roles[rows]
        for index, role in enumerate(ROLES):
            mask = block_roles == index
            start = positions[role]
            end = start + int(mask.sum())
            parts[role][0][start:end] = values[mask]
            parts[role][1][start:end] = labels[mask]
            digests[role].update(np.ascontiguousarray(rows[mask], dtype="<u8"))
            positions[role] = end

    if offset != len(roles) or list(positions.values()) != counts.tolist():
        raise ValueError("Temporal feature replay did not fill the declared role identities")
    source_counts = sum(np.bincount(parts[role][1], minlength=len(names)) for role in ROLES)
    if source_counts.tolist() != spec["source_class_counts"]:
        raise ValueError("Raw label counts differ from the pinned source recipe")
    _verify_raw(source)
    original_digests = {role: digest.hexdigest() for role, digest in digests.items()}
    purge = {"enabled": False}
    if purge_exact:
        original_support = {role: _class_counts(parts[role][1], names) for role in ROLES}
        keep, purge = _purge_temporal_inputs(parts, names, chunk_size)
        purge.update(
            role_rows_before=dict(positions),
            role_class_support_before=original_support,
            source_identity_sha256_by_role_before=original_digests,
            observed_role_time_min_max_before=observed,
        )
        for index, role in enumerate(ROLES):
            retained = keep[role]
            source_rows = np.flatnonzero(roles == index)
            digests[role] = hashlib.sha256(np.ascontiguousarray(source_rows[retained], dtype="<u8"))
            if not retained.all():
                x, y = parts[role]
                parts[role] = x[retained], y[retained]
                del x, y
            positions[role] = int(retained.sum())
            if not positions[role]:
                raise ValueError(f"Exact temporal purge leaves {role.upper()} empty")
        if positions["cal"] < 2:
            raise ValueError(
                "Exact temporal purge needs at least two distinct CAL inputs for fit and holdout"
            )
        del keep
        purge.update(
            role_rows_after=dict(positions),
            role_class_support_after={role: _class_counts(parts[role][1], names) for role in ROLES},
            source_identity_sha256_by_role_after={
                role: digest.hexdigest() for role, digest in digests.items()
            },
            removed_rows_by_role={role: purge["role_rows_before"][role] - positions[role] for role in ROLES},
        )
    exact, novel = _audit_exact_inputs(parts, names, chunk_size)
    if purge_exact and any(
        pair["unique_shared_inputs"] for pair in exact["cross_role_exact_overlap"].values()
    ):
        raise ValueError("Exact temporal purge left cross-role feature overlap")
    return RevisionDataset(
        parts,
        names,
        features,
        {
            "split_protocol": "temporal_purged" if purge_exact else "temporal",
            "evaluation_scope": "later-day holdout within the same pinned benchmark, not an external capture",
            "external_captures": False,
            "timestamp_format": TIMESTAMP_FORMAT,
            "timezone": "not recorded in the pinned CSV",
            "timestamp_used_as_model_feature": False,
            "role_time_windows_left_closed_right_open": ROLE_WINDOWS,
            "observed_role_time_min_max": observed,
            "observed_role_time_min_max_scope": "Original source roles before optional exact-input purge",
            "calibration_scope": "CAL fit and threshold holdout stay within the earlier CAL window",
            "source_recipe_sha256": spec["recipe_sha256"],
            "source_files_sha256": {source["filename"]: source["sha256"]},
            "source_identity_disjoint_by_construction": True,
            "source_identity_encoding": (
                "Little-endian uint64 physical CSV row index in role row order, file ID zero"
            ),
            "source_identity_sha256_by_role": {role: digest.hexdigest() for role, digest in digests.items()},
            "role_rows": positions,
            "role_class_support": {role: _class_counts(parts[role][1], names) for role in ROLES},
            "nb15_data_csv_label_csv_used": False,
            "exact_feature_audit": exact,
            "temporal_exact_purge": purge,
        },
        novel,
    )
