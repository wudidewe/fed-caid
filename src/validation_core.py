"""Explicit development validation, privacy scope and exact feature-overlap audit."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def _indices(value, size):
    x = np.asarray(value)
    if (
        x.ndim != 1
        or not np.issubdtype(x.dtype, np.integer)
        or (x < 0).any()
        or (x >= size).any()
        or len(np.unique(x)) != len(x)
    ):
        raise ValueError("Invalid or duplicate calibration indices")
    return x


def three_way_calibration(size, fit, threshold, seed):
    """Separate selection from adaptation; preserve the old threshold holdout."""
    fit, threshold = _indices(fit, size), _indices(threshold, size)
    if (
        size < 3
        or len(fit) < 2
        or len(threshold) < 1
        or not np.array_equal(np.sort(np.concatenate((fit, threshold))), np.arange(size))
    ):
        raise ValueError("Reference CAL split must be complete and disjoint")
    order = np.random.default_rng(seed).permutation(fit)
    count = min(len(fit) - 1, max(1, int(0.6 * len(fit))))
    return {
        "adapt": order[:count],
        "select": order[count:],
        "threshold": threshold.copy(),
        "final_refit": fit.copy(),
    }


def noninferiority_gate(reference, candidate, *, require_improvement=False, protect_classes=True):
    """Strict observed-data gate, not a statistical noninferiority theorem."""
    if reference.get("class_names") != candidate.get("class_names") or reference.get(
        "rows"
    ) != candidate.get("rows"):
        raise ValueError("Guard requires identical row and class spaces")
    comparisons = {}
    for key in ("accuracy", "macro_f1", "weighted_f1"):
        a, b = float(reference[key]), float(candidate[key])
        if not math.isfinite(a) or not math.isfinite(b):
            raise ValueError("Guard metrics must be finite")
        comparisons[key] = {"reference": a, "candidate": b, "pass": b >= a - 1e-12}
    for key in ("far", "fnr"):
        a, b = reference[key], candidate[key]
        good = a is None and b is None
        if a is not None and b is not None:
            if not math.isfinite(float(a)) or not math.isfinite(float(b)):
                raise ValueError("Guard error rates must be finite")
            good = b <= a + 1e-12
        comparisons[key] = {"reference": a, "candidate": b, "pass": bool(good)}
    if protect_classes:
        for name in reference["class_names"]:
            a, b = reference["per_class"][name], candidate["per_class"][name]
            if a["support"] != b["support"]:
                raise ValueError("Class support changed across guard arms")
            if not all(math.isfinite(float(v)) and 0 <= v <= 1 for v in (a["recall"], b["recall"])):
                raise ValueError("Class recalls must be finite probabilities")
            comparisons["recall:" + name] = {
                "reference": a["recall"],
                "candidate": b["recall"],
                "pass": b["recall"] >= a["recall"] - 1e-12,
            }
    if require_improvement:
        comparisons["macro_improvement"] = {
            "pass": candidate["macro_f1"] > reference["macro_f1"] + 1e-6
        }
    failed = [k for k, v in comparisons.items() if not v["pass"]]
    return {
        "passed": not failed,
        "failed_checks": failed,
        "comparisons": comparisons,
        "guarantee": "observed development rows only; no future-data or statistical guarantee",
        "tolerance": 1e-12,
    }


def validate_role_coverage(partitions, size):
    """Linear-time full roster check; avoid sorting millions of TRAIN indices."""
    seen = np.zeros(size, dtype=bool)
    total = 0
    for rows in partitions:
        rows = np.asarray(rows)
        if (
            rows.ndim != 1
            or not np.issubdtype(rows.dtype, np.integer)
            or (rows < 0).any()
            or (rows >= size).any()
            or seen[rows].any()
        ):
            raise ValueError("Reference client roles are not complete and mutually exclusive")
        seen[rows] = True
        total += len(rows)
    # Internal duplicates necessarily leave a gap when the row total is fixed.
    if total != size or not seen.all():
        raise ValueError("Reference client roles are not complete and mutually exclusive")
    return True


def complete_metrics(metrics):
    """Add aggregate recalls without changing legacy checkpoint replay metrics."""
    result = dict(metrics)
    rows = [metrics["per_class"][c] for c in metrics["class_names"]]
    support = np.asarray([r["support"] for r in rows], dtype=float)
    recall = np.asarray([r["recall"] for r in rows], dtype=float)
    result["macro_recall"] = float(recall.mean())
    result["weighted_recall"] = (
        float(np.dot(support, recall) / support.sum()) if support.sum() else 0.0
    )
    return result


def strict_quality_gate(reference, candidate, *, require_improvement=False):
    """Protect all requested P/R/F1 metrics and every class, on observed rows."""
    reference, candidate = complete_metrics(reference), complete_metrics(candidate)
    result = noninferiority_gate(reference, candidate, require_improvement=require_improvement)
    checks = result["comparisons"]
    for key in ("macro_precision", "weighted_precision", "macro_recall", "weighted_recall"):
        a, b = float(reference[key]), float(candidate[key])
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in (a, b)):
            raise ValueError("Precision/recall must be finite probabilities")
        checks[key] = {"reference": a, "candidate": b, "pass": b >= a - 1e-12}
    for name in reference["class_names"]:
        for key in ("precision", "f1-score"):
            a, b = float(reference["per_class"][name][key]), float(
                candidate["per_class"][name][key]
            )
            if not all(math.isfinite(x) and 0 <= x <= 1 for x in (a, b)):
                raise ValueError("Per-class metrics must be finite probabilities")
            checks[key + ":" + name] = {"reference": a, "candidate": b, "pass": b >= a - 1e-12}
    result["failed_checks"] = [k for k, v in checks.items() if not v["pass"]]
    result["passed"] = not result["failed_checks"]
    return result


def validate_aggregate(clipped_updates, recovered, scale):
    matrix = np.asarray(clipped_updates, dtype=np.float64)
    recovered = np.asarray(recovered, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[0] < 3
        or recovered.shape != (matrix.shape[1],)
        or not np.isfinite(matrix).all()
        or not np.isfinite(recovered).all()
        or type(scale) is not int
        or scale < 1
    ):
        raise ValueError("Invalid private aggregate verification inputs")
    reference = matrix.mean(0)
    quantized = np.rint(matrix * scale).sum(0) / (scale * len(matrix))
    error = float(np.max(np.abs(recovered - reference)))
    exact_error = float(np.max(np.abs(recovered - quantized)))
    tolerance = 8 * np.finfo(float).eps * max(1.0, float(np.abs(matrix).max()))
    if error > 0.5 / scale + tolerance or exact_error > tolerance:
        raise RuntimeError("Secure aggregate fails private arithmetic reference verification")
    return {
        "passed": True,
        "max_coordinate_error": error,
        "mean_coordinate_error_bound": 0.5 / scale,
        "floating_point_tolerance": tolerance,
        "max_error_vs_quantized_mean": exact_error,
        "scope": "trusted researcher private reference; no individual updates in public transcript",
    }


def privacy_scope_validation(public, mode="mpc"):
    """Validate artifact scope; never turn a functional check into a security proof."""
    import json

    public = Path(public)
    forbidden = {
        "share",
        "shares",
        "individual_update",
        "raw_update",
        "local_update",
        "log_prior",
        "alarm_threshold",
        "state_dict",
        "cal_labels",
        "val_labels",
        "X_train",
        "y_train",
        "partition_indices_sha256",
        "backbone_state_sha256",
    }
    forbidden.update(
        {
            "attack_subtype_calibration",
            "subtype_bias",
            "train_probabilities",
            "train_labels",
            "attack_class_counts",
            "train_prediction_cache",
        }
    )
    violations = []

    def walk(value, path):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in forbidden:
                    violations.append(path + ":" + key)
                walk(item, path)
        elif isinstance(value, list):
            for item in value:
                walk(item, path)

    for path in public.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix in {".pt", ".npz", ".npy"} or "client_" in path.name:
            violations.append(path.name)
        if path.suffix == ".json":
            walk(json.loads(path.read_text(encoding="utf-8")), path.name)
        elif path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                walk(json.loads(line), path.name)
    protocol = json.loads((public / "protocol.json").read_text(encoding="utf-8"))
    if mode == "mpc" and (
        protocol.get("differential_privacy") is not False
        or protocol.get("epsilon") is not None
        or protocol.get("delta") is not None
    ):
        violations.append("MPC falsely declares DP accounting")
    if violations:
        raise RuntimeError("Public privacy boundary validation failed: " + ", ".join(violations))
    return {
        "public_individual_artifact_scan": "pass",
        "violations": [],
        "secure_share_construction": "conditional mathematical construction; protocol test evidence reported separately",
        "differential_privacy": False if mode == "mpc" else "separate accountant required",
        "member_inference_resistance": "not_established",
        "collector_noncollusion": "assumed_not_verified",
        "host_admin_isolation": "not_provided",
        "authenticated_remote_transport": "not_provided",
        "actual_blockchain_consensus": "not_deployed",
        "scope": "runtime artifact and metadata checks; not a cryptographic audit or empirical attack proof",
    }


def _canonical_rows(x):
    x = np.asarray(x, dtype=np.float32).copy(order="C")
    x[x == 0] = 0.0  # signed zero is one feature value
    x[np.isnan(x)] = np.nan
    return x


def feature_overlap_audit(train, held, chunk=65536):
    """Count exact held feature rows in TRAIN without changing data or roles.

    Feature repetition alone does not establish source identity or temporal leakage.
    """
    train, held = np.asarray(train), _canonical_rows(held)
    if train.ndim != 2 or held.ndim != 2 or train.shape[1] != held.shape[1] or chunk < 1:
        raise ValueError("Aligned feature matrices required")
    width = held.shape[1]
    rng = np.random.default_rng(72041)
    weights = rng.integers(1, 2**63, size=(2, width), dtype=np.uint64)
    keytype = np.dtype([("a", "<u8"), ("b", "<u8")])

    def keys(x):
        bits = x.view(np.uint32).astype(np.uint64)
        result = np.empty(len(x), dtype=keytype)
        result["a"] = (bits * weights[0]).sum(1, dtype=np.uint64)
        result["b"] = (bits * weights[1]).sum(1, dtype=np.uint64)
        return result

    held_keys = keys(held)
    pending = {}
    for i, key in enumerate(held_keys):
        pending.setdefault(key.tobytes(), []).append(i)
    found = np.zeros(len(held), dtype=bool)
    scanned = 0
    for start in range(0, len(train), chunk):
        if not pending:
            break
        values = _canonical_rows(train[start : start + chunk])
        candidate_keys = keys(values)
        wanted = np.frombuffer(b"".join(pending), dtype=keytype)
        hits = np.flatnonzero(np.isin(candidate_keys, wanted))
        if len(hits):
            unique, at = np.unique(candidate_keys[hits], return_index=True)
            for key, index in zip(unique, hits[at]):
                rawkey = key.tobytes()
                matches = pending.get(rawkey, [])
                for row in matches:
                    if np.array_equal(values[index], held[row], equal_nan=True):
                        found[row] = True
                    else:
                        # Check every hash match because distinct feature rows can collide.
                        colliding = hits[candidate_keys[hits] == key]
                        found[row] = any(
                            np.array_equal(values[j], held[row], equal_nan=True) for j in colliding
                        )
                remaining = [row for row in matches if not found[row]]
                if remaining:
                    pending[rawkey] = remaining
                else:
                    pending.pop(rawkey, None)
        scanned += len(values)
    return {
        "held_rows": len(held),
        "held_rows_with_exact_train_features": int(found.sum()),
        "overlap_fraction": float(found.mean()) if len(found) else None,
        "train_rows_scanned": scanned,
        "source_identity_independence": "not_established_by_feature_comparison",
        "time_or_capture_session_independence": "not_established",
        "interpretation": "exact feature repetition is a diagnostic; it alone does not prove data leakage",
    }
