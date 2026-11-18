"""Validate public artifacts and metadata for the MPC runtime."""

from __future__ import annotations

from pathlib import Path


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
    forbidden.update({
        "attack_subtype_calibration",
        "subtype_bias",
        "train_probabilities",
        "train_labels",
        "attack_class_counts",
        "train_prediction_cache",
    })
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
        "secure_share_construction": (
            "conditional mathematical construction; protocol test evidence reported separately"
        ),
        "differential_privacy": False if mode == "mpc" else "separate accountant required",
        "member_inference_resistance": "not_established",
        "collector_noncollusion": "assumed_not_verified",
        "host_admin_isolation": "not_provided",
        "authenticated_remote_transport": "not_provided",
        "actual_blockchain_consensus": "not_deployed",
        "scope": "runtime artifact and metadata checks; not a cryptographic audit or empirical attack proof",
    }
