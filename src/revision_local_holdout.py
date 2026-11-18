"""Audit local evaluation exposure and class support from verified artifacts."""

import csv
import json


NB15 = "CIC-UNSW-NB15"
DATASETS = ("CIC-IDS-2017", NB15, "CIC-IoV-2024", "Car-Hacking")
ROLES = (("VAL", "val"), ("TEST", "test"), ("TEST_EXACT_UNIQUE", "test_unique_unseen"))
CURRENT_SCOPE = "current_grouped"
HISTORICAL_SCOPE = "historical_temporal"
AGGREGATE_ROLES = ("val", "test")


def _read_metrics(path):
    """Read recorded evaluation identities, leaving score values unused."""
    if not path.is_file():
        return None
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def _support(audit, role):
    """Use row support for VAL/TEST and exact-input group support for unique TEST."""
    if role in AGGREGATE_ROLES:
        counts = audit["role_class_support"][role]
    else:
        counts = audit["exact_feature_audit"]["test_exact_input_unseen_deduplicated"]
    roster = audit["role_class_support"].get("train", counts)
    if (
        "BENIGN" not in counts
        or counts.keys() != roster.keys()
        or any(type(value) is not int or value < 0 for value in counts.values())
    ):
        raise ValueError(f"Invalid audited class support for {role}")
    if role in AGGREGATE_ROLES and sum(counts.values()) != audit["role_rows"][role]:
        raise ValueError(f"Audited class support disagrees with the row count for {role}")
    return counts


def _role_fields(audit, role):
    """Preserve zeroes and distinguish row counts from feature-group counts."""
    support = _support(audit, role)
    attacks = {name: count for name, count in support.items() if name != "BENIGN"}
    exact = audit["exact_feature_audit"]
    deduplicated = role == "test_unique_unseen"
    return {
        "support_unit": "exact_input_groups" if deduplicated else "rows",
        "support": sum(support.values()),
        "benign_support": support["BENIGN"],
        "attack_support": sum(attacks.values()),
        "supported_attack_classes": sum(count > 0 for count in attacks.values()),
        "missing_attack_classes": ", ".join(name for name, count in attacks.items() if count == 0),
        "attack_classes_with_1_to_4_units": ", ".join(
            name for name, count in attacks.items() if 0 < count <= 4
        ),
        "class_support": json.dumps(support, sort_keys=True),
        "role_feature_groups": (
            sum(support.values()) if deduplicated else exact["within_role_repeats"][role]["distinct_inputs"]
        ),
        "ordered_source_identity_sha256": audit.get("source_identity_sha256_by_role", {}).get(role),
    }


def _observed_seeds(metrics, dataset, protocol, role, expected_support, scenario):
    """Identify executed evaluations using row, role and experiment identity only."""
    if metrics is None:
        return None
    matching = (
        row
        for row in metrics
        if (row["dataset"], row["protocol"], row["scenario"], row["role"])
        == (dataset, protocol, scenario, role)
    )
    seeds = set()
    for row in matching:
        if int(row["rows"]) != expected_support:
            raise ValueError(f"Recorded evaluation rows differ from the {dataset} audit for {role}")
        if expected_support > 0:
            seeds.add(int(row["seed"]))
    return seeds


def _temporal_history(temporal):
    """Bind an explicitly supplied temporal run to its own known evaluation roles."""
    if temporal is None:
        return None
    plan, audit, metrics = (temporal[key] for key in ("run", "audit", "metrics"))
    scenarios = [
        (row["dataset"], row["protocol"], case[0]) for row in plan["matrix"] for case in row["scenarios"]
    ]
    if (
        plan.get("synthetic_only") is not False
        or plan.get("data_only") is not False
        or scenarios != [(NB15, "temporal", "supplement_temporal")]
    ):
        raise ValueError("Expected a recorded real NB15 temporal evaluation plan")
    if (
        audit.get("split_protocol") not in {"temporal", "temporal_purged"}
        or audit.get("external_captures") is not False
    ):
        raise ValueError("Expected a local NB15 temporal data-role audit")
    planned = set(plan["seeds"])
    observed = _temporal_seeds(audit, metrics, planned)
    return audit, observed, temporal["source"]


def _temporal_seeds(audit, metrics, planned):
    """Confirm role counts and seed identifiers against the supplied past plan."""
    metrics = tuple(metrics) if metrics is not None else None
    observed = {
        role: _observed_seeds(
            metrics, NB15, "temporal", role, sum(_support(audit, role).values()), "supplement_temporal"
        )
        for _, role in ROLES
    }
    if not planned or any(seeds is not None and not seeds <= planned for seeds in observed.values()):
        raise ValueError("Historical NB15 temporal evaluations differ from the recorded seeds")
    return observed


def _scoring_status(seeds):
    """Distinguish an absent result table from observed missing evaluations."""
    if seeds is None:
        return "unknown"
    return "yes" if seeds else "no"


def _base_row(dataset, scope, protocol, label, artifact):
    """No same-source local partition qualifies as external capture confirmation."""
    return {
        "dataset": dataset,
        "evidence_scope": scope,
        "protocol": protocol,
        "role": label,
        "independent_confirmation_eligible": False,
        "source_artifact": artifact,
    }


def _historical_rows(history):
    """Present past temporal scoring as separate within-source exposure evidence."""
    if history is None:
        return [
            {
                **_base_row(NB15, HISTORICAL_SCOPE, "temporal", "TEST", None),
                "scored_in_current_run": "no",
                "scored_in_verified_history": "unknown",
                "exposure_status": "historical_records_unavailable",
                "note": "A verified historical temporal evaluation record was unavailable.",
            }
        ]
    audit, observed, artifact = history
    return [
        {
            **_base_row(NB15, HISTORICAL_SCOPE, audit["split_protocol"], label, artifact),
            **_role_fields(audit, role),
            "scored_in_current_run": "no",
            "scored_in_verified_history": "yes" if observed[role] else "unknown",
            "seeds_scored_in_verified_history": len(observed[role]) if observed[role] is not None else None,
            "exposure_status": (
                "verified_historical_evaluation" if observed[role] else "historical_exposure_unknown"
            ),
            "note": "Later-day role from the same pinned benchmark capture.",
        }
        for label, role in ROLES
    ]


def _same_source(audit, previous):
    """A matching nonempty source file inventory establishes source identity only."""
    sources = audit.get("source_files_sha256")
    return bool(sources) and sources == previous.get("source_files_sha256")


def _exposure_status(audit, role, history):
    """Use matching ordered-role digests only when source identity also matches."""
    if history is None or not _same_source(audit, history[0]):
        return "historical_exposure_unknown", None
    previous, observed, _ = history
    identity = audit.get("source_identity_sha256_by_role", {}).get(role)
    for previous_role, previous_identity in previous.get("source_identity_sha256_by_role", {}).items():
        if identity and identity == previous_identity and observed.get(previous_role):
            return "same_ordered_role_previously_scored", observed[previous_role]
    return "same_source_overlap_unknown", None


def _current_row(dataset, label, role, audit, metrics, artifact, history):
    """Report local reserve support separately from evidence of model exposure."""
    fields = _role_fields(audit, role)
    observed = _observed_seeds(metrics, dataset, "grouped", role, fields["support"], "main")
    status, history_seeds = _exposure_status(audit, role, history)
    return {
        **_base_row(dataset, CURRENT_SCOPE, "grouped", label, artifact),
        **fields,
        "scored_in_current_run": _scoring_status(observed),
        "seeds_scored_in_current_run": len(observed) if observed is not None else None,
        "scored_in_verified_history": "yes" if history_seeds else "unknown",
        "seeds_scored_in_verified_history": len(history_seeds) if history_seeds else None,
        "exposure_status": status,
        "grouped_test_rows_from_original_train_role": (
            audit.get("grouped_resplit", {}).get("test_rows_formerly_in_original_train")
            if role == "test"
            else None
        ),
        "note": (
            "Current grouped same-capture reserve."
            if role != "val"
            else "Grouped same-capture development VAL."
        ),
    }


def _current_dataset(tables, dataset, history):
    """Use the current matrix audit or the current dedicated IoV audit if available."""
    stages = ("benchmark_matrix", "iov_diagnosis") if dataset == "CIC-IoV-2024" else ("benchmark_matrix",)
    relative = f"{dataset}/grouped/data_audit.json"
    for stage in stages:
        path = tables / stage / relative
        if path.is_file():
            audit = json.loads(path.read_text(encoding="utf-8-sig"))
            if audit.get("split_protocol") != "grouped" or audit.get("external_captures") is not False:
                raise ValueError(f"Expected the local grouped audit for {dataset}")
            metrics = _read_metrics(tables / stage / "metrics_by_seed.csv")
            return [
                _current_row(dataset, label, role, audit, metrics, f"tables/{stage}/{relative}", history)
                for label, role in ROLES
            ]
    return [
        {
            **_base_row(dataset, CURRENT_SCOPE, "grouped", "TEST", None),
            "scored_in_current_run": "unknown",
            "scored_in_verified_history": "unknown",
            "exposure_status": "current_audit_unavailable",
            "note": "A current verified grouped data-role audit was unavailable.",
        }
    ]


def holdout_rows(tables, temporal=None):
    """Create exposure and support rows from already verified report inputs.

    ``tables`` is the caller's verified current-run report table directory.
    The optional ``temporal`` record supplies verified ``run``, ``audit`` and
    ``metrics`` content with a display ``source`` path for the historical run.
    No models, raw inputs, or automatically discovered historic paths are used.
    """
    history = _temporal_history(temporal)
    current = [
        row
        for dataset in DATASETS
        for row in _current_dataset(tables, dataset, history if dataset == NB15 else None)
    ]
    return current + _historical_rows(history)
