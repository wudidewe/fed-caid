"""Freeze and evaluate prespecified NB15 thresholds on one reference detector."""

import numpy as np

from .revision_nb15_low_far_evaluation import (
    FAR_CAP,
    FNR_INCREASE_LIMIT,
    MIN_CAL_BENIGN_ROWS,
    SCOPES,
    _binary_metrics,
    _class_rows,
    _classification_metrics,
    _decide,
    _difference,
    _paired_stats,
    _pool,
    _scope_check,
    _validate_role,
)
from .revision_report import summarize
from .utils import empirical_alarm_threshold


REFERENCE = "reference_current"
MATCHED = "matched_cal_1pct"
CONSERVATIVE = "conservative_cal_0_9pct"
ARMS = (REFERENCE, MATCHED, CONSERVATIVE)
NOMINAL_FARS = {REFERENCE: FAR_CAP, MATCHED: FAR_CAP, CONSERVATIVE: 0.009}
PAIRS = ((MATCHED, REFERENCE), (CONSERVATIVE, REFERENCE), (CONSERVATIVE, MATCHED))
MIN_ASSESSMENT_SEEDS = 5
TARGET_CLASSES = ("Exploits", "Analysis")
CLASS_METRICS = ("binary_recall", "recall", "f1")
CAL_ROLES = ("cal_fit", "cal_check")
COUNT_KEYS = (
    "rows", "benign_rows", "attack_rows", "true_positives", "true_negatives",
    "false_positives", "false_negatives",
)


def _split_indices(subsets, size):
    """Validate the native CAL partition before using its aligned row indices."""
    fit, check = (np.asarray(subsets[key]) for key in ("threshold_fit", "operating_select"))
    for rows in (fit, check):
        if rows.ndim != 1 or not rows.size or rows.dtype.kind not in "ui":
            raise ValueError("NB15 threshold-fit and CAL-check need integer row indices")
    joined = np.concatenate((fit, check))
    if len(joined) != size or not np.array_equal(np.sort(joined), np.arange(size)):
        raise ValueError("NB15 threshold-fit and CAL-check must partition the parent holdout without overlap")
    return fit, check


def _numbers(role, metrics):
    """Prefix output columns to distinguish threshold fitting from checking."""
    return {f"{role}_{key}": value for key, value in metrics.items()}


def _freeze_client(labels, probabilities, fit, check, arm):
    """Choose the threshold from fit rows, then measure it unchanged on CAL-check."""
    independent = arm != REFERENCE
    fit_p, fit_y = (probabilities[fit], labels[fit]) if independent else (probabilities, labels)
    threshold, source = empirical_alarm_threshold(
        fit_p, fit_y, NOMINAL_FARS[arm], floor=0,
        min_benign_rows=MIN_CAL_BENIGN_ROWS, insufficient_policy="disable",
    )
    fit_metrics = _binary_metrics(_decide(fit_p, fit_y, threshold))
    check_metrics = _binary_metrics(_decide(probabilities[check], labels[check], threshold))
    return {
        **source,
        "nominal_cal_far": NOMINAL_FARS[arm],
        "cal_fit_view": "disjoint_fit" if independent else "original_full_holdout",
        "cal_check_independent": independent,
        "cal_check_far_pass": _check_far(check_metrics) if independent else None,
        **_numbers("cal_fit", fit_metrics),
        **_numbers("cal_check", check_metrics),
    }


def _check_far(metrics):
    """Require support and a CAL-check FAR at or below the fixed cap."""
    return (
        metrics["benign_rows"] >= MIN_CAL_BENIGN_ROWS
        and metrics["far"] is not None
        and metrics["far"] <= FAR_CAP
    )


def freeze_thresholds(labels_cal, probs_cal, cal_subsets, classes):
    """Return JSON-serializable threshold locks using only CAL inputs."""
    if not 0 < len(labels_cal) == len(probs_cal) == len(cal_subsets):
        raise ValueError("NB15 threshold locks need matched CAL input for every client")
    result = {arm: [] for arm in ARMS}
    for labels, probabilities, subsets in zip(labels_cal, probs_cal, cal_subsets, strict=True):
        labels, probabilities = _validate_role(labels, probabilities, classes)
        fit, check = _split_indices(subsets, len(labels))
        for arm in ARMS:
            result[arm].append(_freeze_client(labels, probabilities, fit, check, arm))
    return result


def _pooled_cal(locks, role):
    """Calculate pooled CAL rates directly from matched client counts."""
    counts = {key: sum(lock[f"{role}_{key}"] for lock in locks) for key in COUNT_KEYS}
    benign, attack = counts["benign_rows"], counts["attack_rows"]
    return _numbers(role, {
        **counts,
        "far": counts["false_positives"] / benign if benign else None,
        "fnr": counts["false_negatives"] / attack if attack else None,
    })


def _scope_rows(identity, selected, names):
    """Pool the gate outcomes while keeping each client's own threshold."""
    locks = [lock for lock, _ in selected]
    val = _pool([decision for _, decision in selected])
    classes = list(_class_rows(identity, val, names))
    point = {
        **identity, **_classification_metrics(classes),
        **_numbers("val", _binary_metrics(val, ranking=True)),
        **{key: value for role in CAL_ROLES for key, value in _pooled_cal(locks, role).items()},
        "nominal_cal_far": locks[0]["nominal_cal_far"],
        "cal_check_independent": locks[0]["cal_check_independent"],
        "cal_check_clients_passing": (
            sum(lock["cal_check_far_pass"] is True for lock in locks)
            if locks[0]["cal_check_independent"] else None
        ),
        "cal_check_clients": len(locks),
        "cal_clients_with_insufficient_support": sum(lock["alert_gate_disabled_for_support"] for lock in locks),
        "val_ranking_scope": "receiver_raw_scores" if identity["scope"] == "receiver" else "pooled_raw_scores",
    }
    return point, classes


def _evaluate_arm(seed, names, receiver, validated, arm, locks):
    """Apply existing thresholds to VAL and preserve matched population labels."""
    identity = {"seed": seed, "arm": arm, "receiver_client": receiver}
    clients = [
        (lock, _decide(p, labels, lock["threshold"]))
        for (labels, p), lock in zip(validated, locks, strict=True)
    ]
    client_rows = [
        {**identity, "client": index, **lock, **_numbers("val", _binary_metrics(val, ranking=True))}
        for index, (lock, val) in enumerate(clients)
    ]
    scoped = [
        _scope_rows({**identity, "scope": scope}, selected, names)
        for scope, selected in (("receiver", clients[receiver:receiver + 1]), ("overall", clients))
    ]
    return [point for point, _ in scoped], client_rows, [row for _, rows in scoped for row in rows]


def _class_differences(rows):
    """Compare every class against its reference on the same VAL rows."""
    reference = {(row["scope"], row["class_index"]): row for row in rows if row["arm"] == REFERENCE}
    return [
        {
            **row,
            **{
                key + "_delta_vs_reference": _difference(
                    row[key], reference[(row["scope"], row["class_index"])][key],
                )
                for key in CLASS_METRICS
            },
        }
        for row in rows
    ]


def evaluate_locked_seed(seed, names, receiver, labels_val, probabilities_val, locked):
    """Evaluate one seed from already locked thresholds without revisiting CAL."""
    if (
        len(names) < 2 or names[0] != "BENIGN" or len(set(names)) != len(names)
        or not 0 <= receiver < len(labels_val)
        or len(labels_val) != len(probabilities_val)
        or set(locked) != set(ARMS)
        or any(len(locks) != len(labels_val) for locks in locked.values())
    ):
        raise ValueError("Expected all frozen NB15 thresholds and aligned VAL populations")
    validated = [_validate_role(y, p, len(names)) for y, p in zip(labels_val, probabilities_val, strict=True)]
    results = [_evaluate_arm(seed, names, receiver, validated, arm, locked[arm]) for arm in ARMS]
    points, clients, classes = ([row for group in table for row in group] for table in zip(*results, strict=True))
    return {"working_points": points, "client_working_points": clients, "per_class": _class_differences(classes)}


def evaluate_seed(seed, names, receiver, labels_by_client, probabilities_by_client, cal_subsets):
    """Freeze using CAL then evaluate on the corresponding held-back VAL arrays."""
    cal_y, val_y = zip(*labels_by_client, strict=True)
    cal_p, val_p = zip(*probabilities_by_client, strict=True)
    locked = freeze_thresholds(cal_y, cal_p, cal_subsets, len(names))
    return evaluate_locked_seed(seed, names, receiver, val_y, val_p, locked)


def _index_points(rows, seeds):
    """Require the locked evaluation population for every seed, scope and arm."""
    expected = {(seed, scope, arm) for seed in seeds for scope in SCOPES for arm in ARMS}
    index = {(row["seed"], row["scope"], row["arm"]): row for row in rows}
    if len(index) != len(rows) or set(index) != expected:
        raise ValueError("NB15 threshold results lack a complete matched seed/scope inventory")
    return index


def _checks(rows, seeds, candidate):
    """Summarize the per-client independent CAL check without changing predictions."""
    selected = [row for row in rows if row["arm"] == candidate]
    keys = {(row["seed"], row["client"]) for row in selected}
    reference = {(row["seed"], row["client"]) for row in rows if row["arm"] == REFERENCE}
    if len(keys) != len(selected) or keys != reference or {seed for seed, _ in keys} != set(seeds):
        raise ValueError("NB15 threshold check data lack a complete matched seed/client inventory")
    passed = sum(row["cal_check_far_pass"] is True for row in selected)
    failing = [
        {"seed": row["seed"], "client": row["client"], "far": row["cal_check_far"]}
        for row in selected if row["cal_check_far_pass"] is not True
    ]
    return {
        "pass": not failing and all(row["cal_check_independent"] for row in selected),
        "clients_passing": passed, "clients_expected": len(selected), "failing_clients": failing,
    }


def _candidate_checks(tables, index, seeds, candidate, intervals):
    """Apply the existing VAL guard plus one independent CAL check per client."""
    scopes = {scope: _scope_check(index, seeds, scope, candidate, intervals[scope]) for scope in SCOPES}
    cal = _checks(tables["client_working_points"], seeds, candidate)
    flags = (
        "far_pass", "cal_support_pass", "nonzero_attack_detection_each_seed",
        "paired_fnr_pass", "macro_f1_mean_non_decrease",
    )
    passed = cal["pass"] and all(record[flag] for record in scopes.values() for flag in flags)
    return {
        "scopes": scopes,
        "cal_check": cal,
        "decision_scope": "prespecified_candidate" if candidate == CONSERVATIVE else "matched_sampling_control",
        "development_guard_pass": passed if len(seeds) >= MIN_ASSESSMENT_SEEDS else None,
    }


def _target_deltas(index, seeds, scope, name, candidate, reference):
    """Pair target and minority recall and F1 on identical seed support."""
    def observed(seed, metric):
        left = index.get((seed, scope, name, candidate), {})
        right = index.get((seed, scope, name, reference), {})
        return _difference(left.get(metric), right.get(metric))

    return {
        metric: summarize([observed(seed, metric) for seed in seeds], bounds=(-1, 1))
        for metric in CLASS_METRICS
    }


def _target_intervals(rows, seeds):
    """Report descriptive minority effects separately from the binary guard."""
    index = {(row["seed"], row["scope"], row["class_name"], row["arm"]): row for row in rows}
    if len(index) != len(rows):
        raise ValueError("NB15 threshold per-class evidence contains duplicate rows")
    return {
        f"{candidate}_minus_{reference}": {
            scope: {
                name: _target_deltas(index, seeds, scope, name, candidate, reference)
                for name in TARGET_CLASSES
            }
            for scope in SCOPES
        }
        for candidate, reference in PAIRS
    }


def assess_results(tables, seeds):
    """Summarize the fixed comparisons and record the development-only decision."""
    seeds = tuple(seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("NB15 threshold assessment requires distinct prescribed seeds")
    index = _index_points(tables["working_points"], seeds)
    intervals = {
        f"{candidate}_minus_{reference}": {
            scope: _paired_stats(index, seeds, scope, candidate, reference) for scope in SCOPES
        }
        for candidate, reference in PAIRS
    }
    return _assessment(tables, index, seeds, intervals)


def _assessment(tables, index, seeds, intervals):
    """Record development roles, guard settings and all paired results."""
    candidates = {
        arm: _candidate_checks(tables, index, seeds, arm, intervals[f"{arm}_minus_{REFERENCE}"])
        for arm in (MATCHED, CONSERVATIVE)
    }
    return {
        "scope": "grouped same-source development", "seeds": list(seeds),
        "calibration_rule": "per-client benign CAL higher quantile with zero floor and strict score > threshold",
        "selection_on_val": False, "test_evaluated": False, "automatic_promotion": False,
        "far_cap": FAR_CAP, "paired_fnr_ci95_upper_limit": FNR_INCREASE_LIMIT,
        "min_cal_benign_rows": MIN_CAL_BENIGN_ROWS, "nominal_cal_fars": NOMINAL_FARS,
        "primary_candidate": CONSERVATIVE, "development_guard_pass": candidates[CONSERVATIVE]["development_guard_pass"],
        "paired_intervals": intervals, "candidates": candidates,
        "target_class_intervals": _target_intervals(tables["per_class"], seeds),
    }
