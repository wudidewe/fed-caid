"""Evaluate matched NB15 binary detectors with a frozen CAL operating rule."""

import numpy as np

from .revision_binary_diagnostics import _class_loss, _low_far, _rates
from .revision_report import summarize
from .utils import empirical_alarm_threshold, hierarchical_decisions


ARMS = ("reference", "frozen_head", "full_finetune")
SCOPES = ("receiver", "overall")
FAR_CAP = 0.01
FNR_INCREASE_LIMIT = 0.01
MIN_CAL_BENIGN_ROWS = 100
METRICS = ("val_far", "val_fnr", "val_macro_f1", "val_binary_ece_10", "val_tpr_at_far_1pct_diagnostic")
PAIRS = (("frozen_head", "reference"), ("full_finetune", "reference"), ("full_finetune", "frozen_head"))
CLASS_FIELDS = ("support", "predicted", "true_positive", "binary_recall", "precision", "recall", "f1")


def _validate_role(labels, probabilities, classes):
    """Require valid aligned class probabilities and integer class indices."""
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if (
        labels.ndim != 1
        or not labels.size
        or labels.dtype.kind not in "ui"
        or (labels < 0).any()
        or (labels >= classes).any()
        or probabilities.shape != (len(labels), classes)
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or not np.allclose(probabilities.sum(1), 1.0, rtol=0, atol=1e-6)
    ):
        raise ValueError("NB15 probabilities must align with nonempty CAL or VAL class labels")
    return labels, probabilities


def _binary_ece(scores, labels):
    """Return row-weighted binary ECE over ten equal-width probability bins."""
    bins = np.minimum((scores * 10).astype(int), 9)
    score_mass = np.bincount(bins, weights=scores, minlength=10)
    attack_mass = np.bincount(bins, weights=labels > 0, minlength=10)
    return float(np.abs(score_mass - attack_mass).sum() / len(labels))


def _decide(p, labels, threshold):
    """Apply the experiment's strict binary gate and retain subtype outputs."""
    summary = hierarchical_decisions(p, threshold)
    return {"labels": labels, **summary}


def _pool(records):
    """Concatenate row-aligned decisions with their frozen per-client gates."""
    return {key: np.concatenate([record[key] for record in records]) for key in records[0]}


def _binary_metrics(decision, *, ranking=False):
    """Measure the locked gate and optionally add diagnostic score ranking."""
    labels, scores = decision["labels"], decision["attack_score"]
    measurements = _rates(labels, decision["attack_alert"])
    measurements["binary_ece_10"] = _binary_ece(scores, labels)
    if ranking:
        diagnostic = _low_far(scores, labels)
        measurements.update(
            tpr_at_far_1pct_diagnostic=diagnostic["tpr_at_far_1pct"],
            standardized_partial_auc_at_far_1pct_diagnostic=diagnostic["standardized_partial_auc_at_far_1pct"],
            empirical_far_resolution=diagnostic["empirical_far_resolution"],
        )
    return measurements


def _client_results(labels, probabilities, classes):
    """Fit a strict threshold from CAL benign scores and apply it to VAL."""
    cal_y, cal_p = _validate_role(labels[0], probabilities[0], classes)
    val_y, val_p = _validate_role(labels[1], probabilities[1], classes)
    threshold, receipt = empirical_alarm_threshold(
        cal_p, cal_y, FAR_CAP, floor=0, min_benign_rows=MIN_CAL_BENIGN_ROWS, insufficient_policy="disable"
    )
    decisions = {
        "cal": _decide(cal_p, cal_y, threshold),
        "val": _decide(val_p, val_y, threshold),
    }
    return receipt, decisions


def _role_metrics(decisions):
    """Prefix CAL and VAL metrics and keep ROC statistics descriptive."""
    return {
        f"{role}_{metric}": value
        for role, decision in decisions.items()
        for metric, value in _binary_metrics(decision, ranking=role == "val").items()
    }


def _class_rows(identity, val, names):
    """Preserve unsupported classes and all supported zero-recall classes."""
    for index, name in enumerate(names):
        truth = val["labels"] == index
        binary_correct = val["attack_alert"] if index else ~val["attack_alert"]
        metrics = _class_loss(truth, binary_correct, val["predicted"] == index)
        yield {
            **identity,
            "class_index": index,
            "class_name": name,
            **{key: metrics[key] for key in CLASS_FIELDS},
        }


def _classification_metrics(rows):
    """Summarize classification over observed true labels for this VAL scope."""
    supported = [row for row in rows if row["support"]]
    total = sum(row["support"] for row in supported)
    return {
        "val_accuracy": sum(row["true_positive"] for row in supported) / total,
        "val_balanced_accuracy": sum(row["recall"] for row in supported) / len(supported),
        "val_macro_f1": sum(row["f1"] for row in supported) / len(supported),
        "val_weighted_f1": sum(row["f1"] * row["support"] for row in supported) / total,
        "observed_val_classes": len(supported),
        "defined_classes": len(rows),
    }


def _scope_results(identity, clients, names):
    """Pool counts using each client's own threshold and report class outcomes."""
    decisions = {role: _pool([client[1][role] for client in clients]) for role in ("cal", "val")}
    classes = list(_class_rows(identity, decisions["val"], names))
    return {
        **identity,
        **_role_metrics(decisions),
        **_classification_metrics(classes),
        "cal_clients_with_insufficient_support": sum(c[0]["alert_gate_disabled_for_support"] for c in clients),
        "val_ranking_scope": "receiver_raw_scores" if identity["scope"] == "receiver" else "pooled_raw_scores",
    }, classes


def _evaluate_arm(seed, names, receiver, label_rows, probabilities, arm):
    """Report per-client thresholds and aggregate receiver and overall outcomes."""
    clients = [
        _client_results(labels, scores, len(names))
        for labels, scores in zip(label_rows, probabilities, strict=True)
    ]
    identity = {"seed": seed, "arm": arm, "receiver_client": receiver}
    client_rows = [
        {**identity, "client": index, **receipt, **_role_metrics(decisions)}
        for index, (receipt, decisions) in enumerate(clients)
    ]
    scoped = [
        _scope_results({**identity, "scope": scope}, selected, names)
        for scope, selected in (("receiver", clients[receiver:receiver + 1]), ("overall", clients))
    ]
    return [point for point, _ in scoped], client_rows, [row for _, per_class in scoped for row in per_class]


def _difference(value, reference):
    """Keep paired differences undefined when either side lacks support."""
    return value - reference if value is not None and reference is not None else None


def _class_differences(rows):
    """Add supported-class recall pairs without inventing a promotion margin."""
    reference = {(row["scope"], row["class_index"]): row for row in rows if row["arm"] == "reference"}
    return [
        {
            **row,
            **{
                key + "_delta_vs_reference": _difference(
                    row[key], reference[(row["scope"], row["class_index"])][key]
                )
                for key in ("binary_recall", "recall", "f1")
            },
        }
        for row in rows
    ]


def evaluate_seed(seed, names, receiver, labels_by_client, probabilities_by_arm):
    """Evaluate one seed with identical roles and one CAL policy for all arms."""
    if (
        len(names) < 2
        or len(set(names)) != len(names)
        or not 0 <= receiver < len(labels_by_client)
        or set(probabilities_by_arm) != set(ARMS)
    ):
        raise ValueError("Expected aligned NB15 class names, receiver and three arms")
    results = [
        _evaluate_arm(seed, names, receiver, labels_by_client, probabilities_by_arm[arm], arm)
        for arm in ARMS
    ]
    points, clients, classes = ([row for member in table for row in member] for table in zip(*results, strict=True))
    return {"working_points": points, "client_working_points": clients, "per_class": _class_differences(classes)}


def _index_points(rows, seeds):
    """Require one row for every seed, scope and arm before computing effects."""
    expected = {(seed, scope, arm) for seed in seeds for scope in SCOPES for arm in ARMS}
    index = {(row["seed"], row["scope"], row["arm"]): row for row in rows}
    if len(seeds) != len(set(seeds)) or set(index) != expected or len(rows) != len(expected):
        raise ValueError("NB15 paired results are incomplete or ambiguous")
    return index


def _paired_stats(index, seeds, scope, candidate, reference):
    """Summarize within-seed changes for detectors evaluated on matching rows."""
    return {
        metric: summarize(
            [
                _difference(index[seed, scope, candidate][metric], index[seed, scope, reference][metric])
                for seed in seeds
            ],
            bounds=(-1, 1),
        )
        for metric in METRICS
    }


def _absolute_checks(index, seeds, scope, candidate):
    """Check per-seed FAR and reject unavailable or entirely silent gates."""
    rows = [index[seed, scope, candidate] for seed in seeds]
    values = [row["val_far"] for row in rows if row["val_far"] is not None]
    return {
        "max_far": max(values) if values else None,
        "seeds_above_far_cap": sum(value > FAR_CAP for value in values),
        "far_pass": len(values) == len(seeds) and all(value <= FAR_CAP for value in values),
        "cal_support_pass": all(row["cal_clients_with_insufficient_support"] == 0 for row in rows),
        "nonzero_attack_detection_each_seed": all(row["val_true_positives"] > 0 for row in rows),
    }


def _scope_check(index, seeds, scope, candidate, stats):
    """Assess binary and classification guards on one evaluation population."""
    fnr, macro = stats["val_fnr"], stats["val_macro_f1"]
    return {
        **_absolute_checks(index, seeds, scope, candidate),
        "paired_fnr_pass": fnr["seeds_with_metric"] == len(seeds)
        and fnr["ci95_high"] is not None
        and fnr["ci95_high"] <= FNR_INCREASE_LIMIT,
        "macro_f1_mean_non_decrease": macro["seeds_with_metric"] == len(seeds)
        and macro["mean"] is not None
        and macro["mean"] >= 0,
    }


def _candidate_checks(index, intervals, seeds, candidate):
    """Record each candidate independently without automatically promoting it."""
    checks = {
        scope: _scope_check(index, seeds, scope, candidate, intervals[candidate + "_minus_reference"][scope])
        for scope in SCOPES
    }
    flags = (
        "far_pass", "cal_support_pass", "nonzero_attack_detection_each_seed",
        "paired_fnr_pass", "macro_f1_mean_non_decrease",
    )
    passes = all(record[flag] for record in checks.values() for flag in flags) if len(seeds) >= 5 else None
    return {"scopes": checks, "development_guard_pass": passes}


def _paired_intervals(index, seeds):
    """Return all prespecified contrasts across the two evaluation scopes."""
    return {
        f"{candidate}_minus_{reference}": {
            scope: _paired_stats(index, seeds, scope, candidate, reference) for scope in SCOPES
        }
        for candidate, reference in PAIRS
    }


def assess_results(rows, seeds):
    """Compute paired intervals and preregistered development-only checks."""
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError("NB15 assessment requires a nonempty seed plan")
    index = _index_points(rows, seeds)
    intervals = _paired_intervals(index, seeds)
    return {
        "scope": "grouped same-source development",
        "seeds": list(seeds),
        "calibration_rule": "per-client benign CAL higher quantile with zero floor and strict score > threshold",
        "selection_on_val": False,
        "test_evaluated": False,
        "automatic_promotion": False,
        "far_cap": FAR_CAP,
        "paired_fnr_ci95_upper_limit": FNR_INCREASE_LIMIT,
        "min_cal_benign_rows": MIN_CAL_BENIGN_ROWS,
        "paired_intervals": intervals,
        "candidates": {arm: _candidate_checks(index, intervals, seeds, arm) for arm in ARMS[1:]},
    }
