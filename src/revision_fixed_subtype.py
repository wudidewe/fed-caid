"""Evaluate missing-class relabeling on an unchanged local binary alarm."""

from collections.abc import Iterable

import numpy as np


def _label_vector(values, classes, field):
    labels = np.asarray(values)
    if (
        labels.ndim != 1
        or labels.dtype.kind not in "iu"
        or not 2 <= classes
        or (labels < 0).any()
        or (labels >= classes).any()
    ):
        raise ValueError(f"{field} must contain integer labels from the declared class space")
    return labels


def _eligible(eligible: Iterable[int], classes: int):
    indices = tuple(eligible)
    if any(
        isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) or not 1 <= i < classes
        for i in indices
    ) or len(set(indices)) != len(indices):
        raise ValueError("Eligible classes must be distinct, non-benign class indices")
    mask = np.zeros(classes, dtype=bool)
    mask[list(indices)] = True
    return mask


def eligible_missing_classes(train_labels, classes, *, target=None):
    """Select absent attacks using only the routed TRAIN labels."""
    if type(classes) is not int or classes < 2:
        raise ValueError("Expected at least two classes in the declared class space")
    labels = _label_vector(train_labels, classes, "TRAIN labels")
    counts = np.bincount(labels.astype(np.int64, copy=False), minlength=classes)
    if target is not None:
        _eligible((target,), classes)
        return (int(target),) if counts[target] == 0 else ()
    return tuple(map(int, np.flatnonzero(counts[1:] == 0) + 1))


def _attack_top1(subtype_scores, expected_rows, classes):
    scores = np.asarray(subtype_scores)
    if (
        scores.shape != (expected_rows, classes)
        or classes < 2
        or scores.dtype.kind not in "fiu"
        or not np.isfinite(scores).all()
    ):
        raise ValueError("Fixed-alarm subtype scores require a finite, aligned rows-by-classes matrix")
    return scores[:, 1:].argmax(axis=1) + 1


def fixed_subtype_predictions(base_predictions, subtype_scores, eligible):
    """Only replace existing binary alerts whose head-preferred attack is eligible.

    Score columns are ordered across the full common class space. Class zero
    is BENIGN and is excluded from subtype comparison. A tie uses the first
    attack class in this order. Independently fitted thresholds for the head
    play no role in the fixed-alarm comparison.
    """
    scores = np.asarray(subtype_scores)
    if scores.ndim != 2:
        raise ValueError("Expected a rows-by-classes score matrix")
    classes = scores.shape[1]
    base = _label_vector(base_predictions, classes, "Base predictions").astype(np.int64, copy=False)
    eligible_mask = _eligible(eligible, classes)
    top = _attack_top1(scores, len(base), classes)
    replace = (base > 0) & eligible_mask[top]
    return np.where(replace, top, base)


def _fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def subtype_transitions(y, base, predicted, names, eligible, *, subtype_scores=None):
    """Count true-class alarm coverage and every changed subtype direction.

    All metrics use the caller's aligned population. The ceiling for an attack
    class is its baseline binary alert rate. Zero support yields a null ceiling.
    For benign traffic, alerts count false binary alarms and there is no ceiling.
    """
    classes = len(names)
    truth = _label_vector(y, classes, "Observed labels").astype(np.int64, copy=False)
    base = _label_vector(base, classes, "Base predictions").astype(np.int64, copy=False)
    predicted = _label_vector(predicted, classes, "Candidate predictions").astype(np.int64, copy=False)
    if truth.shape != base.shape or truth.shape != predicted.shape:
        raise ValueError("Fixed-alarm labels and predictions must align by row")
    alerts = base > 0
    if not np.array_equal(alerts, predicted > 0):
        raise ValueError("Candidate changed the base binary alarm mask")
    eligible_mask = _eligible(eligible, classes)
    changed = base != predicted
    if not eligible_mask[predicted[changed]].all():
        raise ValueError("Candidate changed an alert to an ineligible subtype")

    attacks = truth > 0
    corrected = changed & (truth == predicted)
    spoiled = changed & (truth == base)
    eligible_attack = eligible_mask[truth]
    eligible_attacks = int(eligible_attack.sum())
    eligible_alerts = int((eligible_attack & alerts).sum())
    stats = {
        "rows": len(truth),
        "base_alert_rows": int(alerts.sum()),
        "changed_rows": int(changed.sum()),
        "corrected_attack_subtypes": int(corrected.sum()),
        "spoiled_attack_subtypes": int(spoiled.sum()),
        "relabeled_benign_alerts": int((changed & ~attacks).sum()),
        "unalerted_attacks": int((~alerts & attacks).sum()),
        "eligible_attack_rows": eligible_attacks,
        "eligible_alert_rows": eligible_alerts,
        "eligible_recall_ceiling": _fraction(eligible_alerts, eligible_attacks),
    }

    def by_truth(condition):
        return np.bincount(truth[condition], minlength=classes)

    actual_counts = np.bincount(truth, minlength=classes)
    alerted_counts = by_truth(alerts)
    correct_before = by_truth(base == truth)
    correct_after = by_truth(predicted == truth)
    predicted_before = np.bincount(base, minlength=classes)
    predicted_after = np.bincount(predicted, minlength=classes)
    benign_prediction_before = np.bincount(base[~attacks], minlength=classes)
    benign_prediction_after = np.bincount(predicted[~attacks], minlength=classes)
    changed_counts = by_truth(changed)
    repaired_counts = by_truth(corrected)
    spoiled_counts = by_truth(spoiled)
    if subtype_scores is not None:
        top = _attack_top1(subtype_scores, len(truth), classes)
        match = (top == truth) & attacks
        top_counts = by_truth(match)
        top_alerted = by_truth(match & alerts)
    rows = []
    for i, name in enumerate(names):
        support, alerted = int(actual_counts[i]), int(alerted_counts[i])
        row = {
            "row_type": "class_summary",
            "class": name,
            "class_index": i,
            "eligible": bool(eligible_mask[i]),
            "support": support,
            "base_alert_support": alerted,
            "unalerted_support": support - alerted,
            "recall_ceiling": _fraction(alerted, support) if i > 0 else None,
            "base_true_positive": int(correct_before[i]),
            "candidate_true_positive": int(correct_after[i]),
            "base_false_positive": int(predicted_before[i] - correct_before[i]),
            "candidate_false_positive": int(predicted_after[i] - correct_after[i]),
            "base_recall_given_alert": _fraction(int(correct_before[i]), alerted) if i > 0 else None,
            "candidate_recall_given_alert": _fraction(int(correct_after[i]), alerted) if i > 0 else None,
            "changed_rows": int(changed_counts[i]),
            "corrected_rows": int(repaired_counts[i]),
            "spoiled_rows": int(spoiled_counts[i]),
        }
        if i > 0:
            row.update(
                base_benign_false_positive=int(benign_prediction_before[i]),
                candidate_benign_false_positive=int(benign_prediction_after[i]),
                base_other_attack_false_positive=int(
                    predicted_before[i] - correct_before[i] - benign_prediction_before[i]
                ),
                candidate_other_attack_false_positive=int(
                    predicted_after[i] - correct_after[i] - benign_prediction_after[i]
                ),
            )
        if subtype_scores is not None and i > 0:
            row.update(
                head_attack_top1_support=int(top_counts[i]),
                head_attack_top1_alerted=int(top_alerted[i]),
                head_attack_top1_blocked=int(top_counts[i] - top_alerted[i]),
            )
        rows.append(row)

    transitions = np.ravel_multi_index(
        (truth[changed], base[changed], predicted[changed]), (classes, classes, classes)
    )
    realized, counts = np.unique(transitions, return_counts=True)
    for encoded, count in zip(realized, counts, strict=True):
        actual, before, after = map(int, np.unravel_index(encoded, (classes, classes, classes)))
        rows.append({
            "row_type": "prediction_change",
            "class": names[actual],
            "class_index": actual,
            "base_prediction": names[before],
            "base_prediction_index": before,
            "candidate_prediction": names[after],
            "candidate_prediction_index": after,
            "count": int(count),
        })
    return stats, rows
