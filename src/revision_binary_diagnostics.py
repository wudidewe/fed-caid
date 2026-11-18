"""Diagnose binary alarm behavior at a fixed CAL operating point.

The report is descriptive: it compares CAL and VAL score distributions, FAR/FNR,
and attack-class losses at the already selected threshold.  VAL labels are used
only for this post-hoc diagnostic, never for threshold selection or fitting.
"""

from __future__ import annotations

import json

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score, roc_curve


def _values(scores):
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Binary diagnostic scores must be finite one-dimensional arrays")
    return values


def _quantile(values, q):
    values = np.asarray(values, dtype=np.float64)
    return float(np.quantile(values, q)) if len(values) else None


def _thresholds(threshold, length, field):
    values = np.asarray(threshold, dtype=np.float64)
    if (
        values.shape not in ((), (length,))
        or not np.isfinite(values).all()
        or (values < 0).any()
        or (values > 1).any()
    ):
        raise ValueError(f"{field} must be a scalar or aligned thresholds in [0, 1]")
    return values


def _alert_mask(scores, threshold, field, provided=None):
    expected = scores > _thresholds(threshold, len(scores), field)
    if provided is not None:
        actual = np.asarray(provided)
        if actual.dtype.kind != "b" or not np.array_equal(expected, actual):
            raise ValueError(f"{field} and supplied binary alerts differ")
    return expected


def _rates(labels, alert):
    benign, attack = labels == 0, labels > 0
    benign_count, attack_count = int(benign.sum()), int(attack.sum())
    fp, fn = int((benign & alert).sum()), int((attack & ~alert).sum())
    return {
        "rows": int(len(labels)),
        "benign_rows": benign_count,
        "attack_rows": attack_count,
        "true_positives": attack_count - fn,
        "true_negatives": benign_count - fp,
        "false_positives": fp,
        "false_negatives": fn,
        "far": fp / benign_count if benign_count else None,
        "fnr": fn / attack_count if attack_count else None,
    }


def _auc(scores, labels):
    attack = np.asarray(labels) > 0
    if not attack.any() or attack.all():
        return None
    return float(roc_auc_score(attack.astype(int), scores))


def _low_far(scores, labels):
    attack = labels > 0
    benign_count = int((~attack).sum())
    result = {
        "tpr_at_far_0_5pct": None,
        "tpr_at_far_1pct": None,
        "standardized_partial_auc_at_far_1pct": None,
        "empirical_far_resolution": 1 / benign_count if benign_count else None,
    }
    if benign_count >= 100 and attack.any():
        fpr, tpr, _ = roc_curve(attack, scores, drop_intermediate=False)
        for far, key in ((0.005, "tpr_at_far_0_5pct"), (0.01, "tpr_at_far_1pct")):
            if 1 / benign_count <= far:
                result[key] = float(tpr[fpr <= far].max())
        result["standardized_partial_auc_at_far_1pct"] = float(roc_auc_score(attack, scores, max_fpr=0.01))
    return result


def _class_loss(truth, alert, predicted=None):
    support = int(truth.sum())
    detected = int((truth & alert).sum())
    binary_missed = support - detected
    result = {
        "support": support,
        "alerted_support": detected,
        "binary_missed": binary_missed,
        "binary_fnr": binary_missed / support if support else None,
        "binary_recall": detected / support if support else None,
    }
    if predicted is None:
        return {**result, "missed": binary_missed, "fnr": result["binary_fnr"]}
    true_positive = int((truth & predicted).sum())
    predicted_count = int(predicted.sum())
    missed = support - true_positive
    return {
        **result,
        "predicted": predicted_count,
        "true_positive": true_positive,
        "false_positive": predicted_count - true_positive,
        "subtype_missed": missed,
        "subtype_miss_rate": missed / support if support else None,
        "subtype_missed_among_alerted": detected - true_positive,
        "subtype_miss_rate_among_alerted": (detected - true_positive) / detected if detected else None,
        "missed": missed,  # Legacy consumers treat `missed` as the final class miss count.
        "precision": true_positive / predicted_count if predicted_count else 0.0 if support else None,
        "recall": true_positive / support if support else None,
        "f1": 2 * true_positive / (support + predicted_count) if support else None,
    }


def _ks(first, second):
    if not len(first) or not len(second):
        return None
    return float(ks_2samp(first, second).statistic)


def summarize_binary_diagnostics(
    cal_scores,
    cal_labels,
    threshold,
    *,
    threshold_fit_scores,
    threshold_fit_labels,
    operating_select_scores,
    operating_select_labels,
    val_scores,
    val_labels,
    reference_threshold=None,
    target=None,
    min_benign_rows=0,
    cal_scope="parent_cal_holdout",
    val_predictions=None,
    cal_thresholds=None,
    threshold_fit_thresholds=None,
    operating_select_thresholds=None,
    val_thresholds=None,
    cal_alerts=None,
    val_alerts=None,
):
    """Return rates and class losses at the existing operating point.

    Role-specific threshold vectors support routing on the frozen pre-gate
    subtype. Optional alert masks assert exact agreement with the threshold.
    Low-FAR statistics describe raw scores and do not determine thresholds.
    """
    cal_scores, cal_labels = _values(cal_scores), np.asarray(cal_labels)
    val_scores, val_labels = _values(val_scores), np.asarray(val_labels)
    fit_scores, fit_labels = _values(threshold_fit_scores), np.asarray(threshold_fit_labels)
    select_scores, select_labels = _values(operating_select_scores), np.asarray(operating_select_labels)
    scopes = (
        ("cal", cal_scores, cal_labels, cal_thresholds, cal_alerts),
        ("threshold_fit", fit_scores, fit_labels, threshold_fit_thresholds, None),
        ("operating_select", select_scores, select_labels, operating_select_thresholds, None),
        ("val", val_scores, val_labels, val_thresholds, val_alerts),
    )
    threshold = _thresholds(threshold, len(val_scores), "Binary diagnostic threshold")
    if threshold.ndim:
        raise ValueError("Use role-specific arguments for vector thresholds")
    masks, rates = {}, {}
    for scope, scores, labels, cutoff, alerts in scopes:
        if (
            scores.shape != labels.shape
            or (labels.size and labels.dtype.kind not in "iu")
            or (labels < 0).any()
        ):
            raise ValueError("Binary diagnostic scores and integer labels must align")
        masks[scope] = _alert_mask(
            scores, threshold if cutoff is None else cutoff, f"{scope} thresholds", alerts
        )
        rates[scope] = _rates(labels, masks[scope])
    cal_benign, val_benign = cal_scores[cal_labels == 0], val_scores[val_labels == 0]
    cal_attack, val_attack = cal_scores[cal_labels > 0], val_scores[val_labels > 0]
    cal_q99, val_q99 = _quantile(cal_benign, 0.99), _quantile(val_benign, 0.99)
    q99_shift = (val_q99 - cal_q99) if cal_q99 is not None and val_q99 is not None else None
    ks_benign, ks_attack = _ks(cal_benign, val_benign), _ks(cal_attack, val_attack)
    cal_auc, val_auc = _auc(cal_scores, cal_labels), _auc(val_scores, val_labels)
    val_attack_median = _quantile(val_attack, 0.5)
    overlap = cal_q99 is not None and val_attack_median is not None and val_attack_median <= cal_q99
    if reference_threshold is not None:
        ref_cal = _rates(cal_labels, _alert_mask(cal_scores, reference_threshold, "CAL reference threshold"))
        ref_val = _rates(val_labels, _alert_mask(val_scores, reference_threshold, "VAL reference threshold"))
    else:
        ref_cal = ref_val = {"far": None, "fnr": None}
    val_cutoff = threshold if val_thresholds is None else np.asarray(val_thresholds)
    support_insufficient = bool(
        rates["threshold_fit"]["benign_rows"] < min_benign_rows
        or (val_cutoff.size and np.all(val_cutoff >= 1))
    )
    benign_shift = bool(
        (ks_benign is not None and ks_benign > 0.1) or (q99_shift is not None and abs(q99_shift) > 0.05)
    )
    attack_overlap = bool((val_auc is not None and val_auc < 0.9) or overlap)
    cal_far, val_far = rates["cal"]["far"], rates["val"]["far"]
    far_mismatch = bool(val_far is not None and cal_far is not None and abs(val_far - cal_far) > 0.005)
    val_far_exceeds_limit = bool(val_far is not None and val_far > 0.01)
    predicted = None if val_predictions is None else np.asarray(val_predictions)
    class_support = set(map(int, np.unique(val_labels[val_labels > 0])))
    if predicted is not None:
        if (
            (predicted.size and predicted.dtype.kind not in "iu")
            or predicted.shape != val_labels.shape
            or (predicted < 0).any()
            or not np.array_equal(predicted > 0, masks["val"])
        ):
            raise ValueError("Binary diagnostic final VAL labels must preserve the binary alert mask")
        class_support.update(map(int, np.unique(predicted[predicted > 0])))
    if target is not None:
        if not isinstance(target, (int, np.integer)) or target <= 0:
            raise ValueError("Binary diagnostic target must be an attack class index")
        class_support.add(int(target))
    losses = {
        str(label): _class_loss(
            val_labels == label, masks["val"], None if predicted is None else predicted == label
        )
        for label in sorted(class_support)
    }
    target_metrics = {}
    if target is not None:
        record = losses[str(target)]
        target_metrics = {
            "target_class_index": int(target),
            **{f"target_{key}": value for key, value in record.items()},
            "target_fnr": record["binary_fnr"],
            "target_recall": record.get("recall", record["binary_recall"]),
        }
    target_loss = bool(target_metrics.get("target_f1") is not None and target_metrics["target_f1"] < 0.5)
    diagnosis = (
        "threshold_support_insufficient"
        if support_insufficient
        else (
            "attack_benign_overlap"
            if attack_overlap
            else (
                "target_class_loss"
                if target_loss
                else (
                    "benign_score_shift"
                    if benign_shift
                    else (
                        "cal_val_far_mismatch"
                        if far_mismatch
                        else "val_far_limit_exceeded" if val_far_exceeds_limit else "consistent"
                    )
                )
            )
        )
    )
    return {
        "cal_scope": cal_scope,
        **{f"{scope}_{key}": value for scope, rate in rates.items() for key, value in rate.items()},
        "reference_cal_far": ref_cal["far"],
        "reference_cal_fnr": ref_cal["fnr"],
        "reference_val_far": ref_val["far"],
        "reference_val_fnr": ref_val["fnr"],
        "cal_benign_q50": _quantile(cal_benign, 0.50),
        "cal_benign_q90": _quantile(cal_benign, 0.90),
        "cal_benign_q95": _quantile(cal_benign, 0.95),
        "cal_benign_q99": cal_q99,
        "val_benign_q50": _quantile(val_benign, 0.50),
        "val_benign_q90": _quantile(val_benign, 0.90),
        "val_benign_q95": _quantile(val_benign, 0.95),
        "val_benign_q99": val_q99,
        "benign_q99_shift": q99_shift,
        "cal_attack_q05": _quantile(cal_attack, 0.05),
        "cal_attack_q50": _quantile(cal_attack, 0.50),
        "cal_attack_q95": _quantile(cal_attack, 0.95),
        "val_attack_q05": _quantile(val_attack, 0.05),
        "val_attack_q50": val_attack_median,
        "val_attack_q95": _quantile(val_attack, 0.95),
        "ks_benign": ks_benign,
        "ks_attack": ks_attack,
        "auc_cal": cal_auc,
        "auc_val": val_auc,
        **{
            f"{scope}_{key}": value
            for scope, metrics in (
                ("cal", _low_far(cal_scores, cal_labels)),
                ("val", _low_far(val_scores, val_labels)),
            )
            for key, value in metrics.items()
        },
        "attack_median_below_cal_benign_q99": bool(overlap),
        "support_insufficient": support_insufficient,
        "benign_score_shift": benign_shift,
        "attack_benign_overlap": attack_overlap,
        "target_class_loss": target_loss,
        "cal_val_far_mismatch": far_mismatch,
        "val_far_exceeds_1pct": val_far_exceeds_limit,
        "primary_diagnosis": diagnosis,
        "subtype_predictions_available": predicted is not None,
        "attack_class_losses": json.dumps(losses, sort_keys=True),
        **target_metrics,
    }


def class_diagnostic_rows(diagnostic, names):
    """Expand per-attack VAL diagnostics while preserving the original case identity."""
    losses = json.loads(diagnostic["attack_class_losses"])
    absent_count = 0 if diagnostic.get("subtype_predictions_available") else None
    rows = []
    for i, name in enumerate(names):
        if i == 0:
            continue
        loss = losses.get(str(i), {})
        rows.append({
            "class": name,
            "class_index": i,
            "val_support": loss.get("support", 0),
            "val_alerted_support": loss.get("alerted_support", 0),
            "binary_missed": loss.get("binary_missed", 0),
            "binary_fnr": loss.get("binary_fnr"),
            "subtype_missed": loss.get("subtype_missed", absent_count),
            "subtype_miss_rate": loss.get("subtype_miss_rate"),
            "subtype_missed_among_alerted": loss.get("subtype_missed_among_alerted", absent_count),
            "subtype_miss_rate_among_alerted": loss.get("subtype_miss_rate_among_alerted"),
            "subtype_true_positives": loss.get("true_positive", absent_count),
            "subtype_false_positives": loss.get("false_positive", absent_count),
            "subtype_predicted": loss.get("predicted", absent_count),
            **{f"subtype_{key}": loss.get(key) for key in ("precision", "recall", "f1")},
        })
    return rows


__all__ = ["class_diagnostic_rows", "summarize_binary_diagnostics"]
