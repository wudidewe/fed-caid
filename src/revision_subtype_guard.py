"""Calibrate subtype replacement on labelled CAL negatives with fixed alarms."""

import math

import numpy as np

from .revision_fixed_subtype import _eligible, _label_vector


GUARDED_ROUTING = "calibrated_missing_attack_margin_v1"
_LOG_SCORE_FLOOR = 1e-12
_GUARD_FIELDS = {
    "schema_version",
    "status",
    "eligible_classes",
    "threshold",
    "max_rewrite_rate",
    "min_negative_groups",
    "fit_negative_groups",
    "fit_rewritten_groups",
}
_CHECK_DISABLED = {"insufficient_check_negative_groups", "check_rewrite_budget_exceeded"}


def _scores(values):
    scores = np.asarray(values)
    if (
        scores.ndim != 2
        or scores.shape[1] < 2
        or scores.dtype.kind not in "fiu"
        or not np.isfinite(scores).all()
        or (scores < 0).any()
    ):
        raise ValueError("Expected aligned finite nonnegative scores for BENIGN and each attack")
    return scores


def _log_score(score):
    return np.log(np.maximum(score.astype(np.float64, copy=False), _LOG_SCORE_FLOOR))


def _candidate_margin(scores, candidates):
    rows = np.arange(len(scores))
    candidate = scores[rows, candidates]
    if scores.shape[1] == 2:
        return _log_score(candidate)
    attacks = scores[:, 1:]
    best = attacks.argmax(axis=1)
    largest = np.partition(attacks, -2, axis=1)[:, -2:]
    competitor = np.where(candidates == best + 1, largest[:, 0], largest[:, 1])
    return _log_score(candidate) - _log_score(competitor)


def target_attack_margin(scores, target):
    """Return log(target score / best other attack score), excluding BENIGN.

    A sole attack class uses its log score as the margin because there is no
    competing attack. Scores at or below 1e-12 share the same finite log floor.
    """
    scores = _scores(scores)
    _eligible((target,), scores.shape[1])
    return _candidate_margin(scores, np.full(len(scores), target, dtype=np.int64))


def _inputs(base_predictions, scores, eligible):
    scores = _scores(scores)
    classes = scores.shape[1]
    base = _label_vector(base_predictions, classes, "Base predictions").astype(np.int64, copy=False)
    eligible_mask = _eligible(eligible, classes)
    if len(base) != len(scores):
        raise ValueError("Base predictions and subtype scores must align by row")
    return base, scores, eligible_mask


def _proposal(base, scores, eligible_mask):
    # Compute the proposal only for existing alarms. Sort by class index so ties
    # are independent of the order in which eligible classes were declared.
    candidate = base.copy()
    margin = np.full(len(base), -np.inf)
    eligible = np.flatnonzero(eligible_mask)
    alerts = np.flatnonzero(base > 0)
    if len(eligible) and len(alerts):
        alert_scores = scores[alerts]
        proposed = eligible[alert_scores[:, eligible].argmax(axis=1)]
        candidate[alerts] = proposed
        margin[alerts] = _candidate_margin(alert_scores, proposed)
    return candidate, margin


def _calibration_inputs(base, labels, eligible_mask, group_ids):
    truth = _label_vector(labels, len(eligible_mask), "CAL labels")
    groups = np.arange(len(base)) if group_ids is None else np.asarray(group_ids)
    if (
        truth.shape != base.shape
        or groups.shape != base.shape
        or groups.dtype.kind not in "iubfSUV"
        or groups.dtype.kind == "f"
        and not np.isfinite(groups).all()
    ):
        raise ValueError("CAL labels and exact feature-group IDs must align by row")
    negatives = (base > 0) & ~eligible_mask[truth]
    unique, inverse = np.unique(groups[negatives], return_inverse=True)
    return negatives, len(unique), inverse


def _validate_budget(rate, support):
    if type(rate) not in (float, int) or not 0 <= rate < 1 or type(support) is not int or support < 1:
        raise ValueError("Expected an empirical rewrite rate in [0, 1) and a positive group support floor")


def validate_subtype_guard(guard, *, eligible=None):
    """Validate the full JSON receipt and optionally bind its eligible classes."""
    if (
        not isinstance(guard, dict)
        or set(guard) != _GUARD_FIELDS
        or type(guard["schema_version"]) is not int
        or guard["schema_version"] != 1
    ):
        raise ValueError("Invalid subtype guard receipt schema")
    allowed = guard["eligible_classes"]
    if (
        not isinstance(allowed, list)
        or any(type(i) is not int or i < 1 for i in allowed)
        or allowed != sorted(set(allowed))
    ):
        raise ValueError("Subtype guard eligible classes differ from routing")
    if eligible is not None:
        expected = tuple(eligible)
        if (
            any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) for i in expected)
            or sorted(expected) != allowed
        ):
            raise ValueError("Subtype guard eligible classes differ from routing")
    support = guard["min_negative_groups"]
    rate = guard["max_rewrite_rate"]
    _validate_budget(rate, support)
    status, threshold = guard["status"], guard["threshold"]
    if type(status) is not str:
        raise ValueError("Invalid subtype guard status or threshold")
    total, rewritten = guard["fit_negative_groups"], guard["fit_rewritten_groups"]
    if (
        type(total) is not int
        or type(rewritten) is not int
        or not 0 <= rewritten <= total
        or total
        and rewritten / total > rate
    ):
        raise ValueError("Invalid subtype guard group counts")
    if not allowed:
        valid = status == "no_eligible_attacks"
    elif total < support:
        valid = status == "insufficient_negative_groups"
    elif status == "calibrated":
        valid = type(threshold) in (float, int) and -math.inf < threshold < math.inf
    else:
        valid = status == "no_rewrite_candidates" or status in _CHECK_DISABLED
    if (
        not valid
        or status != "calibrated"
        and threshold is not None
        or status != "calibrated"
        and status not in _CHECK_DISABLED
        and rewritten
    ):
        raise ValueError("Invalid subtype guard status or threshold")
    return guard


def fit_subtype_guard(
    base_predictions,
    scores,
    labels,
    eligible,
    *,
    max_rewrite_rate=0.05,
    min_negative_groups=30,
    group_ids=None,
):
    """Fit a pooled replacement rule on pre-existing alarms in receiver CAL-fit.

    An exact-input group enters the denominator if it has an alerted, known
    non-target row. It enters the numerator if any such row would be rewritten.
    The largest proposed margin in each group determines its cutoff. Ties at
    the selected margin abstain, and insufficient support disables replacement.
    The fitted rate is empirical and is distinct from the binary FAR.
    """
    _validate_budget(max_rewrite_rate, min_negative_groups)
    base, scores, eligible_mask = _inputs(base_predictions, scores, eligible)
    negative, n_groups, inverse = _calibration_inputs(base, labels, eligible_mask, group_ids)
    guard = {
        "schema_version": 1,
        "status": "insufficient_negative_groups" if eligible_mask.any() else "no_eligible_attacks",
        "eligible_classes": np.flatnonzero(eligible_mask).tolist(),
        "threshold": None,
        "max_rewrite_rate": float(max_rewrite_rate),
        "min_negative_groups": min_negative_groups,
        "fit_negative_groups": n_groups,
        "fit_rewritten_groups": 0,
    }
    if not eligible_mask.any() or n_groups < min_negative_groups:
        return guard
    candidate, margins = _proposal(base[negative], scores[negative], eligible_mask)
    group_margin = np.full(n_groups, -np.inf)
    np.maximum.at(group_margin, inverse, np.where(candidate != base[negative], margins, -np.inf))
    finite = group_margin[np.isfinite(group_margin)]
    if not len(finite):
        guard["status"] = "no_rewrite_candidates"
        return guard
    allowed = math.floor(n_groups * max_rewrite_rate)
    if allowed / n_groups > max_rewrite_rate:
        allowed -= 1
    elif (allowed + 1) / n_groups <= max_rewrite_rate:
        allowed += 1
    if allowed < len(finite):
        k = len(finite) - 1 - allowed
        threshold = float(np.partition(finite, k)[k])
    else:
        threshold = float(np.nextafter(finite.min(), -np.inf))
    guard.update(
        status="calibrated",
        threshold=threshold,
        fit_rewritten_groups=int(np.count_nonzero(group_margin > threshold)),
    )
    return validate_subtype_guard(guard)


def _apply_guard(base, scores, eligible_mask, guard):
    validate_subtype_guard(guard, eligible=np.flatnonzero(eligible_mask))
    threshold = guard["threshold"]
    if threshold is None:
        return base.copy()
    candidate, margin = _proposal(base, scores, eligible_mask)
    return np.where(margin > threshold, candidate, base)


def guarded_subtype_predictions(base_predictions, scores, eligible, guard):
    """Replace an existing alarm only if the eligible target exceeds its CAL cutoff."""
    base, scores, eligible_mask = _inputs(base_predictions, scores, eligible)
    return _apply_guard(base, scores, eligible_mask, guard)


def audit_subtype_guard(base_predictions, scores, labels, eligible, guard, group_ids=None):
    """Report the realized non-target rewrite rate on a separate CAL check split."""
    base, scores, eligible_mask = _inputs(base_predictions, scores, eligible)
    negative, n_groups, inverse = _calibration_inputs(base, labels, eligible_mask, group_ids)
    predicted = _apply_guard(base, scores, eligible_mask, guard)
    changed = predicted[negative] != base[negative]
    negative_rewritten = int(changed.sum())
    groups_rewritten = len(np.unique(inverse[changed]))
    rate = groups_rewritten / n_groups if n_groups else None
    return {
        "negative_rows": int(negative.sum()),
        "negative_groups": n_groups,
        "rewritten_negative_rows": negative_rewritten,
        "rewritten_negative_groups": groups_rewritten,
        "negative_group_rewrite_rate": rate,
        "meets_empirical_budget": rate <= guard["max_rewrite_rate"] if rate is not None else None,
    }


def apply_cal_check(
    base_predictions,
    scores,
    labels,
    eligible,
    guard,
    *,
    min_check_groups,
    check_tolerance=0.0,
    group_ids=None,
):
    """Abstain if a fitted guard exceeds its empirical CAL-check limit or lacks support.

    This check must receive a separate CAL block. It keeps the fit and proposal
    check counts for audit, and never refits the threshold from check labels.
    """
    _validate_budget(check_tolerance, min_check_groups)
    audit = audit_subtype_guard(base_predictions, scores, labels, eligible, guard, group_ids)
    limit = min(1.0, guard["max_rewrite_rate"] + check_tolerance)
    status = "not_applicable"
    if guard["status"] == "calibrated":
        if audit["negative_groups"] < min_check_groups:
            status = "insufficient_check_negative_groups"
        elif audit["negative_group_rewrite_rate"] > limit:
            status = "check_rewrite_budget_exceeded"
        else:
            status = "passed"
    audit.update(
        check_status=status,
        check_limit=limit,
        check_support=min_check_groups,
        check_tolerance=float(check_tolerance),
        original_fit_threshold=guard["threshold"],
    )
    if status in _CHECK_DISABLED:
        guard = validate_subtype_guard({**guard, "status": status, "threshold": None})
    return guard, audit
