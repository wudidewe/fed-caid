"""Protect each original attack route on grouped CAL while preserving binary alarms."""

import numpy as np

from .revision_subtype_guard import _inputs, _proposal, _validate_budget, fit_subtype_guard


def _counts(groups, population, changes):
    return int(np.unique(groups[population]).size), int(np.unique(groups[population & changes]).size)


def fit_source_guard(
    base_predictions,
    scores,
    labels,
    eligible,
    *,
    group_ids,
    seed,
    max_rewrite_rate=0.05,
    min_negative_groups=30,
    min_check_groups=30,
    check_tolerance=0.02,
):
    """Fit on two thirds of receiver CAL-fit groups and validate on the rest.

    Pooled and per-original-route cutoffs constrain non-target rewrites. The
    independent check also restricts damage among originally correct alarms of
    each original class. Unsupported or failing routes abstain. Check labels
    can disable a route, but never change a fitted cutoff.
    """
    base, scores, mask = _inputs(base_predictions, scores, eligible)
    labels, groups = np.asarray(labels), np.asarray(group_ids)
    if (
        labels.shape != base.shape
        or groups.shape != base.shape
        or labels.dtype.kind not in "iu"
        or (labels < 0).any()
        or (labels >= len(mask)).any()
    ):
        raise ValueError("Receiver CAL labels and group IDs must align with scores")
    _validate_budget(max_rewrite_rate, min_negative_groups)
    _validate_budget(check_tolerance, min_check_groups)
    unique, inverse = np.unique(groups, return_inverse=True)
    order = np.random.default_rng(seed).permutation(len(unique))
    check = np.isin(inverse, order[: len(order) // 3])
    fit = ~check
    cutoff_limit = min(1.0, max_rewrite_rate + check_tolerance)
    pooled = fit_subtype_guard(
        base[fit],
        scores[fit],
        labels[fit],
        eligible,
        max_rewrite_rate=max_rewrite_rate,
        min_negative_groups=min_negative_groups,
        group_ids=inverse[fit],
    )
    proposal, margin = _proposal(base, scores, mask)
    negative = (base > 0) & ~mask[labels]
    possible = proposal != base
    routes = {}
    proposed_changes = np.zeros(base.shape, dtype=bool)
    for original in np.unique(base[base > 0]):
        route = base == original
        route_fit, route_check = route & fit, route & check
        candidate = fit_subtype_guard(
            base[route_fit],
            scores[route_fit],
            labels[route_fit],
            eligible,
            max_rewrite_rate=max_rewrite_rate,
            min_negative_groups=min_negative_groups,
            group_ids=inverse[route_fit],
        )
        thresholds = (pooled["threshold"], candidate["threshold"])
        threshold = max(thresholds) if None not in thresholds else None
        changes = (
            (margin > threshold) & possible if threshold is not None else np.zeros(base.shape, dtype=bool)
        )
        correct = route_check & (labels == original) & ~mask[labels]
        negatives, rewritten = _counts(inverse, route_check & negative, changes)
        good, spoiled = _counts(inverse, correct, changes)
        if threshold is None:
            status = "insufficient_fit_support_or_no_proposal"
        elif min(negatives, good) < min_check_groups:
            status = "insufficient_class_check_support"
        elif rewritten / negatives > cutoff_limit or spoiled / good > cutoff_limit:
            status = "class_check_budget_exceeded"
        else:
            status = "enabled"
            proposed_changes |= route_check & changes
        routes[str(int(original))] = {
            "status": status,
            "threshold": threshold if status == "enabled" else None,
            "fitted_threshold": threshold,
            "check_negative_groups": negatives,
            "check_rewritten_negative_groups": rewritten,
            "check_originally_correct_groups": good,
            "check_spoiled_groups": spoiled,
            "fit_negative_groups": candidate["fit_negative_groups"],
        }
    total, altered = _counts(inverse, check & negative, proposed_changes)
    if total < min_check_groups:
        status = "insufficient_pooled_check_support"
    elif altered / total > cutoff_limit:
        status = "pooled_check_budget_exceeded"
    elif any(item["status"] == "enabled" for item in routes.values()):
        status = "enabled"
    else:
        status = "no_eligible_route"
    return {
        "schema_version": 1,
        "policy": "source_class_grouped_cal_guard",
        "status": status,
        "eligible_classes": np.flatnonzero(mask).tolist(),
        "fit_group_count": len(unique) - len(order) // 3,
        "check_group_count": len(order) // 3,
        "fit_check_group_overlap": 0,
        "max_rewrite_rate": float(max_rewrite_rate),
        "check_limit": cutoff_limit,
        "min_negative_groups": min_negative_groups,
        "min_check_groups": min_check_groups,
        "pooled_fit": pooled,
        "pooled_check_negative_groups": total,
        "pooled_check_rewritten_negative_groups": altered,
        "source_routes": routes,
    }


def predict_source_guard(base_predictions, scores, eligible, receipt):
    base, scores, mask = _inputs(base_predictions, scores, eligible)
    if (
        receipt.get("policy") != "source_class_grouped_cal_guard"
        or receipt.get("schema_version") != 1
        or receipt.get("eligible_classes") != np.flatnonzero(mask).tolist()
    ):
        raise ValueError("CAL protection receipt differs from the prediction route")
    if receipt["status"] != "enabled":
        return base.copy()
    proposal, margin = _proposal(base, scores, mask)
    result = base.copy()
    for original, item in receipt["source_routes"].items():
        if item["status"] != "enabled":
            continue
        threshold = item["threshold"]
        source = int(original)
        if not 0 < source < len(mask) or not np.isfinite(threshold):
            raise ValueError("Invalid enabled source-class route")
        change = (base == source) & (margin > threshold)
        result[change] = proposal[change]
    return result
