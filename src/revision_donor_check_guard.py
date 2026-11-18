"""Select a fixed-alert subtype rule on donor CAL and check it on receiver CAL."""

from collections.abc import Mapping

import numpy as np

from .revision_fixed_subtype import _label_vector
from .revision_subtype_guard import _inputs, _proposal, _validate_budget


DEFAULT_THRESHOLDS = (-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0)
EXACT_SEARCH = "exact_donor_cal_group_frontier"
GRID_SEARCH = "fixed_donor_cal_grid"
POLICY = "donor_cal_receiver_check_v1"
SCALAR_BYTES = 8


def _grid(thresholds):
    """Validate a fixed finite grid of log-score margins."""
    thresholds = tuple(thresholds)
    if not thresholds or any(type(value) not in (int, float) for value in thresholds):
        raise ValueError("Expected numeric donor CAL thresholds")
    grid = tuple(float(value) for value in thresholds)
    if not np.isfinite(grid).all() or len(set(grid)) != len(grid):
        raise ValueError("Donor CAL thresholds must be distinct and finite")
    return sorted(grid)


def _cal_block(block, target):
    """Align row labels and global feature-group IDs with the existing binary alarms."""
    if not isinstance(block, Mapping):
        raise ValueError("Expected a CAL block with predictions, scores, labels and feature-group IDs")
    base, scores, eligible = _inputs(block["base_predictions"], block["scores"], (target,))
    labels = _label_vector(block["labels"], scores.shape[1], "CAL labels")
    groups = np.asarray(block["group_ids"])
    if labels.shape != base.shape or groups.shape != base.shape or groups.dtype.kind not in "iuSUV":
        raise ValueError("CAL labels and global integer/byte feature-group IDs must align by row")
    proposed, margins = _proposal(base, scores, eligible)
    return base, labels, groups, (base != proposed), margins, scores.shape[1]


def _count(groups, selected):
    """Count each exact feature-group ID at most once in the selected population."""
    return int(np.unique(groups[selected]).size)


def _group_inventory(blocks):
    """Reject overlap between donors and the receiver check before aggregating counts."""
    _, _, first, _, _, classes = blocks[0]
    inventories = []
    for _, _, groups, _, _, width in blocks:
        if width != classes or groups.dtype != first.dtype:
            raise ValueError("CAL blocks must use identical class widths and feature-group ID encodings")
        inventories.append(np.unique(groups))
    combined = np.concatenate(inventories)
    if len(np.unique(combined)) != len(combined):
        raise ValueError("Donor CAL and receiver CAL-check require mutually disjoint feature groups")
    return [int(len(groups)) for groups in inventories]


def _donor_candidates(cal, target, grid):
    """Return only two rewrite counts per cutoff and two support denominators."""
    base, labels, groups, possible, margins, _ = cal
    missed_target = (base > 0) & (base != target) & (labels == target)
    negative = (base > 0) & (labels != target)
    counts = []
    for threshold in grid:
        changes = possible & (margins > threshold)
        counts.append([_count(groups, missed_target & changes), _count(groups, negative & changes)])
    return {
        "missed_alerted_target_groups": _count(groups, missed_target),
        "non_target_alerted_groups": _count(groups, negative),
        "counts": counts,
    }


def _group_maxima(groups, margins):
    """Reduce eligible rows to one activation margin per feature group."""
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    maxima = np.full(unique_groups.size, -np.inf)
    np.maximum.at(maxima, inverse, margins)
    return maxima


def _donor_maxima(cal, target):
    """Collect group maxima for correct recoveries and erroneous rewrites."""
    base, labels, groups, possible, margins, _ = cal
    missed = (base > 0) & (base != target) & (labels == target)
    negative = (base > 0) & (labels != target)
    hits, wrong = possible & missed, possible & negative
    summary = {
        "missed_alerted_target_groups": _count(groups, missed),
        "non_target_alerted_groups": _count(groups, negative),
        "proposal_margin_rows": int(np.count_nonzero(hits | wrong)),
    }
    return summary, _group_maxima(groups[hits], margins[hits]), _group_maxima(groups[wrong], margins[wrong])


def _frontier_counts(blocks, target):
    """Count distinct groups at every strict-cutoff activation regime."""
    summaries, hit_maxima, wrong_maxima = zip(
        *(_donor_maxima(block, target) for block in blocks), strict=True
    )
    hits = np.sort(np.concatenate(hit_maxima))
    wrong = np.sort(np.concatenate(wrong_maxima))
    breakpoints = np.union1d(hits, wrong)
    if breakpoints.size:
        cutoffs = np.r_[np.nextafter(breakpoints[0], -np.inf), breakpoints]
    else:
        cutoffs = breakpoints
    hit_counts = hits.size - np.searchsorted(hits, cutoffs, side="right")
    wrong_counts = wrong.size - np.searchsorted(wrong, cutoffs, side="right")
    return list(summaries), cutoffs, hit_counts, wrong_counts


def _frontier_audit(cutoffs, hits, wrong, can_recover, feasible, negatives):
    """Summarize the evaluated regimes without retaining every donor breakpoint."""
    return {
        "unique_group_margin_breakpoints": max(int(cutoffs.size) - 1, 0),
        "evaluated_cutoff_regimes": int(cutoffs.size),
        "feasible_cutoff_regimes": int(np.count_nonzero(feasible)),
        "max_corrected_target_groups": int(hits.max(initial=0)),
        "min_wrong_rewrite_rate_with_recovery": (
            float(wrong[can_recover].min() / negatives) if can_recover.any() and negatives else None
        ),
    }


def _select_exact(blocks, target, min_targets, min_negatives, budget):
    """Maximize donor CAL recovery after applying the prespecified risk and support limits."""
    donors, cutoffs, hits, wrong = _frontier_counts(blocks, target)
    targets = sum(donor["missed_alerted_target_groups"] for donor in donors)
    negatives = sum(donor["non_target_alerted_groups"] for donor in donors)
    can_recover = hits > 0
    supported = targets >= min_targets and negatives >= min_negatives
    feasible = supported & can_recover & (wrong / max(negatives, 1) <= budget)
    audit = _frontier_audit(cutoffs, hits, wrong, can_recover, feasible, negatives)
    if not supported:
        return None, "insufficient_donor_support", [], donors, audit
    valid = np.flatnonzero(feasible)
    if not valid.size:
        return None, "no_feasible_donor_threshold", [], donors, audit
    order = np.lexsort((-cutoffs[valid], wrong[valid], -hits[valid]))
    best = valid[order[0]]
    threshold = float(cutoffs[best])
    candidate = _candidate_counts(threshold, hits[best], wrong[best])
    return threshold, "donor_threshold_selected", [candidate], donors, audit


def _candidate_counts(threshold, hits, wrong):
    """Create a portable receipt row for a grouped threshold candidate."""
    return {"threshold": threshold, "corrected_target_groups": int(hits), "wrong_rewrite_groups": int(wrong)}


def _choose_threshold(donors, grid, min_targets, min_negatives, budget):
    """Select on donor grouped counts alone using a deterministic conservative tie break."""
    targets = sum(donor["missed_alerted_target_groups"] for donor in donors)
    negatives = sum(donor["non_target_alerted_groups"] for donor in donors)
    rows = []
    for index, threshold in enumerate(grid):
        hits, wrong = map(sum, zip(*(donor["counts"][index] for donor in donors), strict=True))
        rows.append(_candidate_counts(threshold, hits, wrong))
    if targets < min_targets or negatives < min_negatives:
        return None, "insufficient_donor_support", rows
    feasible = [
        r
        for r in rows
        if r["corrected_target_groups"] > 0 and r["wrong_rewrite_groups"] / negatives <= budget
    ]
    if not feasible:
        return None, "no_feasible_donor_threshold", rows
    winner = max(
        feasible, key=lambda r: (r["corrected_target_groups"], -r["wrong_rewrite_groups"], r["threshold"])
    )
    return winner["threshold"], "donor_threshold_selected", rows


def _select_donor_cutoff(blocks, target, grid, minimums, budget):
    """Use either exact in-process search or the legacy fixed-grid diagnostic."""
    if grid is None:
        return _select_exact(blocks, target, *minimums[:2], budget)
    donors = [_donor_candidates(block, target, grid) for block in blocks]
    threshold, status, candidates = _choose_threshold(donors, grid, *minimums[:2], budget)
    return threshold, status, candidates, donors, None


def _check_source(groups, route, correct, proposals, min_receiver, min_correct, budget, harm):
    """Protect a source class using its supported check denominators."""
    total, wrong = _count(groups, route), _count(groups, route & proposals)
    protected, spoiled = _count(groups, correct), _count(groups, correct & proposals)
    if total < min_receiver or protected < min_correct:
        status = "insufficient_class_check_support"
    elif wrong / total > budget or spoiled / protected > harm:
        status = "class_check_budget_exceeded"
    else:
        status = "enabled"
    return {
        "status": status,
        "negative_groups": total,
        "wrong_rewrite_groups": wrong,
        "originally_correct_groups": protected,
        "spoiled_groups": spoiled,
    }


def _check_receiver(cal, threshold, min_receiver, min_correct, budget, harm):
    """Audit one immutable cutoff on the negative-only receiver check block."""
    base, labels, groups, possible, margins, _ = cal
    proposals = possible & (margins > threshold)
    negative = base > 0
    enabled = np.zeros(base.shape, dtype=bool)
    routes = {}
    for source in np.unique(base[negative]):
        route = base == source
        check = _check_source(
            groups, route, route & (labels == source), proposals, min_receiver, min_correct, budget, harm
        )
        routes[str(int(source))] = check
        if check["status"] == "enabled":
            enabled |= route
    total, rewritten = _count(groups, negative), _count(groups, proposals & enabled)
    audit = {
        "checked_threshold": threshold,
        "negative_groups": total,
        "proposed_wrong_rewrite_groups": _count(groups, proposals),
        "enabled_wrong_rewrite_groups": rewritten,
        "source_routes": routes,
    }
    return _receiver_status(audit, min_receiver, budget), audit


def _receiver_status(audit, min_receiver, budget):
    """Fail closed for unsupported or unsafe receiver proposals."""
    total = audit["negative_groups"]
    if total < min_receiver:
        return "insufficient_receiver_check_support"
    statuses = {row["status"] for row in audit["source_routes"].values()}
    if "class_check_budget_exceeded" in statuses or audit["enabled_wrong_rewrite_groups"] / total > budget:
        return "receiver_check_budget_exceeded"
    if "enabled" not in statuses:
        return "no_supported_receiver_routes"
    return "enabled"


def _communication(donor_count, thresholds, selected):
    """Describe the in-process scope or count the legacy grid's logical payload."""
    if thresholds is None:
        return {
            "mode": "in_process",
            "logical_payload_bytes": None,
            "scope": (
                "Donor CAL margins are processed in memory. Network traffic and protection were not measured."
            ),
        }
    upload = donor_count * SCALAR_BYTES * (2 * len(thresholds) + 2)
    delivery = SCALAR_BYTES if selected is not None else 0
    return {
        "donor_candidate_upload_bytes": upload,
        "receiver_threshold_delivery_bytes": delivery,
        "logical_payload_bytes": upload + delivery,
        "scope": "numeric uint64 counts and a float64 cutoff only; excludes framing, setup and control",
    }


def _fit_blocks(donors, receiver_check, target, rates, support):
    """Validate risk budgets and enforce the controlled-absence split before scoring."""
    for rate, floor in zip((rates[0], *rates), support, strict=True):
        _validate_budget(rate, floor)
    blocks = [_cal_block(block, target) for block in donors]
    if not blocks:
        raise ValueError("Donor threshold selection requires at least one donor CAL block")
    receiver = _cal_block(receiver_check, target)
    if np.any(receiver[1] == target):
        raise ValueError("Receiver CAL-check must be negative-only for the declared target")
    return blocks, receiver, _group_inventory([*blocks, receiver])


def fit_donor_check_guard(
    donors,
    receiver_check,
    target,
    *,
    thresholds=DEFAULT_THRESHOLDS,
    max_donor_error_rate=0.05,
    max_receiver_error_rate=0.07,
    max_class_harm_rate=0.07,
    min_donor_target_groups=30,
    min_donor_negative_groups=30,
    min_receiver_groups=30,
    min_correct_class_groups=30,
):
    """Freeze a donor-selected cutoff, then allow receiver CAL-check only to accept or reject it.

    Each CAL block contains base_predictions, scores, labels and global group_ids.
    The receiver block cannot contain target labels. Low-support source classes
    abstain, and a supported route exceeding its budget disables the whole rule.
    Rates are empirical grouped CAL checks scoped to their recorded populations.
    Set thresholds=None for exact search over the donor group-margin frontier.
    """
    grid = None if thresholds is None else _grid(thresholds)
    limits = (max_donor_error_rate, max_receiver_error_rate, max_class_harm_rate)
    minimums = (
        min_donor_target_groups,
        min_donor_negative_groups,
        min_receiver_groups,
        min_correct_class_groups,
    )
    blocks, receiver, group_counts = _fit_blocks(donors, receiver_check, target, limits, minimums)
    selected, status, candidates, summaries, frontier = _select_donor_cutoff(
        blocks, target, grid, minimums, limits[0]
    )
    check = None
    if selected is not None:
        status, check = _check_receiver(receiver, selected, *minimums[2:], *limits[1:])
    settings = _settings(grid, limits, minimums)
    return _receipt(
        target,
        blocks[0][-1],
        status,
        selected,
        candidates,
        summaries,
        check,
        group_counts,
        settings,
        frontier,
    )


def _settings(grid, rates, support):
    """Record the exact configured budgets and group support floors."""
    return {
        "thresholds": grid,
        "threshold_search": EXACT_SEARCH if grid is None else GRID_SEARCH,
        "max_donor_error_rate": rates[0],
        "max_receiver_error_rate": rates[1],
        "max_class_harm_rate": rates[2],
        "min_donor_target_groups": support[0],
        "min_donor_negative_groups": support[1],
        "min_receiver_groups": support[2],
        "min_correct_class_groups": support[3],
    }


def _receipt(target, width, status, selected, candidates, donors, check, groups, settings, frontier):
    """Serialize enough grouped evidence to reproduce selection and checking."""
    return {
        "schema_version": 1,
        "policy": POLICY,
        "target": int(target),
        "classes": width,
        "status": status,
        "selected_threshold": selected,
        "threshold": selected if status == "enabled" else None,
        "candidates": candidates,
        "frontier": frontier,
        "donor_summaries": donors,
        "receiver_check": check,
        "donor_feature_groups": groups[:-1],
        "receiver_feature_groups": groups[-1],
        "cross_block_feature_group_overlap": 0,
        "settings": settings,
        "communication": _communication(len(donors), settings["thresholds"], selected),
    }


def _frozen_cutoff(receipt, threshold, check):
    """Require the originally selected threshold in either receipt protocol."""
    settings = receipt["settings"]
    if settings.get("threshold_search") == EXACT_SEARCH:
        candidates = receipt["candidates"]
        available = len(candidates) == 1 and threshold == candidates[0]["threshold"]
    else:
        available = threshold in settings["thresholds"]
    return available and threshold == receipt["selected_threshold"] == check["checked_threshold"]


def _checked_routes(receipt, target, classes):
    """Validate the routing receipt and retrieve its frozen cutoff and enabled sources."""
    expected = (1, POLICY, int(target), classes)
    if (
        not isinstance(receipt, dict)
        or tuple(receipt.get(k) for k in ("schema_version", "policy", "target", "classes")) != expected
    ):
        raise ValueError("Donor CAL guard receipt differs from the prediction route")
    if receipt["status"] != "enabled":
        return None, []
    threshold = receipt["threshold"]
    check = receipt["receiver_check"]
    if not _frozen_cutoff(receipt, threshold, check):
        raise ValueError("Enabled donor CAL guard must use its original frozen cutoff")
    routes = check["source_routes"]
    enabled = [int(source) for source, row in routes.items() if row["status"] == "enabled"]
    if not enabled or any(not 0 < source < classes for source in enabled):
        raise ValueError("Invalid enabled receiver source routes")
    return threshold, enabled


def predict_donor_check_guard(base_predictions, scores, target, receipt):
    """Use the checked cutoff on enabled source routes while keeping the binary alarm mask."""
    base, scores, mask = _inputs(base_predictions, scores, (target,))
    threshold, enabled = _checked_routes(receipt, target, scores.shape[1])
    if threshold is None:
        return base.copy()
    proposed, margins = _proposal(base, scores, mask)
    change = (base > 0) & np.isin(base, enabled) & (margins > threshold)
    return np.where(change, proposed, base)
