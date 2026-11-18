"""One CAL feature-group split shared by every client and every method.

Exact float32 model-input groups, not individual rows, are assigned to fit or
holdout. A group shared by clients must stay on the same side globally. Labels
are used only for audit, except an explicitly requested benign holdout floor.
"""

import math
from collections import defaultdict
from hashlib import sha256

import numpy as np

from .revision_data import _feature_keys


def _digest(values):
    return sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def _repair_assignment(fit, membership, benign_counts, min_holdout_benign, seed):
    """Find an exact feasible coloring on a lossless compressed witness set.

    Groups with identical client/benign-presence patterns are interchangeable for
    nonempty constraints. Keeping at least floor+1 witnesses per pattern also
    permits a benign floor while leaving a group for fit. Initial non-witness
    assignments remain fixed. MILP failure is explicit, never a row-level fallback."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csc_array

    groups = len(fit)
    owners = [[] for _ in range(groups)]
    for client, indices in enumerate(membership):
        for group in indices:
            owners[int(group)].append(client)
    patterns = defaultdict(list)
    for group, clients in enumerate(owners):
        benign = (
            tuple(client for client in clients if benign_counts[client].get(group, 0))
            if min_holdout_benign
            else ()
        )
        patterns[(tuple(clients), benign)].append(group)
    rng = np.random.default_rng(seed + 1409)
    anchors = np.array(
        sorted(
            group
            for values in patterns.values()
            for group in rng.permutation(values)[: max(2, min_holdout_benign + 1)]
        ),
        dtype=np.int64,
    )
    lookup = {int(group): i for i, group in enumerate(anchors)}
    anchor_set = set(lookup)
    rows, lower, upper = [], [], []
    for client, indices in enumerate(membership):
        row = np.zeros(len(anchors), dtype=float)
        for group in indices:
            if int(group) in lookup:
                row[lookup[int(group)]] = 1
        fixed_fit = sum(bool(fit[group]) for group in indices if int(group) not in anchor_set)
        rows.append(row)
        lower.append(1 - fixed_fit)
        upper.append(len(indices) - 1 - fixed_fit)
        if min_holdout_benign:
            row = np.zeros(len(anchors), dtype=float)
            for group, count in benign_counts[client].items():
                if group in lookup:
                    row[lookup[group]] = count
            fixed_benign_fit = sum(
                count
                for group, count in benign_counts[client].items()
                if group not in anchor_set and fit[group]
            )
            rows.append(row)
            lower.append(-np.inf)
            upper.append(sum(benign_counts[client].values()) - min_holdout_benign - fixed_benign_fit)
    cost = np.where(fit[anchors], -1.0, 1.0) * (1.0 + rng.uniform(0, 0.1, len(anchors)))
    solution = milp(
        cost,
        integrality=np.ones(len(anchors)),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(csc_array(np.asarray(rows)), lower, upper),
        options={"time_limit": 30.0},
    )
    if not solution.success:
        reason = (
            "infeasible global group constraints"
            if solution.status == 2
            else "group feasibility solver did not complete"
        )
        raise ValueError(f"CAL group split failed: {reason}; no row-level fallback ({solution.message})")
    updated = fit.copy()
    updated[anchors] = np.asarray(solution.x) >= 0.5
    return updated


def cal_group_splits(cal_x, cal_y, roster, fit_fraction, seed, *, min_holdout_benign=0):
    """Return ``([(fit_local, hold_local), ...], audit)`` in roster order.

    ``roster[i]['cal']`` contains source positions into these CAL arrays. Returned
    indices are relative to that client's CAL array, and can index cached model
    probabilities directly. Only owned CAL rows participate in the split. Both
    sides must be nonempty for every client, or this function raises ValueError."""
    x, y = np.asarray(cal_x), np.asarray(cal_y)
    if (
        x.ndim != 2
        or not x.shape[1]
        or y.shape != (len(x),)
        or not np.issubdtype(y.dtype, np.integer)
        or (y < 0).any()
    ):
        raise ValueError("CAL group split needs aligned feature rows and nonnegative integer labels")
    if isinstance(fit_fraction, bool) or not math.isfinite(fit_fraction) or not 0 < fit_fraction < 1:
        raise ValueError("CAL fit fraction must lie strictly between zero and one")
    if type(seed) is not int or seed < 0 or type(min_holdout_benign) is not int or min_holdout_benign < 0:
        raise ValueError("CAL split seed and benign floor must be nonnegative integers")
    if not roster:
        raise ValueError("CAL split requires at least one client")
    indices = [np.asarray(client["cal"]) for client in roster]
    if any(
        rows.ndim != 1
        or not np.issubdtype(rows.dtype, np.integer)
        or len(rows) < 2
        or (rows < 0).any()
        or (rows >= len(y)).any()
        for rows in indices
    ):
        raise ValueError("Each client needs at least two valid CAL row indices")
    owned = np.concatenate(indices).astype(np.int64, copy=False)
    if len(np.unique(owned)) != len(owned):
        raise ValueError("CAL source row identities must have a single client owner")
    keys, inverse = np.unique(_feature_keys(x[owned]), return_inverse=True)
    offsets = np.cumsum([0, *map(len, indices)])
    local_groups = [inverse[offsets[i] : offsets[i + 1]] for i in range(len(indices))]
    membership = [np.unique(groups) for groups in local_groups]
    if any(len(groups) < 2 for groups in membership):
        raise ValueError("CAL group split infeasible: a client owns fewer than two distinct feature groups")
    benign_counts = []
    for rows, groups in zip(indices, local_groups, strict=True):
        unique, counts = np.unique(groups[y[rows] == 0], return_counts=True)
        benign_counts.append({int(group): int(count) for group, count in zip(unique, counts, strict=True)})
        if int((y[rows] == 0).sum()) < min_holdout_benign:
            raise ValueError(
                "CAL group split infeasible: insufficient benign CAL rows for requested holdout floor"
            )
    fit = np.zeros(len(keys), dtype=bool)
    requested_groups = max(1, min(len(keys) - 1, int(round(fit_fraction * len(keys)))))
    fit[np.random.default_rng(seed + 4335).permutation(len(keys))[:requested_groups]] = True

    def valid(assignment):
        return all(
            assignment[group_set].any()
            and (~assignment[group_set]).any()
            and sum(count for group, count in benign_counts[client].items() if not assignment[group])
            >= min_holdout_benign
            for client, group_set in enumerate(membership)
        )

    repaired = not valid(fit)
    if repaired:
        fit = _repair_assignment(fit, membership, benign_counts, min_holdout_benign, seed)
    if not valid(fit):
        raise RuntimeError("CAL group feasibility solver returned an invalid assignment")
    splits, receipts = [], []
    classes = int(y.max()) + 1 if len(y) else 0
    for client, (rows, groups) in enumerate(zip(indices, local_groups, strict=True)):
        local_fit, local_hold = np.flatnonzero(fit[groups]), np.flatnonzero(~fit[groups])
        splits.append((local_fit, local_hold))
        receipts.append({
            "client": client,
            "cal_rows": len(rows),
            "cal_feature_groups": len(membership[client]),
            "fit_rows": len(local_fit),
            "holdout_rows": len(local_hold),
            "fit_feature_groups": int(fit[membership[client]].sum()),
            "holdout_feature_groups": int((~fit[membership[client]]).sum()),
            "fit_class_counts": np.bincount(y[rows[local_fit]], minlength=classes).tolist(),
            "holdout_class_counts": np.bincount(y[rows[local_hold]], minlength=classes).tolist(),
            "fit_benign_rows": int((y[rows[local_fit]] == 0).sum()),
            "holdout_benign_rows": int((y[rows[local_hold]] == 0).sum()),
            "no_benign_holdout": not np.any(y[rows[local_hold]] == 0),
            "fit_local_indices_sha256": _digest(local_fit),
            "holdout_local_indices_sha256": _digest(local_hold),
            "fit_source_indices_sha256": _digest(rows[local_fit]),
            "holdout_source_indices_sha256": _digest(rows[local_hold]),
        })
    # Group assignment is shared globally, not independently randomized per client.
    global_fit = np.unique(
        np.concatenate([groups[split[0]] for groups, split in zip(local_groups, splits, strict=True)])
    )
    global_hold = np.unique(
        np.concatenate([groups[split[1]] for groups, split in zip(local_groups, splits, strict=True)])
    )
    overlap = int(len(np.intersect1d(global_fit, global_hold)))
    if overlap:
        raise RuntimeError("CAL fit/holdout feature group overlap survived global assignment")
    assignment_hash = sha256(keys.tobytes() + fit.tobytes()).hexdigest()
    return splits, {
        "version": 1,
        "scope": "global across every client, including cross-client exact feature duplicates",
        "method": "seeded whole-feature-group assignment with constrained nonempty client sides",
        "feature_key": (
            "canonical pre-normalization float32 model features; signed zero/nonfinite canonicalized; labels"
            " excluded"
        ),
        "label_usage": (
            "audit only" if not min_holdout_benign else "explicit benign holdout floor; no VAL/TEST labels"
        ),
        "seed": seed,
        "fit_fraction_requested": fit_fraction,
        "group_assignment_repaired": repaired,
        "minimum_holdout_benign_rows": min_holdout_benign,
        "source_cal_rows": len(y),
        "owned_cal_rows": len(owned),
        "unowned_cal_rows": len(y) - len(owned),
        "cal_feature_groups": len(keys),
        "global_fit_feature_groups": len(global_fit),
        "global_holdout_feature_groups": len(global_hold),
        "global_exact_feature_overlap": overlap,
        "global_fit_rows": sum(len(pair[0]) for pair in splits),
        "global_holdout_rows": sum(len(pair[1]) for pair in splits),
        "assignment_sha256": assignment_hash,
        "clients": receipts,
        "scope_limit": "Exact model-input groups only; no near-duplicate or session-independence guarantee",
    }
