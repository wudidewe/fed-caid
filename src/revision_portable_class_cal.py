"""Select a predicted attack route threshold using only grouped CAL scores."""

from collections import Counter

import numpy as np

from .revision_class_operating import attack_routes, class_alarm_thresholds
from .revision_operating_selection import DEFAULT_CANDIDATES, _digest
from .revision_report import score
from .utils import empirical_alarm_threshold, hierarchical_decisions


def apply_portable_class_cal(probabilities, thresholds):
    """Apply locked per-route thresholds to the unchanged local detector scores."""
    cutoffs = class_alarm_thresholds(probabilities, thresholds)
    return hierarchical_decisions(probabilities, cutoffs)


def _cal_side(client, prefix, classes):
    probabilities = np.asarray(client[f'{prefix}_probabilities'], dtype=np.float64)
    routes = attack_routes(probabilities)
    labels = np.asarray(client[f'{prefix}_labels'])
    groups = np.asarray(client[f'{prefix}_group_ids'])
    if (
        probabilities.shape[1] != classes
        or labels.shape != routes.shape or groups.shape != routes.shape
        or labels.dtype.kind not in 'iu' or groups.dtype.kind not in 'iuSV'
        or (labels < 0).any() or (labels >= classes).any()
    ):
        raise ValueError('CAL probabilities, class indices and exact input groups must align')
    return probabilities, labels, groups, routes


def _prepare(client, classes, far, floor, min_groups):
    fit_p, fit_y, fit_groups, fit_routes = _cal_side(client, 'fit', classes)
    select_p, select_y, select_groups, select_routes = _cal_side(client, 'select', classes)
    scalar, scalar_audit = empirical_alarm_threshold(fit_p, fit_y, far, floor)
    fit_scores = np.clip(fit_p[:, 1:].sum(axis=1), 0, 1)
    fit_benign, select_benign = fit_y == 0, select_y == 0
    route_support, banks = [], {}
    for route in range(1, classes):
        fit_mask = fit_benign & (fit_routes == route)
        select_mask = select_benign & (select_routes == route)
        fit_count = np.unique(fit_groups[fit_mask]).size
        select_count = np.unique(select_groups[select_mask]).size
        supported = min(fit_count, select_count) >= min_groups
        if supported:
            banks[route] = fit_scores[fit_mask]
        route_support.append({
            'route_index': route, 'fit_benign_groups': fit_count, 'select_benign_groups': select_count,
            'fit_benign_rows': int(fit_mask.sum()), 'select_benign_rows': int(select_mask.sum()),
            'eligible': supported,
        })
    return {
        'scalar': scalar, 'banks': banks, 'fit_groups': np.unique(fit_groups),
        'select_groups': np.unique(select_groups), 'select_p': select_p, 'select_y': select_y,
        'support': {
            'fit_rows': len(fit_y), 'select_rows': len(select_y),
            'fit_benign_groups': np.unique(fit_groups[fit_benign]).size,
            'select_benign_groups': np.unique(select_groups[select_benign]).size,
            'select_attack_groups': np.unique(select_groups[~select_benign]).size,
            'routes': route_support, 'scalar': scalar_audit,
        },
    }


def _assessment(clients, thresholds, classes):
    decisions = [apply_portable_class_cal(client['select_p'], cutoffs)
                 for client, cutoffs in zip(clients, thresholds, strict=True)]
    y = np.concatenate([client['select_y'] for client in clients])
    pred = np.concatenate([row['predicted'] for row in decisions])
    q = np.concatenate([row['attack_score'] for row in decisions])
    metrics, _, matrix = score(y, pred, q, range(classes))
    benign = y == 0
    return {
        **{key: metrics[key] for key in ('accuracy', 'macro_f1', 'far', 'fnr')},
        'benign_rows': int(benign.sum()), 'attack_rows': int((~benign).sum()),
        'false_alarms': sum(matrix[0][1:]), 'missed_attacks': sum(row[0] for row in matrix[1:]),
    }


def _candidate(clients, classes, nominal_far, floor, target_far, reference, macro_tolerance):
    thresholds = [[client['scalar']] * (classes - 1) for client in clients]
    for client, vector in zip(clients, thresholds, strict=True):
        for route, scores in client['banks'].items():
            vector[route - 1] = max(floor, float(np.quantile(scores, 1.0 - nominal_far, method='higher')))
    metrics = _assessment(clients, thresholds, classes)
    comparable = all(metrics[key] is not None and reference[key] is not None for key in ('far', 'fnr', 'macro_f1'))
    macro_delta = metrics['macro_f1'] - reference['macro_f1'] if comparable else None
    miss_delta = metrics['missed_attacks'] - reference['missed_attacks']
    checks = {
        'supported_route': any(client['banks'] for client in clients),
        'cal_select_has_both_binary_labels': comparable,
        'far_at_or_below_cap': comparable and metrics['far'] <= target_far,
        'fnr_does_not_increase': comparable and miss_delta <= 0,
        'macro_f1_within_tolerance': comparable and macro_delta >= -macro_tolerance - 1e-12,
        'fnr_or_macro_f1_improves': comparable and (miss_delta < 0 or (miss_delta == 0 and macro_delta > 1e-12)),
    }
    return {
        'nominal_far': nominal_far, 'class_thresholds': thresholds, 'cal_select': metrics,
        'failed_constraints': [name for name, passed in checks.items() if not passed],
        'fnr_delta': metrics['fnr'] - reference['fnr'] if comparable else None,
        'macro_f1_delta': macro_delta,
    }


def select_portable_class_cal(
    clients, *, target_far=0.01, floor=0.5, candidates=DEFAULT_CANDIDATES,
    min_route_benign_groups=100, macro_tolerance=0.01,
):
    """Fit all thresholds on CAL fit; choose from the grid on independent CAL select.

    Each client supplies fit/select probabilities, integer labels and exact raw
    feature group IDs. Group IDs have the same meaning across clients. Metrics
    and the empirical quantiles use retained rows, with distinct benign feature
    groups on both sides determining route eligibility. An unsupported route
    keeps the client's matched scalar threshold. Returned cutoffs can be locked
    before the caller evaluates VAL. When no candidate passes, the returned
    vector encodes the matched scalar fallback.
    """
    clients = tuple(clients)
    nominal_fars = sorted({float(value) for value in candidates}, reverse=True)
    if (
        not clients or not nominal_fars or any(not 0 < value < 1 for value in (*nominal_fars, target_far))
        or not 0 <= floor <= 1 or not 0 <= macro_tolerance <= 1
        or type(min_route_benign_groups) is not int or min_route_benign_groups < 1
    ):
        raise ValueError('Invalid class threshold search inputs')
    first_fit = np.asarray(clients[0]['fit_probabilities'])
    if first_fit.ndim != 2 or first_fit.shape[1] < 2:
        raise ValueError('Class threshold search requires multiclass CAL probabilities')
    classes = first_fit.shape[1]
    prepared = [_prepare(client, classes, target_far, floor, min_route_benign_groups) for client in clients]
    fit_groups = np.concatenate([client['fit_groups'] for client in prepared])
    select_groups = np.concatenate([client['select_groups'] for client in prepared])
    if np.intersect1d(fit_groups, select_groups).size:
        raise ValueError('An exact input group occurs on both threshold CAL sides')
    scalar_thresholds = [client['scalar'] for client in prepared]
    reference_vectors = [[threshold] * (classes - 1) for threshold in scalar_thresholds]
    reference = _assessment(prepared, reference_vectors, classes)
    records = [_candidate(prepared, classes, far, floor, target_far, reference, macro_tolerance)
               for far in nominal_fars]
    eligible = [row for row in records if not row['failed_constraints']]
    selected = min(eligible, key=lambda row: (
        row['cal_select']['fnr'], -row['cal_select']['macro_f1'], row['cal_select']['far'], -row['nominal_far'],
    ), default=None)
    failures = Counter(name for row in records for name in row['failed_constraints'])
    result = {
        'schema': 'portable_class_cal_v1',
        'status': 'selected' if selected is not None else 'no_feasible_operating_point',
        'reason': None if selected is not None else 'no_candidate_satisfied_constraints',
        'target_far': target_far, 'threshold_floor': floor, 'max_macro_f1_loss': macro_tolerance,
        'min_route_benign_groups': min_route_benign_groups, 'global_fit_select_exact_overlap': 0,
        'selection_rule': 'min_fnr_then_max_macro_f1_then_min_far_then_max_nominal_far',
        'support': [client['support'] for client in prepared],
        'scalar_thresholds': scalar_thresholds, 'reference': reference,
        'class_thresholds': selected['class_thresholds'] if selected is not None else reference_vectors,
        'selected_nominal_far': selected['nominal_far'] if selected is not None else None,
        'failed_constraint_counts': dict(sorted(failures.items())), 'candidates': records,
    }
    return {**result, 'selection_sha256': _digest(result)}
