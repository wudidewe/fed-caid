"""CAL-only thresholds conditional on the frozen local TabM attack route."""

import numpy as np

from .revision_operating_selection import (
    DEFAULT_CANDIDATES,
    DEFAULT_REFERENCE_FAR,
    _digest,
    _rates,
    _reference_far_checks,
)

SCHEMA = 'cal_class_operating_selection_v1'
SELECTION_RULE = 'loosest_first_descending_nominal_far_classwise'


def attack_routes(probabilities):
    """Assign each row its local attack argmax, with lowest class index on ties."""
    p = np.asarray(probabilities)
    if (
        p.ndim != 2
        or p.shape[1] < 2
        or p.dtype.kind not in 'fiu'
        or not np.isfinite(p).all()
        or (p < 0).any()
        or (p > 1).any()
    ):
        raise ValueError('Expected a finite matrix of benign and attack probabilities')
    return p[:, 1:].argmax(axis=1) + 1


def class_alarm_thresholds(probabilities, thresholds):
    """Return one frozen local-gate threshold per row without reading labels."""
    routes = attack_routes(probabilities)
    expected = np.asarray(probabilities).shape[1] - 1
    values = _threshold_vector(thresholds, expected)
    return values[routes - 1]


def _threshold_vector(thresholds, size):
    values = np.asarray(thresholds)
    if (
        values.shape != (size,)
        or values.dtype.kind not in 'fiu'
        or not np.isfinite(values).all()
        or (values < 0).any()
        or (values > 1).any()
    ):
        raise ValueError('Expected one finite threshold in [0, 1] per ordered entry')
    return values.astype(np.float64, copy=False)


def _cal_block(client, prefix, classes):
    scores = np.asarray(client[f'{prefix}_scores'], dtype=np.float64)
    labels = np.asarray(client[f'{prefix}_labels'])
    routes = np.asarray(client[f'{prefix}_routes'])
    if (
        scores.shape != labels.shape
        or scores.shape != routes.shape
        or scores.ndim != 1
        or labels.dtype.kind not in 'iu'
        or (labels < 0).any()
        or (labels >= classes).any()
        or routes.dtype.kind not in 'iu'
        or (routes < 1).any()
        or (routes >= classes).any()
    ):
        raise ValueError('CAL scores, integer labels and positive class routes must align by row')
    # Validate the complete score vector, including attacks.
    _rates(scores, labels, 0.0)
    return scores, labels, routes.astype(np.int64, copy=False)


def _client_calibration(client, classes, limits):
    fit_scores, fit_labels, fit_routes = _cal_block(client, 'threshold_fit', classes)
    select_scores, select_labels, select_routes = _cal_block(client, 'operating_select', classes)
    fit_by_route = np.bincount(fit_routes[fit_labels == 0], minlength=classes)[1:]
    select_by_route = np.bincount(select_routes[select_labels == 0], minlength=classes)[1:]
    minimum = limits['min_route_benign']
    sparse = np.flatnonzero((fit_by_route < minimum) | (select_by_route < minimum)) + 1
    pool_fit = int(fit_by_route[sparse - 1].sum())
    pool_select = int(select_by_route[sparse - 1].sum())
    supported = fit_by_route.sum() >= limits['min_benign_rows'] and bool(fit_by_route.sum())
    pool_supported = not len(sparse) or (pool_fit >= minimum and pool_select >= minimum)
    benign = fit_labels == 0
    banks = {
        int(route): fit_scores[benign & (fit_routes == route)] for route in range(1, classes) if route not in sparse
    }
    if len(sparse):
        banks[0] = fit_scores[benign & np.isin(fit_routes, sparse)]
    return {
        'receiver': bool(client['receiver']),
        'select_scores': select_scores,
        'select_labels': select_labels,
        'select_routes': select_routes,
        'banks': banks,
        'support': {
            'fit_benign_by_route': fit_by_route.tolist(),
            'select_benign_by_route': select_by_route.tolist(),
            'fit_benign_rows': int(fit_by_route.sum()),
            'fit_attack_rows': int((fit_labels > 0).sum()),
            'select_benign_rows': int(select_by_route.sum()),
            'select_attack_rows': int((select_labels > 0).sum()),
            'pooled_sparse_routes': sparse.tolist(),
            'pool_fit_benign_rows': pool_fit,
            'pool_select_benign_rows': pool_select,
            'pool_supported': pool_supported,
            'threshold_support': bool(supported and pool_supported),
        },
    }


def _fit_class_thresholds(client, classes, candidate, floor):
    vector = np.ones(classes - 1, dtype=np.float64)
    for route, benign in client['banks'].items():
        if len(benign) and (route or client['support']['pool_supported']):
            indices = route - 1 if route else np.asarray(client['support']['pooled_sparse_routes']) - 1
            vector[indices] = max(floor, float(np.quantile(benign, 1 - candidate, method='higher')))
    return vector


def _candidate_record(candidate, vectors, reference, clients, limits):
    rows = []
    for index, (client, vector, ref_threshold) in enumerate(zip(clients, vectors, reference, strict=True)):
        scores, labels = client['select_scores'], client['select_labels']
        rates = _rates(scores, labels, vector[client['select_routes'] - 1])
        baseline = _rates(scores, labels, ref_threshold)
        attack = rates['attack_rows']
        rows.append(
            {
                **rates,
                'client': index,
                'receiver': client['receiver'],
                'class_thresholds': vector.tolist(),
                'disabled_class_routes': (np.flatnonzero(vector >= 1) + 1).tolist(),
                'reference_threshold': ref_threshold,
                'reference_far': baseline['far'],
                'reference_false_alarms': baseline['false_alarms'],
                'reference_fnr': baseline['fnr'],
                'reference_missed_attacks': baseline['missed_attacks'],
                'fnr_delta': (rates['missed_attacks'] - baseline['missed_attacks']) / attack if attack else None,
            }
        )
    counts = {
        field: sum(row[field] for row in rows)
        for field in (
            'benign_rows',
            'attack_rows',
            'false_alarms',
            'missed_attacks',
            'reference_false_alarms',
            'reference_missed_attacks',
        )
    }
    benign, attack = counts['benign_rows'], counts['attack_rows']
    ref_far = counts['reference_false_alarms'] / benign if benign else None
    ref_fnr = counts['reference_missed_attacks'] / attack if attack else None
    delta = (counts['missed_attacks'] - counts['reference_missed_attacks']) / attack if attack else None
    overall = {
        **counts,
        'far': counts['false_alarms'] / benign if benign else None,
        'fnr': counts['missed_attacks'] / attack if attack else None,
        'alert_coverage': (attack - counts['missed_attacks']) / attack if attack else None,
    }
    receiver = next(row for row in rows if row['receiver'])
    checks = {
        'threshold_support': all(c['support']['threshold_support'] for c in clients),
        'receiver_routes_active': not receiver['disabled_class_routes'],
        'select_benign_support': all(row['benign_rows'] >= limits['min_select_benign_rows'] for row in rows),
        'overall_attack_support': attack >= limits['min_select_attack_rows'],
        'receiver_attack_support': receiver['attack_rows'] >= limits['min_receiver_attack_rows'],
        'overall_far': overall['far'] is not None and overall['far'] <= limits['max_far'],
        'receiver_far': receiver['far'] is not None and receiver['far'] <= limits['max_far'],
        'overall_fnr_delta': delta is not None and delta <= limits['max_fnr_increase'],
        'receiver_fnr_delta': (
            receiver['fnr_delta'] is not None and receiver['fnr_delta'] <= limits['max_fnr_increase']
        ),
        'attack_coverage': (
            overall['alert_coverage'] is not None and overall['alert_coverage'] >= limits['min_attack_coverage']
        ),
        'reference_active': all(value < 1 for value in reference),
    }
    return {
        'candidate_cal_far': candidate,
        'thresholds': reference,
        'reference_thresholds': reference,
        'class_thresholds': [vector.tolist() for vector in vectors],
        'overall': overall,
        'reference_far': ref_far,
        'reference_fnr': ref_fnr,
        **_reference_far_checks(ref_far, receiver['reference_far'], limits['reference_far']),
        'overall_fnr_delta': delta,
        'receiver': receiver,
        'clients': rows,
        'checks': checks,
        'feasible': all(checks.values()),
        'reason': next((name for name, passed in checks.items() if not passed), None),
    }


def select_class_operating_point(
    clients,
    *,
    classes,
    candidates=DEFAULT_CANDIDATES,
    reference_thresholds,
    max_far=0.01,
    reference_far=DEFAULT_REFERENCE_FAR,
    max_fnr_increase=0.01,
    min_benign_rows=1000,
    min_select_benign_rows=30,
    min_select_attack_rows=20,
    min_receiver_attack_rows=10,
    min_attack_coverage=0.5,
    floor=0.5,
    min_route_benign=30,
):
    """Fit local route thresholds on CAL, then choose a common candidate nominal FAR.

    ``clients`` uses the scalar selector's CAL fields plus ``threshold_fit_routes``
    and ``operating_select_routes``. Model argmax supplies routes on both splits;
    true labels only identify benign and attack rows in the CAL assessments.
    Sparse local routes share a benign calibration bank. If the local pool is
    insufficient on either CAL split, no point is selected. Scalar reference
    thresholds come from the original 1% policy and are not refitted here.
    """
    clients = tuple(clients)
    if type(classes) is not int or classes < 2 or sum(bool(c['receiver']) for c in clients) != 1:
        raise ValueError('Class operating selection needs a declared class space and exactly one receiver')
    candidate_grid = sorted({float(value) for value in candidates}, reverse=True)
    if not candidate_grid or any(not 0 < value < 1 for value in (*candidate_grid, max_far, reference_far)):
        raise ValueError('Candidate, reference and maximum FARs must lie in (0, 1)')
    if not 0 <= max_fnr_increase <= 1 or not 0 <= min_attack_coverage <= 1 or not 0 <= floor < 1:
        raise ValueError('Invalid FNR increase limit, minimum coverage or threshold floor')
    if (
        type(min_route_benign) is not int
        or min_route_benign < 1
        or any(
            type(value) is not int or value < 0
            for value in (
                min_benign_rows,
                min_select_benign_rows,
                min_select_attack_rows,
                min_receiver_attack_rows,
            )
        )
    ):
        raise ValueError('Support minima must be nonnegative integers and route support must be positive')
    limits = {
        'max_far': float(max_far),
        'reference_far': float(reference_far),
        'max_fnr_increase': float(max_fnr_increase),
        'min_attack_coverage': float(min_attack_coverage),
        'threshold_floor': float(floor),
        'min_benign_rows': min_benign_rows,
        'min_select_benign_rows': min_select_benign_rows,
        'min_select_attack_rows': min_select_attack_rows,
        'min_receiver_attack_rows': min_receiver_attack_rows,
        'min_route_benign': min_route_benign,
    }
    reference = _threshold_vector(reference_thresholds, len(clients)).tolist()
    prepared = [_client_calibration(client, classes, limits) for client in clients]
    receiver = next(index for index, client in enumerate(prepared) if client['receiver'])
    audit = [client['support'] for client in prepared]
    support = {
        'threshold_fit_benign_rows': [row['fit_benign_rows'] for row in audit],
        'threshold_fit_attack_rows': [row['fit_attack_rows'] for row in audit],
        'operating_select_benign_rows': [row['select_benign_rows'] for row in audit],
        'operating_select_attack_rows': [row['select_attack_rows'] for row in audit],
        'receiver_operating_select_attack_rows': audit[receiver]['select_attack_rows'],
        'reference_source': 'external_frozen_reference',
        'reference_support': None,
        'reference_valid': all(value < 1 for value in reference),
        'threshold_sources': ['client_class_cal'] * len(clients),
        'reference_disabled_clients': [i for i, value in enumerate(reference) if value == 1],
        'class_route_support': audit,
    }
    records = []
    for candidate in candidate_grid:
        vectors = [_fit_class_thresholds(client, classes, candidate, floor) for client in prepared]
        records.append(_candidate_record(candidate, vectors, reference, prepared, limits))
    selected = next((row for row in records if row['feasible']), None)
    result = {
        'schema': SCHEMA,
        'status': 'selected' if selected else 'no_feasible_operating_point',
        'reason': None if selected else 'no_candidate_satisfied_constraints',
        'selected_cal_far': selected['candidate_cal_far'] if selected else None,
        'reference_cal_far': float(reference_far),
        'candidate_grid': candidate_grid,
        'selection_rule': SELECTION_RULE,
        'constraints': limits,
        'support': support,
        'thresholds': reference if selected else [],
        'reference_thresholds': reference,
        'class_thresholds': selected['class_thresholds'] if selected else [],
        'receiver_client': receiver,
        'receiver_fnr_scope': 'other CAL attack classes; target absent from receiver CAL',
    }
    if not selected:
        result['failed_constraints'] = sorted(
            {reason for row in records for reason, ok in row['checks'].items() if not ok}
        )
    return {**result, 'candidates': records, 'selection_sha256': _digest(result)}
