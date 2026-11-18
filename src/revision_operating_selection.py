"""Pure-CAL operating-point selection with disjoint threshold-fit/select groups.

The selector never reads VAL or TEST.  It fits one alarm threshold per client on
a group-isolated threshold-fit subset, then scores every candidate nominal FAR
on a second, group-disjoint operating-select subset.  A candidate passes only
when receiver and pooled FAR, FNR increase relative to the frozen 1% CAL
reference, and attack-alert coverage all satisfy predeclared limits.
The reference may be supplied from the original parent CAL holdout, so its
performance on the operating-select subset is an empirical comparison.

The receiver in a controlled missing-class case has no target rows in CAL, so
receiver FNR is defined over the other attack classes that remain in CAL.  The
target-class recall is left to the locked evaluation set and is never used here.
"""

from __future__ import annotations

import json
import math
from hashlib import sha256

import numpy as np

from .revision_cal_split import InfeasibleDesign, cal_group_splits
from .utils import empirical_alarm_threshold

SCHEMA = 'cal_operating_selection_v1'
SELECTION_RULE = 'loosest_first_descending_nominal_far'
DEFAULT_CANDIDATES = (0.01, 0.0075, 0.005, 0.0025, 0.001)
DEFAULT_REFERENCE_FAR = 0.01
RATE_TOLERANCE = 1e-15


def _digest(value):
    """Hash the canonical receipt payload."""
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return sha256(payload).hexdigest()


def _rates(scores, labels, threshold):
    """Measure supervised operating rates for a single threshold."""
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels)
    if scores.shape != labels.shape or scores.ndim != 1:
        raise ValueError('Operating scores and labels must be aligned one-dimensional arrays')
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError('Operating scores must be finite probabilities')
    benign, attack = labels == 0, labels > 0
    alert = scores > threshold
    return {
        'benign_rows': int(benign.sum()),
        'attack_rows': int(attack.sum()),
        'false_alarms': int((benign & alert).sum()),
        'missed_attacks': int((attack & ~alert).sum()),
        'far': float((benign & alert).sum() / benign.sum()) if benign.any() else None,
        'fnr': float((attack & ~alert).sum() / attack.sum()) if attack.any() else None,
        'alert_coverage': float((attack & alert).sum() / attack.sum()) if attack.any() else None,
    }


def _fit_threshold(scores, labels, far, floor, min_benign_rows, pooled_benign=None, pooled_fallback=True):
    """Fit one local threshold with an optional pooled benign fallback."""
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels)
    probabilities = np.column_stack((1.0 - scores, scores))
    threshold, record = empirical_alarm_threshold(probabilities, labels, far, floor, min_benign_rows, 'disable')
    if (
        record['alert_gate_disabled_for_support']
        and pooled_fallback
        and pooled_benign is not None
        and len(pooled_benign) >= min_benign_rows
    ):
        threshold = max(floor, float(np.quantile(pooled_benign, 1 - far, method='higher')))
        local_benign = scores[labels == 0]
        record = {
            **record,
            'threshold': threshold,
            'calibration_far': float((local_benign > threshold).mean()) if len(local_benign) else None,
            'calibration_rule': 'pooled_benign_support_fallback',
            'threshold_source': 'pooled_cal_threshold_fit',
            'pooled_benign_rows': int(len(pooled_benign)),
            'alert_gate_disabled_for_support': False,
        }
    return threshold, record


def split_operating_subsets(cal_x, cal_y, roster, splits, fraction=0.5, seed=0, *, min_select_benign=1):
    """Split the existing CAL holdout into group-disjoint fit and select parts.

    ``splits`` are the parent ``(fit, holdout)`` local indices.  The returned
    ``threshold_fit`` and ``operating_select`` are positions into the parent
    holdout, so callers can index their aligned CAL probability arrays directly.
    """
    if isinstance(fraction, bool) or not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError('Operating select fraction must lie strictly between zero and one')
    if type(seed) is not int or seed < 0:
        raise ValueError('Operating selection seed must be a nonnegative integer')
    if len(roster) != len(splits) or not roster:
        raise ValueError('Operating selection requires one split per client')
    temporary = [
        {**binding, 'cal': np.asarray(binding['cal'])[hold]} for binding, (_, hold) in zip(roster, splits, strict=True)
    ]
    if any(len(binding['cal']) < 2 for binding in temporary):
        raise InfeasibleDesign('Operating threshold split needs at least two holdout rows per client')
    sub_splits, audit = cal_group_splits(
        cal_x,
        cal_y,
        temporary,
        fraction,
        seed + 7919,
        min_holdout_benign=min_select_benign,
    )
    result = [{'threshold_fit': fit, 'operating_select': select} for fit, select in sub_splits]
    return result, {
        **audit,
        'scope': 'threshold-fit versus operating-select inside the existing CAL holdout',
        'parent_holdout_rows': [int(len(hold)) for _, hold in splits],
        'parent_cal_rows': [int(len(binding['cal'])) for binding in roster],
    }


def _candidate_client(index, client, threshold, reference):
    """Measure candidate and frozen-reference rates on identical CAL rows."""
    select = _rates(client['operating_select_scores'], client['operating_select_labels'], threshold)
    ref = _rates(client['operating_select_scores'], client['operating_select_labels'], reference)
    return {
        **select,
        'client': index,
        'receiver': bool(client['receiver']),
        'threshold': float(threshold),
        'reference_threshold': float(reference),
        'reference_far': ref['far'],
        'reference_false_alarms': ref['false_alarms'],
        'reference_fnr': ref['fnr'],
        'reference_missed_attacks': ref['missed_attacks'],
        'fnr_delta': (
            (select['missed_attacks'] - ref['missed_attacks']) / select['attack_rows']
            if select['attack_rows']
            else None
        ),
    }


def _candidate_pooled(rows):
    """Pool raw client counts before computing overall rates."""
    benign_rows = sum(row['benign_rows'] for row in rows)
    attack_rows = sum(row['attack_rows'] for row in rows)
    false_alarms = sum(row['false_alarms'] for row in rows)
    missed_attacks = sum(row['missed_attacks'] for row in rows)
    reference_false_alarms = sum(row['reference_false_alarms'] for row in rows)
    reference_missed = sum(row['reference_missed_attacks'] for row in rows)
    overall = {
        'benign_rows': benign_rows,
        'attack_rows': attack_rows,
        'false_alarms': false_alarms,
        'missed_attacks': missed_attacks,
        'far': false_alarms / benign_rows if benign_rows else None,
        'fnr': missed_attacks / attack_rows if attack_rows else None,
        'alert_coverage': 1.0 - missed_attacks / attack_rows if attack_rows else None,
    }
    overall_fnr_delta = (missed_attacks - reference_missed) / attack_rows if attack_rows else None
    reference = {
        'far': reference_false_alarms / benign_rows if benign_rows else None,
        'fnr': reference_missed / attack_rows if attack_rows else None,
    }
    return overall, reference, overall_fnr_delta


def _reference_far_checks(overall_far, receiver_far, target):
    """Check observed reference FAR without changing the declared selector."""
    return {
        'reference_overall_far_within_nominal': (overall_far is not None and overall_far <= target + RATE_TOLERANCE),
        'reference_receiver_far_within_nominal': (receiver_far is not None and receiver_far <= target + RATE_TOLERANCE),
    }


def _candidate_support_checks(clients, threshold_support, overall, receiver, limits):
    """Check every denominator before consulting the operating rates."""
    return {
        'threshold_support': all(threshold_support),
        'select_benign_support': all(
            client['select_benign_rows'] >= limits['min_select_benign_rows'] for client in clients
        ),
        'overall_attack_support': overall['attack_rows'] >= limits['min_select_attack_rows'],
        'receiver_attack_support': (
            receiver is not None and receiver['attack_rows'] >= limits['min_receiver_attack_rows']
        ),
    }


def _receiver_rate_checks(receiver, limits):
    """Verify receiver FAR and FNR change on its own CAL attack population."""
    return (
        receiver is not None and receiver['far'] is not None and receiver['far'] <= limits['max_far'],
        receiver is not None
        and receiver['fnr'] is not None
        and receiver['fnr_delta'] is not None
        and receiver['fnr_delta'] <= limits['max_fnr_increase'],
    )


def _candidate_rate_checks(overall, receiver, overall_fnr_delta, limits):
    """Check the pooled and receiver operating rates in declared order."""
    receiver_far, receiver_fnr_delta = _receiver_rate_checks(receiver, limits)
    return {
        'overall_far': overall['far'] is not None and overall['far'] <= limits['max_far'],
        'receiver_far': receiver_far,
        'overall_fnr_delta': (overall_fnr_delta is not None and overall_fnr_delta <= limits['max_fnr_increase']),
        'receiver_fnr_delta': receiver_fnr_delta,
        'attack_coverage': (
            overall['alert_coverage'] is not None and overall['alert_coverage'] >= limits['min_attack_coverage']
        ),
    }


def _candidate_checks(clients, threshold_support, overall, receiver, overall_fnr_delta, limits, reference_valid):
    """Evaluate all prespecified support and rate constraints in order."""
    return {
        **_candidate_support_checks(clients, threshold_support, overall, receiver, limits),
        **_candidate_rate_checks(overall, receiver, overall_fnr_delta, limits),
        'reference_active': bool(reference_valid),
    }


def _candidate_record(candidate, thresholds, references, clients, threshold_support, limits, reference_valid=True):
    """Return the candidate rates and first failed selection constraint."""
    rows = [
        _candidate_client(index, client, threshold, reference)
        for index, (client, threshold, reference) in enumerate(zip(clients, thresholds, references, strict=True))
    ]
    overall, reference, overall_fnr_delta = _candidate_pooled(rows)
    receiver = next((row for row in rows if row['receiver']), None)
    checks = _candidate_checks(
        clients, threshold_support, overall, receiver, overall_fnr_delta, limits, reference_valid
    )
    reason = next((name for name, passed in checks.items() if not passed), None)
    return {
        'candidate_cal_far': float(candidate),
        'thresholds': [float(value) for value in thresholds],
        'reference_thresholds': [float(value) for value in references],
        'overall': overall,
        'reference_far': reference['far'],
        'reference_fnr': reference['fnr'],
        **_reference_far_checks(
            reference['far'], receiver['reference_far'] if receiver else None, limits['reference_far']
        ),
        'overall_fnr_delta': overall_fnr_delta,
        'receiver': receiver,
        'clients': rows,
        'checks': checks,
        'feasible': all(checks.values()),
        'reason': reason,
    }


def _validate_operating_params(
    candidates, max_far, reference_far, max_fnr_increase, min_attack_coverage, minima, floor
):
    """Validate the supplied rates without looking at evaluation data."""
    candidates = tuple(sorted({float(value) for value in candidates}, reverse=True))
    if not candidates or any(not 0 < value < 1 for value in candidates):
        raise ValueError('Candidate nominal FARs must be distinct rates in (0, 1)')
    if not 0 < reference_far < 1 or not 0 < max_far < 1:
        raise ValueError('Reference and maximum FAR must lie in (0, 1)')
    if not math.isfinite(max_fnr_increase) or not 0 <= max_fnr_increase <= 1:
        raise ValueError('Maximum FNR increase must be a rate in [0, 1]')
    if not math.isfinite(min_attack_coverage) or not 0 <= min_attack_coverage <= 1:
        raise ValueError('Minimum attack coverage must be a rate in [0, 1]')
    if any(type(value) is not int or value < 0 for value in minima):
        raise ValueError('Support minima must be nonnegative integers')
    if not 0 <= floor < 1:
        raise ValueError('Threshold floor must lie in [0, 1)')
    return candidates


def _ordered_clients(clients):
    """Require exactly one receiver and preserve the roster's client order."""
    if not clients:
        raise ValueError('Operating selection requires at least one client')
    receiver_count = sum(bool(client['receiver']) for client in clients)
    if receiver_count != 1:
        raise ValueError('Operating selection requires exactly one receiver')
    return [_client_rows(index, client) for index, client in enumerate(clients)]


def _client_rows(index, client):
    """Validate split arrays and compute the CAL support counts per client."""
    fit_scores = np.asarray(client['threshold_fit_scores'], dtype=np.float64)
    fit_labels = np.asarray(client['threshold_fit_labels'])
    select_scores = np.asarray(client['operating_select_scores'], dtype=np.float64)
    select_labels = np.asarray(client['operating_select_labels'])
    if (
        fit_scores.shape != fit_labels.shape
        or select_scores.shape != select_labels.shape
        or fit_scores.ndim != 1
        or select_scores.ndim != 1
    ):
        raise ValueError('Each client needs aligned one-dimensional threshold-fit and select arrays')
    fit = _rates(fit_scores, fit_labels, 0.0)
    select = _rates(select_scores, select_labels, 0.0)
    return {
        'client': index,
        'receiver': bool(client['receiver']),
        'threshold_fit_scores': fit_scores,
        'threshold_fit_labels': fit_labels,
        'operating_select_scores': select_scores,
        'operating_select_labels': select_labels,
        'threshold_fit_benign_rows': fit['benign_rows'],
        'threshold_fit_attack_rows': fit['attack_rows'],
        'select_benign_rows': select['benign_rows'],
        'select_attack_rows': select['attack_rows'],
    }


def _fit_clients(client_rows, far, limits, pooled_benign):
    """Fit CAL thresholds for every client at this nominal FAR."""
    return [
        _fit_threshold(
            row['threshold_fit_scores'],
            row['threshold_fit_labels'],
            far,
            limits['threshold_floor'],
            limits['min_benign_rows'],
            pooled_benign,
            limits['pooled_benign_fallback'],
        )
        for row in client_rows
    ]


def _frozen_reference_values(reference_thresholds, client_count):
    """Require one ordered finite reference value per client."""
    values = np.asarray(reference_thresholds)
    if (
        values.shape != (client_count,)
        or values.dtype.kind not in 'fiu'
        or not np.isfinite(values).all()
        or (values < 0).any()
        or (values > 1).any()
    ):
        raise ValueError('Frozen reference requires one finite threshold in [0, 1] per ordered client')
    reference = values.tolist()
    reference_valid = all(float(threshold) < 1 for threshold in reference)
    return reference, reference_valid, 'external_frozen_reference', None


def _reference_thresholds(client_rows, reference_thresholds, limits, pooled_benign):
    """Use frozen thresholds when provided, or fit a same-CAL reference."""
    if reference_thresholds is not None:
        return _frozen_reference_values(reference_thresholds, len(client_rows))
    fitted = _fit_clients(client_rows, limits['reference_far'], limits, pooled_benign)
    reference = [float(threshold) for threshold, _ in fitted]
    support = [record for _, record in fitted]
    valid = all(not record['alert_gate_disabled_for_support'] for record in support)
    return reference, valid, 'same_cal_threshold_fit', support


def _reference_support(client_rows, pooled_benign, reference_record):
    """Record the support and source of each reference threshold."""
    reference, reference_valid, reference_source, reference_support = reference_record
    return {
        'threshold_fit_benign_rows': [row['threshold_fit_benign_rows'] for row in client_rows],
        'threshold_fit_attack_rows': [row['threshold_fit_attack_rows'] for row in client_rows],
        'operating_select_benign_rows': [row['select_benign_rows'] for row in client_rows],
        'operating_select_attack_rows': [row['select_attack_rows'] for row in client_rows],
        'receiver_operating_select_attack_rows': next(
            row['select_attack_rows'] for row in client_rows if row['receiver']
        ),
        'pooled_benign_rows': int(len(pooled_benign)),
        'threshold_sources': (
            [record.get('threshold_source', 'client_cal') for record in reference_support]
            if reference_support is not None
            else None
        ),
        'reference_source': reference_source,
        'reference_support': reference_support,
        'reference_valid': reference_valid,
        'reference_disabled_clients': [i for i, threshold in enumerate(reference) if threshold == 1],
    }


def _operating_limits(
    max_far,
    reference_far,
    max_fnr_increase,
    min_benign_rows,
    min_select_benign_rows,
    min_select_attack_rows,
    min_receiver_attack_rows,
    min_attack_coverage,
    floor,
    pooled_fallback,
):
    """Build the persisted constraints in their documented order."""
    return {
        'max_far': float(max_far),
        'reference_far': float(reference_far),
        'max_fnr_increase': float(max_fnr_increase),
        'min_benign_rows': int(min_benign_rows),
        'min_select_benign_rows': int(min_select_benign_rows),
        'min_select_attack_rows': int(min_select_attack_rows),
        'min_receiver_attack_rows': int(min_receiver_attack_rows),
        'min_attack_coverage': float(min_attack_coverage),
        'threshold_floor': float(floor),
        'pooled_benign_fallback': bool(pooled_fallback),
    }


def _assess_operating_candidates(candidates, client_rows, reference_record, limits, pooled_benign):
    """Evaluate every threshold candidate on the same operating-select subset."""
    reference, reference_valid, _, _ = reference_record
    for candidate in candidates:
        fitted = _fit_clients(client_rows, candidate, limits, pooled_benign)
        thresholds = [float(threshold) for threshold, _ in fitted]
        records_for_client = [record for _, record in fitted]
        threshold_support = [not record['alert_gate_disabled_for_support'] for record in records_for_client]
        record = _candidate_record(
            candidate, thresholds, reference, client_rows, threshold_support, limits, reference_valid
        )
        record['threshold_fit'] = records_for_client
        yield record


def _selection_digest(selection, limits, reference, reference_source):
    """Bind the selected configuration while preserving the original digest schema."""
    return _digest(
        {
            'status': selection['status'],
            'reason': selection['reason'],
            'failed_constraints': selection.get('failed_constraints', []),
            'selected_cal_far': selection['selected_cal_far'],
            'candidate_grid': selection['candidate_grid'],
            'constraints': limits,
            'thresholds': selection['thresholds'],
            'reference_thresholds': reference,
            'reference_source': reference_source,
            'receiver_client': selection['receiver_client'],
        }
    )


def _selected_record(candidates, records, support, limits, reference_record, client_rows):
    """Record the loosest feasible candidate or the exact fallback reason."""
    reference, reference_valid, reference_source, reference_support = reference_record
    reference_far = limits['reference_far']
    selected = next((record for record in records if record['feasible']), None)
    reasons = sorted({record['reason'] for record in records if record['reason']})
    reason = (
        None
        if selected
        else (
            'reference_1pct_support_insufficient'
            if reference_support is not None and not reference_valid
            else 'no_candidate_satisfied_constraints'
        )
    )
    selection = {
        'schema': SCHEMA,
        'status': 'selected' if selected else 'no_feasible_operating_point',
        'reason': reason,
        'selected_cal_far': selected['candidate_cal_far'] if selected else None,
        'reference_cal_far': float(reference_far),
        'candidate_grid': list(candidates),
        'selection_rule': SELECTION_RULE,
        'constraints': limits,
        'support': support,
        'thresholds': selected['thresholds'] if selected else [],
        'reference_thresholds': reference,
        'candidates': records,
        'receiver_client': next(i for i, c in enumerate(client_rows) if c['receiver']),
        'receiver_fnr_scope': 'other CAL attack classes; target absent from receiver CAL',
    }
    if not selected:
        selection['failed_constraints'] = reasons
    selection['selection_sha256'] = _selection_digest(selection, limits, reference, reference_source)
    return selection


def select_operating_point(
    clients,
    *,
    candidates=DEFAULT_CANDIDATES,
    max_far=0.01,
    reference_far=DEFAULT_REFERENCE_FAR,
    max_fnr_increase=0.01,
    min_benign_rows=1000,
    min_select_benign_rows=30,
    min_select_attack_rows=20,
    min_receiver_attack_rows=10,
    min_attack_coverage=0.5,
    floor=0.5,
    pooled_fallback=False,
    reference_thresholds=None,
):
    """Select the loosest feasible candidate using CAL-only evidence.

    ``clients`` is a list of mappings with aligned
    ``threshold_fit_scores``, ``threshold_fit_labels``,
    ``operating_select_scores``, ``operating_select_labels`` and a boolean
    ``receiver``.  No evaluation labels or scores are accepted.
    Supply one frozen ``reference_thresholds`` value per ordered client to use
    the parent experiment's operating point. When omitted, the legacy reference
    is fitted on threshold-fit alongside the candidates.
    """
    minima = (min_benign_rows, min_select_benign_rows, min_select_attack_rows, min_receiver_attack_rows)
    candidates = _validate_operating_params(
        candidates, max_far, reference_far, max_fnr_increase, min_attack_coverage, minima, floor
    )
    client_rows = _ordered_clients(clients)
    pooled_benign = np.concatenate(
        [row['threshold_fit_scores'][row['threshold_fit_labels'] == 0] for row in client_rows]
    )
    limits = _operating_limits(
        max_far, reference_far, max_fnr_increase, *minima, min_attack_coverage, floor, pooled_fallback
    )
    reference = _reference_thresholds(client_rows, reference_thresholds, limits, pooled_benign)
    support = _reference_support(client_rows, pooled_benign, reference)
    records = list(_assess_operating_candidates(candidates, client_rows, reference, limits, pooled_benign))
    return _selected_record(candidates, records, support, limits, reference, client_rows)


__all__ = [
    'DEFAULT_CANDIDATES',
    'DEFAULT_REFERENCE_FAR',
    'SCHEMA',
    'SELECTION_RULE',
    'select_operating_point',
    'split_operating_subsets',
]
