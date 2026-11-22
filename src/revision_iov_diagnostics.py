"""Observe frozen CIC-IoV-2024 operating points without selecting a threshold."""

from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from .revision_data import _feature_keys
from .revision_iov_id_gate import ID_GATE_RULE, TrainIdGate, id_thresholds
from .revision_report import summarize, write_csv
from .revision_v12 import _write

DATASET = 'CIC-IoV-2024'
MODES = ('base_threshold', 'local_only', 'full')
FIXED_MODES = ('fixed_base_local', 'fixed_base_full')
ALL_MODES = (*MODES, *FIXED_MODES)
ROLES = ('cal_holdout', 'val')
OPERATING_FILE = 'iov_operating_diagnostics.csv'
OVERLAP_FILE = 'iov_alarm_overlap.csv'
CLASS_FILE = 'iov_class_metrics.csv'
PAIRED_FILE = 'iov_fixed_mask_paired.csv'
STATUS_FILE = 'iov_fixed_mask_status.json'
COMPARISONS = (('local_only', 'base_threshold'), ('full', 'base_threshold'), ('full', 'local_only'))
FIXED_COMPARISONS = (
    ('fixed_base_local', 'base_threshold'),
    ('fixed_base_full', 'base_threshold'),
    ('fixed_base_full', 'fixed_base_local'),
    ('fixed_base_local', 'local_only'),
    ('fixed_base_full', 'full'),
)


@dataclass(frozen=True)
class _Population:
    """Keep labels and local exact-input group indices in their scoring order."""

    labels: np.ndarray
    groups: np.ndarray


def _population(part, rows):
    """Align labels and exact pre-normalization feature groups."""
    features, labels = part
    _, groups = np.unique(_feature_keys(features[rows]), return_inverse=True)
    return _Population(labels[rows], groups)


def _populations(parts, roster, splits):
    """Use the already assigned CAL holdout and receiver-independent VAL routes."""
    return {
        (client, role): population
        for client, (route, (_, holdout)) in enumerate(zip(roster, splits, strict=True))
        for role, population in (
            ('cal_holdout', _population(parts['cal'], route['cal'][holdout])),
            ('val', _population(parts['val'], route['val'])),
        )
    }


def _rate(numerator, denominator):
    """Represent rates without support as unavailable."""
    return numerator / denominator if denominator else None


def _group_count(groups, positive):
    """Count groups that contain at least one selected row."""
    return int(np.unique(groups[positive]).size)


def _counts(population, alarms):
    """Report row rates and groups with any alarm or miss on the relevant class."""
    groups = population.groups
    benign = population.labels == 0
    attack = ~benign
    false_alarm = benign & alarms
    missed_attack = attack & ~alarms
    masks = {'benign': benign, 'attack': attack, 'false_alarm': false_alarm, 'missed_attack': missed_attack}
    counts = {f'{name}_rows': int(mask.sum()) for name, mask in masks.items()}
    counts.update({f'{name}_groups': _group_count(groups, mask) for name, mask in masks.items()})
    return {
        **counts,
        'far': _rate(counts['false_alarm_rows'], counts['benign_rows']),
        'fnr': _rate(counts['missed_attack_rows'], counts['attack_rows']),
        'group_far': _rate(counts['false_alarm_groups'], counts['benign_groups']),
        'group_fnr': _rate(counts['missed_attack_groups'], counts['attack_groups']),
    }


def _tail(values, name):
    """Record score distribution tails without retaining logits or row values."""
    levels = (0.90, 0.95, 0.99, 0.999, 1.0)
    fields = ('q90', 'q95', 'q99', 'q999', 'max')
    quantiles = np.quantile(values, levels, method='higher').tolist() if values.size else [None] * len(fields)
    return {f'{name}_score_{key}': value for key, value in zip(fields, quantiles, strict=True)}


def _false_alarm_concentration(groups, benign, alarms):
    """Measure how many false alarms come from high-frequency exact inputs."""
    _, counts = np.unique(groups[benign & alarms], return_counts=True)
    largest = np.sort(counts)[-5:]
    false_alarm_rows = int(counts.sum())
    top = int(largest[-1]) if largest.size else 0
    return {
        'top_false_alarm_group_rows': top,
        'top_false_alarm_group_share': _rate(top, false_alarm_rows),
        'top_five_false_alarm_group_rows': int(largest.sum()),
        'top_five_false_alarm_group_share': _rate(int(largest.sum()), false_alarm_rows),
        'top_false_alarm_group_far_contribution': _rate(top, int(benign.sum())),
    }


def _calibration_evidence(values, alarms, record):
    """Reproduce the actual CAL quantile, floor and strict calibration FAR."""
    observed_far = _rate(int(alarms.sum()), len(values))
    if observed_far != record['calibration_far'] or len(values) != record['benign_holdout_rows']:
        raise ValueError('IoV CAL counts differ from the original calibration receipt')
    if record['calibration_rule'] == ID_GATE_RULE:
        return {}
    floor, target = float(record['threshold_floor']), float(record['target_far'])
    quantile = float(np.quantile(values, 1 - target, method='higher')) if len(values) else None
    empirical = record['calibration_rule'] == 'empirical_benign_holdout_quantile_higher'
    if empirical and record['threshold'] != max(floor, quantile):
        raise ValueError('IoV diagnostic quantile does not reproduce the original CAL threshold')
    return {
        'cal_benign_quantile_before_floor': quantile,
        'cal_floor_binding': quantile <= floor if empirical else None,
        'cal_all_benign_quantiles_at_or_below_floor': bool(values.max() <= floor) if values.size else None,
    }


def _operating_row(population, scores, alarms, record, role, thresholds):
    """Measure the scored role using the unmodified branch threshold."""
    benign = population.labels == 0
    return {
        **{field: record[field] for field in ('threshold', 'target_far', 'threshold_floor', 'calibration_rule')},
        **{field: record[field] for field in ('threshold_seen_id', 'threshold_unseen_id') if field in record},
        **(_calibration_evidence(scores[benign], alarms[benign], record) if role == 'cal_holdout' else {}),
        **_counts(population, alarms),
        **_tail(scores[benign], 'benign'),
        **_tail(scores[~benign], 'attack'),
        **_false_alarm_concentration(population.groups, benign, alarms),
        'equal_to_threshold_rows': int(np.count_nonzero(scores == thresholds)),
    }


def _overlap_counts(reference, candidate, groups):
    """Measure matched alarm overlap and both directions of alarm change."""
    before, after = np.unique(groups[reference]), np.unique(groups[candidate])
    return {
        'alert_rows_both': int((reference & candidate).sum()),
        'alert_rows_added': int((~reference & candidate).sum()),
        'alert_rows_removed': int((reference & ~candidate).sum()),
        'alert_groups_both': int(np.intersect1d(before, after).size),
        'alert_groups_added': int(np.setdiff1d(after, before, assume_unique=True).size),
        'alert_groups_removed': int(np.setdiff1d(before, after, assume_unique=True).size),
    }


def _alarm_overlap(population, reference, candidate):
    """Compare identical rows separately for actual benign and attack traffic."""
    return {
        f'{label}_{key}': value
        for label, selected in (('benign', population.labels == 0), ('attack', population.labels > 0))
        for key, value in _overlap_counts(reference[selected], candidate[selected], population.groups[selected]).items()
    }


def _alarm_mask(scores, threshold, population):
    """Use the same strict binary gate as the original detector."""
    if scores.shape != population.labels.shape or not np.isfinite(scores).all():
        raise ValueError('IoV diagnostic arrays are not aligned with the recorded population')
    return scores > threshold


def _alarm_sha256(alarms):
    """Hash ordered alarm bits with their length as a parity commitment."""
    length = np.array([alarms.size], dtype='<u8').tobytes()
    return sha256(length + np.packbits(alarms, bitorder='little').tobytes()).hexdigest()


def _score_sha256(scores):
    """Commit to the exact ordered attack scores without publishing row data."""
    return sha256(np.asarray(scores, dtype='<f8').tobytes()).hexdigest()


def _confusion(labels, predictions, classes):
    """Count the full label space and reject invalid subtype assignments."""
    if (
        predictions.shape != labels.shape
        or not np.issubdtype(predictions.dtype, np.integer)
        or np.any((predictions < 0) | (predictions >= classes))
        or np.any((labels < 0) | (labels >= classes))
    ):
        raise ValueError('IoV class prediction is outside the recorded class order')
    return np.bincount(classes * labels + predictions, minlength=classes**2).reshape(classes, classes)


def _class_measures(matrix, class_names):
    """Keep unsupported classes visible and separate from zero recall."""
    for name, support, predicted, true_positives in zip(
        class_names, matrix.sum(1), matrix.sum(0), matrix.diagonal(), strict=True
    ):
        support, predicted, true_positives = int(support), int(predicted), int(true_positives)
        yield {
            'class': name,
            'support': support,
            'predicted': predicted,
            'true_positives': true_positives,
            'precision': (true_positives / predicted if predicted else 0.0) if support or predicted else None,
            'recall': _rate(true_positives, support),
            'f1': _rate(2 * true_positives, support + predicted) if support else None,
        }


def _pooled_measures(matrix, class_rows):
    """Use the native metric convention of averaging observed true classes."""
    observed = [row['f1'] for row in class_rows if row['support']]
    return {
        'far': _rate(int(matrix[0, 1:].sum()), int(matrix[0].sum())),
        'fnr': _rate(int(matrix[1:, 0].sum()), int(matrix[1:].sum())),
        'macro_f1': float(np.mean(observed)) if len(observed) >= 2 else None,
    }


def _validate_val_metrics(rows, evaluations, pooled, modes):
    """Cross-check summed client error counts against the native VAL evaluation."""
    entries = {entry['mode']: entry for entry in evaluations if entry['role'] == 'val'}
    for mode in modes:
        val = [row for row in rows if row['mode'] == mode and row['role'] == 'val']
        by_metric = (('far', 'false_alarm_rows', 'benign_rows'), ('fnr', 'missed_attack_rows', 'attack_rows'))
        for metric, errors, support in by_metric:
            measured = _rate(sum(row[errors] for row in val), sum(row[support] for row in val))
            if measured != entries[mode]['metrics'][metric]:
                raise ValueError(f'IoV diagnostics differ from original VAL {mode}/{metric}')
        if mode in FIXED_MODES and 'matrix' not in entries[mode]:
            raise ValueError('IoV fixed-base evaluation lacks its native VAL class confusion matrix')
        if 'matrix' in entries[mode] and not np.array_equal(pooled[mode, 'val'], entries[mode]['matrix']):
            raise ValueError(f'IoV diagnostic class counts differ from native VAL {mode}')


def _check_fixed_parity(rows):
    """Assert fixed branches use the base threshold, score and mask in both roles."""
    identities = {(row['mode'], row['client'], row['role']): row for row in rows}
    keys = (
        'threshold', 'threshold_seen_id', 'threshold_unseen_id', 'ordered_attack_score_sha256', 'alarm_mask_sha256',
    )
    for (mode, client, role), row in identities.items():
        if mode not in FIXED_MODES:
            continue
        base = identities['base_threshold', client, role]
        if any(row.get(field) != base.get(field) for field in keys):
            raise ValueError(f'IoV fixed-base score or mask differs in {mode}/{client}/{role}')


class IoVOperatingDiagnostics:
    """Observe live CAL/VAL probabilities, retaining only aggregate evidence."""

    def __init__(
        self, parts, roster, cal_splits, *, class_names=None, fixed_base_alarm=False, can_id_strata=False,
        id_gate_column=None,
    ):
        self._populations = _populations(parts, roster, cal_splits)
        self._unseen_ids = {}
        if can_id_strata or id_gate_column is not None:
            for client, (route, (_, holdout)) in enumerate(zip(roster, cal_splits, strict=True)):
                column = 0 if id_gate_column is None else id_gate_column
                train_ids = parts['train'][0][route['train'], column]
                gate = TrainIdGate(train_ids) if id_gate_column is not None else None
                if gate is None:
                    train_ids = np.unique(train_ids)
                for role, data_role, rows in (
                    ('cal_holdout', 'cal', route['cal'][holdout]), ('val', 'val', route['val']),
                ):
                    ids = parts[data_role][0][rows, column]
                    self._unseen_ids[client, role] = ~(gate.seen(ids) if gate is not None else np.isin(ids, train_ids))
        self._names = tuple(class_names) if class_names is not None else None
        self._modes = ALL_MODES if fixed_base_alarm else MODES
        self._alarms = {}
        self._matrices = {}
        self._operating = []

    def observe(
        self,
        mode,
        client,
        cal_probabilities,
        calibration,
        val_scores,
        val_predictions,
        *,
        cal_predictions=None,
    ):
        """Observe one unmodified branch and verify original VAL alarm decisions."""
        if mode not in self._modes:
            return
        if mode in FIXED_MODES and cal_predictions is None:
            raise ValueError('IoV fixed-base branch did not supply its original CAL subtype decisions')
        if self._names is None:
            self._names = tuple(map(str, range(cal_probabilities.shape[1])))
        cal_scores = np.clip(cal_probabilities[:, 1:].sum(1), 0, 1)
        if cal_predictions is None:
            thresholds = self._thresholds(calibration, client, 'cal_holdout')
            cal_predictions = np.where(cal_scores > thresholds, cal_probabilities[:, 1:].argmax(1) + 1, 0)
        self._observe_role((mode, client, 'cal_holdout'), cal_scores, cal_predictions, calibration)
        self._observe_role((mode, client, 'val'), val_scores, val_predictions, calibration)

    def _observe_role(self, key, scores, predictions, calibration):
        """Measure each role against the exact native score and class decisions."""
        mode, client, role = key
        population = self._populations[client, role]
        thresholds = self._thresholds(calibration, client, role)
        alarms = _alarm_mask(scores, thresholds, population)
        if not np.array_equal(alarms, predictions > 0):
            raise ValueError(f'IoV {role.upper()} score/cutoff replay changed the original alarm mask')
        if key in self._alarms:
            raise ValueError('IoV observed a branch/client role twice')
        self._alarms[key] = alarms
        self._matrices[key] = _confusion(population.labels, predictions, len(self._names))
        row = _operating_row(population, scores, alarms, calibration, role, thresholds)
        if self._unseen_ids:
            unseen = self._unseen_ids[client, role]
            for name, mask in (('seen_id', ~unseen), ('unseen_id', unseen)):
                scope = _Population(population.labels[mask], population.groups[mask])
                row.update({f'{name}_{key}': value for key, value in _counts(scope, alarms[mask]).items()})
        row['alarm_mask_sha256'] = _alarm_sha256(alarms)
        row['ordered_attack_score_sha256'] = _score_sha256(scores)
        if role == 'val':
            row['strict_score_alarm_parity'] = True
            row['val_prediction_sha256'] = sha256(np.asarray(predictions, dtype='<i8').tobytes()).hexdigest()
        self._operating.append({'mode': mode, 'client': client, 'role': role, **row})

    def _thresholds(self, calibration, client, role):
        if calibration['calibration_rule'] == ID_GATE_RULE:
            return id_thresholds(~self._unseen_ids[client, role])
        return calibration['threshold']

    def finish(self, evaluations):
        """Return complete aggregates while discarding temporary prediction masks."""
        if len(self._alarms) != len(self._modes) * len(self._populations):
            raise ValueError('Incomplete IoV CAL/VAL observation')
        _check_fixed_parity(self._operating)
        pooled = self._pooled_matrices()
        _validate_val_metrics(self._operating, evaluations, pooled, self._modes)
        pairs = (*COMPARISONS, *FIXED_COMPARISONS) if self._modes == ALL_MODES else COMPARISONS
        overlap = [self._comparison(key, pair) for key in self._populations for pair in pairs]
        classes = self._class_table(pooled)
        measures = self._pooled_table(pooled)
        self._alarms.clear()
        self._matrices.clear()
        return {'operating': self._operating, 'overlap': overlap, 'classes': classes, 'measures': measures}

    def _pooled_matrices(self):
        """Pool class counts across the disjoint client row populations."""
        return {
            (mode, role): np.sum(
                [self._matrices[mode, client, role] for client, kind in self._populations if kind == role],
                axis=0,
            )
            for mode in self._modes
            for role in ROLES
        }

    def _class_table(self, pooled):
        """Export client and pooled per-class rows for both CAL and VAL."""
        matrices = {
            **self._matrices,
            **{(mode, 'overall', role): matrix for (mode, role), matrix in pooled.items()},
        }
        return [
            {
                'mode': mode,
                'client': client,
                'role': role,
                'scope': 'overall' if client == 'overall' else 'client',
                **row,
            }
            for (mode, client, role), matrix in matrices.items()
            for row in _class_measures(matrix, self._names)
        ]

    def _pooled_table(self, pooled):
        """Preserve mode-level FAR, FNR and macro-F1 for paired seed analysis."""
        return [
            {'mode': mode, 'role': role, **_pooled_measures(matrix, _class_measures(matrix, self._names))}
            for (mode, role), matrix in pooled.items()
        ]

    def _comparison(self, key, pair):
        """Measure directional agreement between two matched branches."""
        client, role = key
        candidate, reference = pair
        before, after = (self._alarms[mode, client, role] for mode in (reference, candidate))
        return {
            'client': client,
            'role': role,
            'reference': reference,
            'candidate': candidate,
            **_alarm_overlap(self._populations[key], before, after),
        }


def write_iov_diagnostics(output, cases):
    """Publish matched CAL/VAL evidence without altering the native operating points."""
    for key, name in (('operating', OPERATING_FILE), ('overlap', OVERLAP_FILE)):
        rows = [{'seed': case['seed'], **row} for case in cases for row in case['iov_diagnostics'][key]]
        write_csv(output / name, rows)
    classes = [{'seed': case['seed'], **row} for case in cases for row in case['iov_diagnostics'].get('classes', [])]
    if classes:
        write_csv(output / CLASS_FILE, classes)
    if any(row['mode'] in FIXED_MODES for row in classes):
        comparisons = _fixed_intervals(cases)
        write_csv(output / PAIRED_FILE, comparisons)
        _write(output / STATUS_FILE, _fixed_status(cases, comparisons))


def _seed_measures(evidence):
    """Index metrics for paired overall analysis with class support retained."""
    pooled = {
        (row['role'], row['mode'], '', metric): row[metric]
        for row in evidence['measures']
        for metric in ('far', 'fnr', 'macro_f1')
    }
    pooled.update(
        {
            (row['role'], row['mode'], row['class'], metric): row[metric]
            for row in evidence['classes']
            if row['scope'] == 'overall'
            for metric in ('precision', 'recall', 'f1')
        }
    )
    return pooled


def _delta(candidate, reference):
    """Pair only experiments in which both values are defined."""
    return candidate - reference if candidate is not None and reference is not None else None


def _fixed_intervals(cases):
    """Preserve paired CAL and VAL effects across shared-capture seeds."""
    measures = [_seed_measures(case['iov_diagnostics']) for case in cases]
    if len({case['seed'] for case in cases}) != len(cases) or any(
        item.keys() != measures[0].keys() for item in measures
    ):
        raise ValueError('IoV fixed-base cases have unmatched seeds or metric scope')
    rows = []
    for candidate, reference in FIXED_COMPARISONS:
        for role, mode, cls, metric in measures[0]:
            if mode != candidate:
                continue
            values = [_delta(row[role, candidate, cls, metric], row[role, reference, cls, metric]) for row in measures]
            rows.append(
                {
                    'role': role,
                    'candidate': candidate,
                    'reference': reference,
                    'class': cls,
                    'metric': metric,
                    **summarize(values, bounds=(-1, 1)),
                }
            )
    return rows


def _branch_far(cases):
    """Assemble the expected CAL cap and observed fixed-base group VAL rates."""
    observations = {}
    for mode in ('base_threshold', 'fixed_base_full'):
        observations[mode] = {
            str(case['seed']): next(
                row['far']
                for row in case['iov_diagnostics']['measures']
                if row['role'] == 'val' and row['mode'] == mode
            )
            for case in cases
        }
    caps = {row['target_far'] for case in cases for row in case['iov_diagnostics']['operating']}
    if len(caps) != 1 or observations['base_threshold'] != observations['fixed_base_full']:
        raise ValueError('IoV fixed-base FAR or configured cap differs across matched cases')
    return caps.pop(), observations['fixed_base_full']


def _fixed_status(cases, comparisons):
    """Record what the subtype-only development contrast establishes."""
    cap, observed = _branch_far(cases)
    defined = [value for value in observed.values() if value is not None]
    absolute_far_met = len(defined) == len(cases) and all(value <= cap for value in defined)
    effect = _fixed_effects(comparisons)
    positive = {
        key: row['seeds_with_metric'] == len(cases) and row['ci95_low'] is not None and row['ci95_low'] > 0
        for key, row in effect.items()
        if key in {'fixed_full_vs_base_val_macro_f1', 'fixed_full_vs_fixed_local_val_macro_f1'}
    }
    return {
        'scientific_status': _fixed_verdict(absolute_far_met, positive),
        'evaluation_scope': 'grouped same-source development CAL and VAL across matched seeds',
        'exact_base_binary_mask_and_score_match': True,
        'val_far_cap': cap,
        'val_seed_far': observed,
        'val_max_far': max(defined, default=None),
        'val_seeds_over_far_cap': sum(value > cap for value in defined),
        'val_seed_count': len(cases),
        'val_absolute_far_met': absolute_far_met,
        'positive_macro_f1_intervals': positive,
        'paired_effects': effect,
        'threshold_candidate_evaluated': False,
        'test_evaluated': False,
        'deployment_promoted': False,
    }


def _fixed_effects(comparisons):
    """Surface CAL trade-off and matched VAL subtype controls."""
    selected = {
        ('fixed_base_full', 'base_threshold', 'val', 'macro_f1'): 'fixed_full_vs_base_val_macro_f1',
        ('fixed_base_full', 'fixed_base_local', 'val', 'macro_f1'): 'fixed_full_vs_fixed_local_val_macro_f1',
        ('fixed_base_full', 'full', 'cal_holdout', 'fnr'): 'fixed_full_vs_adapted_cal_fnr',
        ('fixed_base_full', 'full', 'val', 'far'): 'fixed_full_vs_adapted_val_far',
        ('fixed_base_full', 'full', 'val', 'fnr'): 'fixed_full_vs_adapted_val_fnr',
    }
    effects = {}
    for row in comparisons:
        key = tuple(row[field] for field in ('candidate', 'reference', 'role', 'metric'))
        if not row['class'] and key in selected:
            effects[selected[key]] = row
    return effects


def _fixed_verdict(far_pass, macro_pass):
    """Keep empirical operating and subtype criteria explicit."""
    if not far_pass:
        return 'development_fixed_alarm_absolute_far_not_met'
    if len(macro_pass) != 2 or not all(macro_pass.values()):
        return 'development_fixed_alarm_subtype_gain_not_established'
    return 'development_fixed_alarm_subtype_criteria_met'
