"""Summarize NB15 measurements from the current report's verified table copies."""

import csv
import json
from math import isclose

from .revision_report import write_csv

BINARY_TABLE = 'nb15_binary_working_points.csv'
CLASS_TABLE = 'nb15_receiver_class_metrics.csv'
FRONTIER_TABLE = 'nb15_donor_frontier_stages.csv'
BRANCHES = ('reference', 'candidate')
CLASS_BRANCHES = ('base_threshold', 'legacy_head_local', 'legacy_head_federated')
BINARY_PAIR_ROWS = 'nb15_pair/binary_pair_by_seed.csv'
BINARY_STATUS = 'nb15_pair/binary_pair_status.json'
DONOR_METRICS = 'donor_check/metrics.csv'
DONOR_AUDIT = 'donor_check/guard_audit.csv'
RATE_FIELDS = ('benign_rows', 'false_positives', 'attack_rows', 'false_negatives')
CLASS_FIELDS = ('support', 'predicted', 'precision', 'recall', 'f1')
FRONTIER_FIELDS = (
    'unique_group_margin_breakpoints',
    'evaluated_cutoff_regimes',
    'feasible_cutoff_regimes',
    'min_wrong_rewrite_rate_with_recovery',
)


def _read_csv(root, name):
    """Return rows from one current-run copy when its stage supplied it."""
    path = root / name
    if not path.is_file():
        return []
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def _index(rows, fields):
    """Reject ambiguous measurement identities."""
    indexed = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in indexed:
            raise ValueError(f'Repeated NB15 measurement for {key}')
        indexed[key] = row
    return indexed


def _diagnostic_path(branch):
    """Resolve the copied native branch diagnostics relative to tables."""
    return f'nb15_pair/{branch}/binary_diagnostics.csv'


def _diagnostics(tables, branch):
    """Use fixed-subtype diagnostics to describe unchanged scalar TabM alarms."""
    rows = _read_csv(tables, _diagnostic_path(branch))
    selected = [row for row in rows if row.get('role') == 'val' and row.get('mode') == 'fixed_federated']
    return _index(selected, ('seed', 'client'))


def _receiver_rows(diagnostics):
    """Identify the actual receiver client for each matched seed from its flag."""
    rows = [row for row in diagnostics.values() if row['receiver'].lower() == 'true']
    return {seed: row for (seed,), row in _index(rows, ('seed',)).items()}


def _rate(numerator, denominator):
    """Keep unsupported rates missing and reject inconsistent counts."""
    if not 0 <= numerator <= denominator:
        raise ValueError('Invalid NB15 binary or subtype count')
    return numerator / denominator if denominator else None


def _check_rate(value, recorded):
    """Cross-check the rate calculated from counts against the recorded rate."""
    if value is None and recorded in (None, ''):
        return
    if value is None or recorded in (None, '') or not isclose(value, float(recorded), rel_tol=0, abs_tol=1e-12):
        raise ValueError('NB15 diagnostic counts do not reproduce the bound metric')


def _binary_counts(rows, split):
    """Sum client counts for pooled FAR/FNR instead of averaging client rates."""
    counts = {field: sum(int(row[f'{split}_{field}']) for row in rows) for field in RATE_FIELDS}
    rates = {
        'far': _rate(counts['false_positives'], counts['benign_rows']),
        'fnr': _rate(counts['false_negatives'], counts['attack_rows']),
    }
    return {f'{split}_{key}': value for key, value in {**counts, **rates}.items()}


def _working_point(diagnostics, pair, branch, far_limit):
    """Join one objective's diagnostics to the recorded paired VAL working point."""
    rows = [row for (seed, _), row in diagnostics.items() if seed == pair['seed']]
    receiver = _receiver_rows(_index(rows, ('seed', 'client')))[pair['seed']]
    if pair['scope'] == 'receiver':
        if receiver['client'] != pair[f'{branch}_receiver']:
            raise ValueError('NB15 paired measurement and diagnostics have different receivers')
        rows = [receiver]
    counts = {**_binary_counts(rows, 'cal'), **_binary_counts(rows, 'val')}
    for metric in ('far', 'fnr', 'attack_rows', 'benign_rows'):
        _check_rate(counts[f'val_{metric}'], pair[f'{branch}_{metric}'])
    if pair['scope'] == 'receiver':
        _check_rate(counts['cal_far'], pair[f'{branch}_cal_far'])
    values = {**counts, 'val_far_pass': counts['val_far'] is not None and counts['val_far'] <= far_limit}
    values.update({metric: pair.get(f'{branch}_{metric}') for metric in ('ece', 'macro_f1')})
    values['cal_scope'] = '+'.join(sorted({row['cal_scope'] for row in rows}))
    values['cal_thresholds_by_client'] = json.dumps({row['client']: row['threshold'] for row in rows}, sort_keys=True)
    values['diagnostic_source'] = f'tables/{_diagnostic_path(branch)}'
    return {f'{branch}_{key}': value for key, value in values.items()}


def _binary_criteria(status, pair):
    """Report per-seed absolute FAR and scope-wide paired FNR suitability separately."""
    scope = status['scopes'][pair['scope']]
    classification = status['classification_guard']
    return {
        'far_limit': status['criteria']['max_far_each_seed'],
        'fnr_delta': pair['delta_fnr'],
        'fnr_ci95_upper': scope['candidate_minus_reference_fnr']['ci95_high'],
        'fnr_max_increase': status['criteria']['max_paired_fnr_increase_ci95_upper'],
        'fnr_pass': scope['fnr_pass'],
        'macro_f1_mean_delta': classification['candidate_minus_reference_macro_f1']['mean'],
        'macro_f1_pass': classification['pass'],
        'combined_acceptance': status['combined_acceptance'],
        'selection_on_val': status['selection_on_val'],
    }


def _binary_rows(tables, diagnostics):
    """Derive paired CAL and VAL evidence for the two locked detector objectives."""
    pairs = _index(_read_csv(tables, BINARY_PAIR_ROWS), ('seed', 'scope'))
    status_path = tables / BINARY_STATUS
    if not pairs or not status_path.is_file() or not all(diagnostics.values()):
        return []
    status = json.loads(status_path.read_text(encoding='utf-8-sig'))
    result = []
    for (seed, scope), pair in sorted(pairs.items()):
        criteria = _binary_criteria(status, pair)
        row = {'seed': seed, 'scope': scope}
        for branch in BRANCHES:
            row.update(_working_point(diagnostics[branch], pair, branch, criteria['far_limit']))
        result.append(
            {
                **row,
                **criteria,
                'paired_source': f'tables/{BINARY_PAIR_ROWS}',
                'criteria_source': f'tables/{BINARY_STATUS}',
            }
        )
    return result


def _class_metrics(metric_map, seed, branch, scope):
    """Read real scoped class metrics and recover the integer TP from recall."""
    rows = {name: metric_map[seed, branch, scope, name]['value'] for name in CLASS_FIELDS}
    support, predicted = int(rows['support']), int(rows['predicted'])
    if support < 0 or predicted < 0:
        raise ValueError('Invalid NB15 subtype count')
    if not support:
        _check_rate(0.0 if predicted else None, rows['precision'])
        _check_rate(None, rows['recall'])
        _check_rate(None, rows['f1'])
        return {**rows, 'true_positive': 0}
    if rows['recall'] in (None, ''):
        raise ValueError('NB15 subtype with positive support is missing recall')
    true_positive = round(support * float(rows['recall']))
    _check_rate(_rate(true_positive, support), rows['recall'])
    _check_rate(_rate(true_positive, predicted) or 0.0, rows['precision'])
    _check_rate(_rate(2 * true_positive, support + predicted), rows['f1'])
    return {**rows, 'true_positive': true_positive}


def _receiver_classes(metrics, receivers):
    """Keep all declared classes and preserve zero support and zero recall."""
    result = []
    for seed, receiver in sorted(receivers.items()):
        classes = json.loads(receiver['class_names'])
        for branch in CLASS_BRANCHES:
            for label in classes.values():
                values = _class_metrics(metrics, seed, branch, f'receiver_class:{label}')
                result.append(
                    {
                        'seed': seed,
                        'receiver_client': receiver['client'],
                        'scope': 'receiver',
                        'data_role': 'VAL',
                        'branch': branch,
                        'class': label,
                        **values,
                        'source_table': f'tables/{DONOR_METRICS}',
                    }
                )
    return result


def _frontier_stage(audit):
    """Separate donor feasibility from an independent receiver CAL acceptance."""
    receipt = json.loads(audit['receipt'])
    selected = receipt['selected_threshold']
    receiver = receipt.get('receiver_check')
    if (selected is None) != (receiver is None):
        raise ValueError('NB15 receiver check must only follow donor cutoff selection')
    return {
        'threshold_search': audit.get('threshold_search') or 'fixed_donor_cal_grid',
        'donor_status': 'feasible' if selected is not None else receipt['status'],
        'receiver_status': receipt['status'] if receiver is not None else 'not_checked',
        'selected_threshold': selected,
        'deployed_threshold': receipt.get('threshold'),
        **{field: audit.get(field) for field in FRONTIER_FIELDS},
        'receiver_cal_negative_groups': receiver['negative_groups'] if receiver is not None else None,
        'receiver_cal_proposed_wrong_groups': (
            receiver['proposed_wrong_rewrite_groups'] if receiver is not None else None
        ),
    }


def _frontier_rows(tables, metrics, receivers):
    """Join the frontier stages with the corresponding actual VAL subtype measures."""
    audits = _index(_read_csv(tables, DONOR_AUDIT), ('seed', 'branch'))
    result = []
    for (seed, branch), audit in sorted(audits.items()):
        receiver = receivers[seed]
        val = _class_metrics(metrics, seed, branch, 'receiver')
        baseline = _class_metrics(metrics, seed, 'base_threshold', 'receiver')
        if val['support'] != baseline['support']:
            raise ValueError('NB15 donor replay changed receiver VAL support')
        result.append(
            {
                'seed': seed,
                'branch': branch,
                'receiver_client': receiver['client'],
                'target_class': receiver['target_class'],
                **_frontier_stage(audit),
                **{f'val_target_{name}': val[name] for name in CLASS_FIELDS},
                'val_target_tp': val['true_positive'],
                'val_base_target_tp': baseline['true_positive'],
                'val_target_tp_delta': val['true_positive'] - baseline['true_positive'],
                'source_audit': f'tables/{DONOR_AUDIT}',
                'source_metrics': f'tables/{DONOR_METRICS}',
            }
        )
    return result


def write_nb15_review_tables(working, tables):
    """Write paper tables using only the caller's verified current-run copies."""
    diagnostics = {branch: _diagnostics(tables, branch) for branch in BRANCHES}
    receivers = _receiver_rows(diagnostics['reference'])
    metrics = _index(_read_csv(tables, DONOR_METRICS), ('seed', 'branch', 'scope', 'metric'))
    outputs = {BINARY_TABLE: _binary_rows(tables, diagnostics)}
    if metrics and receivers:
        outputs[CLASS_TABLE] = _receiver_classes(metrics, receivers)
        outputs[FRONTIER_TABLE] = _frontier_rows(tables, metrics, receivers)
    counts = {}
    for filename, rows in outputs.items():
        if rows:
            write_csv(working / filename, rows)
            counts[filename] = len(rows)
    return counts
