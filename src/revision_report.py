"""Summarize paired seed runs without pooling seeds as independent records."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from tempfile import NamedTemporaryFile

import numpy as np

from .revision_supplement import REPAIR_PAIRS, repair_conditions, repair_pairs_for_modes

METRICS = ('accuracy', 'weighted_f1', 'macro_f1', 'balanced_accuracy', 'far', 'fnr', 'ece', 'alert_rate')
PAIRS = {
    'base_threshold_minus_base': ('base_threshold', 'base'),
    'local_only_minus_base_threshold': ('local_only', 'base_threshold'),
    'conditional_full_minus_conditional_local': ('conditional_full', 'conditional_local'),
    'conditional_global_minus_conditional_local_rounds': ('conditional_global', 'conditional_local_rounds'),
    'conditional_full_argmax_minus_conditional_local_argmax': (
        'conditional_full_argmax',
        'conditional_local_argmax',
    ),
    'conditional_global_argmax_minus_conditional_local_rounds_argmax': (
        'conditional_global_argmax',
        'conditional_local_rounds_argmax',
    ),
    'conditional_local_argmax_minus_base_threshold': ('conditional_local_argmax', 'base_threshold'),
    'conditional_local_argmax_minus_base': ('conditional_local_argmax', 'base'),
    'conditional_full_minus_full': ('conditional_full', 'full'),
    'repair_rescue_minus_donor_only': ('transfer_repair_rescue', 'fedavg_threshold'),
    'repair_union_minus_donor_only': ('transfer_repair_union', 'fedavg_threshold'),
    'repair_fixed_minus_donor_only': ('transfer_repair_fixed_gate', 'fedavg_threshold'),
    'local_only_minus_base': ('local_only', 'base'),
    'full_minus_local_only': ('full', 'local_only'),
    'full_minus_local_unclipped': ('full', 'local_unclipped'),
    'local_only_minus_local_unclipped': ('local_only', 'local_unclipped'),
    'full_minus_without_prox': ('full', 'without_prox'),
    'full_minus_without_threshold': ('full', 'without_threshold'),
    'local_only_minus_local_fixed': ('local_only', 'local_fixed'),
    'base_threshold_minus_base_fixed': ('base_threshold', 'base_fixed'),
    'full_minus_fedavg': ('full', 'fedavg'),
    'full_minus_fedavg_threshold': ('full', 'fedavg_threshold'),
    'full_global_minus_local_rounds': ('full_global', 'local_rounds'),
    'full_minus_full_global': ('full', 'full_global'),
    'local_only_minus_local_rounds': ('local_only', 'local_rounds'),
    'full_fine_minus_local_only_fine': ('full_fine', 'local_only_fine'),
    'full_fine_minus_full': ('full_fine', 'full'),
    'local_only_fine_minus_local_only': ('local_only_fine', 'local_only'),
    'subtype_minus_base_threshold': ('base_threshold_subtype', 'base_threshold'),
    **REPAIR_PAIRS,
}
OBJECTIVE_MODES = ('base', 'base_threshold', 'local_only', 'full', 'fedavg', 'fedavg_threshold')
NETWORK_DATASETS = {'CIC-IDS-2017', 'CIC-UNSW-NB15'}


def divide(num, den):
    """Return elementwise num divided by den, with zero wherever den is nonpositive."""
    return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den > 0)


def wilson_interval(successes, rows, z=1.96):
    """Wilson score interval for a binomial proportion; None for empty rows."""
    if rows is None or rows <= 0:
        return None, None
    p = min(max(float(successes) / float(rows), 0.0), 1.0)
    denominator = 1.0 + z * z / rows
    centre = (p + z * z / (2.0 * rows)) / denominator
    half = z * math.sqrt(max(p * (1.0 - p) / rows + z * z / (4.0 * rows * rows), 0.0)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def with_class_intervals(row):
    """Attach support-aware Wilson intervals to one per-class metric row."""
    result = dict(row)
    support = row.get('support')
    if support and row.get('recall') is not None:
        low, high = wilson_interval(float(row['recall']) * support, support)
        result.update(recall_ci95_low=low, recall_ci95_high=high)
    predicted = row.get('predicted')
    if predicted and row.get('precision') is not None:
        low, high = wilson_interval(float(row['precision']) * predicted, predicted)
        result.update(precision_ci95_low=low, precision_ci95_high=high)
    return result


def score(y, predicted, q, names):
    """Macro-F1 uses observed true classes; class rows retain the full label map."""
    y, predicted, q = (np.asarray(a) for a in (y, predicted, q))
    k = len(names)
    if (
        y.ndim != 1
        or y.shape != predicted.shape
        or y.shape != q.shape
        or not np.issubdtype(y.dtype, np.integer)
        or not np.issubdtype(predicted.dtype, np.integer)
        or (y < 0).any()
        or (y >= k).any()
        or (predicted < 0).any()
        or (predicted >= k).any()
        or not np.isfinite(q).all()
        or (q < 0).any()
        or (q > 1).any()
    ):
        raise ValueError('Evaluation labels, decisions or attack probabilities are invalid')
    counts = np.bincount(k * y + predicted, minlength=k * k).reshape(k, k)
    support, prediction_count, tp = counts.sum(1), counts.sum(0), counts.diagonal()
    precision, recall, f1 = (
        divide(tp, prediction_count),
        divide(tp, support),
        divide(2 * tp, support + prediction_count),
    )
    observed = support > 0
    class_rows = [
        {
            'class': name,
            'support': int(support[i]),
            'predicted': int(prediction_count[i]),
            'precision': float(precision[i]) if observed[i] or prediction_count[i] else None,
            'recall': float(recall[i]) if observed[i] else None,
            'f1': float(f1[i]) if observed[i] else None,
        }
        for i, name in enumerate(names)
    ]
    benign = y == 0
    bins = np.minimum((q * 10).astype(int), 9)
    calibration_gap = np.bincount(bins, weights=q - (~benign), minlength=10)
    metrics = {
        'rows': len(y),
        'observed_classes': int(observed.sum()),
        'defined_classes': k,
        'accuracy': float(tp.sum() / len(y)) if len(y) else None,
        'weighted_f1': float(support @ f1 / len(y)) if len(y) else None,
        'macro_f1': float(f1[observed].mean()) if observed.sum() >= 2 else None,
        'balanced_accuracy': float(recall[observed].mean()) if observed.sum() >= 2 else None,
        'far': float((predicted[benign] > 0).mean()) if benign.any() else None,
        'fnr': float((predicted[~benign] == 0).mean()) if (~benign).any() else None,
        'ece': float(np.abs(calibration_gap).sum() / len(y)) if len(y) else None,
        'alert_rate': float((predicted > 0).mean()) if len(y) else None,
    }
    return metrics, class_rows, counts.tolist()


def write_csv(path, rows, fields=None):
    """Write rows to a CSV path, replacing it only after serialization succeeds."""
    path = Path(path)
    if not fields:
        rows = list(rows)
        fields = list(dict.fromkeys(field for row in rows for field in row))
    temporary = NamedTemporaryFile(
        mode='w',
        dir=path.parent,
        prefix=f'.{path.name}.',
        suffix='.partial',
        newline='',
        encoding='utf-8',
        delete=False,
    )
    partial = Path(temporary.name)
    try:
        with temporary as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            writer.writerows(rows)
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def _statistics(values):
    return {
        'n': len(values),
        'mean': mean(values) if values else None,
        'sd': stdev(values) if len(values) > 1 else None,
    }


def summarize(values, *, bounds=(0, 1)):
    """Student t interval across seeds on one fixed evaluation population."""
    from scipy.stats import t

    observed = [float(value) for value in values if value is not None]
    if any(not math.isfinite(value) or not bounds[0] <= value <= bounds[1] for value in observed):
        raise ValueError('Invalid metric value')
    stats = _statistics(observed)
    average, sd = stats['mean'], stats['sd']
    low = high = None
    if sd is not None:
        radius = float(t.ppf(0.975, stats['n'] - 1)) * sd / math.sqrt(stats['n'])
        low, high = max(bounds[0], average - radius), min(bounds[1], average + radius)
    return {
        'seeds_with_metric': stats['n'],
        'mean': average,
        'seed_sd': sd,
        'ci95_low': low,
        'ci95_high': high,
    }


def _support(entry):
    return {row['class']: row['support'] for row in entry['per_class']}


def balance_comparisons(results):
    """Pair training objectives by seed and identical ordered client partitions."""
    fields = ('dataset', 'protocol', 'clients', 'alpha')
    cases = {}
    for result in results:
        if result['protocol'] != 'grouped' or result['scenario'] not in ('main', 'balance'):
            continue
        key = tuple(result[name] for name in (*fields, 'seed', 'scenario'))
        if key in cases:
            raise ValueError(f'Duplicate objective comparison case: {key}')
        cases[key] = result
    profiles = {key[: len(fields)] for key in cases if key[-1] == 'balance'}
    for profile in profiles:
        reference_seeds = {key[-2] for key in cases if key[: len(fields)] == profile and key[-1] == 'main'}
        balanced_seeds = {key[-2] for key in cases if key[: len(fields)] == profile and key[-1] == 'balance'}
        if reference_seeds != balanced_seeds:
            raise ValueError(f'The two objectives have different seeds: {profile}')
    matches = defaultdict(list)
    for key, balanced in sorted(cases.items()):
        if key[-1] != 'balance':
            continue
        reference = cases.get((*key[:-1], 'main'))
        if reference is None:
            raise ValueError(f'Missing matched reference objective: {key[:-1]}')
        comparable_settings = (
            'train_rows',
            'base_epochs',
            'batch_size',
            'detector_lr',
            'detector_weight_decay',
            'prior_strength',
            'cal_fit_fraction',
            'target_far',
            'threshold_floor',
            'rounds',
            'adapter_epochs',
            'adapter_lr',
            'adapter_prox',
            'clip',
            'secure_scale',
        )
        if (
            not balanced.get('partition_sha256')
            or balanced['partition_sha256'] != reference.get('partition_sha256')
            or balanced['supports'] != reference['supports']
            or any(balanced['settings'].get(k) != reference['settings'].get(k) for k in comparable_settings)
        ):
            raise ValueError(f'Training budget or partition differs between objectives: {key[:-1]}')
        old_rows = [row for row in reference['evaluations'] if row['mode'] in OBJECTIVE_MODES]
        new_rows = [row for row in balanced['evaluations'] if row['mode'] in OBJECTIVE_MODES]
        original = {(row['role'], row['mode']): row for row in old_rows}
        revised = {(row['role'], row['mode']): row for row in new_rows}
        if original.keys() != revised.keys() or len(original) != len(old_rows) or len(revised) != len(new_rows):
            raise ValueError(f'Evaluation modes differ between objectives: {key[:-1]}')
        for (role, mode), evaluation in revised.items():
            old = original[role, mode]
            if _support(old) != _support(evaluation):
                raise ValueError(f'Evaluation support differs for {key[:-1]} {role} {mode}')
            for metric in METRICS:
                a, b = old['metrics'][metric], evaluation['metrics'][metric]
                values = matches[(*key[: len(fields)], role, mode, metric)]
                if a is not None and b is not None:
                    values.append((a, b))
    rows = []
    for (dataset, protocol, clients, alpha, role, mode, metric), values in sorted(matches.items()):
        deltas = [b - a for a, b in values]
        rows.append(
            {
                'dataset': dataset,
                'protocol': protocol,
                'clients': clients,
                'alpha': alpha,
                'role': role,
                'mode': mode,
                'metric': metric,
                'primary': (role, mode, metric) == ('test_unique_unseen', 'full', 'macro_f1'),
                'reference_mean': mean(a for a, _ in values) if values else None,
                'balanced_mean': mean(b for _, b in values) if values else None,
                'positive_seeds': sum(value > 1e-12 for value in deltas),
                'negative_seeds': sum(value < -1e-12 for value in deltas),
                'tied_seeds': sum(abs(value) <= 1e-12 for value in deltas),
                **summarize(deltas, bounds=(-1, 1)),
            }
        )
    return rows


def write_report(root, results, *, figures=True):
    """Write per-case results, seed summaries, paired effects, and available diagnostics under root.

    Read case dictionaries from results and create plots only when figures is true.
    """
    rows, per_class, counts, resources, communication, round_rows = [], [], [], [], [], []
    supports, settings, calibration, measurement = [], [], [], []
    grouped, paired, class_groups = defaultdict(list), defaultdict(list), defaultdict(list)
    group_conditions, pair_conditions = {}, {}
    for result in results:
        key = {name: result[name] for name in ('dataset', 'protocol', 'scenario', 'seed', 'clients', 'alpha')}
        resources.append({**key, **result['resources']})
        settings.append({**key, **result['settings']})
        supports.extend({**key, **row} for row in result['supports'])
        communication.extend({**key, **row} for row in result['communication'])
        round_rows.extend({**key, **row} for row in result['rounds'])
        calibration.extend({**key, **row} for row in result['calibration'])
        measurement.extend({**key, **row} for row in result['measurement'])
        by_role = defaultdict(dict)
        for entry in result['evaluations']:
            mode, role = entry['mode'], entry['role']
            identity = {**key, 'mode': mode, 'role': role}
            rows.append(
                {
                    **identity,
                    **repair_conditions(mode, result['settings']),
                    'score_kind': entry.get('score_kind', 'binary_attack_probability'),
                    **entry['metrics'],
                }
            )
            per_class.extend({**identity, **with_class_intervals(row)} for row in entry['per_class'])
            names = [row['class'] for row in entry['per_class']]
            for name, row in zip(names, entry['matrix']):
                counts.extend(
                    {**identity, 'actual': name, 'predicted': pred, 'count': count} for pred, count in zip(names, row)
                )
            by_role[role][mode] = entry['metrics']
            group = (
                key['dataset'],
                key['protocol'],
                key['scenario'],
                key['clients'],
                key['alpha'],
                role,
                mode,
            )
            conditions = repair_conditions(mode, result['settings'])
            if group in group_conditions and group_conditions[group] != conditions:
                raise ValueError('Cannot pool different repair policies under the same mode')
            group_conditions[group] = conditions
            for metric in METRICS:
                value = entry['metrics'][metric]
                grouped[group, metric]
                if value is not None:
                    grouped[group, metric].append(value)
            for row in entry['per_class']:
                for metric in ('support', 'precision', 'recall', 'f1'):
                    values = class_groups[(*group, row['class']), metric]
                    if row[metric] is not None:
                        values.append(row[metric])
        for role, modes in by_role.items():
            for contrast, (left, right) in {**PAIRS, **repair_pairs_for_modes(modes)}.items():
                if left not in modes or right not in modes:
                    continue
                group = (
                    key['dataset'],
                    key['protocol'],
                    key['scenario'],
                    key['clients'],
                    key['alpha'],
                    role,
                    contrast,
                )
                pair_conditions[group] = repair_conditions(left, result['settings'])
                for metric in METRICS:
                    a, b = modes[left][metric], modes[right][metric]
                    paired[group, metric]
                    if a is not None and b is not None:
                        paired[group, metric].append(a - b)

    fields = ('dataset', 'protocol', 'scenario', 'clients', 'alpha', 'role')

    def aggregate(groups, labels, *, intervals=False):
        rows = []
        for (group, metric), values in sorted(groups.items()):
            row = {**dict(zip((*fields, *labels), group)), 'metric': metric, **_statistics(values)}
            condition_map = pair_conditions if labels == ('contrast',) else group_conditions
            row.update(condition_map.get(group[:7], {}))
            if intervals:
                interval = summarize(values, bounds=(-1, 1))
                row.update(ci95_low=interval['ci95_low'], ci95_high=interval['ci95_high'])
            rows.append(row)
        return rows

    summary = aggregate(grouped, ('mode',))
    differences = aggregate(paired, ('contrast',), intervals=True)
    class_summary = aggregate(class_groups, ('mode', 'class'))
    objective_changes = balance_comparisons(results)
    for name, data in (
        ('metrics_by_seed', rows),
        ('metrics_mean_sd', summary),
        ('paired_differences', differences),
        ('per_class', per_class),
        ('per_class_mean_sd', class_summary),
        ('confusion_counts', counts),
        ('partition_support', supports),
        ('parameters', settings),
        ('resources', resources),
        ('communication', communication),
        ('adapter_rounds', round_rows),
        ('calibration', calibration),
        ('timings', measurement),
    ):
        write_csv(root / f'{name}.csv', data)
    if objective_changes:
        write_csv(root / 'balance_vs_reference.csv', objective_changes)
    transfer_intervals = []
    if any(row['scenario'].startswith('supplement_') for row in results):
        transfer_intervals = _write_supplement(root, results)
    if figures:
        _plot(root, results, differences, objective_changes, transfer_intervals)


def _write_supplement(root, results):
    from .revision_supplement import TRANSFER_PAIRS, TRANSFER_SCORE_METRICS, TRANSFER_SCORE_STATE_FIELDS

    base_identities = ('dataset', 'protocol', 'scenario', 'seed', 'clients', 'alpha')
    conditions = ('train_sampling', 'repair_gate_target_fraction', 'repair_residual_budget_fraction')
    identities = (*base_identities, *conditions)
    configuration = ('dataset', 'protocol', 'scenario', 'clients', 'alpha', *conditions)
    transfer, pairs, repair_pairs = [], [], []
    groups = defaultdict(list)
    mode_seeds, mode_groups, mode_intervals = [], defaultdict(list), []
    overall = defaultdict(list)
    seen_transfer_runs = set()
    scopes = ('train_and_fit_absent', 'fit_only_absent')
    score_detail, score_seeds = [], []
    score_groups = defaultdict(list)
    score_metrics = {**TRANSFER_SCORE_METRICS, **dict.fromkeys(TRANSFER_SCORE_STATE_FIELDS, (0, 1))}
    score_group_fields = (*configuration, 'transfer_scope', 'contrast')
    other = {'development_subtype': [], 'development_rank': [], 'quantization_diagnostics': []}
    for result in results:
        key = {name: result[name] for name in base_identities}
        key.update({name: result['settings'].get(name) for name in conditions})
        for field, data in (
            ('development_subtype', 'development_subtype'),
            ('development_rank', 'development_rank'),
            ('quantization_diagnostics', 'private_round_summary'),
        ):
            other[field].extend({**key, **row} for row in result.get(data, ()))
        if result['scenario'] != 'supplement_transfer':
            continue
        identity = tuple(key[name] for name in identities)
        if identity in seen_transfer_runs:
            raise ValueError('Duplicate transfer configuration and seed')
        seen_transfer_runs.add(identity)
        detail = result['transfer']
        repair = result['settings'].get('repair_missing_classes', False)
        for entry in result.get('evaluations', ()):
            if entry['role'] == 'test_unique_unseen':
                mode_key = {**key, **repair_conditions(entry['mode'], result['settings'])}
                overall[(*tuple(mode_key[name] for name in configuration), entry['mode'])].append(entry['metrics'])
        transfer.extend({**key, **repair_conditions(row['mode'], result['settings']), **row} for row in detail)
        scored = result.get('transfer_scores')
        if scored is not None:
            score_detail.extend({**key, **row} for row in scored)
        for scope in scopes:
            chosen = [row for row in detail if row['transfer_scope'] == scope]
            for mode in sorted({row['mode'] for row in chosen}):
                cells = [row for row in chosen if row['mode'] == mode]
                # A supported class with no predictions has precision zero for
                # this fixed-cell macro endpoint; never drop such cells.
                mode_row = {
                    **key,
                    **repair_conditions(mode, result['settings']),
                    'transfer_scope': scope,
                    'mode': mode,
                    'eligible_cells': len(cells),
                    'test_rows': sum(row['test_support'] for row in cells),
                    'zero_prediction_cells': sum(row['predicted'] == 0 for row in cells),
                    'macro_recall': mean(row['recall'] for row in cells),
                    'macro_precision': mean(row['precision'] if row['precision'] is not None else 0.0 for row in cells),
                    'macro_f1': mean(row['f1'] for row in cells),
                    'precision_zero_division': 0,
                }
                mode_seeds.append(mode_row)
                mode_groups[(*tuple(mode_row[name] for name in configuration), scope, mode)].append(mode_row)
            repair_contrasts = repair_pairs_for_modes({row['mode'] for row in chosen}) if repair else {}
            contrasts = {**TRANSFER_PAIRS, **repair_contrasts}
            for contrast, (left, right) in contrasts.items():
                a = {(r['client'], r['class']): r for r in chosen if r['mode'] == left}
                b = {(r['client'], r['class']): r for r in chosen if r['mode'] == right}
                if a.keys() != b.keys():
                    raise ValueError('Transfer comparison has unmatched receiver/class cells')
                delta = mean(a[k]['recall'] - b[k]['recall'] for k in a) if a else None
                row = {
                    **repair_conditions(left, result['settings']),
                    'transfer_scope': scope,
                    'contrast': contrast,
                    'eligible_cells': len(a),
                    'test_rows': sum(r['test_support'] for r in a.values()),
                    'delta_macro_recall': delta,
                }
                pairs.append({**key, **row})
                pair_key = {**key, **row}
                groups[(*tuple(pair_key[name] for name in configuration), scope, contrast)].append(delta)
                if contrast in repair_contrasts:
                    eligible = [k for k, value in a.items() if value['repair_eligible']]
                    repair_pairs.append(
                        {
                            **key,
                            **row,
                            'donor_train_supported_cells': len(eligible),
                            'donor_train_supported_test_rows': sum(a[k]['test_support'] for k in eligible),
                            'reference_recall': mean(v['recall'] for v in b.values()) if b else None,
                            'candidate_recall': mean(v['recall'] for v in a.values()) if a else None,
                            'reference_true_positive': sum(v['true_positive'] for v in b.values()),
                            'candidate_true_positive': sum(v['true_positive'] for v in a.values()),
                            'reference_predicted': sum(v['predicted'] for v in b.values()),
                            'candidate_predicted': sum(v['predicted'] for v in a.values()),
                        }
                    )
                if scored is not None and contrast in TRANSFER_PAIRS:
                    cells = [
                        item for item in scored if item['transfer_scope'] == scope and item['contrast'] == contrast
                    ]
                    eligible = {(item['client'], item['class'], item['test_support']) for item in a.values()}
                    observed = {(item['client'], item['class'], item['test_support']) for item in cells}
                    if eligible != observed or len(cells) != len(a):
                        raise ValueError('Score diagnostics and hard transfer use different cells')
                    values = {
                        metric: mean(item[metric] for item in cells) if cells else None for metric in score_metrics
                    }
                    seed_row = {**key, **row, **values}
                    score_seeds.append(seed_row)
                    score_groups[tuple(seed_row[field] for field in score_group_fields)].append(seed_row)
    summary = []
    if pairs:
        repair_fields = ['other_train_rows', 'repair_eligible'] if repair_pairs else []
        write_csv(
            root / 'transfer.csv',
            transfer,
            fields=[
                *identities,
                'client',
                'class',
                'mode',
                'local_train_rows',
                'local_cal_fit_rows',
                'other_cal_fit_rows',
                'transfer_scope',
                'test_support',
                'predicted',
                'true_positive',
                'recall',
                'precision',
                'f1',
                *repair_fields,
            ],
        )
        write_csv(root / 'transfer_by_seed.csv', pairs)
        summary = [
            {
                **dict(zip((*configuration, 'transfer_scope', 'contrast'), group, strict=True)),
                'seeds_total': len(values),
                **summarize(values, bounds=(-1, 1)),
            }
            for group, values in sorted(groups.items(), key=lambda item: repr(item[0]))
        ]
        write_csv(root / 'transfer_intervals.csv', summary)
    if repair_pairs:
        write_csv(root / 'transfer_repair_by_seed.csv', repair_pairs)
    if mode_seeds:
        write_csv(root / 'transfer_metrics_by_seed.csv', mode_seeds)
        paper_rows = []
        for group, seeds in sorted(mode_groups.items(), key=lambda item: repr(item[0])):
            paper = {
                **dict(zip((*configuration, 'transfer_scope', 'mode'), group, strict=True)),
                'seeds_with_cells': len(seeds),
                'receiver_class_cells_across_seeds': sum(row['eligible_cells'] for row in seeds),
                'precision_zero_division': 0,
            }
            for metric in ('macro_recall', 'macro_precision', 'macro_f1'):
                statistics = summarize([row[metric] for row in seeds], bounds=(0, 1))
                mode_intervals.append(
                    {
                        **dict(zip((*configuration, 'transfer_scope', 'mode'), group, strict=True)),
                        'metric': metric,
                        'seeds_total': len(seeds),
                        'precision_zero_division': 0,
                        **statistics,
                    }
                )
                paper.update(
                    {f'{metric}_{field}': statistics[field] for field in ('mean', 'seed_sd', 'ci95_low', 'ci95_high')}
                )
            overall_rows = overall[(*group[:-2], group[-1])]
            for metric in ('far', 'fnr', 'ece', 'alert_rate'):
                values = [row[metric] for row in overall_rows if row.get(metric) is not None]
                paper[f'overall_{metric}_mean'] = mean(values) if values else None
            paper[
                'overall_population'
            ] = 'all test_unique_unseen rows; transfer metrics use declared missing-class cells'
            paper_rows.append(paper)
        write_csv(root / 'transfer_metrics_intervals.csv', mode_intervals)
        write_csv(root / 'transfer_paper_table.csv', paper_rows)
    if score_seeds:
        write_csv(
            root / 'transfer_score_deltas.csv',
            score_detail,
            fields=[
                *identities,
                'client',
                'class',
                'transfer_scope',
                'test_support',
                'local_train_rows',
                'local_cal_fit_rows',
                'other_cal_fit_rows',
                'contrast',
                *score_metrics,
            ],
        )
        write_csv(root / 'transfer_score_by_seed.csv', score_seeds)
        score_summary = [
            {
                **dict(zip(score_group_fields, group, strict=True)),
                'metric': metric,
                'seeds_total': len(seeds),
                **summarize([row[metric] for row in seeds], bounds=bounds),
            }
            for group, seeds in sorted(score_groups.items(), key=lambda item: repr(item[0]))
            for metric, bounds in score_metrics.items()
        ]
        write_csv(root / 'transfer_score_intervals.csv', score_summary)
    for name, rows in other.items():
        if rows:
            write_csv(root / f'{name}.csv', rows)
    return summary


def _plot(root, results, differences, objective_changes, transfer_intervals):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    target = root / 'figures'
    target.mkdir(exist_ok=True)

    def save(fig, name):
        fig.savefig(target / f'{name}.pdf', bbox_inches='tight')
        fig.savefig(target / f'{name}.png', dpi=600, bbox_inches='tight')
        plt.close(fig)

    matrices = defaultdict(list)
    for result in results:
        if (result['scenario'], result['protocol']) != ('main', 'grouped') or result['dataset'] not in NETWORK_DATASETS:
            continue
        for entry in result['evaluations']:
            if entry['mode'] == 'full' and entry['role'] == 'test_unique_unseen':
                matrices[result['dataset']].append(entry)
    for dataset, entries in matrices.items():
        names = [row['class'] for row in entries[0]['per_class']]
        # Average row percentages only across seeds where that row has support.
        all_counts = np.asarray([entry['matrix'] for entry in entries], dtype=float)
        supports = all_counts.sum(axis=2, keepdims=True)
        matrix = 100 * divide(divide(all_counts, supports).sum(axis=0), (supports > 0).sum(axis=0))
        absent = supports.sum(axis=0).ravel() == 0
        plot = np.ma.array(matrix, mask=np.broadcast_to(absent[:, None], matrix.shape))
        fig, ax = plt.subplots(figsize=(8, 7), layout='constrained')
        heat = ax.imshow(plot, vmin=0, vmax=100, cmap='Blues')
        ax.set_xticks(range(len(names)), names, rotation=45, ha='right')
        ax.set_yticks(range(len(names)), names)
        ax.set(xlabel='Predicted class', ylabel='True class')
        for i in range(len(names)):
            for j in range(len(names)):
                if not absent[i] and matrix[i, j] >= 0.5:
                    ax.text(
                        j,
                        i,
                        f'{matrix[i, j]:.0f}',
                        ha='center',
                        va='center',
                        color='white' if matrix[i, j] > 58 else 'black',
                        fontsize=8,
                    )
        fig.colorbar(heat, ax=ax, fraction=0.048, label='Row percent')
        save(fig, f'confusion_grouped_unique_{dataset}')

    def effects(entries, xlabel, name, *, by_objective=False):
        if not entries:
            return
        datasets = sorted({row['dataset'] for row in entries})
        positions = {d: i for i, d in enumerate(datasets)}
        fig, ax = plt.subplots(figsize=(7.4, 3.5), layout='constrained')
        for scenario, offset, color, marker, label in (
            (
                ('main', -0.13, '#12557d', 'o', 'Reference objective'),
                ('balance', 0.13, '#934922', 's', 'Class balanced objective'),
            )
            if by_objective
            else ((None, 0, '#12557d', 'o', None),)
        ):
            rows = [row for row in entries if scenario is None or row['scenario'] == scenario]
            if not rows:
                continue
            middle = [100 * row['mean'] for row in rows]
            limits = [
                [
                    value - 100 * row['ci95_low'] if row['ci95_low'] is not None else 0
                    for value, row in zip(middle, rows)
                ],
                [
                    100 * row['ci95_high'] - value if row['ci95_high'] is not None else 0
                    for value, row in zip(middle, rows)
                ],
            ]
            ax.errorbar(
                middle,
                [positions[row['dataset']] + offset for row in rows],
                xerr=limits,
                fmt=marker,
                color=color,
                capsize=3,
                label=label,
            )
        ax.axvline(0, color='black', linewidth=0.9)
        ax.set_yticks(range(len(datasets)), datasets)
        ax.set_ylim(len(datasets) - 0.5, -0.5)
        ax.set_xlabel(xlabel)
        if by_objective:
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
        save(fig, name)

    aggregation = [
        row
        for row in differences
        if row['scenario'] in ('main', 'balance')
        and row['role'] == 'test_unique_unseen'
        and row['protocol'] == 'grouped'
        and row['metric'] == 'macro_f1'
        and row['contrast'] == 'full_minus_local_only'
        and row['mean'] is not None
    ]
    effects(aggregation, 'Full minus local macro F1 (percentage points)', 'aggregation_effect', by_objective=True)
    effects(
        [row for row in objective_changes if row['primary'] and row['mean'] is not None],
        'Class balanced minus reference macro F1 (percentage points)',
        'training_objective_effect',
    )
    effects(
        [
            row
            for row in transfer_intervals
            if row['transfer_scope'] == 'train_and_fit_absent'
            and row['contrast'] == 'full_global_minus_local_rounds'
            and row['mean'] is not None
        ],
        'Federated minus local recall (percentage points)',
        'supplement_transfer_effect',
    )
    effects(
        [
            row
            for row in differences
            if row['protocol'] == 'temporal'
            and row['role'] == 'test_unique_unseen'
            and row['metric'] == 'macro_f1'
            and row['contrast'] == 'subtype_minus_base_threshold'
            and row['mean'] is not None
        ],
        'Subtype head minus reference macro F1 (percentage points)',
        'supplement_subtype_effect',
    )
