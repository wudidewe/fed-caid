"""Compare predeclared NB15 alarm rules on freshly fitted reference detectors."""

from collections import defaultdict
from math import sqrt
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory

import numpy as np
import torch
from scipy.stats import t

from . import revision_next as native
from . import utils as core
from .revision_nb15_low_far import (
    DATASET,
    FAR_CAP,
    FNR_INCREASE_LIMIT,
    MIN_CAL_BENIGN_ROWS,
    SCOPES,
    SEEDS,
    SYNTHETIC_TARGET,
    TARGET,
    _allocate_output,
    _binary_metrics,
    _class_rows,
    _classification_metrics,
    _client_rows,
    _configuration,
    _decide,
    _difference,
    _fit_reference_client,
    _paired_stats,
    _pool,
    _prepare_seed,
    _scope_check,
    _source_snapshot,
    _validate_role,
)
from .revision_nb15_threshold_report import write_threshold_report
from .revision_operating_selection import split_operating_subsets
from .revision_report import summarize, write_csv
from .revision_v12 import _bindings, _hash, _write
from .utils import empirical_alarm_threshold

OUTPUT_CONTAINER = 'fedcaid_nb15_threshold'
SOURCE_MODULES = (
    'revision_nb15_threshold.py',
    'revision_nb15_threshold_report.py',
    'revision_operating_selection.py',
    'utils.py',
)
CAL_CHECK_FRACTION = 0.5

REFERENCE = 'reference_current'
MATCHED = 'matched_cal_1pct'
CONSERVATIVE = 'conservative_cal_0_9pct'
ARMS = (REFERENCE, MATCHED, CONSERVATIVE)
NOMINAL_FARS = {REFERENCE: FAR_CAP, MATCHED: FAR_CAP, CONSERVATIVE: 0.009}
PAIRS = ((MATCHED, REFERENCE), (CONSERVATIVE, REFERENCE), (CONSERVATIVE, MATCHED))
MIN_ASSESSMENT_SEEDS = 5
TARGET_CLASSES = ('Exploits', 'Analysis')
CLASS_METRICS = ('binary_recall', 'recall', 'f1')
CAL_ROLES = ('cal_fit', 'cal_check')
COUNT_KEYS = (
    'rows',
    'benign_rows',
    'attack_rows',
    'true_positives',
    'true_negatives',
    'false_positives',
    'false_negatives',
)


def _split_indices(subsets, size):
    """Validate the native CAL partition before using its aligned row indices."""
    fit, check = (np.asarray(subsets[key]) for key in ('threshold_fit', 'operating_select'))
    for rows in (fit, check):
        if rows.ndim != 1 or not rows.size or rows.dtype.kind not in 'ui':
            raise ValueError('NB15 threshold-fit and CAL-check need integer row indices')
    joined = np.concatenate((fit, check))
    if len(joined) != size or not np.array_equal(np.sort(joined), np.arange(size)):
        raise ValueError('NB15 threshold-fit and CAL-check must partition the parent holdout without overlap')
    return fit, check


def _numbers(role, metrics):
    """Prefix output columns to distinguish threshold fitting from checking."""
    return {f'{role}_{key}': value for key, value in metrics.items()}


def _freeze_client(labels, probabilities, fit, check, arm):
    """Choose the threshold from fit rows, then measure it unchanged on CAL-check."""
    independent = arm != REFERENCE
    fit_p, fit_y = (probabilities[fit], labels[fit]) if independent else (probabilities, labels)
    threshold, source = empirical_alarm_threshold(
        fit_p,
        fit_y,
        NOMINAL_FARS[arm],
        floor=0,
        min_benign_rows=MIN_CAL_BENIGN_ROWS,
        insufficient_policy='disable',
    )
    fit_metrics = _binary_metrics(_decide(fit_p, fit_y, threshold))
    check_metrics = _binary_metrics(_decide(probabilities[check], labels[check], threshold))
    return {
        **source,
        'nominal_cal_far': NOMINAL_FARS[arm],
        'cal_fit_view': 'disjoint_fit' if independent else 'original_full_holdout',
        'cal_check_independent': independent,
        'cal_check_far_pass': _check_far(check_metrics) if independent else None,
        **_numbers('cal_fit', fit_metrics),
        **_numbers('cal_check', check_metrics),
    }


def _check_far(metrics):
    """Require support and a CAL-check FAR at or below the fixed cap."""
    return metrics['benign_rows'] >= MIN_CAL_BENIGN_ROWS and metrics['far'] is not None and metrics['far'] <= FAR_CAP


def freeze_thresholds(labels_cal, probs_cal, cal_subsets, classes):
    """Return JSON-serializable threshold locks using only CAL inputs."""
    if not 0 < len(labels_cal) == len(probs_cal) == len(cal_subsets):
        raise ValueError('NB15 threshold locks need matched CAL input for every client')
    result = {arm: [] for arm in ARMS}
    for labels, probabilities, subsets in zip(labels_cal, probs_cal, cal_subsets, strict=True):
        labels, probabilities = _validate_role(labels, probabilities, classes)
        fit, check = _split_indices(subsets, len(labels))
        for arm in ARMS:
            result[arm].append(_freeze_client(labels, probabilities, fit, check, arm))
    return result


def _pooled_cal(locks, role):
    """Calculate pooled CAL rates directly from matched client counts."""
    counts = {key: sum(lock[f'{role}_{key}'] for lock in locks) for key in COUNT_KEYS}
    benign, attack = counts['benign_rows'], counts['attack_rows']
    return _numbers(
        role,
        {
            **counts,
            'far': counts['false_positives'] / benign if benign else None,
            'fnr': counts['false_negatives'] / attack if attack else None,
        },
    )


def _scope_rows(identity, selected, names):
    """Pool the gate outcomes while keeping each client's own threshold."""
    locks = [lock for lock, _ in selected]
    val = _pool([decision for _, decision in selected])
    classes = list(_class_rows(identity, val, names))
    point = {
        **identity,
        **_classification_metrics(classes),
        **_numbers('val', _binary_metrics(val, ranking=True)),
        **{key: value for role in CAL_ROLES for key, value in _pooled_cal(locks, role).items()},
        'nominal_cal_far': locks[0]['nominal_cal_far'],
        'cal_check_independent': locks[0]['cal_check_independent'],
        'cal_check_clients_passing': (
            sum(lock['cal_check_far_pass'] is True for lock in locks) if locks[0]['cal_check_independent'] else None
        ),
        'cal_check_clients': len(locks),
        'cal_clients_with_insufficient_support': sum(lock['alert_gate_disabled_for_support'] for lock in locks),
        'val_ranking_scope': 'receiver_raw_scores' if identity['scope'] == 'receiver' else 'pooled_raw_scores',
    }
    return point, classes


def _evaluate_arm(seed, names, receiver, validated, arm, locks):
    """Apply existing thresholds to VAL and preserve matched population labels."""
    identity = {'seed': seed, 'arm': arm, 'receiver_client': receiver}
    clients = [
        (lock, _decide(p, labels, lock['threshold'])) for (labels, p), lock in zip(validated, locks, strict=True)
    ]
    client_rows = [
        {**identity, 'client': index, **lock, **_numbers('val', _binary_metrics(val, ranking=True))}
        for index, (lock, val) in enumerate(clients)
    ]
    scoped = [
        _scope_rows({**identity, 'scope': scope}, selected, names)
        for scope, selected in (('receiver', clients[receiver : receiver + 1]), ('overall', clients))
    ]
    return [point for point, _ in scoped], client_rows, [row for _, rows in scoped for row in rows]


def _class_differences(rows):
    """Compare every class against its reference on the same VAL rows."""
    reference = {(row['scope'], row['class_index']): row for row in rows if row['arm'] == REFERENCE}
    return [
        {
            **row,
            **{
                key
                + '_delta_vs_reference': _difference(
                    row[key],
                    reference[(row['scope'], row['class_index'])][key],
                )
                for key in CLASS_METRICS
            },
        }
        for row in rows
    ]


def evaluate_locked_seed(seed, names, receiver, labels_val, probabilities_val, locked):
    """Evaluate one seed from already locked thresholds without revisiting CAL."""
    if (
        len(names) < 2
        or names[0] != 'BENIGN'
        or len(set(names)) != len(names)
        or not 0 <= receiver < len(labels_val)
        or len(labels_val) != len(probabilities_val)
        or set(locked) != set(ARMS)
        or any(len(locks) != len(labels_val) for locks in locked.values())
    ):
        raise ValueError('Expected all frozen NB15 thresholds and aligned VAL populations')
    validated = [_validate_role(y, p, len(names)) for y, p in zip(labels_val, probabilities_val, strict=True)]
    results = [_evaluate_arm(seed, names, receiver, validated, arm, locked[arm]) for arm in ARMS]
    points, clients, classes = ([row for group in table for row in group] for table in zip(*results, strict=True))
    return {'working_points': points, 'client_working_points': clients, 'per_class': _class_differences(classes)}


def evaluate_seed(seed, names, receiver, labels_by_client, probabilities_by_client, cal_subsets):
    """Freeze using CAL then evaluate on the corresponding held-back VAL arrays."""
    cal_y, val_y = zip(*labels_by_client, strict=True)
    cal_p, val_p = zip(*probabilities_by_client, strict=True)
    locked = freeze_thresholds(cal_y, cal_p, cal_subsets, len(names))
    return evaluate_locked_seed(seed, names, receiver, val_y, val_p, locked)


def _index_points(rows, seeds):
    """Require the locked evaluation population for every seed, scope and arm."""
    expected = {(seed, scope, arm) for seed in seeds for scope in SCOPES for arm in ARMS}
    index = {(row['seed'], row['scope'], row['arm']): row for row in rows}
    if len(index) != len(rows) or set(index) != expected:
        raise ValueError('NB15 threshold results lack a complete matched seed/scope inventory')
    return index


def _checks(rows, seeds, candidate):
    """Summarize the per-client independent CAL check without changing predictions."""
    selected = [row for row in rows if row['arm'] == candidate]
    keys = {(row['seed'], row['client']) for row in selected}
    reference = {(row['seed'], row['client']) for row in rows if row['arm'] == REFERENCE}
    if len(keys) != len(selected) or keys != reference or {seed for seed, _ in keys} != set(seeds):
        raise ValueError('NB15 threshold check data lack a complete matched seed/client inventory')
    passed = sum(row['cal_check_far_pass'] is True for row in selected)
    failing = [
        {'seed': row['seed'], 'client': row['client'], 'far': row['cal_check_far']}
        for row in selected
        if row['cal_check_far_pass'] is not True
    ]
    return {
        'pass': not failing and all(row['cal_check_independent'] for row in selected),
        'clients_passing': passed,
        'clients_expected': len(selected),
        'failing_clients': failing,
    }


def _candidate_checks(tables, index, seeds, candidate, intervals):
    """Apply the existing VAL guard plus one independent CAL check per client."""
    scopes = {scope: _scope_check(index, seeds, scope, candidate, intervals[scope]) for scope in SCOPES}
    cal = _checks(tables['client_working_points'], seeds, candidate)
    flags = (
        'far_pass',
        'cal_support_pass',
        'nonzero_attack_detection_each_seed',
        'paired_fnr_pass',
        'macro_f1_mean_non_decrease',
    )
    passed = cal['pass'] and all(record[flag] for record in scopes.values() for flag in flags)
    return {
        'scopes': scopes,
        'cal_check': cal,
        'decision_scope': 'prespecified_candidate' if candidate == CONSERVATIVE else 'matched_sampling_control',
        'development_guard_pass': passed if len(seeds) >= MIN_ASSESSMENT_SEEDS else None,
    }


def _target_deltas(index, seeds, scope, name, candidate, reference):
    """Pair target and minority recall and F1 on identical seed support."""

    def observed(seed, metric):
        left = index.get((seed, scope, name, candidate), {})
        right = index.get((seed, scope, name, reference), {})
        return _difference(left.get(metric), right.get(metric))

    return {metric: summarize([observed(seed, metric) for seed in seeds], bounds=(-1, 1)) for metric in CLASS_METRICS}


def _target_intervals(rows, seeds):
    """Report descriptive minority effects separately from the binary guard."""
    index = {(row['seed'], row['scope'], row['class_name'], row['arm']): row for row in rows}
    if len(index) != len(rows):
        raise ValueError('NB15 threshold per-class evidence contains duplicate rows')
    return {
        f'{candidate}_minus_{reference}': {
            scope: {name: _target_deltas(index, seeds, scope, name, candidate, reference) for name in TARGET_CLASSES}
            for scope in SCOPES
        }
        for candidate, reference in PAIRS
    }


def assess_results(tables, seeds):
    """Summarize the fixed comparisons and record the development-only decision."""
    seeds = tuple(seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('NB15 threshold assessment requires distinct prescribed seeds')
    index = _index_points(tables['working_points'], seeds)
    intervals = {
        f'{candidate}_minus_{reference}': {
            scope: _paired_stats(index, seeds, scope, candidate, reference) for scope in SCOPES
        }
        for candidate, reference in PAIRS
    }
    return _assessment(tables, index, seeds, intervals)


def _assessment(tables, index, seeds, intervals):
    """Record development roles, guard settings and all paired results."""
    candidates = {
        arm: _candidate_checks(tables, index, seeds, arm, intervals[f'{arm}_minus_{REFERENCE}'])
        for arm in (MATCHED, CONSERVATIVE)
    }
    return {
        'scope': 'grouped same-source development',
        'seeds': list(seeds),
        'calibration_rule': 'per-client benign CAL higher quantile with zero floor and strict score > threshold',
        'selection_on_val': False,
        'test_evaluated': False,
        'automatic_promotion': False,
        'far_cap': FAR_CAP,
        'paired_fnr_ci95_upper_limit': FNR_INCREASE_LIMIT,
        'min_cal_benign_rows': MIN_CAL_BENIGN_ROWS,
        'nominal_cal_fars': NOMINAL_FARS,
        'primary_candidate': CONSERVATIVE,
        'development_guard_pass': candidates[CONSERVATIVE]['development_guard_pass'],
        'paired_intervals': intervals,
        'candidates': candidates,
        'target_class_intervals': _target_intervals(tables['per_class'], seeds),
    }


def _sources(project):
    """Bind the ordinary reference training and the new threshold policy code."""
    return {**_source_snapshot(project), **_bindings(Path(__file__).resolve().parent, SOURCE_MODULES)}


def _fit_clients(data, parts, routes, args, temporary):
    """Use the local training population and CAL without giving the fit any VAL rows."""
    nodes = []
    for client, rows in enumerate(routes):
        training_and_cal = {role: rows[role] for role in ('train', 'cal')}
        node = _fit_reference_client(data, parts, training_and_cal, args, client, temporary)
        del node.parts['train']
        nodes.append(node)
    return nodes


def _score_val(node, features, indices, args):
    """Generate VAL scores only after the CAL threshold receipt exists."""
    model = node.model.to(args.device)
    try:
        return core.probabilities(
            core.log_prediction(model, node.transform(features[indices]), args.device, args.batch_size)
        )
    finally:
        model.cpu()


def _lock_cal(output, seed, nodes, subsets, split_audit, classes):
    """Persist thresholds from the disjoint CAL assignment before accessing VAL."""
    path = output / f'threshold_lock_{seed}.json'
    truth = [node.parts['cal'][1] for node in nodes]
    probabilities = [node.probabilities['cal'] for node in nodes]
    lock = freeze_thresholds(truth, probabilities, subsets, classes)
    _write(path, {'seed': seed, 'locked_before_val_scoring': True, 'thresholds': lock, 'split': split_audit})
    return lock, _hash(path)


def _run_seed(output, data, cli, seed):
    """Apply all policies to identical scores after writing the CAL-only threshold lock."""
    parts, roster, policy, audit, args = _prepare_seed(data, cli, seed)
    routes = [_client_rows(roster, policy, client) for client in range(len(roster))]
    subsets, split_audit = split_operating_subsets(
        *parts['cal'],
        policy['roster'],
        policy['splits'],
        CAL_CHECK_FRACTION,
        seed,
        min_select_benign=0 if cli.smoke else 100,
    )
    with TemporaryDirectory(prefix='fedcaid_nb15_threshold_') as temporary:
        nodes = _fit_clients(data, parts, routes, args, Path(temporary))
        lock, lock_hash = _lock_cal(output, seed, nodes, subsets, split_audit, len(data.names))
        print(f'FedCAID | NB15 seed={seed} | CAL 阈值已锁定，开始 VAL', flush=True)
        val_x, val_y = parts['val']
        scores = [_score_val(node, val_x, rows['val'], args) for node, rows in zip(nodes, routes, strict=True)]
        truth = [val_y[rows['val']] for rows in routes]
        tables = evaluate_locked_seed(seed, data.names, audit['receiver'], truth, scores, lock)
    _write(output / f'seed_{seed}.json', {'seed': seed, 'design': audit, 'threshold_lock_sha256': lock_hash})
    return tables


def _collect(output, project, data, cli, seeds, source):
    """Collect all prespecified outcomes without evaluating any archived scores."""
    tables = defaultdict(list)
    for seed in seeds:
        if _sources(project) != source:
            raise ValueError('NB15 threshold experiment code changed during evaluation')
        for name, rows in _run_seed(output, data, cli, seed).items():
            tables[name].extend(rows)
    if _sources(project) != source:
        raise ValueError('NB15 threshold experiment code changed during evaluation')
    return tables


def _receiver_feasibility(rows, seeds):
    """Bound the paired FNR upper endpoint for any FAR-compliant receiver threshold."""
    reference = {r['seed']: r for r in rows if r['arm'] == 'reference_current' and r['scope'] == 'receiver'}
    if len(seeds) != len(SEEDS) or any(reference[seed]['val_tpr_at_far_1pct_diagnostic'] is None for seed in seeds):
        return None
    minimum = {
        seed: 1 - reference[seed]['val_tpr_at_far_1pct_diagnostic'] - reference[seed]['val_fnr'] for seed in seeds
    }
    mean_min, max_min = mean(minimum.values()), max(minimum.values())
    # With a fixed mean m and one difference >= max_min, SE >= (max_min - m)/(n-1) when m <= max_min.
    n = len(seeds)
    critical = float(t.ppf(0.975, n - 1))
    bound = mean_min + critical / (n - 1) * (max_min - mean_min)
    # By Cauchy-Schwarz the t upper endpoint also bounds any subset mean
    # when the t multiplier on centered L2 dominates the subset weights.
    for count in range(2, n):
        if critical / sqrt(n * (n - 1)) >= sqrt(1 / count - 1 / n):
            bound = max(bound, mean(sorted(minimum.values())[-count:]))
    return {
        'diagnostic_only': True,
        'selection_on_val': False,
        'fnr_budget': FNR_INCREASE_LIMIT,
        'per_seed_lower_bounds': [{'seed': seed, 'min_val_fnr_increment': minimum[seed]} for seed in seeds],
        'mean_increment_lower_bound': mean_min,
        'paired_upper_95_lower_bound': bound,
        'joint_fnr_and_far_impossible_on_this_val': bound > FNR_INCREASE_LIMIT,
    }


def _write_plan(output, cli, seeds, source):
    """Declare score isolation and the complete fresh-run workload."""
    _write(
        output / 'plan.json',
        {
            'dataset': DATASET,
            'target': SYNTHETIC_TARGET if cli.smoke else TARGET,
            'scenario': 'train_absence',
            'protocol': 'grouped',
            'clients': 3,
            'alpha': 0.3,
            'data_root': str(cli.data_root),
            'device': cli.device,
            'seeds': list(seeds),
            'synthetic_only': cli.smoke,
            'previous_run_inputs_allowed': False,
            'training': 'fresh original prior-adjusted TabM on TRAIN; no candidate weight updates',
            'reference_epochs': cli.detector_epochs,
            'cal_check_fraction': CAL_CHECK_FRACTION,
            'arm_targets': {'reference_current': 0.01, 'matched_cal_1pct': 0.01, 'conservative_cal_0_9pct': 0.009},
            'locks_written_before_val_scoring': True,
            'selection_on_val': False,
            'test_evaluated': False,
            'source_sha256': source,
            'temporal_followup': 'eligible only after the primary candidate passes grouped VAL',
        },
    )


def _publish(output, tables, seeds):
    """Seal tables, locked thresholds, feasibility diagnosis and the report."""
    for name, rows in tables.items():
        write_csv(output / f'{name}.csv', rows)
    assessment = assess_results(tables, seeds)
    assessment['receiver_threshold_only_feasibility'] = _receiver_feasibility(tables['working_points'], seeds)
    assessment['temporal_followup'] = (
        'eligible_for_separate_same_source_evaluation'
        if assessment['development_guard_pass'] is True
        else 'not_triggered_by_grouped_development'
    )
    _write(output / 'assessment.json', assessment)
    write_threshold_report(output, tables, assessment)
    names = sorted(path.name for path in output.iterdir() if path.is_file() and path.name != 'complete.json')
    _write(output / 'complete.json', {'status': 'completed', 'artifacts': _bindings(output, names)})
    print(f"FedCAID | NB15 阈值专项完成 | {output / 'SUMMARY.md'}", flush=True)


def run_nb15_threshold(project, *, output=None, device='auto', smoke=False):
    """Execute only the prespecified threshold comparison in a new output directory."""
    project = Path(project).resolve(strict=True)
    output = _allocate_output(project, OUTPUT_CONTAINER, output=output)
    cli = _configuration(project, output, device, smoke)
    seeds, source = SEEDS[:1] if smoke else SEEDS, _sources(project)
    _write_plan(output, cli, seeds, source)
    print(f'FedCAID | NB15 阈值专项 | {output}', flush=True)
    if smoke:
        torch.set_num_threads(1)
    data = (
        native._synthetic(DATASET, 'grouped')
        if smoke
        else native.load_revision_data(
            DATASET,
            cli.data_root,
            split_protocol='grouped',
        )
    )
    _write(output / 'data_audit.json', data.audit)
    _publish(output, _collect(output, project, data, cli, seeds, source), seeds)
    return output
