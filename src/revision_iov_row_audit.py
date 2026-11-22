"""Reconcile every original IoV record with the grouped development run."""

from collections import Counter
from pathlib import Path

from .dataset_preparation import recipe_path
from .revision_v12 import _hash, _read


def _seed_rows(case, seed, role_rows, val_support):
    """Verify row allocation, training visits, calibration and VAL coverage."""
    result = _read(case / 'result.json')
    settings = result['settings']
    if settings['rows_per_client'] != 0 or settings['train_rows'] != role_rows['train']:
        raise ValueError(f'IoV seed {seed}: TRAIN row limit or count differs from full TRAIN')

    clients = {f'client_{index}' for index in range(result['clients'])}
    assigned = Counter()
    for row in result['supports']:
        if row['client'] in clients:
            assigned[row['role'], row['client']] += row['rows']
    for role in ('train', 'cal', 'val'):
        if sum(assigned[role, client] for client in clients) != role_rows[role]:
            raise ValueError(f'IoV seed {seed}: {role.upper()} client allocations omit records')

    measurements = [row for row in result['measurement'] if row['stage'] == 'local_detector']
    if len(measurements) != len(clients) or {row['client'] for row in measurements} != clients:
        raise ValueError(f'IoV seed {seed}: local detector measurements do not cover every client')
    for row in measurements:
        client = row['client']
        if (
            row['training_row_visits'] != settings['base_epochs'] * assigned['train', client]
            or row['inference_rows'] != assigned['cal', client] + assigned['val', client]
            or row['test_inference_rows'] != 0
        ):
            raise ValueError(f'IoV seed {seed}: detector row accounting differs for {client}')

    cal = _read(case / 'cal_group_split.json')
    if (
        cal['source_cal_rows'] != role_rows['cal']
        or cal['owned_cal_rows'] != role_rows['cal']
        or cal['unowned_cal_rows'] != 0
        or cal['global_fit_rows'] + cal['global_holdout_rows'] != role_rows['cal']
        or cal['global_exact_feature_overlap'] != 0
    ):
        raise ValueError(f'IoV seed {seed}: CAL allocations omit or share records')
    for evaluation in result['evaluations']:
        if evaluation['role'] != 'val' or [sum(row) for row in evaluation['matrix']] != val_support:
            raise ValueError(f'IoV seed {seed}: {evaluation["mode"]} differs from full VAL class support')
    return {
        'seed': seed,
        'train_rows': role_rows['train'],
        'train_row_visits': sum(row['training_row_visits'] for row in measurements),
        'cal_fit_rows': cal['global_fit_rows'],
        'cal_holdout_rows': cal['global_holdout_rows'],
        'val_rows_per_branch': role_rows['val'],
    }


def verify_iov_full_rows(output, dataset, seeds):
    """Return a receipt for the checked real-data run without changing its records."""
    grouped = Path(output) / dataset / 'grouped'
    audit = _read(grouped / 'data_audit.json')
    recipe_file = recipe_path(dataset)
    recipe = _read(recipe_file)
    source_rows = {record['filename']: record['csv_data_rows'] for record in recipe['source_files']}
    role_rows = audit['role_rows']
    support = audit['role_class_support']
    classes = recipe['class_names']
    repeats = audit['exact_feature_audit']['within_role_repeats']
    group_counts = audit['grouped_resplit']['assigned_groups']
    if (
        audit['source_recipe_sha256'] != _hash(recipe_file)
        or sum(source_rows.values()) != sum(role_rows.values())
        or [sum(counts[label] for counts in support.values()) for label in classes] != recipe['source_class_counts']
        or any(sum(support[role].values()) != count for role, count in role_rows.items())
        or any(pair['unique_shared_inputs'] for pair in audit['exact_feature_audit']['cross_role_exact_overlap'].values())
    ):
        raise ValueError('IoV grouped roles do not account for the original data and disjoint feature groups')

    val_support = [support['val'][label] for label in classes]
    return {
        'dataset': dataset,
        'source_rows_by_file': source_rows,
        'n_source_rows': sum(source_rows.values()),
        'role_rows': role_rows,
        'rows_beyond_one_per_exact_group': {role: count - group_counts[role] for role, count in role_rows.items()},
        'val_distinct_inputs': repeats['val']['distinct_inputs'],
        'test_evaluated': False,
        'per_seed': [_seed_rows(grouped / 'main' / f'seed_{seed}', seed, role_rows, val_support) for seed in seeds],
    }
