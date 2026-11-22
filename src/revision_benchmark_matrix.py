"""Run the original K−1 adapter benchmark on four grouped development datasets."""

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .experiment_config import DATASETS
from .revision_report import write_csv
from .revision_run import SEEDS, grouped_validation_arguments
from .revision_run import main as run_revision_matrix
from .revision_v12 import _bindings, _output_lock, _read

MODES = ('base', 'base_threshold', 'local_only', 'full', 'fedavg_threshold')
METRICS = ('accuracy', 'weighted_f1', 'macro_f1', 'balanced_accuracy', 'far', 'fnr', 'ece')
CONTRASTS = ('local_only_minus_base_threshold', 'full_minus_local_only', 'full_minus_fedavg_threshold')
CAL_SPLIT = 'exact_feature_group'
BASELINE_SOURCES = (
    'metrics_mean_sd.csv',
    'metrics_by_seed.csv',
    'per_class.csv',
    'parameters.csv',
    'communication.csv',
)
COMMUNICATION_MODES = {'full': 'full', 'fedavg_threshold': 'fedavg'}


def _rows(root, name):
    """Read one native benchmark table."""
    with (root / name).open(encoding='utf-8-sig', newline='') as stream:
        yield from csv.DictReader(stream)


def _validate_plan(root, seeds, smoke):
    """Bind the published matrix to its intended split, scenario and seeds."""
    config = _read(root / 'run.json')
    expected_matrix = [{'dataset': name, 'protocol': 'grouped', 'scenarios': [['main', 3, 0.8]]} for name in DATASETS]
    expected_revision = ('development_only', CAL_SPLIT, 'disable')
    revision = config['revision_v6']
    if (
        config['matrix'] != expected_matrix
        or config['seeds'] != list(seeds)
        or config['synthetic_only'] is not smoke
        or tuple(
            revision[field]
            for field in (
                'evaluation_stage',
                'cal_split_protocol',
                'threshold_insufficient_policy',
            )
        )
        != expected_revision
        or not config['no_tcp']
    ):
        raise ValueError('The benchmark matrix settings differ from the declared comparison')


def _require_grouped_main(row, *, role='val'):
    """Reject evidence outside the benchmark protocol and evaluation role."""
    if (row.get('protocol'), row.get('scenario'), row.get('role')) != ('grouped', 'main', role):
        raise ValueError('The benchmark matrix must contain grouped development VAL only')


def _validate_val_rows(root, seeds):
    """Require every planned matched control and reject scores from other roles."""
    expected = {(name, str(seed), mode) for name in DATASETS for seed in seeds for mode in MODES}
    observed = set()
    for row in _rows(root, 'metrics_by_seed.csv'):
        _require_grouped_main(row)
        key = (row['dataset'], row['seed'], row['mode'])
        if row['mode'] in MODES:
            if key in observed:
                raise ValueError(f'Duplicate benchmark case: {key}')
            observed.add(key)
    if observed != expected:
        raise ValueError('Incomplete matched benchmark matrix')


def _metric_summaries(root, seeds):
    """Read absolute VAL summaries for the five main controls."""
    cases = {}
    observed = set()
    for row in _rows(root, 'metrics_mean_sd.csv'):
        _require_grouped_main(row)
        dataset, mode, metric = (row[key] for key in ('dataset', 'mode', 'metric'))
        key = (dataset, mode, metric)
        if key in observed:
            raise ValueError(f'Duplicate benchmark summary: {key}')
        observed.add(key)
        if mode in MODES and metric in METRICS:
            if int(row['n']) != len(seeds):
                raise ValueError(f'Incomplete seed coverage for benchmark summary: {key}')
            cases.setdefault((dataset, mode), {}).update({f'{metric}_{key}': row[key] for key in ('n', 'mean', 'sd')})
    expected = {(name, mode, metric) for name in DATASETS for mode in MODES for metric in METRICS}
    if not expected <= observed:
        raise ValueError('Incomplete matched benchmark summaries')
    return cases


def _by_case(root, filename, *, role='val'):
    """Group native rows by declared dataset and run mode."""
    rows = defaultdict(list)
    for row in _rows(root, filename):
        _require_grouped_main(row, role=role)
        rows[row['dataset'], row['mode']].append(row)
    return rows


def _uniform(rows, field, cast=str):
    """Keep a stable grouping value without conflating distinct settings."""
    values = {cast(row[field]) for row in rows}
    if len(values) != 1:
        raise ValueError(f'Expected one benchmark value for {field}')
    return values.pop()


def _coverage_values(rows, metrics):
    """Distinguish an absent true class from a supported class with zero recall."""
    support = defaultdict(list)
    zero_recall = []
    for row in rows:
        examples = int(row['support'])
        support[row['class']].append(examples)
        if examples and row['recall'] and float(row['recall']) == 0:
            zero_recall.append(row['class'])
    observed, defined = (_uniform(metrics, key, int) for key in ('observed_classes', 'defined_classes'))
    absent = sorted(label for label, values in support.items() if not any(values))
    if len(support) != defined or defined - len(absent) != observed:
        raise ValueError('Benchmark class metrics do not match observed and defined class counts')
    return {
        'observed_classes': observed,
        'defined_classes': defined,
        'zero_support_classes': ' | '.join(absent),
        'zero_recall_classes_any_seed': ' | '.join(sorted(set(zero_recall))),
        'zero_recall_class_seed_cases': len(zero_recall),
    }


def _communication_values(rows, mode):
    """Measure logical model payload using the correct FedAvg transport branch."""
    fields = ('round_payload_bytes', 'training_payload_bytes', 'shared_preprocessing_setup_payload_bytes')
    if mode not in COMMUNICATION_MODES:
        return {**{f'{field}_mean': 0 for field in fields}, 'payload_scope': 'No inter-client model exchange'}
    values = {field: [int(row[field]) for row in rows if row.get(field)] for field in fields}
    if any(len(values[field]) != len(rows) or not rows for field in fields[:2]):
        raise ValueError(f'Missing logical payload for benchmark mode {mode}')
    return {
        **{f'{field}_mean': mean(items) if items else None for field, items in values.items()},
        'payload_scope': _uniform(rows, 'scope'),
    }


def _operating_values(metrics, target_far):
    """Record the observed FAR worst seed and count of cap exceedances."""
    far = [float(row['far']) for row in metrics if row['far']]
    return {
        'far_cap': target_far,
        'far_max': max(far, default=None),
        'far_observed_seeds': len(far),
        'far_seeds_over_cap': sum(value > target_far for value in far),
    }


def _baseline_row(dataset, mode, metrics, classes, communication, target_far):
    """Summarize a single comparable benchmark branch."""
    return {
        'dataset': dataset,
        'mode': mode,
        'role': _uniform(metrics, 'role'),
        'protocol': _uniform(metrics, 'protocol'),
        'scenario': _uniform(metrics, 'scenario'),
        'clients': _uniform(metrics, 'clients', int),
        'alpha': _uniform(metrics, 'alpha', float),
        'allocation': 'Dirichlet label-skewed non-IID',
        **_coverage_values(classes, metrics),
        **_operating_values(metrics, target_far),
        **_communication_values(communication, mode),
    }


def benchmark_paper_rows(root):
    """Build paper metrics only from native, immutable VAL tables."""
    if not all((root / name).is_file() for name in BASELINE_SOURCES):
        return []
    config = _read(root / 'run.json')
    smoke = config['synthetic_only']
    if type(smoke) is not bool:
        raise ValueError('Invalid synthetic scope in the benchmark run plan')
    seeds = (SEEDS[0],) if smoke else SEEDS
    _validate_plan(root, seeds, smoke)
    _validate_val_rows(root, seeds)
    summaries = _metric_summaries(root, seeds)
    metrics = _by_case(root, 'metrics_by_seed.csv')
    classes = _by_case(root, 'per_class.csv')
    payload = _by_case(root, 'communication.csv', role=None)
    settings = list(_rows(root, 'parameters.csv'))
    far_by_dataset = {
        name: _uniform([row for row in settings if row['dataset'] == name], 'target_far', float) for name in DATASETS
    }
    return [
        {
            **_baseline_row(
                name,
                mode,
                metrics[name, mode],
                classes[name, mode],
                payload.get((name, COMMUNICATION_MODES.get(mode)), []),
                far_by_dataset[name],
            ),
            **summaries[name, mode],
        }
        for name in DATASETS
        for mode in MODES
    ]


def _baseline_table(root):
    """Publish the absolute and scope-aware paper table."""
    rows = benchmark_paper_rows(root)
    if not rows:
        raise ValueError('Missing source tables for the benchmark paper comparison')
    write_csv(root / 'paper_baseline.csv', rows)


def _paired_table(root, seeds):
    """Keep the original adapter contrasts and paired intervals visible."""
    rows = [
        row
        for row in _rows(root, 'paired_differences.csv')
        if row['contrast'] in CONTRASTS and row['metric'] in METRICS
    ]
    expected = {(name, contrast, metric) for name in DATASETS for contrast in CONTRASTS for metric in METRICS}
    observed = set()
    for row in rows:
        _require_grouped_main(row)
        key = (row['dataset'], row['contrast'], row['metric'])
        if key in observed or int(row['n']) != len(seeds):
            raise ValueError(f'Invalid paired benchmark seed coverage: {key}')
        observed.add(key)
    if observed != expected:
        raise ValueError('Incomplete paired benchmark contrasts')
    write_csv(root / 'paper_paired.csv', rows)


def _artifacts(output, seeds):
    """Bind primary tables, role and CAL-group audits, and per-seed case receipts."""
    names = ['run.json', 'complete.json', *(path.name for path in sorted(output.glob('*.csv')))]
    for dataset in DATASETS:
        root = f'{dataset}/grouped'
        names.append(f'{root}/data_audit.json')
        for seed in seeds:
            prefix = f'{root}/main/seed_{seed}'
            names.extend((f'{prefix}/result.json', f'{prefix}/cal_group_split.json'))
    return _bindings(output, names)


def run_benchmark_matrix(output, project, *, device='auto', smoke=False):
    """Run all matched controls. Accept an output and checkout path, and return a stage receipt."""
    output, dataset_root = Path(output).resolve(), Path(project).resolve() / 'dataset'
    output.parent.mkdir(parents=True, exist_ok=True)
    with _output_lock(output.parent, filename=f'.{output.name}.benchmark.lock'):
        return _run_benchmark_locked(output, dataset_root, device, smoke)


def _run_benchmark_locked(output, dataset_root, device, smoke):
    """Train and finalize the benchmark while holding its output lock."""
    seeds = (SEEDS[0],) if smoke else SEEDS
    arguments = grouped_validation_arguments(output, dataset_root, device, DATASETS, seeds, smoke=smoke)
    arguments.extend(('--threshold-insufficient-policy', 'disable'))
    if run_revision_matrix(arguments) != 0:
        raise RuntimeError('Four-dataset grouped benchmark did not finish')
    expected = len(DATASETS) * len(seeds)
    if _read(output / 'complete.json') != {'replicates': expected, 'data_audits': len(DATASETS)}:
        raise ValueError('The benchmark matrix did not complete every declared case')
    config = _read(output / 'run.json')
    if (
        config.get('tabm_attention') or 'tree_alarm_count' in config
        or config.get('tabm_refinement') or config.get('portable_detector')
    ):
        raise ValueError('The benchmark detector differs from the requested model')
    _baseline_table(output)
    _paired_table(output, seeds)
    return {
        'status': 'completed',
        'output': str(output),
        'scientific_status': 'grouped_development_original_adapter',
        'execution_origin': 'current_fresh_training',
        'test_evaluated': False,
        'datasets': list(DATASETS),
        'cases_per_dataset': dict.fromkeys(DATASETS, len(seeds)),
        'artifacts': _artifacts(output, seeds),
    }
