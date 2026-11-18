"""Run and bind fresh local scale, CAN quantization and NB15 temporal evidence."""

import csv
from dataclasses import dataclass
from pathlib import Path

from . import revision_next as shared
from . import revision_run as native
from .revision_supplement import CAN_DATASETS, FLOW_DATASETS
from .revision_v12 import _bindings, _hash, _read, _write

SCALE_BUDGETS = {
    'train_total': 100_000,
    'cal_total': 16_500,
    'train_per_client': 33_000,
    'cal_per_client': 5_500,
}
QUANTIZATION_MODES = ('base_threshold', 'local_only', 'full', 'local_only_fine', 'full_fine')
TEMPORAL_MODES = (*native.MATCHED_MODES, 'fedavg', 'fedavg_threshold')
SCALE_MODES = (
    'base_threshold',
    'head_local',
    'head_federated',
    'fixed_local',
    'fixed_federated',
    'guarded_local',
    'guarded_federated',
)
TEST_ROLES = ('val', 'test', 'test_unique_unseen')
STAGE_RECEIPT = 'stage_complete.json'


@dataclass(frozen=True)
class _Supplement:
    """Declare one historical-style native phase to reproduce from fresh training."""

    protocol: str
    profile: str
    scenario: str
    datasets: tuple[str, ...]
    modes: tuple[str, ...]
    alpha: float = 0.8
    native_flags: tuple[str, ...] = ()


CAN_PHASES = (
    _Supplement('grouped', 'quantization_grouped', 'supplement_quantization', CAN_DATASETS, QUANTIZATION_MODES),
    _Supplement('frozen', 'quantization', 'supplement_quantization', CAN_DATASETS, QUANTIZATION_MODES),
)
NB15_TEMPORAL = _Supplement('temporal', 'temporal', 'supplement_temporal', ('CIC-UNSW-NB15',), TEMPORAL_MODES)
KMINUS1_TRANSFER = _Supplement(
    'grouped',
    'transfer',
    'supplement_transfer',
    FLOW_DATASETS,
    native.MATCHED_MODES,
    0.3,
    ('--transfer-full-train', '--transfer-score-diagnostics', '--transfer-alpha', '0.3'),
)


def _fresh_paths(output, data_root):
    """Keep new experiments separate from the raw dataset and previous outputs."""
    output, data_root = Path(output).resolve(), Path(data_root).resolve()
    if output.exists():
        raise ValueError(f'Supplementary matrix needs a new output directory: {output}')
    if output.is_relative_to(data_root) or data_root.is_relative_to(output):
        raise ValueError('Supplementary output must be separate from the raw dataset')
    return output, data_root


def _common_arguments(output, data_root, device, seeds, smoke):
    """Pin the data, runtime and current-run-only seed set for the backend."""
    return [
        '--out',
        str(output),
        '--data-root',
        str(data_root),
        '--device',
        device,
        '--seeds',
        *map(str, seeds),
        *(['--smoke'] if smoke else []),
    ]


def _rows(root, artifact):
    """Read one native aggregate table inside its fresh stage."""
    with (root / artifact).open(encoding='utf-8-sig', newline='') as stream:
        yield from csv.DictReader(stream)


def _case_key(record):
    """Use the identity columns shared by the two experiment backends."""
    return (
        record['dataset'],
        record['scenario'],
        int(record['clients']),
        float(record['alpha']),
        int(record['seed']),
    )


def _checked_metrics(output, filename, cases, modes, roles, *, protocol=None):
    """Require the paired modes in every planned role and seed."""
    expected = {(case, role, mode) for case in cases for role in roles for mode in modes}
    observed = set()
    for row in _rows(output, filename):
        case, role, mode = _case_key(row), row['role'], row['mode']
        if case not in cases or role not in roles or (protocol is not None and row['protocol'] != protocol):
            raise ValueError(f'Unexpected supplementary metric population at {output}: {(case, role)}')
        if mode in modes:
            key = (case, role, mode)
            if key in observed:
                raise ValueError(f'Duplicate supplementary metric at {output}: {key}')
            observed.add(key)
    if observed != expected:
        raise ValueError(f'Incomplete supplementary metric comparisons at {output}')


def _artifact_names(output, extras):
    """Bind native public tables and the explicit configuration, role and case receipts."""
    paths = [output / 'run.json', output / 'complete.json', *output.glob('*.csv'), *extras]
    if any(path.resolve() != path or not path.is_relative_to(output) for path in paths):
        raise ValueError('Supplementary matrix artifact path is redirected or outside the run')
    return sorted({path.relative_to(output).as_posix() for path in paths})


def _publish(output, names, contents):
    """Write the wrapper receipt after validating native artifacts."""
    report = {
        **contents,
        'datasets': list(contents['datasets']),
        'seeds': list(contents['seeds']),
        'status': 'completed',
        'output': str(output),
        'execution_origin': 'current_fresh_training',
        'independent_confirmation': False,
        'artifacts': _bindings(output, names),
    }
    _write(output / STAGE_RECEIPT, report)
    return {**report, 'artifacts': {**report['artifacts'], STAGE_RECEIPT: _hash(output / STAGE_RECEIPT)}}


def _scale_cases(seeds, smoke):
    """Pin both training budgets and the separate three-client alpha sweep."""
    counts = (3, 5) if smoke else (3, 5, 10)
    conditions = [(f'scale_{budget}_c{clients}', clients, 0.8) for budget in ('total', 'local') for clients in counts]
    conditions += [
        (f'scale_alpha_{label}', 3, alpha) for label, alpha in (('low', 0.3), ('reference', 0.8), ('high', 2.0))
    ]
    return {(dataset, *case, seed) for dataset in FLOW_DATASETS for case in conditions for seed in seeds}


def _scale_arguments(output, data_root, device, seeds, smoke):
    """Reuse the declared V12 shared-head scale and operating-guard recipe."""
    budgets = [
        arg for name, value in SCALE_BUDGETS.items() for arg in (f"--scale-{name.replace('_', '-')}", str(value))
    ]
    return [
        '--task',
        'scale',
        '--dataset',
        *FLOW_DATASETS,
        '--cal-policies',
        'natural',
        '--primary-cal-policy',
        'natural',
        '--operating-guard',
        '--operating-policy',
        'fixed',
        '--no-save-scores',
        *budgets,
        *_common_arguments(output, data_root, device, seeds, smoke),
    ]


def _validate_scale_run(output, data_root, cases, seeds, smoke):
    """Check live settings and complete workload counts before reading cases."""
    run = _read(output / 'run.json')
    settings = run['settings']
    expected = {
        'task': 'scale',
        'dataset': list(FLOW_DATASETS),
        'seeds': list(seeds),
        'data_root': str(data_root),
        'smoke': smoke,
        'check_data': False,
        'head_only': False,
        'operating_guard': True,
        'operating_policy': 'fixed',
        'cal_policies': ['natural'],
        'primary_cal_policy': 'natural',
        'replay_from': None,
        'save_scores': False,
        'save_row_evidence': False,
        **{f'scale_{key}': value for key, value in SCALE_BUDGETS.items()},
    }
    if (
        any(settings.get(key) != value for key, value in expected.items())
        or run['code_sha256'] != shared._source_receipt()
    ):
        raise ValueError('The completed scale plan differs from the declared fresh experiment')
    marker = _read(output / 'complete.json')
    if (marker['planned_case_policies'], marker['evaluated'], marker['test_evaluated']) != (
        len(cases),
        len(cases),
        False,
    ):
        raise ValueError('Scale matrix did not evaluate every planned VAL case')


def _validate_scale_budget(audit, case, smoke):
    """Confirm advertised TRAIN and CAL budgets against each actual allocation."""
    _, scenario, clients, _, _ = case
    budget = {'total': 'total', 'local': 'local', 'alpha': 'heterogeneity'}[scenario.split('_')[1]]
    if audit['test_evaluated'] is not False or audit['budget_policy'] != budget:
        raise ValueError(f'Scale case used a different protocol: {case}')
    if smoke:
        return
    for role, field in (('train', 'train_rows_by_client'), ('cal', 'original_cal_rows_by_client')):
        expected = (
            SCALE_BUDGETS[f'{role}_total'] if budget != 'local' else SCALE_BUDGETS[f'{role}_per_client'] * clients
        )
        counts = audit[field]
        if audit[f'{role}_rows'] != expected or len(counts) != clients or sum(counts) != expected:
            raise ValueError(f'Scale case violated its {role.upper()} budget: {case}')
        if budget == 'local' and set(counts) != {SCALE_BUDGETS[f'{role}_per_client']}:
            raise ValueError(f'Scale case violated its fixed per-client budget: {case}')


def _scale_receipts(output, cases, smoke):
    """Bind per-seed results to the native case ledger and verify no cache was used."""
    receipts = {}
    for row in _rows(output, 'case_status.csv'):
        case = _case_key(row)
        dataset, scenario, _, _, seed = case
        path = output / dataset / scenario / 'target_0' / f'seed_{seed}' / 'result.json'
        if case in receipts or case not in cases or row['cal_policy'] != 'natural' or row['status'] != 'evaluated':
            raise ValueError(f'Unexpected scale workload: {case}')
        if any(row.get(field) for field in ('score_cache', 'frozen_head', 'fixed_subtype_evidence')):
            raise ValueError(f'Scale case contains private replay artifacts: {case}')
        if Path(row['result']) != path or path.resolve() != path or row['result_sha256'] != _hash(path):
            raise ValueError(f'Scale workload result binding differs: {case}')
        result = _read(path)
        if _case_key(result['identity']) != case or result['status'] != 'evaluated':
            raise ValueError(f'Scale case result differs from the workload ledger: {case}')
        audit = result['audit']
        if audit['policies']['natural']['group_audit']['global_exact_feature_overlap'] != 0:
            raise ValueError(f'Scale case has cross-role CAL group overlap: {case}')
        _validate_scale_budget(audit, case, smoke)
        receipts[case] = path
    if receipts.keys() != cases:
        raise ValueError('The scale workload ledger does not cover the declared cases')
    return list(receipts.values())


def run_scale_matrix(output, data_root, device='auto', *, smoke=False):
    """Run the shared-head scale and heterogeneity matrix on grouped VAL."""
    output, data_root = _fresh_paths(output, data_root)
    seeds = native.SEEDS[:1] if smoke else native.SEEDS
    cases = _scale_cases(seeds, smoke)
    if shared.main(_scale_arguments(output, data_root, device, seeds, smoke)) != 0:
        raise RuntimeError('Native shared-head scale matrix did not finish')
    _validate_scale_run(output, data_root, cases, seeds, smoke)
    _checked_metrics(output, 'metrics.csv', cases, SCALE_MODES, ('val',))
    audit_paths = [output / dataset / 'data_audit.json' for dataset in FLOW_DATASETS]
    names = _artifact_names(output, [*audit_paths, *_scale_receipts(output, cases, smoke)])
    return _publish(
        output,
        names,
        {
            'datasets': FLOW_DATASETS,
            'seeds': seeds,
            'cases': len(cases),
            'synthetic_only': smoke,
            'test_evaluated': False,
            'scientific_status': 'grouped_development_scale_and_heterogeneity',
            'evaluation_scope': 'Grouped same-source VAL; fixed-total, fixed-per-client and alpha conditions',
        },
    )


def _supplement_cases(phase, seeds):
    """Declare the ordinary three-client supplement cases."""
    return {(dataset, phase.scenario, 3, phase.alpha, seed) for dataset in phase.datasets for seed in seeds}


def _validate_supplement(output, data_root, phase, seeds, cases, smoke):
    """Verify retrospective roles, profile and absence of a replay/selection input."""
    run = _read(output / 'run.json')
    expected_matrix = [
        {'dataset': dataset, 'protocol': phase.protocol, 'scenarios': [[phase.scenario, 3, phase.alpha]]}
        for dataset in phase.datasets
    ]
    supplement = run['supplement']
    if (
        (run['matrix'], run['seeds'], run['synthetic_only'], run['data_root'])
        != (expected_matrix, list(seeds), smoke, None if smoke else str(data_root))
        or supplement['profile'] != phase.profile
        or not supplement['temporal_purge_exact']
        or supplement['subtype_ablation']
        or run['data_only']
        or run['source_sha256'] != native._source_hashes(supplement=True)
    ):
        raise ValueError(f'Native supplement differs from the declared {phase.profile} comparison')
    marker = _read(output / 'complete.json')
    if (marker['replicates'], marker['data_audits']) != (len(cases), len(phase.datasets)):
        raise ValueError(f'Native supplement incomplete: {phase.profile}')
    if phase == KMINUS1_TRANSFER and (
        supplement.get('transfer_training', {}).get('mode') != 'full'
        or not supplement.get('transfer_score_diagnostics')
        or supplement.get('missing_class_repair')
    ):
        raise ValueError('K−1 transfer requires full TRAIN score diagnostics without donor repair')


def _validate_temporal_audit(path):
    """Identify an exact-purged later-day holdout within the pinned NB15 source."""
    audit = _read(path)
    overlaps = audit['exact_feature_audit']['cross_role_exact_overlap']
    earlier = ('train:cal', 'train:val', 'train:test', 'cal:val', 'cal:test', 'val:test')
    if (
        audit['split_protocol'] != 'temporal_purged'
        or audit['external_captures'] is not False
        or any(overlaps[pair]['unique_shared_inputs'] for pair in earlier)
    ):
        raise ValueError('NB15 temporal stage requires exact-input role separation within the same capture')


def _phase_artifacts(output, phase, cases, smoke):
    """Check data-role and per-case identity receipts for a native phase."""
    paths = [output / dataset / phase.protocol / 'data_audit.json' for dataset in phase.datasets]
    if phase.protocol == 'temporal' and not smoke:
        _validate_temporal_audit(paths[0])
    for dataset, scenario, clients, alpha, seed in sorted(cases):
        path = output / dataset / phase.protocol / scenario / f'seed_{seed}' / 'result.json'
        result = _read(path)
        if result['protocol'] != phase.protocol or _case_key(result) != (
            dataset,
            scenario,
            clients,
            alpha,
            seed,
        ):
            raise ValueError(f'Native supplement case routing differs: {path}')
        if result.get('prediction_cache') is not None:
            raise ValueError(f'Native supplement produced a score cache: {path}')
        _validate_supplement_case(result['settings'], output / dataset / phase.protocol, phase, smoke)
        paths.append(path)
    return _artifact_names(output, paths)


def _validate_supplement_case(settings, dataset_dir, phase, smoke):
    """Check the case-specific quantization precision or full TRAIN population."""
    if phase in CAN_PHASES and (settings['secure_scale'], settings['secure_scale_fine']) != (10**6, 10**9):
        raise ValueError(f'Native CAN fixed-point scales differ: {dataset_dir}')
    if phase == KMINUS1_TRANSFER:
        train = settings['train_rows']
        expected = train if smoke else _read(dataset_dir / 'data_audit.json')['role_rows']['train']
        if (
            train != expected
            or train != settings['train_rows_before_sampling']
            or settings['train_sampling'] != 'all_rows'
            or settings['train_row_limit'] is not None
        ):
            raise ValueError(f'K−1 transfer did not use every TRAIN row: {dataset_dir}')


def _run_supplement(output, data_root, device, seeds, smoke, phase):
    """Run one native supplement phase with a fresh output and validated tables."""
    arguments = [
        '--profile',
        phase.profile,
        '--dataset',
        *phase.datasets,
        '--no-tcp',
        '--no-figures',
        *phase.native_flags,
        *_common_arguments(output, data_root, device, seeds, smoke),
    ]
    if native.main(arguments, supplement=True) != 0:
        raise RuntimeError(f'Native supplement did not finish: {phase.profile}')
    cases = _supplement_cases(phase, seeds)
    _validate_supplement(output, data_root, phase, seeds, cases, smoke)
    _checked_metrics(output, 'metrics_by_seed.csv', cases, phase.modes, TEST_ROLES, protocol=phase.protocol)
    return _phase_artifacts(output, phase, cases, smoke)


def run_can_quantization(output, data_root, device='auto', *, smoke=False):
    """Reproduce the original K−1 fixed-point mechanism on grouped and frozen CAN roles."""
    output, data_root = _fresh_paths(output, data_root)
    seeds = native.SEEDS[:1] if smoke else native.SEEDS
    names = []
    for phase in CAN_PHASES:
        arguments = (output / phase.protocol, data_root, device, seeds, smoke, phase)
        names.extend(f'{phase.protocol}/{name}' for name in _run_supplement(*arguments))
    return _publish(
        output,
        names,
        {
            'datasets': CAN_DATASETS,
            'protocols': [phase.protocol for phase in CAN_PHASES],
            'seeds': seeds,
            'cases': len(CAN_PHASES) * len(CAN_DATASETS) * len(seeds),
            'synthetic_only': smoke,
            'test_evaluated': True,
            'scientific_status': 'same_source_kminus1_quantization_diagnostic',
            'secure_scales': [10**6, 10**9],
            'evaluation_scope': ('Retrospective grouped and frozen CAN VAL and benchmark TEST, original K−1 adapters'),
        },
    )


def run_nb15_temporal(output, data_root, device='auto', *, smoke=False):
    """Evaluate the original pipeline on the exact-purged later-day NB15 split."""
    output, data_root = _fresh_paths(output, data_root)
    seeds = native.SEEDS[:1] if smoke else native.SEEDS
    names = _run_supplement(output, data_root, device, seeds, smoke, NB15_TEMPORAL)
    return _publish(
        output,
        names,
        {
            'datasets': NB15_TEMPORAL.datasets,
            'protocols': ['temporal_purged'],
            'seeds': seeds,
            'cases': len(seeds),
            'synthetic_only': smoke,
            'test_evaluated': True,
            'scientific_status': 'same_source_exact_purged_temporal_test',
            'evaluation_scope': 'Retrospective later-day TEST within the same pinned NB15 capture',
        },
    )


def run_kminus1_transfer(output, data_root, device='auto', *, smoke=False):
    """Run the original K−1 missing-class transfer score comparison with full TRAIN."""
    output, data_root = _fresh_paths(output, data_root)
    seeds = native.SEEDS[:1] if smoke else native.SEEDS
    names = _run_supplement(output, data_root, device, seeds, smoke, KMINUS1_TRANSFER)
    scores = 'transfer_score_deltas.csv'
    counts = len(list(_rows(output, scores))) if scores in names else 0
    return _publish(
        output,
        names,
        {
            'datasets': FLOW_DATASETS,
            'protocols': ['grouped'],
            'seeds': seeds,
            'alpha': KMINUS1_TRANSFER.alpha,
            'cases': len(FLOW_DATASETS) * len(seeds),
            'synthetic_only': smoke,
            'test_evaluated': True,
            'eligible_score_contrast_rows': counts,
            'full_train': True,
            'scientific_status': 'same_source_kminus1_transfer_diagnostic',
            'evaluation_scope': (
                'Retrospective exact-unseen benchmark TEST; original K−1 missing-class transfer strata'
            ),
        },
    )
