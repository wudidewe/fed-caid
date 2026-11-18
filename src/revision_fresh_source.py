"""Produce fresh review artifacts through the native evaluators."""

import csv
from pathlib import Path

from .revision_nb15_source import BINARY_OUTPUTS, STAGES, binary_pair_assessment
from .revision_next import _source_receipt
from .revision_next import main as run_native
from .revision_run import main as run_matrix
from .revision_v12 import _bindings, _hash, _read, _write

SEEDS = (42, 43, 44, 45, 46)
CONTROLS = ('v12_plan.json', 'v12_state.json', 'v12_complete.json')
IOV = 'CIC-IoV-2024'
IOV_STATUS = 'iov_fixed_mask_status.json'
IOV_TABLES = (
    'iov_operating_diagnostics.csv',
    'iov_alarm_overlap.csv',
    'iov_class_metrics.csv',
    'iov_fixed_mask_paired.csv',
    IOV_STATUS,
)
IOV_FIXED_MODES = ('fixed_base_local', 'fixed_base_full')
IOV_MODES = ('base_threshold', 'local_only', 'full', *IOV_FIXED_MODES)
IOV_ALARM_FIELDS = (
    'threshold',
    'alarm_mask_sha256',
    'ordered_attack_score_sha256',
    'benign_rows',
    'attack_rows',
    'false_alarm_rows',
    'missed_attack_rows',
)


def _common_switches(data_root, device):
    """Choose the fixed five-seed dataset and reserved CAL protocol."""
    return [
        '--operating-guard',
        '--data-root',
        str(data_root),
        '--device',
        device,
        '--seeds',
        *map(str, SEEDS),
        '--cal-policies',
        'reserved',
        '--primary-cal-policy',
        'reserved',
    ]


def _nb15_switches(common):
    """Build the two matched detector fits with fresh private scores."""
    fixed = [
        *common,
        '--dataset',
        'CIC-UNSW-NB15',
        '--operating-policy',
        'fixed',
        '--target',
        'Exploits',
        '--save-scores',
        '--export-frozen',
    ]
    return {name: [*fixed, '--detector-objective', objective] for name, objective in STAGES.items()}


def _hulk_switches(common):
    """Build the conditions that export the five version-7 Hulk receiver bundles."""
    return {
        'attribution': [
            *common,
            '--dataset',
            'CIC-IDS-2017',
            '--operating-policy',
            'cal-select',
            '--single-donor',
            '--target',
            'DoS Hulk',
            '--export-frozen',
            '--no-save-scores',
        ]
    }


def _native_stage(path, switches, code):
    """Train into a newly created native stage and bind its evaluated tables."""
    print(f'FedCAID | fresh native training | {path}', flush=True)
    if run_native([*switches, '--out', str(path)]) != 0:
        raise RuntimeError(f'Native experiment failed: {path}')
    receipt = _read(path / 'complete.json')
    expected = {
        'mode': 'evaluation',
        'task': 'train-absence',
        'smoke': False,
        'score_replay': False,
        'test_evaluated': False,
        'evaluated': len(SEEDS),
        'planned_case_policies': len(SEEDS),
    }
    if any(receipt.get(name) != value for name, value in expected.items()):
        raise ValueError(f'Native stage did not complete all five real development cases: {path}')
    if _read(path / 'run.json')['code_sha256'] != code:
        raise ValueError(f'Native training source changed during fresh source generation: {path}')
    audits = sorted(path.glob('*/data_audit.json'))
    if len(audits) != 1:
        raise ValueError(f'Native stage is missing its single dataset-role audit: {path}')
    names = (
        'run.json',
        'complete.json',
        *(p.name for p in sorted(path.glob('*.csv'))),
        *(p.relative_to(path).as_posix() for p in audits),
    )
    if 'case_status.csv' not in names:
        raise ValueError(f'Native stage is missing its case inventory: {path}')
    return receipt, _bindings(path, names)


def _ordered_stages(output, stages, snapshot):
    """Record each successful child while keeping partial runs uncommitted."""
    results, state = {}, {}
    for name, switches in stages.items():
        results[name], state[name] = _native_stage(output / name, switches, snapshot)
        _write(output / 'v12_state.json', state)
    return results, state


def _commit(output, mode, results, state, binary_assessment):
    """Publish a V12-compatible receipt for consumers within this invocation."""
    receipt = {
        'schema_version': 1,
        'mode': mode,
        'scope': 'development',
        'status': 'completed',
        'plan_sha256': _hash(output / 'v12_plan.json'),
        'state_sha256': _hash(output / 'v12_state.json'),
        'test_evaluated': False,
        'stages': results,
    }
    evidence = {}
    if binary_assessment is not None:
        _write(output / BINARY_OUTPUTS[0], binary_assessment)
        evidence = _bindings(output, BINARY_OUTPUTS)
        receipt.update(binary_assessment=binary_assessment, binary_artifacts=evidence)
    _write(output / 'v12_complete.json', receipt)
    artifacts = {f'{stage}/{name}': digest for stage, bindings in state.items() for name, digest in bindings.items()}
    return {
        **receipt,
        'output': str(output),
        'execution_origin': 'current_fresh_training',
        'artifacts': {**_bindings(output, CONTROLS), **artifacts, **evidence},
    }


def _generate(output, project, device, mode, configure):
    """Start a new wrapper with a single reviewed source snapshot."""
    output, project = Path(output).resolve(), Path(project).resolve(strict=True)
    data_root = (project / 'dataset').resolve()
    if output.is_relative_to(data_root) or data_root.is_relative_to(output):
        raise ValueError('Fresh experiment outputs must be separate from the dataset directory')
    stages = configure(_common_switches(data_root, device))
    code = _source_receipt()
    plan = {
        'schema_version': 1,
        'implementation': 'native_current_source',
        'mode': mode,
        'scope': 'development',
        'stages': stages,
        'source_sha256': {**code, Path(__file__).name: _hash(Path(__file__))},
        'reused_prior_outputs': False,
    }
    if mode == 'nb15-pair':
        plan['detector_objectives'] = STAGES
    output.mkdir(parents=True, exist_ok=False)
    _write(output / 'v12_plan.json', plan)
    results, state = _ordered_stages(output, stages, code)
    binary_assessment = binary_pair_assessment(output, state, plan) if mode == 'nb15-pair' else None
    return _commit(output, mode, results, state, binary_assessment)


def run_fresh_nb15_pair(output, project, *, device='auto'):
    """Fit both NB15 detector objectives on the current run's five splits."""
    return _generate(output, project, device, 'nb15-pair', _nb15_switches)


def run_fresh_hulk_attribution(output, project, *, device='auto'):
    """Fit the five Hulk development cases and freeze their new receiver models."""
    return _generate(output, project, device, 'attribution', _hulk_switches)


def _iov_arguments(output, project, device, seeds, smoke):
    """Use the original IoV matrix settings with an additive fixed-alarm control."""
    return [
        '--dataset',
        IOV,
        '--protocol',
        'grouped',
        '--scenarios',
        'main',
        '--seeds',
        *map(str, seeds),
        '--device',
        device,
        '--data-root',
        str(project / 'dataset'),
        '--out',
        str(output),
        '--group-cal',
        '--development-only',
        '--threshold-insufficient-policy',
        'disable',
        '--no-tcp',
        '--no-figures',
        '--iov-diagnostics',
        '--iov-fixed-base-alarm',
        *(['--smoke'] if smoke else []),
    ]


def _iov_case_files(output, seeds):
    """Bind all native case receipts and require VAL to be the sole scored role."""
    prefix = f'{IOV}/grouped'
    names = [f'{prefix}/data_audit.json']
    for seed in seeds:
        case = f'{prefix}/main/seed_{seed}'
        result = _read(output / case / 'result.json')
        identity = tuple(result[key] for key in ('dataset', 'protocol', 'scenario', 'seed'))
        if (
            identity != (IOV, 'grouped', 'main', seed)
            or {row['role'] for row in result['evaluations']} != {'val'}
            or result['prediction_cache'] is not None
            or result['settings'].get('iov_fixed_base_alarm') is not True
            or not set(IOV_MODES) <= {row['mode'] for row in result['evaluations']}
        ):
            raise ValueError(f'Unexpected scope or score cache for IoV seed {seed}')
        names.extend((f'{case}/result.json', f'{case}/cal_group_split.json'))
    return names


def _verify_iov_rows(output, seeds):
    """Require an unambiguous observation for every case, client, role and branch."""
    expected = {
        (str(seed), str(client), role, mode)
        for seed in seeds
        for client in range(3)
        for role in ('cal_holdout', 'val')
        for mode in IOV_MODES
    }
    with (output / IOV_TABLES[0]).open(encoding='utf-8-sig', newline='') as stream:
        records = list(csv.DictReader(stream))
    rows = {tuple(row[field] for field in ('seed', 'client', 'role', 'mode')): row for row in records}
    if len(records) != len(expected) or rows.keys() != expected:
        raise ValueError('The IoV operating diagnostic table has incomplete or duplicate cases')
    _verify_fixed_alarms(rows)


def _verify_fixed_alarms(rows):
    """Require exact base-cutoff and ordered alarm parity for both fixed controls."""
    for (seed, client, role, mode), row in rows.items():
        if mode not in IOV_FIXED_MODES:
            continue
        reference = rows[seed, client, role, 'base_threshold']
        hashes_present = all(row[field] for field in ('alarm_mask_sha256', 'ordered_attack_score_sha256'))
        if not hashes_present or any(row[field] != reference[field] for field in IOV_ALARM_FIELDS):
            raise ValueError(f'The fixed alarm control changed the baseline mask on {seed}/{client}/{role}/{mode}')


def _iov_artifacts(output, seeds, smoke):
    """Verify native completion and bind the aggregate tables for the report."""
    if _read(output / 'complete.json') != {'replicates': len(seeds), 'data_audits': 1}:
        raise ValueError('The IoV diagnostic did not finish every declared case')
    config = _read(output / 'run.json')
    if (
        config['seeds'] != list(seeds)
        or config['synthetic_only'] is not smoke
        or config.get('iov_diagnostics') is not True
        or config.get('iov_fixed_base_alarm') is not True
    ):
        raise ValueError('The IoV result does not match its diagnostic settings')
    _verify_iov_rows(output, seeds)
    names = {'run.json', 'complete.json', *IOV_TABLES, *(p.name for p in output.glob('*.csv'))}
    return _bindings(output, sorted(names) + _iov_case_files(output, seeds))


def run_fresh_iov_diagnosis(output, project, *, device='auto', smoke=False):
    """Fit one matched set of IoV models and compare adapted and fixed alarms."""
    output, project = Path(output).resolve(), Path(project).resolve()
    seeds = SEEDS[:1] if smoke else SEEDS
    if run_matrix(_iov_arguments(output, project, device, seeds, smoke)) != 0:
        raise RuntimeError('CIC-IoV-2024 grouped alarm diagnosis did not finish')
    artifacts = _iov_artifacts(output, seeds, smoke)
    assessment = _read(output / IOV_STATUS)
    return {
        'status': 'completed',
        'scientific_status': assessment['scientific_status'],
        'execution_origin': 'current_fresh_training',
        'output': str(output),
        'seeds': list(seeds),
        'test_evaluated': False,
        'threshold_candidate_evaluated': False,
        'fixed_base_alarm_evaluated': True,
        'synthetic_only': smoke,
        'artifacts': artifacts,
    }
