"""Continue the supplementary suffix of an interrupted full review in place."""

from pathlib import Path

from . import revision_full_matrix as matrix
from . import revision_run as native
from .revision_final_review import (
    CURRENT_REVIEW,
    FULL_WORKFLOW,
    SCHEMA_VERSION,
    STAGE_LABELS,
    _run_stage,
    _verify_source,
    _write_abort,
    _write_completion,
)
from .revision_fresh_report import _artifact_path, _stage_root, _verified_files, run_fresh_report
from .revision_v12 import _bindings, _completion_summary, _hash, _output_lock, _read, _write

SUPPLEMENTARY_STAGES = ('can_quantization', 'nb15_temporal', 'kminus1_transfer', 'report')
TERMINAL_MARKERS = {'completed', 'completed_with_pending'}


def _run_path(project, name):
    """Resolve an explicit run name under the review output container."""
    if Path(name).name != name or not name.startswith('run_'):
        raise ValueError('Expected an explicit review run name')
    output = project / 'outputs' / CURRENT_REVIEW / name
    if output.resolve(strict=True) != output or not output.is_dir():
        raise ValueError(f'Invalid review run directory: {output}')
    return output


def _read_plan(project, output, plan_path, marker):
    """Check the original run identity and all previously bound source files."""
    plan = _read(plan_path)
    if (
        plan['schema_version'] != SCHEMA_VERSION
        or marker['schema_version'] != SCHEMA_VERSION
        or marker['plan_sha256'] != _hash(plan_path)
        or Path(plan['project']) != project
        or Path(plan['run_directory']) != output
        or plan['workflow'] != FULL_WORKFLOW
        or plan['synthetic_only'] is not False
        or plan['previous_run_inputs_allowed'] is not False
        or plan['stages'] != list(STAGE_LABELS)
        or tuple(plan['stages'][-len(SUPPLEMENTARY_STAGES) :]) != SUPPLEMENTARY_STAGES
    ):
        raise ValueError('The interrupted run differs from its declared full-review plan')
    _verify_source(plan)
    return plan


def _verify_stage(output, stage, record, optional):
    """Verify completed evidence or the original optional pending declaration."""
    root = _stage_root(output, stage, record)
    if record['status'] == 'completed' and record.get('artifacts'):
        for _ in _verified_files(root, record['artifacts']):
            pass
        return
    preflight = optional.get(stage, {})
    if preflight.get('status') != 'pending' or not all(record.get(k) == v for k, v in preflight.items()):
        raise ValueError(f'Cannot preserve an unverified stage: {stage}')


def _read_state(output, state_path, marker, plan):
    """Require a contiguous completed prefix and verify each preserved stage."""
    state = _read(state_path)
    stages = plan['stages']
    minimum = len(stages) - len(SUPPLEMENTARY_STAGES)
    if len(state) < minimum or set(state) != set(stages[: len(state)]):
        raise ValueError('Saved stages do not cover the completed prefix through the scale matrix')
    if marker['status'] in TERMINAL_MARKERS and (
        len(state) != len(stages) or marker.get('state_sha256') != _hash(state_path)
    ):
        raise ValueError('Finalized run state differs from its completion receipt')
    ordered = {name: state[name] for name in stages if name in state}
    for name, record in ordered.items():
        _verify_stage(output, name, record, plan['input_preflight'])
    return ordered


def _continuation_receipt(output, plan_path, state_path, state):
    """Bind the original plan, initial progress and the added continuation code."""
    receipt = _artifact_path(output, 'continuation_plan.json')
    project = Path(__file__).resolve().parent.parent
    names = ('main.py', 'src/revision_resume_review.py')
    binding = {'original_plan_sha256': _hash(plan_path), 'source_sha256': _bindings(project, names)}
    if receipt.exists():
        recorded = _read(receipt)
        if any(recorded.get(key) != value for key, value in binding.items()):
            raise ValueError('Continuation code or the original plan changed')
        for _ in _verified_files(output, recorded['preserved_case_sha256']):
            pass
        return
    _write(
        receipt,
        {
            **binding,
            'initial_state_sha256': _hash(state_path),
            'preserved_stages': list(state),
            'preserved_case_sha256': _case_bindings(output, state),
        },
    )


def _case_bindings(output, state):
    """Bind atomic case files saved before the missing stage completed."""
    paths = (
        path
        for stage in SUPPLEMENTARY_STAGES[:-1]
        if stage not in state
        for path in (output / stage).rglob('result.json')
    )
    names = sorted(path.relative_to(output).as_posix() for path in paths)
    return {name: _hash(_artifact_path(output, name)) for name in names}


def _recover_report(output, state):
    """Reuse a verified report published before its stage record was saved."""
    path = _artifact_path(output, 'complete.json')
    report = _read(path)
    _verify_stage(output.parent, 'report', report, {})
    sources = {
        str(_stage_root(output.parent, stage, record) / name): digest
        for stage, record in state.items()
        for name, digest in record.get('artifacts', {}).items()
    }
    evidence = _read(_artifact_path(output, 'evidence_plan.json'))
    if evidence['sources'] != sources or evidence['run_plan_sha256'] != _hash(output.parent / 'final_plan.json'):
        raise ValueError('The published report does not match the current continuation')
    return {**report, 'artifacts': {**report['artifacts'], path.name: _hash(path)}}


def _resume_device(output, plan):
    """Use the actual device recorded by the completed benchmark."""
    config = _read(_artifact_path(output / 'benchmark_matrix', 'run.json'))
    device = config['device']
    if device not in {'cpu', 'cuda'} or plan['requested_device'] not in {'auto', device}:
        raise ValueError('The benchmark device differs from the original run plan')
    return device


def _operations(project, state, device):
    """Use the current run's original supplement settings and reporting pipeline."""
    data = project / 'dataset'
    return {
        'can_quantization': lambda path: resume_can_quantization(path, data, device),
        'nb15_temporal': lambda path: resume_nb15_temporal(path, data, device),
        'kminus1_transfer': lambda path: resume_kminus1_transfer(path, data, device),
        'report': lambda path: _recover_report(path, state) if path.exists() else run_fresh_report(path, state),
    }


def _run_remaining(output, state_path, plan, state, operations):
    """Execute only missing stages and atomically commit each completed record."""
    for name in SUPPLEMENTARY_STAGES:
        if name in state:
            continue
        _verify_source(plan)
        record = _run_stage(name, output / name, operations[name])
        if record['status'] != 'completed':
            raise RuntimeError(f"{STAGE_LABELS[name]}: {record.get('reason', record['status'])}")
        _verify_stage(output, name, record, {})
        state[name] = record
        _write(state_path, state)
    _verify_source(plan)


def _continue_locked(project, output):
    """Bind a stopped run and finish its supplementary stages and report."""
    plan_path, state_path, marker_path = (
        _artifact_path(output, name) for name in ('final_plan.json', 'final_state.json', 'final_complete.json')
    )
    marker = _read(marker_path)
    plan = _read_plan(project, output, plan_path, marker)
    state = _read_state(output, state_path, marker, plan)
    if marker['status'] in TERMINAL_MARKERS:
        summary = _completion_summary(state, plan['stages'])
        print(f"FedCAID | {summary['status']} | {output / 'report' / 'SUMMARY.md'}", flush=True)
        return 2 if summary['required_pending_steps'] or summary['failed_steps'] else 0
    if marker['status'] not in {'running', 'aborted'}:
        raise ValueError(f"Cannot continue a run with status {marker['status']}")
    _continuation_receipt(output, plan_path, state_path, state)
    return _execute_remaining(project, output, marker_path, plan_path, state_path, plan, state)


def _execute_remaining(project, output, marker_path, plan_path, state_path, plan, state):
    """Record an interruption as resumable and publish the normal final receipt."""
    _write(marker_path, {'schema_version': SCHEMA_VERSION, 'status': 'running', 'plan_sha256': _hash(plan_path)})
    print(f'FedCAID | 补跑目录 | {output}', flush=True)
    print(f"FedCAID | 已校验并保留的阶段 | {'、'.join(STAGE_LABELS[name] for name in state)}", flush=True)
    try:
        operations = _operations(project, state, _resume_device(output, plan))
        _run_remaining(output, state_path, plan, state, operations)
    except (KeyboardInterrupt, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        _write_abort(marker_path, plan_path, error)
        raise
    return _write_completion(marker_path, plan_path, state_path, state)


def resume_review(project, name):
    """Resume the designated full review without loading another run's evidence."""
    project = Path(project).resolve(strict=True)
    output = _run_path(project, name)
    with _output_lock(output):
        return _continue_locked(project, output)


def _resume_paths(output, data_root):
    """Keep partial outputs and their children inside the declared stage."""
    output, data_root = Path(output).absolute(), Path(data_root).resolve()
    if output.resolve() != output or output.is_relative_to(data_root) or data_root.is_relative_to(output):
        raise ValueError('Supplementary output must be an unredirected path separate from the raw dataset')
    if any(path.resolve() != path for path in output.rglob('*')):
        raise ValueError(f'Redirected supplementary artifact in {output}')
    if (output / matrix.STAGE_RECEIPT).exists():
        _verify_published_bindings(output)
    return output, data_root


def _verify_published_bindings(output):
    """Reject changes to published evidence before native steps can run."""
    bindings = _read(output / matrix.STAGE_RECEIPT)['artifacts']
    for location, digest in bindings.items():
        path = output / location
        if path.resolve() != path or not path.is_relative_to(output) or not path.is_file() or _hash(path) != digest:
            raise ValueError(f'Existing supplementary evidence changed: {path}')


def _case_paths(output, phase, cases):
    """Map each declared case to the atomic native result location."""
    return {output / case[0] / phase.protocol / case[1] / f'seed_{case[-1]}' / 'result.json': case for case in cases}


def _validate_case(path, phase, case, device, smoke):
    """Check the identity, settings and completeness of an atomic case result."""
    record = _read(path)
    if record['protocol'] != phase.protocol or matrix._case_key(record) != case:
        raise ValueError(f'Saved supplementary case identity differs at {path}')
    settings = record['settings']
    if settings['device'] != device or record.get('prediction_cache') is not None:
        raise ValueError(f'Saved supplementary case used different runtime or cache inputs: {path}')
    matrix._validate_supplement_case(settings, path.parents[2], phase, smoke)
    required = {(role, mode) for role in matrix.TEST_ROLES for mode in phase.modes}
    evaluations = [(row['role'], row['mode']) for row in record['evaluations']]
    if (
        not required.issubset(evaluations)
        or len(evaluations) != len(set(evaluations))
        or any(role not in matrix.TEST_ROLES for role, _ in evaluations)
    ):
        raise ValueError(f'Saved supplementary case has incomplete or duplicate evaluations: {path}')


def _verify_partial_phase(output, data_root, device, seeds, smoke, phase, cases):
    """Reject wrong or incomplete saved case files before entering native resume."""
    saved_plan = _read(output / 'run.json')
    expected_matrix = [
        {'dataset': name, 'protocol': phase.protocol, 'scenarios': [[phase.scenario, 3, phase.alpha]]}
        for name in phase.datasets
    ]
    coordinates = (saved_plan['matrix'], saved_plan['seeds'], saved_plan['synthetic_only'], saved_plan['data_root'])
    saved_device = saved_plan['device']
    if (
        coordinates != (expected_matrix, list(seeds), smoke, None if smoke else str(data_root))
        or saved_plan['supplement']['profile'] != phase.profile
        or saved_plan['source_sha256'] != native._source_hashes(supplement=True)
        or saved_device not in {'cpu', 'cuda'}
        or device not in {'auto', saved_device}
    ):
        raise ValueError(f'Saved supplementary plan differs from this run: {output}')
    _verify_saved_cases(output, phase, cases, saved_device, smoke)
    return saved_device


def _verify_saved_cases(output, phase, cases, device, smoke):
    """Validate the declared file set and require a source audit for every result."""
    paths = _case_paths(output, phase, cases)
    for path in output.rglob('result.json'):
        if path not in paths:
            raise ValueError(f'Saved supplementary case is outside the declared workload: {path}')
        if not (path.parents[2] / 'data_audit.json').is_file():
            raise ValueError(f'Saved supplementary case lacks its original dataset audit: {path}')
        _validate_case(path, phase, paths[path], device, smoke)


def _resume_native(output, data_root, device, seeds, smoke, phase):
    """Use the native runner's exact-plan and dataset-audit resume checks."""
    arguments = [
        '--profile',
        phase.profile,
        '--dataset',
        *phase.datasets,
        '--no-tcp',
        '--no-figures',
        '--resume',
        *phase.native_flags,
        *matrix._common_arguments(output, data_root, device, seeds, smoke),
    ]
    if native.main(arguments, supplement=True) != 0:
        raise RuntimeError(f'Native supplement could not resume: {phase.profile}')


def _clear_uncommitted_phase(output):
    """Release only an empty phase that stopped before publishing its plan."""
    partial = output / 'run.partial'
    entries = list(output.iterdir())
    if entries:
        if entries != [partial] or not partial.is_file() or partial.resolve() != partial:
            raise ValueError(f'Native phase has evidence but no committed run.json: {output}')
        partial.unlink()
    output.rmdir()


def _resume_phase(output, data_root, device, seeds, smoke, phase):
    """Train missing cases and bind the same evidence as the fresh runner."""
    if output.is_dir() and not (output / 'run.json').exists():
        _clear_uncommitted_phase(output)
    if not output.exists():
        return matrix._run_supplement(output, data_root, device, seeds, smoke, phase)
    cases = matrix._supplement_cases(phase, seeds)
    device = _verify_partial_phase(output, data_root, device, seeds, smoke, phase, cases)
    if not (output / 'complete.json').exists():
        _resume_native(output, data_root, device, seeds, smoke, phase)
    matrix._validate_supplement(output, data_root, phase, seeds, cases, smoke)
    matrix._checked_metrics(
        output, 'metrics_by_seed.csv', cases, phase.modes, matrix.TEST_ROLES, protocol=phase.protocol
    )
    return matrix._phase_artifacts(output, phase, cases, smoke)


def _publish_resume(output, names, details):
    """Publish a new stage receipt or preserve an already finished, verified one."""
    marker = output / matrix.STAGE_RECEIPT
    if not marker.exists():
        return matrix._publish(output, names, details)
    receipt = _read(marker)
    expected = {
        **details,
        'datasets': list(details['datasets']),
        'seeds': list(details['seeds']),
        'status': 'completed',
        'output': str(output),
        'execution_origin': 'current_fresh_training',
        'independent_confirmation': False,
        'artifacts': _bindings(output, names),
    }
    if receipt != expected:
        raise ValueError(f'Existing supplementary stage receipt differs from its artifacts: {marker}')
    return {**receipt, 'artifacts': {**receipt['artifacts'], matrix.STAGE_RECEIPT: _hash(marker)}}


def resume_can_quantization(output, data_root, device='auto', *, smoke=False):
    """Resume the grouped and frozen quantization phases, reusing only this run's cases."""
    output, data_root = _resume_paths(output, data_root)
    seeds = native.SEEDS[:1] if smoke else native.SEEDS
    names = []
    for phase in matrix.CAN_PHASES:
        phase_output = output / phase.protocol
        names.extend(
            f'{phase.protocol}/{name}' for name in _resume_phase(phase_output, data_root, device, seeds, smoke, phase)
        )
    details = {
        'datasets': matrix.CAN_DATASETS,
        'protocols': [phase.protocol for phase in matrix.CAN_PHASES],
        'seeds': seeds,
        'cases': len(matrix.CAN_PHASES) * len(matrix.CAN_DATASETS) * len(seeds),
        'synthetic_only': smoke,
        'test_evaluated': True,
        'scientific_status': 'same_source_kminus1_quantization_diagnostic',
        'secure_scales': [10**6, 10**9],
        'evaluation_scope': 'Retrospective grouped and frozen CAN VAL and benchmark TEST, original K−1 adapters',
    }
    return _publish_resume(output, names, details)


def resume_nb15_temporal(output, data_root, device='auto', *, smoke=False):
    """Continue the exact-purged NB15 later-day holdout."""
    output, data_root = _resume_paths(output, data_root)
    seeds = native.SEEDS[:1] if smoke else native.SEEDS
    phase = matrix.NB15_TEMPORAL
    names = _resume_phase(output, data_root, device, seeds, smoke, phase)
    details = {
        'datasets': phase.datasets,
        'protocols': ['temporal_purged'],
        'seeds': seeds,
        'cases': len(seeds),
        'synthetic_only': smoke,
        'test_evaluated': True,
        'scientific_status': 'same_source_exact_purged_temporal_test',
        'evaluation_scope': 'Retrospective later-day TEST within the same pinned NB15 capture',
    }
    return _publish_resume(output, names, details)


def resume_kminus1_transfer(output, data_root, device='auto', *, smoke=False):
    """Continue the full-TRAIN original K−1 transfer score diagnostics."""
    output, data_root = _resume_paths(output, data_root)
    seeds = native.SEEDS[:1] if smoke else native.SEEDS
    phase = matrix.KMINUS1_TRANSFER
    names = _resume_phase(output, data_root, device, seeds, smoke, phase)
    scores = 'transfer_score_deltas.csv'
    details = {
        'datasets': phase.datasets,
        'protocols': [phase.protocol],
        'seeds': seeds,
        'alpha': phase.alpha,
        'cases': len(phase.datasets) * len(seeds),
        'synthetic_only': smoke,
        'test_evaluated': True,
        'eligible_score_contrast_rows': len(list(matrix._rows(output, scores))) if scores in names else 0,
        'full_train': True,
        'scientific_status': 'same_source_kminus1_transfer_diagnostic',
        'evaluation_scope': 'Retrospective exact-unseen benchmark TEST; original K−1 missing-class transfer strata',
    }
    return _publish_resume(output, names, details)
