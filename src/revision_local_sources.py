"""Bind official local evidence and separate historical supplements for reporting."""

import csv
import json
import re
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from .revision_fresh_report import (
    _artifact_path,
    _copy_table,
    _refresh_receipts,
    _stage_root,
    _verified_files,
    regenerate_verified_report,
)
from .revision_v12 import _hash, _read, _write

OFFICIAL_NAME = re.compile(r'run_[0-9]{8}_[0-9]{6}_[a-z0-9]+')
LEGACY_REPORT_NAME = re.compile(r'report_[a-f0-9]{16}')
WORKFLOWS = ('full_review', 'iov_diagnosis')
SEED_SET = {42, 43, 44, 45, 46}
BENCHMARK_DATASETS = {'CIC-IDS-2017', 'CIC-UNSW-NB15', 'CIC-IoV-2024', 'Car-Hacking'}
FULL_CORE = {'nb15_pair', 'targeted', 'donor_check', 'hulk_attribution', 'benchmark_matrix'}
FINAL_STATUSES = {'completed', 'completed_with_pending', 'completed_with_failures'}
INCOMPLETE_STATUSES = {'running', 'interrupted', 'aborted'}

SCALE_CAN_ROOT = Path('v12_final_evidence/report')
SCALE_CAN_PREFIXES = ('scale_alpha_', 'can_')
SCALE_CAN_REQUIRED = {
    'can_support_limits.csv',
    'can_quantization_diagnostics.csv',
    'scale_alpha_actual_workloads.csv',
    'scale_alpha_paired_intervals.csv',
    'scale_alpha_timing_summary.csv',
    'scale_alpha_communication.csv',
}
TEMPORAL_ROOT = Path('revision_v5_suite_20261112_104859_a09ad0')
TEMPORAL_AUDIT = 'temporal_purged/CIC-UNSW-NB15/temporal/data_audit.json'
# The v5 suite saved completion metadata without an artifact digest inventory.
# These are current, locally checked pins for this one historical supplement.
TEMPORAL_PINS = {
    'suite.json': '8b8282ae1c96c0392a83412c30585a4ebfd633922e6aa0cb7c814af23fe65a1c',
    'suite_complete.json': 'c978c25ba4120d10604f6d64ebae26b2157d22c343670b21dbd24c85a2f64dee',
    'progress.json': '1162a475d9b3521c68f761f6d54a01ea0f61269bf9c3186cfd0bf3021236b7af',
    'temporal_purged/run.json': '9f73330a7d103d64d528688c6021a0ede4368315732df98bddbb8dcf8b7242ae',
    'temporal_purged/complete.json': '70a1e39c96ac19a7266b12c3d618ca654b7aebf22f35507c21ad62bf1963ec5d',
    TEMPORAL_AUDIT: '98660399d1913fa22f6abed0de150078b5467ff9a88edf065f9043d546f2cacd',
    'temporal_purged/metrics_by_seed.csv': '32f33cfc1223acdf84ae8e017f583bf3485b86acd85f31e4e1977d522e174360',
    'temporal_purged/paired_differences.csv': ('05f345568b9f0f6e53be8545e209d7c3f50a7eeb66ac456d85370201f9a0d804'),
    'temporal_purged/timings.csv': 'f10ee1c1731b096f58b65f587b44284de2008630be49414540833fd1dbc544ff',
    'temporal_purged/resources.csv': 'cf5b914e6b884ae00015ca2969d3b229e5a9a77b4d87084bf9bf703d826c6e1c',
}


def _csv_rows(path):
    """Read the confirmed archive inventory without modifying its tables."""
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def _digest_checks(root, bindings):
    """Enforce every declared digest before a source is eligible for copying."""
    for _ in _verified_files(root, bindings):
        pass


def _status_at(run):
    """Skip in-progress runs and reject missing receipts after a completed report."""
    receipt = _artifact_path(run, 'final_complete.json')
    if receipt.is_file():
        row = _read(receipt)
        if row.get('status') in INCOMPLETE_STATUSES:
            if 'state_sha256' in row or 'steps' in row:
                raise ValueError(f'Invalid status for a finalized official run in {receipt}')
            _digest_checks(run, {'final_plan.json': row['plan_sha256']})
            return row
        if row.get('status') in FINAL_STATUSES:
            return row
        raise ValueError(f'Unrecognized official run status in {receipt}')
    state = _artifact_path(run, 'final_state.json')
    if (run / 'report/complete.json').is_file() or (
        state.is_file() and _read(state).get('report', {}).get('status') == 'completed'
    ):
        raise ValueError(f'Missing final receipt after a completed official report in {run}')
    return {}


def _run_kind(plan, stages):
    """Recognize the explicit workflow or the genuine schema-five full matrix."""
    workflow = plan.get('workflow')
    if workflow in WORKFLOWS:
        return workflow
    if workflow is None and plan.get('schema_version') == 5 and FULL_CORE <= set(stages):
        return 'full_review'
    return None


def _complete_full(stages):
    """Require every local experiment plus the genuine four-by-five matrix."""
    if not FULL_CORE <= set(stages) or any(stages[name]['status'] != 'completed' for name in FULL_CORE):
        return False
    matrix = stages['benchmark_matrix']
    return (
        matrix.get('execution_origin') == 'current_fresh_training'
        and matrix.get('test_evaluated') is False
        and set(matrix.get('datasets', [])) == BENCHMARK_DATASETS
        and matrix.get('cases_per_dataset') == dict.fromkeys(BENCHMARK_DATASETS, len(SEED_SET))
    )


def _complete_iov(stages):
    """Require exactly the genuine five-seed working-point diagnosis."""
    if set(stages) != {'iov_diagnosis'}:
        return False
    row = stages['iov_diagnosis']
    return (
        row.get('status') == 'completed'
        and row.get('execution_origin') == 'current_fresh_training'
        and row.get('synthetic_only') is False
        and row.get('test_evaluated') is False
        and row.get('threshold_candidate_evaluated') is False
        and set(row.get('seeds', [])) == SEED_SET
    )


def _official_kind(run):
    """Verify the control receipts before allowing fallback or selecting a report."""
    completion = _status_at(run)
    if completion.get('status') not in FINAL_STATUSES:
        return None
    if completion['status'] == 'completed_with_failures':
        _verify_failed_run(run, completion)
        return None
    stages, bindings, _ = _refresh_receipts(run)
    plan_path = _artifact_path(run, 'final_plan.json')
    plan = _read(plan_path)
    _digest_checks(run, bindings)
    kind = _run_kind(plan, stages)
    if not kind or plan.get('synthetic_only') is True:
        return None
    if any(item.get('previous_run_inputs_allowed') is not False for item in (completion, plan)):
        raise ValueError(f'Official workflow permits previous-run input in {run}')
    if not (_complete_full(stages) if kind == 'full_review' else _complete_iov(stages)):
        raise ValueError(f'Official run lacks its completed real-data workflow in {run}')
    for name, record in stages.items():
        if record['status'] == 'completed':
            _digest_checks(_stage_root(run, name, record), record['artifacts'])
    return kind


def _verify_failed_run(run, completion):
    """Accept a declared failure as ineligible after validating its receipts."""
    paths = {
        'final_plan.json': completion['plan_sha256'],
        'final_state.json': completion['state_sha256'],
    }
    plan, state = [_read(path) for _, path, _ in _verified_files(run, paths)]
    failed = {name for name, record in state.items() if record['status'] == 'failed'}
    steps = {name: {k: v for k, v in row.items() if k != 'artifacts'} for name, row in state.items()}
    if not failed or failed != set(completion['failed_steps']) or steps != completion['steps']:
        raise ValueError(f'Invalid failed official run receipt in {run}')
    if set(plan['stages']) != set(state) or any(Path(row['run_directory']) != run for row in (completion, plan)):
        raise ValueError(f'Invalid failed official run plan in {run}')


def discover_official_runs(outputs_root):
    """Return the newest finalized, verified real run by workflow."""
    outputs = Path(outputs_root).resolve()
    parent = _artifact_path(outputs, 'fedcaid_review')
    if not parent.is_dir():
        return {}
    selected = {}
    for run in sorted(parent.iterdir(), key=lambda path: path.name, reverse=True):
        if not OFFICIAL_NAME.fullmatch(run.name):
            continue
        if run.resolve() != run or not run.is_dir():
            raise ValueError(f'Redirected official run directory {run}')
        kind = _official_kind(run)
        if kind and kind not in selected:
            selected[kind] = run
            if set(selected) == set(WORKFLOWS):
                break
    return selected


def copy_official_reports(outputs_root, destination):
    """Refresh only reports from selected sources into a local evidence workspace."""
    selected = discover_official_runs(outputs_root)
    destination = Path(destination).absolute()
    result = {}
    for workflow in WORKFLOWS:
        run = selected.get(workflow)
        if run is None:
            result[workflow] = {'status': 'unavailable', 'reason': 'No finalized official real-data run'}
            continue
        report = regenerate_verified_report(run, destination / workflow)
        provenance = report['refresh_source']
        result[workflow] = {
            'status': 'completed',
            'source_run': run.name,
            'source_run_directory': str(run),
            'source_receipt_sha256': provenance['source_receipt_sha256'],
            'trained_cases': 0,
        }
    return result


def _scale_can_receipts(root):
    """Verify the named legacy snapshot, completion, plan and output inventory."""
    pointer = _read(_artifact_path(root, 'latest.json'))
    snapshot = Path(pointer['report_path'])
    if snapshot.parent != root or not LEGACY_REPORT_NAME.fullmatch(snapshot.name) or snapshot.resolve() != snapshot:
        raise ValueError('Invalid legacy CAN and scale report path')
    marker = _artifact_path(snapshot, 'complete.json')
    marker_sha = _hash(marker)
    receipt = _read(marker)
    if marker_sha != pointer['receipt_sha256'] or receipt.get('status') != 'completed':
        raise ValueError('Invalid legacy CAN and scale completion receipt')
    bindings = receipt['artifacts']
    if pointer['artifacts'] != {**bindings, 'complete.json': marker_sha} or 'evidence_plan.json' not in bindings:
        raise ValueError('Legacy CAN and scale archive inventories differ')
    _digest_checks(snapshot, bindings)
    return snapshot, receipt, marker_sha


def _legacy_native_inputs(outputs, snapshot, receipt):
    """Check every archived native data input while preserving old source-code fingerprints."""
    plan = _read(snapshot / 'evidence_plan.json')
    digest = sha256(json.dumps(plan, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if receipt['source_digest'] != digest or snapshot.name != f'report_{digest[:16]}' or plan['missing_sources']:
        raise ValueError('Legacy CAN and scale source plan is incomplete or changed')
    sources, historical_code = {}, {}
    for name, expected in plan['source_hashes'].items():
        path = Path(name)
        if path.is_relative_to(outputs):
            sources[path.relative_to(outputs).as_posix()] = expected
        elif path.is_relative_to(outputs.parent / 'src'):
            historical_code[str(path)] = expected
        else:
            raise ValueError(f'Unexpected legacy CAN and scale source: {path}')
    if not sources:
        raise ValueError('Legacy CAN and scale plan has no native evidence')
    _digest_checks(outputs, sources)
    return sources, historical_code


def _publish_archive(source, target, bindings, provenance):
    """Write a new verified archive copy without altering source files."""
    if target.exists() or target.resolve() != target or target.is_relative_to(source):
        raise ValueError('Historical evidence requires a new, non-redirected destination outside the source')
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.local-evidence-', dir=target.parent) as temporary:
        working = Path(temporary) / 'archive'
        artifact_hashes = {}
        for name, path, expected in _verified_files(source, bindings):
            destination = Path(name).name
            if destination in artifact_hashes:
                raise ValueError(f'Duplicate local evidence filename: {destination}')
            _copy_table(path, Path(destination), expected, working)
            artifact_hashes[destination] = expected
        _write(working / 'archive_provenance.json', {**provenance, 'copied_artifact_sha256': artifact_hashes})
        working.rename(target)


def _scale_can_archive(outputs, target):
    """Copy verified legacy measurements and support into a distinct evidence slot."""
    root = _artifact_path(outputs, str(SCALE_CAN_ROOT))
    if not root.is_dir():
        return {'status': 'unavailable', 'reason': 'Missing historical CAN and scale report'}
    snapshot, receipt, marker_sha = _scale_can_receipts(root)
    sources, code = _legacy_native_inputs(outputs, snapshot, receipt)
    copied = {
        name: expected
        for name, expected in receipt['artifacts'].items()
        if name.startswith(SCALE_CAN_PREFIXES) and name.endswith('.csv')
    }
    if not SCALE_CAN_REQUIRED <= set(copied):
        raise ValueError('Legacy CAN and scale report is missing expected evidence tables')
    provenance = {
        'status': 'completed',
        'evidence_origin': 'verified_historical_report',
        'evidence_scope': 'legacy CAN adapter and quantization, single-host scale and heterogeneity',
        'source_run_directory': str(snapshot),
        'source_receipt_sha256': marker_sha,
        'source_plan_sha256': receipt['artifacts']['evidence_plan.json'],
    }
    _publish_archive(
        snapshot,
        target,
        copied,
        {
            **provenance,
            'verified_native_artifact_sha256': sources,
            'archived_source_code_sha256': code,
        },
    )
    return provenance


def _validate_temporal_suite(source):
    """Confirm the historical NB15 job was declared and completed on real data."""
    suite = _read(source / 'suite.json')
    suite_complete = _read(source / 'suite_complete.json')
    progress = _read(source / 'progress.json')
    jobs = [job for job in suite['jobs'] if job.get('label') == 'temporal_purged']
    if (
        suite.get('synthetic_only') is not False
        or set(suite.get('seeds', [])) != SEED_SET
        or len(jobs) != 1
        or jobs[0].get('datasets') != ['CIC-UNSW-NB15']
        or not suite_complete['declared_task_complete']
        or suite_complete['synthetic_only'] is not False
        or 'temporal_purged' not in progress['completed_jobs']
    ):
        raise ValueError('Historical NB15 temporal job is not a complete real-data supplement')


def _validate_temporal_rows(source):
    """Require the declared five matched seeds and audited evaluation roles."""
    run = _read(source / 'temporal_purged/run.json')
    complete = _read(source / 'temporal_purged/complete.json')
    audit = _read(source / TEMPORAL_AUDIT)
    if (
        set(run['seeds']) != SEED_SET
        or run['synthetic_only'] is not False
        or complete != {'replicates': len(SEED_SET), 'data_audits': 1}
        or audit['external_captures'] is not False
        or audit['temporal_exact_purge']['enabled'] is not True
    ):
        raise ValueError('Historical NB15 temporal settings differ from the declared scope')
    rows = _csv_rows(source / 'temporal_purged/metrics_by_seed.csv')
    _validate_temporal_inventory(rows)


def _validate_temporal_inventory(rows):
    """Refuse incomplete or cross-dataset metrics under the archival label."""
    roles = {'val', 'test', 'test_unique_unseen'}
    modes = {'base_threshold', 'local_only', 'full'}
    observed = []
    for row in rows:
        if (row['dataset'], row['protocol'], row['scenario']) != (
            'CIC-UNSW-NB15',
            'temporal',
            'supplement_temporal',
        ):
            raise ValueError('Unexpected dataset or split in historical temporal results')
        if row['mode'] in modes:
            observed.append((int(row['seed']), row['role'], row['mode']))
    expected = {(seed, role, mode) for seed in SEED_SET for role in roles for mode in modes}
    if set(observed) != expected or len(observed) != len(expected):
        raise ValueError('Incomplete archived temporal metric inventory')


def _temporal_archive(outputs, target):
    """Copy the locally pinned historical NB15 evidence with limited provenance."""
    source = _artifact_path(outputs, str(TEMPORAL_ROOT))
    if not source.is_dir():
        return {'status': 'unavailable', 'reason': 'Missing historical NB15 temporal supplement'}
    _digest_checks(source, TEMPORAL_PINS)
    _validate_temporal_suite(source)
    _validate_temporal_rows(source)
    provenance = {
        'status': 'completed',
        'evidence_origin': 'local_hash_pinned',
        'evidence_scope': 'same-source historical exploration',
        'source_run_directory': str(source / 'temporal_purged'),
        'source_receipt_sha256': TEMPORAL_PINS['temporal_purged/complete.json'],
        'contemporaneous_artifact_hash_inventory': False,
    }
    _publish_archive(source, target, TEMPORAL_PINS, provenance)
    return provenance


def copy_legacy_evidence(outputs_root, destination):
    """Copy only clean historical evidence, leaving current experiments separate."""
    source, destination = Path(outputs_root).resolve(), Path(destination).absolute()
    return {
        'scale_can': _scale_can_archive(source, destination / 'scale_can'),
        'nb15_temporal': _temporal_archive(source, destination / 'nb15_temporal'),
    }
