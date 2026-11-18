"""Verify and summarize the archived synthetic-update loopback benchmark."""

import csv
import math
import re
from ipaddress import ip_address
from pathlib import Path, PurePosixPath
from shutil import copyfile
from statistics import median, quantiles

from .revision_v12 import _hash, _read, _write

CLIENTS = (3, 5, 10)
DIMENSIONS = (1035, 690)
ROUNDS = 30
SUITE_FILES = ('suite_complete.json', 'suite_plan.json', 'suite_state.json')
MEASUREMENTS = ('protocol_rounds.csv', 'protocol_summary.json')
LOOPBACK = 'same_host_loopback'
SUMMARY_TABLE = 'local_transport_summary.csv'
OUTPUT_RECEIPT = 'local_transport_receipt.json'
ARCHIVE_RELATIVE = Path('outputs/v12_review_suite')
METHOD_SOURCES = ('revision_protocol_validation.py', 'secure_aggregation.py')
NO_RECORDED_DIGEST = object()


def _bound_file(root, name, expected=NO_RECORDED_DIGEST):
    """Check a regular archive artifact and its recorded digest."""
    path = root / name
    if not path.is_relative_to(root) or path.resolve() != path or not path.is_file():
        raise ValueError(f'Missing or redirected loopback archive artifact: {name}')
    digest = _hash(path)
    if expected is not NO_RECORDED_DIGEST and digest != expected:
        raise ValueError(f'Loopback archive artifact changed: {name}')
    return digest


def _suite_receipts(root):
    """Read source files through the original completion receipt."""
    complete, plan, state = SUITE_FILES
    bindings = {complete: _bound_file(root, complete)}
    marker = _read(root / complete)
    if marker['schema_version'] != 1 or marker['status'] not in {'completed', 'completed_with_pending'}:
        raise ValueError('Loopback archive suite has no compatible completion receipt')
    bindings[plan] = _bound_file(root, plan, marker['plan_sha256'])
    bindings[state] = _bound_file(root, state, marker['state_sha256'])
    plan, state = _read(root / plan), _read(root / state)
    expected_design = (list(CLIENTS), list(DIMENSIONS), ROUNDS)
    design = tuple(plan[key] for key in ('protocol_clients', 'protocol_dimensions', 'protocol_rounds'))
    if plan['schema_version'] != 1 or design != expected_design:
        raise ValueError('Loopback archive has a different scale matrix')
    return marker['steps'], plan, state, bindings


def _directory_name(original):
    """Use only the archived output's final path component on any host."""
    return PurePosixPath(original.replace('\\', '/')).name


def _measurement_bindings(root, stages, state, key, expected_name):
    """Require a completed stage and verify its original measurement receipt."""
    receipt = state[key]
    summary = {field: value for field, value in receipt.items() if field != 'artifacts'}
    if summary != stages[key] or receipt['status'] != 'completed':
        raise ValueError(f'Loopback archive stage is incomplete: {key}')
    path = _directory_name(receipt['output'])
    if expected_name.fullmatch(path) is None or set(receipt['artifacts']) != set(MEASUREMENTS):
        raise ValueError(f'Unexpected loopback archive stage path or inventory: {key}')
    digests = {f'{path}/{name}': _bound_file(root, f'{path}/{name}', sha) for name, sha in receipt['artifacts'].items()}
    return path, digests


def _rows(path):
    """Read the original round table without modifying the archive."""
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def _measurement_metadata(summary, dimension):
    """Require protocol-only and same-host measurement labels."""
    expected = {
        'schema_version': 1,
        'complete': True,
        'clients': list(CLIENTS),
        'dimension': dimension,
        'rounds_per_condition': ROUNDS,
        'transport_environment': LOOPBACK,
        'independent_physical_hosts_verified': False,
        'detector_training_measured': False,
        'startup_in_round_times': False,
        'tls': False,
        'authentication': False,
        'differential_privacy': False,
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        raise ValueError(f'Loopback measurement scope or design differs for dimension {dimension}')
    endpoints = summary['endpoints']
    if len(set(endpoints)) != 2 or len(endpoints) != 2:
        raise ValueError('Loopback measurement requires two distinct collector endpoints')
    for endpoint in endpoints:
        address, port = endpoint.rsplit(':', 1)
        if not ip_address(address).is_loopback or not 1 <= int(port) <= 65535:
            raise ValueError('Benchmark does not use two loopback collectors')


def _round_observations(row, summary, clients):
    """Validate bench workload identity, roster completion and byte accounting."""
    elapsed = float(row['elapsed_seconds'])
    sharing = float(row['quantize_and_share_seconds'])
    uploaded, downloaded, transferred = (
        int(row[f'tcp_application_{part}_bytes']) for part in ('upload', 'download', 'total')
    )
    logical = int(row['numeric_share_payload_bytes'])
    if (
        not math.isfinite(elapsed)
        or not 0 <= sharing <= elapsed
        or elapsed <= 0
        or min(uploaded, downloaded) <= 0
        or uploaded + downloaded != transferred
        or int(row['dimension']) != summary['dimension']
        or int(row['seed']) != summary['seed']
        or float(row['clip']) != summary['clip']
        or int(row['scale']) != summary['scale']
        or logical != (2 * clients + 2) * summary['dimension'] * 8
        or row['plaintext_quantized_sum_equal'] != 'True'
        or row['all_clients_accepted'] != 'True'
    ):
        raise ValueError(f"Invalid or incomplete loopback round: {clients} clients, round {row['round']}")
    return elapsed, uploaded, downloaded, transferred, logical


def _condition_rows(summary, rows, clients):
    selected = [row for row in rows if int(row['clients']) == clients]
    if sorted(int(row['round']) for row in selected) != list(range(ROUNDS)):
        raise ValueError(f'Loopback rounds missing or duplicated for {clients} clients')
    times, upload, download, total, logical = zip(
        *(_round_observations(row, summary, clients) for row in selected), strict=True
    )
    return (
        {
            'clients': clients,
            'measured_rounds': len(times),
            'round_median_seconds': median(times),
            'round_p95_seconds': quantiles(times, n=20, method='inclusive')[-1],
            'tcp_application_total_bytes': sum(total),
            'all_plaintext_quantized_sums_equal': True,
        },
        (sum(upload), sum(download), logical[0]),
    )


def _check_condition(recomputed, recorded):
    """Cross-check sums and linear percentiles against the native summary."""
    timings = {'round_median_seconds', 'round_p95_seconds'}
    if recomputed.keys() != recorded.keys() or any(
        not math.isclose(value, recorded[key], rel_tol=1e-12) if key in timings else value != recorded[key]
        for key, value in recomputed.items()
    ):
        raise ValueError('Recomputed loopback round statistics differ from the native summary')


def _display_condition(dimension, summary, bytes_by_direction):
    uploaded, downloaded, logical = bytes_by_direction
    return {
        'dimension': dimension,
        'clients': summary['clients'],
        'measured_rounds': summary['measured_rounds'],
        'round_median_ms': 1000 * summary['round_median_seconds'],
        'round_p95_ms': 1000 * summary['round_p95_seconds'],
        'numeric_share_payload_bytes_per_round': logical,
        'tcp_application_mean_bytes_per_round': (summary['tcp_application_total_bytes'] / summary['measured_rounds']),
        'tcp_application_total_bytes': summary['tcp_application_total_bytes'],
        'tcp_application_upload_total_bytes': uploaded,
        'tcp_application_download_total_bytes': downloaded,
        'all_plaintext_quantized_sums_equal': summary['all_plaintext_quantized_sums_equal'],
        'all_clients_accepted': True,
        'transport_environment': LOOPBACK,
    }


def summarize_loopback_dimension(base, dimension, *, workload=None):
    """Validate and summarize one native same-host protocol measurement."""
    summary = _read(base / MEASUREMENTS[1])
    _measurement_metadata(summary, dimension)
    if workload is not None and any(summary.get(key) != value for key, value in workload.items()):
        raise ValueError(f'Loopback measurement workload differs for dimension {dimension}')
    rows = _rows(base / MEASUREMENTS[0])
    if len(rows) != len(CLIENTS) * ROUNDS:
        raise ValueError(f'Unexpected loopback row count for dimension {dimension}')
    published = []
    for count, recorded in zip(CLIENTS, summary['conditions'], strict=True):
        calculated, byte_counts = _condition_rows(summary, rows, count)
        _check_condition(calculated, recorded)
        published.append(_display_condition(dimension, calculated, byte_counts))
    return rows, published, [{'dimension': dimension, **item} for item in summary['conditions']]


def _verify_index(path, expected_bindings, measurements, conditions):
    index = _read(path / MEASUREMENTS[1])
    expected = {
        'schema_version': 1,
        'complete': True,
        'transport_environment': LOOPBACK,
        'independent_physical_hosts_verified': False,
        'detector_training_measured': False,
        'conditions': conditions,
    }
    if any(index.get(key) != value for key, value in expected.items()):
        raise ValueError('Loopback archive index differs from its source summaries')
    indexed = {_directory_name(name): sha for name, sha in index['source_measurements'].items()}
    if (
        indexed != expected_bindings
        or len(indexed) != len(index['source_measurements'])
        or _rows(path / MEASUREMENTS[0]) != measurements
    ):
        raise ValueError('Loopback archive index differs from its source measurements')


def _read_archive(root):
    """Verify the suite, the two dimensions and their combined source index."""
    steps, plan, state, bindings = _suite_receipts(root)
    rows, summaries, conditions, indexed = [], [], [], {}
    for dimension in DIMENSIONS:
        key = f'protocol_{dimension}'
        name, digests = _measurement_bindings(root, steps, state, key, re.compile(key))
        if state[key]['dimension'] != dimension or state[key]['transport_environment'] != LOOPBACK:
            raise ValueError(f'Loopback suite stage scope differs for dimension {dimension}')
        bindings.update(digests)
        indexed[name] = {file: digests[f'{name}/{file}'] for file in MEASUREMENTS}
        raw, report, original = summarize_loopback_dimension(root / name, dimension)
        rows.extend(raw)
        summaries.extend(report)
        conditions.extend(original)
    name, digests = _measurement_bindings(root, steps, state, 'protocol', re.compile(r'protocol_index_[0-9a-f]{12}'))
    _verify_index(root / name, indexed, rows, conditions)
    source = {name: plan['source_sha256'][name] for name in METHOD_SOURCES}
    if any(re.fullmatch(r'[0-9a-f]{64}', digest) is None for digest in source.values()):
        raise ValueError('Invalid archived protocol implementation digests')
    return summaries, {**bindings, **digests}, source


def _copy_sources(archive, output, bindings):
    """Preserve the exact validated lightweight synthetic measurement files."""
    copied = {}
    for name, digest in bindings.items():
        relative = f'archive/{name}'
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        copyfile(archive / name, target)
        _bound_file(output, relative, digest)
        copied[relative] = digest
    return copied


def _transport_receipt(output, archive, rows, sources, bindings):
    """Describe what the archived measurements exercise and what is outside scope."""
    return {
        'schema_version': 1,
        'status': 'completed',
        'output': str(output),
        'archive': str(archive),
        'evidence_origin': 'verified_archived_protocol_measurement',
        'measurements_reexecuted': False,
        'transport_environment': LOOPBACK,
        'independent_physical_hosts_verified': False,
        'synthetic_updates': True,
        'hardware_metadata_recorded': False,
        'detector_training_measured': False,
        'inference_measured': False,
        'startup_in_round_times': False,
        'rounds_measured': sum(row['measured_rounds'] for row in rows),
        'all_plaintext_quantized_sums_equal': all(row['all_plaintext_quantized_sums_equal'] for row in rows),
        'scope': (
            'Synthetic updates exchanged on one host; time includes clipping, sharing, TCP exchanges and' ' check'
        ),
        'integrity_scope': (
            'Original local suite receipt and index digests checked; acquisition not independently attested'
        ),
        'security_scope': ('Trusted orchestrator, noncolluding honest-but-curious collectors; quantized sums checked'),
        'security_exclusions': [
            'TLS',
            'authentication',
            'DP',
            'malicious-party protection',
            'dropout recovery',
        ],
        'numeric_payload_scope': 'uint64 client shares and collector aggregate shares; framing excluded',
        'tcp_application_byte_scope': ('JSON and 4-byte frame prefixes; excludes TCP/IP overhead and retransmission'),
        'archive_source_sha256': sources,
        'archive_artifact_sha256': bindings,
        'verifier_source_sha256': _hash(Path(__file__)),
    }


def run_local_transport(output, project, *, archive=None):
    """Import bound loopback evidence without starting collectors or training."""
    output = Path(output).resolve()
    archive = Path(archive or Path(project) / ARCHIVE_RELATIVE).resolve()
    if not archive.exists():
        return {
            'status': 'pending',
            'reason': 'Archived same-host transport measurements are missing',
            'output': str(output),
            'archive': str(archive),
            'artifacts': {},
        }
    if output.is_relative_to(archive) or archive.is_relative_to(output):
        raise ValueError('Loopback archive and new evidence output must be separate')
    rows, bindings, sources = _read_archive(archive)
    output.mkdir(parents=True, exist_ok=False)
    copied = _copy_sources(archive, output, bindings)
    with (output / SUMMARY_TABLE).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    receipt = _transport_receipt(output, archive, rows, sources, bindings)
    _write(output / OUTPUT_RECEIPT, receipt)
    artifacts = {**copied, **{name: _bound_file(output, name) for name in (OUTPUT_RECEIPT, SUMMARY_TABLE)}}
    return {**receipt, 'artifacts': artifacts}
