"""Measure a real two-collector TCP message protocol with synthetic updates.

This is a bounded protocol benchmark, not end-to-end federated training or a
production security transport. The orchestrating host is trusted; collectors
must not collude. There is no TLS, authentication, differential privacy,
malicious-client protection or dropout recovery. Default collectors run on the
same host. Explicit endpoints do not prove separate machines or trust domains.
"""

from __future__ import annotations

import csv
import json
import math
import multiprocessing as mp
import socket
import struct
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from uuid import uuid4

import numpy as np

from .secure_aggregation import (
    PRIME,
    SCHEMA,
    SecureAggregationError,
    SecureContract,
    _clip_quantize,
    _split_quantized,
    _validate_base,
    _validate_contract,
)

MAX_FRAME_BYTES = 8 * 1024 * 1024
MAX_DIMENSION = 16384
MAX_CLIENTS = 1000
HEADER = struct.Struct('!I')
SCOPE = (
    'Synthetic-update two-collector TCP protocol benchmark; trusted orchestrator, '
    'noncolluding honest-but-curious collectors. No detector training or independent '
    'client hosts are measured. No TLS, authentication, DP, malicious protection '
    'or dropout recovery. Byte counts include JSON and length prefixes, not TCP/IP headers.'
)


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


class _CleanDisconnect(EOFError):
    """The peer closed before starting the next frame."""


def _read_exact(connection, size, *, allow_clean_disconnect=False):
    data = bytearray()
    while len(data) < size:
        chunk = connection.recv(size - len(data))
        if not chunk:
            if allow_clean_disconnect and not data:
                raise _CleanDisconnect
            raise EOFError('Peer disconnected before the complete frame')
        data.extend(chunk)
    return bytes(data)


def _send_frame(connection, value):
    payload = _json_bytes(value)
    if not 0 < len(payload) <= MAX_FRAME_BYTES:
        raise SecureAggregationError('frame_size_out_of_bounds')
    connection.sendall(HEADER.pack(len(payload)) + payload)
    return HEADER.size + len(payload)


def _receive_frame(connection):
    size = HEADER.unpack(_read_exact(connection, HEADER.size, allow_clean_disconnect=True))[0]
    if not 0 < size <= MAX_FRAME_BYTES:
        raise SecureAggregationError('frame_size_out_of_bounds')
    value = json.loads(_read_exact(connection, size))
    if not isinstance(value, dict):
        raise SecureAggregationError('invalid_message')
    return value, HEADER.size + size


class _CollectorState:
    """One connection, fixed roster, ordered rounds; only a running share sum."""

    def __init__(self):
        self.base = None
        self.session = None
        self.previous_round = None
        self.active = None
        self.seen = set()
        self.total = None
        self.failed = False

    def process(self, request):
        if self.failed:
            raise SecureAggregationError('session_is_not_open')
        try:
            return self._process(request)
        except (ValueError, TypeError, KeyError, SecureAggregationError) as error:
            self.failed = True
            self.active = None
            self.total = None
            self.seen.clear()
            if isinstance(error, SecureAggregationError):
                raise
            raise SecureAggregationError('invalid_message') from error

    def _validate_timeline(self, contract, base):
        _validate_contract(contract, self.base or base)
        if self.active is not None:
            raise SecureAggregationError('round_already_active')
        if self.session is not None and self.session != contract['session']:
            raise SecureAggregationError('changed_session')
        expected = 0 if self.previous_round is None else self.previous_round + 1
        if contract['round_index'] != expected:
            raise SecureAggregationError('replayed_or_skipped_round')

    def _begin(self, contract):
        base = _contract_base(contract)
        self._validate_timeline(contract, base)
        dimension = base['dimension']
        self.base, self.session = base, contract['session']
        self.active = SecureContract(
            contract['session'],
            contract['round_index'],
            contract['parent_model'],
            tuple(base['clients']),
            dimension,
            contract['clip'],
            contract['scale'],
        )
        self.seen, self.total = set(), [0] * dimension
        return {'ok': True, 'contract_hash': self.active.contract_hash}

    def _share(self, identity, share):
        if not isinstance(identity, str) or identity not in self.active.clients:
            raise SecureAggregationError('unregistered_client')
        if identity in self.seen:
            raise SecureAggregationError('duplicate_client')
        if (
            not isinstance(share, list)
            or len(share) != self.active.dimension
            or any(type(value) is not int or not 0 <= value < PRIME for value in share)
        ):
            raise SecureAggregationError('invalid_share')
        self.total = [(left + right) % PRIME for left, right in zip(self.total, share, strict=True)]
        self.seen.add(identity)
        return {'ok': True, 'count': len(self.seen)}

    def _finish(self):
        if self.seen != set(self.active.clients):
            raise SecureAggregationError('incomplete_roster')
        response = {
            'ok': True,
            'count': len(self.seen),
            'aggregate_share': self.total,
            'contract_hash': self.active.contract_hash,
        }
        self.previous_round = self.active.round_index
        self.active, self.total, self.seen = None, None, set()
        return response

    def _process(self, request):
        operation = request.get('op')
        keys = set(request)
        if operation == 'begin' and keys == {'op', 'contract'}:
            return self._begin(request['contract'])
        if self.active is None:
            raise SecureAggregationError('no_active_round')
        if request.get('contract_hash') != self.active.contract_hash:
            raise SecureAggregationError('stale_or_changed_contract')
        if operation == 'share' and keys == {'op', 'contract_hash', 'client', 'share'}:
            return self._share(request['client'], request['share'])
        if operation == 'finish' and keys == {'op', 'contract_hash'}:
            return self._finish()
        raise SecureAggregationError('invalid_operation')


def _contract_base(contract):
    if not isinstance(contract, dict):
        raise SecureAggregationError('invalid_contract')
    roster, dimension = contract.get('clients'), contract.get('dimension')
    _validate_base(roster, dimension, contract.get('clip'), contract.get('scale'))
    if dimension > MAX_DIMENSION or len(roster) > MAX_CLIENTS:
        raise SecureAggregationError('contract_resource_limit')
    base = {key: contract[key] for key in ('schema', 'clients', 'dimension', 'clip', 'scale', 'prime')}
    if base['schema'] != SCHEMA or base['prime'] != PRIME:
        raise SecureAggregationError('unsupported_contract')
    return base


def _serve_connection(connection, timeout):
    state = _CollectorState()
    connection.settimeout(timeout)
    while True:
        try:
            request, _ = _receive_frame(connection)
            response = state.process(request)
            _send_frame(connection, response)
        except _CleanDisconnect:
            return state.previous_round is not None and state.active is None
        except (EOFError, OSError, ValueError, TypeError, SecureAggregationError):
            # No payloads or share values are logged or echoed on failure.
            try:
                _send_frame(connection, {'ok': False, 'error': 'protocol_session_aborted'})
            except (OSError, ValueError):
                pass
            return False


def _announce_listener(listener, ready, sessions):
    address = listener.getsockname()
    if ready is not None:
        ready.send(address)
        ready.close()
    else:
        print(f'COLLECTOR LISTENING {address[0]}:{address[1]}; sessions={sessions}', flush=True)
        print(SCOPE, flush=True)


class _IncompleteCollectorRun(RuntimeError):
    """The declared session count could not be completed."""


def _accept_sessions(listener, sessions, timeout):
    completed = 0
    for _ in range(sessions):
        try:
            connection, _ = listener.accept()
        except socket.timeout as error:
            raise TimeoutError(f'Collector timed out after completing {completed} of {sessions} sessions') from error
        with connection:
            completed += _serve_connection(connection, timeout)
    if completed != sessions:
        raise _IncompleteCollectorRun(f'Collector completed {completed} of {sessions} sessions')
    return completed


def serve_collector(bind='127.0.0.1', port=0, *, sessions=3, timeout=30.0, ready=None):
    """Return the number of completed sessions or raise for an incomplete declared run.

    Bind to the requested address and port, optionally publishing the actual
    address through ready. Apply timeout to accepts and peer socket operations.
    """
    if type(sessions) is not int or not 1 <= sessions <= 1000:
        raise ValueError('sessions must be between 1 and 1000')
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('Positive finite timeout required')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((bind, port))
        listener.listen(2)
        listener.settimeout(timeout)
        _announce_listener(listener, ready, sessions)
        return _accept_sessions(listener, sessions, timeout)


def _serve_local_collector(sessions, timeout, ready):
    try:
        serve_collector(sessions=sessions, timeout=timeout, ready=ready)
    except (_IncompleteCollectorRun, TimeoutError):
        # The benchmark caller receives the corresponding RPC failure.
        raise SystemExit(1) from None


class _LocalCollectors:
    def __init__(self, sessions, timeout):
        self.sessions, self.timeout = sessions, timeout
        self.processes, self.endpoints = [], []

    def __enter__(self):
        context = mp.get_context('spawn')
        try:
            for _ in range(2):
                self._launch(context)
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self.endpoints

    def _launch(self, context):
        local, child = context.Pipe(duplex=False)
        process = context.Process(
            target=_serve_local_collector,
            kwargs={'sessions': self.sessions, 'timeout': self.timeout, 'ready': child},
            daemon=True,
        )
        try:
            try:
                process.start()
            finally:
                child.close()
            self.processes.append(process)
            if not local.poll(max(30.0, self.timeout)):
                raise TimeoutError('Collector startup timed out')
            self.endpoints.append(local.recv())
        finally:
            local.close()

    def __exit__(self, *_):
        for process in self.processes:
            process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)


def _endpoint(value):
    if isinstance(value, tuple) and len(value) == 2:
        host, port = value
    else:
        host, separator, port = str(value).rpartition(':')
        if not separator:
            raise ValueError('Endpoint must be HOST:PORT')
    port = int(port)
    if not host or not 1 <= port <= 65535:
        raise ValueError('Endpoint requires a host and port 1..65535')
    return str(host), port


@dataclass
class _TCPPair:
    endpoints: tuple
    timeout: float

    def __enter__(self):
        self.connections = []
        try:
            for endpoint in self.endpoints:
                connection = socket.create_connection(endpoint, timeout=self.timeout)
                connection.settimeout(self.timeout)
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.connections.append(connection)
        except Exception:
            self.__exit__(None, None, None)
            raise
        self.upload_bytes = self.download_bytes = self.messages = 0
        self.failed = False
        return self

    def rpc(self, payloads):
        if self.failed:
            raise SecureAggregationError('session_is_not_open')
        try:
            for connection, payload in zip(self.connections, payloads, strict=True):
                self.upload_bytes += _send_frame(connection, payload)
                self.messages += 1
            responses = []
            for connection in self.connections:
                response, size = _receive_frame(connection)
                self.download_bytes += size
                self.messages += 1
                if response.get('ok') is not True:
                    raise SecureAggregationError('collector_rejected_round')
                responses.append(response)
            return responses
        except Exception:
            self.failed = True
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        for connection in getattr(self, 'connections', ()):
            connection.close()


def _submit_updates(transport, contract, updates):
    elapsed = 0.0
    encoded = []
    for index, (identity, update) in enumerate(zip(contract.clients, updates, strict=True), 1):
        started = time.perf_counter()
        value = _clip_quantize(update, contract.dimension, contract.clip, contract.scale)
        shares = _split_quantized(value)
        elapsed += time.perf_counter() - started
        encoded.append(value)
        replies = transport.rpc(
            [
                {'op': 'share', 'contract_hash': contract.contract_hash, 'client': identity, 'share': share}
                for share in shares
            ]
        )
        if any(reply.get('count') != index for reply in replies):
            raise SecureAggregationError('inconsistent_roster_count')
    return elapsed, encoded


def _valid_total(reply, contract):
    values = reply.get('aggregate_share')
    return (
        reply.get('count') == len(contract.clients)
        and reply.get('contract_hash') == contract.contract_hash
        and isinstance(values, list)
        and len(values) == contract.dimension
        and all(type(value) is int and 0 <= value < PRIME for value in values)
    )


def _verify_round_sum(transport, contract, encoded):
    replies = transport.rpc([{'op': 'finish', 'contract_hash': contract.contract_hash}] * 2)
    if not all(_valid_total(reply, contract) for reply in replies):
        raise SecureAggregationError('invalid_aggregate_response')
    parts = (reply['aggregate_share'] for reply in replies)
    field_sum = [(left + right) % PRIME for left, right in zip(*parts, strict=True)]
    signed = [value if value <= PRIME // 2 else value - PRIME for value in field_sum]
    expected = [sum(values) for values in zip(*encoded, strict=True)]
    if signed != expected:
        raise SecureAggregationError('plaintext_equivalence_failed')


def _round_metrics(transport, initial, contract, elapsed, sharing_elapsed, clip, seed):
    uploaded, downloaded = transport.upload_bytes - initial[0], transport.download_bytes - initial[1]
    return {
        'clients': len(contract.clients),
        'dimension': contract.dimension,
        'round': contract.round_index,
        'seed': seed,
        'elapsed_seconds': elapsed,
        'quantize_and_share_seconds': sharing_elapsed,
        'tcp_application_upload_bytes': uploaded,
        'tcp_application_download_bytes': downloaded,
        'tcp_application_total_bytes': uploaded + downloaded,
        'messages': transport.messages - initial[2],
        'numeric_share_payload_bytes': (2 * len(contract.clients) + 2) * contract.dimension * 8,
        'plaintext_quantized_sum_equal': True,
        'all_clients_accepted': True,
        'clip': clip,
        'scale': contract.scale,
    }


def _measure_round(transport, contract, updates, clip, seed):
    initial = transport.upload_bytes, transport.download_bytes, transport.messages
    started = time.perf_counter()
    replies = transport.rpc([{'op': 'begin', 'contract': contract.as_dict()}] * 2)
    if any(reply.get('contract_hash') != contract.contract_hash for reply in replies):
        raise SecureAggregationError('changed_contract_response')
    sharing_elapsed, encoded = _submit_updates(transport, contract, updates)
    _verify_round_sum(transport, contract, encoded)
    elapsed = time.perf_counter() - started
    return _round_metrics(transport, initial, contract, elapsed, sharing_elapsed, clip, seed)


def _measure(endpoints, clients, dimension, rounds, seed, timeout, clip, scale):
    rows = []
    for count in clients:
        roster = tuple(f'client_{index}' for index in range(count))
        session = uuid4().hex
        rng = np.random.default_rng(np.random.SeedSequence([seed, count, dimension]))
        with _TCPPair(tuple(endpoints), timeout) as transport:
            for round_index in range(rounds):
                contract = SecureContract(session, round_index, '0' * 64, roster, dimension, float(clip), scale)
                updates = rng.normal(0, clip / max(1, dimension) ** 0.5, size=(count, dimension))
                rows.append(_measure_round(transport, contract, updates, clip, seed))
    return rows


def _validate_benchmark(clients, dimension, rounds, seed, timeout, clip, scale):
    if (
        not clients
        or len(set(clients)) != len(clients)
        or any(type(count) is not int or not 3 <= count <= MAX_CLIENTS for count in clients)
        or type(dimension) is not int
        or not 1 <= dimension <= MAX_DIMENSION
        or type(rounds) is not int
        or not 1 <= rounds <= 10000
        or type(seed) is not int
        or seed < 0
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError('Invalid bounded protocol benchmark configuration')
    for count in clients:
        _validate_base(tuple(map(str, range(count))), dimension, clip, scale)


def _benchmark_summary(clients, dimension, rounds, seed, clip, scale, collector_a):
    return {
        'schema_version': 1,
        'scope': SCOPE,
        'clients': list(clients),
        'dimension': dimension,
        'rounds_per_condition': rounds,
        'seed': seed,
        'clip': clip,
        'scale': scale,
        'transport_environment': 'same_host_loopback' if collector_a is None else 'user_supplied_endpoints',
        'independent_physical_hosts_verified': False,
        'detector_training_measured': False,
        'startup_in_round_times': False,
        'tls': False,
        'authentication': False,
        'differential_privacy': False,
        'complete': False,
    }


def _condition_summary(rows, count):
    selected = [row for row in rows if row['clients'] == count]
    times = [row['elapsed_seconds'] for row in selected]
    return {
        'clients': count,
        'measured_rounds': len(selected),
        'round_median_seconds': float(np.median(times)),
        'round_p95_seconds': float(np.quantile(times, 0.95)),
        'tcp_application_total_bytes': sum(row['tcp_application_total_bytes'] for row in selected),
        'all_plaintext_quantized_sums_equal': all(row['plaintext_quantized_sum_equal'] for row in selected),
    }


def _benchmark_rows(measure, clients, timeout, collector_a, collector_b):
    if collector_a is None:
        with _LocalCollectors(len(clients), timeout) as endpoints:
            return endpoints, measure(endpoints)
    endpoints = (_endpoint(collector_a), _endpoint(collector_b))
    if endpoints[0] == endpoints[1]:
        raise ValueError('Collector endpoints must be distinct')
    return endpoints, measure(endpoints)


def _write_rounds(out, rows):
    with (out / 'protocol_rounds.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_benchmark(out, summary, measure, clients, timeout, collector_a, collector_b):
    try:
        endpoints, rows = _benchmark_rows(measure, clients, timeout, collector_a, collector_b)
        summary['endpoints'] = [f'{host}:{port}' for host, port in endpoints]
        summary['conditions'] = [_condition_summary(rows, count) for count in clients]
        _write_rounds(out, rows)
        summary['complete'] = True
    except Exception as error:
        summary['error_type'] = type(error).__name__
        summary['error'] = str(error)
        raise
    finally:
        (out / 'protocol_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def run_protocol_validation(
    out,
    *,
    clients=(3, 5, 10),
    dimension=14,
    rounds=5,
    seed=2026,
    timeout=10.0,
    collector_a=None,
    collector_b=None,
    clip=0.1,
    scale=1_000_000,
):
    """Measure synthetic updates and write a scoped receipt and round table."""
    clients = tuple(clients)
    _validate_benchmark(clients, dimension, rounds, seed, timeout, clip, scale)
    if (collector_a is None) != (collector_b is None):
        raise ValueError('Provide both collector endpoints, or neither')
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    summary = _benchmark_summary(clients, dimension, rounds, seed, clip, scale, collector_a)
    measure = partial(
        _measure,
        clients=clients,
        dimension=dimension,
        rounds=rounds,
        seed=seed,
        timeout=timeout,
        clip=clip,
        scale=scale,
    )
    return _save_benchmark(out, summary, measure, clients, timeout, collector_a, collector_b)
