"""Local two-worker additive sharing for clipped federated updates.

The host and IPC are trusted; complete-roster reconstruction provides no DP,
remote-client authentication, or protection against malicious clients or collusion.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import secrets
import time
from concurrent.futures import Future
from dataclasses import dataclass
from multiprocessing.reduction import ForkingPickler
from threading import RLock, Thread

import numpy as np

PRIME = (1 << 61) - 1
SCHEMA = 'two_server_additive_sum_v1'


class SecureAggregationError(RuntimeError):
    """A protocol failure invalidates the entire aggregation session."""


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _frame_header_bytes(connection, payload_size):
    # Windows pipes send pickle directly; POSIX adds a 4/12-byte length prefix.
    if type(connection).__name__ == 'PipeConnection':
        return 0
    return 4 if payload_size <= 0x7FFFFFFF else 12


@dataclass(frozen=True)
class SecureContract:
    """Record a session identifier, round index, parent model digest and client roster together with the vector
    dimension, clip norm and encoding settings.
    """
    session: str
    round_index: int
    parent_model: str
    clients: tuple[str, ...]
    dimension: int
    clip: float
    scale: int
    prime: int = PRIME
    schema: str = SCHEMA

    def as_dict(self):
        """Return the canonical contract fields as a JSON-compatible dictionary."""
        return {
            'schema': self.schema,
            'session': self.session,
            'round_index': self.round_index,
            'parent_model': self.parent_model,
            'clients': list(self.clients),
            'dimension': self.dimension,
            'clip': self.clip,
            'scale': self.scale,
            'prime': self.prime,
        }

    @property
    def contract_hash(self):
        """Return the SHA-256 hexadecimal digest of the canonical round contract."""
        return hashlib.sha256(_canonical(self.as_dict())).hexdigest()


def _validate_base(clients, dimension, clip, scale):
    if (
        not isinstance(clients, (tuple, list))
        or len(clients) < 3
        or any(not isinstance(v, str) or not v or len(v) > 128 for v in clients)
        or len(set(clients)) != len(clients)
    ):
        raise ValueError('A fixed roster of at least three distinct clients is required')
    if type(dimension) is not int or dimension < 1:
        raise ValueError('Positive integer dimension required')
    if type(scale) is not int or scale < 1:
        raise ValueError('Positive integer fixed-point scale required')
    if isinstance(clip, bool) or not isinstance(clip, (float, int)) or not math.isfinite(clip) or clip <= 0:
        raise ValueError('Positive finite clipping radius required')
    bound = float(clip) * scale
    # Keep rounding representable in float64 and prevent signed aggregate wrap.
    if not math.isfinite(bound) or bound >= (1 << 52) or math.ceil(bound) * len(clients) >= PRIME // 2:
        raise ValueError('Fixed-point contract could overflow the signed aggregate')


def _validate_contract(value, base):
    expected_keys = {
        'schema',
        'session',
        'round_index',
        'parent_model',
        'clients',
        'dimension',
        'clip',
        'scale',
        'prime',
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise SecureAggregationError('invalid_contract')
    for key in ('schema', 'clients', 'dimension', 'clip', 'scale', 'prime'):
        if value[key] != base[key] or type(value[key]) is not type(base[key]):
            raise SecureAggregationError('changed_contract')
    if (
        not isinstance(value['session'], str)
        or not value['session']
        or len(value['session']) > 128
        or type(value['round_index']) is not int
        or value['round_index'] < 0
    ):
        raise SecureAggregationError('invalid_round')
    parent = value['parent_model']
    if not isinstance(parent, str) or len(parent) != 64 or any(c not in '0123456789abcdef' for c in parent):
        raise SecureAggregationError('invalid_parent_model')


def _worker_main(connection, base, server_index):
    """One worker receives only its share; it retains only a running field sum."""
    active = None
    seen = set()
    aggregate = None
    previous_round = None
    session_name = None
    try:
        connection.send({'ok': True, 'server': server_index})
        while True:
            request = connection.recv()
            if not isinstance(request, dict):
                raise SecureAggregationError('invalid_message')
            operation = request.get('op')
            if operation == 'close' and set(request) == {'op'}:
                return
            if operation == 'begin' and set(request) == {'op', 'contract'}:
                contract = request['contract']
                _validate_contract(contract, base)
                if active is not None:
                    raise SecureAggregationError('round_already_active')
                if session_name is not None and contract['session'] != session_name:
                    raise SecureAggregationError('changed_session')
                if previous_round is not None and contract['round_index'] != previous_round + 1:
                    raise SecureAggregationError('replayed_or_skipped_round')
                active = contract
                session_name = contract['session']
                seen = set()
                aggregate = [0] * base['dimension']
                connection.send({'ok': True, 'count': 0})
            elif operation == 'share' and set(request) == {
                'op',
                'contract_hash',
                'client_id',
                'share',
            }:
                if active is None:
                    raise SecureAggregationError('no_active_round')
                expected_hash = hashlib.sha256(_canonical(active)).hexdigest()
                if request['contract_hash'] != expected_hash:
                    raise SecureAggregationError('stale_or_changed_contract')
                client = request['client_id']
                if not isinstance(client, str) or client not in base['clients']:
                    raise SecureAggregationError('unregistered_client')
                if client in seen:
                    raise SecureAggregationError('duplicate_client')
                share = request['share']
                if (
                    not isinstance(share, list)
                    or len(share) != base['dimension']
                    or any(type(x) is not int or not 0 <= x < PRIME for x in share)
                ):
                    raise SecureAggregationError('invalid_field_share')
                aggregate = [(x + y) % PRIME for x, y in zip(aggregate, share)]
                seen.add(client)
                connection.send({'ok': True, 'count': len(seen)})
                # Do not retain an individual client's share after summation.
                del share, request
            elif operation == 'finish' and set(request) == {'op', 'contract_hash'}:
                if active is None or request['contract_hash'] != hashlib.sha256(_canonical(active)).hexdigest():
                    raise SecureAggregationError('stale_or_missing_round')
                if seen != set(base['clients']):
                    raise SecureAggregationError('incomplete_roster')
                connection.send({'ok': True, 'aggregate_share': aggregate, 'count': len(seen)})
                previous_round = active['round_index']
                active = None
                aggregate = None
                seen = set()
            else:
                raise SecureAggregationError('invalid_operation')
    except SecureAggregationError as error:
        try:
            connection.send({'ok': False, 'error': str(error)})
        except (EOFError, OSError):
            pass
    except (EOFError, OSError):
        pass
    finally:
        connection.close()


def _clip_quantize(update, dimension, clip, scale):
    update = np.asarray(update, dtype=np.float64)
    if update.ndim != 1 or len(update) != dimension or not np.isfinite(update).all():
        raise SecureAggregationError('invalid_update_vector')
    update = update.copy()
    largest = float(np.abs(update).max())
    if largest:
        unit = update / largest
        unit_norm = float(np.linalg.norm(unit))
        if largest > clip / unit_norm:
            update = unit * (clip / unit_norm)
    return [int(v) for v in np.rint(update * scale)]


def _split_quantized(quantized, generator=None):
    generator = secrets.SystemRandom() if generator is None else generator
    first = [generator.randrange(PRIME) for _ in quantized]
    second = [(int(value) - share) % PRIME for value, share in zip(quantized, first)]
    return first, second


def _frame_counts(connection, message, direction, counters):
    size = len(ForkingPickler.dumps(message))
    counters[f'{direction}_payload_bytes'] += size
    counters[f'{direction}_frame_header_bytes'] += _frame_header_bytes(connection, size)
    counters[f'{direction}_messages'] += 1


def _exchange(connections, messages, deadline, counters, completed):
    """Send requests and receive responses without blocking the caller past its deadline."""
    try:
        for connection, message in zip(connections, messages):
            connection.send(message)
            if counters is not None:
                _frame_counts(connection, message, 'sent', counters)
        responses = []
        for connection in connections:
            if not connection.poll(max(0.0, deadline - time.monotonic())):
                raise TimeoutError('aggregation_worker_timeout')
            response = connection.recv()
            if counters is not None:
                _frame_counts(connection, response, 'received', counters)
            responses.append(response)
    except Exception as error:
        completed.set_exception(error)
    else:
        completed.set_result((responses, counters))


class SecureAggregationSession:
    """Keep two workers and reconstruct only complete-roster aggregates.

    Call ``submit`` inside the client's upload method. Session operations are
    serialized, and each exchange has a deadline for sending and receiving.
    """

    def __init__(self, clients, dimension, clip=0.1, scale=1_000_000, timeout=15.0):
        if not isinstance(clients, (list, tuple)):
            raise ValueError('A list or tuple of fixed client identities is required')
        clients = tuple(clients)
        _validate_base(clients, dimension, clip, scale)
        if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('Positive finite timeout required')
        self.clients = clients
        self.dimension = dimension
        self.clip = float(clip)
        self.scale = scale
        self.timeout = float(timeout)
        self._base = {
            'schema': SCHEMA,
            'clients': list(clients),
            'dimension': dimension,
            'clip': float(clip),
            'scale': scale,
            'prime': PRIME,
        }
        self._connections = []
        self._processes = []
        self._active = None
        self._submitted = set()
        self._session_name = None
        self._previous_round = None
        self._closed = False
        self._counters = None
        self._lock = RLock()

    def __enter__(self):
        with self._lock:
            if self._closed or self._processes:
                raise SecureAggregationError('session_cannot_be_reopened')
            context = mp.get_context('spawn')
            try:
                for server_index in range(2):
                    local, remote = context.Pipe(duplex=True)
                    self._connections.append(local)
                    try:
                        process = context.Process(
                            target=_worker_main,
                            args=(remote, self._base, server_index),
                            daemon=True,
                        )
                        process.start()
                        self._processes.append(process)
                    finally:
                        remote.close()
                responses = self._call_workers(())
                if [item.get('server') for item in responses] != [0, 1]:
                    self._fail('invalid_worker_initialization')
            except BaseException:
                self._shutdown(abort=True)
                raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _fail(self, message):
        self._shutdown(abort=True)
        raise SecureAggregationError(message)

    def _call_workers(self, messages):
        completed = Future()
        deadline = time.monotonic() + self.timeout
        counters = None if self._counters is None else dict(self._counters)
        # Once a frame is partially available, Pipe.recv can still block.
        Thread(
            target=_exchange,
            args=(tuple(self._connections), messages, deadline, counters, completed),
            daemon=True,
        ).start()
        try:
            responses, counters = completed.result(timeout=max(0.0, deadline - time.monotonic()))
        except TimeoutError:
            self._fail('aggregation_worker_timeout')
        except (EOFError, OSError):
            self._fail('aggregation_worker_disconnected')
        except BaseException:
            self._shutdown(abort=True)
            raise
        for response in responses:
            if not isinstance(response, dict):
                self._fail('invalid_worker_response')
            if response.get('ok') is not True:
                self._fail(response.get('error', 'invalid_worker_response'))
        self._counters = counters
        return responses

    def _rpc(self, operation, payloads):
        if self._closed or len(self._connections) != 2:
            self._fail('session_is_not_open')
        return self._call_workers(tuple({'op': operation, **payload} for payload in payloads))

    def start_round(self, session, round_index, parent_model):
        """Open a round for the given session and round index, bound to the parent model digest. Return the validated
        contract.
        """
        with self._lock:
            contract = SecureContract(
                session, round_index, parent_model, self.clients, self.dimension, self.clip, self.scale
            )
            try:
                _validate_contract(contract.as_dict(), self._base)
            except SecureAggregationError as error:
                self._fail(str(error))
            if self._active is not None:
                self._fail('round_already_active')
            if self._session_name is not None and session != self._session_name:
                self._fail('changed_session')
            if self._previous_round is not None and round_index != self._previous_round + 1:
                self._fail('replayed_or_skipped_round')
            self._counters = {
                'sent_payload_bytes': 0,
                'received_payload_bytes': 0,
                'sent_frame_header_bytes': 0,
                'received_frame_header_bytes': 0,
                'sent_messages': 0,
                'received_messages': 0,
            }
            self._rpc('begin', [{'contract': contract.as_dict()}] * 2)
            self._active = contract
            self._submitted = set()
            self._session_name = session
            return contract

    def submit(self, client_id, update, contract=None):
        """Client-local clip/quantize/share; return only public acceptance metadata."""
        with self._lock:
            if self._active is None:
                self._fail('no_active_round')
            if contract is not None and (not isinstance(contract, SecureContract) or contract != self._active):
                self._fail('stale_or_changed_contract')
            if not isinstance(client_id, str) or client_id not in self.clients:
                self._fail('unregistered_client')
            if client_id in self._submitted:
                self._fail('duplicate_client')
            try:
                quantized = _clip_quantize(update, self._active.dimension, self._active.clip, self._active.scale)
                first, second = _split_quantized(quantized)
            except (SecureAggregationError, TypeError, ValueError, OverflowError) as error:
                self._fail(str(error) if isinstance(error, SecureAggregationError) else 'invalid_update_vector')
            payloads = [
                {'contract_hash': self._active.contract_hash, 'client_id': client_id, 'share': share}
                for share in (first, second)
            ]
            del quantized, first, second
            responses = self._rpc('share', payloads)
            self._submitted.add(client_id)
            if any(item.get('count') != len(self._submitted) for item in responses):
                self._fail('inconsistent_roster_count')
            return {'accepted': True, 'contract_hash': self._active.contract_hash}

    def aggregate(self):
        """Reconstruct the quantized mean from both collectors after the complete roster submits. Close the active
        round and return the mean and its receipt.
        """
        with self._lock:
            if self._active is None:
                self._fail('no_active_round')
            contract = self._active
            responses = self._rpc('finish', [{'contract_hash': contract.contract_hash}] * 2)
            shares = [item.get('aggregate_share') for item in responses]
            if (
                self._submitted != set(self.clients)
                or any(item.get('count') != len(self.clients) for item in responses)
                or any(
                    not isinstance(share, list)
                    or len(share) != self.dimension
                    or any(type(x) is not int or not 0 <= x < PRIME for x in share)
                    for share in shares
                )
            ):
                self._fail('invalid_aggregate_response')
            summed = [(x + y) % PRIME for x, y in zip(*shares)]
            signed = [x if x <= PRIME // 2 else x - PRIME for x in summed]
            bound = math.ceil(contract.clip * contract.scale) * len(contract.clients)
            if any(abs(x) > bound for x in signed):
                self._fail('aggregate_outside_public_coordinate_bound')
            mean = np.asarray(signed, dtype=np.float64) / (contract.scale * len(contract.clients))
            receipt = {
                'schema': SCHEMA,
                'contract_hash': contract.contract_hash,
                'session': contract.session,
                'round_index': contract.round_index,
                'parent_model': contract.parent_model,
                'dimension': self.dimension,
                'roster_size': len(self.clients),
                'clip': self.clip,
                'fixed_point_scale': self.scale,
                'field_prime': PRIME,
                'mean_coordinate_quantization_bound': 0.5 / self.scale,
                'floating_point_tolerance': 8 * np.finfo(np.float64).eps * max(1.0, self.clip),
                'all_clients_accepted': True,
                'differential_privacy': False,
                'security_scope': 'two_noncolluding_workers_trusted_local_simulation',
            }
            counters = dict(self._counters)
            receipt.update(
                {
                    'numeric_share_upload_bytes': 2 * len(self.clients) * self.dimension * 8,
                    'aggregate_share_download_bytes': 2 * self.dimension * 8,
                    'numeric_payload_encoding': (
                        'analytical uint64 field-element count; actual pickle IPC bytes reported separately'
                    ),
                    'serialized_application_bytes': (
                        counters['sent_payload_bytes'] + counters['received_payload_bytes']
                    ),
                    'local_ipc_frame_bytes': sum(
                        counters[key]
                        for key in (
                            'sent_payload_bytes',
                            'received_payload_bytes',
                            'sent_frame_header_bytes',
                            'received_frame_header_bytes',
                        )
                    ),
                    'communication_counters': counters,
                    'communication_scope': (
                        'complete round begin/share/finish commands and responses; ForkingPickler payload and'
                        ' Connection frame headers; excludes worker startup and OS transport overhead'
                    ),
                }
            )
            self._previous_round = contract.round_index
            self._active = None
            self._submitted = set()
            self._counters = None
            return mean, receipt

    def close(self):
        """Close the local pipes and stop any remaining collector workers."""
        with self._lock:
            self._shutdown(abort=False)

    def _shutdown(self, *, abort):
        if self._closed:
            return
        self._closed = True
        if abort:
            # Terminating the peers also releases transports blocked on a partial frame.
            for process in self._processes:
                if process.is_alive():
                    process.terminate()
        for connection in self._connections:
            connection.close()
        for process in self._processes:
            process.join(timeout=0.2)
            if process.is_alive():
                process.kill()
                process.join(timeout=0.2)
        self._active = None
        self._submitted = set()
        self._counters = None
