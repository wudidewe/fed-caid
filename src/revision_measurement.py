"""Measure execution resources, inference, and private round observations."""

from __future__ import annotations

import json
import math
import os
import socket
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from threading import Event, Thread

import numpy as np
import psutil
import torch

from . import utils as core


class Resources:
    """Sample summed process RSS, including shared pages in each process's RSS."""

    def __init__(self, device):
        self.device = device
        self.process = psutil.Process()
        self.done = Event()
        self.peak_rss = 0
        self.samples = 0

    def _sample(self):
        while True:
            total = 0
            for process in [self.process, *self.process.children(recursive=True)]:
                try:
                    total += process.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            self.peak_rss = max(self.peak_rss, total)
            self.samples += 1
            if self.done.wait(0.1):
                break

    def __enter__(self):
        if self.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        self.start = time.perf_counter()
        self.thread = Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.done.set()
        self.thread.join()
        self.result = {
            'elapsed_seconds': time.perf_counter() - self.start,
            'peak_process_tree_rss_bytes': self.peak_rss,
            'rss_accounting': 'sum of process RSS; shared pages may be counted more than once',
            'rss_sample_interval_s': 0.1,
            'rss_samples': self.samples,
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated() if self.device == 'cuda' else None,
            'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved() if self.device == 'cuda' else None,
        }


class ResidentInference:
    """Time warm one-record inference including transform, detector and decision."""

    def __init__(self, model, transform, raw, device, *, score_enhancer=None):
        self.model, self.transform, self.raw, self.device = model, transform, raw, device
        self.score_enhancer = score_enhancer

    def __call__(self, decide):
        def infer():
            x = self.transform(self.raw)
            p = core.probabilities(core.log_prediction(self.model, x, self.device))
            decide(p if self.score_enhancer is None else self.score_enhancer(self.raw, x, p))
            if self.device == 'cuda':
                torch.cuda.synchronize()

        self.model.to(self.device)
        try:
            for _ in range(3):
                infer()
            times = []
            for _ in range(20):
                start = time.perf_counter()
                infer()
                times.append((time.perf_counter() - start) * 1000)
        finally:
            self.model.cpu()
        return {
            'repeats_per_connection': None,
            'resident_single_repeats': len(times),
            'median_ms': float(np.median(times)),
            'p95_ms': float(np.quantile(times, 0.95)),
        }


def _transfer(listener, size, repeats):
    connection, _ = listener.accept()
    with connection:
        connection.settimeout(15)
        for _ in range(repeats + 1):
            left = size
            while left:
                data = connection.recv(min(left, 65536))
                if not data:
                    raise OSError('TCP benchmark peer disconnected')
                left -= len(data)
            connection.sendall(b'\x01')


def _send(port, payload, repeats):
    with closing(socket.create_connection(('127.0.0.1', port), timeout=15)) as connection:
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        times = []
        for _ in range(repeats + 1):
            start = time.perf_counter()
            connection.sendall(payload)
            if connection.recv(1) != b'\x01':
                raise OSError('TCP benchmark peer acknowledgment missing')
            times.append((time.perf_counter() - start) * 1000)
    return np.asarray(times[1:])


def benchmark_tcp(clients, adapter_share_bytes, weight_message_bytes=None, *, repeats=20):
    """Loopback TCP transfer+ack for synthetic payload sizes, no protocol/security emulation."""
    if type(clients) is not int or clients < 1:
        raise ValueError('TCP client count must be a positive integer')
    if type(repeats) is not int or repeats < 1:
        raise ValueError('TCP repeat count must be a positive integer')
    patterns = [('adapter_share', adapter_share_bytes, 2)]
    if weight_message_bytes is not None:
        patterns.append(('fedavg_weights', weight_message_bytes, 1))
    if any(type(size) is not int or size < 1 for _, size, _ in patterns):
        raise ValueError('TCP payload sizes must be positive integers')
    rows = []
    for name, size, collectors in patterns:
        # Multiple persistent connections, one per client/receiver, measured concurrently.
        count = clients * collectors
        with closing(socket.socket()) as listener, ThreadPoolExecutor(max_workers=2 * count) as pool:
            listener.bind(('127.0.0.1', 0))
            listener.listen(count)
            listener.settimeout(15)
            port = listener.getsockname()[1]
            servers = [pool.submit(_transfer, listener, size, repeats) for _ in range(count)]
            payload = bytes(size)
            transfers = [pool.submit(_send, port, payload, repeats) for _ in range(count)]
            times = np.concatenate([transfer.result() for transfer in transfers])
            for server in servers:
                server.result()
        rows.append(
            {
                'stage': 'loopback_tcp_payload_transfer_ack',
                'mode': name,
                'message_bytes': size,
                'connections': count,
                'repeats_per_connection': repeats,
                'median_ms': float(np.median(times)),
                'p95_ms': float(np.quantile(times, 0.95)),
            }
        )
    return rows


def observe_rounds(
    nodes,
    global_vectors,
    *,
    private_dir,
    dataset,
    training_seed,
    partition_seed,
    original_round_timings=None,
    resources_summary=None,
):
    """Replay private VAL decisions from completed rounds without changing nodes.

    Timings and resources must come from execution, not this replay.
    """
    from .privacy_core import ClassScoreAdapter, fit_class_adapter
    from .privacy_run import empirical_alarm_threshold, risk_metrics

    nodes, snapshots = tuple(nodes), tuple(global_vectors)
    if not nodes or not snapshots:
        raise ValueError('At least one private client and completed round required')
    names = list(nodes[0].names)
    if (
        len(names) < 2
        or any(list(node.names) != names for node in nodes)
        or any(getattr(node, 'framework_version', 3) != 3 for node in nodes)
    ):
        raise ValueError('Matching v3 MPC clients in one public class space required')
    if len({node.client_id for node in nodes}) != len(nodes):
        raise ValueError('Distinct client identities required')
    for node in nodes:
        calibration = getattr(node, 'attack_subtype_calibration', None)
        if calibration is not None and not calibration.identity:
            raise ValueError('Subtype continuation needs a separately scoped observer')
    rounds = [entry.get('round') for entry in snapshots]
    if any(type(index) is not int for index in rounds) or rounds != list(range(1, len(snapshots) + 1)):
        raise ValueError('Consecutive completed rounds starting at one required')
    if (
        not isinstance(dataset, str)
        or not dataset
        or any(type(value) is not int or value < 0 for value in (training_seed, partition_seed))
    ):
        raise ValueError('Dataset identity and nonnegative integer seeds required')
    timings = {entry['round']: entry for entry in original_round_timings or ()}
    result = []
    for snapshot in snapshots:
        global_vector = np.asarray(snapshot['adapter'], dtype=np.float64).copy()
        # Adapter setters validate dimensions, finiteness and radius.
        validation = ClassScoreAdapter(len(names))
        validation.set_vector(global_vector)
        if not np.array_equal(validation.get_vector(), global_vector):
            raise ValueError('Round global adapter is outside its public radius')
        probability_rows, label_rows, threshold_rows = [], [], []
        personalization_seconds, decision_seconds = 0.0, 0.0
        for node in nodes:
            started = time.perf_counter()
            fit = node.fit_cal_rows
            delta = fit_class_adapter(
                global_vector,
                node.probabilities['cal'][fit],
                node.parts['cal'][1][fit],
                node.log_prior,
                node.adapter_epochs,
                node.adapter_lr,
                node.prior_strength,
                node.adapter_prox,
            )
            correction = ClassScoreAdapter(len(names))
            correction.set_vector(global_vector + delta)
            personalization_seconds += time.perf_counter() - started
            started = time.perf_counter()
            threshold_rows_local = node.threshold_cal_rows
            threshold, _ = empirical_alarm_threshold(
                correction.transform(node.probabilities['cal'][threshold_rows_local]),
                node.parts['cal'][1][threshold_rows_local],
                node.target_far,
                node.threshold_floor,
            )
            probabilities = correction.transform(node.probabilities['val'])
            labels = node.parts['val'][1]
            decision_seconds += time.perf_counter() - started
            probability_rows.append(probabilities)
            label_rows.append(labels)
            threshold_rows.append(np.full(len(labels), threshold))
        started = time.perf_counter()
        pooled = risk_metrics(
            np.concatenate(label_rows),
            np.concatenate(probability_rows),
            names,
            np.concatenate(threshold_rows),
        )
        pooled['macro_recall'] = float(statistics.mean(pooled['per_class'][name]['recall'] for name in names))
        decision_seconds += time.perf_counter() - started
        actual = timings.get(snapshot['round'], {})
        for key in ('adapter_fit_seconds', 'aggregation_seconds'):
            value = actual.get(key)
            if value is not None and (type(value) not in (float, int) or not math.isfinite(value) or value < 0):
                raise ValueError('Nonnegative finite actual round timings required')
        result.append(
            {
                'round': snapshot['round'],
                'global_adapter': global_vector.tolist(),
                'metrics': pooled,
                'adapter_fit_seconds': actual.get('adapter_fit_seconds'),
                'aggregation_seconds': actual.get('aggregation_seconds'),
                'cached_personalization_seconds': personalization_seconds,
                'cached_val_decision_seconds': decision_seconds,
                'resources': actual.get('resources'),
            }
        )
    document = {
        'schema': 'round_observability_v1',
        'dataset': dataset,
        'training_seed': training_seed,
        'partition_seed': partition_seed,
        'evaluation': 'VAL development',
        'class_names': names,
        'rows': sum(len(node.parts['val'][1]) for node in nodes),
        'client_count': len(nodes),
        'resources_summary': resources_summary,
        'scope': 'PRIVATE fixed development VAL records repeated across rounds; not independent TEST',
        'personalization': (
            'each recorded global adapter plus the same CAL local fit and held-out benign threshold rule as'
            ' final saved models'
        ),
        'detector_training': 'once before federation; not repeated or attributed to each round',
        'inference': ('cached base CAL/VAL probabilities; observer timings are not backbone prediction timings'),
        'timing_scope': (
            'adapter_fit_seconds and aggregation_seconds measured during actual execution; cached'
            ' personalization/VAL times are post-protocol replay, exclude detector inference'
        ),
        'raw_data_or_full_models_uploaded': False,
        'test_opened': False,
        'rounds': result,
    }
    save_private_observations(Path(private_dir) / 'round_observability_private.json', document)
    return document


def save_private_observations(path, observations):
    """Write only beneath an explicitly private directory, never public/."""
    path = Path(path).resolve()
    if 'private_DO_NOT_UPLOAD' not in path.parts:
        raise ValueError('Round observations must remain in private_DO_NOT_UPLOAD')
    if path.exists():
        raise ValueError('Existing private observations are never overwritten')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(observations, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')


class ResourceSampler:
    """Sample main-process CPU and optional RSS without changing RNG or GPU state.

    Missing memory is None; sampled peaks exclude child processes and GPU usage.
    """

    def __init__(self, interval=0.2):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError('Positive finite resource sampling interval required')
        self.interval = float(interval)
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._thread = None
        self._samples = []
        self._started = None
        try:
            import psutil
        except ImportError:
            self._process = None
        else:
            self._process = psutil.Process(os.getpid())
        self._psutil = self._process is not None

    def _sample(self):
        with self._lock:
            frame = {
                'wall_seconds': time.perf_counter(),
                'parent_cpu_seconds': time.process_time(),
                'parent_rss_bytes': None,
                'unreadable_process_count': 0,
            }
            if self._process is not None:
                try:
                    frame['parent_rss_bytes'] = int(self._process.memory_info().rss)
                except Exception:
                    frame['unreadable_process_count'] += 1
            self._samples.append(frame)
            return len(self._samples) - 1

    def start(self):
        """Take an initial sample and start the background sampling thread. Return this sampler."""
        if self._started is not None:
            raise ValueError('Resource sampler can only be started once')
        self._started = time.perf_counter()
        self._sample()
        self._thread = threading.Thread(target=self._run, name='private-resource-sampler', daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._event.wait(self.interval):
            self._sample()

    def begin_interval(self):
        """Return an interval token; take this at the actual execution boundary."""
        if self._started is None:
            raise ValueError('Start resource sampling before creating an interval')
        return self._sample()

    def end_interval(self, token):
        """Return observations for an interval ending now, during real execution."""
        if self._started is None or self._event.is_set():
            raise ValueError('An active resource sampler is required')
        return self.summary(token, self._sample())

    def summary(self, start=None, stop=None):
        """Summarize a complete run or the actual marked execution interval."""
        if self._started is None:
            raise ValueError('Start resource sampling before requesting observations')
        if stop is None and not self._event.is_set():
            stop = self._sample()
        with self._lock:
            start = 0 if start is None else start
            stop = len(self._samples) - 1 if stop is None else stop
            if type(start) is not int or type(stop) is not int or not 0 <= start <= stop < len(self._samples):
                raise ValueError('Valid resource interval tokens required')
            frames = self._samples[start : stop + 1]
        elapsed = frames[-1]['wall_seconds'] - frames[0]['wall_seconds']
        parent_cpu = frames[-1]['parent_cpu_seconds'] - frames[0]['parent_cpu_seconds']
        rss = [frame['parent_rss_bytes'] for frame in frames if frame['parent_rss_bytes'] is not None]
        result = {
            'elapsed_seconds': elapsed,
            'sample_count': len(frames),
            'sampling_interval_seconds': self.interval,
            'optional_psutil_available': self._psutil,
            'scope': (
                'actual execution; main simulator process including threads; collector subprocesses and GPU' ' excluded'
            ),
            'cpu_seconds': parent_cpu,
            'cpu_percent_mean': 100 * parent_cpu / elapsed if elapsed > 0 else None,
            'ram_peak_mib': max(rss) / (1024**2) if rss else None,
            'process_scope': 'main simulator process including threads; collector subprocesses excluded',
            'cpu_percent_scope': (
                '100% is one fully used logical CPU; may exceed 100%; not normalized by host CPU count'
            ),
            'memory_scope': (
                'sampled main-process RSS peak MiB; excludes child processes and GPU memory; not exact'
                ' lifetime peak or host utilization'
            ),
            'unreadable_process_samples': sum(frame['unreadable_process_count'] for frame in frames),
        }
        return result

    def stop(self):
        """Stop the background thread, take a final sample and return the run summary."""
        if self._started is None:
            raise ValueError('Resource sampler has not been started')
        if not self._event.is_set():
            self._event.set()
            self._thread.join(timeout=max(2.0, self.interval + 1.0))
            self._sample()
        return self.summary()

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        return False
