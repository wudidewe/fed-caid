"""Measure local process resources and a separate TCP payload microbenchmark."""

from __future__ import annotations

import socket
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from threading import Event, Thread

import numpy as np
import psutil
import torch

from . import utils as core


class Resources:
    """Sample parent and child RSS; GPU statistics track PyTorch allocations."""

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
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        self.start = time.perf_counter()
        self.thread = Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.done.set()
        self.thread.join()
        self.result = {
            "elapsed_seconds": time.perf_counter() - self.start,
            "peak_process_tree_rss_bytes": self.peak_rss,
            "rss_sample_interval_s": 0.1,
            "rss_samples": self.samples,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if self.device == "cuda" else None,
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if self.device == "cuda" else None,
        }


class ResidentInference:
    """Time warm one-record inference including transform, detector and decision."""

    def __init__(self, model, transform, raw, device):
        self.model, self.transform, self.raw, self.device = model, transform, raw, device

    def __call__(self, decide):
        def infer():
            prediction = core.log_prediction(self.model, self.transform(self.raw), self.device)
            decide(core.probabilities(prediction))
            if self.device == "cuda":
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
            "repeats_per_connection": None,
            "resident_single_repeats": len(times),
            "median_ms": float(np.median(times)),
            "p95_ms": float(np.quantile(times, 0.95)),
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
                    raise OSError("TCP benchmark peer disconnected")
                left -= len(data)
            connection.sendall(b"\x01")


def _send(port, payload, repeats):
    with closing(socket.create_connection(("127.0.0.1", port), timeout=15)) as connection:
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        times = []
        for _ in range(repeats + 1):
            start = time.perf_counter()
            connection.sendall(payload)
            if connection.recv(1) != b"\x01":
                raise OSError("TCP benchmark peer acknowledgment missing")
            times.append((time.perf_counter() - start) * 1000)
    return np.asarray(times[1:])


def benchmark_tcp(clients, adapter_share_bytes, weight_message_bytes=None, *, repeats=20):
    """Loopback TCP transfer+ack for synthetic payload sizes, no protocol/security emulation."""
    patterns = [("adapter_share", adapter_share_bytes, 2)]
    if weight_message_bytes is not None:
        patterns.append(("fedavg_weights", weight_message_bytes, 1))
    rows = []
    for name, size, collectors in patterns:
        if size < 1:
            raise ValueError("TCP payload size must be positive")
        # Multiple persistent connections, one per client/receiver, measured concurrently.
        count = clients * collectors
        with closing(socket.socket()) as listener, ThreadPoolExecutor(max_workers=2 * count) as pool:
            listener.bind(("127.0.0.1", 0))
            listener.listen(count)
            listener.settimeout(15)
            port = listener.getsockname()[1]
            servers = [pool.submit(_transfer, listener, size, repeats) for _ in range(count)]
            payload = bytes(size)
            transfers = [pool.submit(_send, port, payload, repeats) for _ in range(count)]
            times = np.concatenate([transfer.result() for transfer in transfers])
            for server in servers:
                server.result()
        rows.append({
            "stage": "loopback_tcp_payload_transfer_ack",
            "mode": name,
            "message_bytes": size,
            "connections": count,
            "repeats_per_connection": repeats,
            "median_ms": float(np.median(times)),
            "p95_ms": float(np.quantile(times, 0.95)),
        })
    return rows
