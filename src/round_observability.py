"""Collect private round diagnostics and local resource observations.

Replay cached CAL/VAL predictions only after aggregation and client saving finish.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import statistics
import threading
import time


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
    import numpy as np
    from privacy_core import ClassScoreAdapter, fit_class_adapter
    from privacy_run import empirical_alarm_threshold, risk_metrics

    nodes, snapshots = tuple(nodes), tuple(global_vectors)
    if not nodes or not snapshots:
        raise ValueError("At least one private client and completed round required")
    names = list(nodes[0].names)
    if (
        len(names) < 2
        or any(list(node.names) != names for node in nodes)
        or any(getattr(node, "framework_version", 3) != 3 for node in nodes)
    ):
        raise ValueError("Matching v3 MPC clients in one public class space required")
    if len({node.client_id for node in nodes}) != len(nodes):
        raise ValueError("Distinct client identities required")
    for node in nodes:
        calibration = getattr(node, "attack_subtype_calibration", None)
        if calibration is not None and not calibration.identity:
            raise ValueError("Subtype continuation needs a separately scoped observer")
    rounds = [entry.get("round") for entry in snapshots]
    if any(type(index) is not int for index in rounds) or rounds != list(
        range(1, len(snapshots) + 1)
    ):
        raise ValueError("Consecutive completed rounds starting at one required")
    if (
        not isinstance(dataset, str)
        or not dataset
        or any(type(value) is not int or value < 0 for value in (training_seed, partition_seed))
    ):
        raise ValueError("Dataset identity and nonnegative integer seeds required")
    timings = {entry["round"]: entry for entry in original_round_timings or ()}
    result = []
    for snapshot in snapshots:
        global_vector = np.asarray(snapshot["adapter"], dtype=np.float64).copy()
        # Adapter setters validate dimensions, finiteness and radius.
        validation = ClassScoreAdapter(len(names))
        validation.set_vector(global_vector)
        if not np.array_equal(validation.get_vector(), global_vector):
            raise ValueError("Round global adapter is outside its public radius")
        probability_rows, label_rows, threshold_rows = [], [], []
        personalization_seconds, decision_seconds = 0.0, 0.0
        for node in nodes:
            started = time.perf_counter()
            fit = node.fit_cal_rows
            delta = fit_class_adapter(
                global_vector,
                node.probabilities["cal"][fit],
                node.parts["cal"][1][fit],
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
                correction.transform(node.probabilities["cal"][threshold_rows_local]),
                node.parts["cal"][1][threshold_rows_local],
                node.target_far,
                node.threshold_floor,
            )
            probabilities = correction.transform(node.probabilities["val"])
            labels = node.parts["val"][1]
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
        pooled["macro_recall"] = float(
            statistics.mean(pooled["per_class"][name]["recall"] for name in names)
        )
        decision_seconds += time.perf_counter() - started
        actual = timings.get(snapshot["round"], {})
        for key in ("adapter_fit_seconds", "aggregation_seconds"):
            value = actual.get(key)
            if value is not None and (
                type(value) not in (float, int) or not math.isfinite(value) or value < 0
            ):
                raise ValueError("Nonnegative finite actual round timings required")
        result.append(
            {
                "round": snapshot["round"],
                "global_adapter": global_vector.tolist(),
                "metrics": pooled,
                "adapter_fit_seconds": actual.get("adapter_fit_seconds"),
                "aggregation_seconds": actual.get("aggregation_seconds"),
                "cached_personalization_seconds": personalization_seconds,
                "cached_val_decision_seconds": decision_seconds,
                "resources": actual.get("resources"),
            }
        )
    document = {
        "schema": "round_observability_v1",
        "dataset": dataset,
        "training_seed": training_seed,
        "partition_seed": partition_seed,
        "evaluation": "VAL development",
        "class_names": names,
        "rows": sum(len(node.parts["val"][1]) for node in nodes),
        "client_count": len(nodes),
        "resources_summary": resources_summary,
        "scope": "PRIVATE fixed development VAL records repeated across rounds; not independent TEST",
        "personalization": "each recorded global adapter plus the same CAL local fit and held-out benign threshold rule as final saved models",
        "detector_training": "once before federation; not repeated or attributed to each round",
        "inference": "cached base CAL/VAL probabilities; observer timings are not backbone prediction timings",
        "timing_scope": "adapter_fit_seconds and aggregation_seconds measured during actual execution; cached personalization/VAL times are post-protocol replay, exclude detector inference",
        "raw_data_or_full_models_uploaded": False,
        "test_opened": False,
        "rounds": result,
    }
    save_private_observations(Path(private_dir) / "round_observability_private.json", document)
    return document


def save_private_observations(path, observations):
    """Write only beneath an explicitly private directory, never public/."""
    path = Path(path).resolve()
    if "private_DO_NOT_UPLOAD" not in path.parts:
        raise ValueError("Round observations must remain in private_DO_NOT_UPLOAD")
    if path.exists():
        raise ValueError("Existing private observations are never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(observations, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


class ResourceSampler:
    """Sample main-process CPU and optional RSS without changing RNG or GPU state.

    Missing memory is None; sampled peaks exclude child processes and GPU usage.
    """

    def __init__(self, interval=0.2):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("Positive finite resource sampling interval required")
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
                "wall_seconds": time.perf_counter(),
                "parent_cpu_seconds": time.process_time(),
                "parent_rss_bytes": None,
                "unreadable_process_count": 0,
            }
            if self._process is not None:
                try:
                    frame["parent_rss_bytes"] = int(self._process.memory_info().rss)
                except Exception:
                    frame["unreadable_process_count"] += 1
            self._samples.append(frame)
            return len(self._samples) - 1

    def start(self):
        if self._started is not None:
            raise ValueError("Resource sampler can only be started once")
        self._started = time.perf_counter()
        self._sample()
        self._thread = threading.Thread(
            target=self._run, name="private-resource-sampler", daemon=True
        )
        self._thread.start()
        return self

    def _run(self):
        while not self._event.wait(self.interval):
            self._sample()

    def begin_interval(self):
        """Return an interval token; take this at the actual execution boundary."""
        if self._started is None:
            raise ValueError("Start resource sampling before creating an interval")
        return self._sample()

    def end_interval(self, token):
        """Return observations for an interval ending now, during real execution."""
        if self._started is None or self._event.is_set():
            raise ValueError("An active resource sampler is required")
        return self.summary(token, self._sample())

    def summary(self, start=None, stop=None):
        """Summarize a complete run or the actual marked execution interval."""
        if self._started is None:
            raise ValueError("Start resource sampling before requesting observations")
        if stop is None and not self._event.is_set():
            stop = self._sample()
        with self._lock:
            start = 0 if start is None else start
            stop = len(self._samples) - 1 if stop is None else stop
            if (
                type(start) is not int
                or type(stop) is not int
                or not 0 <= start <= stop < len(self._samples)
            ):
                raise ValueError("Valid resource interval tokens required")
            frames = self._samples[start : stop + 1]
        elapsed = frames[-1]["wall_seconds"] - frames[0]["wall_seconds"]
        parent_cpu = frames[-1]["parent_cpu_seconds"] - frames[0]["parent_cpu_seconds"]
        rss = [
            frame["parent_rss_bytes"] for frame in frames if frame["parent_rss_bytes"] is not None
        ]
        result = {
            "elapsed_seconds": elapsed,
            "sample_count": len(frames),
            "sampling_interval_seconds": self.interval,
            "optional_psutil_available": self._psutil,
            "scope": "actual execution; main simulator process including threads; collector subprocesses and GPU excluded",
            "cpu_seconds": parent_cpu,
            "cpu_percent_mean": 100 * parent_cpu / elapsed if elapsed > 0 else None,
            "ram_peak_mib": max(rss) / (1024**2) if rss else None,
            "process_scope": "main simulator process including threads; collector subprocesses excluded",
            "cpu_percent_scope": "100% is one fully used logical CPU; may exceed 100%; not normalized by host CPU count",
            "memory_scope": "sampled main-process RSS peak MiB; excludes child processes and GPU memory; not exact lifetime peak or host utilization",
            "unreadable_process_samples": sum(
                frame["unreadable_process_count"] for frame in frames
            ),
        }
        return result

    def stop(self):
        if self._started is None:
            raise ValueError("Resource sampler has not been started")
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
