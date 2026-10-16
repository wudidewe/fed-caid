"""Round diagnostics preserve trained state, class spaces and random streams."""

import copy
import json
from pathlib import Path
import random
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import privacy_run
from round_observability import ResourceSampler, observe_rounds, save_private_observations


class UnusableModel:
    def __getattr__(self, name):
        raise AssertionError("Cached diagnostics must not touch the trained detector")


def client(identity, classes):
    names = ["BENIGN"] + [f"ATTACK_{index}" for index in range(1, classes)]
    # Every class is represented globally; a client can lack training classes.
    cal_labels = np.tile(np.arange(classes), 4)
    val_labels = np.tile(np.arange(classes), 3)

    def probabilities(labels):
        p = np.full((len(labels), classes), 0.01 / (classes - 1))
        p[np.arange(len(labels)), labels] = 0.99
        return p

    return SimpleNamespace(
        client_id=identity,
        framework_version=3,
        names=names,
        probabilities={"cal": probabilities(cal_labels), "val": probabilities(val_labels)},
        parts={"cal": (None, cal_labels), "val": (None, val_labels)},
        fit_cal_rows=np.arange(classes * 2),
        threshold_cal_rows=np.arange(classes * 2, classes * 4),
        log_prior=np.zeros(classes),
        adapter_epochs=2,
        adapter_lr=0.1,
        prior_strength=0.5,
        adapter_prox=1.0,
        target_far=0.01,
        threshold_floor=0.5,
        last_fit_initial=np.full(classes - 1, 0.01),
        last_fit_vector=np.full(classes - 1, 0.02),
        performance={"adapter_training_seconds": 12.5},
        model=UnusableModel(),
    )


class ObserverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.temp.cleanup)
        self.private = Path(self.temp.name) / "private_DO_NOT_UPLOAD" / "dataset"

    def observe(self, nodes, **overrides):
        kwargs = {
            "private_dir": self.private,
            "dataset": "fixture",
            "training_seed": 42,
            "partition_seed": 42,
        }
        kwargs.update(overrides)
        classes = len(nodes[0].names)
        return observe_rounds(
            nodes,
            [
                {"round": 1, "adapter": [0.0] * (classes - 1)},
                {"round": 2, "adapter": [0.01] * (classes - 1)},
            ],
            **kwargs,
        )

    def test_observation_does_not_mutate_node_state_or_random_streams(self):
        nodes = [client("client_0", 6), client("client_1", 6)]
        before = [
            {
                key: copy.deepcopy(getattr(node, key))
                for key in (
                    "probabilities",
                    "parts",
                    "fit_cal_rows",
                    "threshold_cal_rows",
                    "last_fit_initial",
                    "last_fit_vector",
                    "performance",
                )
            }
            for node in nodes
        ]
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        torch_rng = torch.get_rng_state().clone()
        document = self.observe(nodes)
        self.assertEqual(random.getstate(), python_rng)
        after_numpy = np.random.get_state()
        self.assertEqual(after_numpy[0], numpy_rng[0])
        np.testing.assert_array_equal(after_numpy[1], numpy_rng[1])
        self.assertEqual(after_numpy[2:], numpy_rng[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))
        for node, original in zip(nodes, before):
            self.assertEqual(node.performance, original["performance"])
            for key in (
                "last_fit_initial",
                "last_fit_vector",
                "fit_cal_rows",
                "threshold_cal_rows",
            ):
                np.testing.assert_array_equal(getattr(node, key), original[key])
            for role in ("cal", "val"):
                np.testing.assert_array_equal(
                    node.probabilities[role], original["probabilities"][role]
                )
                np.testing.assert_array_equal(node.parts[role][1], original["parts"][role][1])
        self.assertEqual(document["schema"], "round_observability_v1")
        self.assertEqual(document["rows"], 36)
        self.assertFalse(document["test_opened"])
        self.assertEqual(document["evaluation"], "VAL development")
        self.assertTrue((self.private / "round_observability_private.json").is_file())
        for row in document["rounds"]:
            self.assertEqual(row["metrics"]["accuracy"], 1.0)
            self.assertEqual(row["metrics"]["macro_recall"], 1.0)
            self.assertIsNone(row["adapter_fit_seconds"])
            self.assertIsNone(row["aggregation_seconds"])
            self.assertIsNone(row["resources"])
            self.assertGreaterEqual(row["cached_personalization_seconds"], 0)
            self.assertGreaterEqual(row["cached_val_decision_seconds"], 0)

    def test_all_four_public_class_spaces_keep_their_dimensions(self):
        for classes in (5, 6, 10, 15):
            with self.subTest(classes=classes):
                destination = self.private / str(classes)
                document = self.observe([client("client_0", classes)], private_dir=destination)
                for row in document["rounds"]:
                    self.assertEqual(len(row["global_adapter"]), classes - 1)
                    self.assertEqual(len(row["metrics"]["confusion_matrix"]), classes)
                    self.assertEqual(row["metrics"]["class_names"], document["class_names"])
                    self.assertEqual(row["metrics"]["macro_f1"], 1.0)

    def test_actual_execution_timing_and_resources_are_distinct_from_replay(self):
        resource = {
            "cpu_percent_mean": 156.0,
            "ram_peak_mib": 125.5,
            "sample_count": 4,
            "scope": "actual main-process observation",
        }
        document = self.observe(
            [client("client_0", 5)],
            original_round_timings=[
                {
                    "round": 1,
                    "adapter_fit_seconds": 0.2,
                    "aggregation_seconds": 0.3,
                    "resources": resource,
                },
                {"round": 2, "adapter_fit_seconds": 0.4, "aggregation_seconds": 0.5},
            ],
            resources_summary=resource,
        )
        self.assertEqual(document["resources_summary"], resource)
        self.assertEqual(document["rounds"][0]["resources"], resource)
        self.assertEqual(document["rounds"][0]["adapter_fit_seconds"], 0.2)
        self.assertEqual(document["rounds"][1]["aggregation_seconds"], 0.5)
        self.assertIsNone(document["rounds"][1]["resources"])
        self.assertNotIn("training_seconds", document["rounds"][0])
        self.assertNotIn("validation_inference_seconds", document["rounds"][0])
        saved = json.loads((self.private / "round_observability_private.json").read_text("utf-8"))
        self.assertEqual(saved, document)

    def test_invalid_class_contract_rounds_and_timing_fail_closed(self):
        nodes = [client("client_0", 5), client("client_1", 6)]
        with self.assertRaisesRegex(ValueError, "class space"):
            self.observe(nodes)
        nodes = [client("client_0", 5)]
        for snapshots in (
            [{"round": 2, "adapter": [0.0] * 4}],
            [{"round": 1, "adapter": [0.0] * 3}],
            [{"round": 1, "adapter": [float("nan")] * 4}],
        ):
            with self.assertRaises(ValueError):
                observe_rounds(
                    nodes,
                    snapshots,
                    private_dir=self.private,
                    dataset="fixture",
                    training_seed=42,
                    partition_seed=42,
                )
        with self.assertRaisesRegex(ValueError, "timings"):
            self.observe(nodes, original_round_timings=[{"round": 1, "adapter_fit_seconds": -0.1}])
        self.assertFalse((self.private / "round_observability_private.json").exists())

    def test_private_boundaries_and_existing_observations_are_protected(self):
        with self.assertRaisesRegex(ValueError, "private_DO_NOT_UPLOAD"):
            save_private_observations(Path(self.temp.name) / "public" / "report.json", {})
        path = self.private / "round_observability_private.json"
        save_private_observations(path, {"original": True})
        with self.assertRaisesRegex(ValueError, "overwritten"):
            save_private_observations(path, {"original": False})
        self.assertEqual(json.loads(path.read_text("utf-8")), {"original": True})


class ResourceTests(unittest.TestCase):
    def test_sampler_reports_actual_intervals_and_missing_memory_honestly(self):
        sampler = ResourceSampler(0.01)
        # Optional psutil absence must not make CPU timing fail or invent RSS.
        sampler._process, sampler._psutil = None, False
        with sampler:
            token = sampler.begin_interval()
            total = sum(index * index for index in range(80000))
            self.assertGreater(total, 0)
            time.sleep(0.025)
            interval = sampler.end_interval(token)
            self.assertGreater(interval["elapsed_seconds"], 0)
            self.assertGreaterEqual(interval["cpu_percent_mean"], 0)
            self.assertGreaterEqual(interval["sample_count"], 2)
            self.assertIsNone(interval["ram_peak_mib"])
            self.assertFalse(interval["optional_psutil_available"])
        final = sampler.stop()
        self.assertGreaterEqual(final["elapsed_seconds"], interval["elapsed_seconds"])
        self.assertIn("collector subprocesses", final["scope"])
        self.assertIn("100%", final["cpu_percent_scope"])
        with self.assertRaises(ValueError):
            sampler.start()

    def test_invalid_sampling_settings_and_lifecycle_fail(self):
        for interval in (0, -0.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                ResourceSampler(interval)
        sampler = ResourceSampler()
        for action in (sampler.begin_interval, sampler.summary, sampler.stop):
            with self.assertRaises(ValueError):
                action()


if __name__ == "__main__":
    unittest.main()
