"""Single-run diagnostics preserve measured timing and missing observations."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from experiment_report_data import load_runs, ReportDataError
from report_timeseries import load_observations
from test_experiment_report_data_loader import fixture, write


def observations(root, metrics, seed=42, *, ram=150.0):
    resources = {
        "cpu_percent_mean": 125.0,
        "ram_peak_mib": ram,
        "sample_count": 2,
        "scope": "Coordinator process; CPU 100% = one logical core; RSS excludes GPU and child processes.",
    }
    value = {
        "schema": "round_observability_v1",
        "dataset": "D",
        "training_seed": seed,
        "partition_seed": 42,
        "evaluation": "VAL development",
        "class_names": ["BENIGN", "ATTACK"],
        "rows": 20,
        "resources_summary": resources,
        "rounds": [
            {
                "round": 1,
                "metrics": metrics,
                "adapter_fit_seconds": 0.03,
                "aggregation_seconds": 0.02,
                "cached_personalization_seconds": 0.01,
                "cached_val_decision_seconds": 0.005,
                "resources": resources,
            }
        ],
    }
    write(root, "private_DO_NOT_UPLOAD/D/round_observability_private.json", value)
    return value


class RoundPresentationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.root = Path(self.temp.name)
        self.one = self.root / "seed_42"
        self.metrics = fixture(self.one)

    def tearDown(self):
        self.temp.cleanup()

    def test_old_results_have_no_reconstructed_round_history(self):
        data = load_runs([self.one])
        result = load_observations([self.one], data)
        self.assertFalse(result["datasets"]["D"]["available"])
        self.assertEqual(result["provenance"], [])

    def test_cpu_above_one_core_and_missing_psutil_rss_are_valid(self):
        observations(self.one, self.metrics, ram=None)
        result = load_observations([self.one], load_runs([self.one]))
        summary = result["datasets"]["D"]["resources_summary"]
        self.assertEqual(summary["cpu_percent_mean"]["mean"], 125.0)
        self.assertIsNone(summary["cpu_percent_mean"]["sample_sd"])
        self.assertIsNone(summary["ram_peak_mib"]["mean"])
        self.assertEqual(len(result["provenance"][0]["sha256"]), 64)

    def test_round_f1_is_the_recorded_weighted_f1(self):
        observations(self.one, self.metrics)
        result = load_observations([self.one], load_runs([self.one]))
        scores = result["datasets"]["D"]["rounds"][0]["metrics"]
        self.assertEqual(list(scores), ["weighted_f1"])
        self.assertEqual(scores["weighted_f1"]["mean"], self.metrics["weighted_f1"])
        self.assertEqual(scores["weighted_f1"]["n"], 1)
        self.assertIsNone(scores["weighted_f1"]["sample_sd"])

    def test_round_loader_rejects_additional_run_directories(self):
        two = self.root / "seed_43"
        fixture(two, seed=43)
        observations(self.one, self.metrics)
        with self.assertRaisesRegex(ReportDataError, "exactly one"):
            load_observations([self.one, two], load_runs([self.one]))

    def test_missing_round_timer_remains_unavailable(self):
        value = observations(self.one, self.metrics)
        value["rounds"][0].pop("aggregation_seconds")
        write(self.one, "private_DO_NOT_UPLOAD/D/round_observability_private.json", value)
        result = load_observations([self.one], load_runs([self.one]))
        timer = result["datasets"]["D"]["rounds"][0]["timing"]["aggregation_seconds"]
        self.assertIsNone(timer["mean"])
        self.assertEqual(timer["n"], 0)

    def test_round_history_respects_dataset_specific_round_override(self):
        protocol = json.loads((self.one / "public/protocol.json").read_text(encoding="utf-8"))
        protocol["dataset_parameters"] = {"D": {"rounds": 2}}
        write(self.one, "public/protocol.json", protocol)
        aggregate = json.loads(
            (self.one / "public/D/r1_aggregate.json").read_text(encoding="utf-8")
        )
        aggregate["round_index"] = 2
        write(self.one, "public/D/r2_aggregate.json", aggregate)
        value = observations(self.one, self.metrics)
        second = deepcopy(value["rounds"][0])
        second["round"] = 2
        value["rounds"].append(second)
        write(self.one, "private_DO_NOT_UPLOAD/D/round_observability_private.json", value)

        data = load_runs([self.one])
        self.assertEqual(data["runs"][0]["settings"]["rounds"], 1)
        self.assertEqual(data["runs"][0]["datasets"]["D"]["aggregation"]["completed_rounds"], 2)
        result = load_observations([self.one], data)
        self.assertEqual([row["round"] for row in result["datasets"]["D"]["rounds"]], [1, 2])

    def test_conflicting_support_and_unmeasured_resource_values_fail(self):
        value = observations(self.one, self.metrics)
        value["rounds"][0]["metrics"] = deepcopy(self.metrics)
        value["rounds"][0]["metrics"]["rows"] = 21
        write(self.one, "private_DO_NOT_UPLOAD/D/round_observability_private.json", value)
        with self.assertRaisesRegex(ReportDataError, "VAL support"):
            load_observations([self.one], load_runs([self.one]))
        value = observations(self.one, self.metrics)
        value["resources_summary"]["sample_count"] = 0
        write(self.one, "private_DO_NOT_UPLOAD/D/round_observability_private.json", value)
        with self.assertRaisesRegex(ReportDataError, "without resource samples"):
            load_observations([self.one], load_runs([self.one]))

    def test_last_round_cannot_contradict_final_confusion_matrix(self):
        value = observations(self.one, self.metrics)
        alternate = deepcopy(self.metrics)
        alternate.update(
            {
                "confusion_matrix": [[10, 0], [0, 10]],
                "accuracy": 1.0,
                "macro_precision": 1.0,
                "weighted_precision": 1.0,
                "macro_f1": 1.0,
                "weighted_f1": 1.0,
                "far": 0.0,
                "fnr": 0.0,
                "per_class": {
                    name: {"precision": 1.0, "recall": 1.0, "f1-score": 1.0, "support": 10}
                    for name in ["BENIGN", "ATTACK"]
                },
            }
        )
        value["rounds"][0]["metrics"] = alternate
        write(self.one, "private_DO_NOT_UPLOAD/D/round_observability_private.json", value)
        with self.assertRaisesRegex(ReportDataError, "final round metrics"):
            load_observations([self.one], load_runs([self.one]))


if __name__ == "__main__":
    unittest.main()
