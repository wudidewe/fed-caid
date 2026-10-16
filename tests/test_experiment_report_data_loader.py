"""Portable JSON-only fixtures for the private experiment report loader."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from experiment_report_data import ReportDataError, load_runs


def write(root, relative, value):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(root, seed=42, version=3):
    """No models, NPZ files, raw data, or numerical dependencies are created."""
    names = ["BENIGN", "ATTACK"]
    p0, p1, r0, r1 = 9 / 11, 8 / 9, .9, .8
    f0, f1 = 2 * p0 * r0 / (p0 + r0), 2 * p1 * r1 / (p1 + r1)
    metrics = {"rows": 20, "class_names": names, "confusion_matrix": [[9, 1], [2, 8]],
               "decision_rule": "total_attack_probability_then_attack_subtype",
               "accuracy": .85, "macro_precision": (p0 + p1) / 2,
               "weighted_precision": (p0 + p1) / 2, "macro_f1": (f0 + f1) / 2,
               "weighted_f1": (f0 + f1) / 2, "far": .1, "fnr": .2,
               "per_class": {"BENIGN": {"precision": p0, "recall": r0, "f1-score": f0, "support": 10},
                             "ATTACK": {"precision": p1, "recall": r1, "f1-score": f1, "support": 10}}}
    settings = {"smoke": True, "mechanism": "mpc", "training_seed": seed, "partition_seed": 42,
                "seed": 42, "clients": 3, "rounds": 1, "base_epochs": 1, "secure_scale": 1000000,
                "clip": .1, "device": "cpu"}
    common = {"framework_version": version, "datasets": ["D"], "differential_privacy": False,
              "epsilon": None, "delta": None, "test_opened": False}
    write(root, "public/completion.json", {**common, "complete": True, "synthetic": True, "mechanism": "mpc"})
    write(root, "public/protocol.json", {**common, "mechanism": "two_server_additive_secure_sum", "settings": settings})
    base = "private_DO_NOT_UPLOAD/D"
    metric_path = "private_DO_NOT_UPLOAD/selected_models/D" if version == 5 else base
    write(root, metric_path + "/pooled_validation_private.json", metrics)
    write(root, "public/D/r1_aggregate.json", {"schema": "two_server_additive_sum_v1", "round_index": 1,
          "dimension": 1, "roster_size": 3, "fixed_point_scale": 1000000, "clip": .1,
          "all_clients_accepted": True, "differential_privacy": False, "adapter": [0.], "accepted_update": False})
    timers = []
    for index in range(3):
        client = f"client_{index}"
        folder = base + "/" + client
        pins = {role: hashlib.sha256((client + role).encode()).hexdigest() for role in ("train", "cal", "val")}
        write(root, folder + "/prediction_cache_manifest_private.json",
              {"class_names": names, "feature_names": ["flow/second"], "partition_indices_sha256": pins})
        write(root, folder + "/calibration_private.json", {"threshold": .5, "target_far": .01,
              "benign_holdout_rows": 5, "calibration_far": 0., "formal_conformal_guarantee": False})
        write(root, folder + "/training_history_private.json", [{"epoch": 1, "training_loss": .2, "seconds": .1}])
        write(root, folder + "/binary_preservation_private.json", {"passed": True, "threshold_exact": True})
        timers.append({"client": client, "detector_training_seconds": .1, "adapter_training_seconds": .01})
    write(root, base + "/performance_private.json", {"total_experiment_seconds": 1., "total_seconds": .2, "clients": timers})
    if version == 5:
        write(root, base + "/release_guard_private.json", {"selected": "reference", "passed": False,
              "comparisons": {"accuracy": {"reference": .85, "candidate": .8}}})
        # Deliberately different, invalid candidate content must not be selected.
        write(root, base + "/pooled_validation_private.json", {"candidate": "not selected"})
    return metrics


class ReportLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.root = Path(self.temp.name)
        self.one = self.root / "seed_42"
        fixture(self.one)

    def tearDown(self):
        self.temp.cleanup()

    def test_single_seed_has_no_sample_sd_and_keeps_private_scope(self):
        data = load_runs([self.one])
        self.assertEqual(data["run_count"], 1)
        self.assertEqual(data["seeds"], [42])
        self.assertEqual(data["summary"]["D"]["metrics"]["accuracy"]["mean"], .85)
        self.assertEqual(data["summary"]["D"]["metrics"]["accuracy"]["n"], 1)
        self.assertIsNone(data["summary"]["D"]["metrics"]["macro_f1"]["sample_sd"])
        self.assertEqual(data["scope"]["visibility"], "private")
        self.assertFalse(data["scope"]["test_opened"])
        self.assertFalse(data["scope"]["input_content_verified"])

    def test_single_run_keeps_integer_confusion_counts_and_support(self):
        data = load_runs([self.one])
        self.assertEqual(data["summary"]["D"]["support"]["BENIGN"], 10)
        confusion = data["summary"]["D"]["confusion"]
        self.assertEqual(confusion["matrix"], [[9, 1], [2, 8]])
        self.assertTrue(all(type(value) is int for row in confusion["matrix"] for value in row))
        self.assertNotIn("mean_matrix", confusion)
        self.assertNotIn("per_seed", confusion)

    def test_v5_uses_selected_models_and_never_follows_reference_history(self):
        five = self.root / "v5"
        fixture(five, version=5)
        data = load_runs([five])
        self.assertEqual(data["summary"]["D"]["metrics"]["accuracy"]["mean"], .85)
        self.assertEqual(data["runs"][0]["datasets"]["D"]["selection"]["arm"], "reference")
        self.assertFalse(data["runs"][0]["datasets"]["D"]["history"]["available"])

    def test_missing_selected_result_does_not_fall_back_to_candidate(self):
        five = self.root / "v5"
        fixture(five, version=5)
        (five / "private_DO_NOT_UPLOAD/selected_models/D/pooled_validation_private.json").unlink()
        with self.assertRaises(ReportDataError):
            load_runs([five])

    def test_multiple_runs_are_rejected_instead_of_selecting_the_first(self):
        two = self.root / "seed_43"
        fixture(two, seed=43)
        with self.assertRaisesRegex(ReportDataError, "Exactly one"):
            load_runs([self.one, two])
        with self.assertRaisesRegex(ReportDataError, "Exactly one"):
            load_runs([self.one, self.one])

    def test_empty_input_is_rejected(self):
        with self.assertRaisesRegex(ReportDataError, "Exactly one"):
            load_runs([])

    def test_support_and_matrix_disagreement_is_rejected(self):
        relative = "private_DO_NOT_UPLOAD/D/pooled_validation_private.json"
        data = json.loads((self.one / relative).read_text())
        data["per_class"]["BENIGN"]["support"] = 11
        write(self.one, relative, data)
        with self.assertRaisesRegex(ReportDataError, "support"):
            load_runs([self.one])

    def test_invalid_partition_hash_is_rejected(self):
        relative = "private_DO_NOT_UPLOAD/D/client_0/prediction_cache_manifest_private.json"
        data = json.loads((self.one / relative).read_text())
        data["partition_indices_sha256"]["val"] = "not-a-sha256"
        write(self.one, relative, data)
        with self.assertRaisesRegex(ReportDataError, "SHA-256"):
            load_runs([self.one])

    def test_client_feature_contract_mismatch_is_rejected(self):
        relative = "private_DO_NOT_UPLOAD/D/client_1/prediction_cache_manifest_private.json"
        data = json.loads((self.one / relative).read_text())
        data["feature_names"] = ["different_feature"]
        write(self.one, relative, data)
        with self.assertRaisesRegex(ReportDataError, "feature_names"):
            load_runs([self.one])

    def test_weighted_and_macro_metric_validation_is_retained(self):
        relative = "private_DO_NOT_UPLOAD/D/pooled_validation_private.json"
        original = json.loads((self.one / relative).read_text())
        for key in ("weighted_f1", "macro_f1"):
            with self.subTest(metric=key):
                data = {**original, key: .1}
                write(self.one, relative, data)
                with self.assertRaisesRegex(ReportDataError, key):
                    load_runs([self.one])
        write(self.one, relative, original)

    def test_loader_does_not_modify_json_inputs(self):
        before = {str(path.relative_to(self.one)): path.read_bytes() for path in self.one.rglob("*.json")}
        load_runs([self.one])
        after = {str(path.relative_to(self.one)): path.read_bytes() for path in self.one.rglob("*.json")}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
