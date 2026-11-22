"""Check evidence interval and support handling without retraining."""

import json
import os
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.experiment_config import DATASETS
from src.review_evidence import (
    MAIN_MODES,
    METRICS,
    _tables,
    generate,
    latest_real_run,
    population_support,
    summarize,
    support_status,
)


class ReviewEvidenceTests(unittest.TestCase):
    def test_seed_interval_matches_t_reference(self):
        stat = summarize([0.4, 0.5, 0.6, 0.5, 0.5])
        self.assertEqual(stat["seeds_with_metric"], 5)
        self.assertAlmostEqual(stat["mean"], 0.5)
        self.assertAlmostEqual(stat["ci95_low"], 0.412201096689)
        self.assertAlmostEqual(stat["ci95_high"], 0.587798903311)

    def test_missing_metrics_and_bounded_intervals(self):
        self.assertIsNone(summarize([None, None])["mean"])
        self.assertIsNone(summarize([None, 1.0])["ci95_low"])
        self.assertEqual(summarize([0.99, 1, 1, 1, 1])["ci95_high"], 1)
        self.assertAlmostEqual(summarize([-0.05] * 5, bounds=(-1, 1))["mean"], -0.05)
        with self.assertRaises(ValueError):
            summarize([float("nan")])

    def test_support_boundary(self):
        self.assertEqual(
            [support_status(n) for n in (0, 1, 29, 30)],
            ["absent", "low_support", "low_support", "at_least_30"],
        )

    @staticmethod
    def _table_inputs(counts=None):
        counts = counts or {"BENIGN": 20, "ATTACK": 10}
        rows = sum(counts.values())
        roles = ("train", "cal", "val", "test")
        audit = {
            "role_rows": {role: rows for role in roles},
            "role_class_support": {role: dict(counts) for role in roles},
            "external_captures": False,
            "exact_feature_audit": {
                "test_exact_input_unseen_deduplicated": dict(counts),
                "cross_role_exact_overlap": {"train:test": {"test_rows": 0, "unique_shared_inputs": 0}},
            },
        }
        metrics = {name: 0.5 if rows else None for name in METRICS}
        metrics.update({
            "rows": rows,
            "macro_f1": 0.5 if sum(n > 0 for n in counts.values()) >= 2 else None,
            "balanced_accuracy": 0.5 if sum(n > 0 for n in counts.values()) >= 2 else None,
            "far": 0.5 if counts["BENIGN"] else None,
            "fnr": 0.5 if rows > counts["BENIGN"] else None,
        })
        case = {
            "evaluations": [
                {
                    "role": role,
                    "mode": mode,
                    "metrics": dict(metrics),
                    "per_class": [
                        {"class": label, "support": n, "recall": 0.5 if n else None}
                        for label, n in counts.items()
                    ],
                }
                for role in ("val", "test", "test_unique_unseen")
                for mode in MAIN_MODES
            ],
            "communication": [
                {"mode": mode, "round_payload_bytes": size}
                for mode, size in (("full", 100), ("fedavg", 2000))
            ],
        }
        identity = ("Car-Hacking", "grouped")
        return {identity: audit}, {identity: [{"seed": seed, **deepcopy(case)} for seed in range(42, 47)]}

    def test_main_tables_skip_supplement_only_pairs(self):
        audits, cases = self._table_inputs()
        tables = _tables(audits, cases)
        self.assertEqual(
            {row["contrast"] for row in tables["main_paired_intervals"]},
            {
                "local_only_minus_base",
                "full_minus_local_only",
                "full_minus_fedavg",
                "full_minus_fedavg_threshold",
            },
        )
        self.assertTrue(all(row["seeds_with_metric"] == 5 for row in tables["main_paired_intervals"]))
        self.assertEqual(len(tables["main_seed_intervals"]), 3 * len(MAIN_MODES) * len(METRICS))

    def test_absent_required_mode_is_still_rejected(self):
        audits, cases = self._table_inputs()
        for case in next(iter(cases.values())):
            case["evaluations"] = [row for row in case["evaluations"] if row["mode"] != "fedavg_threshold"]
        with self.assertRaisesRegex(ValueError, "Incomplete matched evaluations"):
            _tables(audits, cases)

    def test_incomplete_optional_mode_is_still_rejected(self):
        audits, cases = self._table_inputs()
        case = next(iter(cases.values()))[0]
        evaluation = deepcopy(case["evaluations"][0])
        evaluation["mode"] = "local_unclipped"
        case["evaluations"].append(evaluation)
        with self.assertRaisesRegex(ValueError, "Incomplete matched evaluations"):
            _tables(audits, cases)

    def test_population_support_exposes_single_class_and_empty_limits(self):
        benign = population_support({"BENIGN": 10, "ATTACK": 0})
        self.assertEqual((benign["benign_rows"], benign["attack_rows"]), (10, 0))
        self.assertEqual(benign["population_status"], "single_class")
        self.assertFalse(benign["multiclass_metrics_evaluable"])
        self.assertFalse(benign["fnr_evaluable"])
        self.assertTrue(benign["far_evaluable"])
        attack = population_support({"BENIGN": 0, "ATTACK": 10})
        self.assertFalse(attack["far_evaluable"])
        self.assertTrue(attack["fnr_evaluable"])
        self.assertEqual(population_support({"BENIGN": 0, "ATTACK": 0})["population_status"], "empty")
        with self.assertRaises(ValueError):
            population_support({"ATTACK": 10})

    def test_tables_label_unavailable_metrics_without_fabricating_values(self):
        audits, cases = self._table_inputs({"BENIGN": 10, "ATTACK": 0})
        tables = _tables(audits, cases)
        fnr = [row for row in tables["main_seed_intervals"] if row["metric"] == "fnr"]
        self.assertTrue(all(row["mean"] is None and row["seeds_with_metric"] == 0 for row in fnr))
        self.assertTrue(all(row["metric_support_status"] == "unavailable_no_attack_rows" for row in fnr))
        self.assertEqual(tables["split_audit_summary"][0]["test_unique_unseen_attack_rows"], 0)
        self.assertEqual(len(tables["evaluation_population_support"]), 5)

    def test_evidence_reuse_requires_matching_source_and_exports(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "revision"
            output = root / "evidence"
            hashes = {"run.json": "example"}
            cases = {("Car-Hacking", "grouped"): [{}]}
            with (
                patch("src.review_evidence._load", return_value=([42], {}, cases, hashes)),
                patch("src.review_evidence._tables", return_value={"test": [{"count": 5}]}),
                patch("src.review_evidence._payload_figure") as figure,
            ):
                self.assertEqual(generate(run, output, figures=False), output)
                self.assertEqual(generate(run, output, figures=False), output)
                figure.assert_not_called()
                with patch(
                    "src.review_evidence._report_source_hashes", return_value={"review_evidence.py": "changed"}
                ):
                    with self.assertRaises(FileExistsError):
                        generate(run, output, figures=False)
                hashes["run.json"] = "changed"
                with self.assertRaises(FileExistsError):
                    generate(run, output, figures=False)
                hashes["run.json"] = "example"
                (output / "test.csv").write_text("changed", encoding="utf-8")
                with self.assertRaises(FileExistsError):
                    generate(run, output, figures=False)

    def test_latest_real_run_supports_v1_and_v2_full_matrices(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            matrix = [
                {"dataset": dataset, "protocol": protocol, "scenarios": [["main", 3, 0.8]]}
                for dataset in DATASETS
                for protocol in ("frozen", "grouped")
            ]
            for row in matrix:
                if row["protocol"] == "grouped" and row["dataset"] in {"Car-Hacking", "CIC-UNSW-NB15"}:
                    row["scenarios"] += [
                        [name, 3, 0.8]
                        for name in (
                            "scale_reference",
                            "scale_alpha_low",
                            "scale_alpha_high",
                            "scale_clients_5",
                            "scale_clients_10",
                        )
                    ]
            v1 = root / "revision_v1"
            self._save_run_manifest(v1, matrix, 90, 1)
            self.assertEqual(latest_real_run(root), v1)
            for row in matrix:
                if row["protocol"] == "grouped":
                    row["scenarios"].append(["balance", 3, 0.8])
            v2 = root / "revision_v2"
            self._save_run_manifest(v2, matrix, 110, 2)
            self._save_run_manifest(root / "revision_partial", matrix[:-1], 110, 3)
            self.assertEqual(latest_real_run(root), v2)

    @staticmethod
    def _save_run_manifest(directory, matrix, replicas, timestamp):
        directory.mkdir()
        (directory / "run.json").write_text(
            json.dumps({
                "experiment": "fedcaid_review",
                "matrix": matrix,
                "seeds": list(range(42, 47)),
            }),
            encoding="utf-8",
        )
        marker = directory / "complete.json"
        marker.write_text(json.dumps({"replicates": replicas, "data_audits": len(matrix)}), encoding="utf-8")
        os.utime(marker, (timestamp, timestamp))


if __name__ == "__main__":
    unittest.main()
