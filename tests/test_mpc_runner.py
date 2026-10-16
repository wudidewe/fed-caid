"""End-to-end boundaries for the unified MPC mode; fixtures are synthetic."""

import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import privacy_core as pc
import privacy_run as run
from aggregation_audit import AggregationAudit
from secure_aggregation import SecureAggregationError, SecureContract


class MPCRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="mpc_runner_", dir=run.ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def args(self, output="experiment", **options):
        values = [
            "--out",
            str(self.root / output),
            "--smoke",
            "--clients",
            "3",
            "--rounds",
            "2",
            "--base-epochs",
            "1",
            "--adapter-epochs",
            "2",
            "--batch-size",
            "128",
            "--device",
            "cpu",
        ]
        for key, value in options.items():
            values.extend(["--" + key.replace("_", "-"), str(value)])
        return run.parser().parse_args(values)

    def test_release_supports_only_fresh_mpc(self):
        default = run.parser().parse_args([])
        self.assertEqual(default.mechanism, "mpc")
        self.assertIsNone(default.improve_from)
        self.assertIsNone(default.partition_seed)
        self.assertIsNone(default.training_seed)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            run.parser().parse_args(["--mechanism", "dp"])

    def test_fixed_calibration_indices_across_training_seeds(self):
        parts, names, features = run.synthetic_data(0)
        partitions = run.client_partitions(parts, len(names), 3, 0.8, 42, 0)
        models = []
        for training_seed in (42, 43):
            args = self.args(training_seed=training_seed, partition_seed=42)
            # A directly instantiated client receives the partition seed via
            # the legacy alias, exactly as execute's normalization specifies.
            args.seed = args.partition_seed
            with contextlib.redirect_stdout(io.StringIO()):
                node = run.PrivateClient(
                    "client_0",
                    parts,
                    partitions[0],
                    names,
                    features,
                    args,
                    self.root / str(training_seed),
                )
            models.append(node)
        np.testing.assert_array_equal(models[0].fit_cal_rows, models[1].fit_cal_rows)
        np.testing.assert_array_equal(models[0].threshold_cal_rows, models[1].threshold_cal_rows)
        self.assertEqual(models[0].parameter_count, models[1].parameter_count)
        self.assertEqual(
            models[0].performance["training_row_visits"],
            models[1].performance["training_row_visits"],
        )
        self.assertTrue(
            any(
                not torch.equal(a, b)
                for a, b in zip(
                    models[0].model.state_dict().values(), models[1].model.state_dict().values()
                )
            )
        )

    def test_four_domains_secure_transcript_and_private_reload(self):
        args = self.args(training_seed=43, partition_seed=42)
        original_submit = run.PrivateClient.submit_secure
        contracts = []

        def observe_submission(node, channel, initial, contract):
            contracts.append(contract.as_dict())
            return original_submit(node, channel, initial, contract)

        with (
            contextlib.redirect_stdout(io.StringIO()),
            patch.object(run.PrivateClient, "submit_secure", new=observe_submission),
        ):
            self.assertEqual(run.execute(args), 0)
        output = Path(args.out)
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.training_seed, 43)
        self.assertEqual(len(contracts), 4 * 3 * 2)
        summary = json.loads((output / "public" / "summary.json").read_text())
        self.assertEqual(len(summary), 4)
        self.assertEqual(
            {row["domain"] for row in summary}, {"in_vehicle_CAN", "external_network_proxy"}
        )
        self.assertFalse(list(output.rglob("budget.jsonl")))
        self.assertFalse(list(output.rglob("comparison_private.json")))
        for row in summary:
            self.assertEqual(row["mechanism"], "mpc")
            self.assertEqual(row["completed_rounds"], 2)
            self.assertTrue(row["synthetic"])
            metrics = row["privacy_metrics"]
            self.assertIsNone(metrics["epsilon"])
            self.assertIsNone(metrics["delta"])
            guarantee = row.get("privacy_guarantee", metrics.get("privacy_guarantee"))
            self.assertEqual(guarantee, "secure_sum_no_DP")
            directory = output / "public" / row["dataset"]
            self.assertFalse(list(directory.glob("*client_*.json")))
            self.assertEqual(len(list(directory.glob("r*_aggregate.json"))), 2)
            dimension = len(run.synthetic_data(run.DATASETS.index(row["dataset"]))[1]) - 1
            self.assertEqual(row["adapter_dimension"], dimension)
            private = output / "private_DO_NOT_UPLOAD" / row["dataset"]
            performance = json.loads((private / "performance_private.json").read_text())
            self.assertEqual(performance["raw_train_rows"], performance["selected_train_rows"])
            self.assertTrue(performance["full_assigned_train"])
            self.assertEqual(len(performance["clients"]), 3)
            for client in performance["clients"]:
                client_directory = private / client["client"]
                saved = torch.load(
                    client_directory / "detector_private.pt", map_location="cpu", weights_only=False
                )
                self.assertEqual(saved["framework_version"], 3)
                self.assertEqual(saved["mechanism"], "mpc")
                self.assertEqual(len(saved["class_score_adapter"]), dimension)
                with np.load(client_directory / "partition_private.npz") as partition:
                    train_rows, cal_rows = len(partition["train"]), len(partition["cal"])
                    validation_rows = partition["val"]
                self.assertEqual(client["training_row_visits"], train_rows)
                self.assertEqual(
                    client["optimizer_steps"], int(np.ceil(train_rows / args.batch_size))
                )
                with np.load(client_directory / "calibration_split_private.npz") as split:
                    self.assertFalse(np.intersect1d(split["fit"], split["threshold"]).size)
                    np.testing.assert_array_equal(
                        np.sort(np.r_[split["fit"], split["threshold"]]), np.arange(cal_rows)
                    )
                raw_parts, names, _ = run.synthetic_data(run.DATASETS.index(row["dataset"]))
                raw, labels = raw_parts["val"]
                decisions = run.predict_private_decisions(
                    client_directory / "detector_private.pt", raw[validation_rows]
                )
                np.testing.assert_array_equal(
                    decisions["attack_alert"], decisions["attack_score"] > saved["alarm_threshold"]
                )
                np.testing.assert_allclose(decisions["probabilities"].sum(1), 1.0, atol=1e-8)
                actual = run.risk_metrics(
                    labels[validation_rows],
                    decisions["probabilities"],
                    names,
                    saved["alarm_threshold"],
                )
                expected = json.loads((client_directory / "validation_private.json").read_text())
                self.assertEqual(actual, expected)
        text = "\n".join(path.read_text() for path in (output / "public").rglob("*.json"))
        for forbidden in (
            '"share":',
            '"aggregate_share":',
            '"state_dict":',
            '"X_train":',
            '"y_train":',
            '"log_prior":',
            '"alarm_threshold":',
            '"local_class_score_adapter":',
            '"detector_training_seconds":',
            '"adapter_training_seconds":',
            '"inference_seconds":',
        ):
            self.assertNotIn(forbidden, text)
        completion = json.loads((output / "public" / "completion.json").read_text())
        self.assertTrue(completion["complete"])
        with self.assertRaisesRegex(ValueError, "NEW --out"):
            run.execute(args)

    def test_missing_roster_failure_has_no_aggregate_or_plaintext_fallback(self):
        args = self.args(output="failed", scope="can")
        original_submit = run.PrivateClient.submit_secure

        def skip_one_client(node, channel, initial, contract):
            if node.client_id != "client_2":
                return original_submit(node, channel, initial, contract)

        with (
            contextlib.redirect_stdout(io.StringIO()),
            patch.object(run.PrivateClient, "submit_secure", new=skip_one_client),
        ):
            with self.assertRaisesRegex(SecureAggregationError, "incomplete_roster"):
                run.execute(args)
        public = Path(args.out) / "public"
        self.assertFalse((public / "completion.json").exists())
        self.assertFalse(list(public.rglob("r*_aggregate.json")))
        self.assertFalse(list(public.rglob("*client_*.json")))

    def test_invalid_mpc_roster_rejected_before_training_or_output(self):
        args = self.args(output="invalid", clients=2)
        with patch.object(run, "PrivateClient", side_effect=AssertionError("Must not train")):
            with self.assertRaisesRegex(ValueError, "at least three"):
                run.execute(args)
        self.assertFalse(Path(args.out).exists())


class MPCAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="mpc_audit_", dir=run.ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.clients = ("client_0", "client_1", "client_2")
        self.audit = AggregationAudit(
            self.root / "audit.jsonl", "CAN", self.clients, 2, 0.1, 1000000
        )
        self.contract = SecureContract(
            "experiment", 1, pc.digest([0.0, 0.0]), self.clients, 2, 0.1, 1000000
        )

    def receipt(self, contract, model):
        return {
            "contract_hash": contract.contract_hash,
            "parent_model": contract.parent_model,
            "model_hash": pc.digest(model.tolist()),
        }

    def test_policy_and_order_mismatch_rejected_before_appending(self):
        original = self.audit.path.read_bytes()
        for changed in (
            replace(self.contract, clients=("client_0", "client_1", "other")),
            replace(self.contract, dimension=3),
            replace(self.contract, clip=0.2),
            replace(self.contract, scale=1000),
            replace(self.contract, round_index=2),
            replace(self.contract, parent_model="a" * 64),
        ):
            with self.subTest(contract=changed):
                with self.assertRaises(ValueError):
                    self.audit.reserve(changed)
                self.assertEqual(self.audit.path.read_bytes(), original)
        self.audit.reserve(self.contract)
        reserved = self.audit.path.read_bytes()
        for changed in (self.contract, replace(self.contract, round_index=2)):
            with self.assertRaises(ValueError):
                self.audit.reserve(changed)
            self.assertEqual(self.audit.path.read_bytes(), reserved)

    def test_receipt_commit_is_bound_to_model_contract_and_parent(self):
        model = np.array([0.01, -0.02])
        receipt = self.receipt(self.contract, model)
        with self.assertRaises(ValueError):
            self.audit.commit(self.contract, model, receipt)
        self.audit.reserve(self.contract)
        reserved = self.audit.path.read_bytes()
        for field in ("contract_hash", "parent_model", "model_hash"):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    self.audit.commit(self.contract, model, {**receipt, field: "f" * 64})
                self.assertEqual(self.audit.path.read_bytes(), reserved)
        with self.assertRaises(ValueError):
            self.audit.commit(self.contract, np.ones(3), receipt)
        state = self.audit.commit(self.contract, model, receipt)
        self.assertEqual(state["completed_rounds"], 1)
        committed = self.audit.path.read_bytes()
        with self.assertRaises(ValueError):
            self.audit.commit(self.contract, model, receipt)
        self.assertEqual(self.audit.path.read_bytes(), committed)
        with self.assertRaises(ValueError):
            self.audit.reserve(replace(self.contract, round_index=2))
        following = replace(self.contract, round_index=2, parent_model=receipt["model_hash"])
        self.audit.reserve(following)
        self.audit.commit(following, model, self.receipt(following, model))
        self.assertEqual(self.audit.verify()["completed_rounds"], 2)

    def test_chain_tampering_and_changed_pinned_head_are_detected(self):
        head = self.audit.verify()["head"]
        self.audit.reserve(self.contract)
        with self.assertRaisesRegex(ValueError, "pinned head"):
            self.audit.verify(head)
        before = self.audit.path.read_bytes()
        self.audit.path.write_bytes(before.replace(b'"experiment"', b'"tampered"'))
        with self.assertRaisesRegex(ValueError, "hash chain"):
            self.audit.verify()


if __name__ == "__main__":
    unittest.main()
