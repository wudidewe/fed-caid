"""Check that external evaluation seals models and enforces independent input checks."""

import csv
import json
import unittest
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch

from src.privacy_run import PrivateClient
from src.revision_external import (
    CAPTURE_F1_LOWER_CRITERION,
    CLASS_ALARM_ROUTING,
    CRITERIA,
    DONOR_FULL_CONTRAST,
    DONOR_POOL_CONTRAST,
    FIXED_ROUTING,
    GUARDED_ROUTING,
    OPERATING_CONTRAST,
    OPERATING_REFERENCE_MODE,
    REFERENCE_SUBTYPE_CONTRAST,
    REFERENCE_SUBTYPE_MODE,
    SINGLE_DONOR_MODE,
    _compatible_inference_hashes,
    _fingerprints,
    _inference_hashes,
    _sha,
    confirm_external,
    frozen_selection_digest,
    load_development_screen,
    prepare_development_screen,
    save_frozen_head,
    seal_confirmation,
)
from src.revision_tabm_checkpoint import save_tabm_checkpoint


class ExternalConfirmationTests(unittest.TestCase):
    def test_format_compatibility_is_limited_to_exact_source_hashes(self):
        """An unknown implementation digest must invalidate the historical receipt."""
        resources = files("src.fedcaid_resources")
        current = _inference_hashes(7)
        package_table = resources.joinpath("package_import_compatibility.json")
        package_bindings = json.loads(package_table.read_text(encoding="utf-8"))
        unpackaged = current.copy()
        for name in current.keys() & package_bindings.keys():
            self.assertEqual(current[name], package_bindings[name]["packaged_sha256"])
            unpackaged[name] = package_bindings[name]["original_sha256"]
        self.assertTrue(_compatible_inference_hashes(unpackaged, current))
        table = resources.joinpath("frozen_format_compatibility.json")
        bindings = json.loads(table.read_text(encoding="utf-8"))
        original = unpackaged.copy()
        for name in current.keys() & bindings.keys():
            self.assertEqual(unpackaged[name], bindings[name]["formatted_sha256"])
            original[name] = bindings[name]["original_sha256"]
        self.assertTrue(_compatible_inference_hashes(original, current))
        self.assertFalse(_compatible_inference_hashes(original, {**current, "utils.py": "0" * 64}))

    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.raw = self.root / "external_original.txt"
        self.raw.write_text("independently prepared synthetic observations", encoding="utf-8")
        self.csv = self.root / "external.csv"
        self.csv.write_text("f,label,group_id\n-1,BENIGN,g1\n1,Attack,g2\n2,Attack,g2\n", encoding="utf-8")
        dev = {
            "train": (np.array([[8.0]]), None),
            "cal": (np.array([[9.0]]), None),
            "val": (np.array([[10.0]]), None),
        }
        path = self.root / "development.npy"
        self.screen = prepare_development_screen(
            path, dev, {"original.csv": "a" * 64}, development_group_ids=["held-out-of-test"]
        )
        self.assertEqual(load_development_screen(path), self.screen)
        local = np.zeros((2, 2), dtype=np.float32)
        fed = np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        self.models = []
        for seed in (42, 43):
            dest = self.root / f"head-{seed}.npz"
            receipt = save_frozen_head(
                dest,
                dataset="Toy",
                names=["BENIGN", "Attack"],
                features=["f"],
                seed=seed,
                receiver=0,
                target_class="Attack",
                development_screen=self.screen,
                vectors={"head_local": local, "head_federated": fed},
                thresholds={"head_local": 0.5, "head_federated": 0.5},
                settings={"protocol": {"rounds": 1}, "gate_disabled": {}},
            )
            self.assertEqual(receipt["sha256"], _sha(dest))
            self.models.append(dest)
        self.manifest = self.root / "manifest.json"
        self.seal = self.root / "seal.json"
        self.result = self.root / "results"
        self.config = {
            "version": 1,
            "dataset": "Toy",
            "target_class": "Attack",
            "attestation": {"custodian": "synthetic fixture", "unseen_in_development": True},
            "criteria": {**CRITERIA, "min_target_groups": 1, "min_benign_groups": 1, "min_seeds": 2},
            "captures": [
                {
                    "capture_id": "external",
                    "prepared_csv": self.csv.name,
                    "sha256": _sha(self.csv),
                    "raw_files": [{"path": self.raw.name, "sha256": _sha(self.raw)}],
                }
            ],
        }

    def make_seal(self):
        self.manifest.write_text(json.dumps(self.config), encoding="utf-8")
        return seal_confirmation(self.models, self.manifest, self.seal)

    def test_frozen_scoring_seals_metrics_and_reports_groups(self):
        self.make_seal()
        result = confirm_external(self.seal, self.result)
        self.assertEqual(result["status"], "criteria_met_on_declared_external_receiver")
        self.assertTrue(result["checks"]["gates_active"])
        self.assertFalse(result["threshold_selection_on_test"])
        self.assertNotIn("absolute_target_f1_seed_mean_met", result["checks"])
        self.assertEqual(result["absolute_federated_target_metrics"]["target_f1"]["mean"], 1)
        evidence = result["capture_evidence"]
        self.assertEqual(evidence["capture_units"], 1)
        self.assertEqual(evidence["seed_models"], 2)
        self.assertEqual(evidence["row_screen"]["development_roles"], ["train", "cal", "val"])
        self.assertEqual(evidence["row_screen"]["rows"], 3)
        self.assertIsNone(evidence["target_f1_pairing"]["ci95_low"])
        self.assertNotIn("capture_target_f1_lower_limit", result["checks"])
        self.assertEqual(len((self.result / "paired_by_seed.csv").read_text().splitlines()), 5)
        self.assertIn(
            "criteria_met_on_declared_external_receiver", (self.result / "complete.json").read_text()
        )

    def test_prospective_absolute_cutoffs_operate_on_pooled_seed_means(self):
        self.csv.write_text(
            "f,label,group_id\n-1,BENIGN,g1\n-2,Attack,g2\n1,Attack,g3\n2,Attack,g4\n", encoding="utf-8"
        )
        self.config["captures"][0]["sha256"] = _sha(self.csv)
        self.config["criteria"].update(min_absolute_target_f1=0.9, min_absolute_target_precision=1.0)
        plan = self.make_seal()
        self.assertEqual(plan["criteria"]["min_absolute_target_f1"], 0.9)
        result = confirm_external(self.seal, self.result)
        self.assertEqual(result["status"], "criteria_not_met")
        self.assertTrue(result["checks"]["target_f1_interval_positive"])
        self.assertTrue(result["checks"]["absolute_target_f1_complete_seed_support"])
        self.assertFalse(result["checks"]["absolute_target_f1_seed_mean_met"])
        self.assertTrue(result["checks"]["absolute_target_precision_seed_mean_met"])
        self.assertAlmostEqual(result["absolute_federated_target_metrics"]["target_f1"]["mean"], 0.8)

    def test_per_capture_support_and_capture_intervals_are_distinct_from_seed_intervals(self):
        self.csv.write_text("f,label,group_id\n-1,BENIGN,g1\n1,Attack,g2\n2,Attack,g3\n", encoding="utf-8")
        self.config["captures"][0]["sha256"] = _sha(self.csv)
        second = self.root / "external-second.csv"
        second.write_text("f,label,group_id\n-3,BENIGN,g4\n-4,Attack,g5\n", encoding="utf-8")
        raw = self.root / "external-original-second.txt"
        raw.write_text("second acquisition fixture", encoding="utf-8")
        self.config["captures"].append({
            "capture_id": "second",
            "prepared_csv": second.name,
            "sha256": _sha(second),
            "raw_files": [{"path": raw.name, "sha256": _sha(raw)}],
        })
        self.config["criteria"].update(
            min_captures=2, min_target_groups_per_capture=2, min_benign_groups_per_capture=1
        )
        self.make_seal()
        result = confirm_external(self.seal, self.result)
        self.assertEqual(result["status"], "insufficient_support_or_gate")
        self.assertTrue(result["checks"]["target_groups"])
        self.assertFalse(result["checks"]["target_groups_each_capture"])
        self.assertTrue(result["checks"]["benign_groups_each_capture"])
        with (self.result / "capture_metric_summaries.csv").open(newline="") as source:
            means = {
                r["capture_id"]: float(r["mean"])
                for r in csv.DictReader(source)
                if r["mode"] == "head_federated" and r["metric"] == "target_f1"
            }
        self.assertEqual(means, {"external": 1.0, "second": 0.0, "ALL_CAPTURES": 0.8})
        with (self.result / "capture_intervals.csv").open(newline="") as source:
            summary = next(
                r
                for r in csv.DictReader(source)
                if r["statistic"] == "absolute"
                and r["comparison"] == "head_federated"
                and r["metric"] == "target_f1"
            )
        self.assertEqual(summary["captures_with_metric"], "2")
        self.assertAlmostEqual(float(summary["mean"]), 0.5)
        self.assertGreater(float(summary["capture_sd"]), 0.0)

    def test_optional_external_criteria_reject_invalid_types_and_ranges(self):
        self.config["criteria"]["min_captures"] = 2
        for key, invalid in (
            (CAPTURE_F1_LOWER_CRITERION, -1),
            (CAPTURE_F1_LOWER_CRITERION, True),
            ("min_absolute_target_f1", 101),
            ("min_absolute_target_precision", True),
            ("min_target_groups_per_capture", 0),
            ("min_benign_groups_per_capture", 1.5),
        ):
            with self.subTest(key=key):
                self.config["criteria"][key] = invalid
                with self.assertRaisesRegex(ValueError, key):
                    self.make_seal()
                del self.config["criteria"][key]

    def test_capture_bound_requires_multiple_complete_captures(self):
        self.config["criteria"][CAPTURE_F1_LOWER_CRITERION] = 0
        with self.assertRaisesRegex(ValueError, "min_captures >= 2"):
            self.make_seal()
        self.config["criteria"]["min_captures"] = 2
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.make_seal()
        self._add_second_capture("f,label,group_id\n3,Attack,g4\n")
        self.make_seal()
        result = confirm_external(self.seal, self.result)
        self.assertEqual(result["status"], "insufficient_support_or_gate")
        self.assertFalse(result["checks"]["capture_target_f1_paired_support"])
        self.assertFalse(result["checks"]["capture_target_f1_lower_limit"])

    def test_predeclared_capture_bound_uses_paired_capture_means(self):
        self._add_second_capture("f,label,group_id\n-3,BENIGN,g3\n3,Attack,g4\n")
        self.config["criteria"].update({"min_captures": 2, CAPTURE_F1_LOWER_CRITERION: 0})
        self.make_seal()
        result = confirm_external(self.seal, self.result)
        self.assertEqual(result["status"], "criteria_met_on_declared_external_receiver")
        self.assertTrue(result["checks"]["capture_target_f1_lower_limit"])
        pairing = result["capture_evidence"]["target_f1_pairing"]
        self.assertEqual(pairing["captures_with_complete_seed_pairs"], 2)
        self.assertEqual(pairing["ci95_low"], 1.0)

    def _add_second_capture(self, data):
        second = self.root / "external-second.csv"
        second.write_text(data, encoding="utf-8")
        raw = self.root / "external-original-second.txt"
        raw.write_text("second acquisition fixture", encoding="utf-8")
        self.config["captures"].append({
            "capture_id": "second",
            "prepared_csv": second.name,
            "sha256": _sha(second),
            "raw_files": [{"path": raw.name, "sha256": _sha(raw)}],
        })

    def test_only_reviewed_training_revision_is_inference_compatible(self):
        key = "revision_shared_head.py"
        old = "9faba2ca0b1ef0f358346ce1d72de1cec19595c8f419092dee5416409f7b2ed0"
        new = "a4de88679030aedeb0d565b56f1d8613bf33d44600b5a5a8ff54b570f0b518a8"
        frozen, current = {key: old, "utils.py": "x"}, {key: new, "utils.py": "x"}
        self.assertTrue(_compatible_inference_hashes(frozen, current))
        self.assertFalse(_compatible_inference_hashes(frozen, {**current, "utils.py": "y"}))
        self.assertFalse(_compatible_inference_hashes(frozen, {**current, key: "unknown"}))

    def test_seal_does_not_access_data_then_detects_mutation(self):
        self.csv.unlink()
        self.make_seal()
        self.csv.write_text("f,label,group_id\n1,Attack,g1\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "prepared CSV differs"):
            confirm_external(self.seal, self.result)
        self.assertFalse(self.result.exists())

    def test_rejects_development_source_and_declared_group_overlap(self):
        capture = self.config["captures"][0]
        capture["raw_files"][0]["sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "reuses a development source"):
            self.make_seal()
        capture["raw_files"][0]["sha256"] = _sha(self.raw)
        self.csv.write_text("f,label,group_id\n-1,BENIGN,held-out-of-test\n", encoding="utf-8")
        capture["sha256"] = _sha(self.csv)
        self.make_seal()
        with self.assertRaisesRegex(ValueError, "group identifiers overlap"):
            confirm_external(self.seal, self.result)

    def test_whole_group_overlap_exclusion_retains_absent_target_as_na(self):
        self.csv.write_text("f,label,group_id\n8,Attack,g1\n1,Attack,g1\n-1,BENIGN,g2\n", encoding="utf-8")
        self.config["captures"][0]["sha256"] = _sha(self.csv)
        self.config["exact_overlap_policy"] = "exclude_overlapping_groups"
        self.make_seal()
        result = confirm_external(self.seal, self.result)
        self.assertEqual(result["status"], "insufficient_support_or_gate")
        self.assertFalse(result["checks"]["target_groups"])
        audit = (self.result / "capture_support.csv").read_text().splitlines()
        self.assertEqual(audit[1], "external,3,2,1,1,2,1,1,0,1")

    def make_fixed_cases(
        self,
        *,
        guarded=False,
        operating_threshold=0.5,
        operating_reference=None,
        operating_selection=None,
        single_donor_client=None,
    ):
        names, features = ["BENIGN", "Other", "Attack"], ["w", "x", "y", "z"]
        raw = np.arange(144, dtype=np.float32).reshape(36, 4) / 144
        labels = np.arange(36, dtype=np.int64) % 3
        parts = {
            "train": (raw[:24], labels[:24]),
            "cal": (raw[24:30], labels[24:30]),
            "val": (raw[30:], labels[30:]),
        }
        routes = {role: np.arange(len(y)) for role, (_, y) in parts.items()}
        args = SimpleNamespace(
            device="cpu",
            seed=21,
            base_epochs=1,
            batch_size=8,
            adapter_epochs=1,
            adapter_lr=0.1,
            adapter_prox=1,
            prior_strength=0.5,
            target_far=0.01,
            threshold_floor=0.5,
            cal_fit_fraction=0.5,
        )
        node = PrivateClient("client_0", parts, routes, names, features, args, self.root / "working")
        # Uniform base probabilities make every row cross the shared alarm
        # threshold, and break subtype ties toward Other.
        with torch.no_grad():
            for value in node.model.parameters():
                value.zero_()
        self.checkpoint = self.root / "tabm.pt"
        save_tabm_checkpoint(self.checkpoint, node, batch_size=8)
        self.csv.write_text(
            "w,x,y,z,label,group_id\n3,4,5,6,BENIGN,g1\n4,5,6,7,Attack,g2\n7,8,9,10,Attack,g3\n",
            encoding="utf-8",
        )
        self.config["criteria"]["max_far"] = 1.0
        self.config["captures"][0]["sha256"] = _sha(self.csv)
        self.screen = prepare_development_screen(
            self.root / "fixed_development.npy", parts, {"source": "c" * 64}
        )
        self.models = []
        local = np.zeros((3, 5), dtype=np.float32)
        fed = local.copy()
        fed[2, -1] = 1
        fixed_gate = {
            "checkpoint": self.checkpoint,
            "threshold": operating_threshold,
            "gate_disabled": False,
            "routing": FIXED_ROUTING,
        }
        if guarded:
            from src.revision_subtype_guard import fit_subtype_guard

            fit_base = np.array([1, 1])
            fit_labels = np.array([0, 1])
            quiet = np.full((2, 3), 1 / 3)
            strong = np.tile([0.10, 0.05, 0.85], (2, 1))
            regular = fit_subtype_guard(fit_base, quiet, fit_labels, [2], min_negative_groups=2)
            blocked = fit_subtype_guard(fit_base, strong, fit_labels, [2], min_negative_groups=2)
            fixed_gate.update(
                routing=GUARDED_ROUTING,
                subtype_guards={"head_local": regular, "head_federated": regular, "head_donor": blocked},
            )
        if operating_selection is not None:
            fixed_gate.update(
                reference_threshold=operating_selection["receiver_reference_threshold"],
                max_operating_fnr_increase=0.01,
                operating_selection=operating_selection,
            )
        elif operating_reference is not None:
            fixed_gate.update(reference_threshold=operating_reference, max_operating_fnr_increase=0.01)
        protocol = {"rounds": 1}
        if operating_selection is not None:
            protocol.update(
                clients=3,
                operating_reference_cal_far=0.01,
                operating_candidates=operating_selection["candidate_grid"],
                operating_constraints=operating_selection["constraints"],
            )
        vectors = {"head_local": local, "head_federated": fed, "head_donor": fed}
        settings = {"protocol": protocol}
        if single_donor_client is not None:
            vectors[SINGLE_DONOR_MODE] = local
            fixed_gate["subtype_guards"][SINGLE_DONOR_MODE] = fixed_gate["subtype_guards"]["head_local"]
            settings["single_donor_client"] = single_donor_client
        settings["gate_disabled"] = dict.fromkeys(vectors, False)
        for seed in (42, 43):
            dest = self.root / f"fixed-{seed}.npz"
            receipt = save_frozen_head(
                dest,
                dataset="Toy",
                names=names,
                features=features,
                seed=seed,
                receiver=0,
                target_class="Attack",
                development_screen=self.screen,
                vectors=vectors,
                thresholds=dict.fromkeys(vectors, operating_threshold),
                settings=settings,
                fixed_gate=fixed_gate,
            )
            self.assertEqual(receipt["tabm_checkpoint_sha256"], _sha(self.checkpoint))
            self.models.append(dest)

    def test_fixed_pipeline_keeps_gate_and_compares_base_and_donor(self):
        self.make_fixed_cases()
        plan = self.make_seal()
        result = confirm_external(self.seal, self.result)
        self.assertEqual(plan["version"], 2)
        self.assertEqual(result["status"], "criteria_met_on_declared_external_receiver")
        self.assertTrue(result["checks"]["binary_alarm_metrics_equal_to_base"])
        self.assertTrue(result["checks"]["target_f1_gain_over_base"])
        with (self.result / "metrics.csv").open(newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if row["capture_id"] == "ALL_CAPTURES"]
        self.assertEqual(len(rows), 8)
        self.assertEqual({row["far"] for row in rows}, {"1.0"})
        self.assertEqual({row["fnr"] for row in rows}, {"0.0"})
        self.assertEqual(
            {row["mode"] for row in rows}, {"base_threshold", "head_local", "head_federated", "head_donor"}
        )
        self.assertEqual(len({row["ece"] for row in rows}), 1)
        with (self.result / "paired_intervals.csv").open(newline="") as handle:
            contrasts = {row["contrast"] for row in csv.DictReader(handle)}
        self.assertEqual(contrasts, {"aggregation", "against_local_detector", "receiver_participation"})
        self.assertIn("prediction_change", (self.result / "fixed_alarm_transitions.csv").read_text())

    def test_fixed_pipeline_rejects_modified_checkpoint_and_branch_threshold(self):
        self.make_fixed_cases()
        info_path = self.models[0].with_suffix(".json")
        original = json.loads(info_path.read_text())
        changed = {**original, "thresholds": {**original["thresholds"], "head_federated": 0.6}}
        info_path.write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "same TabM threshold"):
            self.make_seal()
        info_path.write_text(json.dumps(original), encoding="utf-8")
        self.make_seal()
        with self.checkpoint.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "checkpoint differs"):
            confirm_external(self.seal, self.result)
        self.assertFalse(self.result.exists())

    def test_guarded_pipeline_replays_a_different_frozen_cutoff_for_each_branch(self):
        self.make_fixed_cases(guarded=True)
        plan = self.make_seal()
        self.assertEqual(plan["version"], 3)
        self.assertIn("revision_subtype_guard.py", plan["analysis_sha256"])
        result = confirm_external(self.seal, self.result)
        self.assertTrue(result["checks"]["binary_alarm_metrics_equal_to_base"])
        self.assertFalse(result["threshold_selection_on_test"])
        with (self.result / "metrics.csv").open(newline="") as handle:
            rows = {
                row["mode"]: row
                for row in csv.DictReader(handle)
                if row["capture_id"] == "ALL_CAPTURES" and row["seed"] == "42"
            }
        self.assertEqual({row["alert_rate"] for row in rows.values()}, {"1.0"})
        self.assertAlmostEqual(float(rows["head_federated"]["target_f1"]), 0.8)
        self.assertEqual(float(rows["head_donor"]["target_f1"]), 0)
        self.assertEqual(float(rows["head_local"]["target_f1"]), 0)
        self.assertEqual(rows["head_federated"]["subtype_guard_status"], "calibrated")
        self.assertNotEqual(
            rows["head_federated"]["subtype_guard_threshold"], rows["head_donor"]["subtype_guard_threshold"]
        )

    def test_guarded_pipeline_rejects_guard_inventory_changes_and_post_seal_mutation(self):
        self.make_fixed_cases(guarded=True)
        info_path = self.models[0].with_suffix(".json")
        info = json.loads(info_path.read_text(encoding="utf-8"))
        guards = info["fixed_gate"]["subtype_guards"]
        fed = guards.pop("head_federated")
        info_path.write_text(json.dumps(info), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "must match the declared head modes"):
            self.make_seal()
        guards["head_federated"] = fed
        info_path.write_text(json.dumps(info), encoding="utf-8")
        self.make_seal()
        guards["head_federated"] = guards["head_donor"]
        info_path.write_text(json.dumps(info), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Seal verification failed"):
            confirm_external(self.seal, self.result)
        self.assertFalse(self.result.exists())

    def test_operating_reference_uses_the_frozen_threshold_and_enforces_fnr_allowance(self):
        self.make_fixed_cases(guarded=True, operating_threshold=0.75, operating_reference=0.5)
        plan = self.make_seal()
        self.assertEqual(plan["version"], 4)
        self.assertEqual(plan["max_operating_fnr_increase"], 0.01)
        result = confirm_external(self.seal, self.result)
        self.assertFalse(result["checks"]["operating_fnr_limit_vs_1pct_reference"])
        self.assertTrue(result["checks"]["operating_fnr_paired_seed_support"])
        self.assertEqual(result["operating_fnr_vs_1pct_reference"]["ci95_high"], 1)
        with (self.result / "metrics.csv").open(newline="") as handle:
            pooled = {
                row["mode"]: row
                for row in csv.DictReader(handle)
                if row["capture_id"] == "ALL_CAPTURES" and row["seed"] == "42"
            }
        self.assertEqual(float(pooled[OPERATING_REFERENCE_MODE]["threshold"]), 0.5)
        self.assertEqual(float(pooled[OPERATING_REFERENCE_MODE]["fnr"]), 0)
        self.assertEqual(float(pooled["head_federated"]["fnr"]), 1)
        self.assertEqual(pooled[OPERATING_REFERENCE_MODE]["ece"], pooled["head_federated"]["ece"])
        self.assertIn(OPERATING_CONTRAST, (self.result / "paired_intervals.csv").read_text())

    def test_operating_reference_values_are_validated_and_sealed(self):
        self.make_fixed_cases(guarded=True, operating_reference=0.5)
        info_path = self.models[0].with_suffix(".json")
        info = json.loads(info_path.read_text(encoding="utf-8"))
        info["fixed_gate"]["max_operating_fnr_increase"] = 1.1
        info_path.write_text(json.dumps(info), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "FNR allowance must be finite rates"):
            self.make_seal()
        info["fixed_gate"]["max_operating_fnr_increase"] = 0.01
        info_path.write_text(json.dumps(info), encoding="utf-8")
        self.make_seal()
        result = confirm_external(self.seal, self.result)
        self.assertTrue(result["checks"]["operating_fnr_limit_vs_1pct_reference"])
        self.assertEqual(result["operating_fnr_vs_1pct_reference"]["ci95_high"], 0)
        info["fixed_gate"]["reference_threshold"] = 0.6
        info_path.write_text(json.dumps(info), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Seal verification failed"):
            confirm_external(self.seal, self.root / "tampered-operating")

    def make_operating_selection(self, *, reference_subtype=False):
        selection = {
            "schema": "cal_operating_selection_v1",
            "status": "selected",
            "reason": None,
            "selection_rule": "loosest_first_descending_nominal_far",
            "candidate_grid": [0.01, 0.005],
            "selected_cal_far": 0.005,
            "reference_cal_far": 0.01,
            "receiver_client": 0,
            "receiver_threshold": 0.75,
            "receiver_reference_threshold": 0.5,
            "thresholds": [0.75, 0.75, 0.75],
            "reference_thresholds": [0.5, 0.5, 0.5],
            "cal_group_assignment_sha256": "a" * 64,
            "group_overlap": 0,
            "selection_sha256": None,
            "constraints": {
                "max_far": 0.01,
                "reference_far": 0.01,
                "max_fnr_increase": 0.01,
                "min_benign_rows": 1,
                "min_select_benign_rows": 1,
                "min_select_attack_rows": 1,
                "min_receiver_attack_rows": 1,
                "min_attack_coverage": 0.5,
                "threshold_floor": 0.5,
                "pooled_benign_fallback": True,
            },
            "support": {"receiver_operating_select_attack_rows": 1},
        }
        if reference_subtype:
            from src.revision_subtype_guard import fit_subtype_guard

            selection["reference_subtype_guard"] = fit_subtype_guard(
                np.array([1, 1]), np.full((2, 3), 1 / 3), np.array([0, 1]), [2], min_negative_groups=2
            )
        selection["selection_sha256"] = frozen_selection_digest(selection)
        return selection

    def test_operating_selection_is_frozen_with_cal_groups_and_replays(self):
        selection = self.make_operating_selection()
        self.make_fixed_cases(guarded=True, operating_threshold=0.75, operating_selection=selection)
        plan = self.make_seal()
        self.assertEqual(plan["version"], 5)
        self.assertEqual(plan["operating_selection_protocol"]["schema"], "cal_operating_selection_v1")
        result = confirm_external(self.seal, self.result)
        self.assertTrue(result["checks"]["cal_operating_selection_complete"])
        info_path = self.models[0].with_suffix(".json")
        info = json.loads(info_path.read_text(encoding="utf-8"))
        info["fixed_gate"]["operating_selection"]["selected_cal_far"] = 0.01
        info_path.write_text(json.dumps(info), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "digest differs"):
            self.make_seal()

    def test_operating_selection_rejects_inconsistent_receipt_even_with_valid_selection_digest(self):
        self.make_fixed_cases(
            guarded=True, operating_threshold=0.75, operating_selection=self.make_operating_selection()
        )
        receipt = self.models[0].with_suffix(".json")
        initial = receipt.read_text(encoding="utf-8")
        for field, value, message in (
            ("receiver_client", 1, "receiver differs"),
            ("max_fnr_increase", 0.02, "FNR allowance differs"),
            ("reference_cal_far", 0.02, "must be 1%"),
            ("reference_far", 0.02, "reference CAL FAR differs"),
            ("clients", 4, "vector length differs"),
            ("operating_reference_cal_far", 0.02, "reference CAL FAR differs"),
            ("receiver_operating_select_attack_rows", 0, "receiver attack support constraint"),
        ):
            with self.subTest(field=field):
                info = json.loads(initial)
                selected = info["fixed_gate"]["operating_selection"]
                if field in ("max_fnr_increase", "reference_far"):
                    selected["constraints"][field] = value
                elif field in ("clients", "operating_reference_cal_far"):
                    info["settings"]["protocol"][field] = value
                elif field == "receiver_operating_select_attack_rows":
                    selected["support"][field] = value
                else:
                    selected[field] = value
                selected["selection_sha256"] = frozen_selection_digest(selected)
                receipt.write_text(json.dumps(info), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    self.make_seal()

    def test_frozen_subtype_reference_reports_and_rejects_regression_on_same_capture(self):
        self.make_fixed_cases(
            guarded=True,
            operating_threshold=0.75,
            operating_selection=self.make_operating_selection(reference_subtype=True),
        )
        plan = self.make_seal()
        self.assertTrue(plan["operating_selection_protocol"]["reference_subtype_nonregression"])
        result = confirm_external(self.seal, self.result)
        self.assertTrue(result["checks"]["reference_subtype_paired_seed_support"])
        self.assertFalse(result["checks"]["target_f1_mean_nonregression_vs_1pct_reference"])
        self.assertFalse(result["checks"]["macro_f1_mean_nonregression_vs_1pct_reference"])
        for metric in ("target_f1", "macro_f1"):
            entry = result["subtype_nonregression_vs_1pct_reference"][metric]
            self.assertLess(entry["mean"], 0)
            self.assertEqual(entry["mean"], entry["ci95_low"])
            self.assertEqual(entry["mean"], entry["ci95_high"])
        self.assertIn(REFERENCE_SUBTYPE_MODE, (self.result / "metrics.csv").read_text())
        self.assertIn(REFERENCE_SUBTYPE_CONTRAST, (self.result / "paired_intervals.csv").read_text())

    def test_seal_cannot_mix_cases_with_and_without_reference_subtype_guard(self):
        self.make_fixed_cases(
            guarded=True,
            operating_threshold=0.75,
            operating_selection=self.make_operating_selection(reference_subtype=True),
        )
        receipt = self.models[0].with_suffix(".json")
        info = json.loads(receipt.read_text(encoding="utf-8"))
        selected = info["fixed_gate"]["operating_selection"]
        selected.pop("reference_subtype_guard")
        selected["selection_sha256"] = frozen_selection_digest(selected)
        receipt.write_text(json.dumps(info), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "must share dataset, target, method protocol"):
            self.make_seal()

    def make_class_operating_selection(self):
        from src.revision_class_operating import SCHEMA, SELECTION_RULE

        selection = self.make_operating_selection(reference_subtype=True)
        selection.update(
            schema=SCHEMA,
            selection_rule=SELECTION_RULE,
            class_routing=CLASS_ALARM_ROUTING,
            class_thresholds=[[0.75, 0.5] for _ in range(3)],
            thresholds=list(selection["reference_thresholds"]),
            receiver_threshold=0.5,
        )
        selection["constraints"].pop("pooled_benign_fallback")
        selection["constraints"]["min_route_benign"] = 1
        selection["selection_sha256"] = frozen_selection_digest(selection)
        return selection

    def test_class_operating_selection_replays_local_attack_route_and_keeps_scalar_reference(self):
        self.make_fixed_cases(guarded=True, operating_selection=self.make_class_operating_selection())
        plan = self.make_seal()
        self.assertEqual(plan["version"], 6)
        self.assertEqual(plan["operating_selection_protocol"]["class_routing"], CLASS_ALARM_ROUTING)
        self.assertIn("revision_class_operating.py", plan["analysis_sha256"])
        result = confirm_external(self.seal, self.result)
        self.assertTrue(result["checks"]["binary_alarm_metrics_equal_to_base"])
        self.assertFalse(result["checks"]["operating_fnr_limit_vs_1pct_reference"])
        self.assertFalse(result["checks"]["target_f1_mean_nonregression_vs_1pct_reference"])
        with (self.result / "metrics.csv").open(newline="") as handle:
            rows = {
                row["mode"]: row
                for row in csv.DictReader(handle)
                if row["capture_id"] == "ALL_CAPTURES" and row["seed"] == "42"
            }
        for mode in ("base_threshold", "head_local", "head_federated", "head_donor"):
            self.assertEqual(rows[mode]["class_alarm_thresholds"], "[0.75, 0.5]")
            self.assertEqual(float(rows[mode]["alert_rate"]), 0)
        for mode in (OPERATING_REFERENCE_MODE, REFERENCE_SUBTYPE_MODE):
            self.assertFalse(rows[mode]["class_alarm_thresholds"])
            self.assertEqual(float(rows[mode]["alert_rate"]), 1)

    def test_class_operating_selection_validates_threshold_width_and_routing(self):
        self.make_fixed_cases(guarded=True, operating_selection=self.make_class_operating_selection())
        receipt = self.models[0].with_suffix(".json")
        initial = receipt.read_text(encoding="utf-8")
        for field, value, message in (
            ("class_thresholds", [[0.75] for _ in range(3)], "ordered attack classes"),
            ("class_thresholds", [[1.0, 1.0], [0.75, 0.5], [0.75, 0.5]], "gate status differs"),
            ("class_routing", "head_argmax", "class alarm routing differs"),
        ):
            with self.subTest(field=field, value=value):
                info = json.loads(initial)
                selection = info["fixed_gate"]["operating_selection"]
                selection[field] = value
                selection["selection_sha256"] = frozen_selection_digest(selection)
                receipt.write_text(json.dumps(info), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    self.make_seal()

    def test_frozen_donor_attribution_compares_each_branch_on_the_same_capture(self):
        selection = self.make_operating_selection(reference_subtype=True)
        selection.update(receiver_threshold=0.5, thresholds=[0.5, 0.5, 0.5])
        selection["selection_sha256"] = frozen_selection_digest(selection)
        self.make_fixed_cases(guarded=True, operating_selection=selection, single_donor_client=1)
        plan = self.make_seal()
        self.assertEqual(plan["version"], 7)
        self.assertEqual([case["single_donor_client"] for case in plan["cases"]], [1, 1])
        confirm_external(self.seal, self.result)
        with (self.result / "metrics.csv").open(newline="") as stream:
            rows = [r for r in csv.DictReader(stream) if r["capture_id"] == "ALL_CAPTURES"]
        self.assertEqual({r["single_donor_client"] for r in rows}, {"1"})
        self.assertEqual({r["alert_rate"] for r in rows}, {"1.0"})
        with (self.result / "paired_intervals.csv").open(newline="") as stream:
            means = {
                r["contrast"]: float(r["mean"])
                for r in csv.DictReader(stream)
                if r["capture_id"] == "ALL_CAPTURES" and r["metric"] == "target_f1"
            }
        self.assertEqual(means[DONOR_POOL_CONTRAST], 0.0)
        self.assertAlmostEqual(means[DONOR_FULL_CONTRAST], 0.8)

    def test_frozen_donor_attribution_rejects_incomplete_or_inconsistent_modes(self):
        selection = self.make_operating_selection(reference_subtype=True)
        self.make_fixed_cases(
            guarded=True, operating_threshold=0.75, operating_selection=selection, single_donor_client=2
        )
        receipt = self.models[0].with_suffix(".json")
        initial = receipt.read_text(encoding="utf-8")
        for donor in (0, 3, True, None):
            with self.subTest(donor=donor):
                info = json.loads(initial)
                info["settings"]["single_donor_client"] = donor
                receipt.write_text(json.dumps(info), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "non-receiver single donor"):
                    self.make_seal()
        info = json.loads(initial)
        del info["thresholds"]["head_donor"]
        receipt.write_text(json.dumps(info), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "donor pool and the single donor"):
            self.make_seal()

    def test_canonical_fingerprints_equal_for_signed_zero_and_nonfinite(self):
        x = np.array([[-0.0, np.nan], [0.0, np.inf], [0.0, -np.inf], [1.0, np.nan]])
        first, second, third, fourth = _fingerprints(x)
        self.assertEqual(first, second)
        self.assertEqual(first, third)
        self.assertNotEqual(first, fourth)


if __name__ == "__main__":
    unittest.main()
