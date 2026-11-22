"""Check isolated V7 external orchestration without accessing real captures."""

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.revision_followup_hulk import RESULT_FILES, run_hulk
from src.revision_v12 import _hash


class HulkFollowupTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / "v12_hulk_attribution"
        self.output = self.root / "followup"
        self.manifest = self.root / "manifest.json"
        self.capture = self.root / "capture.csv"
        self.raw = self.root / "raw.capture"
        self.version = 7

    def _declare(self):
        self.manifest.write_text('{"fixture": true}', encoding="utf-8")
        self.capture.write_bytes(b"independent capture fixture")
        self.raw.write_bytes(b"independent raw fixture")
        stage = self.source / "attribution"
        stage.mkdir(parents=True)
        inventory = []
        for seed in range(42, 47):
            path = stage / f"seed_{seed}" / "head_frozen.npz"
            path.parent.mkdir()
            path.write_bytes(f"model {seed}".encode())
            checkpoint = path.with_name('tabm_receiver.pt')
            checkpoint.write_bytes(f'detector {seed}'.encode())
            route = f'{seed:064x}'
            partition = f'{seed + 1:064x}'
            allocation = f'{seed + 2:064x}'
            path.with_suffix(".json").write_text(
                json.dumps({
                    "version": 7,
                    "dataset": "CIC-IDS-2017",
                    "target_class": "DoS Hulk",
                    "receiver": 0,
                    "seed": seed,
                    'sha256': _hash(path),
                    'settings': {
                        'route_sha256': route,
                        'cal_partition_sha256': partition,
                        'single_donor_client': 1,
                    },
                    'fixed_gate': {
                        'checkpoint': str(checkpoint),
                        'sha256': _hash(checkpoint),
                        'operating_selection': {'cal_group_assignment_sha256': allocation},
                    },
                    "thresholds": {
                        "head_local": 0.5,
                        "head_federated": 0.5,
                        "head_donor": 0.5,
                        "head_donor_single": 0.5,
                    },
                }),
                encoding="utf-8",
            )
            result = path.with_name('result.json')
            result.write_text(
                json.dumps({
                    'identity': {
                        'dataset': 'CIC-IDS-2017',
                        'scenario': 'train_absence',
                        'target_class': 'DoS Hulk',
                        'seed': seed,
                    },
                    'status': 'evaluated',
                    'audit': {
                        'test_evaluated': False,
                        'route_sha256': route,
                        'receiver': 0,
                        'policies': {
                            'reserved': {
                                'partition_sha256': partition,
                                'operating_audit': {'assignment_sha256': allocation},
                            },
                        },
                    },
                    'model_spec': {'donor_subsets': {'head_donor_single': [1]}},
                    'frozen_head_artifacts': {
                        'reserved': {
                            'file': str(path),
                            'sha256': _hash(path),
                            'receipt': str(path.with_suffix('.json')),
                            'receipt_sha256': _hash(path.with_suffix('.json')),
                            'tabm_checkpoint': str(checkpoint),
                            'tabm_checkpoint_sha256': _hash(checkpoint),
                            'seed': seed,
                            'single_donor_client': 1,
                        },
                    },
                }),
                encoding='utf-8',
            )
            inventory.append({
                "dataset": "CIC-IDS-2017",
                "scenario": "train_absence",
                "target_class": "DoS Hulk",
                "cal_policy": "reserved",
                "status": "evaluated",
                "seed": seed,
                "receiver": 0,
                'single_donor_client': 1,
                'partition_sha256': partition,
                'result': str(result.relative_to(stage)),
                'result_sha256': _hash(result),
                "frozen_head": str(path.relative_to(stage)),
            })
        with (stage / "case_status.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(inventory[0]))
            writer.writeheader()
            writer.writerows(inventory)
        (stage / "run.json").write_text("{}", encoding="utf-8")
        audit = stage / "CIC-IDS-2017" / "data_audit.json"
        audit.parent.mkdir()
        audit.write_text('{"scope": "test fixture"}', encoding="utf-8")
        (stage / "complete.json").write_text(
            json.dumps({
                "task": "train-absence",
                "mode": "evaluation",
                "smoke": False,
                "test_evaluated": False,
                "planned_case_policies": 5,
                "evaluated": 5,
            }),
            encoding="utf-8",
        )
        (self.source / "v12_plan.json").write_text(
            json.dumps({"mode": "attribution", "scope": "development"}), encoding="utf-8"
        )
        (self.source / "v12_state.json").write_text(
            json.dumps({
                "attribution": {
                    name: _hash(stage / name)
                    for name in (
                        "run.json",
                        "complete.json",
                        "case_status.csv",
                        "CIC-IDS-2017/data_audit.json",
                    )
                }
            }),
            encoding="utf-8",
        )
        (self.source / "v12_complete.json").write_text(
            json.dumps({
                "mode": "attribution",
                "scope": "development",
                "status": "completed",
                "test_evaluated": False,
                "plan_sha256": _hash(self.source / "v12_plan.json"),
                "state_sha256": _hash(self.source / "v12_state.json"),
            }),
            encoding="utf-8",
        )
        self.capture_record = {
            "prepared_csv": str(self.capture),
            "sha256": _hash(self.capture),
            "raw_files": [[str(self.raw), _hash(self.raw)]],
        }

    def _plan(self, heads, manifest):
        return {
            "version": self.version,
            "dataset": "CIC-IDS-2017",
            "target_class": "DoS Hulk",
            "seeds": list(range(42, 47)),
            "manifest": str(manifest),
            "manifest_sha256": _hash(manifest),
            "cases": [{"file": str(head), "sha256": _hash(head)} for head in heads],
            "captures": [self.capture_record],
        }, []

    def _seal(self, heads, manifest, seal):
        plan, _ = self._plan(heads, manifest)
        with seal.open("x", encoding="utf-8") as stream:
            json.dump(plan, stream)
        return plan

    def _confirm(self, seal, result):
        result.mkdir()
        for name in RESULT_FILES - {"complete.json"}:
            (result / name).write_text("fixture results", encoding="utf-8")
        (result / "complete.json").write_text(
            json.dumps({
                "seal": str(seal),
                "seal_sha256": _hash(seal),
                "seeds": list(range(42, 47)),
                "captures": 1,
                "detector_training": False,
                "threshold_selection_on_test": False,
                "status": "criteria_not_met",
            }),
            encoding="utf-8",
        )

    def _workflow(self, confirm=None):
        patches = (
            patch("src.revision_external._plan", side_effect=self._plan),
            patch("src.revision_external.seal_confirmation", side_effect=self._seal),
            patch("src.revision_external.confirm_external", side_effect=confirm or self._confirm),
        )
        for item in patches:
            self.addCleanup(item.stop)
        return [item.start() for item in patches]

    def _run(self):
        return run_hulk(self.output, self.source, self.manifest)

    def test_missing_manifest_is_pending_without_source_reads_or_writes(self):
        with patch(
            "src.revision_followup_hulk._source_artifacts", side_effect=AssertionError("Unexpected source read")
        ):
            state = self._run()
        self.assertEqual(state["status"], "pending")
        self.assertEqual(state["required_manifest"], str(self.manifest))
        self.assertFalse(self.output.exists())

    def test_completed_v7_reuses_results_even_when_scientific_criteria_fail(self):
        self._declare()
        original = {str(p): p.read_bytes() for p in self.source.rglob("*") if p.is_file()}
        _, seal, confirm = self._workflow()
        first = self._run()
        self.assertEqual(first["status"], "completed")
        self.assertEqual(first["assessment"], "criteria_not_met")
        self.assertEqual(first["frozen_version"], 7)
        self.assertTrue(first["scoring"])
        self.assertTrue(first["capture_contents_read"])
        self.assertEqual(first["confirmation"]["capture_count"], 1)
        self.assertEqual(first["confirmation"]["seed_count"], 5)
        self.assertFalse(first["confirmation"]["criteria_met"])
        self.assertEqual(first, self._run())
        self.assertEqual(first["artifacts"]["seal.json"], _hash(self.output / "seal.json"))
        for path, contents in original.items():
            self.assertEqual(Path(path).read_bytes(), contents)
        self.assertEqual(seal.call_count, 1)
        confirm.assert_called_once_with(self.output / "seal.json", self.output / "result")

    def test_version_five_and_changed_source_are_rejected_before_confirmation(self):
        self._declare()
        _, seal, confirm = self._workflow()
        self.version = 5
        self.assertIn("version-7", self._run()["reason"])
        self.assertFalse(self.output.exists())
        self.version = 7
        (self.source / "attribution" / "run.json").write_text('{"changed": true}', encoding="utf-8")
        self.assertIn("source artifact changed", self._run()["reason"])
        seal.assert_not_called()
        confirm.assert_not_called()

    def test_changed_nested_role_audit_fails_before_sealing(self):
        self._declare()
        _, seal, confirm = self._workflow()
        audit = self.source / "attribution" / "CIC-IDS-2017" / "data_audit.json"
        audit.write_text('{"changed": true}', encoding="utf-8")
        self.assertIn("source artifact changed", self._run()["reason"])
        self.assertFalse(self.output.exists())
        seal.assert_not_called()
        confirm.assert_not_called()

    def test_changed_manifest_or_model_cannot_reseal_completed_run(self):
        self._declare()
        _, seal, confirm = self._workflow()
        self.assertEqual(self._run()["status"], "completed")
        original_seal = (self.output / "seal.json").read_bytes()
        self.manifest.write_text('{"fixture": false}', encoding="utf-8")
        self.assertIn("seal differs", self._run()["reason"])
        self.manifest.write_text('{"fixture": true}', encoding="utf-8")
        (self.source / "attribution" / "seed_42" / "head_frozen.npz").write_bytes(b"changed model")
        self.assertIn('source artifact changed', self._run()['reason'])
        self.assertEqual((self.output / "seal.json").read_bytes(), original_seal)
        seal.assert_called_once()
        confirm.assert_called_once()

    def test_nested_checkpoint_and_result_hashes_are_verified_on_first_run(self):
        self._declare()
        _, seal, confirm = self._workflow()
        case = self.source / 'attribution' / 'seed_42'
        for name in ('result.json', 'head_frozen.json', 'tabm_receiver.pt'):
            with self.subTest(artifact=name):
                path = case / name
                original = path.read_bytes()
                path.write_bytes(original + b'\n')
                self.assertIn('source artifact changed', self._run()['reason'])
                path.write_bytes(original)
        self.assertFalse(self.output.exists())
        seal.assert_not_called()
        confirm.assert_not_called()

    def test_unreceipted_result_requires_matching_recomputation(self):
        self._declare()
        _, _, confirm = self._workflow()
        self.assertEqual(self._run()['status'], 'completed')
        receipt = self.output / 'receipt.json'
        receipt.unlink()
        self.assertEqual(self._run()['status'], 'completed')
        self.assertTrue(receipt.is_file())
        receipt.unlink()
        table = self.output / 'result' / 'metrics.csv'
        table.write_text('changed metrics', encoding='utf-8')
        self.assertIn('differs from sealed rescoring', self._run()['reason'])
        self.assertFalse(receipt.exists())
        self.assertEqual(table.read_text(encoding='utf-8'), 'changed metrics')
        self.assertEqual(confirm.call_count, 3)

    def test_changed_capture_or_result_is_not_replaced(self):
        self._declare()
        _, _, confirm = self._workflow()
        self.assertEqual(self._run()["status"], "completed")
        self.capture.write_bytes(b"changed capture")
        self.assertIn("capture file differs", self._run()["reason"])
        self.capture.write_bytes(b"independent capture fixture")
        artifact = self.output / "result" / "paired_intervals.csv"
        artifact.write_text("changed result", encoding="utf-8")
        self.assertIn("receipt or artifacts changed", self._run()["reason"])
        self.assertEqual(artifact.read_text(encoding="utf-8"), "changed result")
        confirm.assert_called_once()

    def test_partial_confirmation_is_preserved_and_not_restarted(self):
        self._declare()

        def interrupt(seal, result):
            result.mkdir()
            (result / "metrics.csv").write_text("partial", encoding="utf-8")
            raise RuntimeError("interrupted result write")

        _, seal, confirm = self._workflow(interrupt)
        self.assertIn("interrupted result write", self._run()["reason"])
        original_seal = (self.output / "seal.json").read_bytes()
        retry = self._run()
        self.assertEqual(retry["status"], "failed")
        self.assertIn("existing outputs were preserved", retry["reason"])
        self.assertEqual((self.output / "seal.json").read_bytes(), original_seal)
        self.assertFalse((self.output / "receipt.json").exists())
        seal.assert_called_once()
        confirm.assert_called_once()

    def test_existing_unrelated_output_and_source_overlap_are_preserved(self):
        self._declare()
        _, seal, confirm = self._workflow()
        self.output.mkdir()
        sentinel = self.output / "v12_complete.json"
        sentinel.write_text("old V12 result", encoding="utf-8")
        self.assertIn("new output directory", self._run()["reason"])
        self.assertEqual(list(self.output.iterdir()), [sentinel])
        self.output = self.source / "followup"
        self.assertIn("disjoint", self._run()["reason"])
        self.assertFalse(self.output.exists())
        seal.assert_not_called()
        confirm.assert_not_called()

    def test_malformed_source_receipt_returns_failure_without_writing_output(self):
        self._declare()
        (self.source / "v12_complete.json").write_text("[]", encoding="utf-8")
        state = self._run()
        self.assertEqual(state["status"], "failed")
        self.assertIn("JSON objects", state["reason"])
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
