"""Validate the planned execution of fresh review experiments."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.revision_final_review import (
    _nb15_status,
    _operations,
    _with_declaration,
    _write_completion,
    check_optional_inputs,
    run_final_review,
    run_iov_review,
)
from src.revision_fresh_source import IOV_ALARM_FIELDS, _verify_fixed_alarms
from src.revision_v12 import _hash, _read, _write


class CurrentReviewTests(unittest.TestCase):
    def test_required_pending_returns_failure_and_optional_pending_does_not(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, state_path, receipt = (root / file for file in ('plan.json', 'state.json', 'complete.json'))
            state = {'benchmark_matrix': {'status': 'pending'}, 'external': {'status': 'pending'}}
            _write(plan, {'stages': list(state)})
            _write(state_path, state)
            self.assertEqual(_write_completion(receipt, plan, state_path, state), 2)
            result = _read(receipt)
            self.assertEqual(result['required_pending_steps'], ['benchmark_matrix'])
            self.assertEqual(result['optional_pending_steps'], ['external'])
            state['benchmark_matrix']['status'] = 'completed'
            _write(state_path, state)
            self.assertEqual(_write_completion(receipt, plan, state_path, state), 0)

    def test_nb15_status_reads_both_specialist_assessments(self):
        for verdict, expected in (
            (True, "development_criteria_met"),
            (False, "development_criteria_not_met"),
            (None, "development_criteria_not_assessed"),
        ):
            with self.subTest(verdict=verdict):
                weight = {"candidates": {"frozen_head": {"development_guard_pass": verdict}}}
                threshold = {"development_guard_pass": verdict}
                self.assertEqual(_nb15_status(weight), expected)
                self.assertEqual(_nb15_status(threshold), expected)

    def test_focused_clicks_fit_iov_in_two_fresh_directories(self):
        with (
            TemporaryDirectory() as temporary,
            patch("src.revision_final_review._operations", side_effect=AssertionError("Unexpected full workflow")),
            patch("src.revision_final_review.run_fresh_iov_diagnosis", return_value={"status": "completed"}) as fit,
            patch("src.revision_final_review.run_fresh_report", return_value={"status": "completed"}) as report,
        ):
            project = Path(temporary).resolve()
            self.assertEqual(run_iov_review(project), 0)
            self.assertEqual(run_iov_review(project), 0)
            roots = [call.args[0].parent for call in fit.call_args_list]
            self.assertEqual(len(set(roots)), 2)
            for index, root in enumerate(roots):
                plan = _read(root / "final_plan.json")
                self.assertEqual(plan["stages"], ["iov_diagnosis", "report"])
                self.assertEqual(plan["input_preflight"], {})
                self.assertFalse(plan["previous_run_inputs_allowed"])
                self.assertEqual(report.call_args_list[index].args[0], root / "report")

    def test_fixed_iov_alarm_parity_requires_ordered_mask_and_score_hashes(self):
        key = ("42", "0", "val")
        base = dict.fromkeys(IOV_ALARM_FIELDS, "matching")
        rows = {(*key, "base_threshold"): base, (*key, "fixed_base_full"): base.copy()}
        _verify_fixed_alarms(rows)
        for field in ("alarm_mask_sha256", "ordered_attack_score_sha256"):
            with self.subTest(field=field):
                rows[(*key, "fixed_base_full")] = {**base, field: "mismatch"}
                with self.assertRaisesRegex(ValueError, "changed the baseline"):
                    _verify_fixed_alarms(rows)

    def test_two_clicks_retrain_in_distinct_directories(self):
        """A populated legacy directory and the first click cannot supply the second run."""
        targeted = {"status": "completed", "execution_origin": "current_training"}
        with (
            TemporaryDirectory() as temporary,
            patch(
                "src.revision_final_review.run_fresh_nb15_pair", return_value={"status": "completed"}
            ) as source,
            patch("src.revision_final_review.run_targeted", return_value=targeted) as train,
            patch(
                "src.revision_final_review.run_fresh_hulk_attribution", return_value={"status": "completed"}
            ) as hulk,
            patch("src.revision_final_review.preflight_hulk_declaration", return_value={"status": "ready"}),
            patch("src.revision_final_review.run_verified_hulk", return_value={"status": "pending"}) as confirm,
            patch("src.revision_final_review.run_lan", return_value={"status": "pending"}),
            patch(
                "src.revision_final_review.run_donor_check_replay", return_value={"status": "completed"}
            ) as replay,
            patch(
                "src.revision_final_review.run_benchmark_matrix", return_value={"status": "completed"}
            ) as benchmark,
            patch(
                "src.revision_final_review.run_alarm_comparison", return_value={"status": "completed"}
            ) as alarm,
            patch("src.revision_final_review.run_loopback", return_value={"status": "completed"}) as loopback,
            patch("src.revision_final_review.run_fresh_iov_diagnosis", return_value={"status": "completed"}) as iov,
            patch("src.revision_final_review._nb15_stage", return_value={"status": "completed"}) as nb15,
            patch("src.revision_final_review.run_scale_matrix", return_value={"status": "completed"}) as scale,
            patch("src.revision_final_review.run_can_quantization", return_value={"status": "completed"}) as quant,
            patch("src.revision_final_review.run_nb15_temporal", return_value={"status": "completed"}) as temporal,
            patch("src.revision_final_review.run_kminus1_transfer", return_value={"status": "completed"}) as transfer,
            patch("src.revision_final_review.run_fresh_report", return_value={"status": "completed"}) as report,
        ):
            project = Path(temporary).resolve()
            declaration = project / "capture.json"
            _write(declaration, {"fixture": True})
            container = project / "outputs" / "fedcaid_review"
            container.mkdir(parents=True)
            previous = container / "final_complete.json"
            previous.write_bytes(b"old result must not be read")
            self.assertEqual(run_final_review(project, hulk_manifest=declaration), 0)
            self.assertEqual(run_final_review(project, hulk_manifest=declaration), 0)
            roots = [call.args[0].parent for call in source.call_args_list]
            self.assertEqual(len(set(roots)), 2)
            for i, root in enumerate(roots):
                self.assertEqual(root.parent, container)
                self.assertFalse(_read(root / "final_complete.json")["previous_run_inputs_allowed"])
                self.assertEqual(train.call_args_list[i].kwargs["pair_source"], root / "nb15_pair")
                self.assertEqual(replay.call_args_list[i].kwargs["pair_source"], root / "nb15_pair")
                self.assertEqual(hulk.call_args_list[i].args[0], root / "hulk_attribution")
                self.assertEqual(confirm.call_args_list[i].args[1], root / "hulk_attribution")
                self.assertEqual(benchmark.call_args_list[i].args[0], root / "benchmark_matrix")
                self.assertEqual(alarm.call_args_list[i].args[0], root / "alarm_comparison")
                self.assertEqual(alarm.call_args_list[i].kwargs["reference"], root / "benchmark_matrix")
                self.assertTrue(alarm.call_args_list[i].kwargs["primary"])
                for runner, name in (
                    (loopback, "loopback"), (iov, "iov_diagnosis"), (scale, "scale_matrix"),
                    (quant, "can_quantization"), (temporal, "nb15_temporal"), (transfer, "kminus1_transfer"),
                ):
                    self.assertEqual(runner.call_args_list[i].args[0], root / name)
                self.assertEqual(report.call_args_list[i].args[0], root / "report")
                plan = _read(root / "final_plan.json")
                self.assertEqual(plan["input_preflight"]["external"]["manifest_sha256"], _hash(declaration))
                self.assertEqual(plan["input_preflight"]["lan"]["status"], "pending")
                self.assertIn("main.py", plan["entry_sha256"])
                self.assertEqual(plan["primary_detector"], "binary_swa")
                self.assertEqual(plan["primary_alarm_policy"], "local_scalar_cal")
                self.assertEqual(plan["primary_stage"], "alarm_comparison")
                self.assertLess(plan["stages"].index("benchmark_matrix"), plan["stages"].index("alarm_comparison"))
                scale_index = plan["stages"].index("scale_matrix")
                self.assertEqual(plan["stages"][scale_index - 2:scale_index], ["nb15_low_far", "nb15_threshold"])
            self.assertEqual(nb15.call_count, 4)
            self.assertEqual(previous.read_bytes(), b"old result must not be read")

    def test_incomplete_targeted_stage_does_not_start_donor_replay(self):
        project = Path("project")
        state = {"targeted": {"status": "failed"}}
        operations = _operations(project, project / "run", {}, state, "auto")
        with patch("src.revision_final_review.run_donor_check_replay") as replay:
            result = operations["donor_check"](Path("donor_check"))
        replay.assert_not_called()
        self.assertEqual(result["status"], "pending")

    def test_incomplete_benchmark_does_not_start_alarm_comparison(self):
        project = Path("project")
        state = {"benchmark_matrix": {"status": "failed"}}
        operations = _operations(project, project / "run", {}, state, "auto")
        with patch("src.revision_final_review.run_alarm_comparison") as compare:
            result = operations["alarm_comparison"](Path("alarm_comparison"))
        compare.assert_not_called()
        self.assertEqual(result["status"], "pending")

    def test_optional_preflight_checks_both_inputs(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            lan = root / "lan.json"
            _write(lan, {"version": 0})
            checks = check_optional_inputs(root / "missing.json", lan)
        self.assertEqual(checks["external"]["status"], "pending")
        self.assertEqual(checks["lan"]["status"], "failed")

    def test_optional_declaration_cannot_change_after_preflight(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "capture.json"
            _write(path, {"original": True})
            ready = {"status": "ready", "manifest": str(path), "manifest_sha256": _hash(path)}
            _write(path, {"changed": True})
            with patch("src.revision_final_review.run_verified_hulk") as run:
                with self.assertRaisesRegex(ValueError, "declaration changed after startup"):
                    _with_declaration(ready, run)
            run.assert_not_called()
