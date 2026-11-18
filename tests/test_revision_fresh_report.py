"""Restrict one-click reports to evidence from the run being reported."""

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.revision_benchmark_matrix import _baseline_row
from src.revision_fresh_report import (
    ABLATION_TABLE,
    BRANCHES,
    CLASS_SUPPORT_TABLE,
    MATRIX_DATASETS,
    OPTIONAL_PEERS,
    STAGE_LABELS,
    STAGE_TIMING_TABLE,
    _branch_seed,
    _nb15_donor_lines,
    regenerate_verified_report,
    run_fresh_report,
)
from src.revision_v12 import _hash, _read, _write


def _stages(root):
    """Create current-run stage records with one completed synthetic table."""
    records = {stage: {"output": str(root / stage), "status": "pending"} for stage in STAGE_LABELS}
    nb15 = root / "nb15_pair"
    nb15.mkdir()
    table = nb15 / "metrics.csv"
    table.write_bytes(b"metric,value\nmicro_f1,0.70\n")
    records["nb15_pair"].update(status="completed", artifacts={"metrics.csv": _hash(table)})
    records["donor_check"].update(scientific_status="not_evaluated", reason="waiting for dependency")
    return records, table


def _bind(root, stages, stage, sources):
    """Write a fixture stage and bind it as if it completed in this run."""
    bindings = {}
    for name, contents in sources.items():
        path = root / stage / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(contents, dict):
            _write(path, contents)
        else:
            path.write_text(contents, encoding="utf-8")
        bindings[name] = _hash(path)
    stages[stage].update(status="completed", artifacts=bindings)


def _csv_rows(path):
    """Read a generated compact CSV."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _branch_fixture():
    """Create one receiver with complete, independently keyed branch records."""
    case = {"seed": "42", "clients": "1", "receiver": "0", "partition_sha256": "split"}
    seed = {"seed": case["seed"]}
    shared = {"support": "8", "precision": "0.5", "recall": "0.5", "f1": "0.5", "receiver_far": "0.01"}
    records = {
        "metrics": [
            {**seed, "mode": mode, "macro_f1": "0.52", "far": "0.01", "fnr": "0.2"} for mode in BRANCHES
        ],
        "target_metrics": [{**seed, **shared, "mode": mode, "client": "client_0"} for mode in BRANCHES],
        "binary_diagnostics": [
            {**seed, "mode": "guarded_federated", "client": "0", "val_far": "0.01", "val_fnr": "0.3"}
        ],
        "subtype_audit": [
            {**seed, "mode": mode, "client": "0", "base_alarm_mask_sha256": "mask"}
            for mode in BRANCHES
            if mode != "base_threshold"
        ],
        "communication": [
            {**seed, "mode": payload, "round_payload_bytes": index, "scope": "logical"}
            for index, (_, payload) in enumerate(BRANCHES.values())
            if payload
        ],
    }
    return case, records


def _binary_assessment():
    """Set distinct FAR, FNR and macro-F1 decisions for the report fixture."""
    return {
        "status": "development_binary_criteria_met",
        "combined_acceptance": False,
        "criteria": {"max_far_each_seed": 0.01, "max_paired_fnr_increase_ci95_upper": 0.01},
        "scopes": {
            "receiver": {
                "candidate_far_max": 0.008,
                "reference_far_seeds_over_limit": 4,
                "candidate_far_seeds_over_limit": 0,
                "candidate_minus_reference_fnr": {"seeds_with_metric": 5, "ci95_high": 0.006},
                "far_pass": True,
                "fnr_pass": True,
            }
        },
        "classification_guard": {
            "candidate_minus_reference_macro_f1": {"mean": -0.002, "ci95_low": -0.01, "ci95_high": 0},
            "pass": False,
        },
    }


def _finalized_iov(root):
    """Create one finalized diagnostic source without model or dataset dependencies."""
    stage = "iov_diagnosis"
    stages = {stage: {"output": str(root / stage), "status": "pending", "test_evaluated": False}}
    _bind(
        root,
        stages,
        stage,
        {
            "iov_operating_diagnostics.csv": (
                "seed,mode,role,target_far,benign_rows,false_alarm_rows,benign_groups,attack_rows,missed_attack_rows\n"
                "42,base_threshold,cal_holdout,0.01,100,0,10,10,0\n"
                "42,base_threshold,val,0.01,100,2,20,10,1\n"
            )
        },
    )
    stages["report"] = run_fresh_report(root / "report", stages)
    _write(root / "final_plan.json", {"run_directory": str(root), "stages": list(stages)})
    _write(root / "final_state.json", stages)
    _write(
        root / "final_complete.json",
        {
            "status": "completed",
            "run_directory": str(root),
            "plan_sha256": _hash(root / "final_plan.json"),
            "state_sha256": _hash(root / "final_state.json"),
            "steps": {
                name: {k: v for k, v in record.items() if k != "artifacts"} for name, record in stages.items()
            },
        },
    )


def _bind_benchmark_tables(root, stages):
    """Bind native rows for one ablation and two sparse classes."""
    prefix = "CIC-UNSW-NB15,grouped,main"
    _bind(root, stages, "benchmark_matrix", {
        "paired_differences.csv": (
            "dataset,protocol,scenario,clients,alpha,role,contrast,metric,n,mean,ci95_low,ci95_high\n"
            f"{prefix},3,0.8,val,full_minus_without_threshold,macro_f1,5,0.02,0.01,0.03\n"
            f"{prefix},3,0.8,val,full_minus_without_threshold,far,5,-0.005,-0.01,0\n"
            f"{prefix},3,0.8,test,full_minus_without_threshold,fnr,5,0.05,0.03,0.07\n"
        ),
        "per_class.csv": (
            "dataset,protocol,scenario,role,mode,seed,class,support,precision,recall,f1\n"
            f"{prefix},val,full,42,Absent,0,,,\n"
            f"{prefix},val,full,42,Worms,26,0,0,0\n"
            f"{prefix},val,full,43,Worms,26,0.1,0.2,0.1333\n"
        ),
    })


class FreshReportTests(unittest.TestCase):
    def test_reports_ablation_units_missing_class_support_and_resumed_elapsed(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            stages, _ = _stages(root)
            _bind_benchmark_tables(root, stages)
            _bind(root, stages, "can_quantization", {"seed_42/result.json": {"seed": 42}})
            stages["can_quantization"]["elapsed_seconds"] = 12.5
            case = "can_quantization/seed_42/result.json"
            _write(root / "final_plan.json", {"stages": list(stages)})
            _write(root / "continuation_plan.json", {
                "original_plan_sha256": _hash(root / "final_plan.json"),
                "preserved_stages": ["benchmark_matrix"],
                "preserved_case_sha256": {case: _hash(root / case)},
            })
            run_fresh_report(root / "report", stages)
            self._assert_benchmark_summaries(root / "report")

    def _assert_benchmark_summaries(self, report):
        row, = _csv_rows(report / ABLATION_TABLE)
        self.assertEqual(
            (row["macro_f1_mean_pp"], row["far_ci95_low_pp"], row["fnr_paired_seeds"]), ("2.0", "-1.0", "0")
        )
        unsupported, zero = _csv_rows(report / CLASS_SUPPORT_TABLE)
        self.assertEqual((unsupported["supported_seeds"], unsupported["recall_mean"]), ("0", ""))
        self.assertEqual((zero["zero_recall_seeds"], zero["recall_mean"]), ("1", "0.1"))
        timing, = _csv_rows(report / STAGE_TIMING_TABLE)
        self.assertEqual(
            (timing["elapsed_scope"], timing["saved_case_bindings"], timing["concurrent_workload"]),
            ("Continuation attempt", "1", "unrecorded"),
        )

    def test_exports_only_bound_current_run_tables(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            stages, table = _stages(root)
            stages = {name: row for name, row in stages.items() if name not in OPTIONAL_PEERS}
            result = run_fresh_report(root / "report", stages)
            report = root / "report"
            exported = report / "tables/nb15_pair/metrics.csv"
            summary = (report / "SUMMARY.md").read_text(encoding="utf-8")
            self.assertEqual(result["execution_origin"], "current_run_evidence_only")
            self.assertEqual(exported.read_bytes(), table.read_bytes())
            self.assertEqual(_read(report / "evidence_plan.json")["sources"], {str(table): _hash(table)})
            self.assertEqual(_hash(report / "complete.json"), result["artifacts"]["complete.json"])
            self.assertIn("pending | not_evaluated | waiting for dependency", summary)

    def test_refuses_tampered_stage_bytes_without_publishing(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            stages, table = _stages(root)
            table.write_bytes(b"overwritten")
            with self.assertRaisesRegex(ValueError, "Current-run artifact changed"):
                run_fresh_report(root / "report", stages)
            self.assertFalse((root / "report").exists())

    def test_refuses_to_import_another_runs_stage(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            stages, table = _stages(root)
            stages["targeted"].update(
                status="completed", output=str(table.parent), artifacts={table.name: _hash(table)}
            )
            with self.assertRaisesRegex(ValueError, "assigned current-run directory"):
                run_fresh_report(root / "report", stages)
            self.assertFalse((root / "report").exists())

    def test_native_status_and_paired_evidence_share_the_primary_cal_condition(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            stages, _ = _stages(root)
            _bind(
                root,
                stages,
                "hulk_attribution",
                {
                    "attribution/guarded_subtype_decisions.csv": (
                        "cal_policy,primary_cal_policy,status,max_receiver_far\n"
                        "strict,False,sensitivity_only,0.020\n"
                        "reserved,True,development_signal_requires_external_confirmation,0.008\n"
                    ),
                    "attribution/paired_intervals.csv": (
                        "cal_policy,contrast,scope,metric,seeds_with_metric,mean,ci95_low,ci95_high\n"
                        "strict,guarded_aggregation,receiver,f1,5,0.999,0.999,1.000\n"
                        "reserved,guarded_aggregation,receiver,f1,5,0.3421,0.0348,0.6494\n"
                    ),
                },
            )
            _bind(
                root,
                stages,
                "nb15_pair",
                {
                    "binary_pair_status.json": _binary_assessment(),
                    "candidate/run.json": {"settings": {"detector_objective": "hard_benign"}},
                    "candidate/guarded_subtype_decisions.csv": (
                        "cal_policy,primary_cal_policy,status\n"
                        "reserved,True,subtype_gain_absolute_far_not_met\n"
                    ),
                    "candidate/paired_intervals.csv": (
                        "cal_policy,contrast,scope,metric,seeds_with_metric,mean,ci95_low,ci95_high\n"
                        "reserved,guarded_aggregation,receiver,f1,5,0.0140,0.0001,0.0279\n"
                    ),
                },
            )
            report = root / "report"
            run_fresh_report(report, stages)
            status = _read(report / "stage_status.json")["hulk_attribution"]
            decisions = status["native_development_decisions"]
            self.assertEqual([row["max_receiver_far"] for row in decisions], ["0.008"])
            rows = _csv_rows(report / "paired_evidence.csv")
            self.assertEqual([row["mean"] for row in rows], ["0.0140", "0.3421"])
            summary = (report / "SUMMARY.md").read_text(encoding="utf-8")
            self.assertIn(
                "completed | attribution — development_signal_requires_external_confirmation", summary
            )
            self.assertIn("completed | binary — development_binary_criteria_met", summary)
            self.assertIn("binary and macro-F1 — criteria not met", summary)
            self.assertIn("| receiver | 0.800% | 4/5 → 0/5 | +0.600 pp | met / met |", summary)
            self.assertIn("-0.0020 [-0.0100, +0.0000]. Classification guard missed.", summary)
            self.assertIn("(candidate, hard_benign) | Federated − local subtype", summary)
            self.assertIn("+0.3421 [+0.0348, +0.6494]", summary)
            self.assertNotIn("sensitivity_only", summary)

    def test_links_only_receipt_bound_role_audits_and_labels_plan_hash_separately(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            stages, _ = _stages(root)
            _write(root / "final_plan.json", {"source_sha256": {"source.py": "test"}})
            _write(root / "nb15_pair/data_audit.json", {"evaluation_scope": "unbound"})
            _bind(
                root,
                stages,
                "targeted",
                {
                    "data_audit.json": {
                        "evaluation_scope": "grouped development validation",
                        "role_rows": {"train": 12, "cal": 6, "val": 4, "test": 3},
                        "exact_feature_audit": {
                            "cross_role_exact_overlap": {"train:cal": {"unique_shared_inputs": 0}},
                            "within_role_repeats": {"cal": {"rows_beyond_one_per_distinct_input": 2}},
                        },
                    }
                },
            )
            report = root / "report"
            run_fresh_report(report, stages)
            rows = _csv_rows(report / "data_role_audit.csv")
            self.assertEqual([row["stage"] for row in rows], ["targeted"])
            self.assertEqual(rows[0]["cal_repeated_rows"], "2")
            self.assertEqual(rows[0]["audited_role_pairs"], "1")
            audit_path = root / "targeted/data_audit.json"
            evidence = _read(report / "evidence_plan.json")
            self.assertEqual(evidence["sources"][str(audit_path)], _hash(audit_path))
            self.assertEqual(evidence["run_plan_sha256"], _hash(root / "final_plan.json"))
            summary = (report / "SUMMARY.md").read_text(encoding="utf-8")
            self.assertIn("[Run plan and code hashes](../final_plan.json)", summary)
            self.assertIn("[Verified stage artifact hashes](evidence_plan.json)", summary)

    def test_hulk_branches_share_one_alarm_mask_and_keep_distinct_payloads(self):
        case, records = _branch_fixture()
        rows = list(_branch_seed(records, case))
        self.assertEqual([row["mode"] for row in rows], list(BRANCHES))
        self.assertEqual({row["receiver_fnr"] for row in rows}, {"0.3"})
        self.assertEqual({row["receiver_alarm_mask_sha256"] for row in rows}, {"mask"})
        self.assertEqual([row["round_payload_bytes"] for row in rows], list(range(6)))
        records["subtype_audit"][-1]["base_alarm_mask_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "alarm masks are not matched"):
            list(_branch_seed(records, case))

    def test_reports_same_run_matrix_and_independent_capture_coverage(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            stages, _ = _stages(root)
            _bind(root, stages, "benchmark_matrix", {"paper_baseline.csv": "dataset,metric\nsynthetic,f1\n"})
            stages["benchmark_matrix"]["test_evaluated"] = False
            _bind(root, stages, "external", {"result/complete.json": {"criteria_met": False}})
            stages["external"].update(scoring=True, confirmation={"capture_count": 1, "seed_count": 5})
            report = root / "report"
            run_fresh_report(report, stages)
            coverage = _csv_rows(report / "evidence_coverage.csv")
            matrix = [row for row in coverage if row["evidence"].startswith("Dirichlet label skew")]
            self.assertEqual([row["dataset"] for row in matrix], list(MATRIX_DATASETS))
            self.assertEqual(
                {(row["execution"], row["heldout_scored"]) for row in matrix}, {("completed", "no")}
            )
            independent = next(row for row in coverage if row["protocol"] == "Declared capture holdout")
            self.assertEqual((independent["execution"], independent["heldout_scored"]), ("completed", "yes"))
            summary = (report / "SUMMARY.md").read_text(encoding="utf-8")
            self.assertIn("Four-dataset K−1 benchmark metrics", summary)
            self.assertIn("A single capture provides no between-capture interval.", summary)
            self.assertTrue((report / "tables/external/result/complete.json").is_file())

    def test_donor_summary_does_not_substitute_cal_counts_for_val_recovery(self):
        rows = [
            {"donor_status": "feasible", "receiver_status": "enabled", "val_target_tp_delta": "4"},
            {
                "donor_status": "feasible",
                "receiver_status": "receiver_check_budget_exceeded",
                "val_target_tp_delta": "-2",
            },
        ]
        rendered = "\n".join(_nb15_donor_lines(rows))
        self.assertIn("donor-feasible cutoff in 2/2", rendered)
        self.assertIn("Receiver CAL approved it in 1/2", rendered)
        self.assertIn("net target true-positive change versus base TabM was +2", rendered)

    def test_standalone_iov_and_refresh_leave_finalized_source_immutable(self):
        with TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            source, output = parent / "source", parent / "refresh"
            _finalized_iov(source)
            before = _hash(source / "report/SUMMARY.md")
            receipt = regenerate_verified_report(source, output)
            self.assertEqual(_hash(source / "report/SUMMARY.md"), before)
            self.assertEqual(receipt["execution_origin"], "verified_source_report_refresh")
            self.assertEqual(receipt["trained_cases"], 0)
            manifest = _read(output / "refresh_source.json")
            self.assertEqual(
                manifest["source_receipt_sha256"]["final_complete.json"],
                _hash(source / "final_complete.json"),
            )
            self.assertEqual(_hash(output / "source_run_plan.json"), _hash(source / "final_plan.json"))
            summary = (output / "SUMMARY.md").read_text(encoding="utf-8")
            self.assertIn("| base_threshold | val | 20–20 | 2.000% | 1/1 | 10.000% |", summary)
            self.assertIn("Reporting-only export with zero cases trained.", summary)

    def test_refresh_rejects_tampered_original_report_and_control_receipts(self):
        with TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            source, output = parent / "source", parent / "refresh"
            _finalized_iov(source)
            (source / "report/SUMMARY.md").write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Current-run artifact changed"):
                regenerate_verified_report(source, output)
            self.assertFalse(output.exists())

    def test_backend_benchmark_stratifies_absent_class_zero_recall_and_payload(self):
        base = {"protocol": "grouped", "scenario": "main", "clients": 3, "alpha": 0.8}
        metrics = [
            {**base, "observed_classes": 2, "defined_classes": 3, "far": value} for value in ("0.01", "0.021")
        ]
        classes = [
            {"class": name, "support": support, "recall": recall}
            for name, support, recall in (("a", 0, ""), ("b", 10, "0"), ("c", 4, "0.2"))
        ]
        payload = [{"round_payload_bytes": 50, "training_payload_bytes": 150, "scope": "logical"}]
        row = _baseline_row("candidate", "full", metrics, classes, payload, 0.01)
        self.assertEqual((row["observed_classes"], row["defined_classes"]), (2, 3))
        self.assertEqual(row["zero_support_classes"], "a")
        self.assertEqual(row["zero_recall_classes_any_seed"], "b")
        self.assertEqual((row["far_max"], row["far_seeds_over_cap"]), (0.021, 1))
        self.assertEqual(row["round_payload_bytes_mean"], 50)

    def test_iov_fixed_subtype_report_uses_bound_criteria_and_class_support(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            stage = "iov_diagnosis"
            stages = {stage: {"status": "pending", "output": str(root / stage), "test_evaluated": False}}
            status = {
                "scientific_status": "development_fixed_alarm_absolute_far_not_met",
                "val_far_cap": 0.01,
                "val_max_far": 0.03,
                "val_seeds_over_far_cap": 4,
                "val_seed_count": 5,
                "paired_effects": {
                    "fixed_full_vs_adapted_cal_fnr": {"mean": 0.1, "ci95_low": -0.01, "ci95_high": 0.21},
                },
            }
            _bind(
                root,
                stages,
                stage,
                {
                    "iov_fixed_mask_status.json": status,
                    "iov_class_metrics.csv": (
                        "role,scope,mode,class,support\n"
                        "val,overall,fixed_base_full,BENIGN,50\n"
                        "val,overall,fixed_base_full,GAS,0\n"
                    ),
                },
            )
            stages[stage]["scientific_status"] = status["scientific_status"]
            report = root / "report"
            run_fresh_report(report, stages)
            summary = (report / "SUMMARY.md").read_text(encoding="utf-8")
            self.assertIn("Fixed-base VAL maximum 3.000% with 4/5 seeds above the 1.00% cap", summary)
            self.assertIn("VAL class support covered 1/2 classes. Unobserved classes were GAS.", summary)
            self.assertIn(
                "CAL FNR, fixed federated minus original federated | +0.1000 [-0.0100, +0.2100]", summary
            )
            self.assertTrue((report / "tables/iov_diagnosis/iov_fixed_mask_status.json").is_file())

    def test_new_peer_stages_keep_explored_test_and_loopback_scopes_visible(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            stages, _ = _stages(root)
            _bind(root, stages, "scale_matrix", {"metrics.csv": "role,far\nval,0.02\n"})
            stages["scale_matrix"].update(test_evaluated=False, cases=90)
            _bind(
                root, stages, "can_quantization", {"frozen/quantization_diagnostics.csv": "setting\nfine\n"}
            )
            stages["can_quantization"].update(test_evaluated=True, cases=20)
            _bind(
                root,
                stages,
                "nb15_temporal",
                {
                    "metrics_by_seed.csv": "role,f1\ntest,0.6\n",
                    "stage_complete.json": {"test_evaluated": True},
                },
            )
            stages["nb15_temporal"].update(test_evaluated=True, cases=5)
            _bind(
                root,
                stages,
                "kminus1_transfer",
                {
                    "transfer_score_intervals.csv": "metric,mean\ndelta_logp,0.0\n",
                },
            )
            stages["kminus1_transfer"].update(test_evaluated=True)
            _bind(
                root,
                stages,
                "loopback",
                {
                    "loopback_summary.csv": (
                        "dimension,clients,measured_rounds,round_median_ms,round_p95_ms,"
                        "numeric_share_payload_bytes_per_round,tcp_application_mean_bytes_per_round,"
                        "all_plaintext_quantized_sums_equal,all_clients_accepted\n"
                        "690,3,30,0.7,1.1,100,260,True,True\n"
                    )
                },
            )
            report = root / "report"
            run_fresh_report(report, stages)
            ledger = _csv_rows(report / "evidence_coverage.csv")
            by_protocol = {row["protocol"]: row for row in ledger}
            self.assertEqual(
                by_protocol["Chronological split with exact-input purge"]["heldout_scored"], "yes"
            )
            self.assertEqual(by_protocol["Grouped and frozen benchmark splits"]["heldout_scored"], "yes")
            self.assertEqual(by_protocol["Same-host TCP loopback"]["heldout_scored"], "not applicable")
            self.assertEqual(by_protocol["Declared capture holdout"]["execution"], "pending")
            summary = (report / "SUMMARY.md").read_text(encoding="utf-8")
            self.assertIn("90 completed cases.", summary)
            self.assertIn("CAN adapter quantization mechanics (frozen)", summary)
            self.assertIn("previously explored part of the same benchmark source", summary)
            self.assertIn("Original K−1 paired score and rank effects", summary)
            self.assertIn("Every measured round matched the plaintext quantized sum", summary)
            self.assertIn("| 690 | 3 | 30 | 0.700 | 1.100 | 100 | 260.0 |", summary)
            self.assertTrue((report / "tables/nb15_temporal/stage_complete.json").is_file())
