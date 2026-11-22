"""Verify that reporting preserves scoped NB15 counts and failed criteria."""

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.revision_nb15_review_tables import (
    BINARY_TABLE,
    CLASS_BRANCHES,
    CLASS_TABLE,
    FRONTIER_TABLE,
    write_nb15_review_tables,
)
from src.revision_report import write_csv


def _counts(prefix, benign, false_positives, attacks, false_negatives):
    return {
        f"{prefix}_benign_rows": benign,
        f"{prefix}_false_positives": false_positives,
        f"{prefix}_attack_rows": attacks,
        f"{prefix}_false_negatives": false_negatives,
    }


def _diagnostic(client, false_positives, val_false_positives):
    receiver = client == 2
    return {
        "seed": 42,
        "client": client,
        "receiver": receiver,
        "role": "val",
        "mode": "fixed_federated",
        "cal_scope": "parent_cal_holdout",
        "threshold": 0.57,
        "class_names": json.dumps({0: "BENIGN", 1: "Rare", 2: "Target"}),
        "target_class": "Target",
        **_counts("cal", 90 if receiver else 10, false_positives, 4 if receiver else 2, 1),
        **_counts("val", 100, val_false_positives, 20, 3),
    }


def _binary_pair(scope, receiver_mismatch=False):
    receiver = scope == "receiver"
    reference = _pair_branch(100 if receiver else 200, 20 if receiver else 40, 1, 1)
    candidate = _pair_branch(100 if receiver else 200, 20 if receiver else 40, 2, 0)
    if receiver_mismatch:
        candidate["receiver"] = 8
    return {
        "seed": 42,
        "scope": scope,
        "delta_fnr": 0,
        **{f"reference_{key}": value for key, value in reference.items()},
        **{f"candidate_{key}": value for key, value in candidate.items()},
    }


def _pair_branch(benign, attack, false_positives, cal_false_positives):
    return {
        "benign_rows": benign,
        "attack_rows": attack,
        "far": false_positives / benign,
        "fnr": 0.15,
        "receiver": 2,
        "cal_far": cal_false_positives / 90,
    }


def _binary_status():
    return {
        "criteria": {"max_far_each_seed": 0.01, "max_paired_fnr_increase_ci95_upper": 0.01},
        "scopes": {
            scope: {"candidate_minus_reference_fnr": {"ci95_high": 0.04}, "fnr_pass": False}
            for scope in ("overall", "receiver")
        },
        "classification_guard": {"candidate_minus_reference_macro_f1": {"mean": -0.01}, "pass": False},
        "combined_acceptance": False,
        "selection_on_val": False,
    }


def _metric_rows(branch, scope, support, predicted=0, true_positive=0):
    values = {
        "support": support,
        "predicted": predicted,
        "recall": true_positive / support if support else None,
        "precision": true_positive / predicted if predicted else (0 if support else None),
        "f1": 2 * true_positive / (support + predicted) if support else None,
    }
    return [
        {"seed": 42, "branch": branch, "scope": scope, "metric": field, "value": value}
        for field, value in values.items()
    ]


def _frontier_audit(branch, status, cutoff):
    enabled = status == "enabled"
    return {
        "seed": 42,
        "branch": branch,
        "receipt": json.dumps({
            "status": status,
            "selected_threshold": cutoff,
            "threshold": cutoff if enabled else None,
            "receiver_check": {"negative_groups": 40, "proposed_wrong_rewrite_groups": 3} if cutoff else None,
        }),
    }


class NB15ReviewTableTests(unittest.TestCase):
    """Exercise publication-ready rows without using experimental data."""

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.tables = self.root / "tables"

    def _write(self, filename, rows):
        path = self.tables / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        write_csv(path, rows)

    def _read(self, filename):
        with (self.root / filename).open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def _binary_inputs(self, receiver_mismatch=False):
        self._write(
            "nb15_pair/reference/binary_diagnostics.csv", [_diagnostic(2, 1, 1), _diagnostic(1, 2, 0)]
        )
        self._write(
            "nb15_pair/candidate/binary_diagnostics.csv", [_diagnostic(2, 0, 2), _diagnostic(1, 2, 0)]
        )
        pairs = [_binary_pair("overall"), _binary_pair("receiver", receiver_mismatch)]
        self._write("nb15_pair/binary_pair_by_seed.csv", pairs)
        (self.tables / "nb15_pair/binary_pair_status.json").write_text(json.dumps(_binary_status()))

    def test_binary_uses_actual_cal_and_val_counts_for_each_scope(self):
        self._binary_inputs()
        self.assertEqual(write_nb15_review_tables(self.root, self.tables), {BINARY_TABLE: 2})
        overall, receiver = self._read(BINARY_TABLE)
        self.assertEqual(overall["reference_cal_far"], "0.03")
        self.assertEqual(overall["reference_cal_fnr"], str(2 / 6))
        self.assertEqual(receiver["reference_cal_far"], str(1 / 90))
        self.assertEqual(receiver["candidate_val_far"], "0.02")
        self.assertEqual(receiver["candidate_val_far_pass"], "False")
        self.assertEqual(receiver["reference_val_far_pass"], "True")
        self.assertEqual(receiver["fnr_ci95_upper"], "0.04")
        self.assertEqual(receiver["combined_acceptance"], "False")

    def test_binary_rejects_a_receiver_pair_from_another_client(self):
        self._binary_inputs(receiver_mismatch=True)
        with self.assertRaisesRegex(ValueError, "different receivers"):
            write_nb15_review_tables(self.root, self.tables)

    def test_missing_inputs_yield_no_derived_tables(self):
        self.assertEqual(write_nb15_review_tables(self.root, self.tables), {})

    def _donor_inputs(self):
        self._write("nb15_pair/reference/binary_diagnostics.csv", [_diagnostic(2, 1, 1)])
        metrics = [
            row
            for branch in CLASS_BRANCHES
            for row in [
                *_metric_rows(branch, "receiver_class:BENIGN", 100, 100, 97),
                *_metric_rows(branch, "receiver_class:Rare", 0),
                *_metric_rows(branch, "receiver_class:Target", 10),
            ]
        ]
        full, local = "donor_checked_head_federated", "donor_checked_head_local"
        metrics += _metric_rows("base_threshold", "receiver", 10)
        metrics += _metric_rows(full, "receiver", 10, 3, 2) + _metric_rows(local, "receiver", 10)
        self._write("donor_check/metrics.csv", metrics)
        self._write(
            "donor_check/guard_audit.csv",
            [
                _frontier_audit(full, "enabled", 0.3),
                _frontier_audit(local, "no_feasible_donor_threshold", None),
            ],
        )

    def test_class_and_donor_tables_keep_zeroes_and_actual_val_tp(self):
        self._donor_inputs()
        expected = {CLASS_TABLE: 9, FRONTIER_TABLE: 2}
        self.assertEqual(write_nb15_review_tables(self.root, self.tables), expected)
        rows = self._read(CLASS_TABLE)
        rare = [row for row in rows if row["class"] == "Rare"]
        self.assertEqual(
            [(r["receiver_client"], r["support"], r["recall"], r["true_positive"]) for r in rare],
            [("2", "0", "", "0")] * 3,
        )
        enabled, blocked = self._read(FRONTIER_TABLE)
        self.assertEqual((enabled["donor_status"], enabled["receiver_status"]), ("feasible", "enabled"))
        self.assertEqual((enabled["val_target_tp_delta"], enabled["val_target_support"]), ("2", "10"))
        self.assertEqual(blocked["receiver_status"], "not_checked")


if __name__ == "__main__":
    unittest.main()
