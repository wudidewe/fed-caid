"""Exercise IoV diagnostics against the adapter's original operating points."""

import csv
import json
import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from src.revision_adaptation import AdapterClient, run_adapter_ablations
from src.revision_iov_diagnostics import (
    ALL_MODES,
    FIXED_MODES,
    MODES,
    OPERATING_FILE,
    PAIRED_FILE,
    STATUS_FILE,
    IoVOperatingDiagnostics,
    write_iov_diagnostics,
)
from src.revision_report import score
from src.revision_run import _objective_comparison, _parser, _validate_iov_diagnostics
from src.utils import empirical_alarm_threshold


def _fixture():
    """Build repeated exact groups, a score tie, and distinct CAL roles."""
    cal_labels = np.array([0, 1, 0, 0, 1])
    cal_scores = np.array([0.25, 0.85, 0.2, 0.4, 0.8])
    val_labels = np.array([0, 0, 0, 1, 1, 1])
    val_scores = np.array([0.9, 0.9, 0.5, 0.2, 0.2, 0.8])
    cal_p = np.column_stack((1 - cal_scores, cal_scores))
    val_p = np.column_stack((1 - val_scores, val_scores))
    splits = [(np.array([0, 1]), np.array([2, 3, 4]))]
    parts = {
        "cal": (np.array([[1], [2], [3], [4], [5]]), cal_labels),
        "val": (np.array([[10], [10], [11], [12], [12], [13]]), val_labels),
    }
    roster = [{"cal": np.arange(cal_labels.size), "val": np.arange(val_labels.size)}]
    client = AdapterClient(cal_p, cal_labels, val_p, np.zeros(2), cal_split=splits[0])
    return client, parts, roster, splits


def _adapter_args():
    return Namespace(
        adapter_epochs=1,
        adapter_lr=0.01,
        adapter_prox=0.1,
        clip=1.0,
        prior_strength=0.0,
        rounds=1,
        secure_scale=1000,
        target_far=0.01,
        threshold_floor=0.5,
    )


def _evaluation(parts, score_map, prediction_map):
    """Build matching native VAL metrics from the actual branch outputs."""
    labels = parts["val"][1]
    return [
        {
            "mode": mode,
            "role": "val",
            "metrics": score(labels, prediction_map[mode][0], score_map[mode][0], ["BENIGN", "attack"])[0],
        }
        for mode in MODES
    ]


def _fixed_observation(*, shift_fixed_score=False):
    """Produce matched multi-class evidence with a changed subtype assignment."""
    names = ("BENIGN", "attack_a", "attack_b", "unsupported")
    parts = {
        "cal": (np.arange(4).reshape(-1, 1), np.array([0, 0, 1, 2])),
        "val": (np.arange(10, 14).reshape(-1, 1), np.array([0, 0, 1, 2])),
    }
    roster = [{"cal": np.arange(4), "val": np.arange(4)}]
    splits = [(np.array([0]), np.array([1, 2, 3]))]
    observer = IoVOperatingDiagnostics(parts, roster, splits, class_names=names, fixed_base_alarm=True)
    cal = np.array([[0.9, 0.1, 0, 0], [0.1, 0, 0.9, 0], [0.2, 0.8, 0, 0]])
    _, receipt = empirical_alarm_threshold(cal, parts["cal"][1][1:])
    base_scores = np.array([0.9, 0.1, 0.7, 0.8])
    evaluations = []
    for mode in ALL_MODES:
        if mode in FIXED_MODES:
            q = base_scores + np.array([0.01, 0, 0, 0]) if shift_fixed_score else base_scores
            pred = np.array([2, 0, 1, 2])
            observer.observe(mode, 0, cal, receipt, q, pred, cal_predictions=np.array([0, 1, 2]))
        else:
            q = np.array([0.4, 0.1, 0.7, 0.8]) if mode != "base_threshold" else base_scores
            pred = np.array([int(q[0] > 0.5), 0, 2, 1])
            observer.observe(mode, 0, cal, receipt, q, pred)
        metrics, _, matrix = score(parts["val"][1], pred, q, names)
        evaluations.append({"role": "val", "mode": mode, "metrics": metrics, "matrix": matrix})
    return observer.finish(evaluations)


class IoVOperatingDiagnosticsTests(unittest.TestCase):
    """Check aggregate evidence, strict gates and mode parity."""

    def test_observer_preserves_same_case_probabilities_predictions_and_calibration(self):
        client, parts, roster, splits = _fixture()
        cal_before, val_before = client.cal_probabilities.copy(), client.eval_probabilities.copy()
        observer = IoVOperatingDiagnostics(parts, roster, splits)
        with patch("src.revision_adaptation._federated_round_vector", return_value=np.zeros(1)):
            before = run_adapter_ablations([client], _adapter_args(), modes=MODES)
            after = run_adapter_ablations(
                [client], _adapter_args(), modes=MODES, observe_operating=observer.observe
            )
        for mode in MODES:
            np.testing.assert_array_equal(before[0][mode][0], after[0][mode][0])
            np.testing.assert_array_equal(before[1][mode][0], after[1][mode][0])
        np.testing.assert_array_equal(client.cal_probabilities, cal_before)
        np.testing.assert_array_equal(client.eval_probabilities, val_before)
        self.assertEqual(before[2]["calibration"], after[2]["calibration"])
        evidence = observer.finish(_evaluation(parts, after[1], after[0]))
        self.assertEqual(len(evidence["operating"]), 6)
        self.assertEqual(len(evidence["overlap"]), 6)

    def test_strict_gate_groups_and_concentration_match_actual_counts(self):
        client, parts, roster, splits = _fixture()
        observer = IoVOperatingDiagnostics(parts, roster, splits)
        cal, labels = client.cal_probabilities[2:], client.cal_labels[2:]
        cutoff, receipt = empirical_alarm_threshold(cal, labels)
        values = client.eval_probabilities[:, 1]
        changed = np.array([0.4, 0.4, 0.5, 0.2, 0.2, 0.8])
        scores = {"base_threshold": [values], "local_only": [values], "full": [changed]}
        predictions = {mode: [(rows[0] > cutoff).astype(int)] for mode, rows in scores.items()}
        for mode in MODES:
            observer.observe(mode, 0, cal, receipt, scores[mode][0], predictions[mode][0])
        evidence = observer.finish(_evaluation(parts, scores, predictions))
        hold, val = evidence["operating"][:2]
        self.assertEqual(hold["cal_benign_quantile_before_floor"], 0.4)
        self.assertTrue(hold["cal_all_benign_quantiles_at_or_below_floor"])
        self.assertEqual(
            (val["false_alarm_rows"], val["false_alarm_groups"], val["benign_groups"]), (2, 1, 2)
        )
        self.assertEqual((val["far"], val["group_far"], val["top_false_alarm_group_share"]), (2 / 3, 0.5, 1))
        self.assertEqual(val["equal_to_threshold_rows"], 1)
        self.assertTrue(val["strict_score_alarm_parity"])
        self.assertEqual(len(val["val_prediction_sha256"]), 64)
        overlap = next(
            row for row in evidence["overlap"] if row["role"] == "val" and row["candidate"] == "full"
        )
        self.assertEqual(
            (overlap["benign_alert_rows_removed"], overlap["benign_alert_groups_removed"]), (2, 1)
        )
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            write_iov_diagnostics(output, [{"seed": 42, "iov_diagnostics": evidence}])
            with (output / OPERATING_FILE).open(encoding="utf-8", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 6)

    def test_diagnostic_rejects_a_different_val_alarm_mask(self):
        client, parts, roster, splits = _fixture()
        observer = IoVOperatingDiagnostics(parts, roster, splits)
        cal, labels = client.cal_probabilities[2:], client.cal_labels[2:]
        _, receipt = empirical_alarm_threshold(cal, labels)
        with self.assertRaisesRegex(ValueError, "changed the original alarm mask"):
            observer.observe("full", 0, cal, receipt, client.eval_probabilities[:, 1], np.zeros(6, dtype=int))

    def test_fixed_branches_preserve_scores_and_change_only_alerted_subtypes(self):
        evidence = _fixed_observation()
        rows = {(row["mode"], row["role"]): row for row in evidence["operating"]}
        for fixed in FIXED_MODES:
            self.assertEqual(
                rows[fixed, "val"]["ordered_attack_score_sha256"],
                rows["base_threshold", "val"]["ordered_attack_score_sha256"],
            )
            self.assertEqual(
                rows[fixed, "val"]["alarm_mask_sha256"], rows["base_threshold", "val"]["alarm_mask_sha256"]
            )
        classes = [
            row
            for row in evidence["classes"]
            if row["mode"] == "fixed_base_full" and row["scope"] == "overall"
        ]
        holdout = next(row for row in classes if row["class"] == "attack_a" and row["role"] == "cal_holdout")
        self.assertEqual((holdout["recall"], holdout["f1"]), (1, 1))
        missing = next(row for row in classes if row["class"] == "unsupported" and row["role"] == "val")
        self.assertEqual((missing["support"], missing["recall"]), (0, None))
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_iov_diagnostics(root, [{"seed": 42, "iov_diagnostics": evidence}])
            status = json.loads((root / STATUS_FILE).read_text(encoding="utf-8"))
            self.assertEqual(status["scientific_status"], "development_fixed_alarm_absolute_far_not_met")
            self.assertFalse(status["deployment_promoted"])
            self.assertGreater(status["paired_effects"]["fixed_full_vs_base_val_macro_f1"]["mean"], 0)
            self.assertTrue((root / PAIRED_FILE).is_file())

    def test_fixed_branch_cannot_change_binary_scores_even_if_alarm_mask_is_equal(self):
        with self.assertRaisesRegex(ValueError, "fixed-base score or mask differs"):
            _fixed_observation(shift_fixed_score=True)

    def test_native_flags_and_objective_metadata(self):
        args = _parser().parse_args([
            "--iov-diagnostics",
            "--dataset",
            "CIC-IoV-2024",
            "--protocol",
            "grouped",
            "--scenarios",
            "main",
            "--group-cal",
            "--development-only",
            "--smoke",
        ])
        _validate_iov_diagnostics(args)
        args.cache_predictions = True
        with self.assertRaisesRegex(ValueError, "without --cache-predictions"):
            _validate_iov_diagnostics(args)
        main = {("CIC-IoV-2024", "grouped"): [("main", 3, 0.8)]}
        self.assertEqual(_objective_comparison(main, True), {})
        balanced = {("CIC-IoV-2024", "grouped"): [("main", 3, 0.8), ("balance", 3, 0.8)]}
        self.assertIn("grouped VAL", _objective_comparison(balanced, True)["objective_comparison"]["primary"])


if __name__ == "__main__":
    unittest.main()
