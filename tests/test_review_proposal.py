"""Regression checks for configuration isolation and fixed-cell metrics."""

import csv
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from src.revision_report import _write_supplement
from src.revision_run import _parser, execute
from src.revision_supplement import TRANSFER_MODES, supplement_cases


def transfer_case(seed=42, alpha=0.3, clients=3, delta=0.5):
    rows = []
    for mode in TRANSFER_MODES:
        for label in ("A", "B"):
            recall = delta if mode.startswith("full") and label == "A" else 0.0
            tp = round(10 * recall)
            rows.append({
                "client": "client_0",
                "class": label,
                "mode": mode,
                "local_train_rows": 0,
                "local_cal_fit_rows": 0,
                "other_cal_fit_rows": 5,
                "transfer_scope": "train_and_fit_absent",
                "test_support": 10,
                "predicted": tp,
                "true_positive": tp,
                "recall": recall,
                "precision": 1.0 if tp else None,
                "f1": 2 * tp / (10 + tp),
            })
    return {
        "dataset": "CIC-IDS-2017",
        "protocol": "grouped",
        "scenario": "supplement_transfer",
        "seed": seed,
        "clients": clients,
        "alpha": alpha,
        "settings": {"train_sampling": "all_rows"},
        "transfer": rows,
    }


def read_csv(root, name):
    with (root / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


class ReviewProposalTests(unittest.TestCase):
    def test_alpha_clients_and_gate_settings_never_count_as_extra_seeds(self):
        results = [
            transfer_case(seed, alpha, delta=delta)
            for alpha, delta in ((0.3, 0.2), (0.8, 0.8))
            for seed in (42, 43)
        ]
        results += [transfer_case(seed, clients=5) for seed in (42, 43)]
        gated = [transfer_case(seed) for seed in (42, 43)]
        for result in gated:
            result["settings"].update(repair_gate_target_fraction=0.25, repair_residual_budget_fraction=1.0)
        results += gated
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = _write_supplement(root, results)
            strict = [
                row
                for row in summary
                if row["transfer_scope"] == "train_and_fit_absent"
                and row["contrast"] == "full_minus_local_only"
            ]
            self.assertEqual(len(strict), 4)
            self.assertTrue(all(row["seeds_total"] == 2 for row in strict))
            means = {
                row["alpha"]: row["mean"]
                for row in strict
                if row["clients"] == 3 and row["repair_gate_target_fraction"] is None
            }
            self.assertAlmostEqual(means[0.3], 0.1)
            self.assertAlmostEqual(means[0.8], 0.4)

    def test_supported_zero_prediction_cells_stay_in_macro_precision(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_supplement(root, [transfer_case()])
            (row,) = [r for r in read_csv(root, "transfer_metrics_by_seed.csv") if r["mode"] == "full"]
            self.assertEqual(row["eligible_cells"], "2")
            self.assertEqual(row["zero_prediction_cells"], "1")
            self.assertAlmostEqual(float(row["macro_precision"]), 0.5)
            self.assertAlmostEqual(float(row["macro_recall"]), 0.25)
            self.assertAlmostEqual(float(row["macro_f1"]), 1 / 3)
            (local,) = [
                r for r in read_csv(root, "transfer_metrics_by_seed.csv") if r["mode"] == "local_only"
            ]
            self.assertEqual(float(local["macro_precision"]), 0.0)

    def test_duplicate_seed_configuration_is_rejected(self):
        result = transfer_case()
        with TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                _write_supplement(Path(temporary), [result, deepcopy(result)])

    def test_single_configuration_transfer_entry_and_validation(self):
        parser = _parser(supplement=True)
        cli = parser.parse_args([
            "--profile",
            "transfer",
            "--transfer-alpha",
            "0.8",
            "--transfer-clients",
            "5",
            "--repair-missing-classes",
            "--repair-gate-target-fraction",
            "1.0",
        ])
        cases = supplement_cases(
            cli.dataset, cli.profile, transfer_alpha=cli.transfer_alpha, transfer_clients=cli.transfer_clients
        )
        self.assertTrue(
            all(case == ("supplement_transfer", 5, 0.8) for values in cases.values() for case in values)
        )
        for options in (
            ["--transfer-alpha", "nan"],
            ["--transfer-clients", "1"],
            ["--repair-gate-target-fraction", "nan"],
            ["--repair-gate-target-fraction", "0"],
            ["--repair-residual-budget-fraction", "1.1"],
            ["--repair-gate-target-fraction", "0.25"],
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                execute(parser.parse_args(["--profile", "transfer", *options]))


if __name__ == "__main__":
    unittest.main()
