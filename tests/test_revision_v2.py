"""Regression checks for the prespecified imbalance comparison."""

import unittest
from copy import deepcopy

from src.revision_report import METRICS, balance_comparisons
from src.revision_run import DATASETS, SEEDS, experiment_cases


def _case(seed, scenario, unique_macro_f1):
    supports = [
        {"role": role, "client": "pooled", "class": name, "rows": count}
        for role, counts in (("train", (90, 30)), ("cal", (12, 4)), ("val", (6, 2)), ("test", (6, 2)))
        for name, count in zip(("BENIGN", "ATTACK"), counts)
    ]
    evaluations = []
    for role, counts, macro_f1 in (
        ("val", (6, 2), 0.8),
        ("test", (6, 2), 0.7 if scenario == "main" else 0.6),
        ("test_unique_unseen", (2, 2), unique_macro_f1),
    ):
        for mode in ("base", "local_only", "full"):
            metrics = dict.fromkeys(METRICS, 0.5)
            metrics.update(rows=sum(counts), observed_classes=2, defined_classes=2, macro_f1=macro_f1)
            evaluations.append({
                "role": role,
                "mode": mode,
                "metrics": metrics,
                "per_class": [
                    {"class": name, "support": count} for name, count in zip(("BENIGN", "ATTACK"), counts)
                ],
            })
    return {
        "dataset": "CIC-UNSW-NB15",
        "protocol": "grouped",
        "scenario": scenario,
        "seed": seed,
        "clients": 3,
        "alpha": 0.8,
        "partition_sha256": f"partition-{seed}",
        "settings": {"train_rows": 120, "base_epochs": 5},
        "supports": supports,
        "evaluations": evaluations,
    }


class RevisionV2Tests(unittest.TestCase):
    def test_full_plan_preserves_reference_and_adds_grouped_only(self):
        cases = experiment_cases(DATASETS, "both", "all")
        totals = {"main": 0, "scale": 0, "balance": 0}
        self.assertEqual(len(cases), 8)
        for (dataset, protocol), scenarios in cases.items():
            for scenario, clients, alpha in scenarios:
                key = "scale" if scenario.startswith("scale_") else scenario
                totals[key] += len(SEEDS)
                if key != "scale":
                    self.assertEqual((clients, alpha), (3, 0.8))
                if key == "balance":
                    self.assertEqual(protocol, "grouped")
                    self.assertIn(("main", 3, 0.8), scenarios, dataset)
        self.assertEqual(totals, {"main": 40, "scale": 50, "balance": 20})

    def test_balance_plan_runs_only_grouped_matched_cases(self):
        cases = experiment_cases(DATASETS, "both", "balance")
        self.assertEqual(set(cases), {(dataset, "grouped") for dataset in DATASETS})
        for scenarios in cases.values():
            self.assertEqual(scenarios, [("main", 3, 0.8), ("balance", 3, 0.8)])

    def test_report_pairs_by_seed_and_uses_unique_input_test_endpoint(self):
        cases = [
            _case(43, "balance", 0.55),
            _case(42, "main", 0.4),
            _case(43, "main", 0.5),
            _case(42, "balance", 0.6),
        ]
        rows = balance_comparisons(cases)
        (primary,) = [row for row in rows if row["primary"]]
        self.assertEqual(
            (primary["role"], primary["mode"], primary["metric"]), ("test_unique_unseen", "full", "macro_f1")
        )
        self.assertEqual(primary["seeds_with_metric"], 2)
        self.assertEqual(primary["positive_seeds"], 2)
        self.assertAlmostEqual(primary["reference_mean"], 0.45)
        self.assertAlmostEqual(primary["balanced_mean"], 0.575)
        self.assertAlmostEqual(primary["mean"], 0.125)
        self.assertAlmostEqual(primary["seed_sd"], 0.10606601717798213)
        (other,) = [
            row for row in rows if (row["role"], row["mode"], row["metric"]) == ("test", "full", "macro_f1")
        ]
        self.assertAlmostEqual(other["mean"], -0.1)

    def test_report_rejects_unmatched_partitions_budgets_and_support(self):
        reference = _case(42, "main", 0.4)
        balanced = _case(42, "balance", 0.6)
        different_partition = {**balanced, "partition_sha256": "wrong"}
        different_budget = {**balanced, "settings": {"train_rows": 119, "base_epochs": 5}}
        different_support = deepcopy(balanced)
        changed = different_support["evaluations"][-1]["per_class"]
        changed[0]["support"], changed[1]["support"] = 3, 1
        for case in (different_partition, different_budget, different_support):
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    balance_comparisons([reference, case])
        self.assertEqual(balance_comparisons([reference]), [])
        with self.assertRaises(ValueError):
            balance_comparisons([balanced])


if __name__ == "__main__":
    unittest.main()
