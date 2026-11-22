"""Check supplement comparisons without running detector training."""

import unittest
from types import SimpleNamespace

import numpy as np

from src.experiment_config import DATASETS
from src.revision_supplement import (
    REPAIR_FIXED,
    REPAIR_RESCUE,
    REPAIR_UNION,
    TRANSFER_MODES,
    AttackSubtypeHead,
    TransferScoreDiagnostics,
    add_subtype_mode,
    missing_class_repair,
    rebalance_scale,
    supplement_cases,
    transfer_rows,
)


class SupplementTests(unittest.TestCase):
    def test_profiles_keep_supported_datasets_and_count_training_cases(self):
        for profile, count in (
            ("core", 3),
            ("transfer", 2),
            ("temporal", 1),
            ("quantization", 2),
            ("scale", 12),
            ("all", 17),
        ):
            cases = supplement_cases(DATASETS, profile)
            self.assertEqual(sum(map(len, cases.values())), count)
            self.assertTrue(
                all(dataset == "CIC-UNSW-NB15" for dataset, protocol in cases if protocol == "temporal")
            )

    def test_transfer_partition_parameters_do_not_change_other_profiles(self):
        cases = supplement_cases(DATASETS, "core", transfer_clients=5, transfer_alpha=2.0)
        self.assertEqual(cases["CIC-IDS-2017", "grouped"], [("supplement_transfer", 5, 2.0)])
        self.assertEqual(cases["CIC-UNSW-NB15", "grouped"], [("supplement_transfer", 5, 2.0)])
        self.assertEqual(cases["CIC-UNSW-NB15", "temporal"], [("supplement_temporal", 3, 0.8)])

    def test_scale_quotas_preserve_rows_and_eval_routing(self):
        roster = [
            {"train": np.arange(9), "cal": np.arange(6), "test": np.array([0])},
            {"train": np.array([9]), "cal": np.array([6]), "test": np.array([1])},
            {"train": np.array([10, 11]), "cal": np.array([7]), "test": np.array([2])},
        ]
        moves = rebalance_scale(roster, 42)
        self.assertEqual(moves, {"train": 5, "cal": 3})
        self.assertEqual([len(client["train"]) for client in roster], [4, 4, 4])
        self.assertEqual([len(client["cal"]) for client in roster], [3, 3, 2])
        np.testing.assert_array_equal(np.sort(np.concatenate([c["train"] for c in roster])), np.arange(12))
        np.testing.assert_array_equal(np.sort(np.concatenate([c["cal"] for c in roster])), np.arange(8))
        np.testing.assert_array_equal(np.concatenate([c["test"] for c in roster]), np.arange(3))

    def test_transfer_counts_false_positives_across_receiver_test(self):
        roster = [
            {"train": np.array([0]), "val": np.array([0]), "test": np.array([0, 1, 2])},
            {"train": np.array([1]), "val": np.array([1]), "test": np.array([3])},
        ]
        parts = {"train": (None, np.array([0, 1])), "test": (None, np.array([1, 0, 1, 1]))}
        predictions = {mode: [np.array([0, 1, 1, 0]), np.array([1, 1])] for mode in TRANSFER_MODES}
        rows = transfer_rows(
            parts, roster, [[5, 0], [0, 5]], ["BENIGN", "A"], predictions, np.array([True, True, False, True])
        )
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["test_support"] == 1 and row["predicted"] == 2 for row in rows))
        self.assertTrue(all(row["precision"] == 0.5 and row["recall"] == 1 for row in rows))

    def test_score_pairs_use_novel_targets_and_separate_rank_from_alert(self):
        roster = [
            {"train": np.array([0]), "val": np.array([0]), "test": np.array([0, 1, 2])},
            {"train": np.array([1]), "val": np.array([1]), "test": np.array([3])},
        ]
        parts = {"train": (None, np.array([0, 1])), "test": (None, np.array([1, 1, 1, 2]))}
        names, unseen = ["BENIGN", "A", "B"], np.array([True, True, False, True])
        local_prob = np.array([[0.8, 0.1, 0.1], [0.2, 0.3, 0.5], [0.6, 0.15, 0.25], [0.2, 0.7, 0.1]])
        fed_prob = np.array([[0.8, 0.1, 0.1], [0.3, 0.5, 0.2], [0.6, 0.3, 0.1], [0.2, 0.1, 0.7]])
        diagnostics = TransferScoreDiagnostics(parts, roster, names, unseen)
        predictions = {}
        for mode in TRANSFER_MODES:
            fed = mode.startswith("full")
            guesses = np.array([0, 1, 0, 2] if fed else [0, 2, 0, 1])
            predictions[mode] = [guesses, np.array([0, 2])]
            diagnostics.observe(mode, 0, fed_prob if fed else local_prob, guesses)
            diagnostics.observe(mode, 1, np.array([[0.8, 0.1, 0.1], [0.2, 0.2, 0.6]]), predictions[mode][1])
        hard = transfer_rows(parts, roster, [[5, 0, 4], [5, 6, 4]], names, predictions, unseen)
        scores = diagnostics.paired_rows(hard)
        self.assertEqual(len(scores), 2)
        for row in scores:
            self.assertEqual(row["test_support"], 2)
            self.assertAlmostEqual(row["delta_logp_mean"], (np.log(5 / 3) + np.log(2)) / 2)
            self.assertEqual(row["logp_increase_rate"], 1.0)
            self.assertEqual(row["delta_attack_rank_mean"], -1.0)
            self.assertGreater(row["delta_attack_log_margin_mean"], 0)
            self.assertEqual(row["delta_top1_rate"], 1.0)
            self.assertEqual(row["delta_alert_rate"], 0.0)
            self.assertEqual(row["delta_recall"], 0.5)
            self.assertEqual(row["federated_top1_no_alert_rate"], 0.5)
            self.assertEqual(row["local_alert_wrong_subtype_rate"], 0.5)
        np.testing.assert_allclose(fed_prob[:, 1], [0.1, 0.5, 0.3, 0.1])

    def test_subtype_fallback_and_preservation_of_binary_gate(self):
        y = np.array([0, 2, 2, 4, 4, 4])
        x = np.array([[0.0], [-3.0], [-2.0], [1.0], [2.0], [3.0]])
        fallback = np.array([2] * len(y))
        head = AttackSubtypeHead(x, y, 42)
        self.assertEqual(head.training_rows, 5)
        self.assertEqual(set(head.predict(x, fallback)), {2, 4})
        self.assertIs(AttackSubtypeHead(x, np.ones(len(y), dtype=int), 42).predict(x, fallback), fallback)
        predicted = {"base_threshold": [np.array([0, 2, 4])]}
        scores = {"base_threshold": [np.array([0.2, 0.8, 0.9])]}
        diagnostics = {"calibration": [{"mode": "base_threshold", "threshold": 0.5}]}
        add_subtype_mode(predicted, scores, diagnostics, [np.array([4, 4, 2])])
        np.testing.assert_array_equal(predicted["base_threshold_subtype"][0], [0, 4, 2])
        self.assertIs(scores["base_threshold_subtype"], scores["base_threshold"])

    def test_missing_class_repair_separates_subtype_from_new_alerts(self):
        cal = np.tile([0.8, 0.1, 0.1], (200, 1))
        shared_cal = np.tile([0.7, 0.1, 0.2], (200, 1))
        client = SimpleNamespace(
            cal_labels=np.zeros(200, dtype=int),
            cal_probabilities=cal,
            eval_probabilities=np.array([[0.8, 0.08, 0.12], [0.3, 0.2, 0.5], [0.3, 0.2, 0.5]]),
        )
        args = SimpleNamespace(target_far=0.01, threshold_floor=0.5)
        shared_eval = np.array([[0.2, 0.7, 0.1], [0.2, 0.7, 0.1], [0.2, 0.1, 0.7]])
        base = np.array([0, 2, 2])
        predictions, _, calibration = missing_class_repair(
            client,
            np.arange(200),
            shared_cal,
            shared_eval,
            base,
            np.array([False, True, False]),
            args,
        )
        np.testing.assert_array_equal(predictions[REPAIR_FIXED], [0, 1, 2])
        np.testing.assert_array_equal(predictions[REPAIR_FIXED] > 0, base > 0)
        np.testing.assert_array_equal(predictions[REPAIR_UNION], [1, 1, 2])
        np.testing.assert_array_equal(predictions[REPAIR_RESCUE], [1, 1, 2])
        self.assertEqual(calibration[1]["calibration_far"], 0.0)
        predictions, probabilities, _ = missing_class_repair(
            client,
            np.arange(200),
            shared_cal,
            shared_eval,
            base,
            np.zeros(3, dtype=bool),
            args,
        )
        for values in predictions.values():
            np.testing.assert_array_equal(values, base)
        for values in probabilities.values():
            np.testing.assert_allclose(values, [0.2, 0.7, 0.7])

    def test_routed_probability_forecasts_use_selected_branch_attack_mass(self):
        local_cal = np.tile([0.8, 0.1, 0.1], (200, 1))
        shared_cal = np.tile([0.7, 0.2, 0.1], (200, 1))
        local_eval = np.array([[0.8, 0.1, 0.1], [0.3, 0.2, 0.5], [0.6, 0.1, 0.3], [0.7, 0.2, 0.1]])
        shared_eval = np.array([[0.35, 0.4, 0.25], [0.1, 0.5, 0.4], [0.1, 0.05, 0.85], [0.8, 0.15, 0.05]])
        local_before, shared_before = local_eval.copy(), shared_eval.copy()
        client = SimpleNamespace(
            cal_labels=np.zeros(200, dtype=int), cal_probabilities=local_cal, eval_probabilities=local_eval
        )
        predicted, probabilities, records = missing_class_repair(
            client,
            np.arange(200),
            shared_cal,
            shared_eval,
            np.array([0, 2, 0, 0]),
            np.array([False, True, False]),
            SimpleNamespace(target_far=0.01, threshold_floor=0.5),
        )
        np.testing.assert_array_equal(predicted[REPAIR_FIXED], [0, 1, 0, 0])
        np.testing.assert_array_equal(predicted[REPAIR_RESCUE], [1, 1, 0, 0])
        np.testing.assert_allclose(probabilities[REPAIR_FIXED], [0.2, 0.7, 0.4, 0.3])
        np.testing.assert_allclose(probabilities[REPAIR_UNION], [0.65, 0.9, 0.4, 0.3])
        np.testing.assert_allclose(probabilities[REPAIR_RESCUE], [0.65, 0.7, 0.4, 0.3])
        for probability in probabilities.values():
            self.assertTrue(np.isfinite(probability).all())
            self.assertTrue(((probability >= 0) & (probability <= 1)).all())
            np.testing.assert_allclose(np.column_stack((1 - probability, probability)).sum(axis=1), 1.0)
        for record in records:
            self.assertEqual(record["score_kind"], "selected_branch_binary_probability")
            self.assertIn("calibration", record["score_notes"])
        np.testing.assert_array_equal(local_eval, local_before)
        np.testing.assert_array_equal(shared_eval, shared_before)
        # ECE evaluates this defined forecast, independently of its operational alert gate.
        from src.revision_report import score

        actual = np.array([1, 0, 1, 0])
        metrics, _, _ = score(actual, predicted[REPAIR_RESCUE], probabilities[REPAIR_RESCUE], ["B", "A", "C"])
        self.assertAlmostEqual(metrics["ece"], (0.35 + 0.7 + 0.6 + 0.3) / 4)

    def test_transfer_includes_actual_repair_sensitivity_modes(self):
        roster = [
            {"train": np.array([0]), "val": np.array([0]), "test": np.array([0, 1, 2])},
            {"train": np.arange(1, 6), "val": np.array([1]), "test": np.array([3])},
        ]
        parts = {"train": (None, np.array([0, 1, 1, 1, 1, 1])), "test": (None, np.array([1, 0, 1, 1]))}
        sensitivity = f"{REPAIR_RESCUE}__g0.5_r0.25"
        modes = (*TRANSFER_MODES, "base_threshold", sensitivity)
        predictions = {mode: [np.array([0, 1, 1, 0]), np.array([1, 1])] for mode in modes}
        rows = transfer_rows(
            parts, roster, [[5, 0], [0, 5]], ["BENIGN", "A"], predictions, np.array([True, True, False, True])
        )
        self.assertEqual({row["mode"] for row in rows}, set(modes))
        self.assertEqual(len(rows), len(modes))
        self.assertTrue(all(row["repair_eligible"] for row in rows))
        self.assertTrue(all(row["other_train_rows"] == 5 for row in rows))

    def test_rescue_spends_only_residual_benign_cal_budget(self):
        cal = np.tile([0.8, 0.1, 0.1], (200, 1))
        cal[-2:] = [[0.1, 0.1, 0.8], [0.05, 0.05, 0.9]]
        shared = np.tile([0.7, 0.1, 0.2], (200, 1))
        shared[:2] = [[0.15, 0.8, 0.05], [0.25, 0.7, 0.05]]
        client = SimpleNamespace(
            cal_labels=np.zeros(200, dtype=int),
            cal_probabilities=cal,
            eval_probabilities=cal,
        )
        base = np.zeros(200, dtype=int)
        base[-1] = 2
        predictions, _, calibration = missing_class_repair(
            client,
            np.arange(200),
            shared,
            shared,
            base,
            np.array([False, True, False]),
            SimpleNamespace(target_far=0.01, threshold_floor=0.5),
        )
        np.testing.assert_array_equal(np.flatnonzero(predictions[REPAIR_RESCUE] > 0), [0, 199])
        self.assertEqual(calibration[2]["remaining_benign_cal_false_alerts"], 1)
        self.assertEqual(calibration[2]["rescue_threshold"], 0.7)
        self.assertEqual(calibration[2]["calibration_far"], 0.01)

    def test_gate_fraction_changes_donor_decision_with_complementary_local_budget(self):
        donor_scores = np.linspace(0.1, 0.9, 100)
        shared_cal = np.column_stack((1 - donor_scores, donor_scores, np.zeros(100)))
        local_cal = np.tile([0.8, 0.1, 0.1], (100, 1))
        client = SimpleNamespace(
            cal_labels=np.zeros(100, dtype=int),
            cal_probabilities=local_cal,
            eval_probabilities=np.array([[0.8, 0.1, 0.1]]),
        )
        alerts = []
        for fraction in (0.25, 0.5, 1.0):
            predicted, _, records = missing_class_repair(
                client,
                np.arange(100),
                shared_cal,
                np.array([[0.15, 0.85, 0.0]]),
                np.array([0]),
                np.array([False, True, False]),
                SimpleNamespace(target_far=0.2, threshold_floor=0.5, repair_gate_target_fraction=fraction),
            )
            alerts.append(predicted[REPAIR_UNION][0])
            union = records[1]
            self.assertAlmostEqual(union["local_target_far"] + union["donor_target_far"], 0.2)
            self.assertLessEqual(union["calibration_far"], 0.2)
            for record in records:
                self.assertEqual(record["repair_gate_target_fraction"], fraction)
                self.assertEqual(record["repair_residual_budget_fraction"], 1.0)
                self.assertEqual(record["source_mode"], "base_threshold")
                self.assertEqual(record["donor_kind"], "full_weight_fedavg")
        self.assertEqual(alerts, [0, 1, 1])

    def test_full_donor_budget_disables_local_union_even_above_cal_maximum(self):
        local_cal = np.tile([0.8, 0.1, 0.1], (100, 1))
        shared_cal = np.tile([0.7, 0.1, 0.2], (100, 1))
        client = SimpleNamespace(
            cal_labels=np.zeros(100, dtype=int),
            cal_probabilities=local_cal,
            eval_probabilities=np.array([[0.0, 0.9, 0.1]]),
        )
        predicted, _, records = missing_class_repair(
            client,
            np.arange(100),
            shared_cal,
            np.array([[0.7, 0.1, 0.2]]),
            np.array([1]),
            np.array([False, True, False]),
            SimpleNamespace(target_far=0.01, threshold_floor=0.5, repair_gate_target_fraction=1.0),
        )
        np.testing.assert_array_equal(predicted[REPAIR_UNION], [0])
        np.testing.assert_array_equal(predicted[REPAIR_RESCUE], [1])
        self.assertEqual(records[1]["threshold"], 1.0)
        self.assertEqual(records[1]["local_target_far"], 0.0)
        self.assertEqual(records[1]["local_calibration_far"], 0.0)
        predicted, _, records = missing_class_repair(
            client,
            np.arange(100),
            shared_cal,
            np.array([[0.7, 0.1, 0.2]]),
            np.array([1]),
            np.zeros(3, dtype=bool),
            SimpleNamespace(target_far=0.01, threshold_floor=0.5, repair_gate_target_fraction=1.0),
        )
        np.testing.assert_array_equal(predicted[REPAIR_UNION], [1])
        self.assertEqual(records[1]["local_target_far"], 0.01)
        self.assertEqual(records[1]["donor_target_far"], 0.0)

    def test_residual_fraction_limits_new_alerts_and_zero_disables_rescue(self):
        local_cal = np.tile([0.8, 0.1, 0.1], (400, 1))
        donor_scores = np.full(400, 0.1)
        donor_scores[:4] = [0.9, 0.8, 0.7, 0.6]
        shared = np.column_stack((1 - donor_scores, donor_scores, np.zeros(400)))
        client = SimpleNamespace(
            cal_labels=np.zeros(400, dtype=int), cal_probabilities=local_cal, eval_probabilities=local_cal
        )
        for fraction, expected_alerts in ((0.0, 0), (0.5, 2), (1.0, 4)):
            predicted, _, records = missing_class_repair(
                client,
                np.arange(400),
                shared,
                shared,
                np.zeros(400, dtype=int),
                np.array([False, True, False]),
                SimpleNamespace(
                    target_far=0.01, threshold_floor=0.5, repair_residual_budget_fraction=fraction
                ),
            )
            self.assertEqual(int((predicted[REPAIR_RESCUE] > 0).sum()), expected_alerts)
            self.assertEqual(records[2]["available_benign_cal_false_alerts"], 4)
            self.assertEqual(records[2]["remaining_benign_cal_false_alerts"], expected_alerts)
            if fraction == 0:
                self.assertEqual(records[2]["rescue_threshold"], 1.0)
                # Even a TEST score exceeding every CAL score cannot create a new alert.
                predicted, _, _ = missing_class_repair(
                    client,
                    np.arange(400),
                    shared,
                    np.tile([0.0, 1.0, 0.0], (400, 1)),
                    np.zeros(400, dtype=int),
                    np.array([False, True, False]),
                    SimpleNamespace(
                        target_far=0.01, threshold_floor=0.5, repair_residual_budget_fraction=0.0
                    ),
                )
                self.assertFalse(predicted[REPAIR_RESCUE].any())
        # One existing benign alarm leaves three slots; a half budget permits only one addition.
        local_cal[-1] = [0.1, 0.1, 0.8]
        base = np.zeros(400, dtype=int)
        base[-1] = 2
        predicted, _, records = missing_class_repair(
            client,
            np.arange(400),
            shared,
            shared,
            base,
            np.array([False, True, False]),
            SimpleNamespace(target_far=0.01, threshold_floor=0.5, repair_residual_budget_fraction=0.5),
        )
        np.testing.assert_array_equal(np.flatnonzero(predicted[REPAIR_RESCUE]), [0, 399])
        self.assertEqual(records[2]["available_benign_cal_false_alerts"], 3)
        self.assertEqual(records[2]["remaining_benign_cal_false_alerts"], 1)

    def test_explicit_default_fractions_match_existing_default_behavior(self):
        local = np.tile([0.8, 0.1, 0.1], (200, 1))
        shared = np.tile([0.7, 0.2, 0.1], (200, 1))
        client = SimpleNamespace(
            cal_labels=np.zeros(200, dtype=int),
            cal_probabilities=local,
            eval_probabilities=np.array([[0.3, 0.2, 0.5]]),
        )
        values = []
        for settings in ({}, {"repair_gate_target_fraction": 0.5, "repair_residual_budget_fraction": 1.0}):
            values.append(
                missing_class_repair(
                    client,
                    np.arange(200),
                    shared,
                    np.array([[0.1, 0.8, 0.1]]),
                    np.array([2]),
                    np.array([False, True, False]),
                    SimpleNamespace(target_far=0.01, threshold_floor=0.5, **settings),
                )
            )
        for mode in values[0][0]:
            np.testing.assert_array_equal(values[0][0][mode], values[1][0][mode])
            np.testing.assert_array_equal(values[0][1][mode], values[1][1][mode])
        self.assertEqual(values[0][2], values[1][2])

    def test_invalid_budget_fractions_are_rejected(self):
        client = SimpleNamespace()
        for field, invalid in (
            ("repair_gate_target_fraction", (0.0, -0.1, 1.1, float("nan"))),
            ("repair_residual_budget_fraction", (-0.1, 1.1, float("inf"))),
        ):
            for value in invalid:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    missing_class_repair(
                        client, None, None, None, None, None, SimpleNamespace(**{field: value})
                    )


if __name__ == "__main__":
    unittest.main()
