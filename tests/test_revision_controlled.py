"""Protocol invariants, target selection and cache integrity without real training."""

import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from src.revision_controlled import (
    PredictionCache,
    choose_controlled_receiver,
    development_cells,
    load_prediction_cache,
    plan_targets,
    remove_receiver_class,
)
from src.revision_run import _evaluate, allocate_clients


def data():
    names = ["BENIGN", "Rare", "Medium", "Frequent"]
    parts = {}
    for role in ("train", "cal", "val", "test"):
        labels = np.repeat(np.arange(4), [200, 40, 80, 120])
        parts[role] = (np.arange(len(labels) * 2).reshape(-1, 2).astype(float), labels)
    return names, parts


class ControlledTests(unittest.TestCase):
    def test_target_plan_reads_only_train_cal_and_is_stable(self):
        names, parts = data()
        allowed = {role: parts[role] for role in ("train", "cal")}
        first = plan_targets(allowed, names, 3)
        self.assertEqual([row["class"] for row in first["targets"]], names[1:])
        self.assertEqual(first, plan_targets(allowed, names, 3))
        with self.assertRaises(ValueError):
            plan_targets(allowed, names, 3, ["BENIGN"])
        with self.assertRaises(ValueError):
            plan_targets(allowed, names, 4)

    def test_cal_only_absence_keeps_receiver_train_target(self):
        names, parts = data()
        roster, _ = allocate_clients(parts, len(names), 3, 0.3, 42)
        changed, receipt = remove_receiver_class(parts, roster, 1, 0, 42, 0.5, absence_scope="cal_only")
        labels = parts["train"][1]
        self.assertGreater(int((labels[changed[0]["train"]] == 1).sum()), 0)
        self.assertEqual(int((parts["cal"][1][changed[0]["cal"]] == 1).sum()), 0)
        self.assertFalse(receipt["roles"]["train"]["modified"])
        self.assertTrue(receipt["roles"]["cal"]["modified"])
        self.assertIn("TRAIN retains", receipt["absence_scope"])

    def test_grouped_receiver_choice_uses_only_train_cal_and_skips_ineligible_preference(self):
        names, parts = data()
        roster, _ = allocate_clients(parts, len(names), 3, 0.3, 42)
        # Give the preferred client zero target TRAIN support; another client
        # already has both TRAIN/CAL support and must be selected instead.
        moved = roster[0]["train"][parts["train"][1][roster[0]["train"]] == 1]
        roster[0]["train"] = roster[0]["train"][parts["train"][1][roster[0]["train"]] != 1]
        roster[1]["train"] = np.concatenate((roster[1]["train"], moved))
        allowed_parts = {role: parts[role] for role in ("train", "cal")}
        allowed_roster = [{role: client[role] for role in ("train", "cal")} for client in roster]
        receiver, eligibility = choose_controlled_receiver(allowed_parts, allowed_roster, 1, 42, 0.5)
        self.assertNotEqual(receiver, 0)
        self.assertFalse(eligibility["val_test_used"])
        self.assertEqual(eligibility["seed_retries"], 0)
        self.assertFalse(eligibility["candidates_checked"][0]["eligible"])
        self.assertEqual(
            (receiver, eligibility), choose_controlled_receiver(allowed_parts, allowed_roster, 1, 42, 0.5)
        )
        changed, receipt = remove_receiver_class(
            parts, roster, 1, receiver, 42, 0.5, "cal_only", grouped_cal=True
        )
        self.assertGreater(receipt["roles"]["train"]["receiver_target_rows_after"], 0)
        self.assertGreater(receipt["roles"]["cal"]["receiver_target_rows_before"], 0)
        self.assertGreaterEqual(receipt["donor_cal_fit_rows"], 5)
        self.assertEqual(receipt["cal_group_split"]["global_exact_feature_overlap"], 0)
        self.assertEqual(
            receipt["cal_group_split"]["assignment_sha256"], eligibility["cal_group_assignment_sha256"]
        )
        for role in ("train", "cal"):
            np.testing.assert_array_equal(
                np.sort(np.concatenate([client[role] for client in changed])), np.arange(len(parts[role][1]))
            )

    def test_cal_only_requires_real_intervention_and_train_target(self):
        names, parts = data()
        roster, _ = allocate_clients(parts, len(names), 3, 0.3, 42)
        for role in ("train", "cal"):
            invalid = deepcopy(roster)
            invalid[0][role] = invalid[0][role][parts[role][1][invalid[0][role]] != 1]
            with self.assertRaisesRegex(ValueError, "positive target " + role.upper()):
                remove_receiver_class(parts, invalid, 1, 0, 42, 0.5, "cal_only", grouped_cal=True)

    def test_controlled_removal_preserves_global_budget_and_eval_routes(self):
        names, parts = data()
        roster, _ = allocate_clients(parts, len(names), 3, 0.3, 42)
        original = deepcopy(roster)
        changed, receipt = remove_receiver_class(parts, roster, 1, 0, 42, 0.5)
        for role in ("train", "cal"):
            combined = np.concatenate([c[role] for c in changed])
            np.testing.assert_array_equal(np.sort(combined), np.arange(len(parts[role][1])))
            self.assertFalse(np.any(parts[role][1][changed[0][role]] == 1))
        for i in range(3):
            np.testing.assert_array_equal(changed[i]["test"], original[i]["test"])
            for role in parts:
                np.testing.assert_array_equal(roster[i][role], original[i][role])
        self.assertEqual(receipt["client_cal_fit_target_rows"][0], 0)
        self.assertGreaterEqual(receipt["donor_cal_fit_rows"], 5)
        np.testing.assert_array_equal(
            np.sort(np.concatenate([client["val"] for client in changed])), np.arange(len(parts["val"][1]))
        )

    def test_controlled_val_routing_ignores_labels_and_target(self):
        names, parts = data()
        roster, _ = allocate_clients(parts, len(names), 3, 0.3, 42)
        first, _ = remove_receiver_class(parts, roster, 1, 0, 42, 0.5)
        second, _ = remove_receiver_class(parts, roster, 2, 0, 42, 0.5)
        relabeled = {**parts, "val": (parts["val"][0], parts["val"][1][::-1])}
        third, _ = remove_receiver_class(relabeled, roster, 1, 0, 42, 0.5)
        for i in range(3):
            np.testing.assert_array_equal(first[i]["val"], second[i]["val"])
            np.testing.assert_array_equal(first[i]["val"], third[i]["val"])

    def test_missing_donor_support_rejects_without_retry(self):
        names, parts = data()
        x, y = parts["cal"]
        parts["cal"] = (x[:10], y[:10])
        roster, _ = allocate_clients(parts, len(names), 3, 0.3, 42)
        with self.assertRaisesRegex(ValueError, "fewer than five"):
            remove_receiver_class(parts, roster, 1, 0, 42, 0.5)

    def test_val_target_cells_include_donor_and_do_not_use_test_predictions(self):
        parts = {"val": (None, np.array([0, 1, 1]))}
        roster = [{"val": np.arange(3)}]
        predictions = {"full": [np.array([0, 0, 0, 1, 1])], "fedavg_threshold": [np.array([0, 1, 1, 0, 0])]}
        rows = development_cells(parts, roster, predictions, 1, 0, ["BENIGN", "Attack"])
        self.assertEqual(rows[0]["recall"], 0)
        self.assertEqual(rows[1]["recall"], 1)
        self.assertTrue(all(row["support"] == 2 and row["role"] == "val" for row in rows))

    def test_evaluation_can_exclude_test_entirely(self):
        parts = {"val": (None, np.array([0, 1])), "test": (None, np.array([1, 1]))}
        roster = [{"val": np.array([0, 1]), "test": np.array([], dtype=int)}]
        evaluated = _evaluate(
            parts,
            roster,
            {"full": [np.array([0, 1])]},
            {"full": [np.array([0.1, 0.9])]},
            ["BENIGN", "Attack"],
            np.zeros(0, dtype=bool),
            roles=("val",),
        )
        self.assertEqual(len(evaluated), 1)
        self.assertEqual(evaluated[0]["role"], "val")

    def test_private_cache_roundtrip_and_corruption_rejected(self):
        names, parts = data()
        parts["test"] = (parts["test"][0][:0], parts["test"][1][:0])
        roster, _ = allocate_clients(parts, len(names), 3, 0.3, 42)
        with TemporaryDirectory() as temporary:
            cache = PredictionCache(temporary, names, parts, roster, development_only=True)
            local = [
                SimpleNamespace(
                    cal_probabilities=np.full((len(c["cal"]), 4), 0.25),
                    eval_probabilities=np.full((len(c["val"]), 4), 0.25),
                    log_prior=np.zeros(4),
                )
                for c in roster
            ]
            cache.local(local)
            predictions = {"full": [np.zeros(len(c["val"]), dtype=int) for c in roster]}
            scores = {"full": [np.full(len(c["val"]), 0.75) for c in roster]}
            cache.save(predictions, scores)
            meta, arrays = load_prediction_cache(temporary)
            self.assertTrue(meta["private"] and meta["development_only"])
            self.assertEqual(len(arrays["client_0_test_labels"]), 0)
            with (Path(temporary) / "predictions_private.npz").open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                load_prediction_cache(temporary)


if __name__ == "__main__":
    unittest.main()
