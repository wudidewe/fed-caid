"""Verify class-balanced attack-only training and its matched controls."""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from src.revision_shared_head import (
    ATTACK_ONLY,
    BALANCED_ATTACK,
    DONOR,
    FEDERATED,
    LOCAL,
    MULTICLASS,
    fit_shared_heads,
)


def _args(**options):
    return SimpleNamespace(**{
        "seed": 42,
        "device": "cpu",
        "batch_size": 8,
        "rounds": 1,
        "head_epochs": 1,
        "head_lr": 0.5,
        "clip": 0.1,
        **options,
    })


class BalancedAttackHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_one_batch_weights_equalize_class_mass_without_benign_gradient(self):
        y = np.array([0, 1, 1, 1, 1, 2])
        x = np.zeros((len(y), 1), dtype=np.float32)
        data = [(x, y)]
        result = fit_shared_heads(data, 3, _args(head_objective=BALANCED_ATTACK), modes=(LOCAL,))
        unweighted = fit_shared_heads(data, 3, _args(head_objective=ATTACK_ONLY), modes=(LOCAL,))
        np.testing.assert_array_equal(result["states"][LOCAL][0], np.zeros((3, 2)))
        unweighted_bias = unweighted["states"][LOCAL][0][:, -1]
        np.testing.assert_allclose(unweighted_bias, [0, 0.1 / np.sqrt(2), -0.1 / np.sqrt(2)], atol=1e-6)
        self.assertEqual(result["model_spec"]["local_train_class_support"], [[1, 4, 1]])
        self.assertEqual(result["model_spec"]["settings"]["objective"], BALANCED_ATTACK)
        self.assertEqual(unweighted["model_spec"]["settings"]["objective"], ATTACK_ONLY)
        for output in (result, unweighted):
            self.assertEqual(output["performance"][0]["training_row_visits"], 5)
            self.assertEqual(output["performance"][0]["optimizer_steps"], 1)

    def test_attack_controls_match_visits_and_keep_benign_only_client(self):
        x = np.array([[0.0], [100.0], [200.0], [1000.0], [-100.0]], dtype=np.float32)
        train = [
            (x, np.zeros(5, dtype=np.int64)),
            (x, np.array([0, 1, 1, 1, 2])),
            (x, np.array([0, 2, 1, 1, 2])),
        ]
        for objective in (ATTACK_ONLY, BALANCED_ATTACK):
            with self.subTest(objective=objective):
                result = fit_shared_heads(train, 3, _args(head_objective=objective, batch_size=3), receiver=0)
                states, performance = result["states"], result["performance"]
                np.testing.assert_array_equal(states[LOCAL][0], np.zeros((3, 2)))
                self.assertGreater(float(np.linalg.norm(states[FEDERATED][0])), 0)
                for client_states in states.values():
                    for state in client_states:
                        np.testing.assert_array_equal(state[0], np.zeros(2))
                np.testing.assert_allclose(states[FEDERATED][0], np.mean(states[LOCAL], axis=0), atol=1e-8)
                np.testing.assert_allclose(states[DONOR][0], np.mean(states[LOCAL][1:], axis=0), atol=1e-8)
                self.assertEqual(
                    [row["training_row_visits_by_client"] for row in performance], [[0, 4, 4]] * 3
                )
                self.assertEqual([row["optimizer_steps"] for row in performance], [4, 4, 4])
                (round_row,) = (row for row in result["rounds"] if row["mode"] == FEDERATED)
                self.assertEqual(round_row["participating_clients"], 3)
                self.assertEqual(round_row["participating_clients_without_training_rows"], 1)
                self.assertEqual(round_row["raw_update_l2_min"], 0.0)
                self.assertGreater(round_row["raw_update_l2_max"], round_row["raw_update_l2_mean"])
                self.assertTrue(round_row["all_clients_accepted"])

    def test_default_multiclass_and_explicit_multiclass_match(self):
        x = np.array([[0.0], [1.0], [-1.0]], dtype=np.float32)
        train = [(x, np.array([0, 1, 2]))]
        default = fit_shared_heads(train, 3, _args(), modes=(LOCAL,))
        explicit = fit_shared_heads(train, 3, _args(head_objective=MULTICLASS), modes=(LOCAL,))
        np.testing.assert_array_equal(default["states"][LOCAL][0], explicit["states"][LOCAL][0])
        self.assertEqual(default["performance"][0]["training_row_visits"], 3)
        with self.assertRaises(ValueError):
            fit_shared_heads(train, 3, _args(head_objective="unknown"), modes=(LOCAL,))


if __name__ == "__main__":
    unittest.main()
