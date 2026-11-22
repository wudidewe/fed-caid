"""Cover matched TRAIN visits, fixed-point aggregation and frozen inference."""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from src.revision_shared_head import (
    ATTACK_ONLY,
    DONOR,
    FEDERATED,
    LOCAL,
    SINGLE_DONOR,
    fit_shared_heads,
    head_probabilities,
)


def _fixture():
    negative = np.array([[-100.0, -5.0], [-200.0, -10.0], [-1000.0, -30.0]], dtype=np.float32)
    positive = -negative
    train = [
        (negative, np.zeros(len(negative), dtype=np.int64)),
        (np.concatenate([negative, positive]), np.array([0, 0, 0, 1, 1, 1])),
        (np.concatenate([negative, positive]), np.array([0, 0, 0, 1, 1, 1])),
    ]
    args = SimpleNamespace(
        seed=42,
        device="cpu",
        batch_size=3,
        rounds=1,
        head_epochs=1,
        head_lr=0.5,
        clip=0.1,
        secure_scale=1_000_000,
    )
    return train, positive, args


class SharedHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_one_round_matches_clipped_local_updates_and_tracks_donor(self):
        train, features, args = _fixture()
        result = fit_shared_heads(train, 2, args, receiver=0)
        states, performance = result["states"], result["performance"]
        np.testing.assert_allclose(states[FEDERATED][0], np.mean(states[LOCAL], axis=0), atol=1e-8)
        np.testing.assert_allclose(states[DONOR][0], np.mean(states[LOCAL][1:], axis=0), atol=1e-8)
        self.assertEqual([row["receiver_train_row_visits"] for row in performance], [3, 3, 0])
        visits = [row["training_row_visits_by_client"] for row in performance]
        self.assertEqual(visits, [[3, 6, 6], [3, 6, 6], [0, 6, 6]])
        self.assertFalse(states[LOCAL][0].flags.writeable)
        local = head_probabilities(states[LOCAL][0], features, batch_size=2)
        federated = head_probabilities(states[FEDERATED][0], features, batch_size=2)
        self.assertTrue(np.all(federated[:, 1] > local[:, 1]))
        np.testing.assert_allclose(federated.sum(axis=1), 1)
        comm = {row["mode"]: row for row in result["communication"]}
        self.assertEqual(comm[FEDERATED]["round_payload_bytes"], 6 * (2 * 3 * 8 + 2 * 8 + 3 * 4))
        self.assertEqual(comm[DONOR]["total_payload_bytes"], 6 * (2 * 8 + 2 * 4 + 4))
        self.assertEqual(comm[LOCAL]["total_payload_bytes"], 0)
        (secure_round,) = (row for row in result["rounds"] if row["mode"] == FEDERATED)
        self.assertTrue(secure_round["all_clients_accepted"])
        self.assertGreater(secure_round["local_ipc_frame_bytes"], 0)

    def test_local_visit_schedule_and_inference_are_reproducible(self):
        train, features, args = _fixture()
        args.rounds, args.head_epochs = 2, 3
        modes = (LOCAL,)
        a = fit_shared_heads(train, 2, args, modes=modes)
        b = fit_shared_heads(train, 2, args, modes=modes)
        self.assertEqual(a["model_spec"]["epochs_by_round"], [2, 1])
        self.assertEqual(a["performance"][0]["optimizer_steps"], 15)
        for u, v in zip(a["states"][LOCAL], b["states"][LOCAL], strict=True):
            np.testing.assert_array_equal(u, v)
        state = a["states"][LOCAL][0]
        self.assertEqual(head_probabilities(state, features[:0]).shape, (0, 2))
        np.testing.assert_array_equal(
            head_probabilities(state, np.array([[np.nan, np.inf]])),
            head_probabilities(state, np.zeros((1, 2))),
        )

    def test_explicit_single_donor_matches_its_local_fit_over_multiple_rounds(self):
        train, _, args = _fixture()
        train[2] = (train[2][0] * 10, np.ones_like(train[2][1]))
        args.rounds, args.head_epochs = 3, 5
        original = fit_shared_heads(train, 2, args, receiver=0)
        compared = fit_shared_heads(train, 2, args, receiver=0, donor_subsets={SINGLE_DONOR: (2,)})
        self.assertEqual(tuple(compared["states"]), (LOCAL, FEDERATED, DONOR, SINGLE_DONOR))
        self.assertEqual(compared["model_spec"]["donor_subsets"], {SINGLE_DONOR: [2]})
        self.assertNotIn("donor_subsets", original["model_spec"])
        self.assertEqual(len(compared["states"][SINGLE_DONOR]), len(train))
        self.assertFalse(np.array_equal(compared["states"][LOCAL][2], compared["states"][LOCAL][1]))
        for result in compared["states"][SINGLE_DONOR]:
            np.testing.assert_array_equal(result, compared["states"][LOCAL][2])
            self.assertFalse(result.flags.writeable)
        for mode in original["states"]:
            for a, b in zip(original["states"][mode], compared["states"][mode], strict=True):
                np.testing.assert_array_equal(a, b)
        fit = compared["performance"][-1]
        self.assertEqual(fit["training_row_visits_by_client"], [0, 0, 30])
        self.assertEqual(fit["receiver_train_row_visits"], 0)
        self.assertEqual(fit["optimizer_steps"], 10)
        rounds = [r for r in compared["rounds"] if r["mode"] == SINGLE_DONOR]
        self.assertEqual([r["local_epochs"] for r in rounds], [2, 2, 1])
        self.assertTrue(all(r["active_donor_clients"] == [2] for r in rounds))
        comm = compared["communication"][-1]
        self.assertEqual(comm["active_donor_clients"], [2])
        self.assertEqual(comm["round_payload_bytes"], 6 * (8 + 4))
        self.assertEqual(comm["receiver_delivery_payload_bytes"], 6 * 4)
        self.assertEqual(comm["total_payload_bytes"], 3 * 6 * (8 + 4) + 6 * 4)

    def test_generic_subset_and_benign_only_attack_head_preserve_zero_visits(self):
        train, _, args = _fixture()
        args.head_objective = ATTACK_ONLY
        mode = "head_donor_0"
        result = fit_shared_heads(train, 2, args, receiver=1, modes=(mode,), donor_subsets={mode: (0,)})
        self.assertEqual(tuple(result["states"]), (mode,))
        for state in result["states"][mode]:
            np.testing.assert_array_equal(state, np.zeros((2, 3), dtype=np.float32))
        self.assertEqual(result["rounds"][0]["participating_clients_without_training_rows"], 1)
        self.assertEqual(result["performance"][0]["training_row_visits_by_client"], [0, 0, 0])

    def test_invalid_inputs_are_rejected(self):
        train, _, args = _fixture()
        with self.assertRaises(ValueError):
            fit_shared_heads(train[:2], 2, args)
        with self.assertRaises(ValueError):
            fit_shared_heads(train, 2, args, modes=(DONOR,))
        with self.assertRaises(ValueError):
            fit_shared_heads([*train[:2], (train[2][0], np.array(0))], 2, args)
        with self.assertRaises(ValueError):
            head_probabilities(np.zeros((2, 2)), np.zeros((2, 2)))

    def test_invalid_donor_selection_is_rejected_before_fit(self):
        train, _, args = _fixture()
        for receiver, subset in (
            (None, {SINGLE_DONOR: (1,)}),
            (0, {SINGLE_DONOR: (0,)}),
            (0, {SINGLE_DONOR: (3,)}),
            (0, {SINGLE_DONOR: (True,)}),
            (0, {SINGLE_DONOR: (1, 2)}),
            (0, {SINGLE_DONOR: ()}),
            (0, {"head_donor_pair": (1, 1)}),
            (0, {DONOR: (1,)}),
            (0, [(SINGLE_DONOR, (1,))]),
        ):
            with (
                self.subTest(receiver=receiver, subset=subset),
                self.assertRaisesRegex(ValueError, "[Dd]onor subset"),
            ):
                fit_shared_heads(train, 2, args, receiver=receiver, donor_subsets=subset)
        with self.assertRaisesRegex(ValueError, "roster"):
            fit_shared_heads(train, 2, args, receiver=0, modes=(SINGLE_DONOR,))
