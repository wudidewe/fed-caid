"""Validate matched low-FAR training over the actual CrossTabM backend."""

import unittest

import numpy as np
import torch
from torch.nn import functional as F

from src import revision_nb15_low_far as low_far
from src import utils as core


class NB15LowFARFitTests(unittest.TestCase):
    """Check the starting score, training scope and upper-tail objective."""

    def setUp(self):
        core.seed(42)
        self.features = np.random.default_rng(42).normal(size=(60, 3)).astype(np.float32)
        self.labels = np.tile([0, 0, 1, 2, 0], 12)
        self.reference, _ = core.make_model(3, 3, "ple_cross", x=self.features)

    def test_zero_readout_recovers_the_original_ensemble_attack_probability(self):
        batch = torch.as_tensor(self.features)
        self.reference.eval()
        with torch.no_grad():
            expected = self.reference(batch).softmax(-1)[..., 1:].sum(-1).mean(1)
            for mode in ("frozen", "full"):
                scorer = low_far.BinaryAttackScorer(self.reference, mode, "cpu")
                torch.testing.assert_close(scorer(batch).sigmoid(), expected)

    def test_both_arms_cover_same_train_budget_and_only_full_changes_tabm(self):
        state = {name: value.clone() for name, value in self.reference.state_dict().items()}
        result = {
            mode: low_far.fit_low_far(
                self.reference, self.features, self.labels, mode=mode, seed=42, batch_size=20, device="cpu"
            )
            for mode in ("frozen", "full")
        }
        frozen, full = result["frozen"], result["full"]
        for name in ("train_rows", "optimizer_steps", "train_rows_visited", "tail_pairs"):
            self.assertGreater(frozen.receipt[name], 0)
            self.assertEqual(frozen.receipt[name], full.receipt[name])
        self.assertEqual(frozen.receipt["base_parameters_updated"], [])
        self.assertTrue(full.receipt["base_parameters_updated"])
        for arm in (frozen, full):
            self.assertTrue(arm.receipt["binary_readout_updated"])
            scores = arm.score(self.features)
            self.assertEqual(scores.shape, self.labels.shape)
            self.assertTrue(np.all((scores >= 0) & (scores <= 1)))
        for name, value in self.reference.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)

    def test_only_top_benign_score_enters_the_small_batch_ranking_term(self):
        logits = torch.tensor([-3.0, 1.0, -2.0, 2.0, 0.5])
        attacks = torch.tensor([False, False, False, True, True])
        expected = F.softplus(logits[1] - logits[attacks]).mean()
        expected += low_far.BCE_WEIGHT * F.binary_cross_entropy_with_logits(logits, attacks.float())
        loss, pairs = low_far._tail_loss(logits, attacks, benign_count=3)
        self.assertEqual(pairs, 2)
        torch.testing.assert_close(loss, expected)
