"""Check the isolated detector objective and the matched FedAvg control."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch

from src.privacy_run import PrivateClient
from src.revision_baseline import run_fedavg
from src.revision_detector_objective import HARD_BENIGN, hard_benign_weights, training_losses
from src.revision_nb15_source import _binary_summary, _classification_summary
from src.utils import CLASS_BALANCE_MAX_RATIO, class_weights, classification_losses, detector_class_weights


class DetectorBalanceTests(unittest.TestCase):
    def test_operating_assessment_separates_binary_and_macro_f1_requirements(self):
        rows = [
            {
                "scope": "overall",
                "candidate_far": 0.009,
                "reference_far": 0.012,
                "delta_far": -0.003,
                "delta_fnr": 0.005,
                "delta_macro_f1": -0.002,
            }
            for _ in range(5)
        ]
        binary = _binary_summary(rows)
        self.assertTrue(binary["far_pass"])
        self.assertTrue(binary["fnr_pass"])
        self.assertEqual(binary["reference_far_seeds_over_limit"], 5)
        self.assertFalse(_classification_summary(rows)["pass"])
        rows[0].update(candidate_far=0.0101, delta_fnr=0.03, delta_macro_f1=0.02)
        binary = _binary_summary(rows)
        self.assertFalse(binary["far_pass"])
        self.assertFalse(binary["fnr_pass"])
        self.assertTrue(_classification_summary(rows)["pass"])

    def test_train_hard_benign_weighting_preserves_class_mass(self):
        logits = torch.tensor([[4.0, 0.0], [0.0, 4.0], [4.0, 0.0], [0.0, 4.0]], requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1])
        ensemble = logits[:, None, :].expand(-1, 2, -1)
        weights = hard_benign_weights(ensemble, labels)
        torch.testing.assert_close(weights, hard_benign_weights(logits, labels))
        self.assertFalse(weights.requires_grad)
        self.assertLess(weights[0], 1)
        self.assertGreater(weights[1], 1)
        self.assertAlmostEqual(float(weights[:2].mean()), 1)
        torch.testing.assert_close(weights[2:], torch.ones(2))
        torch.testing.assert_close(hard_benign_weights(logits[2:], labels[2:]), torch.ones(2))
        prior = torch.tensor([-0.22, -1.6])
        loss = training_losses(ensemble, labels, prior, None, HARD_BENIGN)
        torch.testing.assert_close(loss, classification_losses(ensemble + prior, labels) * weights)
        loss.mean().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_inverse_quartic_normalization_with_missing_classes(self):
        y = np.array([0] * 10_000 + [2] * 16 + [3], dtype=np.int64)
        weights = detector_class_weights(y, 5, "class_balanced")
        np.testing.assert_array_equal(weights[[1, 4]], 0)
        self.assertAlmostEqual(weights[y].mean(), 1.0)
        self.assertAlmostEqual(weights[3] / weights[0], CLASS_BALANCE_MAX_RATIO)
        self.assertAlmostEqual(weights[2] / weights[0], (10_000 / 16) ** 0.25)
        self.assertIsNone(detector_class_weights(y, 5, "prior_adjusted"))
        with self.assertRaises(ValueError):
            detector_class_weights(y, 5, "unknown")

    def test_existing_weights_and_shared_loss_remain_compatible(self):
        y = np.array([0, 0, 0, 1, 1, 2], dtype=np.int64)
        counts = np.bincount(y)
        original = counts.astype(float) ** -0.25
        np.testing.assert_array_equal(class_weights(y), original / np.sum(counts / len(y) * original))
        z = torch.zeros((len(y), 2, 3))
        labels = torch.as_tensor(y)
        weights = torch.as_tensor(detector_class_weights(y, 3, "class_balanced"), dtype=torch.float32)
        reference = classification_losses(z, labels)
        torch.testing.assert_close(classification_losses(z, labels, weights), reference * weights[labels])

    def test_private_and_fedavg_share_train_only_objective(self):
        rng = np.random.default_rng(20)
        y = np.array([0] * 20 + [2] * 4, dtype=np.int64)
        x = rng.normal(size=(len(y), 4)).astype(np.float32)
        probe = rng.normal(size=(6, 4)).astype(np.float32)
        labels = np.arange(6, dtype=np.int64) % 3
        parts = {"train": (x, y), "cal": (probe, labels), "val": (probe, labels)}
        shifted = {**parts, "cal": (probe + 17, labels[::-1]), "val": (probe - 12, labels[::-1])}
        rows = {role: np.arange(len(values)) for role, (values, _) in parts.items()}
        args = SimpleNamespace(
            device="cpu",
            seed=19,
            base_epochs=1,
            rounds=1,
            batch_size=8,
            detector_lr=0.002,
            detector_weight_decay=0.0001,
            prior_strength=0.5,
            adapter_epochs=1,
            adapter_lr=0.1,
            adapter_prox=1.0,
            target_far=0.01,
            threshold_floor=0.5,
            cal_fit_fraction=0.5,
        )
        names, features = ("BENIGN", "A", "B"), tuple(range(4))
        with TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = PrivateClient("client_0", parts, rows, names, features, args, root / "legacy")
            args.detector_objective = "prior_adjusted"
            reference = PrivateClient("client_0", parts, rows, names, features, args, root / "reference")
            args.detector_objective = "class_balanced"
            balanced = PrivateClient("client_0", parts, rows, names, features, args, root / "balanced")
            replay = PrivateClient("client_0", shifted, rows, names, features, args, root / "replay")
            args.detector_objective = HARD_BENIGN
            hard_benign = PrivateClient("client_0", parts, rows, names, features, args, root / "hard")
            hard_shifted = PrivateClient(
                "client_0", shifted, rows, names, features, args, root / "hard_shifted"
            )
        for old, explicit, weighted, unseen in zip(
            legacy.model.parameters(),
            reference.model.parameters(),
            balanced.model.parameters(),
            replay.model.parameters(),
            strict=True,
        ):
            torch.testing.assert_close(old, explicit, atol=0, rtol=0)
            torch.testing.assert_close(weighted, unseen, atol=0, rtol=0)
        self.assertTrue(
            any(
                not torch.equal(old, weighted)
                for old, weighted in zip(legacy.model.parameters(), balanced.model.parameters(), strict=True)
            )
        )
        for left, right in zip(hard_benign.model.parameters(), hard_shifted.model.parameters(), strict=True):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        self.assertTrue(
            any(
                not torch.equal(old, hard)
                for old, hard in zip(
                    reference.model.parameters(), hard_benign.model.parameters(), strict=True
                )
            )
        )
        hard_fedavg = run_fedavg([parts["train"]], [probe], len(names), args)
        self.assertEqual(hard_fedavg["settings"]["detector_objective"], HARD_BENIGN)
        self.assertEqual(
            hard_fedavg["performance"]["training_row_visits"], reference.performance["training_row_visits"]
        )
        self.assertEqual(hard_benign.performance["optimizer_steps"], reference.performance["optimizer_steps"])
        args.detector_objective = "class_balanced"
        fedavg = run_fedavg([parts["train"]], [probe], len(names), args)
        self.assertEqual(fedavg["settings"]["detector_objective"], "class_balanced")
        np.testing.assert_array_equal(
            fedavg["settings"]["class_weights_by_client"][0], balanced.train_class_weights
        )
        self.assertEqual(
            fedavg["performance"]["training_row_visits"], balanced.performance["training_row_visits"]
        )


if __name__ == "__main__":
    unittest.main()
