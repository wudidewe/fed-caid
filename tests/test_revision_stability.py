"""Regress narrow-bin PLE extrapolation and TRAIN-only boundary policies."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch

from src import utils as core
from src.privacy_run import PrivateClient, _predict_private_saved
from src.revision_baseline import _train_client, run_fedavg


class PLEStabilityTests(unittest.TestCase):
    def setUp(self):
        core.seed(23)

    def _narrow_model(self, policy="clip_to_train_edges"):
        bins = [torch.tensor([0.0, 1e-8])] * 2
        model, _ = core.make_model(2, 3, "ple_cross", bins=bins, ple_boundary_policy=policy)
        return model, bins

    def test_narrow_bins_clip_before_both_ple_branches(self):
        model, _ = self._narrow_model()
        x = torch.tensor([[-1.0, 12.0], [0.0, 1e-8]])
        clipped = model.prepare_ple_input(x)
        self.assertTrue(torch.isfinite(model(x)).all())
        torch.testing.assert_close(clipped, torch.tensor([[0.0, 1e-8], [0.0, 1e-8]]))
        encoded = model.base.num_module.impl(clipped)
        self.assertLessEqual(float(encoded.abs().max()), 1.0)
        diagnostics = core.ple_input_diagnostics(model, x.numpy())
        self.assertEqual(diagnostics["sample_rows_outside_train_edges"], 1)
        self.assertEqual(diagnostics["sample_encoding_abs_max"], 1.0)
        self.assertAlmostEqual(diagnostics["minimum_bin_width"], 1e-8)

    def test_legacy_policy_reproduces_extrapolation(self):
        model, _ = self._narrow_model("extrapolate")
        x = torch.tensor([[1.0, 12.0]])
        torch.testing.assert_close(model.prepare_ple_input(x), x)
        self.assertGreater(float(model.base.num_module.impl(x).abs().max()), 1e9)
        # Version B starts its piecewise branch at zero. A tiny learned weight
        # demonstrates why finite inputs can later create enormous embeddings.
        with torch.no_grad():
            model.base.num_module.linear.weight.fill_(1e-4)
        self.assertGreater(float(model.base.num_module(x).abs().max()), 1e5)

    def test_old_checkpoint_preserves_policy_and_strict_state_loading(self):
        model, bins = self._narrow_model("extrapolate")
        with torch.no_grad():
            model.base.num_module.linear.weight.fill_(1e-4)
        self.assertFalse(any(key.startswith("ple_") for key in model.state_dict()))
        x = np.array([[0.0, 0.0], [1.0, 12.0]], dtype=np.float32)
        transform = core.Transform.restore({
            "median": [0.0, 0.0],
            "mean": [0.0, 0.0],
            "scale": [1.0, 1.0],
            "neural": True,
        })
        expected = core.probabilities(core.log_prediction(model, transform(x), "cpu"))
        saved = {
            "state": model.state_dict(),
            "bins": bins,
            "transform": transform.state(),
            "framework_version": 3,
            "mechanism": "mpc",
            "adapter_schema": "class_logit_bias_v2",
            "feature_names": ["x", "y"],
            "class_names": ["BENIGN", "A", "B"],
            "class_score_adapter": [0.0, 0.0],
            "alarm_threshold": 0.5,
        }
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / "old.pt"
            torch.save(saved, path)
            actual, _, _ = _predict_private_saved(path, x, return_subtypes=True)
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_both_training_paths_use_same_explicit_policy_and_train_bounds(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(16, 2)).astype(np.float32)
        labels = np.arange(16, dtype=np.int64) % 3
        probe = np.full((6, 2), 1e3, dtype=np.float32)
        parts = {"train": (x, labels), "cal": (probe, np.arange(6) % 3), "val": (probe, np.arange(6) % 3)}
        rows = {role: np.arange(len(y)) for role, (_, y) in parts.items()}
        args = SimpleNamespace(
            device="cpu",
            seed=5,
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
        for policy in core.PLE_BOUNDARY_POLICIES:
            args.ple_boundary_policy = policy
            with (
                self.subTest(policy=policy),
                TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory,
            ):
                client = PrivateClient(
                    "client_0",
                    parts,
                    rows,
                    ("BENIGN", "A", "B"),
                    ("x", "y"),
                    args,
                    Path(directory) / "client",
                )
                result = run_fedavg([parts["train"]], [probe], 3, args)
                self.assertEqual(client.model.ple_boundary_policy, policy)
                self.assertEqual(result["settings"]["ple_boundary_policy"], policy)
                local = client.ple_stability["train"]
                shared = result["settings"]["ple_stability"]["train_by_client"][0]
                np.testing.assert_allclose(local["train_lower"], shared["train_lower"])
                np.testing.assert_allclose(local["train_upper"], shared["train_upper"])
                self.assertTrue(Path(directory, "client", "ple_stability_private.json").exists())
                self.assertEqual(client.ple_stability["cal"]["sample_outside_row_fraction"], 1.0)

    def test_finite_loss_with_nonfinite_gradient_fails_training(self):
        model = torch.nn.Linear(2, 3)
        model.weight.register_hook(lambda gradient: torch.full_like(gradient, float("nan")))
        args = SimpleNamespace(detector_lr=0.002, detector_weight_decay=0.0, batch_size=4)
        x = np.ones((4, 2), dtype=np.float32)
        with self.assertRaisesRegex(FloatingPointError, "loss or gradient"):
            _train_client(
                model,
                lambda values: values,
                (x, np.arange(4) % 3),
                (np.zeros(3), None),
                1,
                np.random.default_rng(2),
                args,
                torch.device("cpu"),
            )


if __name__ == "__main__":
    unittest.main()
