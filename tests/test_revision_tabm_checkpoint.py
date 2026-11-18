"""Prove that the frozen first-stage detector preserves raw TabM decisions."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch

from src import utils as core
from src.privacy_run import PrivateClient
from src.revision_run import _train_clients
from src.revision_tabm_checkpoint import (
    load_tabm_checkpoint,
    predict_tabm_checkpoint,
    save_tabm_checkpoint,
    tabm_node_predictor,
)


class FrozenTabMCheckpointTests(unittest.TestCase):
    def _fitted(self, folder):
        rng = np.random.default_rng(23)
        x = rng.normal(size=(41, 4)).astype(np.float32)
        x[4, 2] = np.nan
        x[32, 0] = np.inf
        y = np.arange(len(x), dtype=np.int64) % 3
        parts = {"train": (x[:24], y[:24]), "cal": (x[24:32], y[24:32]), "val": (x[32:], y[32:])}
        routes = {role: np.arange(len(labels)) for role, (_, labels) in parts.items()}
        args = SimpleNamespace(
            device="cpu",
            seed=23,
            base_epochs=1,
            batch_size=4,
            adapter_epochs=1,
            adapter_lr=0.1,
            adapter_prox=1,
            prior_strength=0.75,
            target_far=0.01,
            threshold_floor=0.5,
            cal_fit_fraction=0.5,
        )
        node = PrivateClient(
            "client_2", parts, routes, ("BENIGN", "A", "B"), ("w", "x", "y", "z"), args, folder / "working"
        )
        return node, parts["val"][0]

    def test_raw_checkpoint_matches_private_client_predictions(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            node, val = self._fitted(root)
            path = root / "tabm.pt"
            receipt = save_tabm_checkpoint(path, node, batch_size=4)
            predictor = load_tabm_checkpoint(
                path, expected_sha256=receipt["sha256"], names=node.names, features=node.features
            )
            self.assertEqual(predictor.client_id, "client_2")
            self.assertEqual(predictor.batch_size, 4)
            self.assertEqual(predictor.inference_device, "cpu")
            direct = tabm_node_predictor(node, batch_size=4)
            np.testing.assert_array_equal(predictor(val), node.probabilities["val"])
            np.testing.assert_array_equal(direct(val), predictor(val))
            np.testing.assert_array_equal(predict_tabm_checkpoint(path, val), predictor(val))
            self.assertEqual(predictor(val[:0]).shape, (0, 3))
            np.testing.assert_array_equal(
                core.hierarchical_decisions(predictor(val), 0.5)["predicted"],
                core.hierarchical_decisions(node.probabilities["val"], 0.5)["predicted"],
            )

    def test_experiment_callback_exports_before_scratch_cleanup(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            x = np.random.default_rng(38).normal(size=(12, 4)).astype(np.float32)
            y = np.arange(len(x), dtype=np.int64) % 3
            parts = {"train": (x, y), "cal": (x[:6], y[:6]), "val": (x[6:], y[6:]), "test": (x[:0], y[:0])}
            roster = [{role: np.arange(len(labels)) for role, (_, labels) in parts.items()}]
            args = SimpleNamespace(
                device="cpu",
                seed=23,
                base_epochs=1,
                batch_size=4,
                adapter_epochs=1,
                adapter_lr=0.1,
                adapter_prox=1,
                prior_strength=0.75,
                target_far=0.01,
                threshold_floor=0.5,
                cal_fit_fraction=0.5,
            )
            names, features = ("BENIGN", "A", "B"), ("w", "x", "y", "z")
            receipts = []

            def export(index, node):
                self.assertEqual((index, node.client_id), (0, "client_0"))
                receipts.append(save_tabm_checkpoint(root / "tabm.pt", node, batch_size=4))

            with TemporaryDirectory(dir=root) as scratch:
                clients, _, _ = _train_clients(
                    parts, roster, names, features, args, Path(scratch), on_fitted_client=export
                )
            predictor = load_tabm_checkpoint(receipts[0]["file"], expected_sha256=receipts[0]["sha256"])
            np.testing.assert_array_equal(predictor(x[6:]), clients[0].eval_probabilities)

    def test_checkpoint_rejects_mismatch_and_invalid_transform(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            node, val = self._fitted(root)
            path = root / "tabm.pt"
            receipt = save_tabm_checkpoint(path, node, batch_size=4)
            with self.assertRaisesRegex(FileExistsError, "tabm"):
                save_tabm_checkpoint(path, node, batch_size=4)
            with self.assertRaisesRegex(ValueError, "SHA256"):
                load_tabm_checkpoint(path, expected_sha256="0" * 64)
            with self.assertRaisesRegex(ValueError, "order"):
                load_tabm_checkpoint(path, features=tuple(reversed(node.features)))
            with self.assertRaisesRegex(ValueError, "matrix"):
                load_tabm_checkpoint(path, expected_sha256=receipt["sha256"])(val[:, :2])
            payload = torch.load(path, weights_only=True)
            payload["transform"]["scale"][0] = 0
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "TRAIN preprocessing"):
                load_tabm_checkpoint(path)

    def test_checkpoint_rejects_unrecognized_tensor_structure(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            node, _ = self._fitted(root)
            path = root / "tabm.pt"
            save_tabm_checkpoint(path, node, batch_size=4)
            payload = torch.load(path, weights_only=True)
            key = next(iter(payload["state"]))
            payload["state"][key] = torch.zeros(1)
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "tensor schema"):
                load_tabm_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
