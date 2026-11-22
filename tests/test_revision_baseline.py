import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from src.revision_baseline import _epoch_schedule, _pooled_train_sample, run_fedavg


class FedAvgTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(83)
        self.train = [
            (rng.normal(size=(n, 4)).astype(np.float32), np.arange(n, dtype=np.int64) % 3)
            for n in (9, 13, 17)
        ]
        self.evaluation = [rng.normal(size=(n, 4)).astype(np.float32) for n in (5, 0, 4)]
        self.args = SimpleNamespace(
            device="cpu",
            seed=7,
            base_epochs=3,
            rounds=2,
            batch_size=8,
            detector_lr=0.002,
            detector_weight_decay=0.0001,
            prior_strength=0.5,
        )

    def test_train_only_sample_and_epoch_budget(self):
        x = np.arange(60).reshape(15, 4)
        train = [(x[:5], np.zeros(5)), (x[5:], np.zeros(10))]
        expected = np.sort(np.random.default_rng(29).choice(15, size=8, replace=False))
        np.testing.assert_array_equal(_pooled_train_sample(train, 8, 29), x[expected])
        self.assertEqual(_epoch_schedule(5, 3), [2, 2, 1])
        with self.assertRaisesRegex(ValueError, "rounds <= detector epochs"):
            _epoch_schedule(2, 3)

    def test_crossple_end_to_end_and_evaluation_isolation(self):
        output = run_fedavg(self.train, self.evaluation, 3, self.args, preprocessing_rows=20)
        shifted = run_fedavg(
            self.train, [x + 10 for x in self.evaluation], 3, self.args, preprocessing_rows=20
        )
        self.assertEqual(output["settings"]["epochs_by_round"], [2, 1])
        self.assertEqual(output["performance"]["training_row_visits"], 117)
        self.assertEqual(output["performance"]["optimizer_steps"], 21)
        self.assertEqual(output["performance"]["inference_rows"], 9)
        self.assertEqual(output["settings"]["preprocessing_rows"], 20)
        self.assertEqual(output["settings"]["training_rows_by_client"], [9, 13, 17])
        for before, after in zip(output["rounds"], shifted["rounds"], strict=True):
            self.assertEqual(before["mean_training_loss"], after["mean_training_loss"])
            self.assertEqual(before["global_update_l2"], after["global_update_l2"])
            self.assertGreater(before["global_update_l2"], 0)
        for p, x in zip(output["probabilities"], self.evaluation, strict=True):
            self.assertEqual(p.shape, (len(x), 3))
            np.testing.assert_allclose(p.sum(axis=1), 1.0)
        self.assertFalse(np.allclose(output["probabilities"][0], shifted["probabilities"][0]))
        communication = output["communication"]
        self.assertEqual(communication["model_payload_bytes"], 4 * communication["parameter_count"])
        self.assertEqual(communication["round_payload_bytes"], 6 * communication["model_payload_bytes"])
        self.assertEqual(communication["training_payload_bytes"], 2 * communication["round_payload_bytes"])

    def test_aggregation_is_weighted_by_client_train_rows(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weights = torch.nn.Parameter(torch.zeros(3))

            def forward(self, x):
                return self.weights.expand(len(x), -1)

        def fake_train(model, transform, train, prior, epochs, rng, args, device):
            with torch.no_grad():
                model.weights[0] += len(train[1])
            return 0.0

        self.args.base_epochs = self.args.rounds = 1
        with (
            patch("src.revision_baseline.core.make_model", return_value=(TinyModel(), [])),
            patch("src.revision_baseline._train_client", side_effect=fake_train),
        ):
            output = run_fedavg(self.train, self.evaluation, 3, self.args)
        expected = (9**2 + 13**2 + 17**2) / (9 + 13 + 17)
        self.assertAlmostEqual(output["rounds"][0]["global_update_l2"], expected, places=6)
        self.assertEqual(output["communication"]["round_payload_bytes"], 72)


if __name__ == "__main__":
    unittest.main()
