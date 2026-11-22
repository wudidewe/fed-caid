"""Regression tests for the portable binary and conditional subtype detector."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch.nn import functional

from src import utils
from src.privacy_core import ClassScoreAdapter
from src.privacy_run import PrivateClient, predict_private_decisions
from src.revision_iov_binary import PORTABLE_BINARY_MODES, conditional_attack_loss
from src.revision_tabm_checkpoint import load_tabm_checkpoint, save_tabm_checkpoint


class PortableBinaryTests(unittest.TestCase):
    def test_auxiliary_loss_averages_attack_rows_and_has_no_alarm_gradient(self):
        alarm = torch.tensor([[1., 2.], [0., -1.], [2., 1.]], requires_grad=True)
        logits = torch.arange(18, dtype=torch.float32).reshape(3, 2, 3).requires_grad_()
        y = torch.tensor([0, 3, 1])
        joint = torch.cat((
            functional.logsigmoid(-alarm[..., None]),
            functional.logsigmoid(alarm[..., None]) + logits.log_softmax(-1),
        ), dim=-1)
        loss = conditional_attack_loss(joint, y)
        expected = functional.cross_entropy(logits[1:].reshape(4, 3), torch.tensor([2, 2, 0, 0]))
        torch.testing.assert_close(loss, expected)
        loss.backward()
        torch.testing.assert_close(alarm.grad, torch.zeros_like(alarm), atol=1e-7, rtol=0)
        self.assertEqual(torch.count_nonzero(logits.grad[0]), 0)
        self.assertEqual(conditional_attack_loss(joint.detach(), torch.zeros_like(y)).item(), 0.)

    def test_trained_binary_checkpoints_preserve_scores_masks_and_subtypes(self):
        rng = np.random.default_rng(509)
        parts = {
            role: (rng.normal(size=(n, 4)).astype(np.float32), np.arange(n, dtype=np.int64) % 3)
            for role, n in [('train', 24), ('cal', 12), ('val', 8)]
        }
        rows = {role: np.arange(len(y)) for role, (_, y) in parts.items()}
        args = SimpleNamespace(
            device='cpu', seed=7, base_epochs=1, batch_size=8, prior_strength=.5, adapter_epochs=1,
            adapter_lr=.1, adapter_prox=1., target_far=.01, threshold_floor=.5,
            cal_fit_fraction=.5, tabm_attention=True,
        )
        for mode in PORTABLE_BINARY_MODES:
            with self.subTest(mode=mode), TemporaryDirectory() as directory:
                args.tabm_binary_head = mode
                path = Path(directory)
                client = PrivateClient('client_0', parts, rows, ('BENIGN', 'A', 'B'), tuple('abcd'), args, path / 'client')
                x = parts['val'][0]
                p = client.probabilities['val']
                raw = path / 'raw.pt'
                save_tabm_checkpoint(raw, client, batch_size=args.batch_size)
                np.testing.assert_array_equal(load_tabm_checkpoint(raw)(x), p)
                vector = np.array([.4, -.3])
                adapter = ClassScoreAdapter(3)
                adapter.set_vector(vector)
                adapted = adapter.transform(p)
                threshold = float(np.median(p[:, 1:].sum(axis=1)))
                for fixed in (False, True):
                    branch = path / f'branch_{fixed}.pt'
                    client.save_detector_checkpoint(branch, vector, threshold, fixed_base_alarm=fixed)
                    scores, subtypes = (p, adapted[:, 1:]) if fixed else (adapted, None)
                    expected = utils.hierarchical_decisions(scores, threshold, subtypes)
                    actual = predict_private_decisions(branch, x)
                    for key in ('attack_score', 'attack_alert', 'predicted', 'threshold'):
                        np.testing.assert_array_equal(actual[key], expected[key])
                    np.testing.assert_array_equal(actual['probabilities'], utils.subtype_joint_probabilities(scores, subtypes))
