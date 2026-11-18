"""Test matched decisions and the local additive sharing failure contract."""

import unittest
from argparse import Namespace
from collections import Counter
from unittest.mock import patch

import numpy as np
import torch

from src.privacy_core import fit_class_adapter
from src.revision_adaptation import MODES, AdapterClient, run_adapter_ablations
from src.secure_aggregation import SecureAggregationError, SecureAggregationSession, _split_quantized


def _arguments():
    return Namespace(
        rounds=1,
        seed=42,
        adapter_epochs=2,
        adapter_lr=0.1,
        prior_strength=0.5,
        adapter_prox=1.0,
        cal_fit_fraction=0.5,
        target_far=0.01,
        threshold_floor=0.5,
        clip=0.1,
        secure_scale=1_000_000,
        secure_timeout=30.0,
    )


class RevisionAdaptationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls._threads)

    def test_adaptation_diagnostics_preserve_fit(self):
        args = (np.zeros(1), np.array([[0.4, 0.6], [0.8, 0.2]]), np.array([1, 0]), np.zeros(2))
        details = {}
        np.testing.assert_array_equal(fit_class_adapter(*args), fit_class_adapter(*args, diagnostics=details))
        self.assertGreater(details["first_gradient_l2"], 0)
        self.assertEqual(details["projection_steps"], 0)

    def test_matched_modes_share_evaluation_inputs_and_use_real_collectors(self):
        cal_labels = np.arange(12) % 3
        cal = np.full((12, 3), 0.23)
        cal[np.arange(12), cal_labels] = 0.54
        eval_p = np.array([[0.40, 0.33, 0.27], [0.20, 0.21, 0.59]])
        client = AdapterClient(cal, cal_labels, eval_p, np.zeros(3))
        predictions, scores, details = run_adapter_ablations([client] * 3, _arguments())
        self.assertEqual(tuple(predictions), MODES)
        np.testing.assert_array_equal(predictions["base"][0], [0, 2])
        np.testing.assert_array_equal(predictions["base_fixed"][0], [1, 2])
        np.testing.assert_array_equal(scores["full"][0], scores["without_threshold"][0])
        self.assertEqual({r["mode"] for r in details["rounds"]}, {"full", "without_prox"})
        self.assertTrue(all(r["all_clients_accepted"] for r in details["rounds"]))
        self.assertTrue(all(r["local_ipc_frame_bytes"] > 0 for r in details["rounds"]))
        self.assertTrue(all(r["cal_fit_rows"] == r["cal_holdout_rows"] == 6 for r in details["calibration"]))

    def test_saturated_scores_can_lose_local_updates_in_quantization(self):
        cal_labels = np.arange(12) % 2
        client = AdapterClient(np.eye(2)[cal_labels], cal_labels, np.eye(2), np.zeros(2))
        _, _, details = run_adapter_ablations([client] * 3, _arguments(), modes=("full",))
        self.assertFalse(details["rounds"][0]["accepted_update"])
        private = details["private_rounds"][0]
        self.assertEqual(private["clients_with_zero_encoded_delta"], 3)
        self.assertEqual(private["base_true_class_saturation_fraction"], 1.0)
        self.assertIn(
            private["outcome"],
            {
                "all_local_deltas_exactly_zero",
                "all_local_deltas_zero_after_clipping_and_encoding",
            },
        )

    def test_no_benign_in_cal_holdout_is_explicit(self):
        probabilities = np.array([[0.0, 1.0], [0.0, 1.0]])
        client = AdapterClient(probabilities, np.array([1, 1]), probabilities, np.zeros(2))
        predictions, _, details = run_adapter_ablations([client], _arguments(), modes=("base_threshold",))
        self.assertEqual(details["calibration"][0]["threshold"], 1.0)
        self.assertTrue(details["calibration"][0]["no_benign_fallback"])
        np.testing.assert_array_equal(predictions["base_threshold"][0], [0, 0])


class SharingBoundaryTest(unittest.TestCase):
    def test_incomplete_roster_does_not_produce_an_aggregate(self):
        with SecureAggregationSession(["a", "b", "c"], 1, timeout=30.0) as session:
            contract = session.start_round("test", 1, "0" * 64)
            session.submit("a", [0.04], contract=contract)
            session.submit("b", [0.02], contract=contract)
            with self.assertRaisesRegex(SecureAggregationError, "incomplete_roster"):
                session.aggregate()
            with self.assertRaisesRegex(SecureAggregationError, "session_is_not_open"):
                session.start_round("test", 2, "1" * 64)

    def test_single_share_is_uniform_under_uniform_mask_exhaustive_toy_field(self):
        class FixedMask:
            def __init__(self, mask):
                self.mask = mask

            def randrange(self, _):
                return self.mask

        # Enumerate every mask in a toy field instead of using a flaky frequency test.
        with patch("src.secure_aggregation.PRIME", 7):
            for value in range(-3, 4):
                shares = [_split_quantized([value], FixedMask(mask)) for mask in range(7)]
                for collector in (0, 1):
                    self.assertEqual(Counter(pair[collector][0] for pair in shares), Counter(range(7)))
                self.assertTrue(all((a[0] + b[0]) % 7 == value % 7 for a, b in shares))


if __name__ == "__main__":
    unittest.main()
