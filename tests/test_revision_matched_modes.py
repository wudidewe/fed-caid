"""Check matched personalization budgets and development-only diagnostics."""

import unittest
from argparse import Namespace
from unittest.mock import patch

import numpy as np
import torch

from src import revision_adaptation as adaptation
from src.secure_aggregation import _clip_quantize


def _arguments():
    return Namespace(
        rounds=2,
        seed=42,
        adapter_epochs=1,
        adapter_lr=0.1,
        prior_strength=0.5,
        adapter_prox=1.0,
        cal_fit_fraction=0.5,
        target_far=0.01,
        threshold_floor=0.5,
        clip=0.1,
        secure_scale=1_000_000_000,
        secure_timeout=30.0,
    )


class _LocalCollector:
    """Replay the clipping and mean used by the subprocess protocol."""

    def __init__(self, clients, dimension, *, clip, scale, timeout):
        self.dimension, self.clip, self.scale = dimension, clip, scale

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def start_round(self, *_):
        self.updates = []
        return None

    def submit(self, client, update, *, contract):
        self.updates.append(_clip_quantize(update, self.dimension, self.clip, self.scale))

    def aggregate(self):
        return np.mean(self.updates, axis=0) / self.scale, {}


class MatchedModesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls._threads)

    def test_shared_trajectories_and_matched_final_fit_budgets(self):
        args = _arguments()
        cal = np.tile([0.4, 0.32, 0.28], (12, 1))
        eval_p = np.array([[0.45, 0.30, 0.25]])
        clients = [
            adaptation.AdapterClient(cal, np.full(len(cal), label), eval_p, np.zeros(3)) for label in range(3)
        ]
        with (
            patch.object(adaptation, "SecureAggregationSession", _LocalCollector),
            patch.object(adaptation, "_fit", wraps=adaptation._fit) as fit,
            patch.object(adaptation, "_classify", wraps=adaptation._classify) as classify,
        ):
            predictions, _, diagnostics = adaptation.run_adapter_ablations(
                clients,
                args,
                modes=adaptation.MATCHED_MODES,
                include_development=True,
            )
        self.assertEqual(tuple(predictions), adaptation.MATCHED_MODES)
        self.assertEqual(fit.call_count, 2 * (args.rounds + 1) * len(clients))
        self.assertEqual([row["mode"] for row in diagnostics["rounds"]], ["full"] * args.rounds)
        vectors = {
            mode: [call.args[4] for call in classify.call_args_list if call.args[0] == mode]
            for mode in ("full_global", "local_rounds", "full", "local_only")
        }
        np.testing.assert_array_equal(vectors["full_global"][0], vectors["full_global"][1])
        self.assertFalse(np.array_equal(vectors["local_rounds"][0], vectors["local_rounds"][1]))
        self.assertFalse(np.array_equal(vectors["full_global"][0], vectors["full"][0]))
        self.assertFalse(np.array_equal(vectors["local_rounds"][0], vectors["local_only"][0]))
        self.assertEqual(diagnostics["cal_fit_class_counts"], [[6, 0, 0], [0, 6, 0], [0, 0, 6]])
        self.assertTrue(
            all(
                row["threshold_source"] == ("native argmax" if row["mode"] == "base" else "CAL holdout")
                for row in diagnostics["calibration"]
            )
        )

    def test_diagnostics_distinguish_alarm_failure_from_wrong_subtype(self):
        cal = np.array([[0.9, 0.09, 0.01], [0.2, 0.79, 0.01], [0.8, 0.1, 0.1]] * 4)
        labels = np.tile([2, 2, 0], 4)
        client = adaptation.AdapterClient(cal, labels, np.array([[0.0, 0.0, 1.0]]), np.zeros(3))
        _, _, diagnostics = adaptation.run_adapter_ablations(
            [client],
            _arguments(),
            modes=("base_threshold",),
            include_development=True,
        )
        rows = diagnostics["development_class_diagnostics"]
        self.assertEqual({r["role"] for r in rows}, {"cal_fit", "cal_holdout"})
        self.assertEqual({r["class_index"] for r in rows}, {2})
        for key, count in {
            "support": 8,
            "missed_alarm": 4,
            "alerted_wrong_subtype": 4,
            "attack_top1_correct": 0,
            "attack_top3_correct": 8,
            "outside_bias_margin": 4,
        }.items():
            self.assertEqual(sum(row[key] for row in rows), count)


if __name__ == "__main__":
    unittest.main()
