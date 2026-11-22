"""Check the fixed alarm policy against aligned local TRAIN membership."""

import unittest
from types import SimpleNamespace

import numpy as np

from src.privacy_core import ClassScoreAdapter
from src.revision_adaptation import AdapterClient, _classify, _fixed_base_decisions, _split_clients
from src.revision_iov_id_gate import ID_GATE_MODE, TrainIdGate, id_thresholds
from src.utils import hierarchical_decisions


class IdGateTests(unittest.TestCase):
    def test_client_local_membership_and_strict_cutoff(self):
        ids = np.array([21, 42, 21, 42, np.nan])
        gate = TrainIdGate([21, np.nan, -3, 4.5])
        self.assertEqual(gate.training_id_count, 1)
        np.testing.assert_array_equal(gate.seen(ids), [True, False, True, False, False])
        self.assertFalse(TrainIdGate([42]).seen(ids)[0])

        scores = np.array([0.50, 0.70, 0.51, 0.71, 0.65])
        probabilities = np.column_stack((1 - scores, scores))
        alarms = hierarchical_decisions(probabilities, id_thresholds(gate.seen(ids)))['attack_alert']
        np.testing.assert_array_equal(alarms, [False, False, True, True, False])

    def test_base_and_adapted_labels_share_the_fixed_gate(self):
        probabilities = np.array([[.4, .4, .2], [.4, .2, .4], [.1, .1, .8], [.3, .6, .1]])
        client = AdapterClient(
            probabilities, np.array([0, 1, 2, 0]), probabilities, np.zeros(3),
            cal_split=(np.array([0, 2]), np.array([1, 3])),
            cal_id_seen=np.array([True, False, False, True]),
            eval_id_seen=np.array([True, False, False, True]),
        )
        adapter = ClassScoreAdapter(3)
        args = SimpleNamespace(tabm_binary_head=ID_GATE_MODE, target_far=.01)
        split, = _split_clients([client], adapter, args)
        predicted, scores, receipt, _, _, _ = _classify('base_threshold', client, split, adapter, None, args)
        np.testing.assert_array_equal(predicted, [1, 0, 2, 1])
        np.testing.assert_allclose(scores, [.6, .6, .9, .7])
        self.assertIsNone(receipt['threshold'])
        self.assertEqual((receipt['threshold_seen_id'], receipt['threshold_unseen_id']), (.5, .7))

        swapped = probabilities[:, [0, 2, 1]]
        fixed_cal, fixed_val = _fixed_base_decisions(
            split, receipt, predicted, swapped[split.holdout_indices], swapped,
            cal_id_seen=client.cal_id_seen[split.holdout_indices],
        )
        np.testing.assert_array_equal(fixed_cal, [0, 2])
        np.testing.assert_array_equal(fixed_val, [2, 0, 1, 2])


if __name__ == '__main__':
    unittest.main()
