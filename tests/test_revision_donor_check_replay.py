"""Verify the isolated calibration roles used by the donor-CAL replay."""

import unittest

import numpy as np

from src.revision_donor_check_replay import _cal_owners, _receiver_check_mask


class DonorReplaySplitTests(unittest.TestCase):
    """Check exact-input disjointness independently of labels and model scores."""

    def test_receiver_groups_are_not_used_for_donor_selection(self):
        groups = {0: np.array([1, 2]), 1: np.array([4, 5, 4]), 2: np.array([2, 3, 4, 6])}
        keep = _cal_owners(groups, receiver=0)
        np.testing.assert_array_equal(groups[0][keep[0]], [1, 2])
        np.testing.assert_array_equal(groups[1][keep[1]], [4, 5, 4])
        np.testing.assert_array_equal(groups[2][keep[2]], [3, 6])

    def test_receiver_check_keeps_exact_groups_together(self):
        features = np.repeat(np.arange(18).reshape(9, 2), repeats=2, axis=0)
        check = _receiver_check_mask(features, seed=42)
        self.assertEqual(check.sum(), 6)
        np.testing.assert_array_equal(check[::2], check[1::2])
        np.testing.assert_array_equal(check, _receiver_check_mask(features, seed=42))
