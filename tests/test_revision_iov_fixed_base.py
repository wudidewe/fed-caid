"""Check native IoV subtype controls at the original CAL and VAL alarm masks."""

import unittest
from argparse import Namespace
from unittest.mock import patch

import numpy as np

from src.revision_adaptation import AdapterClient, _fixed_alarm_labels, run_adapter_ablations
from src.revision_run import _parser, _validate_iov_diagnostics
from src.utils import hierarchical_decisions


def _client():
    """Provide a benign CAL quantile above the floor and a VAL score tie."""
    cal = np.array([[0.8, 0.1, 0.1], [0.1, 0.6, 0.3], [0.3, 0.35, 0.35], [0.2, 0.43, 0.37], [0.6, 0.2, 0.2]])
    val = np.array([[0.25, 0.4, 0.35], [0.3, 0.4, 0.3], [0.05, 0.45, 0.5]])
    return AdapterClient(
        cal, np.array([0, 1, 0, 1, 0]), val, np.zeros(3), cal_split=(np.arange(2), np.arange(2, 5))
    )


def _run_fixed_case(client, enabled, observe=None):
    """Use deterministic adapted vectors and run the actual branch scoring code."""
    local, complete = np.array([1.0, 0.0]), np.array([1.0, -1.0])
    args = Namespace(target_far=0.01, threshold_floor=0.5, adapter_prox=1.0)
    with (
        patch("src.revision_adaptation._local_round_vectors", return_value=[local]),
        patch("src.revision_adaptation._personalize_vectors", side_effect=lambda initial, *_: initial),
        patch("src.revision_adaptation._federated_round_vector", return_value=complete),
    ):
        return run_adapter_ablations(
            [client],
            args,
            modes=("base_threshold", "local_only", "full"),
            observe_operating=observe,
            fixed_base_alarm=enabled,
        )


class IoVFixedBaseAlarmTests(unittest.TestCase):
    """Require immutable base scores and alarms when adding adapted subtype labels."""

    def test_opt_in_keeps_original_branches_and_cal_val_alarm_masks(self):
        client = _client()
        observed = {}

        def observe(mode, index, holdout, receipt, scores, labels, *, cal_predictions=None):
            observed[mode] = (index, holdout.copy(), receipt, scores.copy(), labels.copy(), cal_predictions)

        ordinary = _run_fixed_case(client, False)
        pred, scores, audit = _run_fixed_case(client, True, observe)
        for mode in ordinary[0]:
            np.testing.assert_array_equal(pred[mode][0], ordinary[0][mode][0])
            np.testing.assert_array_equal(scores[mode][0], ordinary[1][mode][0])
        _, base_cal, reference, base_score, base_val, _ = observed["base_threshold"]
        self.assertEqual(reference["threshold"], 0.7)
        np.testing.assert_array_equal(base_val, [1, 0, 2])
        base_cal_val = hierarchical_decisions(base_cal, reference["threshold"])["predicted"]
        np.testing.assert_array_equal(base_cal_val, [0, 1, 0])
        self._check_fixed_outputs(observed, audit, base_cal, base_cal_val, base_score, base_val)

    def _check_fixed_outputs(self, observed, audit, base_cal, base_cal_val, base_score, base_val):
        for mode, adapted in (("fixed_base_local", "local_only"), ("fixed_base_full", "full")):
            _, cal, receipt, val_score, val_labels, cal_labels = observed[mode]
            self.assertEqual(receipt["threshold"], 0.7)
            self.assertEqual(receipt["subtype_source_mode"], adapted)
            np.testing.assert_array_equal(cal, base_cal)
            np.testing.assert_array_equal(val_score, base_score)
            np.testing.assert_array_equal(val_labels > 0, base_val > 0)
            np.testing.assert_array_equal(cal_labels > 0, base_cal_val > 0)
            np.testing.assert_array_equal(val_labels, [2, 0, 2])
            np.testing.assert_array_equal(cal_labels, [0, 2, 0])
            original = next(row for row in audit["calibration"] if row["mode"] == mode)
            self.assertEqual(original["threshold_reference_mode"], "base_threshold")

    def test_unusable_subtype_rows_fall_back_and_ties_use_class_order(self):
        base = np.array([2, 2, 2, 2, 0])
        values = np.array([
            [0.4, 0.3, 0.3], [1.0, 0.0, 0.0], [np.nan, 0.1, 0.9], [0.1, -0.1, 1.0], [0.1, 0.1, 0.8]
        ])
        np.testing.assert_array_equal(_fixed_alarm_labels(base, values), [1, 2, 2, 2, 0])
        np.testing.assert_array_equal(_fixed_alarm_labels(base, None), base)

    def test_fixed_control_requires_explicit_diagnostic_opt_in(self):
        cli = _parser().parse_args(["--iov-fixed-base-alarm"])
        with self.assertRaisesRegex(ValueError, "requires --iov-diagnostics"):
            _validate_iov_diagnostics(cli)


if __name__ == "__main__":
    unittest.main()
