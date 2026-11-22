import unittest

import numpy as np

from src.revision_portable_class_cal import apply_portable_class_cal, select_portable_class_cal


def probabilities(rows):
    """Build benign plus two attack probabilities from an attack score and route."""
    return np.asarray([(1 - q, q if route == 1 else 0, q if route == 2 else 0) for q, route in rows])


class PortableClassCalTests(unittest.TestCase):
    def setUp(self):
        self.client = dict(
            fit_probabilities=probabilities([(.90, 1), (.80, 1), (.54, 2), (.55, 2)]),
            fit_labels=np.zeros(4, dtype=int),
            fit_group_ids=np.arange(4),
            select_probabilities=probabilities([(.88, 1), (.86, 1), (.51, 2), (.52, 2), (.70, 2)]),
            select_labels=np.array([0, 0, 0, 0, 2]),
            select_group_ids=np.arange(4, 9),
        )

    def test_supported_routes_improve_recall_at_same_far(self):
        selection = select_portable_class_cal([self.client], min_route_benign_groups=2)
        self.assertEqual(selection['status'], 'selected')
        self.assertEqual(selection['scalar_thresholds'], [.90])
        self.assertEqual(selection['class_thresholds'], [[.90, .55]])
        self.assertEqual(selection['reference']['missed_attacks'], 1)
        self.assertTrue(all(not row['failed_constraints'] for row in selection['candidates']))
        pred = apply_portable_class_cal(self.client['select_probabilities'], [.90, .55])['predicted']
        np.testing.assert_array_equal(pred, [0, 0, 0, 0, 2])

    def test_duplicate_benign_rows_do_not_inflate_route_support(self):
        self.client['fit_group_ids'] = np.array([0, 1, 2, 2])
        selection = select_portable_class_cal([self.client], min_route_benign_groups=2)
        self.assertEqual(selection['status'], 'no_feasible_operating_point')
        self.assertEqual(selection['class_thresholds'], [[.90, .90]])
        self.assertEqual(selection['support'][0]['routes'][1]['fit_benign_groups'], 1)

    def test_rejects_cross_client_group_overlap(self):
        second = {**self.client, 'fit_group_ids': np.arange(8, 12), 'select_group_ids': np.arange(12, 17)}
        with self.assertRaisesRegex(ValueError, 'occurs on both'):
            select_portable_class_cal([self.client, second], min_route_benign_groups=2)

    def test_false_alarm_on_independent_cal_prevents_promotion(self):
        self.client['select_probabilities'][2] = (.40, 0, .60)
        selection = select_portable_class_cal([self.client], min_route_benign_groups=2)
        self.assertEqual(selection['status'], 'no_feasible_operating_point')
        self.assertEqual(selection['class_thresholds'], [[.90, .90]])
        self.assertIn('far_at_or_below_cap', selection['candidates'][0]['failed_constraints'])
