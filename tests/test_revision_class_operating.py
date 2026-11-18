from unittest import TestCase

import numpy as np

from src.revision_class_operating import (
    SCHEMA,
    attack_routes,
    class_alarm_thresholds,
    select_class_operating_point,
)


def calibration_client(batches, *, receiver=False):
    client = {"receiver": receiver}
    for prefix, (scores, labels, routes) in zip(("threshold_fit", "operating_select"), batches, strict=True):
        client[f"{prefix}_scores"] = np.asarray(scores)
        client[f"{prefix}_labels"] = np.asarray(labels, dtype=np.int64)
        client[f"{prefix}_routes"] = np.asarray(routes, dtype=np.int64)
    return client


class ClassOperatingTests(TestCase):
    def choose(self, clients, reference, **overrides):
        settings = dict(
            classes=3,
            candidates=(0.005, 0.01, 0.0075),
            max_far=0.01,
            max_fnr_increase=0.01,
            min_route_benign=2,
            min_benign_rows=2,
            min_select_benign_rows=2,
            min_select_attack_rows=1,
            min_receiver_attack_rows=1,
        )
        settings.update(overrides)
        return select_class_operating_point(clients, reference_thresholds=reference, **settings)

    def test_inference_uses_attack_argmax_and_frozen_threshold_vector(self):
        p = np.array([[0.8, 0.15, 0.05], [0.05, 0.475, 0.475], [0.1, 0.15, 0.75]])
        np.testing.assert_array_equal(attack_routes(p), [1, 1, 2])
        np.testing.assert_array_equal(class_alarm_thresholds(p, [0.6, 0.85]), [0.6, 0.6, 0.85])
        self.assertEqual(class_alarm_thresholds(np.empty((0, 3)), [0.6, 0.85]).shape, (0,))
        with self.assertRaisesRegex(ValueError, "ordered entry"):
            class_alarm_thresholds(p, [0.6])
        with self.assertRaisesRegex(ValueError, "finite matrix"):
            attack_routes([[0.2, 0.4, float("nan")]])

    def test_routes_reduce_misses_without_multiple_far_budgets(self):
        blocks = [
            ([0.5, 0.6, 0.7, 0.85], [0, 0, 0, 0], [1, 1, 2, 2]),
            ([0.5, 0.6, 0.7, 0.85, 0.72, 0.95], [0, 0, 0, 0, 1, 2], [1, 1, 2, 2, 1, 2]),
        ]
        clients = [calibration_client(blocks, receiver=True), calibration_client(blocks)]
        result = self.choose(clients, [0.85, 0.85])
        self.assertEqual(result["schema"], SCHEMA)
        self.assertEqual(result["status"], "selected")
        self.assertEqual(result["selected_cal_far"], 0.01)
        self.assertEqual(result["class_thresholds"], [[0.6, 0.85], [0.6, 0.85]])
        self.assertEqual(result["thresholds"], [0.85, 0.85])
        self.assertEqual(
            [candidate["candidate_cal_far"] for candidate in result["candidates"]], [0.01, 0.0075, 0.005]
        )
        candidate = result["candidates"][0]
        self.assertEqual(candidate["receiver"]["far"], 0)
        self.assertEqual(candidate["overall"]["benign_rows"], 8)
        self.assertEqual(candidate["overall_fnr_delta"], -0.5)

    def test_sparse_routes_share_local_benign_bank(self):
        fit = ([0.55, 0.6, 0.7, 0.83], [0, 0, 0, 0], [1, 1, 2, 3])
        select = ([0.5, 0.59, 0.69, 0.8, 0.92, 0.73], [0, 0, 0, 0, 1, 1], [1, 1, 2, 3, 3, 1])
        result = self.choose([calibration_client([fit, select], receiver=True)], [0.8], classes=4)
        self.assertEqual(result["class_thresholds"], [[0.6, 0.83, 0.83]])
        support = result["support"]["class_route_support"][0]
        self.assertEqual(support["pooled_sparse_routes"], [2, 3])
        self.assertEqual(support["pool_fit_benign_rows"], 2)
        self.assertEqual(support["pool_select_benign_rows"], 2)

    def test_sparse_route_pool_cannot_borrow_benign_from_other_clients(self):
        fit = ([0.55, 0.6, 0.83], [0, 0, 0], [1, 1, 2])
        select = ([0.5, 0.59, 0.8, 0.92], [0, 0, 0, 1], [1, 1, 2, 2])
        clients = [calibration_client([fit, select], receiver=True), calibration_client([fit, select])]
        result = self.choose(clients, [0.8, 0.8])
        self.assertEqual(result["status"], "no_feasible_operating_point")
        self.assertEqual(result["class_thresholds"], [])
        self.assertIn("threshold_support", result["failed_constraints"])

    def test_candidates_never_compare_against_a_disabled_parent_gate(self):
        blocks = [
            ([0.52, 0.6, 0.83, 0.85], [0, 0, 0, 0], [1, 1, 2, 2]),
            ([0.5, 0.55, 0.72, 0.81, 0.9], [0, 0, 0, 0, 1], [1, 1, 2, 2, 2]),
        ]
        clients = [calibration_client(blocks, receiver=True)]
        result = self.choose(clients, [1])
        self.assertEqual(result["status"], "no_feasible_operating_point")
        self.assertIn("reference_active", result["failed_constraints"])
        with self.assertRaisesRegex(ValueError, "ordered entry"):
            self.choose(clients, [0.5, 0.9])
        clients[0]["operating_select_routes"][0] = 0
        with self.assertRaisesRegex(ValueError, "positive class routes"):
            self.choose(clients, [0.5])
