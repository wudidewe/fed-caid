"""Check serialized collector requests and defensive experiment inputs."""

import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event

import numpy as np

from src.privacy_core import ClassScoreAdapter, fit_class_adapter
from src.revision_measurement import benchmark_tcp
from src.revision_nb15_low_far import _candidate_seed
from src.secure_aggregation import SecureAggregationError, SecureAggregationSession


class AggregationIntegrityTests(unittest.TestCase):
    def test_concurrent_clients_reconstruct_the_same_complete_roster_mean(self):
        clients = ('a', 'b', 'c')
        updates = ([0.03], [0.05], [-0.02])
        barrier = Barrier(len(clients))
        with SecureAggregationSession(clients, 1, timeout=30) as session:
            contract = session.start_round('concurrent', 0, '0' * 64)

            def submit(pair):
                barrier.wait(timeout=5)
                return session.submit(*pair, contract=contract)

            with ThreadPoolExecutor(max_workers=len(clients)) as executor:
                results = list(executor.map(submit, zip(clients, updates)))
            self.assertTrue(all(result['accepted'] for result in results))
            average, receipt = session.aggregate()
        np.testing.assert_array_equal(average, [0.02])
        self.assertEqual(receipt['communication_counters']['sent_messages'], 10)
        self.assertEqual(receipt['communication_counters']['received_messages'], 10)

    def test_deadline_covers_blocked_send_and_partial_receive(self):
        for blocked in ('send', 'recv'):
            with self.subTest(operation=blocked):
                released = Event()

                class StalledConnection:
                    def send(self, message):
                        if blocked == 'send' and not released.wait(5):
                            raise RuntimeError('send was not released')

                    def poll(self, timeout):
                        return True

                    def recv(self):
                        if blocked == 'recv' and not released.wait(5):
                            raise RuntimeError('receive was not released')
                        return {'ok': True}

                    def close(self):
                        released.set()

                session = SecureAggregationSession(('a', 'b', 'c'), 1, timeout=0.1)
                session._connections = [StalledConnection(), StalledConnection()]
                started = time.monotonic()
                with self.assertRaisesRegex(SecureAggregationError, 'aggregation_worker_timeout'):
                    session.start_round('stalled', 0, '0' * 64)
                self.assertLess(time.monotonic() - started, 2)
                self.assertTrue(session._closed)
                self.assertTrue(released.is_set())


class DefensiveExperimentInputTests(unittest.TestCase):
    def test_outside_radius_and_projected_vector_have_the_same_training_anchor(self):
        p = [[0.7, 0.3], [0.4, 0.6], [0.55, 0.45]]
        y = [0, 1, 1]
        outside = [10.0]
        projected = ClassScoreAdapter(2).project(outside)
        options = {'epochs': 2, 'prox': 0.5}
        expected = fit_class_adapter(projected, p, y, [0.0, 0.0], **options)
        actual = fit_class_adapter(outside, p, y, [0.0, 0.0], **options)
        np.testing.assert_array_equal(actual, expected)

    def test_tcp_probe_rejects_invalid_dimensions_before_creating_connections(self):
        for clients, payload, repeats in ((0, 1, 1), (1, 0, 1), (1, 1, 0), (True, 1, 1)):
            with self.subTest(clients=clients, payload=payload, repeats=repeats):
                with self.assertRaises(ValueError):
                    benchmark_tcp(clients, payload, repeats=repeats)

    def test_nb15_rng_reference_mapping_and_opt_in_independent_streams(self):
        first, second = (42, 2), (44, 0)
        legacy = [_candidate_seed(*item, independent_client_rng=False) for item in (first, second)]
        independent = [_candidate_seed(*item, independent_client_rng=True) for item in (first, second)]
        self.assertEqual(legacy, [44, 44])
        self.assertEqual(independent, [list(first), list(second)])
        streams = [np.random.default_rng(seed).integers(0, 256, size=32) for seed in independent]
        self.assertFalse(np.array_equal(*streams))
