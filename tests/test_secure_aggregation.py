"""Protocol properties, rather than model-specific implementation snapshots."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from secure_aggregation import (
    PRIME,
    SecureAggregationError,
    SecureAggregationSession,
    _clip_quantize,
    _split_quantized,
)


CLIENTS = ("vehicle-a", "vehicle-b", "vehicle-c")
PARENT = "a" * 64


class _FixedRandom:
    def randrange(self, upper):
        assert upper == PRIME
        return 123456789


class SecureAggregationTests(unittest.TestCase):
    def session(self, dimension=3, **kwargs):
        return SecureAggregationSession(CLIENTS, dimension, **kwargs)

    def assert_closed(self, channel):
        self.assertTrue(channel._closed)
        self.assertTrue(all(not process.is_alive() for process in channel._processes))

    def test_exact_fixed_point_aggregate_and_persistent_workers(self):
        with self.session() as channel:
            processes = tuple(p.pid for p in channel._processes)
            for round_index in range(3):
                contract = channel.start_round("experiment", round_index, PARENT)
                updates = [
                    np.array([0.02, -0.03, 0.01]),
                    np.array([0.01, 0.04, -0.005]),
                    np.array([-0.01, 0.02, 0.03]),
                ]
                for client, update in zip(CLIENTS, updates):
                    ack = channel.submit(client, update, contract)
                    self.assertEqual(set(ack), {"accepted", "contract_hash"})
                actual, receipt = channel.aggregate()
                np.testing.assert_allclose(actual, np.mean(updates, axis=0), atol=1e-15, rtol=0)
                self.assertEqual(tuple(p.pid for p in channel._processes), processes)
                self.assertEqual(receipt["roster_size"], 3)
                self.assertFalse(receipt["differential_privacy"])
                self.assertNotIn("aggregate_share", receipt)
                self.assertNotIn("clients", receipt)
                self.assertEqual(receipt["numeric_share_upload_bytes"], 2 * 3 * 3 * 8)
                self.assertEqual(receipt["aggregate_share_download_bytes"], 2 * 3 * 8)
                self.assertEqual(receipt["communication_counters"]["sent_messages"], 10)
                self.assertEqual(receipt["communication_counters"]["received_messages"], 10)
                self.assertGreater(receipt["serialized_application_bytes"], 192)
                self.assertGreaterEqual(
                    receipt["local_ipc_frame_bytes"], receipt["serialized_application_bytes"]
                )
        self.assert_closed(channel)

    def test_large_finite_update_is_clipped_without_overflow(self):
        updates = [
            np.array([1e300, -1e300, 0.0]),
            np.array([0.100000123, 0.0, 0.0]),
            np.array([0.01234567, -0.02345678, 0.00345678]),
        ]
        expected = []
        for update in updates:
            maximum = np.max(np.abs(update))
            direction = update / maximum
            expected.append(direction * min(maximum, 0.1 / np.linalg.norm(direction)))
        with self.session() as channel:
            channel.start_round("clipping", 0, PARENT)
            for client, update in zip(CLIENTS, updates):
                channel.submit(client, update)
            actual, receipt = channel.aggregate()
        error = np.max(np.abs(actual - np.mean(expected, axis=0)))
        self.assertLessEqual(error, receipt["mean_coordinate_quantization_bound"] + 1e-15)
        self.assertLessEqual(np.linalg.norm(actual), 0.1 + np.sqrt(3) * 0.5e-6)

    def test_single_share_is_compatible_with_different_updates(self):
        # For a uniform first share, EVERY field input maps to one unique other
        # share. This verifies reconstruction without pretending to prove MPC.
        first, second = _split_quantized([-11, 0, 37], _FixedRandom())
        other_first, other_second = _split_quantized([5, -9, -42], _FixedRandom())
        self.assertEqual(first, other_first)
        self.assertNotEqual(second, other_second)
        self.assertEqual([(a + b) % PRIME for a, b in zip(first, second)], [(-11) % PRIME, 0, 37])
        self.assertTrue(all(0 <= value < PRIME for value in first + second))

    def test_fresh_system_random_shares(self):
        first, second = _split_quantized([1, 2, 3])
        repeated, _ = _split_quantized([1, 2, 3])
        self.assertNotEqual(first, repeated)
        self.assertEqual([(a + b) % PRIME for a, b in zip(first, second)], [1, 2, 3])

    def test_duplicate_client_aborts_both_workers(self):
        with self.session() as channel:
            channel.start_round("duplicate", 0, PARENT)
            channel.submit(CLIENTS[0], [0, 0, 0])
            with self.assertRaisesRegex(SecureAggregationError, "duplicate_client"):
                channel.submit(CLIENTS[0], [0, 0, 0])
            self.assert_closed(channel)

    def test_missing_client_never_releases_partial_sum(self):
        with self.session() as channel:
            channel.start_round("missing", 0, PARENT)
            channel.submit(CLIENTS[0], [0, 0, 0])
            channel.submit(CLIENTS[1], [0, 0, 0])
            with self.assertRaisesRegex(SecureAggregationError, "incomplete_roster"):
                channel.aggregate()
            self.assert_closed(channel)

    def test_stale_parent_contract_aborts(self):
        with self.session() as channel:
            contract = channel.start_round("stale", 0, PARENT)
            stale = replace(contract, parent_model="b" * 64)
            with self.assertRaisesRegex(SecureAggregationError, "stale_or_changed_contract"):
                channel.submit(CLIENTS[0], [0, 0, 0], stale)
            self.assert_closed(channel)

    def test_replayed_round_aborts(self):
        with self.session() as channel:
            channel.start_round("replay", 0, PARENT)
            for client in CLIENTS:
                channel.submit(client, [0, 0, 0])
            channel.aggregate()
            with self.assertRaisesRegex(SecureAggregationError, "replayed_or_skipped_round"):
                channel.start_round("replay", 0, PARENT)
            self.assert_closed(channel)

    def test_unregistered_client_and_bad_vectors_abort(self):
        cases = [
            ("unknown", [0, 0, 0], "unregistered_client"),
            (CLIENTS[0], [0, 0], "invalid_update_vector"),
            (CLIENTS[0], [0, np.nan, 0], "invalid_update_vector"),
            (CLIENTS[0], [0, np.inf, 0], "invalid_update_vector"),
        ]
        for client, update, expected in cases:
            with self.subTest(expected=expected, update=update):
                with self.session() as channel:
                    channel.start_round("bad-update", 0, PARENT)
                    with self.assertRaisesRegex(SecureAggregationError, expected):
                        channel.submit(client, update)
                    self.assert_closed(channel)

    def test_worker_rejects_out_of_range_share(self):
        with self.session() as channel:
            contract = channel.start_round("field-range", 0, PARENT)
            payload = {
                "contract_hash": contract.contract_hash,
                "client_id": CLIENTS[0],
                "share": [PRIME, 0, 0],
            }
            with self.assertRaisesRegex(SecureAggregationError, "invalid_field_share"):
                channel._rpc("share", [payload, payload])
            self.assert_closed(channel)

    def test_worker_rejects_changed_round_metadata(self):
        with self.session() as channel:
            channel.start_round("metadata", 0, PARENT)
            payload = {"contract_hash": "f" * 64, "client_id": CLIENTS[0], "share": [1, 2, 3]}
            with self.assertRaisesRegex(SecureAggregationError, "stale_or_changed_contract"):
                channel._rpc("share", [payload, payload])
            self.assert_closed(channel)

    def test_timeout_closes_both_workers_without_fallback(self):
        with self.session() as channel:
            # Force a transport timeout after workers have started; no sleeps.
            with patch.object(channel._connections[0], "poll", return_value=False):
                with self.assertRaisesRegex(SecureAggregationError, "aggregation_worker_timeout"):
                    channel.start_round("timeout", 0, PARENT)
            self.assert_closed(channel)

    def test_worker_death_fails_closed(self):
        with self.session() as channel:
            channel.start_round("worker-death", 0, PARENT)
            channel._processes[0].terminate()
            channel._processes[0].join(timeout=2)
            with self.assertRaisesRegex(SecureAggregationError, "aggregation_worker_disconnected"):
                channel.submit(CLIENTS[0], [0, 0, 0])
            self.assert_closed(channel)

    def test_invalid_public_parameters_rejected_before_spawn(self):
        for roster in [("a", "b"), ("a", "a", "b"), "abc"]:
            with self.assertRaises(ValueError):
                SecureAggregationSession(roster, 2)
        for extra in [
            {"dimension": 0},
            {"scale": 0},
            {"scale": True},
            {"clip": float("inf")},
            {"clip": 1e15, "scale": 1_000_000},
            {"timeout": 0},
        ]:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                values = {"dimension": 2, **extra}
                SecureAggregationSession(CLIENTS, **values)

    def test_quantizer_coordinate_error(self):
        values = np.array([0.013579123, -0.024681234, 0.032165489])
        integers = _clip_quantize(values, 3, 0.1, 1_000_000)
        actual = np.asarray(integers) / 1_000_000
        self.assertLessEqual(np.max(np.abs(actual - values)), 0.5e-6 + 1e-16)


if __name__ == "__main__":
    unittest.main()
