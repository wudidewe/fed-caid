"""Fault contracts and actual TCP framing without detector training."""

import json
import socket
import struct
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

import numpy as np

from src.revision_protocol_validation import (
    MAX_FRAME_BYTES,
    _CollectorState,
    _LocalCollectors,
    _accept_sessions,
    _receive_frame,
    _send_frame,
    _TCPPair,
    run_protocol_validation,
)
from src.secure_aggregation import (
    SecureAggregationError,
    SecureAggregationSession,
    SecureContract,
    _clip_quantize,
)


class ExistingSharingFaultTests(unittest.TestCase):
    def test_quantized_mean_matches_plaintext_with_clipping_and_negative_values(self):
        updates = [[-2.0, 0.3], [0.02, -0.04], [0.07, 0.08]]
        with SecureAggregationSession(["a", "b", "c"], 2, timeout=30) as session:
            contract = session.start_round("test", 0, "0" * 64)
            for identity, update in zip(session.clients, updates):
                session.submit(identity, update, contract=contract)
            actual, receipt = session.aggregate()
            expected = (
                np.mean([_clip_quantize(update, 2, 0.1, 1_000_000) for update in updates], axis=0) / 1_000_000
            )
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)
            self.assertTrue(receipt["all_clients_accepted"])

    def test_duplicate_submission_aborts_and_closes_session(self):
        with SecureAggregationSession(["a", "b", "c"], 1, timeout=30) as session:
            contract = session.start_round("test", 0, "0" * 64)
            session.submit("a", [0.03], contract=contract)
            with self.assertRaisesRegex(SecureAggregationError, "duplicate_client"):
                session.submit("a", [0.03], contract=contract)
            self.assertTrue(session._closed)
            with self.assertRaises(SecureAggregationError):
                session.aggregate()

    def test_stale_contract_aborts_before_update(self):
        with SecureAggregationSession(["a", "b", "c"], 1, timeout=30) as session:
            contract = session.start_round("test", 0, "0" * 64)
            with self.assertRaisesRegex(SecureAggregationError, "stale_or_changed_contract"):
                session.submit("a", [0.03], contract=replace(contract, round_index=1))
            self.assertTrue(session._closed)

    def test_nan_vector_aborts_session(self):
        with SecureAggregationSession(["a", "b", "c"], 1, timeout=30) as session:
            contract = session.start_round("test", 0, "0" * 64)
            with self.assertRaises(SecureAggregationError):
                session.submit("a", [float("nan")], contract=contract)
            self.assertTrue(session._closed)

    def test_worker_exit_prevents_aggregate_and_reopening(self):
        with SecureAggregationSession(["a", "b", "c"], 1, timeout=30) as session:
            contract = session.start_round("test", 0, "0" * 64)
            session._processes[0].terminate()
            session._processes[0].join(timeout=3)
            with self.assertRaisesRegex(SecureAggregationError, "disconnected"):
                session.submit("a", [0.03], contract=contract)
            self.assertTrue(session._closed)
            with self.assertRaises(SecureAggregationError):
                session.start_round("test", 1, "0" * 64)

    def test_timeout_branch_is_deterministic_and_closes_session(self):
        class UnresponsiveConnection:
            def send(self, _):
                pass

            def poll(self, _):
                return False

            def close(self):
                pass

        session = SecureAggregationSession(["a", "b", "c"], 1, timeout=0.001)
        session._connections = [UnresponsiveConnection(), UnresponsiveConnection()]
        with self.assertRaisesRegex(SecureAggregationError, "timeout"):
            session.start_round("test", 0, "0" * 64)
        self.assertTrue(session._closed)

    def test_unsafe_fixed_point_bound_is_rejected_before_workers(self):
        with self.assertRaisesRegex(ValueError, "overflow"):
            SecureAggregationSession(["a", "b", "c"], 1, clip=1e15, scale=100)


class TCPProtocolTests(unittest.TestCase):
    def contract(self):
        return SecureContract("test", 0, "0" * 64, ("a", "b", "c"), 2, 0.1, 1_000_000)

    def test_collector_rejects_unfinished_declared_sessions(self):
        listener = Mock()
        listener.accept.side_effect = socket.timeout
        with self.assertRaisesRegex(TimeoutError, '0 of 2 sessions'):
            _accept_sessions(listener, 2, 0.1)

        one, two = socket.socketpair()
        two.close()
        listener.accept.side_effect = None
        listener.accept.return_value = one, None
        with self.assertRaisesRegex(RuntimeError, '0 of 1 sessions'):
            _accept_sessions(listener, 1, 0.1)

    def test_incomplete_tcp_roster_has_no_aggregate_and_session_is_terminal(self):
        state, contract = _CollectorState(), self.contract()
        state.process({"op": "begin", "contract": contract.as_dict()})
        state.process({
            "op": "share", "contract_hash": contract.contract_hash, "client": "a", "share": [1, 2]
        })
        with self.assertRaisesRegex(SecureAggregationError, "incomplete_roster"):
            state.process({"op": "finish", "contract_hash": contract.contract_hash})
        self.assertIsNone(state.total)
        self.assertTrue(state.failed)
        with self.assertRaisesRegex(SecureAggregationError, "session_is_not_open"):
            state.process({"op": "begin", "contract": contract.as_dict()})

    def test_tcp_duplicate_stale_and_invalid_share_abort(self):
        contract = self.contract()
        for kind in ("duplicate", "stale", "invalid"):
            with self.subTest(kind=kind):
                state = _CollectorState()
                state.process({"op": "begin", "contract": contract.as_dict()})
                request = {
                    "op": "share",
                    "contract_hash": contract.contract_hash,
                    "client": "a",
                    "share": [1, 2],
                }
                if kind == "duplicate":
                    state.process(request)
                elif kind == "stale":
                    request["contract_hash"] = "0" * 64
                else:
                    request["share"] = [True, 2]
                with self.assertRaises(SecureAggregationError):
                    state.process(request)
                self.assertTrue(state.failed)
                self.assertIsNone(state.total)

    def test_framing_counts_bytes_and_caps_untrusted_length(self):
        one, two = socket.socketpair()
        try:
            message = {"op": "test", "value": [1, 2]}
            sent = _send_frame(one, message)
            received, size = _receive_frame(two)
            self.assertEqual(received, message)
            self.assertEqual(sent, size)
            one.sendall(struct.pack("!I", MAX_FRAME_BYTES + 1))
            with self.assertRaisesRegex(SecureAggregationError, "frame_size"):
                _receive_frame(two)
        finally:
            one.close()
            two.close()

    def test_real_tcp_benchmark_checks_all_rounds_and_reports_scope(self):
        with TemporaryDirectory() as temporary:
            out = Path(temporary) / "protocol"
            report = run_protocol_validation(out, clients=(3, 5), dimension=3, rounds=2, timeout=30)
            self.assertTrue(report["complete"])
            self.assertFalse(report["independent_physical_hosts_verified"])
            self.assertEqual(report["transport_environment"], "same_host_loopback")
            self.assertEqual([row["measured_rounds"] for row in report["conditions"]], [2, 2])
            self.assertTrue(all(row["all_plaintext_quantized_sums_equal"] for row in report["conditions"]))
            self.assertTrue(all(row["tcp_application_total_bytes"] > 0 for row in report["conditions"]))
            self.assertEqual(json.loads((out / "protocol_summary.json").read_text())["complete"], True)

    def test_network_failure_never_returns_partial_aggregate(self):
        contract = self.contract()
        with _LocalCollectors(1, 30) as endpoints, _TCPPair(tuple(endpoints), 30) as pair:
            pair.rpc([{"op": "begin", "contract": contract.as_dict()}] * 2)
            pair.rpc(
                [{"op": "share", "contract_hash": contract.contract_hash, "client": "a", "share": [1, 2]}] * 2
            )
            with self.assertRaisesRegex(SecureAggregationError, "collector_rejected"):
                pair.rpc([{"op": "finish", "contract_hash": contract.contract_hash}] * 2)
            self.assertTrue(pair.failed)
            with self.assertRaisesRegex(SecureAggregationError, "session_is_not_open"):
                pair.rpc([{"op": "finish", "contract_hash": contract.contract_hash}] * 2)

    def test_partial_endpoint_configuration_fails_before_output(self):
        with TemporaryDirectory() as temporary:
            out = Path(temporary) / "invalid"
            with self.assertRaisesRegex(ValueError, "both collector"):
                run_protocol_validation(out, collector_a="127.0.0.1:12345")
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
