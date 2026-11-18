"""Verify archived loopback transport evidence without running a new benchmark."""

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.revision_local_transport import CLIENTS, DIMENSIONS, LOOPBACK, MEASUREMENTS, METHOD_SOURCES, ROUNDS
from src.revision_local_transport import _condition_rows, run_local_transport
from src.revision_v12 import _bindings, _hash, _write


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fake_rounds(dimension):
    for clients in CLIENTS:
        for round_index in range(ROUNDS):
            yield {
                "clients": str(clients),
                "dimension": str(dimension),
                "round": str(round_index),
                "seed": "2026",
                "clip": "0.1",
                "scale": "1000000",
                "elapsed_seconds": str(0.001 * (round_index + 1)),
                "quantize_and_share_seconds": "0.0005",
                "tcp_application_upload_bytes": "100000",
                "tcp_application_download_bytes": "60000",
                "tcp_application_total_bytes": "160000",
                "numeric_share_payload_bytes": str(8 * (2 * clients + 2) * dimension),
                "plaintext_quantized_sum_equal": "True",
                "all_clients_accepted": "True",
            }


def _base_summary():
    return {
        "schema_version": 1,
        "complete": True,
        "transport_environment": LOOPBACK,
        "independent_physical_hosts_verified": False,
        "detector_training_measured": False,
    }


def _dimension(root, dimension, *, bad_round):
    name = f"protocol_{dimension}"
    summary = {
        **_base_summary(),
        "dimension": dimension,
        "clients": list(CLIENTS),
        "rounds_per_condition": ROUNDS,
        "seed": 2026,
        "clip": 0.1,
        "scale": 1000000,
        "startup_in_round_times": False,
        "tls": False,
        "authentication": False,
        "differential_privacy": False,
        "endpoints": ["127.0.0.1:123", "127.0.0.1:456"],
    }
    raw = list(_fake_rounds(dimension))
    summary["conditions"] = [_condition_rows(summary, raw, count)[0] for count in CLIENTS]
    if bad_round:
        raw[0]["all_clients_accepted"] = "False"
    receipt = _save_measurement(root, name, raw, summary)
    conditions = [{"dimension": dimension, **row} for row in summary["conditions"]]
    return receipt, raw, conditions


def _save_measurement(root, name, raw, summary):
    folder = root / name
    folder.mkdir()
    _write_csv(folder / MEASUREMENTS[0], raw)
    _write(folder / MEASUREMENTS[1], summary)
    return {
        "status": "completed",
        "output": f"E:\\archive\\{name}",
        "artifacts": _bindings(folder, MEASUREMENTS),
    }


def _make_archive(root, *, bad_round=False):
    root.mkdir()
    state, measurements, conditions, sources = {}, [], [], {}
    for dimension in DIMENSIONS:
        receipt, raw, details = _dimension(root, dimension, bad_round=bad_round)
        state[f"protocol_{dimension}"] = {
            **receipt,
            "dimension": dimension,
            "transport_environment": LOOPBACK,
        }
        measurements.extend(raw)
        conditions.extend(details)
        sources[receipt["output"]] = receipt["artifacts"]
    state["protocol"] = _save_measurement(
        root,
        "protocol_index_0123456789ab",
        measurements,
        {**_base_summary(), "conditions": conditions, "source_measurements": sources},
    )
    _write(
        root / "suite_plan.json",
        {
            "schema_version": 1,
            "protocol_clients": list(CLIENTS),
            "protocol_dimensions": list(DIMENSIONS),
            "protocol_rounds": ROUNDS,
            "source_sha256": {name: "a" * 64 for name in METHOD_SOURCES},
        },
    )
    _write(root / "suite_state.json", state)
    _write(
        root / "suite_complete.json",
        {
            "schema_version": 1,
            "status": "completed",
            "plan_sha256": _hash(root / "suite_plan.json"),
            "state_sha256": _hash(root / "suite_state.json"),
            "steps": {
                name: {key: val for key, val in stage.items() if key != "artifacts"}
                for name, stage in state.items()
            },
        },
    )


class LocalTransportEvidenceTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.project = Path(temporary.name).resolve()
        self.archive = self.project / "archived"
        self.output = self.project / "finished"

    def _run(self):
        return run_local_transport(self.output, self.project, archive=self.archive)

    def test_verified_archive_has_all_rounds_and_preserves_byte_domains(self):
        _make_archive(self.archive)
        state = self._run()
        self.assertEqual(state["rounds_measured"], 180)
        self.assertEqual(state["transport_environment"], LOOPBACK)
        self.assertFalse(state["independent_physical_hosts_verified"])
        self.assertFalse(state["measurements_reexecuted"])
        table = "local_transport_summary.csv"
        with (self.output / table).open(encoding="utf-8", newline="") as stream:
            cases = list(csv.DictReader(stream))
        self.assertEqual(len(cases), 6)
        self.assertEqual(int(cases[0]["numeric_share_payload_bytes_per_round"]), 66240)
        self.assertEqual(float(cases[0]["tcp_application_mean_bytes_per_round"]), 160000)
        self.assertAlmostEqual(float(cases[0]["round_p95_ms"]), 28.55)
        self.assertEqual(state["artifacts"][table], _hash(self.output / table))
        for name, digest in state["archive_artifact_sha256"].items():
            self.assertEqual(_hash(self.archive / name), digest)
            self.assertEqual(state["artifacts"][f"archive/{name}"], digest)

    def test_changed_measurement_rejected_before_output(self):
        _make_archive(self.archive)
        (self.archive / "protocol_690/protocol_rounds.csv").write_text("invalid", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "archive artifact changed"):
            self._run()
        self.assertFalse(self.output.exists())

    def test_verified_digest_cannot_hide_incomplete_roster(self):
        _make_archive(self.archive, bad_round=True)
        with self.assertRaisesRegex(ValueError, "incomplete loopback round"):
            self._run()
        self.assertFalse(self.output.exists())

    def test_missing_archive_is_pending_without_new_files(self):
        self.assertEqual(self._run()["status"], "pending")
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
