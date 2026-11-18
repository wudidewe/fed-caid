import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.revision_lan_evidence import preflight_lan, run_lan, run_loopback
from src.revision_v12 import _hash


class LanEvidenceTests(unittest.TestCase):
    def test_fresh_loopback_measures_real_collectors_and_binds_receipts(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "fresh_loopback"
            receipt = run_loopback(output)
            self.assertEqual(receipt["rounds_measured"], 180)
            self.assertEqual(receipt["conditions_measured"], 6)
            self.assertEqual(receipt["transport_environment"], "same_host_loopback")
            self.assertTrue(receipt["all_plaintext_quantized_sums_equal"])
            self.assertFalse(receipt["independent_physical_hosts_verified"])
            self.assertFalse(receipt["detector_training_measured"])
            self.assertFalse(receipt["inference_measured"])
            self._verify_loopback_files(output, receipt["artifacts"])

    def _verify_loopback_files(self, output, artifacts):
        self.assertEqual(len(artifacts), 7)
        for name, digest in artifacts.items():
            self.assertEqual(_hash(output / name), digest)
        with (output / "loopback_summary.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(
            {(int(row["dimension"]), int(row["clients"]), int(row["measured_rounds"])) for row in rows},
            {(dimension, clients, 30) for dimension in (690, 1035) for clients in (3, 5, 10)},
        )
        self.assertTrue(all(float(row["round_p95_ms"]) >= float(row["round_median_ms"]) > 0 for row in rows))
        self.assertTrue(
            all(
                int(row["tcp_application_total_bytes"])
                == int(row["tcp_application_upload_total_bytes"])
                + int(row["tcp_application_download_total_bytes"])
                for row in rows
            )
        )

    def test_loopback_refuses_existing_directory_and_does_not_publish_failure(self):
        with (
            TemporaryDirectory() as temporary,
            patch(
                "src.revision_protocol_validation.run_protocol_validation", side_effect=RuntimeError("failure")
            ) as run,
        ):
            output = Path(temporary) / "fresh_loopback"
            with self.assertRaisesRegex(RuntimeError, "failure"):
                run_loopback(output)
            self.assertTrue((output / "loopback_plan.json").is_file())
            self.assertFalse((output / "complete.json").exists())
            with self.assertRaises(FileExistsError):
                run_loopback(output)
            run.assert_called_once()

    def test_missing_manifest_never_contacts_a_collector(self):
        with (
            TemporaryDirectory() as temporary,
            patch("src.revision_protocol_validation.run_protocol_validation") as run,
        ):
            root = Path(temporary)
            record = run_lan(root / "out", root / "missing.json")
            self.assertEqual(record["status"], "pending")
            self.assertFalse((root / "out").exists())
            run.assert_not_called()

    def test_preflight_rejects_loopback_and_keeps_physical_verification_false(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "lan.json"
            row = {"os": "Linux", "cpu": "CPU", "ram": "8 GiB"}
            config = {
                "version": 1,
                "topology": "controlled private wired switch",
                "orchestrator": {**row, "physical_host_id": "A"},
                "collector_a": {**row, "physical_host_id": "B", "endpoint": "127.0.0.1:8761"},
                "collector_b": {**row, "physical_host_id": "C", "endpoint": "192.168.1.22:8762"},
            }
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-loopback"):
                preflight_lan(path)
            config["collector_a"]["endpoint"] = "192.168.1.21:8761"
            path.write_text(json.dumps(config), encoding="utf-8")
            check = preflight_lan(path)
            self.assertEqual(check["status"], "ready")
            self.assertFalse(check["independent_physical_hosts_verified"])
