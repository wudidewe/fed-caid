"""Exercise CPU replay of a run whose model scores were generated on CUDA."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.revision_next import (
    PRE_V12_EVALUATORS,
    PRE_V12_SHARED_HEAD_SHA256,
    V12_COMPATIBLE_SHARED_HEAD_SHA256,
    _checked_replay_configuration,
    _normalize,
    _parser,
    _runtime,
)


class ReplayConfigurationTests(unittest.TestCase):
    def test_original_training_device_is_not_a_replay_requirement(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            (source / "run.json").write_text(json.dumps({"settings": {"device": "cuda"}}), encoding="utf-8")
            parser = _parser()
            cli = parser.parse_args([
                "--operating-guard",
                "--smoke",
                "--device",
                "cpu",
                "--replay-from",
                str(source),
                "--out",
                str(source.with_name("replay")),
            ])
            with (
                patch("src.revision_next.torch.cuda.is_available", return_value=False),
                patch(
                    "src.revision_next.torch.cuda.get_device_name",
                    side_effect=AssertionError("Replay queried GPU"),
                ),
            ):
                _normalize(parser, cli)
                runtime = _runtime(cli)
        self.assertEqual(runtime["training_device"], "cuda")
        self.assertEqual(runtime["execution_device"], "cpu")
        self.assertIsNone(runtime["gpu"])

    def test_v12_replays_only_the_recognized_legacy_scoring_receipt(self):
        evaluator, external = sorted(PRE_V12_EVALUATORS)[0]
        original = {
            "version": 4,
            "settings": {"operating_policy": "fixed"},
            "code_sha256": {
                "revision_next.py": evaluator,
                "revision_external.py": external,
                "revision_shared_head.py": PRE_V12_SHARED_HEAD_SHA256,
                "utils.py": "same",
            },
        }
        current = {
            **original,
            "code_sha256": {
                "revision_next.py": "current",
                "revision_external.py": "current",
                "revision_shared_head.py": V12_COMPATIBLE_SHARED_HEAD_SHA256,
                "utils.py": "same",
            },
        }
        self.assertEqual(_checked_replay_configuration(original, current)["code"], original["code_sha256"])
        for name in ("revision_shared_head.py", "utils.py"):
            altered = {**current, "code_sha256": {**current["code_sha256"], name: "unrecognized"}}
            with self.subTest(name=name), self.assertRaises(ValueError):
                _checked_replay_configuration(original, altered)
