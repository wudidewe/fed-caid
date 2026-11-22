"""Check score replay binding and reject modified arrays and reordered rows."""

import unittest
from hashlib import file_digest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.revision_score_cache import load_score_cache, save_score_cache


class ScoreCacheTests(unittest.TestCase):
    def test_bound_round_trip_and_tampering(self):
        union = [{"cal": np.array([4, 7]), "val": np.array([16, 1])}]
        configuration = {"objective": "multiclass"}
        identity = {"seed": 42}
        route = {"route_sha256": "recorded"}
        scores = np.arange(12, dtype=np.float32).reshape(4, 3) / 12
        branches = {"base_threshold": [scores], "head_local": [scores[::-1]]}
        names = ["BENIGN", "Attack 1", "Attack 2"]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = save_score_cache(
                root, identity, route, configuration, union, branches, class_names=names
            )
            recovered = load_score_cache(
                root,
                receipt,
                identity,
                route,
                configuration,
                union,
                expected_modes=set(branches),
                expected_classes=names,
            )
            self.assertEqual(set(recovered), set(branches))
            np.testing.assert_array_equal(recovered["head_local"][0], scores[::-1])
            with self.assertRaisesRegex(ValueError, "embedded metadata differs"):
                load_score_cache(
                    root, {**receipt, "modes": receipt["modes"][::-1]}, identity, route, configuration, union
                )
            with self.assertRaisesRegex(ValueError, "expected branches"):
                load_score_cache(
                    root, receipt, identity, route, configuration, union, expected_modes={"head_local"}
                )
            for classes in (4, names[::-1]):
                with self.assertRaisesRegex(ValueError, "expected columns"):
                    load_score_cache(
                        root, receipt, identity, route, configuration, union, expected_classes=classes
                    )
            for changed_config, changed_union in (
                ({"objective": "balanced_attack"}, union),
                (configuration, [{**union[0], "val": union[0]["val"][::-1]}]),
            ):
                with self.assertRaisesRegex(ValueError, "different configuration or route"):
                    load_score_cache(root, receipt, identity, route, changed_config, changed_union)
            with (root / receipt["file"]).open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                load_score_cache(root, receipt, identity, route, configuration, union)

    def test_legacy_schema_is_explicitly_rejected(self):
        union = [{"cal": np.array([1]), "val": np.array([2])}]
        scores = np.full((2, 2), 0.5)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = save_score_cache(root, {}, {}, {}, union, {"head_local": [scores]})
            with self.assertRaisesRegex(ValueError, "unsupported legacy schema"):
                old = {key: value for key, value in receipt.items() if key != "schema_version"}
                load_score_cache(root, old, {}, {}, {}, union)
            path = root / receipt["file"]
            np.savez_compressed(path, m0_c0=scores)
            with path.open("rb") as stream:
                receipt["sha256"] = file_digest(stream, "sha256").hexdigest()
            with self.assertRaisesRegex(ValueError, "no embedded metadata"):
                load_score_cache(root, receipt, {}, {}, {}, union)
