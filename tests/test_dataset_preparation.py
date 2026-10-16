"""Source replay checks use small CSV fixtures, never network downloads."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import dataset_access
import dataset_preparation as preparation


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(root):
    name = "CIC-IoV-2024"
    profile = dataset_access.DATASET_PROFILES[name]
    names, features = profile["class_names"], profile["feature_names"]
    directory = root / "preparation" / name
    directory.mkdir(parents=True)
    raw = root / "raw" / name / "source.csv"
    raw.parent.mkdir(parents=True)
    labels = np.repeat(np.arange(len(names)), 25)
    x = np.arange(len(labels) * len(features), dtype=np.float32).reshape(len(labels), len(features))
    frame = pd.DataFrame(x, columns=features)
    frame["specific_class"] = [names[c] for c in labels]
    frame.to_csv(raw, index=False, lineterminator="\n")
    heldout = []
    for role, offset in (("cal", 0), ("val", 2)):
        # Reverse class order verifies that physical source order cannot replace row_in_role.
        rows = [int(c * 25 + offset + j) for c in reversed(range(len(names))) for j in range(2)]
        heldout.extend(
            {
                "role": role,
                "class": names[labels[row]],
                "source_file_id": 0,
                "zero_based_data_row": row,
                "row_in_role": i,
            }
            for i, row in enumerate(rows)
        )
    heldout = pd.DataFrame(heldout)
    heldout.to_csv(directory / "heldout_identities.csv", index=False, lineterminator="\n")
    # An additional reserved row per class must not become TRAIN.
    excluded = sorted(
        heldout.zero_based_data_row.tolist() + [c * 25 + 4 for c in range(len(names))]
    )
    pd.DataFrame(
        {
            "source_file_id": 0,
            "zero_based_data_row": excluded,
            "class": [names[labels[row]] for row in excluded],
        }
    ).to_csv(directory / "excluded_identities.csv", index=False, lineterminator="\n")
    recipe = {
        "schema_version": 1,
        "name": name,
        "seed": 42,
        "fraction": "full",
        "official_source": "https://example.org/dataset",
        "class_names": names,
        "feature_names": features,
        "pipeline": None,
        "encoding": "utf-8-sig",
        "label_column": "specific_class",
        "source_class_counts": [25] * len(names),
        "train_rows": 20 * len(names),
        "train_class_counts": [20] * len(names),
        "heldout_class_counts": {r: [2] * len(names) for r in ("cal", "val")},
        "source_files": [
            {
                "filename": raw.name,
                "bytes": raw.stat().st_size,
                "sha256": sha(raw),
                "csv_data_rows": len(labels),
                "url": None,
            }
        ],
        "heldout_identities": {
            "filename": "heldout_identities.csv",
            "sha256": sha(directory / "heldout_identities.csv"),
        },
        "excluded_identities": {
            "filename": "excluded_identities.csv",
            "sha256": sha(directory / "excluded_identities.csv"),
        },
    }
    (directory / "recipe.json").write_text(json.dumps(recipe), encoding="utf-8")
    return name, raw, recipe, x, labels, heldout, excluded


class DatasetPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.name, self.raw, self.recipe, self.x, self.y, self.heldout, self.excluded = fixture(
            self.root
        )

    def save_recipe(self):
        preparation.recipe_path(self.name, self.root).write_text(
            json.dumps(self.recipe), encoding="utf-8"
        )

    def test_full_replay_preserves_role_order_exclusions_and_seed_permutation(self):
        with patch(
            "dataset_access._download_archive", side_effect=AssertionError("network forbidden")
        ):
            preparation.check_raw_sources(self.name, self.root, allow_download=False)
            result = preparation.prepare_dataset(
                self.name, self.root, allow_download=False, chunk_size=13
            )
        eligible = np.asarray([row for row in range(len(self.y)) if row not in self.excluded])
        expected = eligible[np.random.default_rng(42).permutation(len(eligible))]
        with np.load(result, allow_pickle=False) as data:
            np.testing.assert_array_equal(data["X_train"], self.x[expected])
            np.testing.assert_array_equal(data["y_train"], self.y[expected])
            for role in ("cal", "val"):
                rows = (
                    self.heldout.loc[self.heldout.role == role]
                    .sort_values("row_in_role")
                    .zero_based_data_row
                )
                np.testing.assert_array_equal(data["X_" + role], self.x[rows])
                np.testing.assert_array_equal(data["y_" + role], self.y[rows])
            self.assertFalse(any(key.endswith("_test") for key in data.files))
        with np.load(result.parent / "selected_train_identities.npz", allow_pickle=False) as data:
            np.testing.assert_array_equal(data["source_row"], expected)
            np.testing.assert_array_equal(data["source_file_id"], np.zeros(len(expected), np.int32))
        audit = json.loads((result.parent / "preparation.json").read_text())
        self.assertEqual(audit["data_sha256"], sha(result))
        self.assertEqual(
            audit["identity_sha256"], sha(result.parent / "selected_train_identities.npz")
        )
        self.assertEqual(audit["verified_frozen_heldout_source_rows"], len(self.heldout))
        before = sha(result)
        self.assertEqual(
            preparation.prepare_dataset(self.name, self.root, allow_download=False), result
        )
        self.assertEqual(sha(result), before)

    def test_missing_raw_preflight_does_not_download_or_create_cache(self):
        self.raw.unlink()
        with patch(
            "dataset_access._download_archive", side_effect=AssertionError("network forbidden")
        ):
            with self.assertRaisesRegex(FileNotFoundError, "No verified direct download URL"):
                preparation.check_raw_sources(self.name, self.root)
            with self.assertRaisesRegex(FileNotFoundError, "Downloads are disabled"):
                preparation.check_raw_sources(self.name, self.root, allow_download=False)
        self.assertFalse((self.root / self.name).exists())

    def test_configured_source_preflight_is_non_mutating_and_offline_still_fails(self):
        self.raw.unlink()
        self.recipe["kaggle"] = {"handle": "owner/dataset", "version": 1}
        self.save_recipe()
        preparation.check_raw_sources(self.name, self.root)
        with self.assertRaisesRegex(FileNotFoundError, "Downloads are disabled"):
            preparation.check_raw_sources(self.name, self.root, allow_download=False)
        self.assertFalse((self.root / self.name).exists())

    def test_wrong_raw_hash_fails_before_preparing(self):
        contents = self.raw.read_bytes()
        self.raw.write_bytes(contents.replace(b"0.0", b"9.9", 1))
        with self.assertRaisesRegex(ValueError, "SHA256"):
            preparation.prepare_dataset(self.name, self.root, allow_download=False)
        self.assertFalse((self.root / self.name).exists())

    def test_source_row_metadata_tampering_is_rejected(self):
        path = preparation.recipe_path(self.name, self.root).parent / "heldout_identities.csv"
        path.write_text(path.read_text() + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "metadata hash differs"):
            preparation.check_raw_sources(self.name, self.root)

    def test_heldout_overlap_or_nonconsecutive_positions_is_rejected(self):
        path = preparation.recipe_path(self.name, self.root).parent / "heldout_identities.csv"
        rows = self.heldout.copy()
        rows.loc[0, "row_in_role"] = 10
        rows.to_csv(path, index=False, lineterminator="\n")
        self.recipe["heldout_identities"]["sha256"] = sha(path)
        self.save_recipe()
        with self.assertRaisesRegex(ValueError, "consecutive"):
            preparation.prepare_dataset(self.name, self.root, allow_download=False)

    def test_raw_row_count_mismatch_is_rejected(self):
        self.recipe["source_files"][0]["csv_data_rows"] += 1
        self.save_recipe()
        with self.assertRaisesRegex(ValueError, "Physical source row count changed"):
            preparation.prepare_dataset(self.name, self.root, allow_download=False)

    def test_declared_source_path_cannot_escape_raw_directory(self):
        self.recipe["source_files"][0]["filename"] = "../source.csv"
        self.save_recipe()
        with self.assertRaisesRegex(ValueError, "plain file name"):
            preparation.check_raw_sources(self.name, self.root)

    def test_archive_layout_is_flat_and_replays_without_downloading(self):
        archive = self.root / "archive"
        archive.mkdir()
        self.raw.rename(archive / self.raw.name)
        self.recipe["raw_directory"] = "archive"
        self.save_recipe()
        preparation.check_raw_sources(self.name, self.root, allow_download=False)
        result = preparation.prepare_dataset(self.name, self.root, allow_download=False)
        self.assertTrue(result.is_file())
        self.assertEqual(preparation.resolve_raw_directory(self.recipe, self.root), archive)

    def test_raw_directory_cannot_escape_dataset_root(self):
        for relative in (
            "../elsewhere",
            "/absolute",
            "C:/absolute",
            "raw/../elsewhere",
            "raw\\elsewhere",
        ):
            with self.subTest(relative=relative):
                self.recipe["raw_directory"] = relative
                self.save_recipe()
                with self.assertRaisesRegex(ValueError, "safe relative path"):
                    preparation.check_raw_sources(self.name, self.root)


if __name__ == "__main__":
    unittest.main()
