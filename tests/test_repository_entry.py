"""Exercise the default entry's verified prepared-package path without a network."""

import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import main
from test_dataset_access import fixture, sha
from test_dataset_preparation import fixture as raw_fixture


class RepositoryEntryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def package(self, name):
        source = self.root / "source" / name
        fixture(source, name)
        archive = self.root / (name + ".zip")
        files = ("development.npz", "preparation.json", "selected_train_identities.npz")
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for filename in files:
                bundle.write(source / filename, filename)
        entry = {
            "url": "https://example.org/" + name + ".zip",
            "archive_sha256": sha(archive),
            "files": {filename: sha(source / filename) for filename in files},
        }
        return archive, entry

    def test_missing_default_cache_downloads_before_execute_and_then_reuses_it(self):
        name = main.DATASETS[0]
        archive, entry = self.package(name)
        root = self.root / "dataset"
        root.mkdir()
        (root / "download_sources.json").write_text(
            json.dumps({"schema_version": 1, "datasets": {name: entry}}), encoding="utf-8"
        )
        arguments = ["--dataset", name, "--data-root", str(root)]
        with (
            patch(
                "dataset_access._download_archive",
                side_effect=lambda url, dest, limit: shutil.copyfile(archive, dest),
            ) as download,
            patch("privacy_run.execute", return_value=9) as execute,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(main.main(arguments), 9)
        download.assert_called_once()
        self.assertTrue((root / name / "development.npz").is_file())
        self.assertEqual(sha(root / name / "development.npz"), entry["files"]["development.npz"])
        execute.assert_called_once()
        with (
            patch(
                "dataset_access._download_archive",
                side_effect=AssertionError("cache must be reused"),
            ),
            patch("privacy_run.execute", return_value=9),
        ):
            self.assertEqual(main.main(arguments + ["--offline"]), 9)

    def test_bad_hash_stops_before_any_model_is_trained(self):
        name = main.DATASETS[0]
        archive, entry = self.package(name)
        entry["archive_sha256"] = "0" * 64
        manifest = self.root / "bad_sources.json"
        manifest.write_text(
            json.dumps({"schema_version": 1, "datasets": {name: entry}}), encoding="utf-8"
        )
        root = self.root / "dataset"
        with (
            patch(
                "dataset_access._download_archive",
                side_effect=lambda url, dest, limit: shutil.copyfile(archive, dest),
            ),
            patch("privacy_run.execute") as execute,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(
                main.main(
                    [
                        "--dataset",
                        name,
                        "--data-root",
                        str(root),
                        "--download-manifest",
                        str(manifest),
                    ]
                ),
                2,
            )
        execute.assert_not_called()
        self.assertFalse((root / name).exists())

    def test_smoke_never_checks_or_downloads_real_datasets(self):
        with (
            patch(
                "dataset_access.check_repository_source",
                side_effect=AssertionError("real data used"),
            ),
            patch("privacy_run.execute", return_value=9),
        ):
            self.assertEqual(main.main(["--smoke"]), 9)

    def test_default_reader_prepares_flat_archive_in_offline_mode(self):
        name, raw, recipe, *_ = raw_fixture(self.root)
        archive = self.root / "archive"
        archive.mkdir()
        raw.rename(archive / raw.name)
        recipe["raw_directory"] = "archive"
        (self.root / "preparation" / name / "recipe.json").write_text(
            json.dumps(recipe), encoding="utf-8"
        )
        with (
            patch(
                "kaggle_source.ensure_kaggle_sources",
                side_effect=AssertionError("offline download"),
            ),
            patch("privacy_run.execute", return_value=9),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(
                main.main(["--dataset", name, "--data-root", str(self.root), "--offline"]), 9
            )
        self.assertTrue((self.root / name / "development.npz").is_file())

    def test_main_downloads_missing_csv_then_replays_the_declared_roles(self):
        name, raw, recipe, *_ = raw_fixture(self.root)
        original = raw.read_bytes()
        raw.unlink()
        recipe["raw_directory"] = "archive"
        recipe["kaggle"] = {"handle": "pushpakattarde/ciciov2024decimalcsv", "version": 1}
        (self.root / "preparation" / name / "recipe.json").write_text(
            json.dumps(recipe), encoding="utf-8"
        )

        def transport(handle, filename, output):
            output.mkdir(parents=True)
            saved = output / filename
            saved.write_bytes(original)
            return saved

        with (
            patch("kaggle_source._download_file", side_effect=transport) as download,
            patch("kaggle_source._temporary_parent", return_value=self.root),
            patch("privacy_run.execute", return_value=9),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(main.main(["--dataset", name, "--data-root", str(self.root)]), 9)
        download.assert_called_once()
        self.assertEqual((self.root / "archive" / raw.name).read_bytes(), original)
        args = main.parser().parse_args(["--dataset", name, "--data-root", str(self.root)])
        parts, classes, features = main.load_local_for_dataset(args, name)
        self.assertEqual(set(parts), {"train", "cal", "val"})
        self.assertEqual((len(features), len(classes)), (9, 6))


if __name__ == "__main__":
    unittest.main()
