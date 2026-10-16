"""Prepared-data identities and dormant downloader; tests never use the network."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import dataset_access as access


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture(directory, name="CIC-IoV-2024", *, extra=None):
    directory.mkdir(parents=True, exist_ok=True)
    profile = access.DATASET_PROFILES[name]
    arrays = {key: np.asarray(value) for key, value in profile.items()}
    width, classes = len(profile["feature_names"]), len(profile["class_names"])
    for role in ("train", "cal", "val"):
        arrays["X_" + role] = np.arange(classes * width, dtype=np.float32).reshape(classes, width)
        arrays["y_" + role] = np.arange(classes, dtype=np.int64)
    if extra:
        arrays.update(extra)
    path = directory / "development.npz"
    np.savez_compressed(path, **arrays)
    np.savez_compressed(directory / "selected_train_identities.npz", rows=np.arange(classes))
    (directory / "preparation.json").write_text(json.dumps({
        "data_sha256": sha(path), "identity_sha256": sha(directory / "selected_train_identities.npz")
    }), encoding="utf-8")
    return path


class DatasetAccessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def package(self, dataset="CIC-IoV-2024", members=None):
        source = self.root / "source"
        fixture(source, dataset)
        archive = self.root / "package.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
            for filename in sorted(access._PACKAGE_FILES):
                if members and filename in members:
                    output.writestr(members[filename], (source / filename).read_bytes())
                else:
                    output.write(source / filename, filename)
        entry = {"url": "https://example.org/prepared.zip", "archive_sha256": sha(archive),
                 "files": {filename: sha(source / filename) for filename in access._PACKAGE_FILES}}
        return archive, {"schema_version": 1, "datasets": {dataset: entry}}

    def mock_download(self, archive):
        return patch.object(access, "_download_archive", side_effect=lambda url, destination, limit:
                            shutil.copyfile(archive, destination))

    def test_all_four_complete_signatures_identify_and_load_without_mutation(self):
        for name in access.DATASETS:
            with self.subTest(dataset=name):
                path = fixture(self.root / name, name)
                before = sha(path)
                self.assertEqual(access.identify_dataset(path), name)
                parts, classes, features = access.read_local_dataset(path, name)
                self.assertEqual(classes, access.DATASET_PROFILES[name]["class_names"])
                self.assertEqual(features, access.DATASET_PROFILES[name]["feature_names"])
                self.assertEqual(set(parts), {"train", "cal", "val"})
                for x, y in parts.values():
                    self.assertEqual(x.shape, (len(classes), len(features)))
                    np.testing.assert_array_equal(y, np.arange(len(classes)))
                self.assertEqual(sha(path), before)

    def test_same_feature_count_does_not_confuse_network_datasets(self):
        self.assertEqual(len(access.DATASET_PROFILES["CIC-IDS-2017"]["feature_names"]), 68)
        self.assertEqual(len(access.DATASET_PROFILES["CIC-UNSW-NB15"]["feature_names"]), 68)
        path = fixture(self.root / "misleading_CIC-IDS-2017", "CIC-UNSW-NB15")
        self.assertEqual(access.identify_dataset(path), "CIC-UNSW-NB15")
        with self.assertRaisesRegex(ValueError, "conflicts"):
            access.read_local_dataset(path, "CIC-IDS-2017")

    def test_unknown_or_reordered_metadata_is_rejected(self):
        profile = access.DATASET_PROFILES["Car-Hacking"]
        for replacement in ({"feature_names": profile["feature_names"][::-1]},
                            {"class_names": profile["class_names"][::-1]},
                            {"feature_names": ["unknown"] * 10}):
            with self.subTest(replacement=replacement):
                path = fixture(self.root / "bad", "Car-Hacking", extra=replacement)
                with self.assertRaisesRegex(ValueError, "exact ordered"):
                    access.identify_dataset(path)

    def test_unknown_requested_identifier_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown dataset"):
            access.identify_dataset(self.root / "absent.npz", expected="CAN")
        with self.assertRaisesRegex(ValueError, "Unknown dataset"):
            access.read_repository_dataset("../CIC-IoV-2024", self.root / "repository")

    def test_extra_test_and_confirmation_members_are_forbidden(self):
        for key in ("X_test", "y_test", "confirmation", "unexpected"):
            with self.subTest(key=key):
                path = fixture(self.root / "forbidden", extra={key: np.zeros(1)})
                with self.assertRaisesRegex(ValueError, "exactly TRAIN/VAL/CAL"):
                    access.read_local_dataset(path)

    def test_original_shape_and_hash_guards_stay_active(self):
        profile = access.DATASET_PROFILES["CIC-IoV-2024"]
        path = fixture(self.root / "wrong_width", extra={"X_val": np.zeros((6, 8), np.float32)})
        self.assertEqual(access.identify_dataset(path), "CIC-IoV-2024")
        with self.assertRaises(ValueError):
            access.read_local_dataset(path)
        path = fixture(self.root / "bad_hash")
        audit = json.loads((path.parent / "preparation.json").read_text())
        audit["data_sha256"] = "0" * 64
        (path.parent / "preparation.json").write_text(json.dumps(audit))
        with self.assertRaisesRegex(ValueError, "hash differs"):
            access.read_local_dataset(path)
        path = fixture(self.root / "bad_identity")
        (path.parent / "selected_train_identities.npz").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "identity cache differs"):
            access.read_local_dataset(path)

    def test_repository_existing_cache_never_downloads_or_changes(self):
        path = fixture(self.root / "repo" / "Car-Hacking", "Car-Hacking")
        before = {p.name: sha(p) for p in path.parent.iterdir()}
        with patch.object(access, "_download_archive", side_effect=AssertionError("must not download")):
            result = access.read_repository_dataset("Car-Hacking", self.root / "repo")
        self.assertEqual(result[1], access.DATASET_PROFILES["Car-Hacking"]["class_names"])
        self.assertEqual(before, {p.name: sha(p) for p in path.parent.iterdir()})

    def test_unconfigured_download_stops_without_creating_directories(self):
        target = self.root / "missing_repository"
        with patch.object(access, "_download_archive", side_effect=AssertionError("must not download")):
            with self.assertRaisesRegex(ValueError, "No verified"):
                access.read_repository_dataset("Car-Hacking", target)
            empty = {"schema_version": 1, "datasets": {"Car-Hacking": None}}
            with self.assertRaisesRegex(ValueError, "No verified"):
                access.read_repository_dataset("Car-Hacking", target, empty)
        self.assertFalse(target.exists())

    def test_pinned_flat_package_installs_only_after_validation(self):
        archive, manifest = self.package("CIC-IDS-2017")
        repository = self.root / "repository"
        with self.mock_download(archive) as download:
            result = access.read_repository_dataset("CIC-IDS-2017", repository, manifest)
        self.assertEqual(download.call_count, 1)
        self.assertEqual(result[1], access.DATASET_PROFILES["CIC-IDS-2017"]["class_names"])
        self.assertEqual(set(p.name for p in (repository / "CIC-IDS-2017").iterdir()), access._PACKAGE_FILES)
        self.assertEqual(set(p.name for p in repository.iterdir()), {"CIC-IDS-2017"})

    def test_archive_and_member_hash_mismatch_leave_no_cache(self):
        archive, manifest = self.package()
        for kind in ("archive", "member"):
            with self.subTest(kind=kind):
                bad = copy.deepcopy(manifest)
                entry = bad["datasets"]["CIC-IoV-2024"]
                if kind == "archive":
                    entry["archive_sha256"] = "0" * 64
                else:
                    entry["files"]["development.npz"] = "0" * 64
                repository = self.root / kind
                with self.mock_download(archive), self.assertRaisesRegex(ValueError, "hash differs|SHA256 differs"):
                    access.read_repository_dataset("CIC-IoV-2024", repository, bad)
                self.assertEqual(list(repository.iterdir()), [])

    def test_wrong_dataset_package_cannot_install(self):
        archive, manifest = self.package("Car-Hacking")
        manifest["datasets"]["CIC-IoV-2024"] = manifest["datasets"].pop("Car-Hacking")
        repository = self.root / "repository"
        with self.mock_download(archive), self.assertRaisesRegex(ValueError, "conflicts"):
            access.read_repository_dataset("CIC-IoV-2024", repository, manifest)
        self.assertEqual(list(repository.iterdir()), [])

    def test_zip_traversal_duplicate_extra_and_symlink_are_rejected(self):
        archive, manifest = self.package()
        entry = manifest["datasets"]["CIC-IoV-2024"]
        cases = ("../outside", "C:/outside", "sub/development.npz", "sub\\outside")
        for index, filename in enumerate(cases):
            with self.subTest(filename=filename):
                unsafe = self.root / f"unsafe_{index}.zip"
                with zipfile.ZipFile(unsafe, "w") as bundle:
                    bundle.writestr(filename, b"bad")
                    bundle.writestr("preparation.json", b"{}")
                    bundle.writestr("selected_train_identities.npz", b"bad")
                bad = copy.deepcopy(manifest)
                bad["datasets"]["CIC-IoV-2024"]["archive_sha256"] = sha(unsafe)
                with self.mock_download(unsafe), self.assertRaisesRegex(ValueError, "Unsafe"):
                    access.read_repository_dataset("CIC-IoV-2024", self.root / f"repo_{index}", bad)
                self.assertFalse((self.root / "outside").exists())
        unsafe = self.root / "symlink.zip"
        with zipfile.ZipFile(unsafe, "w") as bundle:
            member = zipfile.ZipInfo("development.npz")
            member.create_system = 3
            member.external_attr = (stat.S_IFLNK | 0o777) << 16
            bundle.writestr(member, "../../outside")
            bundle.writestr("preparation.json", "{}")
            bundle.writestr("selected_train_identities.npz", b"bad")
        entry["archive_sha256"] = sha(unsafe)
        with self.mock_download(unsafe), self.assertRaisesRegex(ValueError, "Unsafe"):
            access.read_repository_dataset("CIC-IoV-2024", self.root / "symlink_repo", manifest)

    def test_package_limit_and_invalid_manifests_fail_before_installation(self):
        archive, manifest = self.package()
        for replacement in ({"max_uncompressed_bytes": 1}, {"max_download_bytes": False},
                            {"archive_sha256": "missing"}, {"files": {}}):
            with self.subTest(replacement=replacement):
                bad = copy.deepcopy(manifest)
                bad["datasets"]["CIC-IoV-2024"].update(replacement)
                with self.mock_download(archive), self.assertRaises(ValueError):
                    access.read_repository_dataset("CIC-IoV-2024", self.root / "limited", bad)
                self.assertFalse((self.root / "limited" / "CIC-IoV-2024").exists())

    def test_https_public_host_and_redirect_guards(self):
        for url in ("http://example.org/package.zip", "file:///dataset.zip", "https://localhost/x",
                    "https://127.0.0.1/x", "https://10.0.0.1/x", "https://[::1]/x",
                    "https://user:secret@example.org/x", "https://example.org:8443/x",
                    "https://example.org/x#fragment", "https://service.local/x"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                access._validate_download_url(url)
        private_dns = [(2, 1, 6, "", ("192.168.1.1", 443))]
        with patch.object(access.socket, "getaddrinfo", return_value=private_dns), self.assertRaises(ValueError):
            access._validate_download_url("https://example.org/x", resolve=True)
        handler = access._HttpsRedirect()
        for destination in ("file:///secret", "http://example.org/x", "https://localhost/x"):
            with self.subTest(destination=destination), self.assertRaises(ValueError):
                handler.redirect_request(None, None, 302, "", {}, destination)

    def test_incomplete_existing_directory_is_preserved_without_download(self):
        repository = self.root / "repository"
        target = repository / "CIC-IoV-2024"
        target.mkdir(parents=True)
        (target / "user_file.txt").write_text("keep")
        with patch.object(access, "_download_archive", side_effect=AssertionError("must not download")):
            with self.assertRaises(FileNotFoundError):
                access.read_repository_dataset("CIC-IoV-2024", repository)
        self.assertEqual((target / "user_file.txt").read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
