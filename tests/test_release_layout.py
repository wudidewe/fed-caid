"""Integration checks for source relocation and reproducible source archives."""

import importlib
import os
import pickle
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import ids_core
import privacy_run
import utils
from _iov_layout import SNAPSHOT_SUPPORT_FILES, source_path


ROOT = Path(__file__).resolve().parents[1]


class ReleaseLayoutTests(unittest.TestCase):
    def test_shared_modules_resolve_to_the_canonical_source(self):
        for name in ("ids_core", "privacy_run", "secure_aggregation", "dataset_access"):
            first = importlib.import_module(name)
            self.assertIs(first, importlib.import_module(name))
            self.assertEqual(Path(first.__file__).resolve(), ROOT / "src" / f"{name}.py")
        self.assertIs(ids_core.Transform, utils.Transform)
        self.assertEqual(ids_core.Transform.__module__, "utils")
        self.assertIs(pickle.loads(b"cUtils\nTransform\n."), utils.Transform)
        self.assertEqual(privacy_run.ROOT, ROOT)

    def test_source_archive_contains_only_present_release_modules(self):
        self.assertEqual(privacy_run.DEFAULT_DATA, ROOT / "dataset")
        for relative in SNAPSHOT_SUPPORT_FILES:
            self.assertTrue(source_path(relative).is_file(), relative)
        for removed in (
            "privacy_history.py",
            "conditional_run.py",
            "attack_calibration.py",
            "aggregation_comparison.py",
        ):
            self.assertNotIn(removed, SNAPSHOT_SUPPORT_FILES)
            self.assertFalse((ROOT / "src" / removed).exists())

    def test_flat_execution_archive_runs_without_checkout_imports(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            archive = Path(folder)
            for relative in SNAPSHOT_SUPPORT_FILES:
                target = archive / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_path(relative), target)
            environment = dict(os.environ)
            environment.pop("PYTHONPATH", None)
            result = subprocess.run(
                [sys.executable, "-B", str(archive / "main.py"), "--help"],
                cwd=archive,
                env=environment,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--data-root", result.stdout)
            self.assertIn("--dataset", result.stdout)
            self.assertNotIn("--mechanism", result.stdout)
            self.assertNotIn("--improve-from", result.stdout)

    def test_module_entry_refuses_to_report_zero_tests_outside_checkout(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
            # Simulate a site-packages installation for the missing-tests guard.
            installed = Path(folder) / "site-packages"
            installed.mkdir()
            for filename in ("main.py", "_iov_layout.py", "iov_instruction.py"):
                shutil.copyfile(ROOT / "src" / filename, installed / filename)
            environment["PYTHONPATH"] = str(installed)
            result = subprocess.run(
                [sys.executable, "-B", "-m", "iov_instruction", "--check"],
                cwd=folder,
                env=environment,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("source checkout", result.stdout)


if __name__ == "__main__":
    unittest.main()
