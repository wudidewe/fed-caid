"""Exercise the CAL-lock and fresh scoring sequence."""

from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
import unittest

from src.revision_nb15_threshold import _receiver_feasibility, run_nb15_threshold
from src.revision_v12 import _hash, _read


class Nb15ThresholdTests(unittest.TestCase):
    def test_synthetic_outputs_bind_the_global_cal_lock(self):
        checkout = Path(__file__).resolve().parent.parent
        with TemporaryDirectory() as directory:
            project = Path(directory).resolve()
            copyfile(checkout / "main.py", project / "main.py")
            output = run_nb15_threshold(project, device="cpu", smoke=True)
            path = output / "threshold_lock_42.json"
            lock, case = _read(path), _read(output / "seed_42.json")
            self.assertTrue(lock["locked_before_val_scoring"])
            self.assertEqual(lock["split"]["global_exact_feature_overlap"], 0)
            self.assertEqual(case["threshold_lock_sha256"], _hash(path))
            self.assertFalse(_read(output / "plan.json")["previous_run_inputs_allowed"])
            self.assertIn("SUMMARY.md", _read(output / "complete.json")["artifacts"])

    def test_feasibility_does_not_use_val_as_a_threshold_selector(self):
        seeds = tuple(range(5))
        minima = (0.00468, 0.00538, 0.01807, 0.01623, 0.00205)
        rows = [
            {
                "seed": seed, "arm": "reference_current", "scope": "receiver",
                "val_fnr": 0.1, "val_tpr_at_far_1pct_diagnostic": 0.9 - delta,
            }
            for seed, delta in zip(seeds, minima, strict=True)
        ]
        result = _receiver_feasibility(rows, seeds)
        self.assertFalse(result["selection_on_val"])
        self.assertGreater(result["paired_upper_95_lower_bound"], 0.015)
        self.assertTrue(result["joint_fnr_and_far_impossible_on_this_val"])
