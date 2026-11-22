"""Exercise the NB15-only entry and development artifacts on synthetic data."""

import csv
import json
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
import unittest

from src.revision_nb15_low_far import run_nb15_low_far


class Nb15LowFarTests(unittest.TestCase):
    def test_synthetic_workflow_writes_fresh_matched_outputs(self):
        checked_out = Path(__file__).resolve().parent.parent
        with TemporaryDirectory() as temporary:
            project = Path(temporary).resolve()
            copyfile(checked_out / "main.py", project / "main.py")
            output = run_nb15_low_far(project, device="cpu", smoke=True)
            plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
            receipt = json.loads((output / "complete.json").read_text(encoding="utf-8"))
            self.assertFalse(plan["previous_run_inputs_allowed"])
            self.assertFalse(plan["test_evaluated"])
            self.assertEqual(plan["seeds"], [42])
            self.assertEqual(receipt["status"], "completed")
            self.assertIn("per_class.csv", receipt["artifacts"])
            with (output / "working_points.csv").open(encoding="utf-8-sig", newline="") as stream:
                points = list(csv.DictReader(stream))
            self.assertEqual({row["arm"] for row in points}, set(plan["arms"]))
            self.assertEqual({row["scope"] for row in points}, {"receiver", "overall"})
