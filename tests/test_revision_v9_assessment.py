"""Guard the distinct subtype and absolute-FAR decisions of V9."""

import unittest

from src.revision_next import CONDITIONS, DEVELOPMENT_THRESHOLDS, _fixed_assessment


class FixedSubtypeAssessmentTests(unittest.TestCase):
    def assess(
        self,
        *,
        receiver_far=0.008,
        macro_lower=-0.004,
        baseline_gain=0.08,
        binary_increase=0.0,
        seeds=5,
        supported=None,
        selection=None,
    ):
        key = ("CIC-IDS-2017", "train_absence", 0.3, 3, "DoS Hulk", "reserved")
        identity = dict(zip(CONDITIONS, key, strict=True))

        def record(scope, contrast, metric, mean, low, high, absolute=None):
            return {
                **identity,
                "scope": scope,
                "contrast": contrast,
                "metric": metric,
                "mean": mean,
                "ci95_low": low,
                "ci95_high": high,
                "candidate_max": absolute,
                "candidate_min": absolute,
                "seed_sd": 0.0,
                "seeds_with_metric": seeds if supported is None else supported,
            }

        rows = [
            record("receiver", "fixed_aggregation", "f1", 0.08, 0.04, 0.12),
            record(
                "receiver",
                "fixed_against_base",
                "f1",
                baseline_gain,
                baseline_gain - 0.02,
                baseline_gain + 0.02,
            ),
            record("overall", "fixed_against_base", "macro_f1", 0.004, macro_lower, 0.01),
            record(
                "overall",
                "fixed_against_base",
                "far",
                binary_increase,
                binary_increase,
                binary_increase,
                0.007,
            ),
            record("overall", "fixed_against_base", "fnr", 0.0, 0.0, 0.0, 0.05),
            record("receiver", "fixed_against_base", "receiver_far", 0.0, 0.0, 0.0, receiver_far),
            record("coverage", "fixed_against_base", "enabled_attack_fraction", 0.0, 0.0, 0.0, 1.0),
        ]
        (answer,) = _fixed_assessment(
            rows, {key: ["evaluated"] * seeds}, DEVELOPMENT_THRESHOLDS, "reserved", selection=selection
        )
        return answer

    def test_positive_signal_and_absolute_far_report_separately(self):
        eligible = self.assess()
        self.assertEqual(eligible["status"], "development_signal_requires_external_confirmation")
        self.assertTrue(eligible["subtype_signal"])
        too_noisy = self.assess(receiver_far=0.012)
        self.assertEqual(too_noisy["status"], "subtype_gain_absolute_far_not_met")
        self.assertTrue(too_noisy["binary_metrics_unchanged"])
        self.assertFalse(too_noisy["observed_absolute_far_met"])

    def test_regression_against_base_and_same_alarm_requirements(self):
        for kwargs in ({"macro_lower": -0.02}, {"baseline_gain": 0.0}, {"binary_increase": 0.001}):
            with self.subTest(kwargs=kwargs):
                self.assertEqual(self.assess(**kwargs)["status"], "criterion_not_met")
        self.assertEqual(self.assess(supported=4)["status"], "insufficient_paired_support")
        self.assertEqual(self.assess(seeds=1)["status"], "insufficient_paired_support")

    def test_no_feasible_operating_point_is_explicit(self):
        key = ("CIC-IDS-2017", "train_absence", 0.3, 3, "DoS Hulk", "reserved")
        answer = self.assess(
            selection={key: [{"status": "no_feasible_operating_point", "selected_cal_far": None}]}
        )
        self.assertEqual(answer["status"], "no_feasible_operating_point")
        self.assertEqual(answer["operating_selection_failures"], 1)
        self.assertFalse(answer["operating_selection_complete"])


if __name__ == "__main__":
    unittest.main()
