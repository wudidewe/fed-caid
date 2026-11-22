from unittest import TestCase

from src.revision_binary_diagnostics import summarize_fnr_guard


class OperatingFNRGuardTests(TestCase):
    def test_complete_constant_effect_at_limit_passes(self):
        seeds = (43, 44, 45, 46, 47)
        rows = [{"seed": seed, "fnr_delta": 0.02} for seed in reversed(seeds)]
        result = summarize_fnr_guard(rows, seeds, 0.02)
        self.assertEqual(result["operating_fnr_planned_seeds"], 5)
        self.assertEqual(result["operating_fnr_observed_seeds"], 5)
        self.assertEqual(result["operating_fnr_delta_mean"], 0.02)
        self.assertEqual(result["operating_fnr_delta_ci95_low"], 0.02)
        self.assertEqual(result["operating_fnr_delta_ci95_high"], 0.02)
        self.assertTrue(result["operating_fnr_guard_met"])

    def test_upper_bound_blocks_small_mean_with_high_variability(self):
        rows = [{"seed": 42, "fnr_delta": -0.01}, {"seed": 43, "fnr_delta": 0.02}]
        result = summarize_fnr_guard(rows, (42, 43), 0.02)
        self.assertLess(result["operating_fnr_delta_mean"], 0.02)
        self.assertGreater(result["operating_fnr_delta_ci95_high"], 0.02)
        self.assertFalse(result["operating_fnr_guard_met"])

    def test_missing_or_undefined_seed_and_unavailable_interval_fail(self):
        rows = [{"seed": 42, "fnr_delta": 0}, {"seed": 43, "fnr_delta": 0}]
        for records, planned, observed, measured in (
            (rows, (42, 43, 44), 2, 2),
            (rows, (42, 43), 2, 2),
            ([*rows, {"seed": 44, "fnr_delta": None}], (42, 43, 44), 3, 2),
            (rows[:1], (42,), 1, 1),
            ([], (), 0, 0),
        ):
            with self.subTest(planned=planned, observed=observed):
                result = summarize_fnr_guard(records, planned, 0.02)
                self.assertEqual(result["operating_fnr_observed_seeds"], observed)
                self.assertEqual(result["operating_fnr_seeds_with_metric"], measured)
                self.assertFalse(result["operating_fnr_guard_met"])

    def test_mismatched_or_duplicate_seed_and_invalid_rates_are_rejected(self):
        row = {"seed": 42, "fnr_delta": 0}
        with self.assertRaisesRegex(ValueError, "unique planned"):
            summarize_fnr_guard([row, row], (42, 43), 0.02)
        with self.assertRaisesRegex(ValueError, "unique planned"):
            summarize_fnr_guard([row], (43,), 0.02)
        with self.assertRaisesRegex(ValueError, "distinct planned"):
            summarize_fnr_guard([row], (42, 42), 0.02)
        with self.assertRaisesRegex(ValueError, "maximum FNR"):
            summarize_fnr_guard([row], (42,), float("nan"))
        with self.assertRaisesRegex(ValueError, "Invalid metric"):
            summarize_fnr_guard([{"seed": 42, "fnr_delta": -1.01}], (42,), 0.02)
