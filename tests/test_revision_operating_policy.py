"""Check legacy fixed defaults, bounded cache reuse and candidate adoption rules."""

from unittest import TestCase

from src.revision_next import (
    CONDITIONS,
    ORIGINAL_V11_EVALUATION_SOURCES,
    _checked_replay_configuration,
    _normalize,
    _parser,
    _reference_results,
    _tables,
)


class OperatingPolicyTests(TestCase):
    def test_scale_keeps_the_original_fixed_cal(self):
        parser = _parser()
        cli = parser.parse_args(["--operating-guard", "--task", "scale", "--smoke"])
        _normalize(parser, cli)
        self.assertEqual(cli.operating_policy, "fixed")
        self.assertIn("binary_operating_guard", _tables(cli))
        self.assertNotIn("operating_selection", _tables(cli))

    def test_replay_uses_original_binding_and_rejects_training_code_change(self):
        original = {
            "version": 4,
            "settings": {"head_objective": "multiclass"},
            "code_sha256": {**ORIGINAL_V11_EVALUATION_SOURCES, "revision_shared_head.py": "trained"},
        }
        candidate = {
            "version": 4,
            "settings": {"head_objective": "multiclass", "operating_policy": "cal-select"},
            "code_sha256": {
                "revision_next.py": "new",
                "revision_external.py": "new",
                "revision_operating_selection.py": "new",
                "revision_shared_head.py": "trained",
            },
        }
        self.assertEqual(_checked_replay_configuration(original, candidate)["code"], original["code_sha256"])
        candidate["code_sha256"]["revision_shared_head.py"] = "different"
        with self.assertRaises(ValueError):
            _checked_replay_configuration(original, candidate)

    def test_adoption_requires_complete_nonnegative_paired_means(self):
        key = ("CIC-UNSW-NB15", "train_absence", 0.3, 3, "Exploits", "reserved")
        identity = dict(zip(CONDITIONS, key, strict=True))
        rows = [
            {**identity, "seed": seed, "metric": metric, "delta": delta}
            for seed in (42, 43)
            for metric, delta in (("f1", 0.0), ("macro_f1", -0.01))
        ]
        result = _reference_results(rows, [key], [42, 43])[key]
        self.assertTrue(result["reference_f1_nonregression_met"])
        self.assertFalse(result["subtype_nonregression_met"])
        self.assertFalse(_reference_results(rows[:-1], [key], [42, 43])[key]["subtype_nonregression_met"])
