"""Core detector/data regressions; synthetic scores are not research results."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import data_io as main
import main as entry
import ids_core as core
import privacy_run


class Case(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="unified_test_", dir=main.ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def data(self):
        rng = np.random.default_rng(42)
        arrays = {"class_names": ["BENIGN", "A", "B"], "feature_names": ["x0", "x1", "x2", "x3"]}
        for role, n in (("train", 240), ("val", 60), ("cal", 60)):
            y = np.arange(n, dtype=np.int64) % 3
            arrays["X_" + role] = (rng.normal(size=(n, 4)) + 0.5 * y[:, None]).astype(np.float32)
            arrays["y_" + role] = y
        path = self.root / "data.npz"
        np.savez_compressed(path, **arrays)
        return path


class LossTests(Case):
    def test_weighted_member_ce_matches_formula_and_gradients(self):
        torch.manual_seed(42)
        weights = torch.tensor([0.5, 1.5, 2.0], dtype=torch.float64)
        labels = torch.arange(7) % 3
        for shape in ((7, 3), (7, 4, 3)):
            logits = torch.randn(shape, dtype=torch.float64, requires_grad=True)
            loss = core.classification_losses(logits, labels, weights)
            expanded = logits[:, None, :] if logits.ndim == 2 else logits
            logp = expanded.log_softmax(-1)
            expected = -logp[torch.arange(7), :, labels].mean(1) * weights[labels]
            torch.testing.assert_close(loss, expected, rtol=1e-12, atol=1e-12)
            loss.mean().backward()
            self.assertTrue(torch.isfinite(logits.grad).all())

    def test_weights_keep_original_power_and_normalization(self):
        y = np.r_[np.zeros(80, int), np.ones(15, int), np.full(5, 2)]
        expected = np.array([80, 15, 5], float) ** -0.25
        expected /= np.sum(np.array([0.8, 0.15, 0.05]) * expected)
        np.testing.assert_allclose(core.class_weights(y), expected, rtol=1e-14)
        for labels in (np.array([]), np.array([0, 2]), np.array([0.0, 1.0]), np.array([-1, 0])):
            with self.assertRaises(ValueError):
                core.class_weights(labels)

    def test_diagnostics_protect_complete_rare_classes_when_budget_allows(self):
        y = np.r_[np.zeros(20000, int), np.ones(145, int), np.full(212, 2), np.full(125, 3)]
        rows = core.diagnostic_rows(y, 1000)
        self.assertLessEqual(len(rows), 1000)
        np.testing.assert_array_equal(np.bincount(y[rows])[1:], [145, 212, 125])
        self.assertEqual(len(rows), len(set(rows)))
        np.testing.assert_array_equal(rows, core.diagnostic_rows(y, 1000))

    def test_diagnostic_fallback_covers_classes_with_small_cap(self):
        y = np.tile(np.arange(14), 300)
        rows = core.diagnostic_rows(y, 100)
        self.assertLessEqual(len(rows), 100)
        self.assertEqual(set(y[rows]), set(y))
        with self.assertRaises(ValueError):
            core.diagnostic_rows(y, 10)

    def test_weighted_and_macro_metrics_are_distinct(self):
        y = np.r_[np.zeros(90, int), np.ones(10, int)]
        p = np.tile([0.9, 0.1], (100, 1))
        result = core.prediction_metrics(y, p)
        self.assertAlmostEqual(result["weighted_f1"], 0.9 * 180 / 190)
        self.assertAlmostEqual(result["macro_f1"], 0.5 * 180 / 190)
        self.assertAlmostEqual(result["weighted_precision"], 0.81)
        self.assertAlmostEqual(result["accuracy"], 0.9)

    def test_selection_prioritizes_weighted_f1_then_macro_and_other_metrics(self):
        a = dict(weighted_f1=0.9, macro_f1=0.5, accuracy=0.91, weighted_precision=0.9, nll=0.2)
        b = dict(a, weighted_f1=0.89, macro_f1=0.99)
        self.assertGreater(core.selection_key(a), core.selection_key(b))
        for key in ("macro_f1", "accuracy", "weighted_precision"):
            with self.subTest(key=key):
                self.assertGreater(
                    core.selection_key(dict(a, **{key: a[key] + 0.01})), core.selection_key(a)
                )
        self.assertGreater(core.selection_key(dict(a, nll=0.19)), core.selection_key(a))
        self.assertEqual(
            core.selection_key(a), core.selection_key({"val_" + k: v for k, v in a.items()})
        )


class DataTests(Case):
    def test_transform_uses_fixed_training_statistics_and_clips_extremes(self):
        x = np.array([[0.0, 1.0], [2.0, np.nan], [4.0, 3.0]], np.float32)
        transform = core.Transform().fit(x, True)
        state = transform.state()
        extreme = np.array([[1e9, 2.0]], np.float32)
        self.assertEqual(transform(extreme)[0, 0], 12)
        self.assertEqual(transform.state(), state)
        self.assertTrue(np.isfinite(transform(x)).all())
        np.testing.assert_array_equal(transform(x), core.Transform.restore(state)(x))

    def test_development_loader_rejects_test_members(self):
        data = self.data()
        parts, names, features = main.load_data(data)
        self.assertEqual(set(parts), {"train", "val", "cal"})
        with np.load(data) as source:
            arrays = {k: source[k] for k in source.files}
        arrays["X_test"] = np.zeros((1, 4))
        np.savez_compressed(self.root / "bad.npz", **arrays)
        with self.assertRaises(ValueError):
            main.load_data(self.root / "bad.npz")

    def test_car_short_dlc_and_hex_values_retain_r_t_meaning(self):
        frame = pd.DataFrame(
            [["1", "021", "1", "21", "R"] + [""] * 7, ["2", "0A0", "2", "0F", "FF", "T"] + [""] * 6]
        )
        labels, x = main.car_chunk(frame, "gear_dataset.csv", True)
        np.testing.assert_array_equal(labels, ["BENIGN", "Gear spoofing"])
        np.testing.assert_array_equal(x[0], [33, 1, 33, 0, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(x[1, :4], [160, 2, 15, 255])
        self.assertEqual(x[1, 4:].sum(), 0)

    def test_car_invalid_observed_byte_dlc_and_shifted_flag_fail(self):
        rows = [
            ["1", "21", "9", "R"] + [""] * 8,
            ["1", "21", "1", "ZZ", "T"] + [""] * 7,
            ["1", "21", "1", "FF", "T", "extra"] + [""] * 6,
        ]
        for row in rows:
            with self.assertRaises(ValueError):
                main.car_chunk(pd.DataFrame([row]), "DoS_dataset.csv", True)

    def test_frozen_balance_formula_and_label_canonicalization(self):
        x = np.array([[2.0, 1.0], [np.nan, 1.0], [-1.0, 3.0]], np.float32)
        recipes = [{"operation": "balance", "columns": ["a", "b"]}]
        result = main.apply_recipes(x, ["a", "b"], recipes)
        self.assertEqual(result[0, 0], 0.25)
        self.assertTrue(np.isnan(result[1:]).all())
        self.assertEqual(main.label(" normal "), "BENIGN")
        self.assertEqual(main.label("Web Attack \ufffd XSS"), "Web Attack - XSS")

    def test_one_third_stream_retains_heldout_exclusions_and_exact_replay(self):
        names, features = ["BENIGN", "A"], ["f"]
        labels = np.tile(names, 30)
        raw = pd.DataFrame({"f": np.arange(60, dtype=float), "Label": labels})
        path = self.root / "source.csv"
        raw.to_csv(path, index=False)
        data = self.root / "old.npz"
        selected = pd.DataFrame(
            {
                "role": ["val", "val", "cal", "cal", "test", "test"],
                "class": ["BENIGN", "A"] * 3,
                "source_file_id": 0,
                "zero_based_data_row": np.arange(6),
                "row_in_role": [0, 1] * 3,
            }
        )
        identities = self.root / "selected_source_rows.csv"
        selected.to_csv(identities, index=False)
        exclusion = self.root / "excluded.csv"
        selected[["class", "source_file_id", "zero_based_data_row"]].to_csv(exclusion, index=False)
        np.savez_compressed(
            data,
            class_names=names,
            feature_names=features,
            X_val=np.array([[0.0], [1.0]], np.float32),
            y_val=np.array([0, 1]),
            X_cal=np.array([[2.0], [3.0]], np.float32),
            y_cal=np.array([0, 1]),
        )
        st = path.stat()
        spec = {
            "name": "synthetic",
            "source_files": [
                {
                    "path": str(path),
                    "sha256": core.sha(path),
                    "bytes": st.st_size,
                    "mtime_ns": st.st_mtime_ns,
                    "csv_data_rows": 60,
                }
            ],
            "class_names": names,
            "feature_names": features,
            "pipeline": None,
            "encoding": "utf-8",
            "label_column": "Label",
            "source_class_counts": [30, 30],
            "available_train_counts": [27, 27],
            "third_train_counts": [9, 9],
            "train_rows": 18,
            "data": str(data),
            "data_sha256": core.sha(data),
            "excluded_identity_file": str(exclusion),
            "excluded_identity_sha256": core.sha(exclusion),
            "metadata_pins": [{"path": str(identities), "sha256": core.sha(identities)}],
        }
        third_spec = main.training_spec(spec, "third")
        with contextlib.redirect_stdout(io.StringIO()):
            prepared = main.prepare_data(third_spec, self.root / "third", 7)
        parts, _, _ = main.load_data(prepared)
        self.assertEqual(len(parts["train"][1]), 18)
        self.assertFalse(np.isin(parts["train"][0][:, 0], np.arange(6)).any())
        np.testing.assert_array_equal(parts["val"][0], [[0.0], [1.0]])
        with np.load(prepared.parent / "selected_train_identities.npz") as ids:
            self.assertEqual(len(set(ids["source_row"])), 18)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main.prepare_data(third_spec, self.root / "third", 7), prepared)
        full_spec = main.training_spec(spec, "full")
        with contextlib.redirect_stdout(io.StringIO()):
            full = main.prepare_data(full_spec, self.root / "full", 7)
        full_parts, _, _ = main.load_data(full)
        self.assertEqual(len(full_parts["train"][1]), 54)
        np.testing.assert_array_equal(np.sort(full_parts["train"][0][:, 0]), np.arange(6, 60))
        for role in ("val", "cal"):
            for actual, expected in zip(full_parts[role], parts[role]):
                np.testing.assert_array_equal(actual, expected)
        self.assertTrue(
            main.read_json(full.parent / "preparation.json")["all_eligible_train_included"]
        )
        with self.assertRaisesRegex(ValueError, "Prepared TRAIN cache differs"):
            main.prepare_data(third_spec, full.parent, 7)

    def test_full_fraction_preserves_all_available_class_counts(self):
        spec = {
            "class_names": ["BENIGN", "A", "B"],
            "available_train_counts": [101, 5, 1],
            "third_train_counts": [33, 1, 1],
            "train_rows": 35,
        }
        full = main.training_spec(spec, "full")
        self.assertEqual(full["selected_train_counts"], [101, 5, 1])
        self.assertEqual(full["train_rows"], 107)
        self.assertEqual(spec["train_rows"], 35)
        self.assertEqual(main.training_spec(spec, "third")["train_rows"], 35)
        with self.assertRaises(ValueError):
            main.training_spec({**spec, "available_train_counts": [100, 0, 1]}, "full")


class TrainingTests(Case):
    def test_cross_initialization_and_bins_are_repeatable(self):
        x = np.random.default_rng(1).normal(size=(80, 4)).astype(np.float32)
        core.seed()
        one, bins = core.make_model(4, 3, "ple_cross", x)
        core.seed()
        two, other = core.make_model(4, 3, "ple_cross", x)
        for k, v in one.state_dict().items():
            self.assertTrue(torch.equal(v, two.state_dict()[k]), k)
        for a, b in zip(bins, other):
            self.assertTrue(torch.equal(a, b))

    def test_simple_temperature_has_no_route_or_test_input(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        p = np.full((6, 3), 0.1)
        p[np.arange(6), y] = 0.8
        cal = core.calibrate(np.log(p), y)
        self.assertEqual(cal["fit_role"], "CAL")
        self.assertGreater(cal["temperature"], 0)

    def test_only_active_neural_architecture_is_supported(self):
        self.assertEqual(core.NEURAL_KINDS, ("ple_cross",))
        for kind in ("mlp", "tabm", "ple_balanced", "extra_trees", "xgboost"):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                core.make_model(4, 3, kind)


class UnifiedDecisionTests(Case):
    def test_local_train_prior_is_cached_finite_and_neutral_for_absent_classes(self):
        y = np.r_[np.zeros(80, dtype=int), np.ones(15, dtype=int), np.full(5, 2, dtype=int)]
        prior = np.asarray(privacy_run.cached_train_log_prior(y, 4))
        self.assertEqual(prior.shape, (4,))
        self.assertTrue(np.isfinite(prior).all())
        self.assertGreater(prior[0], prior[1])
        self.assertGreater(prior[1], prior[2])
        self.assertEqual(prior[3], 0.0)
        labels = torch.tensor([2])
        for members in (1, 4):
            logits = torch.zeros((1, members, 4), dtype=torch.float64, requires_grad=True)
            adjustment = 0.5 * torch.as_tensor(prior)
            loss = core.classification_losses(logits + adjustment, labels).mean()
            loss.backward()
            self.assertTrue(torch.isfinite(logits.grad).all())
            self.assertLess(float(logits.grad[0, :, 2].sum()), -0.75)
        for bad in (
            np.array([], dtype=int),
            np.array([-1, 0]),
            np.array([0, 4]),
            np.array([0.0, 1.0]),
        ):
            with self.assertRaises(ValueError):
                privacy_run.cached_train_log_prior(bad, 4)

    def test_calibration_split_is_disjoint_deterministic_and_label_independent(self):
        fit, threshold = privacy_run.split_calibration(51, 0.5, 42)
        replay = privacy_run.split_calibration(51, 0.5, 42)
        np.testing.assert_array_equal(fit, replay[0])
        np.testing.assert_array_equal(threshold, replay[1])
        self.assertFalse(np.intersect1d(fit, threshold).size)
        np.testing.assert_array_equal(np.sort(np.r_[fit, threshold]), np.arange(51))
        self.assertGreater(len(fit), 0)
        self.assertGreater(len(threshold), 0)
        one_fit, one_threshold = privacy_run.split_calibration(1, 0.5, 42)
        np.testing.assert_array_equal(one_fit, [0])
        self.assertEqual(len(one_threshold), 0)

    def test_alarm_threshold_uses_only_benign_holdout_and_suppresses_empty_benign(self):
        p = np.array([[0.99, 0.005, 0.005], [0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.3, 0.2, 0.5]])
        labels = np.array([0, 0, 1, 2])
        threshold, _ = privacy_run.empirical_alarm_threshold(p, labels, 0.01, 0.5)
        self.assertEqual(threshold, 0.5)
        changed_attack_scores = p.copy()
        changed_attack_scores[2:] = [0.99, 0.005, 0.005]
        unchanged, _ = privacy_run.empirical_alarm_threshold(
            changed_attack_scores, labels, 0.01, 0.5
        )
        self.assertEqual(threshold, unchanged)
        no_benign, _ = privacy_run.empirical_alarm_threshold(p[2:], labels[2:], 0.01, 0.5)
        self.assertGreaterEqual(no_benign, 1.0)
        self.assertFalse(np.any(p[2:, 1:].sum(1) > no_benign))

    def test_primary_decision_uses_attack_sum_then_attack_subtype(self):
        # Total attack probability exceeds .5 although BENIGN is the largest
        # single class; subtype prediction therefore must select ATTACK_A.
        p = np.array([[0.4, 0.35, 0.25], [0.8, 0.15, 0.05]])
        labels = np.array([1, 0])
        result = privacy_run.risk_metrics(labels, p, ["BENIGN", "A", "B"], threshold=0.5)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["argmax_accuracy"], 0.5)
        self.assertEqual(result["fnr_argmax"], 1.0)
        self.assertEqual(result["confusion_matrix"], [[1, 0, 0], [0, 1, 0], [0, 0, 0]])
        boundary = privacy_run.risk_metrics(
            np.array([0]), np.array([[0.5, 0.3, 0.2]]), threshold=0.5
        )
        self.assertEqual(boundary["accuracy"], 1.0)
        self.assertEqual(boundary["far_at_half"], 0.0)
        attack_boundary = privacy_run.risk_metrics(
            np.array([1]), np.array([[0.5, 0.3, 0.2]]), threshold=0.5
        )
        self.assertEqual(attack_boundary["fnr_at_half"], 1.0)
        self.assertEqual(attack_boundary["fnr"], 1.0)


class CrossTests(Case):
    def test_low_rank_cross_formula_and_autograd(self):
        core.seed()
        layer = core.LowRankCross(4, 2).double()
        with torch.no_grad():
            layer.project_out.weight.normal_(0, 0.2)
            layer.bias.normal_(0, 0.1)
        original = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
        current = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
        expected = current + original * (
            current @ layer.project_in.weight.T @ layer.project_out.weight.T + layer.bias
        )
        torch.testing.assert_close(layer(original, current), expected, atol=1e-12, rtol=1e-12)
        self.assertTrue(torch.autograd.gradcheck(layer, (original, current)))
        for width, rank in ((0, 1), (4, 0), (4, 5)):
            with self.assertRaises(ValueError):
                core.LowRankCross(width, rank)

    def test_cross_identity_start_preserves_base_weights_rng_and_dropout_logits(self):
        x = np.random.default_rng(1).normal(size=(40, 4)).astype(np.float32)
        core.seed()
        reference, bins = core.make_model(4, 3, "ple_cross", x)
        plain_logits = reference.base(torch.as_tensor(x))
        next_random = torch.rand(10)
        core.seed()
        crossed, other = core.make_model(4, 3, "ple_cross", x)
        crossed_logits = crossed(torch.as_tensor(x))
        torch.testing.assert_close(torch.rand(10), next_random, rtol=0, atol=0)
        torch.testing.assert_close(crossed_logits, plain_logits, rtol=0, atol=0)
        self.assertEqual(crossed_logits.shape, (40, 4, 3))
        for key, value in reference.base.state_dict().items():
            torch.testing.assert_close(crossed.base.state_dict()[key], value, rtol=0, atol=0)
        for a, b in zip(bins, other):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        expected_extra = 2 * (2 * 32 * 16 + 32)
        self.assertEqual(
            sum(p.numel() for p in crossed.parameters())
            - sum(p.numel() for p in reference.base.parameters()),
            expected_extra,
        )

    def test_cross_layers_receive_gradients_and_affect_predictions(self):
        x = np.random.default_rng(2).normal(size=(40, 4)).astype(np.float32)
        core.seed()
        model, _ = core.make_model(4, 3, "ple_cross", x)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
        labels = torch.arange(len(x)) % 3
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss = core.classification_losses(model(torch.as_tensor(x)), labels).mean()
            loss.backward()
            for layer in model.cross_layers:
                self.assertTrue(torch.isfinite(layer.project_out.weight.grad).all())
                self.assertGreater(float(layer.project_out.weight.grad.norm()), 0)
            optimizer.step()
        model.eval()
        active = model(torch.as_tensor(x))
        info = model.diagnostics(torch.as_tensor(x))
        self.assertGreater(info["mean_relative_residual_norm"], 0)
        bypass = model.base(torch.as_tensor(x))
        self.assertGreater(float((active - bypass).detach().abs().max()), 0)
        self.assertFalse(hasattr(model, "cross_enabled"))
        with self.assertRaises(ValueError):
            model(torch.zeros(2, 3))
        with self.assertRaises(ValueError):
            model(torch.zeros(2, 4, 4))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cross_cuda_forward_backward(self):
        x = np.random.default_rng(3).normal(size=(40, 4)).astype(np.float32)
        core.seed()
        model, _ = core.make_model(4, 3, "ple_cross", x)
        model = model.cuda()
        logits = model(torch.as_tensor(x, device="cuda"))
        loss = core.classification_losses(logits, torch.arange(40, device="cuda") % 3).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        )


class EntryPointTests(unittest.TestCase):
    def test_default_dispatches_to_private_federation(self):
        arguments = ["--scope", "all", "--device", "cpu"]
        with (
            patch.object(entry, "prepare_selected_datasets") as prepare,
            patch.object(privacy_run, "execute", return_value=7) as private,
        ):
            self.assertEqual(entry.main(arguments), 7)
        prepare.assert_called_once()
        private.assert_called_once()
        self.assertEqual(private.call_args.args[0].scope, "all")
        self.assertEqual(private.call_args.args[0].device, "cpu")

    def test_privacy_alias_remains_compatible(self):
        arguments = ["--smoke", "--device", "cpu"]
        with patch.object(privacy_run, "execute", return_value=0) as private:
            self.assertEqual(entry.main(["--privacy", *arguments]), 0)
        private.assert_called_once()
        self.assertTrue(private.call_args.args[0].smoke)

    def test_check_runs_regression_suite_and_propagates_status(self):
        with (
            patch.object(entry.subprocess, "run") as checks,
            patch.object(privacy_run, "main") as private,
        ):
            checks.return_value.returncode = 3
            self.assertEqual(entry.main(["--check"]), 3)
        checks.assert_called_once()
        self.assertEqual(
            checks.call_args.args[0],
            [entry.sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-v"],
        )
        self.assertEqual(checks.call_args.kwargs["cwd"], entry.ROOT)
        self.assertTrue(
            checks.call_args.kwargs["env"]["PYTHONPATH"].startswith(str(entry.ROOT / "src"))
        )
        private.assert_not_called()

    def test_removed_experiment_controls_are_rejected(self):
        self.assertEqual(privacy_run.parser().parse_args([]).scope, "all")
        for arguments in (
            ["--xgb-rounds", "1"],
            ["--group", "next"],
            ["--fusion", "mean"],
            ["--ablation"],
            ["--train-fraction", "third"],
            ["--dimension", "9"],
        ):
            with (
                self.subTest(arguments=arguments),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                privacy_run.parser().parse_args(arguments)
        for field in ("ROSTER", "experiment_roster", "fit_one", "run", "predict_saved"):
            self.assertFalse(hasattr(entry, field), field)

    def test_unified_defaults_keep_fixed_training_budget_and_automatic_coordinates(self):
        args = privacy_run.parser().parse_args([])
        self.assertEqual(args.base_epochs, 5)
        self.assertEqual(args.batch_size, 1024)
        self.assertEqual(args.rows_per_client, 0)
        self.assertEqual(args.prior_strength, 0.5)
        self.assertEqual(args.adapter_prox, 1.0)
        self.assertEqual(args.target_far, 0.01)
        self.assertEqual(args.threshold_floor, 0.5)
        self.assertEqual(args.cal_fit_fraction, 0.5)
        self.assertFalse(hasattr(args, "dimension"))


if __name__ == "__main__":
    unittest.main()
