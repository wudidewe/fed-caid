"""Numerical and information-boundary tests for the conditional candidate."""

import unittest
from argparse import Namespace
from unittest.mock import patch

import numpy as np

from src import revision_conditional as candidate
from src.revision_adaptation import AdapterClient
from src.secure_aggregation import _clip_quantize
from src.utils import split_calibration


def arguments(**overrides):
    values = dict(
        rounds=2,
        seed=42,
        adapter_epochs=3,
        adapter_lr=0.15,
        adapter_prox=0.2,
        prior_strength=0.5,
        cal_fit_fraction=0.5,
        target_far=0.1,
        threshold_floor=0.5,
        clip=0.2,
        secure_scale=10**9,
        secure_timeout=30.0,
        conditional_rank=2,
        conditional_fit_limit=7,
    )
    return Namespace(**{**values, **overrides})


def clients():
    result = []
    for index in range(3):
        rng = np.random.default_rng(400 + index)
        probability = rng.dirichlet([2, 2, 2], 40)
        labels = (np.arange(40) + index) % 3
        evaluation = rng.dirichlet([1, 2, 3], 11)
        result.append(AdapterClient(probability, labels, evaluation, np.zeros(3)))
    return result


class PlainCollector:
    """Plain oracle using exactly the same clip, rounding and equal mean."""

    def __init__(self, clients, dimension, *, clip, scale, timeout):
        self.dimension, self.clip, self.scale = dimension, clip, scale

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def start_round(self, *_):
        self.updates = []
        return None

    def submit(self, client, update, *, contract):
        self.updates.append(_clip_quantize(update, self.dimension, self.clip, self.scale))

    def aggregate(self):
        return np.asarray(self.updates).mean(axis=0) / self.scale, {}


class ConditionalModelTests(unittest.TestCase):
    def test_identity_bounded_features_and_public_map_determinism(self):
        one = candidate.ConditionalScoreAdapter(5, 3)
        two = candidate.ConditionalScoreAdapter(5, 3)
        probabilities = np.array([[0, 1, 0, 0, 0], [0.1, 0.2, 0.3, 0.15, 0.25]])
        transformed = one.transform(probabilities)
        np.testing.assert_allclose(transformed, probabilities, atol=5e-12)
        np.testing.assert_array_equal(one.feature_basis, two.feature_basis)
        np.testing.assert_allclose(one.basis.T @ one.basis, np.eye(4), atol=1e-15)
        _, features = one.features(probabilities)
        self.assertTrue(np.all(np.linalg.norm(features[:, 1:], axis=1) <= 1 + 1e-15))
        # Public class positions are not secretly sorted or fitted per client.
        self.assertFalse(np.array_equal(one.features(probabilities[:, ::-1])[1], features))
        self.assertEqual(one.dimension, 16)
        self.assertEqual(one.transform(np.empty((0, 5))).shape, (0, 5))

    def test_rank_zero_is_a_matched_intercept_only_model(self):
        model = candidate.ConditionalScoreAdapter(4, 0)
        self.assertEqual(model.dimension, 3)
        model.set_vector(np.array([0.15, -0.09, 0.22]))
        probabilities = np.array([[0.5, 0.25, 0.15, 0.10], [0.1, 0.2, 0.3, 0.4]])
        result = model.transform(probabilities)
        change = np.log(result[:, 1] / result[:, 0]) - np.log(probabilities[:, 1] / probabilities[:, 0])
        np.testing.assert_allclose(change, [change[0], change[0]], atol=1e-14)
        self.assertEqual(model.spec()["feature_map"], "intercept only")

    def test_low_rank_map_includes_late_public_class_positions(self):
        model = candidate.ConditionalScoreAdapter(15, 4)
        # Every contrast of two public class positions changes the fixed map;
        # a prefix-only Helmert map would fail for classes beyond position four.
        for first in range(15):
            for second in range(first + 1, 15):
                self.assertGreater(
                    np.linalg.norm(model.feature_basis[first] - model.feature_basis[second]), 1e-6
                )
        np.testing.assert_allclose(model.feature_basis.T @ model.feature_basis, np.eye(4), atol=1e-14)
        self.assertEqual(
            model.spec()["feature_basis_sha256"],
            candidate.ConditionalScoreAdapter(15, 4).spec()["feature_basis_sha256"],
        )

    def test_residual_is_input_conditioned_and_stable_at_saturation(self):
        model = candidate.ConditionalScoreAdapter(3, 2)
        vector = np.zeros(model.dimension).reshape(2, 3)
        vector[0, 1] = 0.4
        model.set_vector(vector.reshape(-1))
        p = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.2, 0.3, 0.5]])
        result = model.transform(p)
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_allclose(result.sum(axis=1), 1)
        residual = np.log(result) - np.log(np.clip(p, 1e-12, 1))
        # The ratio between classes changes differently for different inputs.
        self.assertNotAlmostEqual(residual[0, 0] - residual[0, 1], residual[1, 0] - residual[1, 1])
        projected = model.project(np.full(model.dimension, 1e300))
        self.assertTrue(np.isfinite(projected).all())
        self.assertAlmostEqual(np.linalg.norm(projected), model.radius)

    def test_one_step_gradient_matches_numerical_loss_derivative(self):
        model = candidate.ConditionalScoreAdapter(3, 2)
        p = np.array([[0.7, 0.2, 0.1], [0.1, 0.4, 0.5], [0.4, 0.4, 0.2]])
        y = np.array([1, 2, 0])
        data = candidate._fit_data(model, p, y)
        anchor = np.linspace(-0.1, 0.1, model.dimension)
        ridge = 0.03
        delta, details = candidate.fit_conditional(
            anchor, data, model, epochs=1, lr=0.01, prox=0.2, ridge=ridge
        )

        def objective(vector):
            model.set_vector(vector)
            return candidate._log_loss(model.transform(p), y) + ridge * np.dot(vector, vector) / 2

        derivative = []
        for position in range(model.dimension):
            step = np.zeros(model.dimension)
            step[position] = 1e-6
            derivative.append((objective(anchor + step) - objective(anchor - step)) / 2e-6)
        np.testing.assert_allclose(delta / -0.01, derivative, atol=2e-9)
        self.assertEqual(details["gradient_steps"], 1)

    def test_rejects_invalid_settings_and_probabilities(self):
        for rank in (-1, True):
            with self.assertRaises(ValueError):
                candidate.ConditionalScoreAdapter(3, rank)
        model = candidate.ConditionalScoreAdapter(3)
        for p in ([[1, 1, 1]], [[np.nan, 0, 1]], [[-1, 1, 1]], [[0.5, 0.5]]):
            with self.assertRaises(ValueError):
                model.transform(p)
        with self.assertRaises(ValueError):
            candidate.run_conditional_ablations(clients(), arguments(conditional_fit_limit=-1))


class MatchedConditionalTests(unittest.TestCase):
    def run_plain(self, inputs, args):
        with patch.object(candidate, "SecureAggregationSession", PlainCollector):
            return candidate.run_conditional_ablations(inputs, args)

    def test_matched_fit_budget_classblind_subset_and_payload(self):
        args = arguments()
        inputs = clients()
        with patch.object(candidate, "fit_conditional", wraps=candidate.fit_conditional) as fitter:
            predictions, scores, diagnostics = self.run_plain(inputs, args)
        self.assertEqual(fitter.call_count, 2 * len(inputs) * (args.rounds + 1))
        self.assertTrue(all(len(call.args[1].labels) == 7 for call in fitter.call_args_list))
        self.assertEqual(set(predictions), set(candidate.MODES))
        self.assertEqual(set(scores), set(candidate.MODES))
        for index in range(3):
            chosen = [
                call.args[1]
                for call in fitter.call_args_list
                if call.args[1] is fitter.call_args_list[index * args.rounds].args[1]
            ]
            self.assertEqual(len(chosen), 2 * (args.rounds + 1))
        specification = diagnostics["model_spec"]
        self.assertEqual(specification["parameter_count"], 6)
        self.assertEqual(specification["cal_fit_limit"], 7)
        for mode in diagnostics["communication"]:
            self.assertEqual(mode["parameter_count"], 6)
            self.assertEqual(
                mode["round_payload_bytes"],
                528 if mode["mode"] in ("conditional_full", "conditional_global") else 0,
            )
        self.assertTrue(all(row["cal_holdout_rows"] == 20 for row in diagnostics["calibration"]))

    def test_single_method_profiling_does_not_fit_the_other_branch(self):
        args, inputs = arguments(), clients()
        for mode in ("conditional_local", "conditional_full"):
            with (
                patch.object(candidate, "SecureAggregationSession", PlainCollector),
                patch.object(candidate, "fit_conditional", wraps=candidate.fit_conditional) as fitter,
            ):
                predictions, _, _ = candidate.run_conditional_ablations(inputs, args, modes=(mode,))
            self.assertEqual(set(predictions), {mode})
            self.assertEqual(fitter.call_count, len(inputs) * (args.rounds + 1))

    def test_holdout_labels_cannot_change_fitted_scores(self):
        args = arguments()
        original = clients()
        changed = []
        for index, client in enumerate(original):
            _, holdout = split_calibration(
                len(client.cal_labels), args.cal_fit_fraction, args.seed + 335 + index
            )
            labels = client.cal_labels.copy()
            labels[holdout] = (labels[holdout] + 1) % 3
            changed.append(
                AdapterClient(client.cal_probabilities, labels, client.eval_probabilities, client.log_prior)
            )
        _, first, diag_first = self.run_plain(original, args)
        _, second, diag_second = self.run_plain(changed, args)
        self.assertEqual(diag_first["fit_selection_private"], diag_second["fit_selection_private"])
        for mode in candidate.MODES:
            for a, b in zip(first[mode], second[mode], strict=True):
                np.testing.assert_array_equal(a, b)

    def test_evaluation_values_do_not_change_fits_or_development_results(self):
        original = clients()
        changed = [
            AdapterClient(
                c.cal_probabilities, c.cal_labels, np.tile([0.99, 0.005, 0.005], (9, 1)), c.log_prior
            )
            for c in original
        ]
        _, _, first = self.run_plain(original, arguments())
        _, _, second = self.run_plain(changed, arguments())
        self.assertEqual(first["development_metrics"], second["development_metrics"])
        self.assertEqual(first["fit_selection_private"], second["fit_selection_private"])
        with self.assertRaises(ValueError):
            candidate.run_conditional_ablations(original[:2], arguments())

    def test_real_two_collector_result_matches_plain_quantized_mean(self):
        args = arguments(rounds=1, adapter_epochs=2)
        inputs = clients()
        plain_pred, plain_score, _ = self.run_plain(inputs, args)
        real_pred, real_score, diagnostics = candidate.run_conditional_ablations(inputs, args)
        for mode in candidate.MODES:
            for expected, actual in zip(plain_pred[mode], real_pred[mode], strict=True):
                np.testing.assert_array_equal(actual, expected)
            for expected, actual in zip(plain_score[mode], real_score[mode], strict=True):
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)
        self.assertTrue(diagnostics["rounds"][0]["all_clients_accepted"])
        self.assertGreater(diagnostics["rounds"][0]["serialized_application_bytes"], 0)


class CoverageAndMaskTests(unittest.TestCase):
    def test_coverage_sampler_reserves_every_present_class(self):
        labels = np.array([0] * 100 + [1] * 3 + [2] * 1, dtype=np.int64)
        chosen = candidate.coverage_sample(labels, 24, 3, 4, np.random.default_rng(7))
        self.assertEqual(len(chosen), 24)
        self.assertEqual(len(np.unique(chosen)), 24)
        counts = np.bincount(labels[chosen], minlength=3)
        self.assertTrue((counts > 0).all())
        self.assertGreaterEqual(counts[1], 3)
        self.assertEqual(counts[2], 1)

    def test_class_weights_are_uniform_with_zero_beta_and_mask_absent(self):
        labels = np.array([0, 0, 0, 1, 1, 2], dtype=np.int64)
        weights = candidate.class_weights_for(labels, 3, 0.0)
        np.testing.assert_array_equal(weights, [1.0, 1.0, 1.0])
        masked = candidate.class_mask_for(np.array([0, 0, 1]), 3)
        np.testing.assert_array_equal(masked, [1.0, 0.0])

    def test_absent_class_coordinates_are_not_updated(self):
        model = candidate.ConditionalScoreAdapter(3, 2)
        p = np.array([[0.7, 0.2, 0.1], [0.1, 0.4, 0.5], [0.4, 0.4, 0.2]])
        y = np.array([0, 1, 0])
        data = candidate._fit_data(model, p, y)
        unmasked, _ = candidate.fit_conditional(
            np.zeros(model.dimension), data, model, epochs=2, lr=0.1, prox=0.0, ridge=0.0
        )
        masked, details = candidate.fit_conditional(
            np.zeros(model.dimension),
            data,
            model,
            epochs=2,
            lr=0.1,
            prox=0.0,
            ridge=0.0,
            class_mask=np.array([1.0, 0.0]),
        )
        missing_vs_benign = model.basis[2] - model.basis[0]
        np.testing.assert_allclose(missing_vs_benign @ masked.reshape(2, 3), 0.0, atol=1e-12)
        anchor = model.project(np.arange(1, 7, dtype=float))
        constrained, _ = candidate.fit_conditional(
            anchor,
            data,
            model,
            epochs=3,
            lr=5.0,
            prox=0.3,
            ridge=0.2,
            class_mask=np.array([1.0, 0.0]),
            optimizer="adam",
        )
        np.testing.assert_allclose(missing_vs_benign @ constrained.reshape(2, 3), 0.0, atol=1e-12)
        self.assertLessEqual(np.linalg.norm(anchor + constrained), model.radius + 1e-12)
        self.assertEqual(details["masked_class_count"], 1)
        self.assertFalse(np.allclose(unmasked, masked))

    def test_uniform_weights_match_unweighted_fit(self):
        model = candidate.ConditionalScoreAdapter(3, 2)
        p = np.array([[0.7, 0.2, 0.1], [0.1, 0.4, 0.5], [0.4, 0.4, 0.2]])
        y = np.array([0, 1, 2])
        data = candidate._fit_data(model, p, y)
        first, _ = candidate.fit_conditional(
            np.zeros(model.dimension), data, model, epochs=2, lr=0.1, prox=0.2, ridge=0.01
        )
        second, _ = candidate.fit_conditional(
            np.zeros(model.dimension),
            data,
            model,
            epochs=2,
            lr=0.1,
            prox=0.2,
            ridge=0.01,
            class_weights=np.ones(3),
        )
        np.testing.assert_allclose(first, second)

    def test_argmax_modes_are_matched_and_present(self):
        args = arguments()
        args.conditional_argmax_modes = True
        with patch.object(candidate, "SecureAggregationSession", PlainCollector):
            predictions, scores, diagnostics = candidate.run_conditional_ablations(clients(), args)
        for mode in candidate.MODES:
            self.assertIn(f"{mode}_argmax", predictions)
            self.assertIn(f"{mode}_argmax", scores)
            for values in predictions[f"{mode}_argmax"]:
                self.assertTrue(((values >= 0) & (values < 3)).all())
        self.assertIn("threshold_min_benign_rows", diagnostics["model_spec"])


if __name__ == "__main__":
    unittest.main()
