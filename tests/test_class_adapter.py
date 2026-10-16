"""Class adapter mathematics using synthetic probability scores."""

import unittest

import numpy as np

import privacy_core as pc


class ClassAdapterTests(unittest.TestCase):
    def test_fixed_helmert_basis_is_orthonormal_and_zero_sum(self):
        for classes in (2, 3, 10):
            adapter = pc.ClassScoreAdapter(classes)
            basis = adapter.basis.numpy()
            self.assertEqual(basis.shape, (classes, classes - 1))
            self.assertEqual(adapter.get_vector().shape, (classes - 1,))
            np.testing.assert_allclose(basis.T @ basis, np.eye(classes - 1), atol=1e-14)
            np.testing.assert_allclose(basis.sum(0), 0.0, atol=1e-14)
            np.testing.assert_array_equal(basis, pc.ClassScoreAdapter(classes).basis.numpy())

    def test_identity_and_bias_can_correct_attack_subtype_ratio(self):
        p = np.array([[0.6, 0.3, 0.1], [0.1, 0.45, 0.45], [1.0, 0.0, 0.0]])
        adapter = pc.ClassScoreAdapter(3)
        np.testing.assert_allclose(adapter.transform(p), p, atol=3e-12)
        adapter.set_vector([0.0, 1.0])
        result = adapter.transform(p)
        np.testing.assert_allclose(result.sum(1), 1.0, atol=1e-14)
        self.assertFalse(np.allclose(result[:2, 1] / result[:2, 2], p[:2, 1] / p[:2, 2]))
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue((result >= 0.0).all())

    def test_projection_bounds_whole_vector_with_extreme_finite_input(self):
        adapter = pc.ClassScoreAdapter(4)
        for values in ([3.0, 4.0, 0.0], [1e308, -1e308, 1e308]):
            adapter.set_vector(values)
            self.assertLessEqual(np.linalg.norm(adapter.get_vector()), 2.0 + 1e-14)
        with self.assertRaises(ValueError):
            adapter.set_vector([0.0, 0.0])
        with self.assertRaises(ValueError):
            adapter.set_vector([0.0, np.nan, 0.0])

    def test_calibration_learns_attack_subtype_without_detector_parameters(self):
        p = np.tile([0.1, 0.8, 0.1], (40, 1))
        labels = np.full(40, 2, dtype=int)
        delta = pc.fit_class_adapter(
            np.zeros(2),
            p,
            labels,
            np.log([0.8, 0.15, 0.05]),
            epochs=30,
            lr=0.2,
            prior_strength=0.5,
            prox=0.1,
        )
        adapter = pc.ClassScoreAdapter(3)
        adapter.set_vector(delta)
        corrected = adapter.transform(p)
        self.assertGreater(corrected[:, 2].mean(), p[:, 2].mean())
        self.assertLess(corrected[:, 1].mean(), p[:, 1].mean())
        self.assertLessEqual(np.linalg.norm(adapter.get_vector()), 2.0 + 1e-14)
        self.assertEqual(sum(parameter.numel() for parameter in adapter.parameters()), 2)

    def test_prior_adjustment_and_proximal_penalty_affect_local_fit(self):
        p = np.tile([0.1, 0.8, 0.1], (20, 1))
        labels = np.full(20, 2, dtype=int)
        prior = np.log([0.8, 0.15, 0.05])
        kwargs = dict(epochs=10, lr=0.1)
        unadjusted = pc.fit_class_adapter(
            np.zeros(2), p, labels, prior, prior_strength=0.0, prox=0.0, **kwargs
        )
        adjusted = pc.fit_class_adapter(
            np.zeros(2), p, labels, prior, prior_strength=0.5, prox=0.0, **kwargs
        )
        proximal = pc.fit_class_adapter(
            np.zeros(2), p, labels, prior, prior_strength=0.5, prox=2.0, **kwargs
        )
        self.assertFalse(np.allclose(unadjusted, adjusted))
        self.assertLess(np.linalg.norm(proximal), np.linalg.norm(adjusted))

    def test_invalid_local_inputs_are_rejected(self):
        p = np.tile([0.2, 0.3, 0.5], (4, 1))
        for labels in ([0, 1, 2], [0.0, 1.0, 2.0, 0.0], [0, 1, 3, 0], [0, 1, -1, 0]):
            with self.assertRaises(ValueError):
                pc.fit_class_adapter(np.zeros(2), p, labels, np.log([0.2, 0.3, 0.5]))
        for prior in ([0.0, 0.0], [0.0, np.nan, 0.0]):
            with self.assertRaises(ValueError):
                pc.fit_class_adapter(np.zeros(2), p, [0, 1, 2, 0], prior)
        for kwargs in (
            dict(epochs=0),
            dict(epochs=1.5),
            dict(lr=0),
            dict(prior_strength=-1),
            dict(prox=np.inf),
        ):
            with self.assertRaises(ValueError):
                pc.fit_class_adapter(
                    np.zeros(2), p, [0, 1, 2, 0], np.log([0.2, 0.3, 0.5]), **kwargs
                )
        with self.assertRaises(ValueError):
            pc.ClassScoreAdapter(1)
        with self.assertRaises(ValueError):
            pc.ClassScoreAdapter(3).transform(p * 2)


if __name__ == "__main__":
    unittest.main()
