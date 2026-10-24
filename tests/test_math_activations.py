"""
Unit tests for activation functions and numerical stability.
"""

import unittest
import math
from edupredict.math.activations import (
    clip,
    sigmoid,
    sigmoid_vector,
    clip_probability,
    safe_log,
    softmax,
    pure_exp,
    pure_log,
)


class TestActivations(unittest.TestCase):

    def test_sigmoid_values(self):
        self.assertAlmostEqual(sigmoid(0.0), 0.5)
        # Should be strictly bounded in (0, 1) even for extreme values without overflow
        self.assertAlmostEqual(sigmoid(100.0), 1.0)
        self.assertAlmostEqual(sigmoid(-100.0), 0.0)

    def test_sigmoid_vector(self):
        res = sigmoid_vector([0.0, 2.0, -2.0])
        self.assertEqual(len(res), 3)
        self.assertAlmostEqual(res[0], 0.5)

    def test_clip_probability(self):
        eps = 1e-15
        self.assertEqual(clip_probability(-1.0, eps), eps)
        self.assertEqual(clip_probability(2.0, eps), 1.0 - eps)
        self.assertEqual(clip_probability(0.6, eps), 0.6)

    def test_safe_log(self):
        # Should not throw domain error on 0.0
        self.assertTrue(math.isfinite(safe_log(0.0)))
        self.assertAlmostEqual(safe_log(math.e), 1.0)

    def test_softmax(self):
        logits = [2.0, 1.0, 0.1]
        probs = softmax(logits)
        self.assertEqual(len(probs), 3)
        self.assertAlmostEqual(sum(probs), 1.0)
        self.assertTrue(probs[0] > probs[1] > probs[2])

        # Large numbers (numerical stability)
        large_logits = [1000.0, 1001.0, 1002.0]
        large_probs = softmax(large_logits)
        self.assertAlmostEqual(sum(large_probs), 1.0)

    def test_pure_exp_accuracy_and_boundaries(self):
        """Verify pure_exp produces sub-1e-10 relative error and handles extremes."""
        self.assertEqual(pure_exp(0.0), 1.0)
        self.assertEqual(pure_exp(-800.0), 0.0)
        self.assertEqual(pure_exp(800.0), float("inf"))

        test_points = [-650.0, -100.0, -10.0, -1.0, -0.5, 0.0, 0.5, 1.0, 5.0, 20.0, 100.0, 650.0]
        for x in test_points:
            expected = math.exp(x)
            actual = pure_exp(x)
            rel_err = abs(actual - expected) / expected
            self.assertLess(rel_err, 1e-10, f"pure_exp({x}) rel error {rel_err} exceeds 1e-10")

    def test_pure_log_accuracy_and_boundaries(self):
        """Verify pure_log produces sub-1e-13 relative error and handles extremes."""
        self.assertEqual(pure_log(1.0), 0.0)
        with self.assertRaises(ValueError):
            pure_log(0.0)
        with self.assertRaises(ValueError):
            pure_log(-1.0)

        test_points = [1e-100, 1e-10, 1e-3, 0.5, 1.0, 2.0, math.e, 10.0, 1000.0, 1e50, 1e150]
        for x in test_points:
            expected = math.log(x)
            actual = pure_log(x)
            rel_err = abs(actual - expected) / max(1.0, abs(expected))
            self.assertLess(rel_err, 1e-13, f"pure_log({x}) rel error {rel_err} exceeds 1e-13")


if __name__ == "__main__":
    unittest.main()
