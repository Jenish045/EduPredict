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


if __name__ == "__main__":
    unittest.main()
