"""
Unit tests for gradient descent optimizer.
"""

import unittest
import random
from edupredict.optimization.gradient_descent import GradientDescent


class TestOptimization(unittest.TestCase):

    def test_optimizer_initialization(self):
        opt = GradientDescent(learning_rate=0.01, max_epochs=100, momentum=0.9)
        self.assertEqual(opt.learning_rate, 0.01)
        self.assertEqual(opt.momentum, 0.9)

        with self.assertRaises(ValueError):
            GradientDescent(learning_rate=-0.1)

    def test_batching(self):
        opt = GradientDescent(batch_size=10)
        rng = random.Random(42)
        batches = opt.get_batches(n_samples=25, rng=rng)
        # Should create batches of size 10, 10, 5
        self.assertEqual(len(batches), 3)
        self.assertEqual(len(batches[0]), 10)
        self.assertEqual(len(batches[1]), 10)
        self.assertEqual(len(batches[2]), 5)

    def test_momentum_update(self):
        opt = GradientDescent(learning_rate=0.1, momentum=0.5)
        weights = [1.0]
        bias = 0.5
        grad_w = [2.0]
        grad_b = 1.0

        # Step 1:
        # v_w = 0.5 * 0 + 0.1 * 2.0 = 0.2 -> w = 1.0 - 0.2 = 0.8
        # v_b = 0.5 * 0 + 0.1 * 1.0 = 0.1 -> b = 0.5 - 0.1 = 0.4
        new_w, new_b = opt.update(weights, bias, grad_w, grad_b)
        self.assertAlmostEqual(new_w[0], 0.8)
        self.assertAlmostEqual(new_b, 0.4)


if __name__ == "__main__":
    unittest.main()
