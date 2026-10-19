"""
Unit tests for gradient descent optimizer and parameter initializers.
"""

import unittest
import random
from edupredict.optimization.gradient_descent import GradientDescent, initialize_parameters


class TestOptimization(unittest.TestCase):

    def test_parameter_initialization(self):
        w_zeros, b_zeros = initialize_parameters(n_features=5, method="zeros")
        self.assertEqual(w_zeros, [0.0] * 5)
        self.assertEqual(b_zeros, 0.0)

        w_rand, b_rand = initialize_parameters(n_features=10, method="random", seed=42)
        self.assertEqual(len(w_rand), 10)
        self.assertEqual(b_rand, 0.0)

        w_xav, b_xav = initialize_parameters(n_features=8, method="xavier", seed=42)
        self.assertEqual(len(w_xav), 8)

        with self.assertRaises(ValueError):
            initialize_parameters(n_features=5, method="unsupported_init")

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

        new_w, new_b = opt.update(weights, bias, grad_w, grad_b)
        self.assertAlmostEqual(new_w[0], 0.8)
        self.assertAlmostEqual(new_b, 0.4)

    def test_convergence_and_early_stopping(self):
        opt = GradientDescent(early_stopping=True, patience=3, min_delta=0.01, tolerance=1e-4)

        # Check tolerance convergence
        self.assertTrue(opt.check_convergence(1.00005, 1.00002))
        self.assertFalse(opt.check_convergence(1.05, 1.00))

        # Check early stopping patience
        w = [1.0]
        b = 0.0
        # Epoch 1: improvement from inf to 1.0
        self.assertFalse(opt.should_stop_early(1.0, w, b))
        # Epoch 2: improvement to 0.8
        self.assertFalse(opt.should_stop_early(0.8, w, b))
        # Epoch 3: plateau (0.795, delta < 0.01) -> patience count = 1
        self.assertFalse(opt.should_stop_early(0.795, w, b))
        # Epoch 4: plateau -> patience count = 2
        self.assertFalse(opt.should_stop_early(0.794, w, b))
        # Epoch 5: plateau -> patience count = 3 -> STOP!
        self.assertTrue(opt.should_stop_early(0.793, w, b))


if __name__ == "__main__":
    unittest.main()
