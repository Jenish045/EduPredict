"""
Unit tests for regression and classification loss functions.
"""

import unittest
from edupredict.losses.regression import mean_squared_error, mean_absolute_error, mse_gradient
from edupredict.losses.classification import binary_cross_entropy, bce_gradient_wrt_logits


class TestLosses(unittest.TestCase):

    def test_mean_squared_error(self):
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.0, 2.0, 5.0]  # error = [0, 0, 2], sq = [0, 0, 4], mean = 4/3
        self.assertAlmostEqual(mean_squared_error(y_true, y_pred), 4.0 / 3.0)

    def test_mean_absolute_error(self):
        y_true = [1.0, 2.0, 3.0]
        y_pred = [2.0, 2.0, 0.0]  # abs diff = [1, 0, 3], mean = 4/3
        self.assertAlmostEqual(mean_absolute_error(y_true, y_pred), 4.0 / 3.0)

    def test_mse_gradient(self):
        y_true = [1.0, 2.0]
        y_pred = [3.0, 2.0]
        # grad = 2/N * (y_pred - y_true) = 2/2 * [2.0, 0.0] = [2.0, 0.0]
        grad = mse_gradient(y_true, y_pred)
        self.assertEqual(grad, [2.0, 0.0])

    def test_binary_cross_entropy(self):
        y_true = [1, 0]
        # Near perfect predictions
        y_prob_good = [0.99, 0.01]
        loss_good = binary_cross_entropy(y_true, y_prob_good)
        self.assertTrue(loss_good < 0.05)

        # Bad predictions
        y_prob_bad = [0.01, 0.99]
        loss_bad = binary_cross_entropy(y_true, y_prob_bad)
        self.assertTrue(loss_bad > 4.0)

    def test_mismatched_inputs(self):
        with self.assertRaises(ValueError):
            mean_squared_error([1.0], [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
