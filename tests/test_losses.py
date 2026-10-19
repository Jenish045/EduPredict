"""
Unit tests for regression and classification loss functions,
hand-calculated baseline verifications, and numerical finite-difference gradient checks.
"""

import unittest
import math
from edupredict.losses.regression import mean_squared_error, mean_absolute_error, mse_gradient
from edupredict.losses.classification import (
    binary_cross_entropy,
    bce_gradient_wrt_logits,
    hinge_loss,
    hinge_subgradient,
)
from edupredict.math.activations import sigmoid


class TestLosses(unittest.TestCase):

    def test_hand_calculated_mse(self):
        # y_true = [2.0, 4.0, 6.0]
        # y_pred = [2.5, 3.5, 7.0]
        # errors = [+0.5, -0.5, +1.0]
        # sq_err = [0.25, 0.25, 1.00] -> sum = 1.50 -> MSE = 1.50 / 3 = 0.50
        y_true = [2.0, 4.0, 6.0]
        y_pred = [2.5, 3.5, 7.0]
        self.assertAlmostEqual(mean_squared_error(y_true, y_pred), 0.50)

    def test_hand_calculated_mae(self):
        # y_true = [2.0, 4.0, 6.0]
        # y_pred = [2.5, 3.5, 7.0]
        # abs_err = [0.5, 0.5, 1.0] -> sum = 2.0 -> MAE = 2.0 / 3 = 0.6666666...
        y_true = [2.0, 4.0, 6.0]
        y_pred = [2.5, 3.5, 7.0]
        self.assertAlmostEqual(mean_absolute_error(y_true, y_pred), 2.0 / 3.0)

    def test_hand_calculated_bce(self):
        # y_true = [1, 0]
        # y_prob = [0.8, 0.2]
        # Loss_0 = - (1 * ln(0.8) + 0) = -ln(0.8) approx 0.22314355
        # Loss_1 = - (0 + 1 * ln(1 - 0.2)) = -ln(0.8) approx 0.22314355
        # Total BCE = (0.22314355 + 0.22314355) / 2 = 0.22314355
        y_true = [1, 0]
        y_prob = [0.8, 0.2]
        expected_bce = -math.log(0.8)
        self.assertAlmostEqual(binary_cross_entropy(y_true, y_prob), expected_bce, places=7)

    def test_hand_calculated_hinge_loss(self):
        # y_true = [1, -1, 1]
        # y_margin = [0.5, -1.5, 2.0]
        # Sample 0: max(0, 1 - 1*0.5) = 0.5
        # Sample 1: max(0, 1 - (-1)*(-1.5)) = max(0, 1 - 1.5) = 0.0
        # Sample 2: max(0, 1 - 1*2.0) = max(0, -1.0) = 0.0
        # Total = (0.5 + 0.0 + 0.0) / 3 = 0.5 / 3 = 0.1666666...
        y_true = [1, -1, 1]
        y_margin = [0.5, -1.5, 2.0]
        self.assertAlmostEqual(hinge_loss(y_true, y_margin), 0.5 / 3.0, places=7)

    def test_mse_gradient_finite_difference(self):
        """
        Verify analytical MSE gradient matches numerical centered finite difference:
        grad_num = (MSE(y_pred + eps) - MSE(y_pred - eps)) / (2 * eps)
        """
        y_true = [3.0, 7.0, 2.5, 9.0]
        y_pred = [2.8, 7.4, 2.1, 9.2]
        eps = 1e-6

        analytical_grads = mse_gradient(y_true, y_pred)

        for i in range(len(y_pred)):
            y_pred_plus = list(y_pred)
            y_pred_minus = list(y_pred)
            y_pred_plus[i] += eps
            y_pred_minus[i] -= eps

            loss_plus = mean_squared_error(y_true, y_pred_plus)
            loss_minus = mean_squared_error(y_true, y_pred_minus)
            numerical_grad = (loss_plus - loss_minus) / (2.0 * eps)

            self.assertAlmostEqual(analytical_grads[i], numerical_grad, places=5)

    def test_bce_gradient_finite_difference(self):
        """
        Verify analytical BCE gradient with respect to linear logits z matches numerical finite difference:
        d(BCE)/dz_i = (BCE(z + eps) - BCE(z - eps)) / (2 * eps)
        """
        y_true = [1, 0, 1, 0]
        logits = [1.5, -0.8, 0.4, -1.2]
        probs = [sigmoid(z) for z in logits]
        eps = 1e-6

        analytical_grads = bce_gradient_wrt_logits(y_true, probs)

        for i in range(len(logits)):
            logits_plus = list(logits)
            logits_minus = list(logits)
            logits_plus[i] += eps
            logits_minus[i] -= eps

            probs_plus = [sigmoid(z) for z in logits_plus]
            probs_minus = [sigmoid(z) for z in logits_minus]

            loss_plus = binary_cross_entropy(y_true, probs_plus)
            loss_minus = binary_cross_entropy(y_true, probs_minus)
            numerical_grad = (loss_plus - loss_minus) / (2.0 * eps)

            self.assertAlmostEqual(analytical_grads[i], numerical_grad, places=5)

    def test_mismatched_inputs(self):
        with self.assertRaises(ValueError):
            mean_squared_error([1.0], [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
