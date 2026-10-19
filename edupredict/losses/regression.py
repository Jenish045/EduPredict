"""
Regression loss functions and gradients implemented from scratch.

Provides:
- mean_squared_error: J = (1 / (2*m)) * sum((y_pred - y_true)^2) or (1 / m) * sum((y_pred - y_true)^2)
- mean_absolute_error: MAE = (1 / m) * sum(|y_pred - y_true|)
- mse_gradient: dJ/d(y_pred) = (1 / m) * (y_pred - y_true)
"""

import math
from typing import Sequence, List, Union

Number = Union[int, float]


def _check_loss_inputs(y_true: Sequence[Number], y_pred: Sequence[Number]) -> None:
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Inputs cannot be empty")
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_true)} != len(y_pred)={len(y_pred)}")


def mean_squared_error(y_true: Sequence[Number], y_pred: Sequence[Number]) -> float:
    """
    Mean Squared Error:
    MSE = (1 / N) * sum_{i=1}^N (y_true_i - y_pred_i)^2.
    """
    _check_loss_inputs(y_true, y_pred)
    n = len(y_true)
    total_sq_err = 0.0
    for i in range(n):
        err = float(y_pred[i]) - float(y_true[i])
        total_sq_err += err * err
    return total_sq_err / float(n)


def mean_absolute_error(y_true: Sequence[Number], y_pred: Sequence[Number]) -> float:
    """
    Mean Absolute Error:
    MAE = (1 / N) * sum_{i=1}^N |y_true_i - y_pred_i|.
    """
    _check_loss_inputs(y_true, y_pred)
    n = len(y_true)
    total_abs_err = 0.0
    for i in range(n):
        err = abs(float(y_pred[i]) - float(y_true[i]))
        total_abs_err += err
    return total_abs_err / float(n)


def mse_gradient(y_true: Sequence[Number], y_pred: Sequence[Number]) -> List[float]:
    """
    Gradient of MSE with respect to predictions y_pred:
    d(MSE) / d(y_pred_i) = (2 / N) * (y_pred_i - y_true_i).
    """
    _check_loss_inputs(y_true, y_pred)
    n = len(y_true)
    factor = 2.0 / float(n)
    return [factor * (float(y_pred[i]) - float(y_true[i])) for i in range(n)]
