"""
Regression evaluation metrics implemented from scratch.

Strictly zero third-party dependencies and zero math/statistics library imports.
"""

from typing import Sequence, Union
from edupredict.losses.regression import mean_squared_error, mean_absolute_error
from edupredict.math.stats import mean, safe_sqrt

Number = Union[int, float]


def root_mean_squared_error(y_true: Sequence[Number], y_pred: Sequence[Number]) -> float:
    """Root Mean Squared Error: sqrt(MSE)."""
    mse = mean_squared_error(y_true, y_pred)
    return safe_sqrt(mse)


def r2_score(y_true: Sequence[Number], y_pred: Sequence[Number]) -> float:
    """
    Coefficient of Determination R^2:
    R^2 = 1 - (SS_res / SS_tot)
    where SS_res = sum((y_i - y_hat_i)^2) and SS_tot = sum((y_i - y_mean)^2).
    Returns 1.0 if both y_true and y_pred are identical constant sequences.
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Cannot compute R2 score on empty sequence")
    if len(y_true) != len(y_pred):
        raise ValueError(f"Lengths do not match: len(y_true)={len(y_true)} != len(y_pred)={len(y_pred)}")

    n = len(y_true)
    y_true_f = [float(v) for v in y_true]
    y_pred_f = [float(v) for v in y_pred]

    ss_res = 0.0
    for i in range(n):
        diff = y_true_f[i] - y_pred_f[i]
        ss_res += diff * diff

    y_bar = mean(y_true_f)
    ss_tot = 0.0
    for i in range(n):
        diff = y_true_f[i] - y_bar
        ss_tot += diff * diff

    if ss_tot == 0.0:
        # Constant target
        return 1.0 if ss_res == 0.0 else 0.0

    return 1.0 - (ss_res / ss_tot)
