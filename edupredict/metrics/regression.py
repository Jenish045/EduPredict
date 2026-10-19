"""
Regression evaluation metrics implemented from first principles.

Provides:
- mean_squared_error (MSE)
- root_mean_squared_error (RMSE)
- mean_absolute_error (MAE)
- r2_score (Coefficient of Determination)
"""

import math
from typing import Sequence, Union
from edupredict.math.stats import mean, safe_sqrt
from edupredict.losses.regression import mean_squared_error, mean_absolute_error

Number = Union[int, float]


def root_mean_squared_error(y_true: Sequence[Number], y_pred: Sequence[Number]) -> float:
    """
    Root Mean Squared Error: sqrt(MSE).
    """
    mse = mean_squared_error(y_true, y_pred)
    return safe_sqrt(mse)


def r2_score(y_true: Sequence[Number], y_pred: Sequence[Number]) -> float:
    """
    Coefficient of Determination R^2:
    R^2 = 1 - (SS_res / SS_tot)
    where SS_res = sum((y_i - y_hat_i)^2) and SS_tot = sum((y_i - y_mean)^2).
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Cannot compute R2 score on empty sequence")
    if len(y_true) != len(y_pred):
        raise ValueError(f"Lengths do not match: len(y_true)={len(y_true)} != len(y_pred)={len(y_pred)}")

    y_bar = mean(y_true)
    ss_res = 0.0
    ss_tot = 0.0

    for i in range(len(y_true)):
        y_i = float(y_true[i])
        y_hat_i = float(y_pred[i])
        res = y_i - y_hat_i
        ss_res += res * res
        diff_tot = y_i - y_bar
        ss_tot += diff_tot * diff_tot

    if ss_tot < 1e-12:
        # All target values are identical
        return 1.0 if ss_res < 1e-12 else 0.0

    return 1.0 - (ss_res / ss_tot)
