"""
Loss functions module for EduPredict.
"""

from edupredict.losses.regression import (
    mean_squared_error,
    mean_absolute_error,
    mse_gradient,
)
from edupredict.losses.classification import (
    binary_cross_entropy,
    bce_gradient_wrt_logits,
)

__all__ = [
    "mean_squared_error",
    "mean_absolute_error",
    "mse_gradient",
    "binary_cross_entropy",
    "bce_gradient_wrt_logits",
]
