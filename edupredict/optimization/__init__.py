"""
Optimization module for EduPredict.
"""

from edupredict.optimization.gradient_descent import (
    GradientDescent,
    initialize_parameters,
)

__all__ = ["GradientDescent", "initialize_parameters"]
