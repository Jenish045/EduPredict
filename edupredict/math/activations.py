"""
Activation and numerical stability functions implemented from first principles.

Strictly zero third-party dependencies.
Includes numerically stable sigmoid, probability clipping, and safe logarithm.
"""

import math
from typing import Sequence, List, Union

Number = Union[int, float]


def clip(x: Number, min_val: float = -500.0, max_val: float = 500.0) -> float:
    """Clamp a numerical value within [min_val, max_val]."""
    val = float(x)
    if val < min_val:
        return min_val
    if val > max_val:
        return max_val
    return val


def sigmoid(z: Number) -> float:
    """
    Numerically stable sigmoid function:
    sigma(z) = 1 / (1 + exp(-z)).
    
    Prevents floating-point overflow for large positive or negative z:
    - If z >= 0: 1 / (1 + exp(-z))
    - If z < 0:  exp(z) / (1 + exp(z))
    """
    if not isinstance(z, (int, float)) or isinstance(z, bool):
        raise TypeError(f"z must be a numeric value, got {type(z).__name__}")
    if math.isnan(z):
        raise ValueError("Cannot compute sigmoid of NaN")
    
    val = float(z)
    if val >= 0.0:
        # For positive z, exp(-z) <= 1, preventing overflow
        return 1.0 / (1.0 + math.exp(-val))
    else:
        # For negative z, exp(z) <= 1, preventing overflow
        exp_z = math.exp(val)
        return exp_z / (1.0 + exp_z)


def sigmoid_vector(z_vec: Sequence[Number]) -> List[float]:
    """Apply sigmoid element-wise to a vector."""
    return [sigmoid(z) for z in z_vec]


def clip_probability(p: Number, eps: float = 1e-15) -> float:
    """
    Clip probability strictly into [eps, 1 - eps] to prevent log(0) in cross-entropy.
    """
    val = float(p)
    if val < eps:
        return eps
    if val > 1.0 - eps:
        return 1.0 - eps
    return val


def safe_log(x: Number, eps: float = 1e-15) -> float:
    """
    Compute log(max(x, eps)) to guard against zero or negative inputs.
    """
    val = float(x)
    if val < eps:
        val = eps
    return math.log(val)


def softmax(z_vec: Sequence[Number]) -> List[float]:
    """
    Numerically stable softmax for a vector of logits:
    p_i = exp(z_i - max(z)) / sum_j exp(z_j - max(z))
    """
    if len(z_vec) == 0:
        raise ValueError("Cannot compute softmax on empty vector")
    
    # Subtract max logit for stability
    max_z = float(z_vec[0])
    for item in z_vec:
        v = float(item)
        if v > max_z:
            max_z = v

    exp_vals = [math.exp(float(z) - max_z) for z in z_vec]
    sum_exp = sum(exp_vals)
    return [e / sum_exp for e in exp_vals]
