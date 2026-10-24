"""
Activation and numerical stability functions implemented from first principles.

Strictly zero third-party dependencies and zero standard math library imports.
Includes numerically stable sigmoid, probability clipping, pure Python exponential,
logarithm, and softmax implementations.
"""

from typing import Sequence, List, Union

Number = Union[int, float]

_LN2 = 0.693147180559945309417232


def _is_nan(val: Number) -> bool:
    """Check if value is NaN without importing math."""
    return val != val


def pure_exp(x: Number) -> float:
    """
    Compute e^x using range reduction and Taylor series expansion from scratch.
    """
    val = float(x)
    if _is_nan(val):
        return float("nan")
    if val > 709.0:
        return float("inf")
    if val < -709.0:
        return 0.0

    # Scale by 2048 (2^11) to bring input close to 0
    u = val / 2048.0
    term = 1.0
    s = 1.0
    for i in range(1, 16):
        term = term * u / i
        s += term

    # Repeated squaring: (e^(x/2048))^2048
    for _ in range(11):
        s = s * s

    return s


def pure_log(x: Number) -> float:
    """
    Compute natural logarithm ln(x) using range reduction and series expansion from scratch.
    """
    val = float(x)
    if _is_nan(val) or val <= 0.0:
        raise ValueError(f"Cannot compute logarithm of non-positive value or NaN: {val}")

    k = 0
    m = val
    while m > 1.5:
        m /= 2.0
        k += 1
    while m < 0.75:
        m *= 2.0
        k -= 1

    # For m in [0.75, 1.5], let z = (m - 1) / (m + 1)
    z = (m - 1.0) / (m + 1.0)
    z2 = z * z
    term = z
    s = 0.0
    for i in range(1, 27, 2):
        s += term / i
        term *= z2

    return k * _LN2 + 2.0 * s


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
    """
    if not isinstance(z, (int, float)) or isinstance(z, bool):
        raise TypeError(f"z must be a numeric value, got {type(z).__name__}")
    if _is_nan(z):
        raise ValueError("Cannot compute sigmoid of NaN")

    val = float(z)
    if val >= 0.0:
        return 1.0 / (1.0 + pure_exp(-val))
    else:
        exp_z = pure_exp(val)
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
    Compute log(max(x, eps)) safely.
    """
    val = float(x)
    if val < eps:
        val = eps
    return pure_log(val)


def softmax(z_vec: Sequence[Number]) -> List[float]:
    """
    Numerically stable softmax for a vector of logits:
    p_i = exp(z_i - max(z)) / sum_j exp(z_j - max(z))
    """
    if len(z_vec) == 0:
        raise ValueError("Cannot compute softmax on empty vector")

    max_z = float(z_vec[0])
    for item in z_vec:
        v = float(item)
        if v > max_z:
            max_z = v

    exp_vals = [pure_exp(float(z) - max_z) for z in z_vec]
    sum_exp = sum(exp_vals)
    return [e / sum_exp for e in exp_vals]
