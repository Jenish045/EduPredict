"""
Statistical and mathematical utility functions implemented from first principles.

Strictly zero third-party dependencies and zero external library imports (no NumPy, no math, no statistics module).
Formulas are coded explicitly from fundamental mathematical definitions using pure Python.
"""

from typing import Sequence, List, Union, Tuple, Optional, Any
from collections import Counter

Number = Union[int, float]


def _is_nan(val: Number) -> bool:
    """Check if value is NaN without importing math."""
    return val != val


def safe_sqrt(x: Number) -> float:
    """
    Compute square root safely: sqrt(max(0.0, x)) = x ** 0.5.
    Raises TypeError if not a number, ValueError if x is NaN or negative beyond tolerance.
    """
    if not isinstance(x, (int, float)) or isinstance(x, bool):
        raise TypeError(f"Input must be numeric, got {type(x).__name__}")
    if _is_nan(x):
        raise ValueError("Cannot compute sqrt of NaN")
    if x < -1e-12:
        raise ValueError(f"Cannot compute square root of negative number: {x}")
    if x < 0.0:
        return 0.0
    return float(x) ** 0.5


def abs_val(x: Number) -> float:
    """Compute absolute value: |x|."""
    if not isinstance(x, (int, float)) or isinstance(x, bool):
        raise TypeError(f"Input must be numeric, got {type(x).__name__}")
    if _is_nan(x):
        raise ValueError("Cannot compute abs of NaN")
    val = float(x)
    return val if val >= 0.0 else -val


def summation(values: Sequence[Number]) -> float:
    """
    Compute sum of a sequence: sum_{i=1}^N x_i.
    Raises ValueError on empty sequence.
    """
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"values must be a sequence, got {type(values).__name__}")
    if len(values) == 0:
        raise ValueError("Cannot compute summation of empty sequence")
    total = 0.0
    for idx, v in enumerate(values):
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise TypeError(f"Item at index {idx} must be numeric, got {type(v).__name__}")
        if _is_nan(v):
            raise ValueError(f"Item at index {idx} is NaN")
        total += float(v)
    return total


def min_val(values: Sequence[Number]) -> float:
    """Compute minimum value in sequence."""
    if not values:
        raise ValueError("Cannot compute min of empty sequence")
    current_min = float("inf")
    for v in values:
        if v is not None and not _is_nan(v):
            flt = float(v)
            if flt < current_min:
                current_min = flt
    if current_min == float("inf"):
        raise ValueError("No valid numeric values in sequence")
    return current_min


def max_val(values: Sequence[Number]) -> float:
    """Compute maximum value in sequence."""
    if not values:
        raise ValueError("Cannot compute max of empty sequence")
    current_max = float("-inf")
    for v in values:
        if v is not None and not _is_nan(v):
            flt = float(v)
            if flt > current_max:
                current_max = flt
    if current_max == float("-inf"):
        raise ValueError("No valid numeric values in sequence")
    return current_max


def mean(values: Sequence[Number]) -> float:
    """
    Arithmetic mean:
    mu = (1 / N) * sum_{i=1}^N x_i.
    """
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"values must be a sequence, got {type(values).__name__}")
    if len(values) == 0:
        raise ValueError("Cannot compute mean of empty sequence")
    return summation(values) / float(len(values))


def median(values: Sequence[Number]) -> float:
    """
    Median: middle value of sorted sequence (or average of two middle values).
    """
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"values must be a sequence, got {type(values).__name__}")
    if len(values) == 0:
        raise ValueError("Cannot compute median of empty sequence")
    sorted_vals = sorted([float(x) for x in values])
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 1:
        return sorted_vals[mid]
    else:
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0


def mode(values: Sequence[Any]) -> Any:
    """
    Mode: most frequently occurring value in sequence.
    Breaks ties deterministically by selecting the smallest value.
    """
    if not values or len(values) == 0:
        raise ValueError("Cannot compute mode of empty sequence")
    counts = Counter(values)
    max_count = max(counts.values())
    candidates = [val for val, count in counts.items() if count == max_count]
    return sorted(candidates, key=lambda x: str(x))[0]


def variance(values: Sequence[Number], sample: bool = True) -> float:
    """
    Variance:
    s^2 = (1 / (N - 1)) * sum_{i=1}^N (x_i - mu)^2  (sample=True, Bessel's correction)
    sigma^2 = (1 / N) * sum_{i=1}^N (x_i - mu)^2    (sample=False, population variance)
    """
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"values must be a sequence, got {type(values).__name__}")
    n = len(values)
    if n == 0:
        raise ValueError("Cannot compute variance of empty sequence")
    if sample and n == 1:
        raise ValueError("Sample variance requires at least 2 observations (division by N-1)")

    mu = mean(values)
    sq_sum = 0.0
    for v in values:
        diff = float(v) - mu
        sq_sum += diff * diff

    divisor = float(n - 1) if sample else float(n)
    return sq_sum / divisor


def standard_deviation(values: Sequence[Number], sample: bool = True) -> float:
    """Standard deviation: sqrt(variance)."""
    return safe_sqrt(variance(values, sample=sample))


def covariance(x: Sequence[Number], y: Sequence[Number], sample: bool = True) -> float:
    """
    Sample or population covariance between two sequences x and y:
    Cov(x, y) = (1 / (N - 1)) * sum_{i=1}^N (x_i - mu_x)(y_i - mu_y).
    """
    if not isinstance(x, (list, tuple)) or not isinstance(y, (list, tuple)):
        raise TypeError("x and y must be sequence collections")
    if len(x) != len(y):
        raise ValueError(f"Lengths do not match: len(x)={len(x)}, len(y)={len(y)}")
    n = len(x)
    if n == 0:
        raise ValueError("Cannot compute covariance of empty sequences")
    if sample and n == 1:
        raise ValueError("Sample covariance requires at least 2 observations")

    mu_x = mean(x)
    mu_y = mean(y)

    cov_sum = 0.0
    for i in range(n):
        cov_sum += (float(x[i]) - mu_x) * (float(y[i]) - mu_y)

    divisor = float(n - 1) if sample else float(n)
    return cov_sum / divisor


def pearson_correlation(x: Sequence[Number], y: Sequence[Number]) -> float:
    """
    Pearson product-moment correlation coefficient r_xy:
    r_xy = Cov(x, y) / (s_x * s_y).
    Returns 0.0 if either sequence has zero standard deviation (constant feature).
    """
    std_x = standard_deviation(x, sample=True)
    std_y = standard_deviation(y, sample=True)

    if std_x < 1e-12 or std_y < 1e-12:
        return 0.0

    cov = covariance(x, y, sample=True)
    r = cov / (std_x * std_y)
    # Clamp to [-1.0, 1.0] to guard against floating-point roundoff
    if r > 1.0:
        return 1.0
    elif r < -1.0:
        return -1.0
    return r


def correlation_matrix(columns: Sequence[Sequence[Number]]) -> List[List[float]]:
    """
    Compute pairwise Pearson correlation matrix for multiple numeric column sequences.
    """
    n_cols = len(columns)
    matrix = [[1.0 if i == j else 0.0 for j in range(n_cols)] for i in range(n_cols)]
    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            r = pearson_correlation(columns[i], columns[j])
            matrix[i][j] = r
            matrix[j][i] = r
    return matrix
