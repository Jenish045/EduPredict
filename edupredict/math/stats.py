"""
Statistical and mathematical utility functions implemented from first principles.

Strictly zero third-party dependencies.
Does NOT delegate to Python's statistics module; formulas are coded explicitly
from fundamental mathematical definitions for clear educational inspection and viva explanation.
"""

import math
from typing import Sequence, List, Union, Tuple, Optional

Number = Union[int, float]


def safe_sqrt(x: Number) -> float:
    """
    Compute square root safely: sqrt(max(0.0, x)).
    Raises TypeError if not a number, ValueError if x is NaN or negative beyond tolerance.
    """
    if not isinstance(x, (int, float)) or isinstance(x, bool):
        raise TypeError(f"Input must be numeric, got {type(x).__name__}")
    if math.isnan(x):
        raise ValueError("Cannot compute sqrt of NaN")
    if x < -1e-12:
        raise ValueError(f"Cannot compute square root of negative number: {x}")
    if x < 0.0:
        return 0.0
    return math.sqrt(float(x))


def abs_val(x: Number) -> float:
    """Compute absolute value: |x|."""
    if not isinstance(x, (int, float)) or isinstance(x, bool):
        raise TypeError(f"Input must be numeric, got {type(x).__name__}")
    if math.isnan(x):
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
        if math.isnan(v):
            raise ValueError(f"Item at index {idx} is NaN")
        total += float(v)
    return total


def min_val(values: Sequence[Number]) -> float:
    """
    Compute minimum value from scratch.
    """
    if not isinstance(values, (list, tuple)) or len(values) == 0:
        raise ValueError("Cannot compute min of empty or invalid sequence")
    lowest = float(values[0])
    for idx, item in enumerate(values):
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise TypeError(f"Item at index {idx} must be numeric, got {type(item).__name__}")
        if math.isnan(item):
            raise ValueError(f"Item at index {idx} is NaN")
        v = float(item)
        if v < lowest:
            lowest = v
    return lowest


def max_val(values: Sequence[Number]) -> float:
    """
    Compute maximum value from scratch.
    """
    if not isinstance(values, (list, tuple)) or len(values) == 0:
        raise ValueError("Cannot compute max of empty or invalid sequence")
    highest = float(values[0])
    for idx, item in enumerate(values):
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise TypeError(f"Item at index {idx} must be numeric, got {type(item).__name__}")
        if math.isnan(item):
            raise ValueError(f"Item at index {idx} is NaN")
        v = float(item)
        if v > highest:
            highest = v
    return highest


def mean(values: Sequence[Number]) -> float:
    """
    Compute arithmetic mean from first principles:
    mu = (1 / N) * sum_{i=1}^N x_i.
    """
    n = len(values) if isinstance(values, (list, tuple)) else 0
    if n == 0:
        raise ValueError("Cannot compute mean of empty sequence")
    return summation(values) / float(n)


def median(values: Sequence[Number]) -> float:
    """
    Compute median from first principles:
    Sorts sequence; returns middle element (if odd) or average of two middle elements (if even).
    """
    if not isinstance(values, (list, tuple)) or len(values) == 0:
        raise ValueError("Cannot compute median of empty sequence")
    
    cleaned = []
    for idx, x in enumerate(values):
        if not isinstance(x, (int, float)) or isinstance(x, bool):
            raise TypeError(f"Item at index {idx} must be numeric, got {type(x).__name__}")
        if math.isnan(x):
            raise ValueError(f"Item at index {idx} is NaN")
        cleaned.append(float(x))

    # Sort using Python standard Timsort (O(N log N))
    sorted_vals = sorted(cleaned)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 1:
        return sorted_vals[mid]
    else:
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0


def mode(values: Sequence[Union[Number, str]]) -> Union[Number, str]:
    """
    Compute mode (most frequent value) from first principles.
    In case of ties, returns the first encountered highest-frequency element.
    """
    if not isinstance(values, (list, tuple)) or len(values) == 0:
        raise ValueError("Cannot compute mode of empty sequence")

    freq: dict = {}
    for item in values:
        if isinstance(item, float) and math.isnan(item):
            continue
        freq[item] = freq.get(item, 0) + 1

    if not freq:
        raise ValueError("No valid items to compute mode")

    best_item = None
    max_count = -1
    for k, count in freq.items():
        if count > max_count:
            max_count = count
            best_item = k
    return best_item


def variance(values: Sequence[Number], sample: bool = True) -> float:
    """
    Compute variance from first principles:
    Sample:    s^2 = sum_{i=1}^N (x_i - mu)^2 / (N - 1)
    Population: sigma^2 = sum_{i=1}^N (x_i - mu)^2 / N
    """
    if not isinstance(values, (list, tuple)):
        raise TypeError("values must be a sequence")
    n = len(values)
    if n == 0:
        raise ValueError("Cannot compute variance of empty sequence")
    if sample and n < 2:
        raise ValueError("Sample variance requires at least 2 data points (degrees of freedom n-1)")

    mu = mean(values)
    sum_sq_diff = 0.0
    for v in values:
        diff = float(v) - mu
        sum_sq_diff += diff * diff

    denom = float(n - 1) if sample else float(n)
    return sum_sq_diff / denom


def standard_deviation(values: Sequence[Number], sample: bool = True) -> float:
    """
    Compute standard deviation: sqrt(variance).
    """
    var = variance(values, sample=sample)
    return safe_sqrt(var)


def covariance(x: Sequence[Number], y: Sequence[Number], sample: bool = True) -> float:
    """
    Compute covariance between two series x and y from first principles:
    Cov(x, y) = sum_{i=1}^N (x_i - mu_x)(y_i - mu_y) / (N - 1) [sample]
    """
    if not isinstance(x, (list, tuple)) or not isinstance(y, (list, tuple)):
        raise TypeError("x and y must be sequences")
    if len(x) == 0 or len(y) == 0:
        raise ValueError("x and y cannot be empty")
    if len(x) != len(y):
        raise ValueError(f"Lengths do not match: len(x)={len(x)}, len(y)={len(y)}")
    n = len(x)
    if sample and n < 2:
        raise ValueError("Sample covariance requires at least 2 observations")

    mu_x = mean(x)
    mu_y = mean(y)

    total = 0.0
    for i in range(n):
        diff_x = float(x[i]) - mu_x
        diff_y = float(y[i]) - mu_y
        total += diff_x * diff_y

    denom = float(n - 1) if sample else float(n)
    return total / denom


def pearson_correlation(x: Sequence[Number], y: Sequence[Number]) -> float:
    """
    Compute Pearson correlation coefficient r:
    r = Cov(x, y) / (sigma_x * sigma_y)
    Bounded in [-1.0, 1.0].
    Returns 0.0 if either series has zero variance (degenerate constant).
    """
    cov = covariance(x, y, sample=True)
    std_x = standard_deviation(x, sample=True)
    std_y = standard_deviation(y, sample=True)

    denom = std_x * std_y
    if denom <= 1e-12:
        return 0.0

    r = cov / denom
    # Clamp to [-1.0, 1.0] to guard against slight floating point overflow
    if r > 1.0:
        return 1.0
    if r < -1.0:
        return -1.0
    return r
