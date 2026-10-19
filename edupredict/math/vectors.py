"""
Vector operations implemented from scratch using Python lists.

Strictly zero third-party dependencies (no NumPy).
Handles dimension checking, empty inputs, and numerical edge cases explicitly.
"""

import math
from typing import Sequence, List, Union

Number = Union[int, float]


def _check_non_empty_vector(v: Sequence[Number], name: str = "vector") -> None:
    """Validate that vector is a non-empty sequence of numbers."""
    if not isinstance(v, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple, got {type(v).__name__}")
    if len(v) == 0:
        raise ValueError(f"{name} cannot be empty")
    for idx, item in enumerate(v):
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise TypeError(f"{name}[{idx}] must be a numeric float/int, got {type(item).__name__}")
        if math.isnan(item) or math.isinf(item):
            raise ValueError(f"{name}[{idx}] contains invalid numerical value: {item}")


def _check_matching_dimensions(u: Sequence[Number], v: Sequence[Number]) -> None:
    """Ensure both vectors have matching dimensions."""
    _check_non_empty_vector(u, "u")
    _check_non_empty_vector(v, "v")
    if len(u) != len(v):
        raise ValueError(f"Vector dimensions do not match: len(u)={len(u)} != len(v)={len(v)}")


def vector_add(u: Sequence[Number], v: Sequence[Number]) -> List[float]:
    """
    Element-wise vector addition: w_i = u_i + v_i.
    """
    _check_matching_dimensions(u, v)
    return [float(u[i] + v[i]) for i in range(len(u))]


def vector_sub(u: Sequence[Number], v: Sequence[Number]) -> List[float]:
    """
    Element-wise vector subtraction: w_i = u_i - v_i.
    """
    _check_matching_dimensions(u, v)
    return [float(u[i] - v[i]) for i in range(len(u))]


def vector_mul(u: Sequence[Number], v: Sequence[Number]) -> List[float]:
    """
    Hadamard (element-wise) vector multiplication: w_i = u_i * v_i.
    """
    _check_matching_dimensions(u, v)
    return [float(u[i] * v[i]) for i in range(len(u))]


def vector_scalar_mul(v: Sequence[Number], c: Number) -> List[float]:
    """
    Scalar multiplication: w_i = c * v_i.
    """
    _check_non_empty_vector(v, "v")
    if not isinstance(c, (int, float)) or isinstance(c, bool):
        raise TypeError(f"Scalar multiplier must be numeric, got {type(c).__name__}")
    if math.isnan(c) or math.isinf(c):
        raise ValueError(f"Scalar multiplier cannot be NaN or Inf: {c}")
    c_flt = float(c)
    return [float(val * c_flt) for val in v]


def vector_scalar_add(v: Sequence[Number], c: Number) -> List[float]:
    """
    Scalar addition: w_i = v_i + c.
    """
    _check_non_empty_vector(v, "v")
    if not isinstance(c, (int, float)) or isinstance(c, bool):
        raise TypeError(f"Scalar must be numeric, got {type(c).__name__}")
    if math.isnan(c) or math.isinf(c):
        raise ValueError(f"Scalar cannot be NaN or Inf: {c}")
    c_flt = float(c)
    return [float(val + c_flt) for val in v]


def dot_product(u: Sequence[Number], v: Sequence[Number]) -> float:
    """
    Dot product (inner product) of two vectors: sum(u_i * v_i).
    """
    _check_matching_dimensions(u, v)
    total = 0.0
    for i in range(len(u)):
        total += float(u[i]) * float(v[i])
    return total


def vector_norm(v: Sequence[Number], p: float = 2.0) -> float:
    """
    Calculate Lp norm of vector: (sum(|v_i|^p))^(1/p).
    Default p=2 is Euclidean norm (L2).
    """
    _check_non_empty_vector(v, "v")
    if p <= 0:
        raise ValueError(f"Norm order p must be positive, got {p}")
    total = 0.0
    for item in v:
        total += abs(float(item)) ** p
    return total ** (1.0 / p)


def euclidean_distance(u: Sequence[Number], v: Sequence[Number]) -> float:
    """
    Euclidean distance between two vectors: sqrt(sum((u_i - v_i)^2)).
    """
    _check_matching_dimensions(u, v)
    diff_sq_sum = 0.0
    for i in range(len(u)):
        diff = float(u[i]) - float(v[i])
        diff_sq_sum += diff * diff
    return math.sqrt(diff_sq_sum)


def manhattan_distance(u: Sequence[Number], v: Sequence[Number]) -> float:
    """
    Manhattan distance (L1) between two vectors: sum(|u_i - v_i|).
    """
    _check_matching_dimensions(u, v)
    diff_abs_sum = 0.0
    for i in range(len(u)):
        diff_abs_sum += abs(float(u[i]) - float(v[i]))
    return diff_abs_sum
