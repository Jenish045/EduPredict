"""
Math package providing vectors, matrices, statistics, and activation functions
implemented completely from scratch without external dependencies.
"""

from edupredict.math.vectors import (
    vector_add,
    vector_sub,
    vector_mul,
    vector_scalar_mul,
    vector_scalar_add,
    dot_product,
    vector_norm,
    euclidean_distance,
    manhattan_distance,
)
from edupredict.math.matrices import (
    get_shape,
    create_zeros_matrix,
    create_identity_matrix,
    matrix_transpose,
    matrix_multiply,
    matrix_vector_mul,
    matrix_add,
    matrix_scalar_mul,
    invert_matrix,
)
from edupredict.math.stats import (
    safe_sqrt,
    abs_val,
    summation,
    min_val,
    max_val,
    mean,
    median,
    mode,
    variance,
    standard_deviation,
    covariance,
    pearson_correlation,
    correlation_matrix,
)
from edupredict.math.activations import (
    clip,
    sigmoid,
    sigmoid_vector,
    clip_probability,
    safe_log,
    softmax,
)

__all__ = [
    # Vectors
    "vector_add",
    "vector_sub",
    "vector_mul",
    "vector_scalar_mul",
    "vector_scalar_add",
    "dot_product",
    "vector_norm",
    "euclidean_distance",
    "manhattan_distance",
    # Matrices
    "get_shape",
    "create_zeros_matrix",
    "create_identity_matrix",
    "matrix_transpose",
    "matrix_multiply",
    "matrix_vector_mul",
    "matrix_add",
    "matrix_scalar_mul",
    "invert_matrix",
    # Stats
    "safe_sqrt",
    "abs_val",
    "summation",
    "min_val",
    "max_val",
    "mean",
    "median",
    "mode",
    "variance",
    "standard_deviation",
    "covariance",
    "pearson_correlation",
    "correlation_matrix",
    # Activations
    "clip",
    "sigmoid",
    "sigmoid_vector",
    "clip_probability",
    "safe_log",
    "softmax",
]
