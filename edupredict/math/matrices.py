"""
Matrix operations implemented from scratch using 2D Python lists (List[List[float]]).

Strictly zero third-party dependencies (no NumPy).
Supports transpose, multiplication, matrix-vector multiplication,
shape inspection, Gaussian elimination, and matrix inversion.
"""

import math
from typing import List, Tuple, Sequence, Union

Number = Union[int, float]
Matrix = List[List[float]]
Vector = List[float]


def get_shape(A: Sequence[Sequence[Number]]) -> Tuple[int, int]:
    """
    Get the (rows, cols) shape of a 2D matrix.
    Raises ValueError if empty or non-rectangular.
    """
    if not isinstance(A, (list, tuple)) or len(A) == 0:
        raise ValueError("Matrix must be a non-empty 2D sequence")
    num_rows = len(A)
    if not isinstance(A[0], (list, tuple)) or len(A[0]) == 0:
        raise ValueError("Matrix rows must be non-empty sequences")
    num_cols = len(A[0])
    for r_idx, row in enumerate(A):
        if not isinstance(row, (list, tuple)):
            raise TypeError(f"Row {r_idx} must be a list or tuple, got {type(row).__name__}")
        if len(row) != num_cols:
            raise ValueError(f"Matrix is ragged: row 0 has {num_cols} cols, row {r_idx} has {len(row)} cols")
        for c_idx, val in enumerate(row):
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise TypeError(f"A[{r_idx}][{c_idx}] must be a float or int, got {type(val).__name__}")
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"A[{r_idx}][{c_idx}] contains invalid numerical value: {val}")
    return num_rows, num_cols


def create_zeros_matrix(rows: int, cols: int) -> Matrix:
    """Create a matrix of zeros with given rows and cols."""
    if rows <= 0 or cols <= 0:
        raise ValueError(f"Dimensions must be positive integers, got rows={rows}, cols={cols}")
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def create_identity_matrix(n: int) -> Matrix:
    """Create an n x n identity matrix."""
    if n <= 0:
        raise ValueError(f"Identity matrix dimension must be positive, got {n}")
    I = create_zeros_matrix(n, n)
    for i in range(n):
        I[i][i] = 1.0
    return I


def matrix_transpose(A: Sequence[Sequence[Number]]) -> Matrix:
    """
    Transpose matrix A (m x n) -> A^T (n x m).
    A^T[j][i] = A[i][j].
    """
    rows, cols = get_shape(A)
    T = create_zeros_matrix(cols, rows)
    for i in range(rows):
        for j in range(cols):
            T[j][i] = float(A[i][j])
    return T


def matrix_multiply(A: Sequence[Sequence[Number]], B: Sequence[Sequence[Number]]) -> Matrix:
    """
    Multiply matrix A (m x k) with matrix B (k x n) to produce C (m x n).
    C[i][j] = sum_p A[i][p] * B[p][j].
    """
    r_A, c_A = get_shape(A)
    r_B, c_B = get_shape(B)
    if c_A != r_B:
        raise ValueError(f"Cannot multiply matrix shapes ({r_A}, {c_A}) and ({r_B}, {c_B}): columns of A must match rows of B")

    C = create_zeros_matrix(r_A, c_B)
    for i in range(r_A):
        for k in range(c_A):
            a_ik = float(A[i][k])
            if a_ik == 0.0:
                continue
            for j in range(c_B):
                C[i][j] += a_ik * float(B[k][j])
    return C


def matrix_vector_mul(A: Sequence[Sequence[Number]], x: Sequence[Number]) -> Vector:
    """
    Multiply matrix A (m x n) with vector x (n) to produce vector y (m).
    y[i] = sum_j A[i][j] * x[j].
    """
    r_A, c_A = get_shape(A)
    if len(x) != c_A:
        raise ValueError(f"Matrix columns ({c_A}) do not match vector length ({len(x)})")
    
    y = [0.0] * r_A
    for i in range(r_A):
        acc = 0.0
        row = A[i]
        for j in range(c_A):
            acc += float(row[j]) * float(x[j])
        y[i] = acc
    return y


def matrix_add(A: Sequence[Sequence[Number]], B: Sequence[Sequence[Number]]) -> Matrix:
    """
    Element-wise matrix addition: C[i][j] = A[i][j] + B[i][j].
    """
    r_A, c_A = get_shape(A)
    r_B, c_B = get_shape(B)
    if r_A != r_B or c_A != c_B:
        raise ValueError(f"Matrix shapes ({r_A}, {c_A}) and ({r_B}, {c_B}) must match for addition")
    C = create_zeros_matrix(r_A, c_A)
    for i in range(r_A):
        for j in range(c_A):
            C[i][j] = float(A[i][j]) + float(B[i][j])
    return C


def matrix_scalar_mul(A: Sequence[Sequence[Number]], c: Number) -> Matrix:
    """
    Multiply matrix A by scalar c: C[i][j] = c * A[i][j].
    """
    r_A, c_A = get_shape(A)
    c_flt = float(c)
    C = create_zeros_matrix(r_A, c_A)
    for i in range(r_A):
        for j in range(c_A):
            C[i][j] = float(A[i][j]) * c_flt
    return C


def invert_matrix(A: Sequence[Sequence[Number]], eps: float = 1e-12) -> Matrix:
    """
    Invert square matrix A using Gauss-Jordan elimination with partial pivoting.
    Strictly standard library implementation.
    Raises ValueError if matrix is not square or singular (non-invertible).
    """
    n, cols = get_shape(A)
    if n != cols:
        raise ValueError(f"Matrix must be square to invert, got shape ({n}, {cols})")

    # Create augmented matrix [A | I]
    aug = [[float(A[i][j]) for j in range(n)] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for col in range(n):
        # Partial pivoting: find max element in current column
        pivot_row = col
        max_val = abs(aug[col][col])
        for r in range(col + 1, n):
            candidate = abs(aug[r][col])
            if candidate > max_val:
                max_val = candidate
                pivot_row = r

        if max_val < eps:
            raise ValueError(f"Matrix is singular or near-singular (pivot {max_val} < {eps}) and cannot be inverted")

        # Swap rows if needed
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

        # Normalize pivot row
        pivot = aug[col][col]
        for c in range(2 * n):
            aug[col][c] /= pivot

        # Eliminate other rows
        for r in range(n):
            if r != col:
                factor = aug[r][col]
                if abs(factor) > 0.0:
                    for c in range(2 * n):
                        aug[r][c] -= factor * aug[col][c]

    # Extract right side
    inv = create_zeros_matrix(n, n)
    for i in range(n):
        for j in range(n):
            inv[i][j] = aug[i][n + j]
    return inv
