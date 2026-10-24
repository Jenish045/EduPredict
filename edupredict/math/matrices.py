"""
Matrix operations implemented from first principles using 2D Python lists.

Strictly zero third-party dependencies and zero external mathematical library imports (no NumPy, no math module).
Supports matrix shape validation, transposition, multiplication, matrix-vector multiplication,
identity matrix creation, and Gauss-Jordan elimination with partial row pivoting for matrix inversion.
"""

from typing import List, Tuple, Sequence, Union

Number = Union[int, float]
Matrix = List[List[float]]
Vector = List[float]


def _is_nan(val: Number) -> bool:
    """Check if value is NaN without importing math."""
    return val != val


def _is_inf(val: Number) -> bool:
    """Check if value is infinity without importing math."""
    return val == float("inf") or val == float("-inf")


def get_shape(A: Sequence[Sequence[Number]]) -> Tuple[int, int]:
    """
    Get (rows, cols) shape of a 2D matrix.
    Raises ValueError if empty or ragged, and TypeError if non-numeric.
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
            if _is_nan(val) or _is_inf(val):
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


def matrix_add(A: Sequence[Sequence[Number]], B: Sequence[Sequence[Number]]) -> Matrix:
    """Add two matrices element-wise: C = A + B."""
    rows_A, cols_A = get_shape(A)
    rows_B, cols_B = get_shape(B)
    if rows_A != rows_B or cols_A != cols_B:
        raise ValueError(
            f"Matrix dimensions must match for addition: ({rows_A}, {cols_A}) vs ({rows_B}, {cols_B})"
        )
    return [[float(A[r][c]) + float(B[r][c]) for c in range(cols_A)] for r in range(rows_A)]


def matrix_scalar_mul(A: Sequence[Sequence[Number]], scalar: Number) -> Matrix:
    """Multiply a matrix by a scalar: C = scalar * A."""
    rows, cols = get_shape(A)
    s = float(scalar)
    return [[float(A[r][c]) * s for c in range(cols)] for r in range(rows)]


def matrix_transpose(A: Sequence[Sequence[Number]]) -> Matrix:
    """Transpose matrix A (m x n) -> A^T (n x m)."""
    rows, cols = get_shape(A)
    return [[float(A[r][c]) for r in range(rows)] for c in range(cols)]


def matrix_multiply(A: Sequence[Sequence[Number]], B: Sequence[Sequence[Number]]) -> Matrix:
    """
    Compute standard matrix multiplication: C = A * B.
    Dimensions: A is (m x p), B is (p x n) -> C is (m x n).
    """
    rows_A, cols_A = get_shape(A)
    rows_B, cols_B = get_shape(B)

    if cols_A != rows_B:
        raise ValueError(
            f"Inner matrix dimensions must match: A cols ({cols_A}) != B rows ({rows_B})"
        )

    C = create_zeros_matrix(rows_A, cols_B)
    for i in range(rows_A):
        for k in range(cols_A):
            a_ik = float(A[i][k])
            for j in range(cols_B):
                C[i][j] += a_ik * float(B[k][j])
    return C


def matrix_vector_mul(A: Sequence[Sequence[Number]], v: Sequence[Number]) -> Vector:
    """Compute matrix-vector product: y = A * v."""
    rows_A, cols_A = get_shape(A)
    if not isinstance(v, (list, tuple)) or len(v) == 0:
        raise ValueError("Vector v must be a non-empty sequence")
    if len(v) != cols_A:
        raise ValueError(f"Dimension mismatch: A has {cols_A} cols, but v has length {len(v)}")

    result: Vector = [0.0] * rows_A
    for i in range(rows_A):
        dot = 0.0
        for j in range(cols_A):
            dot += float(A[i][j]) * float(v[j])
        result[i] = dot
    return result


def invert_matrix(A: Sequence[Sequence[Number]], tol: float = 1e-12) -> Matrix:
    """
    Compute inverse of an n x n square matrix using Gauss-Jordan elimination with partial row pivoting.
    
    Augments matrix A with identity matrix I: [A | I], performs elementary row operations
    to reduce the left side to reduced row echelon form (I), resulting in [I | A^(-1)].
    Raises ValueError if matrix is singular (non-invertible) or non-square.
    """
    rows, cols = get_shape(A)
    if rows != cols:
        raise ValueError(f"Only square matrices can be inverted, got shape ({rows}, {cols})")
    n = rows

    # Construct augmented matrix [A | I]
    augmented: Matrix = []
    for i in range(n):
        row = [float(A[i][j]) for j in range(n)]
        row.extend([1.0 if i == j else 0.0 for j in range(n)])
        augmented.append(row)

    # Gauss-Jordan elimination with partial pivoting
    for col in range(n):
        # 1. Partial pivoting: find largest absolute value in current column at or below diagonal
        max_row = col
        max_val = augmented[col][col]
        max_val_abs = max_val if max_val >= 0.0 else -max_val
        for r in range(col + 1, n):
            val = augmented[r][col]
            val_abs = val if val >= 0.0 else -val
            if val_abs > max_val_abs:
                max_val_abs = val_abs
                max_row = r

        if max_val_abs < tol:
            raise ValueError(f"Matrix is singular or nearly singular (pivot={max_val_abs:.2e} < {tol:.2e})")

        # Swap rows if necessary
        if max_row != col:
            augmented[col], augmented[max_row] = augmented[max_row], augmented[col]

        # 2. Scale pivot row so diagonal element equals 1.0
        pivot = augmented[col][col]
        for c in range(2 * n):
            augmented[col][c] /= pivot

        # 3. Eliminate all other entries in current column (both above and below diagonal)
        for r in range(n):
            if r != col:
                factor = augmented[r][col]
                if factor != 0.0:
                    for c in range(2 * n):
                        augmented[r][c] -= factor * augmented[col][c]

    # Extract right-hand side [A^(-1)]
    inv_A: Matrix = []
    for i in range(n):
        inv_A.append(augmented[i][n:])
    return inv_A
