"""
Unit tests for matrix operations and inversion.
"""

import unittest
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


class TestMatrixMath(unittest.TestCase):

    def test_shape_and_validation(self):
        A = [[1, 2, 3], [4, 5, 6]]
        self.assertEqual(get_shape(A), (2, 3))
        # Empty
        with self.assertRaises(ValueError):
            get_shape([])
        # Ragged
        with self.assertRaises(ValueError):
            get_shape([[1, 2], [3]])

    def test_identity_and_zeros(self):
        I = create_identity_matrix(3)
        self.assertEqual(I, [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        Z = create_zeros_matrix(2, 2)
        self.assertEqual(Z, [[0.0, 0.0], [0.0, 0.0]])

    def test_transpose(self):
        A = [[1, 2, 3], [4, 5, 6]]
        AT = matrix_transpose(A)
        self.assertEqual(AT, [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])

    def test_matrix_multiply(self):
        # 2x2 by 2x2
        A = [[1, 2], [3, 4]]
        B = [[2, 0], [1, 2]]
        # Row 0: [1*2 + 2*1, 1*0 + 2*2] = [4, 4]
        # Row 1: [3*2 + 4*1, 3*0 + 4*2] = [10, 8]
        C = matrix_multiply(A, B)
        self.assertEqual(C, [[4.0, 4.0], [10.0, 8.0]])

    def test_matrix_vector_mul(self):
        A = [[1, 2], [3, 4]]
        x = [2, 1]
        # [1*2 + 2*1, 3*2 + 4*1] = [4, 10]
        self.assertEqual(matrix_vector_mul(A, x), [4.0, 10.0])

    def test_invert_matrix(self):
        # Invert 2x2 matrix: [[4, 7], [2, 6]]
        # det = 24 - 14 = 10
        # inv = 1/10 * [[6, -7], [-2, 4]] = [[0.6, -0.7], [-0.2, 0.4]]
        A = [[4, 7], [2, 6]]
        invA = invert_matrix(A)
        self.assertAlmostEqual(invA[0][0], 0.6)
        self.assertAlmostEqual(invA[0][1], -0.7)
        self.assertAlmostEqual(invA[1][0], -0.2)
        self.assertAlmostEqual(invA[1][1], 0.4)

        # A * invA should equal Identity
        I_check = matrix_multiply(A, invA)
        for i in range(2):
            for j in range(2):
                expected = 1.0 if i == j else 0.0
                self.assertAlmostEqual(I_check[i][j], expected, places=7)

    def test_singular_matrix_inversion(self):
        # Singular matrix (col 2 is multiple of col 1)
        A = [[1, 2], [2, 4]]
        with self.assertRaises(ValueError):
            invert_matrix(A)


if __name__ == "__main__":
    unittest.main()
