"""
Unit tests for vector operations.
"""

import unittest
import math
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


class TestVectorMath(unittest.TestCase):

    def test_vector_add(self):
        self.assertEqual(vector_add([1, 2, 3], [4, 5, 6]), [5.0, 7.0, 9.0])
        self.assertEqual(vector_add([0.5, -1.0], [-0.5, 2.0]), [0.0, 1.0])

    def test_vector_sub(self):
        self.assertEqual(vector_sub([10, 20], [3, 5]), [7.0, 15.0])

    def test_vector_mul(self):
        self.assertEqual(vector_mul([2, 3, 4], [3, 4, 5]), [6.0, 12.0, 20.0])

    def test_vector_scalar_mul(self):
        self.assertEqual(vector_scalar_mul([1.0, 2.0, -3.0], 2.5), [2.5, 5.0, -7.5])

    def test_vector_scalar_add(self):
        self.assertEqual(vector_scalar_add([1.0, 2.0], 3.0), [4.0, 5.0])

    def test_dot_product(self):
        # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        self.assertAlmostEqual(dot_product([1, 2, 3], [4, 5, 6]), 32.0)
        # Orthogonal vectors
        self.assertAlmostEqual(dot_product([1, 0], [0, 1]), 0.0)

    def test_vector_norm(self):
        # 3-4-5 triangle norm
        self.assertAlmostEqual(vector_norm([3.0, 4.0], p=2.0), 5.0)
        # L1 norm
        self.assertAlmostEqual(vector_norm([-3.0, 4.0], p=1.0), 7.0)

    def test_euclidean_distance(self):
        self.assertAlmostEqual(euclidean_distance([0, 0], [3, 4]), 5.0)
        self.assertAlmostEqual(euclidean_distance([1, 2, 3], [1, 2, 3]), 0.0)

    def test_manhattan_distance(self):
        self.assertAlmostEqual(manhattan_distance([1, 2], [4, 6]), 7.0)

    def test_edge_cases(self):
        # Empty vector
        with self.assertRaises(ValueError):
            vector_add([], [])
        # Dimension mismatch
        with self.assertRaises(ValueError):
            vector_add([1, 2], [1, 2, 3])
        # Non-numeric input
        with self.assertRaises(TypeError):
            vector_add(["a", "b"], [1, 2])
        # NaN detection
        with self.assertRaises(ValueError):
            vector_add([float("nan"), 1.0], [2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
