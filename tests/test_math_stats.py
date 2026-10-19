"""
Unit tests for statistics and mathematical utilities.
"""

import unittest
import math
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
)


class TestStatsMath(unittest.TestCase):

    def test_safe_sqrt(self):
        self.assertAlmostEqual(safe_sqrt(16.0), 4.0)
        self.assertAlmostEqual(safe_sqrt(0.0), 0.0)
        with self.assertRaises(ValueError):
            safe_sqrt(-5.0)

    def test_abs_val(self):
        self.assertEqual(abs_val(-7.5), 7.5)
        self.assertEqual(abs_val(4.2), 4.2)

    def test_summation_min_max(self):
        data = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0]
        self.assertAlmostEqual(summation(data), 23.0)
        self.assertEqual(min_val(data), 1.0)
        self.assertEqual(max_val(data), 9.0)

    def test_mean(self):
        self.assertAlmostEqual(mean([10.0, 20.0, 30.0]), 20.0)
        with self.assertRaises(ValueError):
            mean([])

    def test_median(self):
        # Odd
        self.assertEqual(median([1, 5, 2]), 2.0)
        # Even
        self.assertEqual(median([1, 2, 3, 4]), 2.5)

    def test_mode(self):
        self.assertEqual(mode(["A", "B", "A", "C"]), "A")
        self.assertEqual(mode([10, 20, 20, 30]), 20)

    def test_variance_and_std(self):
        data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        # Mean = 40 / 8 = 5.0
        # Sum sq diff = 9 + 1 + 1 + 1 + 0 + 0 + 4 + 16 = 32
        # Sample variance = 32 / 7 = 4.571428...
        # Population variance = 32 / 8 = 4.0
        self.assertAlmostEqual(variance(data, sample=False), 4.0)
        self.assertAlmostEqual(variance(data, sample=True), 32.0 / 7.0)
        self.assertAlmostEqual(standard_deviation(data, sample=False), 2.0)

    def test_covariance_and_correlation(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]  # Perfect linear correlation r = 1.0
        self.assertAlmostEqual(pearson_correlation(x, y), 1.0)

        # Inversely correlated r = -1.0
        y_inv = [-2.0, -4.0, -6.0, -8.0, -10.0]
        self.assertAlmostEqual(pearson_correlation(x, y_inv), -1.0)

        # Uncorrelated / zero variance
        const_y = [5.0, 5.0, 5.0, 5.0, 5.0]
        self.assertEqual(pearson_correlation(x, const_y), 0.0)


if __name__ == "__main__":
    unittest.main()
