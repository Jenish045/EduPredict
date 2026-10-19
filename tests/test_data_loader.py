"""
Unit tests for data loading, dataset container, and generator.
"""

import unittest
import os
import tempfile
from edupredict.data.loader import load_csv, save_csv, try_cast_numeric
from edupredict.data.dataset import Dataset
from edupredict.data.generator import generate_student_data


class TestDataLoader(unittest.TestCase):

    def test_try_cast_numeric(self):
        self.assertEqual(try_cast_numeric("42"), 42)
        self.assertEqual(try_cast_numeric("3.14"), 3.14)
        self.assertEqual(try_cast_numeric("hello"), "hello")
        self.assertIsNone(try_cast_numeric(""))
        self.assertIsNone(try_cast_numeric("NaN"))
        self.assertIsNone(try_cast_numeric("null"))

    def test_save_and_load_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.csv")
            headers = ["id", "score", "category"]
            rows = [
                ["A1", 85.5, "Pass"],
                ["A2", 45.0, "Fail"],
                ["A3", None, "Pass"],
            ]
            save_csv(file_path, headers, rows)

            loaded_headers, loaded_rows = load_csv(file_path)
            self.assertEqual(loaded_headers, headers)
            self.assertEqual(len(loaded_rows), 3)
            self.assertEqual(loaded_rows[0][1], 85.5)
            self.assertIsNone(loaded_rows[2][1])

    def test_dataset_operations(self):
        headers = ["f1", "f2", "target"]
        rows = [
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, None, 300.0],
        ]
        ds = Dataset(headers, rows, target_col="target")
        self.assertEqual(ds.shape, (3, 3))
        self.assertEqual(ds.get_column("f1"), [1.0, 2.0, 3.0])
        self.assertEqual(ds.count_missing()["f2"], 1)

        X, y = ds.split_features_target()
        self.assertEqual(len(X), 3)
        self.assertEqual(len(X[0]), 2)
        self.assertEqual(y, [100.0, 200.0, 300.0])

    def test_generator_deterministic(self):
        ds1 = generate_student_data(n_samples=50, random_seed=123)
        ds2 = generate_student_data(n_samples=50, random_seed=123)
        self.assertEqual(ds1.rows, ds2.rows)
        self.assertEqual(len(ds1), 50)


if __name__ == "__main__":
    unittest.main()
