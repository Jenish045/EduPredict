"""
Unit tests for data loading, dataset container, validation, and generator.
"""

import unittest
import os
import tempfile
from edupredict.data.loader import (
    load_csv,
    save_csv,
    try_cast_numeric,
    validate_dataset,
    ValidationReport,
)
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
        self.assertIsNone(try_cast_numeric("?"))

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

    def test_malformed_csv_row(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "malformed.csv")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("col_a,col_b,col_c\n")
                f.write("1,2,3\n")
                f.write("4,5\n")  # Missing column at line 3

            with self.assertRaises(ValueError) as ctx:
                load_csv(file_path)
            self.assertIn("line 3", str(ctx.exception))
            self.assertIn("expected 3", str(ctx.exception))

    def test_empty_and_nonexistent_csv(self):
        with self.assertRaises(FileNotFoundError):
            load_csv("non_existent_file_path_123.csv")

        with tempfile.TemporaryDirectory() as tmpdir:
            empty_file = os.path.join(tmpdir, "empty.csv")
            with open(empty_file, "w", encoding="utf-8") as f:
                pass  # 0 bytes
            with self.assertRaises(ValueError):
                load_csv(empty_file)

    def test_duplicate_headers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dup_file = os.path.join(tmpdir, "dup.csv")
            with open(dup_file, "w", encoding="utf-8") as f:
                f.write("id,score,score\n")
                f.write("1,50,50\n")
            with self.assertRaises(ValueError) as ctx:
                load_csv(dup_file)
            self.assertIn("Duplicate", str(ctx.exception))

    def test_validate_dataset_schema_and_ranges(self):
        headers = ["id", "age", "grade", "passed"]
        rows = [
            ["S1", 20, 85.0, "Yes"],
            ["S2", 150, "NotANumber", "Yes"],  # age out of range, grade type violation
            ["S3", 22, 90.0, "InvalidCategory"],  # category violation
            ["S4", None, 75.0, "No"],  # missing age
        ]
        schema = {"age": int, "grade": float, "passed": str}
        numeric_ranges = {"age": (15, 60)}
        allowed_cats = {"passed": {"Yes", "No"}}

        report = validate_dataset(
            headers=headers,
            rows=rows,
            schema=schema,
            allowed_categories=allowed_cats,
            numeric_ranges=numeric_ranges
        )

        self.assertTrue(report.is_valid)  # no fatal structure errors
        self.assertEqual(report.missing_counts["age"], 1)
        self.assertTrue(len(report.invalid_values) >= 3)
        summary = report.summary()
        self.assertIn("CSV Validation Report", summary)
        self.assertIn("Missing Value Counts", summary)

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
