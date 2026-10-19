"""
Unit tests for data preprocessing: imputer, encoder, scaler, and leakage prevention.
"""

import unittest
from edupredict.preprocessing.imputer import SimpleImputer
from edupredict.preprocessing.encoder import LabelEncoder, OneHotEncoder
from edupredict.preprocessing.scaler import StandardScaler, MinMaxScaler


class TestPreprocessing(unittest.TestCase):

    def test_simple_imputer_mean(self):
        X = [
            [1.0, 10.0],
            [3.0, None],
            [None, 30.0],
        ]
        # Col 0 mean: (1+3)/2 = 2.0
        # Col 1 mean: (10+30)/2 = 20.0
        imputer = SimpleImputer(strategy="mean")
        X_imp = imputer.fit_transform(X)
        self.assertEqual(X_imp[1][1], 20.0)
        self.assertEqual(X_imp[2][0], 2.0)

    def test_simple_imputer_mode(self):
        X = [
            ["Cat", "Low"],
            ["Dog", "High"],
            [None, "Low"],
        ]
        imputer = SimpleImputer(strategy="mode")
        X_imp = imputer.fit_transform(X)
        self.assertIn(X_imp[2][0], ["Cat", "Dog"])
        self.assertEqual(X_imp[1][1], "High")

    def test_simple_imputer_all_missing_column(self):
        # Column completely missing
        X = [
            [None, 5.0],
            [None, 15.0],
        ]
        imputer = SimpleImputer(strategy="mean", fill_value=0.0)
        X_imp = imputer.fit_transform(X)
        self.assertEqual(X_imp[0][0], 0.0)
        self.assertEqual(X_imp[1][0], 0.0)

    def test_label_encoder(self):
        labels = ["Pass", "Fail", "Pass", "Absent"]
        le = LabelEncoder(handle_unknown="error")
        encoded = le.fit_transform(labels)
        self.assertEqual(len(encoded), 4)
        recovered = le.inverse_transform(encoded)
        self.assertEqual(recovered, labels)

        # Handle unknown error
        with self.assertRaises(ValueError):
            le.transform(["UnknownLabel"])

        # Handle unknown ignore
        le_ignore = LabelEncoder(handle_unknown="ignore")
        le_ignore.fit(labels)
        res = le_ignore.transform(["UnknownLabel"])
        self.assertEqual(res, [-1])
        self.assertEqual(le_ignore.inverse_transform([-1]), ["UNKNOWN"])

    def test_one_hot_encoder(self):
        X = [
            ["Red", "S"],
            ["Blue", "M"],
            ["Red", "L"],
        ]
        ohe = OneHotEncoder(drop_first=False)
        X_enc = ohe.fit_transform(X)
        # Red, Blue -> 2 categories; S, M, L -> 3 categories => 5 columns
        self.assertEqual(len(X_enc[0]), 5)

        # Unknown category handling: should output zeros without error when handle_unknown='ignore'
        X_test = [["Green", "XL"]]
        X_test_enc = ohe.transform(X_test)
        self.assertEqual(X_test_enc[0], [0.0, 0.0, 0.0, 0.0, 0.0])

    def test_standard_scaler(self):
        X = [
            [10.0, 100.0],
            [20.0, 200.0],
            [30.0, 300.0],
        ]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # Center should have mean ~ 0
        col0_scaled = [row[0] for row in X_scaled]
        self.assertAlmostEqual(sum(col0_scaled), 0.0, places=7)

        # Inverse transform should restore original
        X_orig = scaler.inverse_transform(X_scaled)
        for r in range(3):
            for c in range(2):
                self.assertAlmostEqual(X_orig[r][c], X[r][c], places=5)

    def test_constant_features_handling(self):
        # Column 0 is constant 5.0, Column 1 varies
        X = [
            [5.0, 10.0],
            [5.0, 20.0],
            [5.0, 30.0],
        ]
        # StandardScaler on constant column: should center to 0.0 and scale to 1.0 without dividing by zero
        std_scaler = StandardScaler()
        X_std = std_scaler.fit_transform(X)
        for row in X_std:
            self.assertAlmostEqual(row[0], 0.0)

        # MinMaxScaler on constant column: should map to min_range without dividing by zero
        mm_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        X_mm = mm_scaler.fit_transform(X)
        for row in X_mm:
            self.assertAlmostEqual(row[0], 0.0)

    def test_data_leakage_prevention(self):
        # Training set
        X_train = [[10.0], [20.0], [30.0]]  # mean = 20.0
        # Test set with different mean
        X_test = [[100.0], [200.0]]  # mean = 150.0

        scaler = StandardScaler()
        scaler.fit(X_train)
        # Ensure scaler parameters were calculated strictly on X_train
        self.assertAlmostEqual(scaler.mean_[0], 20.0)

        # Transform test set using train statistics
        X_test_scaled = scaler.transform(X_test)

        # Verify that transforming X_test did NOT modify the fitted parameters
        self.assertAlmostEqual(scaler.mean_[0], 20.0)
        # X_test_scaled should be (100 - 20) / std_train, NOT (100 - 150) / std_test
        expected_test_val = (100.0 - 20.0) / scaler.scale_[0]
        self.assertAlmostEqual(X_test_scaled[0][0], expected_test_val)


if __name__ == "__main__":
    unittest.main()
