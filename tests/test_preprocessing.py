"""
Unit tests for data preprocessing: imputer, encoder, scaler.
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

    def test_label_encoder(self):
        labels = ["Pass", "Fail", "Pass", "Absent"]
        le = LabelEncoder()
        encoded = le.fit_transform(labels)
        self.assertEqual(len(encoded), 4)
        recovered = le.inverse_transform(encoded)
        self.assertEqual(recovered, labels)

        with self.assertRaises(ValueError):
            le.transform(["UnknownLabel"])

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

    def test_min_max_scaler(self):
        X = [[0.0], [5.0], [10.0]]
        scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        X_scaled = scaler.fit_transform(X)
        self.assertEqual(X_scaled, [[0.0], [0.5], [1.0]])


if __name__ == "__main__":
    unittest.main()
