"""
Unit tests for train/test splitting and cross-validation.
"""

import unittest
from edupredict.model_selection.split import train_test_split, KFold
from edupredict.model_selection.validation import cross_val_score
from edupredict.models.logistic_regression import LogisticRegression


class TestModelSelection(unittest.TestCase):

    def test_train_test_split(self):
        X = [[i] for i in range(100)]
        y = [i % 2 for i in range(100)]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_seed=42
        )
        self.assertEqual(len(X_train), 75)
        self.assertEqual(len(X_test), 25)
        self.assertEqual(len(y_train), 75)
        self.assertEqual(len(y_test), 25)

    def test_stratified_split(self):
        # 80 class 0, 20 class 1
        X = [[i] for i in range(100)]
        y = [0] * 80 + [1] * 20

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_seed=42, stratify=True
        )
        # Test set (20 samples) should have 16 of class 0 and 4 of class 1
        self.assertEqual(y_test.count(0), 16)
        self.assertEqual(y_test.count(1), 4)

    def test_kfold_splits(self):
        X = list(range(10))
        kf = KFold(n_splits=5, shuffle=False)
        splits = kf.split(X)
        self.assertEqual(len(splits), 5)

        # Check disjointness and coverage of test indices
        all_test_indices = []
        for train_idx, test_idx in splits:
            self.assertEqual(len(test_idx), 2)
            self.assertEqual(len(train_idx), 8)
            all_test_indices.extend(test_idx)

        self.assertEqual(sorted(all_test_indices), list(range(10)))

    def test_cross_val_score(self):
        # Linearly separable data
        X = [[-2.0], [-1.0], [1.0], [2.0]] * 5
        y = [0, 0, 1, 1] * 5

        model = LogisticRegression(learning_rate=0.2, max_epochs=200, random_seed=42)
        scores = cross_val_score(model, X, y, cv=3, scoring="accuracy")
        self.assertEqual(len(scores), 3)
        for s in scores:
            self.assertTrue(0.0 <= s <= 1.0)


if __name__ == "__main__":
    unittest.main()
