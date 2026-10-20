"""
Unit tests for train/test splitting, stratified K-fold, and cross-validation.
"""

import unittest
from edupredict.model_selection.split import train_test_split, KFold, StratifiedKFold
from edupredict.model_selection.validation import cross_val_score
from edupredict.models import LogisticRegression


class TestModelSelection(unittest.TestCase):

    def test_train_test_split_determinism(self):
        X = [[float(i), float(i * 2)] for i in range(100)]
        y = [i % 2 for i in range(100)]

        # Two splits with same seed must be identical
        X_tr1, X_te1, y_tr1, y_te1 = train_test_split(X, y, test_size=0.2, random_seed=42)
        X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X, y, test_size=0.2, random_seed=42)

        self.assertEqual(X_tr1, X_tr2)
        self.assertEqual(X_te1, X_te2)
        self.assertEqual(y_tr1, y_tr2)
        self.assertEqual(y_te1, y_te2)

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

    def test_stratified_kfold(self):
        # 60 samples: 40 class 0, 20 class 1
        X = [[float(i)] for i in range(60)]
        y = [0] * 40 + [1] * 20

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_seed=42)
        splits = skf.split(X, y)
        self.assertEqual(len(splits), 5)

        all_test_indices = []
        for train_idx, test_idx in splits:
            # Each fold test set: 12 samples (8 of class 0, 4 of class 1)
            self.assertEqual(len(test_idx), 12)
            test_y = [y[i] for i in test_idx]
            self.assertEqual(test_y.count(0), 8)
            self.assertEqual(test_y.count(1), 4)
            all_test_indices.extend(test_idx)

        # All samples must be tested exactly once
        self.assertEqual(sorted(all_test_indices), list(range(60)))

    def test_cross_val_score(self):
        # Linearly separable data
        X = [[-2.0], [-1.0], [1.0], [2.0]] * 5
        y = [0, 0, 1, 1] * 5

        model = LogisticRegression(max_iter=200, random_state=42)
        scores = cross_val_score(model, X, y, cv=3, scoring="accuracy")
        self.assertEqual(len(scores), 3)
        for s in scores:
            self.assertTrue(0.0 <= s <= 1.0)


if __name__ == "__main__":
    unittest.main()
