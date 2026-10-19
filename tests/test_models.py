"""
Unit tests for all implemented Classical ML models.
"""

import unittest
from edupredict.models.linear_regression import LinearRegression
from edupredict.models.logistic_regression import LogisticRegression
from edupredict.models.knn import KNNClassifier, KNNRegressor
from edupredict.models.decision_tree import DecisionTreeClassifier
from edupredict.models.naive_bayes import GaussianNaiveBayes
from edupredict.metrics.regression import r2_score
from edupredict.metrics.classification import accuracy_score


class TestModels(unittest.TestCase):

    def test_unfitted_model_error(self):
        lr = LinearRegression()
        with self.assertRaises(RuntimeError):
            lr.predict([[1.0, 2.0]])

        clf = LogisticRegression()
        with self.assertRaises(RuntimeError):
            clf.predict([[1.0, 2.0]])

    def test_linear_regression_normal_equation(self):
        # Line: y = 2*x1 + 3*x2 + 5
        X = [
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 3.0],
            [4.0, 2.0],
            [5.0, 5.0],
        ]
        y = [2*r[0] + 3*r[1] + 5 for r in X]

        lr = LinearRegression(solver="normal_equation")
        lr.fit(X, y)
        preds = lr.predict(X)

        self.assertAlmostEqual(lr.bias_, 5.0, places=4)
        self.assertAlmostEqual(lr.weights_[0], 2.0, places=4)
        self.assertAlmostEqual(lr.weights_[1], 3.0, places=4)
        self.assertAlmostEqual(r2_score(y, preds), 1.0, places=4)

    def test_linear_regression_gradient_descent(self):
        # Synthetic linear data
        X = [[float(i)] for i in range(1, 21)]
        y = [3.0 * float(i) + 4.0 for i in range(1, 21)]

        # Scale X manually for clean GD convergence
        lr = LinearRegression(solver="gd", learning_rate=0.005, max_epochs=2000, random_seed=42)
        lr.fit(X, y)
        preds = lr.predict(X)

        self.assertTrue(r2_score(y, preds) > 0.98)

    def test_logistic_regression(self):
        # Linearly separable clusters in 1D
        X = [[-3.0], [-2.0], [-1.5], [1.5], [2.0], [3.0]]
        y = [0, 0, 0, 1, 1, 1]

        clf = LogisticRegression(learning_rate=0.2, max_epochs=1000, random_seed=42)
        clf.fit(X, y)
        preds = clf.predict(X)
        self.assertEqual(preds, y)

        probs = clf.predict_proba(X)
        for p in probs:
            self.assertTrue(0.0 <= p <= 1.0)

    def test_knn_classifier(self):
        X = [[1.0, 1.0], [1.5, 1.2], [5.0, 5.0], [5.5, 5.2]]
        y = ["Low", "Low", "High", "High"]

        knn = KNNClassifier(n_neighbors=1)
        knn.fit(X, y)
        test_X = [[1.1, 1.1], [5.2, 5.1]]
        preds = knn.predict(test_X)
        self.assertEqual(preds, ["Low", "High"])

    def test_knn_regressor(self):
        X = [[1.0], [2.0], [3.0], [4.0]]
        y = [10.0, 20.0, 30.0, 40.0]

        knn = KNNRegressor(n_neighbors=2)
        knn.fit(X, y)
        test_X = [[2.5]]
        # Neighbors of 2.5 are 2.0 and 3.0 -> mean of 20 and 30 is 25.0
        preds = knn.predict(test_X)
        self.assertAlmostEqual(preds[0], 25.0)

    def test_decision_tree_classifier(self):
        # XOR-like or simple split: feature 0 > 2.0 -> 1 else 0
        X = [[1.0, 5.0], [1.5, 3.0], [3.0, 1.0], [4.0, 2.0]]
        y = [0, 0, 1, 1]

        dt = DecisionTreeClassifier(criterion="gini", max_depth=3)
        dt.fit(X, y)
        preds = dt.predict(X)
        self.assertEqual(preds, y)
        self.assertEqual(len(dt.feature_importances_), 2)

    def test_gaussian_naive_bayes(self):
        X = [[1.0, 2.0], [1.2, 1.9], [8.0, 9.0], [8.2, 9.1]]
        y = [0, 0, 1, 1]

        gnb = GaussianNaiveBayes()
        gnb.fit(X, y)
        preds = gnb.predict(X)
        self.assertEqual(preds, y)

        probs = gnb.predict_proba(X)
        for row in probs:
            self.assertAlmostEqual(sum(row), 1.0)


if __name__ == "__main__":
    unittest.main()
