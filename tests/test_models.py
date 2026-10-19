"""
Unit tests for all implemented Classical ML models:
Linear Regression, Logistic Regression, KNN, and Decision Tree.
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

        knn = KNNClassifier()
        with self.assertRaises(RuntimeError):
            knn.predict([[1.0, 2.0]])

        dt = DecisionTreeClassifier()
        with self.assertRaises(RuntimeError):
            dt.predict([[1.0, 2.0]])

    def test_invalid_feature_dimensions(self):
        lr = LinearRegression(solver="gd")
        lr.fit([[1.0, 2.0], [3.0, 4.0]], [10.0, 20.0])
        # Predict with 3 features instead of 2
        with self.assertRaises(ValueError):
            lr.predict([[1.0, 2.0, 3.0]])

    def test_linear_regression_gradient_descent_single_and_multi(self):
        # Single feature: y = 2.5 * x + 1.0
        X_single = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        y_single = [2.5 * r[0] + 1.0 for r in X_single]

        lr_single = LinearRegression(solver="gd", learning_rate=0.05, max_epochs=1000, random_seed=42)
        lr_single.fit(X_single, y_single)
        preds = lr_single.predict(X_single)
        self.assertTrue(r2_score(y_single, preds) > 0.99)
        self.assertAlmostEqual(lr_single.weights_[0], 2.5, places=1)
        self.assertAlmostEqual(lr_single.bias_, 1.0, places=1)

        # Without intercept: y = 3.0 * x
        X_no_int = [[1.0], [2.0], [3.0]]
        y_no_int = [3.0 * r[0] for r in X_no_int]
        lr_no_int = LinearRegression(solver="gd", fit_intercept=False, learning_rate=0.05, max_epochs=500)
        lr_no_int.fit(X_no_int, y_no_int)
        self.assertEqual(lr_no_int.bias_, 0.0)
        self.assertAlmostEqual(lr_no_int.weights_[0], 3.0, places=2)

    def test_linear_regression_normal_equation(self):
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

    def test_logistic_regression_probabilities_and_threshold(self):
        # Linearly separable clusters in 1D
        X = [[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]]
        y = [0, 0, 0, 1, 1, 1]

        clf = LogisticRegression(learning_rate=0.2, max_epochs=800, threshold=0.5, random_seed=42)
        clf.fit(X, y)
        preds = clf.predict(X)
        self.assertEqual(preds, y)

        probs = clf.predict_proba(X)
        for p in probs:
            self.assertTrue(0.0 <= p <= 1.0)
        # Negative cluster probabilities should be low, positive high
        self.assertTrue(probs[0] < 0.2)
        self.assertTrue(probs[-1] > 0.8)

        # Test threshold variation
        preds_strict = clf.predict(X, threshold=0.99)
        self.assertTrue(sum(preds_strict) <= sum(preds))

    def test_knn_deterministic_tie_breaking(self):
        # 4 training points forming a symmetric square:
        # (-1, 0) -> Class 'A'
        # (1, 0)  -> Class 'A'
        # (0, -1) -> Class 'B'
        # (0, 1)  -> Class 'B'
        # Query at (0, 0) with k=4 yields an exact 2-2 tie between 'A' and 'B'.
        X_train = [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]
        y_train = ["A", "A", "B", "B"]

        knn = KNNClassifier(n_neighbors=4, weights="uniform")
        knn.fit(X_train, y_train)

        # Predict query at center
        pred = knn.predict([[0.0, 0.0]])
        # Tie-breaking rules ensure deterministic pick (sorted string label 'A' < 'B')
        self.assertEqual(pred, ["A"])

        # Predict probabilities
        probs = knn.predict_proba([[0.0, 0.0]])
        self.assertEqual(probs[0]["A"], 0.5)
        self.assertEqual(probs[0]["B"], 0.5)

    def test_knn_regressor(self):
        X = [[1.0], [2.0], [3.0], [4.0]]
        y = [10.0, 20.0, 30.0, 40.0]

        knn = KNNRegressor(n_neighbors=2)
        knn.fit(X, y)
        test_X = [[2.5]]
        # Neighbors of 2.5 are 2.0 and 3.0 -> mean of 20 and 30 is 25.0
        preds = knn.predict(test_X)
        self.assertAlmostEqual(preds[0], 25.0)

    def test_decision_tree_splitting_and_probabilities(self):
        # Feature 0 < 2.0 -> Class 0; Feature 0 >= 2.0 -> Class 1
        X = [[1.0, 5.0], [1.5, 3.0], [3.0, 1.0], [4.0, 2.0]]
        y = [0, 0, 1, 1]

        dt = DecisionTreeClassifier(criterion="gini", max_depth=3)
        dt.fit(X, y)
        preds = dt.predict(X)
        self.assertEqual(preds, y)

        # Probabilities
        probs = dt.predict_proba(X)
        self.assertEqual(probs[0][0], 1.0)
        self.assertEqual(probs[2][1], 1.0)

        # Out of sample prediction
        test_pred = dt.predict([[0.5, 10.0], [5.0, 0.0]])
        self.assertEqual(test_pred, [0, 1])

    def test_gaussian_naive_bayes(self):
        X = [[1.0, 2.0], [1.2, 1.9], [8.0, 9.0], [8.2, 9.1]]
        y = [0, 0, 1, 1]

        gnb = GaussianNaiveBayes()
        gnb.fit(X, y)
        preds = gnb.predict(X)
        self.assertEqual(preds, y)


if __name__ == "__main__":
    unittest.main()
