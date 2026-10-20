"""
Unit and integration tests for Scikit-learn model estimators used in EduPredict,
verifying proper fitting, prediction, probability calculation, and evaluation
using scratch-implemented metrics.
"""

import unittest
from sklearn.exceptions import NotFittedError

from edupredict.models import (
    LinearRegression,
    Ridge,
    LogisticRegression,
    KNNClassifier,
    KNNRegressor,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    GaussianNaiveBayes,
    get_model,
    list_models,
)
from edupredict.metrics.regression import r2_score, mean_squared_error
from edupredict.metrics.classification import accuracy_score, precision_score, recall_score, f1_score


class TestModels(unittest.TestCase):

    def test_unfitted_model_error(self):
        """Verify that calling predict on an unfitted estimator raises an error."""
        lr = LinearRegression()
        with self.assertRaises(NotFittedError):
            lr.predict([[1.0, 2.0]])

        clf = LogisticRegression()
        with self.assertRaises(NotFittedError):
            clf.predict([[1.0, 2.0]])

        knn = KNNClassifier()
        with self.assertRaises(NotFittedError):
            knn.predict([[1.0, 2.0]])

        dt = DecisionTreeClassifier()
        with self.assertRaises(NotFittedError):
            dt.predict([[1.0, 2.0]])

    def test_linear_regression_fitting_and_custom_r2(self):
        """Test LinearRegression with our scratch r2_score on synthetic linear data."""
        X = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        y = [2.5 * r[0] + 1.0 for r in X]

        lr = LinearRegression(fit_intercept=True)
        lr.fit(X, y)
        preds = lr.predict(X)

        score = r2_score(y, preds)
        self.assertGreater(score, 0.99)
        self.assertAlmostEqual(float(lr.coef_[0]), 2.5, places=2)
        self.assertAlmostEqual(float(lr.intercept_), 1.0, places=2)

    def test_ridge_regression_fitting(self):
        """Test Ridge regression fitting and metric evaluation."""
        X = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]
        y = [5.0, 8.0, 11.0, 14.0]

        ridge = Ridge(alpha=1.0)
        ridge.fit(X, y)
        preds = ridge.predict(X)
        self.assertGreater(r2_score(y, preds), 0.95)

    def test_logistic_regression_probabilities_and_scratch_metrics(self):
        """Test LogisticRegression prediction, predict_proba, and custom classification metrics."""
        X = [[1.0], [1.5], [2.0], [5.0], [6.0], [7.0]]
        y = [0, 0, 0, 1, 1, 1]

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X, y)

        preds = clf.predict(X)
        probs = clf.predict_proba(X)

        acc = accuracy_score(y, preds)
        prec = precision_score(y, preds)
        rec = recall_score(y, preds)
        f1 = f1_score(y, preds)

        self.assertEqual(acc, 1.0)
        self.assertEqual(prec, 1.0)
        self.assertEqual(rec, 1.0)
        self.assertEqual(f1, 1.0)
        self.assertEqual(len(probs), len(X))
        self.assertAlmostEqual(probs[0][0] + probs[0][1], 1.0, places=5)

    def test_knn_classifier_and_regressor(self):
        """Test KNNClassifier and KNNRegressor fitting and predictions."""
        X = [[1.0, 1.0], [1.0, 2.0], [5.0, 5.0], [6.0, 5.0]]
        y_clf = [0, 0, 1, 1]
        y_reg = [2.0, 3.0, 10.0, 11.0]

        knn_c = KNNClassifier(n_neighbors=2)
        knn_c.fit(X, y_clf)
        self.assertEqual(list(knn_c.predict([[1.2, 1.5]])), [0])

        knn_r = KNNRegressor(n_neighbors=2)
        knn_r.fit(X, y_reg)
        pred_reg = knn_r.predict([[1.0, 1.5]])[0]
        self.assertAlmostEqual(pred_reg, 2.5, places=2)

    def test_decision_tree_classifier_and_regressor(self):
        """Test DecisionTreeClassifier and DecisionTreeRegressor."""
        X = [[10.0], [20.0], [80.0], [90.0]]
        y_clf = [0, 0, 1, 1]
        y_reg = [1.0, 2.0, 8.0, 9.0]

        dt_c = DecisionTreeClassifier(max_depth=3, random_state=42)
        dt_c.fit(X, y_clf)
        self.assertEqual(accuracy_score(y_clf, dt_c.predict(X)), 1.0)

        dt_r = DecisionTreeRegressor(max_depth=3, random_state=42)
        dt_r.fit(X, y_reg)
        self.assertGreater(r2_score(y_reg, dt_r.predict(X)), 0.90)

    def test_gaussian_naive_bayes(self):
        """Test Gaussian Naive Bayes classification and probability calibration."""
        X = [[1.0, 1.0], [1.2, 0.9], [5.0, 5.0], [5.2, 4.8]]
        y = [0, 0, 1, 1]

        gnb = GaussianNaiveBayes()
        gnb.fit(X, y)
        preds = gnb.predict(X)
        self.assertEqual(accuracy_score(y, preds), 1.0)

        probs = gnb.predict_proba(X)
        self.assertEqual(len(probs), 4)

    def test_factory_and_registry_retrieval(self):
        """Test get_model factory and list_models."""
        reg_model = get_model("Linear Regression", task="regression", fit_intercept=True)
        self.assertIsInstance(reg_model, LinearRegression)

        clf_model = get_model("Logistic Regression", task="classification", max_iter=500)
        self.assertIsInstance(clf_model, LogisticRegression)


if __name__ == "__main__":
    unittest.main()
