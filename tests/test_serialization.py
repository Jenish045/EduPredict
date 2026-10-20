"""
Unit tests for model serialization and deserialization.
"""

import unittest
import os
import tempfile
from edupredict.models import (
    LinearRegression,
    LogisticRegression,
    KNNClassifier,
    KNNRegressor,
    DecisionTreeClassifier,
    GaussianNaiveBayes,
    save_model,
    load_model,
)


class TestModelSerialization(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        # Synthetic dataset: 2 features, 6 samples
        self.X_train = [
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 3.0],
            [5.0, 6.0],
            [6.0, 7.0],
            [7.0, 8.0],
        ]
        self.y_reg = [10.0, 15.0, 16.0, 28.0, 33.0, 38.0]
        self.y_clf = [0, 0, 0, 1, 1, 1]
        self.X_test = [[1.5, 2.5], [6.5, 7.5]]

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_linear_regression_serialization(self):
        lr = LinearRegression()
        lr.fit(self.X_train, self.y_reg)
        orig_preds = list(lr.predict(self.X_test))

        path = os.path.join(self.temp_dir.name, "lr.pkl")
        save_model(lr, path)
        self.assertTrue(os.path.exists(path))

        loaded_lr = load_model(path)
        loaded_preds = list(loaded_lr.predict(self.X_test))

        for p1, p2 in zip(orig_preds, loaded_preds):
            self.assertAlmostEqual(p1, p2, places=5)

    def test_logistic_regression_serialization(self):
        logr = LogisticRegression(max_iter=200, random_state=42)
        logr.fit(self.X_train, self.y_clf)
        orig_probs = list(logr.predict_proba(self.X_test))

        path = os.path.join(self.temp_dir.name, "logr.pkl")
        save_model(logr, path)
        loaded_logr = load_model(path)

        loaded_probs = list(loaded_logr.predict_proba(self.X_test))
        for p1, p2 in zip(orig_probs, loaded_probs):
            self.assertAlmostEqual(float(p1[0]), float(p2[0]), places=5)
            self.assertAlmostEqual(float(p1[1]), float(p2[1]), places=5)

    def test_knn_serialization(self):
        # Classifier
        knnc = KNNClassifier(n_neighbors=3, weights="distance")
        knnc.fit(self.X_train, self.y_clf)
        orig_preds_c = list(knnc.predict(self.X_test))

        path_c = os.path.join(self.temp_dir.name, "knnc.pkl")
        save_model(knnc, path_c)
        loaded_knnc = load_model(path_c)
        self.assertEqual(orig_preds_c, list(loaded_knnc.predict(self.X_test)))

        # Regressor
        knnr = KNNRegressor(n_neighbors=3, weights="uniform")
        knnr.fit(self.X_train, self.y_reg)
        orig_preds_r = list(knnr.predict(self.X_test))

        path_r = os.path.join(self.temp_dir.name, "knnr.pkl")
        save_model(knnr, path_r)
        loaded_knnr = load_model(path_r)
        self.assertEqual(orig_preds_r, list(loaded_knnr.predict(self.X_test)))

    def test_decision_tree_serialization(self):
        dt = DecisionTreeClassifier(max_depth=3, criterion="gini", random_state=42)
        dt.fit(self.X_train, self.y_clf)
        orig_preds = list(dt.predict(self.X_test))

        path = os.path.join(self.temp_dir.name, "dt.pkl")
        save_model(dt, path)
        loaded_dt = load_model(path)

        self.assertEqual(orig_preds, list(loaded_dt.predict(self.X_test)))

    def test_gaussian_naive_bayes_serialization(self):
        gnb = GaussianNaiveBayes()
        gnb.fit(self.X_train, self.y_clf)
        orig_preds = list(gnb.predict(self.X_test))

        path = os.path.join(self.temp_dir.name, "gnb.pkl")
        save_model(gnb, path)
        loaded_gnb = load_model(path)

        self.assertEqual(orig_preds, list(loaded_gnb.predict(self.X_test)))


if __name__ == "__main__":
    unittest.main()
