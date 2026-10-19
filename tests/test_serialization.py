"""
Unit tests for model serialization and deserialization from scratch.
"""

import unittest
import os
import tempfile
from edupredict.models.linear_regression import LinearRegression
from edupredict.models.logistic_regression import LogisticRegression
from edupredict.models.knn import KNNClassifier, KNNRegressor
from edupredict.models.decision_tree import DecisionTreeClassifier
from edupredict.models.naive_bayes import GaussianNaiveBayes
from edupredict.models.serialization import save_model, load_model


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
        lr = LinearRegression(solver="gd", max_epochs=200, learning_rate=0.01)
        lr.fit(self.X_train, self.y_reg)
        orig_preds = lr.predict(self.X_test)

        path = os.path.join(self.temp_dir.name, "lr.json")
        save_model(lr, path)
        self.assertTrue(os.path.exists(path))

        loaded_lr = load_model(path)
        self.assertTrue(loaded_lr.is_fitted)
        loaded_preds = loaded_lr.predict(self.X_test)

        for p1, p2 in zip(orig_preds, loaded_preds):
            self.assertAlmostEqual(p1, p2, places=5)

    def test_logistic_regression_serialization(self):
        logr = LogisticRegression(max_epochs=200, learning_rate=0.05)
        logr.fit(self.X_train, self.y_clf)
        orig_probs = logr.predict_proba(self.X_test)

        path = os.path.join(self.temp_dir.name, "logr.json")
        save_model(logr, path)
        loaded_logr = load_model(path)

        loaded_probs = loaded_logr.predict_proba(self.X_test)
        for p1, p2 in zip(orig_probs, loaded_probs):
            self.assertAlmostEqual(p1, p2, places=5)

    def test_knn_serialization(self):
        # Classifier
        knnc = KNNClassifier(n_neighbors=3, weights="distance")
        knnc.fit(self.X_train, self.y_clf)
        orig_preds_c = knnc.predict(self.X_test)

        path_c = os.path.join(self.temp_dir.name, "knnc.json")
        save_model(knnc, path_c)
        loaded_knnc = load_model(path_c)
        self.assertEqual(orig_preds_c, loaded_knnc.predict(self.X_test))

        # Regressor
        knnr = KNNRegressor(n_neighbors=3, weights="uniform")
        knnr.fit(self.X_train, self.y_reg)
        orig_preds_r = knnr.predict(self.X_test)

        path_r = os.path.join(self.temp_dir.name, "knnr.json")
        save_model(knnr, path_r)
        loaded_knnr = load_model(path_r)
        self.assertEqual(orig_preds_r, loaded_knnr.predict(self.X_test))

    def test_decision_tree_serialization(self):
        dt = DecisionTreeClassifier(max_depth=3, criterion="gini")
        dt.fit(self.X_train, self.y_clf)
        orig_preds = dt.predict(self.X_test)

        path = os.path.join(self.temp_dir.name, "dt.json")
        save_model(dt, path)
        loaded_dt = load_model(path)

        self.assertEqual(orig_preds, loaded_dt.predict(self.X_test))
        self.assertEqual(dt.feature_importances_, loaded_dt.feature_importances_)

    def test_gaussian_naive_bayes_serialization(self):
        gnb = GaussianNaiveBayes()
        gnb.fit(self.X_train, self.y_clf)
        orig_preds = gnb.predict(self.X_test)

        path = os.path.join(self.temp_dir.name, "gnb.json")
        save_model(gnb, path)
        loaded_gnb = load_model(path)

        self.assertEqual(orig_preds, loaded_gnb.predict(self.X_test))


if __name__ == "__main__":
    unittest.main()
