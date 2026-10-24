"""
Unit tests for model serialization and deserialization of EduPredict custom models.
"""

import unittest
import os
import tempfile
from edupredict.models import (
    LinearRegression,
    KNNClassifier,
    DecisionTreeClassifier,
    save_model,
    load_model,
)


class TestModelSerialization(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
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

    def test_linear_regression_serialization_pickle(self):
        lr = LinearRegression(solver="normal_equation")
        lr.fit(self.X_train, self.y_reg)
        orig_preds = list(lr.predict(self.X_test))

        path = os.path.join(self.temp_dir.name, "lr.pkl")
        save_model(lr, path)
        self.assertTrue(os.path.exists(path))

        loaded_lr = load_model(path)
        loaded_preds = list(loaded_lr.predict(self.X_test))

        for p1, p2 in zip(orig_preds, loaded_preds):
            self.assertAlmostEqual(p1, p2, places=5)

    def test_linear_regression_serialization_json(self):
        lr = LinearRegression(solver="normal_equation")
        lr.fit(self.X_train, self.y_reg)
        orig_preds = list(lr.predict(self.X_test))

        path = os.path.join(self.temp_dir.name, "lr.json")
        save_model(lr, path)
        self.assertTrue(os.path.exists(path))

        loaded_lr = load_model(path)
        loaded_preds = list(loaded_lr.predict(self.X_test))

        for p1, p2 in zip(orig_preds, loaded_preds):
            self.assertAlmostEqual(p1, p2, places=5)

    def test_knn_serialization_pickle_and_json(self):
        knn = KNNClassifier(n_neighbors=3, weights="distance")
        knn.fit(self.X_train, self.y_clf)
        orig_preds = list(knn.predict(self.X_test))

        # Pickle
        path_pkl = os.path.join(self.temp_dir.name, "knn.pkl")
        save_model(knn, path_pkl)
        loaded_pkl = load_model(path_pkl)
        self.assertEqual(orig_preds, list(loaded_pkl.predict(self.X_test)))

        # JSON
        path_json = os.path.join(self.temp_dir.name, "knn.json")
        save_model(knn, path_json)
        loaded_json = load_model(path_json)
        self.assertEqual(orig_preds, list(loaded_json.predict(self.X_test)))

    def test_decision_tree_serialization_pickle_and_json(self):
        dt = DecisionTreeClassifier(max_depth=3, random_seed=42)
        dt.fit(self.X_train, self.y_clf)
        orig_preds = list(dt.predict(self.X_test))

        # Pickle
        path_pkl = os.path.join(self.temp_dir.name, "dt.pkl")
        save_model(dt, path_pkl)
        loaded_pkl = load_model(path_pkl)
        self.assertEqual(orig_preds, list(loaded_pkl.predict(self.X_test)))

        # JSON
        path_json = os.path.join(self.temp_dir.name, "dt.json")
        save_model(dt, path_json)
        loaded_json = load_model(path_json)
        self.assertEqual(orig_preds, list(loaded_json.predict(self.X_test)))


if __name__ == "__main__":
    unittest.main()
