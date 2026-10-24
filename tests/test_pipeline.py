"""
Unit and integration tests for DatasetSchema, MLPipeline, ExperimentSuite,
dynamic inference, and leak-free preprocessing execution.
"""

import unittest
import random
from typing import List, Dict, Any

from edupredict.data.schema import infer_schema, DatasetSchema
from edupredict.data.dataset import Dataset
from edupredict.pipeline.pipeline import MLPipeline, ExperimentSuite
from edupredict.models.registry import get_model, list_models, REGRESSION_MODELS, CLASSIFICATION_MODELS


class TestPipeline(unittest.TestCase):

    def setUp(self):
        # Generate arbitrary non-educational synthetic dataset (e.g. Housing domain)
        random.seed(42)
        self.n_samples = 120
        self.headers = ["house_id", "area_sqft", "bedrooms", "neighborhood", "has_garage", "price", "is_luxury"]
        self.rows = []

        neighborhoods = ["North", "South", "East", "West"]
        for i in range(self.n_samples):
            h_id = f"H_{i:04d}"
            area = 800.0 + (i % 30) * 50.0 + random.uniform(-20, 20)
            beds = 1 + (i % 4)
            # Inject intermittent missing values
            if i % 15 == 0:
                area = None
            neigh = neighborhoods[i % len(neighborhoods)]
            garage = "Yes" if i % 2 == 0 else "No"
            price = 100000.0 + (area or 1200.0) * 150.0 + beds * 20000.0 + (50000.0 if garage == "Yes" else 0.0)
            is_lux = 1 if price > 300000.0 else 0

            self.rows.append([h_id, area, beds, neigh, garage, price, is_lux])

        self.dataset = Dataset(headers=self.headers, data=self.rows)

    def test_schema_inference_automatic(self):
        """Test automatic inference of numerical features, categorical features, ID columns, and task."""
        schema = infer_schema(self.headers, self.rows, target_column="price")
        self.assertEqual(schema.target_column, "price")
        self.assertEqual(schema.task_type, "regression")
        self.assertIn("house_id", schema.ignored_columns)
        self.assertIn("area_sqft", schema.numerical_features)
        self.assertIn("bedrooms", schema.numerical_features)
        self.assertIn("neighborhood", schema.categorical_features)
        self.assertIn("has_garage", schema.categorical_features)
        self.assertNotIn("price", schema.feature_names)

    def test_schema_inference_classification(self):
        """Test schema inference for binary classification target."""
        schema = infer_schema(self.headers, self.rows, target_column="is_luxury", ignored_columns=["house_id", "price"])
        self.assertEqual(schema.target_column, "is_luxury")
        self.assertEqual(schema.task_type, "binary_classification")
        self.assertIn("is_luxury", schema.target_column)
        self.assertNotIn("price", schema.feature_names)

    def test_ml_pipeline_regression_end_to_end(self):
        """Test complete leak-free regression pipeline on arbitrary synthetic data."""
        schema = infer_schema(self.headers, self.rows, target_column="price", ignored_columns=["house_id", "is_luxury"])
        raw_X, y = self.dataset.split_features_target(feature_cols=schema.feature_names, target_col="price")
        y = [float(v) for v in y]

        pipe = MLPipeline(schema=schema, model_name="Linear Regression", random_seed=42)
        pipe.fit(raw_X, y, test_size=0.25)

        self.assertTrue(pipe.is_fitted)
        self.assertIn("r2", pipe.test_metrics)
        self.assertIn("mse", pipe.test_metrics)
        self.assertIn("mae", pipe.test_metrics)
        self.assertGreater(pipe.test_metrics["r2"], 0.70)
        self.assertEqual(len(pipe.y_test_pred), len(pipe.y_test))

    def test_ml_pipeline_classification_end_to_end(self):
        """Test complete leak-free classification pipeline on arbitrary synthetic data."""
        schema = infer_schema(self.headers, self.rows, target_column="is_luxury", ignored_columns=["house_id", "price"])
        raw_X, y = self.dataset.split_features_target(feature_cols=schema.feature_names, target_col="is_luxury")
        y = [int(v) for v in y]

        pipe = MLPipeline(schema=schema, model_name="KNN Classifier", random_seed=42)
        pipe.fit(raw_X, y, test_size=0.25)

        self.assertTrue(pipe.is_fitted)
        self.assertIn("accuracy", pipe.test_metrics)
        self.assertIn("precision", pipe.test_metrics)
        self.assertIn("recall", pipe.test_metrics)
        self.assertIn("f1", pipe.test_metrics)
        self.assertGreaterEqual(pipe.test_metrics["accuracy"], 0.70)
        self.assertIsNotNone(pipe.test_metrics.get("roc_auc"))

    def test_single_record_prediction_dynamic(self):
        """Test predicting single records passed as dictionaries."""
        schema = infer_schema(self.headers, self.rows, target_column="is_luxury", ignored_columns=["house_id", "price"])
        raw_X, y = self.dataset.split_features_target(feature_cols=schema.feature_names, target_col="is_luxury")
        y = [int(v) for v in y]

        pipe = MLPipeline(schema=schema, model_name="KNN Classifier", random_seed=42)
        pipe.fit(raw_X, y, test_size=0.25)

        sample_record = {
            "area_sqft": 1500.0,
            "bedrooms": 3,
            "neighborhood": "North",
            "has_garage": "Yes",
        }
        pred = pipe.predict_single(sample_record)
        self.assertIn(pred, [0, 1])

        proba = pipe.predict_single_proba(sample_record)
        self.assertIsNotNone(proba)
        self.assertAlmostEqual(sum(proba.values()), 1.0, places=4)

    def test_unseen_category_handling_in_pipeline(self):
        """Test that unseen categories in inference data are safely encoded without errors."""
        schema = infer_schema(self.headers, self.rows, target_column="price", ignored_columns=["house_id", "is_luxury"])
        raw_X, y = self.dataset.split_features_target(feature_cols=schema.feature_names, target_col="price")
        y = [float(v) for v in y]

        pipe = MLPipeline(schema=schema, model_name="Linear Regression", random_seed=42)
        pipe.fit(raw_X, y, test_size=0.25)

        unseen_record = {
            "area_sqft": 1200.0,
            "bedrooms": 2,
            "neighborhood": "NeverBeforeSeenTown",
            "has_garage": "UnknownOption",
        }
        pred = pipe.predict_single(unseen_record)
        self.assertIsInstance(pred, float)

    def test_cross_validation_leak_free(self):
        """Test k-fold cross validation with fold-isolated preprocessing."""
        schema = infer_schema(self.headers, self.rows, target_column="price", ignored_columns=["house_id", "is_luxury"])
        raw_X, y = self.dataset.split_features_target(feature_cols=schema.feature_names, target_col="price")
        y = [float(v) for v in y]

        pipe = MLPipeline(schema=schema, model_name="Linear Regression", random_seed=42)
        cv_res = pipe.cross_validate(raw_X, y, k=3)

        self.assertIn("cv_r2_mean", cv_res)
        self.assertIn("cv_mse_mean", cv_res)
        self.assertGreater(cv_res["cv_r2_mean"], 0.70)

    def test_experiment_suite_run_all(self):
        """Test ExperimentSuite benchmarks all available models for a task."""
        schema = infer_schema(self.headers, self.rows, target_column="price", ignored_columns=["house_id", "is_luxury"])
        raw_X, y = self.dataset.split_features_target(feature_cols=schema.feature_names, target_col="price")
        y = [float(v) for v in y]

        suite = ExperimentSuite(schema, raw_X, y, random_seed=42)
        report = suite.run_all()

        self.assertEqual(len(report.records), len(REGRESSION_MODELS))
        for r in report.records:
            self.assertIn("Model", r)
            self.assertIn("R2", r)
            self.assertIn("MSE", r)

    def test_model_registry_functions(self):
        """Test model registry retrieval, listing, and validation."""
        reg_models = list_models("regression")
        self.assertIn("Linear Regression", reg_models)

        clf_models = list_models("classification")
        self.assertIn("KNN Classifier", clf_models)
        self.assertIn("Decision Tree Classifier", clf_models)

        with self.assertRaises(ValueError):
            get_model("NonExistentModel", task="regression")

    def test_three_way_split_and_data_isolation(self):
        """Verify strict pairwise disjointness and leakage isolation in 3-way partition."""
        from edupredict.model_selection.split import train_test_split_indices
        n_samples = 395
        # 1. Split into Dev (80% = 316) and Holdout Test (20% = 79)
        dev_idx, test_idx = train_test_split_indices(n_samples, test_size=0.20, random_seed=42)
        self.assertEqual(len(dev_idx), 316)
        self.assertEqual(len(test_idx), 79)

        # 2. Split Dev into Train-Sub (80% = 253) and Validation (20% = 63)
        train_sub_subidx, val_subidx = train_test_split_indices(len(dev_idx), test_size=0.20, random_seed=42)
        train_sub_idx = [dev_idx[i] for i in train_sub_subidx]
        val_idx = [dev_idx[i] for i in val_subidx]

        self.assertEqual(len(train_sub_idx), 253)
        self.assertEqual(len(val_idx), 63)

        # Pairwise disjointness
        set_tr = set(train_sub_idx)
        set_va = set(val_idx)
        set_te = set(test_idx)

        self.assertEqual(set_tr.intersection(set_va), set())
        self.assertEqual(set_tr.intersection(set_te), set())
        self.assertEqual(set_va.intersection(set_te), set())

        # Completeness
        union_all = set_tr.union(set_va).union(set_te)
        self.assertEqual(union_all, set(range(n_samples)))


if __name__ == "__main__":
    unittest.main()
