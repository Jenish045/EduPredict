"""
Programmatic verification of all core logic, data flows, and prediction paths in app.py.
Exercises all 6 tab workflows without requiring a GUI browser or Streamlit AppTest.
"""

import unittest
import os
import sys

from edupredict.data.loader import load_csv, validate_dataset
from edupredict.data.dataset import Dataset
from edupredict.data.schema import infer_schema
from edupredict.pipeline.pipeline import MLPipeline
from edupredict.data.uci_metadata import (
    UCI_FEATURE_CONFIG,
    EDUCATION_LEVEL_LABELS,
    TRAVEL_TIME_LABELS,
    STUDY_TIME_LABELS,
    ADDRESS_LABELS,
    FAMILY_SIZE_LABELS,
    COHABITATION_LABELS,
    get_column_display_name,
    format_preview_records,
)


class TestAppLogic(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.uci_path = os.path.join(os.path.dirname(__file__), "..", "data", "student-mat.csv")
        headers, rows = load_csv(cls.uci_path)
        g3_idx = headers.index("G3")
        new_headers = list(headers) + ["academic_risk"]
        new_rows = []
        for row in rows:
            g3_val = row[g3_idx]
            risk = 1 if (g3_val is not None and float(g3_val) < 10.0) else 0
            new_rows.append(list(row) + [risk])
        cls.dataset = Dataset(headers=new_headers, data=new_rows)

    def test_tab1_dataset_overview_logic(self):
        """Verify Tab 1: Dataset loading, shape, missing values, and validation report."""
        self.assertEqual(self.dataset.shape, (395, 34))
        missing_dict = self.dataset.count_missing()
        self.assertEqual(sum(missing_dict.values()), 0)
        report = validate_dataset(self.dataset.headers, self.dataset.rows)
        self.assertTrue(report.is_valid)

    def test_tab1_preview_friendly_labels_and_dataset_immutability(self):
        """Verify Tab 1: Preview table renders friendly column labels while dataset/schema remain unchanged."""
        table_headers = self.dataset.headers
        table_rows = self.dataset.rows[:8]
        preview_records = format_preview_records(table_headers, table_rows)

        # 1. Preview row count and column count
        self.assertEqual(len(preview_records), 8)
        self.assertEqual(len(preview_records[0]), 34)

        # 2. Check friendly display labels in preview keys
        preview_columns = list(preview_records[0].keys())
        expected_friendly = [
            ("school", "School"),
            ("sex", "Sex"),
            ("age", "Age (years)"),
            ("address", "Residential Area"),
            ("famsize", "Family Size"),
            ("Pstatus", "Parents' Cohabitation Status"),
            ("Medu", "Mother's Education Level"),
            ("Fedu", "Father's Education Level"),
            ("Mjob", "Mother's Occupation"),
            ("Fjob", "Father's Occupation"),
            ("traveltime", "Travel Time to School"),
            ("studytime", "Weekly Study Time"),
            ("failures", "Previous Class Failures"),
            ("absences", "School Absences"),
        ]
        for raw_col, friendly_col in expected_friendly:
            self.assertIn(friendly_col, preview_columns, f"Friendly column '{friendly_col}' missing from preview")
            self.assertNotIn(raw_col, preview_columns, f"Raw column '{raw_col}' was unexpectedly found in preview")
            # Verify it directly matches UCI_FEATURE_CONFIG
            self.assertEqual(get_column_display_name(raw_col), friendly_col)

        # 3. Preserve column order and exact values
        expected_headers = [get_column_display_name(h) for h in table_headers]
        self.assertEqual(preview_columns, expected_headers)
        for r_idx, row in enumerate(table_rows):
            for c_idx, h in enumerate(expected_headers):
                self.assertEqual(preview_records[r_idx][h], row[c_idx])

        # 4. Underlying dataset headers and schema are strictly immutable and unchanged
        self.assertEqual(self.dataset.headers[0], "school")
        self.assertEqual(self.dataset.headers[2], "age")
        self.assertIn("Medu", self.dataset.headers)
        self.assertIn("absences", self.dataset.headers)
        self.assertIn("G3", self.dataset.headers)

        schema = infer_schema(
            self.dataset.headers,
            self.dataset.rows,
            target_column="G3",
            task_type="regression",
            ignored_columns=["G1", "G2", "academic_risk"],
        )
        self.assertIn("school", schema.feature_names)
        self.assertIn("age", schema.feature_names)
        self.assertIn("absences", schema.feature_names)
        self.assertEqual(len(schema.feature_names), 30)

    def test_tab2_regression_studio_logic(self):
        """Verify Tab 2: Linear Regression training, metrics, and single-record inference."""
        schema = infer_schema(
            self.dataset.headers,
            self.dataset.rows,
            target_column="G3",
            task_type="regression",
            ignored_columns=["G1", "G2", "academic_risk"],
        )
        self.assertNotIn("G1", schema.feature_names)
        self.assertNotIn("G2", schema.feature_names)

        raw_X, y = self.dataset.split_features_target(schema.feature_names, "G3")
        y = [float(v) for v in y]

        # Normal equation
        pipe_ne = MLPipeline(
            schema=schema,
            model_name="Linear Regression",
            model_params={"solver": "normal_equation", "l2_reg": 10.0},
            random_seed=42,
        )
        pipe_ne.fit(raw_X, y, test_size=0.20)
        self.assertIn("r2", pipe_ne.test_metrics)
        self.assertIn("mse", pipe_ne.test_metrics)

        # Gradient descent
        pipe_gd = MLPipeline(
            schema=schema,
            model_name="Linear Regression",
            model_params={"solver": "gradient_descent", "learning_rate": 0.01, "max_epochs": 100},
            random_seed=42,
        )
        pipe_gd.fit(raw_X, y, test_size=0.20)
        self.assertTrue(len(pipe_gd.estimator.loss_history_) > 0)

        # Single record prediction
        sample_dict = {f: raw_X[0][i] for i, f in enumerate(schema.feature_names)}
        pred_val = pipe_ne.predict_single(sample_dict)
        self.assertIsInstance(pred_val, float)
        self.assertGreaterEqual(pred_val, -5.0)
        self.assertLessEqual(pred_val, 25.0)

    def test_tab3_classification_studio_logic(self):
        """Verify Tab 3: KNN & Decision Tree training, metrics, confusion matrix, single inference."""
        schema = infer_schema(
            self.dataset.headers,
            self.dataset.rows,
            target_column="academic_risk",
            task_type="binary_classification",
            ignored_columns=["G1", "G2", "G3"],
        )
        self.assertNotIn("G1", schema.feature_names)
        self.assertNotIn("G2", schema.feature_names)
        self.assertNotIn("G3", schema.feature_names)

        raw_X, y = self.dataset.split_features_target(schema.feature_names, "academic_risk")
        y = [int(v) for v in y]

        # KNN
        pipe_knn = MLPipeline(
            schema=schema,
            model_name="KNN Classifier",
            model_params={"n_neighbors": 5, "metric": "euclidean", "weights": "uniform"},
            random_seed=42,
        )
        pipe_knn.fit(raw_X, y, test_size=0.20)
        self.assertIn("accuracy", pipe_knn.test_metrics)
        self.assertIn("confusion_matrix", pipe_knn.test_metrics)

        # Decision Tree
        pipe_dt = MLPipeline(
            schema=schema,
            model_name="Decision Tree Classifier",
            model_params={"max_depth": 3, "min_samples_split": 2, "random_seed": 42},
            random_seed=42,
        )
        pipe_dt.fit(raw_X, y, test_size=0.20)
        self.assertIn("accuracy", pipe_dt.test_metrics)

        # Single record risk prediction and probability
        sample_dict = {f: raw_X[0][i] for i, f in enumerate(schema.feature_names)}
        pred_risk = pipe_knn.predict_single(sample_dict)
        self.assertIn(pred_risk, [0, 1])
        proba = pipe_knn.predict_single_proba(sample_dict)
        self.assertIsNotNone(proba)
        self.assertAlmostEqual(sum(proba.values()), 1.0, places=4)

    def test_tab4_and_tab5_artifacts_exist(self):
        """Verify Tab 4 & 5: All required diagnostic visualization artifacts exist on disk."""
        artifacts_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts")
        required_artifacts = [
            "knn_overfitting_curve.png",
            "tree_overfitting_curve.png",
            "linear_regression_regularization_curve.png",
            "classification_comparison.png",
            "roc_curves.png",
            "actual_vs_predicted_scores.png",
            "linear_regression_loss_curve.png",
            "feature_distributions.png",
            "correlation_heatmap.png",
            "confusion_matrix.png",
        ]
        for art in required_artifacts:
            path = os.path.join(artifacts_dir, art)
            self.assertTrue(os.path.exists(path), f"Artifact missing for dashboard: {art}")

    def test_visible_labels_map_to_internal_columns(self):
        """Verify 1: All visible labels map exactly to the correct internal UCI column names."""
        expected_mappings = {
            "school": "School",
            "sex": "Sex",
            "age": "Age (years)",
            "address": "Residential Area",
            "famsize": "Family Size",
            "Pstatus": "Parents' Cohabitation Status",
            "Medu": "Mother's Education Level",
            "Fedu": "Father's Education Level",
            "Mjob": "Mother's Occupation",
            "Fjob": "Father's Occupation",
            "reason": "Reason for Choosing School",
            "guardian": "Student's Guardian",
            "traveltime": "Travel Time to School",
            "studytime": "Weekly Study Time",
            "failures": "Previous Class Failures",
            "schoolsup": "Extra Educational Support",
            "famsup": "Family Educational Support",
            "paid": "Extra Paid Classes",
            "activities": "Extracurricular Activities",
            "nursery": "Attended Nursery School",
            "higher": "Wants Higher Education",
            "internet": "Internet Access at Home",
            "romantic": "In a Romantic Relationship",
            "famrel": "Family Relationship Quality",
            "freetime": "Free Time After School",
            "goout": "Going Out with Friends",
            "Dalc": "Workday Alcohol Consumption",
            "Walc": "Weekend Alcohol Consumption",
            "health": "Self-Rated Health",
            "absences": "School Absences",
        }
        self.assertEqual(len(expected_mappings), 30)
        for col, expected_label in expected_mappings.items():
            self.assertIn(col, UCI_FEATURE_CONFIG, f"Column '{col}' missing from UCI_FEATURE_CONFIG")
            actual_label = UCI_FEATURE_CONFIG[col]["label"]
            self.assertEqual(actual_label, expected_label, f"Label mismatch for '{col}': '{actual_label}' != '{expected_label}'")

    def test_dropdown_options_submit_raw_dataset_values(self):
        """Verify 2: Dropdown options submit the correct encoded/raw dataset values."""
        # Verify coded categories retain underlying dataset tokens
        self.assertEqual(UCI_FEATURE_CONFIG["address"]["options"], ["U", "R"])
        self.assertEqual(ADDRESS_LABELS["U"], "U (Urban)")
        self.assertEqual(ADDRESS_LABELS["R"], "R (Rural)")

        self.assertIn("LE3", UCI_FEATURE_CONFIG["famsize"]["options"])
        self.assertIn("GT3", UCI_FEATURE_CONFIG["famsize"]["options"])
        self.assertEqual(FAMILY_SIZE_LABELS["LE3"], "LE3 (3 or fewer)")
        self.assertEqual(FAMILY_SIZE_LABELS["GT3"], "GT3 (More than 3)")

        self.assertEqual(UCI_FEATURE_CONFIG["Pstatus"]["options"], ["T", "A"])
        self.assertEqual(COHABITATION_LABELS["T"], "T (Together)")
        self.assertEqual(COHABITATION_LABELS["A"], "A (Apart)")

        # Verify all dropdown options exist in student-mat.csv
        for col, cfg in UCI_FEATURE_CONFIG.items():
            if cfg.get("control") == "select":
                col_idx = self.dataset.header_to_idx[col]
                csv_vals = set(row[col_idx] for row in self.dataset.rows if row[col_idx] is not None)
                for opt in cfg["options"]:
                    match_found = (opt in csv_vals) or (str(opt) in [str(v) for v in csv_vals])
                    self.assertTrue(match_found, f"Option '{opt}' for '{col}' not present in dataset values: {csv_vals}")

    def test_ordinal_controls_return_valid_integers_in_expected_ranges(self):
        """Verify 3: Ordinal and integer controls return valid integers in their documented ranges."""
        # Medu & Fedu (0 to 4)
        for edu_col in ("Medu", "Fedu"):
            cfg = UCI_FEATURE_CONFIG[edu_col]
            self.assertEqual(cfg["control"], "select")
            self.assertEqual(cfg["options"], [0, 1, 2, 3, 4])
            self.assertTrue(all(isinstance(x, int) for x in cfg["options"]))

        self.assertEqual(EDUCATION_LEVEL_LABELS[0], "0 — None")
        self.assertEqual(EDUCATION_LEVEL_LABELS[1], "1 — Primary education (4th grade)")
        self.assertEqual(EDUCATION_LEVEL_LABELS[2], "2 — 5th–9th grade")
        self.assertEqual(EDUCATION_LEVEL_LABELS[3], "3 — Secondary education")
        self.assertEqual(EDUCATION_LEVEL_LABELS[4], "4 — Higher education")

        # traveltime (1 to 4)
        self.assertEqual(UCI_FEATURE_CONFIG["traveltime"]["options"], [1, 2, 3, 4])
        self.assertEqual(TRAVEL_TIME_LABELS[1], "1 — Less than 15 minutes")
        self.assertEqual(TRAVEL_TIME_LABELS[2], "2 — 15–30 minutes")
        self.assertEqual(TRAVEL_TIME_LABELS[3], "3 — 30 minutes to 1 hour")
        self.assertEqual(TRAVEL_TIME_LABELS[4], "4 — More than 1 hour")

        # studytime (1 to 4)
        self.assertEqual(UCI_FEATURE_CONFIG["studytime"]["options"], [1, 2, 3, 4])
        self.assertEqual(STUDY_TIME_LABELS[1], "1 — Less than 2 hours/week")
        self.assertEqual(STUDY_TIME_LABELS[2], "2 — 2–5 hours/week")
        self.assertEqual(STUDY_TIME_LABELS[3], "3 — 5–10 hours/week")
        self.assertEqual(STUDY_TIME_LABELS[4], "4 — More than 10 hours/week")

        # failures (0 to 3)
        self.assertEqual(UCI_FEATURE_CONFIG["failures"]["options"], [0, 1, 2, 3])

        # 1-5 scales with documented help text
        scale_5_cols = ["famrel", "freetime", "goout", "Dalc", "Walc", "health"]
        for col in scale_5_cols:
            cfg = UCI_FEATURE_CONFIG[col]
            self.assertEqual(cfg["options"], [1, 2, 3, 4, 5])
            self.assertIn("1 is the lowest level and 5 is the highest", cfg["help"])

        # age (15 to 22)
        age_cfg = UCI_FEATURE_CONFIG["age"]
        self.assertEqual(age_cfg["min"], 15)
        self.assertEqual(age_cfg["max"], 22)
        self.assertEqual(age_cfg["step"], 1)

        # absences (non-negative integer, upper limit covers observed max of 75)
        abs_cfg = UCI_FEATURE_CONFIG["absences"]
        self.assertEqual(abs_cfg["control"], "number")
        self.assertEqual(abs_cfg["min"], 0)
        self.assertGreaterEqual(abs_cfg["max"], 75)
        self.assertEqual(abs_cfg["step"], 1)

    def test_regression_and_classification_forms_supply_all_required_features(self):
        """Verify 4: Forms supply all 30 required predictor features in the expected schema."""
        schema_reg = infer_schema(
            self.dataset.headers,
            self.dataset.rows,
            target_column="G3",
            task_type="regression",
            ignored_columns=["G1", "G2", "academic_risk"],
        )
        schema_clf = infer_schema(
            self.dataset.headers,
            self.dataset.rows,
            target_column="academic_risk",
            task_type="binary_classification",
            ignored_columns=["G1", "G2", "G3"],
        )

        self.assertEqual(len(schema_reg.feature_names), 30)
        self.assertEqual(len(schema_clf.feature_names), 30)
        self.assertEqual(schema_reg.feature_names, schema_clf.feature_names)

        # Confirm every schema feature has an exact configuration entry
        for feature in schema_reg.feature_names:
            self.assertIn(feature, UCI_FEATURE_CONFIG, f"Feature '{feature}' missing from UCI configuration")

        # Confirm building a default sample produces all 30 schema features
        default_sample = {col: cfg["default"] for col, cfg in UCI_FEATURE_CONFIG.items()}
        for feature in schema_reg.feature_names:
            self.assertIn(feature, default_sample)

    def test_pipeline_inference_with_form_inputs(self):
        """Verify 5: Existing preprocessing and inference pipelines work seamlessly with form inputs."""
        # 1. Regression Pipeline
        schema_reg = infer_schema(
            self.dataset.headers,
            self.dataset.rows,
            target_column="G3",
            task_type="regression",
            ignored_columns=["G1", "G2", "academic_risk"],
        )
        raw_X_reg, y_reg = self.dataset.split_features_target(schema_reg.feature_names, "G3")
        pipe_reg = MLPipeline(schema_reg, "Linear Regression", {"solver": "normal_equation", "l2_reg": 5.0})
        pipe_reg.fit(raw_X_reg, [float(v) for v in y_reg], test_size=0.20)

        # 2. Classification Pipeline
        schema_clf = infer_schema(
            self.dataset.headers,
            self.dataset.rows,
            target_column="academic_risk",
            task_type="binary_classification",
            ignored_columns=["G1", "G2", "G3"],
        )
        raw_X_clf, y_clf = self.dataset.split_features_target(schema_clf.feature_names, "academic_risk")
        pipe_clf = MLPipeline(schema_clf, "Decision Tree Classifier", {"max_depth": 2})
        pipe_clf.fit(raw_X_clf, [int(v) for v in y_clf], test_size=0.20)

        # Construct single student inputs from form configuration defaults
        sample_inputs = {col: cfg["default"] for col, cfg in UCI_FEATURE_CONFIG.items()}

        pred_grade = pipe_reg.predict_single(sample_inputs)
        self.assertIsInstance(pred_grade, float)
        self.assertGreaterEqual(pred_grade, 0.0)
        self.assertLessEqual(pred_grade, 20.0)

        pred_risk = pipe_clf.predict_single(sample_inputs)
        self.assertIn(int(pred_risk), [0, 1])
        proba = pipe_clf.predict_single_proba(sample_inputs)
        self.assertAlmostEqual(sum(proba.values()), 1.0, places=4)

    def test_no_target_or_leakage_columns_in_predictor_controls(self):
        """Verify 6: No target or leakage-prone grade columns are exposed as predictors."""
        forbidden_targets = ["G1", "G2", "G3", "academic_risk", "final_exam_score", "student_id"]
        for target in forbidden_targets:
            self.assertNotIn(target, UCI_FEATURE_CONFIG, f"Forbidden column '{target}' exposed in UCI_FEATURE_CONFIG")

        schema_reg = infer_schema(
            self.dataset.headers,
            self.dataset.rows,
            target_column="G3",
            task_type="regression",
            ignored_columns=["G1", "G2", "academic_risk"],
        )
        schema_clf = infer_schema(
            self.dataset.headers,
            self.dataset.rows,
            target_column="academic_risk",
            task_type="binary_classification",
            ignored_columns=["G1", "G2", "G3"],
        )

        for col in ["G1", "G2", "G3", "academic_risk"]:
            self.assertNotIn(col, schema_reg.feature_names)
            self.assertNotIn(col, schema_clf.feature_names)


if __name__ == "__main__":
    unittest.main()
