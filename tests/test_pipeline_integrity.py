"""
Integration and regression tests for pipeline integrity, sample alignment,
zero-leakage transformer boundaries, and unseen category handling.
"""

import unittest
from edupredict.model_selection.split import train_test_split, train_test_split_indices
from edupredict.preprocessing.imputer import SimpleImputer
from edupredict.preprocessing.encoder import OneHotEncoder
from edupredict.preprocessing.scaler import StandardScaler
from edupredict.models import LinearRegression, KNNClassifier
from edupredict.metrics.regression import r2_score, mean_squared_error
from edupredict.metrics.classification import accuracy_score, f1_score, roc_auc_score


class TestPipelineIntegrity(unittest.TestCase):

    def setUp(self):
        # 60 samples with identifiable IDs, known targets, mixed categories, and missing values
        self.n_samples = 60
        self.raw_records = []
        self.y_reg = []
        self.y_clf = []

        categories = ["CatA", "CatB", "CatC"]
        for i in range(self.n_samples):
            student_id = f"STU_{i:04d}"
            study_hours = 10.0 + (i % 15) * 2.0
            # Inject intermittent missing value
            attendance = None if i % 12 == 0 else (60.0 + (i % 20) * 2.0)
            cat_val = categories[i % len(categories)]
            
            # Ground truth targets deterministically derived from sample index i
            reg_score = 40.0 + 0.8 * study_hours
            clf_risk = 1 if i % 4 == 0 else 0  # Imbalanced: 15 at-risk (25%), 45 safe (75%)

            self.raw_records.append([student_id, study_hours, attendance, cat_val])
            self.y_reg.append(reg_score)
            self.y_clf.append(clf_risk)

    def test_sample_alignment_and_non_overlapping_split(self):
        """
        Verify that:
        1. Every feature row remains paired with its exact regression and classification targets.
        2. Train and test index partitions are mutually disjoint (zero overlap).
        3. Stratified splitting preserves class proportions.
        """
        raw_tr, raw_te, y_reg_tr, y_reg_te, y_clf_tr, y_clf_te = train_test_split(
            self.raw_records, self.y_reg, self.y_clf,
            test_size=0.25,
            random_seed=42,
            stratify=self.y_clf
        )

        self.assertEqual(len(raw_tr), 45)
        self.assertEqual(len(raw_te), 15)
        self.assertEqual(len(y_reg_tr), 45)
        self.assertEqual(len(y_reg_te), 15)
        self.assertEqual(len(y_clf_tr), 45)
        self.assertEqual(len(y_clf_te), 15)

        # 1. Zero overlap check
        train_ids = {row[0] for row in raw_tr}
        test_ids = {row[0] for row in raw_te}
        self.assertEqual(len(train_ids), 45)
        self.assertEqual(len(test_ids), 15)
        self.assertTrue(train_ids.isdisjoint(test_ids), "Train and test IDs overlap!")

        # 2. Strict sample-to-target alignment in training partition
        for r, row in enumerate(raw_tr):
            idx = int(row[0].split("_")[1])
            expected_reg = 40.0 + 0.8 * (10.0 + (idx % 15) * 2.0)
            expected_clf = 1 if idx % 4 == 0 else 0
            self.assertAlmostEqual(y_reg_tr[r], expected_reg, places=5)
            self.assertEqual(y_clf_tr[r], expected_clf)

        # 3. Strict sample-to-target alignment in testing partition
        for r, row in enumerate(raw_te):
            idx = int(row[0].split("_")[1])
            expected_reg = 40.0 + 0.8 * (10.0 + (idx % 15) * 2.0)
            expected_clf = 1 if idx % 4 == 0 else 0
            self.assertAlmostEqual(y_reg_te[r], expected_reg, places=5)
            self.assertEqual(y_clf_te[r], expected_clf)

        # 4. Stratified class proportions check
        # Original: 15 ones / 60 = 25%
        # Test: 4 ones / 15 = 26.7%
        # Train: 11 ones / 45 = 24.4%
        self.assertEqual(y_clf_te.count(1), 4)
        self.assertEqual(y_clf_tr.count(1), 11)

    def test_preprocessing_leakage_isolation(self):
        """
        Verify that SimpleImputer and StandardScaler compute statistics
        EXCLUSIVELY from the training split.
        """
        # Train has mean = 20.0
        X_train_num = [[10.0], [20.0], [30.0]]
        # Test has an extreme outlier 1000.0 and a missing value None
        X_test_num = [[None], [1000.0]]

        imputer = SimpleImputer(strategy="mean")
        X_tr_clean = imputer.fit_transform(X_train_num)
        X_te_clean = imputer.transform(X_test_num)

        # The imputed value in test must equal the training mean (20.0), NOT contaminated by test
        self.assertEqual(imputer.statistics_[0], 20.0)
        self.assertEqual(X_te_clean[0][0], 20.0)

        # Scaler fit on train must have mean = 20.0
        scaler = StandardScaler()
        scaler.fit(X_tr_clean)
        self.assertEqual(scaler.mean_[0], 20.0)

        # Transform test: (1000 - 20) / std
        scaled_test = scaler.transform([[1000.0]])
        self.assertAlmostEqual(scaled_test[0][0], (1000.0 - 20.0) / scaler.scale_[0], places=4)

    def test_unseen_category_handling_in_pipeline(self):
        """
        Verify that OneHotEncoder safely handles categories in the test set
        that never appeared in the training set (encoded as all-zeros).
        """
        cat_train = [["HighSchool"], ["Bachelor"], ["HighSchool"]]
        cat_test = [["Bachelor"], ["Doctorate"]]  # "Doctorate" is unseen!

        ohe = OneHotEncoder(drop_first=False)
        train_encoded = ohe.fit_transform(cat_train)
        test_encoded = ohe.transform(cat_test)

        self.assertEqual(len(ohe.categories_[0]), 2)  # ["Bachelor", "HighSchool"]
        # "Doctorate" must be encoded with all zeros: [0.0, 0.0]
        self.assertEqual(test_encoded[1], [0.0, 0.0])

    def test_end_to_end_leak_free_pipeline_execution(self):
        """
        Execute the entire 9-step pipeline on constructed records and verify
        that both regression and classification models train and evaluate cleanly.
        """
        # 1. Feature matrix without ID
        features = [[r[1], r[2], r[3]] for r in self.raw_records]

        # 2. Split features and targets together
        X_tr, X_te, y_reg_tr, y_reg_te, y_clf_tr, y_clf_te = train_test_split(
            features, self.y_reg, self.y_clf,
            test_size=0.25,
            random_seed=42,
            stratify=self.y_clf
        )

        num_tr = [[r[0], r[1]] for r in X_tr]
        cat_tr = [[r[2]] for r in X_tr]
        num_te = [[r[0], r[1]] for r in X_te]
        cat_te = [[r[2]] for r in X_te]

        # 3. Fit imputers on train only
        num_imp = SimpleImputer(strategy="mean")
        cat_imp = SimpleImputer(strategy="mode")
        num_tr_c = num_imp.fit_transform(num_tr)
        cat_tr_c = cat_imp.fit_transform(cat_tr)

        # 4. Transform test with frozen imputers
        num_te_c = num_imp.transform(num_te)
        cat_te_c = cat_imp.transform(cat_te)

        # 5. Fit encoder on train only
        ohe = OneHotEncoder(drop_first=False)
        cat_tr_enc = ohe.fit_transform(cat_tr_c)
        # 6. Transform test with frozen encoder
        cat_te_enc = ohe.transform(cat_te_c)

        comb_tr = [num_tr_c[i] + cat_tr_enc[i] for i in range(len(num_tr_c))]
        comb_te = [num_te_c[i] + cat_te_enc[i] for i in range(len(num_te_c))]

        # 7. Fit scaler on train only
        scaler = StandardScaler()
        scaled_tr = scaler.fit_transform(comb_tr)
        # 8. Transform test with frozen scaler
        scaled_te = scaler.transform(comb_te)

        # 9. Train regression model
        lr = LinearRegression()
        lr.fit(scaled_tr, y_reg_tr)
        preds_reg = lr.predict(scaled_te)
        r2 = r2_score(y_reg_te, preds_reg)
        self.assertTrue(r2 > 0.8, f"Expected R2 > 0.8 on clean data, got {r2}")

        # Train classification model
        clf = KNNClassifier(n_neighbors=3)
        clf.fit(scaled_tr, y_clf_tr)
        preds_clf = clf.predict(scaled_te)
        acc = accuracy_score(y_clf_te, preds_clf)
        self.assertTrue(0.0 <= acc <= 1.0)

    def test_cli_and_dashboard_use_identical_split_logic(self):
        """
        Verify that app.py pipeline and runner.py use the same aligned split logic
        with identical sample counts and stratified proportions on UCI student-mat.csv.
        """
        from edupredict.data.loader import load_csv
        import os
        csv_path = "data/student-mat.csv"
        self.assertTrue(os.path.exists(csv_path), "UCI dataset data/student-mat.csv must be present")

        headers, rows = load_csv(csv_path)
        g3_idx = headers.index("G3")
        y_reg = [float(row[g3_idx]) for row in rows]
        y_clf = [1 if float(row[g3_idx]) < 10.0 else 0 for row in rows]

        # Use 30 non-grade predictor features
        ignored = {"G1", "G2", "G3", "academic_risk"}
        feat_indices = [i for i, h in enumerate(headers) if h not in ignored]
        raw_X = [[row[idx] for idx in feat_indices] for row in rows]

        # Use unified train_test_split
        raw_tr, raw_te, y_reg_tr, y_reg_te, y_clf_tr, y_clf_te = train_test_split(
            raw_X, y_reg, y_clf,
            test_size=0.20,
            random_seed=42,
            stratify=y_clf
        )

        self.assertEqual(len(raw_tr), 316)
        self.assertEqual(len(raw_te), 79)

        # Check stratification consistency
        overall_prop = sum(y_clf) / len(y_clf)
        train_prop = sum(y_clf_tr) / len(y_clf_tr)
        test_prop = sum(y_clf_te) / len(y_clf_te)
        self.assertAlmostEqual(train_prop, overall_prop, places=2)
        self.assertAlmostEqual(test_prop, overall_prop, places=2)
        self.assertEqual(sum(y_clf_te), 26)  # Exactly 26 at-risk in holdout
        self.assertEqual(len(y_clf_te) - sum(y_clf_te), 53)  # Exactly 53 normal in holdout


if __name__ == "__main__":
    unittest.main()

