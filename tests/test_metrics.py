"""
Unit tests for evaluation metrics: regression, classification, and ROC/AUC.
"""

import unittest
from edupredict.metrics.regression import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from edupredict.metrics.classification import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_curve,
    roc_auc_score,
)


class TestMetrics(unittest.TestCase):

    def test_regression_metrics(self):
        y_true = [10.0, 20.0, 30.0]
        y_pred = [12.0, 18.0, 32.0]
        # errors: [2, -2, 2] -> sq: [4, 4, 4] -> MSE = 4.0, RMSE = 2.0, MAE = 2.0
        self.assertAlmostEqual(mean_squared_error(y_true, y_pred), 4.0)
        self.assertAlmostEqual(root_mean_squared_error(y_true, y_pred), 2.0)
        self.assertAlmostEqual(mean_absolute_error(y_true, y_pred), 2.0)
        # y_mean = 20 -> tot sq = (100 + 0 + 100) = 200
        # ss_res = 12 -> R2 = 1 - 12/200 = 0.94
        self.assertAlmostEqual(r2_score(y_true, y_pred), 0.94)

    def test_constant_target_r2(self):
        y_true = [5.0, 5.0, 5.0]
        y_pred = [5.0, 5.0, 5.0]
        self.assertAlmostEqual(r2_score(y_true, y_pred), 1.0)

        y_bad = [4.0, 6.0, 5.0]
        self.assertAlmostEqual(r2_score(y_true, y_bad), 0.0)

    def test_classification_metrics(self):
        y_true = [1, 1, 0, 0, 1]
        y_pred = [1, 0, 0, 0, 1]
        # TP=2, FN=1, TN=2, FP=0
        # Accuracy: 4/5 = 0.8
        # Precision: 2/(2+0) = 1.0
        # Recall: 2/(2+1) = 2/3 = 0.6666...
        # F1: 2*(1.0*2/3)/(1.0+2/3) = 4/3 / 5/3 = 0.8
        self.assertAlmostEqual(accuracy_score(y_true, y_pred), 0.8)
        self.assertAlmostEqual(precision_score(y_true, y_pred, pos_label=1), 1.0)
        self.assertAlmostEqual(recall_score(y_true, y_pred, pos_label=1), 2.0 / 3.0)
        self.assertAlmostEqual(f1_score(y_true, y_pred, pos_label=1), 0.8)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        self.assertEqual(cm[0][0], 2)  # TN
        self.assertEqual(cm[0][1], 0)  # FP
        self.assertEqual(cm[1][0], 1)  # FN
        self.assertEqual(cm[1][1], 2)  # TP

        report = classification_report(y_true, y_pred, labels=[0, 1])
        self.assertIn("Accuracy", report)
        self.assertIn("F1-Score", report)

    def test_zero_division_handling(self):
        # No predicted positives: TP=0, FP=0
        y_true = [0, 0, 1]
        y_pred = [0, 0, 0]
        self.assertEqual(precision_score(y_true, y_pred, pos_label=1, zero_division=0.0), 0.0)
        self.assertEqual(f1_score(y_true, y_pred, pos_label=1, zero_division=0.0), 0.0)

    def test_roc_curve_and_auc(self):
        # Hand-calculated example:
        # y_true = [0, 0, 1, 1]
        # y_score = [0.1, 0.4, 0.35, 0.8]
        # Expected AUC = 0.75
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.4, 0.35, 0.8]

        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
        self.assertTrue(len(fpr) > 1)
        self.assertEqual(fpr[0], 0.0)
        self.assertEqual(tpr[0], 0.0)
        self.assertEqual(fpr[-1], 1.0)
        self.assertEqual(tpr[-1], 1.0)

        auc = roc_auc_score(y_true, y_score, pos_label=1)
        self.assertAlmostEqual(auc, 0.75, places=5)

        # Perfect ranking: AUC = 1.0
        y_perf_true = [0, 0, 1, 1]
        y_perf_score = [0.1, 0.2, 0.8, 0.9]
        self.assertAlmostEqual(roc_auc_score(y_perf_true, y_perf_score), 1.0)


if __name__ == "__main__":
    unittest.main()
