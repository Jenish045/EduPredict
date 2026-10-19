"""
Unit tests for OneVsRestClassifier multiclass extension.
"""

import unittest
from edupredict.models.multiclass import OneVsRestClassifier
from edupredict.models.logistic_regression import LogisticRegression


class TestMulticlass(unittest.TestCase):

    def test_one_vs_rest_classifier(self):
        # 3 classes: 0 (Low), 1 (Medium), 2 (High)
        X = [
            [1.0, 1.0],
            [1.5, 2.0],
            [5.0, 5.0],
            [5.5, 6.0],
            [9.0, 9.0],
            [9.5, 10.0],
        ]
        y = [0, 0, 1, 1, 2, 2]

        ovr = OneVsRestClassifier(
            base_estimator_cls=LogisticRegression,
            estimator_kwargs={"max_epochs": 300, "learning_rate": 0.1, "random_seed": 42}
        )
        ovr.fit(X, y)
        self.assertEqual(ovr.classes_, [0, 1, 2])
        self.assertEqual(len(ovr.estimators_), 3)

        # Predict probabilities
        probs = ovr.predict_proba(X)
        self.assertEqual(len(probs), 6)
        for p_dict in probs:
            total_p = sum(p_dict.values())
            self.assertAlmostEqual(total_p, 1.0, places=4)

        # Predict classes
        preds = ovr.predict(X)
        self.assertEqual(len(preds), 6)
        # Verify extreme classes are distinguished
        self.assertEqual(preds[0], 0)
        self.assertEqual(preds[-1], 2)


if __name__ == "__main__":
    unittest.main()
