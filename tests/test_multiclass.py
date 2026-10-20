"""
Unit tests for multiclass classification using Scikit-learn estimators.
"""

import unittest
from edupredict.models import LogisticRegression, DecisionTreeClassifier, GaussianNaiveBayes


class TestMulticlass(unittest.TestCase):

    def test_multiclass_logistic_regression(self):
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

        clf = LogisticRegression(random_state=42)
        clf.fit(X, y)
        self.assertEqual(list(clf.classes_), [0, 1, 2])

        probs = clf.predict_proba(X)
        self.assertEqual(len(probs), 6)
        for p_row in probs:
            self.assertAlmostEqual(sum(p_row), 1.0, places=4)

        preds = clf.predict(X)
        self.assertEqual(len(preds), 6)
        self.assertEqual(preds[0], 0)
        self.assertEqual(preds[-1], 2)

    def test_multiclass_decision_tree(self):
        X = [[1.0], [2.0], [5.0], [6.0], [9.0], [10.0]]
        y = [0, 0, 1, 1, 2, 2]

        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X, y)
        self.assertEqual(list(dt.classes_), [0, 1, 2])
        preds = dt.predict(X)
        self.assertEqual(list(preds), y)


if __name__ == "__main__":
    unittest.main()
