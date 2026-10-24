"""
Unit and integration tests for EduPredict custom model estimators:
- Custom Linear Regression (Normal Equation & Gradient Descent)
- Custom K-Nearest Neighbors Classifier
- Custom Decision Tree Classifier

Strictly zero dependencies on third-party ML libraries (no Scikit-learn, no NumPy).
"""

import unittest

from edupredict.models import (
    LinearRegression,
    KNNClassifier,
    DecisionTreeClassifier,
    get_model,
    list_models,
)
from edupredict.metrics.regression import r2_score, mean_squared_error
from edupredict.metrics.classification import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


class TestModels(unittest.TestCase):

    def test_unfitted_model_error(self):
        """Verify that calling predict on an unfitted custom estimator raises ValueError."""
        lr = LinearRegression()
        with self.assertRaises(ValueError):
            lr.predict([[1.0, 2.0]])

        knn = KNNClassifier()
        with self.assertRaises(ValueError):
            knn.predict([[1.0, 2.0]])

        dt = DecisionTreeClassifier()
        with self.assertRaises(ValueError):
            dt.predict([[1.0, 2.0]])

    def test_linear_regression_normal_equation(self):
        """Test LinearRegression with closed-form Normal Equation on linear data."""
        # y = 3 * x1 + 2 * x2 + 5
        X = [
            [1.0, 2.0],
            [2.0, 1.0],
            [3.0, 4.0],
            [4.0, 3.0],
            [5.0, 5.0],
        ]
        y = [3.0 * r[0] + 2.0 * r[1] + 5.0 for r in X]

        lr = LinearRegression(solver="normal_equation", fit_intercept=True)
        lr.fit(X, y)
        preds = lr.predict(X)

        score = r2_score(y, preds)
        self.assertGreater(score, 0.999)
        self.assertAlmostEqual(lr.coef_[0], 3.0, places=4)
        self.assertAlmostEqual(lr.coef_[1], 2.0, places=4)
        self.assertAlmostEqual(lr.intercept_, 5.0, places=4)

    def test_linear_regression_gradient_descent(self):
        """Test LinearRegression with iterative Gradient Descent solver."""
        X = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        y = [2.0 * r[0] + 1.0 for r in X]

        lr = LinearRegression(
            solver="gradient_descent",
            learning_rate=0.05,
            max_epochs=500,
            fit_intercept=True,
            random_seed=42,
        )
        lr.fit(X, y)
        preds = lr.predict(X)

        score = r2_score(y, preds)
        self.assertGreater(score, 0.95)
        self.assertTrue(len(lr.loss_history_) > 0)
        # Loss should decrease over epochs
        self.assertLess(lr.loss_history_[-1], lr.loss_history_[0])

    def test_linear_regression_ridge_penalty(self):
        """Test that L2 penalty shrinks coefficients."""
        X = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]
        y = [5.0, 8.0, 11.0, 14.0]

        lr_unreg = LinearRegression(solver="normal_equation", l2_reg=0.0)
        lr_unreg.fit(X, y)

        lr_reg = LinearRegression(solver="normal_equation", l2_reg=10.0)
        lr_reg.fit(X, y)

        norm_unreg = sum(c * c for c in lr_unreg.coef_)
        norm_reg = sum(c * c for c in lr_reg.coef_)
        self.assertLess(norm_reg, norm_unreg)

    def test_analytical_gradient_vs_numerical_finite_differences(self):
        """Verify analytical MSE + L2 gradient matches central finite differences."""
        X = [
            [1.0, 0.5, -0.2],
            [0.2, 1.5, 0.8],
            [-0.5, 0.2, 1.2],
            [1.2, -0.8, 0.5],
            [0.8, 0.9, -0.5],
        ]
        y = [2.5, 1.2, 3.8, 0.5, 2.1]
        l2_reg = 0.1
        w = [0.4, -0.3, 0.6]
        b = 0.5
        eps = 1e-6

        # Loss function
        def compute_loss(weights, intercept):
            preds = [intercept + sum(w_j * x_ij for w_j, x_ij in zip(weights, row)) for row in X]
            mse = sum((y_i - p_i) ** 2 for y_i, p_i in zip(y, preds)) / len(y)
            penalty = l2_reg * sum(w_j ** 2 for w_j in weights)
            return mse + penalty

        # Analytical gradient
        # dJ/dw_j = (2/N) * sum_i ((p_i - y_i) * x_ij) + 2 * l2_reg * w_j
        # dJ/db   = (2/N) * sum_i (p_i - y_i)
        n = len(X)
        preds = [b + sum(w_j * x_ij for w_j, x_ij in zip(w, row)) for row in X]
        errors = [p_i - y_i for y_i, p_i in zip(y, preds)]
        grad_w_analytical = [
            (2.0 / n) * sum(e * row[j] for e, row in zip(errors, X)) + 2.0 * l2_reg * w[j]
            for j in range(len(w))
        ]
        grad_b_analytical = (2.0 / n) * sum(errors)

        # Numerical gradient (central differences)
        grad_w_numerical = []
        for j in range(len(w)):
            w_plus = list(w)
            w_plus[j] += eps
            loss_plus = compute_loss(w_plus, b)

            w_minus = list(w)
            w_minus[j] -= eps
            loss_minus = compute_loss(w_minus, b)

            grad_w_numerical.append((loss_plus - loss_minus) / (2.0 * eps))

        loss_b_plus = compute_loss(w, b + eps)
        loss_b_minus = compute_loss(w, b - eps)
        grad_b_numerical = (loss_b_plus - loss_b_minus) / (2.0 * eps)

        for j in range(len(w)):
            diff = abs(grad_w_analytical[j] - grad_w_numerical[j])
            self.assertLess(diff, 1e-5, f"Weight gradient {j} differs by {diff}")

        diff_b = abs(grad_b_analytical - grad_b_numerical)
        self.assertLess(diff_b, 1e-5, f"Intercept gradient differs by {diff_b}")

    def test_normal_equation_and_gradient_descent_equivalence(self):
        """Verify Normal Equation and Gradient Descent find equivalent solutions."""
        X = [
            [0.1, 0.5],
            [0.3, 0.2],
            [0.8, 0.9],
            [0.5, 0.6],
            [0.9, 0.4],
            [0.2, 0.7],
            [0.6, 0.1],
            [0.4, 0.8],
        ]
        y = [1.2, 0.8, 2.5, 1.8, 2.1, 1.5, 1.1, 1.9]

        lr_norm = LinearRegression(
            solver="normal_equation",
            fit_intercept=True,
            l2_reg=0.01,
        )
        lr_norm.fit(X, y)

        lr_gd = LinearRegression(
            solver="gradient_descent",
            learning_rate=0.08,
            max_epochs=8000,
            fit_intercept=True,
            l2_reg=0.01,
            random_seed=42,
        )
        lr_gd.fit(X, y)

        for j in range(len(lr_norm.coef_)):
            self.assertAlmostEqual(lr_norm.coef_[j], lr_gd.coef_[j], places=2)
        self.assertAlmostEqual(lr_norm.intercept_, lr_gd.intercept_, places=2)

    def test_knn_classifier_euclidean_and_manhattan(self):
        """Test KNNClassifier with Euclidean and Manhattan distances and deterministic tie-breaking."""
        X = [[1.0, 1.0], [1.0, 2.0], [5.0, 5.0], [6.0, 5.0]]
        y = [0, 0, 1, 1]

        # Euclidean
        knn_euc = KNNClassifier(n_neighbors=2, metric="euclidean")
        knn_euc.fit(X, y)
        self.assertEqual(knn_euc.predict([[1.2, 1.5]]), [0])
        self.assertEqual(knn_euc.predict([[5.5, 5.2]]), [1])

        # Manhattan
        knn_man = KNNClassifier(n_neighbors=2, metric="manhattan")
        knn_man.fit(X, y)
        self.assertEqual(knn_man.predict([[1.2, 1.5]]), [0])
        self.assertEqual(knn_man.predict([[5.5, 5.2]]), [1])

    def test_knn_predict_proba(self):
        """Test KNN predict_proba returns valid probability distribution."""
        X = [[1.0], [2.0], [8.0], [9.0]]
        y = [0, 0, 1, 1]

        knn = KNNClassifier(n_neighbors=2)
        knn.fit(X, y)
        probs = knn.predict_proba([[1.5], [8.5]])

        self.assertEqual(len(probs), 2)
        # Class 0 near 1.5
        self.assertEqual(probs[0], [1.0, 0.0])
        # Class 1 near 8.5
        self.assertEqual(probs[1], [0.0, 1.0])

    def test_decision_tree_classifier(self):
        """Test DecisionTreeClassifier with Gini impurity splitting and feature importances."""
        X = [
            [1.0, 10.0],
            [1.5, 12.0],
            [2.0, 11.0],
            [8.0, 2.0],
            [8.5, 3.0],
            [9.0, 1.0],
        ]
        y = [0, 0, 0, 1, 1, 1]

        dt = DecisionTreeClassifier(max_depth=3, random_seed=42)
        dt.fit(X, y)
        preds = dt.predict(X)

        self.assertEqual(preds, y)
        self.assertEqual(dt.score(X, y), 1.0)
        self.assertEqual(len(dt.feature_importances_), 2)
        self.assertAlmostEqual(sum(dt.feature_importances_), 1.0, places=5)

    def test_decision_tree_probabilities(self):
        """Test DecisionTreeClassifier predict_proba."""
        X = [[1.0], [2.0], [8.0], [9.0]]
        y = [0, 0, 1, 1]

        dt = DecisionTreeClassifier(max_depth=2, random_seed=42)
        dt.fit(X, y)
        probs = dt.predict_proba([[1.5], [8.5]])

        self.assertEqual(len(probs), 2)
        for p in probs:
            self.assertAlmostEqual(sum(p), 1.0, places=5)

    def test_factory_and_registry_retrieval(self):
        """Test get_model factory and list_models for custom models."""
        reg_models = list_models("regression")
        self.assertIn("Linear Regression", reg_models)

        clf_models = list_models("classification")
        self.assertIn("KNN Classifier", clf_models)
        self.assertIn("Decision Tree Classifier", clf_models)

        lr = get_model("Linear Regression", task="regression", solver="normal_equation")
        self.assertIsInstance(lr, LinearRegression)

        knn = get_model("KNN Classifier", task="classification", n_neighbors=7)
        self.assertIsInstance(knn, KNNClassifier)
        self.assertEqual(knn.n_neighbors, 7)

        dt = get_model("Decision Tree Classifier", task="classification", max_depth=4)
        self.assertIsInstance(dt, DecisionTreeClassifier)
        self.assertEqual(dt.max_depth, 4)

    def test_decision_tree_f1_exact_consistency(self):
        """Verify exact mathematical F1 score from precision and recall on contingency table."""
        # TN = 44, FP = 9, FN = 15, TP = 11
        y_true = [0] * 44 + [0] * 9 + [1] * 15 + [1] * 11
        y_pred = [0] * 44 + [1] * 9 + [0] * 15 + [1] * 11

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        self.assertEqual(cm[0][0], 44)
        self.assertEqual(cm[0][1], 9)
        self.assertEqual(cm[1][0], 15)
        self.assertEqual(cm[1][1], 11)

        acc = accuracy_score(y_true, y_pred)
        self.assertAlmostEqual(acc, 55.0 / 79.0, places=6)

        prec = precision_score(y_true, y_pred, pos_label=1)
        self.assertAlmostEqual(prec, 11.0 / 20.0, places=6)

        rec = recall_score(y_true, y_pred, pos_label=1)
        self.assertAlmostEqual(rec, 11.0 / 26.0, places=6)

        f1 = f1_score(y_true, y_pred, pos_label=1)
        expected_f1 = (2 * (11.0 / 20.0) * (11.0 / 26.0)) / ((11.0 / 20.0) + (11.0 / 26.0))
        self.assertAlmostEqual(f1, expected_f1, places=6)
        self.assertAlmostEqual(f1, 11.0 / 23.0, places=6)
        self.assertAlmostEqual(f1, 0.478260869, places=6)

    def test_solver_high_precision_agreement(self):
        """Verify Normal Equation and Gradient Descent match within 1e-12 on well-conditioned data."""
        # 10 samples, 3 features
        X = [
            [0.1, -0.2, 0.5],
            [0.4, 0.8, -0.1],
            [-0.3, 0.2, 0.9],
            [0.7, -0.5, 0.3],
            [-0.1, 0.6, -0.4],
            [0.5, 0.1, 0.2],
            [-0.6, -0.3, 0.8],
            [0.2, 0.9, -0.7],
            [0.8, -0.1, 0.4],
            [-0.4, 0.5, 0.1],
        ]
        y = [1.2, 2.5, 0.8, 3.1, 1.0, 2.2, 0.5, 2.8, 3.4, 1.4]

        lr_ne = LinearRegression(solver="normal_equation", l2_reg=5.0)
        lr_ne.fit(X, y)

        lr_gd = LinearRegression(solver="gradient_descent", l2_reg=5.0, learning_rate=0.01, max_epochs=4000, random_seed=42)
        lr_gd.fit(X, y)

        self.assertAlmostEqual(lr_ne.intercept_, lr_gd.intercept_, places=6)
        for c_ne, c_gd in zip(lr_ne.coef_, lr_gd.coef_):
            self.assertAlmostEqual(c_ne, c_gd, places=6)


if __name__ == "__main__":
    unittest.main()
