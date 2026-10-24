"""
Custom Linear Regression implemented entirely from scratch without external libraries.

Strictly zero third-party dependencies and zero external library imports (no NumPy, no math module).
Supports:
- Ordinary Least Squares and Ridge L2 Regularization.
- Closed-form Normal Equation solver: (X^T X + lambda I)^(-1) X^T y.
- Iterative Gradient Descent solver with Polyak momentum and early stopping.
- Full-batch, mini-batch, and stochastic gradient updates.
- Transparent training loss tracking and R^2 evaluation.
"""

from typing import List, Sequence, Optional, Dict, Any, Union
import random

from edupredict.models.base import BaseModel
from edupredict.math.matrices import (
    matrix_transpose,
    matrix_multiply,
    matrix_vector_mul,
    create_identity_matrix,
    invert_matrix,
)
from edupredict.losses.regression import mean_squared_error
from edupredict.metrics.regression import r2_score
from edupredict.optimization.gradient_descent import GradientDescent, initialize_parameters

Number = Union[int, float]
Matrix = Sequence[Sequence[Number]]
Vector = Sequence[Number]


class LinearRegression(BaseModel):
    """
    Classical Linear Regression estimator with Normal Equation and Gradient Descent solvers.
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        solver: str = "gradient_descent",
        learning_rate: float = 0.01,
        max_epochs: int = 1000,
        batch_size: Optional[int] = None,
        momentum: float = 0.0,
        l2_reg: float = 0.0,
        early_stopping: bool = False,
        patience: int = 10,
        min_delta: float = 1e-4,
        random_seed: Optional[int] = 42,
    ) -> None:
        super().__init__()
        valid_solvers = ("gradient_descent", "normal_equation")
        if solver not in valid_solvers:
            raise ValueError(f"Unknown solver '{solver}'. Expected one of {valid_solvers}")
        if l2_reg < 0.0:
            raise ValueError(f"l2_reg must be non-negative, got {l2_reg}")

        self.fit_intercept = fit_intercept
        self.solver = solver
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.momentum = momentum
        self.l2_reg = l2_reg
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.random_seed = random_seed

        # Learned parameters
        self.coef_: List[float] = []
        self.intercept_: float = 0.0
        self.loss_history_: List[float] = []
        self.n_iter_: int = 0

    def fit(self, X: Matrix, y: Vector) -> "LinearRegression":
        """
        Fit linear model to training data (X, y).
        """
        self._validate_inputs(X, y)
        n_samples = len(X)
        n_features = len(X[0])
        self.n_features_in_ = n_features
        y_float = [float(v) for v in y]

        if self.solver == "normal_equation":
            self._fit_normal_equation(X, y_float, n_samples, n_features)
        else:
            self._fit_gradient_descent(X, y_float, n_samples, n_features)

        self.is_fitted = True
        return self

    def _fit_normal_equation(
        self,
        X: Matrix,
        y: List[float],
        n_samples: int,
        n_features: int,
    ) -> None:
        """
        Solve parameters using the analytical closed-form normal equation:
        theta = (X_aug^T * X_aug + lambda * I)^(-1) * (X_aug^T * y)
        """
        if self.fit_intercept:
            # Augment X with leading column of 1.0s
            X_aug = [[1.0] + [float(val) for val in row] for row in X]
            dim = n_features + 1
        else:
            X_aug = [[float(val) for val in row] for row in X]
            dim = n_features

        # Compute X^T and X^T * X
        XT = matrix_transpose(X_aug)
        XT_X = matrix_multiply(XT, X_aug)

        # Add L2 ridge regularization to diagonal: (XT * X + N * lambda * I)
        # Scaled by n_samples so that l2_reg consistently minimizes the exact same objective
        # as Gradient Descent: J(w, b) = MSE(y, y_hat) + l2_reg * sum(w_j^2).
        # Intercept term at index 0 is strictly excluded from regularization.
        # Numerical safeguard (Tikhonov regularization): if l2_reg is 0, add 1e-6 to avoid singularity
        # from collinear one-hot dummy variables.
        ridge_diag = (self.l2_reg * n_samples) if self.l2_reg > 0.0 else 1e-6
        for i in range(dim):
            if self.fit_intercept and i == 0:
                continue
            XT_X[i][i] += float(ridge_diag)

        # Invert (XT_X + lambda * I)
        inv_XT_X = invert_matrix(XT_X)

        # Compute X^T * y
        XT_y = matrix_vector_mul(XT, y)

        # Compute theta = inv_XT_X * XT_y
        theta = matrix_vector_mul(inv_XT_X, XT_y)

        if self.fit_intercept:
            self.intercept_ = theta[0]
            self.coef_ = theta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = list(theta)

        # Record training loss
        preds = self._predict_raw(X)
        final_loss = mean_squared_error(y, preds)
        self.loss_history_ = [final_loss]
        self.n_iter_ = 1

    def _fit_gradient_descent(
        self,
        X: Matrix,
        y: List[float],
        n_samples: int,
        n_features: int,
    ) -> None:
        """
        Optimize parameters using first-order gradient descent.
        """
        optimizer = GradientDescent(
            learning_rate=self.learning_rate,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            momentum=self.momentum,
            early_stopping=self.early_stopping,
            patience=self.patience,
            min_delta=self.min_delta,
            random_seed=self.random_seed,
        )

        weights, bias = initialize_parameters(
            n_features=n_features,
            method="zeros",
            seed=self.random_seed,
        )
        if not self.fit_intercept:
            bias = 0.0

        optimizer.init_velocity(n_features)
        rng = random.Random(self.random_seed)
        X_clean = [[float(val) for val in row] for row in X]
        self.loss_history_ = []

        for epoch in range(self.max_epochs):
            batches = optimizer.get_batches(n_samples, rng)

            for batch_indices in batches:
                m_batch = len(batch_indices)
                grad_w = [0.0] * n_features
                grad_b = 0.0

                for idx in batch_indices:
                    xi = X_clean[idx]
                    yi = y[idx]
                    # Prediction: y_hat = w . x + b
                    pred_i = sum(weights[j] * xi[j] for j in range(n_features))
                    if self.fit_intercept:
                        pred_i += bias
                    err_i = pred_i - yi

                    for j in range(n_features):
                        grad_w[j] += err_i * xi[j]
                    if self.fit_intercept:
                        grad_b += err_i

                # Scale gradients: 2/M * sum(err * x) + 2 * lambda * w
                scale = 2.0 / m_batch
                for j in range(n_features):
                    grad_w[j] = scale * grad_w[j] + 2.0 * self.l2_reg * weights[j]
                grad_b = scale * grad_b if self.fit_intercept else 0.0

                weights, bias = optimizer.update(weights, bias, grad_w, grad_b)
                if not self.fit_intercept:
                    bias = 0.0

            # Compute full training MSE for epoch tracking
            epoch_preds = []
            for i in range(n_samples):
                p = sum(weights[j] * X_clean[i][j] for j in range(n_features))
                if self.fit_intercept:
                    p += bias
                epoch_preds.append(p)

            epoch_loss = mean_squared_error(y, epoch_preds)
            self.loss_history_.append(epoch_loss)

            if optimizer.should_stop_early(epoch_loss, weights, bias):
                if optimizer.best_weights is not None:
                    weights = list(optimizer.best_weights)
                    bias = optimizer.best_bias if optimizer.best_bias is not None else 0.0
                break

        self.coef_ = weights
        self.intercept_ = bias
        self.n_iter_ = len(self.loss_history_)

    def _predict_raw(self, X: Matrix) -> List[float]:
        """Generate numerical predictions without checking input validation repeatedly."""
        n_features = len(self.coef_)
        preds: List[float] = []
        for row in X:
            val = sum(self.coef_[j] * float(row[j]) for j in range(n_features))
            if self.fit_intercept:
                val += self.intercept_
            preds.append(val)
        return preds

    def predict(self, X: Matrix) -> List[float]:
        """
        Predict target values for feature matrix X.
        """
        self._check_is_fitted()
        self._validate_inputs(X, check_y=False)
        if len(X[0]) != len(self.coef_):
            raise ValueError(
                f"Feature count mismatch: model expects {len(self.coef_)} features, got {len(X[0])}"
            )
        return self._predict_raw(X)

    def score(self, X: Matrix, y: Vector) -> float:
        """
        Compute coefficient of determination R^2 on test data.
        """
        preds = self.predict(X)
        return r2_score([float(v) for v in y], preds)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model state to dictionary."""
        return {
            "model_type": "LinearRegression",
            "fit_intercept": self.fit_intercept,
            "solver": self.solver,
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "momentum": self.momentum,
            "l2_reg": self.l2_reg,
            "early_stopping": self.early_stopping,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "random_seed": self.random_seed,
            "coef_": self.coef_,
            "intercept_": self.intercept_,
            "loss_history_": self.loss_history_,
            "n_iter_": self.n_iter_,
            "is_fitted": self.is_fitted,
            "n_features_in_": self.n_features_in_,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LinearRegression":
        """Instantiate and restore model from serialized state."""
        model = cls(
            fit_intercept=data.get("fit_intercept", True),
            solver=data.get("solver", "gradient_descent"),
            learning_rate=data.get("learning_rate", 0.01),
            max_epochs=data.get("max_epochs", 1000),
            batch_size=data.get("batch_size"),
            momentum=data.get("momentum", 0.0),
            l2_reg=data.get("l2_reg", 0.0),
            early_stopping=data.get("early_stopping", False),
            patience=data.get("patience", 10),
            min_delta=data.get("min_delta", 1e-4),
            random_seed=data.get("random_seed", 42),
        )
        model.coef_ = list(data.get("coef_", []))
        model.intercept_ = float(data.get("intercept_", 0.0))
        model.loss_history_ = list(data.get("loss_history_", []))
        model.n_iter_ = int(data.get("n_iter_", 0))
        model.is_fitted = bool(data.get("is_fitted", False))
        model.n_features_in_ = data.get("n_features_in_")
        return model
