"""
Linear Regression implemented from first principles.

Supports both:
1. Analytical closed-form solution via Normal Equation: theta = (X^T * X)^(-1) * X^T * y
2. Iterative optimization via Gradient Descent (Batch, Mini-batch, SGD)
"""

import random
from typing import List, Sequence, Optional, Union
from edupredict.models.base import BaseModel
from edupredict.math.vectors import dot_product
from edupredict.math.matrices import (
    matrix_transpose,
    matrix_multiply,
    matrix_vector_mul,
    invert_matrix,
)
from edupredict.losses.regression import mean_squared_error
from edupredict.optimization.gradient_descent import GradientDescent

Number = Union[int, float]


class LinearRegression(BaseModel):
    """
    Ordinary Least Squares Linear Regression.
    Hypothesis: y_hat = X * w + b.
    """

    def __init__(
        self,
        solver: str = "gd",
        learning_rate: float = 0.01,
        max_epochs: int = 1000,
        batch_size: Optional[int] = None,
        momentum: float = 0.0,
        l2_penalty: float = 0.0,
        random_seed: Optional[int] = 42
    ) -> None:
        super().__init__()
        if solver not in ("gd", "normal_equation"):
            raise ValueError(f"Unknown solver '{solver}'. Supported: 'gd', 'normal_equation'")
        self.solver = solver
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.momentum = momentum
        self.l2_penalty = l2_penalty
        self.random_seed = random_seed

        self.weights_: List[float] = []
        self.bias_: float = 0.0
        self.loss_history_: List[float] = []

    def fit(self, X: Sequence[Sequence[Number]], y: Sequence[Number]) -> "LinearRegression":
        """
        Fit linear model to training data X and targets y.
        """
        self._validate_inputs(X, y)
        n_samples = len(X)
        n_features = len(X[0])
        self._is_fitted = True

        if self.solver == "normal_equation":
            self._fit_normal_equation(X, y, n_samples, n_features)
        else:
            self._fit_gradient_descent(X, y, n_samples, n_features)

        return self

    def _fit_normal_equation(
        self,
        X: Sequence[Sequence[Number]],
        y: Sequence[Number],
        n_samples: int,
        n_features: int
    ) -> None:
        """
        Compute closed-form solution: theta = (X_aug^T * X_aug)^(-1) * X_aug^T * y
        where X_aug includes a column of 1s for the bias/intercept.
        """
        # Augment X with leading column of 1s for intercept
        X_aug = [[1.0] + [float(val) for val in row] for row in X]
        y_vec = [float(val) for val in y]

        XT = matrix_transpose(X_aug)
        XT_X = matrix_multiply(XT, X_aug)

        # Apply L2 ridge regularization to diagonal if l2_penalty > 0 to guarantee invertibility
        if self.l2_penalty > 0.0:
            for i in range(1, len(XT_X)):  # Do not regularize intercept
                XT_X[i][i] += self.l2_penalty

        try:
            XT_X_inv = invert_matrix(XT_X)
        except ValueError as e:
            raise RuntimeError(
                f"Normal equation failed: X^T * X is singular and cannot be inverted. "
                f"Features may be perfectly collinear. Suggest using solver='gd'. Details: {e}"
            )

        XT_y = matrix_vector_mul(XT, y_vec)
        theta = matrix_vector_mul(XT_X_inv, XT_y)

        self.bias_ = theta[0]
        self.weights_ = theta[1:]

        # Record final MSE
        preds = self.predict(X)
        self.loss_history_ = [mean_squared_error(y, preds)]

    def _fit_gradient_descent(
        self,
        X: Sequence[Sequence[Number]],
        y: Sequence[Number],
        n_samples: int,
        n_features: int
    ) -> None:
        """
        Optimize weights and bias iteratively via Gradient Descent.
        """
        rng = random.Random(self.random_seed)
        
        # Initialize small random weights
        self.weights_ = [rng.gauss(0.0, 0.01) for _ in range(n_features)]
        self.bias_ = 0.0
        self.loss_history_ = []

        optimizer = GradientDescent(
            learning_rate=self.learning_rate,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            momentum=self.momentum,
            random_seed=self.random_seed
        )
        optimizer.init_velocity(n_features)

        y_vec = [float(v) for v in y]

        for epoch in range(self.max_epochs):
            batches = optimizer.get_batches(n_samples, rng)

            for batch_indices in batches:
                m_b = len(batch_indices)
                grad_w = [0.0] * n_features
                grad_b = 0.0

                for idx in batch_indices:
                    x_i = X[idx]
                    y_i = y_vec[idx]
                    y_hat = dot_product(self.weights_, x_i) + self.bias_
                    error = y_hat - y_i

                    for j in range(n_features):
                        grad_w[j] += error * float(x_i[j])
                    grad_b += error

                # Average gradient over batch
                for j in range(n_features):
                    grad_w[j] = (2.0 / m_b) * grad_w[j]
                    if self.l2_penalty > 0.0:
                        grad_w[j] += (2.0 * self.l2_penalty * self.weights_[j])

                grad_b = (2.0 / m_b) * grad_b

                self.weights_, self.bias_ = optimizer.update(
                    self.weights_, self.bias_, grad_w, grad_b
                )

            # Record epoch MSE loss
            all_preds = self.predict(X)
            epoch_loss = mean_squared_error(y_vec, all_preds)
            self.loss_history_.append(epoch_loss)

    def predict(self, X: Sequence[Sequence[Number]]) -> List[float]:
        """
        Predict target values for samples in X.
        """
        self._check_is_fitted()
        self._validate_inputs(X, expected_features=len(self.weights_))
        return [dot_product(self.weights_, row) + self.bias_ for row in X]
